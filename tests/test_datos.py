"""Tests de la capa de datos (WP1, contrato C1)."""

import json
import shutil
import time

import pandas as pd
import pytest

from agente.config import datos_dir
from agente.datos import (
    DIAS_SEMANA,
    ErrorValidacionDatos,
    cargar_datos,
    reporte_calidad,
    tasa_emparejada,
)

NUEVAS = {"S015": "2026-03-16", "S033": "2026-03-16", "S037": "2026-03-16", "S059": "2026-03-16",
          "S076": "2026-03-17", "S082": "2026-03-17"}
INVALIDAS = {("2026-01-10", "S036", 99999), ("2026-02-09", "S092", 99999), ("2026-02-18", "S085", -5),
             ("2026-02-28", "S016", -44), ("2026-03-13", "S112", -14), ("2026-03-16", "S081", 99999),
             ("2026-03-20", "S035", 99999), ("2026-03-27", "S100", 99999)}


def _tasa(datos, **filtro):
    d = datos.diario_sucursal
    for col, valor in filtro.items():
        d = d[d[col] == valor]
    return tasa_emparejada(d)


# --- sucursales -------------------------------------------------------------

def test_sucursales_y_regiones(datos):
    s = datos.sucursales
    assert len(s) == 120
    assert s["region"].value_counts().to_dict() == {"Norte": 30, "Sur": 30, "Centro": 30, "Costa": 30}


def test_sucursales_nuevas(datos):
    s = datos.sucursales
    nuevas = s.loc[s["es_nueva"]]
    assert dict(zip(nuevas["sucursal_id"], nuevas["fecha_apertura"].dt.strftime("%Y-%m-%d"))) == NUEVAS
    assert (s.loc[~s["es_nueva"], "fecha_inicio_datos"] == pd.Timestamp("2026-01-05")).all()
    assert (nuevas["fecha_inicio_datos"] == nuevas["fecha_apertura"]).all()


def test_dias_operados(datos):
    s = datos.sucursales.set_index("sucursal_id")
    por_suc = datos.ventas.groupby("sucursal_id").size()
    for sid, fila in s.iterrows():
        esperado = {"2026-03-16": 14, "2026-03-17": 13}.get(NUEVAS.get(sid), 84)
        assert fila["dias_operados"] == esperado == por_suc[sid], sid


def test_capacitaciones_en_sucursales(datos):
    s = datos.sucursales
    assert int(s["capacitada"].sum()) == len(datos.capacitaciones) == 36
    assert s.loc[~s["capacitada"], "metodo_asignacion"].isna().all()


# --- quejas -----------------------------------------------------------------

def test_quejas_deduplicadas(datos):
    assert len(datos.quejas) == 6646
    assert not datos.quejas["queja_id"].duplicated().any()
    rep = reporte_calidad(datos)
    assert rep["filas_crudas"]["quejas"] == 6845
    assert rep["duplicados_quejas"]["filas_duplicadas"] == 199


def test_quejas_en_dia_invalido(datos):
    q = datos.quejas
    assert int(q["en_dia_invalido"].sum()) == 7
    claves_inv = {(f, s) for f, s, _ in INVALIDAS}
    marcadas = {(f"{r.fecha:%Y-%m-%d}", r.sucursal_id) for r in q.loc[q["en_dia_invalido"]].itertuples()}
    assert marcadas <= claves_inv


def test_ids_con_contenido_distinto_avisa(tmp_path):
    copiar = _copiar(tmp_path)
    q = pd.read_csv(copiar / "quejas.csv", dtype=str)
    dup = q.iloc[[0]].copy()
    dup.loc[:, "comentario"] = "otro texto"
    pd.concat([q, dup]).to_csv(copiar / "quejas.csv", index=False)
    d = cargar_datos(copiar)
    assert len(d.quejas) == 6646
    assert any(q.iloc[0]["queja_id"] in a and "contenido distinto" in a for a in d.avisos)
    assert d.quejas.set_index("queja_id").loc[q.iloc[0]["queja_id"], "comentario"] == q.iloc[0]["comentario"]


# --- ventas -----------------------------------------------------------------

def test_filas_invalidas(datos):
    v = datos.ventas
    inv = v.loc[v["dato_invalido"]]
    assert {(f"{r.fecha:%Y-%m-%d}", r.sucursal_id, int(r.pedidos)) for r in inv.itertuples()} == INVALIDAS
    assert inv["pedidos_validos"].isna().all()
    assert v.loc[~v["dato_invalido"], "pedidos_validos"].notna().all()
    assert str(v["pedidos_validos"].dtype) == "Int64"


def test_totales_pedidos(datos):
    v = datos.ventas
    assert int(v["pedidos_validos"].sum()) == 3_246_856
    por_region = v.groupby("region")["pedidos_validos"].sum()
    assert int(por_region["Costa"]) == 691_516
    assert int(por_region["Norte"]) == 833_561


def test_calendario(datos):
    v = datos.ventas
    assert v["fecha"].min() == pd.Timestamp("2026-01-05") and v["fecha"].max() == pd.Timestamp("2026-03-29")
    assert sorted(v["semana_n"].unique().tolist()) == list(range(1, 13))
    assert (v["semana_inicio"].dt.dayofweek == 0).all()
    assert ((v["fecha"] - v["semana_inicio"]).dt.days.between(0, 6)).all()
    assert set(v["mes"]) == {"2026-01", "2026-02", "2026-03"}
    assert set(v["dia_semana"]) == set(DIAS_SEMANA)
    assert v.loc[v["fecha"] == pd.Timestamp("2026-01-10"), "dia_semana"].iloc[0] == "sábado"
    assert v.loc[v["fecha"] == pd.Timestamp("2026-03-29"), "semana_n"].iloc[0] == 12
    q = datos.quejas
    assert {"semana_inicio", "semana_n", "mes", "dia_semana", "region"} <= set(q.columns)


# --- diario y tasas ---------------------------------------------------------

def test_diario_sucursal(datos):
    d = datos.diario_sucursal
    assert len(d) == len(datos.ventas)
    assert not d.duplicated(["fecha", "sucursal_id"]).any()
    assert int(d["quejas"].sum()) == 6646
    cats = [c for c in d.columns if c.startswith("quejas_")]
    assert len(cats) == 5
    assert (d[cats].sum(axis=1) == d["quejas"]).all()
    assert int(d.loc[d["dato_invalido"], "quejas"].sum()) == 7


def test_tasas_emparejadas(datos):
    assert _tasa(datos, sucursal_id="S035")["tasa"] == pytest.approx(1.393, abs=5e-4)
    assert _tasa(datos, sucursal_id="S100")["tasa"] == pytest.approx(2.293, abs=5e-4)
    assert _tasa(datos, region="Centro")["tasa"] == pytest.approx(2.1468, abs=5e-4)
    s082 = _tasa(datos, sucursal_id="S082")
    assert s082["tasa"] == 0.0 and s082["quejas"] == 0
    total = tasa_emparejada(datos.diario_sucursal)
    assert total["quejas"] + total["quejas_excluidas"] == 6646
    assert total["dias_excluidos"] == 8
    assert total["pedidos"] == 3_246_856


def test_tasa_sin_pedidos_es_none(datos):
    vacio = datos.diario_sucursal.iloc[0:0]
    assert tasa_emparejada(vacio)["tasa"] is None


# --- DuckDB y reporte -------------------------------------------------------

def test_vistas_duckdb(datos):
    con = datos.con
    assert con.execute("SELECT count(*) FROM quejas").fetchone()[0] == 6646
    for vista in ("sucursales", "ventas", "quejas", "capacitaciones", "diario_sucursal"):
        assert con.execute(f"SELECT count(*) FROM {vista}").fetchone()[0] > 0
    assert con.execute("SELECT sum(pedidos_validos) FROM ventas").fetchone()[0] == 3_246_856
    assert con.execute("SELECT count(*) FROM ventas WHERE pedidos_validos IS NULL").fetchone()[0] == 8
    tipo = con.execute("SELECT typeof(fecha) FROM ventas LIMIT 1").fetchone()[0]
    assert tipo == "DATE"


def test_reporte_calidad_json(datos):
    rep = reporte_calidad(datos)
    json.dumps(rep)
    assert len(rep["filas_invalidas_ventas"]) == 8
    assert rep["quejas_en_dias_invalidos"]["total"] == 7
    assert {s["sucursal_id"] for s in rep["sucursales_nuevas"]} == set(NUEVAS)
    assert rep["filas_limpias"]["quejas"] == 6646


def test_carga_rapida():
    t0 = time.perf_counter()
    cargar_datos()
    assert time.perf_counter() - t0 < 2.0


# --- validaciones con CSV corruptos ----------------------------------------

def _copiar(tmp_path):
    destino = tmp_path / "datos"
    shutil.copytree(datos_dir(), destino)
    return destino


def _alterar(directorio, archivo, fila, columna, valor):
    ruta = directorio / archivo
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    df.loc[fila, columna] = valor
    df.to_csv(ruta, index=False)


@pytest.mark.parametrize(
    "archivo,columna,valor,mensaje",
    [
        ("quejas.csv", "categoria", "ruido", "categoría desconocida"),
        ("quejas.csv", "canal", "fax", "canal desconocido"),
        ("quejas.csv", "sucursal_id", "S999", "sucursal inexistente"),
        ("ventas_diarias.csv", "sucursal_id", "S999", "sucursal inexistente"),
        ("capacitaciones.csv", "sucursal_id", "S999", "sucursal inexistente"),
        ("quejas.csv", "fecha", "2026-04-02", "fuera del periodo"),
        ("ventas_diarias.csv", "fecha", "2025-12-31", "fuera del periodo"),
        ("ventas_diarias.csv", "fecha", "05/01/2026", "formato inválido"),
        ("ventas_diarias.csv", "pedidos", "muchos", "tipos inválidos"),
    ],
)
def test_validacion_falla(tmp_path, archivo, columna, valor, mensaje):
    directorio = _copiar(tmp_path)
    _alterar(directorio, archivo, 0, columna, valor)
    with pytest.raises(ErrorValidacionDatos, match=mensaje):
        cargar_datos(directorio)


def test_ventas_antes_de_apertura(tmp_path):
    directorio = _copiar(tmp_path)
    ventas = pd.read_csv(directorio / "ventas_diarias.csv", dtype=str)
    extra = pd.DataFrame([{"fecha": "2026-03-10", "sucursal_id": "S015", "pedidos": "100"}])
    pd.concat([ventas, extra]).to_csv(directorio / "ventas_diarias.csv", index=False)
    with pytest.raises(ErrorValidacionDatos, match="antes de la apertura"):
        cargar_datos(directorio)


def test_columna_faltante(tmp_path):
    directorio = _copiar(tmp_path)
    q = pd.read_csv(directorio / "quejas.csv", dtype=str).drop(columns=["canal"])
    q.to_csv(directorio / "quejas.csv", index=False)
    with pytest.raises(ErrorValidacionDatos, match="faltan columnas"):
        cargar_datos(directorio)


def test_queja_en_dia_sin_ventas(tmp_path):
    directorio = _copiar(tmp_path)
    _alterar(directorio, "quejas.csv", 0, "sucursal_id", "S015")  # S015 no tiene ventas el 2026-01-05
    with pytest.raises(ErrorValidacionDatos, match="días sin fila de ventas"):
        cargar_datos(directorio)


def test_datos_dir_por_entorno(tmp_path, monkeypatch):
    directorio = _copiar(tmp_path)
    monkeypatch.setenv("DATOS_DIR", str(directorio))
    assert reporte_calidad(cargar_datos())["filas_limpias"]["quejas"] == 6646

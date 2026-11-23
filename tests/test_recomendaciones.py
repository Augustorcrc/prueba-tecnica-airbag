"""Tests del score determinístico de recomendaciones (contrato C7, parte sin LLM)."""

import dataclasses
import json
from datetime import date

import pytest

from agente.datos import REGIONES, tasa_emparejada
from agente.recomendaciones import (
    ACCIONES_POR_CATEGORIA,
    PREFIJO_PREVENTIVA,
    _candidatos_visita,
    candidatos_visita,
)


def _todas(datos, **kw):
    """Lista completa, sin el tope de 10 de la herramienta (solo para tests)."""
    return _candidatos_visita(datos, n=200, n_maximo=None, **kw)


def _ids(r):
    return [c["sucursal_id"] for c in r["resultado"]["candidatos"]]


def _tam(r):
    return len(json.dumps(r, ensure_ascii=False).encode("utf-8"))


def test_default_cadena(datos):
    r = candidatos_visita(datos)
    assert _ids(r) == ["S056", "S024", "S107", "S053", "S045"]
    excesos = [c["exceso_quejas"] for c in r["resultado"]["candidatos"]]
    assert excesos == pytest.approx([44.7, 21.8, 15.0, 13.8, 13.7], abs=0.1)
    assert excesos == sorted(excesos, reverse=True)
    assert all(c["supera_umbral"] and not c["preventiva"] for c in r["resultado"]["candidatos"])
    assert r["periodo"] == {"desde": "2026-03-02", "hasta": "2026-03-29"}
    p = r["resultado"]["parametros"]
    assert p["umbral_ratio"] == 1.5 and p["n_supera_umbral"] >= 5 and p["n_elegibles"] > 0
    assert p["ventana_anterior"] == "2026-02-02..2026-03-01"
    assert r["n_registros"] == 5


def test_componentes_consistentes(datos):
    c = candidatos_visita(datos)["resultado"]["candidatos"][0]
    claves = {"sucursal_id", "nombre", "region", "ciudad", "tasa_4sem", "tasa_region_4sem",
              "ratio_regional", "quejas_4sem", "pedidos_4sem", "exceso_quejas", "tasa_4sem_anterior", "tendencia",
              "categoria_dominante", "pct_categoria_dominante", "capacitada", "fecha_capacitacion",
              "metodo_asignacion", "supera_umbral", "preventiva", "accion_sugerida_base", "advertencias"}
    assert claves <= set(c)
    assert c["tasa_4sem"] == pytest.approx(c["quejas_4sem"] / c["pedidos_4sem"] * 1000, abs=0.01)
    assert c["ratio_regional"] == pytest.approx(c["tasa_4sem"] / c["tasa_region_4sem"], abs=0.01)
    esperado = c["quejas_4sem"] - c["pedidos_4sem"] * c["tasa_region_4sem"] / 1000
    assert c["exceso_quejas"] == pytest.approx(esperado, abs=0.2)
    assert c["categoria_dominante"] in ACCIONES_POR_CATEGORIA


def test_region_costa(datos):
    r = candidatos_visita(datos, region="Costa")
    cands = r["resultado"]["candidatos"]
    assert 3 <= len(cands) <= 5
    assert all(c["region"] == "Costa" for c in cands)


def test_menos_de_tres_sobre_umbral_completa_con_preventivas(datos):
    # Datos manipulados: en Costa todas las sucursales tienen la misma tasa salvo S096 (triple).
    diario = datos.diario_sucursal.copy()
    costa = diario["region"] == "Costa"
    base = (diario.loc[costa, "pedidos_validos"].fillna(0).astype("int64") // 100)
    diario.loc[costa, "quejas"] = base
    diario.loc[costa & (diario["sucursal_id"] == "S096"), "quejas"] = base * 3
    manipulado = dataclasses.replace(datos, diario_sucursal=diario)
    r = candidatos_visita(manipulado, region="Costa")
    cands = r["resultado"]["candidatos"]
    assert len(cands) == 3
    assert cands[0]["sucursal_id"] == "S096" and cands[0]["preventiva"] is False
    assert all(c["preventiva"] and not c["supera_umbral"] for c in cands[1:])
    assert r["resultado"]["parametros"]["n_supera_umbral"] == 1
    assert any(PREFIJO_PREVENTIVA in a for a in r["advertencias"])
    assert PREFIJO_PREVENTIVA.startswith("Atención preventiva: no supera el umbral, pero")


@pytest.mark.parametrize("n", [2, 7])
def test_n_exacto(datos, n):
    r = candidatos_visita(datos, n=n)
    assert len(r["resultado"]["candidatos"]) == n
    assert _ids(r)[:min(n, 5)] == ["S056", "S024", "S107", "S053", "S045"][:n]


def test_n_marca_preventivas(datos):
    r = _todas(datos)
    for c in r["resultado"]["candidatos"]:
        assert c["preventiva"] == (c["ratio_regional"] < 1.5)
    assert any(c["preventiva"] for c in r["resultado"]["candidatos"])


def test_nuevas(datos):
    nuevas = set(datos.sucursales.loc[datos.sucursales["es_nueva"], "sucursal_id"])
    todas = _todas(datos)
    assert not (set(_ids(todas)) & nuevas)
    con = _todas(datos, incluir_nuevas=True)
    entradas = [c for c in con["resultado"]["candidatos"] if c["sucursal_id"] in nuevas]
    assert entradas
    assert all(any(a.startswith("abrió el ") and "poco confiable" in a for a in c["advertencias"])
               for c in entradas)


def test_tendencias_y_seguimiento(datos):
    r = _todas(datos)
    por_id = {c["sucursal_id"]: c for c in r["resultado"]["candidatos"]}
    for sid in ("S012", "S067", "S107"):
        assert por_id[sid]["tendencia"]["etiqueta"] == "baja"
    s012 = por_id["S012"]
    assert s012["capacitada"] is True
    accion = s012["accion_sugerida_base"].lower()
    assert accion.startswith("seguimiento") and "capacitar" not in accion
    assert s012["fecha_capacitacion"] in accion


def test_orden_superan_primero_y_por_exceso(datos):
    cands = _todas(datos)["resultado"]["candidatos"]
    supera = [c["supera_umbral"] for c in cands]
    assert supera == sorted(supera, reverse=True)  # primero todas las que superan el umbral
    for grupo in (True, False):
        excesos = [c["exceso_quejas"] for c in cands if c["supera_umbral"] is grupo]
        assert excesos == sorted(excesos, reverse=True)


def test_categorias_exceso(datos):
    por_id = {c["sucursal_id"]: c for c in candidatos_visita(datos)["resultado"]["candidatos"]}
    for c in por_id.values():
        assert len(c["categorias_exceso"]) <= 2
        for x in c["categorias_exceso"]:
            assert x["quejas"] - x["esperadas"] >= 3 - 0.05 and x["quejas"] >= 1.5 * x["esperadas"] - 0.05
    # S024: lo que más se aparta de su región es limpieza, y la acción arranca por limpieza.
    s024 = por_id["S024"]
    assert s024["categorias_exceso"][0] == {"categoria": "limpieza", "quejas": 9, "esperadas": 2.3}
    assert s024["accion_sugerida_base"].startswith(f"escalar: {ACCIONES_POR_CATEGORIA['limpieza']}; además, ")
    # S056: demora y pedido incorrecto.
    assert [x["categoria"] for x in por_id["S056"]["categorias_exceso"]] == ["demora", "pedido_incorrecto"]
    # Con categoría pedida no se calcula (la acción es la de esa categoría).
    assert "categorias_exceso" not in candidatos_visita(datos, categoria="limpieza")["resultado"]["candidatos"][0]


def test_capacitada_en_suba_escala(datos):
    por_id = {c["sucursal_id"]: c for c in _todas(datos)["resultado"]["candidatos"]}
    s024 = por_id["S024"]
    assert s024["capacitada"] is True and s024["tendencia"]["etiqueta"] == "sube"
    accion = s024["accion_sugerida_base"].lower()
    assert accion.startswith("escalar:") and "capacitar" not in accion
    assert s024["fecha_capacitacion"] in accion


def test_categoria_usa_solo_esa_categoria(datos):
    r = candidatos_visita(datos, categoria="limpieza")
    c = r["resultado"]["candidatos"][0]
    sub = datos.diario_sucursal
    sub = sub[(sub["sucursal_id"] == c["sucursal_id"]) & (sub["fecha"] >= "2026-03-02")
              & (sub["fecha"] <= "2026-03-29") & ~sub["dato_invalido"].astype(bool)]
    assert c["quejas_4sem"] == int(sub["quejas_limpieza"].sum())
    assert r["filtros_aplicados"]["categoria"] == "limpieza"


def test_fecha_hasta_string_y_date(datos):
    a = candidatos_visita(datos, fecha_hasta="2026-03-29")
    b = candidatos_visita(datos, fecha_hasta=date(2026, 3, 29))
    assert _ids(a) == _ids(b) == _ids(candidatos_visita(datos))


@pytest.mark.parametrize("kwargs", [
    {"region": "Atlántida"}, {"categoria": "ruido"}, {"n": 0}, {"n": -3},
    {"fecha_hasta": "2025-12-31"}, {"fecha_hasta": "2026-04-01"}, {"fecha_hasta": "no-fecha"},
])
def test_errores(datos, kwargs):
    r = candidatos_visita(datos, **kwargs)
    assert "error" in r and isinstance(r["valores_validos"], list) and r["valores_validos"]
    json.dumps(r)


def test_region_inexistente_valores_validos(datos):
    r = candidatos_visita(datos, region="Atlántida")
    assert r["valores_validos"] == list(REGIONES)


def test_serializable_y_tamano(datos):
    for kwargs in ({}, {"region": "Costa"}, {"categoria": "demora"}, {"incluir_nuevas": True}):
        r = candidatos_visita(datos, **kwargs)
        assert json.loads(json.dumps(r, ensure_ascii=False)) == r
        assert _tam(r) < 5200


def test_tasa_emparejada_s035(datos):
    d = datos.diario_sucursal
    t = tasa_emparejada(d[d["sucursal_id"] == "S035"])["tasa"]
    assert round(t, 3) == 1.393


# --- Fix B: categoría (I4) y tope/tamaño (M5) ---------------------------------------------------

def test_categoria_define_la_accion(datos):
    r = candidatos_visita(datos, categoria="limpieza")
    cands = r["resultado"]["candidatos"]
    assert cands
    base = ACCIONES_POR_CATEGORIA["limpieza"]
    for c in cands:
        accion = c["accion_sugerida_base"]
        assert "demora" not in accion.lower() and "dotación" not in accion
        assert accion == base or accion.startswith((f"seguimiento: {base};", f"escalar: {base};"))
        assert "categoria_dominante" in c  # queda como dato extra
    # S024 tiene "demora" como dominante pero la acción es la de limpieza.
    assert any(c["categoria_dominante"] != "limpieza" for c in cands)


def test_categoria_informa_quejas_y_pct(datos):
    r = candidatos_visita(datos, categoria="limpieza")
    d = datos.diario_sucursal
    for c in r["resultado"]["candidatos"]:
        sub = d[(d["sucursal_id"] == c["sucursal_id"]) & (d["fecha"] >= "2026-03-02") & (d["fecha"] <= "2026-03-29")]
        q_cat, q_tot = int(sub["quejas_limpieza"].sum()), int(sub["quejas"].sum())
        assert c["quejas_categoria_4sem"] == q_cat
        assert c["pct_categoria_4sem"] == round(100 * q_cat / q_tot, 1)
    # Sin categoría no aparecen.
    assert "pct_categoria_4sem" not in candidatos_visita(datos)["resultado"]["candidatos"][0]


def test_n_tope_10(datos):
    r = candidatos_visita(datos, n=30)
    assert len(r["resultado"]["candidatos"]) == 10
    assert any("se limitó a 10" in a for a in r["advertencias"])
    assert _ids(r) == _ids(_todas(datos))[:10]


@pytest.mark.parametrize("kwargs, limite", [
    ({"n": 5}, 5200), ({"n": 5, "categoria": "limpieza"}, 5200),
    ({"n": 5, "categoria": "demora", "incluir_nuevas": True}, 5200),
    ({"n": 10}, 10400), ({"n": 10, "categoria": "trato", "incluir_nuevas": True}, 10400),
])
def test_tamano_con_recorte_de_campos(datos, kwargs, limite):
    r = candidatos_visita(datos, **kwargs)
    assert len(json.dumps(r)) < limite
    assert len(r["resultado"]["candidatos"]) == kwargs["n"]  # se recortan campos, no ítems
    for c in r["resultado"]["candidatos"]:
        assert {"sucursal_id", "ratio_regional", "accion_sugerida_base"} <= set(c)

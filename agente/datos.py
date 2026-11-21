"""Capa de datos (contrato C1): carga, limpieza y vistas DuckDB.

Única fuente de verdad: todo el resto del agente consume `Datos`. El LLM nunca ve los CSV.

Decisiones documentadas
-----------------------
- Periodo: 2026-01-05 (lunes) a 2026-03-29 (domingo) = 84 días, 12 semanas.
- Fechas: columnas `datetime64[ns]` normalizadas (sin hora) en los DataFrames; en DuckDB se
  exponen como `DATE`.
- Ventas (D1.2): `dato_invalido = pedidos < 0 or pedidos >= 99999`. `pedidos_validos` es dtype
  nullable `Int64` con `<NA>` en las filas inválidas (NULL en DuckDB). Además corre un chequeo
  estadístico (z robusto con mediana/MAD por sucursal, |z| > 8) que SOLO agrega avisos.
- Quejas: se deduplican por `queja_id` (se queda la primera aparición; si hay IDs repetidos con
  contenido distinto se agrega un aviso). Cada queja lleva `en_dia_invalido` (D1.1, opción B):
  la queja cae en un (fecha, sucursal) cuya fila de ventas es inválida. Los conteos de quejas
  incluyen todas; las tasas usan `tasa_emparejada` (excluye esos días del numerador y del
  denominador).
- Sucursales (D1.3): `es_nueva` = `fecha_apertura` dentro del periodo.
- Calendario (ventas, quejas y diario_sucursal):
    * `semana_inicio`: lunes de la semana (datetime64 / DATE).
    * `semana_n`: 1..12 (semana 1 = 2026-01-05..2026-01-11).
    * `mes`: texto "AAAA-MM" ("2026-01", "2026-02", "2026-03").
    * `dia_semana`: minúscula en español, con tilde: "lunes", "martes", "miércoles", "jueves",
      "viernes", "sábado", "domingo".
- Columnas extra: `region` en ventas, quejas y diario_sucursal; `dato_invalido` en
  diario_sucursal; `fecha_capacitacion`, `metodo_asignacion` y `capacitada` en sucursales.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from agente.config import FECHA_INICIO as _FECHA_INICIO
from agente.config import FECHA_REFERENCIA as _FECHA_REFERENCIA

if TYPE_CHECKING:  # pragma: no cover
    import duckdb

FECHA_REFERENCIA: date = _FECHA_REFERENCIA
FECHA_INICIO: date = _FECHA_INICIO

REGIONES = ("Norte", "Sur", "Centro", "Costa")
CATEGORIAS = ("demora", "pedido_incorrecto", "trato", "limpieza", "calidad_producto")
CANALES = ("app", "telefono", "redes", "en_tienda")
METODOS_ASIGNACION = ("aleatorio", "por_desempeno")
DIAS_SEMANA = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")

UMBRAL_PEDIDOS_INVALIDO = 99999  # pedidos >= este valor es un centinela de error
UMBRAL_Z_ROBUSTO = 8.0

FORMATO_FECHA = "%Y-%m-%d"

# archivo -> (columnas esperadas con dtype de lectura, columnas de fecha)
ESQUEMAS: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {
    "sucursales.csv": (
        {"sucursal_id": "str", "nombre": "str", "region": "str", "ciudad": "str", "fecha_apertura": "str"},
        ("fecha_apertura",),
    ),
    "ventas_diarias.csv": ({"fecha": "str", "sucursal_id": "str", "pedidos": "int64"}, ("fecha",)),
    "quejas.csv": (
        {"queja_id": "str", "fecha": "str", "sucursal_id": "str", "categoria": "str", "canal": "str",
         "comentario": "str"},
        ("fecha",),
    ),
    "capacitaciones.csv": (
        {"sucursal_id": "str", "fecha_capacitacion": "str", "metodo_asignacion": "str"},
        ("fecha_capacitacion",),
    ),
}


@dataclass
class Datos:
    """DataFrames limpios más una conexión DuckDB con vistas registradas."""

    sucursales: pd.DataFrame  # + es_nueva, dias_operados, fecha_inicio_datos
    ventas: pd.DataFrame  # + pedidos_validos (NULL si inválido), dato_invalido, semana_inicio, mes, dia_semana, semana_n
    quejas: pd.DataFrame  # deduplicada por queja_id; + en_dia_invalido, mismas columnas de calendario
    capacitaciones: pd.DataFrame
    diario_sucursal: pd.DataFrame  # fecha, sucursal_id, pedidos_validos, quejas, quejas por categoría
    con: Optional["duckdb.DuckDBPyConnection"]  # vistas: sucursales, ventas, quejas, capacitaciones, diario_sucursal
    avisos: list[str] = field(default_factory=list)


class ErrorValidacionDatos(ValueError):
    """Los CSV no cumplen las validaciones duras de la carga."""


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------

def _leer_csv(directorio: Path, archivo: str) -> pd.DataFrame:
    """Lee un CSV con dtypes explícitos y fechas con formato fijo; valida columnas."""
    columnas, fechas = ESQUEMAS[archivo]
    ruta = directorio / archivo
    if not ruta.exists():
        raise ErrorValidacionDatos(f"No existe el archivo {ruta}")
    encabezado = pd.read_csv(ruta, nrows=0).columns.tolist()
    faltantes = [c for c in columnas if c not in encabezado]
    if faltantes:
        raise ErrorValidacionDatos(f"{archivo}: faltan columnas {faltantes}")
    try:
        df = pd.read_csv(ruta, dtype=columnas, usecols=list(columnas), keep_default_na=False)
    except (ValueError, TypeError) as error:
        raise ErrorValidacionDatos(f"{archivo}: tipos inválidos ({error})") from error
    df = df[list(columnas)]
    for col in fechas:
        try:
            df[col] = pd.to_datetime(df[col], format=FORMATO_FECHA)
        except (ValueError, TypeError) as error:
            raise ErrorValidacionDatos(f"{archivo}: fecha con formato inválido en '{col}' ({error})") from error
    return df


# ---------------------------------------------------------------------------
# Validaciones duras
# ---------------------------------------------------------------------------

def _fallar_si(mascara: pd.Series, df: pd.DataFrame, mensaje: str, columnas: list[str]) -> None:
    if bool(mascara.any()):
        ejemplos = df.loc[mascara, columnas].head(5).astype(str).to_dict("records")
        raise ErrorValidacionDatos(f"{mensaje}: {int(mascara.sum())} filas, p. ej. {ejemplos}")


def _validar(suc: pd.DataFrame, ventas: pd.DataFrame, quejas: pd.DataFrame, cap: pd.DataFrame) -> None:
    inicio, fin = pd.Timestamp(FECHA_INICIO), pd.Timestamp(FECHA_REFERENCIA)

    _fallar_si(suc["sucursal_id"].duplicated(), suc, "sucursales.csv: sucursal_id repetido", ["sucursal_id"])
    _fallar_si(~suc["region"].isin(REGIONES), suc, "sucursales.csv: región desconocida", ["sucursal_id", "region"])
    ids = set(suc["sucursal_id"])

    for nombre, df in (("ventas_diarias.csv", ventas), ("quejas.csv", quejas), ("capacitaciones.csv", cap)):
        _fallar_si(~df["sucursal_id"].isin(ids), df, f"{nombre}: sucursal inexistente", ["sucursal_id"])

    for nombre, df, col in (("ventas_diarias.csv", ventas, "fecha"), ("quejas.csv", quejas, "fecha"),
                            ("capacitaciones.csv", cap, "fecha_capacitacion")):
        fuera = (df[col] < inicio) | (df[col] > fin)
        _fallar_si(fuera, df, f"{nombre}: fecha fuera del periodo {FECHA_INICIO}..{FECHA_REFERENCIA}",
                   ["sucursal_id", col])

    _fallar_si(~quejas["categoria"].isin(CATEGORIAS), quejas, "quejas.csv: categoría desconocida",
               ["queja_id", "categoria"])
    _fallar_si(~quejas["canal"].isin(CANALES), quejas, "quejas.csv: canal desconocido", ["queja_id", "canal"])
    _fallar_si(~cap["metodo_asignacion"].isin(METODOS_ASIGNACION), cap,
               "capacitaciones.csv: método de asignación desconocido", ["sucursal_id", "metodo_asignacion"])
    _fallar_si(cap["sucursal_id"].duplicated(), cap, "capacitaciones.csv: sucursal repetida", ["sucursal_id"])

    _fallar_si(ventas.duplicated(["fecha", "sucursal_id"]), ventas,
               "ventas_diarias.csv: (fecha, sucursal) repetido", ["fecha", "sucursal_id"])
    apertura = ventas["sucursal_id"].map(suc.set_index("sucursal_id")["fecha_apertura"])
    _fallar_si(ventas["fecha"] < apertura, ventas, "ventas_diarias.csv: ventas antes de la apertura",
               ["fecha", "sucursal_id"])

    claves_ventas = pd.MultiIndex.from_frame(ventas[["fecha", "sucursal_id"]])
    sin_ventas = ~pd.MultiIndex.from_frame(quejas[["fecha", "sucursal_id"]]).isin(claves_ventas)
    _fallar_si(pd.Series(sin_ventas, index=quejas.index), quejas,
               "quejas.csv: quejas en días sin fila de ventas", ["queja_id", "fecha", "sucursal_id"])


# ---------------------------------------------------------------------------
# Transformaciones
# ---------------------------------------------------------------------------

def _agregar_calendario(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega semana_inicio, semana_n, mes y dia_semana a partir de `fecha`."""
    fecha = df["fecha"]
    df = df.copy()
    df["semana_inicio"] = fecha - pd.to_timedelta(fecha.dt.dayofweek, unit="D")
    df["semana_n"] = ((fecha - pd.Timestamp(FECHA_INICIO)).dt.days // 7 + 1).astype("int64")
    df["mes"] = fecha.dt.strftime("%Y-%m")
    df["dia_semana"] = fecha.dt.dayofweek.map(dict(enumerate(DIAS_SEMANA)))
    return df


def _avisos_estadisticos(ventas: pd.DataFrame) -> list[str]:
    """z robusto (mediana/MAD por sucursal) sobre filas válidas. Solo informa, no limpia."""
    validas = ventas.loc[~ventas["dato_invalido"], ["fecha", "sucursal_id", "pedidos"]].copy()
    validas["pedidos"] = validas["pedidos"].astype("float64")
    grupos = validas.groupby("sucursal_id")["pedidos"]
    mediana = grupos.transform("median")
    mad = (validas["pedidos"] - mediana).abs().groupby(validas["sucursal_id"]).transform("median")
    z = 0.6745 * (validas["pedidos"] - mediana) / mad.where(mad > 0)
    raros = validas.loc[z.abs() > UMBRAL_Z_ROBUSTO]
    return [
        f"ventas: valor atípico (z robusto {z.loc[i]:.1f}) en {r.sucursal_id} el {r.fecha:%Y-%m-%d}: "
        f"{int(r.pedidos)} pedidos; se conserva"
        for i, r in raros.iterrows()
    ]


def _deduplicar_quejas(quejas: pd.DataFrame, avisos: list[str]) -> tuple[pd.DataFrame, dict]:
    exactos = int(quejas.duplicated().sum())
    sin_exactos = quejas.drop_duplicates()
    conflictivos = sin_exactos.loc[sin_exactos["queja_id"].duplicated(keep=False), "queja_id"].unique().tolist()
    for qid in conflictivos:
        avisos.append(f"quejas: el ID {qid} aparece con contenido distinto; se conserva la primera fila")
    limpio = quejas.drop_duplicates("queja_id", keep="first").reset_index(drop=True)
    return limpio, {
        "filas_duplicadas": len(quejas) - len(limpio),
        "duplicados_exactos": exactos,
        "ids_con_contenido_distinto": [str(q) for q in conflictivos],
    }


def _construir_diario(ventas: pd.DataFrame, quejas: pd.DataFrame) -> pd.DataFrame:
    conteo = (
        quejas.groupby(["fecha", "sucursal_id", "categoria"]).size().unstack("categoria")
        .reindex(columns=list(CATEGORIAS)).fillna(0).astype("int64")
    )
    conteo.columns = [f"quejas_{c}" for c in conteo.columns]
    conteo = conteo.reset_index()
    base = ventas[["fecha", "sucursal_id", "region", "pedidos_validos", "dato_invalido",
                   "semana_inicio", "semana_n", "mes", "dia_semana"]]
    diario = base.merge(conteo, on=["fecha", "sucursal_id"], how="left")
    cols_cat = [f"quejas_{c}" for c in CATEGORIAS]
    diario[cols_cat] = diario[cols_cat].fillna(0).astype("int64")
    diario["quejas"] = diario[cols_cat].sum(axis=1).astype("int64")
    orden = ["fecha", "sucursal_id", "region", "pedidos_validos", "quejas", *cols_cat, "dato_invalido",
             "semana_inicio", "semana_n", "mes", "dia_semana"]
    return diario[orden].sort_values(["fecha", "sucursal_id"]).reset_index(drop=True)


def _registrar_duckdb(tablas: dict[str, pd.DataFrame]):
    """Conexión en memoria con una vista por tabla limpia; las fechas quedan como DATE."""
    import duckdb

    con = duckdb.connect(":memory:")
    for nombre, df in tablas.items():
        interno = f"_df_{nombre}"
        con.register(interno, df)
        fechas = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        reemplazo = ", ".join(f'CAST("{c}" AS DATE) AS "{c}"' for c in fechas)
        seleccion = f"* REPLACE ({reemplazo})" if fechas else "*"
        con.execute(f'CREATE VIEW "{nombre}" AS SELECT {seleccion} FROM "{interno}"')
    return con


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def cargar_datos(dir: Path | None = None) -> Datos:
    """Carga, valida y limpia los CSV de `dir` (por defecto `DATOS_DIR` o `./datos`).

    Lanza `ErrorValidacionDatos` si falla una validación dura (columnas, tipos, IDs de sucursal,
    categorías, canales, fechas fuera del periodo, ventas antes de la apertura, quejas en días
    sin ventas). Ver el docstring del módulo para las decisiones de limpieza.
    """
    from agente.config import datos_dir

    t0 = time.perf_counter()
    directorio = Path(dir) if dir is not None else datos_dir()
    avisos: list[str] = []

    suc = _leer_csv(directorio, "sucursales.csv")
    ventas = _leer_csv(directorio, "ventas_diarias.csv")
    quejas_crudas = _leer_csv(directorio, "quejas.csv")
    cap = _leer_csv(directorio, "capacitaciones.csv")
    crudas = {"sucursales": len(suc), "ventas": len(ventas), "quejas": len(quejas_crudas),
              "capacitaciones": len(cap)}

    _validar(suc, ventas, quejas_crudas, cap)
    region = suc.set_index("sucursal_id")["region"]

    # --- ventas
    ventas = ventas.sort_values(["fecha", "sucursal_id"]).reset_index(drop=True)
    ventas["dato_invalido"] = (ventas["pedidos"] < 0) | (ventas["pedidos"] >= UMBRAL_PEDIDOS_INVALIDO)
    ventas["pedidos_validos"] = ventas["pedidos"].astype("Int64").mask(ventas["dato_invalido"])
    ventas["region"] = ventas["sucursal_id"].map(region)
    ventas = _agregar_calendario(ventas)
    ventas = ventas[["fecha", "sucursal_id", "region", "pedidos", "pedidos_validos", "dato_invalido",
                     "semana_inicio", "semana_n", "mes", "dia_semana"]]
    for _, r in ventas.loc[ventas["dato_invalido"]].iterrows():
        avisos.append(f"ventas: valor inválido en {r.sucursal_id} el {r.fecha:%Y-%m-%d} ({int(r.pedidos)} "
                      "pedidos); se trata como dato faltante")
    avisos.extend(_avisos_estadisticos(ventas))

    # --- quejas
    quejas, info_dup = _deduplicar_quejas(quejas_crudas, avisos)
    claves_invalidas = pd.MultiIndex.from_frame(ventas.loc[ventas["dato_invalido"], ["fecha", "sucursal_id"]])
    quejas["en_dia_invalido"] = pd.MultiIndex.from_frame(quejas[["fecha", "sucursal_id"]]).isin(claves_invalidas)
    quejas["region"] = quejas["sucursal_id"].map(region)
    quejas = _agregar_calendario(quejas)
    quejas = quejas[["queja_id", "fecha", "sucursal_id", "region", "categoria", "canal", "comentario",
                     "en_dia_invalido", "semana_inicio", "semana_n", "mes", "dia_semana"]]
    quejas = quejas.sort_values(["fecha", "queja_id"]).reset_index(drop=True)

    # --- sucursales
    inicio, fin = pd.Timestamp(FECHA_INICIO), pd.Timestamp(FECHA_REFERENCIA)
    suc["es_nueva"] = (suc["fecha_apertura"] >= inicio) & (suc["fecha_apertura"] <= fin)
    suc["dias_operados"] = suc["sucursal_id"].map(ventas.groupby("sucursal_id").size()).fillna(0).astype("int64")
    suc["fecha_inicio_datos"] = suc["fecha_apertura"].where(suc["fecha_apertura"] > inicio, inicio)
    cap_idx = cap.set_index("sucursal_id")
    suc["fecha_capacitacion"] = suc["sucursal_id"].map(cap_idx["fecha_capacitacion"])
    suc["metodo_asignacion"] = suc["sucursal_id"].map(cap_idx["metodo_asignacion"])
    suc["capacitada"] = suc["sucursal_id"].isin(cap_idx.index)
    suc = suc.sort_values("sucursal_id").reset_index(drop=True)

    # --- diario
    diario = _construir_diario(ventas, quejas)

    con = _registrar_duckdb({"sucursales": suc, "ventas": ventas, "quejas": quejas,
                             "capacitaciones": cap, "diario_sucursal": diario})

    datos = Datos(sucursales=suc, ventas=ventas, quejas=quejas, capacitaciones=cap,
                  diario_sucursal=diario, con=con, avisos=avisos)
    # Metadatos de calidad: atributo privado (no es campo del contrato C1).
    datos._calidad = {  # type: ignore[attr-defined]
        "directorio": str(directorio),
        "filas_crudas": crudas,
        "duplicados_quejas": info_dup,
        "segundos_carga": round(time.perf_counter() - t0, 3),
    }
    return datos


def tasa_emparejada(diario: pd.DataFrame) -> dict:
    """Tasa de quejas por 1.000 pedidos de un recorte de `diario_sucursal`.

    REGLA ÚNICA DE TASAS DE TODA LA APP (D1.1, opción B): numerador y denominador emparejados,
    sin los días con dato de ventas inválido:

        tasa = sum(quejas en días válidos) / sum(pedidos_validos) * 1000

    Los conteos de quejas (no las tasas) sí incluyen las quejas de días inválidos.
    `diario` es cualquier filtro de `Datos.diario_sucursal` (por sucursal, región, semana, ...).

    Devuelve {"tasa", "quejas", "pedidos", "quejas_excluidas", "dias_excluidos"}:
    - tasa: float, o None si no hay pedidos válidos en el recorte (una sucursal sin quejas da 0.0).
    - quejas: quejas usadas en el numerador (días válidos).
    - pedidos: suma de pedidos válidos.
    - quejas_excluidas / dias_excluidos: quejas y filas (fecha, sucursal) dejadas afuera.
    """
    invalido = diario["dato_invalido"].astype(bool)
    validos = diario.loc[~invalido]
    quejas = int(validos["quejas"].sum())
    pedidos = int(validos["pedidos_validos"].fillna(0).sum())
    return {
        "tasa": (quejas / pedidos * 1000) if pedidos > 0 else None,
        "quejas": quejas,
        "pedidos": pedidos,
        "quejas_excluidas": int(diario.loc[invalido, "quejas"].sum()),
        "dias_excluidos": int(invalido.sum()),
    }


def reporte_calidad(datos: Datos) -> dict:
    """Resumen (serializable a JSON) de lo encontrado y tratado en los datos."""
    meta = getattr(datos, "_calidad", {})
    crudas = meta.get("filas_crudas", {})
    limpias = {"sucursales": len(datos.sucursales), "ventas": len(datos.ventas), "quejas": len(datos.quejas),
               "capacitaciones": len(datos.capacitaciones), "diario_sucursal": len(datos.diario_sucursal)}
    inval = datos.ventas.loc[datos.ventas["dato_invalido"]]
    q_inv = datos.quejas.loc[datos.quejas["en_dia_invalido"]]
    nuevas = datos.sucursales.loc[datos.sucursales["es_nueva"]]
    return {
        "periodo": {"inicio": FECHA_INICIO.isoformat(), "fin": FECHA_REFERENCIA.isoformat(), "dias": 84},
        "filas_crudas": {k: int(v) for k, v in crudas.items()},
        "filas_limpias": {k: int(v) for k, v in limpias.items()},
        "duplicados_quejas": meta.get("duplicados_quejas"),
        "filas_invalidas_ventas": [
            {"fecha": f"{r.fecha:%Y-%m-%d}", "sucursal_id": str(r.sucursal_id), "pedidos": int(r.pedidos)}
            for r in inval.itertuples()
        ],
        "quejas_en_dias_invalidos": {
            "total": int(len(q_inv)),
            "queja_ids": [str(q) for q in q_inv["queja_id"]],
        },
        "sucursales_nuevas": [
            {"sucursal_id": str(r.sucursal_id), "fecha_apertura": f"{r.fecha_apertura:%Y-%m-%d}",
             "dias_operados": int(r.dias_operados)}
            for r in nuevas.itertuples()
        ],
        "avisos": list(datos.avisos),
        "segundos_carga": meta.get("segundos_carga"),
    }

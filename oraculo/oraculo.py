"""Oráculo independiente: recalcula las respuestas esperadas con pandas puro sobre los CSV crudos.

Prohibido importar el paquete del agente: así no se valida el código contra sí mismo.
Toda la limpieza se hace acá, a partir de los CSV de ``datos/``.

Reglas de negocio aplicadas:

- Periodo 2026-01-05 (lunes) a 2026-03-29 (domingo); semanas de lunes a domingo.
- Quejas deduplicadas por ``queja_id``.
- Pedidos inválidos si ``pedidos < 0`` o ``pedidos >= 99999``: esa fila no suma pedidos.
- Tasa = sum(quejas) / sum(pedidos) * 1000 (ratio de sumas), emparejada: las quejas de un
  día-sucursal con pedido inválido no entran al numerador (sí cuentan en los conteos).
- Rankings por tasa: se excluyen sucursales con menos de 28 días abiertas al final de la
  ventana; desempate por ``sucursal_id``.
- Sucursal nueva: ``fecha_apertura`` dentro del periodo.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

INICIO = pd.Timestamp("2026-01-05")
FIN = pd.Timestamp("2026-03-29")
FECHA_REFERENCIA = FIN
FECHA_CAPACITACION = pd.Timestamp("2026-02-09")
DIAS_MINIMOS_RANKING = 28
PEDIDO_CENTINELA = 99999

DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

_RAIZ = Path(__file__).resolve().parent.parent


def _dir_datos(dir: str | os.PathLike | None) -> Path:
    """Resuelve la carpeta de datos: argumento, variable DATOS_DIR o ``<raíz>/datos``."""
    if dir is not None:
        return Path(dir)
    if os.environ.get("DATOS_DIR"):
        return Path(os.environ["DATOS_DIR"])
    return _RAIZ / "datos"


@lru_cache(maxsize=8)
def _cargar_cache(ruta: str) -> dict[str, pd.DataFrame]:
    base = Path(ruta)
    suc = pd.read_csv(base / "sucursales.csv", dtype=str)
    suc["fecha_apertura"] = pd.to_datetime(suc["fecha_apertura"])
    suc["region"] = suc["region"].str.strip()

    quejas = pd.read_csv(base / "quejas.csv", dtype=str)
    quejas = quejas.drop_duplicates(subset="queja_id", keep="first").reset_index(drop=True)
    quejas["fecha"] = pd.to_datetime(quejas["fecha"])
    for col in ("categoria", "canal", "sucursal_id"):
        quejas[col] = quejas[col].str.strip()

    ventas = pd.read_csv(base / "ventas_diarias.csv", dtype={"sucursal_id": str})
    ventas["fecha"] = pd.to_datetime(ventas["fecha"])
    ventas["pedidos"] = pd.to_numeric(ventas["pedidos"], errors="coerce")
    ventas["valido"] = ventas["pedidos"].notna() & (ventas["pedidos"] >= 0) & (ventas["pedidos"] < PEDIDO_CENTINELA)

    cap = pd.read_csv(base / "capacitaciones.csv", dtype=str)
    cap["fecha_capacitacion"] = pd.to_datetime(cap["fecha_capacitacion"])

    # Marca en quejas si el día-sucursal tiene pedido inválido (para la tasa emparejada).
    claves_invalidas = pd.MultiIndex.from_frame(ventas.loc[~ventas["valido"], ["fecha", "sucursal_id"]])
    claves_quejas = pd.MultiIndex.from_frame(quejas[["fecha", "sucursal_id"]])
    quejas["dia_invalido"] = claves_quejas.isin(claves_invalidas)

    region = suc.set_index("sucursal_id")["region"]
    quejas["region"] = quejas["sucursal_id"].map(region)
    ventas["region"] = ventas["sucursal_id"].map(region)
    return {"sucursales": suc, "quejas": quejas, "ventas": ventas, "capacitaciones": cap}


def cargar(dir: str | os.PathLike | None = None) -> dict[str, pd.DataFrame]:
    """Devuelve los cuatro DataFrames ya limpios (copias, se pueden modificar)."""
    datos = _cargar_cache(str(_dir_datos(dir).resolve()))
    return {k: v.copy() for k, v in datos.items()}


def _fecha(x) -> pd.Timestamp | None:
    return None if x is None else pd.Timestamp(x)


def _filtrar(df: pd.DataFrame, region=None, sucursal=None, desde=None, hasta=None) -> pd.DataFrame:
    """Filtra por región, sucursal (id o lista de ids) y rango de fechas inclusivo."""
    m = pd.Series(True, index=df.index)
    if region is not None:
        m &= df["region"] == region
    if sucursal is not None:
        ids = [sucursal] if isinstance(sucursal, str) else list(sucursal)
        m &= df["sucursal_id"].isin(ids)
    if desde is not None:
        m &= df["fecha"] >= _fecha(desde)
    if hasta is not None:
        m &= df["fecha"] <= _fecha(hasta)
    return df[m]


# --- Conteos básicos -------------------------------------------------------------------------

def contar_sucursales(region=None, dir=None) -> int:
    """Cantidad de sucursales (opcionalmente de una región)."""
    suc = cargar(dir)["sucursales"]
    if region is not None:
        suc = suc[suc["region"] == region]
    return int(len(suc))


def contar_quejas(region=None, sucursal=None, desde=None, hasta=None, categoria=None, canal=None, dir=None) -> int:
    """Cantidad de quejas deduplicadas (incluye las de días con pedido inválido)."""
    q = _filtrar(cargar(dir)["quejas"], region, sucursal, desde, hasta)
    if categoria is not None:
        q = q[q["categoria"] == categoria]
    if canal is not None:
        q = q[q["canal"] == canal]
    return int(len(q))


def pedidos_total(region=None, sucursal=None, desde=None, hasta=None, dir=None) -> int:
    """Suma de pedidos válidos."""
    v = _filtrar(cargar(dir)["ventas"], region, sucursal, desde, hasta)
    return int(v.loc[v["valido"], "pedidos"].sum())


def tasa(region=None, sucursal=None, desde=None, hasta=None, dir=None) -> float:
    """Tasa emparejada de quejas por cada 1000 pedidos (ratio de sumas)."""
    d = cargar(dir)
    q = _filtrar(d["quejas"], region, sucursal, desde, hasta)
    v = _filtrar(d["ventas"], region, sucursal, desde, hasta)
    pedidos = v.loc[v["valido"], "pedidos"].sum()
    if pedidos <= 0:
        return float("nan")
    return float((~q["dia_invalido"]).sum() / pedidos * 1000)


def _mas_comun(serie: pd.Series) -> str:
    """Valor más frecuente; empate por orden alfabético."""
    conteo = serie.value_counts()
    maximo = conteo.max()
    return sorted(conteo[conteo == maximo].index)[0]


def categoria_mas_comun(region=None, sucursal=None, desde=None, hasta=None, dir=None) -> str:
    """Categoría de queja más frecuente."""
    return _mas_comun(_filtrar(cargar(dir)["quejas"], region, sucursal, desde, hasta)["categoria"])


def canal_mas_comun(region=None, desde=None, hasta=None, dir=None) -> str:
    """Canal por el que llegan más quejas."""
    return _mas_comun(_filtrar(cargar(dir)["quejas"], region, None, desde, hasta)["canal"])


# --- Sucursales nuevas y rankings ------------------------------------------------------------

def sucursales_nuevas(dir=None) -> list[str]:
    """Sucursales con fecha de apertura dentro del periodo."""
    s = cargar(dir)["sucursales"]
    m = (s["fecha_apertura"] >= INICIO) & (s["fecha_apertura"] <= FIN)
    return sorted(s.loc[m, "sucursal_id"])


def contar_sucursales_nuevas(dir=None) -> int:
    return len(sucursales_nuevas(dir))


def tasas_por_sucursal(desde=None, hasta=None, excluir_nuevas=True, solo_periodo_completo=False,
                       region=None, dir=None) -> pd.DataFrame:
    """Tabla por sucursal con quejas (emparejadas), pedidos válidos y tasa.

    ``excluir_nuevas`` saca las sucursales con menos de 28 días abiertas al final de la
    ventana; ``solo_periodo_completo`` deja solo las abiertas antes del inicio del periodo.
    Ordenada por tasa descendente y desempate por ``sucursal_id``.
    """
    d = cargar(dir)
    desde_ts = _fecha(desde) or INICIO
    hasta_ts = _fecha(hasta) or FIN
    suc = d["sucursales"]
    if region is not None:
        suc = suc[suc["region"] == region]
    if excluir_nuevas:
        suc = suc[(hasta_ts - suc["fecha_apertura"]).dt.days >= DIAS_MINIMOS_RANKING]
    if solo_periodo_completo:
        suc = suc[suc["fecha_apertura"] <= INICIO]
    ids = list(suc["sucursal_id"])
    q = _filtrar(d["quejas"], None, ids, desde_ts, hasta_ts)
    v = _filtrar(d["ventas"], None, ids, desde_ts, hasta_ts)
    quejas = q[~q["dia_invalido"]].groupby("sucursal_id").size()
    pedidos = v[v["valido"]].groupby("sucursal_id")["pedidos"].sum()
    t = pd.DataFrame({"sucursal_id": ids}).set_index("sucursal_id")
    t["quejas"] = quejas.reindex(t.index).fillna(0).astype(int)
    t["pedidos"] = pedidos.reindex(t.index).fillna(0).astype(int)
    t = t[t["pedidos"] > 0]
    t["tasa"] = t["quejas"] / t["pedidos"] * 1000
    t = t.reset_index().sort_values(["tasa", "sucursal_id"], ascending=[False, True], kind="mergesort")
    return t.reset_index(drop=True)


def top_tasa(n=3, desde=None, hasta=None, excluir_nuevas=True, solo_periodo_completo=False,
             region=None, dir=None) -> list[str]:
    """Las ``n`` sucursales con mayor tasa de quejas en la ventana."""
    t = tasas_por_sucursal(desde, hasta, excluir_nuevas, solo_periodo_completo, region, dir)
    return list(t["sucursal_id"].head(n))


def pedidos_por_sucursal(desde=None, hasta=None, region=None, dir=None) -> pd.Series:
    """Pedidos válidos por sucursal, orden descendente con desempate por id."""
    v = _filtrar(cargar(dir)["ventas"], region, None, desde, hasta)
    s = v[v["valido"]].groupby("sucursal_id")["pedidos"].sum()
    df = s.rename("pedidos").reset_index().sort_values(["pedidos", "sucursal_id"], ascending=[False, True])
    return df.set_index("sucursal_id")["pedidos"]


def top_pedidos_semana(lunes, n=1, region=None, dir=None):
    """Sucursal (o top ``n``) con más pedidos válidos en la semana lunes–domingo que empieza en ``lunes``."""
    ini = pd.Timestamp(lunes)
    if ini.dayofweek != 0:
        raise ValueError(f"{ini.date()} no es lunes")
    s = pedidos_por_sucursal(ini, ini + pd.Timedelta(days=6), region, dir)
    top = list(s.index[:n])
    return top[0] if n == 1 else top


def pedidos_promedio_por_dia_semana(dir=None) -> pd.Series:
    """Promedio de pedidos por fila válida (día-sucursal) según día de la semana."""
    v = cargar(dir)["ventas"]
    v = v[v["valido"]]
    prom = v.groupby(v["fecha"].dt.dayofweek)["pedidos"].mean()
    prom.index = [DIAS_SEMANA[i] for i in prom.index]
    return prom


def dia_semana_mas_pedidos_promedio(dir=None) -> str:
    """Día de la semana con más pedidos en promedio."""
    return str(pedidos_promedio_por_dia_semana(dir).idxmax())


# --- Capacitación ----------------------------------------------------------------------------

def capacitacion_resumen(dir=None) -> dict:
    """Cantidad de sucursales capacitadas, fechas y métodos de asignación."""
    c = cargar(dir)["capacitaciones"]
    fechas = sorted({f.strftime("%Y-%m-%d") for f in c["fecha_capacitacion"]})
    return {
        "n": int(c["sucursal_id"].nunique()),
        "fechas": fechas,
        "fecha": fechas[0] if len(fechas) == 1 else None,
        "por_metodo": {k: int(v) for k, v in c["metodo_asignacion"].value_counts().sort_index().items()},
    }


def grupos_did_norte(dir=None) -> tuple[list[str], list[str]]:
    """(tratado, control) del Norte: capacitadas aleatorias vs. no capacitadas, periodo completo."""
    d = cargar(dir)
    suc = d["sucursales"]
    suc = suc[(suc["region"] == "Norte") & (suc["fecha_apertura"] <= INICIO)]
    cap = d["capacitaciones"]
    aleatorias = set(cap.loc[cap["metodo_asignacion"] == "aleatorio", "sucursal_id"])
    capacitadas = set(cap["sucursal_id"])
    tratado = sorted(s for s in suc["sucursal_id"] if s in aleatorias)
    control = sorted(s for s in suc["sucursal_id"] if s not in capacitadas)
    return tratado, control


def _ventanas_did():
    antes = (INICIO, FECHA_CAPACITACION - pd.Timedelta(days=1))
    despues = (FECHA_CAPACITACION, FIN)
    return antes, despues


def did_norte(dir=None) -> dict:
    """Diferencia en diferencias de la tasa en el Norte (ratio de sumas por grupo)."""
    tratado, control = grupos_did_norte(dir)
    antes, despues = _ventanas_did()
    r = {
        "tratado_antes": tasa(sucursal=tratado, desde=antes[0], hasta=antes[1], dir=dir),
        "tratado_despues": tasa(sucursal=tratado, desde=despues[0], hasta=despues[1], dir=dir),
        "control_antes": tasa(sucursal=control, desde=antes[0], hasta=antes[1], dir=dir),
        "control_despues": tasa(sucursal=control, desde=despues[0], hasta=despues[1], dir=dir),
    }
    r["did"] = (r["tratado_despues"] - r["tratado_antes"]) - (r["control_despues"] - r["control_antes"])
    r["n_tratado"], r["n_control"] = len(tratado), len(control)
    return r


def did_norte_promedio_sucursal(dir=None) -> float:
    """Variante del DiD: promedio simple de las tasas por sucursal en cada grupo y ventana."""
    tratado, control = grupos_did_norte(dir)
    antes, despues = _ventanas_did()

    def prom(ids, ventana):
        return sum(tasa(sucursal=s, desde=ventana[0], hasta=ventana[1], dir=dir) for s in ids) / len(ids)

    return (prom(tratado, despues) - prom(tratado, antes)) - (prom(control, despues) - prom(control, antes))


# --- Resumen de sucursal (P10) ---------------------------------------------------------------

def comentario_base(texto: str) -> str:
    """Primera oración del comentario (sin los agregados genéricos como «Vine con mi familia.»)."""
    t = str(texto).strip()
    i = t.find(".")
    return t if i == -1 else t[: i + 1]


def resumen_sucursal(sucursal_id: str, n_comentarios=2, dir=None) -> dict:
    """Quejas totales, % por categoría y comentarios base más frecuentes de una sucursal."""
    q = _filtrar(cargar(dir)["quejas"], sucursal=sucursal_id)
    total = len(q)
    pct = (q["categoria"].value_counts() / total * 100).round(1) if total else pd.Series(dtype=float)
    base = q["comentario"].map(comentario_base).value_counts()
    base = pd.DataFrame({"comentario": list(base.index), "n": list(base.values)})
    base = base.sort_values(["n", "comentario"], ascending=[False, True])
    return {
        "sucursal_id": sucursal_id,
        "quejas": int(total),
        "pct_por_categoria": {k: float(v) for k, v in pct.items()},
        "categoria_principal": _mas_comun(q["categoria"]) if total else None,
        "comentarios_frecuentes": list(base["comentario"].head(n_comentarios)),
        "conteo_comentarios": {r.comentario: int(r.n) for r in base.head(10).itertuples()},
    }


# --- Respuestas esperadas --------------------------------------------------------------------

def respuestas_esperadas(dir=None) -> dict:
    """Respuestas esperadas de P01–P09 y A01–A06 calculadas desde los datos."""
    cap = capacitacion_resumen(dir)
    return {
        "P01": contar_sucursales("Costa", dir=dir),
        "P02": contar_quejas("Sur", desde="2026-02-01", hasta="2026-02-28", dir=dir),
        "P03": round(tasa("Centro", dir=dir), 2),
        "P04": categoria_mas_comun("Norte", dir=dir),
        "P05": f"{cap['n']} sucursales, el {cap['fecha']}",
        "P06": top_tasa(3, "2026-02-01", "2026-02-28", dir=dir),
        "P07": top_pedidos_semana("2026-03-02", dir=dir),
        "P08": canal_mas_comun(dir=dir),
        "P09": contar_sucursales_nuevas(dir),
        "A01": pedidos_total("Costa", dir=dir),
        "A02": top_pedidos_semana("2026-01-05", dir=dir),
        "A03": dia_semana_mas_pedidos_promedio(dir),
        "A04": top_tasa(3, INICIO, FIN, solo_periodo_completo=True, dir=dir),
        "A05": round(did_norte(dir)["did"], 2),
        "A06": contar_quejas(dir=dir),
    }

"""Score determinístico de sucursales a visitar (contrato C7).

El LLM nunca decide a quién visitar: esta función ordena las sucursales con una regla fija y
devuelve los componentes del score para que el agente arme el motivo y la evidencia.

Regla (calibrada con los datos al 2026-03-29)
---------------------------------------------
- Ventana reciente: los 28 días que terminan en `fecha_hasta` (default `FECHA_REFERENCIA`).
  Comparación: los 28 días anteriores a la ventana.
- Elegibles: sucursales con >= 28 días abiertas al `fecha_hasta`
  (`fecha_hasta - fecha_apertura >= 28`), filtradas por región. Con `incluir_nuevas=True`
  entran también las nuevas, con una advertencia en su ítem.
- `tasa_4sem`: tasa emparejada (`datos.tasa_emparejada`, regla única) de la sucursal en la
  ventana. `tasa_region_4sem`: tasa emparejada de TODA su región en la ventana (todas las
  sucursales de la región, incluidas las nuevas, sin importar el filtro de elegibilidad).
- `ratio_regional = tasa_4sem / tasa_region_4sem`: criterio de orden (desc; desempate por
  sucursal_id).
- Con `categoria`, todas las tasas usan solo las quejas de esa categoría.
- Umbral `ratio_regional >= 1.5`. Se devuelven las que lo superan (hasta 5); si son menos de
  3 se completa hasta 3 con las siguientes, marcadas `preventiva`. Con `n` se devuelven
  exactamente n (o las elegibles que haya), marcando preventivas las que no superan el umbral.
- Tendencia: `tasa_4sem / tasa_4sem_anterior`; "sube" si > 1.10, "baja" si < 0.90, si no
  "estable" ("sin_datos" si no hay periodo anterior comparable).
- Acción base: `ACCIONES_POR_CATEGORIA[categoria_dominante]` (con `categoria`, la de ESA
  categoría); si la tendencia baja o la sucursal ya fue capacitada, la acción es de seguimiento
  (nunca "capacitar"). Con `categoria`, cada ítem informa además `quejas_categoria_4sem` y
  `pct_categoria_4sem` (% de esa categoría dentro de todas las quejas de la ventana, igual base
  que `pct_categoria_dominante`); `categoria_dominante` queda como dato extra.
- `n` admite como máximo 10 (n > 10 se usa 10 con advertencia). Tamaño de salida: < ~4 KB con
  hasta 5 candidatos y < ~8 KB con 6 a 10 (json.dumps). Si no entra, se omiten campos
  secundarios (`ciudad`, `metodo_asignacion`, `nombre`, en ese orden) en todos los ítems, nunca
  ítems, y se avisa en `advertencias`.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd

from agente.datos import CATEGORIAS, FECHA_INICIO, FECHA_REFERENCIA, REGIONES, tasa_emparejada

if TYPE_CHECKING:  # pragma: no cover
    from agente.datos import Datos

DIAS_VENTANA = 28
DIAS_MINIMOS_ABIERTA = 28
UMBRAL_RATIO = 1.5
MAX_SUPERAN = 5
MIN_RECOMENDADAS = 3
UMBRAL_TENDENCIA = 0.10  # ±10 % vs las 4 semanas anteriores
N_MAX_CANDIDATOS = 10
LIMITE_BYTES_HASTA_5 = 3950  # json.dumps (ensure_ascii=True) con <= 5 candidatos
LIMITE_BYTES_HASTA_10 = 7900  # con 6 a 10 candidatos
CAMPOS_SECUNDARIOS = ("ciudad", "metodo_asignacion", "nombre")  # se omiten en este orden si no entra

PREFIJO_PREVENTIVA = "Atención preventiva: no supera el umbral, pero…"

ACCIONES_POR_CATEGORIA: dict[str, str] = {
    "limpieza": "auditoría de limpieza de baños y salón",
    "demora": "revisar dotación y tiempos de preparación en horas pico",
    "pedido_incorrecto": "revisar el proceso de toma y armado de pedidos",
    "trato": "coaching de atención al cliente al equipo",
    "calidad_producto": "control de calidad de insumos y recetas",
}


def _error(mensaje: str, valores_validos: list) -> dict:
    return {"error": mensaje, "valores_validos": valores_validos}


def _parsear_fecha(valor: Any) -> date | None:
    if valor is None:
        return FECHA_REFERENCIA
    if isinstance(valor, pd.Timestamp):
        return valor.date()
    if isinstance(valor, date):
        return valor if not hasattr(valor, "hour") else valor.date()  # datetime -> date
    try:
        return date.fromisoformat(str(valor).strip()[:10])
    except ValueError:
        return None


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(float(x), nd)


def _tasa(diario: pd.DataFrame, categoria: str | None) -> dict:
    """Tasa emparejada; con categoría, el numerador son solo las quejas de esa categoría."""
    if categoria is not None:
        diario = diario.assign(quejas=diario[f"quejas_{categoria}"])
    return tasa_emparejada(diario)


def _tendencia(actual: float | None, anterior: float | None) -> tuple[float | None, str]:
    if actual is None or anterior is None or anterior == 0:
        return None, "sin_datos"
    ratio = actual / anterior
    if ratio > 1 + UMBRAL_TENDENCIA:
        return ratio, "sube"
    if ratio < 1 - UMBRAL_TENDENCIA:
        return ratio, "baja"
    return ratio, "estable"


def _accion(cat_dom: str | None, tendencia: str, capacitada: bool, fecha_cap: str | None) -> str:
    base = ACCIONES_POR_CATEGORIA.get(cat_dom or "", "revisión general de la operación")
    if tendencia == "baja" or capacitada:
        motivos = []
        if tendencia == "baja":
            motivos.append("tasa en baja")
        if capacitada:
            motivos.append(f"ya fue capacitada el {fecha_cap}")
        return f"seguimiento: {base}; " + "; ".join(motivos)
    return base


def _tamano(salida: dict) -> int:
    return len(json.dumps(salida))


def _compactar(salida: dict) -> dict:
    """Omite campos secundarios de todos los candidatos hasta que la salida entre en el límite."""
    candidatos = salida["resultado"]["candidatos"]
    limite = LIMITE_BYTES_HASTA_5 if len(candidatos) <= 5 else LIMITE_BYTES_HASTA_10
    omitidos = []
    for campo in CAMPOS_SECUNDARIOS:
        if _tamano(salida) + 130 <= limite:  # margen para la advertencia de abajo
            break
        for c in candidatos:
            c.pop(campo, None)
        omitidos.append(campo)
    if omitidos:
        salida["advertencias"].append(f"por tamaño se omitieron campos secundarios de cada candidato: "
                                      f"{', '.join(omitidos)}")
    return salida


def candidatos_visita(
    datos: "Datos",
    region: str | None = None,
    categoria: str | None = None,
    n: int | None = None,
    fecha_hasta: date | str | None = None,
    incluir_nuevas: bool = False,
) -> dict:
    """Sucursales a visitar ordenadas por `ratio_regional` (formato C2).

    La ventana reciente es `periodo`; tasas en quejas cada 1.000 pedidos. `n` máximo 10.

    Devuelve `{"resultado": {"parametros": {...}, "candidatos": [...]}, "filtros_aplicados",
    "periodo", "n_registros", "advertencias"}` o `{"error", "valores_validos"}`.
    """
    return _candidatos_visita(datos, region=region, categoria=categoria, n=n, fecha_hasta=fecha_hasta,
                              incluir_nuevas=incluir_nuevas, n_maximo=N_MAX_CANDIDATOS)


def _candidatos_visita(
    datos: "Datos",
    region: str | None = None,
    categoria: str | None = None,
    n: int | None = None,
    fecha_hasta: date | str | None = None,
    incluir_nuevas: bool = False,
    n_maximo: int | None = N_MAX_CANDIDATOS,
) -> dict:
    """Implementación de `candidatos_visita`. `n_maximo=None` (solo tests/análisis) no limita `n`
    ni compacta la salida por tamaño."""
    # --- validación
    if region is not None and region not in REGIONES:
        return _error(f"región inexistente: {region!r}", list(REGIONES))
    if categoria is not None and categoria not in CATEGORIAS:
        return _error(f"categoría inexistente: {categoria!r}", list(CATEGORIAS))
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            return _error(f"n debe ser un entero >= 1 (recibido {n!r})", ["entero >= 1"])
    hasta = _parsear_fecha(fecha_hasta)
    rango = [FECHA_INICIO.isoformat(), FECHA_REFERENCIA.isoformat()]
    if hasta is None:
        return _error(f"fecha_hasta inválida: {fecha_hasta!r} (formato AAAA-MM-DD)", rango)
    if not (FECHA_INICIO <= hasta <= FECHA_REFERENCIA):
        return _error(f"fecha_hasta fuera del periodo con datos: {hasta.isoformat()}", rango)

    advertencias: list[str] = []
    if n is not None and n_maximo is not None and n > n_maximo:
        advertencias.append(f"n se limitó a {n_maximo} (se pidieron {n}) para mantener la salida chica")
        n = n_maximo
    desde = hasta - timedelta(days=DIAS_VENTANA - 1)
    ant_hasta = desde - timedelta(days=1)
    ant_desde = ant_hasta - timedelta(days=DIAS_VENTANA - 1)
    if desde < FECHA_INICIO:
        advertencias.append(f"la ventana empieza antes de los datos ({FECHA_INICIO.isoformat()}): "
                            "se usan solo los días disponibles")
    if ant_desde < FECHA_INICIO:
        advertencias.append("el periodo de comparación está incompleto o vacío: tendencia poco confiable")

    diario = datos.diario_sucursal
    f = diario["fecha"]
    ventana = diario.loc[(f >= pd.Timestamp(desde)) & (f <= pd.Timestamp(hasta))]
    anterior = diario.loc[(f >= pd.Timestamp(ant_desde)) & (f <= pd.Timestamp(ant_hasta))]

    # --- elegibles
    suc = datos.sucursales
    dias_abierta = (pd.Timestamp(hasta) - suc["fecha_apertura"]).dt.days
    abierta = dias_abierta >= 0
    madura = dias_abierta >= DIAS_MINIMOS_ABIERTA
    mascara = (madura | abierta) if incluir_nuevas else madura
    if region is not None:
        mascara &= suc["region"] == region
    elegibles = suc.loc[mascara]

    tasas_region = {r: _tasa(ventana.loc[ventana["region"] == r], categoria)["tasa"] for r in REGIONES}
    cols_cat = [f"quejas_{c}" for c in CATEGORIAS]
    v_por_suc = dict(tuple(ventana.groupby("sucursal_id")))
    a_por_suc = dict(tuple(anterior.groupby("sucursal_id")))
    vacio = ventana.iloc[0:0]

    filas = []
    for s in elegibles.itertuples(index=False):
        dv = v_por_suc.get(s.sucursal_id, vacio)
        da = a_por_suc.get(s.sucursal_id, vacio)
        tv = _tasa(dv, categoria)
        ta = _tasa(da, categoria)
        t_reg = tasas_region.get(s.region)
        if tv["tasa"] is None or not t_reg:
            continue
        ratio = tv["tasa"] / t_reg
        filas.append((s, dv, tv, ta, t_reg, ratio))
    filas.sort(key=lambda x: (-round(x[5], 9), x[0].sucursal_id))

    n_supera = sum(1 for x in filas if x[5] >= UMBRAL_RATIO)
    if n is not None:
        cantidad = min(n, len(filas))
        if n > len(filas):
            advertencias.append(f"se pidieron {n} y hay solo {len(filas)} sucursales elegibles")
    else:
        cantidad = min(len(filas), max(min(n_supera, MAX_SUPERAN), MIN_RECOMENDADAS))

    candidatos = []
    for s, dv, tv, ta, t_reg, ratio in filas[:cantidad]:
        # categoría dominante dentro de las quejas (todas) de la ventana
        conteos = {c: int(dv[f"quejas_{c}"].sum()) for c in CATEGORIAS} if len(dv) else {}
        total_q = sum(conteos.values())
        if total_q:
            cat_dom = max(CATEGORIAS, key=lambda c: (conteos[c], -CATEGORIAS.index(c)))
            pct_dom = round(conteos[cat_dom] / total_q * 100, 1)
        else:
            cat_dom, pct_dom = None, None
        if categoria is not None:
            q_cat = conteos.get(categoria, 0)
            pct_cat = round(q_cat / total_q * 100, 1) if total_q else None
        ratio_tend, etiqueta = _tendencia(tv["tasa"], ta["tasa"])
        capacitada = bool(s.capacitada)
        fecha_cap = None if pd.isna(s.fecha_capacitacion) else f"{s.fecha_capacitacion:%Y-%m-%d}"
        metodo = None if pd.isna(s.metodo_asignacion) else str(s.metodo_asignacion)
        supera = ratio >= UMBRAL_RATIO
        adv_item = []
        dias = int((pd.Timestamp(hasta) - s.fecha_apertura).days)
        if dias < DIAS_MINIMOS_ABIERTA:
            adv_item.append(f"abrió el {s.fecha_apertura:%Y-%m-%d}: solo {dias + 1} días de datos, "
                            "tasa poco confiable")
        if tv["dias_excluidos"]:
            adv_item.append(f"{tv['dias_excluidos']} día(s) con ventas inválidas excluidos de la tasa")
        item = {
            "sucursal_id": str(s.sucursal_id),
            "nombre": str(s.nombre),
            "region": str(s.region),
            "ciudad": str(s.ciudad),
            "tasa_4sem": _r(tv["tasa"], 2),
            "tasa_region_4sem": _r(t_reg, 2),
            "ratio_regional": _r(ratio, 2),
            "quejas_4sem": tv["quejas"],
            "pedidos_4sem": tv["pedidos"],
            "tasa_4sem_anterior": _r(ta["tasa"], 2),
            "tendencia": {"ratio": _r(ratio_tend, 2), "etiqueta": etiqueta},
            "categoria_dominante": cat_dom,
            "pct_categoria_dominante": pct_dom,
            **({"quejas_categoria_4sem": q_cat, "pct_categoria_4sem": pct_cat} if categoria is not None else {}),
            "capacitada": capacitada,
            "fecha_capacitacion": fecha_cap,
            "metodo_asignacion": metodo,
            "supera_umbral": supera,
            "preventiva": not supera,
            "accion_sugerida_base": _accion(categoria if categoria is not None else cat_dom,
                                            etiqueta, capacitada, fecha_cap),
            "advertencias": adv_item,
        }
        candidatos.append(item)

    if any(c["preventiva"] for c in candidatos):
        advertencias.append(f"los ítems con preventiva=true no superan el umbral; su motivo debe "
                            f"empezar con: {PREFIJO_PREVENTIVA}")
    if not filas:
        advertencias.append("no hay sucursales elegibles con los filtros dados")

    salida = {
        "resultado": {
            "parametros": {
                "ventana": f"{max(desde, FECHA_INICIO).isoformat()}..{hasta.isoformat()}",
                "ventana_anterior": f"{ant_desde.isoformat()}..{ant_hasta.isoformat()}",
                "criterio": "ratio_regional = tasa_4sem / tasa_region_4sem",
                "umbral_ratio": UMBRAL_RATIO,
                "dias_minimos_abierta": DIAS_MINIMOS_ABIERTA,
                "n_elegibles": len(filas),
                "n_supera_umbral": n_supera,
            },
            "candidatos": candidatos,
        },
        "filtros_aplicados": {"region": region, "categoria": categoria, "n": n,
                              "fecha_hasta": hasta.isoformat(), "incluir_nuevas": bool(incluir_nuevas)},
        "periodo": {"desde": max(desde, FECHA_INICIO).isoformat(), "hasta": hasta.isoformat()},
        "n_registros": len(candidatos),
        "advertencias": advertencias,
    }
    return _compactar(salida) if n_maximo is not None else salida

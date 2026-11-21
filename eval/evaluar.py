"""Script de evaluación (contrato C6).

Uso:
    python -m eval.evaluar <ruta.jsonl> [<ruta2.jsonl> ...] [--repeticiones 3] [--salida reportes/]
                           [--paralelo 1] [--sin-juez] [--umbral 0.0]

Cada pregunta se corre `repeticiones` veces. Una pregunta cuenta como acertada "por mayoría" si
acierta en más de la mitad de las corridas, y "estricta" si acierta en todas. Una excepción de
`responder()` cuenta como fallo de esa corrida (con el error registrado) y el eval sigue.

Códigos de salida: 0 = ok; 1 = error de configuración (clave faltante, JSONL inválido);
2 = la tasa de aciertos por mayoría quedó bajo `--umbral`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional

from agente.esquemas import Respuesta
from eval.comparadores import TIPOS_VALIDOS, comparar

CAMPOS_OBLIGATORIOS = ("id", "pregunta", "respuesta_esperada", "tipo")


class ErrorJSONL(ValueError):
    """JSONL de preguntas inválido; el mensaje incluye archivo y número de línea."""


# ---------------------------------------------------------------- carga y validación


def _validar_item(item: Any) -> Optional[str]:
    """Devuelve un mensaje de error o None si el ítem es válido."""
    if not isinstance(item, dict):
        return "cada línea debe ser un objeto JSON"
    faltan = [c for c in CAMPOS_OBLIGATORIOS if c not in item]
    if faltan:
        return f"faltan campos obligatorios: {', '.join(faltan)}"
    if not isinstance(item["id"], str) or not item["id"].strip():
        return "`id` debe ser un string no vacío"
    if not isinstance(item["pregunta"], str) or not item["pregunta"].strip():
        return "`pregunta` debe ser un string no vacío"
    tipo = item["tipo"]
    if tipo not in TIPOS_VALIDOS:
        return f"`tipo` inválido: {tipo!r} (válidos: {', '.join(TIPOS_VALIDOS)})"
    esp = item["respuesta_esperada"]
    es_num = isinstance(esp, (int, float)) and not isinstance(esp, bool)
    if tipo == "numero" and not es_num:
        return "`respuesta_esperada` debe ser un número para tipo numero"
    if tipo == "lista" and not (isinstance(esp, list) and all(isinstance(x, str) for x in esp)):
        return "`respuesta_esperada` debe ser una lista de strings para tipo lista"
    if tipo == "texto" and not isinstance(esp, str):
        return "`respuesta_esperada` debe ser un string para tipo texto"
    tol = item.get("tolerancia")
    if tol is not None and (isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol < 0):
        return "`tolerancia` debe ser un número ≥ 0 o null"
    return None


def cargar_preguntas(ruta: str | Path, ids_vistos: Optional[dict[str, str]] = None) -> list[dict]:
    """Lee y valida un JSONL de preguntas. Agrega `_archivo` (nombre del archivo) a cada ítem.

    `ids_vistos` (id → "archivo:línea") permite exigir ids únicos entre varios archivos.
    Lanza `ErrorJSONL` con el número de línea si algo no cumple el formato.
    """
    ruta = Path(ruta)
    ids_vistos = {} if ids_vistos is None else ids_vistos
    preguntas = []
    try:
        lineas = ruta.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise ErrorJSONL(f"{ruta}: no se pudo leer ({e})") from e
    for n, linea in enumerate(lineas, start=1):
        if not linea.strip():
            continue
        try:
            item = json.loads(linea)
        except json.JSONDecodeError as e:
            raise ErrorJSONL(f"{ruta}:{n}: JSON inválido ({e.msg})") from e
        error = _validar_item(item)
        if error:
            raise ErrorJSONL(f"{ruta}:{n}: {error}")
        if item["id"] in ids_vistos:
            raise ErrorJSONL(f"{ruta}:{n}: id duplicado {item['id']!r} (ya estaba en {ids_vistos[item['id']]})")
        ids_vistos[item["id"]] = f"{ruta}:{n}"
        item.setdefault("tolerancia", None)
        item["_archivo"] = ruta.name
        preguntas.append(item)
    if not preguntas:
        raise ErrorJSONL(f"{ruta}: el archivo no tiene preguntas")
    return preguntas


# ---------------------------------------------------------------- ejecución


def es_rate_limit(error: BaseException) -> bool:
    """Heurística: el nombre de la clase o el mensaje mencionan "rate" o "429"."""
    texto = f"{type(error).__name__} {error}".lower()
    return "rate" in texto or "429" in texto


def _llamar_con_reintentos(responder, pregunta: str, reintentos: int, espera_base_s: float) -> Respuesta:
    intento = 0
    while True:
        try:
            return responder(pregunta)
        except Exception as e:
            if intento < reintentos and es_rate_limit(e):
                time.sleep(espera_base_s * (2**intento))
                intento += 1
                continue
            raise


def _a_jsonable(x: Any) -> Any:
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)


def _costo_agente(traza: dict) -> Optional[float]:
    costo = traza.get("costo_usd")
    if costo is not None:
        return float(costo)
    uso = traza.get("uso") or {}
    modelo = traza.get("modelo")
    if modelo and uso:
        from agente.config import costo_usd

        return costo_usd(modelo, int(uso.get("tokens_entrada", 0)), int(uso.get("tokens_salida", 0)))
    return None


def _una_corrida(item: dict, k: int, responder, usar_juez: bool, juez, reintentos: int, espera_base_s: float) -> dict:
    """Corre y compara una repetición de una pregunta. Nunca lanza."""
    inicio = time.perf_counter()
    corrida: dict[str, Any] = {"corrida": k + 1, "error": None}
    try:
        respuesta = _llamar_con_reintentos(responder, item["pregunta"], reintentos, espera_base_s)
    except Exception as e:
        corrida.update(
            ok=False, obtenida=None, metodo="error", motivo="excepción en responder()",
            error=f"{type(e).__name__}: {e}", latencia_s=time.perf_counter() - inicio,
            tokens_entrada=0, tokens_salida=0, costo_usd=None, juez={}, texto=None,
        )
        return corrida
    medida = time.perf_counter() - inicio
    traza = respuesta.traza or {}
    uso = traza.get("uso") or {}
    try:
        res = comparar(respuesta, item, usar_juez=usar_juez, juez=juez)
        ok, obtenida, metodo, motivo, uso_juez = res.ok, res.obtenida, res.metodo, res.motivo, res.uso_juez
    except Exception as e:  # defensivo: un bug del comparador no corta el eval
        ok, obtenida, metodo, motivo, uso_juez = False, None, "error", f"error al comparar: {e}", {}
        corrida["error"] = f"{type(e).__name__}: {e}"
    corrida.update(
        ok=ok, obtenida=_a_jsonable(obtenida), metodo=metodo, motivo=motivo,
        latencia_s=float(traza.get("latencia_s", medida)),
        tokens_entrada=int(uso.get("tokens_entrada", 0) or 0),
        tokens_salida=int(uso.get("tokens_salida", 0) or 0),
        costo_usd=_costo_agente(traza), juez=uso_juez or {},
        texto=respuesta.texto, valor=_a_jsonable(respuesta.valor),
        advertencias=traza.get("advertencias") or [],
    )
    return corrida


def _sumar_costos(valores: list[Optional[float]]) -> Optional[float]:
    conocidos = [v for v in valores if v is not None]
    return sum(conocidos) if conocidos else None


def _resultado_pregunta(item: dict, corridas: list[dict]) -> dict:
    corridas = sorted(corridas, key=lambda c: c["corrida"])
    oks = [c["ok"] for c in corridas]
    n = len(oks)
    aciertos = sum(oks)
    latencias = [c["latencia_s"] for c in corridas]
    errores = [c["error"] for c in corridas if c["error"]]
    juez_tokens_e = sum(c["juez"].get("tokens_entrada", 0) for c in corridas)
    juez_tokens_s = sum(c["juez"].get("tokens_salida", 0) for c in corridas)
    return {
        "id": item["id"],
        "archivo": item.get("_archivo"),
        "tipo": item["tipo"],
        "pregunta": item["pregunta"],
        "esperada": item["respuesta_esperada"],
        "tolerancia": item.get("tolerancia"),
        "obtenidas": [c["obtenida"] for c in corridas],
        "ok_por_corrida": oks,
        "aciertos": aciertos,
        "n": n,
        "aciertos_n": f"{aciertos}/{n}",
        "ok_mayoria": aciertos > n / 2,
        "ok_estricto": n > 0 and aciertos == n,
        "consistente": len(set(oks)) <= 1,
        "metodos": [c["metodo"] for c in corridas],
        "latencias_s": latencias,
        "latencia_media_s": sum(latencias) / n if n else None,
        "error": errores[0] if errores else None,
        "errores": errores,
        "tokens_entrada": sum(c["tokens_entrada"] for c in corridas),
        "tokens_salida": sum(c["tokens_salida"] for c in corridas),
        "costo_usd": _sumar_costos([c["costo_usd"] for c in corridas]),
        "juez_tokens_entrada": juez_tokens_e,
        "juez_tokens_salida": juez_tokens_s,
        "juez_costo_usd": _sumar_costos([c["juez"].get("costo_usd") for c in corridas if c["juez"]]),
        "corridas": corridas,
    }


def percentil(valores: list[float], p: float) -> Optional[float]:
    """Percentil con interpolación lineal (como `numpy.percentile` por defecto)."""
    if not valores:
        return None
    v = sorted(valores)
    pos = (len(v) - 1) * p / 100
    bajo, alto = math.floor(pos), math.ceil(pos)
    return v[bajo] + (v[alto] - v[bajo]) * (pos - bajo)


def _agrupar(resultados: list[dict], clave: str) -> dict:
    grupos: dict[str, dict] = {}
    for r in resultados:
        g = grupos.setdefault(str(r[clave]), {"preguntas": 0, "aciertos_mayoria": 0, "aciertos_estrictos": 0})
        g["preguntas"] += 1
        g["aciertos_mayoria"] += int(r["ok_mayoria"])
        g["aciertos_estrictos"] += int(r["ok_estricto"])
    for g in grupos.values():
        g["tasa_mayoria"] = g["aciertos_mayoria"] / g["preguntas"]
        g["tasa_estricta"] = g["aciertos_estrictos"] / g["preguntas"]
    return grupos


def resumir(resultados: list[dict]) -> dict:
    """Resumen global: aciertos, consistencia, latencias, tokens y costo."""
    n = len(resultados)
    latencias = [l for r in resultados for l in r["latencias_s"]]
    t_e = sum(r["tokens_entrada"] for r in resultados)
    t_s = sum(r["tokens_salida"] for r in resultados)
    jt_e = sum(r["juez_tokens_entrada"] for r in resultados)
    jt_s = sum(r["juez_tokens_salida"] for r in resultados)
    costo_agente = _sumar_costos([r["costo_usd"] for r in resultados])
    costo_juez = _sumar_costos([r["juez_costo_usd"] for r in resultados])
    costo_total = _sumar_costos([costo_agente, costo_juez])
    consistentes = sum(r["consistente"] for r in resultados)
    mayoria = sum(r["ok_mayoria"] for r in resultados)
    estrictos = sum(r["ok_estricto"] for r in resultados)
    return {
        "preguntas": n,
        "corridas": sum(r["n"] for r in resultados),
        "aciertos_mayoria": mayoria,
        "tasa_mayoria": mayoria / n if n else 0.0,
        "aciertos_estrictos": estrictos,
        "tasa_estricta": estrictos / n if n else 0.0,
        "consistentes": consistentes,
        "tasa_consistencia": consistentes / n if n else 0.0,
        "corridas_con_error": sum(len(r["errores"]) for r in resultados),
        "por_tipo": _agrupar(resultados, "tipo"),
        "por_archivo": _agrupar(resultados, "archivo"),
        "latencia_media_s": sum(latencias) / len(latencias) if latencias else None,
        "latencia_p95_s": percentil(latencias, 95),
        "tokens_agente": {"entrada": t_e, "salida": t_s, "total": t_e + t_s},
        "tokens_juez": {"entrada": jt_e, "salida": jt_s, "total": jt_e + jt_s},
        "tokens_totales": t_e + t_s + jt_e + jt_s,
        "costo_agente_usd": costo_agente,
        "costo_juez_usd": costo_juez,
        "costo_total_usd": costo_total,
        "por_pregunta_promedio": {
            "tokens": (t_e + t_s + jt_e + jt_s) / n if n else None,
            "costo_usd": costo_total / n if (n and costo_total is not None) else None,
        },
    }


def correr_eval(
    preguntas: list[dict],
    responder: Callable[[str], Respuesta],
    repeticiones: int = 3,
    usar_juez: bool = True,
    *,
    juez: Any = None,
    paralelo: int = 1,
    al_terminar_pregunta: Optional[Callable[[dict], None]] = None,
    reintentos: int = 3,
    espera_base_s: float = 2.0,
) -> dict:
    """Corre el eval y devuelve `{"resultados": [...], "resumen": {...}, "config": {...}}`.

    - `juez`: juez inyectable (ver `eval.juez.juzgar`); None = juez real si hace falta.
    - `paralelo`: hilos para correr (pregunta, repetición) en paralelo; 1 = secuencial.
    - `al_terminar_pregunta(resultado)`: callback cuando terminan todas las corridas de una pregunta.
    - `reintentos`/`espera_base_s`: backoff exponencial ante errores de rate limit.
    """
    if repeticiones < 1:
        raise ValueError("repeticiones debe ser ≥ 1")
    inicio = time.perf_counter()
    pendientes = {i: repeticiones for i in range(len(preguntas))}
    corridas: dict[int, list[dict]] = {i: [] for i in range(len(preguntas))}
    resultados: dict[int, dict] = {}
    candado = Lock()

    def tarea(i: int, k: int) -> None:
        c = _una_corrida(preguntas[i], k, responder, usar_juez, juez, reintentos, espera_base_s)
        terminado = None
        with candado:
            corridas[i].append(c)
            pendientes[i] -= 1
            if pendientes[i] == 0:
                terminado = resultados[i] = _resultado_pregunta(preguntas[i], corridas[i])
        if terminado is not None and al_terminar_pregunta is not None:
            al_terminar_pregunta(terminado)

    trabajos = [(i, k) for i in range(len(preguntas)) for k in range(repeticiones)]
    if paralelo <= 1:
        for i, k in trabajos:
            tarea(i, k)
    else:
        with ThreadPoolExecutor(max_workers=paralelo) as ejecutor:
            for f in [ejecutor.submit(tarea, i, k) for i, k in trabajos]:
                f.result()

    lista = [resultados[i] for i in range(len(preguntas))]
    return {
        "resultados": lista,
        "resumen": resumir(lista),
        "config": {
            "repeticiones": repeticiones,
            "usar_juez": usar_juez,
            "paralelo": paralelo,
            "duracion_s": time.perf_counter() - inicio,
            "fecha": datetime.now().isoformat(timespec="seconds"),
        },
    }


# ---------------------------------------------------------------- reportes


def _corto(x: Any, largo: int = 40) -> str:
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    s = " ".join(str(s).split())
    return s if len(s) <= largo else s[: largo - 1] + "…"


def _obtenida_representativa(r: dict) -> Any:
    """La obtenida más frecuente entre las corridas (para la tabla)."""
    vistas = [json.dumps(o, ensure_ascii=False, sort_keys=True) for o in r["obtenidas"]]
    if not vistas:
        return None
    mas = max(set(vistas), key=vistas.count)
    return json.loads(mas)


def filas_tabla(resultado: dict) -> list[list[str]]:
    filas = []
    for r in resultado["resultados"]:
        lat = r["latencia_media_s"]
        filas.append([
            r["id"], r["tipo"], _corto(r["esperada"]), _corto(_obtenida_representativa(r)),
            "sí" if r["ok_mayoria"] else "no", r["aciertos_n"],
            f"{lat:.2f}" if lat is not None else "-",
        ])
    return filas


ENCABEZADO = ["id", "tipo", "esperada", "obtenida", "ok", "aciertos", "latencia_s"]


def formatear_tabla(resultado: dict) -> str:
    """Tabla de texto para la consola."""
    filas = [ENCABEZADO] + filas_tabla(resultado)
    anchos = [max(len(f[i]) for f in filas) for i in range(len(ENCABEZADO))]
    lineas = ["  ".join(c.ljust(a) for c, a in zip(f, anchos)) for f in filas]
    lineas.insert(1, "  ".join("-" * a for a in anchos))
    return "\n".join(lineas)


def _fmt(x: Optional[float], dec: int = 2, pref: str = "") -> str:
    return "-" if x is None else f"{pref}{x:.{dec}f}"


def lineas_resumen(resumen: dict) -> list[str]:
    s = resumen
    out = [
        f"Aciertos por mayoría: {s['aciertos_mayoria']}/{s['preguntas']} ({s['tasa_mayoria']:.0%})",
        f"Aciertos estrictos (N de N): {s['aciertos_estrictos']}/{s['preguntas']} ({s['tasa_estricta']:.0%})",
        f"Consistencia: {s['consistentes']}/{s['preguntas']} ({s['tasa_consistencia']:.0%})",
        f"Corridas con error: {s['corridas_con_error']} de {s['corridas']}",
        f"Latencia media: {_fmt(s['latencia_media_s'])} s · p95: {_fmt(s['latencia_p95_s'])} s",
        f"Tokens: agente {s['tokens_agente']['total']} · juez {s['tokens_juez']['total']} · total {s['tokens_totales']}",
        f"Costo: agente {_fmt(s['costo_agente_usd'], 4, 'US$')} · juez {_fmt(s['costo_juez_usd'], 4, 'US$')}"
        f" · total {_fmt(s['costo_total_usd'], 4, 'US$')}",
        f"Por pregunta (promedio): {_fmt(s['por_pregunta_promedio']['tokens'], 0)} tokens · "
        f"{_fmt(s['por_pregunta_promedio']['costo_usd'], 4, 'US$')}",
    ]
    for nombre, clave in (("Por tipo", "por_tipo"), ("Por archivo", "por_archivo")):
        partes = [
            f"{k}: {g['aciertos_mayoria']}/{g['preguntas']} mayoría, {g['aciertos_estrictos']}/{g['preguntas']} estrictos"
            for k, g in s[clave].items()
        ]
        out.append(f"{nombre}: " + "; ".join(partes))
    return out


def formatear_markdown(resultado: dict) -> str:
    """Reporte en Markdown: tabla + resumen + fallos."""
    def celda(x: str) -> str:
        return x.replace("|", "\\|")

    cfg = resultado.get("config", {})
    md = [
        "# Reporte de evaluación",
        "",
        f"Fecha: {cfg.get('fecha', '-')} · repeticiones: {cfg.get('repeticiones')} · "
        f"juez: {'sí' if cfg.get('usar_juez') else 'no'} · paralelo: {cfg.get('paralelo')}",
        "",
        "| " + " | ".join(ENCABEZADO) + " |",
        "|" + "|".join("---" for _ in ENCABEZADO) + "|",
    ]
    md += ["| " + " | ".join(celda(c) for c in f) + " |" for f in filas_tabla(resultado)]
    md += ["", "## Resumen", ""] + [f"- {l}" for l in lineas_resumen(resultado["resumen"])]
    fallos = [
        (r["id"], c) for r in resultado["resultados"] for c in r["corridas"] if not c["ok"]
    ]
    if fallos:
        md += ["", "## Corridas fallidas", ""]
        for pid, c in fallos:
            detalle = c.get("error") or c.get("motivo") or ""
            md.append(f"- {pid} corrida {c['corrida']} ({c['metodo']}): {_corto(c['obtenida'], 60)} — {detalle}")
    return "\n".join(md) + "\n"


def escribir_reportes(resultado: dict, salida: str | Path) -> dict[str, Path]:
    """Escribe `reporte_<timestamp>.{json,md}` y las copias `reporte.{json,md}` (último)."""
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    contenido_json = json.dumps(resultado, ensure_ascii=False, indent=2, default=str)
    contenido_md = formatear_markdown(resultado)
    rutas = {
        "json": salida / f"reporte_{marca}.json",
        "md": salida / f"reporte_{marca}.md",
        "json_ultimo": salida / "reporte.json",
        "md_ultimo": salida / "reporte.md",
    }
    for clave, ruta in rutas.items():
        ruta.write_text(contenido_json if clave.startswith("json") else contenido_md, encoding="utf-8")
    return rutas


# ---------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m eval.evaluar", description="Evalúa el agente contra JSONL de preguntas.")
    p.add_argument("rutas", nargs="+", help="Uno o más archivos JSONL de preguntas.")
    p.add_argument("--repeticiones", type=int, default=3)
    p.add_argument("--salida", default="reportes/")
    p.add_argument("--paralelo", type=int, default=1)
    p.add_argument("--sin-juez", action="store_true", help="No usar el LLM juez (texto largo falla con 'requiere juez').")
    p.add_argument("--umbral", type=float, default=0.0, help="Tasa mínima de aciertos por mayoría (0 a 1).")
    return p


def main(argv: Optional[list[str]] = None, *, responder: Optional[Callable[[str], Respuesta]] = None, juez: Any = None) -> int:
    """Punto de entrada. `responder` y `juez` son inyectables (tests); si no, se usan los reales."""
    args = _parser().parse_args(argv)
    usar_juez = not args.sin_juez

    try:
        ids: dict[str, str] = {}
        preguntas = [p for ruta in args.rutas for p in cargar_preguntas(ruta, ids)]
    except ErrorJSONL as e:
        print(f"Error en las preguntas: {e}", file=sys.stderr)
        return 1

    from agente.config import FaltaVariableDeEntorno, clave_api, modelo_juez, modelo_llm, variable_clave_para

    try:
        if responder is None:
            clave_api(variable_clave_para(modelo_llm()))
        if usar_juez and juez is None:
            clave_api(variable_clave_para(modelo_juez()))
    except FaltaVariableDeEntorno as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if responder is None:
        from agente.agente import responder as responder_real

        responder = responder_real

    def progreso(r: dict) -> None:
        print(f"  {r['id']}: {r['aciertos_n']} {'ok' if r['ok_mayoria'] else 'FALLA'}", flush=True)

    print(f"Evaluando {len(preguntas)} preguntas × {args.repeticiones} repeticiones…", flush=True)
    resultado = correr_eval(
        preguntas, responder, repeticiones=args.repeticiones, usar_juez=usar_juez,
        juez=juez, paralelo=args.paralelo, al_terminar_pregunta=progreso,
    )
    resultado["config"]["archivos"] = [str(r) for r in args.rutas]
    resultado["config"]["umbral"] = args.umbral

    print()
    print(formatear_tabla(resultado))
    print()
    for linea in lineas_resumen(resultado["resumen"]):
        print(linea)
    rutas = escribir_reportes(resultado, args.salida)
    print(f"\nReportes: {rutas['json']} · {rutas['md']} (y reporte.json / reporte.md)")

    tasa = resultado["resumen"]["tasa_mayoria"]
    if tasa < args.umbral:
        print(f"La tasa de aciertos por mayoría ({tasa:.0%}) quedó bajo el umbral ({args.umbral:.0%}).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

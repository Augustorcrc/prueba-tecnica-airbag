"""Tests del script de evaluación con un agente falso (sin clave ni red)."""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

import pytest

from agente.esquemas import Respuesta
from eval import evaluar as ev
from eval.juez import Veredicto

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
EJEMPLO = EVAL_DIR / "preguntas_ejemplo.jsonl"
PROPIAS = EVAL_DIR / "preguntas_propias.jsonl"


def _todas() -> list[dict]:
    ids: dict = {}
    return ev.cargar_preguntas(EJEMPLO, ids) + ev.cargar_preguntas(PROPIAS, ids)


def _formato_texto(esperada) -> str:
    if isinstance(esperada, list):
        return "Son " + ", ".join(esperada) + "."
    if isinstance(esperada, (int, float)):
        return f"El resultado es {str(esperada).replace('.', ',')}."
    return f"Respuesta: {esperada}"


def agente_perfecto(preguntas: list[dict]):
    """Agente falso que contesta la esperada (solo en texto, para ejercitar el parser)."""
    por_pregunta = {p["pregunta"]: p for p in preguntas}

    def responder(pregunta: str) -> Respuesta:
        p = por_pregunta[pregunta]
        return Respuesta(
            tipo_respuesta="dato", texto=_formato_texto(p["respuesta_esperada"]),
            traza={"latencia_s": 0.5, "uso": {"tokens_entrada": 100, "tokens_salida": 10},
                   "costo_usd": 0.001, "modelo": "anthropic:claude-sonnet-5-5"},
        )

    return responder


def juez_falso(pregunta, esperada, obtenida):
    return Veredicto(correcto=str(esperada)[:20] in obtenida, razon="falso")


def test_ejemplo_y_propias_completo(tmp_path):
    preguntas = _todas()
    assert len(preguntas) == 16
    terminadas = []
    res = ev.correr_eval(preguntas, agente_perfecto(preguntas), repeticiones=2, juez=juez_falso,
                         al_terminar_pregunta=lambda r: terminadas.append(r["id"]))
    s = res["resumen"]
    assert s["preguntas"] == 16 and s["corridas"] == 32
    assert s["aciertos_mayoria"] == 16 and s["aciertos_estrictos"] == 16
    assert s["tasa_consistencia"] == 1.0
    assert s["latencia_media_s"] == pytest.approx(0.5) and s["latencia_p95_s"] == pytest.approx(0.5)
    assert s["tokens_agente"]["total"] == 32 * 110
    assert s["costo_agente_usd"] == pytest.approx(0.032)
    assert set(s["por_archivo"]) == {"preguntas_ejemplo.jsonl", "preguntas_propias.jsonl"}
    assert s["por_tipo"]["lista"]["preguntas"] == 2
    assert sorted(terminadas) == sorted(p["id"] for p in preguntas)
    p10 = next(r for r in res["resultados"] if r["id"] == "P10")
    assert p10["metodos"] == ["texto", "texto"]  # la esperada está literal en el texto

    rutas = ev.escribir_reportes(res, tmp_path)
    datos = json.loads((tmp_path / "reporte.json").read_text(encoding="utf-8"))
    assert datos["resumen"]["aciertos_mayoria"] == 16
    md = (tmp_path / "reporte.md").read_text(encoding="utf-8")
    assert "| P03 | numero |" in md and "Aciertos por mayoría: 16/16" in md
    assert rutas["json"].name.startswith("reporte_") and rutas["json"].exists()
    assert "P06" in ev.formatear_tabla(res)


def test_consistencia_con_agente_alternante():
    preguntas = [{"id": "Q1", "pregunta": "¿Cuántas?", "respuesta_esperada": 30, "tipo": "numero",
                  "tolerancia": 0}]
    contador = count()

    def responder(_):
        n = next(contador)
        return Respuesta(tipo_respuesta="dato", texto="x", valor=30.0 if n % 2 == 0 else 31.0)

    res = ev.correr_eval(preguntas, responder, repeticiones=3, usar_juez=False)
    r = res["resultados"][0]
    assert r["ok_por_corrida"] == [True, False, True]
    assert r["aciertos_n"] == "2/3" and r["ok_mayoria"] and not r["ok_estricto"]
    assert not r["consistente"]
    assert r["obtenidas"] == [30.0, 31.0, 30.0]
    assert res["resumen"]["tasa_consistencia"] == 0.0
    # sin latencia en la traza: se mide alrededor de responder()
    assert all(l >= 0 for l in r["latencias_s"])

    contador2 = count()

    def responder2(_):
        n = next(contador2)
        return Respuesta(tipo_respuesta="dato", valor=30.0 if n == 0 else 31.0)

    r2 = ev.correr_eval(preguntas, responder2, repeticiones=2, usar_juez=False)["resultados"][0]
    assert r2["aciertos_n"] == "1/2" and not r2["ok_mayoria"]  # 1 de 2 no es mayoría


def test_excepcion_no_corta_el_eval():
    preguntas = [
        {"id": "Q1", "pregunta": "a", "respuesta_esperada": 1, "tipo": "numero", "tolerancia": 0},
        {"id": "Q2", "pregunta": "b", "respuesta_esperada": 2, "tipo": "numero", "tolerancia": 0},
    ]

    def responder(p):
        if p == "a":
            raise ValueError("se rompió")
        return Respuesta(tipo_respuesta="dato", valor=2.0)

    res = ev.correr_eval(preguntas, responder, repeticiones=2, usar_juez=False)
    q1, q2 = res["resultados"]
    assert q1["aciertos"] == 0 and q1["error"] == "ValueError: se rompió" and q1["metodos"] == ["error", "error"]
    assert q2["ok_estricto"]
    assert res["resumen"]["corridas_con_error"] == 2


def test_reintenta_rate_limit_y_paralelo():
    class RateLimitError(Exception):
        pass

    preguntas = [{"id": f"Q{i}", "pregunta": str(i), "respuesta_esperada": i, "tipo": "numero",
                  "tolerancia": 0} for i in range(6)]
    fallos: dict[str, int] = {}

    def responder(p):
        if fallos.get(p, 0) < 1:
            fallos[p] = fallos.get(p, 0) + 1
            raise RateLimitError("429 Too Many Requests")
        return Respuesta(tipo_respuesta="dato", valor=float(p))

    res = ev.correr_eval(preguntas, responder, repeticiones=1, usar_juez=False, paralelo=3, espera_base_s=0)
    assert res["resumen"]["aciertos_estrictos"] == 6
    assert [r["id"] for r in res["resultados"]] == [f"Q{i}" for i in range(6)]
    assert ev.es_rate_limit(RateLimitError("x")) and not ev.es_rate_limit(ValueError("x"))


@pytest.mark.parametrize(
    "lineas, fragmento",
    [
        (['{"id": "A", "pregunta": "p", "respuesta_esperada": 1, "tipo": "numero", "tolerancia": 0}', "{no json"],
         ":2: JSON inválido"),
        (['{"id": "A", "pregunta": "p", "tipo": "numero"}'], ":1: faltan campos obligatorios: respuesta_esperada"),
        (['{"id": "A", "pregunta": "p", "respuesta_esperada": 1, "tipo": "fecha"}'], ":1: `tipo` inválido"),
        (['{"id": "A", "pregunta": "p", "respuesta_esperada": 1, "tipo": "numero"}', "",
          '{"id": "A", "pregunta": "q", "respuesta_esperada": 2, "tipo": "numero"}'], ":3: id duplicado"),
        (['{"id": "A", "pregunta": "p", "respuesta_esperada": "x", "tipo": "numero"}'], ":1: `respuesta_esperada`"),
    ],
)
def test_jsonl_invalido(tmp_path, lineas, fragmento):
    ruta = tmp_path / "malo.jsonl"
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    with pytest.raises(ev.ErrorJSONL, match=fragmento.replace("(", r"\(")):
        ev.cargar_preguntas(ruta)


def test_main_codigo_de_salida(tmp_path, capsys):
    preguntas = _todas()
    perfecto = agente_perfecto(preguntas)
    args = [str(EJEMPLO), str(PROPIAS), "--repeticiones", "1", "--salida", str(tmp_path)]
    assert ev.main(args + ["--umbral", "1.0"], responder=perfecto, juez=juez_falso) == 0
    assert (tmp_path / "reporte.json").exists() and (tmp_path / "reporte.md").exists()

    def malo(_):
        return Respuesta(tipo_respuesta="dato", texto="no sé")

    assert ev.main(args + ["--sin-juez", "--umbral", "0.5"], responder=malo) == 2
    assert ev.main(args + ["--sin-juez"], responder=malo) == 0  # umbral 0 por defecto
    salida = capsys.readouterr().out
    assert "aciertos" in salida and "P01" in salida


def test_main_jsonl_invalido_y_falta_clave(tmp_path, monkeypatch, capsys):
    malo = tmp_path / "malo.jsonl"
    malo.write_text('{"id": "A"}\n', encoding="utf-8")
    assert ev.main([str(malo)], responder=lambda p: None) == 1
    assert "malo.jsonl:1" in capsys.readouterr().err

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MODELO_LLM", raising=False)
    monkeypatch.delenv("MODELO_JUEZ", raising=False)
    assert ev.main([str(EJEMPLO), "--salida", str(tmp_path)]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    # con agente inyectado pero juez real habilitado también exige la clave del juez
    assert ev.main([str(EJEMPLO), "--salida", str(tmp_path)], responder=lambda p: None) == 1


# ---------------------------------------------------------------- Fix C: costo por pregunta, ids, rate limit


def test_costo_y_tokens_por_pregunta_no_se_multiplican_por_repeticiones(tmp_path):
    preguntas = [{"id": "Q1", "pregunta": "¿Cuántas?", "respuesta_esperada": 30, "tipo": "numero", "tolerancia": 0}]
    latencias = iter([1.0, 3.0])

    def responder(_):
        return Respuesta(tipo_respuesta="dato", valor=30.0,
                         traza={"latencia_s": next(latencias), "uso": {"tokens_entrada": 1000, "tokens_salida": 100},
                                "costo_usd": 0.01})

    res = ev.correr_eval(preguntas, responder, repeticiones=2, usar_juez=False)
    s = res["resumen"]
    pp = s["por_pregunta_promedio"]
    assert pp["tokens"] == 1100 and pp["tokens_entrada"] == 1000 and pp["tokens_salida"] == 100
    assert pp["costo_usd"] == pytest.approx(0.01)
    assert pp["latencia_media_s"] == pytest.approx(2.0) and pp["latencia_p95_s"] == pytest.approx(2.9)
    assert pp["corridas_respondidas"] == 2
    assert s["agente"] == {"tokens_entrada": 2000, "tokens_salida": 200, "costo_usd": pytest.approx(0.02)}
    assert s["juez"]["tokens_entrada"] == 0 and s["juez"]["costo_usd"] is None
    r = res["resultados"][0]
    assert r["tokens_medios_por_corrida"] == 1100 and r["costo_medio_por_corrida_usd"] == pytest.approx(0.01)
    assert ev.filas_tabla(res)[0][-2:] == ["1100", "0.0100"]

    ev.escribir_reportes(res, tmp_path)
    md = (tmp_path / "reporte.md").read_text(encoding="utf-8")
    seccion = md.split("## Costo y latencia por pregunta (agente)")[1]
    assert "| tokens totales | 1100 |" in seccion and "| costo | US$0.0100 |" in seccion
    assert "| latencia p95 | 2.90 s |" in seccion


def test_por_pregunta_separa_juez_y_excluye_excepciones():
    preguntas = [{"id": "Q1", "pregunta": "a", "respuesta_esperada": "demora", "tipo": "texto", "tolerancia": None}]
    contador = count()

    def responder(_):
        if next(contador) == 0:
            raise ValueError("se rompió")
        return Respuesta(tipo_respuesta="dato", texto="No pude determinarlo.",
                         traza={"uso": {"tokens_entrada": 500, "tokens_salida": 50}, "costo_usd": 0.005})

    class Mensaje:
        usage_metadata = {"input_tokens": 300, "output_tokens": 30}

    class Juez:
        def invoke(self, _):
            return {"raw": Mensaje(), "parsed": Veredicto(correcto=False, razon="no"), "parsing_error": None}

    res = ev.correr_eval(preguntas, responder, repeticiones=2, juez=Juez())
    s = res["resumen"]
    assert s["por_pregunta_promedio"]["tokens"] == 550  # la corrida con excepción no cuenta, el juez tampoco
    assert s["por_pregunta_promedio"]["costo_usd"] == pytest.approx(0.005)
    assert s["juez"]["tokens_entrada"] == 300 and s["juez"]["tokens_salida"] == 30


def test_id_no_string_se_convierte(tmp_path):
    ruta = tmp_path / "ids.jsonl"
    ruta.write_text('{"id": 7, "pregunta": "p", "respuesta_esperada": 1, "tipo": "numero"}\n', encoding="utf-8")
    assert ev.cargar_preguntas(ruta)[0]["id"] == "7"


@pytest.mark.parametrize(
    "error, esperado",
    [
        (ValueError("failed to generate the answer"), False),
        (ValueError("not accurate"), False),
        (RuntimeError("the rate of errors is high"), False),
        (RuntimeError("Error code: 429"), True),
        (RuntimeError("rate_limit_error: slow down"), True),
        (RuntimeError("Rate limit exceeded"), True),
        (RuntimeError("Too Many Requests"), True),
    ],
)
def test_es_rate_limit(error, esperado):
    assert ev.es_rate_limit(error) is esperado

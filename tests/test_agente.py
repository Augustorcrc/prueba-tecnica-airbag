"""Tests de la integración del agente: `responder()` con el grafo real y modelo falso."""

from __future__ import annotations

import json
import os

import pytest
from langchain_anthropic.chat_models import convert_to_anthropic_tool

from agente import agente as modulo_agente
from agente.agente import responder
from agente.config import FECHA_REFERENCIA, FaltaVariableDeEntorno
from agente.herramientas import HERRAMIENTAS
from agente.herramientas_lc import crear_herramientas
from agente.prompts import SYSTEM_PROMPT
from agente.recomendaciones import PREFIJO_PREVENTIVA
from agente.sql_seguro import ESQUEMA_VISTAS
from tests.fakes import ModeloFalso, ai, final, llamada

SIN_CLAVE = not os.environ.get("ANTHROPIC_API_KEY", "").strip()
requiere_clave = [pytest.mark.requiere_clave, pytest.mark.skipif(SIN_CLAVE, reason="falta ANTHROPIC_API_KEY")]

P01 = "¿Cuántas sucursales tiene la región Costa?"
P09 = "¿Cuántas sucursales abrieron durante el periodo?"
CAMPOS_TRAZA = {"pregunta", "modelo", "latencia_s", "uso", "costo_usd", "pasos", "llamadas", "verificacion",
                "advertencias", "error"}


@pytest.fixture(autouse=True)
def trazas_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAZAS_DIR", str(tmp_path / "trazas"))
    return tmp_path / "trazas"


def guion_p01(valor=30, texto="La región Costa tiene 30 sucursales."):
    return [
        ai(llamada("buscar_sucursales", region="Costa")),
        final(tipo_respuesta="dato", texto=texto, valor=valor, tipo_valor="numero"),
    ]


# --------------------------------------------------------------------------- herramientas


def test_una_tool_por_herramienta_con_schema_serializable(datos):
    tools = crear_herramientas(datos)
    assert [t.name for t in tools] == list(HERRAMIENTAS)
    total = 0
    for t in tools:
        esquema = convert_to_anthropic_tool(t)
        assert esquema["name"] == t.name and esquema["description"]
        assert esquema["input_schema"]["type"] == "object"
        total += len(json.dumps(esquema, ensure_ascii=False))
    # Todas juntas: ~17 KB (~4.200 tokens); tope holgado para detectar crecimiento accidental.
    assert total < 22_000


def test_tools_ejecutan_con_datos_y_respetan_defaults(datos):
    tools = {t.name: t for t in crear_herramientas(datos)}
    res = tools["buscar_sucursales"].invoke({"region": "Costa", "ciudad": None})
    assert res["resultado"]["total"] == 30
    assert "error" not in tools["ranking_sucursales"].invoke({"metrica": "tasa", "n": None})
    assert tools["info_datos"].invoke({})["periodo"]["hasta"] == "2026-03-29"
    assert tools["consultar_sql"].invoke({"sql": "SELECT COUNT(*) AS n FROM sucursales"})["resultado"]["filas"] == [[120]]
    assert tools["candidatos_visita"].invoke({})["resultado"]["candidatos"]


def test_region_invalida_llega_a_la_herramienta_y_vuelve_error_c2(datos):
    """region/categoria/canal son str; la herramienta pura normaliza o devuelve error con válidos."""
    tools = {t.name: t for t in crear_herramientas(datos)}
    res = tools["contar_quejas"].invoke({"region": "Atlántida"})
    assert "error" in res and "Norte" in res["valores_validos"]


@pytest.mark.parametrize(
    "args", [{"region": "norte"}, {"canal": "teléfono"}, {"categoria": "Limpieza"}, {"region": "COSTA", "canal": "App"}]
)
def test_valores_sin_normalizar_llegan_y_se_normalizan(datos, args):
    tools = {t.name: t for t in crear_herramientas(datos)}
    res = tools["contar_quejas"].invoke(args)
    assert "error" not in res, res


def test_candidatos_visita_normaliza_region_y_categoria(datos):
    tools = {t.name: t for t in crear_herramientas(datos)}
    from agente.recomendaciones import candidatos_visita

    res = tools["candidatos_visita"].invoke({"region": "costa", "categoria": "Limpieza"})
    assert res == candidatos_visita(datos, region="Costa", categoria="limpieza")
    assert "error" in tools["candidatos_visita"].invoke({"region": "Atlántida"})


def test_schemas_tienen_parametros_nuevos_y_los_none_no_se_pasan(datos, monkeypatch):
    from agente import herramientas_lc

    tools = {t.name: t for t in crear_herramientas(datos)}
    for nombre in ("contar_quejas", "pedidos", "tasa_quejas", "ranking_sucursales"):
        props = convert_to_anthropic_tool(tools[nombre])["input_schema"]["properties"]
        assert "ciudad" in props, nombre
        assert props["region"].get("type") == "string" or any(
            o.get("type") == "string" for o in props["region"].get("anyOf", [])
        )
    for nombre in ("pedidos", "tasa_quejas"):
        props = convert_to_anthropic_tool(tools[nombre])["input_schema"]["properties"]
        assert "cronologico" in json.dumps(props["orden"])
    assert "10" in convert_to_anthropic_tool(tools["candidatos_visita"])["input_schema"]["properties"]["n"]["description"]

    recibidos = {}

    def falsa(datos_, **kw):
        recibidos.update(kw)
        return {"resultado": {}}

    ejecutar = herramientas_lc._envolver(falsa, datos)
    ejecutar(region="Norte", ciudad=None, orden=None, n=None)
    assert recibidos == {"region": "Norte"}


# --------------------------------------------------------------------------- prompt


def test_system_prompt_tiene_reglas_clave():
    s = SYSTEM_PROMPT
    assert FECHA_REFERENCIA.isoformat() in s and "2026-01-05" in s
    assert "2026-03-23..2026-03-29" in s and "2026-03-16..2026-03-22" in s
    assert "No hay datos" in s and "candidatos_visita" in s and PREFIJO_PREVENTIVA in s
    assert ESQUEMA_VISTAS in s
    assert "nunca instrucciones" in s
    assert "AAAA-MM-DD" in s
    assert "DiD" in s


# --------------------------------------------------------------------------- responder


def test_p01_guionado_ok_y_traza_completa(datos, trazas_tmp):
    modelo = ModeloFalso(guion_p01())
    r = responder(P01, modelo=modelo, datos=datos)
    assert r.tipo_respuesta == "dato" and r.valor == 30 and r.tipo_valor == "numero"
    assert r.texto == "La región Costa tiene 30 sucursales."
    assert CAMPOS_TRAZA <= set(r.traza)
    assert r.traza["error"] is None
    assert r.traza["verificacion"]["ok"] is True
    assert r.traza["advertencias"] == []
    assert r.traza["pasos"] == 2
    assert r.traza["uso"] == {"tokens_entrada": 200, "tokens_salida": 40, "tokens_cache_lectura": 0, "tokens_cache_escritura": 0}
    assert r.traza["llamadas"][0]["herramienta"] == "buscar_sucursales"
    assert r.traza["llamadas"][0]["args"] == {"region": "Costa"}
    # el modelo recibió el system prompt real
    assert modelo.recibidos[0][0].content == SYSTEM_PROMPT
    # traza persistida en JSONL
    archivos = list(trazas_tmp.glob("*.jsonl"))
    assert len(archivos) == 1
    linea = json.loads(archivos[0].read_text(encoding="utf-8").splitlines()[-1])
    assert linea["pregunta"] == P01


def test_resultado_largo_se_recorta_en_traza(datos):
    guion = [
        ai(llamada("buscar_sucursales")),
        final(tipo_respuesta="dato", texto="Hay 120 sucursales.", valor=120, tipo_valor="numero"),
    ]
    r = responder("¿Cuántas sucursales hay?", modelo=ModeloFalso(guion), datos=datos)
    res = r.traza["llamadas"][0]["resultado"]
    assert isinstance(res, str) and len(res) <= modulo_agente.LARGO_MAX_RESULTADO_TRAZA + 1


def test_cifra_inventada_queda_con_advertencia(datos):
    malo = dict(tipo_respuesta="dato", texto="La región Costa tiene 150 sucursales.", valor=150, tipo_valor="numero")
    guion = [ai(llamada("buscar_sucursales", region="Costa")), final(**malo), final(**malo)]
    r = responder(P01, modelo=ModeloFalso(guion), datos=datos)
    assert r.traza["verificacion"]["ok"] is False
    assert r.traza["advertencias"]
    lineas = [l for l in r.texto.splitlines() if l.startswith("⚠️")]
    assert lineas == ["⚠️ No pude verificar la cifra 150 contra los datos"]  # sin duplicado "valor: 150"


def test_id_inexistente_queda_con_advertencia(datos):
    malo = dict(tipo_respuesta="dato", texto="La peor es S999.", valor="S999", tipo_valor="texto")
    guion = [ai(llamada("ranking_sucursales", n=1)), final(**malo), final(**malo)]
    r = responder("¿Cuál es la peor sucursal?", modelo=ModeloFalso(guion), datos=datos)
    assert "S999" in r.traza["verificacion"]["ids_invalidos"]
    assert "⚠️ No pude verificar el ID S999 contra los datos" in r.texto


class ModeloQueFalla(ModeloFalso):
    def invoke(self, messages, config=None, **kw):
        raise RuntimeError("overloaded_error: el proveedor no responde")


def test_error_del_modelo_no_propaga(datos, trazas_tmp):
    r = responder(P01, modelo=ModeloQueFalla([]), datos=datos)
    assert r.tipo_respuesta == "dato" and r.valor is None and r.tipo_valor == "ninguno"
    assert r.texto.startswith("No pude completar la consulta:")
    assert "overloaded_error" in r.traza["error"]
    assert CAMPOS_TRAZA <= set(r.traza)
    assert list(trazas_tmp.glob("*.jsonl"))


def test_sin_clave_y_sin_modelo_error_claro(monkeypatch, datos):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MODELO_LLM", "anthropic:claude-sonnet-5-5")
    with pytest.raises(FaltaVariableDeEntorno, match="ANTHROPIC_API_KEY"):
        responder(P01, datos=datos)


def test_main_sin_clave_devuelve_1(monkeypatch, capsys):
    import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MODELO_LLM", "anthropic:claude-sonnet-5-5")
    assert main.main(["hola"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_datos_cacheados(monkeypatch, datos):
    monkeypatch.setattr(modulo_agente, "_cache_datos", None)
    llamadas = []

    def cargar():
        llamadas.append(1)
        return datos

    monkeypatch.setattr("agente.datos.cargar_datos", cargar)
    assert modulo_agente.obtener_datos() is datos
    assert modulo_agente.obtener_datos() is datos
    assert len(llamadas) == 1


# --------------------------------------------------------------------------- integración con el eval


def test_integracion_eval_p01_p09(datos):
    from agente.config import RAIZ
    from eval.evaluar import cargar_preguntas, correr_eval

    preguntas = [p for p in cargar_preguntas(RAIZ / "eval" / "preguntas_ejemplo.jsonl") if p["id"] in ("P01", "P09")]
    assert len(preguntas) == 2
    guiones = {
        P01: guion_p01,
        P09: lambda: [
            ai(llamada("buscar_sucursales", es_nueva=True)),
            final(tipo_respuesta="dato", texto="Abrieron 6 sucursales durante el periodo.", valor=6,
                  tipo_valor="numero"),
        ],
    }

    def responder_eval(pregunta):
        return responder(pregunta, modelo=ModeloFalso(guiones[pregunta]()), datos=datos)

    resultado = correr_eval(preguntas, responder_eval, repeticiones=2, usar_juez=False)
    for r in resultado["resultados"]:
        assert all(c["ok"] for c in r["corridas"]), r
        assert all(c["error"] is None for c in r["corridas"]), r


# --------------------------------------------------------------------------- con modelo real (se saltean sin clave)


def _preguntas_ejemplo():
    from agente.config import RAIZ
    from eval.evaluar import cargar_preguntas

    return cargar_preguntas(RAIZ / "eval" / "preguntas_ejemplo.jsonl")


@pytest.mark.parametrize("indice", range(10), ids=[f"P{i:02d}" for i in range(1, 11)])
@pytest.mark.requiere_clave
@pytest.mark.skipif(SIN_CLAVE, reason="falta ANTHROPIC_API_KEY")
def test_preguntas_ejemplo_reales(indice):
    from eval.comparadores import comparar

    item = _preguntas_ejemplo()[indice]
    r = responder(item["pregunta"])
    assert r.traza["error"] is None
    res = comparar(r, item, usar_juez=True)
    assert res.ok, (item["id"], res.motivo, r.texto, r.valor)


def test_rate_limit_se_propaga_para_que_el_eval_reintente(monkeypatch, tmp_path):
    """Agregado por el orquestador: un 429 no se convierte en respuesta, se propaga."""
    import pytest

    from agente import agente as ag

    monkeypatch.setenv("TRAZAS_DIR", str(tmp_path))

    class RateLimitError(Exception):
        pass

    class ModeloQueRateLimitea:
        def bind_tools(self, *a, **k):
            return self

        def invoke(self, *a, **k):
            raise RateLimitError("Error code: 429 - rate_limit_error")

    assert ag.es_rate_limit(RateLimitError("429"))
    assert not ag.es_rate_limit(ValueError("otra cosa"))
    with pytest.raises(RateLimitError):
        ag.responder("¿Cuántas sucursales tiene Costa?", modelo=ModeloQueRateLimitea())

"""C1: el payload que se mandaría a Anthropic se arma sin errores para el modelo real (offline).

Construye el `ChatAnthropic` REAL como `agente.crear_modelo()` (con una clave falsa), le liga las
herramientas reales con el `tool_choice` que elige `construir_grafo` y arma el payload con
`_get_request_payload`, que es lo que `invoke` llama antes de ir a la red. No hace ninguna llamada HTTP.
Este test hubiera detectado que Sonnet 5.5 rechaza `temperature=0` y `tool_choice="any"`.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from agente import agente as modulo_agente
from agente.config import kwargs_modelo, soporta_tool_choice_forzado, tool_choice_para
from agente.grafo import construir_grafo
from agente.herramientas import HERRAMIENTAS
from agente.herramientas_lc import crear_herramientas
from agente.prompts import SYSTEM_PROMPT


class _Espia:
    """Envuelve el modelo real para capturar lo que `construir_grafo` liga (sin tocar el grafo)."""

    def __init__(self, modelo):
        self.modelo = modelo
        self.model = modelo.model
        self.ligado = None

    def bind_tools(self, herramientas, **kw):
        self.ligado = self.modelo.bind_tools(herramientas, **kw)
        return self.ligado


def _payload(monkeypatch, datos, nombre: str, *, tool_choice=None):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-falsa")
    monkeypatch.setenv("MODELO_LLM", nombre)
    modelo = modulo_agente.crear_modelo()  # mismo camino que responder()
    espia = _Espia(modelo)
    construir_grafo(espia, crear_herramientas(datos), tool_choice=tool_choice)
    ligado = espia.ligado
    mensajes = [SystemMessage(SYSTEM_PROMPT), HumanMessage("¿Cuántas sucursales tiene la región Costa?")]
    return ligado.bound._get_request_payload(mensajes, **ligado.kwargs)


def _temperatura(payload: dict):
    """La temperatura que viaja: top-level o en `extra_body` (langchain-anthropic la reubica si el SDK
    instalado no acepta el parámetro con nombre; en el JSON de la request queda igual)."""
    if "temperature" in payload:
        return payload["temperature"]
    return (payload.get("extra_body") or {}).get("temperature")


def test_sonnet_5_5_sin_temperature_y_tool_choice_auto(monkeypatch, datos):
    payload = _payload(monkeypatch, datos, "anthropic:claude-sonnet-5-5")
    assert payload["model"] == "claude-sonnet-5-5"
    assert _temperatura(payload) is None
    assert payload["tool_choice"] == {"type": "auto"}
    nombres = [t["name"] for t in payload["tools"]]
    assert nombres == list(HERRAMIENTAS) + ["RespuestaFinal"]  # ninguna herramienta descartada


def test_haiku_5_5_sin_temperature_y_tool_choice_any(monkeypatch, datos):
    """La API rechaza `temperature` para Haiku 5.5 (400), aunque la librería la deja pasar."""
    payload = _payload(monkeypatch, datos, "anthropic:claude-haiku-5-5")
    assert payload["model"] == "claude-haiku-5-5"
    assert _temperatura(payload) is None
    assert payload["tool_choice"] == {"type": "any"}
    assert len(payload["tools"]) == len(HERRAMIENTAS) + 1


def test_responder_pasa_el_tool_choice_explicito(monkeypatch, datos):
    """`responder()` con el modelo real calcula el tool_choice desde `modelo_llm()`."""
    capturado = {}
    original = modulo_agente.construir_grafo

    def espia(modelo, herramientas, **kw):
        capturado.update(kw)
        raise RuntimeError("corte del test antes de ir a la red")

    monkeypatch.setattr(modulo_agente, "construir_grafo", espia)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-falsa")
    monkeypatch.setenv("MODELO_LLM", "anthropic:claude-sonnet-5-5")
    monkeypatch.setenv("TRAZAS_DIR", "/tmp/trazas_test_payload")
    r = modulo_agente.responder("¿Cuántas sucursales tiene Costa?", datos=datos)
    assert capturado["tool_choice"] == "auto"
    assert "corte del test" in r.traza["error"]
    assert original is not None


def test_la_config_vieja_hubiera_fallado(monkeypatch, datos):
    """Documenta C1: con temperature=0 la librería lanza ValueError para Sonnet 5.5."""
    from langchain.chat_models import init_chat_model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-falsa")
    modelo = init_chat_model("anthropic:claude-sonnet-5-5", temperature=0)
    with pytest.raises(ValueError, match="temperature"):
        modelo._get_request_payload([HumanMessage("hola")])


@pytest.mark.parametrize(
    "nombre, temperatura, forzado",
    [
        ("anthropic:claude-sonnet-5-5", False, False),
        ("claude-sonnet-5-5-20261101", False, False),
        ("anthropic:claude-fable-5-1", False, False),
        ("claude-fable-5", False, True),
        ("anthropic:claude-opus-5-5", True, False),
        ("anthropic:claude-haiku-5-5", False, True),
        ("openai:gpt-x", True, True),
    ],
)
def test_helpers_de_config(nombre, temperatura, forzado):
    kw = kwargs_modelo(nombre)
    assert ("temperature" in kw) is temperatura
    assert kw["max_retries"] == 2 and kw["timeout"] > 0
    assert soporta_tool_choice_forzado(nombre) is forzado
    assert tool_choice_para(nombre) == ("any" if forzado else "auto")


def test_system_prompt_con_breakpoint_de_cache(monkeypatch, datos):
    """El system viaja como bloque con `cache_control`: el cache cubre tools + system (prompt caching)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-falsa")
    monkeypatch.setenv("MODELO_LLM", "anthropic:claude-sonnet-5-5")
    modelo = modulo_agente.crear_modelo()
    espia = _Espia(modelo)
    construir_grafo(espia, crear_herramientas(datos))
    ligado = espia.ligado
    mensajes = [modulo_agente.mensaje_sistema(modelo.model), HumanMessage("¿Cuántas sucursales tiene Costa?")]
    payload = ligado.bound._get_request_payload(mensajes, **ligado.kwargs)
    assert payload["system"] == [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    assert len(payload["tools"]) == len(HERRAMIENTAS) + 1


def test_system_prompt_sin_cache_fuera_de_anthropic():
    assert modulo_agente.mensaje_sistema("openai:gpt-x").content == SYSTEM_PROMPT
    assert modulo_agente.mensaje_sistema("ModeloFalso").content == SYSTEM_PROMPT

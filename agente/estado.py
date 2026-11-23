"""Estado del grafo."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from agente.esquemas import RespuestaFinal


class Estado(TypedDict):
    """Estado que recorre el grafo.

    Solo `messages` tiene reductor (`add_messages`); el resto lo reemplaza cada nodo con el valor nuevo.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    pregunta: str
    pasos: int  # llamadas al modelo
    intentos_verificacion: int  # 0 o 1
    respuesta: RespuestaFinal | None
    verificacion: dict | None
    advertencias: list[str]
    uso: dict  # {"tokens_entrada", "tokens_salida", "tokens_cache_lectura", "tokens_cache_escritura"}
    llamadas: list[dict]  # traza: [{"herramienta", "args", "resultado"}]


CLAVES_USO = ("tokens_entrada", "tokens_salida", "tokens_cache_lectura", "tokens_cache_escritura")


def uso_vacio() -> dict:
    """Acumulador de tokens. `tokens_entrada` incluye los leídos y escritos en el cache."""
    return dict.fromkeys(CLAVES_USO, 0)


def uso_de_mensaje(mensaje) -> dict:
    """Tokens de un AIMessage según su `usage_metadata` (0 en lo que falte, p. ej. modelos falsos)."""
    meta = getattr(mensaje, "usage_metadata", None) or {}
    detalle = meta.get("input_token_details") or {}
    escritura = sum(int(detalle.get(k) or 0)
                    for k in ("cache_creation", "ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"))
    return {
        "tokens_entrada": int(meta.get("input_tokens") or 0),
        "tokens_salida": int(meta.get("output_tokens") or 0),
        "tokens_cache_lectura": int(detalle.get("cache_read") or 0),
        "tokens_cache_escritura": escritura,
    }


def sumar_uso(a: dict | None, b: dict | None) -> dict:
    return {k: int((a or {}).get(k, 0) or 0) + int((b or {}).get(k, 0) or 0) for k in CLAVES_USO}


def estado_inicial(pregunta: str, mensajes: list[BaseMessage] | None = None) -> dict:
    """Estado inicial para `grafo.invoke`.

    `mensajes` son los mensajes iniciales (típicamente `[SystemMessage(...), HumanMessage(pregunta)]`,
    los arma quien llama). Si no se pasan, se usa solo `HumanMessage(pregunta)`.
    """
    return {
        "messages": list(mensajes) if mensajes is not None else [HumanMessage(pregunta)],
        "pregunta": pregunta,
        "pasos": 0,
        "intentos_verificacion": 0,
        "respuesta": None,
        "verificacion": None,
        "advertencias": [],
        "uso": uso_vacio(),
        "llamadas": [],
    }

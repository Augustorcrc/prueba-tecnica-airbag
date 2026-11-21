"""Estado del grafo (Etapa 3)."""

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
    uso: dict  # {"tokens_entrada": int, "tokens_salida": int}
    llamadas: list[dict]  # traza: [{"herramienta", "args", "resultado"}]


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
        "uso": {"tokens_entrada": 0, "tokens_salida": 0},
        "llamadas": [],
    }

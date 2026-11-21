"""Spike de LangGraph (WP-S): grafo mínimo modelo <-> herramientas con RespuestaFinal como tool.

Modo offline (sin clave): valida que el schema de RespuestaFinal se serialice como tool de Anthropic
y que bind_tools acepte tool_choice="any".
Modo real (con ANTHROPIC_API_KEY): corre una pregunta contra Claude Sonnet 5.5 y comprueba que el
modelo termina llamando RespuestaFinal con un `valor` del tipo correcto.

Uso: python spikes/spike_grafo.py [--real]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402

from agente.esquemas import RespuestaFinal  # noqa: E402


@tool
def contar_sucursales(region: str) -> dict:
    """Cuenta las sucursales de una región (Norte, Sur, Centro o Costa)."""
    return {"resultado": {"region": region, "sucursales": 30}, "n_registros": 30, "advertencias": []}


class Estado(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def chequeo_offline() -> None:
    from langchain_anthropic import ChatAnthropic
    from langchain_anthropic.chat_models import convert_to_anthropic_tool

    esquema = convert_to_anthropic_tool(RespuestaFinal)
    valor = esquema["input_schema"]["properties"]["valor"]
    assert "anyOf" in valor, valor
    tipos = {v.get("type") for v in valor["anyOf"]}
    assert {"number", "string", "array", "null"} <= tipos, tipos
    ligado = ChatAnthropic(model="claude-sonnet-5-5", api_key="sk-falsa").bind_tools(
        [contar_sucursales, RespuestaFinal], tool_choice="any"
    )
    assert ligado.kwargs["tool_choice"] == {"type": "any"}
    print("offline OK: schema", len(json.dumps(esquema)), "bytes; tool_choice any aceptado")


def chequeo_real() -> None:
    from langchain.chat_models import init_chat_model

    modelo = init_chat_model("anthropic:claude-sonnet-5-5", temperature=0, timeout=60, max_retries=2)
    ligado = modelo.bind_tools([contar_sucursales, RespuestaFinal], tool_choice="any")

    def nodo_modelo(estado: Estado) -> dict:
        return {"messages": [ligado.invoke(estado["messages"])]}

    def nodo_herramientas(estado: Estado) -> dict:
        salida = []
        for llamada in estado["messages"][-1].tool_calls:
            if llamada["name"] == "contar_sucursales":
                res = contar_sucursales.invoke(llamada["args"])
                salida.append(ToolMessage(json.dumps(res), tool_call_id=llamada["id"]))
        return {"messages": salida}

    def ruta(estado: Estado) -> str:
        nombres = {c["name"] for c in estado["messages"][-1].tool_calls}
        return END if nombres == {"RespuestaFinal"} else "herramientas"

    g = StateGraph(Estado)
    g.add_node("modelo", nodo_modelo)
    g.add_node("herramientas", nodo_herramientas)
    g.add_edge(START, "modelo")
    g.add_conditional_edges("modelo", ruta, {"herramientas": "herramientas", END: END})
    g.add_edge("herramientas", "modelo")
    grafo = g.compile()
    final = grafo.invoke({"messages": [HumanMessage("¿Cuántas sucursales tiene la región Costa?")]},
                         config={"recursion_limit": 10})
    args = final["messages"][-1].tool_calls[0]["args"]
    respuesta = RespuestaFinal.model_validate(args)
    print("real OK:", respuesta.model_dump())
    assert respuesta.tipo_valor == "numero" and float(respuesta.valor) == 30


if __name__ == "__main__":
    chequeo_offline()
    if "--real" in sys.argv:
        chequeo_real()

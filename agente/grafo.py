"""Grafo LangGraph del agente (contrato C5)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph


def construir_grafo(modelo: "BaseChatModel", herramientas: "list[BaseTool]") -> "CompiledStateGraph":
    raise NotImplementedError

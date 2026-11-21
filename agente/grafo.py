"""Grafo LangGraph del agente (contrato C5).

```
START → modelo
modelo ─┬─ solo RespuestaFinal         → extraer_respuesta
        ├─ otras herramientas          → herramientas → modelo
        ├─ sin tool calls              → recordatorio → modelo
        └─ pasos >= MAX_PASOS          → cerrar_con_advertencia → END
extraer_respuesta ─┬─ válida   → verificar
                   └─ inválida → modelo  (o cerrar_con_advertencia si se agotaron los pasos)
verificar ─┬─ ok                                       → END
           ├─ falla e intentos_verificacion == 0       → feedback → modelo
           └─ falla e intentos >= 1 (o pasos agotados) → cerrar_con_advertencia → END
```

Sin checkpointer. El `recursion_limit` va en el config del invoke: `grafo.invoke(estado, config_invocacion())`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from agente import nodos
from agente.config import RECURSION_LIMIT, tool_choice_para
from agente.esquemas import RespuestaFinal
from agente.estado import Estado

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph


def config_invocacion(**extra: Any) -> dict:
    """Config para `grafo.invoke`: `recursion_limit=RECURSION_LIMIT` (30) más lo que se agregue."""
    return {"recursion_limit": RECURSION_LIMIT, **extra}


def tool_choice_elegido(modelo: Any) -> str:
    """"any" si el modelo lo soporta, "auto" si no. Si no se puede saber el nombre (modelo falso), "any"."""
    for atributo in ("model", "model_name"):
        nombre = getattr(modelo, atributo, None)
        if isinstance(nombre, str) and nombre:
            return tool_choice_para(nombre)
    return "any"


def construir_grafo(
    modelo: BaseChatModel,
    herramientas: list[BaseTool],
    *,
    verificador: nodos.Verificador | None = None,
    ids_validos: set[str] | None = None,
    completar_recomendaciones: nodos.CompletarRecomendaciones | None = None,
    tool_choice: str | None = None,
) -> CompiledStateGraph:
    """Arma y compila el grafo.

    - `modelo`: cualquier objeto con `bind_tools(...)` (no se exige `BaseChatModel` en runtime).
    - `verificador`: función con la firma de `agente.verificador.verificar_respuesta`; por defecto se
      importa esa en tiempo de llamada.
    - `ids_validos`: IDs de sucursal válidos; `None` = no se validan IDs (se ignora `ids_invalidos`).
    - `completar_recomendaciones`: hook de WP-R para reponer ítems descartados al cerrar con advertencia.
    - `tool_choice`: "any" (obliga a llamar alguna herramienta) o "auto". Si no se pasa, se elige según
      el modelo (`tool_choice_elegido`): Sonnet 5.5, Opus 5.5 y Fable 5.1 no aceptan "any".
    """
    herramientas = list(herramientas)
    if tool_choice is None:
        tool_choice = tool_choice_elegido(modelo)
    ligado = modelo.bind_tools(herramientas + [RespuestaFinal], tool_choice=tool_choice)

    g = StateGraph(Estado)
    g.add_node("modelo", nodos.crear_nodo_modelo(ligado))
    g.add_node("herramientas", nodos.crear_nodo_herramientas(herramientas))
    g.add_node("recordatorio", nodos.recordatorio)
    g.add_node("extraer_respuesta", nodos.extraer_respuesta)
    g.add_node("verificar", nodos.crear_nodo_verificar(verificador, ids_validos))
    g.add_node("feedback", nodos.feedback)
    g.add_node(
        "cerrar_con_advertencia", nodos.crear_nodo_cerrar(verificador, ids_validos, completar_recomendaciones)
    )

    g.add_edge(START, "modelo")
    g.add_conditional_edges(
        "modelo",
        nodos.ruta_modelo,
        {
            "extraer_respuesta": "extraer_respuesta",
            "herramientas": "herramientas",
            "recordatorio": "recordatorio",
            "cerrar_con_advertencia": "cerrar_con_advertencia",
        },
    )
    g.add_edge("herramientas", "modelo")
    g.add_edge("recordatorio", "modelo")
    g.add_conditional_edges(
        "extraer_respuesta",
        nodos.ruta_extraer,
        {"verificar": "verificar", "modelo": "modelo", "cerrar_con_advertencia": "cerrar_con_advertencia"},
    )
    g.add_conditional_edges(
        "verificar",
        nodos.ruta_verificar,
        {"fin": END, "feedback": "feedback", "cerrar_con_advertencia": "cerrar_con_advertencia"},
    )
    g.add_edge("feedback", "modelo")
    g.add_edge("cerrar_con_advertencia", END)
    return g.compile()

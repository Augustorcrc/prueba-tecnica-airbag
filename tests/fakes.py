"""Dobles de prueba para el grafo: modelo guionado, herramientas stub y verificador falso."""

from __future__ import annotations

import itertools
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool

from agente.esquemas import RespuestaFinal

_contador = itertools.count(1)


def llamada(nombre: str, **args: Any) -> dict:
    """Tool call con id único."""
    return {"name": nombre, "args": args, "id": f"call_{next(_contador)}", "type": "tool_call"}


def ai(*llamadas: dict, texto: str = "", entrada: int = 100, salida: int = 20) -> AIMessage:
    """AIMessage con tool calls y `usage_metadata`."""
    return AIMessage(
        content=texto,
        tool_calls=list(llamadas),
        usage_metadata={"input_tokens": entrada, "output_tokens": salida, "total_tokens": entrada + salida},
    )


def final(**args: Any) -> AIMessage:
    """AIMessage que llama solo a RespuestaFinal."""
    return ai(llamada("RespuestaFinal", **args))


class ModeloFalso:
    """Modelo de chat guionado: devuelve los AIMessage en orden y guarda lo que recibe.

    No hereda de BaseChatModel a propósito: el grafo no debe exigir el tipo.
    """

    def __init__(self, guion: list[AIMessage]):
        self.guion = list(guion)
        self.recibidos: list[list[BaseMessage]] = []  # mensajes de cada invocación
        self.ligadas: list = []
        self.tool_choice = None

    def bind_tools(self, herramientas, tool_choice=None, **kw):
        self.ligadas = list(herramientas)
        self.tool_choice = tool_choice
        return self

    def invoke(self, messages, config=None, **kw):
        self.recibidos.append(list(messages))
        if not self.guion:
            raise AssertionError("ModeloFalso: se terminó el guion")
        return self.guion.pop(0)

    @property
    def n_invocaciones(self) -> int:
        return len(self.recibidos)


# --------------------------------------------------------------------------- herramientas stub


def _c2(resultado: Any, **filtros) -> dict:
    return {
        "resultado": resultado,
        "filtros_aplicados": filtros,
        "periodo": {"desde": "2026-01-05", "hasta": "2026-03-29"},
        "n_registros": 42,
        "advertencias": [],
    }


@tool
def contar_quejas_stub(region: str) -> dict:
    """Cuenta las quejas de una región (Norte, Sur, Centro o Costa)."""
    return _c2({"region": region, "quejas": 137}, region=region)


@tool
def ranking_stub(n: int = 3) -> dict:
    """Ranking de sucursales por tasa de quejas."""
    filas = [
        {"sucursal_id": "S012", "tasa": 4.5, "quejas": 61},
        {"sucursal_id": "S007", "tasa": 3.8, "quejas": 52},
        {"sucursal_id": "S021", "tasa": 3.1, "quejas": 40},
    ]
    return _c2(filas[:n], n=n)


@tool
def herramienta_rota(x: str) -> dict:
    """Siempre falla (para probar el manejo de errores)."""
    raise RuntimeError("se cayó la base")


# --------------------------------------------------------------------------- verificador falso

_NUMERO = re.compile(r"\d+(?:[.,]\d+)?")


def _numeros(texto: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUMERO.findall(texto or "")}


class VerificadorFalso:
    """Verificador con la firma de C4.

    - Con `guion` (lista de dicts): devuelve uno por llamada, en orden.
    - Sin guion: verificación simple: toda cifra del texto / motivo / evidencia tiene que aparecer en
      los resultados o en la pregunta; IDs fuera de `ids_validos` (si no está vacío) son inválidos.
    Guarda cada llamada en `llamadas`.
    """

    def __init__(self, guion: list[dict] | None = None):
        self.guion = list(guion) if guion is not None else None
        self.llamadas: list[dict] = []

    def __call__(self, respuesta: RespuestaFinal, resultados: list, pregunta: str, ids_validos: set[str]) -> dict:
        self.llamadas.append(
            {"respuesta": respuesta, "resultados": list(resultados), "pregunta": pregunta, "ids_validos": ids_validos}
        )
        if self.guion is not None:
            return self.guion.pop(0)
        conocidos = _numeros(json.dumps(resultados, ensure_ascii=False, default=str)) | _numeros(pregunta)
        textos = [respuesta.texto or ""]
        ids = []
        for item in respuesta.recomendaciones or []:
            textos += [item.motivo, item.evidencia]
            ids.append(item.sucursal_id)
        no_verificados = sorted({n for t in textos for n in _numeros(t)} - conocidos)
        # Los IDs (S012) no son cifras: se excluyen sus dígitos.
        no_verificados = [n for n in no_verificados if not any(n in i for i in ids)]
        ids_invalidos = [i for i in ids if ids_validos and i not in ids_validos]
        ids_repetidos = sorted({i for i in ids if ids.count(i) > 1})
        ok = not (no_verificados or ids_invalidos or ids_repetidos)
        return {"ok": ok, "no_verificados": no_verificados, "ids_invalidos": ids_invalidos, "ids_repetidos": ids_repetidos}


OK = {"ok": True, "no_verificados": [], "ids_invalidos": [], "ids_repetidos": []}

"""Script de evaluación (contrato C6). Uso: python -m eval.evaluar <ruta.jsonl> [...]"""

from __future__ import annotations

from typing import Callable

from agente.esquemas import Respuesta


def correr_eval(
    preguntas: list[dict],
    responder: Callable[[str], Respuesta],
    repeticiones: int = 3,
    usar_juez: bool = True,
) -> dict:
    raise NotImplementedError

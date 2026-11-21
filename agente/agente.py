"""Fachada `responder()` (contrato C5)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agente.esquemas import Respuesta

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models import BaseChatModel

    from agente.datos import Datos


def responder(pregunta: str, *, modelo: "BaseChatModel | None" = None, datos: "Datos | None" = None) -> Respuesta:
    raise NotImplementedError

"""SQL de respaldo restringido (contrato C7): solo SELECT sobre vistas limpias, ≤200 filas, timeout."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from agente.datos import Datos


def consultar_sql(datos: "Datos", sql: str) -> dict:
    raise NotImplementedError

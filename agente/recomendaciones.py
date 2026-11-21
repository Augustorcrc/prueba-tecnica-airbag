"""Score determinístico de sucursales a visitar (contrato C7)."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from agente.datos import Datos

ACCIONES_POR_CATEGORIA: dict[str, str] = {}


def candidatos_visita(
    datos: "Datos",
    region: str | None = None,
    categoria: str | None = None,
    n: int | None = None,
    fecha_hasta: date | None = None,
    incluir_nuevas: bool = False,
) -> dict:
    """Formato C2; los componentes del score van en `resultado`."""
    raise NotImplementedError

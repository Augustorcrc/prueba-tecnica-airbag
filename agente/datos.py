"""Capa de datos (contrato C1): carga, limpieza y vistas DuckDB.

Única fuente de verdad: todo el resto del agente consume `Datos`. El LLM nunca ve los CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from agente.config import FECHA_REFERENCIA as _FECHA_REFERENCIA

if TYPE_CHECKING:  # pragma: no cover
    import duckdb

FECHA_REFERENCIA: date = _FECHA_REFERENCIA


@dataclass
class Datos:
    """DataFrames limpios más una conexión DuckDB con vistas registradas."""

    sucursales: pd.DataFrame  # + es_nueva, dias_operados, fecha_inicio_datos
    ventas: pd.DataFrame  # + pedidos_validos (NULL si inválido), dato_invalido, semana_inicio, mes, dia_semana, semana_n
    quejas: pd.DataFrame  # deduplicada por queja_id; + en_dia_invalido, mismas columnas de calendario
    capacitaciones: pd.DataFrame
    diario_sucursal: pd.DataFrame  # fecha, sucursal_id, pedidos_validos, quejas, quejas por categoría
    con: Optional["duckdb.DuckDBPyConnection"]  # vistas: sucursales, ventas, quejas, capacitaciones, diario_sucursal
    avisos: list[str] = field(default_factory=list)


class ErrorValidacionDatos(ValueError):
    """Los CSV no cumplen las validaciones duras de la carga."""


def cargar_datos(dir: Path | None = None) -> Datos:
    """Carga y limpia los CSV de `dir` (por defecto `DATOS_DIR` o `./datos`)."""
    raise NotImplementedError


def reporte_calidad(datos: Datos) -> dict:
    """Resumen de lo encontrado y tratado en los datos."""
    raise NotImplementedError

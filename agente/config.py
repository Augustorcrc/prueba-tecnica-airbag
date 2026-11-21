"""Configuración central: rutas, modelos, clave y precios.

La clave del proveedor se lee solo de variables de entorno (o de un `.env` local, nunca commiteado).
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

try:  # python-dotenv es opcional en tiempo de import (los tests no lo necesitan)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

RAIZ = Path(__file__).resolve().parent.parent

FECHA_INICIO = date(2026, 1, 5)
FECHA_REFERENCIA = date(2026, 3, 29)  # último día con datos: "hoy" para el agente

MODELO_LLM_DEFAULT = "anthropic:claude-sonnet-5-5"
MODELO_JUEZ_DEFAULT = "anthropic:claude-haiku-5-5"

MAX_PASOS = 8
RECURSION_LIMIT = 25
TIMEOUT_LLM_S = 60

# USD por millón de tokens (lista de precios de Anthropic, noviembre de 2026).
# Haiku 5.5: tarifa para prompts de hasta 100.000 tokens.
PRECIOS_USD_POR_MTOK: dict[str, dict[str, float]] = {
    "claude-sonnet-5-5": {"entrada": 2.00, "salida": 10.00},
    "claude-haiku-5-5": {"entrada": 0.10, "salida": 0.50},
}


class FaltaVariableDeEntorno(RuntimeError):
    """Se lanza cuando falta una variable de entorno obligatoria."""


def datos_dir() -> Path:
    """Directorio de los CSV: `DATOS_DIR` o `./datos` del repo."""
    valor = os.environ.get("DATOS_DIR")
    return Path(valor) if valor else RAIZ / "datos"


def modelo_llm() -> str:
    return os.environ.get("MODELO_LLM", MODELO_LLM_DEFAULT)


def modelo_juez() -> str:
    return os.environ.get("MODELO_JUEZ", MODELO_JUEZ_DEFAULT)


def clave_api(variable: str = "ANTHROPIC_API_KEY") -> str:
    """Devuelve la clave o falla con un mensaje que dice qué variable falta."""
    valor = os.environ.get(variable, "").strip()
    if not valor:
        raise FaltaVariableDeEntorno(
            f"Falta la variable de entorno {variable}. Copiá .env.example a .env y completala, "
            f"o exportala: export {variable}=..."
        )
    return valor


def variable_clave_para(modelo: str) -> str:
    """Variable de entorno de la clave según el proveedor de `proveedor:modelo`."""
    proveedor = modelo.split(":", 1)[0] if ":" in modelo else "anthropic"
    return {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(
        proveedor, f"{proveedor.upper()}_API_KEY"
    )


def costo_usd(modelo: str, tokens_entrada: int, tokens_salida: int) -> float | None:
    """Costo estimado en USD; None si el modelo no está en la tabla."""
    nombre = modelo.split(":", 1)[-1]
    precios = PRECIOS_USD_POR_MTOK.get(nombre)
    if precios is None:
        return None
    return (tokens_entrada * precios["entrada"] + tokens_salida * precios["salida"]) / 1_000_000

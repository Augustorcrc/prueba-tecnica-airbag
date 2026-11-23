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

MAX_PASOS = 10
RECURSION_LIMIT = 30
TIMEOUT_LLM_S = 60

# Restricciones de la API por modelo. Espejan las de langchain-anthropic 1.7.x
# (`ChatAnthropic._assert_valid_model_configuration` y `_supports_forced_tool_choice` en
# `langchain_anthropic/chat_models.py`): si se actualiza la librería, revisar estas tuplas.
# - Estos modelos rechazan `temperature`: Sonnet 5.5 y Fable 5 con ValueError local (distinta de 1, y
#   también `top_p` != 1 o `top_k`); Haiku 5.5 la deja pasar la librería, pero la API devuelve 400
#   ("`temperature` is deprecated for this model", visto en el eval del 2026-11-23).
MODELOS_SIN_TEMPERATURA = ("claude-sonnet-5-5", "claude-fable-5", "claude-haiku-5-5")
# - Estos modelos no aceptan `tool_choice` de tipo "any" ni "tool" (la API devuelve 400): se usa "auto".
MODELOS_SIN_TOOL_CHOICE_FORZADO = ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5")

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


def _sin_proveedor(nombre: str) -> str:
    """'anthropic:claude-sonnet-5-5' -> 'claude-sonnet-5-5'."""
    return nombre.split(":", 1)[-1].strip()


def kwargs_modelo(nombre: str) -> dict:
    """Kwargs para `init_chat_model(nombre, **kwargs_modelo(nombre))`.

    temperature=0 salvo en los modelos de `MODELOS_SIN_TEMPERATURA`, donde se omite (la librería o la
    API rechazan el parámetro). Siempre `timeout` y `max_retries=2`.
    """
    kwargs: dict = {"timeout": TIMEOUT_LLM_S, "max_retries": 2}
    if not _sin_proveedor(nombre).startswith(MODELOS_SIN_TEMPERATURA):
        kwargs["temperature"] = 0
    return kwargs


def soporta_tool_choice_forzado(nombre: str) -> bool:
    """False si el modelo no acepta `tool_choice` "any"/"tool" (ver `MODELOS_SIN_TOOL_CHOICE_FORZADO`)."""
    return not _sin_proveedor(nombre).startswith(MODELOS_SIN_TOOL_CHOICE_FORZADO)


def tool_choice_para(nombre: str) -> str:
    """"any" si el modelo lo soporta; si no, "auto" (el nodo `recordatorio` cubre el texto libre)."""
    return "any" if soporta_tool_choice_forzado(nombre) else "auto"


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

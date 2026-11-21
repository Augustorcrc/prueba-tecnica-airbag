"""Herramientas de consulta (contrato C2): funciones puras `f(datos, **parametros) -> dict`.

Formato de salida: {"resultado", "filtros_aplicados", "periodo", "n_registros", "advertencias"}.
Errores: {"error": "...", "valores_validos": [...]}, nunca excepciones hacia el LLM.
"""

from __future__ import annotations

from agente.datos import Datos
from agente.recomendaciones import candidatos_visita  # noqa: F401  (reexport, dueño: WP5)
from agente.sql_seguro import consultar_sql  # noqa: F401  (reexport, dueño: WP6)


def info_datos(datos: Datos) -> dict:
    raise NotImplementedError


def buscar_sucursales(datos: Datos, **parametros) -> dict:
    raise NotImplementedError


def contar_quejas(datos: Datos, **parametros) -> dict:
    raise NotImplementedError


def pedidos(datos: Datos, **parametros) -> dict:
    raise NotImplementedError


def tasa_quejas(datos: Datos, **parametros) -> dict:
    raise NotImplementedError


def ranking_sucursales(datos: Datos, **parametros) -> dict:
    raise NotImplementedError


def comentarios_sucursal(datos: Datos, **parametros) -> dict:
    raise NotImplementedError


def efecto_capacitacion(datos: Datos) -> dict:
    raise NotImplementedError

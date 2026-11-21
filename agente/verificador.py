"""Verificador de cifras (contrato C4): toda cifra del texto tiene que salir de una herramienta."""

from __future__ import annotations

from agente.esquemas import RespuestaFinal


def verificar_numeros(texto: str, resultados: list[dict | str], pregunta: str) -> list[str]:
    """Devuelve las cifras de `texto` que NO aparecen en ningún resultado (vacía = ok)."""
    raise NotImplementedError


def verificar_respuesta(
    respuesta: RespuestaFinal, resultados: list[dict | str], pregunta: str, ids_validos: set[str]
) -> dict:
    """{"ok": bool, "no_verificados": [...], "ids_invalidos": [...], "ids_repetidos": [...]}"""
    raise NotImplementedError

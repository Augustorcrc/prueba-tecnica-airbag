"""Esquemas Pydantic de la salida del agente (contrato C3).

`RespuestaFinal` se registra como herramienta de salida del grafo: la única forma de terminar.
`Respuesta` es lo que devuelve `responder()` y consume el eval.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

TipoRespuesta = Literal["dato", "recomendaciones"]
TipoValor = Literal["numero", "texto", "lista", "ninguno"]
Valor = Optional[Union[float, str, list[str]]]


class ItemRecomendacion(BaseModel):
    """Una sucursal recomendada para visitar."""

    sucursal_id: str = Field(description="ID de la sucursal, por ejemplo S012.")
    motivo: str = Field(description="Por qué esta sucursal necesita atención.")
    evidencia: str = Field(description="Cifras de las herramientas que respaldan el motivo.")
    accion_sugerida: str = Field(description="Qué debería hacer el gerente.")


class RespuestaFinal(BaseModel):
    """Respuesta final al gerente. Llamala una sola vez, cuando ya tengas todas las cifras."""

    tipo_respuesta: TipoRespuesta = Field(
        description="'recomendaciones' si la pregunta pide recomendaciones; si no, 'dato'."
    )
    texto: Optional[str] = Field(
        default=None, description="Respuesta en español para el gerente (si tipo_respuesta = dato)."
    )
    valor: Valor = Field(
        default=None,
        description=(
            "Valor principal para la evaluación: número (punto decimal, sin separador de miles), "
            "texto corto (palabra clave) o lista de IDs en orden. None si no aplica o no hay datos."
        ),
    )
    tipo_valor: TipoValor = Field(default="ninguno", description="Tipo de `valor`.")
    recomendaciones: Optional[list[ItemRecomendacion]] = Field(
        default=None, description="Solo si tipo_respuesta = recomendaciones."
    )


class Respuesta(BaseModel):
    """Lo que devuelve `responder()`: la respuesta final más la traza de ejecución."""

    tipo_respuesta: TipoRespuesta
    texto: Optional[str] = None
    valor: Valor = None
    tipo_valor: TipoValor = "ninguno"
    recomendaciones: Optional[list[ItemRecomendacion]] = None
    traza: dict = Field(default_factory=dict)

    def salida_para_usuario(self) -> str:
        """Texto que se imprime: JSON puro si son recomendaciones, si no el texto."""
        import json

        if self.tipo_respuesta == "recomendaciones":
            items = [r.model_dump() for r in (self.recomendaciones or [])]
            return json.dumps(items, ensure_ascii=False, indent=2)
        return self.texto or ""

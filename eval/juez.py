"""LLM juez para preguntas de texto largo (Etapa 5).

Se usa solo cuando las reglas deterministas de `eval.comparadores` no alcanzan (por ejemplo, el
resumen de P10). Modelo: `agente.config.modelo_juez()` con salida estructurada `Veredicto`
(temperatura 0 si el modelo la acepta; Haiku 5.5 no). El modelo se cachea. Para tests se inyecta un juez falso.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from agente.config import costo_usd, modelo_juez


class Veredicto(BaseModel):
    """Veredicto del juez sobre una respuesta.

    `razon` va ANTES que `correcto`: el modelo completa los campos en orden, así que primero razona y
    después decide. Con el orden inverso el booleano salía antes del razonamiento y podía contradecirlo
    (eval del 2026-11-23: la razón concluía "sí" y `correcto` era False).
    """

    razon: str = Field(description="Justificación breve, en español. Escribila antes de decidir.")
    correcto: bool = Field(description="True si la respuesta contiene los hechos esperados y no los contradice. "
                                       "Tiene que ser coherente con la razón.")


RUBRICA = (
    "Sos un evaluador estricto de hechos. Te doy una pregunta, la respuesta esperada y la respuesta "
    "de un asistente. Criterio: ¿La respuesta contiene los hechos de la respuesta esperada (cifras, "
    "categorías, IDs)? ¿Contradice alguno? Ignorá diferencias de redacción y de formato numérico "
    "(81,7% y 82% son lo mismo redondeado). Ignorá también la forma: la longitud, la cantidad de "
    "frases y el detalle extra no restan, salvo que contradigan la esperada. Si además hay un 'valor "
    "estructurado', es un campo aparte que usa el sistema de evaluación, no parte del texto. "
    "Primero escribí una razón breve en español y después decidí: correcto=true solo si contiene los "
    "hechos esenciales y no contradice ninguno."
)


def armar_mensajes(pregunta: str, esperada: Any, obtenida: str) -> list[tuple[str, str]]:
    """Mensajes (system + user) para el juez."""
    usuario = (
        f"Pregunta:\n{pregunta}\n\n"
        f"Respuesta esperada:\n{esperada}\n\n"
        f"Respuesta del asistente:\n{obtenida}"
    )
    return [("system", RUBRICA), ("human", usuario)]


@lru_cache(maxsize=4)
def _modelo_juez_cacheado(nombre: str):
    """Modelo juez con salida estructurada (cacheado por nombre)."""
    from langchain.chat_models import init_chat_model

    from agente.config import kwargs_modelo

    # temperature=0 salvo en modelos que no la aceptan (Haiku 5.5, el juez por defecto, y Sonnet 5.5).
    return init_chat_model(nombre, **kwargs_modelo(nombre)).with_structured_output(Veredicto, include_raw=True)


def obtener_juez():
    """Juez real (requiere clave del proveedor)."""
    return _modelo_juez_cacheado(modelo_juez())


def _uso_de_mensaje(mensaje: Any) -> dict:
    """Tokens y costo de un AIMessage, si están disponibles."""
    meta = getattr(mensaje, "usage_metadata", None) or {}
    entrada = int(meta.get("input_tokens", 0) or 0)
    salida = int(meta.get("output_tokens", 0) or 0)
    if not entrada and not salida:
        return {}
    return {
        "tokens_entrada": entrada,
        "tokens_salida": salida,
        "costo_usd": costo_usd(modelo_juez(), entrada, salida),
    }


def _a_veredicto(x: Any) -> Veredicto:
    if isinstance(x, Veredicto):
        return x
    if isinstance(x, bool):
        return Veredicto(correcto=x, razon="(juez sin razón)")
    if isinstance(x, dict):
        return Veredicto.model_validate(x)
    raise TypeError(f"salida del juez no reconocida: {type(x).__name__}")


def formatear_obtenida(texto: str, valor: str | None = None) -> str:
    """Respuesta del asistente para el juez: el texto y, aparte y rotulado, el valor estructurado.

    No se concatenan sin rótulo: un valor como "limpieza" pegado al final del texto se lee como un
    fragmento suelto y el juez lo penaliza.
    """
    texto = (texto or "").strip()
    valor = (valor or "").strip()
    if not valor or valor == texto:
        return texto
    if not texto:
        return f"Valor estructurado: {valor}"
    return f"{texto}\n\n(Valor estructurado, campo aparte para el eval: {valor})"


def juzgar(pregunta: str, esperada: Any, obtenida: str, *, valor: str | None = None,
           juez: Any = None) -> tuple[Veredicto, dict]:
    """Pide el veredicto al juez. Devuelve `(veredicto, uso)`; `uso` puede ser `{}`.

    `obtenida` es el texto de la respuesta; `valor`, si se pasa, es el valor estructurado, que se le
    muestra al juez aparte y rotulado (ver `formatear_obtenida`).

    `juez` puede ser:
    - un runnable con `.invoke(mensajes)` que devuelve `Veredicto`, un dict con `parsed`/`raw`
      (formato de `include_raw=True`) o un dict con los campos del veredicto;
    - una función `(pregunta, esperada, obtenida) -> Veredicto | bool | dict` (útil en tests);
    - None: se usa el juez real cacheado.
    """
    obtenida = formatear_obtenida(obtenida, valor)
    if juez is None:
        juez = obtener_juez()
    if hasattr(juez, "invoke"):
        salida = juez.invoke(armar_mensajes(pregunta, esperada, obtenida))
    else:
        salida = juez(pregunta, esperada, obtenida)
    uso: dict = {}
    if isinstance(salida, dict) and "parsed" in salida:
        if salida.get("parsed") is None:
            raise ValueError(f"el juez no devolvió un veredicto válido: {salida.get('parsing_error')}")
        uso = _uso_de_mensaje(salida.get("raw"))
        salida = salida["parsed"]
    return _a_veredicto(salida), uso

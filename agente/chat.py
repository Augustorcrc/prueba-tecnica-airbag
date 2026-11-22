"""Modo chat (opcional): varias preguntas seguidas con memoria de la conversación.

Complementa a `responder()`, que sigue siendo la forma de contestar una pregunta suelta (la que usan
el eval y `python main.py "pregunta"`). Cada turno llama a `responder()` con:

- `historial`: las últimas preguntas y respuestas, como mensajes Human/AI compactos (solo el texto
  final, no los resultados de herramientas), para que "¿y en marzo?" se entienda.
- `llamadas_previas`: los resultados de herramientas de los turnos anteriores, que no se mandan al
  modelo pero sí los mira el verificador. Así una cifra que ya salió antes no cuenta como inventada.

La memoria es solo de la sesión (en memoria del proceso): no se guarda entre ejecuciones.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agente.esquemas import Respuesta

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models import BaseChatModel

    from agente.datos import Datos

MAX_TURNOS_HISTORIAL = 6  # turnos (pregunta + respuesta) que ve el modelo
MAX_LLAMADAS_MEMORIA = 60  # resultados de herramientas que conserva el verificador
LARGO_MAX_RESPUESTA_HISTORIAL = 2000  # caracteres de cada respuesta guardada en el historial

AYUDA = "Comandos: /nuevo (borra la memoria), /salir (o Ctrl+D)."


class SesionChat:
    """Conversación con memoria. `preguntar()` devuelve la misma `Respuesta` que `responder()`."""

    def __init__(self, *, modelo: "BaseChatModel | None" = None, datos: "Datos | None" = None,
                 max_turnos: int = MAX_TURNOS_HISTORIAL) -> None:
        self.modelo = modelo
        self.datos = datos
        self.max_turnos = max_turnos
        self.turnos: list[tuple[str, str]] = []  # (pregunta, respuesta mostrada)
        self.llamadas: list[dict] = []

    def reiniciar(self) -> None:
        self.turnos.clear()
        self.llamadas.clear()

    def historial(self) -> list[BaseMessage]:
        mensajes: list[BaseMessage] = []
        for pregunta, respuesta in self.turnos[-self.max_turnos:]:
            mensajes.append(HumanMessage(pregunta))
            mensajes.append(AIMessage(respuesta[:LARGO_MAX_RESPUESTA_HISTORIAL] or "(sin respuesta)"))
        return mensajes

    def preguntar(self, pregunta: str) -> Respuesta:
        from agente.agente import responder

        modelo = self.modelo
        if modelo is None:  # el modelo real se crea una vez por sesión
            from agente.agente import crear_modelo

            modelo = self.modelo = crear_modelo()
        respuesta = responder(pregunta, modelo=modelo, datos=self.datos,
                              historial=self.historial(), llamadas_previas=self.llamadas)
        del self.llamadas[:-MAX_LLAMADAS_MEMORIA]
        self.turnos.append((pregunta, respuesta.salida_para_usuario()))
        return respuesta


def bucle_chat(sesion: SesionChat, *, leer: Callable[[str], str] = input,
               escribir: Callable[[str], None] = print, mostrar_traza: bool = False) -> int:
    """Loop interactivo. Devuelve el código de salida. `leer`/`escribir` se inyectan en los tests."""
    import json

    from agente.agente import es_rate_limit

    escribir("Chat con el asistente de Café Tostadero (datos del 2026-01-05 al 2026-03-29). " + AYUDA)
    while True:
        try:
            pregunta = leer("\nVos: ").strip()
        except (EOFError, KeyboardInterrupt):
            escribir("")
            return 0
        if not pregunta:
            continue
        comando = pregunta.lower()
        if comando in {"/salir", "salir", "exit", "quit"}:
            return 0
        if comando == "/nuevo":
            sesion.reiniciar()
            escribir("Memoria borrada: empezamos de nuevo.")
            continue
        if comando in {"/ayuda", "/help"}:
            escribir(AYUDA)
            continue
        try:
            respuesta = sesion.preguntar(pregunta)
        except Exception as e:  # noqa: BLE001  (el chat no se corta por un error de un turno)
            if es_rate_limit(e):
                escribir(f"Asistente: límite de tasa del proveedor ({e}). Probá de nuevo en unos segundos.")
            else:
                escribir(f"Asistente: error: {e}")
            continue
        escribir(f"Asistente: {respuesta.salida_para_usuario()}")
        if respuesta.traza.get("error"):
            escribir(f"(aviso: hubo un error del proveedor: {respuesta.traza['error']})")
        if mostrar_traza:
            escribir(json.dumps(respuesta.traza, ensure_ascii=False, indent=2, default=str))

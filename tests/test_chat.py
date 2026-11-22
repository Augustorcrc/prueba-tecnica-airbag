"""Modo chat (opcional): memoria entre turnos sin cambiar el modo de una pregunta."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agente.agente import responder
from agente.chat import SesionChat, bucle_chat
from agente.esquemas import Respuesta
from tests.fakes import ModeloFalso, ai, final, llamada


@pytest.fixture(autouse=True)
def _trazas_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAZAS_DIR", str(tmp_path))


def _guion():
    return [
        ai(llamada("contar_quejas", region="Sur", mes="2026-02")),
        final(tipo_respuesta="dato", texto="El Sur tuvo 442 quejas en febrero de 2026.", valor=442,
              tipo_valor="numero"),
        # turno 2: repregunta que reutiliza la cifra del turno 1 sin volver a consultar
        final(tipo_respuesta="dato", texto="Sí: fueron 442 quejas, las de febrero que te pasé recién.",
              valor=442, tipo_valor="numero"),
    ]


def test_dos_turnos_con_memoria(datos):
    modelo = ModeloFalso(_guion())
    sesion = SesionChat(modelo=modelo, datos=datos)

    r1 = sesion.preguntar("¿Cuántas quejas recibió el Sur en febrero de 2026?")
    assert r1.valor == 442 and r1.traza["verificacion"]["ok"]
    assert [c["herramienta"] for c in r1.traza["llamadas"]] == ["contar_quejas"]

    r2 = sesion.preguntar("¿Me confirmás el número?")
    assert r2.valor == 442
    assert r2.traza["verificacion"]["ok"], r2.traza["verificacion"]  # la cifra sale del turno anterior
    assert "⚠️" not in (r2.texto or "")
    assert r2.traza["llamadas"] == []  # la traza del turno 2 solo tiene sus propias llamadas

    # el modelo, en el turno 2, recibió el historial del turno 1
    mensajes = modelo.recibidos[-1]
    humanos = [m.content for m in mensajes if isinstance(m, HumanMessage)]
    assert humanos[0].startswith("¿Cuántas quejas recibió el Sur") and humanos[-1] == "¿Me confirmás el número?"
    assert any(isinstance(m, AIMessage) and "442" in str(m.content) and not m.tool_calls for m in mensajes)


def test_sin_sesion_la_misma_repregunta_no_verifica(datos):
    """Control: con responder() suelto (modo de siempre) no hay memoria, así que la cifra no verifica."""
    guion = _guion()[2:] + _guion()[2:]  # dos intentos (con el reintento por feedback)
    r = responder("¿Me confirmás el número?", modelo=ModeloFalso(guion), datos=datos)
    assert not r.traza["verificacion"]["ok"]


def test_reiniciar_borra_la_memoria(datos):
    sesion = SesionChat(modelo=ModeloFalso(_guion()), datos=datos)
    sesion.preguntar("¿Cuántas quejas recibió el Sur en febrero de 2026?")
    assert sesion.turnos and sesion.llamadas
    sesion.reiniciar()
    assert sesion.turnos == [] and sesion.llamadas == [] and sesion.historial() == []


def test_historial_acotado(datos):
    sesion = SesionChat(modelo=ModeloFalso([]), datos=datos, max_turnos=2)
    sesion.turnos = [(f"p{i}", f"r{i}") for i in range(5)]
    assert [m.content for m in sesion.historial()] == ["p3", "r3", "p4", "r4"]


class _SesionFalsa:
    def __init__(self):
        self.preguntas = []
        self.reinicios = 0

    def preguntar(self, pregunta):
        self.preguntas.append(pregunta)
        if pregunta == "rompe":
            raise RuntimeError("falla de prueba")
        return Respuesta(tipo_respuesta="dato", texto=f"respuesta a {pregunta}", tipo_valor="ninguno")

    def reiniciar(self):
        self.reinicios += 1


def test_bucle_chat_comandos():
    entradas = iter(["hola", "", "rompe", "/nuevo", "otra", "/salir", "no llega"])
    salida = []
    sesion = _SesionFalsa()
    codigo = bucle_chat(sesion, leer=lambda _: next(entradas), escribir=salida.append)
    assert codigo == 0
    assert sesion.preguntas == ["hola", "rompe", "otra"] and sesion.reinicios == 1
    texto = "\n".join(salida)
    assert "respuesta a hola" in texto and "error: falla de prueba" in texto and "Memoria borrada" in texto


def test_bucle_chat_sale_con_ctrl_d():
    def leer(_):
        raise EOFError

    assert bucle_chat(_SesionFalsa(), leer=leer, escribir=lambda _: None) == 0


def test_main_chat_usa_el_bucle(monkeypatch):
    import main
    from agente import chat

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-falsa")
    llamado = {}
    monkeypatch.setattr(chat, "bucle_chat", lambda sesion, mostrar_traza=False: llamado.setdefault("ok", 0))
    assert main.main(["--chat"]) == 0 and "ok" in llamado


def test_main_sin_argumentos_sigue_pidiendo_pregunta(monkeypatch, capsys):
    import main

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-falsa")
    assert main.main([]) == 2
    assert "--chat" in capsys.readouterr().err

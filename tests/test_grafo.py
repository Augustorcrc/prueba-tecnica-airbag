"""Tests del grafo (WP7) con modelo falso: sin clave ni red."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agente import nodos
from agente.config import MAX_PASOS, RECURSION_LIMIT
from agente.esquemas import RespuestaFinal
from agente.estado import estado_inicial
from agente.grafo import config_invocacion, construir_grafo
from tests.fakes import (
    OK,
    ModeloFalso,
    VerificadorFalso,
    ai,
    contar_quejas_stub,
    final,
    herramienta_rota,
    llamada,
    ranking_stub,
)

RAIZ = Path(__file__).resolve().parent.parent
PREGUNTA = "¿Cuántas quejas tuvo la región Norte?"
HERRAMIENTAS = [contar_quejas_stub, ranking_stub, herramienta_rota]


def correr(guion, *, verificador=None, ids_validos=None, pregunta=PREGUNTA, **kw):
    modelo = ModeloFalso(guion)
    verificador = verificador if verificador is not None else VerificadorFalso()
    grafo = construir_grafo(modelo, HERRAMIENTAS, verificador=verificador, ids_validos=ids_validos, **kw)
    inicial = estado_inicial(pregunta, [SystemMessage("system prompt"), HumanMessage(pregunta)])
    salida = grafo.invoke(inicial, config_invocacion())
    chequear_invariante_tool_messages(modelo)
    return salida, modelo, verificador


def chequear_invariante_tool_messages(modelo: ModeloFalso) -> None:
    """Antes de cada invocación del modelo, toda tool_call previa tiene su ToolMessage."""
    for mensajes in modelo.recibidos:
        respondidas = {m.tool_call_id for m in mensajes if isinstance(m, ToolMessage)}
        for i, m in enumerate(mensajes):
            if isinstance(m, AIMessage):
                for c in m.tool_calls:
                    assert c["id"] in respondidas, f"tool_call {c['name']} ({c['id']}) sin ToolMessage"
                    # y el ToolMessage viene antes de cualquier AIMessage posterior
                    siguiente_ai = next(
                        (j for j in range(i + 1, len(mensajes)) if isinstance(mensajes[j], AIMessage)), len(mensajes)
                    )
                    ids_entre = {
                        x.tool_call_id for x in mensajes[i + 1 : siguiente_ai] if isinstance(x, ToolMessage)
                    }
                    assert c["id"] in ids_entre


def final_dato(texto, valor, tipo_valor="numero"):
    return final(tipo_respuesta="dato", texto=texto, valor=valor, tipo_valor=tipo_valor)


# --------------------------------------------------------------------------- camino feliz


def test_camino_feliz_acumula_uso_pasos_y_llamadas():
    salida, modelo, verificador = correr(
        [ai(llamada("contar_quejas_stub", region="Norte"), entrada=100, salida=20), final_dato("Hubo 137 quejas.", 137)]
    )
    assert salida["respuesta"] == RespuestaFinal(tipo_respuesta="dato", texto="Hubo 137 quejas.", valor=137, tipo_valor="numero")
    assert salida["pasos"] == 2
    assert salida["uso"] == {"tokens_entrada": 200, "tokens_salida": 40}
    assert salida["llamadas"][0]["herramienta"] == "contar_quejas_stub"
    assert salida["llamadas"][0]["args"] == {"region": "Norte"}
    assert salida["llamadas"][0]["resultado"]["resultado"]["quejas"] == 137
    assert salida["verificacion"]["ok"] is True
    assert salida["advertencias"] == []
    # el verificador recibió los resultados de las herramientas y la pregunta
    assert verificador.llamadas[0]["pregunta"] == PREGUNTA
    assert verificador.llamadas[0]["resultados"] == [salida["llamadas"][0]["resultado"]]
    # ligado con RespuestaFinal y tool_choice any
    assert modelo.tool_choice == "any" and RespuestaFinal in modelo.ligadas
    # el ToolMessage lleva JSON con ensure_ascii=False
    tool_msg = next(m for m in salida["messages"] if isinstance(m, ToolMessage))
    assert json.loads(tool_msg.content)["resultado"]["quejas"] == 137


def test_tool_message_sin_escapar_acentos():
    estado = {"messages": [ai(llamada("contar_quejas_stub", region="Señorío"))], "llamadas": []}
    salida = nodos.crear_nodo_herramientas([contar_quejas_stub])(estado)
    assert "Señorío" in salida["messages"][0].content


# --------------------------------------------------------------------------- verificación


def test_cifra_inventada_feedback_y_correccion():
    salida, modelo, _ = correr(
        [
            ai(llamada("contar_quejas_stub", region="Norte")),
            final_dato("Hubo 999 quejas.", 999),
            final_dato("Hubo 137 quejas.", 137),
        ]
    )
    assert salida["respuesta"].texto == "Hubo 137 quejas."
    assert salida["intentos_verificacion"] == 1
    assert salida["pasos"] == 3
    feedback = modelo.recibidos[-1][-1]
    assert isinstance(feedback, ToolMessage)
    assert "999" in feedback.content and "volvé a consultar y corregí" in feedback.content


def test_cifra_inventada_dos_veces_cierra_con_advertencia():
    salida, _, _ = correr(
        [
            ai(llamada("contar_quejas_stub", region="Norte")),
            final_dato("Hubo 999 quejas.", 999),
            final_dato("Hubo 998 quejas.", 998),
        ]
    )
    texto = salida["respuesta"].texto
    assert texto.startswith("Hubo 998 quejas.")
    assert texto.splitlines()[-1] == "⚠️ No pude verificar la cifra 998 contra los datos"
    assert salida["advertencias"]


def test_recomendaciones_descarta_item_no_verificado():
    def recos(cifra_s007):
        return final(
            tipo_respuesta="recomendaciones",
            recomendaciones=[
                {"sucursal_id": "S012", "motivo": "Tasa más alta", "evidencia": "tasa 4.5, 61 quejas", "accion_sugerida": "Visitar"},
                {"sucursal_id": "S007", "motivo": "Muchas quejas", "evidencia": f"{cifra_s007} quejas", "accion_sugerida": "Visitar"},
            ],
        )

    salida, _, _ = correr(
        [ai(llamada("ranking_stub", n=3)), recos(777), recos(778)],
        ids_validos={"S012", "S007", "S021"},
        pregunta="¿Qué sucursales debería visitar?",
    )
    respuesta = salida["respuesta"]
    assert [r.sucursal_id for r in respuesta.recomendaciones] == ["S012"]
    assert respuesta.texto is None  # no se agrega texto
    assert any("S007" in a for a in salida["advertencias"])


def test_recomendaciones_descarta_id_invalido_y_repetido_y_llama_hook():
    item = {"motivo": "Tasa alta", "evidencia": "tasa 4.5", "accion_sugerida": "Visitar"}
    recos = lambda: final(  # noqa: E731
        tipo_respuesta="recomendaciones",
        recomendaciones=[{"sucursal_id": "S012", **item}, {"sucursal_id": "S999", **item}, {"sucursal_id": "S012", **item}],
    )
    vistos = {}

    def completar(resp, descartados, estado):
        vistos["descartados"] = [d.sucursal_id for d in descartados]
        return resp

    salida, _, _ = correr(
        [ai(llamada("ranking_stub", n=3)), recos(), recos()],
        ids_validos={"S012", "S007", "S021"},
        completar_recomendaciones=completar,
    )
    assert [r.sucursal_id for r in salida["respuesta"].recomendaciones] == ["S012"]
    assert sorted(vistos["descartados"]) == ["S012", "S999"]


def test_ids_validos_none_no_valida_ids():
    verificador = VerificadorFalso([{"ok": False, "no_verificados": [], "ids_invalidos": ["S999"], "ids_repetidos": []}])
    salida, _, _ = correr(
        [final(tipo_respuesta="recomendaciones", recomendaciones=[
            {"sucursal_id": "S999", "motivo": "m", "evidencia": "e", "accion_sugerida": "a"}])],
        verificador=verificador,
        ids_validos=None,
    )
    assert salida["verificacion"]["ok"] is True
    assert verificador.llamadas[0]["ids_validos"] == set()


def test_verificador_por_defecto_se_importa_en_tiempo_de_llamada(monkeypatch):
    llamado = []

    def falso(respuesta, resultados, pregunta, ids_validos):
        llamado.append(True)
        return OK

    monkeypatch.setattr("agente.verificador.verificar_respuesta", falso)
    modelo = ModeloFalso([final_dato("Sin datos.", None, "ninguno")])
    grafo = construir_grafo(modelo, HERRAMIENTAS)
    salida = grafo.invoke(estado_inicial(PREGUNTA), config_invocacion())
    assert llamado and salida["verificacion"]["ok"]


# --------------------------------------------------------------------------- RespuestaFinal mal usada


def test_respuesta_final_invalida_vuelve_a_modelo():
    salida, modelo, _ = correr(
        [
            ai(llamada("contar_quejas_stub", region="Norte")),
            final(tipo_respuesta="opinion", texto="Hubo 137 quejas."),
            final_dato("Hubo 137 quejas.", 137),
        ]
    )
    assert salida["respuesta"].valor == 137
    error = modelo.recibidos[-1][-1]
    assert isinstance(error, ToolMessage) and "error" in json.loads(error.content)
    assert salida["intentos_verificacion"] == 0


def test_respuesta_final_junto_con_otra_herramienta_se_ignora():
    salida, modelo, _ = correr(
        [
            ai(llamada("contar_quejas_stub", region="Norte"), llamada("RespuestaFinal", tipo_respuesta="dato", texto="Hubo 5.", valor=5, tipo_valor="numero")),
            final_dato("Hubo 137 quejas.", 137),
        ]
    )
    assert salida["respuesta"].valor == 137
    assert [c["herramienta"] for c in salida["llamadas"]] == ["contar_quejas_stub"]
    tool_msgs = [m for m in modelo.recibidos[1] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2
    assert any(m.content == nodos.MENSAJE_FINAL_IGNORADA for m in tool_msgs)


def test_dos_respuestas_finales_en_el_mismo_turno():
    salida, _, _ = correr(
        [ai(llamada("RespuestaFinal", tipo_respuesta="dato", texto="a", valor=None, tipo_valor="ninguno"),
            llamada("RespuestaFinal", tipo_respuesta="dato", texto="b", valor=None, tipo_valor="ninguno"))]
    )
    assert salida["respuesta"].texto == "a"


def test_texto_sin_tool_calls_recibe_recordatorio():
    salida, modelo, _ = correr([ai(texto="Creo que fueron 137."), final_dato("Sin datos suficientes.", None, "ninguno")])
    assert salida["respuesta"].texto == "Sin datos suficientes."
    recordatorio = modelo.recibidos[1][-1]
    assert isinstance(recordatorio, HumanMessage) and recordatorio.content == nodos.RECORDATORIO
    assert salida["pasos"] == 2


# --------------------------------------------------------------------------- límites y errores


def test_corte_por_pasos():
    guion = [ai(llamada("contar_quejas_stub", region="Norte")) for _ in range(MAX_PASOS)]
    salida, modelo, _ = correr(guion)
    assert modelo.n_invocaciones == MAX_PASOS
    assert salida["pasos"] == MAX_PASOS
    assert salida["respuesta"] == RespuestaFinal(
        tipo_respuesta="dato", texto=nodos.TEXTO_SIN_RESPUESTA, valor=None, tipo_valor="ninguno"
    )
    assert salida["advertencias"]


def test_ultimo_paso_con_respuesta_final_se_procesa():
    guion = [ai(llamada("contar_quejas_stub", region="Norte")) for _ in range(MAX_PASOS - 1)]
    guion.append(final_dato("Hubo 137 quejas.", 137))
    salida, _, _ = correr(guion)
    assert salida["respuesta"].valor == 137 and salida["pasos"] == MAX_PASOS


def test_peor_caso_no_supera_recursion_limit():
    """Mezcla de recordatorios, finales inválidas y verificación fallida dentro de RECURSION_LIMIT supersteps."""
    relleno = [
        lambda: ai(texto="hola"),
        lambda: final(tipo_respuesta="malo"),
        lambda: ai(llamada("contar_quejas_stub", region="Sur")),
    ]
    guion = [
        ai(texto="hola"),
        final(tipo_respuesta="malo"),
        ai(llamada("contar_quejas_stub", region="Norte")),
        final_dato("Hubo 999.", 999),  # falla la verificación → feedback (el camino más largo)
    ]
    while len(guion) < MAX_PASOS - 1:
        guion.append(relleno[len(guion) % len(relleno)]())
    guion.append(final_dato("Hubo 998.", 998))  # segunda falla → cerrar_con_advertencia
    assert len(guion) == MAX_PASOS
    salida, modelo, _ = correr(guion)
    assert modelo.n_invocaciones == MAX_PASOS
    assert "⚠️ No pude verificar la cifra 998" in salida["respuesta"].texto


def test_tool_choice_segun_modelo():
    class ConNombre(ModeloFalso):
        def __init__(self, guion, model):
            super().__init__(guion)
            self.model = model

    for nombre, esperado in [
        ("claude-sonnet-5-5", "auto"),
        ("claude-opus-5-5", "auto"),
        ("claude-fable-5-1", "auto"),
        ("claude-haiku-5-5", "any"),
    ]:
        modelo = ConNombre([], nombre)
        construir_grafo(modelo, HERRAMIENTAS)
        assert modelo.tool_choice == esperado, nombre
    # sin nombre (modelo falso) → "any"; el valor explícito manda
    sin_nombre = ModeloFalso([])
    construir_grafo(sin_nombre, HERRAMIENTAS)
    assert sin_nombre.tool_choice == "any"
    explicito = ConNombre([], "claude-haiku-5-5")
    construir_grafo(explicito, HERRAMIENTAS, tool_choice="auto")
    assert explicito.tool_choice == "auto"


def test_texto_libre_con_auto_pasa_por_recordatorio():
    """Con tool_choice="auto" el modelo puede contestar texto: el recordatorio lo reencauza."""
    salida, modelo, _ = correr(
        [ai(texto="Son 137."), ai(llamada("contar_quejas_stub", region="Norte")), final_dato("Hubo 137 quejas.", 137)],
        tool_choice="auto",
    )
    assert modelo.tool_choice == "auto"
    assert salida["respuesta"].valor == 137
    assert any(isinstance(m, HumanMessage) and m.content == nodos.RECORDATORIO for m in modelo.recibidos[1])


def test_herramienta_que_lanza_excepcion():
    salida, modelo, _ = correr([ai(llamada("herramienta_rota", x="a")), final_dato("No pude consultar.", None, "ninguno")])
    error = json.loads(modelo.recibidos[1][-1].content)
    assert "se cayó la base" in error["error"] and error["valores_validos"] == []
    assert salida["llamadas"][0]["resultado"]["error"]
    assert salida["respuesta"].texto == "No pude consultar."


def test_herramienta_desconocida_y_args_invalidos():
    nodo = nodos.crear_nodo_herramientas(HERRAMIENTAS)
    salida = nodo({"messages": [ai(llamada("no_existe"), llamada("ranking_stub", n="muchos"))], "llamadas": []})
    desconocida, invalida = (json.loads(m.content) for m in salida["messages"])
    assert "contar_quejas_stub" in desconocida["valores_validos"]
    assert "error" in invalida
    assert len(salida["llamadas"]) == 2


# --------------------------------------------------------------------------- config y diagrama


def test_config_invocacion():
    assert config_invocacion() == {"recursion_limit": RECURSION_LIMIT}
    assert MAX_PASOS == 10 and RECURSION_LIMIT == 30


def test_diagrama_mermaid_actualizado():
    grafo = construir_grafo(ModeloFalso([]), HERRAMIENTAS, verificador=VerificadorFalso())
    mermaid = grafo.get_graph().draw_mermaid()
    archivo = RAIZ / "docs" / "grafo.mmd"
    assert archivo.exists(), "falta docs/grafo.mmd (python -m tests.test_grafo lo genera)"
    assert archivo.read_text(encoding="utf-8") == mermaid
    for nodo in ["modelo", "herramientas", "extraer_respuesta", "verificar", "feedback", "cerrar_con_advertencia"]:
        assert nodo in mermaid


if __name__ == "__main__":  # regenera el diagrama
    g = construir_grafo(ModeloFalso([]), HERRAMIENTAS, verificador=VerificadorFalso())
    (RAIZ / "docs").mkdir(exist_ok=True)
    (RAIZ / "docs" / "grafo.mmd").write_text(g.get_graph().draw_mermaid(), encoding="utf-8")
    print("docs/grafo.mmd generado")

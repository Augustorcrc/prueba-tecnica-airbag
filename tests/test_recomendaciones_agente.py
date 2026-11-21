"""Tests de la integración de recomendaciones con el agente (WP-R).

Offline con `ModeloFalso` y datos reales; los que usan el modelo real se saltean sin clave.
"""

from __future__ import annotations

import json
import os

import pytest

from agente.agente import alinear_con_candidatos, item_deterministico, responder
from agente.recomendaciones import PREFIJO_PREVENTIVA, candidatos_visita
from agente.verificador import verificar_respuesta
from agente.esquemas import RespuestaFinal
from tests.fakes import ModeloFalso, ai, final, llamada

SIN_CLAVE = not os.environ.get("ANTHROPIC_API_KEY", "").strip()
CLAVES = {"sucursal_id", "motivo", "evidencia", "accion_sugerida"}
CADENA = ["S112", "S012", "S067", "S056", "S024"]
PREGUNTA = "¿Qué sucursales debería visitar esta semana?"


@pytest.fixture(autouse=True)
def trazas_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAZAS_DIR", str(tmp_path / "trazas"))


def _n(x):
    return str(x).replace(".", ",")


def item_llm(c: dict, *, tasa=None) -> dict:
    """Ítem como lo redactaría el LLM, con cifras copiadas del candidato (o una inventada)."""
    return {
        "sucursal_id": c["sucursal_id"],
        "motivo": f"Su tasa es {_n(c['ratio_regional'])} veces la de la región {c['region']}.",
        "evidencia": (
            f"Tasa 4 semanas {_n(tasa if tasa is not None else c['tasa_4sem'])} vs "
            f"{_n(c['tasa_region_4sem'])} de la región; {c['quejas_4sem']} quejas; "
            f"{c['categoria_dominante']} {_n(c['pct_categoria_dominante'])}%."
        ),
        "accion_sugerida": c["accion_sugerida_base"],
    }


def final_recos(items):
    return final(tipo_respuesta="recomendaciones", recomendaciones=items)


def candidatos(datos, **kw):
    return candidatos_visita(datos, **kw)["resultado"]["candidatos"]


def ids(r):
    return [i.sucursal_id for i in r.recomendaciones]


def chequear_json(r):
    salida = json.loads(r.salida_para_usuario())
    assert isinstance(salida, list) and salida
    for it in salida:
        assert isinstance(it, dict) and set(it) == CLAVES
        assert all(isinstance(v, str) and v for v in it.values())
    return salida


# --------------------------------------------------------------------------- offline


def test_cadena_json_puro_con_los_5(datos):
    cs = candidatos(datos)
    guion = [ai(llamada("candidatos_visita")), final_recos([item_llm(c) for c in cs])]
    r = responder(PREGUNTA, modelo=ModeloFalso(guion), datos=datos)
    assert r.tipo_respuesta == "recomendaciones" and r.texto is None
    salida_txt = r.salida_para_usuario()
    assert salida_txt.lstrip().startswith("[") and salida_txt.rstrip().endswith("]")
    salida = chequear_json(r)
    assert [i["sucursal_id"] for i in salida] == CADENA
    assert r.traza["verificacion"]["ok"] is True
    assert r.traza["advertencias"] == []


def test_cifra_inventada_dos_veces_se_completa_con_el_siguiente(datos):
    cs = candidatos(datos)
    items = [item_llm(c, tasa=9.99 if c["sucursal_id"] == "S067" else None) for c in cs]
    guion = [ai(llamada("candidatos_visita")), final_recos(items), final_recos(items)]
    r = responder(PREGUNTA, modelo=ModeloFalso(guion), datos=datos)
    siguiente = candidatos(datos, n=6)[5]
    assert ids(r) == ["S112", "S012", "S056", "S024", siguiente["sucursal_id"]]
    assert any("S067" in a for a in r.traza["advertencias"])
    nuevo = r.recomendaciones[-1]
    assert nuevo.evidencia == item_deterministico(siguiente, "2026-03-02..2026-03-29").evidencia
    # la evidencia determinística verifica contra el resultado de candidatos_visita
    res = verificar_respuesta(
        RespuestaFinal(tipo_respuesta="recomendaciones", recomendaciones=[nuevo]),
        [candidatos_visita(datos, n=6)], PREGUNTA, {c["sucursal_id"] for c in candidatos(datos, n=6)},
    )
    assert res["ok"], res
    chequear_json(r)


def test_orden_cambiado_por_el_llm_se_corrige(datos):
    cs = candidatos(datos)
    items = [item_llm(c) for c in reversed(cs)]
    guion = [ai(llamada("candidatos_visita")), final_recos(items)]
    r = responder(PREGUNTA, modelo=ModeloFalso(guion), datos=datos)
    assert ids(r) == CADENA
    assert any("orden" in a for a in r.traza["advertencias"])
    # se conserva el texto del LLM de cada sucursal
    por_id = {i["sucursal_id"]: i for i in items}
    assert all(it.evidencia == por_id[it.sucursal_id]["evidencia"] for it in r.recomendaciones)


def test_sucursal_ajena_se_reemplaza_por_el_candidato(datos):
    cs = candidatos(datos)
    items = [item_llm(c) for c in cs]
    items[2] = {**items[2], "sucursal_id": "S001"}  # el LLM metió otra sucursal (válida, no candidata)
    guion = [ai(llamada("candidatos_visita")), final_recos(items)]
    r = responder(PREGUNTA, modelo=ModeloFalso(guion), datos=datos)
    assert ids(r) == CADENA
    assert any("S001" in a for a in r.traza["advertencias"])
    assert r.recomendaciones[2] == item_deterministico(cs[2], "2026-03-02..2026-03-29")


def test_region_costa(datos):
    cs = candidatos(datos, region="Costa")
    guion = [ai(llamada("candidatos_visita", region="Costa")), final_recos([item_llm(c) for c in cs])]
    r = responder("¿Qué sucursales de Costa necesitan atención?", modelo=ModeloFalso(guion), datos=datos)
    region = dict(zip(datos.sucursales["sucursal_id"], datos.sucursales["region"]))
    assert 3 <= len(r.recomendaciones) <= 5
    assert all(region[i] == "Costa" for i in ids(r))
    chequear_json(r)


def test_recomendaciones_sin_candidatos_visita_se_marca(datos):
    items = [{"sucursal_id": "S012", "motivo": "Muchas quejas.", "evidencia": "Ver datos.",
              "accion_sugerida": "Visitar."}]
    guion = [final_recos(items)]
    r = responder(PREGUNTA, modelo=ModeloFalso(guion), datos=datos)
    assert ids(r) == ["S012"]
    assert any("candidatos_visita" in a for a in r.traza["advertencias"])


def test_item_deterministico_preventiva_y_nuevas(datos):
    c = dict(candidatos(datos)[0], preventiva=True, supera_umbral=False)
    item = item_deterministico(c)
    assert item.motivo.startswith(PREFIJO_PREVENTIVA)
    nuevas = [c for c in candidatos(datos, incluir_nuevas=True) if any("abrió" in a for a in c["advertencias"])]
    assert nuevas
    assert "abrió el" in item_deterministico(nuevas[0]).evidencia


def test_alinear_no_toca_respuestas_de_dato():
    r = RespuestaFinal(tipo_respuesta="dato", texto="Hay 30.", valor=30, tipo_valor="numero")
    salida, adv = alinear_con_candidatos(r, {"llamadas": []})
    assert salida == r and adv == []


# --------------------------------------------------------------------------- con modelo real

PIDEN_RECOMENDACION = [
    "¿qué sucursales debería visitar?",
    "¿a qué locales tendría que ir esta semana?",
    "recomendame 3 sucursales para visitar",
    "¿qué debería revisar esta semana?",
    "¿dónde tengo que reforzar la limpieza?",
    "¿qué categoría de queja ataco primero y dónde?",
    "¿qué sucursales de Costa necesitan atención?",
    "dame un plan de visitas",
]
NO_PIDEN_RECOMENDACION = [
    "¿cuál es la peor sucursal?",
    "¿cuántas quejas hubo en febrero?",
    "¿qué canal recibe más quejas?",
    "¿cuál es la tasa de quejas del Norte?",
    "¿qué sucursales abrieron en marzo?",
]


@pytest.mark.parametrize("pregunta", PIDEN_RECOMENDACION)
@pytest.mark.requiere_clave
@pytest.mark.skipif(SIN_CLAVE, reason="falta ANTHROPIC_API_KEY")
def test_real_pide_recomendacion_da_json(pregunta, datos):
    r = responder(pregunta)
    assert r.tipo_respuesta == "recomendaciones", (pregunta, r.texto)
    salida = chequear_json(r)
    validos = set(datos.sucursales["sucursal_id"])
    assert all(i["sucursal_id"] in validos for i in salida)
    if "3 sucursales" in pregunta:
        assert len(salida) == 3
    if "Costa" in pregunta:
        region = dict(zip(datos.sucursales["sucursal_id"], datos.sucursales["region"]))
        assert all(region[i["sucursal_id"]] == "Costa" for i in salida)


@pytest.mark.parametrize("pregunta", NO_PIDEN_RECOMENDACION)
@pytest.mark.requiere_clave
@pytest.mark.skipif(SIN_CLAVE, reason="falta ANTHROPIC_API_KEY")
def test_real_no_pide_recomendacion_no_da_json(pregunta):
    r = responder(pregunta)
    assert r.tipo_respuesta == "dato", pregunta
    try:
        salida = json.loads(r.salida_para_usuario())
    except ValueError:
        salida = None
    assert not isinstance(salida, list)


@pytest.mark.requiere_clave
@pytest.mark.skipif(SIN_CLAVE, reason="falta ANTHROPIC_API_KEY")
def test_real_estabilidad():
    vistas = [ids(responder("¿qué sucursales debería visitar?")) for _ in range(3)]
    assert vistas[0] == vistas[1] == vistas[2] == CADENA


@pytest.mark.requiere_clave
@pytest.mark.skipif(SIN_CLAVE, reason="falta ANTHROPIC_API_KEY")
def test_real_incluir_nuevas_advierte_en_evidencia(datos):
    r = responder("¿qué sucursales debería visitar? Incluí también las sucursales nuevas.")
    assert r.tipo_respuesta == "recomendaciones"
    nuevas = {c["sucursal_id"] for c in candidatos(datos, incluir_nuevas=True)
              if any("abrió" in a for a in c["advertencias"])}
    recomendadas_nuevas = [i for i in r.recomendaciones if i.sucursal_id in nuevas]
    assert recomendadas_nuevas
    assert all(("abrió" in i.evidencia or "días de datos" in i.evidencia or "nueva" in i.evidencia.lower())
               for i in recomendadas_nuevas)


@pytest.mark.parametrize(
    "kw", [{}, {"region": "Costa"}, {"categoria": "limpieza"}, {"n": 10}, {"incluir_nuevas": True},
           {"region": "Norte", "categoria": "trato", "n": 5}],
)
def test_items_deterministicos_verifican_por_construccion(datos, kw):
    res = candidatos_visita(datos, **kw)
    periodo = res["resultado"]["parametros"]["ventana"]
    items = [item_deterministico(c, periodo) for c in res["resultado"]["candidatos"]]
    v = verificar_respuesta(RespuestaFinal(tipo_respuesta="recomendaciones", recomendaciones=items),
                            [res], "¿qué sucursales visito?", set(datos.sucursales["sucursal_id"]))
    assert v["ok"], v
    for item, c in zip(items, res["resultado"]["candidatos"]):
        assert item.motivo.startswith(PREFIJO_PREVENTIVA) == bool(c["preventiva"])

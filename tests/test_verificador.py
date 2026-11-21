"""Tests del verificador de cifras (contrato C4). No necesitan clave ni red."""

import json

import pytest

from agente.esquemas import ItemRecomendacion, RespuestaFinal
from agente.verificador import (
    extraer_ids,
    extraer_numeros,
    numeros_disponibles,
    verificar_numeros,
    verificar_respuesta,
)

TASA = [{"resultado": {"tasa": 2.1468}, "n_registros": 1}]


def c2(resultado, advertencias=None):
    """Resultado en formato C2."""
    return {
        "resultado": resultado,
        "filtros_aplicados": {},
        "periodo": {"desde": "2026-01-05", "hasta": "2026-03-29"},
        "n_registros": 42,
        "advertencias": advertencias or [],
    }


# --- Extracción ------------------------------------------------------------------------------


def test_extraer_formatos():
    cifras = extraer_numeros("2,15 y 2.15, 3.246.856 quejas, 82 % y +15%; -0,29 y −0,29")
    textos = [c.texto for c in cifras]
    assert textos == ["2,15", "2.15", "3.246.856", "82 %", "+15%", "-0,29", "−0,29"]
    assert cifras[2].valores == ((3246856.0, 0),)
    assert cifras[3].es_porcentaje and cifras[4].es_porcentaje
    assert cifras[5].valores[0][0] == pytest.approx(-0.29)
    assert cifras[6].valores[0][0] == pytest.approx(-0.29)


def test_extraer_ambiguo_miles_o_decimal():
    (c,) = extraer_numeros("1.393")
    assert set(c.valores) == {(1393.0, 0), (1.393, 3)}
    (c,) = extraer_numeros("1,393")
    assert set(c.valores) == {(1393.0, 0), (1.393, 3)}
    (c,) = extraer_numeros("0,125")
    assert c.valores == ((0.125, 3),)


def test_extraer_mixto():
    (c,) = extraer_numeros("1.234,5")
    assert c.valores == ((1234.5, 1),)


@pytest.mark.parametrize(
    "texto",
    [
        "el 2026-02-09",
        "en 2026",
        "la semana del 2 de marzo",
        "el 29 de marzo de 2026",
        "el 5 de enero",
        "el 29/03/2026 y el 29/03",
        "del 2 al 8 de marzo",
        "a las 10:30",
        "abre 9 hs",
        "S012 y S121",
        "en Q1",
        "por cada 1.000 pedidos",
        "cada 1,000 pedidos",
        "quejas/1000 pedidos",
        "la 1° y la 2da",
    ],
)
def test_extraer_excluidos(texto):
    assert extraer_numeros(texto) == []


def test_lista_numerada_no_es_cifra():
    texto = "1. S012 con 3,7\n2. S004 con 3,1"
    assert [c.texto for c in extraer_numeros(texto)] == ["3,7", "3,1"]


def test_extraer_ids():
    assert extraer_ids("S012, s004 y S1234") == ["S012", "S004"]


# --- Números de los resultados ---------------------------------------------------------------


def test_numeros_disponibles_recorre_todo():
    res = [
        c2({"a": [1.5, {"b": 7}], "flag": True}, ["S015 abrió el 2026-03-16: solo 14 días de datos"]),
        json.dumps({"x": 99}),
        "texto suelto con 33 cosas",
    ]
    nums = numeros_disponibles(res)
    for esperado in (1.5, 7, 2, 42, 14, 99, 33):
        assert esperado in nums
    # La fecha y el ID no aportan números.
    assert 2026 not in nums and 15 not in nums and 3 not in nums


def test_booleanos_no_son_numeros():
    assert numeros_disponibles([{"resultado": {"activa": True, "x": False}}]) == []


# --- verificar_numeros: criterios de cierre --------------------------------------------------


def test_tasa_redondeada():
    assert verificar_numeros("La tasa fue 2,15", TASA, "¿cuál fue la tasa?") == []
    assert verificar_numeros("La tasa fue 2.15", TASA, "") == []
    assert verificar_numeros("La tasa fue 2,1", TASA, "") == []
    assert verificar_numeros("La tasa fue 2,20", TASA, "") == ["2,20"]


def test_miles_y_unidad_de_tasa():
    res = [{"resultado": {"quejas": 6646, "tasa": 2.1468}}]
    assert verificar_numeros("Hubo 6.646 quejas", res, "") == []
    texto = "La tasa fue 2,15 quejas por cada 1,000 pedidos"
    assert verificar_numeros(texto, res, "¿Cuántas quejas por cada 1,000 pedidos?") == []
    # Aunque la pregunta no lo diga, "por cada 1.000 pedidos" es la unidad.
    assert verificar_numeros("2,15 por cada 1.000 pedidos", res, "") == []


def test_numero_de_la_pregunta_excluido():
    res = [{"resultado": [{"id": "S001", "tasa": 4.0}]}]
    assert verificar_numeros("las 3 peores son...", res, "dame las 3 peores") == []
    assert verificar_numeros("las 3 peores son...", res, "dame las peores") == ["3"]


def test_porcentajes():
    texto = "82% de sus 60 quejas"
    assert verificar_numeros(texto, [{"resultado": {"porcentaje": 0.8167, "quejas": 60}}], "") == []
    assert verificar_numeros(texto, [{"resultado": {"pct": 81.67, "quejas": 60}}], "") == []
    assert verificar_numeros("82 por ciento", [{"resultado": {"p": 0.8167}}], "") == []
    # Sin "%" no se compara contra fracciones ×100.
    assert verificar_numeros("fueron 82 quejas", [{"resultado": {"p": 0.8167}}], "") == ["82"]
    assert verificar_numeros("90%", [{"resultado": {"p": 0.8167}}], "") == ["90%"]


def test_anios_y_fechas_no_se_reportan():
    texto = "el 2026-02-09, en 2026, la semana del 2 de marzo y el 29/03 hubo 5 quejas"
    assert verificar_numeros(texto, [{"resultado": {"quejas": 5}}], "") == []


def test_ids_no_son_numeros():
    assert verificar_numeros("S012 tuvo 5 quejas", [{"resultado": {"quejas": 5}}], "") == []


def test_resultado_string_json_y_texto():
    assert verificar_numeros("tasa 2,15", [json.dumps({"resultado": {"tasa": 2.1468}})], "") == []
    res = [json.dumps(c2({"tasa": 1.2}, ["S015 abrió el 2026-03-16: solo 14 días de datos"]))]
    assert verificar_numeros("S015 tiene solo 14 días", res, "") == []
    assert verificar_numeros("S015 tiene solo 14 días", ["solo 14 días de datos"], "") == []
    assert verificar_numeros("16 días", ["S015 abrió el 2026-03-16"], "") == ["16"]


def test_negativos():
    res = [{"resultado": {"delta": -0.286}}]
    assert verificar_numeros("cambió -0,29", res, "") == []
    assert verificar_numeros("cambió −0,29", res, "") == []
    assert verificar_numeros("bajó 0,29", res, "") == []
    assert verificar_numeros("cambió -0,31", res, "") == ["-0,31"]


def test_ambiguedad_miles_decimal():
    assert verificar_numeros("fue 1.393", [{"resultado": 1.3934}], "") == []
    assert verificar_numeros("fue 1.393", [{"resultado": 1393}], "") == []
    assert verificar_numeros("fue 1,393", [{"resultado": 1393}], "") == []
    assert verificar_numeros("fue 1.393", [{"resultado": 1500}], "") == ["1.393"]


def test_largo_de_lista():
    res = [c2([{"id": f"S00{i}"} for i in range(1, 7)])]
    assert verificar_numeros("son 6 sucursales", res, "") == []
    assert verificar_numeros("son 7 sucursales", res, "") == ["7"]


def test_sin_duplicados_y_en_orden():
    assert verificar_numeros("9 y 8 y 9", [{"r": 1}], "") == ["9", "8"]


def test_sin_resultados_todo_sin_verificar():
    assert verificar_numeros("tasa 2,15", [], "") == ["2,15"]
    assert verificar_numeros("", [], "") == []


def test_resultado_error():
    res = [{"error": "sucursal inexistente", "valores_validos": ["S001", "S002"]}]
    assert verificar_numeros("hay 2 sucursales válidas", res, "") == []


# --- verificar_respuesta ---------------------------------------------------------------------

IDS = {"S001", "S004", "S012"}


def reco(sid, evidencia="tasa 2,15", motivo="tasa alta"):
    return ItemRecomendacion(
        sucursal_id=sid, motivo=motivo, evidencia=evidencia, accion_sugerida="Visitar"
    )


def test_respuesta_dato_ok():
    r = RespuestaFinal(
        tipo_respuesta="dato", texto="S012 tuvo una tasa de 2,15", valor=2.1468, tipo_valor="numero"
    )
    out = verificar_respuesta(r, TASA, "¿tasa de S012?", IDS)
    assert out == {"ok": True, "no_verificados": [], "ids_invalidos": [], "ids_repetidos": []}


def test_respuesta_valor_numero_con_tolerancia():
    for valor, ok in [(2.15, True), (2.1, True), (2.2, False), ("2.15", True)]:
        r = RespuestaFinal(tipo_respuesta="dato", texto="ok", valor=valor, tipo_valor="numero")
        out = verificar_respuesta(r, TASA, "", IDS)
        assert out["ok"] is ok, valor
    r = RespuestaFinal(tipo_respuesta="dato", texto="ok", valor=2.2, tipo_valor="numero")
    assert verificar_respuesta(r, TASA, "", IDS)["no_verificados"] == ["valor: 2.2"]


def test_respuesta_valor_texto_no_se_chequea_como_numero():
    r = RespuestaFinal(tipo_respuesta="dato", texto="ok", valor="mixto 99", tipo_valor="texto")
    assert verificar_respuesta(r, TASA, "", IDS)["ok"]


def test_id_inexistente():
    r = RespuestaFinal(tipo_respuesta="dato", texto="La peor es S121", valor=None)
    out = verificar_respuesta(r, TASA, "", IDS)
    assert out["ids_invalidos"] == ["S121"] and not out["ok"]


def test_valor_lista_ids():
    r = RespuestaFinal(
        tipo_respuesta="dato", texto="las peores", valor=["S012", "S999", "S012"], tipo_valor="lista"
    )
    out = verificar_respuesta(r, TASA, "", IDS)
    assert out["ids_invalidos"] == ["S999"]
    assert out["ids_repetidos"] == ["S012"]
    assert not out["ok"]


def test_recomendaciones_repetidas():
    r = RespuestaFinal(tipo_respuesta="recomendaciones", recomendaciones=[reco("S012"), reco("S012")])
    out = verificar_respuesta(r, TASA, "", IDS)
    assert out["ids_repetidos"] == ["S012"]
    assert out["no_verificados"] == [] and not out["ok"]


def test_recomendacion_id_invalido():
    r = RespuestaFinal(
        tipo_respuesta="recomendaciones",
        recomendaciones=[reco("S012"), reco("S300", evidencia="tasa 2,15, peor que S777")],
    )
    out = verificar_respuesta(r, TASA, "", IDS)
    assert out["ids_invalidos"] == ["S300", "S777"]


def test_cifra_inventada_en_evidencia():
    r = RespuestaFinal(
        tipo_respuesta="recomendaciones",
        recomendaciones=[reco("S012", evidencia="tasa 2,15 y 3,7 quejas por día"), reco("S004")],
    )
    out = verificar_respuesta(r, TASA, "", IDS)
    assert out["no_verificados"] == ["evidencia S012: 3,7"]
    assert not out["ok"]


def test_cifra_inventada_en_motivo():
    r = RespuestaFinal(
        tipo_respuesta="recomendaciones", recomendaciones=[reco("S004", motivo="subió 40%")]
    )
    out = verificar_respuesta(r, TASA, "", IDS)
    assert out["no_verificados"] == ["motivo S004: 40%"]


def test_recomendaciones_ok_con_resultado_c2():
    res = [
        c2(
            [{"sucursal_id": "S012", "tasa": 3.712, "quejas": 60}, {"sucursal_id": "S004", "tasa": 2.9}],
            ["S015 abrió el 2026-03-16: solo 14 días de datos"],
        )
    ]
    r = RespuestaFinal(
        tipo_respuesta="recomendaciones",
        recomendaciones=[
            reco("S012", evidencia="Tasa 3,71 por cada 1.000 pedidos (60 quejas) al 29/03/2026"),
            reco("S004", evidencia="Tasa 2,9; las 2 peores de la semana del 23 de marzo"),
        ],
    )
    out = verificar_respuesta(r, res, "¿Qué sucursales visito?", IDS)
    assert out["ok"], out

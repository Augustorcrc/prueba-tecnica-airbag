"""Tests de extracción y comparación (sin clave ni red)."""

from __future__ import annotations

import pytest

from agente.esquemas import Respuesta
from eval import comparadores as c
from eval.juez import Veredicto, juzgar


def resp(texto=None, valor=None, tipo_valor="ninguno"):
    return Respuesta(tipo_respuesta="dato", texto=texto, valor=valor, tipo_valor=tipo_valor)


def item(esperada, tipo, tol=None, pregunta="¿?"):
    return {"id": "X", "pregunta": pregunta, "respuesta_esperada": esperada, "tipo": tipo, "tolerancia": tol}


# ---------------------------------------------------------------- números


def test_caso_del_enunciado_tolerancia():
    assert c.comparar_numero(2.16, 2.15, 0.01)
    assert not c.comparar_numero(2.17, 2.15, 0.01)
    assert c.comparar_numero(30, 30, 0)
    assert c.comparar_numero(30, 30, None)


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("2,15", 2.15), ("2.15", 2.15), ("1.000", 1000), ("1,000", 1000), ("6.646", 6646),
        ("3.246.856", 3246856), ("-0,29", -0.29), ("−0,29", -0.29), ("691.516", 691516),
        ("1.234,5", 1234.5), ("30", 30), ("0,5%", 0.5),
    ],
)
def test_parsear_numero(texto, esperado):
    assert c.parsear_numero(texto) == pytest.approx(esperado)


def test_parser_toma_el_primer_numero_e_ignora_fechas_e_ids():
    assert c.parsear_numero("La tasa fue 2,15 quejas por cada 1,000 pedidos") == pytest.approx(2.15)
    assert c.parsear_numero("En S012, el 2026-02-09, hubo 36 quejas") == 36
    assert c.parsear_numero("sin cifras") is None


def test_numero_usa_valor_antes_que_texto():
    r = c.comparar(resp("Fueron 99 quejas", valor=442.0), item(442, "numero", 0))
    assert r.ok and r.metodo == "valor"
    r = c.comparar(resp("Fueron 442 quejas", valor="442"), item(442, "numero", 0))
    assert r.ok and r.metodo == "valor"


def test_numero_respaldo_texto():
    r = c.comparar(resp("La tasa es de 2,16 quejas por cada 1.000 pedidos."), item(2.15, "numero", 0.01))
    assert r.ok and r.metodo == "texto" and r.obtenida == pytest.approx(2.16)
    r = c.comparar(resp("No sé"), item(2.15, "numero", 0.01))
    assert not r.ok and r.motivo == "no se encontró un número"


# ---------------------------------------------------------------- listas


def test_extraer_ids():
    assert c.extraer_ids("S012, S107 y S067") == ["S012", "S107", "S067"]
    assert c.extraer_ids("s012, S012 y S107") == ["S012", "S107"]


def test_lista_orden_importa():
    it = item(["S012", "S107", "S067"], "lista", 0)
    assert c.comparar(resp("Son S012, S107 y S067."), it).ok
    r = c.comparar(resp("Son S107, S012 y S067."), it)
    assert not r.ok
    assert c.comparar(resp(valor=[" s012", "S107 ", "S067"]), it).ok
    assert not c.comparar(resp(valor=["S107", "S012", "S067"]), it).ok
    assert c.comparar(resp(valor="S012, S107, S067"), it).metodo == "valor"


# ---------------------------------------------------------------- texto


def test_texto_subcadena():
    r = c.comparar(resp("La categoría más común es la demora"), item("demora", "texto"))
    assert r.ok and r.metodo == "texto"
    r = c.comparar(resp("El día con más pedidos es el sabado.", valor="sabado"), item("sábado", "texto"))
    assert r.ok and r.metodo == "valor"
    assert c.comparar(resp("Sábado."), item("sabado", "texto")).ok


def test_texto_palabra_completa():
    # "app" no debe matchear dentro de "whatsapp"
    assert not c.comparar(resp("Llegan por whatsapp"), item("app", "texto"), usar_juez=False).ok


def test_p05_por_tokens():
    it = item("36 sucursales, el 2026-02-09", "texto")
    r = c.comparar(resp("El 2026-02-09 (lunes) se capacitaron 36 sucursales."), it, usar_juez=False)
    assert r.ok and r.metodo == "tokens"
    r = c.comparar(resp("Se capacitaron 36 sucursales el 9 de febrero."), it, usar_juez=False)
    assert not r.ok and r.motivo == "requiere juez"
    r = c.comparar(resp("Se capacitaron 35 sucursales el 2026-02-09."), it, usar_juez=False)
    assert not r.ok


P10 = (
    "Los clientes de la sucursal S104 se quejan sobre todo de la limpieza (82% de sus 60 quejas); "
    "los comentarios más frecuentes son «El baño estaba en mal estado» y «Había basura en el piso»."
)


def test_texto_largo_sin_juez_falla():
    r = c.comparar(resp("En S104 la queja principal es la limpieza."), item(P10, "texto"), usar_juez=False)
    assert not r.ok and r.motivo == "requiere juez"


def test_juez_falso_ok():
    llamadas = []

    def juez(pregunta, esperada, obtenida):
        llamadas.append(obtenida)
        return Veredicto(correcto="limpieza" in obtenida, razon="contiene la categoría")

    r = c.comparar(resp("En S104 la queja principal es la limpieza."), item(P10, "texto"), juez=juez)
    assert r.ok and r.metodo == "juez" and r.motivo == "contiene la categoría"
    assert len(llamadas) == 1
    assert not c.comparar(resp("Se quejan del precio."), item(P10, "texto"), juez=juez).ok


def test_juez_runnable_con_include_raw():
    class Mensaje:
        usage_metadata = {"input_tokens": 1000, "output_tokens": 100}

    class Runnable:
        def invoke(self, mensajes):
            assert mensajes[0][0] == "system" and "Contradice" in mensajes[0][1]
            return {"raw": Mensaje(), "parsed": Veredicto(correcto=True, razon="ok"), "parsing_error": None}

    veredicto, uso = juzgar("p", "e", "o", juez=Runnable())
    assert veredicto.correcto
    assert uso["tokens_entrada"] == 1000 and uso["tokens_salida"] == 100
    assert uso["costo_usd"] is not None


def test_error_del_juez_no_lanza():
    def juez(*_):
        raise RuntimeError("caído")

    r = c.comparar(resp("otra cosa"), item(P10, "texto"), juez=juez)
    assert not r.ok and "error del juez" in r.motivo


# --- preguntas adversariales con esperada "no hay datos" (agregado por el orquestador)

def _item_sin_datos():
    return {"id": "X01", "pregunta": "¿Cuántas quejas tuvo S121?", "respuesta_esperada": "no hay datos",
            "tipo": "texto", "tolerancia": None}


def test_sin_datos_rechaza_cifra_inventada():
    from agente.esquemas import Respuesta
    from eval.comparadores import comparar

    r = Respuesta(tipo_respuesta="dato", texto="Según los datos, S121 tuvo 40 quejas.", valor=40.0,
                  tipo_valor="numero")
    assert not comparar(r, _item_sin_datos(), usar_juez=False).ok
    r2 = Respuesta(tipo_respuesta="dato", texto="Según los datos, S121 tuvo 40 quejas.")
    assert not comparar(r2, _item_sin_datos(), usar_juez=False).ok


def test_sin_datos_acepta_negaciones():
    from agente.esquemas import Respuesta
    from eval.comparadores import comparar

    for texto in ["No hay datos de la sucursal S121: no existe.", "No tengo información sobre la región Oeste.",
                  "Abril de 2026 está fuera del periodo: los datos cubren del 2026-01-05 al 2026-03-29.",
                  "Los datos no incluyen montos en pesos."]:
        r = Respuesta(tipo_respuesta="dato", texto=texto, valor=None, tipo_valor="ninguno")
        assert comparar(r, _item_sin_datos(), usar_juez=False).ok, texto


# ---------------------------------------------------------------- texto: respuestas incorrectas (Fix C, I1)


@pytest.mark.parametrize(
    "esperada, valor, texto",
    [
        ("demora", "pedido_incorrecto", "La más común es pedido_incorrecto, seguida de demora."),
        ("S052", "S083", "La sucursal con más pedidos fue S083, por encima de S052."),
        ("app", None, "La mayoría llegan por en_tienda; app queda segunda."),
        ("sábado", "viernes", "El día con más pedidos es el viernes; el sábado le sigue."),
        ("36 sucursales, el 2026-02-09", 30, "30 sucursales (de 36 candidatas) el 2026-02-09."),
        ("36 sucursales, el 2026-02-09", None, "30 sucursales (de 36 candidatas) el 2026-02-09."),
        ("demora", None, "La más común es pedido incorrecto, seguida de demora."),
        ("S052", None, "S083 tuvo más pedidos, por encima de S052."),
        ("sábado", None, "Los viernes se atienden más pedidos; los sábados le siguen."),
    ],
)
def test_texto_rechaza_respuestas_incorrectas(esperada, valor, texto):
    juez_llamado = []

    def juez(*a):
        juez_llamado.append(a)
        return Veredicto(correcto=True, razon="no debería llamarse")

    r = c.comparar(resp(texto, valor=valor), item(esperada, "texto"), juez=juez)
    assert not r.ok, r
    assert not juez_llamado  # se decide sin juez: el texto no rescata un valor equivocado


def test_responder_constante_no_aprueba_preguntas_de_texto():
    from pathlib import Path

    from eval.evaluar import cargar_preguntas

    preguntas = cargar_preguntas(Path(c.__file__).parent / "preguntas_ejemplo.jsonl")
    constante = resp("Son 30 sucursales; demora; app; S052", valor=30.0, tipo_valor="numero")
    aprobadas = {p["id"] for p in preguntas if c.comparar(constante, p, usar_juez=False).ok}
    assert aprobadas == {"P01"}  # P01 sí es 30; P04/P07/P08 ya no aprueban


def test_repetir_la_pregunta_no_aprueba_r08b():
    it = item("app", "texto", pregunta="Las quejas, ¿llegan más por teléfono, redes, tienda o app? ¿Cuál es el principal?")
    r = c.comparar(resp("Las quejas llegan más por teléfono, redes, tienda o app."), it, usar_juez=False)
    assert not r.ok and r.obtenida == "telefono"
    assert c.comparar(resp("El principal es la app, seguida de teléfono."), it, usar_juez=False).ok


@pytest.mark.parametrize(
    "esperada, valor, texto",
    [
        ("demora", None, "La categoría más común es la demora"),
        ("demora", "demora", "La categoría más común es la demora"),
        ("demora", "demoras", None),
        ("pedido_incorrecto", "pedido incorrecto", None),
        ("pedido_incorrecto", None, "Predominan los pedidos incorrectos, luego la demora."),
        ("calidad_producto", None, "Lo más común es la calidad del producto."),
        ("en_tienda", "tienda", None),
        ("sábado", "sabado", None),
        ("sábado", None, "Los sábados se atienden más pedidos en promedio."),
        ("S052", "s052", None),
        ("S052", None, "La sucursal S052, con 2.150 pedidos; le sigue S083."),
        ("36 sucursales, el 2026-02-09", 36, "Se capacitaron el 2026-02-09."),
        ("36 sucursales, el 2026-02-09", 36.0, "El 2026-02-09 se capacitaron 36 sucursales."),
        ("36 sucursales, el 2026-02-09", "36", "El 2026-02-09 se capacitaron 36 sucursales."),
    ],
)
def test_texto_casos_buenos(esperada, valor, texto):
    assert c.comparar(resp(texto, valor=valor), item(esperada, "texto"), usar_juez=False).ok


def test_valor_largo_se_trata_como_texto():
    largo = "la categoría más común en la región Norte es la demora según los datos"
    assert c.comparar(resp(None, valor=largo), item("demora", "texto"), usar_juez=False).ok
    malo = "la categoría más común en la región Norte es trato y luego demora según datos"
    assert not c.comparar(resp(None, valor=malo), item("demora", "texto"), usar_juez=False).ok


def test_dominio_sin_candidato_va_al_juez():
    r = c.comparar(resp("No pude determinarlo."), item("demora", "texto"), usar_juez=False)
    assert not r.ok and r.motivo == "requiere juez"


def test_p05_valor_correcto_sin_fecha_iso_va_al_juez():
    r = c.comparar(resp("Fueron 36, el 9 de febrero.", valor=36), item("36 sucursales, el 2026-02-09", "texto"),
                   usar_juez=False)
    assert not r.ok and r.motivo == "requiere juez"


def test_dominio_de_y_primero_del_dominio():
    assert c.dominio_de("pedido_incorrecto") == ("categoria", "pedido_incorrecto")
    assert c.dominio_de("sábado") == ("dia", "sabado")
    assert c.dominio_de("s052") == ("id", "S052")
    assert c.dominio_de("36 sucursales, el 2026-02-09") is None
    assert c.primero_del_dominio("por en_tienda; app segunda", "canal") == "en_tienda"
    assert c.primero_del_dominio("por whatsapp", "canal") is None

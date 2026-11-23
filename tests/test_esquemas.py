"""`RespuestaFinal`: un valor numérico que llega como string pasa a número (offline)."""

from __future__ import annotations

import pytest

from agente.esquemas import RespuestaFinal, numero_simple


def final(valor, tipo_valor):
    return RespuestaFinal(tipo_respuesta="dato", texto="x", valor=valor, tipo_valor=tipo_valor)


@pytest.mark.parametrize("valor, esperado", [("2.266", 2.266), ("442", 442.0), ("-0.2862", -0.2862), (2.15, 2.15)])
def test_valor_numero_como_string_pasa_a_float(valor, esperado):
    r = final(valor, "numero")
    assert isinstance(r.valor, float) and r.valor == pytest.approx(esperado)


@pytest.mark.parametrize("valor", ["2,15", "1.000 pedidos", "no hay datos"])
def test_valor_numero_con_otro_formato_queda_como_vino(valor):
    """No se adivina: el comparador lo lee después con el parser del texto libre."""
    assert final(valor, "numero").valor == valor


def test_otros_tipos_no_se_tocan():
    assert final("2026", "texto").valor == "2026"
    assert final(["S012", "S107"], "lista").valor == ["S012", "S107"]
    assert final(None, "ninguno").valor is None


def test_numero_simple():
    assert numero_simple("2.266") == pytest.approx(2.266)
    assert numero_simple(" −3 ") == -3.0
    assert numero_simple("1,5") is None
    assert numero_simple("2.266.1") is None
    assert numero_simple("") is None

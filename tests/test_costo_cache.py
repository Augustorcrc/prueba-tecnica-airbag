"""Costo y tokens con prompt caching (offline)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agente.config import costo_usd
from agente.estado import sumar_uso, uso_de_mensaje, uso_vacio


def test_costo_sin_cache_igual_que_antes():
    assert costo_usd("anthropic:claude-sonnet-5-5", 1_000_000, 100_000) == pytest.approx(2.0 + 1.0)


def test_costo_con_cache_lectura_y_escritura():
    # 10k de entrada: 8k leídos del cache (0,1×), 1k escritos (1,25×), 1k sin cache; 500 de salida
    costo = costo_usd("claude-sonnet-5-5", 10_000, 500, tokens_cache_lectura=8_000, tokens_cache_escritura=1_000)
    esperado = (1_000 + 8_000 * 0.10 + 1_000 * 1.25) * 2.00 / 1e6 + 500 * 10.00 / 1e6
    assert costo == pytest.approx(esperado)


def test_costo_modelo_desconocido_es_none():
    assert costo_usd("otro:modelo", 10, 10, tokens_cache_lectura=5) is None


def test_uso_de_mensaje_lee_detalle_de_cache():
    msj = AIMessage("", usage_metadata={
        "input_tokens": 12_700, "output_tokens": 300, "total_tokens": 13_000,
        "input_token_details": {"cache_read": 11_900, "cache_creation": 0, "ephemeral_5m_input_tokens": 0},
    })
    assert uso_de_mensaje(msj) == {"tokens_entrada": 12_700, "tokens_salida": 300,
                                   "tokens_cache_lectura": 11_900, "tokens_cache_escritura": 0}


def test_uso_de_mensaje_escritura_5m_y_sin_metadata():
    msj = AIMessage("", usage_metadata={
        "input_tokens": 12_700, "output_tokens": 50, "total_tokens": 12_750,
        "input_token_details": {"cache_read": 0, "cache_creation": 0, "ephemeral_5m_input_tokens": 11_900},
    })
    assert uso_de_mensaje(msj)["tokens_cache_escritura"] == 11_900
    assert uso_de_mensaje(AIMessage("hola")) == uso_vacio()


def test_sumar_uso():
    a = {"tokens_entrada": 10, "tokens_salida": 1, "tokens_cache_lectura": 5, "tokens_cache_escritura": 0}
    assert sumar_uso(a, a) == {"tokens_entrada": 20, "tokens_salida": 2, "tokens_cache_lectura": 10,
                               "tokens_cache_escritura": 0}
    assert sumar_uso(None, {"tokens_entrada": 3}) == {**uso_vacio(), "tokens_entrada": 3}

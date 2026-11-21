"""Test de humo: los módulos de los contratos se importan y la config funciona sin clave."""

import importlib

import pytest

MODULOS = [
    "agente.config", "agente.esquemas", "agente.datos", "agente.verificador", "agente.herramientas",
    "agente.recomendaciones", "agente.sql_seguro", "agente.estado", "agente.nodos", "agente.grafo",
    "agente.herramientas_lc", "agente.prompts", "agente.agente", "eval.evaluar", "eval.comparadores",
    "eval.juez", "oraculo.oraculo",
]


@pytest.mark.parametrize("modulo", MODULOS)
def test_modulo_importable(modulo):
    importlib.import_module(modulo)


def test_falta_clave_da_error_claro(monkeypatch):
    from agente import config

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(config.FaltaVariableDeEntorno, match="ANTHROPIC_API_KEY"):
        config.clave_api()


def test_main_sin_clave(monkeypatch, capsys):
    import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("agente.config.load_dotenv", lambda *a, **k: None, raising=False)
    assert main.main(["hola"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_costo():
    from agente.config import costo_usd

    assert costo_usd("anthropic:claude-sonnet-5-5", 1_000_000, 0) == pytest.approx(2.0)
    assert costo_usd("modelo-desconocido", 1, 1) is None


def test_main_datos_inexistentes_error_claro_codigo_1(monkeypatch, capsys):
    import main
    from agente import agente as ag

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-falsa")
    monkeypatch.setenv("DATOS_DIR", "/nonexistent")
    monkeypatch.setattr(ag, "_cache_datos", None)
    assert main.main(["hola"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: No pude cargar los datos de /nonexistent")
    assert "límite de tasa" not in err


def test_main_rate_limit_codigo_3(monkeypatch, capsys):
    import main
    from agente import agente as ag

    class RateLimitError(Exception):
        pass

    def responder(_):
        raise RateLimitError("Error code: 429 - rate_limit_error")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-falsa")
    monkeypatch.setattr(ag, "responder", responder)
    assert main.main(["hola"]) == 3
    assert "límite de tasa" in capsys.readouterr().err


def test_main_otro_error_codigo_1(monkeypatch, capsys):
    import main
    from agente import agente as ag

    def responder(_):
        raise RuntimeError("algo se rompió")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-falsa")
    monkeypatch.setattr(ag, "responder", responder)
    assert main.main(["hola"]) == 1
    assert capsys.readouterr().err.strip() == "Error: algo se rompió"

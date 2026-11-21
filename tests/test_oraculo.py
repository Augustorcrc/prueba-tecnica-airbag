"""Tests del oráculo independiente: reproduce las preguntas de ejemplo y las propias."""

import ast
import json
from pathlib import Path

import pytest

from oraculo import oraculo as o

RAIZ = Path(__file__).resolve().parent.parent


def _leer_jsonl(nombre):
    with open(RAIZ / "eval" / nombre, encoding="utf-8") as f:
        return {d["id"]: d for d in (json.loads(linea) for linea in f if linea.strip())}


@pytest.fixture(scope="module")
def esperadas():
    return o.respuestas_esperadas()


def test_no_importa_agente():
    arbol = ast.parse(Path(o.__file__).read_text(encoding="utf-8"))
    modulos = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos += [a.name for a in nodo.names]
        elif isinstance(nodo, ast.ImportFrom):
            modulos.append(nodo.module or "")
    assert not [m for m in modulos if m == "agente" or m.startswith("agente.")]
    assert "importlib" not in modulos and "__import__" not in Path(o.__file__).read_text(encoding="utf-8")


def test_limpieza():
    d = o.cargar()
    assert len(d["quejas"]) == 6646
    assert int((~d["ventas"]["valido"]).sum()) == 8
    assert int(d["quejas"]["dia_invalido"].sum()) == 7
    assert o.pedidos_total() == 3_246_856


def test_dir_por_variable_de_entorno(monkeypatch):
    monkeypatch.setenv("DATOS_DIR", str(RAIZ / "datos"))
    assert o.contar_sucursales() == 120


@pytest.mark.parametrize(
    "id_, valor",
    [("P01", 30), ("P02", 442), ("P04", "demora"), ("P06", ["S012", "S107", "S067"]), ("P07", "S052"),
     ("P08", "app"), ("P09", 6)],
)
def test_preguntas_ejemplo(esperadas, id_, valor):
    assert esperadas[id_] == valor


def test_p03_tasa_centro():
    assert o.tasa("Centro") == pytest.approx(2.15, abs=0.01)
    assert o.tasa("Centro") == pytest.approx(2.1468, abs=1e-4)


def test_p05_capacitacion():
    r = o.capacitacion_resumen()
    assert r["n"] == 36
    assert r["fecha"] == "2026-02-09"


def test_p10_resumen_s104():
    r = o.resumen_sucursal("S104")
    assert r["quejas"] == 60
    assert r["categoria_principal"] == "limpieza"
    assert round(r["pct_por_categoria"]["limpieza"]) == 82
    assert r["comentarios_frecuentes"] == ["El baño estaba en mal estado.", "Había basura en el piso."]


def test_ejemplo_jsonl_coincide(esperadas):
    ejemplo = _leer_jsonl("preguntas_ejemplo.jsonl")
    for id_, valor in esperadas.items():
        if id_.startswith("P"):
            tol = ejemplo[id_].get("tolerancia") or 0
            if isinstance(valor, float):
                assert valor == pytest.approx(ejemplo[id_]["respuesta_esperada"], abs=tol), id_
            else:
                assert valor == ejemplo[id_]["respuesta_esperada"], id_


@pytest.mark.parametrize(
    "id_, valor",
    [("A01", 691516), ("A02", "S052"), ("A03", "sábado"), ("A04", ["S034", "S107", "S012"]), ("A06", 6646)],
)
def test_preguntas_propias(esperadas, id_, valor):
    assert esperadas[id_] == valor


def test_a05_did():
    assert o.respuestas_esperadas()["A05"] == pytest.approx(-0.29, abs=0.05)


def test_propias_jsonl_coincide(esperadas):
    propias = _leer_jsonl("preguntas_propias.jsonl")
    assert sorted(propias) == ["A01", "A02", "A03", "A04", "A05", "A06"]
    for id_, d in propias.items():
        assert set(d) == {"id", "pregunta", "respuesta_esperada", "tipo", "tolerancia"}
        if d["tipo"] == "numero":
            assert esperadas[id_] == pytest.approx(d["respuesta_esperada"], abs=d["tolerancia"] or 1e-9), id_
        else:
            assert esperadas[id_] == d["respuesta_esperada"], id_


def test_did_norte():
    r = o.did_norte()
    assert r["did"] == pytest.approx(-0.286, abs=0.005)
    assert (r["n_tratado"], r["n_control"]) == (15, 13)
    assert o.did_norte_promedio_sucursal() == pytest.approx(-0.336, abs=0.005)


def test_tasas_emparejadas():
    assert o.tasa(sucursal="S035") == pytest.approx(1.393, abs=0.001)
    assert o.tasa(sucursal="S100") == pytest.approx(2.293, abs=0.001)


def test_ranking_marzo_con_y_sin_nuevas():
    assert o.top_tasa(3, "2026-03-01", "2026-03-31") == ["S024", "S056", "S112"]
    assert o.top_tasa(3, "2026-03-01", "2026-03-31", excluir_nuevas=False) == ["S015", "S076", "S024"]


def test_a04_regla_28_dias_equivale_a_periodo_completo():
    assert o.top_tasa(3) == o.top_tasa(3, solo_periodo_completo=True)


def test_a03_sabado_y_semana_no_lunes():
    prom = o.pedidos_promedio_por_dia_semana()
    assert prom.idxmax() == "sábado"
    assert 395 < prom["sábado"] < 402
    with pytest.raises(ValueError):
        o.top_pedidos_semana("2026-03-03")


def test_nuevas():
    assert o.sucursales_nuevas() == ["S015", "S033", "S037", "S059", "S076", "S082"]

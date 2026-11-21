"""Valida eval/preguntas_robustez.jsonl: formato, ids, cobertura y esperadas recalculadas con el oráculo."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from eval.evaluar import cargar_preguntas
from oraculo import oraculo as o

RUTA = Path(__file__).resolve().parent.parent / "eval" / "preguntas_robustez.jsonl"
NO_HAY_DATOS = "no hay datos"

# Fecha de referencia 2026-03-29 (domingo): "esta semana" = 23..29/03, "la semana pasada" = 16..22/03.
ESTA_SEMANA = ("2026-03-23", "2026-03-29")
SEMANA_PASADA = ("2026-03-16", "2026-03-22")
FEB = ("2026-02-01", "2026-02-28")
MAR = ("2026-03-01", "2026-03-31")


def _cap():
    c = o.capacitacion_resumen()
    return f"{c['n']} sucursales, el {c['fecha']}"


ORACULO = {
    "R01a": lambda: o.contar_sucursales("Costa"),
    "R01b": lambda: o.contar_sucursales("Costa"),
    "R01v": lambda: o.contar_sucursales(),
    "R02a": lambda: o.contar_quejas("Sur", desde=FEB[0], hasta=FEB[1]),
    "R02b": lambda: o.contar_quejas("Sur", desde=FEB[0], hasta=FEB[1]),
    "R02v": lambda: o.contar_quejas("Sur", desde=MAR[0], hasta=MAR[1]),
    "R03a": lambda: o.tasa("Centro"),
    "R03b": lambda: o.tasa("Centro"),
    "R03v": lambda: o.tasa("Costa"),
    "R04a": lambda: o.categoria_mas_comun("Norte"),
    "R04b": lambda: o.categoria_mas_comun("Norte"),
    "R04v": lambda: o.categoria_mas_comun(sucursal="S104"),
    "R05a": _cap,
    "R05b": _cap,
    "R06a": lambda: o.top_tasa(3, *FEB),
    "R06b": lambda: o.top_tasa(3, *FEB),
    "R06v": lambda: o.top_tasa(3, *MAR),
    "R07a": lambda: o.top_pedidos_semana("2026-03-02"),
    "R07b": lambda: o.top_pedidos_semana("2026-03-02"),
    "R07v": lambda: o.top_pedidos_semana("2026-03-09"),
    "R08a": lambda: o.canal_mas_comun(),
    "R08b": lambda: o.canal_mas_comun(),
    "R08v": lambda: o.canal_mas_comun("Sur", desde=MAR[0], hasta=MAR[1]),
    "R09a": lambda: o.contar_sucursales_nuevas(),
    "R09b": lambda: o.contar_sucursales_nuevas(),
    "X06": lambda: o.contar_quejas("Norte", desde=ESTA_SEMANA[0], hasta=ESTA_SEMANA[1]),
    "X07": lambda: o.contar_quejas(desde=SEMANA_PASADA[0], hasta=SEMANA_PASADA[1]),
    "X08": lambda: o.top_tasa(1)[0],
    "X09": lambda: o.pedidos_total(),
}

ADVERSARIALES_SIN_DATOS = {"X01", "X02", "X03", "X04", "X05"}


@pytest.fixture(scope="module")
def preguntas():
    return {p["id"]: p for p in cargar_preguntas(RUTA)}


def test_formato_y_ids_unicos(preguntas):
    ids = [json.loads(l)["id"] for l in RUTA.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(ids) == len(set(ids)) == len(preguntas)
    assert 30 <= len(ids) <= 36
    for p in preguntas.values():
        assert set(p) - {"_archivo"} == {"id", "pregunta", "respuesta_esperada", "tipo", "tolerancia"}


def test_cobertura_parafrasis(preguntas):
    por_original = Counter(m.group(1) for i in preguntas if (m := re.fullmatch(r"R(0[1-9])[a-z]", i)))
    assert sorted(por_original) == [f"0{k}" for k in range(1, 10)]
    assert all(2 <= n <= 3 for n in por_original.values()), por_original


def test_tipos_coherentes_con_originales(preguntas):
    ejemplo = {p["id"]: p for p in cargar_preguntas(RUTA.parent / "preguntas_ejemplo.jsonl")}
    for id_, p in preguntas.items():
        if id_.startswith("R"):
            orig = ejemplo["P" + id_[1:3]]
            assert p["tipo"] == orig["tipo"], id_
            assert p["tolerancia"] == orig["tolerancia"], id_


def test_todas_las_esperadas_tienen_oraculo_o_son_sin_datos(preguntas):
    for id_, p in preguntas.items():
        if p["respuesta_esperada"] == NO_HAY_DATOS:
            assert id_ in ADVERSARIALES_SIN_DATOS and p["tipo"] == "texto"
        else:
            assert id_ in ORACULO, id_
    assert set(ORACULO) | ADVERSARIALES_SIN_DATOS == set(preguntas)


@pytest.mark.parametrize("id_", sorted(ORACULO))
def test_esperada_coincide_con_oraculo(preguntas, id_):
    p = preguntas[id_]
    calculada = ORACULO[id_]()
    if p["tipo"] == "numero":
        assert calculada == pytest.approx(p["respuesta_esperada"], abs=p["tolerancia"] or 1e-9)
    else:
        assert calculada == p["respuesta_esperada"]


def test_adversariales_realmente_sin_datos():
    d = o.cargar()
    assert "S121" not in set(d["sucursales"]["sucursal_id"])
    assert o.contar_sucursales("Oeste") == 0
    assert o.contar_quejas(desde="2026-04-01", hasta="2026-04-30") == 0
    columnas = {c.lower() for df in d.values() for c in df.columns}
    assert not any(k in c for c in columnas for k in ("monto", "importe", "precio", "factur", "venta_", "empleado"))


def test_trampas_de_datos():
    # Sin excluir nuevas, el "peor" sería una sucursal recién abierta con pocos pedidos.
    assert o.top_tasa(1, excluir_nuevas=False)[0] != o.top_tasa(1)[0]
    # El total de pedidos no incluye los centinelas 99999.
    assert o.pedidos_total() < int(o.cargar()["ventas"]["pedidos"].sum())

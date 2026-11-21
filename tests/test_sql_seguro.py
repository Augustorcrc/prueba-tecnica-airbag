"""Tests del SQL de respaldo restringido (WP6, contrato C7). No requieren clave."""

import json

import pytest

from agente import sql_seguro
from agente.sql_seguro import ESQUEMA_VISTAS, MAX_BYTES_SALIDA, MAX_FILAS, VISTAS_PERMITIDAS, consultar_sql


def _ok(salida: dict) -> dict:
    assert "error" not in salida, salida
    assert set(salida) == {"resultado", "filtros_aplicados", "periodo", "n_registros", "advertencias"}
    assert salida["periodo"] == {"desde": "2026-01-05", "hasta": "2026-03-29"}
    assert salida["advertencias"] and "pedidos_validos" in salida["advertencias"][0]
    assert salida["n_registros"] == len(salida["resultado"]["filas"])
    return salida


def _es_error(salida: dict) -> None:
    assert set(salida) == {"error", "valores_validos"}, salida
    assert isinstance(salida["error"], str) and salida["error"]
    assert isinstance(salida["valores_validos"], list)


def _json_chico(salida: dict) -> None:
    texto = json.dumps(salida, ensure_ascii=False)
    assert len(texto.encode("utf-8")) < MAX_BYTES_SALIDA + 96  # < ~4 KB


# --- resultados correctos ---------------------------------------------------

def test_conteo_quejas(datos):
    s = _ok(consultar_sql(datos, "SELECT count(*) FROM quejas"))
    assert s["resultado"]["filas"] == [[6646]]
    assert s["resultado"]["truncado"] is False
    assert s["filtros_aplicados"]["sql"].upper().startswith("SELECT")


def test_suma_pedidos_validos(datos):
    s = _ok(consultar_sql(datos, "SELECT sum(pedidos_validos) FROM ventas"))
    assert s["resultado"]["filas"] == [[3246856]]


def test_sucursales_por_region(datos):
    s = _ok(consultar_sql(datos, "SELECT region, count(*) AS n FROM sucursales GROUP BY 1 ORDER BY 1"))
    assert s["resultado"]["columnas"] == ["region", "n"]
    assert {r[0]: r[1] for r in s["resultado"]["filas"]} == {"Centro": 30, "Costa": 30, "Norte": 30, "Sur": 30}


def test_with_y_union_permitidos(datos):
    sql = ("WITH r AS (SELECT region, sum(quejas) AS q, sum(pedidos_validos) AS p FROM diario_sucursal "
           "WHERE NOT dato_invalido GROUP BY region) SELECT region, q * 1000.0 / p AS tasa FROM r "
           "UNION ALL SELECT 'total', sum(q) * 1000.0 / sum(p) FROM r")
    s = _ok(consultar_sql(datos, sql))
    assert len(s["resultado"]["filas"]) == 5


def test_fechas_como_iso(datos):
    s = _ok(consultar_sql(datos, "SELECT min(fecha), max(fecha) FROM ventas"))
    assert s["resultado"]["filas"] == [["2026-01-05", "2026-03-29"]]


def test_nulos_a_none(datos):
    s = _ok(consultar_sql(datos, "SELECT pedidos_validos FROM ventas WHERE dato_invalido"))
    assert len(s["resultado"]["filas"]) == 8
    assert all(f == [None] for f in s["resultado"]["filas"])


# --- límite de filas y tamaño -----------------------------------------------

def test_limite_200_filas(datos):
    s = _ok(consultar_sql(datos, "SELECT sucursal_id FROM diario_sucursal"))
    assert len(s["resultado"]["filas"]) == MAX_FILAS == 200
    assert s["resultado"]["truncado"] is True
    assert any("truncado" in a.lower() for a in s["advertencias"])
    _json_chico(s)


def test_select_estrella_diario_truncado_y_chico(datos):
    s = _ok(consultar_sql(datos, "SELECT * FROM diario_sucursal"))
    assert s["resultado"]["truncado"] is True
    assert 0 < len(s["resultado"]["filas"]) <= 200
    assert any("recortada" in a.lower() for a in s["advertencias"])
    _json_chico(s)


# --- rechazos ---------------------------------------------------------------

RECHAZADAS = [
    "DROP TABLE quejas",
    "INSERT INTO quejas VALUES ('x')",
    "SELECT 1; SELECT 2",
    "UPDATE quejas SET canal = 'app'",
    "DELETE FROM quejas",
    "CREATE TABLE x AS SELECT * FROM quejas",
    "ATTACH 'otra.db' AS otra",
    "COPY quejas TO 'salida.csv'",
    "COPY (SELECT * FROM quejas) TO 'salida.csv'",
    "PRAGMA show_tables",
    "SET threads = 1",
    "INSTALL httpfs",
    "LOAD httpfs",
    "EXPORT DATABASE 'x'",
    "CALL pragma_version()",
    "DESCRIBE quejas",
    "SELECT * FROM read_csv('datos/quejas.csv')",
    "SELECT * FROM read_csv_auto('datos/quejas.csv')",
    "SELECT * FROM read_parquet('x.parquet')",
    "SELECT * FROM read_json_auto('x.json')",
    "SELECT * FROM glob('*')",
    "SELECT * FROM parquet_scan('x.parquet')",
    "SELECT * FROM sniff_csv('datos/quejas.csv')",
    "SELECT * FROM 'datos/quejas.csv'",
    "SELECT * FROM \"datos/quejas.csv\"",
    "SELECT * FROM query_table('quejas')",
    "SELECT getenv('HOME')",
    "SELECT * FROM range(10)",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM duckdb_tables()",
    "SELECT * FROM duckdb_settings()",
    "SELECT * FROM pg_catalog.pg_tables",
    "SELECT * FROM main.quejas",
    "SELECT * FROM _interno_sql_ventas",
    "SELECT * FROM _df_ventas",
    "SELECT * FROM ventas_diarias",
    "SELECT * FROM quejas WHERE sucursal_id IN (SELECT sucursal_id FROM read_csv('x.csv'))",
    "WITH q AS (SELECT * FROM read_csv('x.csv')) SELECT * FROM q",
    "",
    "   ",
]


@pytest.mark.parametrize("sql", RECHAZADAS)
def test_rechaza(datos, sql):
    s = consultar_sql(datos, sql)
    _es_error(s)
    _json_chico(s)


def test_error_lista_vistas_permitidas(datos):
    s = consultar_sql(datos, "DROP TABLE quejas")
    assert set(VISTAS_PERMITIDAS) <= set(s["valores_validos"])


def test_sql_que_no_parsea(datos):
    s = consultar_sql(datos, "SELEC * FRM quejas ((")
    _es_error(s)


def test_columna_cruda_pedidos_no_existe(datos):
    s = consultar_sql(datos, "SELECT pedidos FROM ventas")
    _es_error(s)
    assert "pedidos" in s["error"]
    s = consultar_sql(datos, "SELECT sum(pedidos) FROM ventas")
    _es_error(s)


def test_defensa_en_profundidad_conexion(datos):
    """Aunque se saltee la validación, la conexión no accede a archivos ni cambia la configuración."""
    con, _ = sql_seguro._conexion(datos)
    with pytest.raises(Exception):
        con.execute("SELECT * FROM read_csv('datos/quejas.csv')").fetchall()
    with pytest.raises(Exception):
        con.execute("SET enable_external_access = true")
    nombres = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables "
                                         "WHERE table_type = 'VIEW'").fetchall()}
    assert set(VISTAS_PERMITIDAS) <= nombres
    assert "ventas_diarias" not in nombres
    assert con is not datos.con


def test_no_modifica_conexion_original(datos):
    consultar_sql(datos, "SELECT count(*) FROM ventas")
    cols = [r[0] for r in datos.con.execute("DESCRIBE ventas").fetchall()]
    assert "pedidos" in cols  # la conexión de Datos sigue intacta


# --- timeout ----------------------------------------------------------------

def test_timeout_y_conexion_usable(datos):
    lenta = "SELECT count(*) FROM diario_sucursal a, diario_sucursal b, sucursales c WHERE a.quejas + b.quejas > c.dias_operados"
    s = consultar_sql(datos, lenta, timeout_s=0.3)
    _es_error(s)
    assert "tiempo máximo" in s["error"]
    s = _ok(consultar_sql(datos, "SELECT count(*) FROM quejas"))
    assert s["resultado"]["filas"] == [[6646]]


# --- esquema ----------------------------------------------------------------

def test_esquema_vistas_coincide_con_columnas_reales(datos):
    con, _ = sql_seguro._conexion(datos)
    assert sql_seguro._describir_esquema(con) in ESQUEMA_VISTAS
    for vista in VISTAS_PERMITIDAS:
        reales = [r[0] for r in con.execute(f'DESCRIBE "{vista}"').fetchall()]
        linea = next(l for l in ESQUEMA_VISTAS.splitlines() if l.startswith(f"- {vista}("))
        declaradas = [c.strip().split(" ")[0] for c in linea[len(vista) + 3:-1].split(",")]
        assert declaradas == reales, vista


# --- serialización ----------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT * FROM quejas",
    "SELECT * FROM sucursales",
    "SELECT * FROM capacitaciones",
    "SELECT * FROM ventas",
    "SELECT avg(pedidos_validos), stddev(pedidos_validos), sum(pedidos_validos) / 3.0 FROM ventas",
    "SELECT fecha, semana_inicio, dato_invalido FROM ventas WHERE dato_invalido",
    "SELECT CAST(1.5 AS DECIMAL(10,2)) AS d, NULL AS n, 'NaN'::DOUBLE AS nan",
])
def test_salidas_json_y_chicas(datos, sql):
    s = consultar_sql(datos, sql)
    _ok(s)
    _json_chico(s)


def test_nan_a_none(datos):
    s = _ok(consultar_sql(datos, "SELECT 'NaN'::DOUBLE AS x"))
    assert s["resultado"]["filas"] == [[None]]

"""SQL de respaldo restringido (contrato C7).

`consultar_sql(datos, sql)` ejecuta una consulta de solo lectura escrita por el LLM cuando ninguna
herramienta tipada alcanza. Capas de defensa:

1. Validación estática con `sqlglot` (dialecto duckdb): una única sentencia SELECT (se permiten
   `WITH ... SELECT` y UNION/INTERSECT/EXCEPT de SELECTs); solo se pueden leer las 5 vistas
   limpias (o CTEs de la misma consulta); se rechazan funciones de archivos/red/catálogo
   (`read_*`, `glob`, `getenv`, `duckdb_*`, ...), tablas como función, rutas entre comillas,
   esquemas calificados (`information_schema.*`, `main.*`) y los nombres internos.
2. Conexión DuckDB en memoria propia del módulo (no `datos.con`), donde solo existen las vistas
   limpias. La vista `ventas` NO trae la columna cruda `pedidos` (solo `pedidos_validos` y
   `dato_invalido`), así un `SUM(pedidos)` no puede dar un total erróneo.
3. `enable_external_access=false` y `lock_configuration=true` en esa conexión.
4. Máximo 200 filas, timeout (~5 s, `con.interrupt()` desde un `threading.Timer`) y salida JSON
   nativa de menos de ~4 KB.

Nunca lanza excepciones: los errores vuelven como `{"error": ..., "valores_validos": [...]}`.
"""

from __future__ import annotations

import datetime as _dt
import decimal
import json
import math
import threading
from typing import TYPE_CHECKING, Any

import pandas as pd

from agente.config import FECHA_INICIO, FECHA_REFERENCIA

if TYPE_CHECKING:  # pragma: no cover
    import duckdb

    from agente.datos import Datos

VISTAS_PERMITIDAS: tuple[str, ...] = ("sucursales", "ventas", "quejas", "capacitaciones", "diario_sucursal")
PREFIJO_INTERNO = "_interno_sql_"
MAX_FILAS = 200
TIMEOUT_S = 5.0
MAX_BYTES_SALIDA = 4000

ADVERTENCIA_FIJA = (
    "Preferí las herramientas tipadas; las tasas son ratio de sumas sobre días válidos: "
    "SUM(quejas)/SUM(pedidos_validos)*1000 excluyendo dato_invalido"
)

# Columnas que no se exponen en la conexión restringida.
_COLUMNAS_EXCLUIDAS = {"ventas": ("pedidos",)}

# Funciones prohibidas (en minúscula). Además se rechaza todo lo que empiece con estos prefijos.
_FUNCIONES_PROHIBIDAS = {
    "glob", "parquet_scan", "sniff_csv", "query_table", "query", "getenv", "read_blob", "read_text",
    "parquet_metadata", "parquet_schema", "parquet_file_metadata", "parquet_kv_metadata",
    "iceberg_scan", "delta_scan", "postgres_scan", "sqlite_scan", "mysql_scan", "httpfs",
    "current_setting", "load_extension", "install_extension", "json_execute_serialized_sql",
}
_PREFIJOS_PROHIBIDOS = ("read_", "duckdb_", "pg_", "pragma_", "sqlite_", "postgres_", "mysql_", "iceberg_",
                        "delta_", "parquet_", "json_execute", "scan_", "glob")
_ESQUEMAS_PROHIBIDOS = ("information_schema", "duckdb_", "pg_", "main", "temp", "system", "memory")

_REGLA = ("Solo una sentencia SELECT (o WITH ... SELECT) sobre las vistas: " + ", ".join(VISTAS_PERMITIDAS)
          + "; sin funciones de archivos, red ni catálogo.")

_CONEXIONES: dict[int, tuple[Any, "duckdb.DuckDBPyConnection", threading.Lock]] = {}
_LOCK_CACHE = threading.Lock()


class _Rechazo(Exception):
    """Consulta rechazada por la validación estática."""


# ---------------------------------------------------------------------------
# Conexión restringida
# ---------------------------------------------------------------------------

def _crear_conexion(datos: "Datos") -> "duckdb.DuckDBPyConnection":
    import duckdb

    con = duckdb.connect(":memory:")
    for nombre in VISTAS_PERMITIDAS:
        df: pd.DataFrame = getattr(datos, nombre)
        df = df.drop(columns=[c for c in _COLUMNAS_EXCLUIDAS.get(nombre, ()) if c in df.columns])
        interno = f"{PREFIJO_INTERNO}{nombre}"
        con.register(interno, df)
        columnas = []
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                columnas.append(f'CAST("{c}" AS DATE) AS "{c}"')
            else:
                columnas.append(f'"{c}"')
        con.execute(f'CREATE VIEW "{nombre}" AS SELECT {", ".join(columnas)} FROM "{interno}"')
    con.execute("SET enable_external_access=false")
    try:
        con.execute("SET lock_configuration=true")
    except Exception:  # pragma: no cover - versiones viejas sin esta opción
        pass
    return con


def _conexion(datos: "Datos") -> tuple["duckdb.DuckDBPyConnection", threading.Lock]:
    """Conexión cacheada por id de `datos` (se guarda la referencia para que el id no se recicle)."""
    with _LOCK_CACHE:
        entrada = _CONEXIONES.get(id(datos))
        if entrada is None or entrada[0] is not datos:
            entrada = (datos, _crear_conexion(datos), threading.Lock())
            _CONEXIONES[id(datos)] = entrada
        return entrada[1], entrada[2]


def _describir_esquema(con: "duckdb.DuckDBPyConnection") -> str:
    lineas = []
    for nombre in VISTAS_PERMITIDAS:
        cols = con.execute(f'DESCRIBE "{nombre}"').fetchall()
        lineas.append(f"- {nombre}(" + ", ".join(f"{c[0]} {c[1]}" for c in cols) + ")")
    return "\n".join(lineas)


ESQUEMA_VISTAS: str = """Vistas disponibles para consultar_sql (DuckDB; fechas como DATE):
- sucursales(sucursal_id VARCHAR, nombre VARCHAR, region VARCHAR, ciudad VARCHAR, fecha_apertura DATE, es_nueva BOOLEAN, dias_operados BIGINT, fecha_inicio_datos DATE, fecha_capacitacion DATE, metodo_asignacion VARCHAR, capacitada BOOLEAN)
- ventas(fecha DATE, sucursal_id VARCHAR, region VARCHAR, pedidos_validos BIGINT, dato_invalido BOOLEAN, semana_inicio DATE, semana_n BIGINT, mes VARCHAR, dia_semana VARCHAR)
- quejas(queja_id VARCHAR, fecha DATE, sucursal_id VARCHAR, region VARCHAR, categoria VARCHAR, canal VARCHAR, comentario VARCHAR, en_dia_invalido BOOLEAN, semana_inicio DATE, semana_n BIGINT, mes VARCHAR, dia_semana VARCHAR)
- capacitaciones(sucursal_id VARCHAR, fecha_capacitacion DATE, metodo_asignacion VARCHAR)
- diario_sucursal(fecha DATE, sucursal_id VARCHAR, region VARCHAR, pedidos_validos BIGINT, quejas BIGINT, quejas_demora BIGINT, quejas_pedido_incorrecto BIGINT, quejas_trato BIGINT, quejas_limpieza BIGINT, quejas_calidad_producto BIGINT, dato_invalido BOOLEAN, semana_inicio DATE, semana_n BIGINT, mes VARCHAR, dia_semana VARCHAR)
Notas: pedidos_validos es NULL en los días con dato inválido (dato_invalido = true); la columna cruda
`pedidos` no existe. Tasa = SUM(quejas)/SUM(pedidos_validos)*1000 sobre filas con NOT dato_invalido.
mes es texto 'AAAA-MM'; dia_semana en español con tilde ('miércoles', 'sábado'); semana_n 1..12."""


# ---------------------------------------------------------------------------
# Validación estática
# ---------------------------------------------------------------------------

def _nombre_funcion(nodo) -> str:
    from sqlglot import exp

    if isinstance(nodo, exp.Anonymous):
        return str(nodo.name).lower()
    try:
        return str(nodo.sql_name()).lower()
    except Exception:  # pragma: no cover
        return type(nodo).__name__.lower()


def _validar(sql: str) -> str:
    """Valida la consulta y devuelve el SQL normalizado. Lanza `_Rechazo`."""
    import sqlglot
    from sqlglot import exp

    if not isinstance(sql, str) or not sql.strip():
        raise _Rechazo("La consulta está vacía.")
    try:
        sentencias = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except Exception as error:  # sqlglot.errors.ParseError y afines
        raise _Rechazo(f"No se pudo interpretar el SQL: {str(error)[:300]}") from None
    if len(sentencias) != 1:
        raise _Rechazo("Se permite una única sentencia SQL.")
    arbol = sentencias[0]

    raiz_ok = (exp.Select, exp.Union, exp.Intersect, exp.Except)
    if not isinstance(arbol, raiz_ok):
        raise _Rechazo(f"Solo se permiten consultas SELECT (se recibió {type(arbol).__name__.upper()}).")

    prohibidos = tuple(
        getattr(exp, n) for n in (
            "Insert", "Update", "Delete", "Create", "Drop", "Alter", "Command", "Copy", "Pragma", "Set",
            "Attach", "Detach", "Use", "Merge", "TruncateTable", "Install", "Load", "Export", "Summarize",
            "Describe", "Show", "Into", "LoadData", "Grant", "Transaction", "Commit", "Rollback", "Cache",
            "Uncache", "Refresh",
        ) if hasattr(exp, n)
    )
    for nodo in arbol.walk():
        if isinstance(nodo, prohibidos):
            raise _Rechazo(f"Operación no permitida: {type(nodo).__name__.upper()}.")

    # Identificadores internos en cualquier parte
    for ident in arbol.find_all(exp.Identifier):
        if str(ident.name).lower().startswith(PREFIJO_INTERNO):
            raise _Rechazo("Referencia a un objeto interno no permitida.")

    # Funciones
    for func in arbol.find_all(exp.Func):
        nombre = _nombre_funcion(func)
        if (nombre in _FUNCIONES_PROHIBIDAS or nombre.startswith(_PREFIJOS_PROHIBIDOS)
                or type(func).__name__.startswith("Read")):
            raise _Rechazo(f"Función no permitida: {nombre}.")

    # Tablas: solo vistas permitidas o CTEs definidos en la consulta
    ctes = {str(c.alias_or_name).lower() for c in arbol.find_all(exp.CTE)}
    permitidas = set(VISTAS_PERMITIDAS) | ctes
    for tabla in arbol.find_all(exp.Table):
        if not isinstance(tabla.this, exp.Identifier):
            raise _Rechazo("No se permiten funciones de tabla ni rutas de archivo en FROM.")
        esquema = (tabla.db or "").lower()
        catalogo = (tabla.catalog or "").lower()
        if esquema or catalogo:
            raise _Rechazo(f"No se permiten tablas calificadas con esquema ({esquema or catalogo}.{tabla.name}).")
        nombre = str(tabla.name).lower()
        if nombre.startswith(_ESQUEMAS_PROHIBIDOS) and nombre not in permitidas:
            raise _Rechazo(f"Acceso al catálogo no permitido: {tabla.name}.")
        if nombre not in permitidas:
            raise _Rechazo(f"Tabla o vista no permitida: {tabla.name}.")

    return arbol.sql(dialect="duckdb")


# ---------------------------------------------------------------------------
# Conversión a JSON
# ---------------------------------------------------------------------------

def _a_json(valor: Any) -> Any:
    if valor is None:
        return None
    if isinstance(valor, bool):
        return valor
    if isinstance(valor, (_dt.datetime, _dt.date, _dt.time)):
        return valor.isoformat()
    if isinstance(valor, decimal.Decimal):
        valor = float(valor)
    if isinstance(valor, float):
        return None if (math.isnan(valor) or math.isinf(valor)) else valor
    if isinstance(valor, int):
        return int(valor)
    if isinstance(valor, str):
        return valor
    if isinstance(valor, _dt.timedelta):
        return str(valor)
    try:
        if pd.isna(valor):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(valor, "item"):  # numpy escalar
        return _a_json(valor.item())
    if isinstance(valor, (list, tuple)):
        return [_a_json(v) for v in valor]
    if isinstance(valor, dict):
        return {str(k): _a_json(v) for k, v in valor.items()}
    if isinstance(valor, (bytes, bytearray)):
        return valor.hex()
    return str(valor)


def _error(mensaje: str, valores_validos: list | None = None) -> dict:
    return {"error": mensaje, "valores_validos": valores_validos if valores_validos is not None
            else [*VISTAS_PERMITIDAS, _REGLA]}


def _tamano(salida: dict) -> int:
    return len(json.dumps(salida, ensure_ascii=False).encode("utf-8"))


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def consultar_sql(datos: "Datos", sql: str, timeout_s: float = TIMEOUT_S) -> dict:
    """Ejecuta un SELECT de solo lectura sobre las vistas limpias (contrato C7, formato C2).

    Devuelve `{"resultado": {"columnas", "filas", "truncado"}, "filtros_aplicados": {"sql"},
    "periodo", "n_registros", "advertencias"}` o `{"error", "valores_validos"}`. Nunca lanza.
    `timeout_s` es inyectable para los tests.
    """
    try:
        sql_normalizado = _validar(sql)
    except _Rechazo as rechazo:
        return _error(str(rechazo))
    except Exception as error:  # pragma: no cover - defensa
        return _error(f"No se pudo validar el SQL: {error}")

    try:
        con, lock = _conexion(datos)
    except Exception as error:  # pragma: no cover
        return _error(f"No se pudo preparar la conexión de consulta: {error}", [])

    envuelta = f"SELECT * FROM ({sql_normalizado}) AS _sub_consulta LIMIT {MAX_FILAS + 1}"
    with lock:
        cortada = threading.Event()

        def _cortar() -> None:
            cortada.set()
            con.interrupt()

        temporizador = threading.Timer(timeout_s, _cortar)
        temporizador.daemon = True
        temporizador.start()
        try:
            cursor = con.execute(envuelta)
            columnas = [d[0] for d in cursor.description]
            filas = cursor.fetchall()
        except Exception as error:
            if cortada.is_set() or "interrupt" in str(error).lower():
                return _error(f"La consulta superó el tiempo máximo ({timeout_s:g} s).",
                              ["Simplificá la consulta: filtrá antes de unir o agregá con GROUP BY."])
            return _error(f"Error al ejecutar la consulta: {str(error)[:400]}")
        finally:
            temporizador.cancel()

    advertencias = [ADVERTENCIA_FIJA]
    truncado = len(filas) > MAX_FILAS
    filas = filas[:MAX_FILAS]
    if truncado:
        advertencias.append(f"Resultado truncado: había más de {MAX_FILAS} filas; agregá o filtrá.")

    filas_json = [[_a_json(v) for v in fila] for fila in filas]
    salida = {
        "resultado": {"columnas": columnas, "filas": filas_json, "truncado": truncado},
        "filtros_aplicados": {"sql": sql_normalizado},
        "periodo": {"desde": FECHA_INICIO.isoformat(), "hasta": FECHA_REFERENCIA.isoformat()},
        "n_registros": len(filas_json),
        "advertencias": advertencias,
    }

    if _tamano(salida) > MAX_BYTES_SALIDA:
        total = len(filas_json)
        aviso = "Salida recortada por tamaño (máx. ~4 KB): se muestran {n} de {t} filas; agregá o filtrá."
        advertencias.append(aviso.format(n=total, t=total))
        salida["resultado"]["truncado"] = True
        if len(sql_normalizado) > 1500:
            salida["filtros_aplicados"]["sql"] = sql_normalizado[:1500] + "…"
        # Búsqueda binaria de la cantidad de filas que entra.
        bajo, alto = 0, total
        while bajo < alto:
            medio = (bajo + alto + 1) // 2
            salida["resultado"]["filas"] = filas_json[:medio]
            salida["n_registros"] = medio
            advertencias[-1] = aviso.format(n=medio, t=total)
            if _tamano(salida) <= MAX_BYTES_SALIDA:
                bajo = medio
            else:
                alto = medio - 1
        salida["resultado"]["filas"] = filas_json[:bajo]
        salida["n_registros"] = bajo
        advertencias[-1] = aviso.format(n=bajo, t=total)
        if _tamano(salida) > MAX_BYTES_SALIDA:  # demasiadas columnas: se recortan también
            salida["resultado"]["columnas"] = columnas[:20]
            salida["resultado"]["filas"] = []
            salida["n_registros"] = 0
            advertencias[-1] = "Salida demasiado grande incluso sin filas; elegí menos columnas."
    return salida

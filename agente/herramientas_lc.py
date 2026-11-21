"""Wrappers LangChain (`StructuredTool`) sobre las herramientas puras de `herramientas.py` (WP-I).

Cada tool tiene un `args_schema` Pydantic explícito (tipos, `Literal` para los valores cerrados y
descripción de cada campo en español). La función envuelta cierra sobre `datos`, descarta los
argumentos `None` (así rigen los defaults de la herramienta pura) y devuelve el dict C2 tal cual.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Literal, Optional, Union

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from agente.herramientas import HERRAMIENTAS

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from agente.datos import Datos

Region = Literal["Norte", "Sur", "Centro", "Costa"]
Categoria = Literal["demora", "pedido_incorrecto", "trato", "limpieza", "calidad_producto"]
Canal = Literal["app", "telefono", "redes", "en_tienda"]

_D_REGION = "Región (solo si la pregunta la nombra)."
_D_SUC = "ID de sucursal (p. ej. 'S012') o lista de IDs."
_D_CAT = "Categoría de queja."
_D_CANAL = "Canal por el que llegó la queja."
_D_DESDE = "Fecha inicial inclusiva, ISO AAAA-MM-DD."
_D_HASTA = "Fecha final inclusiva, ISO AAAA-MM-DD."
_D_MES = "Mes 'AAAA-MM' (p. ej. '2026-02'); alternativa a fecha_desde/fecha_hasta."
_D_N = "Cantidad máxima de grupos/filas a listar (defecto 20, máx. 30)."


class _SinArgs(BaseModel):
    """Sin parámetros."""


class _Filtros(BaseModel):
    region: Optional[Region] = Field(default=None, description=_D_REGION)
    sucursal_id: Optional[Union[str, list[str]]] = Field(default=None, description=_D_SUC)
    fecha_desde: Optional[str] = Field(default=None, description=_D_DESDE)
    fecha_hasta: Optional[str] = Field(default=None, description=_D_HASTA)
    mes: Optional[str] = Field(default=None, description=_D_MES)


class ArgsBuscarSucursales(BaseModel):
    region: Optional[Region] = Field(default=None, description=_D_REGION)
    ciudad: Optional[str] = Field(default=None, description="Ciudad de la sucursal.")
    es_nueva: Optional[bool] = Field(default=None, description="True: solo las que abrieron dentro del periodo.")
    capacitada: Optional[bool] = Field(default=None, description="True: solo las capacitadas el 2026-02-09.")
    abiertas_desde: Optional[str] = Field(default=None, description="fecha_apertura >= esta fecha (AAAA-MM-DD).")
    abiertas_hasta: Optional[str] = Field(default=None, description="fecha_apertura <= esta fecha (AAAA-MM-DD).")
    limite: Optional[int] = Field(default=None, ge=1, description="Máximo de filas listadas (defecto 40).")


class ArgsContarQuejas(_Filtros):
    categoria: Optional[Categoria] = Field(default=None, description=_D_CAT)
    canal: Optional[Canal] = Field(default=None, description=_D_CANAL)
    agrupar_por: Optional[
        Literal["region", "sucursal", "categoria", "canal", "mes", "semana", "dia_semana", "fecha"]
    ] = Field(default=None, description="Agrupación de los conteos (grupos de mayor a menor).")
    n: Optional[int] = Field(default=None, ge=1, description=_D_N)


class ArgsPedidos(_Filtros):
    agrupar_por: Optional[Literal["region", "sucursal", "mes", "semana", "dia_semana", "fecha"]] = Field(
        default=None, description="Agrupación de los pedidos."
    )
    agregacion: Optional[Literal["suma", "promedio_por_sucursal_dia", "promedio_diario_total"]] = Field(
        default=None, description="Cómo agregar los pedidos válidos (defecto 'suma')."
    )
    n: Optional[int] = Field(default=None, ge=1, description=_D_N)


class ArgsTasaQuejas(_Filtros):
    categoria: Optional[Categoria] = Field(default=None, description=_D_CAT + " Numerador solo de esa categoría.")
    canal: Optional[Canal] = Field(default=None, description=_D_CANAL + " Numerador solo de ese canal.")
    agrupar_por: Optional[Literal["region", "sucursal", "mes", "semana", "dia_semana"]] = Field(
        default=None, description="Agrupación: tasa de cada grupo."
    )
    n: Optional[int] = Field(default=None, ge=1, description=_D_N)


class ArgsRanking(_Filtros):
    metrica: Optional[Literal["quejas", "tasa", "pedidos"]] = Field(
        default=None, description="Métrica del ranking (defecto 'tasa' = quejas por 1.000 pedidos)."
    )
    categoria: Optional[Categoria] = Field(default=None, description=_D_CAT)
    canal: Optional[Canal] = Field(default=None, description=_D_CANAL)
    n: Optional[int] = Field(default=None, ge=1, description="Cantidad de sucursales (defecto 5, máx. 30).")
    orden: Optional[Literal["desc", "asc"]] = Field(
        default=None, description="'desc' = las de más (defecto), 'asc' = las de menos."
    )
    incluir_nuevas: Optional[bool] = Field(
        default=None, description="Solo para tasa: incluir sucursales con menos de 28 días abiertas."
    )


class ArgsComentarios(BaseModel):
    sucursal_id: str = Field(description="ID de UNA sucursal, p. ej. 'S104'.")
    categoria: Optional[Categoria] = Field(default=None, description="Restringe los comentarios a esa categoría.")
    top_k: Optional[int] = Field(default=None, ge=1, description="Comentarios a listar (defecto 5, máx. 10).")
    fecha_desde: Optional[str] = Field(default=None, description=_D_DESDE)
    fecha_hasta: Optional[str] = Field(default=None, description=_D_HASTA)
    mes: Optional[str] = Field(default=None, description=_D_MES)


class ArgsCandidatos(BaseModel):
    region: Optional[Region] = Field(default=None, description=_D_REGION)
    categoria: Optional[Categoria] = Field(
        default=None, description="Si la pregunta se enfoca en una categoría (p. ej. limpieza)."
    )
    n: Optional[int] = Field(default=None, ge=1, description="Cantidad exacta pedida (si la pregunta la da).")
    fecha_hasta: Optional[str] = Field(
        default=None, description="Fin de la ventana de 4 semanas, AAAA-MM-DD (defecto 2026-03-29)."
    )
    incluir_nuevas: Optional[bool] = Field(
        default=None, description="Incluir sucursales con menos de 28 días abiertas (con advertencia)."
    )


class ArgsSQL(BaseModel):
    sql: str = Field(description="Un único SELECT de solo lectura sobre las vistas documentadas en el prompt.")


DESC_CANDIDATOS = (
    "Sucursales a visitar / donde actuar, ordenadas por un score determinístico "
    "(ratio_regional = tasa de quejas de las últimas 4 semanas / tasa de su región). Usala SIEMPRE que la "
    "pregunta pida recomendaciones. Devuelve `candidatos` en orden, con tasa_4sem, tasa_region_4sem, "
    "ratio_regional, quejas_4sem, pedidos_4sem, tendencia, categoria_dominante, pct_categoria_dominante, "
    "capacitada, preventiva y accion_sugerida_base."
)
DESC_SQL = (
    "Respaldo: ejecuta un SELECT de solo lectura (DuckDB) sobre las vistas limpias. Usala solo si ninguna "
    "herramienta tipada responde la pregunta. Poné alias a las columnas agregadas."
)

_ESQUEMAS: dict[str, type[BaseModel]] = {
    "info_datos": _SinArgs,
    "buscar_sucursales": ArgsBuscarSucursales,
    "contar_quejas": ArgsContarQuejas,
    "pedidos": ArgsPedidos,
    "tasa_quejas": ArgsTasaQuejas,
    "ranking_sucursales": ArgsRanking,
    "comentarios_sucursal": ArgsComentarios,
    "efecto_capacitacion": _SinArgs,
    "candidatos_visita": ArgsCandidatos,
    "consultar_sql": ArgsSQL,
}
_DESCRIPCIONES_PROPIAS = {"candidatos_visita": DESC_CANDIDATOS, "consultar_sql": DESC_SQL}


def _descripcion(nombre: str, funcion: Callable) -> str:
    if nombre in _DESCRIPCIONES_PROPIAS:
        return _DESCRIPCIONES_PROPIAS[nombre]
    return inspect.cleandoc(funcion.__doc__ or nombre)


def _envolver(funcion: Callable[..., dict], datos: "Datos") -> Callable[..., dict]:
    def ejecutar(**kwargs: Any) -> dict:
        return funcion(datos, **{k: v for k, v in kwargs.items() if v is not None})

    return ejecutar


def crear_herramientas(datos: "Datos") -> list[BaseTool]:
    """Un `StructuredTool` por cada entrada de `HERRAMIENTAS`, con `datos` cerrado en el closure."""
    faltan = set(HERRAMIENTAS) - set(_ESQUEMAS)
    if faltan:  # pragma: no cover - defensa ante herramientas nuevas sin esquema
        raise RuntimeError(f"Herramientas sin args_schema: {sorted(faltan)}")
    return [
        StructuredTool.from_function(
            func=_envolver(funcion, datos),
            name=nombre,
            description=_descripcion(nombre, funcion),
            args_schema=_ESQUEMAS[nombre],
        )
        for nombre, funcion in HERRAMIENTAS.items()
    ]

"""Fachada `responder()` (contrato C5) y post-proceso de recomendaciones (WP-R).

- `responder(pregunta)` arma el grafo con las herramientas LangChain, corre la pregunta y devuelve una
  `Respuesta` con la traza (latencia, tokens, costo, llamadas, verificación, advertencias). Nunca
  propaga errores del proveedor: devuelve una `Respuesta` de tipo dato con el error en la traza.
- Recomendaciones (D4.2: el score decide, el LLM solo redacta): las sucursales recomendadas son
  exactamente las de `candidatos_visita`, en el mismo orden. Si el LLM cambia el orden, agrega otra
  sucursal o le falta alguna, se corrige en forma determinística y se deja en `traza.advertencias`.
  Los ítems descartados por el verificador se reponen con el hook `completar_recomendaciones`.
- Red de seguridad (C2): si la respuesta es de recomendaciones, o se llamó a `candidatos_visita`, o la
  pregunta pide recomendaciones según `es_pregunta_de_recomendacion`, y la salida no es una lista de
  recomendaciones alineada con `candidatos_visita`, el JSON se arma en forma determinística
  (`candidatos_visita` + `item_deterministico`). Cubre el corte por pasos, la respuesta de tipo dato,
  los argumentos inválidos repetidos, las recomendaciones sin candidatos (ítems inventados) y los errores
  del modelo (salvo el límite de tasa, que se propaga).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from agente.config import (
    RAIZ,
    clave_api,
    costo_usd,
    datos_dir,
    kwargs_modelo,
    modelo_llm,
    tool_choice_para,
    variable_clave_para,
)
from agente.datos import CATEGORIAS, REGIONES
from agente.esquemas import ItemRecomendacion, Respuesta, RespuestaFinal
from agente.estado import estado_inicial
from agente.grafo import config_invocacion, construir_grafo
from agente.herramientas_lc import crear_herramientas
from agente.prompts import SYSTEM_PROMPT
from agente.recomendaciones import PREFIJO_PREVENTIVA, candidatos_visita

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models import BaseChatModel

    from agente.datos import Datos

LARGO_MAX_RESULTADO_TRAZA = 1500
NOMBRE_CANDIDATOS = "candidatos_visita"

_cache_datos: "Datos | None" = None
_candado = threading.Lock()


# --------------------------------------------------------------------------- datos y modelo


class ErrorDeDatos(RuntimeError):
    """No se pudieron cargar los CSV (directorio inexistente, archivo faltante o ilegible)."""


def obtener_datos() -> "Datos":
    """Datos cargados una sola vez por proceso. Falla con `ErrorDeDatos` y un mensaje claro."""
    global _cache_datos
    with _candado:
        if _cache_datos is None:
            from agente.datos import cargar_datos

            try:
                _cache_datos = cargar_datos()
            except Exception as error:  # noqa: BLE001  (se re-lanza con contexto)
                raise ErrorDeDatos(
                    f"No pude cargar los datos de {datos_dir()} (revisá DATOS_DIR): {type(error).__name__}: {error}"
                ) from error
        return _cache_datos


def crear_modelo() -> "BaseChatModel":
    """Modelo real: falla con `FaltaVariableDeEntorno` si no está la clave del proveedor.

    Los kwargs dependen del modelo (`kwargs_modelo`): Sonnet 5.5 no acepta `temperature`.
    """
    nombre = modelo_llm()
    clave_api(variable_clave_para(nombre))
    from langchain.chat_models import init_chat_model

    return init_chat_model(nombre, **kwargs_modelo(nombre))


def _nombre_modelo(modelo: Any) -> str:
    for atributo in ("model", "model_name"):
        valor = getattr(modelo, atributo, None)
        if isinstance(valor, str) and valor:
            return valor
    return type(modelo).__name__


def trazas_dir() -> Path:
    valor = os.environ.get("TRAZAS_DIR")
    return Path(valor) if valor else RAIZ / "reportes" / "trazas"


# --------------------------------------------------------------------------- recomendaciones (WP-R)


def _num(x: Any) -> str:
    """Cifra tal cual del resultado, con coma decimal (sin separador de miles)."""
    if x is None:
        return "s/d"
    if isinstance(x, float) and x.is_integer():
        x = int(x) if abs(x) >= 100 else x
    return str(x).replace(".", ",")


def ultima_llamada_candidatos(estado: dict) -> dict | None:
    """Última llamada exitosa a `candidatos_visita` registrada en `estado["llamadas"]`."""
    for llamada in reversed(estado.get("llamadas") or []):
        resultado = llamada.get("resultado")
        if llamada.get("herramienta") == NOMBRE_CANDIDATOS and isinstance(resultado, dict) and "resultado" in resultado:
            return llamada
    return None


def item_deterministico(candidato: dict, periodo: str | None = None) -> ItemRecomendacion:
    """Ítem armado solo con los componentes del score: las cifras verifican por construcción."""
    c = candidato
    tendencia = c.get("tendencia") or {}
    etiqueta = tendencia.get("etiqueta") or "sin_datos"
    region = c.get("region")
    cuerpo = (
        f"su tasa de quejas de las últimas 4 semanas ({_num(c.get('tasa_4sem'))} por 1.000 pedidos) es "
        f"{_num(c.get('ratio_regional'))} veces la de la región {region}"
    )
    if c.get("preventiva"):
        motivo = f"{PREFIJO_PREVENTIVA} {cuerpo}."
    else:
        motivo = cuerpo[0].upper() + cuerpo[1:] + ", por encima del umbral."
    if etiqueta in ("sube", "baja", "estable"):
        motivo += f" Tendencia: {etiqueta}."
    if c.get("categoria_dominante"):
        motivo += f" Predomina {c['categoria_dominante']}."

    partes = [
        f"Tasa 4 semanas{f' ({periodo})' if periodo else ''}: {_num(c.get('tasa_4sem'))} quejas por 1.000 pedidos "
        f"vs {_num(c.get('tasa_region_4sem'))} de la región {region} (ratio {_num(c.get('ratio_regional'))})",
        f"{_num(c.get('quejas_4sem'))} quejas sobre {_num(c.get('pedidos_4sem'))} pedidos",
    ]
    if etiqueta == "sin_datos":
        partes.append("tendencia: sin datos del periodo anterior")
    else:
        partes.append(
            f"tendencia: {etiqueta} (ratio {_num(tendencia.get('ratio'))}; tasa anterior "
            f"{_num(c.get('tasa_4sem_anterior'))})"
        )
    if c.get("categoria_dominante"):
        partes.append(
            f"categoría dominante: {c['categoria_dominante']} ({_num(c.get('pct_categoria_dominante'))}% de las quejas)"
        )
    if c.get("capacitada"):
        partes.append(f"capacitada el {c.get('fecha_capacitacion')}")
    evidencia = "; ".join(partes) + "."
    for adv in c.get("advertencias") or []:
        evidencia += f" Advertencia: {adv}."
    accion = str(c.get("accion_sugerida_base") or "revisión general de la operación")
    return ItemRecomendacion(
        sucursal_id=str(c["sucursal_id"]),
        motivo=motivo,
        evidencia=evidencia,
        accion_sugerida=accion[0].upper() + accion[1:],
    )


def _items_por_id(items: list[ItemRecomendacion] | None) -> dict[str, ItemRecomendacion]:
    salida: dict[str, ItemRecomendacion] = {}
    for item in items or []:
        salida.setdefault(item.sucursal_id.strip().upper(), item)
    return salida


def completar_recomendaciones(
    respuesta_filtrada: RespuestaFinal,
    items_descartados: list[ItemRecomendacion],
    estado: dict,
    *,
    datos: "Datos | None" = None,
    registro: dict | None = None,
) -> RespuestaFinal:
    """Hook del nodo de cierre: repone los ítems descartados con los siguientes candidatos.

    Toma los args de la última llamada a `candidatos_visita`, vuelve a pedir candidatos con `n` mayor si
    hace falta y completa, en orden, con candidatos que no estén ya ni hayan sido descartados, hasta la
    cantidad original. Los ítems nuevos se arman en forma determinística (`item_deterministico`).
    `registro` (opcional) recibe `esperado` (IDs en orden) y `resultado` (el resultado usado).
    """
    llamada = ultima_llamada_candidatos(estado)
    if llamada is None:
        return respuesta_filtrada
    original = estado.get("respuesta")
    ids_originales = list(_items_por_id(original.recomendaciones if original else respuesta_filtrada.recomendaciones))
    conservados = _items_por_id(respuesta_filtrada.recomendaciones)
    excluidos = {i.sucursal_id.strip().upper() for i in items_descartados} - set(conservados)
    objetivo = max(len(ids_originales), len(conservados))

    resultado = llamada["resultado"]
    candidatos = resultado["resultado"].get("candidatos") or []
    necesario = objetivo + len(excluidos)
    if len(candidatos) < necesario and datos is not None:
        args = {k: v for k, v in (llamada.get("args") or {}).items() if v is not None}
        args["n"] = necesario
        nuevo = candidatos_visita(datos, **args)
        if isinstance(nuevo, dict) and "resultado" in nuevo:
            resultado = nuevo
            candidatos = nuevo["resultado"].get("candidatos") or []

    periodo = (resultado.get("resultado", {}).get("parametros") or {}).get("ventana")
    elegidos = [c for c in candidatos if str(c["sucursal_id"]).upper() not in excluidos][:objetivo]
    items = [conservados.get(str(c["sucursal_id"]).upper()) or item_deterministico(c, periodo) for c in elegidos]
    if registro is not None:
        registro["esperado"] = [str(c["sucursal_id"]) for c in elegidos]
        registro["resultado"] = resultado
        registro["candidatos"] = elegidos
    return respuesta_filtrada.model_copy(update={"recomendaciones": items})


def alinear_con_candidatos(
    respuesta: RespuestaFinal, estado: dict, registro: dict | None = None
) -> tuple[RespuestaFinal, list[str]]:
    """D4.2: deja exactamente las sucursales de `candidatos_visita`, en su orden.

    Devuelve (respuesta corregida, advertencias). No inventa: si no hubo `candidatos_visita`, deja la
    respuesta como vino y lo advierte.
    """
    advertencias: list[str] = []
    llamada = ultima_llamada_candidatos(estado)
    if respuesta.tipo_respuesta != "recomendaciones":
        if llamada is not None:
            advertencias.append("Se llamó a candidatos_visita pero la respuesta es de tipo dato.")
        return respuesta, advertencias
    if llamada is None:
        advertencias.append(
            "Respuesta de recomendaciones sin llamar a candidatos_visita: no se pudo validar el orden."
        )
        return respuesta, advertencias

    registro = registro or {}
    if "candidatos" in registro:
        candidatos = registro["candidatos"]
        resultado = registro["resultado"]
    else:
        resultado = llamada["resultado"]
        candidatos = resultado["resultado"].get("candidatos") or []
    periodo = (resultado.get("resultado", {}).get("parametros") or {}).get("ventana")
    esperado = [str(c["sucursal_id"]).upper() for c in candidatos]
    del_llm = _items_por_id(respuesta.recomendaciones)
    ids_llm = [i.sucursal_id.strip().upper() for i in respuesta.recomendaciones or []]

    items = [del_llm.get(i) or item_deterministico(c, periodo) for i, c in zip(esperado, candidatos)]
    if ids_llm != esperado:
        extra = [i for i in ids_llm if i not in esperado]
        faltan = [i for i in esperado if i not in del_llm]
        if extra:
            advertencias.append(
                "Reemplacé sucursales que no están en candidatos_visita: " + ", ".join(dict.fromkeys(extra)) + "."
            )
        if faltan:
            advertencias.append("Completé con texto determinístico: " + ", ".join(faltan) + ".")
        if not extra and not faltan:
            advertencias.append("Corregí el orden de las recomendaciones al de candidatos_visita.")
        elif [i for i in ids_llm if i in esperado] != [i for i in esperado if i in del_llm]:
            advertencias.append("Corregí el orden de las recomendaciones al de candidatos_visita.")
        if len(ids_llm) != len(set(ids_llm)):
            advertencias.append("Quité sucursales repetidas.")
    corregida = respuesta.model_copy(
        update={"recomendaciones": items, "texto": None, "valor": None, "tipo_valor": "ninguno"}
    )
    return corregida, advertencias


# --------------------------------------------------------------------------- red de seguridad (C2)


def _normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto).lower())
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


# Patrones sobre el texto en minúsculas y sin tildes.
_PATRONES_RECOMENDACION = [
    r"\bvisit",  # visitar, visita(s), visito, plan de visitas
    r"\brecomend",  # recomendame, recomendación, recomendás
    r"\bsuger|\bsugier",  # sugerime, sugerencias, sugiere
    r"\bque (deberia|tendria que|tengo que|hay que|conviene) (revisar|hacer|atacar|priorizar|mirar|mejorar)",
    r"\bdonde\b.*\b(reforz|refuerz|actu|interven|enfoc|foco|poner el ojo)",
    r"\bnecesit\w* atencion",
    r"\ba que (locales|sucursales|tiendas|cafeterias)\b",
    r"\bpriori",  # priorizar, prioridad(es)
    r"\batac",  # atacar, ataco
    r"\bplan de accion",
    r"\bpor donde (empiezo|empezar|arranco|arrancar)",
]


def es_pregunta_de_recomendacion(pregunta: str) -> bool:
    """Clasificador determinístico por palabras clave: ¿la pregunta pide recomendaciones / un plan?

    "¿Cuál es la peor sucursal?" o "¿cuántas quejas...?" son datos: dan False.
    """
    texto = _normalizar(pregunta)
    return any(re.search(p, texto) for p in _PATRONES_RECOMENDACION)


_NUMEROS = {"una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
            "ocho": 8, "nueve": 9, "diez": 10}
_CATEGORIAS_PREGUNTA = [
    (r"\blimpi", "limpieza"),
    (r"\bdemor|\btardanz|\bespera", "demora"),
    (r"\btrato\b", "trato"),
    (r"\bpedidos? (incorrecto|equivocado|errone)", "pedido_incorrecto"),
    (r"\bcalidad", "calidad_producto"),
]


def parametros_de_pregunta(pregunta: str) -> dict:
    """Args de `candidatos_visita` detectados por palabras clave: region, categoria y n."""
    texto = _normalizar(pregunta)
    args: dict = {}
    for region in REGIONES:
        if re.search(rf"\b{_normalizar(region)}\b", texto):
            args["region"] = region
            break
    for patron, categoria in _CATEGORIAS_PREGUNTA:
        if re.search(patron, texto) and categoria in CATEGORIAS:
            args["categoria"] = categoria
            break
    numero = r"(\d{1,2}|" + "|".join(_NUMEROS) + r")"
    m = re.search(rf"\b(?:dame|decime|recomendame|sugerime|top|las|los|primeras|primeros)\s+{numero}\b", texto) or re.search(
        rf"\b{numero}\s+(?:sucursales|locales|tiendas|cafeterias|lugares)\b", texto
    )
    if m:
        crudo = m.group(1)
        n = int(crudo) if crudo.isdigit() else _NUMEROS[crudo]
        if n >= 1:
            args["n"] = n
    return args


def _resultado_candidatos(estado: dict, pregunta: str, datos: "Datos") -> dict | None:
    """Resultado C2 de candidatos a usar: el de la última llamada exitosa del modelo; si no hubo, se
    llama a `candidatos_visita` con lo detectado en la pregunta (y, si eso da error, con los defaults)."""
    llamada = ultima_llamada_candidatos(estado)
    if llamada is not None:
        return llamada["resultado"]
    for args in (parametros_de_pregunta(pregunta), {}):
        resultado = candidatos_visita(datos, **args)
        if isinstance(resultado, dict) and "resultado" in resultado:
            return resultado
    return None


def recomendaciones_deterministicas(estado: dict, pregunta: str, datos: "Datos") -> RespuestaFinal | None:
    """Lista de recomendaciones armada solo con `candidatos_visita` + `item_deterministico`."""
    resultado = _resultado_candidatos(estado, pregunta, datos)
    if resultado is None:
        return None
    candidatos = resultado["resultado"].get("candidatos") or []
    if not candidatos:
        return None
    periodo = (resultado["resultado"].get("parametros") or {}).get("ventana")
    return RespuestaFinal(
        tipo_respuesta="recomendaciones",
        texto=None,
        valor=None,
        tipo_valor="ninguno",
        recomendaciones=[item_deterministico(c, periodo) for c in candidatos],
    )


def _motivo_reposicion(respuesta: RespuestaFinal | None, estado: dict) -> str | None:
    """Por qué la salida no sirve como recomendaciones (None si es una lista válida y alineada)."""
    llamada = ultima_llamada_candidatos(estado)
    if respuesta is None:
        return "el modelo no devolvió una respuesta"
    if respuesta.tipo_respuesta != "recomendaciones":
        if any("máximo de" in a and "pasos" in a for a in estado.get("advertencias") or []):
            return "se alcanzó el máximo de pasos sin una respuesta final válida"
        return "la respuesta del modelo fue de tipo dato"
    if llamada is None:
        return "el modelo respondió recomendaciones sin llamar a candidatos_visita (se descartaron sus ítems)"
    if not respuesta.recomendaciones:
        return "la respuesta de recomendaciones quedó vacía"
    return None


def asegurar_recomendaciones(
    respuesta: RespuestaFinal, estado: dict, pregunta: str, datos: "Datos"
) -> tuple[RespuestaFinal, list[str]]:
    """Regla final de `responder()`: garantiza el JSON de recomendaciones cuando corresponde.

    Corresponde si la respuesta es de recomendaciones, si se llamó a `candidatos_visita` o si la
    pregunta pide recomendaciones. Una respuesta de tipo dato a una pregunta que no es de recomendación
    y sin `candidatos_visita` no se toca.
    """
    # Un dato ("¿cuál es la peor sucursal?") no se convierte en JSON solo porque el modelo haya mirado
    # candidatos_visita: hace falta que la pregunta pida recomendaciones, o que el modelo no haya podido
    # terminar (corte por pasos o error) después de consultar candidatos.
    sin_terminar = respuesta.tipo_respuesta == "dato" and (
        (respuesta.texto or "").startswith(("No pude completar la consulta",))
    )
    corresponde = (
        respuesta.tipo_respuesta == "recomendaciones"
        or es_pregunta_de_recomendacion(pregunta)
        or (ultima_llamada_candidatos(estado) is not None and sin_terminar)
    )
    if not corresponde:
        return respuesta, []
    motivo = _motivo_reposicion(respuesta, estado)
    if motivo is None:
        return respuesta, []
    nueva = recomendaciones_deterministicas(estado, pregunta, datos)
    if nueva is None:
        return respuesta, [f"No pude armar recomendaciones determinísticas ({motivo}): candidatos_visita sin resultados."]
    return nueva, [f"Recomendaciones armadas en forma determinística porque {motivo}."]


# --------------------------------------------------------------------------- traza


def _recortar(resultado: Any) -> Any:
    texto = json.dumps(resultado, ensure_ascii=False, default=str)
    if len(texto) <= LARGO_MAX_RESULTADO_TRAZA:
        return resultado
    return texto[:LARGO_MAX_RESULTADO_TRAZA] + "…"


def guardar_traza(traza: dict) -> Path | None:
    """Agrega la traza como una línea JSONL en `<TRAZAS_DIR>/AAAAMMDD.jsonl`. Nunca lanza."""
    try:
        carpeta = trazas_dir()
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / f"{datetime.now():%Y%m%d}.jsonl"
        with ruta.open("a", encoding="utf-8") as f:
            f.write(json.dumps(traza, ensure_ascii=False, default=str) + "\n")
        return ruta
    except Exception:  # noqa: BLE001  (la traza nunca rompe una respuesta)
        return None


# --------------------------------------------------------------------------- responder


def es_rate_limit(error: BaseException) -> bool:
    """True si el error es un límite de tasa del proveedor (429): ese error se propaga."""
    texto = f"{type(error).__name__} {error}".lower()
    return "ratelimit" in texto.replace("_", "") or "rate limit" in texto or "429" in texto


def responder(
    pregunta: str, *, modelo: "BaseChatModel | None" = None, datos: "Datos | None" = None
) -> Respuesta:
    """Responde una pregunta del gerente (C5). Ver docstring del módulo.

    Lanza `FaltaVariableDeEntorno` (sin clave), `ErrorDeDatos` (no se pudieron cargar los CSV) y los
    errores de límite de tasa. Cualquier otro error del proveedor o del grafo vuelve como `Respuesta`.
    """
    tool_choice = None
    if modelo is None:
        modelo = crear_modelo()  # lanza FaltaVariableDeEntorno si falta la clave
        tool_choice = tool_choice_para(modelo_llm())
    datos = datos if datos is not None else obtener_datos()  # lanza ErrorDeDatos con mensaje claro
    nombre_modelo = _nombre_modelo(modelo)
    inicio = time.perf_counter()
    traza: dict[str, Any] = {"pregunta": pregunta, "modelo": nombre_modelo, "error": None}

    registro: dict = {}

    def hook(respuesta_filtrada, descartados, estado):
        return completar_recomendaciones(respuesta_filtrada, descartados, estado, datos=datos, registro=registro)

    try:
        grafo = construir_grafo(
            modelo,
            crear_herramientas(datos),
            ids_validos=set(datos.sucursales["sucursal_id"].astype(str)),
            completar_recomendaciones=hook,
            tool_choice=tool_choice,
        )
        inicial = estado_inicial(pregunta, [SystemMessage(SYSTEM_PROMPT), HumanMessage(pregunta)])
        estado = grafo.invoke(inicial, config_invocacion())
        final, adv_alineacion = alinear_con_candidatos(estado["respuesta"], estado, registro)
        final, adv_red = asegurar_recomendaciones(final, estado, pregunta, datos)
        if adv_red and final.tipo_respuesta == "recomendaciones":
            # la advertencia de "sin candidatos_visita" ya no aplica: los ítems se rearmaron
            adv_alineacion = [a for a in adv_alineacion if "candidatos_visita" not in a]
        respuesta = Respuesta(**final.model_dump())
        uso = estado.get("uso") or {}
        traza.update(
            pasos=estado.get("pasos", 0),
            llamadas=[
                {"herramienta": c["herramienta"], "args": c.get("args"), "resultado": _recortar(c.get("resultado"))}
                for c in estado.get("llamadas") or []
            ],
            verificacion=estado.get("verificacion"),
            advertencias=list(estado.get("advertencias") or []) + adv_alineacion + adv_red,
        )
    except Exception as error:  # noqa: BLE001  (error del proveedor u otro: el eval sigue)
        if es_rate_limit(error):
            raise  # el eval reintenta con backoff; main.py lo informa
        detalle = f"{type(error).__name__}: {error}"
        advertencias: list[str] = []
        recos = recomendaciones_deterministicas({}, pregunta, datos) if es_pregunta_de_recomendacion(pregunta) else None
        if recos is not None:
            respuesta = Respuesta(**recos.model_dump())
            advertencias.append(
                f"Recomendaciones armadas en forma determinística porque el modelo falló ({detalle})."
            )
        else:
            respuesta = Respuesta(
                tipo_respuesta="dato",
                texto=f"No pude completar la consulta: {detalle}",
                valor=None,
                tipo_valor="ninguno",
            )
        uso = {}
        traza.update(pasos=None, llamadas=[], verificacion=None, advertencias=advertencias, error=detalle)

    uso = {"tokens_entrada": int(uso.get("tokens_entrada", 0)), "tokens_salida": int(uso.get("tokens_salida", 0))}
    traza["latencia_s"] = round(time.perf_counter() - inicio, 3)
    traza["uso"] = uso
    traza["costo_usd"] = costo_usd(nombre_modelo, uso["tokens_entrada"], uso["tokens_salida"])
    respuesta.traza = traza
    guardar_traza({"fecha": datetime.now().isoformat(timespec="seconds"), **traza})
    return respuesta

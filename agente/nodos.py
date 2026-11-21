"""Nodos del grafo (Etapa 3).

Cada nodo es una función `estado -> actualización parcial`. Los que necesitan el modelo, las
herramientas o el verificador se construyen con fábricas (`crear_nodo_*`) que devuelven closures, así
se pueden probar sueltos.

Invariante de la API de Anthropic: todo `tool_use` de un AIMessage tiene su ToolMessage (mismo
`tool_call_id`) antes de la próxima llamada al modelo. Por eso también se responden las llamadas a
`RespuestaFinal` que se ignoran, que tienen argumentos inválidos o que no pasan la verificación.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from agente.config import MAX_PASOS
from agente.esquemas import ItemRecomendacion, RespuestaFinal
from agente.estado import Estado

NOMBRE_FINAL = RespuestaFinal.__name__  # "RespuestaFinal"

RECORDATORIO = "Respondé llamando a RespuestaFinal"
TEXTO_SIN_RESPUESTA = "No pude completar la consulta con los datos disponibles."
MENSAJE_FINAL_IGNORADA = (
    "RespuestaFinal ignorada: llamala sola, en un turno aparte, después de tener los resultados de las "
    "herramientas."
)
MENSAJE_FINAL_DUPLICADA = "RespuestaFinal duplicada en el mismo turno: solo se considera la primera."

Verificador = Callable[[RespuestaFinal, list, str, set], dict]
# Hook de WP-R: recibe (respuesta_filtrada, items_descartados, estado) y devuelve la respuesta
# completada con los siguientes candidatos. Si es None, la respuesta queda solo con los ítems que
# verificaron.
CompletarRecomendaciones = Callable[[RespuestaFinal, list[ItemRecomendacion], Estado], RespuestaFinal]


# --------------------------------------------------------------------------- utilidades


def a_json(valor: Any) -> str:
    """Serializa un resultado de herramienta para el ToolMessage."""
    if isinstance(valor, str):
        return valor
    return json.dumps(valor, ensure_ascii=False, default=str)


def ultimo_ai(estado: Estado) -> AIMessage | None:
    """Último AIMessage del historial (o None)."""
    for mensaje in reversed(estado["messages"]):
        if isinstance(mensaje, AIMessage):
            return mensaje
    return None


def _llamadas_finales(mensaje: AIMessage | None) -> list[dict]:
    if mensaje is None:
        return []
    return [c for c in mensaje.tool_calls if c["name"] == NOMBRE_FINAL]


def resultados_de(estado: Estado) -> list[dict | str]:
    """Todos los resultados de herramientas de esta ejecución (para el verificador)."""
    return [llamada["resultado"] for llamada in estado.get("llamadas") or []]


def _error(mensaje: str, valores_validos: list | None = None) -> dict:
    return {"error": mensaje, "valores_validos": list(valores_validos or [])}


# --------------------------------------------------------------------------- modelo


def crear_nodo_modelo(modelo_ligado: Any) -> Callable[[Estado], dict]:
    """Nodo `modelo`: invoca el modelo ya ligado a las herramientas y acumula pasos y tokens."""

    def modelo(estado: Estado) -> dict:
        mensaje = modelo_ligado.invoke(estado["messages"])
        uso_previo = estado.get("uso") or {}
        metadatos = getattr(mensaje, "usage_metadata", None) or {}
        uso = {
            "tokens_entrada": int(uso_previo.get("tokens_entrada", 0)) + int(metadatos.get("input_tokens", 0) or 0),
            "tokens_salida": int(uso_previo.get("tokens_salida", 0)) + int(metadatos.get("output_tokens", 0) or 0),
        }
        return {"messages": [mensaje], "pasos": int(estado.get("pasos", 0)) + 1, "uso": uso}

    return modelo


def recordatorio(estado: Estado) -> dict:
    """Nodo `recordatorio`: el modelo respondió texto sin tool calls; se le pide usar RespuestaFinal."""
    return {"messages": [HumanMessage(RECORDATORIO)]}


# --------------------------------------------------------------------------- herramientas


def crear_nodo_herramientas(herramientas: list) -> Callable[[Estado], dict]:
    """Nodo `herramientas`: ejecuta las tool calls (salvo RespuestaFinal) y registra cada llamada.

    Nunca lanza: una excepción o argumentos inválidos se devuelven como `{"error", "valores_validos"}`.
    """
    por_nombre = {h.name: h for h in herramientas}
    nombres_validos = sorted(por_nombre)

    def ejecutar(nombre: str, args: dict) -> Any:
        herramienta = por_nombre.get(nombre)
        if herramienta is None:
            return _error(f"Herramienta desconocida: {nombre!r}.", nombres_validos)
        try:
            return herramienta.invoke(args)
        except ValidationError as exc:
            return _error(f"Argumentos inválidos para {nombre}: {exc.errors(include_url=False)}")
        except Exception as exc:  # noqa: BLE001  (el error vuelve al modelo, nunca rompe el grafo)
            return _error(f"La herramienta {nombre} falló: {type(exc).__name__}: {exc}")

    def nodo_herramientas(estado: Estado) -> dict:
        mensaje = ultimo_ai(estado)
        salida: list[ToolMessage] = []
        llamadas = list(estado.get("llamadas") or [])
        for llamada in mensaje.tool_calls if mensaje is not None else []:
            if llamada["name"] == NOMBRE_FINAL:
                salida.append(ToolMessage(MENSAJE_FINAL_IGNORADA, tool_call_id=llamada["id"], name=NOMBRE_FINAL))
                continue
            args = llamada.get("args") or {}
            resultado = ejecutar(llamada["name"], args)
            llamadas.append({"herramienta": llamada["name"], "args": args, "resultado": resultado})
            salida.append(ToolMessage(a_json(resultado), tool_call_id=llamada["id"], name=llamada["name"]))
        return {"messages": salida, "llamadas": llamadas}

    return nodo_herramientas


# --------------------------------------------------------------------------- respuesta final


def extraer_respuesta(estado: Estado) -> dict:
    """Nodo `extraer_respuesta`: valida con Pydantic los args de la (primera) RespuestaFinal.

    Si no validan, responde la tool call con el error y deja `respuesta=None` (el grafo vuelve a
    `modelo`). Las RespuestaFinal extra del mismo turno se responden como duplicadas.
    """
    finales = _llamadas_finales(ultimo_ai(estado))
    primera, extras = finales[0], finales[1:]
    mensajes = [ToolMessage(MENSAJE_FINAL_DUPLICADA, tool_call_id=c["id"], name=NOMBRE_FINAL) for c in extras]
    try:
        respuesta = RespuestaFinal.model_validate(primera.get("args") or {})
    except ValidationError as exc:
        error = _error(f"Argumentos inválidos para RespuestaFinal: {exc.errors(include_url=False)}. Corregilos.")
        mensajes.insert(0, ToolMessage(a_json(error), tool_call_id=primera["id"], name=NOMBRE_FINAL))
        return {"messages": mensajes, "respuesta": None, "verificacion": None}
    return {"messages": mensajes, "respuesta": respuesta, "verificacion": None}


def _verificador_por_defecto() -> Verificador:
    # Import en tiempo de llamada: el módulo lo implementa otro WP y se puede reemplazar en tests.
    from agente import verificador

    return verificador.verificar_respuesta


def _correr_verificador(
    verificador: Verificador | None, respuesta: RespuestaFinal, estado: Estado, ids_validos: set[str] | None
) -> dict:
    """Corre el verificador y normaliza su salida.

    Con `ids_validos=None` no se validan IDs: se le pasa un set vacío y se descarta `ids_invalidos`
    (recalculando `ok`).
    """
    funcion = verificador or _verificador_por_defecto()
    crudo = funcion(respuesta, resultados_de(estado), estado.get("pregunta", ""), set(ids_validos or ()))
    resultado = {
        "no_verificados": list(crudo.get("no_verificados") or []),
        "ids_invalidos": list(crudo.get("ids_invalidos") or []) if ids_validos is not None else [],
        "ids_repetidos": list(crudo.get("ids_repetidos") or []),
    }
    ok = bool(crudo.get("ok", True)) if ids_validos is not None else True
    resultado["ok"] = ok and not any(resultado[k] for k in ("no_verificados", "ids_invalidos", "ids_repetidos"))
    return resultado


def crear_nodo_verificar(
    verificador: Verificador | None = None, ids_validos: set[str] | None = None
) -> Callable[[Estado], dict]:
    """Nodo `verificar`: chequea la respuesta contra todos los resultados de herramientas."""

    def verificar(estado: Estado) -> dict:
        return {"verificacion": _correr_verificador(verificador, estado["respuesta"], estado, ids_validos)}

    return verificar


def texto_feedback(verificacion: dict) -> str:
    """Mensaje que se le devuelve al modelo cuando la verificación falla."""
    partes = []
    if verificacion.get("no_verificados"):
        cifras = ", ".join(str(c) for c in verificacion["no_verificados"])
        partes.append(f"Las cifras {cifras} no aparecen en ningún resultado de herramienta")
    if verificacion.get("ids_invalidos"):
        partes.append("IDs inválidos: " + ", ".join(map(str, verificacion["ids_invalidos"])))
    if verificacion.get("ids_repetidos"):
        partes.append("IDs repetidos: " + ", ".join(map(str, verificacion["ids_repetidos"])))
    return "; ".join(partes) + "; volvé a consultar y corregí."


def feedback(estado: Estado) -> dict:
    """Nodo `feedback`: responde la RespuestaFinal rechazada con el detalle y suma un intento."""
    primera = _llamadas_finales(ultimo_ai(estado))[0]
    mensaje = ToolMessage(texto_feedback(estado["verificacion"]), tool_call_id=primera["id"], name=NOMBRE_FINAL)
    return {"messages": [mensaje], "intentos_verificacion": int(estado.get("intentos_verificacion", 0)) + 1}


# --------------------------------------------------------------------------- cierre


def _cerrar_por_pasos(estado: Estado) -> dict:
    respuesta = RespuestaFinal(tipo_respuesta="dato", texto=TEXTO_SIN_RESPUESTA, valor=None, tipo_valor="ninguno")
    advertencia = f"Se alcanzó el máximo de {MAX_PASOS} pasos sin una respuesta final válida."
    return {"respuesta": respuesta, "advertencias": list(estado.get("advertencias") or []) + [advertencia]}


def _cifras_visibles(no_verificados: list) -> list[str]:
    """Cifras a mostrar en el texto: las de `texto` (sin prefijo de origen); si no hay, las demás sin
    el prefijo `valor: `. Sin duplicados (por valor numérico cuando se puede interpretar)."""
    sin_prefijo = [str(c) for c in no_verificados if ": " not in str(c)]
    candidatas = sin_prefijo or [str(c).split(": ", 1)[-1] for c in no_verificados]
    vistas: set = set()
    salida: list[str] = []
    for c in candidatas:
        try:
            clave: object = float(c.replace(",", ".")) if c.count(",") + c.count(".") <= 1 else c
        except ValueError:
            clave = c
        if clave not in vistas:
            vistas.add(clave)
            salida.append(c)
    return salida


def _cerrar_dato(estado: Estado) -> dict:
    respuesta: RespuestaFinal = estado["respuesta"]
    verificacion = estado["verificacion"]
    lineas = [f"⚠️ No pude verificar la cifra {c} contra los datos" for c in _cifras_visibles(verificacion["no_verificados"])]
    lineas += [f"⚠️ No pude verificar el ID {i} contra los datos" for i in verificacion["ids_invalidos"]]
    lineas += [f"⚠️ El ID {i} aparece repetido" for i in verificacion["ids_repetidos"]]
    texto = "\n".join([t for t in [respuesta.texto] if t] + lineas)
    advertencias = list(estado.get("advertencias") or []) + [
        "Respuesta entregada sin verificar por completo: " + texto_feedback(verificacion)
    ]
    return {"respuesta": respuesta.model_copy(update={"texto": texto}), "advertencias": advertencias}


def _cerrar_recomendaciones(
    estado: Estado,
    verificador: Verificador | None,
    ids_validos: set[str] | None,
    completar: CompletarRecomendaciones | None,
) -> dict:
    """Descarta los ítems que no verifican (cifras, id inválido o repetido); no toca el texto.

    Cada ítem se verifica solo (una `RespuestaFinal` con ese único ítem) para saber a cuál pertenecen
    las cifras no verificadas. Los IDs repetidos se resuelven quedándose con la primera aparición.
    """
    respuesta: RespuestaFinal = estado["respuesta"]
    invalidos_global = set(estado["verificacion"].get("ids_invalidos") or [])
    conservados: list[ItemRecomendacion] = []
    descartados: list[ItemRecomendacion] = []
    advertencias = list(estado.get("advertencias") or [])
    vistos: set[str] = set()
    for item in respuesta.recomendaciones or []:
        if item.sucursal_id in vistos:
            descartados.append(item)
            advertencias.append(f"Descarté la recomendación repetida de {item.sucursal_id}.")
            continue
        vistos.add(item.sucursal_id)
        solo = RespuestaFinal(tipo_respuesta="recomendaciones", recomendaciones=[item])
        res = _correr_verificador(verificador, solo, estado, ids_validos)
        if item.sucursal_id in invalidos_global or res["ids_invalidos"]:
            descartados.append(item)
            advertencias.append(f"Descarté la recomendación de {item.sucursal_id}: ID inválido.")
        elif res["no_verificados"]:
            descartados.append(item)
            cifras = ", ".join(map(str, res["no_verificados"]))
            advertencias.append(
                f"Descarté la recomendación de {item.sucursal_id}: no pude verificar las cifras {cifras}."
            )
        else:
            conservados.append(item)
    nueva = respuesta.model_copy(update={"recomendaciones": conservados})
    # HOOK WP-R: completar con los siguientes candidatos de `candidatos_visita` hasta llegar a N.
    if completar is not None and descartados:
        nueva = completar(nueva, descartados, estado)
    return {"respuesta": nueva, "advertencias": advertencias}


def crear_nodo_cerrar(
    verificador: Verificador | None = None,
    ids_validos: set[str] | None = None,
    completar_recomendaciones: CompletarRecomendaciones | None = None,
) -> Callable[[Estado], dict]:
    """Nodo `cerrar_con_advertencia`.

    - Hay respuesta y su verificación falló: se entrega con advertencias (texto en `dato`, ítems
      descartados en `recomendaciones`).
    - Si no (corte por pasos): respuesta fija "No pude completar la consulta..." y advertencia.
    """

    def cerrar_con_advertencia(estado: Estado) -> dict:
        respuesta = estado.get("respuesta")
        verificacion = estado.get("verificacion")
        if respuesta is None or verificacion is None or verificacion.get("ok"):
            return _cerrar_por_pasos(estado)
        if respuesta.tipo_respuesta == "recomendaciones":
            return _cerrar_recomendaciones(estado, verificador, ids_validos, completar_recomendaciones)
        return _cerrar_dato(estado)

    return cerrar_con_advertencia


# --------------------------------------------------------------------------- rutas


def ruta_modelo(estado: Estado) -> str:
    """Después de `modelo`: extraer_respuesta | herramientas | recordatorio | cerrar_con_advertencia."""
    mensaje = ultimo_ai(estado)
    llamadas = mensaje.tool_calls if mensaje is not None else []
    if llamadas and all(c["name"] == NOMBRE_FINAL for c in llamadas):
        return "extraer_respuesta"  # se procesa aunque sea el último paso
    if estado.get("pasos", 0) >= MAX_PASOS:
        return "cerrar_con_advertencia"
    return "herramientas" if llamadas else "recordatorio"


def ruta_extraer(estado: Estado) -> str:
    if estado.get("respuesta") is not None:
        return "verificar"
    return "cerrar_con_advertencia" if estado.get("pasos", 0) >= MAX_PASOS else "modelo"


def ruta_verificar(estado: Estado) -> str:
    if estado["verificacion"]["ok"]:
        return "fin"
    if estado.get("intentos_verificacion", 0) >= 1 or estado.get("pasos", 0) >= MAX_PASOS:
        return "cerrar_con_advertencia"
    return "feedback"

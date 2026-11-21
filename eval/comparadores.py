"""Extracción y comparación de respuestas según el tipo (Etapa 5).

Flujo por corrida:

1. Se intenta usar `respuesta.valor` (el campo estructurado que el agente llena para el eval),
   normalizado según el `tipo` de la pregunta.
2. Si el valor falta o no sirve para ese tipo, se usa un parser de respaldo sobre el texto.
3. Se compara con la respuesta esperada con la regla del tipo.

Reglas documentadas:

- **Números.** Un número es una secuencia de dígitos con separadores `.` o `,`, opcionalmente con
  signo `-` o `−` (signo menos Unicode). Cada separador seguido de *exactamente* 3 dígitos (y sin
  más dígitos pegados) es separador de miles; si no, es separador decimal. Así "2,15" y "2.15"
  son 2.15, y "1.000", "1,000", "6.646" y "3.246.856" son enteros. Consecuencia conocida: "2,150"
  se lee como 2150 (ambigüedad inherente; el agente debería llenar `valor` con punto decimal).
  Las fechas ISO (`2026-02-09`) y los IDs (`S012`) no se leen como números.
  Si el texto tiene varios números se toma **el primero que aparece** (no el más cercano a la
  esperada, que sería hacer trampa). Por eso el `valor` estructurado tiene prioridad: si el agente
  lo llena, el texto no se mira.
- **Listas.** IDs `S\\d{3}` en orden de aparición, sin duplicados. Igualdad exacta y en el mismo
  orden tras normalizar mayúsculas/espacios.
- **Texto.** Se normaliza (minúsculas, sin tildes, sin puntuación, `_` como espacio, espacios
  colapsados; singular/plural equivalentes al comparar claves). Reglas, en orden:

  1. **Esperada corta (≤ 8 palabras) con `valor` informativo** (no vacío y ≤ 8 palabras): se decide
     SOLO por el valor; el texto no puede "rescatar" un valor equivocado.
     - Si la esperada es de un dominio conocido (categoría de queja, canal, día de la semana o ID
       `S###`), el primer elemento de ese dominio que aparece en el valor tiene que ser la esperada
       ("pedido_incorrecto" = "pedido incorrecto", "sabado" = "sábado", "S052" = "s052").
     - Si la esperada es compuesta con número (P05: "36 sucursales, el 2026-02-09"), el primer número
       del valor tiene que ser el de la esperada; si coincide, alcanza con que las fechas ISO e IDs
       de la esperada estén en texto + valor (si no tiene fechas ni IDs, se exigen todos los tokens);
       si faltan, juez.
     - Si no, igualdad de claves normalizadas.
  2. **Sin valor informativo** (un valor de más de 8 palabras se trata como texto):
     - Esperada de dominio conocido: el PRIMER candidato de ese dominio que aparece en el texto tiene
       que ser la esperada. "pedido_incorrecto, seguida de demora" no aprueba "demora", y repetir la
       pregunta ("¿teléfono, redes, tienda o app?") tampoco aprueba "app". Si el texto no nombra
       ningún candidato del dominio, decide el juez.
     - Si no: pasa si la esperada es subcadena (por palabras completas) del texto; o, si tiene ≤ 8
       palabras, si todos sus tokens clave (números, fechas ISO, IDs y palabras de ≥ 4 letras que no
       sean stopwords) aparecen en el texto. Si la esperada tiene un número, además el PRIMER número
       del texto (fuera de fechas e IDs) tiene que ser ese ("30 sucursales (de 36 candidatas)" falla).
  Las fechas se buscan en formato ISO: el prompt del agente le pide escribir fechas como `AAAA-MM-DD`;
  "9 de febrero" no alcanza para el token `2026-02-09` y la corrida pasa al juez. Si nada de eso
  decide se usa el LLM juez (si está habilitado; siempre para esperadas largas como P10); sin juez la
  corrida falla con motivo "requiere juez".
- **"No hay datos".** Si la esperada es "no hay datos", la corrida pasa solo si el texto dice que no
  hay datos (frases de negación) y `valor` está vacío: cualquier número o lista se considera inventado.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from agente.esquemas import Respuesta

TIPOS_VALIDOS = ("numero", "texto", "lista")
EPSILON = 1e-9
MAX_PALABRAS_TOKENS = 8

_RE_FECHA_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_RE_ID = re.compile(r"\b[sS]\d{3}\b")
# Signo opcional (guion o signo menos Unicode) y dígitos con separadores; no pegado a letras/dígitos.
_RE_NUMERO = re.compile(r"(?<![\w.,])([-−]?)(\d+(?:[.,]\d+)*)")

STOPWORDS = {
    "para", "pero", "como", "cada", "esta", "este", "estos", "estas", "esos", "esas", "ese", "esa",
    "desde", "hasta", "entre", "sobre", "todo", "toda", "todos", "todas", "tiene", "tienen",
    "fueron", "fue", "sido", "son", "sus", "más", "mas", "menos", "muy", "donde", "cuando",
    "cual", "cuales", "porque", "segun", "tambien", "solo", "otro", "otra", "otros", "otras",
    "unos", "unas", "algo", "alguna", "alguno", "hubo", "habia", "eran", "esto", "aquel",
    "aquella", "durante", "mismo", "misma", "contra", "tras", "dentro", "ellos", "ellas",
}


@dataclass
class ResultadoComparacion:
    """Resultado de comparar una corrida con la esperada."""

    ok: bool
    obtenida: Any
    metodo: str  # valor | texto | tokens | juez | ninguno
    motivo: Optional[str] = None
    uso_juez: dict = field(default_factory=dict)  # tokens_entrada, tokens_salida, costo_usd


# ---------------------------------------------------------------- normalización y parsers


def quitar_tildes(texto: str) -> str:
    """Quita tildes y diacríticos ("sábado" → "sabado", "ñ" → "n")."""
    descompuesto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def normalizar_texto(texto: Any) -> str:
    """Minúsculas, sin tildes, sin puntuación y con espacios colapsados."""
    if texto is None:
        return ""
    t = quitar_tildes(str(texto).lower())
    t = re.sub(r"[^\w\s]|_", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _convertir_numero(signo: str, cuerpo: str) -> float:
    """Convierte el cuerpo de un número aplicando la regla de miles/decimal."""
    partes = re.split(r"[.,]", cuerpo)
    entero = partes[0]
    decimal = ""
    for parte in partes[1:]:
        if len(parte) == 3 and not decimal:
            entero += parte  # separador de miles
        elif not decimal:
            decimal = parte  # separador decimal
        else:
            break  # un segundo decimal ("1.5.2"): se corta ahí
    valor = float(f"{entero}.{decimal}") if decimal else float(entero)
    return -valor if signo else valor


def extraer_numeros(texto: Any) -> list[float]:
    """Todos los números del texto, en orden de aparición (sin fechas ISO ni IDs)."""
    if texto is None:
        return []
    t = _RE_FECHA_ISO.sub(" ", str(texto))
    t = _RE_ID.sub(" ", t)
    return [_convertir_numero(m.group(1), m.group(2)) for m in _RE_NUMERO.finditer(t)]


def parsear_numero(texto: Any) -> Optional[float]:
    """Primer número que aparece en el texto (regla documentada en el módulo), o None."""
    numeros = extraer_numeros(texto)
    return numeros[0] if numeros else None


def extraer_ids(texto: Any) -> list[str]:
    """IDs de sucursal `S\\d{3}` en orden de aparición, sin duplicados, en mayúsculas."""
    if texto is None:
        return []
    vistos: list[str] = []
    for m in _RE_ID.finditer(str(texto)):
        i = m.group(0).upper()
        if i not in vistos:
            vistos.append(i)
    return vistos


def normalizar_item_lista(item: Any) -> str:
    """Mayúsculas y sin espacios sobrantes."""
    return re.sub(r"\s+", " ", str(item)).strip().upper()


def _es_numero(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _formatear_valor(valor: Any) -> str:
    if valor is None:
        return ""
    if isinstance(valor, list):
        return ", ".join(str(v) for v in valor)
    if _es_numero(valor):
        return str(int(valor)) if float(valor).is_integer() else str(valor)
    return str(valor)


def texto_de(respuesta: Respuesta) -> str:
    """Texto de la respuesta; si no hay (recomendaciones), su salida para el usuario."""
    if respuesta.texto:
        return respuesta.texto
    try:
        return respuesta.salida_para_usuario()
    except Exception:  # pragma: no cover - defensivo
        return ""


# ---------------------------------------------------------------- extracción por tipo


def valor_numerico(valor: Any) -> Optional[float]:
    """Normaliza `respuesta.valor` a número, o None si no sirve."""
    if _es_numero(valor):
        return float(valor)
    if isinstance(valor, str):
        return parsear_numero(valor)
    return None


def valor_lista(valor: Any) -> Optional[list[str]]:
    """Normaliza `respuesta.valor` a lista de IDs, o None si no sirve."""
    if isinstance(valor, list) and valor:
        return [normalizar_item_lista(v) for v in valor]
    if isinstance(valor, str):
        ids = extraer_ids(valor)
        return ids or None
    return None


# ---------------------------------------------------------------- comparadores


def comparar_numero(obtenida: Optional[float], esperada: Any, tolerancia: Any) -> bool:
    """`abs(r - e) <= tol + 1e-9` (tolerancia None = 0)."""
    if obtenida is None:
        return False
    tol = float(tolerancia) if tolerancia is not None else 0.0
    return abs(float(obtenida) - float(esperada)) <= tol + EPSILON


def comparar_lista(obtenida: Optional[list], esperada: list) -> bool:
    """Igualdad exacta y en el mismo orden tras normalizar mayúsculas/espacios."""
    if obtenida is None:
        return False
    return [normalizar_item_lista(x) for x in obtenida] == [normalizar_item_lista(x) for x in esperada]


def _contiene_frase(frase: str, texto: str) -> bool:
    """¿`frase` (normalizada) aparece en `texto` (normalizado) como palabras completas?"""
    return bool(frase) and f" {frase} " in f" {texto} "


def tokens_clave(esperada: str) -> dict[str, list]:
    """Tokens clave de una esperada corta: fechas ISO, IDs, números y palabras significativas."""
    crudo = str(esperada)
    fechas = _RE_FECHA_ISO.findall(crudo)
    ids = extraer_ids(crudo)
    numeros = extraer_numeros(crudo)
    sin_fechas_ids = _RE_ID.sub(" ", _RE_FECHA_ISO.sub(" ", crudo))
    palabras = [
        p for p in normalizar_texto(sin_fechas_ids).split()
        if p.isalpha() and len(p) >= 4 and p not in STOPWORDS
    ]
    return {"fechas": fechas, "ids": ids, "numeros": numeros, "palabras": palabras}


def coinciden_tokens(esperada: str, combinado: str) -> bool:
    """¿Aparecen TODOS los tokens clave de la esperada en `combinado` (texto + valor)?"""
    tk = tokens_clave(esperada)
    if not any(tk.values()):
        return False
    fechas_obt = set(_RE_FECHA_ISO.findall(combinado))
    ids_obt = set(extraer_ids(combinado))
    numeros_obt = extraer_numeros(combinado)
    palabras_obt = set(normalizar_texto(combinado).split())
    return (
        all(f in fechas_obt for f in tk["fechas"])
        and all(i in ids_obt for i in tk["ids"])
        and all(any(abs(n - m) <= EPSILON for m in numeros_obt) for n in tk["numeros"])
        and all(p in palabras_obt for p in tk["palabras"])
    )



# ---------------------------------------------------------------- dominios conocidos (texto corto)


def _singular(token: str) -> str:
    """Singular aproximado para comparar claves ("demoras" → "demora", "sucursales" → "sucursal")."""
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def clave(texto: Any) -> str:
    """Texto normalizado con cada palabra en singular (para igualdad tolerante)."""
    return " ".join(_singular(t) for t in normalizar_texto(texto).split())


# Dominio → {valor canónico: alias}. Los alias se comparan con `clave()` (por eso van en singular).
DOMINIOS: dict[str, dict[str, list[str]]] = {
    "categoria": {
        "demora": ["demora"],
        "pedido_incorrecto": ["pedido incorrecto"],
        "trato": ["trato"],
        "limpieza": ["limpieza"],
        "calidad_producto": ["calidad producto", "calidad del producto", "calidad de producto",
                             "calidad de los productos", "calidad de lo producto"],
    },
    "canal": {
        "app": ["app", "aplicacion"],
        "telefono": ["telefono"],
        "redes": ["redes", "redes sociales"],
        "en_tienda": ["en tienda", "tienda", "presencial"],
    },
    "dia": {d: [d] for d in ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo")},
}


def dominio_de(esperada: Any) -> Optional[tuple[str, str]]:
    """(dominio, valor canónico) si la esperada es una categoría, canal, día o un ID; si no, None."""
    crudo = str(esperada).strip()
    if re.fullmatch(r"[sS]\d{3}", crudo):
        return "id", crudo.upper()
    k = clave(crudo)
    for dominio, valores in DOMINIOS.items():
        for canonico, alias in valores.items():
            if k == clave(canonico) or k in {clave(a) for a in alias}:
                return dominio, canonico
    return None


def primero_del_dominio(texto: Any, dominio: str) -> Optional[str]:
    """Primer elemento del `dominio` que aparece en el texto (valor canónico), o None."""
    if dominio == "id":
        ids = extraer_ids(texto)
        return ids[0] if ids else None
    t = f" {clave(texto)} "
    mejor: Optional[tuple[int, int, str]] = None  # (posición, -largo, canónico)
    for canonico, alias in DOMINIOS[dominio].items():
        for a in {clave(canonico), *(clave(x) for x in alias)}:
            pos = t.find(f" {a} ")
            if pos >= 0 and (mejor is None or (pos, -len(a)) < mejor[:2]):
                mejor = (pos, -len(a), canonico)
    return mejor[2] if mejor else None


def _primer_numero_ok(esperada: str, texto: Any) -> bool:
    """Si la esperada tiene un número, ¿el primer número del texto es ese?"""
    numeros_esp = extraer_numeros(esperada)
    if not numeros_esp:
        return True
    obtenido = parsear_numero(texto)
    return obtenido is not None and abs(obtenido - numeros_esp[0]) <= EPSILON


# ---------------------------------------------------------------- preguntas sin datos

ESPERADAS_SIN_DATOS = {"no hay datos", "sin datos", "no hay informacion"}
_RE_NEGACION_DATOS = re.compile(
    r"\b(no hay (datos|informacion|registros?)|no tengo (datos|informacion)|no (existe|figura|aparece)n?"
    r"|no (se )?encontr|no cuento con|sin datos|no dispongo|fuera del periodo|no (esta|estan) disponibles?"
    r"|no (incluyen|contienen|registran)|no se registr)"
)


def es_esperada_sin_datos(esperada: Any) -> bool:
    return isinstance(esperada, str) and normalizar_texto(esperada) in ESPERADAS_SIN_DATOS


def comparar_sin_datos(respuesta: Respuesta) -> ResultadoComparacion:
    """Esperada "no hay datos": pasa solo si el agente dice que no hay datos y NO da un valor.

    Un número o una lista en `valor` cuenta como cifra inventada (falla aunque el texto mencione
    "datos"). Se evalúa sin juez y sin tokens, para que "según los datos, S121 tuvo 40" no apruebe.
    """
    texto = texto_de(respuesta)
    valor = respuesta.valor
    if valor not in (None, "", []) and normalizar_texto(_formatear_valor(valor)) not in ESPERADAS_SIN_DATOS:
        return ResultadoComparacion(False, _formatear_valor(valor), "sin_datos", "dio un valor en vez de decir que no hay datos")
    if _RE_NEGACION_DATOS.search(normalizar_texto(texto)):
        return ResultadoComparacion(True, texto, "sin_datos")
    return ResultadoComparacion(False, texto, "sin_datos", "no dijo que no hay datos")


def _usar_juez(item: dict, esperada: Any, combinado: str, obtenida: Any, usar_juez: bool, juez: Any,
               motivo_sin_juez: str = "requiere juez") -> ResultadoComparacion:
    if not usar_juez:
        return ResultadoComparacion(False, obtenida, "ninguno", motivo_sin_juez)
    from eval.juez import juzgar

    try:
        veredicto, uso = juzgar(item.get("pregunta", ""), esperada, combinado.strip(), juez=juez)
    except Exception as e:  # el juez falla → la corrida falla, el eval sigue
        return ResultadoComparacion(False, obtenida, "juez", f"error del juez: {type(e).__name__}: {e}")
    return ResultadoComparacion(bool(veredicto.correcto), obtenida, "juez", veredicto.razon, uso_juez=uso)


def _comparar_texto(respuesta: Respuesta, item: dict, texto: str, *, usar_juez: bool, juez: Any) -> ResultadoComparacion:
    """Regla de texto documentada en el módulo (valor primero; dominio por primer candidato; juez)."""
    esperada = str(item["respuesta_esperada"])
    esp_norm = normalizar_texto(esperada)
    corta = len(esp_norm.split()) <= MAX_PALABRAS_TOKENS
    dominio = dominio_de(esperada) if corta else None
    valor_str = _formatear_valor(respuesta.valor)
    n_palabras_valor = len(normalizar_texto(valor_str).split())
    valor_informativo = 0 < n_palabras_valor <= MAX_PALABRAS_TOKENS

    # 1) Esperada corta con valor informativo: decide el valor.
    if corta and valor_informativo:
        if dominio:
            obtenido = primero_del_dominio(valor_str, dominio[0])
            ok = obtenido == dominio[1]
            return ResultadoComparacion(ok, valor_str, "valor", None if ok else f"valor distinto ({obtenido or valor_str})")
        if extraer_numeros(esperada) and extraer_numeros(valor_str):
            if not _primer_numero_ok(esperada, valor_str):
                return ResultadoComparacion(False, valor_str, "valor", "número del valor distinto")
            # El número (la respuesta) ya coincide: del texto se exigen las fechas e IDs de la esperada.
            combinado = f"{texto} {valor_str}"
            tk = tokens_clave(esperada)
            if (tk["fechas"] or tk["ids"]) and all(f in _RE_FECHA_ISO.findall(combinado) for f in tk["fechas"]) \
                    and all(i in extraer_ids(combinado) for i in tk["ids"]):
                return ResultadoComparacion(True, valor_str, "valor")
            if _contiene_frase(esp_norm, normalizar_texto(combinado)) or coinciden_tokens(esperada, combinado):
                return ResultadoComparacion(True, valor_str, "valor")
            return _usar_juez(item, esperada, combinado, valor_str, usar_juez, juez)
        if clave(valor_str) == clave(esperada):
            return ResultadoComparacion(True, valor_str, "valor")
        if not extraer_numeros(esperada):
            return ResultadoComparacion(False, valor_str, "valor", "valor distinto")
        # esperada compuesta con número pero valor sin número: se sigue con el texto

    # 2) Sin valor informativo: se mira el texto (un valor largo cuenta como texto).
    texto_eval = texto if valor_informativo else f"{texto} {valor_str}".strip()
    obtenida = texto_eval if not valor_informativo else (valor_str or texto)
    if dominio:
        obtenido = primero_del_dominio(texto_eval, dominio[0])
        if obtenido is not None:
            ok = obtenido == dominio[1]
            return ResultadoComparacion(ok, obtenido, "texto", None if ok else f"el primer candidato del texto es {obtenido}")
        return _usar_juez(item, esperada, texto_eval, obtenida, usar_juez, juez)

    if corta and not _primer_numero_ok(esperada, texto_eval):
        return ResultadoComparacion(False, obtenida, "tokens", "el primer número del texto no es el esperado")
    if _contiene_frase(esp_norm, normalizar_texto(texto_eval)):
        return ResultadoComparacion(True, obtenida, "texto")
    if corta and coinciden_tokens(esperada, texto_eval):
        return ResultadoComparacion(True, obtenida, "tokens")
    return _usar_juez(item, esperada, f"{texto} {valor_str}" if valor_informativo else texto_eval, obtenida, usar_juez, juez)


# ---------------------------------------------------------------- entrada principal


def comparar(
    respuesta: Respuesta,
    item: dict,
    *,
    usar_juez: bool = True,
    juez: Any = None,
) -> ResultadoComparacion:
    """Compara una `Respuesta` con el ítem del JSONL (`respuesta_esperada`, `tipo`, `tolerancia`).

    `juez` permite inyectar un juez falso (ver `eval.juez.juzgar`).
    """
    tipo = item["tipo"]
    esperada = item["respuesta_esperada"]
    texto = texto_de(respuesta)

    if tipo == "numero":
        obtenida = valor_numerico(respuesta.valor)
        metodo = "valor"
        if obtenida is None:
            obtenida, metodo = parsear_numero(texto), "texto"
        if obtenida is None:
            return ResultadoComparacion(False, None, "ninguno", "no se encontró un número")
        ok = comparar_numero(obtenida, esperada, item.get("tolerancia"))
        return ResultadoComparacion(ok, obtenida, metodo, None if ok else "número distinto")

    if tipo == "lista":
        obtenida = valor_lista(respuesta.valor)
        metodo = "valor"
        if obtenida is None:
            obtenida, metodo = extraer_ids(texto), "texto"
        ok = comparar_lista(obtenida, esperada)
        return ResultadoComparacion(ok, obtenida, metodo, None if ok else "lista distinta")

    if tipo == "texto" and es_esperada_sin_datos(esperada):
        return comparar_sin_datos(respuesta)

    if tipo == "texto":
        return _comparar_texto(respuesta, item, texto, usar_juez=usar_juez, juez=juez)

    raise ValueError(f"tipo inválido: {tipo!r} (válidos: {', '.join(TIPOS_VALIDOS)})")


ComparadorFn = Callable[..., ResultadoComparacion]

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
- **Texto.** Se normaliza (minúsculas, sin tildes, sin puntuación, espacios colapsados). Pasa si la
  esperada es subcadena (por palabras completas) del texto o del valor. Si no, y la esperada tiene
  ≤ 8 palabras, pasa si todos sus tokens clave (números, fechas ISO, IDs y palabras de ≥ 4 letras
  que no sean stopwords) aparecen en texto + valor. Las fechas se buscan en formato ISO: el prompt
  del agente le pide escribir fechas como `AAAA-MM-DD`; "9 de febrero" no alcanza para el token
  `2026-02-09` y la corrida pasa al juez. Si nada de eso alcanza se usa el LLM juez (si está
  habilitado); sin juez la corrida falla con motivo "requiere juez".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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

    if tipo == "texto":
        valor_str = _formatear_valor(respuesta.valor)
        esp_norm = normalizar_texto(esperada)
        obtenida = valor_str or texto
        if _contiene_frase(esp_norm, normalizar_texto(valor_str)):
            return ResultadoComparacion(True, obtenida, "valor")
        if _contiene_frase(esp_norm, normalizar_texto(texto)):
            return ResultadoComparacion(True, obtenida, "texto")
        combinado = f"{texto} {valor_str}"
        if len(esp_norm.split()) <= MAX_PALABRAS_TOKENS and coinciden_tokens(str(esperada), combinado):
            return ResultadoComparacion(True, obtenida, "tokens")
        if not usar_juez:
            return ResultadoComparacion(False, obtenida, "ninguno", "requiere juez")
        from eval.juez import juzgar

        try:
            veredicto, uso = juzgar(item.get("pregunta", ""), esperada, combinado.strip(), juez=juez)
        except Exception as e:  # el juez falla → la corrida falla, el eval sigue
            return ResultadoComparacion(False, obtenida, "juez", f"error del juez: {type(e).__name__}: {e}")
        return ResultadoComparacion(
            bool(veredicto.correcto), obtenida, "juez", veredicto.razon, uso_juez=uso
        )

    raise ValueError(f"tipo inválido: {tipo!r} (válidos: {', '.join(TIPOS_VALIDOS)})")


ComparadorFn = Callable[..., ResultadoComparacion]

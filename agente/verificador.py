"""Verificador de cifras: toda cifra del texto tiene que salir de una herramienta.

Idea general:

1. `extraer_numeros(texto)` encuentra las cifras de un texto en español y, para cada una, todas
   sus interpretaciones razonables (p. ej. "1.393" puede ser 1393 o 1,393). Antes de buscar
   cifras se "enmascaran" (se reemplazan por espacios) los fragmentos que no son cifras a
   verificar: fechas, horas, IDs (S012, Q3), la unidad de la tasa ("por cada 1.000 pedidos"),
   marcadores de listas numeradas ("1. ") y ordinales ("1°", "2do"). Los años sueltos
   (1900–2100, 4 dígitos sin separador) se descartan después de extraer.
2. `numeros_disponibles(resultados)` junta todos los números de los resultados de herramientas:
   recorre dicts/listas, parsea strings JSON, extrae números de strings con texto
   (advertencias) y agrega el largo de cada lista. Los booleanos no cuentan.
3. Una cifra con k decimales x verifica contra un valor v si |v| y |x| difieren a lo sumo
   0,5·10^−k (+1e-9). Se compara en valor absoluto: "bajó 0,29" con un delta de -0.286 es
   válido. Si la cifra va seguida de "%", "por ciento" o "puntos porcentuales", también se
   compara contra v·100 (fracciones).
4. Las cifras que aparecen literalmente en la pregunta (misma normalización) no se reportan.

Formato de `no_verificados` en `verificar_respuesta`: las cifras de `texto` van tal cual
aparecen ("2,20"); las demás llevan prefijo de origen: "valor: 3.1", "motivo S012: 40%",
"evidencia S012: 3,7".
"""

from __future__ import annotations

import bisect
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal

from agente.esquemas import RespuestaFinal

__all__ = [
    "Cifra",
    "extraer_numeros",
    "numeros_disponibles",
    "verificar_numeros",
    "verificar_respuesta",
    "extraer_ids",
]

# ---------------------------------------------------------------------------------------------
# Patrones
# ---------------------------------------------------------------------------------------------

_MESES = (
    r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|sept?iembre|octubre|noviembre|diciembre)"
)

# Fragmentos que se enmascaran antes de extraer cifras (el orden importa).
_PATRONES_EXCLUIDOS = [
    # Fechas ISO, con hora opcional: 2026-03-29, 2026-03-29T10:00:00, y año-mes 2026-03.
    re.compile(r"\b\d{4}-\d{1,2}(?:-\d{1,2})?(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?\b"),
    # Rangos de días: "del 2 al 8 de marzo", "2 y 9 de marzo", "2-8 de marzo".
    re.compile(
        rf"\b\d{{1,2}}\s*(?:al|y|a|-|–)\s*\d{{1,2}}\s+de\s+{_MESES}(?:\s+(?:de|del)\s+\d{{4}})?",
        re.IGNORECASE,
    ),
    # "2 de marzo", "29 de marzo de 2026".
    re.compile(rf"\b\d{{1,2}}\s+de\s+{_MESES}(?:\s+(?:de|del)\s+\d{{4}})?", re.IGNORECASE),
    # 29/03/2026, 29/03, 29-03-2026.
    re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"),
    re.compile(r"\b\d{1,2}-\d{1,2}-\d{4}\b"),
    # Horas: 10:30, 10:30:00, "10 hs", "10h", "9 am", "a las 10 horas".
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    re.compile(r"\b\d{1,2}\s*(?:hs|h|am|pm|a\.\s?m\.|p\.\s?m\.)(?!\w)", re.IGNORECASE),
    re.compile(r"\ba\s+las?\s+\d{1,2}\s*(?:horas|en\s+punto)\b", re.IGNORECASE),
    # IDs: S012, Q3 (sucursales, preguntas, trimestres).
    re.compile(r"\b[SsQq]\d+\b"),
    # Unidad de la tasa: "por cada 1.000 pedidos", "cada 1,000", "/1000", "c/1.000", "x 1.000".
    re.compile(r"(?:cada|c/|/|\bx|\bpor)\s*1[.,]?000(?![\d.,]*\d)", re.IGNORECASE),
    # Marcadores de lista numerada al comienzo de línea: "1. ", "2) ".
    re.compile(r"(?m)^[ \t]*(?:[-*•][ \t]*)?\d{1,2}[.)](?=\s)"),
    # Ordinales: 1°, 1º, 2ª, 1ro, 2do, 3er, 3ra, 4to, 10mo...
    re.compile(
        r"\b\d{1,3}(?:°|º|ª|ros?|ras?|er|dos?|das?|tos?|tas?|mos?|mas?|vos?|vas?|nos?|nas?)(?!\w)",
        re.IGNORECASE,
    ),
]

# Cifra: signo opcional, número con separadores, sufijo de porcentaje opcional.
_RE_NUMERO = re.compile(
    r"(?<![\w.,])"
    r"(?P<signo>[+\-−–±]?)"
    r"(?P<num>\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
    r"(?!\d)"
    r"(?P<pct>\s*(?:%|por\s*ciento|puntos?\s+porcentuales|p\.\s?p\.|pp\b))?",
    re.IGNORECASE,
)

_RE_ID_SUCURSAL = re.compile(r"\b[Ss]\d{3}\b")


# ---------------------------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cifra:
    """Una cifra encontrada en un texto.

    - `texto`: la cifra tal como aparece (con signo y "%" si los tiene), p. ej. "-0,29", "82%".
    - `valores`: interpretaciones posibles como pares (valor, decimales), p. ej. "1.393" →
      ((1393.0, 0), (1.393, 3)). La cifra verifica si cualquiera coincide.
    - `es_porcentaje`: va seguida de "%", "por ciento" o "puntos porcentuales".
    - `inicio`: posición en el texto.
    """

    texto: str
    valores: tuple[tuple[float, int], ...]
    es_porcentaje: bool
    inicio: int


def _enmascarar(texto: str) -> str:
    """Reemplaza por espacios los fragmentos que no son cifras a verificar (mismo largo)."""
    for patron in _PATRONES_EXCLUIDOS:
        texto = patron.sub(lambda m: " " * len(m.group(0)), texto)
    return texto


def _interpretar(num: str) -> list[tuple[float, int]]:
    """Interpretaciones (valor, decimales) de un número escrito con separadores '.' o ','."""
    seps = [c for c in num if c in ".,"]
    if not seps:
        return [(float(int(num)), 0)]
    if len(set(seps)) == 2:
        # Ambos separadores: el último es el decimal, el otro es de miles ("1.234,5", "1,234.5").
        pos = max(num.rfind("."), num.rfind(","))
        entero = re.sub(r"[.,]", "", num[:pos])
        dec = num[pos + 1 :]
        return [(float(f"{entero}.{dec}"), len(dec))]
    if len(seps) > 1:
        # El mismo separador repetido: solo puede ser de miles ("3.246.856").
        return [(float(num.replace(seps[0], "")), 0)]
    entero, dec = re.split(r"[.,]", num)
    interpretaciones = [(float(f"{entero}.{dec}"), len(dec))]
    if len(dec) == 3 and 1 <= len(entero) <= 3 and not (entero == "0" or entero.startswith("0")):
        # Ambiguo: "1.393" puede ser 1393 (miles) o 1,393 (decimal).
        interpretaciones.insert(0, (float(entero + dec), 0))
    return interpretaciones


def _es_anio(num: str, signo: str, pct: str | None) -> bool:
    return not signo and not pct and len(num) == 4 and num.isdigit() and 1900 <= int(num) <= 2100


def extraer_numeros(texto: str) -> list[Cifra]:
    """Extrae las cifras de un texto en español, sin fechas, horas, IDs, años ni ordinales.

    Pensada para reusarse (p. ej. desde el eval). Devuelve las cifras en orden de aparición.
    """
    if not texto:
        return []
    enmascarado = _enmascarar(texto)
    cifras: list[Cifra] = []
    for m in _RE_NUMERO.finditer(enmascarado):
        signo, num, pct = m.group("signo"), m.group("num"), m.group("pct")
        if _es_anio(num, signo, pct):
            continue
        negativo = signo in ("-", "−", "–")
        valores = tuple((-v if negativo else v, k) for v, k in _interpretar(num))
        cifras.append(
            Cifra(
                texto=texto[m.start() : m.end()].strip(),
                valores=valores,
                es_porcentaje=bool(pct),
                inicio=m.start(),
            )
        )
    return cifras


def numeros_disponibles(resultados: list[dict | str]) -> list[float]:
    """Todos los números presentes en los resultados de herramientas (con repeticiones).

    Recorre dicts (solo valores) y listas recursivamente, parsea strings JSON, extrae números
    de strings con texto (todas sus interpretaciones) y agrega el largo de cada lista.
    """
    valores: list[float] = []

    def visitar(obj) -> None:
        if obj is None or isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            if math.isfinite(obj):
                valores.append(float(obj))
            return
        if isinstance(obj, str):
            s = obj.strip()
            if s[:1] in ("{", "["):
                try:
                    visitar(json.loads(s))
                    return
                except (ValueError, TypeError):
                    pass
            for cifra in extraer_numeros(s):
                valores.extend(v for v, _ in cifra.valores)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                visitar(v)
            return
        if isinstance(obj, (list, tuple)):
            valores.append(float(len(obj)))
            for v in obj:
                visitar(v)
            return
        # Otros tipos (Decimal, numpy, etc.): se intenta convertir a float.
        try:
            f = float(obj)
        except (TypeError, ValueError):
            return
        if math.isfinite(f):
            valores.append(f)

    for r in resultados or []:
        visitar(r)
    return valores


# ---------------------------------------------------------------------------------------------
# Coincidencia
# ---------------------------------------------------------------------------------------------


class _Pozo:
    """Valores absolutos ordenados de los resultados (y su escala ×100) para buscar rápido."""

    def __init__(self, resultados: list[dict | str]):
        nums = numeros_disponibles(resultados)
        self.abs = sorted({abs(v) for v in nums})
        self.abs100 = sorted({abs(v) * 100 for v in nums})

    @staticmethod
    def _hay(ordenados: list[float], x: float, tol: float) -> bool:
        i = bisect.bisect_left(ordenados, x - tol)
        return i < len(ordenados) and ordenados[i] <= x + tol

    def coincide(self, valores, es_porcentaje: bool) -> bool:
        for x, k in valores:
            tol = 0.5 * 10.0 ** (-k) + 1e-9
            if self._hay(self.abs, abs(x), tol):
                return True
            if es_porcentaje and self._hay(self.abs100, abs(x), tol):
                return True
        return False


def _valores_pregunta(pregunta: str) -> list[float]:
    return [v for c in extraer_numeros(pregunta or "") for v, _ in c.valores]


def _esta_en_pregunta(cifra: Cifra, en_pregunta: list[float]) -> bool:
    return any(abs(x - p) < 1e-9 for x, _ in cifra.valores for p in en_pregunta)


def _no_verificadas(texto: str, pozo: _Pozo, en_pregunta: list[float]) -> list[str]:
    salida: list[str] = []
    for cifra in extraer_numeros(texto or ""):
        if _esta_en_pregunta(cifra, en_pregunta):
            continue
        if not pozo.coincide(cifra.valores, cifra.es_porcentaje) and cifra.texto not in salida:
            salida.append(cifra.texto)
    return salida


def verificar_numeros(texto: str, resultados: list[dict | str], pregunta: str) -> list[str]:
    """Devuelve las cifras de `texto` que NO aparecen en ningún resultado (vacía = ok).

    Las cifras se devuelven tal como aparecen en el texto, sin duplicados y en orden.
    """
    return _no_verificadas(texto, _Pozo(resultados), _valores_pregunta(pregunta))


# ---------------------------------------------------------------------------------------------
# Respuesta completa
# ---------------------------------------------------------------------------------------------


def extraer_ids(texto: str) -> list[str]:
    """IDs de sucursal (S + 3 dígitos) en un texto, en mayúsculas y en orden de aparición."""
    return [m.group(0).upper() for m in _RE_ID_SUCURSAL.finditer(texto or "")]


def _decimales_de(x: float) -> int:
    if x.is_integer():
        return 0
    exp = Decimal(repr(x)).as_tuple().exponent
    return max(0, -int(exp))


def _agregar(lista: list[str], item: str) -> None:
    if item not in lista:
        lista.append(item)


def verificar_respuesta(
    respuesta: RespuestaFinal, resultados: list[dict | str], pregunta: str, ids_validos: set[str]
) -> dict:
    """{"ok": bool, "no_verificados": [...], "ids_invalidos": [...], "ids_repetidos": [...]}

    - Cifras: de `texto`, de `valor` (si `tipo_valor == "numero"`) y de `motivo`/`evidencia` de
      cada recomendación. Prefijos en `no_verificados`: ninguno para `texto`; "valor: ",
      "motivo <ID>: " y "evidencia <ID>: " para el resto.
    - IDs: los S### de `valor`, `texto` y recomendaciones (incluido `sucursal_id`) que no están
      en `ids_validos` → `ids_invalidos`. Los `sucursal_id` repetidos en `recomendaciones` (o
      IDs repetidos en `valor` lista) → `ids_repetidos`.
    - `ok` = las tres listas vacías.
    """
    pozo = _Pozo(resultados)
    en_pregunta = _valores_pregunta(pregunta)
    validos = {i.strip().upper() for i in (ids_validos or set())}

    no_verificados: list[str] = []
    ids_vistos: list[str] = []
    ids_repetidos: list[str] = []

    # Texto.
    for c in _no_verificadas(respuesta.texto or "", pozo, en_pregunta):
        _agregar(no_verificados, c)
    ids_vistos.extend(extraer_ids(respuesta.texto or ""))

    # Valor.
    valor = respuesta.valor
    if respuesta.tipo_valor == "numero" and valor is not None:
        if isinstance(valor, (int, float)) and not isinstance(valor, bool):
            x = float(valor)
            # Se acepta también la escala ×100 (valor 82 contra una fracción 0.8167).
            if not pozo.coincide([(x, _decimales_de(x))], es_porcentaje=True):
                _agregar(no_verificados, f"valor: {valor}")
        elif isinstance(valor, str):
            cifras = extraer_numeros(valor)
            if not cifras:
                _agregar(no_verificados, f"valor: {valor}")
            for c in cifras:
                if not pozo.coincide(c.valores, es_porcentaje=True):
                    _agregar(no_verificados, f"valor: {c.texto}")
    if isinstance(valor, list):
        ids_valor: list[str] = []
        for item in valor:
            s = str(item).strip().upper()
            ids_valor.extend(extraer_ids(s) or [s])
        ids_vistos.extend(ids_valor)
        for i, n in Counter(ids_valor).items():
            if n > 1:
                _agregar(ids_repetidos, i)
    elif isinstance(valor, str):
        ids_vistos.extend(extraer_ids(valor))

    # Recomendaciones.
    ids_recos: list[str] = []
    for reco in respuesta.recomendaciones or []:
        sid = (reco.sucursal_id or "").strip().upper()
        ids_recos.append(sid)
        ids_vistos.append(sid)
        for campo in ("motivo", "evidencia"):
            contenido = getattr(reco, campo) or ""
            for c in _no_verificadas(contenido, pozo, en_pregunta):
                _agregar(no_verificados, f"{campo} {sid}: {c}")
            ids_vistos.extend(extraer_ids(contenido))
        ids_vistos.extend(extraer_ids(reco.accion_sugerida or ""))
    for i, n in Counter(ids_recos).items():
        if n > 1:
            _agregar(ids_repetidos, i)

    ids_invalidos: list[str] = []
    for i in ids_vistos:
        if i not in validos:
            _agregar(ids_invalidos, i)

    return {
        "ok": not (no_verificados or ids_invalidos or ids_repetidos),
        "no_verificados": no_verificados,
        "ids_invalidos": ids_invalidos,
        "ids_repetidos": ids_repetidos,
    }

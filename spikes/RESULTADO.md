# Resultado del spike de LangGraph (WP-S)

Fecha: 2026-11-21. Versiones: langgraph 1.2.24, langchain 1.4.10, langchain-core 1.7.7, langchain-anthropic 1.7.10.

## Parte offline: aprobada
- `RespuestaFinal` se serializa como tool de Anthropic. `valor: float | str | list[str] | None` queda como
  `anyOf [number, string, array<string>, null]` y `ItemRecomendacion` queda embebido sin problemas.
- `bind_tools([...], tool_choice="any")` se traduce a `{"type": "any"}`, **pero Sonnet 5.5 no lo acepta**
  (ver hallazgo abajo).

## Hallazgo posterior (revisión independiente, 2026-11-21): la config original fallaba con el modelo real
Leyendo `langchain_anthropic/chat_models.py` (1.7.10):
- `_assert_valid_model_configuration`: para modelos `claude-sonnet-5-5*` y `claude-fable-5*` lanza
  `ValueError` si `temperature` != 1 (también con `top_p` != 1 o `top_k`). El agente creaba el modelo con
  `temperature=0`, así que **toda** llamada fallaba antes de salir a la red. La parte offline del spike
  no lo detectó porque solo miraba `bind_tools`, no armaba el payload.
- `_supports_forced_tool_choice`: `claude-fable-5-1`, `claude-opus-5-5` y `claude-sonnet-5-5` no aceptan
  `tool_choice` de tipo `any` ni `tool` (la API devuelve 400).

Arreglo: `agente/config.py` tiene `kwargs_modelo(nombre)` (omite `temperature` para esos modelos; 0 para
el resto, p. ej. Haiku 5.5) y `soporta_tool_choice_forzado(nombre)`. `construir_grafo` usa `"any"` si el
modelo lo soporta y `"auto"` si no; con `"auto"` el modelo puede contestar texto libre y el nodo
`recordatorio` le pide `RespuestaFinal` (por eso `MAX_PASOS` pasó de 8 a 10 y `RECURSION_LIMIT` de 25 a 30).
`tests/test_payload_anthropic.py` arma el payload real (offline, clave falsa) para Sonnet 5.5 y Haiku 5.5:
es el test que hubiera detectado el problema. El spike offline ahora también arma el payload.

Consecuencia: sin temperatura 0 el modelo es menos determinista. Las cifras siguen saliendo de las
herramientas (y el verificador las controla) y el orden de las recomendaciones lo decide el score
determinístico de `candidatos_visita`, así que la variabilidad queda en la redacción y en la elección de
herramientas.

## Parte real (con Claude Sonnet 5.5): pendiente
No se pudo correr en la sesión de desarrollo porque no había `ANTHROPIC_API_KEY`. Para correrla:
`python spikes/spike_grafo.py --real`.

## Decisión (compuerta G0)
Se congela C3 **sin plan B** (se mantiene el `Union` de `valor`). Si la corrida real mostrara que el modelo
manda `valor` con el tipo equivocado (por ejemplo, "30" como string), el plan B es
`valor_numero / valor_texto / valor_lista`. Mitigación ya prevista: el eval normaliza `valor` según `tipo`
(un string numérico se convierte a número), así que un `valor` mal tipado no rompe la comparación.

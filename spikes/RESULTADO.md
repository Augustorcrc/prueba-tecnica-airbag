# Resultado del spike de LangGraph (WP-S)

Fecha: 2026-11-21. Versiones: langgraph 1.2.24, langchain 1.4.10, langchain-core 1.7.7, langchain-anthropic 1.7.10.

## Parte offline: aprobada
- `RespuestaFinal` se serializa como tool de Anthropic. `valor: float | str | list[str] | None` queda como
  `anyOf [number, string, array<string>, null]` y `ItemRecomendacion` queda embebido sin problemas.
- `bind_tools([...], tool_choice="any")` se traduce a `{"type": "any"}`.

## Parte real (con Claude Sonnet 5.5): pendiente
No se pudo correr en la sesión de desarrollo porque no había `ANTHROPIC_API_KEY`. Para correrla:
`python spikes/spike_grafo.py --real`.

## Decisión (compuerta G0)
Se congela C3 **sin plan B** (se mantiene el `Union` de `valor`). Si la corrida real mostrara que el modelo
manda `valor` con el tipo equivocado (por ejemplo, "30" como string), el plan B es
`valor_numero / valor_texto / valor_lista`. Mitigación ya prevista: el eval normaliza `valor` según `tipo`
(un string numérico se convierte a número), así que un `valor` mal tipado no rompe la comparación.

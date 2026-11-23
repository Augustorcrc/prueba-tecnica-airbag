# Ejemplo de reporte de evaluación

> Ejemplo estático, versionado en el repo. Es la salida de `make eval REPS=1` del 2026-11-23 (preguntas de ejemplo + propias, 1 repetición, con juez) con el código del commit `9478a4d`. Los reportes que genera cada corrida (`reportes/reporte.md`, `reportes/reporte.json` y sus copias con fecha) no se versionan.

Fecha: 2026-11-23T13:09:15 · repeticiones: 1 · juez: sí · paralelo: 1

| id | tipo | esperada | obtenida | ok | aciertos | latencia_s | tokens_corrida | costo_corrida_usd |
|---|---|---|---|---|---|---|---|---|
| P01 | numero | 30 | 30.0 | sí | 1/1 | 3.94 | 25663 | 0.0531 |
| P02 | numero | 442 | 442.0 | sí | 1/1 | 2.68 | 25561 | 0.0531 |
| P03 | numero | 2.15 | 2.1468 | sí | 1/1 | 4.17 | 25708 | 0.0539 |
| P04 | texto | demora | demora | sí | 1/1 | 3.65 | 25855 | 0.0542 |
| P05 | texto | 36 sucursales, el 2026-02-09 | 36 | sí | 1/1 | 3.51 | 25804 | 0.0538 |
| P06 | lista | ["S012", "S107", "S067"] | ["S012", "S107", "S067"] | sí | 1/1 | 4.30 | 26066 | 0.0554 |
| P07 | texto | S052 | S052 | sí | 1/1 | 4.01 | 25937 | 0.0547 |
| P08 | texto | app | app | sí | 1/1 | 4.01 | 25876 | 0.0543 |
| P09 | numero | 6 | 6.0 | sí | 1/1 | 3.04 | 26030 | 0.0548 |
| P10 | texto | Los clientes de la sucursal S104 se que… | limpieza | sí | 1/1 | 3.16 | 25975 | 0.0547 |
| A01 | numero | 691516 | 691516.0 | sí | 1/1 | 2.77 | 25721 | 0.0541 |
| A02 | texto | S052 | S052 | sí | 1/1 | 4.25 | 26007 | 0.0551 |
| A03 | texto | sábado | sábado | sí | 1/1 | 4.05 | 26144 | 0.0554 |
| A04 | lista | ["S034", "S107", "S012"] | ["S034", "S107", "S012"] | sí | 1/1 | 9.17 | 44007 | 0.0972 |
| A05 | numero | -0.29 | -0.2862 | sí | 1/1 | 4.20 | 26386 | 0.0564 |
| A06 | numero | 6646 | 6646.0 | sí | 1/1 | 4.10 | 25642 | 0.0534 |

## Resumen

- Aciertos por mayoría: 16/16 (100%)
- Aciertos estrictos (N de N): 16/16 (100%)
- Consistencia: 16/16 (100%)
- Corridas con error: 0 de 16
- Latencia media: 4.06 s · p95: 5.52 s
- Agente: 426285 tokens de entrada · 6097 de salida · US$0.9135
- Juez: 1331 tokens de entrada · 329 de salida · US$0.0003
- Costo total (agente + juez): US$0.9138
- Por pregunta respondida una vez (agente): 27024 tokens (26643 entrada · 381 salida) · US$0.0571 · latencia media 4.06 s · p95 5.52 s
- Por tipo: numero: 7/7 mayoría, 7/7 estrictos; texto: 7/7 mayoría, 7/7 estrictos; lista: 2/2 mayoría, 2/2 estrictos
- Por archivo: preguntas_ejemplo.jsonl: 10/10 mayoría, 10/10 estrictos; preguntas_propias.jsonl: 6/6 mayoría, 6/6 estrictos

## Costo y latencia por pregunta (agente)

Promedio sobre 16 corridas respondidas (cada corrida = una pregunta respondida una vez; no incluye el LLM juez).

| métrica | valor |
|---|---|
| tokens de entrada | 26643 |
| tokens de salida | 381 |
| tokens totales | 27024 |
| costo | US$0.0571 |
| latencia media | 4.06 s |
| latencia p95 | 5.52 s |

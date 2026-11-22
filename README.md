# Agente Café Tostadero

Asistente en español para el gerente regional de Café Tostadero: responde preguntas sobre pedidos,
quejas, sucursales y capacitaciones, y recomienda qué sucursales visitar. Prueba técnica de Airbag
(ML Engineer). Lo escribí en Python con LangGraph, DuckDB y pandas.

## Cómo correrlo

```bash
uv sync                                   # crea .venv e instala dependencias (Python >= 3.11)
cp .env.example .env                      # y completo ANTHROPIC_API_KEY
make test                                 # tests offline, no necesitan clave ni red
make test-clave                           # tests contra el modelo real (gastan créditos)
make ask Q="¿Cuántas quejas hubo en total?"
python main.py "¿Qué sucursales debería visitar esta semana?"
make eval                                 # preguntas de ejemplo + propias, 3 repeticiones
```

La clave se lee solo de `ANTHROPIC_API_KEY` (variable de entorno o `.env`, que está en `.gitignore`).
Si falta, el CLI dice qué variable exportar. Variables opcionales en `.env.example`: `MODELO_LLM`,
`MODELO_JUEZ`, `DATOS_DIR`.

El eval también se corre directo:

```bash
python -m eval.evaluar eval/preguntas_ejemplo.jsonl eval/preguntas_propias.jsonl \
  --repeticiones 3 --salida reportes/ [--sin-juez] [--paralelo N] [--umbral 0.8]
```

Imprime una tabla por pregunta y deja `reportes/reporte.{json,md}` (más una copia con fecha y hora).
Una pregunta cuenta como acertada si acierta en la mayoría de las repeticiones. `--umbral` hace que el
proceso salga con error si la tasa de aciertos queda por debajo (pensado para CI).

### Con Docker (alternativa)

La imagen instala exactamente las versiones de `uv.lock` (`uv sync --frozen`), corre como usuario sin
privilegios y recibe la clave por variable de entorno (nunca queda dentro de la imagen).

```bash
make docker-test                          # build de la etapa de tests: corre la suite offline
make docker-build                         # imagen final
make docker-ask Q="¿Qué sucursales debería visitar esta semana?"   # usa --env-file .env
make docker-eval REPS=1                   # deja el reporte en ./reportes
```

En `.env`, la clave va sin comillas (`ANTHROPIC_API_KEY=sk-ant-...`): `docker --env-file` no las saca.

## Proveedor y modelo

- **Agente:** Anthropic, Claude Sonnet 5.5 (`anthropic:claude-sonnet-5-5`), con la temperatura por defecto:
  Sonnet 5.5 no acepta `temperature=0` ni `tool_choice` forzado (langchain-anthropic lo valida del lado del
  cliente). Por eso es menos determinista que un modelo a temperatura 0. Las cifras igual salen de las
  herramientas y pasan por el verificador, y el orden de las recomendaciones lo decide un score
  determinístico. La variabilidad queda en la redacción y en qué herramientas elige.
- **Juez del eval:** Claude Haiku 5.5 (`anthropic:claude-haiku-5-5`), temperatura 0.

El modelo se crea con `init_chat_model`, así que el código es agnóstico: se cambia con
`MODELO_LLM=proveedor:modelo` (la variable de la clave se deduce del proveedor). Solo lo probé con
Anthropic.

## Qué encontré en los datos y cómo lo traté

| Problema | Tratamiento | Por qué importa |
|---|---|---|
| 199 quejas duplicadas por `queja_id` | Deduplico (6.845 → 6.646) | Sin esto P02, P03, P06 y P10 no cierran |
| 8 filas de pedidos inválidos (5 con 99999, 3 negativos) | `pedidos_validos` = NULL; el día queda fuera de sumas y tasas | Con los crudos el total sube 15%, cambia el día de la semana con más pedidos y la sucursal top de la semana del 5 de enero |
| 7 quejas caen en esos días inválidos | Cuentan en los conteos, pero no en el numerador de las tasas | Numerador y denominador sobre los mismos días. Mantenerlas cambia la tasa 0,02–0,07 en 8 sucursales |
| 6 sucursales nuevas (abiertas el 16 y 17 de marzo, 13–14 días) | Fuera de los rankings por tasa (regla de 28 días abiertas), pero las menciono en el texto | Con pocos días la tasa es puro ruido |
| Capacitación confundida: en el Norte se asignó al azar, en el resto se capacitó a las peores | Solo el diferencias-en-diferencias del Norte es causal: **-0,29 quejas por 1.000 pedidos (~-15% respecto del contrafactual)** con ratio de sumas (-0,34 promediando por sucursal) | La comparación ingenua capacitadas vs. no capacitadas después de la capacitación concluye lo contrario |
| S082 no tiene quejas (abrió el 17 de marzo) | Tasa 0, no NULL; queda fuera de los rankings por tasa como el resto de las nuevas | Evita leerla como "la mejor sucursal" sin contexto |

Periodo: 2026-01-05 (lunes) a 2026-03-29 (domingo), 12 semanas. Tasa = quejas / pedidos × 1.000
(ratio de sumas).

## Decisiones de diseño

**Capa de datos única** (`agente/datos.py`). Cargo y limpio los CSV una vez, en DuckDB en memoria + pandas.
Todo lo demás consume esa capa; el LLM nunca ve los CSV.

**Herramientas tipadas y genéricas** (`agente/herramientas.py`). En vez de una herramienta por pregunta,
un catálogo chico que se combina:

| Herramienta | Para qué |
|---|---|
| `info_datos` | Periodo, regiones, categorías, reglas de limpieza |
| `buscar_sucursales` | Filtrar por región, nueva/capacitada, etc. |
| `contar_quejas` / `pedidos` | Conteos y sumas con filtros y agrupaciones |
| `tasa_quejas` / `ranking_sucursales` | Tasas emparejadas y rankings (con la regla de 28 días) |
| `comentarios_sucursal` | Texto de las quejas para resumir |
| `efecto_capacitacion` | DiD del Norte, con la advertencia sobre el resto |
| `candidatos_visita` | Score determinístico de recomendaciones |
| `consultar_sql` | Escape hatch: SQL de solo lectura, validado con sqlglot (`agente/sql_seguro.py`) |

Todas devuelven la misma estructura (`resultado`, `filtros`, `periodo`, `n_registros`, `advertencias`).
Si un argumento es inválido, el error lista los valores válidos para que el modelo se corrija solo.

**Grafo LangGraph propio** (`agente/grafo.py`, `docs/grafo.mmd`):

```mermaid
graph TD;
	__start__([__start__]) --> modelo;
	modelo -.-> herramientas;
	modelo -.-> recordatorio;
	modelo -.-> extraer_respuesta;
	modelo -.-> cerrar_con_advertencia;
	herramientas --> modelo;
	recordatorio --> modelo;
	extraer_respuesta -.-> verificar;
	extraer_respuesta -.-> modelo;
	extraer_respuesta -.-> cerrar_con_advertencia;
	verificar -. fin .-> __end__([__end__]);
	verificar -.-> feedback;
	verificar -.-> cerrar_con_advertencia;
	feedback --> modelo;
	cerrar_con_advertencia --> __end__;
```

- El modelo alterna con las herramientas. La respuesta sale por una tool `RespuestaFinal`. Con modelos que
  aceptan `tool_choice="any"` se fuerza; con Sonnet 5.5 va `"auto"` y, si contesta en texto libre, el nodo
  `recordatorio` le pide que llame a `RespuestaFinal`.
- `RespuestaFinal` tiene `texto` (para el gerente) y `valor` estructurado (número, texto o lista, para el eval).
- Después se verifican las cifras; si falla, hay **un** reintento con feedback y, si vuelve a fallar,
  se cierra con una advertencia visible. Máximo 10 llamadas al modelo.

Elegí un grafo propio y no un agente prearmado para controlar la verificación, el reintento y el corte.

**Recomendaciones** (`agente/recomendaciones.py`). El LLM no decide a quién visitar. Un score
determinístico ordena las sucursales por tasa de las últimas 4 semanas / tasa de su región; entran las que
superan 1,5× (de 3 a 5 sucursales). Si no llegan a 3, completo con las siguientes marcadas "Atención
preventiva". El LLM solo redacta el motivo y la acción, y el JSON lo imprime el código con `json.dumps`
(nunca el modelo). Toda pregunta que pida recomendaciones devuelve ese JSON.

**Fecha de referencia** 2026-03-29, el último día con datos: "esta semana" = 23 al 29 de marzo.

**Oráculo independiente** (`oraculo/oraculo.py`). Recalcula las respuestas esperadas con pandas puro sobre
los CSV y tiene prohibido importar el agente, para no validar el código contra sí mismo.

## Cómo evito que invente números

1. Limpieza determinística en la capa de datos, antes de que el modelo vea nada.
2. El modelo solo obtiene datos con las herramientas; el prompt le prohíbe hacer cálculos mentales
   (si necesita una cifra derivada, la pide a una herramienta).
3. **Verificador** (`agente/verificador.py`): busca cada cifra del texto, del `valor` y de la evidencia
   entre los números de los resultados de herramientas. Normaliza 2,15/2.15, separadores de miles y
   porcentajes; ignora años, fechas, IDs y los números que ya estaban en la pregunta.
4. Si hay cifras sin respaldo: un reintento con la lista de cifras; si persiste, advertencia visible.
5. Los IDs de sucursal se validan contra la tabla.
6. Las recomendaciones son JSON armado por el código. Si el modelo no termina, contesta en texto o
   inventa sucursales en una pregunta que pide recomendaciones, el JSON se arma de forma determinística
   con `candidatos_visita`. Sus cifras salen del score, así que verifican por construcción.

Límite conocido: el verificador busca cada cifra entre *todos* los números de los resultados, así que
un entero chico (3, 6, 13) puede pasar por coincidencia con un conteo. Protege bien contra cifras
inventadas "plausibles" (tasas, totales, porcentajes), no contra todos los casos.

## Costo y latencia

`TODO(costo/latencia)`: reemplazar con lo que mida el eval (el reporte ya junta tokens, costo y percentiles
de latencia por corrida).

Fórmula: `costo = tokens_entrada × precio_entrada + tokens_salida × precio_salida` (`agente/config.py`).

| Modelo | Entrada (USD/M tokens) | Salida (USD/M tokens) |
|---|---|---|
| Claude Sonnet 5.5 (agente) | 2,00 | 10,00 |
| Claude Haiku 5.5 (juez, prompts < 100k) | 0,10 | 0,50 |

Estimación previa, sin medir: cada llamada al modelo manda ~5.900 tokens fijos (system prompt ~1.250,
schemas de las 10 herramientas ~4.200, `RespuestaFinal` ~450) más el historial de la pregunta. Con 3–5
llamadas por pregunta son ~20–35k tokens de entrada y ~1k de salida → **~USD 0,05–0,08 por pregunta**.
Latencia: `TODO(costo/latencia)`. Cada corrida deja su traza (herramientas, argumentos, tokens, costo,
latencia, verificación) en `reportes/trazas/AAAAMMDD.jsonl`.

## Cómo funciona el eval

**Extracción.** Primero uso el `valor` estructurado de la respuesta, normalizado según el tipo de la
pregunta. Si falta o no sirve, un parser lee el texto:
- números: un `.` o `,` seguido de exactamente 3 dígitos es separador de miles; si no, es decimal
  ("2,15" = 2.15, "6.646" = 6646). Tomo el primer número del texto, no el más cercano a la esperada;
- listas: IDs `S###` en orden de aparición, sin duplicados, y comparo en el mismo orden.

**Textos.** Normalizo (minúsculas, sin tildes ni puntuación) y busco la esperada o sus palabras/tokens
clave (números, fechas ISO, IDs, palabras de 4+ letras). Si la esperada es larga o no matchea, la decide
un LLM juez (Haiku 5.5, temperatura 0) con una rúbrica cerrada: ¿contiene los hechos esperados?
¿contradice alguno? Con `--sin-juez` esas corridas fallan con motivo "requiere juez".
Si la esperada es "no hay datos" (preguntas adversariales), la corrida pasa solo si el texto lo dice
explícitamente y `valor` está vacío: un número en la respuesta cuenta como inventado.

**Preguntas propias** (`eval/preguntas_propias.jsonl`): cada una tiene una trampa de los datos.

| ID | Pregunta (resumida) | Trampa |
|---|---|---|
| A01 | Pedidos de la región Costa | Los 99999: sin limpiar, el total se infla |
| A02 | Sucursal con más pedidos la semana del 5 de enero | El 99999 de S036 la pone primera |
| A03 | Día de la semana con más pedidos en promedio | Los outliers inflan el viernes |
| A04 | Top 3 de peor tasa entre las que operaron todo el periodo | Duplicados + sucursales nuevas |
| A05 | Efecto de la capacitación | Sesgo de selección: solo el Norte es causal |
| A06 | Total de quejas | Duplicados |

Además, `eval/preguntas_robustez.jsonl` (`make eval-robustez`) tiene 34 preguntas: 2–3 paráfrasis por
cada pregunta de ejemplo y adversariales (sucursal S121, región "Oeste", abril de 2026, ventas en pesos,
empleados, "esta semana", "la semana pasada", peor sucursal por tasa, total de pedidos). Todas las
esperadas salen del oráculo.

## Estado y pendientes

Todo lo que corre sin clave está hecho y testeado: 524 tests offline en verde, más 25 que necesitan clave
y se saltean. Lo que falta necesita `ANTHROPIC_API_KEY`:

- `TODO(eval)`: correr `make eval` con clave real y pegar la tasa de aciertos por pregunta (los tests
  offline usan un modelo falso).
- `TODO(costo/latencia)`: completar con los números medidos por el eval.
- `TODO(spike)`: correr la parte real del spike (`python spikes/spike_grafo.py --real`); la parte
  offline está aprobada (`spikes/RESULTADO.md`).
- `TODO(tests con clave)`: `ANTHROPIC_API_KEY=... make test` corre los 25 tests `requiere_clave` (P01–P10
  reales, 8 pedidos de recomendación, 5 que no lo son, estabilidad).
- `uv.lock`: generarlo con `uv lock` y commitearlo, para fijar versiones (sobre todo `langchain-anthropic`,
  que valida parámetros por modelo).

## Qué mejoraría con más tiempo

- Intervalos de confianza de Poisson o shrinkage bayesiano para rankear sucursales con poco volumen.
- Prompt caching del system prompt y del catálogo de herramientas (baja costo y latencia).
- Probar con otros proveedores (el código ya lo permite) y comparar.
- REPL con memoria de conversación para repreguntas.
- Más preguntas adversariales (ambiguas, fuera del periodo, sucursales inexistentes).
- Correr el eval en CI con clave, con `--umbral` como compuerta.

## Estructura del repo

```
agente/        capa de datos, herramientas, SQL seguro, grafo, nodos, prompts, verificador, recomendaciones
datos/         CSV de entrada (copia de los provistos por Airbag)
eval/          evaluar.py, comparadores, juez, preguntas de ejemplo y propias
oraculo/       recálculo independiente de las respuestas esperadas (pandas puro)
spikes/        spike de LangGraph + resultado
tests/         tests offline (modelo falso, sin red)
docs/grafo.mmd diagrama del grafo
main.py        CLI
```

## Tests

```bash
make test                          # usa uv run python
make test PY=python                # con otro intérprete (por ejemplo, un venv ya activado)
```

Corren offline: no necesitan clave ni red. Los marcados `requiere_clave` llaman al modelo real y se
saltean siempre, aunque haya una clave en `.env`, salvo que se pidan con `make test-clave`.
CI los corre en cada push (`.github/workflows/tests.yml`).

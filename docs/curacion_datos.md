# Curación de datos

Este documento explica dónde viven los datos, cómo llegan al agente y qué transformaciones se aplican sobre los CSV provistos. Cada regla incluye el motivo y su impacto medido.

Periodo de los datos: 2026-01-05 (lunes) a 2026-03-29 (domingo), 84 días, 120 sucursales (30 por región).

---

## 1. Dónde viven los datos y cómo los consume el agente

- **Fuente única:** los CSV (`sucursales.csv`, `ventas_diarias.csv`, `quejas.csv`, `capacitaciones.csv`) en `datos/`, que es una copia sin cambios de los provistos por Airbag. La ruta se puede cambiar con `DATOS_DIR` (`agente/config.py`, función `datos_dir()`). Los CSV **nunca se modifican**.
- **No hay un dataset curado en disco.** La curación corre en memoria cada vez que arranca el agente, el eval o los tests: `cargar_datos(dir)` en `agente/datos.py` lee los CSV, los valida y limpia, y devuelve:
  - un objeto `Datos` con los DataFrames limpios (pandas);
  - una conexión **DuckDB en memoria** con las vistas limpias registradas.
- La carga tarda menos de 2 segundos. Tener una sola función determinística evita archivos intermedios que se desincronicen del código: lo que está escrito acá es exactamente lo que se ejecuta.

```
CSV crudos ──► cargar_datos() ──► DataFrames limpios + DuckDB en memoria (vistas)
                                          │
                                          ▼
                               agente/herramientas.py (funciones puras)
                                          │  JSON: resultado, filtros, periodo, advertencias
                                          ▼
                     LLM (wrappers en herramientas_lc.py) ──► RespuestaFinal ──► verificador
```

- **El LLM no ve nunca los datos crudos.** Solo llama herramientas tipadas, que consultan los datos limpios y devuelven JSON.
- `consultar_sql` (respaldo, `agente/sql_seguro.py`) usa **otra** conexión DuckDB, que solo tiene las 5 vistas limpias. En esa conexión, la vista `ventas` ni siquiera expone la columna cruda `pedidos`: solo `pedidos_validos`. La consulta se valida con `sqlglot`: tiene que ser una única sentencia `SELECT` (se aceptan `WITH` y `UNION`) y solo puede leer esas vistas. Devuelve como máximo 200 filas, con un timeout de 5 segundos.
- El verificador comprueba que cada número de la respuesta aparezca en algún resultado de herramienta.

---

## 2. Hallazgos por archivo

| Archivo | Filas | Problema | Tratamiento |
|---|---|---|---|
| `sucursales.csv` | 120 | Ninguno. 6 sucursales abrieron durante el periodo (16 y 17 de marzo) | Se agregan `es_nueva` y `dias_operados`; se tratan aparte en los rankings por tasa de quejas |
| `ventas_diarias.csv` | 9.658 | 5 filas con `99999` (centinela) y 3 con pedidos negativos | `pedidos_validos = NULL` y `dato_invalido = True` |
| `quejas.csv` | 6.845 | 199 duplicados exactos por `queja_id` | Deduplicación → 6.646 |
| `capacitaciones.csv` | 36 | Datos limpios, pero la asignación está confundida con la región | Se analiza con DiD solo en el Norte (ver sección 5) |

Chequeos que dieron OK: formato de fechas, IDs consistentes entre tablas, sin días faltantes, sin ventas ni quejas antes de la apertura. La `categoria` siempre coincide con el `comentario`, así que no hace falta reclasificar texto. S082 no tiene quejas.

---

## 3. Transformaciones, en el orden en que se aplican

Todo está en `agente/datos.py`, función `cargar_datos()`.

### 3.1 Lectura con tipos explícitos
- Cada archivo tiene su esquema en `ESQUEMAS`, con las columnas esperadas y el `dtype` de cada una.
- Se verifica que estén todas las columnas, se leen con ese `dtype` y las fechas se parsean con formato fijo `%Y-%m-%d`.
- Así no hay inferencia de tipos que cambie entre versiones de pandas (se probó con 2.3 y 3.0).

### 3.2 Validaciones bloqueantes
Si alguna falla, la carga se corta con `ErrorValidacionDatos`, que incluye ejemplos de las filas problemáticas:
- `sucursal_id` único en `sucursales.csv` y en `capacitaciones.csv`;
- regiones, categorías, canales y métodos de asignación dentro de los conjuntos permitidos;
- todo `sucursal_id` de ventas, quejas y capacitaciones existe en `sucursales`;
- las fechas caen dentro del periodo;
- no hay pares (fecha, sucursal) repetidos en ventas;
- no hay ventas antes de la `fecha_apertura` de la sucursal;
- toda queja cae en un día que tiene fila de ventas.

### 3.3 Quejas: deduplicación por `queja_id`
- 6.845 filas → **6.646 quejas**. Los duplicados se concentran en algunas sucursales (S052 tiene 10, S115 tiene 7).
- `_deduplicar_quejas()` primero cuenta los duplicados exactos y después deja una fila por `queja_id`, la primera.
- Si dos filas tuvieran el mismo ID con distinto contenido, se registra un aviso. En estos datos no pasa: los 199 son duplicados exactos.
- **Impacto:** las preguntas de ejemplo provistas por Airbag (`eval/preguntas_ejemplo.jsonl`) solo dan la respuesta esperada con los datos deduplicados:

| Pregunta | Crudo | Deduplicado (= esperado) |
|---|---|---|
| Quejas del Sur en febrero | 455 | 442 |
| Tasa de quejas del Centro por 1.000 pedidos | 2,21 | 2,15 |
| Top 3 por tasa de quejas en febrero | S107, S012, S111 | S012, S107, S067 |
| Quejas de S104 y peso de limpieza | 63 quejas, 81% limpieza | 60 quejas, 82% limpieza |

### 3.4 Ventas: pedidos inválidos
- **Regla explícita:** un valor es inválido si `pedidos < 0` o `pedidos >= 99999`. Marca exactamente 8 filas:

| fecha | sucursal | pedidos |
|---|---|---|
| 2026-01-10 | S036 | 99999 |
| 2026-02-09 | S092 | 99999 |
| 2026-02-18 | S085 | -5 |
| 2026-02-28 | S016 | -44 |
| 2026-03-13 | S112 | -14 |
| 2026-03-16 | S081 | 99999 |
| 2026-03-20 | S035 | 99999 |
| 2026-03-27 | S100 | 99999 |

- Las filas **no se borran**. Se agregan el flag `dato_invalido` y `pedidos_validos`, de tipo `Int64` nullable, que vale `<NA>` (NULL en DuckDB) en esas filas. La columna cruda `pedidos` se conserva en el DataFrame para auditar. Todas las sumas y tasas de quejas usan `pedidos_validos`.
- **Chequeo estadístico complementario:** z robusto por sucursal, calculado con mediana y MAD sobre las filas válidas. Si `|z| > 8`, **solo emite un aviso** y no limpia. Sirve para detectar outliers nuevos sin borrar datos reales.
- No se imputa ningún valor, para no inventar datos.
- **Impacto:**
  - Total de pedidos: 3.746.788 en crudo contra **3.246.856** limpio (+15%).
  - Costa: 891.509 contra **691.516**.
  - Día de la semana con más pedidos en promedio: en crudo sale el viernes; limpio, **el sábado**.
  - Sucursal con más pedidos en la semana del 5 de enero: en crudo sale S036 (101.367); limpio, **S052** (6.575).

### 3.5 Quejas en días con pedidos inválidos
- Cada queja lleva `en_dia_invalido`, que vale `True` si su (fecha, sucursal) tiene la fila de ventas inválida. Son 7.
- La regla vive en una única función, `tasa_emparejada()`, que usan todas las herramientas que calculan tasas de quejas.
- **Por qué esta definición:** las preguntas que importan ("¿qué sucursal está peor?", "¿funcionó la capacitación?") comparan sucursales, regiones y periodos de volumen muy distinto. Por eso hace falta una medida que no dependa del tamaño.
  - El conteo de quejas no sirve para comparar: una sucursal con el doble de pedidos tiene más quejas aunque atienda igual de bien. Normalizar por 1.000 pedidos resuelve eso.
  - Se usa el ratio de sumas y no un promedio de tasas de quejas: así cada pedido pesa lo mismo. Un día o una sucursal con pocos pedidos daría tasas de quejas extremas que, promediadas, pesarían igual que una sucursal grande. Además, es la definición con la que coinciden las respuestas esperadas de las preguntas de ejemplo.
  - Numerador y denominador se calculan sobre los mismos días: en un día inválido no hay un denominador confiable, y dejar sus quejas sin sus pedidos sesgaría la tasa de quejas hacia arriba.
- **Conteos** (`contar_quejas`): incluyen las 7, porque la queja existió aunque el dato de pedidos sea malo.
- **Tasas de quejas:** numerador y denominador se calculan sobre los mismos días:
  `tasa de quejas = sum(quejas en días válidos) / sum(pedidos_validos) × 1000`
- Las herramientas de tasa de quejas devuelven `quejas_excluidas` y `dias_excluidos`, para que el agente pueda explicar por qué el numerador no coincide con el conteo.
- **Alternativa descartada:** excluir el día solo del denominador. Da los mismos rankings (por periodo y por mes) y mueve entre 0,02 y 0,07 la tasa de quejas de esas 8 sucursales. Se eligió la versión emparejada porque es la estadísticamente consistente. Ejemplos: S035 = 1,393 y S100 = 2,293.

### 3.6 Columnas derivadas
- **Sucursales:**
  - `es_nueva`: `fecha_apertura` dentro del periodo;
  - `dias_operados`: cantidad de filas de ventas;
  - `fecha_inicio_datos`: la apertura o el 2026-01-05, la que sea posterior;
  - `capacitada`, `fecha_capacitacion` y `metodo_asignacion`, que se traen de `capacitaciones.csv`.
- **Ventas, quejas y diario:** `region`, más las columnas de calendario:
  - `semana_inicio`: el lunes de la semana (las semanas van de lunes a domingo);
  - `semana_n`: de 1 a 12, donde la semana 1 es del 2026-01-05 al 2026-01-11;
  - `mes`: con formato `"AAAA-MM"`;
  - `dia_semana`: en español, en minúscula y con tilde (`"sábado"`).

### 3.7 Tabla diaria y vistas DuckDB
- `diario_sucursal` tiene una fila por (fecha, sucursal) con `pedidos_validos`, `quejas`, una columna `quejas_<categoria>` por categoría, `dato_invalido` y las columnas de calendario. Es la base de todas las tasas de quejas.
- Vistas registradas en DuckDB en memoria, con las fechas como `DATE`: `sucursales`, `ventas` (limpia), `quejas` (deduplicada), `capacitaciones` y `diario_sucursal`.

### 3.8 Fecha de referencia
`FECHA_REFERENCIA = 2026-03-29`, el último día con datos. "Esta semana" significa del 2026-03-23 al 2026-03-29, no la fecha real.

### 3.9 Reporte de calidad
`reporte_calidad(datos)` devuelve un dict serializable a JSON con:
- las filas crudas y limpias por tabla;
- el detalle de los duplicados;
- las 8 filas inválidas y las 7 quejas afectadas;
- las sucursales nuevas;
- todos los avisos;
- el tiempo de carga.

Con eso se puede auditar la curación sin leer el código. La herramienta `info_datos` le expone esa información al agente, para que pueda explicar la limpieza si se lo preguntan.

---

## 4. Reglas que viven en las herramientas (no en la limpieza)

La limpieza deja los datos correctos. Estas reglas definen **cómo se calculan** las métricas sobre esos datos:

- **Tasa de quejas = ratio de sumas**, nunca un promedio de tasas de quejas diarias o por sucursal. Así sale el 2,15 esperado para la tasa de quejas del Centro.
- **Sucursales nuevas en rankings por tasa de quejas** (`DIAS_MINIMOS_RANKING_TASA = 28` en `agente/herramientas.py`): entra solo una sucursal con al menos 28 días abierta al final de la ventana. Con `incluir_nuevas=True` entran todas. Quedan fuera S015, S033, S037, S059, S076 y S082, que tienen entre 13 y 14 días y unos 700 a 900 pedidos (1 queja mueve la tasa de quejas de S015 en 1,34, frente a 0,04 en una sucursal típica). Las excluidas se devuelven igual en `excluidas`, con la posición que tendrían. Los rankings por conteo o por pedidos no excluyen a nadie.
- **Periodos sin datos** (por ejemplo, abril de 2026): resultado vacío con advertencia, nunca un 0 sin aviso.
- **Empates:** se desempata por `sucursal_id`.

---

## 5. Capacitaciones

- `capacitaciones.csv` no necesitó limpieza: 36 sucursales capacitadas el 2026-02-09, sin duplicados ni IDs inexistentes.
- Se valida en la carga: sucursal existente y sin repetir, fecha dentro del periodo y `metodo_asignacion` igual a `aleatorio` o `por_desempeno`.
- Se une a `sucursales` con tres columnas: `capacitada`, `fecha_capacitacion` y `metodo_asignacion`.
- Se usa en dos herramientas:
  - `efecto_capacitacion`, que responde si la capacitación redujo las quejas. La metodología y el resultado están en el README, en la sección "Qué encontré en los datos y cómo lo traté".
  - `candidatos_visita`: si una sucursal recomendada ya fue capacitada, la acción sugerida nunca es "capacitar". Es de seguimiento, salvo que la tasa de quejas siga subiendo después de la capacitación: en ese caso es "escalar".

---

## 6. Cómo se controla que la curación es correcta

- **Oráculo independiente** (`oraculo/oraculo.py`): recalcula las respuestas esperadas con pandas puro, **sin reutilizar** `datos.py` ni `herramientas.py`. Así un error de limpieza no queda validado contra sí mismo.
- **Tests de la capa de datos** (`tests/test_datos.py`), entre otros:
  - 120 sucursales, 30 por región, 6 nuevas;
  - 6.646 quejas y 0 `queja_id` duplicados;
  - exactamente 8 filas con `dato_invalido` y 7 quejas con `en_dia_invalido`;
  - total de pedidos válidos 3.246.856 y Costa 691.516;
  - 84 días de ventas por sucursal, o los días desde la apertura en las nuevas;
  - la carga falla con un CSV corrupto (por ejemplo, una categoría desconocida);
  - el oráculo reproduce las respuestas esperadas de las preguntas de ejemplo.
- **Preguntas propias del eval** (`eval/preguntas_propias.jsonl`): cada una cae en una de las trampas de arriba. A01 (Costa) y A02 (semana del 5 de enero) en el 99999; A03 (día de la semana) en los outliers; A04 (peores tasas de quejas) en las sucursales nuevas y los duplicados; A05 en la capacitación; A06 (total de quejas) en los duplicados.

# Cómo funciona el eval

Complementa la sección "Cómo funciona el eval" del [README](../README.md). El código está en
`eval/comparadores.py` (extracción y comparación) y `eval/juez.py` (LLM juez).

**Extracción.** Primero uso el `valor` estructurado de la respuesta, normalizado según el tipo de la
pregunta. Si falta o no sirve, un parser lee el texto:
- números: un `.` o `,` seguido de exactamente 3 dígitos es separador de miles; si no, es decimal
  ("2,15" = 2.15, "6.646" = 6646). Tomo el primer número del texto, no el más cercano a la esperada.
  Esta regla no se aplica al `valor`: ahí el schema pide punto decimal sin miles, así que "2.266" se
  lee como 2,266 (con la regla del texto daba 2266 y fallaba R03v en el eval de robustez);
- listas: IDs `S###` en orden de aparición, sin duplicados, y comparo en el mismo orden.

**Textos.** Normalizo (minúsculas, sin tildes ni puntuación) y busco la esperada o sus palabras/tokens
clave (números, fechas ISO, IDs, palabras de 4+ letras). Si la esperada es larga o no matchea, la decide
un LLM juez (Haiku 5.5) con una rúbrica cerrada: ¿contiene los hechos esperados?
¿contradice alguno? Juzga hechos, no forma (la longitud o el detalle extra no restan), recibe el texto y
el `valor` rotulados por separado y escribe su razón antes del veredicto, para que el sí/no salga del
razonamiento. Con `--sin-juez` esas corridas fallan con motivo "requiere juez".
Si la esperada es "no hay datos" (preguntas adversariales), la corrida pasa solo si el texto lo dice
explícitamente y `valor` está vacío: un número en la respuesta cuenta como inventado.

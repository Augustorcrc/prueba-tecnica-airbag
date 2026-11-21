.PHONY: instalar test eval eval-robustez ask lint

# PY se puede sobrescribir: make test PY=.venv-vm/bin/python
PY ?= uv run python
REPS ?= 3

instalar:
	uv sync

test:
	$(PY) -m pytest

eval:
	$(PY) -m eval.evaluar eval/preguntas_ejemplo.jsonl eval/preguntas_propias.jsonl --repeticiones $(REPS) --salida reportes/

eval-robustez:
	$(PY) -m eval.evaluar eval/preguntas_robustez.jsonl --repeticiones $(REPS) --salida reportes/robustez/

ask:
	$(PY) main.py "$(Q)"

lint:
	$(PY) -m ruff check .

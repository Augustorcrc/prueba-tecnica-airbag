.PHONY: instalar test test-clave chat eval eval-robustez ask lint docker-test docker-build docker-ask docker-eval

# PY se puede sobrescribir, por ejemplo: make test PY=python (con un venv ya activado)
PY ?= uv run python
REPS ?= 3

instalar:
	uv sync

test:
	$(PY) -m pytest

# Tests que llaman al modelo real (gastan créditos; necesitan ANTHROPIC_API_KEY).
test-clave:
	PRUEBAS_CON_CLAVE=1 $(PY) -m pytest -m requiere_clave

eval:
	$(PY) -m eval.evaluar eval/preguntas_ejemplo.jsonl eval/preguntas_propias.jsonl --repeticiones $(REPS) --salida reportes/

eval-robustez:
	$(PY) -m eval.evaluar eval/preguntas_robustez.jsonl --repeticiones $(REPS) --salida reportes/robustez/

ask:
	$(PY) main.py "$(Q)"

# Conversación con memoria (opcional; no la usa el eval).
chat:
	$(PY) main.py --chat

lint:
	$(PY) -m ruff check .

IMAGEN ?= agente-cafe

docker-test:
	docker build --target test -t $(IMAGEN):test .

docker-build:
	docker build -t $(IMAGEN) .

docker-ask:
	docker run --rm --env-file .env $(IMAGEN) "$(Q)"

docker-eval:
	docker run --rm --env-file .env -v "$(CURDIR)/reportes:/app/reportes" --entrypoint python $(IMAGEN) \
		-m eval.evaluar eval/preguntas_ejemplo.jsonl eval/preguntas_propias.jsonl --repeticiones $(REPS) --salida reportes/

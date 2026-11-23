# syntax=docker/dockerfile:1
# Imagen del agente Café Tostadero. Las versiones salen de uv.lock (uv sync --frozen): el build falla
# si el lock no existe o no coincide con pyproject.toml, así la imagen es reproducible.
#   docker build --target test -t agente-cafe:test .   # corre los tests offline en el build
#   docker build -t agente-cafe .                       # imagen final (sin dependencias de desarrollo)
#   docker run --rm --env-file .env agente-cafe "¿Cuántas quejas hubo en total?"

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
# Primero solo las dependencias: esta capa se cachea mientras no cambie el lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY agente ./agente
COPY eval ./eval
COPY oraculo ./oraculo
COPY datos ./datos
COPY main.py ./
ENV PATH="/app/.venv/bin:$PATH"

# Etapa de tests: agrega las dependencias de desarrollo y corre la suite offline (sin clave).
FROM base AS test
RUN uv sync --frozen --no-install-project
COPY tests ./tests
COPY docs ./docs
RUN python -m pytest -o addopts="" -q

# Imagen final: usuario sin privilegios; la clave llega por variable de entorno, nunca en la imagen.
FROM base AS runtime
RUN useradd --create-home --uid 1000 agente && mkdir -p /app/reportes && chown -R agente /app/reportes
USER agente
ENTRYPOINT ["python", "main.py"]

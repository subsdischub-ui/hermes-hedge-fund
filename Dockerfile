# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12

# ----------------------------- builder -------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install build deps for torch + chroma's onnxruntime (slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Vendor Samsung-SAIL TRM source for the verifier to import (MIT — embedding is fine).
RUN git clone --depth 1 https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git /opt/trm

COPY pyproject.toml ./
# uv lock + sync in one cached layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && uv pip install --python /opt/venv/bin/python -e .

COPY src ./src
COPY config ./config
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python -e .

# --------------------------- runtime --------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

RUN groupadd -r hermes && useradd -r -g hermes -u 1001 -m -d /home/hermes hermes \
 && apt-get update && apt-get install -y --no-install-recommends curl tini ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HERMES_DATA_DIR=/data \
    PYTHONPATH="/opt/trm:/app/src"

WORKDIR /app
COPY --from=builder --chown=hermes:hermes /opt/venv /opt/venv
COPY --from=builder --chown=hermes:hermes /opt/trm  /opt/trm
COPY --chown=hermes:hermes src ./src
COPY --chown=hermes:hermes config ./config

RUN mkdir -p /data /data/trm /data/chroma && chown -R hermes:hermes /data
USER hermes
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","hermes.api:app","--host","0.0.0.0","--port","8000"]

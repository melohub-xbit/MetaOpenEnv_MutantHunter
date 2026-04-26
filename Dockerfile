# MutantHunter env server.
# Local: docker build -t mutant-hunter:latest -f Dockerfile .
# HF Spaces (Docker SDK) sets PORT=7860; this image obeys it.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MUTANT_HUNTER_HOME=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        util-linux \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY evaluation/ ./evaluation/

RUN pip install --no-cache-dir -e .

# Baselines are committed under src/mutant_hunter/corpus/_baselines/ and
# copied in by `COPY src/ ./src/` above. --skip-existing makes this a no-op
# when the cache is present, but still self-heals if a baseline is missing.
RUN python scripts/precompute_baselines.py --skip-existing

# Drop privileges. The agent's submitted test code runs in a further
# subprocess sandbox (RLIMITs + unshare -n), but defense in depth.
RUN useradd --create-home --uid 1000 mutant \
    && chown -R mutant:mutant /app
USER mutant

EXPOSE 7860
ENV PORT=7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn mutant_hunter.server.app:app --host 0.0.0.0 --port ${PORT}"]

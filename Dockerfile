# Multi-stage: wheels are built in a throwaway stage so the runtime image never
# carries pip's cache or any build tooling.
#
# Runs the API and the built UI from one container. Honours $PORT, so the same
# image works on Render, Fly, or a Hugging Face Space.

# ---------- builder ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
# Resolve once into a prefix we can copy wholesale into the runtime stage.
RUN pip install --prefix=/install --no-cache-dir ".[web,nvidia]"

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Run unprivileged: a container that never needs to write shouldn't be root.
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/api

COPY --from=builder /install /usr/local

COPY --chown=app:app src ./src
COPY --chown=app:app api.py ask.py pyproject.toml README.md ./
COPY --chown=app:app data/quran_full.jsonl data/embeddings.npz ./data/
COPY --chown=app:app scripts/build_tafseer_db.py ./scripts/
COPY --chown=app:app web/dist ./web/dist

# Derive the commentary database in the image rather than committing 28 MB of
# generated data. The JSONL stays the single source of truth; the server reads
# tafseer from SQLite so it never occupies resident memory.
RUN python scripts/build_tafseer_db.py data/quran_full.jsonl data/tafseer.sqlite3

USER app

ENV QURAN_DATA_PATH=/home/app/api/data/quran_full.jsonl \
    EMBEDDINGS_PATH=/home/app/api/data/embeddings.npz \
    TAFSEER_DB=/home/app/api/data/tafseer.sqlite3 \
    PORT=7860

EXPOSE 7860

# Liveness only — never gate the container's health on a network dependency.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz',timeout=4)" || exit 1

# One worker on purpose: the free instance has 512 MB and each worker would load
# its own copy of the corpus and index. Scale with instances, not workers.
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --timeout-keep-alive 65 --proxy-headers --forwarded-allow-ips '*'"]

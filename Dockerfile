# Single container: the FastAPI API plus the built React UI. Honours $PORT, so
# the same image runs on Render, Fly, or a Hugging Face Space.
#
# Deliberately single-stage with `pip install --user`: this is the pattern that
# builds and boots cleanly on Render's free tier. An earlier multi-stage variant
# (pip --prefix into a throwaway stage, then COPY into /usr/local) shaved image
# size but never deployed — the copied environment failed to import at runtime,
# so the health check never passed and Render kept the previous image. Reliability
# beats a smaller image here.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Run unprivileged: the container never needs to write outside its own data dir.
RUN useradd --create-home --uid 1000 app
USER app
WORKDIR /home/app/api

# Dependencies first, as their own cache layer. --user installs console scripts
# (uvicorn) into ~/.local/bin, which the PATH below picks up.
COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src ./src
RUN pip install --user --no-cache-dir ".[web,nvidia]"
ENV PATH="/home/app/.local/bin:$PATH"

# App code, data, and the built UI.
COPY --chown=app:app api.py ask.py ./
COPY --chown=app:app data/quran_full.jsonl data/embeddings.npz ./data/
COPY --chown=app:app scripts/build_tafseer_db.py ./scripts/
COPY --chown=app:app web/dist ./web/dist

# Derive the commentary database in the image rather than committing 28 MB of
# generated data. The JSONL stays the single source of truth; the server reads
# tafseer from SQLite so it never occupies resident memory. Built and read by the
# same (app) user, so no cross-user permission problem.
RUN python scripts/build_tafseer_db.py data/quran_full.jsonl data/tafseer.sqlite3

ENV QURAN_DATA_PATH=/home/app/api/data/quran_full.jsonl \
    EMBEDDINGS_PATH=/home/app/api/data/embeddings.npz \
    TAFSEER_DB=/home/app/api/data/tafseer.sqlite3 \
    PORT=7860

EXPOSE 7860

# Liveness only — /healthz touches no dependency, so a slow upstream can never
# make the platform restart a container that is actually fine.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/healthz',timeout=4)" || exit 1

# One worker on purpose: the free instance has 512 MB and each worker would load
# its own copy of the corpus and index. Scale with instances, not workers.
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --timeout-keep-alive 65 --proxy-headers --forwarded-allow-ips '*'"]

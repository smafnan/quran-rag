# Hugging Face Space (Docker SDK) — serves the API and the built UI together.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user

# Spaces run as a non-root user
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --user --no-warn-script-location -e ".[web,nvidia]"

COPY --chown=user api.py ask.py ./
COPY --chown=user data ./data
COPY --chown=user web/dist ./web/dist

ENV PATH="/home/user/.local/bin:$PATH" \
    QURAN_DATA_PATH=/home/user/app/data/quran_full.jsonl \
    EMBEDDINGS_PATH=/home/user/app/data/embeddings.npz

# HF Spaces expect 7860; Render/Fly inject $PORT. Honour whichever is set.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]

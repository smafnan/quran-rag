"""FastAPI backend for the Quran RAG web UI.

Serves the grounded search over HTTP and the built React frontend (web/dist).
Run:  uvicorn api:app --reload  →  http://localhost:8000
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.groundedrag import (NvidiaEmbedder, RateLimiter, Retriever, TafseerStore,
                             client_key, explain_passage, get_llm, load_corpus)

ROOT = Path(__file__).resolve().parent
SOURCE_NAME = "the Quran"

app = FastAPI(title="Quran RAG API", version="1.0.0")
# Open by default so a clone works anywhere. When the UI is hosted separately
# (e.g. Netlify) set ALLOWED_ORIGINS to that origin so only it may spend the
# deployment's API quota from a browser.
_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins or ["*"],
                   allow_methods=["*"], allow_headers=["*"])

_default_corpus = ROOT / "data" / "quran_full.jsonl"
if not _default_corpus.exists():
    _default_corpus = ROOT / "data" / "quran_sample.jsonl"
_data_path = os.environ.get("QURAN_DATA_PATH", str(_default_corpus))

# When the tafseer database is present, keep the commentary on disk (~54 MB of
# RAM saved) and read it per verse on demand. Falls back to the JSONL's inline
# tafseer when it is absent, so a plain clone still works.
_tafseer_db = Path(os.environ.get("TAFSEER_DB", str(ROOT / "data" / "tafseer.sqlite3")))
_tafseer_store = TafseerStore(_tafseer_db) if _tafseer_db.exists() else None
_passages = load_corpus(_data_path, load_tafseer=_tafseer_store is None)

# Semantic layer: needs the offline index (scripts/build_embeddings.py) plus a
# key for query-time embeddings. Without either, search still works via the
# keyword + TF-IDF layers.
_embed_key = os.environ.get("NVIDIA_EMBED_API_KEY") or os.environ.get("NVIDIA_API_KEY")
_embeddings_path = os.environ.get("EMBEDDINGS_PATH", str(ROOT / "data" / "embeddings.npz"))
_embedder = NvidiaEmbedder(_embed_key) if _embed_key else None

_retriever = Retriever(_passages, embeddings_path=_embeddings_path,
                       embed_query=_embedder)
_passage_by_ref = {p.ref: p for p in _passages}

# The "explain" endpoint needs a real text-generation LLM (not the retriever).
# Configure via EXPLAIN_PROVIDER ("nvidia" | "anthropic") + the matching API key
# env var. Left unconfigured, /api/explain reports that clearly instead of
# silently failing or fabricating an explanation.
# Each provider gets its OWN key — never fall back across vendors, or an
# explicit EXPLAIN_PROVIDER would ship one vendor's secret to the other's API.
_PROVIDER_KEY_ENV = {"nvidia": "NVIDIA_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
_explain_provider = os.environ.get(
    "EXPLAIN_PROVIDER",
    "nvidia" if os.environ.get("NVIDIA_API_KEY") else
    "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "",
)
_explain_api_key = os.environ.get(_PROVIDER_KEY_ENV.get(_explain_provider, ""), "")
_explain_model = os.environ.get("EXPLAIN_MODEL")
_llm = None
if _explain_provider:
    if not _explain_api_key:
        print(f"warning: EXPLAIN_PROVIDER={_explain_provider} but "
              f"{_PROVIDER_KEY_ENV.get(_explain_provider, '<key>')} is unset — "
              f"explanations disabled", file=sys.stderr)
    else:
        try:
            _llm = get_llm(_explain_provider, api_key=_explain_api_key,
                           model=_explain_model)
        except Exception as e:  # unknown provider, missing SDK, ...
            print(f"warning: could not initialise '{_explain_provider}' provider "
                  f"({e}) — explanations disabled", file=sys.stderr)


# Both paid paths are capped per client. A public deployment otherwise hands the
# deployer's API quota to anyone with the URL. Set the limit to 0 to disable.
_ask_limiter = RateLimiter(int(os.environ.get("ASK_RATE_LIMIT", "60")), 300)
_explain_limiter = RateLimiter(int(os.environ.get("EXPLAIN_RATE_LIMIT", "15")), 3600)


def _has_tafseer(passage) -> bool:
    return _tafseer_store.has(passage.ref) if _tafseer_store else bool(passage.tafseer)


def _tafseer_for(passage) -> str:
    return _tafseer_store.get(passage.ref) if _tafseer_store else passage.tafseer


def _enforce(limiter: RateLimiter, request: Request, what: str) -> None:
    allowed, retry_after = limiter.check(client_key(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached for {what}. Try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=100)
    mode: str = "top"  # "top" = best k results; "all" = every occurrence corpus-wide


class ExplainRequest(BaseModel):
    ref: str = Field(min_length=1, max_length=32)
    question: str = Field(default="", max_length=2000)


@app.get("/healthz")
def healthz():
    """Liveness: the process is up. Touches nothing, so a slow dependency can
    never cause the platform to kill an otherwise healthy container."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness: the corpus and index are actually loaded and searchable."""
    if not _passages or _retriever is None:
        raise HTTPException(status_code=503, detail="corpus not loaded")
    return {
        "status": "ready",
        "passages": len(_passages),
        "semantic": _retriever._sem_matrix is not None,
        "tafseer_store": _tafseer_store is not None,
    }


@app.get("/api/info")
def info():
    return {
        "source": SOURCE_NAME,
        "passages": len(_passages),
        "explain_available": _llm is not None,
        "semantic_available": _retriever._sem_matrix is not None and _embedder is not None,
    }


@app.post("/api/ask")
def ask(req: AskRequest, request: Request):
    # only the semantic layer costs an API call; lexical-only search is free
    if _embedder is not None and _retriever._sem_matrix is not None:
        _enforce(_ask_limiter, request, "search")
    mode = req.mode if req.mode in ("top", "all") else "top"
    hits = _retriever.search(req.question, top_k=req.k, mode=mode)
    return {
        "found": bool(hits),
        "source": SOURCE_NAME,
        "count": len(hits),
        "results": [
            {
                "ref": h.passage.ref,
                "text": h.passage.text,
                "arabic": h.passage.arabic,
                "score": round(h.score, 3),
                "matched": h.matched,
                "has_tafseer": _has_tafseer(h.passage),
            }
            for h in hits
        ],
    }


@app.get("/api/tafseer")
def tafseer(ref: str):
    passage = _passage_by_ref.get(ref)
    if passage is None:
        raise HTTPException(status_code=404, detail=f"No passage found for ref '{ref}'.")
    return {
        "ref": passage.ref,
        "arabic": passage.arabic,
        "text": passage.text,
        "tafseer": _tafseer_for(passage),
    }


@app.post("/api/explain")
def explain(req: ExplainRequest, request: Request):
    passage = _passage_by_ref.get(req.ref)
    if passage is None:
        raise HTTPException(status_code=404, detail=f"No passage found for ref '{req.ref}'.")
    _enforce(_explain_limiter, request, "AI explanations")
    if _llm is None:
        raise HTTPException(
            status_code=503,
            detail="No LLM configured for explanations. Set NVIDIA_API_KEY or ANTHROPIC_API_KEY.",
        )
    if _tafseer_store is not None:
        passage = replace(passage, tafseer=_tafseer_store.get(passage.ref))
    try:
        explanation = explain_passage(passage, req.question, _llm, SOURCE_NAME)
    except HTTPError as e:
        # surface the provider's own message instead of a bare 500
        detail = e.read().decode("utf-8", "replace")[:400] if e.fp else str(e)
        raise HTTPException(status_code=502,
                            detail=f"Explanation provider returned {e.code}: {detail}")
    except (URLError, TimeoutError) as e:
        raise HTTPException(status_code=504,
                            detail=f"Could not reach the explanation provider: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Explanation failed: {e}")
    return {
        "ref": passage.ref,
        "verse": passage.text,
        "arabic": passage.arabic,
        "has_tafseer": bool(passage.tafseer),
        "explanation": explanation,
    }


_dist = ROOT / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="web")

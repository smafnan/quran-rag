"""FastAPI backend for the Quran RAG web UI.

Serves the grounded search over HTTP and the built React frontend (web/dist).
Run:  uvicorn api:app --reload  →  http://localhost:8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.groundedrag import Retriever, load_corpus

ROOT = Path(__file__).resolve().parent
SOURCE_NAME = "the Quran"

app = FastAPI(title="Quran RAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_passages = load_corpus(ROOT / "data" / "quran_sample.jsonl")
_retriever = Retriever(_passages)


class AskRequest(BaseModel):
    question: str
    k: int = 5


@app.get("/api/info")
def info():
    return {"source": SOURCE_NAME, "passages": len(_passages)}


@app.post("/api/ask")
def ask(req: AskRequest):
    hits = _retriever.search(req.question, top_k=req.k)
    return {
        "found": bool(hits),
        "source": SOURCE_NAME,
        "results": [
            {"ref": h.passage.ref, "text": h.passage.text, "score": round(h.score, 3)}
            for h in hits
        ],
    }


_dist = ROOT / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="web")

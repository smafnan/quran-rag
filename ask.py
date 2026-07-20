"""Ask a question answered ONLY from the Quran.

    # Return the relevant verses with citations (offline, no key):
    python ask.py "What does it say about hardship and ease?"

    # Compose a grounded prose answer with an LLM (cites every verse):
    python ask.py "..." --provider anthropic --api-key sk-ant-... --compose

By default it loads data/quran_sample.jsonl. Replace that file with the full
translation you trust (one JSON object per line: {"ref": "2:255", "text": "..."}).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.groundedrag import GroundedAnswerer, Retriever, get_llm, load_corpus

SOURCE_NAME = "the Quran"


def _default_corpus() -> str:
    """Prefer the full corpus (Arabic + tafseer) when it is present."""
    here = Path(__file__).resolve().parent
    full = here / "data" / "quran_full.jsonl"
    return str(full if full.exists() else here / "data" / "quran_sample.jsonl")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ask the Quran (grounded RAG).")
    p.add_argument("question")
    p.add_argument("--data", default=_default_corpus())
    p.add_argument("--provider", default="mock",
                   choices=["mock", "anthropic", "nvidia"])
    p.add_argument("--api-key", default=None)
    p.add_argument("--compose", action="store_true",
                   help="Compose a prose answer (needs a real provider).")
    p.add_argument("--all", action="store_true",
                   help="List every matching verse corpus-wide instead of the top k.")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args(argv)

    if args.all and args.compose:
        p.error("--all lists verses verbatim; it cannot be combined with --compose")

    passages = load_corpus(args.data)
    retriever = Retriever(passages)

    if args.all:
        hits = retriever.search(args.question, mode="all")
        if not hits:
            print(f"This topic does not appear to be addressed in {SOURCE_NAME}.")
            return 0
        print(f"{len(hits)} verses connected to '{args.question}':\n")
        for h in hits:
            if h.passage.arabic:
                print(f"[{h.passage.ref}] {h.passage.arabic}")
                print(f"    {h.passage.text}\n")
            else:
                print(f"[{h.passage.ref}] {h.passage.text}\n")
        return 0

    answerer = GroundedAnswerer(
        retriever, SOURCE_NAME,
        llm=get_llm(args.provider,
                    api_key=args.api_key or os.environ.get("ANTHROPIC_API_KEY")),
    )
    ans = answerer.ask(args.question, top_k=args.k, compose=args.compose)

    print(ans.text)
    if ans.found:
        print("\nSources:", ", ".join(p.ref for p in ans.citations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

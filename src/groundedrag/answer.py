"""The grounded answerer — answers ONLY from the source, with citations.

Two modes:
  * **passages** (default, no LLM): return the relevant passages verbatim with
    their citations. This is trivially faithful — it *is* the book's words.
  * **compose** (optional LLM): write a prose answer built strictly from those
    passages, under a grounding prompt that forbids outside knowledge and requires
    a citation for every statement.

If retrieval finds nothing above the relevance gate, the answerer declines —
"this topic is not addressed in <source>" — rather than inventing an answer. That
refusal is the core promise: it never speaks beyond the book.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Passage
from .providers import LLM
from .retriever import Retriever


@dataclass
class Answer:
    found: bool
    text: str
    citations: list[Passage] = field(default_factory=list)


def _grounding_system(source_name: str) -> str:
    return (
        f"You answer strictly and only from {source_name}. You are given numbered "
        f"passages from {source_name}. Compose an answer using ONLY the content of "
        f"those passages. Do not add outside knowledge, interpretation beyond the "
        f"text, or any fact not present in them. Cite the reference (e.g. [2:255]) "
        f"after each statement. If the passages do not address the question, say "
        f"so plainly."
    )


class GroundedAnswerer:
    def __init__(self, retriever: Retriever, source_name: str,
                 llm: LLM | None = None) -> None:
        self.retriever = retriever
        self.source_name = source_name
        self.llm = llm

    def ask(self, question: str, top_k: int = 5, compose: bool = False) -> Answer:
        hits = self.retriever.search(question, top_k=top_k)
        if not hits:
            return Answer(
                found=False,
                text=f"This topic does not appear to be addressed in "
                     f"{self.source_name}.",
            )
        passages = [h.passage for h in hits]

        if compose and self.llm is not None:
            numbered = "\n".join(f"[{p.ref}] {p.text}" for p in passages)
            body = self.llm.complete(
                _grounding_system(self.source_name),
                f"Question: {question}\n\nPassages from {self.source_name}:\n{numbered}",
            )
            return Answer(found=True, text=body, citations=passages)

        # Default: return the passages themselves with citations.
        text = "\n\n".join(f"[{p.ref}] {p.text}" for p in passages)
        return Answer(found=True, text=text, citations=passages)

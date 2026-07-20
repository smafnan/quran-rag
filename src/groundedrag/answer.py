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


TAFSEER_CHAR_BUDGET = 12000


def _explain_system(source_name: str) -> str:
    return (
        f"You are a knowledgeable guide explaining a single verse from {source_name} "
        f"using its accompanying tafseer (commentary and hadith). You are given the "
        f"verse's translation, its citation, the tafseer text, and the question the "
        f"reader was originally exploring. Write a clear, well-organized explanation "
        f"that:\n"
        f"- opens with the verse's core meaning in plain terms\n"
        f"- draws out the deeper meaning, context, and connections the tafseer offers\n"
        f"- calls out cross-references to other verses or hadith mentioned in the "
        f"tafseer, citing them like [2:255]\n"
        f"- relates the explanation back to what the reader was asking about, if given\n"
        f"Use ONLY the tafseer text provided — do not invent hadith, sources, or claims "
        f"not present in it. If the tafseer is thin or absent, say so honestly rather "
        f"than filling gaps from outside knowledge."
    )


def explain_passage(passage: Passage, question: str, llm: LLM, source_name: str) -> str:
    """Deep-dive explanation of a single passage using its tafseer, via LLM."""
    tafseer = passage.tafseer[:TAFSEER_CHAR_BUDGET]
    truncated_note = ("\n\n[tafseer truncated for length]"
                       if len(passage.tafseer) > TAFSEER_CHAR_BUDGET else "")
    user = (
        f"Verse [{passage.ref}]: {passage.text}\n\n"
        f"Tafseer:\n{tafseer or '(no tafseer available for this verse)'}{truncated_note}\n\n"
        f"Reader's original question: {question or '(none given)'}"
    )
    return llm.complete(_explain_system(source_name), user, max_tokens=1200)

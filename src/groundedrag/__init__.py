"""groundedrag - a RAG engine that answers ONLY from a single source text."""

from .corpus import Passage, load_corpus
from .retriever import Retriever, Hit
from .answer import GroundedAnswerer, Answer
from .providers import LLM, MockLLM, get_llm

__all__ = [
    "Passage", "load_corpus", "Retriever", "Hit",
    "GroundedAnswerer", "Answer", "LLM", "MockLLM", "get_llm",
]
__version__ = "1.0.0"

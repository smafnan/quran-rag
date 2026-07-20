"""groundedrag - a RAG engine that answers ONLY from a single source text."""

from .corpus import Passage, load_corpus
from .retriever import Retriever, Hit
from .answer import GroundedAnswerer, Answer, explain_passage
from .providers import LLM, MockLLM, AnthropicLLM, NvidiaLLM, NvidiaEmbedder, get_llm
from .ratelimit import RateLimiter, client_key
from .tafseer_store import TafseerStore
from .query_understanding import QueryUnderstanding, Interpretation, translit_key

__all__ = [
    "Passage", "load_corpus", "Retriever", "Hit", "TafseerStore",
    "GroundedAnswerer", "Answer", "explain_passage",
    "LLM", "MockLLM", "AnthropicLLM", "NvidiaLLM", "NvidiaEmbedder", "get_llm",
    "RateLimiter", "client_key",
    "QueryUnderstanding", "Interpretation", "translit_key",
]
__version__ = "1.0.0"

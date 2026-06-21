"""Optional LLM layer for composing a grounded answer from retrieved passages.

The system works WITHOUT an LLM: it returns the relevant passages verbatim with
their citations, which is by definition "only from the book". An LLM, if
provided, composes those passages into a prose answer under a strict grounding
prompt (no outside knowledge, cite every claim). MockLLM keeps it testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable


class LLM(ABC):
    name = "base"

    @abstractmethod
    def complete(self, system: str, user: str, *, max_tokens: int = 600) -> str:
        raise NotImplementedError


class MockLLM(LLM):
    name = "mock"

    def __init__(self, fn: Callable[[str, str], str] | None = None) -> None:
        self._fn = fn

    def complete(self, system: str, user: str, *, max_tokens: int = 600) -> str:
        return self._fn(system, user) if self._fn else user


class AnthropicLLM(LLM):  # pragma: no cover - needs network + key
    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-opus-4-8") -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, *, max_tokens: int = 600) -> str:
        resp = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


def get_llm(name: str, api_key: str | None = None, model: str | None = None) -> LLM:
    name = (name or "mock").lower()
    if name == "mock":
        return MockLLM()
    if name == "anthropic":
        return AnthropicLLM(api_key, model) if model else AnthropicLLM(api_key)
    raise ValueError(f"Unknown provider '{name}'")

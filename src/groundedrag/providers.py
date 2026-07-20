"""Optional LLM layer for composing a grounded answer from retrieved passages.

The system works WITHOUT an LLM: it returns the relevant passages verbatim with
their citations, which is by definition "only from the book". An LLM, if
provided, composes those passages into a prose answer under a strict grounding
prompt (no outside knowledge, cite every claim). MockLLM keeps it testable.
"""

from __future__ import annotations

import json
import ssl
import urllib.request
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


class NvidiaLLM(LLM):  # pragma: no cover - needs network + key
    """NVIDIA NIM chat-completions (OpenAI-compatible), stdlib HTTP only."""

    name = "nvidia"
    BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "nvidia/nemotron-3-super-120b-a12b") -> None:
        self._api_key = api_key
        self.model = model

    def complete(self, system: str, user: str, *, max_tokens: int = 600) -> str:
        # Nemotron reasoning models dump their chain-of-thought into the response
        # unless told not to; "detailed thinking off" keeps `content` clean and
        # routes any reasoning into a separate `reasoning_content` field instead.
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"detailed thinking off\n\n{system}"},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.4,
        }
        req = urllib.request.Request(
            self.BASE_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


def _ssl_context() -> ssl.SSLContext:
    # Some Python installs (notably python.org builds on macOS) don't wire up
    # the system CA store, so plain urlopen() fails cert verification. Use
    # certifi's bundle when available; fall back to the stdlib default.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class NvidiaEmbedder:  # pragma: no cover - needs network + key
    """Query-side embeddings via NVIDIA NIM (OpenAI-compatible), for the
    Retriever's semantic layer. The passage-side index is built offline by
    scripts/build_embeddings.py with the same model."""

    URL = "https://integrate.api.nvidia.com/v1/embeddings"

    def __init__(self, api_key: str, model: str = "nvidia/nemotron-3-embed-1b") -> None:
        self._api_key = api_key
        self.model = model

    def __call__(self, query: str) -> list[float]:
        payload = {
            "model": self.model,
            "input": [query],
            "input_type": "query",
            "encoding_format": "float",
            "truncate": "END",
        }
        req = urllib.request.Request(
            self.URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["data"][0]["embedding"]


def get_llm(name: str, api_key: str | None = None, model: str | None = None) -> LLM:
    name = (name or "mock").lower()
    if name == "mock":
        return MockLLM()
    if name not in ("anthropic", "nvidia"):
        raise ValueError(f"Unknown provider '{name}'")
    # fail loudly here rather than sending "Bearer None" upstream and having the
    # caller believe the provider is available
    if not api_key:
        raise ValueError(f"provider '{name}' requires an API key")
    cls = AnthropicLLM if name == "anthropic" else NvidiaLLM
    return cls(api_key, model) if model else cls(api_key)

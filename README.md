# Quran RAG — Answers Only From the Quran

> **Bonus project** (beyond the AI Engineer Roadmap).
> A retrieval system that answers questions **strictly and only from the Quran**,
> always returning **verse citations** — and declines when a topic isn't in the
> text rather than guessing.

```bash
python -m venv .venv && source .venv/bin/activate   # Win: .\.venv\Scripts\activate
pip install -e ".[dev]"

python ask.py "What does it say about hardship and ease?"
#   [94:5] So, surely with hardship comes ease.
#   [94:6] Surely with that hardship comes more ease.
#   Sources: 94:5, 94:6

# Compose a grounded prose answer (cites every verse) with an LLM:
pip install -e ".[anthropic]"
python ask.py "..." --provider anthropic --api-key sk-ant-... --compose

pytest -q   # 7 tests
```

## How it stays faithful to the source

- **Default (no LLM)**: returns the relevant **verses verbatim** with their
  references — by definition only the Quran's words.
- **Compose (optional LLM)**: writes a prose answer built strictly from the
  retrieved verses, under a grounding prompt that forbids outside knowledge and
  requires a `[chapter:verse]` citation per statement.
- **Relevance gate**: if no verse clears the similarity threshold, it says the
  topic isn't addressed — it never speaks beyond the book. (A test asserts the
  answer's text comes verbatim from the corpus, and another asserts off-topic
  questions are declined.)

## Add the full text

The repo ships a **small sample** (`data/quran_sample.jsonl`) so it runs out of
the box. Replace it with the full translation you trust — one JSON object per line:

```json
{"ref": "2:255", "text": "Allah! There is no god but He, the Living ..."}
```

Point `ask.py --data` at your file, or overwrite the sample. No code changes
needed; the system indexes whatever you provide.

## Layout

```
src/groundedrag/
├── corpus.py      # Passage + JSONL loader
├── retriever.py   # TF-IDF cosine retrieval + relevance gate
├── answer.py      # grounded answerer (passages mode / LLM-compose mode)
└── providers.py   # MockLLM + Anthropic
ask.py             # CLI
data/quran_sample.jsonl   # small placeholder — replace with full text
tests/             # 7 tests incl. grounding guarantees
```

## Note on the text

Translations of the Quran vary; use the translation you and your community trust.
The included sample uses widely-circulated public-domain English renderings purely
as placeholder data to demonstrate the system. This is a study/search aid — it
surfaces and cites verses; it is not a substitute for scholarship.

## License

MIT (code). The Quran text you supply is governed by its own translation's terms.

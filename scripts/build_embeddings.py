"""Build a semantic-search index: embed every passage with NVIDIA NIM embeddings.

Embeds the English translation text of each corpus row (queries arrive in
English) and stores an .npz with the refs and a float16 matrix, loaded by the
Retriever at startup for hybrid search.

Usage:
    NVIDIA_EMBED_API_KEY=nvapi-... python scripts/build_embeddings.py \
        data/quran_full.jsonl data/embeddings.npz

Falls back to NVIDIA_API_KEY if NVIDIA_EMBED_API_KEY is unset.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from groundedrag.providers import _ssl_context  # noqa: E402

URL = "https://integrate.api.nvidia.com/v1/embeddings"
MODEL = os.environ.get("EMBED_MODEL", "nvidia/nemotron-3-embed-1b")
BATCH = 32
RETRIES = 4


def embed_batch(texts: list[str], api_key: str, input_type: str) -> list[list[float]]:
    payload = {
        "model": MODEL,
        "input": texts,
        "input_type": input_type,
        "encoding_format": "float",
        "truncate": "END",
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    last_err: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            rows = sorted(body["data"], key=lambda d: d["index"])
            return [r["embedding"] for r in rows]
        except Exception as e:  # noqa: BLE001 - retry then surface
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"embedding batch failed after {RETRIES} tries: {last_err}")


def main() -> int:
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "data/quran_full.jsonl"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "data/embeddings.npz"
    api_key = os.environ.get("NVIDIA_EMBED_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("set NVIDIA_EMBED_API_KEY (or NVIDIA_API_KEY)", file=sys.stderr)
        return 1

    refs, texts = [], []
    for line in Path(corpus_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        refs.append(row["ref"])
        texts.append(row["text"])

    vectors: list[list[float]] = []
    total_batches = (len(texts) + BATCH - 1) // BATCH
    for b in range(total_batches):
        chunk = texts[b * BATCH:(b + 1) * BATCH]
        vectors.extend(embed_batch(chunk, api_key, "passage"))
        print(f"[{b + 1}/{total_batches}] embedded {len(vectors)}/{len(texts)}",
              file=sys.stderr)

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms

    np.savez_compressed(out_path, refs=np.array(refs), matrix=matrix.astype(np.float16))
    print(f"wrote {out_path}: {matrix.shape[0]} x {matrix.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

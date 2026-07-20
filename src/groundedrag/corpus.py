"""The source corpus: a list of cited passages.

Each passage is a unit of the book with a citation reference (e.g. a Quran verse
"2:255" or a Nahjul Balagha "Sermon 1"). The corpus is stored as JSON Lines so
it is easy to extend with the full text later — one passage per line:

    {"ref": "2:255", "text": "Allah - there is no deity except Him ..."}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Passage:
    ref: str          # citation reference (verse, sermon, saying number...)
    text: str
    tafseer: str = ""  # optional commentary/exegesis trailing this passage, if the corpus has it
    arabic: str = ""   # optional original-language text of the passage


def load_corpus(path: str | Path) -> list[Passage]:
    """Load passages from a JSON Lines file."""
    passages: list[Passage] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        passages.append(Passage(ref=str(row["ref"]), text=str(row["text"]),
                                 tafseer=str(row.get("tafseer", "")),
                                 arabic=str(row.get("arabic", ""))))
    return passages

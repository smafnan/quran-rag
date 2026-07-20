"""Merge canonical Arabic text into the corpus JSONL, keyed by chapter:verse.

Source: alquran.cloud Uthmani-script JSON (all 6236 ayahs, standard Kufan
numbering). That source prepends the Basmalah to the first ayah of every
surah; it is stripped everywhere except 1:1 (where the Basmalah *is* the
verse — surah 9 has no Basmalah in the source to begin with).

Usage:
    python scripts/merge_arabic.py <arabic_quran.json> <corpus.jsonl> [out.jsonl]

If out.jsonl is omitted the corpus file is rewritten in place.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

def load_arabic(path: str | Path) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    surahs = data["data"]["surahs"]
    # the Basmalah exactly as this source spells it = surah 1, ayah 1
    basmalah = surahs[0]["ayahs"][0]["text"].lstrip("﻿").strip()
    arabic: dict[str, str] = {}
    for surah in surahs:
        num = surah["number"]
        for ayah in surah["ayahs"]:
            text = ayah["text"].lstrip("﻿").strip()
            if ayah["numberInSurah"] == 1 and num != 1 and text.startswith(basmalah):
                text = text[len(basmalah):].strip()
            arabic[f"{num}:{ayah['numberInSurah']}"] = text
    return arabic


def main() -> int:
    arabic_path, corpus_path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else corpus_path

    arabic = load_arabic(arabic_path)
    rows = []
    missing = []
    for line in Path(corpus_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ar = arabic.get(row["ref"], "")
        if not ar:
            missing.append(row["ref"])
        row["arabic"] = ar
        rows.append(row)

    # write to a temp file in the same directory, then atomically replace, so a
    # crash mid-write can never leave the corpus truncated
    out = Path(out_path)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, out)
    except BaseException:
        os.unlink(tmp)
        raise

    print(f"merged arabic into {len(rows)} rows -> {out_path}")
    print(f"rows without arabic match: {len(missing)}")
    if missing:
        print("  refs:", missing[:30], "..." if len(missing) > 30 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

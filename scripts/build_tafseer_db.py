"""Extract tafseer from the corpus JSONL into a read-only SQLite database.

Keeps ~54 MB of commentary out of the server's resident memory; it is read from
disk only when a reader opens a verse.

Usage:
    python scripts/build_tafseer_db.py data/quran_full.jsonl data/tafseer.sqlite3
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/quran_full.jsonl")
    dest = Path(sys.argv[2] if len(sys.argv) > 2 else "data/tafseer.sqlite3")

    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append((row["ref"], row.get("tafseer", "")))

    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".sqlite3.tmp")
    os.close(fd)
    try:
        con = sqlite3.connect(tmp)
        con.execute("PRAGMA journal_mode = OFF")
        con.execute("CREATE TABLE tafseer (ref TEXT PRIMARY KEY, tafseer TEXT NOT NULL)")
        con.executemany("INSERT INTO tafseer (ref, tafseer) VALUES (?, ?)", rows)
        con.commit()
        con.execute("VACUUM")          # compact before shipping in the image
        con.close()
        os.replace(tmp, dest)          # atomic: never leave a half-written db
    except BaseException:
        os.unlink(tmp)
        raise

    with_text = sum(1 for _, t in rows if t)
    size_mb = dest.stat().st_size / 1e6
    print(f"wrote {dest}: {len(rows)} rows ({with_text} with tafseer), {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

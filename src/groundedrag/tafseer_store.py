"""Read-only SQLite store for tafseer text.

The tafseer is 94% of the corpus by bytes but is only read when someone clicks a
verse. Holding it in RAM costs ~54 MB for data that most requests never touch —
a poor trade on a 512 MB instance. SQLite gives indexed lazy reads from disk at
no infrastructural cost.

Why SQLite and not a managed database: the data is read-only reference material
that ships with the image, there is one instance and no writes, and a network
round trip per click would be slower than a local page read. A hosted Postgres
would add a dependency, a credential and a failure mode to buy nothing — and on
free tiers it typically expires after a trial period, which would take the
feature down without warning.

Connections are per-thread because FastAPI runs sync endpoints in a threadpool
and SQLite connections are not shareable across threads.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class TafseerStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._local = threading.local()
        # cheap membership set so /api/ask can flag has_tafseer without a query
        # per result; refs are short, so this is a few hundred KB at most
        with sqlite3.connect(self.db_path) as con:
            self._refs_with_text = {
                r[0] for r in con.execute(
                    "SELECT ref FROM tafseer WHERE length(tafseer) > 0")
            }

    @property
    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.db_path, check_same_thread=False)
            con.execute("PRAGMA query_only = ON")
            self._local.con = con
        return con

    def has(self, ref: str) -> bool:
        return ref in self._refs_with_text

    def get(self, ref: str) -> str:
        row = self._con.execute(
            "SELECT tafseer FROM tafseer WHERE ref = ?", (ref,)).fetchone()
        return row[0] if row else ""

    def __len__(self) -> int:
        return len(self._refs_with_text)

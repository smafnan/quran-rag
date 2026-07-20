"""Audit that each corpus row's English translation matches its verse ref.

Compares the corpus text against an independent reference translation
(alquran.cloud en.sahih) using content-word overlap, and reports refs where the
corpus text looks like a *neighbouring* verse instead — the signature of an
off-by-one in the source PDFs' verse numbering.

A spurious duplicate row in the source PDFs shifts every later verse in that
chapter down by one, so misalignments come in runs rather than singly.

Usage:
    python scripts/audit_alignment.py <ref_en.json> <corpus.jsonl> [--realign out.jsonl]

Without --realign it only reports. With --realign each row is re-keyed to
whichever nearby ref its text actually matches (collisions resolved in favour of
the better match, losers dropped) — recovering shifted runs instead of
discarding them, while guaranteeing no row keeps a ref its text contradicts.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

STOP = {
    "the", "and", "of", "to", "a", "in", "is", "it", "you", "that", "he", "was",
    "for", "on", "are", "with", "as", "his", "they", "be", "at", "have", "this",
    "from", "or", "had", "by", "but", "not", "what", "all", "were", "we", "when",
    "your", "there", "so", "if", "who", "them", "then", "would", "will", "their",
    "him", "has", "our", "said", "which", "do", "did", "no", "an", "my", "us",
}
_W = re.compile(r"[a-z]+")


def bag(text: str) -> set[str]:
    return {w for w in _W.findall(text.lower()) if len(w) > 2 and w not in STOP}


def sim(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


MARGIN = 0.15  # a neighbour must beat the row's own ref by this to win it


def main() -> int:
    ref_path, corpus_path = sys.argv[1], sys.argv[2]
    fix_to = None
    if "--realign" in sys.argv:
        fix_to = sys.argv[sys.argv.index("--realign") + 1]

    data = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    ref = {f"{s['number']}:{a['numberInSurah']}": a["text"]
           for s in data["data"]["surahs"] for a in s["ayahs"]}
    ref_bags = {k: bag(v) for k, v in ref.items()}

    rows = [json.loads(l) for l in Path(corpus_path).read_text(encoding="utf-8").splitlines() if l.strip()]

    # claim: each row proposes the ref its text actually matches best
    claims = []          # (target_ref, score, row, moved)
    misaligned, unmatched = [], 0
    for row in rows:
        r = row["ref"]
        c, v = (int(x) for x in r.split(":"))
        b = bag(row["text"])
        own = sim(b, ref_bags.get(r, set()))
        target, best = r, own
        for off in (-2, -1, 1, 2):
            nb = f"{c}:{v + off}"
            if nb in ref_bags:
                s = sim(b, ref_bags[nb])
                if s > best + MARGIN:
                    target, best = nb, s
        if target != r:
            misaligned.append((r, target, round(own, 2), round(best, 2)))
        elif own < 0.15:
            unmatched += 1
        claims.append((target, best, row, target != r))

    print(f"rows: {len(rows)}")
    print(f"misaligned (text matches a nearby ref better): {len(misaligned)}")
    print(f"weak match to own ref but no better neighbour: {unmatched}")
    from collections import Counter
    by_chapter = Counter(int(r.split(':')[0]) for r, *_ in misaligned)
    if by_chapter:
        print("worst chapters:", by_chapter.most_common(12))
        print("examples:", misaligned[:8])

    if fix_to:
        # resolve collisions: best-scoring claimant keeps the ref, others drop
        winner: dict[str, tuple[float, dict]] = {}
        for target, score, row, moved in claims:
            cur = winner.get(target)
            if cur is None or score > cur[0]:
                winner[target] = (score, row)

        def key(r: str) -> tuple[int, int]:
            c, v = r.split(":")
            return int(c), int(v)

        out_rows = []
        for target in sorted(winner, key=key):
            row = dict(winner[target][1])
            row["ref"] = target          # re-key; arabic is re-merged separately
            out_rows.append(row)

        with Path(fix_to).open("w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        moved = sum(1 for _, _, _, m in claims if m)
        print(f"wrote {len(out_rows)} rows -> {fix_to} "
              f"({moved} re-keyed, {len(rows) - len(out_rows)} dropped as duplicates)")
        print("NOTE: refs changed — re-run merge_arabic.py and build_embeddings.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

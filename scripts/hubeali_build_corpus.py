"""Rebuild the full corpus (translation + tafseer) from the Tafseer Hub-e-Ali PDFs.

Prerequisite: download the chapter PDFs listed in hubeali_urls.txt into
scripts/pdfs/ first, e.g.:

    while IFS='|' read -r chap fname; do
      curl -sL -o "scripts/pdfs/${fname}.pdf" \
        "https://hubeali.com/books/English-Books/TafseerHub-e-Ali/${fname}.pdf"
    done < scripts/hubeali_urls.txt

Then:  python scripts/hubeali_build_corpus.py
(afterwards merge Arabic with scripts/merge_arabic.py and rebuild embeddings)
"""

import json
import re
import sys
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hubeali_parser import parse_chapter  # noqa: E402

HERE = Path(__file__).parent
PDF_DIR = HERE / 'pdfs'
URLS_FILE = HERE / 'hubeali_urls.txt'
OUT_JSONL = HERE / 'quran_hubeali_tafseer.jsonl'
REPORT = HERE / 'build_report.txt'


def load_entries():
    entries = []
    for line in URLS_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        chap, fname = line.split('|')
        entries.append((int(chap), fname))
    return entries


def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return '\n'.join((p.extract_text() or '') for p in pdf.pages)


def main():
    entries = load_entries()
    all_results = {}
    all_tafseer = {}
    report_lines = []
    for i, (chap, fname) in enumerate(entries, 1):
        pdf_path = PDF_DIR / f'{fname}.pdf'
        if not pdf_path.exists():
            report_lines.append(f'MISSING PDF: {fname}')
            print(f'[{i}/{len(entries)}] MISSING {fname}', file=sys.stderr)
            continue
        try:
            full_text = extract_text(pdf_path)
        except Exception as e:
            report_lines.append(f'EXTRACT ERROR: {fname}: {e}')
            print(f'[{i}/{len(entries)}] EXTRACT ERROR {fname}: {e}', file=sys.stderr)
            continue
        try:
            results, tafseer, debug = parse_chapter(full_text, chap)
        except Exception as e:
            report_lines.append(f'PARSE ERROR: {fname}: {e}')
            print(f'[{i}/{len(entries)}] PARSE ERROR {fname}: {e}', file=sys.stderr)
            continue

        new_count = 0
        for ref, text in results.items():
            if ref not in all_results:
                all_results[ref] = text
                all_tafseer[ref] = tafseer.get(ref, '')
                new_count += 1
        report_lines.append(f'{fname}: chapter={chap} parsed={len(results)} new={new_count}')
        print(f'[{i}/{len(entries)}] {fname}: parsed={len(results)} new={new_count}', file=sys.stderr)

    def sort_key(ref):
        c, v = ref.split(':')
        return (int(c), int(v))

    with OUT_JSONL.open('w', encoding='utf-8') as f:
        for ref in sorted(all_results, key=sort_key):
            row = {'ref': ref, 'text': all_results[ref], 'tafseer': all_tafseer.get(ref, '')}
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    # per-chapter counts
    from collections import Counter
    chap_counts = Counter(int(r.split(':')[0]) for r in all_results)
    report_lines.append('')
    report_lines.append(f'TOTAL VERSES: {len(all_results)}')
    report_lines.append('')
    report_lines.append('Per-chapter verse counts:')
    for c in range(1, 115):
        report_lines.append(f'  chapter {c}: {chap_counts.get(c, 0)}')

    REPORT.write_text('\n'.join(report_lines), encoding='utf-8')
    print(f'\nTOTAL VERSES: {len(all_results)}', file=sys.stderr)
    print(f'Wrote {OUT_JSONL}', file=sys.stderr)
    print(f'Report: {REPORT}', file=sys.stderr)


if __name__ == '__main__':
    main()

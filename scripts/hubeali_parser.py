import re
import sys

HEADER_RE = re.compile(r'\n(VERSES?\s+[\dA-Za-z&\-–, ]+?)\n')
CITE_RE = re.compile(r'\[(\d+):(\d+)\]')
FOOTNOTE_RE = re.compile(r'(?<=[A-Za-z])\d{1,3}(?=[\s.,;:!?\'‘’")\]]|$)')
LATIN_LINE_RE = re.compile(r'[A-Za-z]{2,}')

PAGE_HEADER_RE = re.compile(r'^Tafseer Hub-e-Ali')
PAGE_FOOTER_RE = re.compile(r'^\d+\s+out\s+of\s+\d+$')
FOOTNOTE_LINE_RE = re.compile(r'^\d{1,3}\s+[A-Z]')
REPEATED_ANNOTATION_RE = re.compile(r'^\(REPEATED', re.IGNORECASE)


def is_boilerplate(line):
    s = line.strip()
    if not s:
        return True
    if PAGE_HEADER_RE.match(s):
        return True
    if PAGE_FOOTER_RE.match(s):
        return True
    if FOOTNOTE_LINE_RE.match(s):
        return True
    if REPEATED_ANNOTATION_RE.match(s):
        return True
    return False


def strip_boilerplate(full_text):
    return '\n'.join(l for l in full_text.split('\n') if not is_boilerplate(l))


def clean_block(block):
    lines = [l for l in block.split('\n') if LATIN_LINE_RE.search(l)]
    text = ' '.join(lines).strip()
    text = re.sub(r'\[\d+:\d+\]', '', text).strip()
    text = FOOTNOTE_RE.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = text.strip(' .')
    return text


def clean_tafseer(block):
    # unlike clean_block (pure verse text), tafseer keeps citation brackets -
    # cross-references to other verses/hadith are exactly the "connections"
    # this field exists to surface.
    lines = [l for l in block.split('\n') if LATIN_LINE_RE.search(l)]
    text = ' '.join(lines).strip()
    text = FOOTNOTE_RE.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = text.strip(' .')
    return text


def parse_chapter(full_text, chapter_num):
    full_text = strip_boilerplate(full_text)
    all_headers = [m for m in HEADER_RE.finditer(full_text)]
    # if the same header text (e.g. "VERSES 1 - 4") appears more than once, only the
    # LAST occurrence is the real content-start header; earlier ones are title/overview
    # mentions of the same verse range with no per-verse translation immediately following.
    last_by_text = {}
    for h in all_headers:
        last_by_text[h.group(1).strip()] = h
    candidates = sorted(last_by_text.values(), key=lambda h: h.start())

    # a real content header's first citation always matches its own declared starting
    # verse number (within +/-1 - the source occasionally mislabels a header's start
    # by one, e.g. "VERSES 16 - 21" whose content actually opens on verse 15). An
    # "overview" header spanning a wide range (e.g. "VERSES 1 - 78") that just happens
    # to appear once, ahead of extensive front-matter, fails even that loose check
    # because the first citation encountered belongs to whatever verse the front-matter
    # discusses first, not the header's declared start.
    headers = []
    declared_counts = {}
    for idx, h in enumerate(candidates):
        nums = [int(n) for n in re.findall(r'\d+', h.group(1))]
        if not nums:
            continue
        declared_start, declared_end = min(nums), max(nums)
        search_end = candidates[idx + 1].start() if idx + 1 < len(candidates) else len(full_text)
        c = CITE_RE.search(full_text, h.end(), search_end)
        if c and int(c.group(1)) == chapter_num and abs(int(c.group(2)) - declared_start) <= 1:
            actual_start = min(declared_start, int(c.group(2)))
            headers.append(h)
            declared_counts[h.start()] = declared_end - actual_start + 1

    boundaries = [h.start() for h in headers[1:]] + [len(full_text)]
    # boundaries[i] is the end (exclusive) of header i's section
    results = {}
    tafseer = {}
    debug = []
    for i, h in enumerate(headers):
        section_start = h.end()
        section_end = boundaries[i]
        section = full_text[section_start:section_end]

        declared_count = declared_counts.get(h.start())

        pos = 0
        last_cite_end = 0
        last_matched_end = 0
        expected = None
        got = []
        while True:
            if declared_count is not None and len(got) >= declared_count:
                # this header's declared verse range is fully captured; anything
                # else in the section is trailing commentary that may happen to
                # cite the next verse early - don't let it steal that verse's slot.
                break
            m = CITE_RE.search(section, pos)
            if not m:
                break
            c_chap, c_verse = int(m.group(1)), int(m.group(2))
            # every citation (matching or not) is a natural sentence boundary; anchor
            # this candidate's block on whatever citation immediately preceded it, so
            # unrelated cross-references or out-of-sequence repeats in between don't
            # get swept into the extracted text.
            block_start = last_cite_end
            last_cite_end = m.end()
            pos = m.end()

            if c_chap != chapter_num:
                # cross-reference to another chapter inside a hadith quote; not a
                # signal that the translation zone has ended, just noise to skip.
                continue
            if expected is None:
                expected = c_verse
            if c_verse != expected:
                # an already-seen or out-of-order verse number for this chapter -
                # a hadith re-quoting an earlier verse. Skip, keep scanning; the
                # section boundary (next header) is what actually ends the scan.
                continue

            block = section[block_start:m.start()]
            text = clean_block(block)
            match_end = m.end()

            if not text:
                nxt = CITE_RE.search(section, m.end())
                cap = m.end() + 400
                fwd_end = min(nxt.start(), cap) if nxt else cap
                fwd_block = section[m.end():fwd_end]
                fwd_lines = fwd_block.split('\n')
                keep = []
                for l in fwd_lines:
                    if LATIN_LINE_RE.search(l) and not is_boilerplate(l):
                        keep.append(l)
                    else:
                        break
                text = clean_block('\n'.join(keep))
                if keep:
                    match_end = m.end() + sum(len(l) + 1 for l in keep)

            ref = f'{c_chap}:{c_verse}'
            if ref not in results and text:
                results[ref] = text
                got.append(ref)
                last_matched_end = match_end

            expected += 1

        # everything after the last matched verse's translation, up to the next
        # header, is this section's tafseer (commentary) - shared across all
        # verses in the group, since the source doesn't separate it further.
        if got:
            tafseer_text = clean_tafseer(section[last_matched_end:])
            for ref in got:
                tafseer[ref] = tafseer_text

        debug.append((h.group(1).strip(), got))
    return results, tafseer, debug


if __name__ == '__main__':
    path = sys.argv[1]
    chapter_num = int(sys.argv[2])
    full = open(path, encoding='utf-8').read()
    results, tafseer, debug = parse_chapter(full, chapter_num)
    for header, got in debug:
        print(f'{header}: {got}')
    print()
    print(f'Total verses parsed: {len(results)}')
    for ref in sorted(results, key=lambda r: int(r.split(":")[1])):
        print(ref, '->', results[ref])
        print('   tafseer:', tafseer.get(ref, '')[:200])

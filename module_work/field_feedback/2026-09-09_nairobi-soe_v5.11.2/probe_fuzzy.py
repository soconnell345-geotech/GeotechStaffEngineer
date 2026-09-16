"""Measure the fuzzy-search threshold on a REAL submittal's drawing callouts.

planlens' house rule: measure on real documents before choosing a threshold.
The question this answers is the only one that matters for `Document.search
(fuzzy=True)`:

    At what rapidfuzz partial-ratio score does a corrupted query still find
    the line it came from, WITHOUT dragging in unrelated lines?

Method
------
1. Take N real callout strings off this submittal's drawing sheets — both the
   PDF text layer and the hidden AutoCAD SHX strings, which is where letter
   errors actually come from.
2. Corrupt each three ways: one substituted letter, one dropped letter, one
   transposition of two adjacent letters. That is the error shape an optical
   read or a re-typed callout produces.
3. Score EVERY search candidate in the whole document against every corrupted
   query once, at cutoff 0, exactly the way `Document._search_fuzzy` scores
   them (partial_ratio over the same groups, lowercased). Every threshold is
   then evaluated off that one table, so the sweep costs one pass per query
   rather than one per query per threshold.
4. Also score M words that are NOT in the document, to see where a threshold
   starts inventing hits.

Nothing about this document leaves this folder: the report prints counts and
scores, never the strings themselves. Run it from the repo root with the
GeotechStaffEngineer venv.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Tuple

PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "files",
                   "315000-001-00 Excavation Support Dwg & Calcs.pdf")

N_CALLOUTS = 15
THRESHOLDS = [60, 65, 70, 75, 80, 82, 85, 88, 90, 92, 95]

#: Words no geotechnical excavation-support submittal contains. Checked
#: against the document before use; any that turns out to be present is
#: dropped and reported, because a "negative" that is really there measures
#: nothing.
NEGATIVES = [
    "PERMAFROST", "MYCORRHIZAL", "ZEPPELIN", "CHLOROPHYLL", "TRILOBITE",
    "SAXOPHONE", "PTEROSAUR", "BUTTERMILK", "KALEIDOSCOPE", "HARPSICHORD",
]


def corrupt(word: str, rng: random.Random) -> Dict[str, str]:
    """One substituted letter, one dropped letter, one transposition."""
    idx = [i for i, c in enumerate(word) if c.isalpha()]
    out: Dict[str, str] = {}

    i = idx[len(idx) // 2]
    c = word[i]
    repl = "B" if c.upper() != "B" else "R"
    out["substitute"] = word[:i] + (repl if c.isupper() else repl.lower()) + word[i + 1:]

    j = idx[len(idx) // 3]
    out["drop"] = word[:j] + word[j + 1:]

    k = next(p for p in idx[::-1] if p + 1 < len(word) and word[p + 1].isalpha())
    out["transpose"] = word[:k] + word[k + 1] + word[k] + word[k + 2:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=PDF)
    args = ap.parse_args()

    from rapidfuzz import fuzz
    from planlens.document import open_document
    from planlens.document.model import SOURCE_CAD_HIDDEN

    def score(query: str, hay: str) -> float:
        """What `Document._search_fuzzy` computes: the query's best place in
        the candidate, with a candidate SHORTER than the query scored whole.

        Without that second clause rapidfuzz slides the shorter string — the
        candidate — over the query, and every one-character line on a drawing
        sheet scores 100 against anything containing that character.
        """
        if len(hay) < len(query):
            return fuzz.ratio(query, hay)
        return fuzz.partial_ratio(query, hay)

    rng = random.Random(20260916)
    t0 = time.time()
    doc = open_document(args.pdf)
    sheets = [s.page for s in doc.page_map() if s.kind == "drawing_sheet"]
    print(f"{doc.n_pages} pages, {len(sheets)} drawing sheets, "
          f"page map {time.time() - t0:.1f}s")

    # -- every search candidate in the document, scored once per query -------
    t0 = time.time()
    cand: List[Tuple[int, Tuple[str, ...], str, str]] = []   # page, ids, text, source
    for index in range(doc.n_pages):
        groups, _markups = doc._search_groups(index)
        for g in groups:
            text, _spans = doc._join_group(g)
            if text.strip():
                cand.append((index, tuple(ln.id for ln in g), text,
                             g[0].source))
    print(f"{len(cand)} search candidates, gathered in {time.time() - t0:.1f}s")
    lowered = [c[2].lower() for c in cand]

    # -- 15 real callouts off the drawing sheets -----------------------------
    pool: List[Tuple[int, str, str, str]] = []      # page, line id, text, source
    seen = set()
    for index in sheets:
        groups, _ = doc._search_groups(index)
        for g in groups:
            if len(g) != 1:
                continue
            ln = g[0]
            t = " ".join(ln.text.split())
            letters = sum(c.isalpha() for c in t)
            if not (10 <= len(t) <= 45) or letters < 8:
                continue
            if t.lower() in seen:
                continue
            seen.add(t.lower())
            pool.append((index, ln.id, t, ln.source))
    pool.sort(key=lambda r: (r[0], r[1]))
    step = max(1, len(pool) // N_CALLOUTS)
    picked = pool[::step][:N_CALLOUTS]
    n_cad = sum(1 for p in picked if p[3] == SOURCE_CAD_HIDDEN)
    print(f"{len(pool)} candidate callouts on the sheets; sampled "
          f"{len(picked)} ({n_cad} hidden CAD text, {len(picked) - n_cad} "
          f"text layer); lengths "
          f"{min(len(p[2]) for p in picked)}-{max(len(p[2]) for p in picked)} "
          f"chars, median {statistics.median(len(p[2]) for p in picked):.0f}")

    # -- score every corrupted query against every candidate -----------------
    rows = []          # (kind, source_score, [scores of unrelated candidates])
    t0 = time.time()
    for page, line_id, text, source in picked:
        for kind, query in corrupt(text, rng).items():
            q = query.lower()
            src_score, others = 0.0, []
            for (cp, ids, _t, _s), low in zip(cand, lowered):
                sc = score(q, low)
                if cp == page and line_id in ids:
                    src_score = max(src_score, sc)
                elif sc > 0:
                    others.append(sc)
            rows.append((kind, len(text), src_score, others))
    print(f"{len(rows)} corrupted queries scored in {time.time() - t0:.1f}s")

    neg_rows = []
    kept_negatives = []
    for word in NEGATIVES:
        if doc.search(word, max_hits=1)["n_hits"]:
            print(f"  (negative {word!r} IS in the document — dropped)")
            continue
        kept_negatives.append(word)
        q = word.lower()
        neg_rows.append([score(q, low) for low in lowered])
    print(f"{len(kept_negatives)} negatives confirmed absent")

    # -- the table -----------------------------------------------------------
    print()
    print("threshold | corrupted queries whose SOURCE line is found "
          "| median / max unrelated hits per query | negatives with any hit "
          "| median / max hits for a negative")
    for thr in THRESHOLDS:
        found = sum(1 for _k, _n, src, _o in rows if src >= thr)
        unrel = [sum(1 for s in others if s >= thr) for _k, _n, _src, others in rows]
        neg_hits = [sum(1 for s in scores if s >= thr) for scores in neg_rows]
        print(f"{thr:>9} | {found:>2}/{len(rows)} "
              f"| {statistics.median(unrel):.0f} / {max(unrel)} "
              f"| {sum(1 for n in neg_hits if n)}/{len(neg_rows)} "
              f"| {statistics.median(neg_hits):.0f} / {max(neg_hits)}")

    # -- the harder case: a reviewer types ONE WORD ---------------------------
    # A whole callout line is 10-45 characters and absorbs a letter error
    # easily. The query a reviewer actually types is often a single word, and
    # one wrong letter in a short word costs far more of the ratio -- this is
    # what actually sets the floor.
    words: List[Tuple[int, str, str]] = []
    wseen = set()
    for page, line_id, text, _src in pool:
        for w in text.replace('"', " ").split():
            w = "".join(c for c in w if c.isalnum())
            if 5 <= len(w) <= 10 and sum(c.isalpha() for c in w) >= 5 \
                    and w.lower() not in wseen:
                wseen.add(w.lower())
                words.append((page, line_id, w))
    words.sort(key=lambda r: (r[0], r[2]))
    wstep = max(1, len(words) // N_CALLOUTS)
    wpicked = words[::wstep][:N_CALLOUTS]
    wrows = []
    for page, line_id, w in wpicked:
        for kind, query in corrupt(w, rng).items():
            q = query.lower()
            src_score, others = 0.0, []
            for (cp, ids, _t, _s), low in zip(cand, lowered):
                sc = score(q, low)
                if cp == page and line_id in ids:
                    src_score = max(src_score, sc)
                elif sc > 0:
                    others.append(sc)
            wrows.append((kind, len(w), src_score, others))
    print()
    print(f"SINGLE WORDS ({len(wpicked)} words, "
          f"{min(len(w) for _p, _l, w in wpicked)}-"
          f"{max(len(w) for _p, _l, w in wpicked)} letters, "
          f"{len(wrows)} corrupted queries)")
    print("threshold | source line found | median / max unrelated hits")
    for thr in THRESHOLDS:
        found = sum(1 for _k, _n, src, _o in wrows if src >= thr)
        unrel = [sum(1 for s in o if s >= thr) for _k, _n, _src, o in wrows]
        print(f"{thr:>9} | {found:>2}/{len(wrows)} "
              f"| {statistics.median(unrel):.0f} / {max(unrel)}")
    for kind in ("substitute", "drop", "transpose"):
        s = [src for k, _n, src, _o in wrows if k == kind]
        print(f"  {kind:<11} min {min(s):.1f}  median {statistics.median(s):.1f}")

    # -- does the junk BURY the true hit? ------------------------------------
    # Hits come back ordered by score, so unrelated hits below the source line
    # cost tokens, not the answer. What would cost the answer is the source
    # line ranking behind them.
    print()
    print("rank of the source line among the hits (hits are score-ordered)")
    print("threshold | callout queries rank 1 / top 5 | word queries rank 1 / top 5")
    for thr in (70, 75, 80, 85, 88):
        def ranks(rs):
            out = []
            for _k, _n, src, others in rs:
                if src < thr:
                    continue
                out.append(1 + sum(1 for s in others if s > src))
            return out
        a, b = ranks(rows), ranks(wrows)
        print(f"{thr:>9} | {sum(1 for r in a if r == 1):>2}/{len(a)} "
              f"{sum(1 for r in a if r <= 5):>2}/{len(a)} "
              f"| {sum(1 for r in b if r == 1):>2}/{len(b)} "
              f"{sum(1 for r in b if r <= 5):>2}/{len(b)}")

    print()
    print("by corruption kind: lowest source score seen")
    for kind in ("substitute", "drop", "transpose"):
        s = [src for k, _n, src, _o in rows if k == kind]
        print(f"  {kind:<11} min {min(s):.1f}  median {statistics.median(s):.1f}"
              f"  max {max(s):.1f}")
    print()
    print("source score vs query length (the short-query problem)")
    for lo, hi in ((10, 19), (20, 29), (30, 45)):
        s = [src for _k, n, src, _o in rows if lo <= n <= hi]
        if s:
            print(f"  {lo}-{hi} chars: {len(s)} queries, min {min(s):.1f}, "
                  f"median {statistics.median(s):.1f}")
    doc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

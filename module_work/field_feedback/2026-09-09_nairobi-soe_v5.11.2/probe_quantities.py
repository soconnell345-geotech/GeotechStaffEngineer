"""Bake-off: planlens' native quantity regex vs quantulum3, on two corpora.

The question: does a general-purpose quantity extractor recover mentions the
native regex misses, WITHOUT adding junk? If it does, it becomes an optional
second pass; if it does not, it stays out of the dependency list and the
measured reason goes in DESIGN.md.

Corpus A is 40 invented geotechnical sentences (below, safe to publish) that
cover every form the native extractor claims. Corpus B is 40 real sentences
off this submittal's narrative and calculation pages; they are written to
`raw/quantities_real.txt` — inside the gitignored `raw/` folder — for hand
scoring, and are NEVER printed to the report.

Scoring, applied identically to both extractors, is planlens' own rule: a
value with no unit is not a mention. So
  TP  a produced mention whose value is expected and which carries a unit
  FN  an expected value nothing produced
  FP  a produced united mention whose value is neither expected nor allowed
A range may be reported as one mention (low value) or two; the high value is
"allowed", so neither shape is penalized. Dimensionless output is counted
separately AND as FP, because a bare number is exactly what the rule excludes.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "raw", "files",
                   "315000-001-00 Excavation Support Dwg & Calcs.pdf")

# (sentence, must-find values, also-allowed values)
INVENTED: Sequence[Tuple[str, Sequence[float], Sequence[float]]] = [
    ("Borings were advanced at approximately 40-foot centers across the site.",
     [40], []),
    ("The sheet pile toe shall extend 7'-6\" below the excavation subgrade.",
     [7.5], [7, 6]),
    ("A minimum embedment of 12 ft is required at every panel.", [12], []),
    ("Groundwater was measured 18 feet below existing grade.", [18], []),
    ("The diaphragm wall extends 6300 mm below the lowest slab.", [6300], []),
    ("Provide 6.3 m of free length on each ground anchor.", [6.3], []),
    ("The back slope rises at 21 degrees from the crest.", [21], []),
    ("Use an effective friction angle of 34 deg for the granular fill.",
     [34], []),
    ("The wall batter is 5° from vertical.", [5], []),
    ("An allowable bearing pressure of 2,500 psf was assumed.", [2500], []),
    ("Backfill unit weight shall be taken as 120 pcf.", [120], []),
    ("Adopt a saturated unit weight of 18 kN/m³ for the residual soil.",
     [18], []),
    ("The undrained shear strength is 150 kPa at the anchor elevation.",
     [150], []),
    ("End bearing of 2 tsf was used in the capacity check.", [2], []),
    ("Each strut carries a design load of 50 kN.", [50], []),
    ("Lock off the upper tieback at 171 kN pre-load.", [171], []),
    ("Top of the capping beam is at EL. 1684.", [1684], []),
    ("The base slab soffit sits at Elev. 12.5 m.", [12.5], []),
    ("The support of excavation begins at STA 10+50 and runs east.",
     [1050], [10, 50]),
    ("Transition the section between 3+25 and 4+00 on the north face.",
     [325, 400], [3, 25, 4]),
    ("Temporary cut slopes shall be no steeper than 2H:1V.", [2], [1]),
    ("Regrade the stockpile area to 1V:3H before the rainy season.",
     [3], [1]),
    ("Allow 1% of the retained height for lateral wall movement.", [1], []),
    ("Compact each lift to at least 95 percent of maximum dry density.",
     [95], []),
    ("Explorations extended to depths of 20 to 35 feet below grade.",
     [20], [35]),
    ("Design pressures range from 2,500 to 3,200 psf along the north wall.",
     [2500], [3200]),
    ("Excavate 1,200 cy of unsuitable material from the footprint.",
     [1200], []),
    ("The slab on grade covers 4,500 sf of the lower level.", [4500], []),
    ("A 3 m³ test pit was opened at the southeast corner.", [3], []),
    ("Twelve borings and 4 test pits were completed in the first phase.",
     [4], []),
    ("Three anchor levels are shown; 2 levels were installed to date.",
     [2], []),
    ("Cover over the reinforcement shall be 3 in. minimum.", [3], []),
    ("Waler spacing is 8 ft typ. along the south elevation.", [8], []),
    ("The anchor free length is 45 ft ± 2 ft as installed.", [45, 2], []),
    ("Blow counts in excess of 50 blows per 300 mm were recorded.",
     [50, 300], []),
    ("The contract was awarded in 2024 and the report is dated 12 March.",
     [], [2024, 12]),
    ("See Section 4 and Table 3 for the design parameters.", [], [4, 3]),
    ("Borings were advanced 12 in the northern block of the site.", [], [12]),
    ("Page 7 of 42 summarizes the laboratory testing programme.", [], [7, 42]),
    ("The factor of safety against basal heave is 1.4 under drained "
     "conditions.", [], [1.4]),
]


def score(produced: Sequence[Tuple[float, str]],
          must: Sequence[float], allowed: Sequence[float]) -> Tuple[int, int, int, int]:
    """(tp, fn, fp, dimensionless) for one sentence."""
    united = [(v, u) for v, u in produced if u]
    bare = len(produced) - len(united)
    values = [v for v, _u in united]

    def near(a: float, b: float) -> bool:
        return abs(a - b) <= max(1e-6, abs(b) * 1e-6)

    tp = sum(1 for want in must if any(near(v, want) for v in values))
    fn = len(must) - tp
    ok = list(must) + list(allowed)
    fp = sum(1 for v in values if not any(near(v, w) for w in ok))
    return tp, fn, fp + bare, bare


def run_regex(sentence: str) -> List[Tuple[float, str]]:
    from planlens.document.quantities import scan_text
    return [(m.value, m.units) for m in scan_text(sentence)]


def q3_available() -> bool:
    try:
        import quantulum3  # noqa: F401
    except ImportError:
        return False
    return True


def run_q3(sentence: str) -> Tuple[List[Tuple[float, str]], bool]:
    """(mentions, crashed). quantulum3 raises on surfaces its disambiguator
    cannot resolve without the optional classifier models.

    It is NOT installed in the working venv — that is the outcome this probe
    reached (see planlens/document/DESIGN.md, "Quantities"). `pip install
    quantulum3` to reproduce the comparison; without it the quantulum3 rows
    report nothing and the regex rows still run.
    """
    if not q3_available():
        return [], False
    from quantulum3 import parser
    try:
        out = parser.parse(sentence)
    except Exception:
        return [], True
    got = []
    for q in out:
        name = "" if q.unit.name in ("dimensionless", "") else q.unit.name
        got.append((float(q.value), name))
    return got, False


def table(title: str, rows: Dict[str, Tuple[int, int, int, int, int]]) -> None:
    print()
    print(title)
    print("extractor    |  TP |  FN |  FP | precision | recall | crashes")
    for name, (tp, fn, fp, bare, crash) in rows.items():
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"{name:<12} | {tp:>3} | {fn:>3} | {fp:>3} "
              f"|   {p:>5.2f}   |  {r:>5.2f} | {crash}"
              + (f"   ({bare} bare numbers)" if bare else ""))


def corpus_a() -> None:
    rows = {}
    for name, fn in (("regex", None), ("quantulum3", None)):
        tp = fnc = fp = bare = crash = 0
        detail = []
        for sentence, must, allowed in INVENTED:
            if name == "regex":
                got, crashed = run_regex(sentence), False
            else:
                got, crashed = run_q3(sentence)
            crash += int(crashed)
            a, b, c, d = score(got, must, allowed)
            tp, fnc, fp, bare = tp + a, fnc + b, fp + c, bare + d
            if b or c:
                detail.append((sentence, must, allowed, got, b, c))
        rows[name] = (tp, fnc, fp, bare, crash)
        print(f"\n--- {name}: sentences with a miss or a false positive ---")
        for sentence, must, allowed, got, b, c in detail:
            print(f"  {sentence}")
            print(f"     want {list(must)} allowed {list(allowed)} -> got {got}"
                  f"   (FN {b}, FP {c})")
    table("CORPUS A — 40 invented geotechnical sentences", rows)


def corpus_b() -> None:
    from planlens.document import open_document
    import re
    doc = open_document(PDF)
    pages = [s.page for s in doc.page_map() if s.kind in ("text", "mixed")]
    sentences: List[Tuple[int, str]] = []
    seen = set()
    for index in pages:
        for group, _mk in [(g, None) for g in doc._search_groups(index)[0]]:
            text, _spans = doc._join_group(group)
            for piece in re.split(r"(?<=[.!?])\s+", text):
                piece = " ".join(piece.split())
                if not (40 <= len(piece) <= 240):
                    continue
                if not re.search(r"\d", piece) or piece.lower() in seen:
                    continue
                seen.add(piece.lower())
                sentences.append((index, piece))
    # The calc package is 245 pages of program printout, so an even sample of
    # every sentence lands almost entirely on numeric tables and never tests
    # the narrative. Sample the two halves separately: PROSE is mostly letters
    # and real words, PRINTOUT is everything else.
    def is_prose(piece: str) -> bool:
        letters = sum(c.isalpha() or c == " " for c in piece)
        words = [w for w in piece.split() if len(w) > 2 and w.isalpha()]
        return letters / len(piece) >= 0.7 and len(words) >= 6

    prose = [s for s in sentences if is_prose(s[1])]
    printout = [s for s in sentences if not is_prose(s[1])]
    picked = []
    for pool, want in ((prose, 20), (printout, 20)):
        step = max(1, len(pool) // want)
        picked.extend(pool[::step][:want])
    print(f"\n{len(sentences)} candidate real sentences on {len(pages)} "
          f"narrative/calc pages ({len(prose)} prose, {len(printout)} "
          f"printout); sampled {len(picked)}")

    # Under raw/, which .gitignore excludes: this file holds real submittal
    # sentences, and the probe scripts beside it do not.
    out_path = os.path.join(HERE, "raw", "quantities_real.txt")
    with open(out_path, "w", encoding="utf-8") as fh:
        n_regex = n_q3 = n_crash = 0
        for n, (page, sentence) in enumerate(picked, 1):
            r = run_regex(sentence)
            q, crashed = run_q3(sentence)
            n_regex += len(r)
            n_q3 += len([x for x in q if x[1]])
            n_crash += int(crashed)
            fh.write(f"[{n:>2}] p{page}  {sentence}\n")
            fh.write(f"     regex      : {r}\n")
            fh.write(f"     quantulum3 : {q}"
                     + ("  <CRASHED>" if crashed else "") + "\n\n")
    print(f"wrote {out_path} for hand scoring")
    print(f"  regex produced {n_regex} mentions; quantulum3 produced {n_q3} "
          f"united mentions and crashed on {n_crash} of {len(picked)} sentences")
    # Counts and units only — the sentences themselves stay in the file.
    unique_to_each([s for _p, s in picked], "CORPUS B (real)", show=False)
    doc.close()


def unique_to_each(sentences: Sequence[str], label: str,
                   show: bool = True) -> None:
    """The brief's actual decision rule: does quantulum3 RECOVER anything the
    regex misses? A united mention whose value neither extractor's other half
    produced is what "recovers" means."""
    only_q3: List[Tuple[str, float, str]] = []
    only_rx: List[Tuple[str, float, str]] = []
    for s in sentences:
        rx = [(v, u) for v, u in run_regex(s) if u]
        q3 = [(v, u) for v, u in run_q3(s)[0] if u]
        rv = [v for v, _ in rx]
        qv = [v for v, _ in q3]
        for v, u in q3:
            if not any(abs(v - w) <= max(1e-6, abs(w) * 1e-6) for w in rv):
                only_q3.append((s, v, u))
        for v, u in rx:
            if not any(abs(v - w) <= max(1e-6, abs(w) * 1e-6) for w in qv):
                only_rx.append((s, v, u))
    print(f"\n{label}: united mentions unique to one extractor")
    print(f"  only quantulum3 found: {len(only_q3)}")
    for s, v, u in only_q3:
        print(f"     {v!r} {u!r}" + (f"   <- {s[:70]}" if show else ""))
    print(f"  only the regex found: {len(only_rx)}")
    if show:
        for s, v, u in only_rx[:12]:
            print(f"     {v!r} {u!r}   <- {s[:70]}")
    else:
        from collections import Counter
        print("     by unit:", dict(Counter(u for _s, _v, u in only_rx)))


def main() -> int:
    t0 = time.time()
    if not q3_available():
        print("NOTE: quantulum3 is not installed (that is the decision this "
              "probe reached). `pip install quantulum3` to reproduce its "
              "rows; the regex rows run either way.\n")
    corpus_a()
    unique_to_each([s for s, _m, _a in INVENTED], "CORPUS A")
    corpus_b()
    print(f"\ntotal {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

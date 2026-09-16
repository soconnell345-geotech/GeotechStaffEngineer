"""Can a picture hash tell duplicate pages apart, and what does it cost?

`duplicate_of` on planlens' page map is decided from a page's TEXT and its
path count, which is the one thing a scanned page does not have. This probe
asks whether a difference hash of a tiny grayscale render separates the pages
a reviewer would call the same from the pages they would not — and by how
much, so the threshold is measured rather than picked.

Two corpora:

- this submittal (private; `raw/` is gitignored). Only page indices, kinds and
  distances are printed — never a title, a string or a file name;
- the ten public Mecklenburg drawing sheets, one sheet per file, whose
  inter-sheet distances are publishable numbers.

Sections

  A. cost, and the distances between pages of one kind and size — both over
     every page and over only the pages the gate lets through, because an
     ungated page never gets a hash at all;
  B. the ten public sheets: inter-sheet distances, a re-render of one sheet,
     and the same sheet turned 90 degrees;
  C. how far a TRUE copy drifts: one real page rebuilt as an image of itself
     at other resolutions, JPEG qualities, offsets and skews — the way a
     second copy of a page actually arrives;
  D. grid contrast per page, which is what decides whether a hash carries
     information at all (a uniform page hashes to all zeros, and so does the
     next uniform page);
  E. a whole run of same-template pages rebuilt as scans — a scanned appendix
     of boring logs is the population most likely to produce a wrong claim;
  F. whether a finer grid (16x16 = 256 bits) separates the classes better
     than 8x8 = 64 bits.

Run:
  <python> module_work/field_feedback/2026-09-09_nairobi-soe_v5.11.2/probe_dup_hash.py
"""

from __future__ import annotations

import glob
import itertools
import os
import statistics
import time
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "raw", "files",
                   "315000-001-00 Excavation Support Dwg & Calcs.pdf")
CORPUS = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                      "drawing_ground_truth", "mecklenburg", "*.pdf")

#: Runs of same-template pages in the private document (boring-log forms, the
#: drawing set), used by section E.
TEMPLATE_RUNS = (("form pages", range(28, 35)),
                 ("drawing sheets", range(7, 14)))


def _stats(values: Sequence[int]) -> str:
    if not values:
        return "(none)"
    return (f"n={len(values)} min={min(values)} "
            f"median={int(statistics.median(values))} max={max(values)}")


def _pairs_table(name: str, pairs: List[Tuple[int, int, int]],
                 show: int = 8) -> None:
    print(f"\n{name}: {_stats([d for _, _, d in pairs])}")
    for a, b, d in sorted(pairs, key=lambda r: r[2])[:show]:
        print(f"    {a:>4} vs {b:>4}   distance {d}")


def _rebuild(src, zoom: float = 1.0, rotate: float = 0.0,
             jpeg: Optional[int] = None, dx: float = 0.0, dy: float = 0.0):
    """One page rebuilt as an IMAGE of itself, the way a rescan arrives.

    A second copy of a page is rarely byte-identical: it has been printed and
    scanned again, at another resolution, through another JPEG quality,
    landing a millimetre off and a fraction of a degree askew. This makes
    those copies from a real page so the drift can be measured against the
    distance between pages that are genuinely different.
    """
    import fitz
    matrix = fitz.Matrix(zoom, zoom)
    if rotate:
        matrix = matrix.prerotate(rotate)
    pix = src.get_pixmap(matrix=matrix)
    data = (pix.tobytes("jpeg", jpg_quality=jpeg) if jpeg
            else pix.tobytes("png"))
    out = fitz.open()
    page = out.new_page(width=src.rect.width, height=src.rect.height)
    rect = fitz.Rect(src.rect.x0 + dx, src.rect.y0 + dy,
                     src.rect.x1 + dx, src.rect.y1 + dy)
    page.insert_image(rect, stream=data)
    return out, page


def probe_submittal() -> List[int]:
    import fitz
    from planlens.document import open_document
    from planlens.document.imagehash import (
        hamming, page_dhash, wants_image_hash,
    )

    print("=" * 72)
    print("A. the real submittal (private: indices and distances only)")
    print("=" * 72)
    t0 = time.perf_counter()
    doc = open_document(PDF)
    summaries = doc.page_map()
    t_map = time.perf_counter() - t0
    print(f"pages {len(summaries)}   page map {t_map:.1f} s")

    raw = fitz.open(PDF)
    hashes: Dict[int, str] = {}
    times: Dict[int, float] = {}
    gated: List[int] = []
    for s in summaries:
        t = time.perf_counter()
        h = page_dhash(raw[s.page])
        times[s.page] = time.perf_counter() - t
        if h is not None:
            hashes[s.page] = h
        if wants_image_hash(s.kind, s.n_text_chars,
                            bool(s.evidence.get("needs_ocr"))):
            gated.append(s.page)

    total = sum(times.values())
    gated_cost = sum(times[p] for p in gated)
    kinds: Dict[str, int] = {}
    for s in summaries:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    print(f"kinds: {kinds}")
    print(f"COST  every page ({len(times)}): {total:.2f} s")
    print(f"COST  gated only ({len(gated)}): {gated_cost:.2f} s "
          f"({100.0 * len(gated) / max(1, len(times)):.0f}% of pages)")
    print(f"pages with no hash at all: "
          f"{[s.page for s in summaries if s.page not in hashes]}")
    slow = sorted(times.items(), key=lambda kv: -kv[1])[:5]
    print("slowest pages: " + ", ".join(f"p{p} {t * 1000:.0f} ms"
                                        for p, t in slow))

    size = {s.page: (round(s.width, 1), round(s.height, 1)) for s in summaries}
    kind = {s.page: s.kind for s in summaries}
    gate = set(gated)

    text_dups = [(s.page, s.duplicate_of) for s in summaries
                 if s.duplicate_of is not None]
    print(f"\ntext-rule duplicates: {len(text_dups)} page(s)")
    for a, b in text_dups:
        if a in hashes and b in hashes:
            print(f"    {b} -> {a}   distance {hamming(hashes[a], hashes[b])}"
                  f"   kind {kind[a]}")
    dup_set = {(min(a, b), max(a, b)) for a, b in text_dups}

    same: Dict[Tuple[str, Tuple[float, float]], List[int]] = {}
    for p in sorted(hashes):
        same.setdefault((kind[p], size[p]), []).append(p)
    different: List[Tuple[int, int, int]] = []
    per_kind: Dict[str, List[Tuple[int, int, int]]] = {}
    gated_pairs: List[Tuple[int, int, int]] = []
    for (k, _sz), pages in same.items():
        for i, a in enumerate(pages):
            for b in pages[i + 1:]:
                if (a, b) in dup_set:
                    continue
                d = hamming(hashes[a], hashes[b])
                different.append((a, b, d))
                per_kind.setdefault(k, []).append((a, b, d))
                if a in gate and b in gate:
                    gated_pairs.append((a, b, d))
    _pairs_table("DIFFERENT pages, same kind and size (all pairs)", different)
    for k in sorted(per_kind):
        _pairs_table(f"  ... of kind {k}", per_kind[k], show=4)
    # The pairs the RULE actually compares: both pages gated in. Everything
    # above is context — an ungated page never gets a hash at all.
    _pairs_table("PAIRS THE RULE COMPARES (both pages gated in)", gated_pairs)

    forms = [s.page for s in summaries if s.kind == "form"]
    consec = [(a, a + 1, hamming(hashes[a], hashes[a + 1]))
              for a in forms if a + 1 in hashes and kind.get(a + 1) == "form"
              and size[a] == size[a + 1]]
    _pairs_table("consecutive form pages (the text rule covers these)", consec)

    raw2 = fitz.open(PDF)
    probe_pages = (list(hashes)[:6] + gated[:6])[:10]
    unstable = [p for p in probe_pages if page_dhash(raw2[p]) != hashes[p]]
    print(f"\nstability: {len(probe_pages)} pages re-hashed from a second "
          f"open, {len(unstable)} differ -> {unstable}")
    raw2.close()
    raw.close()
    doc.close()

    samples: List[int] = []
    for want in ("scanned", "figure", "drawing_sheet"):
        page = next((p for p in gated if kind[p] == want), None)
        if page is not None:
            samples.append(page)
    return samples


DEGRADATIONS = [
    ("re-rendered 1:1, PNG", dict(zoom=1.0)),
    ("150 dpi PNG", dict(zoom=150 / 72)),
    ("100 dpi JPEG q40", dict(zoom=100 / 72, jpeg=40)),
    ("100 dpi JPEG q20", dict(zoom=100 / 72, jpeg=20)),
    ("150 dpi, placed 3 pt low/right", dict(zoom=150 / 72, dx=3, dy=3)),
    ("150 dpi, placed 8 pt low/right", dict(zoom=150 / 72, dx=8, dy=8)),
    ("150 dpi, skewed 0.3 deg", dict(zoom=150 / 72, rotate=0.3)),
    ("150 dpi, skewed 1.0 deg", dict(zoom=150 / 72, rotate=1.0)),
]


def probe_degradations(pages: Sequence[int], side: int = 8) -> None:
    """How far does a page drift from ITSELF when it comes back as a scan?"""
    import fitz
    from planlens.document.imagehash import hamming, page_dhash

    print("\n" + "=" * 72)
    print(f"C. how far a TRUE copy drifts ({side}x{side} = {side * side} bits)")
    print("=" * 72)
    raw = fitz.open(PDF)
    for index in pages:
        base = page_dhash(raw[index], side)
        print(f"\n  page {index} ({round(raw[index].rect.width)}x"
              f"{round(raw[index].rect.height)} pt)")
        for name, kwargs in DEGRADATIONS:
            doc2, page2 = _rebuild(raw[index], **kwargs)
            got = page_dhash(page2, side)
            doc2.close()
            print(f"    {name:<32} distance "
                  f"{hamming(base, got) if got else '(no hash)'}")
    raw.close()


def probe_contrast() -> None:
    """Does the page carry a picture at all? A uniform page hashes to zero."""
    import fitz
    from planlens.document import open_document
    from planlens.document.imagehash import _gray_grid, wants_image_hash

    print("\n" + "=" * 72)
    print("D. grid contrast (max - min grey of the 9x8 render, 0-255)")
    print("=" * 72)
    doc = open_document(PDF)
    summaries = doc.page_map()
    raw = fitz.open(PDF)
    gated_spread, flat = [], []
    for s in summaries:
        grid = _gray_grid(raw[s.page], 9, 8)
        spread = int(grid.max()) - int(grid.min())
        if wants_image_hash(s.kind, s.n_text_chars,
                            bool(s.evidence.get("needs_ocr"))):
            gated_spread.append((spread, s.page, s.kind))
        else:
            flat.append((spread, s.page, s.kind))
    gated_spread.sort()
    flat.sort()
    print(f"gated pages   : {_stats([v for v, _, _ in gated_spread])}")
    print(f"   lowest five: {gated_spread[:5]}")
    print(f"ungated pages : {_stats([v for v, _, _ in flat])}")
    print(f"   lowest five: {flat[:5]}")
    print(f"   ungated pages under 20: "
          f"{sum(1 for v, _, _ in flat if v < 20)} of {len(flat)}")
    raw.close()
    doc.close()

    # The failure mode a contrast floor exists to stop.
    from planlens.document.imagehash import hamming, page_dhash
    out = fitz.open()
    out.new_page(width=612, height=792)                       # plain white
    p2 = out.new_page(width=612, height=792)
    p2.draw_rect(p2.rect, color=None, fill=(0.97, 0.97, 0.97))  # grey wash
    p3 = out.new_page(width=612, height=792)
    p3.draw_rect(fitz.Rect(0, 0, 612, 400), color=None,
                 fill=(0.98, 0.98, 0.98))                     # half wash
    print("\nthree DIFFERENT near-uniform pages:")
    for i in range(3):
        grid = _gray_grid(out[i], 9, 8)
        print(f"    page {i}: spread {int(grid.max()) - int(grid.min()):>3}  "
              f"hash {page_dhash(out[i], guard=False)}")
    hs = [page_dhash(out[i], guard=False) for i in range(3)]
    print(f"    distances: 0-1 {hamming(hs[0], hs[1])}, "
          f"0-2 {hamming(hs[0], hs[2])}  <- why a contrast floor exists")
    print(f"    with the floor on: {[page_dhash(out[i]) for i in range(3)]}")
    out.close()


def probe_template_runs() -> None:
    """Different pages off ONE template, rebuilt as scans: the worst case."""
    import fitz
    from planlens.document.imagehash import hamming, page_dhash

    print("\n" + "=" * 72)
    print("E. same-template pages rebuilt as scans (150 dpi JPEG q60)")
    print("=" * 72)
    raw = fitz.open(PDF)
    for label, pages in TEMPLATE_RUNS:
        pages = list(pages)
        hashes, keep = {}, []
        for p in pages:
            out, page = _rebuild(raw[p], zoom=150 / 72, jpeg=60)
            hashes[p] = page_dhash(page, guard=False)
            keep.append(out)
        pairs = [(a, b, hamming(hashes[a], hashes[b]))
                 for a, b in itertools.combinations(pages, 2)]
        _pairs_table(f"{label} rescanned, every pair", pairs, show=5)
        print("    each rescan vs its OWN original: " + ", ".join(
            f"{p}:{hamming(page_dhash(raw[p], guard=False), hashes[p])}"
            for p in pages))
        for out in keep:
            out.close()
    raw.close()


def probe_grid_size() -> None:
    """Does a finer grid separate the two classes any better?"""
    import fitz
    from planlens.document.imagehash import hamming, page_dhash

    print("\n" + "=" * 72)
    print("F. 64 bits vs 256 bits on the hard pairs")
    print("=" * 72)
    raw = fitz.open(PDF)
    pairs = [(96, 115, "two figures off one template"),
             (163, 256, "two program outputs, one template"),
             (32, 33, "two boring logs"), (7, 8, "two drawing sheets")]
    for side in (8, 16):
        print(f"  --- {side}x{side} = {side * side} bits ---")
        for a, b, what in pairs:
            d = hamming(page_dhash(raw[a], side), page_dhash(raw[b], side))
            print(f"    {a:>4} vs {b:>4}  {d:>3} bits "
                  f"({100.0 * d / (side * side):4.1f}%)  {what}")
        drift = []
        for name, kwargs in DEGRADATIONS[:4]:
            out, page = _rebuild(raw[7], **kwargs)
            drift.append(hamming(page_dhash(raw[7], side),
                                 page_dhash(page, side)))
            out.close()
        print(f"    a TRUE copy of page 7 drifts {drift} bits "
              f"({[round(100.0 * v / (side * side), 1) for v in drift]}%)")
    raw.close()


def probe_corpus() -> None:
    import fitz
    from planlens.document import open_document
    from planlens.document.imagehash import hamming, page_dhash

    print("\n" + "=" * 72)
    print("B. the ten public Mecklenburg sheets (one sheet per file)")
    print("=" * 72)
    files = sorted(glob.glob(CORPUS))
    hashes: List[Tuple[str, str, str, Tuple[float, float]]] = []
    t0 = time.perf_counter()
    for path in files:
        name = os.path.basename(path)
        with open_document(path) as doc:
            s = doc.summary(0)
        raw = fitz.open(path)
        h = page_dhash(raw[0])
        size = (round(raw[0].rect.width, 1), round(raw[0].rect.height, 1))
        raw.close()
        if h:
            hashes.append((name, h, s.kind, size))
    print(f"{len(hashes)} sheets hashed in {time.perf_counter() - t0:.2f} s "
          f"(includes opening each file and its page summary)")
    for name, h, k, size in hashes:
        print(f"    {name:<12} kind {k:<13} size {size}")
    sizes = {n: s for n, _, _, s in hashes}
    dists = [(na, nb, hamming(ha, hb))
             for (na, ha, _ka, _sa), (nb, hb, _kb, _sb)
             in itertools.combinations(hashes, 2)]
    print(f"\ninter-sheet distances: {_stats([d for _, _, d in dists])}")
    for na, nb, d in sorted(dists, key=lambda r: r[2])[:6]:
        print(f"    {na:<12} vs {nb:<12} distance {d}")
    same_size = [d for na, nb, d in dists if sizes[na] == sizes[nb]]
    print(f"of those, pairs of equal page size: {_stats(same_size)}")

    if hashes:
        raw = fitz.open(files[0])
        again = page_dhash(raw[0])
        rot = fitz.open(files[0])
        rot[0].set_rotation((rot[0].rotation + 90) % 360)
        turned = page_dhash(rot[0])
        print(f"\nsame sheet re-rendered: distance "
              f"{hamming(hashes[0][1], again)}")
        print(f"same sheet with /Rotate +90: distance "
              f"{hamming(hashes[0][1], turned)} "
              f"(displayed size {round(rot[0].rect.width, 1)}x"
              f"{round(rot[0].rect.height, 1)} vs "
              f"{round(raw[0].rect.width, 1)}x{round(raw[0].rect.height, 1)})")
        rot.close()
        raw.close()


if __name__ == "__main__":
    samples = probe_submittal()
    probe_corpus()
    probe_degradations(samples)
    probe_contrast()
    probe_template_runs()
    probe_grid_size()

"""WP0 measurement: what planlens sees in all 38 corpus reports.

Run from the repo root::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp0

Prints one row per report, then the hand-label counts and the sheet-to-ID
matching, and appends the lot to the private measurements ledger as a
``## WP0 -- corpus harness`` section. Reports are named by ID only; nothing
this script writes carries a file name, a project or a place.

The two OCR columns are deliberately before-and-after. ``needs OCR`` is what
the PDF's OWN text layer cannot deliver, which is the number that decides
where ``di="auto"`` spends a DI result; ``left`` is what still needs optical
help once ``auto`` has attached one.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import warnings
from collections import Counter
from datetime import date
from pathlib import Path
from typing import List, Optional

from module_work.report_ingest_harness import corpus, labels

LEDGER = corpus.RAW_DIR.parent / "MEASUREMENTS.md"
SECTION = "## WP0 -- corpus harness"

#: The report whose text-rule duplicate claims the lead wants to eyeball.
DUP_CHECK_ID = "R28"
DUP_CHECK_DPI = 40


def _kinds(summaries) -> str:
    counts = Counter(s.kind for s in summaries)
    return ", ".join(f"{k} {n}" for k, n in counts.most_common())


class Row:
    """One measured report. Everything here is safe to publish."""

    def __init__(self, rid: str):
        self.rid = rid
        self.pages = 0
        self.kinds = ""
        self.image_dups = 0
        self.text_dups = 0
        self.unreliable_pre = 0
        self.needs_ocr_pre = 0
        self.needs_ocr_left = 0
        self.di_pages: Optional[int] = None
        self.di_attached = 0
        self.seconds = 0.0
        self.note = ""

    @property
    def di_cell(self) -> str:
        if self.di_pages is None:
            return "n"
        return f"y {self.di_pages}"

    def cells(self) -> List[str]:
        return [
            self.rid, str(self.pages), self.kinds,
            str(self.image_dups), str(self.text_dups),
            str(self.unreliable_pre), str(self.needs_ocr_pre),
            str(self.needs_ocr_left), self.di_cell,
            str(self.di_attached), f"{self.seconds:.1f}",
        ]


HEADER = ["ID", "pages", "planlens kinds", "img dup", "text dup",
          "text_reliable False", "needs OCR", "left", "DI", "DI used", "s"]


def measure(rid: str) -> Row:
    """Open one report twice -- plain, then ``di="auto"`` -- and count."""
    from planlens.document.azure_di import pages_needing_ocr

    row = Row(rid)
    start = time.perf_counter()
    with corpus.open_report(rid, di="none") as doc:
        plain = doc.page_map()
        row.pages = len(plain)
        row.unreliable_pre = sum(1 for s in plain if not s.text_reliable)
        wanted = pages_needing_ocr(doc)
        row.needs_ocr_pre = len(wanted)
        kinds_plain = _kinds(plain)

    if corpus.has_di(rid):
        row.di_pages = corpus.di_page_count(rid)
    with corpus.open_report(rid, di="auto", warn=False,
                            ocr_pages=wanted) as doc:
        auto = doc.page_map()
        row.kinds = _kinds(auto)
        row.image_dups = sum(1 for s in auto if s.duplicate_rule == "image")
        row.text_dups = sum(1 for s in auto if s.duplicate_rule == "text")
        row.needs_ocr_left = len(pages_needing_ocr(doc))
        row.di_attached = sum(
            1 for s in auto if s.evidence.get("text_source") == "azure_di")
        if rid == DUP_CHECK_ID:
            row.note = dup_check(doc, auto)
    if row.kinds != kinds_plain:
        moved = f"page kinds moved once DI text was attached ({kinds_plain})"
        row.note = f"{row.note}; {moved}" if row.note else moved
    row.seconds = time.perf_counter() - start
    gc.collect()
    return row


# -- the R28 duplicate check ------------------------------------------------

def _pairs(summaries):
    return [(s.page, s.duplicate_of) for s in summaries
            if s.duplicate_rule == "text"]


def _render_pair(doc, a: int, b: int, out: Path) -> bool:
    """Two pages side by side at 40 dpi, so a human can see they differ."""
    try:
        import io

        from PIL import Image
    except ImportError:                              # pragma: no cover
        return False
    tiles = []
    for index in (a, b):
        png, _info = doc.render(index, dpi=DUP_CHECK_DPI)
        tiles.append(Image.open(io.BytesIO(png)).convert("RGB"))
    gap = 8
    width = sum(t.width for t in tiles) + gap
    height = max(t.height for t in tiles)
    sheet = Image.new("RGB", (width, height), "white")
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width + gap
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return True


def _markup_text(doc, index: int) -> List[str]:
    return sorted(" ".join((m.text or "").split())
                  for m in doc.markups(pages=[index]) if m.text)


def dup_check(doc, summaries) -> str:
    """List the text-rule duplicate pairs and render the first two.

    The text rule hashes a page's TEXT LINES. Anything a page says only in an
    annotation is invisible to it, so the check also reports whether the two
    pages carry different markup text -- which is the difference a reader
    sees.
    """
    pairs = _pairs(summaries)
    print(f"\n{DUP_CHECK_ID} text-rule duplicate claims "
          f"({len(pairs)}), page -> duplicate_of:")
    print("  " + ", ".join(f"{p} -> {q}" for p, q in pairs))
    made: List[str] = []
    differ: List[str] = []
    for a, b in pairs[:2]:
        out = corpus.CHECKS_DIR / f"{DUP_CHECK_ID}_textdup_{b}_{a}.png"
        if _render_pair(doc, b, a, out):
            made.append(out.name)
            print(f"  rendered {out}")
        if _markup_text(doc, a) != _markup_text(doc, b):
            differ.append(f"{b}/{a}")
            print(f"  pages {b} and {a} carry DIFFERENT markup text")
    if not made:
        return f"{len(pairs)} text-rule pairs; render unavailable"
    note = (f"{len(pairs)} text-rule pairs; first two rendered at "
            f"{DUP_CHECK_DPI} dpi ({', '.join(made)}) -- AWAITING a visual "
            f"check by the lead")
    if differ:
        note += (f". Pair(s) {', '.join(differ)} have identical text lines "
                 f"but DIFFERENT markup text, so the claim is wrong on the "
                 f"page a reader sees")
    return note


# -- output -----------------------------------------------------------------

def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def build_report(rows: List[Row]) -> List[str]:
    lines = [SECTION, ""]
    lines.append(f"Measured {date.today().isoformat()} with the app venv and "
                 f"the editable planlens checkout, every report opened "
                 f'`di="auto"`. `needs OCR` is what the PDF text layer cannot '
                 f"deliver; `left` is what remains after `auto` attached a DI "
                 f"result; `DI used` is the pages whose text actually came "
                 f"from DI.")
    lines.append("")
    lines += _table(HEADER, [r.cells() for r in rows])
    lines.append("")
    total_pages = sum(r.pages for r in rows)
    with_di = sum(1 for r in rows if r.di_pages is not None)
    lines.append(
        f"{len(rows)} reports, {total_pages} pages, {with_di} with a usable "
        f"DI result, {sum(r.di_attached for r in rows)} pages read by DI "
        f"under `auto`, "
        f"{sum(r.needs_ocr_left for r in rows)} pages still without readable "
        f"text, {sum(r.seconds for r in rows):.0f} s in total.")
    notes = [r for r in rows if r.note]
    if notes:
        lines.append("")
        lines.append("Notes:")
        lines.append("")
        for r in notes:
            lines.append(f"- **{r.rid}**: {r.note}")

    lines += ["", "### Hand labels", ""]
    counts = labels.label_counts()
    lines += _table(["label", "pages"],
                    [[k, str(counts[k])] for k in labels.LABELS])
    lines.append("")
    mapped = labels.mapped_ids()
    lines.append(f"{sum(counts.values())} labelled pages over "
                 f"{len(mapped)} mapped reports ({', '.join(mapped)}).")
    lines += ["", "### Sheet-to-ID matching", ""]
    for m in labels.sheet_map():
        lines.append(f"- {m.describe()}")
    unmatched = [m for m in labels.sheet_map() if m.rid is None]
    if unmatched:
        lines.append("")
        lines.append(
            f"{len(unmatched)} sheet(s) unmatched. The spreadsheet covers a "
            f"report whose PDF was never copied into the corpus, so its rows "
            f"are out of the scorecard until the PDF arrives. It is NOT "
            f"matched on page count alone: an unrelated report of exactly "
            f"that length exists, and matching it would have scored those "
            f"labels against the wrong document.")
    lines.append("")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", metavar="ID",
                        help="measure just these IDs (default: all 38)")
    parser.add_argument("--no-append", action="store_true",
                        help="print the section but do not touch the ledger")
    args = parser.parse_args(argv)

    if not corpus.raw_available():
        print("the private corpus is not on this machine; nothing to measure",
              file=sys.stderr)
        return 1

    warnings.simplefilter("ignore", RuntimeWarning)
    wanted = args.only or [r.id for r in corpus.list_reports()]
    rows: List[Row] = []
    widths = [max(len(h), 5) for h in HEADER]
    print("| " + " | ".join(HEADER) + " |")
    for rid in wanted:
        row = measure(rid)
        rows.append(row)
        print("| " + " | ".join(row.cells()) + " |", flush=True)
    del widths

    lines = build_report(rows)
    print()
    print("\n".join(lines))

    if not args.no_append and not args.only:
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write("\n---\n\n")
            fh.write("\n".join(lines))
        print(f"appended to {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Synthetic boring logs on which no depth ruler can be fitted.

Built for :mod:`report_ingest.printed_depths` (BACKLOG RI-18). Each page is a
ruled log form whose depths are on the paper only as TEXT, in one of the ways
a real log prints them:

- :func:`build_tabular_log` -- a log not drawn to scale. Every row is the
  same height whatever its depth, and the depth column prints an interval for
  each stratum and a single depth for each sample. No straight line runs
  through the depth column, so ``log_grid`` fits no ruler.
- :func:`build_sample_depths_log` -- a form whose depth scale was never
  printed (the column is there, empty) and whose sample column prints each
  sample's own interval, ``S-1  2.5-4.0``. The descriptions print no depth.

Nothing here is copied from a real log; the wording is the trade's and the
numbers are made up. The answers are stated beside each builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

LETTER = (612.0, 792.0)
X_LEFT = 40.0
X_RIGHT = 570.0
Y_FORM_TOP = 110.0
Y_HEADER_BOTTOM = 150.0
Y_FORM_BOTTOM = 720.0


@dataclass
class NoRulerGT:
    """One synthetic page, and what it prints."""

    pdf: bytes
    unit: str
    #: ``(top, bottom, first words)`` of each stratum whose interval is
    #: printed; a stratum printed with one depth only is not here.
    layers: List[Tuple[float, float, str]] = field(default_factory=list)
    #: ``(label, top, bottom, blows, N)`` of each sample, as printed.
    samples: List[Tuple[str, float, Optional[float], str, Optional[int]]] = \
        field(default_factory=list)
    #: Every depth printed on the page, tops and bottoms.
    printed: Tuple[float, ...] = ()


def _form(page, edges: Sequence[float], headers: Sequence[str]) -> None:
    for x in edges:
        page.draw_line((x, Y_FORM_TOP), (x, Y_FORM_BOTTOM), width=0.6)
    for y in (Y_FORM_TOP, Y_HEADER_BOTTOM, Y_FORM_BOTTOM):
        page.draw_line((X_LEFT, y), (X_RIGHT, y), width=0.6)
    for i, head in enumerate(headers):
        page.insert_text((edges[i] + 3, Y_HEADER_BOTTOM - 8), head,
                         fontsize=6)


def _fields(page, hole: str, unit: str) -> None:
    page.insert_text((X_LEFT + 2, 40), f"BORING LOG {hole}", fontsize=13,
                     fontname="hebo")
    page.insert_text((X_LEFT + 2, 62), "PROJECT:  Hollow Creek Culvert",
                     fontsize=9)
    page.insert_text((X_LEFT + 2, 80), "Hammer Type:  Automatic", fontsize=7)
    page.insert_text((X_LEFT + 250, 80), "Driller:  Northfield Drilling",
                     fontsize=7)


#: The tabular log, in metres. Rows of one height; the depth column prints
#: an interval for a stratum and a single depth for a sample.
_TAB_EDGES = (40.0, 100.0, 320.0, 380.0, 450.0, 500.0, 570.0)
_TAB_HEADERS = ("DEPTH (m)", "DESCRIPTION", "SAMPLE", "BLOWS", "N-VALUE",
                "MC (%)")
_ROW_PT = 36.0
#: ``(depth as printed, description lines, sample, blows, N, MC)``. The MC
#: of S-2 is printed a line BELOW its row, where no depth stands beside it.
#: The SILTY CLAY's description runs on for four lines, its last on the
#: line of S-2. From position alone that line could as well start a new
#: stratum beside the sample, so the floor keeps the clay's first three
#: lines and leaves the fourth to the reader.
_TAB_ROWS = (
    ("0.00 - 1.20", ("FILL: brown silty sand with gravel",
                     "and brick fragments"), "", "", "", ""),
    ("0.50", (), "S-1", "2-3-4", "12", "25"),
    ("1.20 - 4.50", ("SILTY CLAY (CL), firm to stiff, grey,",
                     "with fine sand partings",
                     "and occasional rootlets,",
                     "mottled orange-brown"), "", "", "", ""),
    ("1.50", (), "S-2", "3-5-6", "8", "BELOW:14"),
    ("3.00", (), "S-3", "4-6-9", "30", "31"),
    ("4.50 - 7.00", ("SAND (SP), dense, brown, wet",), "", "", "", ""),
    ("6.00", (), "S-4", "10-14-16", "9", "18"),
    ("7.00", ("End of borehole",), "", "", "", ""),
)


def build_tabular_log() -> NoRulerGT:
    """A boring log NOT drawn to scale, its depths printed in a depth column.

    Strata 0.00-1.20 (FILL), 1.20-4.50 (SILTY CLAY), 4.50-7.00 (SAND); a
    final row printing the single depth 7.00 beside "End of borehole".
    Samples S-1 to S-4 at 0.50, 1.50, 3.00 and 6.00, each printed with its
    blows and N on its own line.
    """
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    _fields(page, "BH-3", "m")
    _form(page, _TAB_EDGES, _TAB_HEADERS)
    for i, (depth, desc, sample, blows, n_value, mc) in enumerate(_TAB_ROWS):
        y = Y_HEADER_BOTTOM + i * _ROW_PT + 12.0
        page.insert_text((_TAB_EDGES[0] + 4, y), depth, fontsize=7)
        for k, line in enumerate(desc):
            page.insert_text((_TAB_EDGES[1] + 4, y + 10.0 * k), line,
                             fontsize=7)
        for col, text in ((2, sample), (3, blows), (4, n_value), (5, mc)):
            if not text:
                continue
            dy = 0.0
            if text.startswith("BELOW:"):
                text, dy = text[len("BELOW:"):], 12.0
            page.insert_text((_TAB_EDGES[col] + 4, y + dy), text, fontsize=7)
    pdf = doc.tobytes()
    doc.close()
    return NoRulerGT(
        pdf=pdf, unit="m",
        layers=[(0.0, 1.2, "FILL"), (1.2, 4.5, "SILTY CLAY"),
                (4.5, 7.0, "SAND")],
        samples=[("S-1", 0.5, None, "2-3-4", 12),
                 ("S-2", 1.5, None, "3-5-6", 8),
                 ("S-3", 3.0, None, "4-6-9", 30),
                 ("S-4", 6.0, None, "10-14-16", 9)],
        printed=(0.0, 1.2, 0.5, 4.5, 1.5, 3.0, 7.0, 6.0))


#: The sample-depths log, in feet: the form has a depth column, but its scale
#: was never printed; each sample prints its own interval.
_SMP_EDGES = (40.0, 100.0, 320.0, 420.0, 480.0, 530.0, 570.0)
_SMP_HEADERS = ("DEPTH (ft)", "DESCRIPTION", "SAMPLE", "BLOWS", "N-VALUE",
                "MC (%)")
_SMP_PER_FT = 20.0


def build_sample_depths_log(first: float = 2.5, hole: str = "B-7",
                            depths: bool = True) -> NoRulerGT:
    """A log whose only printed depths are its samples' own intervals.

    Four split-spoon samples, S-1 to S-4, each printed ``S-n  top-bottom``
    with its blows and N on the same line, at ``first``, ``first + 2.5``,
    ``first + 7.5`` and ``first + 12.5`` ft. Two strata described with no
    depth printed anywhere.

    With ``depths`` False the samples print their labels alone, and the log
    prints no depth at all: not in a column, not in the header, not in the
    descriptions. The rows sit where they would have, so a reader that took
    a depth from a row's position would still find one to take.
    """
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    _fields(page, hole, "ft")
    _form(page, _SMP_EDGES, _SMP_HEADERS)

    def y_of(depth: float) -> float:
        return Y_HEADER_BOTTOM + (depth - first + 1.0) * _SMP_PER_FT

    page.insert_text((_SMP_EDGES[1] + 4, y_of(first - 0.5)),
                     "LEAN CLAY (CL), brown, stiff", fontsize=7)
    page.insert_text((_SMP_EDGES[1] + 4, y_of(first + 6.0)),
                     "POORLY GRADED SAND (SP), medium dense", fontsize=7)
    tops = (first, first + 2.5, first + 7.5, first + 12.5)
    blows = ("4-5-7", "3-6-8", "7-9-12", "5-6-6")
    n_values = (12, 9, 20, 8)
    samples = []
    for k, (top, blow, n_value) in enumerate(zip(tops, blows, n_values)):
        label = f"S-{k + 1}"
        y = y_of(top)
        page.insert_text((_SMP_EDGES[2] + 4, y),
                         f"{label}  {top:.1f}-{top + 1.5:.1f}" if depths
                         else label, fontsize=7)
        page.insert_text((_SMP_EDGES[3] + 4, y), blow, fontsize=7)
        page.insert_text((_SMP_EDGES[4] + 4, y), str(n_value), fontsize=7)
        samples.append((label, top, top + 1.5, blow, n_value))
    page.insert_text((_SMP_EDGES[5] + 4, y_of(tops[0])), "22", fontsize=7)
    page.insert_text((_SMP_EDGES[5] + 4, y_of(tops[2])), "18", fontsize=7)
    pdf = doc.tobytes()
    doc.close()
    if not depths:
        return NoRulerGT(pdf=pdf, unit="ft")
    printed = tuple(v for s in samples for v in (s[1], s[2]))
    return NoRulerGT(pdf=pdf, unit="ft", layers=[], samples=samples,
                     printed=printed)


def two_page_log(first_pdf: bytes, second_pdf: bytes) -> bytes:
    """The two one-page logs as one PDF, page 0 then page 1: a log whose
    first sheet has a ruler and whose continuation sheet does not."""
    import pymupdf
    doc = pymupdf.open()
    for pdf in (first_pdf, second_pdf):
        part = pymupdf.open(stream=pdf, filetype="pdf")
        doc.insert_pdf(part)
        part.close()
    out = doc.tobytes()
    doc.close()
    return out

"""A synthetic run of boring logs: three holes on five pages (RI-117).

The case the item builder got wrong on real reports: consecutive logs of
DIFFERENT holes whose title blocks all print the same code before the hole's
name (here a form number, ``FORM QZ-7``), so planlens' rules key every page
by that code and make the five pages one work item.

* page 0: hole HX-3, sheet 1 of 2;
* page 1: the same hole, its name printed with a leading zero (``HX-03``),
  sheet 2 of 2;
* page 2: hole HX-4, sheet 1 of 1;
* page 3: hole HX-5, sheet 1 of 2;
* page 4: HX-5's sheet 2 of 2, which prints no hole name at all.

The form is the one :mod:`planlens.testing.loggrid_fixtures` draws (its
column edges and headings), with each page's own depth ruler, layers and
driven samples. Every name and number here is made up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: (hole name printed in the header or None, sheet, first ruler tick,
#:  layers (top, description), samples (depth, blows, N)).
PageSpec = Tuple[Optional[str], str, float,
                 Sequence[Tuple[float, str]],
                 Sequence[Tuple[float, str, str]]]

#: The form number every page's title block prints first.
FORM_CODE = "QZ-7"

PAGES: Tuple[PageSpec, ...] = (
    ("HX-3", "Sheet 1 of 2", 0.0,
     [(0.0, "SANDY LEAN CLAY (CL), brown, stiff"),
      (9.0, "CLAYEY SAND (SC), gray, medium dense")],
     [(2.5, "3-5-6", "N=11"), (7.5, "4-6-7", "N=13"),
      (12.5, "6-8-9", "N=17"), (17.5, "5-7-9", "N=16")]),
    ("HX-03", "Sheet 2 of 2", 20.0,
     [(20.0, "CLAYEY SAND (SC), gray, medium dense"),
      (29.0, "LEAN CLAY (CL), olive, very stiff")],
     [(22.5, "7-9-11", "N=20"), (27.5, "8-10-12", "N=22"),
      (32.5, "6-9-10", "N=19")]),
    ("HX-4", "Sheet 1 of 1", 0.0,
     [(0.0, "SILTY SAND (SM), tan, loose"),
      (11.0, "POORLY GRADED SAND (SP), white, dense")],
     [(2.5, "2-3-3", "N=6"), (7.5, "3-4-4", "N=8"),
      (12.5, "11-14-17", "N=31")]),
    ("HX-5", "Sheet 1 of 2", 0.0,
     [(0.0, "FAT CLAY (CH), dark gray, soft"),
      (13.0, "SANDY SILT (ML), brown, stiff")],
     [(2.5, "1-1-2", "N=3"), (7.5, "1-2-2", "N=4"),
      (12.5, "3-4-5", "N=9"), (17.5, "4-5-6", "N=11")]),
    (None, "Sheet 2 of 2", 20.0,
     [(20.0, "SANDY SILT (ML), brown, stiff")],
     [(22.5, "5-6-8", "N=14"), (27.5, "6-7-9", "N=16")]),
)

#: The hole each page belongs to, by page.
HOLE_OF_PAGE: Dict[int, str] = {0: "HX-3", 1: "HX-3", 2: "HX-4", 3: "HX-5",
                                4: "HX-5"}

#: Points per foot of depth, and the ruler's tick step.
PER_FOOT = 22.0
TICK_STEP = 5.0


@dataclass
class MultiHoleGT:
    pdf: bytes
    pages: Tuple[PageSpec, ...]
    #: hole -> its pages, in order (pages of the whole PDF).
    holes: Dict[str, List[int]] = field(default_factory=dict)
    #: The PDF page the first log page is on (after any report text).
    first_log_page: int = 0

    @property
    def log_pages(self) -> List[int]:
        return [self.first_log_page + i for i in range(len(self.pages))]


#: Two pages of report text put before the logs by ``with_text=True``, then
#: an appendix tab, so the rules read the logs as an appendix of logs (with
#: no tab, a title block every page repeats reads as the running header of
#: a narrative).
TEXT_PAGES: Tuple[Tuple[str, Sequence[str]], ...] = (
    ("1.0 INTRODUCTION", (
        "This report presents the results of a subsurface exploration made "
        "for a synthetic test site.",
        "Three borings were drilled with a truck-mounted rig and sampled "
        "with a split spoon at regular intervals.",
        "The logs of the borings are given in the appendix that follows.")),
    ("2.0 SUBSURFACE CONDITIONS", (
        "The borings met lean clay and clayey sand over sand and silt.",
        "Groundwater was not measured during drilling.",
        "The conditions between the borings may differ from those shown.")),
)


def _draw_text_page(page, heading: str, lines: Sequence[str]) -> None:
    page.insert_text((72, 40), "SYNTHETIC SITE STUDY", fontsize=8)
    page.insert_text((72, 100), heading, fontsize=12, fontname="hebo")
    y = 130.0
    for paragraph in lines:
        words = paragraph.split()
        while words:
            row = ""
            while words and len(row) + len(words[0]) < 90:
                row = (row + " " + words.pop(0)).strip()
            page.insert_text((72, y), row, fontsize=10)
            y += 14.0
        y += 10.0


def _draw_page(page, spec: PageSpec) -> None:
    import pymupdf
    from planlens.testing.loggrid_fixtures import (
        EDGES, HEADERS, X_LEFT, X_RIGHT, Y_FORM_BOTTOM, Y_FORM_TOP,
        Y_HEADER_BOTTOM)

    hole, sheet, top, layers, samples = spec
    # -- the title block -------------------------------------------------
    page.insert_text((X_LEFT + 2, 40), f"BORING LOG   FORM {FORM_CODE}",
                     fontsize=13, fontname="hebo")
    if hole is not None:
        page.insert_text((X_LEFT + 2, 62), "Boring No.:", fontsize=9)
        page.insert_text((X_LEFT + 70, 62), hole, fontsize=9)
    page.insert_text((X_LEFT + 400, 62), sheet, fontsize=9)
    page.insert_text((X_LEFT + 2, 78), "PROJECT:  Synthetic Test Site",
                     fontsize=7)
    page.insert_text((X_LEFT + 250, 78), "Hammer Type:  Automatic",
                     fontsize=7)

    # -- the ruled form ----------------------------------------------------
    for x in EDGES:
        page.draw_line((x, Y_FORM_TOP), (x, Y_FORM_BOTTOM), width=0.6)
    for y in (Y_FORM_TOP, Y_HEADER_BOTTOM, Y_FORM_BOTTOM):
        page.draw_line((X_LEFT, y), (X_RIGHT, y), width=0.6)
    for i, head in enumerate(h.format(u="ft") for h in HEADERS):
        x0, x1 = EDGES[i], EDGES[i + 1]
        if i == 1:
            page.insert_text((x0 + 3, Y_HEADER_BOTTOM - 4), head, fontsize=6)
        else:
            page.insert_text(pymupdf.Point((x0 + x1) / 2.0 - 3,
                                           Y_HEADER_BOTTOM - 4),
                             head, fontsize=5.5, rotate=90)

    def y_of(depth: float) -> float:
        return Y_HEADER_BOTTOM + (depth - top) * PER_FOOT

    # -- the depth ruler ---------------------------------------------------
    tick = top + TICK_STEP
    while y_of(tick) < Y_FORM_BOTTOM - 10:
        page.insert_text((EDGES[2] + 6, y_of(tick) + 3), f"{tick:g}",
                         fontsize=8)
        tick += TICK_STEP

    # -- the layers and samples --------------------------------------------
    for depth, text in layers:
        if depth > top:
            page.draw_line((EDGES[1] + 2, y_of(depth)),
                           (EDGES[2] - 2, y_of(depth)), width=0.6)
            page.insert_text((EDGES[1] + 4, y_of(depth) - 1), f"{depth:g}",
                             fontsize=6.5)
        page.insert_text((EDGES[1] + 40, y_of(depth) + 9), text,
                         fontsize=7.5)
    for depth, blows, n_value in samples:
        page.insert_text((EDGES[3] + 6, y_of(depth) + 4), "SS", fontsize=7)
        page.insert_text((EDGES[4] + 6, y_of(depth) + 4), blows, fontsize=7)
        page.insert_text((EDGES[4] + 12, y_of(depth) + 13), n_value,
                         fontsize=7)


def build_multi_hole_log(pages: Sequence[PageSpec] = PAGES,
                         with_text: bool = False) -> MultiHoleGT:
    """The five pages as one PDF, with the hole each page belongs to;
    ``with_text`` puts :data:`TEXT_PAGES` before them, as a report would."""
    import pymupdf
    from planlens.testing.loggrid_fixtures import LETTER

    doc = pymupdf.open()
    first = 0
    if with_text:
        for heading, lines in TEXT_PAGES:
            _draw_text_page(doc.new_page(width=LETTER[0], height=LETTER[1]),
                            heading, lines)
        tab = doc.new_page(width=LETTER[0], height=LETTER[1])
        tab.insert_text((200, 360), "APPENDIX A", fontsize=24,
                        fontname="hebo")
        tab.insert_text((200, 400), "BORING LOGS", fontsize=18)
        first = len(TEXT_PAGES) + 1
    for spec in pages:
        _draw_page(doc.new_page(width=LETTER[0], height=LETTER[1]), spec)
    pdf = doc.tobytes()
    doc.close()
    holes: Dict[str, List[int]] = {}
    for page, hole in sorted(HOLE_OF_PAGE.items()):
        if page < len(pages):
            holes.setdefault(hole, []).append(first + page)
    return MultiHoleGT(pdf=pdf, pages=tuple(pages), holes=holes,
                       first_log_page=first)

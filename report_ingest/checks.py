"""The builders' check mode: refuse, as today, or flag (RI-87).

Each reader's ``_Builder`` turns the model's answer into record objects and
checks the values on the way. ``checks="refuse"``, the default, is today's
behaviour, unchanged. ``checks="flag"`` is model-first ingest's
(``module_work/REPORT_INGEST_PLAN.md`` section 8.6): nothing is silently
dropped, and the checks split into two kinds.

* **Impossible**: no real sheet prints it (a fraction outside 0 to 100, a
  water content or an Atterberg limit below 0, a liquid limit below the
  plastic limit, a grading in which more passes the smaller sieve, a
  negative depth, a depth past the lab reader's ``MAX_DEPTH_M``). The value
  stays out of its slot, so nothing downstream takes it as a reading, and
  its QA entry keeps it as read. The entry's words end with
  :data:`IMPOSSIBLE`, which ranks it ``open``
  (:func:`report_ingest.model.qa_rank`).
* **Doubtful**: a check of where the value sits on the paper (a log depth
  outside its sheet's ruler window, a water depth below the hole's total
  depth). The value stays in its slot at :data:`FLAG_CONFIDENCE`, with an
  ``out_of_range`` entry that says so (:func:`doubtful`), also ranked
  ``open``.

Under ``flag`` the pages one answer cites are read one way for the whole
answer (:func:`pdf_local`): as the report's pages, or, when every one is a
position in the pages the call was sent (0 to n-1) and at least one is not
a page of the call, all of them as positions. That is the class RI-74 fixed
for one-page printouts, for every stage. A page outside the call on the
first reading is handled as today, and an answer that cites both pages of
the call and pages outside it gets one note saying so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

__all__ = ["CHECKS", "FLAG_CONFIDENCE", "IMPOSSIBLE", "LISTED",
           "PageNumbers", "as_read", "check_mode", "doubtful", "pdf_local"]

#: The two modes: today's, and model-first ingest's.
CHECKS: Tuple[str, ...] = ("refuse", "flag")
#: What a doubtful value is worth: plan section 8.6, "kept at confidence 0.5".
FLAG_CONFIDENCE = 0.5
#: How the QA entry of a value kept out of its slot ends.
#: ``report_ingest.model`` ranks an ``out_of_range`` entry that says "listed
#: here as read" as ``open``: which reading is right is for a person to say.
LISTED = "kept out of the record and listed here as read"
#: How an impossible value's QA entry ends.
IMPOSSIBLE = f"no real sheet prints this, so it is {LISTED}"


def check_mode(checks: str) -> str:
    """``checks`` checked: one of :data:`CHECKS`."""
    if checks not in CHECKS:
        raise ValueError(f"checks must be one of {CHECKS}, not {checks!r}")
    return checks


def doubtful(why: str) -> str:
    """A doubtful value's QA words: why, and that it was kept."""
    return (f"{why}; kept at confidence {FLAG_CONFIDENCE:g} for review "
            f"rather than refused")


@dataclass(frozen=True)
class PageNumbers:
    """How the pages one answer cites are read (:func:`pdf_local`).

    ``positions``: every cited page is a position in ``pages``, the pages
    the call was sent, in order. ``mixed``: the QA note to list when the
    answer cites both pages of the call and pages outside it; empty when it
    does not.
    """

    pages: Tuple[int, ...]
    positions: bool = False
    mixed: str = ""

    def page(self, cited: int) -> Tuple[int, str]:
        """The report's page for ``cited``, and the note that says it was
        read as a position ("" when it is read as cited)."""
        cited = int(cited)
        if not (self.positions and 0 <= cited < len(self.pages)):
            return cited, ""
        page = self.pages[cited]
        return page, (f"the answer cited page {cited}, a position in the "
                      f"{len(self.pages)} page(s) it was sent; read as the "
                      f"report's page {page}")


def pdf_local(cited: Iterable[Optional[int]],
              pages: Sequence[int]) -> PageNumbers:
    """Plan section 8.4 step 1, decided once for one answer.

    The call's instruction maps each PDF page to the report's page index,
    and the model mostly gives the report's. When every page an answer cites
    is 0 to n-1 and at least one is not a page of the call, the answer gave
    positions in the PDF it was sent, and all its cites are read so: one
    value cannot be read one way and its neighbour the other. Otherwise each
    cite is the report's page. When the answer cites both pages of the
    call and pages outside it, :attr:`PageNumbers.mixed` says so and which
    reading was taken; when the call's pages also start below n, the
    numbers could mean either, and it says that too.
    """
    pages = tuple(int(p) for p in pages)
    n = len(pages)
    cites = sorted({int(c) for c in cited if c is not None})
    inside = [c for c in cites if c in pages]
    outside = [c for c in cites if c not in pages]
    positions = bool(outside) and all(0 <= c < n for c in cites)
    mixed = ""
    if inside and outside:
        either = (f", and with the call's pages {_listed(pages)} each could "
                  f"also be a position among the {n} pages sent"
                  if min(pages) < n else "")
        mixed = (
            f"the page numbers this answer cites mix this call's pages and "
            f"others: {_listed(inside)} {_are(inside)} among this call's "
            f"pages and {_listed(outside)} {_are(outside)} not{either}; "
            + ("all were read as positions" if positions
               else "they were read as the report's pages"))
    return PageNumbers(pages, positions, mixed)


def _listed(pages: Sequence[int]) -> str:
    return ", ".join(str(p) for p in pages)


def _are(pages: Sequence[int]) -> str:
    return "is" if len(pages) == 1 else "are"


def as_read(reading: Any) -> str:
    """A reading as compact JSON, for the QA entry of an object kept out of
    the record whole: what the model gave, its provenance and empty fields
    left out."""
    if hasattr(reading, "model_dump"):
        reading = reading.model_dump(mode="json", exclude={"prov"},
                                     exclude_defaults=True)
    return json.dumps(reading, ensure_ascii=False, separators=(", ", ": "))

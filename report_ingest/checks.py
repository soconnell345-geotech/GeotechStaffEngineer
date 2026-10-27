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

A depth below the ground surface is never negative (RI-123), in either mode:
:func:`surface` is the one rule every path that writes a depth asks (the
log floor, the log reader's builder and its merge, the DIGGS writer). A
depth that reads above the surface by no more than :data:`SURFACE_TOL_M` is
the surface itself, as a ruler fitted to the printed ticks reads it at the
top of a log, and is recorded as 0; one further above is refused under
``refuse`` and, under ``flag``, kept out of its slot and listed as read.
Elevations are not depths and have their own fields (an investigation's and
a water level's ``elevation``, a lab test's ``elevation``, a summary row's
``elevation_top``, a cone sounding whose ``vertical_axis`` is an elevation);
nothing here looks at them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["ABOVE", "CHECKS", "FLAG_CONFIDENCE", "IMPOSSIBLE", "LISTED",
           "SNAPPED", "SURFACE_TOL_M", "PageNumbers", "above_words",
           "as_read", "check_mode", "doubtful", "ground_investigation",
           "pdf_local", "snapped_words", "surface"]

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

#: How far above the ground surface, in metres, a depth below it may read
#: and still be taken for the surface itself (RI-123). Two thirds of the
#: scorer's own depth tolerance (``floor.DEPTH_TOL_M``, 0.15 m): a depth
#: read this close to 0 scores as a printed 0 either way, so setting it to 0
#: moves no score, while a depth further above the surface than the scorer
#: could absorb is a different reading, not the fit's noise. It is also what
#: planlens' own log grid takes for the top of a 1 m ruler (a tenth of a
#: step); it covers the fit's noise at the top of the logs of the saved
#: Foundry runs with room to spare, and stays short of the nearest reading
#: there that is misplaced rather than noisy.
SURFACE_TOL_M = 0.10
#: :func:`surface`'s verdicts: recorded as 0, or above the surface.
SNAPPED = "snapped"
ABOVE = "above"


def _metres(value: float, unit: str) -> float:
    """``value`` in metres; the number itself when its unit is unknown,
    which of feet and metres is the larger reading of it."""
    from report_ingest.model import Quantity

    converted = Quantity(value=float(value), unit=(unit or "").strip()).si_value
    return float(value) if converted is None else float(converted)


def surface(value: float, unit: str = "") -> Tuple[float, str]:
    """A depth below the ground surface against the surface (RI-123).

    ``(value, "")`` for a depth at or below it, as read; ``(0.0,``
    :data:`SNAPPED` ``)`` for one above it by no more than
    :data:`SURFACE_TOL_M`; ``(value,`` :data:`ABOVE` ``)`` for one further
    above. A unit the record cannot convert is read as metres, the larger
    of its two likely readings, so the tolerance is never looser than 0.1 m.
    """
    value = float(value)
    if value >= 0.0:
        return value, ""
    if -_metres(value, unit) <= SURFACE_TOL_M:
        return 0.0, SNAPPED
    return value, ABOVE


def _as_printed(value: float, unit: str) -> str:
    return f"{value:g} {unit}".strip() if unit else f"{value:g}"


def above_words(value: float, unit: str = "", water: bool = False) -> str:
    """Why a depth above the ground surface cannot be one below it. For a
    water level, also where a head above the surface belongs."""
    said = (f"depth {value:g} is negative: {_as_printed(abs(value), unit)} "
            f"above the ground surface, more than the {SURFACE_TOL_M:g} m a "
            f"ruler's fit may read there, and a depth below the surface "
            f"cannot be above it")
    if water:
        said += ("; a water level above the ground (an artesian head) is "
                 "an elevation, which the observation holds in its own slot")
    return said


def snapped_words(value: float, unit: str = "") -> str:
    """What was done with a depth read just above the surface."""
    return (f"depth {_as_printed(value, unit)} reads above the ground "
            f"surface by no more than {SURFACE_TOL_M:g} m, as a ruler's fit "
            f"reads the surface at the top of a log; recorded as 0, the "
            f"surface")


def _page_of(obj: Any, inv: Any) -> Optional[int]:
    prov = getattr(obj, "prov", None)
    if prov is not None and not isinstance(prov, list):
        return int(prov.page)
    pages = list(getattr(inv, "pages", None) or [])
    return int(pages[0]) if pages else None


def ground_investigation(inv: Any, *,
                         refused: str = "refused rather than accepted"
                         ) -> Tuple[List[Dict[str, Any]], int]:
    """Every depth of one investigation against the ground surface, in
    place (RI-123; :func:`surface`).

    A depth just above the surface is set to 0 and its object's provenance
    says so. One further above is taken out of its slot: a layer, sample or
    driven record whose TOP it is leaves the record whole (the record holds
    none without its top), listed with what was read; a base, a water,
    casing or caved depth, a pit's own depth or a dynamic probe's interval
    top, or a sounding's refusal depth, is emptied and its object kept; a
    sounding point leaves its series. A total depth is a header value, not a
    ruler's: any negative one is emptied. A cone sounding whose vertical
    axis is an elevation is not looked at, nor is any elevation.

    Returns the rows for the investigation's ``unresolved`` list (one per
    value taken out, ``refused_by`` python, its words ending ``refused``)
    and how many depths were set to 0.
    """
    unit = (getattr(inv, "depth_unit", "") or "").strip()
    name = (getattr(inv, "investigation_id", "") or "").strip()
    where = f"{name}: " if name else ""
    rows: List[Dict[str, Any]] = []
    snapped = 0

    def check(obj: Any, slot: str, what: str, top: bool = False,
              water: bool = False) -> bool:
        """``obj``'s depth in ``slot`` against the surface; False when it is
        a ``top`` refused, and ``obj`` must leave its list."""
        nonlocal snapped
        q = getattr(obj, slot, None)
        if q is None or q.value is None:
            return True
        q_unit = (q.unit or unit).strip()
        read = float(q.value)
        value, verdict = surface(read, q_unit)
        if verdict == SNAPPED:
            q.value = value
            prov = getattr(obj, "prov", None)
            # (_noted is defined below: a holder with a ``note`` gets it.)
            _noted(None if isinstance(prov, list) else prov,
                   snapped_words(read, q_unit))
            snapped += 1
            return True
        if verdict != ABOVE:
            return True
        row: Dict[str, Any] = {
            "what": f"{where}{what}", "page": _page_of(obj, inv),
            "value": read, "refused_by": "python"}
        if top:
            row["why"] = (f"{above_words(read, q_unit)}; the record holds "
                          f"none without its top, so it goes whole; "
                          f"{refused}")
            row["read"] = as_read(obj)
        else:
            row["why"] = f"{above_words(read, q_unit, water)}; {refused}"
            setattr(obj, slot, None)
        rows.append(row)
        return not top

    for noun, items, top, bottom in (
            ("layer", "layers", "top", "bottom"),
            ("sample", "samples", "top", "bottom"),
            ("driven record", "spt", "depth_top", "depth_bottom")):
        kept = []
        for n, obj in enumerate(getattr(inv, items, None) or []):
            label = (getattr(obj, "sample_id", "") or "").strip() \
                or str(n + 1)
            if not check(obj, top, f"{noun} {label} top", top=True):
                continue
            check(obj, bottom, f"{noun} {label} base")
            kept.append(obj)
        setattr(inv, items, kept)
    for n, water in enumerate(getattr(inv, "water", None) or []):
        for slot, what in (("depth", ""), ("casing_depth", " casing"),
                           ("caved_depth", " caved")):
            check(water, slot, f"water observation {n + 1}{what}",
                  water=slot == "depth")
    if getattr(inv, "pit", None) is not None:
        check(inv.pit, "depth", "pit depth")
    total = getattr(inv, "total_depth", None)
    if total is not None and total.value is not None \
            and float(total.value) < 0.0:
        rows.append({
            "what": f"{where}total depth", "page": None,
            "value": float(total.value),
            "why": f"a hole cannot have a negative total depth; {refused}",
            "refused_by": "python"})
        inv.total_depth = None
    cpt = getattr(inv, "cpt", None)
    if cpt is not None and cpt.vertical_axis == "depth":
        cpt.points = [p for n, p in enumerate(cpt.points)
                      if check(p, "depth", f"cone point {n + 1}", top=True)]
        check(cpt, "refusal_depth", "cone refusal depth")
    dcp = getattr(inv, "dcp", None)
    if dcp is not None:
        points = []
        for n, point in enumerate(dcp.points):
            if check(point, "depth", f"probe increment {n + 1}", top=True):
                check(point, "depth_top", f"probe increment {n + 1} top")
                points.append(point)
        dcp.points = points
        check(dcp, "refusal_depth", "probe refusal depth")
    return rows, snapped


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


def _span(pages: Sequence[int]) -> str:
    pages = list(pages)
    if len(pages) == 1:
        return f"page {pages[0]}"
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"pages {pages[0]}-{pages[-1]}"
    return "pages " + ", ".join(str(p) for p in pages)


def _on_call_pages(holders: Sequence[Any], pages: Sequence[int],
                   one_page: bool = False
                   ) -> Tuple[List[Dict[str, Any]], str]:
    """Each ``page`` the holders cite (a value's provenance, a citation, an
    unsettled row) read once for the whole answer against the pages of its
    own call, in place (:func:`~report_ingest.checks.pdf_local`); a holder
    with a ``note`` gets the reading's words. With ``one_page`` on a call of
    one page, every cite is that page (the calculation reader's RI-74 rule).

    Returns the entry saying the answer mixed the call's pages with others
    (none when it did not), and a line for the warnings when any page was
    read otherwise than cited ("" when none was)."""
    pages = [int(p) for p in pages]
    holders = [h for h in holders if getattr(h, "page", None) is not None]
    if one_page and len(pages) == 1:
        page = pages[0]
        moved = sorted({int(h.page) for h in holders if int(h.page) != page})
        for holder in holders:
            if int(holder.page) != page:
                _noted(holder, f"the answer cited page {holder.page}; its "
                               f"call carried page {page} alone, so it was "
                               f"read as page {page}")
                holder.page = page
        return [], (f"page {page}: the answer cited page(s) "
                    f"{', '.join(str(p) for p in moved)}; its call carried "
                    f"this page alone, so they were read as page {page}"
                    if moved else "")
    numbers = pdf_local([h.page for h in holders], pages)
    if numbers.positions:
        for holder in holders:
            holder.page, said = numbers.page(holder.page)
            _noted(holder, said)
    rows = ([{"what": "page numbers", "why": numbers.mixed,
              "flag": "page_numbers"}] if numbers.mixed else [])
    return rows, (f"{_span(pages)}: the answer cited its pages as positions "
                  f"in the {len(pages)} page(s) its call was sent; they were "
                  f"read as the report's pages" if numbers.positions else "")


def _noted(holder: Any, said: str) -> None:
    """``said`` added to a holder's note, where it has one."""
    if said and isinstance(getattr(holder, "note", None), str):
        holder.note = "; ".join(n for n in (holder.note.strip(), said) if n)

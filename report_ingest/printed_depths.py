"""Depths a log PRINTS, on the pages whose depth ruler could not be fitted.

WHY THIS EXISTS (BACKLOG RI-18). ``log_grid`` gives a row its depth by
reading the row's y through the depth ruler it fits on the page. Where no
ruler fits, no row has a depth: the floor seeds no sample and no layer, and
the log reader refuses every depth the model returns. A ruler fails to fit
for reasons that say nothing about whether the log states its depths:

- the log is not drawn to scale: a tabular log gives every row the same
  height whatever its depth, so its depth column fits no straight line;
- the depth column prints intervals (``1.50 - 3.00``) or numbers with a unit
  beside them, which are not the lone numbers a ruler is made of;
- the form's scale was never printed, and the depths are written against the
  samples instead (``S-1  2.5-4.0``);
- the page draws no ruled columns and its header names none, so nothing on
  it is placed at all.

In each case the depths are still on the paper, as text. This module reads
them, in order of trust, and nothing else:

1. ``depth_column``: a number or an interval in a column the form heads
   DEPTH, or a lone number set into the description's margin (which
   ``log_grid`` already names a depth cell). The form's own statement of the
   depth of the row it stands on.
2. ``sample_column``: an interval printed in a sample column after the
   sample's label, ``S-1  2.5-4.0``, or a depth after ``@``. The sample's own
   stated interval; but a sample column also carries labels that can look
   like an interval, so a pair of whole numbers with no label and no unit is
   not read.
3. ``description``: an interval at the head of a line of the description,
   ``0.0 - 1.2 m: FILL ...``. Prose, where a pair of numbers can also be a
   size or an amount, so one followed by a unit of size or a percentage is
   not read.
4. ``header``: the depth the header's groundwater or total-depth field
   prints, ``Groundwater encountered at 10 ft``, read only when the number
   carries its own unit. It places no row.
5. ``page_text``: an interval at the head of a line, on a page where the
   grid placed nothing at all. No column says what it is, so it is only ever
   support for a depth the reader returns, never seeded on its own.

WHAT IT NEVER DOES. It never reads a depth off a row's POSITION -- with no
ruler, a y is not a depth -- and it never interpolates between two printed
depths. A value is given a depth only when a printed depth stands on its own
line (:func:`on_line`), and a value with no printed depth beside it keeps no
depth. Depth grows down a log, so a printed number that breaks the downward
order of its column is not read as a depth: it is listed in ``dropped``.

Nothing here runs on a page whose ruler was fitted, so a log whose every page
has a ruler reads exactly as it did before this module existed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["PrintedDepth", "PrintedRow", "PrintedDepths", "printed_depths",
           "on_line", "TRUST", "SAME_DEPTH"]

#: How far each source is trusted, as the confidence its readings carry.
#: The number itself is exact as printed; what is uncertain is which row it
#: belongs to, and that is surest where the form prints it in a column of
#: its own and least sure in running text.
TRUST: Dict[str, float] = {
    "depth_column": 0.8, "sample_column": 0.7, "description": 0.6,
    "header": 0.5, "page_text": 0.4,
}

#: How the record says each depth was read, in words.
HOW: Dict[str, str] = {
    "depth_column": "in the depth column",
    "sample_column": "as the sample's own interval",
    "description": "at the head of the description",
    "header": "in the header",
    "page_text": "in the page's text",
}

#: The sources that place a ROW: a value on the same line as one of these is
#: at that depth. A range at the head of a description opens a stratum; it
#: says nothing about a sample printed beside it.
ROW_SOURCES = ("depth_column", "sample_column")

#: The sources that stand on a line of the log's columns. The header and a
#: page's loose text stand on none.
LINE_SOURCES = ("depth_column", "sample_column", "description")

#: The header fields that state a depth, as ``log_grid`` names them.
HEADER_DEPTHS = ("groundwater", "total_depth")

#: A number the reader returns is the printed one when it is within this of
#: it: the model copies a printed number, it does not measure one.
SAME_DEPTH = 0.005

_NUM = r"\d+(?:\.\d+)?"
#: The depth units a log prints beside a depth. Inches and millimetres are
#: the length of a drive or the size of a particle, never a depth, and are
#: deliberately not here.
_UNIT = r"(?:(?:feet|foot|ft|metres|meters|m)\b\.?|'|′)"
_DASH = r"(?:-|–|—|\bto\b)"
_RANGE = rf"({_NUM})\s*({_UNIT})?\s*{_DASH}\s*({_NUM})\s*({_UNIT})?"
_WHOLE_ONE = re.compile(rf"^\s*({_NUM})\s*({_UNIT})?\s*$", re.I)
_WHOLE_RANGE = re.compile(rf"^\s*\(?\s*{_RANGE}\s*\)?\s*$", re.I)
_SAMPLE_RANGE = re.compile(
    rf"^(?P<label>.*?)(?:^|(?<=[\s(@:,]))\(?\s*{_RANGE}\s*\)?\s*$", re.I)
_SAMPLE_AT = re.compile(rf"^(?P<label>.*?)@\s*({_NUM})\s*({_UNIT})?\s*$",
                        re.I)
_WITH_UNIT = re.compile(rf"({_NUM})\s*({_UNIT})", re.I)
_LEAD_RANGE = re.compile(
    rf"^\s*\(?\s*{_RANGE}\s*\)?\s*[:;,.]?\s*(?P<rest>[A-Za-z(].*)?$", re.I)
#: A pair of numbers followed by one of these is a size, an amount or a
#: pressure ("2-3 inch gravel", "1-2 percent organics"), not a stratum.
_NOT_A_DEPTH_WORD = frozenset({
    "in", "inch", "inches", "mm", "cm", "mesh", "percent", "pct", "blows",
    "bpf", "psi", "psf", "tsf", "ksf", "kpa", "mpa", "pcf", "times",
})


def _unit(token: Optional[str]) -> str:
    """A printed depth unit folded to ``ft`` or ``m``; ``""`` for none."""
    word = (token or "").strip().lower().rstrip(".")
    if not word:
        return ""
    return "ft" if word in ("'", "′", "ft", "feet", "foot") else "m"


@dataclass(frozen=True)
class PrintedDepth:
    """One depth, or one interval, exactly as the page prints it."""

    page: int
    top: float
    bottom: Optional[float]
    source: str
    text: str
    bbox: Tuple[float, float, float, float]
    column_id: str = ""
    unit: str = ""
    #: What the cell prints before the depth (a sample's label) or after it
    #: (the words of a description), as printed.
    label: str = ""
    rest: str = ""

    @property
    def confidence(self) -> float:
        return TRUST[self.source]

    @property
    def how(self) -> str:
        return HOW[self.source]

    def shows(self, value: float) -> bool:
        """Is ``value`` one of the numbers this reading prints?"""
        return any(v is not None and abs(float(value) - v) <= SAME_DEPTH
                   for v in (self.top, self.bottom))

    def to_dict(self) -> Dict[str, Any]:
        return {"page": self.page, "top": self.top, "bottom": self.bottom,
                "source": self.source, "text": self.text,
                "bbox": [round(v, 1) for v in self.bbox],
                "unit": self.unit or None, "confidence": self.confidence}


@dataclass
class PrintedRow:
    """The printed depths standing on one line of one page, most trusted
    first; ``top`` is the most trusted one's."""

    readings: List[PrintedDepth]

    @property
    def first(self) -> PrintedDepth:
        return self.readings[0]

    @property
    def page(self) -> int:
        return self.first.page

    @property
    def top(self) -> float:
        return self.first.top

    def sample_bottom(self, beside_words: bool) -> Optional[float]:
        """The base of a sample on this line, where the page prints it.

        The sample's own interval (``S-2  1.5-3.0``) gives it. So does an
        interval in the depth column -- but not on a line where a
        description stands too, since there the interval may as well be
        the stratum's; the base is then left unknown.
        """
        own = next((r.bottom for r in self.readings
                    if r.source == "sample_column" and r.bottom is not None
                    and abs(r.top - self.top) <= SAME_DEPTH), None)
        if own is not None:
            return own
        if self.first.source == "depth_column" and not beside_words:
            return self.first.bottom
        return None

    def stratum(self) -> Optional[PrintedDepth]:
        """The reading on this line that can mark a stratum: an interval at
        the head of the description, else the depth column's number."""
        return next((r for r in self.readings if r.source == "description"),
                    None) or next((r for r in self.readings
                                   if r.source == "depth_column"), None)

    @property
    def label(self) -> str:
        return next((r.label for r in self.readings if r.label), "")

    @property
    def places_values(self) -> bool:
        """Does this line place the samples and tests printed on it? Only a
        depth column or a sample's own interval does."""
        return self.first.source in ROW_SOURCES

    def note(self) -> str:
        return (f"no depth ruler on page {self.page}; the depth is the one "
                f"printed {self.first.how} on this line ('{self.first.text}')")


@dataclass
class PrintedDepths:
    """Every depth printed on the pages of one log that have no ruler."""

    #: This log's pages with no fitted ruler, in order. Empty on a log whose
    #: every page has one, and then there is nothing else here either.
    pages: List[int] = field(default_factory=list)
    readings: List[PrintedDepth] = field(default_factory=list)
    #: Numbers that look like depths and were NOT read as depths, with why.
    dropped: List[Dict[str, Any]] = field(default_factory=list)
    #: Page -> why its ruler was not fitted, in the grid's own words.
    reasons: Dict[int, str] = field(default_factory=dict)
    #: The depth unit printed with the depths or over the depth column.
    unit: str = ""
    unit_from: str = ""

    def __bool__(self) -> bool:
        return bool(self.readings)

    def on_page(self, page: int) -> List[PrintedDepth]:
        return [r for r in self.readings if r.page == page]

    def supporting(self, value: float, page: Optional[int]
                   ) -> Optional[PrintedDepth]:
        """The most trusted reading on ``page`` that prints ``value``."""
        if page is None or int(page) not in self.pages:
            return None
        hits = [r for r in self.on_page(int(page)) if r.shows(value)]
        return max(hits, key=lambda r: r.confidence) if hits else None

    def rows(self, page: int) -> List[PrintedRow]:
        """The page's printed depths gathered by the line of the log they
        stand on, in page order. The header and a page's loose text stand on
        no line of it."""
        rows: List[PrintedRow] = []
        here = sorted((r for r in self.on_page(page)
                       if r.source in LINE_SOURCES),
                      key=lambda r: (r.bbox[1], -r.confidence))
        for reading in here:
            home = next((row for row in rows
                         if on_line(reading.bbox, row.first)), None)
            if home is None:
                rows.append(PrintedRow([reading]))
            else:
                home.readings.append(reading)
                home.readings.sort(key=lambda r: -r.confidence)
        return rows

    def qa_rows(self, grid: Any, value_columns: Sequence[str]
                ) -> List[Dict[str, Any]]:
        """One QA row per page with no ruler: that the ruler was not fitted,
        why, what was read from print instead, and what still has no depth.

        ``value_columns`` are the column names whose cells are values of a
        sample or a test (the floor's sample family); they are what is
        counted as placed or not.
        """
        out: List[Dict[str, Any]] = []
        for page in self.pages:
            why = (f"the depth ruler on page {page} could not be fitted "
                   f"({self.reasons.get(page) or 'no reason was given'})")
            here = self.on_page(page)
            if here:
                counts: Dict[str, int] = {}
                for r in here:
                    counts[r.how] = counts.get(r.how, 0) + 1
                lines = [r for r in here if r.source in ROW_SOURCES]
                values = [c for c in getattr(grid, "rows", ())
                          if c.page == page and _names(grid, c)
                          & set(value_columns)]
                placed = [c for c in values
                          if any(on_line(c.bbox, r) for r in lines)]
                why += (f"; {len(here)} depth(s) were read from what the page "
                        f"prints instead ("
                        + ", ".join(f"{n} {how}" for how, n in counts.items())
                        + "), each marked with how it was read; "
                        f"{len(values) - len(placed)} of the {len(values)} "
                        f"sample and test value(s) on the page stand on no "
                        f"printed depth and have none")
            else:
                why += ("; the page prints no depth in a depth column, as a "
                        "sample's interval, at the head of a description or "
                        "in its header either, so no value on it has a depth")
            left = [d for d in self.dropped if d.get("page") == page]
            if left:
                why += (f"; {len(left)} printed number(s) were not read as "
                        f"depths: " + "; ".join(
                            f"'{d['text']}' {d['why']}" for d in left[:3])
                        + ("" if len(left) <= 3 else "; ..."))
            out.append({"what": f"depth scale, page {page}", "page": page,
                        "why": why, "refused_by": "no_ruler"})
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"pages": list(self.pages), "unit": self.unit or None,
                "unit_from": self.unit_from or None,
                "readings": [r.to_dict() for r in self.readings],
                "dropped": [dict(d) for d in self.dropped],
                "reasons": {str(k): v for k, v in self.reasons.items()}}


def on_line(bbox: Sequence[float], reading: PrintedDepth) -> bool:
    """Does a box stand on the same printed line as ``reading``?

    Its vertical centre lies inside the reading's own line, give or take a
    quarter of the line's height. A value one line lower is on another line
    and takes nothing from it.
    """
    y0, y1 = float(reading.bbox[1]), float(reading.bbox[3])
    slack = 0.25 * max(0.0, y1 - y0)
    centre = (float(bbox[1]) + float(bbox[3])) / 2.0
    return y0 - slack <= centre <= y1 + slack


# ---------------------------------------------------------------------------
# reading one cell
# ---------------------------------------------------------------------------

def _names(grid: Any, cell: Any) -> set:
    column = grid.column(cell.column_id) if hasattr(grid, "column") else None
    return set(column.names) if column is not None else set()


def _range(match: "re.Match[str]", base: int = 1
           ) -> Optional[Tuple[float, float, str]]:
    top, bottom = float(match.group(base)), float(match.group(base + 2))
    if not top < bottom:
        return None                       # an interval goes down the hole
    unit = _unit(match.group(base + 1)) or _unit(match.group(base + 3))
    return top, bottom, unit


def _depth_cell(text: str) -> Optional[Tuple[float, Optional[float], str]]:
    """A depth column's cell: one number or one interval, nothing else."""
    one = _WHOLE_ONE.match(text)
    if one:
        return float(one.group(1)), None, _unit(one.group(2))
    whole = _WHOLE_RANGE.match(text)
    if whole:
        return _range(whole)
    return None


def _sample_cell(text: str
                 ) -> Optional[Tuple[float, Optional[float], str, str]]:
    """A sample column's cell: a label, then its interval or ``@ depth``."""
    at = _SAMPLE_AT.match(text)
    if at:
        label = at.group("label").strip(" ,:(")
        return float(at.group(2)), None, _unit(at.group(3)), label
    match = _SAMPLE_RANGE.match(text)
    if not match:
        return None
    found = _range(match, base=2)
    if found is None:
        return None
    top, bottom, unit = found
    label = match.group("label").strip(" ,:(@")
    whole = "." not in match.group(2) and "." not in match.group(4)
    if whole and not unit and not re.search(r"[A-Za-z]", label):
        # "5-7" alone in a sample column: as likely a label as a depth.
        return None
    return top, bottom, unit, label


def _lead_cell(text: str) -> Optional[Tuple[float, float, str, str]]:
    """A line that opens with an interval: ``0.0 - 1.2 m: FILL, brown``."""
    match = _LEAD_RANGE.match(text)
    if not match:
        return None
    rest = (match.group("rest") or "").strip()
    first = (re.split(r"[\s,.;:()]+", rest.lower(), maxsplit=1)[0]
             if rest else "")
    if first in _NOT_A_DEPTH_WORD:
        return None
    found = _range(match)
    if found is None:
        return None
    top, bottom, unit = found
    return top, bottom, unit, rest


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------

def _downward(readings: List[PrintedDepth]) -> Tuple[List[PrintedDepth],
                                                     List[PrintedDepth]]:
    """The longest run of readings whose tops do not rise down the page, and
    the rest. Depth grows down a log; a number that breaks that order is a
    count, a label or an elevation that happens to stand in the column."""
    ordered = sorted(readings, key=lambda r: (r.bbox[1], r.bbox[0]))
    n = len(ordered)
    if n < 2:
        return ordered, []
    best, prev = [1] * n, [-1] * n
    for i in range(n):
        for j in range(i):
            if ordered[j].top <= ordered[i].top and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    end = max(range(n), key=lambda i: best[i])
    keep: List[int] = []
    while end >= 0:
        keep.append(end)
        end = prev[end]
    kept = set(keep)
    return ([ordered[i] for i in sorted(kept)],
            [ordered[i] for i in range(n) if i not in kept])


def _reason(grid: Any, page: int) -> str:
    """Why the grid fitted no ruler on ``page``, in its own words."""
    said = [str(w) for w in getattr(grid, "warnings", ()) or ()
            if str(w).startswith(f"page {page}:")]
    for marker in ("no text to place", "no columns could be laid out",
                   "no column holds"):
        hit = next((w for w in said if marker in w), None)
        if hit:
            return hit.split(":", 1)[1].strip()
    hit = next((w for w in said if "no depth ruler" in w), None)
    return hit.split(":", 1)[1].strip() if hit else ""


def printed_depths(grid: Any, pages: Sequence[int],
                   lines: Optional[Dict[int, Sequence[Any]]] = None
                   ) -> PrintedDepths:
    """Every depth printed on the pages of this log that have no ruler.

    ``lines`` maps a page to its text lines; it is read only on a page where
    the grid placed nothing, and only for ``page_text`` support. Nothing is
    read on a page that has a ruler.
    """
    rulers = getattr(grid, "rulers", None) or {}
    unruled = [int(p) for p in pages if int(p) not in rulers]
    result = PrintedDepths(pages=unruled)
    if not unruled:
        return result
    found: Dict[Tuple[int, str, str], List[PrintedDepth]] = {}

    def add(reading: PrintedDepth) -> None:
        key = (reading.page, reading.source, reading.column_id)
        found.setdefault(key, []).append(reading)

    for cell in getattr(grid, "rows", ()) or ():
        if cell.page not in unruled:
            continue
        text = " ".join(str(cell.text or "").split())
        names = _names(grid, cell)
        box = tuple(float(v) for v in cell.bbox)
        common = dict(page=int(cell.page), text=text, bbox=box,
                      column_id=str(cell.column_id))
        # A column headed DEPTH, unless it is headed ELEVATION too (its
        # numbers could be either); or a lone number the grid found set
        # into the description's margin, which it names a depth cell.
        is_depth = ("depth" in names and "elevation" not in names) or (
            getattr(cell, "column", "") == "depth" and "depth" not in names)
        if is_depth:
            got = _depth_cell(text)
            if got is not None:
                add(PrintedDepth(top=got[0], bottom=got[1], unit=got[2],
                                 source="depth_column", **common))
                continue
        if names & {"sample_id", "sample_type"}:
            got = _sample_cell(text)
            if got is not None:
                add(PrintedDepth(top=got[0], bottom=got[1], unit=got[2],
                                 label=got[3], source="sample_column",
                                 **common))
                continue
        if "description" in names:
            got = _lead_cell(text)
            if got is not None:
                add(PrintedDepth(top=got[0], bottom=got[1], unit=got[2],
                                 rest=got[3], source="description", **common))
    boxes = getattr(grid, "field_boxes", None) or {}
    for key in HEADER_DEPTHS:
        raw = " ".join(str((getattr(grid, "fields", None) or {}).get(key)
                           or "").split())
        hit = _WITH_UNIT.search(raw)
        if not hit:
            continue                 # a number with no unit is not read here
        page, box = boxes.get(key, (unruled[0], (0.0, 0.0, 0.0, 0.0)))
        if int(page) not in unruled:
            continue
        add(PrintedDepth(page=int(page), top=float(hit.group(1)), bottom=None,
                         unit=_unit(hit.group(2)), source="header", text=raw,
                         bbox=tuple(float(v) for v in box), column_id=key,
                         label=key.replace("_", " ")))
    placed = {int(c.page) for c in getattr(grid, "rows", ()) or ()}
    for page in unruled:
        if page in placed:
            continue
        for ln in (lines or {}).get(page, ()) or ():
            text = " ".join(str(getattr(ln, "text", "") or "").split())
            got = _lead_cell(text)
            if got is not None:
                add(PrintedDepth(page=page, top=got[0], bottom=got[1],
                                 unit=got[2], rest=got[3], source="page_text",
                                 text=text,
                                 bbox=tuple(float(v) for v in ln.bbox)))
    log_unit = str(getattr(grid, "unit", None) or "").strip()
    for (page, source, _column), readings in sorted(found.items()):
        kept, dropped = _downward(readings)
        for r in dropped:
            result.dropped.append({
                "page": page, "text": r.text, "source": source,
                "why": "breaks the downward order of its column, so it is "
                       "not read as a depth"})
        for r in kept:
            if log_unit and r.unit and r.unit != log_unit:
                # The log's ruled sheets read in one unit and this number is
                # printed in another; nothing here converts, so it is left.
                result.dropped.append({
                    "page": page, "text": r.text, "source": source,
                    "why": f"printed in {r.unit} where the log's ruled "
                           f"sheets read {log_unit}"})
            else:
                result.readings.append(r)
    result.readings.sort(key=lambda r: (r.page, r.bbox[1], -r.confidence))
    for page in unruled:
        result.reasons[page] = _reason(grid, page)
    # The unit: the one printed beside the depths or over the depth column.
    # Where the page prints two, it is left unknown rather than chosen.
    units = {r.unit for r in result.readings if r.unit}
    heads = {_unit(str(c.unit)) for c in getattr(grid, "columns", ()) or ()
             if c.page in unruled and "depth" in (c.names or ()) and c.unit}
    heads.discard("")
    if len(units | heads) == 1:
        result.unit = (units | heads).pop()
        result.unit_from = ("printed beside the depths" if units
                            else "the depth column's heading")
    elif len(units | heads) > 1:
        result.unit_from = ("the page prints both "
                            + " and ".join(sorted(units | heads))
                            + ", so the unit is left unknown")
    return result

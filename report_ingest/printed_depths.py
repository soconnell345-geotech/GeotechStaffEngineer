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
  samples instead (``S-1  2.5-4.0 ft``);
- the page draws no ruled columns and its header names none, so nothing on
  it is placed at all.

In each case the depths are still on the paper, as text. This module reads
them, in order of trust, and nothing else:

1. ``depth_column``: a number or an interval in a column the form heads
   DEPTH, or a lone number set into the description's margin (which
   ``log_grid`` already names a depth cell). The form's own statement of the
   depth of the row it stands on.
2. ``sample_column``: an interval printed after a sample's label,
   ``S-1  2.5-4.0 ft``, or a depth after ``@``. A sample column also prints
   blow records that look like an interval (``S-1 6-8``), so the depth is
   read only where a unit is printed with it or the column is headed DEPTH
   as well.
3. ``description``: an interval at the head of a line of the description,
   ``0.0 - 1.2 m: FILL ...``. Prose, where a pair of numbers can also be a
   size, a thickness, a count or an amount (``3-4 feet of fill``, ``2-3
   cobbles``, ``10-12 N``), so one followed by such a word is not read; and
   one is read only where it prints a depth unit of its own or CHAINS with
   the page's other intervals there (its top another's base, or its base
   another's top), as a stratum's interval does.
4. ``header``: the depth the header's groundwater or total-depth field
   prints, ``Groundwater encountered at 10 ft``: the number the unit is
   bound to, never the first number on the line, which may be a date, a
   time or a year. It places no row.
5. ``page_text``: an interval at the head of a line, on a page where the
   grid placed nothing at all; prose, read by the same rules as a
   description's.

WHAT IT NEVER DOES. It never reads a depth off a row's POSITION -- with no
ruler, a y is not a depth -- and it never interpolates between two printed
depths. A value is given a depth only when a printed depth stands on its own
line (:func:`on_line`). A number labelled an elevation (EL, Elev., datum,
NGVD, NAVD, msl) and a bare year are never read as depths.

PLACING AND CHECKING. A depth places the values printed on its line only
when its column can be trusted to be a run of depths: a depth or sample
column whose numbers rise all the way down the page. A column with any
number out of order (depths interleaved with elevations or blow counts), a
lone number in the margin, and loose page text are kept to CHECK a depth the
reader returns, and place nothing. A depth the reader returns is backed only
by a printed number of the same kind (a sample's depth by the depth or
sample column, a stratum's by the depth column or a description, a water
level by the header's groundwater field) and in the same unit.

Nothing here runs on a page whose ruler was fitted, so a log whose every page
has a ruler reads exactly as it did before this module existed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["PrintedDepth", "PrintedRow", "PrintedDepths", "printed_depths",
           "on_line", "chained", "fold_unit", "TRUST", "SAME_DEPTH",
           "SERVES"]

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

#: The sources that are prose, where an interval may be a thickness, a count
#: or an amount: one stands as a depth only with a unit of its own, or where
#: it chains with the page's other intervals from the same source.
PROSE_SOURCES = ("description", "page_text")

#: The header fields that state a depth, as ``log_grid`` names them.
HEADER_DEPTHS = ("groundwater", "total_depth")

#: What each kind of depth the reader returns may stand on. A total depth is
#: a header value the gate does not check, so the total-depth field backs
#: nothing here; it is read for the floor.
SERVES: Dict[str, Tuple[str, ...]] = {
    "layer": ("depth_column", "description", "page_text"),
    "sample": ("depth_column", "sample_column", "page_text"),
    "water": ("header:groundwater",),
}

#: A number the reader returns is the printed one when it is within this of
#: it: the model copies a printed number, it does not measure one.
SAME_DEPTH = 0.005

_NUM = r"\d+(?:\.\d+)?"
#: The depth units a log prints beside a depth. Inches and millimetres are
#: the length of a drive or the size of a particle, never a depth, and are
#: deliberately not here.
_UNIT = r"(?:(?:feet|foot|ft|metres|meters|metre|meter|m)\b\.?|'|′)"
_DASH = r"(?:-|–|—|\bto\b)"
_RANGE = rf"({_NUM})\s*({_UNIT})?\s*{_DASH}\s*({_NUM})\s*({_UNIT})?"
_WHOLE_ONE = re.compile(rf"^\s*({_NUM})\s*({_UNIT})?\s*$", re.I)
_WHOLE_RANGE = re.compile(rf"^\s*\(?\s*{_RANGE}\s*\)?\s*$", re.I)
_SAMPLE_RANGE = re.compile(
    rf"^(?P<label>.*?)(?:^|(?<=[\s(@:,]))\(?\s*{_RANGE}\s*\)?\s*$", re.I)
_SAMPLE_AT = re.compile(rf"^(?P<label>.*?)@\s*({_NUM})\s*({_UNIT})?\s*$",
                        re.I)
_LEAD_RANGE = re.compile(
    rf"^\s*\(?\s*{_RANGE}\s*\)?\s*(?P<sep>[:;,.])?\s*"
    rf"(?P<rest>[A-Za-z(].*)?$", re.I)
#: A number bound to a depth unit: ``12.5 ft``. Not the tail of a date, a
#: time or a range (``3/4/2019``, ``8:30``, ``10-12``), which is no depth.
_BOUND = re.compile(rf"(?<![\d/:.\-])({_NUM})\s*({_UNIT})", re.I)
#: An elevation label just before a number, or a datum just after its
#: unit (``98.5 ft msl``, ``98.5 ft NAVD88``). An ``(EL. ...)`` after a
#: depth labels the number that follows it, not the depth.
_ELEVATION_BEFORE = re.compile(
    r"\b(?:el|elev|elevation|datum)\b\.?\s*[:=]?\s*$", re.I)
_ELEVATION_AFTER = re.compile(
    r"^\s*\(?\s*(?:msl|amsl|asl|datum|ngvd\s*\d*|navd\s*\d*)\b", re.I)
#: A four-figure whole number from 1900 to 2099 printed with no unit.
_YEAR = re.compile(r"(?<![\d.])(?:19|20)\d\d(?![\d.])")
#: A pair of numbers followed by one of these is a size, a thickness, an
#: amount, a blow count, a pressure or an elevation ("2-3 inch gravel", "2
#: to 4 ft thick lenses", "3-4 feet of fill", "1-2 percent organics", "10-12
#: N"), not a stratum. A ``%`` after the pair is not read either: the line
#: must go on with a word.
_NOT_A_DEPTH_WORD = frozenset({
    "in", "inch", "inches", "mm", "cm", "mesh", "percent", "pct", "blow",
    "blows", "bpf", "n", "of", "psi", "psf", "tsf", "ksf", "kpa", "mpa",
    "pcf", "times", "thick", "thickness", "wide", "width", "long", "length",
    "diameter", "dia", "el", "elev", "elevation", "msl", "amsl", "asl",
    "ngvd", "navd", "datum",
})
#: Things a log counts. A pair of numbers run straight into one of these is
#: a count ("2-3 cobbles") or a size ("2-3 ft boulders"). The same word can
#: also name a stratum, so after a colon or a comma (``0.0 - 1.5 m:
#: COBBLES and BOULDERS``) it is read as the stratum's words.
_COUNTED = frozenset({
    "cobbles", "boulders", "pebbles", "stones", "pieces", "fragments",
    "nodules", "lenses", "seams", "layers",
})


def fold_unit(token: Optional[str]) -> str:
    """A depth unit folded to ``ft`` or ``m``; ``""`` for none, and any
    other word lower-cased, so it matches neither."""
    word = (token or "").strip().lower().rstrip(".")
    if not word:
        return ""
    if word in ("'", "′", "ft", "feet", "foot"):
        return "ft"
    if word in ("m", "metre", "metres", "meter", "meters"):
        return "m"
    return word


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
    #: The unit printed with this number, folded; ``""`` when none is.
    unit: str = ""
    #: What the cell prints before the depth (a sample's label) or after it
    #: (the words of a description), as printed.
    label: str = ""
    rest: str = ""
    #: Whether this depth places the values on its line (see the module
    #: notes); when False it only checks a depth the reader returns.
    seeds: bool = True
    #: For an interval read from prose: whether it chains with the page's
    #: other intervals from the same source (see :func:`chained`).
    chains: bool = False

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

    def serves(self, kind: str) -> bool:
        """Can this reading back a depth of ``kind`` (``layer``,
        ``sample``, ``water``)?"""
        key = (f"header:{self.column_id}" if self.source == "header"
               else self.source)
        return key in SERVES.get(kind, ())

    def to_dict(self) -> Dict[str, Any]:
        return {"page": self.page, "top": self.top, "bottom": self.bottom,
                "source": self.source, "text": self.text,
                "bbox": [round(v, 1) for v in self.bbox],
                "unit": self.unit or None, "confidence": self.confidence,
                "places_rows": self.seeds}


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

    @property
    def label(self) -> str:
        return next((r.label for r in self.readings if r.label), "")

    @property
    def places_values(self) -> bool:
        """Does this line place the samples and tests printed on it? Only a
        depth column or a sample's own interval does."""
        return self.first.source in ROW_SOURCES

    def sample_bottom(self, beside_words: bool) -> Optional[float]:
        """The base of a sample on this line, where the page prints it.

        The sample's own interval (``S-2  1.5-3.0 ft``) gives it. So does an
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
    #: Page -> the depth units it prints: with its depths, or over its depth
    #: column.
    page_units: Dict[int, List[str]] = field(default_factory=dict)
    #: The one depth unit these pages print, or ``""``.
    unit: str = ""
    unit_from: str = ""

    def __bool__(self) -> bool:
        return bool(self.readings)

    def on_page(self, page: int) -> List[PrintedDepth]:
        return [r for r in self.readings if r.page == page]

    def unit_on(self, reading: PrintedDepth) -> Optional[str]:
        """The unit of ``reading``: the one printed with it, else the one
        its page prints; ``""`` when the page prints none, and None when the
        page prints two and none is bound to this number."""
        if reading.unit:
            return reading.unit
        units = self.page_units.get(reading.page, [])
        if not units:
            return ""
        return units[0] if len(units) == 1 else None

    def _fits(self, reading: PrintedDepth, unit: str) -> bool:
        printed = self.unit_on(reading)
        if printed is None:
            return False
        return printed == "" or printed == fold_unit(unit)

    def supporting(self, value: float, page: Optional[int], unit: str,
                   kind: str) -> Optional[PrintedDepth]:
        """The most trusted reading on ``page`` that prints ``value`` in
        ``unit`` and can back a depth of ``kind``."""
        if page is None or int(page) not in self.pages:
            return None
        hits = [r for r in self.on_page(int(page)) if r.shows(value)
                and r.serves(kind) and self._fits(r, unit)]
        return max(hits, key=lambda r: r.confidence) if hits else None

    def why_not(self, value: float, page: Optional[int], unit: str,
                kind: str) -> str:
        """Why no printed number backs ``value``, for the refusal."""
        lead = f"no depth ruler could be fitted on page {page}"
        shown = [r for r in self.on_page(int(page)) if r.shows(value)] \
            if page is not None else []
        if not shown:
            left = next((d for d in self.dropped if d.get("page") == page
                         and any(v is not None
                                 and abs(float(value) - v) <= SAME_DEPTH
                                 for v in (d.get("top"), d.get("bottom")))),
                        None)
            if left is not None:
                return (f"{lead}; it prints {value:g} only in "
                        f"'{left['text']}', which was not read as a depth "
                        f"({left['why']}); refused rather than accepted")
            return (f"{lead} and nothing printed on it reads {value:g}, so "
                    f"the depth stands on nothing on the page; refused "
                    f"rather than accepted")
        r = shown[0]
        if not any(s.serves(kind) for s in shown):
            return (f"{lead}; it prints {value:g} only {r.how} "
                    f"('{r.text}'), which is not a {kind}'s depth; refused "
                    f"rather than accepted")
        r = next(s for s in shown if s.serves(kind))
        printed = self.unit_on(r)
        return (f"{lead}; it prints {value:g} "
                + (f"in {printed}" if printed else "with no unit it can be "
                   "read in, as the page prints two")
                + f" ('{r.text}'), and the reader gave "
                f"{fold_unit(unit) or 'no unit'}; refused rather than "
                f"accepted")

    def rows(self, page: int, unit: Optional[str] = None,
             placing: bool = True) -> List[PrintedRow]:
        """The page's depths gathered by the line of the log they stand on,
        in page order: only the PLACING ones unless ``placing`` is False,
        and with ``unit`` only those in that unit. The header and a page's
        loose text stand on no line of it."""
        rows: List[PrintedRow] = []
        here = sorted((r for r in self.on_page(page)
                       if r.source in LINE_SOURCES
                       and (r.seeds or not placing)
                       and (unit is None or self.unit_on(r) == unit)),
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
        """One QA row per page with no ruler, for the engineer: that the
        ruler was not fitted and why, which depths were read from print
        instead and need checking against the log, and what has none.

        ``value_columns`` are the column names whose cells are values of a
        sample or a test (the floor's sample family); they are what is
        counted as placed or not. The row is flagged ``no_ruler``; it is a
        gap in the reading, not a refusal.
        """
        out: List[Dict[str, Any]] = []
        for page in self.pages:
            why = (f"No depth ruler could be fitted on page {page} "
                   f"({self.reasons.get(page) or 'no reason was given'}).")
            here = self.on_page(page)
            if here:
                counts: Dict[str, int] = {}
                for r in here:
                    counts[r.how] = counts.get(r.how, 0) + 1
                why += (f" Check these {len(here)} depth(s) against the "
                        f"printed log: they were read from what the page "
                        f"prints ("
                        + ", ".join(f"{n} {how}" for how, n in counts.items())
                        + ") and no scale checked them.")
                lines = [r for row in self.rows(page) for r in row.readings
                         if r.source in ROW_SOURCES]
                values = [c for c in getattr(grid, "rows", ())
                          if c.page == page and _names(grid, c)
                          & set(value_columns)]
                if values:
                    placed = [c for c in values
                              if any(on_line(c.bbox, r) for r in lines)]
                    why += (f" {len(values) - len(placed)} of the "
                            f"{len(values)} sample and test value(s) on the "
                            f"page stand on no printed depth and were left "
                            f"without one.")
                checking = [r for r in here if not r.seeds]
                if checking:
                    why += (f" {len(checking)} of the depths, from a column "
                            f"whose numbers do not rise down the page, the "
                            f"margin or loose text, only check the reader's "
                            f"depths and place nothing.")
            else:
                why += (" The page prints no depth in a depth column, as a "
                        "sample's interval, at the head of a description or "
                        "in its header either, so nothing on it has a depth;"
                        " check the printed log for depths written another "
                        "way.")
            left = [d for d in self.dropped if d.get("page") == page]
            if left:
                why += (f" Not read as depths: " + "; ".join(
                    f"'{d['text']}' ({d['why']})" for d in left[:3])
                    + ("" if len(left) <= 3 else "; ...") + ".")
            out.append({"what": f"depth scale, page {page}", "page": page,
                        "why": why, "flag": "no_ruler"})
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"pages": list(self.pages), "unit": self.unit or None,
                "unit_from": self.unit_from or None,
                "page_units": {str(k): list(v)
                               for k, v in self.page_units.items()},
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
    unit = fold_unit(match.group(base + 1)) or fold_unit(match.group(base + 3))
    return top, bottom, unit


def _depth_cell(text: str) -> Optional[Tuple[float, Optional[float], str]]:
    """A depth column's cell: one number or one interval, nothing else."""
    one = _WHOLE_ONE.match(text)
    if one:
        return float(one.group(1)), None, fold_unit(one.group(2))
    whole = _WHOLE_RANGE.match(text)
    if whole:
        return _range(whole)
    return None


def _sample_cell(text: str, in_depth_column: bool = False
                 ) -> Optional[Tuple[float, Optional[float], str, str]]:
    """A sample column's cell: a label, then its interval or ``@ depth``.

    A sample column prints blow records that look like intervals
    (``S-1 6-8``), so the depth is read only when a unit is printed with
    it, or when the column is headed DEPTH as well.
    """
    at = _SAMPLE_AT.match(text)
    if at:
        label = at.group("label").strip(" ,:(")
        unit = fold_unit(at.group(3))
        if not (unit or in_depth_column):
            return None
        return float(at.group(2)), None, unit, label
    match = _SAMPLE_RANGE.match(text)
    if not match:
        return None
    label = match.group("label").strip(" ,:(@")
    if _ELEVATION_BEFORE.search(label):
        return None
    found = _range(match, base=2)
    if found is None:
        return None
    top, bottom, unit = found
    if not (unit or in_depth_column):
        return None
    if not unit and not re.search(r"[A-Za-z]", label) \
            and "." not in match.group(2) and "." not in match.group(4):
        return None          # "5-7" alone: as likely a label as a depth
    return top, bottom, unit, label


def _lead_cell(text: str) -> Optional[Tuple[float, float, str, str]]:
    """A line that opens with an interval: ``0.0 - 1.2 m: FILL, brown``."""
    match = _LEAD_RANGE.match(text)
    if not match:
        return None
    rest = (match.group("rest") or "").strip()
    first = (re.split(r"[\s,.;:()]+", rest.lower(), maxsplit=1)[0]
             if rest else "")
    if first in _NOT_A_DEPTH_WORD or \
            first.rstrip("0123456789") in _NOT_A_DEPTH_WORD:
        return None
    if first in _COUNTED and not match.group("sep"):
        return None
    found = _range(match)
    if found is None:
        return None
    top, bottom, unit = found
    return top, bottom, unit, rest


def _header_depth(raw: str) -> Optional[Tuple[float, str]]:
    """The depth a header field prints: the first number BOUND to a depth
    unit that is not labelled an elevation. ``After 24 hrs 12.5 ft`` reads
    12.5; ``3/4/2019 at 8 ft`` reads 8; ``EL. 98.5 ft`` reads nothing."""
    for match in _BOUND.finditer(raw):
        if _ELEVATION_BEFORE.search(raw[:match.start()]) \
                or _ELEVATION_AFTER.match(raw[match.end():]):
            continue
        return float(match.group(1)), fold_unit(match.group(2))
    return None


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------

def chained(intervals: Sequence[PrintedDepth]) -> List[bool]:
    """Which of these intervals chain: each one whose top is another's
    base, or whose base is another's top. A stratum's interval ends where
    the next one's begins; an interval that meets no other may as well be a
    seam, a lens, a thickness or an amount. A seam printed between two
    strata does not break their chain."""
    def meets(a: PrintedDepth, b: PrintedDepth) -> bool:
        return a.bottom is not None and b.bottom is not None and (
            abs(b.bottom - a.top) <= SAME_DEPTH
            or abs(a.bottom - b.top) <= SAME_DEPTH)
    return [any(meets(a, b) for j, b in enumerate(intervals) if j != i)
            for i, a in enumerate(intervals)]


def _rises(readings: List[PrintedDepth]) -> bool:
    """Do these readings' tops rise (or hold) all the way down the page?
    Depth grows down a log; a column where any number goes back up is
    interleaved with something else -- elevations, blow counts -- and its
    numbers cannot be told apart by order alone."""
    ordered = sorted(readings, key=lambda r: (r.bbox[1], r.bbox[0]))
    return all(a.top <= b.top for a, b in zip(ordered, ordered[1:]))


def _is_year(reading: PrintedDepth) -> bool:
    return not reading.unit and bool(_YEAR.search(reading.text)) and any(
        v is not None and float(v).is_integer() and 1900 <= v <= 2099
        for v in (reading.top, reading.bottom))


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
    the grid placed nothing, and only for ``page_text``. Nothing is read on
    a page that has a ruler.
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
        # into the description's margin, which it names a depth cell but
        # which may as well be an elevation, so it only checks.
        in_depth = "depth" in names and "elevation" not in names
        margin = getattr(cell, "column", "") == "depth" and "depth" not in names
        if in_depth or margin:
            got = _depth_cell(text)
            if got is not None:
                add(PrintedDepth(top=got[0], bottom=got[1], unit=got[2],
                                 source="depth_column", seeds=not margin,
                                 **common))
                continue
        if names & {"sample_id", "sample_type"}:
            got = _sample_cell(text, in_depth_column=in_depth)
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
        got = _header_depth(raw)
        if got is None:
            continue
        page, box = boxes.get(key, (unruled[0], (0.0, 0.0, 0.0, 0.0)))
        if int(page) not in unruled:
            continue
        add(PrintedDepth(page=int(page), top=got[0], bottom=None,
                         unit=got[1], source="header", text=raw,
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
                                 text=text, seeds=False,
                                 bbox=tuple(float(v) for v in ln.bbox)))

    # An interval read from prose stands as a depth only where it prints a
    # unit of its own or chains with the page's other intervals from the
    # same source; an interval standing alone with no unit is as likely a
    # count or a thickness, so it is set aside before anything else.
    for page in unruled:
        for source in PROSE_SOURCES:
            keys = [k for k in found if k[0] == page and k[1] == source]
            every = [r for k in keys for r in found[k]]
            links = {id(r): link for r, link in zip(every, chained(every))}
            for key in keys:
                kept_here: List[PrintedDepth] = []
                for r in found[key]:
                    if links[id(r)]:
                        kept_here.append(replace(r, chains=True))
                    elif r.unit:
                        kept_here.append(r)
                    else:
                        result.dropped.append({
                            "page": page, "text": r.text, "source": source,
                            "top": r.top, "bottom": r.bottom,
                            "why": "an interval in prose with no unit of its "
                                   "own that chains with no other, which may "
                                   "be a count or a thickness"})
                found[key] = kept_here

    # The units each page prints: bound to its depths, or over its depth
    # column. A number printed with no unit takes its page's, if one.
    for page in unruled:
        units = {r.unit for (p, _s, _c), rs in found.items() if p == page
                 for r in rs if r.unit}
        units |= {fold_unit(str(c.unit))
                  for c in getattr(grid, "columns", ()) or ()
                  if c.page == page and "depth" in (c.names or ())
                  and "elevation" not in (c.names or ()) and c.unit}
        units.discard("")
        result.page_units[page] = sorted(units)

    log_unit = fold_unit(str(getattr(grid, "unit", None) or ""))
    for (page, source, _column), readings in sorted(found.items()):
        kept: List[PrintedDepth] = []
        for r in readings:
            if _is_year(r):
                result.dropped.append({
                    "page": page, "text": r.text, "source": source,
                    "top": r.top, "bottom": r.bottom,
                    "why": "reads as a year, not a depth"})
            elif log_unit and r.unit and r.unit != log_unit:
                # The log's ruled sheets read in one unit and this number is
                # printed in another; nothing here converts, so it is left.
                result.dropped.append({
                    "page": page, "text": r.text, "source": source,
                    "top": r.top, "bottom": r.bottom,
                    "why": f"printed in {r.unit} where the log's ruled "
                           f"sheets read {log_unit}"})
            else:
                kept.append(r)
        if source in LINE_SOURCES and not _rises(kept):
            # Out of order somewhere: every number in the column only
            # checks, since order cannot say which of them are depths.
            kept = [replace(r, seeds=False) for r in kept]
        result.readings.extend(kept)
    result.readings.sort(key=lambda r: (r.page, r.bbox[1], -r.confidence))
    for page in unruled:
        result.reasons[page] = _reason(grid, page)

    every = sorted({u for units in result.page_units.values() for u in units})
    bound = {r.unit for r in result.readings if r.unit}
    if len(every) == 1:
        result.unit = every[0]
        result.unit_from = ("printed beside the depths" if bound
                            else "the depth column's heading")
    elif len(every) > 1:
        result.unit_from = ("the pages print both " + " and ".join(every)
                            + ", so the unit is left unknown")
    return result

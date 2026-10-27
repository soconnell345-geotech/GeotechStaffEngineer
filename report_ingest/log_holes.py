"""Which hole each log page names, and one log work item per hole (RI-117).

planlens groups consecutive log pages into one work item, and it starts a new
item where the first exploration-shaped code in a page's title block changes.
On many printed forms that code is not the hole's name: it is the number of a
standard or of the form, a field and its value, a column heading, the same on
every page. So consecutive logs of DIFFERENT holes became one item, test pits
and borings alike (BACKLOG RI-117). The log reader returns one hole per item,
so the other holes were never in the record, and the floor folded their rows
into the first hole.

This module reads the hole name a log page prints in its HEADER, the way an
engineer finds it on the sheet:

* a LABELLED name: a field such as "Boring No.:", "Test Pit Number:",
  "Borehole:", "ID:" with the name beside it, under it, or after it on the
  same line; or a title that names the hole, "LOG OF BORING B-1",
  "BORING NO. B-1", "TEST PIT TP-M", or prints it beside it on its
  baseline ("Log of Boring" and "B-1");
* failing that, and only on a page with no hole field at all, a name
  STANDING ALONE: the topmost line of the header band that is nothing but a
  hole-shaped code ("QX-1"), as a scanned form with no field labels prints
  it at the head of its log column; never a sample's or a core run's number
  ("S-1", "RUN-2"), a page part ("SHEET 2"), a month, a code of four digits
  or one written closed up with letters after its number (a logo).

A page whose hole field gives no readable name (left blank on a
continuation sheet, or "SHEET 1" read beside it) names none: the first
sample's number at the head of its log table is not a hole. A page whose
labelled fields name two different holes (a key sheet, a page that refers
to a neighbouring hole) names none. Within one item, a name that stands
alone counts only when it is of the series the item's labelled names are
(or, with no labelled name in the item, the series of its first named
page), and names standing alone that go up one a page ("A-3", "A-4") are
plate numbers, not holes.

:func:`split_log_items` then cuts each log item at every page whose header
names a hole other than the current part's. A continuation sheet that prints
the same hole, written with or without its leading zeros ("B-1", "B-01",
:func:`report_ingest.reconciler.hole_key`) or with "#" before its number
("B # 1"), or no hole at all, stays with its hole. Nothing here calls a
model, and nothing here is model first: today's reader and the floor use it
(``report_ingest.log_reader``), and the graph lists the holes a record lacks
that a field or a title prints (:func:`missing_holes_qa`).

LAYOUT JUDGMENT: the header and footer bands, the label-to-value gap and the
words that are never a hole's prefix below are read off log forms, not taken
from any standard. Measured on the corpus's forms: a value stands 15 to 50 pt
to the right of its label; a hole name sits in the top fifth of the page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from report_ingest.reconciler import hole_key

__all__ = [
    "LOG_ITEM_KINDS", "HEADER_BAND", "FOOTER_BAND", "VALUE_GAP_PT",
    "HoleName", "page_hole", "lines_hole", "item_holes", "hole_parts",
    "split_log_items", "other_hole_pages", "names_match",
    "missing_holes_qa",
]

#: The work-item kinds the log reader reads. A sounding is a series and is
#: read by its own reader; its items are not split here.
LOG_ITEM_KINDS: Tuple[str, ...] = ("boring_log", "test_pit_log")

#: The share of the page, from the top, read as the header. LAYOUT JUDGMENT.
HEADER_BAND = 0.25
#: The share of the page, from the bottom, read as a title block at the foot
#: (labelled names only: a gINT log prints "BORING NUMBER" there).
FOOTER_BAND = 0.12
#: How far right of its label a value may stand, in points.
VALUE_GAP_PT = 80.0
#: How far below its label a value set under it may stand, in points, and
#: how far off the label's left edge when the two do not overlap across
#: (a value centred in its cell under a label set at the cell's left
#: overlaps it). Measured on the corpus's header tables: up to 10 pt below.
VALUE_BELOW_PT = 12.0
VALUE_INDENT_PT = 12.0

#: Between a series and its number: a dash, a space or nothing ("B-1",
#: "QX- 1", "QX 1", "QX7"), or "#" or "No." ("QX # 01", "QX No. 1").
_JOIN = r"(?:\s?[-–_]\s?|\s?#\s?|\s?NO\.\s?|\sNO\s|\s)?"
#: A series, which may itself be in parts ("QV-K-QX-04").
_SERIES = r"(?:[A-Z]{1,6}[-–_]){0,2}[A-Z]{1,6}"
#: A hole's code, upper case: letters and a number ("B-1", "QX- 1",
#: "QX-01", "QX7A", "QX # 01"), letters and letters ("TP-M"), or a bare
#: number.
_LETTERED = rf"{_SERIES}{_JOIN}\d{{1,4}}[A-Z]{{0,2}}"
_LETTERS = r"[A-Z]{1,5}\s?[-–]\s?[A-Z]{1,2}"
#: A bare number, not the first half of a decimal comma ("41,7").
_NUMBER = r"\d{1,4}[A-Z]?(?!,\d)"
#: Where a code ends: not inside a longer word or number, and not followed
#: by a second number ("QK 31-207", "4-MAR-21", "B-1/B-2" name no one hole).
_END = r"(?![A-Z0-9./–-])"
_CODE = rf"(?:{_LETTERED}|{_LETTERS}|{_NUMBER})"
_CODE_AT_START = re.compile(rf"^\s*({_CODE}){_END}")
_CODE_TO_END = re.compile(rf"({_CODE})\s*")
#: A name standing alone is joined by a dash or written closed up ("QX-1",
#: "QX- 1", "B12"): a word and a number set apart by a space ("START 7"),
#: or by "#" or "No." ("JOB # 12"), is a field, not a name.
_CODE_ALONE = re.compile(
    r"^\s*([A-Z]{1,5}(?:\s?[-–_]\s?)?\d{1,4}[A-Z]{0,2})\s*$")
#: "#" or "No." between a series and its number, left out of the key:
#: "QX # 01" and "QX-1" are one hole.
_JOIN_WORD = re.compile(r"(?<=[A-Z])\s*(?:#|NO\.|\sNO\s)\s*(?=\d)")
#: What, after a bare number, makes it a measure or a count, not a hole:
#: "4 in", "96 mm", "1 of 2".
_MEASURE_AFTER = re.compile(
    r"\s*(?:of\s+\d|(?:in|inch(?:es)?|mm|cm|m|ft|feet|met(?:er|re)s?)\b"
    r"|[\"'%])", re.I)
#: Where a word or a number starts, for reading a title's last code.
_WORD_START = re.compile(r"(?<![A-Z0-9])(?=[A-Z0-9])")
#: What a header may print after a hole's name: "(continued)", "cont'd".
_CARRIED = re.compile(
    r"\s*\(?\s*(?<![A-Za-z])cont(?:inued|inuation|'?d)?\.?\s*\)?\s*$", re.I)

#: The words that name a hole on a log, lower case. LAYOUT JUDGMENT, in the
#: languages the corpus is in.
_HOLE_WORDS = frozenset((
    "boring", "test boring", "borehole", "bore hole", "drill hole",
    "drillhole", "test hole", "hole", "test pit", "trial pit", "pit",
    "exploration", "exploration point", "log of boring", "log of borehole",
    "log of test pit", "log of test boring", "boring log", "borehole log",
    "test pit log", "test boring log", "sondage", "forage", "puits",
    "sondeo", "calicata", "perforacion", "bohrung", "schurf"))
#: The words that follow a hole word in a label and say "its name".
_NAME_WORDS = frozenset(("no", "nr", "num", "number", "id", "name"))
#: Short labels that name a hole only with a colon or a name word after
#: them: "ID:", "TP No.".
_SHORT_LABELS = frozenset(("id", "tp", "bh"))
#: The hole words a title may print with the name set beside it, on its
#: baseline, and no name word ("Log of Boring" and "QX-14" beside it). Not
#: "hole" or "pit" alone: "Hole" beside "4 in" is a size.
_TITLE_WORDS = _HOLE_WORDS - {"hole", "pit", "exploration"}
#: Words that head a page part, not a hole ("SHEET 2", "FIG 3", "N 18", an
#: SPT blow count, "ELEV 125"), whether they stand alone or beside a hole
#: field. LAYOUT JUDGMENT.
_NOT_A_HOLE = frozenset((
    "SHEET", "PAGE", "PG", "FIG", "FIGURE", "PLATE", "TABLE", "APP", "REV",
    "DWG", "SEC", "NO", "NR", "N", "EL", "ELEV"))
#: Series that, standing alone, are a sample's or a core run's number at
#: the head of the log table ("S-1", "SS-3", "ST-2", "U-1", "D-4", "C-1",
#: "R-2", "RUN-1"), or a month in a date ("MAR-21"), not a hole. LAYOUT
#: JUDGMENT.
_NOT_ALONE = frozenset((
    "S", "SS", "ST", "U", "D", "C", "R", "RUN",
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "SEPT",
    "OCT", "NOV", "DEC"))
#: The most digits a name standing alone has: "QX-1234" is a project's or
#: a drawing's number. LAYOUT JUDGMENT.
ALONE_DIGITS = 3


@dataclass(frozen=True)
class HoleName:
    """A hole's name as one page's header prints it."""

    #: As printed, upper case, each run of spaces one space ("QX- 1").
    text: str
    #: :func:`report_ingest.reconciler.hole_key` of it: "B-01" is "B1".
    key: str
    page: int
    #: ``label`` (a field or a title names it) or ``alone`` (it stands on
    #: a line of its own in the header band).
    how: str

    @property
    def series(self) -> str:
        """The letters the name starts with ("QX" of "QX-01")."""
        found = re.match(r"[A-Z]*", self.text.upper())
        return found.group(0) if found else ""


def _flat(text: Any) -> str:
    return " ".join(str(text or "").split())


#: How far off level a line may run and still be read as set level: a scan
#: is fed through crooked, and its text runs a degree or two off.
SKEW_DEG = 5.0


def _upright(line: Any) -> bool:
    rotation = getattr(line, "rotation", 0.0)
    if rotation is None:
        return True
    turn = float(rotation) % 360.0
    return turn <= SKEW_DEG or turn >= 360.0 - SKEW_DEG


def _label_words(text: str) -> Tuple[str, bool]:
    """``text`` as a label: its hole words, lower case, and whether a name
    word ("No.", "Number", "#", "ID") followed them."""
    flat = text.lower()
    flat = re.sub(r"n\s*[°º]|#", " no ", flat)
    flat = re.sub(r"[^a-zà-ÿ ]+", " ", flat)
    flat = flat.replace("ó", "o")
    words = flat.split()
    named = False
    while len(words) > 1 and words[-1] in _NAME_WORDS:
        words.pop()
        named = True
    return " ".join(words), named


def _is_label(text: str, colon: bool) -> bool:
    """Does ``text`` (what stands before a colon, or before the code on a
    title line) name a hole field?"""
    words, named = _label_words(text)
    if words in _SHORT_LABELS:
        return colon or named
    return words in _HOLE_WORDS


def _series_of(code: str) -> str:
    found = re.match(r"[A-Z]*", code.upper())
    return found.group(0) if found else ""


def _key(code: Any) -> str:
    """:func:`hole_key` of a printed name, "#" or "No." before its number
    left out ("QX # 01" is "QX1")."""
    return hole_key(_JOIN_WORD.sub("", str(code or "").upper()))


#: A word that heads a telephone line, with what may follow it ("Tel.",
#: "Tel:", "Phone", "Fax", "Mobile"). "Ph" and "Cell" are also hole-like
#: ("PH-1", "CELL-1"), so they head a telephone line only with a colon or a
#: full stop, or with a number of five digits or more after them.
_PHONE_WORD = re.compile(
    r"^\s*(TEL(?:EPHONE)?|PHONE|FAX|MOBILE|PH|CELL)(?=[\s.:\-–_\d]|$)"
    r"(?P<rest>.*)$")
#: A number shaped like a telephone number: a leading "+" and a country code,
#: or three digit groups or more set apart by spaces, dots, dashes or
#: brackets ("(555) 010-0199", "555.010.0199"). LAYOUT JUDGMENT.
_PHONE_SHAPE = re.compile(
    r"^\s*(?:\+\s?\d{1,3}[\s.\-()]*\d"
    r"|\(?\d{2,4}\)?(?:[\s.\-]\s?\(?\d{2,4}\)?){2,}(?![\d]))")


def _is_phone(text: str) -> bool:
    """Is ``text`` a telephone or fax line: it follows a word that heads one
    ("Tel", "Phone", "Fax", "Mobile", "Cell", "Ph.") or it is shaped like a
    telephone number? Neither is a hole's name."""
    upper = _flat(text).upper()
    if _PHONE_SHAPE.match(upper):
        return True
    found = _PHONE_WORD.match(upper)
    if not found:
        return False
    word, rest = found.group(1), found.group("rest")
    if word not in ("PH", "CELL"):
        return True
    return rest[:1] in (".", ":") or len(re.findall(r"\d", rest)) >= 5


def _code_at_start(text: str) -> Optional[str]:
    """The hole code a field's value starts with: not a page part ("SHEET
    1", "ELEV 125", :data:`_NOT_A_HOLE`), and not a bare number that is a
    measure or a count ("4 in", "1 of 2")."""
    if _is_phone(text):
        return None
    upper = _CARRIED.sub("", text.upper())
    found = _CODE_AT_START.match(upper)
    if not found:
        return None
    code = _flat(found.group(1))
    if _series_of(code) in _NOT_A_HOLE:
        return None
    if code[:1].isdigit() and _MEASURE_AFTER.match(upper[found.end():]):
        return None
    return code


def _code_alone(text: str) -> Optional[str]:
    """A line that is nothing but a hole-shaped code, kept only when it is
    not a page part, a sample or run number, or a month
    (:data:`_NOT_A_HOLE`, :data:`_NOT_ALONE`), has at most
    :data:`ALONE_DIGITS` digits, and is not written closed up with letters
    after its number ("QRS6TU", a logo's or a machine's name read as text;
    "QX-6A" is kept)."""
    if _is_phone(text):
        return None
    found = _CODE_ALONE.match(_CARRIED.sub("", text.upper()))
    if not found:
        return None
    code = _flat(found.group(1))
    series = _series_of(code)
    if not series or series in _NOT_A_HOLE or series in _NOT_ALONE:
        return None
    number = re.search(r"\d+", code)
    if number is None or len(number.group(0)) > ALONE_DIGITS:
        return None
    if code[len(series):][:1].isdigit() and number.end() < len(code):
        return None
    return code


def _same_line(label: Any, other: Any) -> bool:
    a, b = label.bbox, other.bbox
    height = max(1.0, min(a[3] - a[1], b[3] - b[1]))
    return min(a[3], b[3]) - max(a[1], b[1]) >= 0.4 * height


def _value_beside(label: Any, lines: Sequence[Any], below_too: bool = True,
                  numbers: bool = True) -> Optional[str]:
    """The hole code a label line points at: the nearest line to its right
    on its baseline, else (``below_too``) the nearest line set under it, at
    its left margin or overlapping it across (the leftmost on a tie).
    Whichever line is nearest is the value, and it names a hole only when it
    is a code: a field left blank does not reach past the next label ("Elev.:
    125") to take that label's value. ``numbers`` False takes no bare
    number."""
    right: Optional[Tuple[float, float, Any]] = None
    below: Optional[Tuple[float, float, Any]] = None
    a = label.bbox
    for other in lines:
        if other is label or not _upright(other) or not _flat(other.text):
            continue
        b = other.bbox
        if _same_line(label, other) and b[0] >= a[2] - 1.0:
            gap = b[0] - a[2]
            if gap <= VALUE_GAP_PT and (right is None or gap < right[0]):
                right = (gap, b[0], other)
        elif below_too and (abs(b[0] - a[0]) <= VALUE_INDENT_PT
                            or min(a[2], b[2]) > max(a[0], b[0])) \
                and -2.0 <= b[1] - a[3] <= VALUE_BELOW_PT:
            gap = b[1] - a[3]
            if below is None or (round(gap, 1), b[0]) < (round(below[0], 1),
                                                         below[1]):
                below = (gap, b[0], other)
    for hit in (right, below):
        if hit is None:
            continue
        code = _code_at_start(hit[2].text)
        if code is not None and (numbers or not code[:1].isdigit()):
            return code
    return None


def _title_code(text: str) -> Optional[str]:
    """The hole a title line names at its end ("LOG OF BORING B-4",
    "BORING NO. QX- 1", "TEST PIT TP-M"): the last code on the line that a
    hole label stands right before. A bare number counts only after a name
    word or "log of" ("BORING NO. 3", "LOG OF BORING 3"), never after a
    hole word alone ("PIT 3" may be a size)."""
    upper = text.upper()
    starts = [m.start() for m in _WORD_START.finditer(upper)]
    for start in reversed(starts[-6:]):
        if not _CODE_TO_END.fullmatch(upper[start:]):
            continue
        before = text[:start].rstrip(" .-#:–")
        # A label prints no number of its own: "Pit: 1" before "of 2" is a
        # field and its value.
        if not before or re.search(r"\d", before) \
                or not _is_label(before, colon=False):
            continue
        code = _flat(upper[start:])
        if _series_of(code) in _NOT_A_HOLE:
            continue
        words, named = _label_words(before)
        if not code.isdigit() or named or words.startswith("log of"):
            return code
    return None


#: A field written on one line: "Boring No.: B-1   Sheet 1 of 2".
_PAIR = re.compile(r"([^:]{1,40}?)\s*:\s*([^:]*?)(?=\s{2,}[^:\s][^:]{0,40}:|$)")


def _labelled(line: Any, lines: Sequence[Any]) -> Tuple[List[str], bool]:
    """Every hole code this line labels (usually none, at most a few), and
    whether it is a hole FIELD, whatever its value.

    A field with a colon ("Boring No.: B-1", "ID:" with the name beside it),
    then, in each stretch of the line set apart by two spaces or more, a
    title that names its hole at its end ("LOG OF BORING B-4", "BORING NO.
    B-1"), then a label standing alone with a name word ("BORING NUMBER")
    and the name beside it or under it, then a title with no name word
    ("Log of Boring", "Test Pit Log", "Boring") and the name beside it on
    its baseline. The fields are the colon labels and the labels with a
    name word: a page whose field gives no readable name names no hole
    (:func:`lines_hole`); a title with nothing beside it is only a title.
    """
    raw = str(line.text or "").strip()
    if not raw:
        return [], False
    found: List[str] = []
    field = False
    for pair in _PAIR.finditer(raw):
        key, value = pair.group(1), pair.group(2)
        if not _is_label(key, colon=True):
            continue
        field = True
        code = (_code_at_start(value) if value.strip()
                else _value_beside(line, lines))
        if code:
            found.append(code)
    if found or field:
        return found, field
    for stretch in re.split(r"\s{2,}", raw):
        code = _title_code(_CARRIED.sub("", _flat(stretch)))
        if code:
            found.append(code)
    if found or ":" in raw:
        return found, field
    words, named = _label_words(_CARRIED.sub("", _flat(raw)))
    if named and (words in _HOLE_WORDS or words in _SHORT_LABELS):
        code = _value_beside(line, lines)
        return ([code] if code else []), True
    if not named and words in _TITLE_WORDS:
        # A bare number beside a title only after "log of", as in
        # :func:`_title_code`.
        code = _value_beside(line, lines, below_too=False,
                             numbers=words.startswith("log of"))
        return ([code] if code else []), False
    return [], False


def lines_hole(lines: Sequence[Any], height: float, page: int
               ) -> Tuple[Optional[HoleName], List[str]]:
    """The hole a page's header names, from its text lines, and the names
    its labelled fields gave when they gave more than one (then none is
    taken). ``height`` is the page's height in the lines' own frame."""
    upright = [ln for ln in lines if _upright(ln) and _flat(ln.text)]
    if not upright:
        return None, []
    tall = max([float(height or 0.0)]
               + [float(ln.bbox[3]) for ln in upright])
    head = HEADER_BAND * tall
    foot = (1.0 - FOOTER_BAND) * tall
    ordered = sorted(upright, key=lambda ln: (float(ln.bbox[1]),
                                              float(ln.bbox[0])))
    banded = [ln for ln in ordered
              if float(ln.bbox[1]) < head or float(ln.bbox[1]) >= foot]
    named: Dict[str, str] = {}
    field = False
    for line in banded:
        codes, is_field = _labelled(line, banded)
        field = field or is_field
        for code in codes:
            key = _key(code)
            if key and key not in named:
                named[key] = code
    if len(named) == 1:
        (key, text), = named.items()
        return HoleName(text=text, key=key, page=int(page), how="label"), []
    if len(named) > 1:
        return None, list(named.values())
    if field:
        # The page has a hole field and its text gives no name for it (left
        # blank on a continuation sheet, or not read): what stands alone in
        # its header is then something else, a sample's number or a logo.
        return None, []
    for line in ordered:
        if float(line.bbox[1]) >= head:
            break
        code = _code_alone(line.text)
        if code:
            return HoleName(text=code, key=_key(code), page=int(page),
                            how="alone"), []
    return None, []


def _numbers_the_pages(alone: Sequence[HoleName]) -> bool:
    """Do these names, each standing alone on its page, number the pages
    rather than the holes? One letter and a number ("A-3", "A-4"), going up
    one a page across two pages or more: a plate or appendix page number.
    One-page logs of one-letter holes that print no field label are then
    read as one item, as they were before RI-117. LAYOUT JUDGMENT."""
    if len(alone) < 2:
        return False
    offsets = set()
    for hole in alone:
        found = re.fullmatch(r"([A-Z])\s?[-–_]?\s?(\d{1,4})", hole.text)
        if found is None:
            return False
        offsets.add((found.group(1), int(found.group(2)) - hole.page))
    return len(offsets) == 1


def page_hole(doc: Any, page: int) -> Optional[HoleName]:
    """The hole one log page's header names, or None (:func:`lines_hole`)."""
    try:
        content = doc.page(int(page))
    except Exception:                        # a page that will not read
        return None
    found, _several = lines_hole(list(content.lines),
                                 float(getattr(content, "height", 0.0)
                                       or 0.0), int(page))
    return found


def item_holes(doc: Any, pages: Iterable[int],
               lines: Optional[Mapping[int, Tuple[Sequence[Any], float]]]
               = None) -> Dict[int, Optional[HoleName]]:
    """``page -> the hole its header names`` over one item's pages, with
    the series rule (module docstring) applied. ``lines``, when the caller
    has read the pages, maps a page to ``(its text lines, its height)``."""
    pages = [int(p) for p in pages]
    found: Dict[int, Optional[HoleName]] = {}
    for page in pages:
        if lines is not None and page in lines:
            page_lines, height = lines[page]
            found[page], _several = lines_hole(page_lines, height, page)
        else:
            found[page] = page_hole(doc, page)
    labelled = {h.series for h in found.values()
                if h is not None and h.how == "label"}
    if not labelled and _numbers_the_pages(
            [h for h in found.values() if h is not None]):
        return {page: None for page in found}
    if not labelled:
        first = next((h for h in found.values() if h is not None), None)
        labelled = {first.series} if first is not None else set()
    return {page: (h if h is None or h.how == "label"
                   or h.series in labelled else None)
            for page, h in found.items()}


def hole_parts(pages: Sequence[int],
               names: Mapping[int, Optional[HoleName]]
               ) -> List[Tuple[List[int], Optional[HoleName]]]:
    """``pages`` cut at every page whose header names a hole other than the
    current part's, each part with the hole it names first. Pages naming no
    hole stay with the part they follow; pages before the first named one
    belong to that first hole. A name :func:`names_match` takes for the
    part's own ("1" in a field, then "B-1" in a title when the form changes
    on the second sheet) does not cut."""
    parts: List[Tuple[List[int], Optional[HoleName]]] = []
    for page in pages:
        hole = names.get(int(page))
        if not parts:
            parts.append(([int(page)], hole))
            continue
        run, current = parts[-1]
        if hole is not None and current is not None \
                and not names_match(hole.text, current.text):
            parts.append(([int(page)], hole))
            continue
        run.append(int(page))
        if current is None and hole is not None:
            parts[-1] = (run, hole)
    return parts


def split_log_items(doc: Any, items: Sequence[Any]) -> List[Any]:
    """Every work item, with each log item cut into one item per hole.

    Items of every other kind, and a log item whose pages name one hole or
    none, pass through untouched and in order. A cut item's parts keep its
    kind and evidence, take ids of the form ``<id>a``, ``<id>b`` ... (as
    :func:`report_ingest.calc_reader.split_calc_items` names its parts), and
    are titled with the hole their pages name.
    """
    out: List[Any] = []
    for item in items:
        pages = [int(p) for p in (getattr(item, "pages", ()) or ())]
        if getattr(item, "kind", "") not in LOG_ITEM_KINDS or len(pages) < 2:
            out.append(item)
            continue
        parts = hole_parts(pages, item_holes(doc, pages))
        if len(parts) <= 1:
            out.append(item)
            continue
        evidence = dict(getattr(item, "evidence", {}) or {})
        for n, (run, hole) in enumerate(parts):
            suffix = chr(ord("a") + n) if n < 26 else f"_{n + 1}"
            out.append(item.__class__(
                id=f"{item.id}{suffix}", kind=item.kind, pages=list(run),
                title=(hole.text if hole is not None
                       else getattr(item, "title", None)),
                evidence={**evidence, "first_page": run[0],
                          "split_from": item.id,
                          "hole": hole.text if hole is not None else None,
                          "split_reason": "the page's header names a hole "
                                          "other than the pages before it "
                                          "(RI-117)"}))
    return out


def other_hole_pages(pages: Sequence[int],
                     names: Mapping[int, Optional[HoleName]]
                     ) -> Tuple[Optional[HoleName], Dict[int, HoleName]]:
    """The hole an item's pages name first, and every page whose header
    names another one (by :func:`names_match`, as :func:`hole_parts`
    cuts)."""
    own = next((names.get(int(p)) for p in pages
                if names.get(int(p)) is not None), None)
    if own is None:
        return None, {}
    return own, {int(p): names[int(p)] for p in pages
                 if names.get(int(p)) is not None
                 and not names_match(names[int(p)].text, own.text)}


#: A name's first series part and what follows it, when what follows is a
#: name of its own ("QV-" of "QV-K-QX-04").
_LEADING_PART = re.compile(r"^[A-Z]{1,6}[-–_]\s?(?=[A-Z][^\d]*\d)")


def _tails(text: Any) -> List[str]:
    """``text`` and the shorter names its leading series parts drop to:
    "QV-K-QX-04", "K-QX-04", "QX-04"."""
    upper = _flat(text).upper()
    out = [upper]
    while True:
        found = _LEADING_PART.match(upper)
        if found is None:
            return out
        upper = upper[found.end():]
        out.append(upper)


def names_match(printed: Any, recorded: Any) -> bool:
    """Is the hole a page prints the one a record names? The same name with
    leading zeros folded (:func:`hole_key`); one written with leading series
    parts the other drops ("QV-K-QX-04" and "QX-4": a site's or an area's
    prefix); or a bare number printed under a hole label against the same
    number with a series ("3" and "B-3")."""
    a, b = _key(printed), _key(recorded)
    if not a or not b:
        return False
    if a == b or a in {_key(t) for t in _tails(recorded)} \
            or b in {_key(t) for t in _tails(printed)}:
        return True
    short, long_ = sorted((a, b), key=len)
    return short.isdigit() and long_.endswith(short) and \
        long_[:-len(short)].isalpha()


def _pages(pages: Sequence[int]) -> str:
    """``[3, 4, 5, 9]`` as "pages 3-5, 9", ``[3]`` as "page 3"."""
    pages = sorted({int(p) for p in pages})
    runs: List[str] = []
    for page in pages:
        if runs and page == int(runs[-1].split("-")[-1]) + 1:
            runs[-1] = f"{runs[-1].split('-')[0]}-{page}"
        else:
            runs.append(str(page))
    return ("pages " if len(pages) > 1 else "page ") + ", ".join(runs)


def missing_holes_qa(doc: Any, items: Sequence[Any], record: Any) -> list:
    """A ``partial`` QA entry for each log item whose page headers print,
    in a field or a title, hole names no log in the record carries: the
    reader may have read them under other names, or not read them. One
    entry per item, naming each hole and the pages that print it.

    A name that stands alone in a header (no field or title gives it) is
    not listed: it is as often a sample's or a plate's number as a hole's.
    """
    from report_ingest.model import QAEntry

    recorded = [inv.investigation_id for inv in
                getattr(record, "investigations", ()) or ()
                if inv.investigation_id]
    out: List[Any] = []
    for item in items:
        if getattr(item, "kind", "") not in LOG_ITEM_KINDS:
            continue
        pages = [int(p) for p in (getattr(item, "pages", ()) or ())]
        printed: Dict[str, Tuple[str, List[int]]] = {}
        for page, hole in item_holes(doc, pages).items():
            if hole is None or hole.how != "label":
                continue
            printed.setdefault(hole.key, (hole.text, []))[1].append(page)
        lacking = [(text, where) for key, (text, where) in printed.items()
                   if not any(names_match(text, name) for name in recorded)]
        if not lacking:
            continue
        where = sorted({p for _text, at in lacking for p in at})
        many = len(lacking) > 1
        span = _pages(where)
        out.append(QAEntry(
            kind="partial", where=f"items.{item.kind}.{item.id}",
            detail=(f"{span} {'print' if len(where) > 1 else 'prints'} the "
                    f"hole name{'s' if many else ''} "
                    f"{', '.join(text for text, _at in lacking)}; no log in "
                    f"the record carries "
                    f"{'those names' if many else 'that name'} (the reader "
                    f"may have read "
                    f"{'them under other names' if many else 'it under another name'}"
                    f", or not read {'them' if many else 'it'})"),
            values=[f"{text}: {_pages(at)}" for text, at in lacking],
            pages=where))
    return out

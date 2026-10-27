"""Joining the record up, and writing down everything that does not join.

The readers each see one thing: one log, one laboratory sheet, the narrative.
This pass is the only one that sees all of them at once, and its job is the
one nothing else can do -- put them beside each other and say where they
disagree.

IT NEVER RESOLVES A DISAGREEMENT. A summary table that says 31 and a sheet
that says 29 for the same specimen are both recorded, as a ``conflict`` QA
entry carrying both values and both pages. Picking one would destroy the only
evidence a reviewer has that there is something to look at, and the reviewer
is the one who can go to the paper. The same holds for a narrative that says
four borings over an appendix that carries five: the count is not corrected,
the mismatch is recorded, because a report saying that about itself is
information.

WHAT IT DOES DO.

**Links a laboratory test to the ground.** A sheet prints a hole and a depth;
this finds the investigation of that name and the sample at that depth, within
0.15 m and whatever units each of them printed. "MW-2" is the log's "MW-02",
and a hole the logs hold at the head of a longer label -- "BH-7 U3", or a
sample label "BH-03 S-2 1.5-1.95 m" on a sheet that prints no boring -- is
that hole, said once in a QA note (RI-07). The printed values are never
touched -- ``LabTest.investigation_id`` stays the sheet's own words -- and the
link goes in fields of its own, empty when nothing matched.

**Joins a hole read from two logs.** A continuation sheet that became a work
item of its own comes back as a second log of the same hole on the next page.
Where the sheets' own "sheet 2 of 2" says so, or the second log's depths begin
where the first's leave off, it is folded into the first, said once in a QA
entry (RI-08).

**Counts what is there.** The narrative's stated counts and named identifiers
against the investigations the appendix actually yielded.

**Cross-checks the summary table.** Most reports print a "Summary of
Laboratory Tests" table; every value in it that also appears on a sheet is
compared, and only the disagreements are recorded. A row is set beside a
sheet of its hole and depth only when the two share a test, and of several
such sheets of one kind the closest is taken; two that cannot be told apart
are both left out and said in a note (RI-140).

**Checks each log against itself.** A driven record's drives must add up to
its interval (read the way the DIGGS writer will write them), its number of
drives must match the log's other complete records, and no layer, sample,
driven record or water level may sit below the hole's total depth.

**Says what was not read.** A page the labels called a boring log that ended
up in no work item, a page whose text layer is unreliable and which no Azure
Document Intelligence result covered, every ``unresolved`` line the readers
returned, and every quantity whose printed unit has no conversion -- each
becomes a QA entry, because a record that is silent about what it missed
cannot be reviewed.

A MODEL IS OPTIONAL AND ONLY COMMENTS. Pass an engine and one call is spent
asking for a sentence about each conflict -- which value looks like the
misreading, what would settle it. The comment is appended to the entry; the
values and the verdict are untouched.
"""

from __future__ import annotations

import math
import re
from typing import (Any, Dict, Iterable, List, Optional, Sequence, Set,
                    Tuple)

from pydantic import BaseModel, Field

from report_ingest.model import (
    UNITLESS_FIELDS, Investigation, LabTest, QAEntry, Quantity, ReportRecord,
    Sample, lab_hole, si_numbers, unit_printed,
)
from report_ingest.printed_depths import _WHOLE_RANGE, _range, _sample_cell
from report_ingest.scoring import KEY_CONTENT

__all__ = [
    "reconcile", "si_view", "DEPTH_TOL_M", "CROSS_CHECKS", "ITEM_ROLES",
    "fold_id", "hole_key", "depth_m", "depth_unit",
]

#: A laboratory specimen belongs to a sample when its depth lands within this,
#: in metres. The log scorer's tolerance, for the same reason: a sheet prints
#: the depth the sampler was at and a log prints the interval it came from.
DEPTH_TOL_M = 0.15

#: How close two readings of the same value have to be before they are the
#: same reading: a hundredth, or one percent of the value, whichever is
#: larger. The absolute part catches an index value printed to two figures;
#: the relative part keeps a pressure converted from psf to kPa from reading
#: as a disagreement with itself.
ABS_TOL = 0.01
REL_TOL = 0.01

#: The roles that OUGHT to become a work item. ``plan`` and ``profile`` are
#: key content and are deliberately not here: they are figures, and figures
#: are collected into one item rather than read one at a time.
ITEM_ROLES: Tuple[str, ...] = tuple(
    role for role in KEY_CONTENT if role not in ("plan", "profile"))

#: ``summary-table column -> the fields a per-sheet result calls it``. The
#: table is the cross-check the plan asks for, and this is the whole of what
#: can be cross-checked: a column with no counterpart on any sheet is not a
#: disagreement, it is a column.
CROSS_CHECKS: Dict[str, Tuple[str, ...]] = {
    "ll": ("ll",),
    "pl": ("pl",),
    "pi": ("pi",),
    "wc": ("water_content", "wc"),
    "fines_percent": ("fines_percent",),
    "sand_percent": ("sand_percent",),
    "gravel_percent": ("gravel_percent",),
    "dry_density": ("dry_density",),
    "wet_density": ("wet_density",),
    "qu": ("qu",),
    "su": ("su",),
    "c": ("c",),
    "phi_deg": ("phi_deg",),
    "swell_percent": ("swell_percent",),
    "organic_percent": ("organic_percent",),
    "pH": ("pH",),
    "resistivity": ("resistivity",),
    "sulfate": ("sulfate",),
    "chloride": ("chloride",),
    "sulfides": ("sulfides",),
    "redox": ("redox",),
}

#: ``narrative count -> (the investigation kinds it counts, its own name)``.
_COUNT_KINDS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "boringCount": (("boring",), "borings"),
    "testPitCount": (("test_pit",), "test pits"),
    "cptCount": (("cpt",), "CPT soundings"),
}


# ---------------------------------------------------------------------------
# small comparisons
# ---------------------------------------------------------------------------

def fold_id(text: Any) -> str:
    """An exploration identifier folded for comparison.

    ``"B-1"``, ``"B 1"`` and ``"b1"`` are one hole written three ways, and a
    laboratory sheet and a log routinely write it two of them. Folding is for
    MATCHING only: what is stored stays exactly as each of them printed it.
    """
    return "".join(ch for ch in str(text or "").upper() if ch.isalnum())


def hole_key(text: Any) -> str:
    """:func:`fold_id` with the leading zeros of each number dropped.

    ``"MW-02"`` on a log and ``"MW-2"`` on a laboratory sheet are one hole:
    a log template pads the number and the laboratory does not. Only zeros
    that START a number go, so ``"B-10"`` stays ten. For matching a sheet to
    a hole only, as :func:`fold_id` is.
    """
    return re.sub(r"(?<!\d)0+(?=\d)", "", fold_id(text))


#: Two numbers with only a separator between them: "1 2.0", "1-2", "1, 2".
#: Folding would run them into one number, "B-1 2.0 ft" into B-12.
_JOINED_NUMBERS = re.compile(r"\d[^0-9A-Za-z]+\d")
#: A mark or word joining a second hole's number to a hole's name
#: ("BH-47/48", "TP-47, 48", "BH-47 & 48", "BH-47 to 50"). A hyphen run on
#: to the name with no space before it is left out: "B-1-2" is as often
#: B-1's sample 2 as two holes.
_KEY_JOIN = re.compile(r"\s*(?:/|&|\+|,|;|\band\b|\bto\b|–)\s*|\s+-\s*",
                       re.I)
#: A whole number at the head of a label's text: a hole's number, not a
#: depth ("2.0", "2 m", "2'", "2-2.45") a label prints after its hole, nor
#: a number a letter runs on to ("3A").
_JOINED_WHOLE = re.compile(
    r"\d+(?![\d.,A-Za-z]|\s*(?:m|ft|cm|mm|in)\b|\s*['\"’]|\s*[-–]\s*\d)",
    re.I)


def _leading_hole(label: Any, keys: Any) -> Tuple[str, str]:
    """The longest of ``keys`` that BEGINS ``label``, and what follows it;
    ``("", "")`` when none does or the label names more than one.

    A sheet that prints no boring heading names the specimen by its hole --
    ``"BH-7 U3"``, ``"BH4, U1"``, a hole, a sample and a depth run together
    -- and a lab test carrying that label as its boring, or as its sample
    with no boring at all, names that hole first. Only holes the report's
    own logs hold are looked for, and only ones with a letter in them: a
    bare ``"2"`` is as likely a sample number as a hole. The hole has to
    end where a word of the label ends -- at a space, a comma, a slash or
    the end -- so ``"B-1"`` is not found at the head of ``"B-12"`` or of
    ``"B-1A"``; and the search stops at the first number that follows
    another across a separator, so ``"B-1 2.0 ft"`` and ``"B-1-2"`` are
    B-1 and never B-12, whose folded spelling they share.

    A label naming a second logged hole further on names no one hole, and
    nothing is returned for it (RI-143): another hole by name, of this
    series or another -- ``"B-1/B-2"``, ``"B-1 to B-4 composite"``,
    ``"BH-47 / TP-71 composite"`` -- or a whole number joined straight after
    the first that is a logged hole's number in its series -- ``"BH-47/48"``
    and ``"BH-47 & 48"`` beside a logged BH-48. A joined number no logged
    hole of the series carries is the sample's (``"BH-47, 3"``).
    """
    text = str(label or "")
    best, end = _head_hole(text, keys)
    if not best:
        return "", ""
    rest = text[end:]
    joined = _KEY_JOIN.match(rest)
    number = _JOINED_WHOLE.match(rest[joined.end():]) if joined else None
    if number is not None:
        series = re.match(r"[A-Z]*", best).group(0)
        other = hole_key(series + number.group(0))
        if other != best and other in keys:
            return "", ""
    for start in range(end, len(text)):
        if text[start].isalnum() and not text[start - 1].isalnum():
            other, _end = _head_hole(text[start:], keys)
            if other and other != best:
                return "", ""
    return best, rest


def _head_hole(text: str, keys: Any) -> Tuple[str, int]:
    """The longest key at the head of ``text`` and where it ends."""
    best, best_end = "", 0
    for end in range(1, len(text) + 1):
        if end < len(text) and text[end].isalnum():
            continue                     # not the end of a word
        if _JOINED_NUMBERS.search(text[:end]):
            break                        # "B-1 2.0 ft": a depth, not B-12
        key = hole_key(text[:end])
        if key and key in keys and not key.isdigit() and \
                len(key) > len(best):
            best, best_end = key, end
    return best, best_end


def depth_m(value: Optional[Quantity]) -> Optional[float]:
    """A depth in metres, or None when there is none or it will not convert."""
    if value is None:
        return None
    converted = value.to_si()
    return None if converted is None else converted.value


def _unit_named(value: Optional[Quantity]) -> str:
    """``"m"`` or ``"ft"`` for a depth printed in metres or in feet (any
    spelling the record converts), the printed unit for one in anything
    else, ``""`` for none or no unit."""
    if value is None or not unit_printed(value.unit):
        return ""
    per = depth_m(Quantity(value=1.0, unit=value.unit))
    if per is None:
        return str(value.unit).strip()
    for name, metres in (("m", 1.0), ("ft", 0.3048)):
        if abs(per - metres) < 1e-9:
            return name
    return str(value.unit).strip()


def depth_unit(record: ReportRecord) -> str:
    """The one unit the record's depths are printed in: ``"m"`` or ``"ft"``
    when every depth that prints a unit -- the logs' samples, layers and
    total depths, the laboratory sheets' and the summary rows' depths --
    prints that one; ``""`` when they print two, or none (BACKLOG RI-143).

    A lab sheet that prints its depths with no unit (``0``, ``1.7``, a
    legend ``0-1.7``) is read in it for linking and matching only
    (:func:`_depth_view`); a report that mixes feet and metres gives none,
    and such a depth stays unread, as RI-20 has it."""
    units: Set[str] = set()

    def add(*values: Optional[Quantity]) -> None:
        for value in values:
            unit = _unit_named(value)
            if unit:
                units.add(unit)

    for inv in record.investigations:
        add(getattr(inv, "total_depth", None))
        for sample in inv.samples:
            add(sample.top, sample.bottom)
        for layer in getattr(inv, "layers", None) or ():
            add(layer.top, layer.bottom)
    for test in record.lab_tests:
        add(test.depth_top, test.depth_bottom)
        if test.kind == "summary_table":
            for row in getattr(test.result, "rows", None) or ():
                add(row.depth_top, row.depth_bottom)
    return units.pop() if len(units) == 1 and \
        next(iter(units)) in ("m", "ft") else ""


def _label_depth(text: str, unit: str
                 ) -> Optional[Tuple[float, Optional[float], str, str]]:
    """``(top, bottom, unit, the label before it)`` for a label that ends
    with its depth -- ``"QX-2, U9, 3.10-3.55m"``, ``"S-3 @ 4.5 ft"`` --
    read as a log's sample column reads one (``printed_depths._sample_cell``:
    with a unit printed), or for a label that is only an interval with a
    decimal in it (``"0-1.7"``: a sample's name is never one), in ``unit``
    where it prints none. None when neither, or when no unit is known."""
    text = str(text or "").strip()
    if not text:
        return None
    found = _sample_cell(text)
    if found is not None:
        top, bottom, printed, label = found
        return top, bottom, printed, label
    whole = _WHOLE_RANGE.match(text)
    if whole and ("." in whole.group(1) or "." in whole.group(3)):
        found_range = _range(whole)
        if found_range is not None:
            top, bottom, printed = found_range
            if printed or unit:
                return top, bottom, printed or unit, ""
    return None


def _is_the_interval(label: str, top: Optional[Quantity],
                     bottom: Optional[Quantity]) -> bool:
    """Whether a sample label prints nothing but the depth interval the
    test prints (``"0-1.7"`` beside 0 and 1.7): a depth, not a name."""
    whole = _WHOLE_RANGE.match(str(label or ""))
    if not whole or top is None:
        return False
    found = _range(whole)
    if found is None:
        return False
    low, high, _unit = found
    return abs(low - float(top.value)) < 1e-6 and (
        bottom is None or abs(high - float(bottom.value)) < 1e-6)


def _depth_view(test: Any, unit: str, need_hole: bool = True
                ) -> Tuple[Any, str]:
    """``(the test as linking and matching read its depth, how)`` (BACKLOG
    RI-143): the test itself and ``""`` where its depth converts, or there
    is nothing to read. Otherwise a copy, with ``how``:

    - ``"unit"``: the test prints its depths with no unit, and they are
      read in ``unit``, the one the record prints its depths in
      (:func:`depth_unit`);
    - ``"label"``: it prints no depth, and the sample label (else the
      boring as printed) ends with one (:func:`_label_depth`), read with
      the unit the label prints, else ``unit``. ``need_hole``: only for a
      test placed in a hole (:func:`~report_ingest.model.lab_hole`), since
      a depth with no hole places nothing (a summary table's second line,
      ``0.00 - 1.70``, under no boring).

    In the copy, a sample label that prints only the interval names no
    sample (:func:`_is_the_interval`), and one the depth was read from
    keeps the words before it. The record's own fields never change."""
    top, bottom = test.depth_top, test.depth_bottom
    if depth_m(top) is not None:
        return test, ""
    update: Dict[str, Any] = {}
    how = ""
    if top is not None and unit and not unit_printed(top.unit):
        update["depth_top"] = Quantity(value=float(top.value), unit=unit)
        if bottom is not None and not unit_printed(bottom.unit):
            update["depth_bottom"] = Quantity(value=float(bottom.value),
                                              unit=unit)
        how = "unit"
    elif top is None and (not need_hole or hole_key(lab_hole(test))):
        for field in ("sample_id", "investigation_id"):
            found = _label_depth(getattr(test, field, ""), unit)
            if found is None:
                continue
            low, high, printed, label = found
            update["depth_top"] = Quantity(value=low, unit=printed)
            update["depth_bottom"] = (Quantity(value=high, unit=printed)
                                      if high is not None else None)
            if field == "sample_id":
                update["sample_id"] = label
            how = "label"
            break
    if not how:
        return test, ""
    sample = update.get("sample_id", test.sample_id)
    if _is_the_interval(sample, test.depth_top, test.depth_bottom):
        update["sample_id"] = ""
    return test.model_copy(update=update), how


def _same(a: Any, b: Any) -> Optional[bool]:
    """Are these two readings of one value the same? None = not comparable.

    Not comparable is a real answer and the important one: a value with a
    unit and a value without are not the same measurement written twice, they
    are two things, and calling them a disagreement would fill the QA section
    with noise a reviewer then has to wade through.
    """
    if a is None or b is None:
        return None
    if isinstance(a, Quantity) != isinstance(b, Quantity):
        return None
    if isinstance(a, Quantity):
        left, right = a.to_si(), b.to_si()
        if left is None or right is None or left.unit != right.unit:
            return None
        return abs(left.value - right.value) <= max(
            ABS_TOL, abs(right.value) * REL_TOL)
    left, right = _number(a), _number(b)
    if left is not None and right is not None:
        return abs(left - right) <= max(ABS_TOL, abs(right) * REL_TOL)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _show(value: Any) -> str:
    if isinstance(value, Quantity):
        return f"{value.value:g} {value.unit}".strip()
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


# ---------------------------------------------------------------------------
# the SI view
# ---------------------------------------------------------------------------

def si_view(record: ReportRecord) -> Dict[str, Dict[str, Any]]:
    """Every number in the record in SI, by section and path.

    The record itself keeps every number in the unit it was printed in -- that
    rule is what lets a reviewer set it beside the page. This is the derived
    view for everything that wants one set of units: each quantity through
    :meth:`~report_ingest.model.Quantity.to_si`, walked off the models
    themselves so it cannot drift from the record.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for index, inv in enumerate(record.investigations):
        name = inv.investigation_id or f"investigation[{index}]"
        out[f"investigations.{name}"] = {
            path: {"value": value, "unit": unit}
            for path, value, unit in si_numbers(inv) if unit}
    for index, test in enumerate(record.lab_tests):
        name = f"{test.kind}[{index}]"
        out[f"lab_tests.{name}"] = {
            path: {"value": value, "unit": unit}
            for path, value, unit in si_numbers(test) if unit}
    bearing = {}
    for index, value in enumerate(record.general.bearingCapacityValues):
        converted = value.value.to_si()
        if converted is not None:
            bearing[f"bearingCapacityValues[{index}]"] = {
                "value": converted.value, "unit": converted.unit}
    if bearing:
        out["general"] = bearing
    return out


# ---------------------------------------------------------------------------
# what a model is asked, and only asked
# ---------------------------------------------------------------------------

class _ConflictComment(BaseModel):
    index: int = Field(description="the conflict's number, as listed")
    comment: str = Field(
        description="which value looks like the misreading and what would "
                    "settle it, in 30 words or fewer")


class _ConflictComments(BaseModel):
    comments: List[_ConflictComment] = Field(default_factory=list)


_CONFLICT_SYSTEM = """\
You are commenting on disagreements found inside one geotechnical report --
the same value read off two different pages and coming back different. You do
NOT decide which is right and nothing you say changes the record: both values
stay, and a person will look at the pages. Say, in 30 words or fewer for each,
which of the two looks like the misreading and what would settle it (which
page to open, which column to check). Where you cannot tell, say so.
"""


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------

def reconcile(record: ReportRecord, *,
              labels: Optional[Dict[int, str]] = None,
              items: Optional[Sequence[Any]] = None,
              no_text_pages: Optional[Sequence[int]] = None,
              di_pages: Optional[Sequence[int]] = None,
              reader_unresolved: Optional[Sequence[Dict[str, Any]]] = None,
              engine: Any = None, doc: Any = None,
              outline: Any = None) -> ReportRecord:
    """Join the record up and record everything that does not join.

    Pure Python. ``labels`` and ``items`` are the final page labels and the
    work items -- the graph has both and passes them; without them the page
    checks are skipped rather than guessed. ``reader_unresolved`` is every
    line the readers could not settle, each already carrying its own ``what``
    and ``why``. ``engine`` is optional and buys ONE call, which comments on
    the conflicts and changes nothing.

    Holes printed under one name that are not one hole's continuation
    sheets (a second series of logs numbered from 1 again) are kept apart
    by what their pages print, each given a ``qualifier`` beside its
    printed name, and a laboratory sheet naming such a name is linked to
    one of them only where it prints what tells them apart
    (:mod:`report_ingest.shared_names`, RI-135). Logs of one name that read
    as one hole's sheets the fold did not join (a later sheet a page
    further on, a sounding carried on to the next page) are kept as read,
    counted as one hole, and said; a sheet naming them goes to the log
    holding its sample, else the first, named by its pages in
    ``linked_log_pages``. ``doc`` (the planlens document) and
    ``outline`` (its outline) give the report sections and the logs' titles
    for that; the graph passes both, and without them only what the record
    holds is used.

    The record is modified in place and returned.
    """
    from report_ingest.shared_names import tell_apart

    qa: List[QAEntry] = []

    _fold_continuations(record, qa)
    twins, said = tell_apart(record, doc=doc, outline=outline)
    qa.extend(said)
    _link_lab_tests(record, qa, twins)
    _link_calculations(record, qa)
    _check_calculations(record, qa)
    _check_counts(record, qa)
    _check_dictionaries(record, qa)
    _cross_check_summary_table(record, qa)
    _check_drives(record, qa)
    _check_below_total_depth(record, qa)
    _check_units(record, qa)
    _check_pages(record, qa, labels, items, no_text_pages, di_pages)
    _carry_unresolved(qa, reader_unresolved)

    if engine is not None:
        _comment_on_conflicts(qa, engine)

    record.qa.extend(qa)
    _fill_found_counts(record)
    return record


# -- one hole read from two logs --------------------------------------------
#
# planlens starts a new log item at a page whose title block it cannot read a
# hole from, so a continuation sheet that prints no "sheet 2 of 2" where the
# item builder looks, or prints its hole somewhere else, is read as a log of
# its own (BACKLOG RI-08; nine in stage b, in two reports and a bound one).
# Every check below then sees two holes of one name: a lab test at 12 m in
# B-3 is looked for on the sheet that stops at 11 m, the narrative's count is
# one short, and the DIGGS file holds two features with one name.

#: Kinds whose log is layers, samples and driven records, which can be joined
#: sheet to sheet. A sounding is a series and is not folded here.
_FOLDED_KINDS = ("boring", "test_pit", "hand_auger", "well", "other")

#: A later sheet repeats the bottom of the one before it -- the sample it
#: broke on, the header row at its top -- so its first depth may sit above
#: the earlier sheet's last one by up to this share of the earlier sheet's
#: own depth range. A log that starts again at the ground surface does not.
CONTINUATION_OVERLAP = 0.25

#: "Sheet 2 of 3", "Page 2 of 3", "2 of 3", "2/3".
_SHEET_OF = re.compile(r"(\d+)\s*(?:of|/)\s*(\d+)", re.I)


def _sheet_number(text: str) -> Optional[Tuple[int, int]]:
    match = _SHEET_OF.search(str(text or ""))
    if not match:
        return None
    k, n = int(match.group(1)), int(match.group(2))
    return (k, n) if 1 <= k <= n else None


def _depth_range(inv: Investigation) -> Optional[Tuple[float, float]]:
    """(shallowest, deepest) of what a log's rows print, in metres.

    The samples and driven records first, the layers only where there are
    none: a layer top read off a description runs further from the ruler
    than a sample's. Negative depths are left out (a misread is not a
    depth). The total depth is the header's, not the sheet's, and is not
    used.
    """
    rows = ([s.top for s in inv.samples] + [s.bottom for s in inv.samples]
            + [s.depth_top for s in inv.spt]
            + [s.depth_bottom for s in inv.spt])
    found = [d for d in (depth_m(q) for q in rows) if d is not None and d >= 0]
    if not found:
        found = [d for d in (depth_m(q) for ly in inv.layers
                             for q in (ly.top, ly.bottom))
                 if d is not None and d >= 0]
    return (min(found), max(found)) if found else None


def _same_hole(first: Investigation, later: Investigation,
               by_sheet: bool) -> bool:
    """Do these two logs name one hole?

    The same identifier, leading zeros folded; or, where the printed sheet
    numbers already say the later sheet continues the first, the later one
    printing only the hole's number ("8" after "BH No. 8").
    """
    a, b = hole_key(first.investigation_id), hole_key(later.investigation_id)
    if not a or not b:
        return False
    if a == b:
        return True
    if not by_sheet:
        return False
    short, long_ = sorted((a, b), key=len)
    return short.isdigit() and long_.endswith(short) and \
        long_[:-len(short)].isalpha()


def _continues(first: Investigation, later: Investigation) -> str:
    """Why ``later`` is a continuation sheet of ``first``, or ``""``."""
    if later.kind != first.kind or first.kind not in _FOLDED_KINDS:
        return ""
    if not first.pages or not later.pages or \
            min(later.pages) != max(first.pages) + 1:
        return ""
    if later.depth_unit != first.depth_unit or \
            later.units_known != first.units_known:
        return ""
    here, there = _depth_range(first), _depth_range(later)
    carries_on = here is not None and there is not None and \
        there[0] >= here[1] - CONTINUATION_OVERLAP * (here[1] - here[0])
    a, b = _sheet_number(first.sheet), _sheet_number(later.sheet)
    if b is not None:
        if b[0] == 1 or (a is not None and (b[1] != a[1] or b[0] <= a[0])):
            return ""           # a first sheet, or one of another set
        if not _same_hole(first, later, by_sheet=True):
            return ""
        why = f"it prints sheet {b[0]} of {b[1]}"
        if here is not None and there is not None and not carries_on:
            # The printed number joins them; the depths do not, so a person
            # should look (a misread ruler, or two holes numbered as one).
            why += (f", but its depths ({there[0]:.2f}-{there[1]:.2f} m) "
                    f"start above where the first's "
                    f"({here[0]:.2f}-{here[1]:.2f} m) leave off")
        return why
    if not _same_hole(first, later, by_sheet=False):
        return ""
    if carries_on and there[1] > here[1]:
        return (f"its depths ({there[0]:.2f}-{there[1]:.2f} m) begin where "
                f"the first's ({here[0]:.2f}-{here[1]:.2f} m) leave off")
    return ""


def _blank(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _unset(row: Any, name: str) -> bool:
    """Is this field of ``row`` empty, or still its default (a sample kind
    'other' is a symbol nobody could read)?"""
    value = getattr(row, name)
    if _blank(value):
        return True
    field = type(row).model_fields[name]
    return value == field.get_default(call_default_factory=True)


def _restated(kept: Any, other: Any, top: str, bottom: str) -> bool:
    """Is ``other`` the same row as ``kept``, printed again on the next
    sheet: the same depth, and nothing printed on both that differs?"""
    a, b = depth_m(getattr(kept, top)), depth_m(getattr(other, top))
    if a is None or b is None or abs(a - b) > DEPTH_TOL_M:
        return False
    if bottom:
        a, b = depth_m(getattr(kept, bottom)), depth_m(getattr(other, bottom))
        if a is not None and b is not None and abs(a - b) > DEPTH_TOL_M:
            return False
    for name in type(kept).model_fields:
        if name in (top, bottom, "prov", "note"):   # how, not what
            continue
        if not _unset(kept, name) and not _unset(other, name) and \
                getattr(kept, name) != getattr(other, name):
            return False
    return True


def _join_rows(kept: List[Any], more: Sequence[Any], top: str,
               bottom: str) -> int:
    """``more`` onto ``kept``; a row ``kept`` already holds is not added
    again, and fills what ``kept``'s copy left blank. Returns how many."""
    again = 0
    earlier = list(kept)            # the later sheet's own rows stay apart
    for row in more:
        twin = next((k for k in earlier if _restated(k, row, top, bottom)),
                    None)
        if twin is None:
            kept.append(row)
            continue
        again += 1
        for name in type(twin).model_fields:
            if _unset(twin, name) and not _unset(row, name):
                setattr(twin, name, getattr(row, name))
    return again


def _fold_into(first: Investigation, later: Investigation) -> Dict[str, Any]:
    """Join ``later``'s sheet onto ``first``. What it says goes in the QA."""
    said: Dict[str, Any] = {"restated": 0, "total_depth": None}
    said["restated"] += _join_rows(first.samples, later.samples,
                                   "top", "bottom")
    said["restated"] += _join_rows(first.spt, later.spt,
                                   "depth_top", "depth_bottom")
    for layer in later.layers:
        if not any(_restated(k, layer, "top", "bottom")
                   and " ".join(k.description.split()) ==
                   " ".join(layer.description.split())
                   for k in first.layers):
            first.layers.append(layer)
    for water in later.water:
        if not any(w.when == water.when and depth_m(w.depth) is not None
                   and depth_m(water.depth) is not None
                   and abs(depth_m(w.depth) - depth_m(water.depth))
                   <= DEPTH_TOL_M for w in first.water):
            first.water.append(water)
    mine, theirs = depth_m(first.total_depth), depth_m(later.total_depth)
    if theirs is not None and (mine is None or theirs > mine + DEPTH_TOL_M):
        if mine is not None:
            said["total_depth"] = (first.total_depth, later.total_depth)
        first.total_depth = later.total_depth
    elif mine is not None and theirs is not None and \
            abs(mine - theirs) > DEPTH_TOL_M:
        said["total_depth"] = (later.total_depth, first.total_depth)
    for name in ("x", "y", "coordinate_system", "elevation", "date_started",
                 "date_finished", "station", "offset", "source_report",
                 "pit"):
        if _blank(getattr(first, name)) and \
                not _blank(getattr(later, name)):
            setattr(first, name, getattr(later, name))
    for name in type(first.drilling).model_fields:
        if _unset(first.drilling, name) and \
                not _unset(later.drilling, name):
            setattr(first.drilling, name, getattr(later.drilling, name))
    if later.remarks and later.remarks not in first.remarks:
        first.remarks = "\n".join(x for x in (first.remarks, later.remarks)
                                  if x)
    first.fields = {**later.fields, **first.fields}
    first.pages = sorted(set(first.pages) | set(later.pages))
    first.prov = list(first.prov) + list(later.prov)
    return said


def _fold_continuations(record: ReportRecord, qa: List[QAEntry]) -> None:
    """Fold each continuation sheet read as a log of its own into the log
    before it, page order, one QA entry per hole."""
    if len(record.investigations) < 2:
        return
    ordered = sorted(record.investigations,
                     key=lambda inv: min(inv.pages) if inv.pages else 10 ** 9)
    kept: List[Investigation] = []
    folds: Dict[int, List[Tuple[Investigation, str, Dict[str, Any]]]] = {}
    for inv in ordered:
        first = kept[-1] if kept else None
        why = _continues(first, inv) if first is not None else ""
        if not why:
            kept.append(inv)
            continue
        pages = list(inv.pages)
        said = _fold_into(first, inv)
        said["pages"] = pages
        folds.setdefault(id(first), []).append((inv, why, said))
    if not folds:
        return
    gone = {id(inv) for rows in folds.values() for inv, _w, _s in rows}
    record.investigations[:] = [inv for inv in record.investigations
                                if id(inv) not in gone]
    for first in record.investigations:
        rows = folds.get(id(first))
        if not rows:
            continue
        name = first.investigation_id or "an exploration"
        parts = []
        values = []
        for later, why, said in rows:
            spelt = (f", printed {later.investigation_id!r}"
                     if later.investigation_id != first.investigation_id
                     else "")
            n = len(said["pages"])
            parts.append(("the next page" if n == 1 else
                          f"the next {n} pages") + f"{spelt}: {why}")
            if said["restated"]:
                values.append(f"{said['restated']} sample or driven record"
                              f"(s) printed again on the later sheet, kept "
                              f"once")
            if said["total_depth"] is not None:
                old, new = said["total_depth"]
                values.append(f"total depth {_show(new)} kept as the "
                              f"deeper; {_show(old)} on the other sheet")
        qa.append(QAEntry(
            kind="note", where=f"{name}.sheets",
            detail=(f"{name} was read as {len(rows) + 1} logs, one per work "
                    f"item, and folded into one exploration: "
                    + "; ".join(parts)
                    + ". Layers are kept as each sheet printed them"),
            values=values, pages=list(first.pages)))


# -- the lab-to-ground link -------------------------------------------------

#: The kinds of exploration a hole is taken for from a SAMPLE label: the
#: ones samples are taken from and named after. A sounding or a probe
#: called "S-1" is as likely the sample's own number. Layout judgment
#: (RI-07 review); a hand auger is a boring made by hand, and a sampled
#: monitoring well is drilled as a boring.
_SAMPLED_KINDS = ("boring", "test_pit", "hand_auger", "well")

#: The words a label or a log names an exploration by before its own name,
#: and the kinds of investigation each may name (BACKLOG RI-143): a pit
#: word a pit, a cone word a cone, a boring word a hole drilled or augered,
#: a general word ("hole", "location") any.
_PIT_WORDS = r"t\s*/\s*p|tp|test\s*pit|pit|trench"
_BORING_WORDS = r"b\s*/\s*h|b\.\s*h\.?|bh|bore\s*hole|boring|forage|sondeo"
_EXPLORATION_HEAD = re.compile(
    rf"^\s*[(\[]?\s*(?P<word>{_PIT_WORDS}|{_BORING_WORDS}|cpt|well|hole|"
    rf"location|exploration|sondage)(?![A-Za-z])\.?\s*"
    rf"(?:(?:no|nr|n[°º]|num|number)(?![A-Za-z])\.?\s*|#\s*)?"
    rf"[-#.:]?\s*[(\[]?\s*", re.I)
#: A number at the head of what follows such a word: a hole's number, not
#: a decimal, a range or a number a letter runs on to ("Pit 4Q").
_HEAD_NUMBER = re.compile(
    r"(\d+)(?![\d]|[.,+]\d|\s*(?:-|–|to)\s*\d|[A-Za-z])")
#: A mark or word joining a second hole to the first in a label naming
#: several ("KW-12 / KW-13", "Boring 47 & 48", "BH 47 to BH 52").
_JOIN = r"\s*(?:/|&|\+|,|;|\band\b|\bto\b|–|-)\s*"


def _word_kinds(word: str) -> Optional[Tuple[str, ...]]:
    """The kinds of investigation an exploration word may name; None for
    a general word, which may name any."""
    word = str(word or "")
    if re.fullmatch(_PIT_WORDS, word, re.I):
        return ("test_pit",)
    if re.fullmatch(_BORING_WORDS, word, re.I):
        return ("boring", "hand_auger", "well", "other")
    if word.lower() == "cpt":
        return ("cpt",)
    if word.lower() == "well":
        return ("well", "boring")
    return None


def _another_hole_after(rest: str, by_id: Dict[str, Investigation],
                        word: str = "") -> bool:
    """Whether what follows a hole's name or number in a label names a
    second hole: a logged hole at the head of any word of it, or, after a
    joining mark or word, a word for an exploration and a number ("/ BH
    48"), or a whole number the logs give a hole of a kind ``word`` may name
    (``"& 48"`` beside a logged 48, :func:`_numbered`). A sample ("U9", the
    "3" of "BH 47, 3" where no hole 3 is logged) or a depth is no hole."""
    joined = re.match(_JOIN, rest)
    if joined:
        after = rest[joined.end():]
        number = _JOINED_WHOLE.match(after)
        if number and _numbered(hole_key(number.group(0)), word,
                                by_id.values()):
            return True
        head = _EXPLORATION_HEAD.match(after)
        if head and head.end() and _HEAD_NUMBER.match(after[head.end():]):
            return True
    return any(_head_hole(rest[i:], by_id)[0]
               for i in range(len(rest)) if rest[i].isalnum()
               and (i == 0 or not rest[i - 1].isalnum()))


def _numbered(number: str, word: str, invs: Iterable[Investigation]
              ) -> List[Investigation]:
    """The holes the logs name by ``number`` alone (``"4"``) or by an
    exploration word and that number (``"BH No. 61"``), of a kind ``word``
    may name. ``number`` is folded as :func:`hole_key` folds it."""
    kinds = _word_kinds(word)
    out: List[Investigation] = []
    for inv in invs:
        name = str(inv.investigation_id or "")
        key = hole_key(name)
        if not key.isdigit():
            head = _EXPLORATION_HEAD.match(name)
            rest = name[head.end():] if head else ""
            key = hole_key(rest) if re.fullmatch(r"\s*\d+\s*\)?\s*",
                                                 rest) else ""
        if key != number or (kinds and inv.kind not in kinds):
            continue
        if not any(inv is other for other in out):
            out.append(inv)
    return out


def _hole_after_word(label: Any, by_id: Dict[str, Investigation]
                     ) -> Optional[Investigation]:
    """The hole a label names after a word for an exploration (BACKLOG
    RI-143): ``"BORE HOLE NO. (KW-12)"`` and ``"BH. KW - 12"`` are KW-12,
    the word, its "No." and the bracket dropped; ``"BH47, U9"``,
    ``"B/H 83 U4"`` and ``"BH 61, U17"`` are the hole the logs name 47, 83
    or "BH No. 61" -- a number the logs give one hole of a kind the word
    may name, by itself or after a word of their own. A word twice over
    ("Location: BH 47") is read through. None where nothing is named, or
    where the label names a second hole after the first ("BH 47 / BH 48
    composite", "Boring 47 & 48", :func:`_another_hole_after`)."""
    text = str(label or "")
    for _ in range(2):
        head = _EXPLORATION_HEAD.match(text)
        if not head or not head.end():
            return None
        word, text = head.group("word"), text[head.end():]
        key, rest = _leading_hole(text, by_id)
        if key:
            return None if _another_hole_after(rest, by_id, word) \
                else by_id[key]
        if _head_hole(text, by_id)[0]:
            return None                  # a label naming two holes
        number = _HEAD_NUMBER.match(text)
        if number:
            if _another_hole_after(text[number.end():], by_id, word):
                return None
            found = _numbered(hole_key(number.group(1)), word,
                              by_id.values())
            return found[0] if len(found) == 1 else None
    return None


def _bare_number_hole(number: str, by_id: Dict[str, Investigation]
                      ) -> Optional[Investigation]:
    """The hole a boring printed as a bare number names (RI-143): the one
    the logs name by an exploration word and that number (``"61"`` for
    ``"BH No. 61"``, :func:`_numbered`), where no other logged hole carries
    the number in a series of its own -- beside KW-038 and BH-38, a bare
    ``"38"`` names neither."""
    key = hole_key(number)
    found = _numbered(key, "", by_id.values())
    if len(found) != 1:
        return None
    for other in by_id.values():
        tail = re.search(r"(\d+)$", hole_key(other.investigation_id))
        if other is not found[0] and tail and tail.group(1) == key:
            return None
    return found[0]


#: A summary-table column heading that names the exploration, where the
#: row has no boring column of its own (BACKLOG RI-143): "Boring",
#: "Borehole", "Test pit", "Location", "Site ref."; not one that names
#: the sample, a depth or another property of the hole.
_HOLE_HEADING = re.compile(
    r"\b(?:bore\s*holes?|boreholes?|borings?|holes?|test\s*pits?|pits?|"
    r"trench(?:es)?|locations?|explorations?|site\s+(?:mark|ref)\w*|"
    r"sondages?|"
    r"sondeos?|forages?|bh|b/h|tp)\b", re.I)
_NOT_HOLE_HEADING = re.compile(
    r"\b(?:samples?|specimens?|depths?|elev\w*|dates?|diam\w*|dia|sizes?|"
    r"types?|logged|muestras?|[ée]chantillons?|prof\w*)\b", re.I)


def _hole_heading(heading: Any) -> bool:
    text = str(heading or "")
    return bool(_HOLE_HEADING.search(text)) and \
        not _NOT_HOLE_HEADING.search(text)


def _hole_in_column(row: Any, by_id: Dict[str, Investigation],
                    samples: Dict[str, List[Tuple[Investigation, Sample]]],
                    by_fold: Dict[str, Investigation]
                    ) -> Tuple[Optional[Investigation], str, str, bool]:
    """``(the hole, the heading, the cell, whether the row prints a cell
    under such a heading)`` for a summary row that prints no boring of its
    own and names its hole in a column of its ``other`` headed for one
    (:func:`_hole_heading`) -- a hole the logs hold, read as a boring
    column's cell is (:func:`_hole_for`; ``"61"`` for "BH No. 61",
    :func:`_bare_number_hole`)."""
    printed = False
    for heading, cell in getattr(row, "other", None) or ():
        cell = str(cell or "").strip()
        if not cell or not _hole_heading(heading):
            continue
        printed = True
        view = row.model_copy(update={"investigation_id": cell})
        inv, _through = _hole_for(view, by_id, samples, by_fold)
        if inv is not None:
            return inv, str(heading), cell, True
    return None, "", "", printed


def _hole_for(test: Any, by_id: Dict[str, Investigation],
              samples: Optional[Dict[str, List[Tuple[Investigation,
                                                     Sample]]]] = None,
              by_fold: Optional[Dict[str, Investigation]] = None
              ) -> Tuple[Optional[Investigation], str]:
    """The investigation a lab test (or a summary row) names, and where the
    name was found.

    The printed boring first, matched whole: exactly as the logs spell it
    (``fold_id``), then with leading zeros folded (``hole_key``, used only
    when one logged hole has that key -- "B-07" and "B-7" logged as two
    holes leave a sheet's "B-007" unlinked). Then a hole the report's logs
    hold at the HEAD of that printed text (``"BH-7 U3"``); then, only where
    the sheet printed no boring at all, a hole at the head of the sample
    label (``"BH-03 S-2 1.5-1.95 m"``), which is how many sheets name a
    specimen. The second string says which, for the QA note: empty for a
    whole match. Where no logged hole heads the label, one named after a
    word for an exploration (``"BORE HOLE NO. (KW-12)"``, ``"BH47, U9"``
    for the log's ``47``; :func:`_hole_after_word`, RI-143), and for a
    boring printed as a bare number, the one hole the logs give that number
    after a word of their own (``"61"`` for the log's ``"BH No. 61"``). A
    label headed by a logged hole that names a second logged hole
    (``"KW-12 / KW-13 composite"``, ``"KW-12/13"``, :func:`_leading_hole`)
    names no one hole, and no word is read through it.

    A SAMPLE NUMBER NAMES NO HOLE, so a hole is not taken from a label
    that could equally be a sample's name:

    * a name the logs also give a sample (``samples``) -- a boring B-1
      beside a pit's bulk sample B-1 -- is refused when another hole holds
      a sample of that name at the sheet's depth, or when this hole holds
      none there;
    * a hole is taken from a sample label only when it is a boring, test
      pit, hand auger or well, since a sounding or probe of that name may
      as well be the sample's number -- whether or not the logs read any
      sample names, and whatever else the label prints.
    """
    folded = fold_id(test.investigation_id)
    if folded and by_fold and folded in by_fold:
        return by_fold[folded], ""
    printed = hole_key(test.investigation_id)
    if printed in by_id:
        return by_id[printed], ""
    label, through = ((test.investigation_id, "the boring as printed")
                      if printed else (test.sample_id, "the sample label"))
    key, _rest = _leading_hole(label, by_id)
    if not key:
        if _head_hole(str(label or ""), by_id)[0]:
            return None, ""              # a label naming two holes
        # The word names the label a hole's, so no sample's name is feared.
        inv = _hole_after_word(label, by_id)
        if inv is None and printed and \
                re.fullmatch(r"\s*\d{1,4}\s*", str(label or "")):
            inv = _bare_number_hole(str(label), by_id)
        if inv is None or (through == "the sample label"
                           and inv.kind not in _SAMPLED_KINDS):
            return None, ""
        return inv, through
    inv = by_id[key]
    named = (samples or {}).get(key, [])
    if named:
        if _sample_at_depth(inv, test)[0] is None:
            return None, ""
        if any(other is not inv and _at_depth(sample, test) is not None
               for other, sample in named):
            return None, ""
    if through == "the sample label" and inv.kind not in _SAMPLED_KINDS:
        return None, ""
    return inv, through


def _link_lab_tests(record: ReportRecord, qa: List[QAEntry],
                    twins: Any = None) -> None:
    """Each laboratory sheet, and each row of a summary table, linked to
    the hole and the sample it names (:func:`_hole_for`,
    :func:`_sample_for`). A name two or more holes are printed under is
    linked to one of them only where the sheet prints what tells them apart
    (``twins``, :class:`report_ingest.shared_names.Twins`); otherwise to
    neither, and one QA entry for the name says so (RI-135). Where a hole is
    read as several logs of its name, the sheet goes to the log holding its
    sample (:func:`_part_for`), and ``linked_log_pages`` names that log by
    its pages, so the DIGGS file writes the result in that log's feature."""
    from report_ingest.shared_names import pages_span

    # name -> [(what, pages, why, chosen hole or None)]
    shared: Dict[str, List[Tuple[str, List[int], List[str], Any]]] = {}
    names = [inv.investigation_id for inv in record.investigations]
    # id(log) -> its pages, for each log of a hole read as several logs.
    several: Dict[int, str] = {
        id(log): pages_span(log.pages)
        for holes in (twins.holes.values() if twins else ())
        for hole in holes if len(hole) > 1 for log in hole}
    for original in record.lab_tests:
        original.linked_log_pages = ""
        for row in getattr(original.result, "rows", None) or ():
            row.linked_log_pages = ""

    def among_twins(inv: Investigation, original: Any, pages: List[int],
                    what: str, view: Any = None) -> Optional[Investigation]:
        """``inv``, or the log of its name the sheet says, or None.
        ``view`` is the sheet or the row with its depth read for the link
        (:func:`_depth_view`)."""
        if not twins or not twins.group(inv):
            original.linked_qualifier = ""
            return inv
        head, why = twins.pick(inv, original, pages)
        if len(twins.holes_of(inv)) > 1:
            shared.setdefault(inv.investigation_id, []).append(
                (what, list(pages), why, head))
        if head is None:
            original.linked_investigation_id = ""
            original.linked_sample_id = ""
            original.linked_qualifier = ""
            if hasattr(original, "linked_depth_delta_m"):
                original.linked_depth_delta_m = None
            return None
        part = _part_for(twins.parts(head), view, names)
        original.linked_qualifier = part.qualifier
        return part

    _link_all(record, qa, among_twins, lambda inv: several.get(id(inv), ""))
    for name, placed in shared.items():
        qa.extend(_shared_name_links(name, placed, twins))


def _part_for(parts: Sequence[Investigation], view: Any,
              names: Sequence[str]) -> Investigation:
    """The log, of one hole read as several logs of its name
    (:func:`report_ingest.shared_names.hole_parts`), a sheet or a row is
    linked to: the one holding the sample it names or a sample at its depth
    (:func:`_sample_for`), where one log does. Where several do (a sample
    name printed on two of the hole's sheets), the one whose sample's
    interval also meets the depth the sheet prints, the nearest where more
    than one does; else the first, as when the logs were not told apart."""
    if len(parts) < 2 or view is None:
        return parts[0]
    holding = [(p, _sample_for(p, view, names)) for p in parts]
    holding = [(p, found) for p, found in holding if found[0] is not None]
    if len(holding) == 1:
        return holding[0][0]
    agree: List[Tuple[float, int]] = []      # (how far off, which)
    for n, (_log, (sample, _delta, clash)) in enumerate(holding):
        gap = _interval_gap(sample, view) if clash is None else None
        if gap is not None:
            agree.append((gap, n))
    return holding[min(agree)[1]][0] if agree else parts[0]


def _capped(values: Sequence[str], n: int = 12) -> List[str]:
    """The first ``n`` values, and how many more there are."""
    values = list(values)
    if len(values) <= n:
        return values
    return values[:n] + [f"... {len(values) - n} more"]


def _shared_name_links(name: str,
                       placed: Sequence[Tuple[str, List[int], List[str], Any]],
                       twins: Any) -> List[QAEntry]:
    """The QA for the laboratory results naming ``name``, a name two or
    more holes are printed under: those linked to neither (``partial``,
    open), and those linked to one by what they print (a note)."""
    from report_ingest.shared_names import pages_span

    holes = len(twins.holes.get(fold_id(name), ())) if twins else 2
    count = {2: "two", 3: "three", 4: "four"}.get(holes, str(holes))
    out: List[QAEntry] = []
    refused = [p for p in placed if p[3] is None]
    kept = [p for p in placed if p[3] is not None]
    if refused:
        out.append(QAEntry(
            kind="partial", where=f"lab_tests.{name}", rank="open",
            detail=(f"{_counted(refused)} name {name!r}, which {count} holes "
                    f"of this report are printed under, and print nothing "
                    f"that tells which of them they are from (no report "
                    f"section, project, firm, date or place that is one "
                    f"hole's): they are linked to neither, and the hole is "
                    f"for a person to say"),
            values=_capped([f"{what}: {pages_span(pages)}"
                            for what, pages, _why, _hole in refused]),
            pages=sorted({int(p) for _w, pages, _y, _h in refused
                          for p in pages})))
    if kept:
        out.append(QAEntry(
            kind="note", where=f"lab_tests.{name}",
            detail=(f"{_counted(kept)} name {name!r}, which {count} holes of "
                    f"this report are printed under, and were linked to one "
                    f"of them by what they print"),
            values=_capped(list(dict.fromkeys(
                f"{what} -> {name} ({hole.qualifier}): {'; '.join(why)}"
                for what, _pages_, why, hole in kept))),
            pages=sorted({int(p) for _w, pages, _y, _h in kept
                          for p in pages})))
    return out


def _counted(placed: Sequence[Tuple[str, List[int], List[str], Any]]) -> str:
    rows = sum(1 for what, *_rest in placed if what.startswith("summary row"))
    tests = len(placed) - rows
    return " and ".join(part for part in (
        f"{tests} laboratory test(s)" if tests else "",
        f"{rows} summary row(s)" if rows else "") if part)


def _link_all(record: ReportRecord, qa: List[QAEntry],
              among_twins: Any, log_pages: Any = None) -> None:
    """Link every sheet and summary row (:func:`_link_lab_tests`).
    ``log_pages(log)`` gives the pages of a log of a hole read as several
    logs, empty for any other, for ``linked_log_pages``."""
    by_fold: Dict[str, Investigation] = {}
    by_key: Dict[str, List[Investigation]] = {}
    for inv in record.investigations:
        if fold_id(inv.investigation_id):
            by_fold.setdefault(fold_id(inv.investigation_id), inv)
            by_key.setdefault(hole_key(inv.investigation_id), []).append(inv)
    # The zero fold names a hole only when one logged spelling has its key:
    # "B-07" and "B-7" logged as two holes are two holes.
    by_id = {key: invs[0] for key, invs in by_key.items()
             if len({fold_id(i.investigation_id) for i in invs}) == 1}
    sample_names: Dict[str, List[Tuple[Investigation, Sample]]] = {}
    for inv in record.investigations:
        for sample in inv.samples:
            if hole_key(sample.sample_id):
                sample_names.setdefault(hole_key(sample.sample_id),
                                        []).append((inv, sample))

    # (the label it was read from, the hole, the pages, a summary row?)
    from_labels: List[Tuple[str, Investigation, List[int], bool]] = []
    clashes: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    # A depth printed with no unit, or only in the sample label, is read
    # for the link in the record's one depth unit, or the label's (RI-143);
    # (how, a summary row?, the pages) for one note.
    unit = depth_unit(record)
    read_depths: List[Tuple[str, bool, List[int]]] = []
    for original in record.lab_tests:
        if original.kind == "summary_table":
            _link_summary_rows(original, by_id, sample_names, by_fold,
                               from_labels, clashes, qa, unit, read_depths,
                               among_twins, log_pages)
            continue
        test, how = _depth_view(original, unit, need_hole=False)
        inv, through = _hole_for(test, by_id, sample_names, by_fold)
        if inv is not None:
            inv = among_twins(inv, original, list(test.pages), test.kind,
                              test)
            if inv is None:
                continue                  # its shared-name entry says why
        if inv is not None and through:
            from_labels.append((_label_read(original, through), inv,
                                list(test.pages), False))
        printed = fold_id(test.investigation_id)
        if not printed and inv is None:
            qa.append(QAEntry(
                kind="partial", where=f"lab_tests.{test.kind}",
                detail="the sheet names no exploration, so this test could "
                       "not be linked to the ground",
                pages=list(test.pages)))
            continue
        if inv is None:
            qa.append(QAEntry(
                kind="partial", where=f"lab_tests.{test.kind}",
                detail=f"the sheet names {test.investigation_id!r}, which is "
                       f"not among the explorations read from this report",
                values=[test.investigation_id],
                pages=list(test.pages)))
            continue
        original.linked_investigation_id = inv.investigation_id
        original.linked_qualifier = inv.qualifier
        original.linked_log_pages = log_pages(inv) if log_pages else ""
        if how:
            read_depths.append((how, False, list(original.pages)))
        sample, delta, clash = _sample_for(inv, test, by_fold)
        if clash is not None:
            _gather(clashes, clash, test.kind, test.pages)
        if sample is None:
            if clash is not None:
                continue                         # its conflict entry says why
            wanted = depth_m(test.depth_top)
            qa.append(QAEntry(
                kind="partial", where=f"lab_tests.{test.kind}",
                detail=(f"linked to {inv.investigation_id} but no sample of "
                        f"it sits within {DEPTH_TOL_M} m of the sheet's "
                        f"depth" if wanted is not None else
                        f"linked to {inv.investigation_id}; the sheet prints "
                        f"no depth, so no sample could be matched"),
                values=([_show(original.depth_top)] if original.depth_top
                        else []),
                pages=list(test.pages)))
            continue
        original.linked_sample_id = sample.sample_id
        original.linked_depth_delta_m = (None if delta is None
                                         else round(delta, 3))

    if from_labels:
        # One note for the report, not one a test: a reviewer wants to know
        # that these links were read out of a longer label, and which.
        shown: List[str] = []
        for label, inv, _pages, _row in from_labels:
            line = f"{label!r} -> {inv.investigation_id}"
            if line not in shown:
                shown.append(line)
        rows = sum(1 for *_rest, row in from_labels if row)
        tests = len(from_labels) - rows
        counted = " and ".join(
            part for part in (f"{tests} lab test(s)" if tests else "",
                              f"{rows} summary row(s)" if rows else "")
            if part)
        qa.append(QAEntry(
            kind="note", where="lab_tests",
            detail=(f"{counted} were linked to the hole at the head of a "
                    f"longer label -- the boring as printed (a hole, sample "
                    f"and depth run together), or the sample label where "
                    f"the sheet printed no boring; only holes read from this "
                    f"report's logs were looked for. A hole named after a "
                    f"word for an exploration (\"BH No.\", \"Bore hole\") "
                    f"by its own name, or by a number the logs give one "
                    f"hole, is linked so too"),
            values=shown[:12] + ([f"... {len(shown) - 12} more"]
                                 if len(shown) > 12 else []),
            pages=sorted({int(p) for _l, _i, pages, _r in from_labels
                          for p in pages})))
    if read_depths:
        qa.append(_read_depths_entry(read_depths, unit))
    qa.extend(_label_depth_entry(group) for group in clashes.values())


def _read_depths_entry(read_depths: Sequence[Tuple[str, bool, List[int]]],
                       unit: str) -> QAEntry:
    """One note for the tests and rows linked by a depth read as
    :func:`_depth_view` reads it (RI-143): how many printed their depths
    with no unit and were read in the record's one unit, and how many
    printed theirs only in the sample label."""
    def counted(how: str) -> str:
        tests = sum(1 for h, row, _p in read_depths if h == how and not row)
        rows = sum(1 for h, row, _p in read_depths if h == how and row)
        return " and ".join(part for part in (
            f"{tests} lab test(s)" if tests else "",
            f"{rows} summary row(s)" if rows else "") if part)
    parts: List[str] = []
    if any(h == "unit" for h, _r, _p in read_depths):
        parts.append(f"{counted('unit')} print their depths with no unit; "
                     f"they were read in {unit}, the one unit every other "
                     f"depth in this record is printed in")
    if any(h == "label" for h, _r, _p in read_depths):
        parts.append(f"{counted('label')} print a depth only in the sample "
                     f"label, or the boring as printed; it was read from "
                     f"there")
    return QAEntry(
        kind="note", where="lab_tests",
        detail="; ".join(parts) + " to link them to the logs. The printed "
                                  "values are kept as printed",
        pages=sorted({int(p) for _h, _r, pages in read_depths
                      for p in pages}))


def _label_read(test: Any, through: str) -> str:
    """The printed words a hole was read out of (see :func:`_hole_for`)."""
    return (test.investigation_id if through == "the boring as printed"
            else test.sample_id)


def _sample_for(inv: Investigation, test: Any,
                holes: Iterable[str] = ()
                ) -> Tuple[Optional[Sample], Optional[float],
                           Optional[Dict[str, Any]]]:
    """The sample this specimen came from, how far off it was, and what the
    label and the depth disagree on (None when they do not).

    ``test`` is a sheet or a summary row: anything with a printed
    ``investigation_id``, ``sample_id`` and ``depth_top``. ``holes`` are the
    folded names of the record's explorations.

    THE LABEL FIRST (RI-56). A label that names one of this hole's samples
    -- the sample label whole, as it always was, or a word of a longer label
    that IS one of the hole's sample names ("BH-7 U3" -> U3), where the
    label names no other hole -- is that sample, wherever the log puts it.
    A word names another hole only if it has a letter in it, or it is a
    number after a word for an exploration ("BH 2", "Hole 2") or standing
    alone as the sample label's first word ("2/S-3"): a number later in the
    label ("S-3 Jar 9"), one a letter runs on to ("Pit 9A"), and the
    numbers of a printed depth range ("4.0-5.5") or of a depth with its
    unit are not a hole (:func:`_numbers_named`). The
    depth decides only where the label names none of the hole's samples,
    or which of two the label named. A sheet's depth agrees with a sample
    when the printed interval, widened by :data:`DEPTH_TOL_M` each side,
    meets the log's interval for it (:func:`_interval_gap`): a sheet
    prints the interval it sampled, a log may hold one point inside it.
    Where they do not meet, the label is followed and the disagreement
    comes back for a QA entry. A label that names several of the hole's
    samples (a composite, "S-2,S-3") is linked to no sample where none of
    them, or more than one, meets the printed interval, and to the one that
    meets it where one does, which comes back to be said too (RI-71); a
    range of names ("U-21 to U-23") names every sample between its ends
    (:func:`_in_range`); a name several of the hole's samples carry is
    settled by the depth or not at all (:func:`_one_name`).
    """
    if test.sample_id:
        wanted = fold_id(test.sample_id)
        whole = [s for s in inv.samples if fold_id(s.sample_id) == wanted]
        if whole:
            return _one_name(inv, test, whole)
    label = f"{test.investigation_id} {test.sample_id}"
    words = {fold_id(w) for w in re.split(r"[^0-9A-Za-z-]+", label)
             if fold_id(w)}
    own = fold_id(inv.investigation_id)
    others = {fold_id(h) for h in holes
              if re.search(r"[A-Za-z]", fold_id(h))} - {own, ""}
    numbered = {hole_key(h) for h in holes
                if fold_id(h).isdigit()} - {hole_key(own), ""}
    named = [] if words & others or _numbers_named(
        test.investigation_id, test.sample_id) & numbered else [
        s for s in inv.samples
        if fold_id(s.sample_id) and fold_id(s.sample_id) in words
        and fold_id(s.sample_id) != own
        and not fold_id(s.sample_id).isdigit()]
    if not named:
        sample, delta = _sample_at_depth(inv, test)
        return sample, delta, None
    named = _in_range(inv, label, named)
    meets = [s for s in named if _interval_gap(s, test) is not None]
    if len({fold_id(s.sample_id) for s in meets}) == 1:
        nearest = _nearest(meets, test)
        return nearest, _interval_gap(nearest, test), _one_of(
            inv, test, named, nearest)
    if len({fold_id(s.sample_id) for s in named}) > 1:
        return None, None, _clash(inv, test, named, None)
    return _one_name(inv, test, named)


def _one_name(inv: Investigation, test: Any, same: Sequence[Sample]
              ) -> Tuple[Optional[Sample], Optional[float],
                         Optional[Dict[str, Any]]]:
    """:func:`_sample_for` for a label naming ``same``, the samples of the
    hole that carry one name.

    One sample is that sample, wherever the log puts it; a disagreement
    with the depth comes back to be said. Several samples of one hole can
    carry one name (a log that starts its numbering again, a sample type
    read as the name): the printed depth decides, the nearest of those
    within the printed interval widened by :data:`DEPTH_TOL_M` (RI-71).
    Where none is within it, or nothing printed gives a depth to tell them
    apart, the label names no one of them: the test is linked to the hole
    and to no sample, and the reason comes back to be said.
    """
    if len(same) == 1:
        return same[0], _delta(same[0], test), _clash(inv, test, same,
                                                      same[0])
    meets = [s for s in same if _interval_gap(s, test) is not None]
    if meets:
        nearest = _nearest(meets, test)
        return nearest, _delta(nearest, test), None
    there, _delta_there = _sample_at_depth(inv, test)
    return None, None, {
        "hole": inv.investigation_id, "named": [same[0].sample_id],
        "label": " ".join(str(x).strip() for x in (
            test.investigation_id, test.sample_id) if str(x).strip()),
        "depth": test.depth_top, "linked": None, "there": there,
        "within": [], "shared": [s.top for s in same]}


def _nearest(samples: Sequence[Sample], test: Any) -> Sample:
    """The sample nearest the printed interval, then nearest the printed
    top; the first of equals."""
    def key(sample: Sample) -> Tuple[float, float]:
        gap, delta = _interval_gap(sample, test), _delta(sample, test)
        return (math.inf if gap is None else gap,
                math.inf if delta is None else delta)
    return min(samples, key=key)


#: The words a label prints between the two ends of a range of names.
_RANGE_WORD = re.compile(r"\s*(?:-|–|—|to|thru|through|a|à)\s*", re.I)
#: A folded sample name as its letters, its number and any letters after.
_NAME_NUMBER = re.compile(r"(\D*)(\d+)([A-Z]*)")


def _in_range(inv: Investigation, label: str, named: Sequence[Sample]
              ) -> List[Sample]:
    """``named``, with every sample of ``inv`` a range of names in
    ``label`` covers (RI-71).

    Two named samples with only a range word between them ("U-21 to
    U-23", "U21 - U23", "U-21 a U-23") name the samples between them as
    well: those of the hole whose names are the ends' letters with a number
    between the ends' numbers ("U-22", "U-22A"). Ends whose letters differ,
    or that carry no number, name themselves only.
    """
    names = {fold_id(s.sample_id) for s in named}
    if len(names) < 2:
        return list(named)
    words = [m for m in re.finditer(r"[0-9A-Za-z-]+", label)
             if fold_id(m.group(0)) in names]
    out = list(named)
    for first, last in zip(words, words[1:]):
        if not _RANGE_WORD.fullmatch(label[first.end():last.start()]):
            continue
        ends = [_NAME_NUMBER.fullmatch(fold_id(m.group(0)))
                for m in (first, last)]
        if None in ends or ends[0].group(1) != ends[1].group(1):
            continue
        low, high = sorted(int(end.group(2)) for end in ends)
        for sample in inv.samples:
            got = _NAME_NUMBER.fullmatch(fold_id(sample.sample_id))
            if got and got.group(1) == ends[0].group(1) and \
                    low < int(got.group(2)) < high and \
                    not any(sample is s for s in out):
                out.append(sample)
    if len(out) == len(named):
        return out
    # In the log's order, as a reviewer reads them there.
    order = {id(s): i for i, s in enumerate(inv.samples)}
    return sorted(out, key=lambda s: order.get(id(s), len(order)))


def _one_of(inv: Investigation, test: Any, named: Sequence[Sample],
            linked: Sample) -> Optional[Dict[str, Any]]:
    """What is said of a label naming several of the hole's samples, of
    which only ``linked`` meets the printed interval, or None for a label
    naming one (RI-71): the test is linked to that one, and the others the
    label names are not linked."""
    names: List[str] = []
    for s in named:
        if s.sample_id not in names:
            names.append(s.sample_id)
    if len({fold_id(name) for name in names}) < 2:
        return None
    return {"hole": inv.investigation_id, "named": names,
            "label": " ".join(str(x).strip() for x in (
                test.investigation_id, test.sample_id) if str(x).strip()),
            "depth": test.depth_top, "linked": linked, "there": linked,
            "within": [linked.sample_id], "one of": True}


#: A number after a word for an exploration ("BH 2", "Hole-2", "TP2"),
#: not the start of a decimal, and not one a letter runs on to: "Pit 9A"
#: and "BH-9A" are not hole 9.
_HOLE_NUMBER = re.compile(
    r"\b(?:bh|bore ?hole|boring|hole|tp|test ?pit|pit|trench|well|cpt)"
    r"\s*[-#.:]?\s*(\d+)(?!\d|[.,]\d|[A-Za-z])", re.I)
#: A number standing alone: not part of a word, a range ("4-5", "4 to 5",
#: "4 a 5", "4 à 5", "4 through 5") or a decimal, and not a depth (a unit
#: after it).
_LONE_NUMBER = re.compile(
    r"(?<![\w.,\-–])(?<![-–]\s)(?<!to\s)(\d+)"
    r"(?![\w\-–]|[.,]\d|\s*(?:-|–|to|a|à|through|thru)\s*\d|"
    r"\s*(?:mm|cm|m|ft|feet|in)\b|\s*['\"])", re.I)


def _numbers_named(boring: Any, sample: Any = "") -> Set[str]:
    """The bare numbers a printed boring and sample label give as a hole's
    name, folded as :func:`hole_key` folds them.

    A number after a word for an exploration, anywhere ("BH 2, S-3"), and
    a number standing alone as the FIRST word of the sample label ("2/S-3",
    "9 S-3"). A number later in the label is the sample's own or a depth
    ("S-3 Jar 9", "S-3 #9", "S-3 (9)", "S-3 @ 9"), never a hole (RI-71).
    Never one a letter runs on to ("Pit 9A"), one inside a word, a range or
    a decimal, nor a depth with its unit.
    """
    found = [m.group(1) for m in _HOLE_NUMBER.finditer(
        f"{boring or ''} {sample or ''}")]
    lone = _LONE_NUMBER.match(str(sample or "").strip())
    if lone:
        found.append(lone.group(1))
    return {hole_key(n) for n in found}


def _clash(inv: Investigation, test: Any, named: Sequence[Sample],
           linked: Optional[Sample]) -> Optional[Dict[str, Any]]:
    """What the label and the depth disagree on, or None.

    None where they agree, and where there is nothing to disagree with: the
    sheet prints no depth that converts, or the log gives none of the named
    samples a depth.
    """
    if depth_m(test.depth_top) is None:
        return None
    if linked is not None and _interval_gap(linked, test) is not None:
        return None
    if all(depth_m(s.top) is None for s in named):
        return None
    names: List[str] = []
    for s in named:
        if s.sample_id not in names:
            names.append(s.sample_id)
    there, _delta_there = _sample_at_depth(inv, test)
    within: List[str] = []
    for s in named:
        if _interval_gap(s, test) is not None and s.sample_id not in within:
            within.append(s.sample_id)
    return {"hole": inv.investigation_id, "named": names,
            "label": " ".join(str(x).strip() for x in (
                test.investigation_id, test.sample_id) if str(x).strip()),
            "depth": test.depth_top, "linked": linked,
            "there": there, "within": within}


def _gather(clashes: Dict[Tuple[Any, ...], Dict[str, Any]],
            clash: Dict[str, Any], kind: str, pages: Sequence[int]) -> None:
    """Add one label/depth disagreement to its group.

    One entry for each sample a label names at each printed depth, however
    many sheets and rows print it: five sheets cut from one sample are one
    disagreement between its label and the log, not five.
    """
    key = (clash["hole"], tuple(clash["named"]), _show(clash["depth"]),
           "shared" in clash, "one of" in clash)
    group = clashes.setdefault(key, {**clash, "kinds": [], "labels": [],
                                     "pages": set()})
    group["kinds"].append(kind)
    if clash["label"] not in group["labels"]:
        group["labels"].append(clash["label"])
    group["pages"].update(int(p) for p in pages)


def _label_depth_entry(group: Dict[str, Any]) -> QAEntry:
    """The QA entry for labels that name a sample the depth does not."""
    hole, depth = group["hole"], _show(group["depth"])
    rows = group["kinds"].count("summary_table")
    sheets = len(group["kinds"]) - rows
    who = " and ".join(part for part in (
        f"{sheets} lab sheet{'s' if sheets > 1 else ''}" if sheets else "",
        f"{rows} summary row{'s' if rows > 1 else ''}" if rows else "")
        if part)
    name, prints = (("names", "prints") if len(group["kinds"]) == 1
                    else ("name", "print"))
    there = group["there"]
    linked = group["linked"]
    unit = getattr(group["depth"], "unit", "")
    if "shared" in group:
        return _shared_name_entry(group, who, name, prints)
    kind = "conflict"
    if "one of" in group:
        others = [n for n in group["named"]
                  if fold_id(n) != fold_id(linked.sample_id)]
        kind = "partial"
        detail = (f"{who} {name} samples {', '.join(group['named'])} of "
                  f"{hole} and {prints} a depth of {depth}; its log puts "
                  f"only {linked.sample_id} within {DEPTH_TOL_M} m of the "
                  f"printed interval. Linked to {linked.sample_id}; "
                  f"{', '.join(others)} not linked")
    elif linked is not None:
        has = (f"has {there.sample_id or 'an unnamed sample'} there"
               if there is not None else "has no sample there")
        detail = (f"{who} {name} sample {linked.sample_id} of {hole} and "
                  f"{prints} a depth of {depth}; its log puts "
                  f"{linked.sample_id} at {_depth_in(linked.top, unit)} and "
                  f"{has}. Linked to {linked.sample_id} by the label; the "
                  f"depth is kept as printed")
    elif len(group.get("within", [])) > 1:
        detail = (f"{who} {name} samples {', '.join(group['named'])} of "
                  f"{hole} and {prints} a depth of {depth}; its log puts "
                  f"{', '.join(group['within'])} within {DEPTH_TOL_M} m of "
                  f"the printed interval, so the specimen is not one of them "
                  f"alone. Linked to {hole} and to no sample")
    else:
        detail = (f"{who} {name} samples {', '.join(group['named'])} of "
                  f"{hole} and {prints} a depth of {depth}; its log puts none "
                  f"of them within {DEPTH_TOL_M} m of the printed interval. "
                  f"Linked to {hole} and to no sample")
    labels = group["labels"]
    return QAEntry(kind=kind,
                   where=f"{hole}.lab_tests",
                   detail=detail,
                   values=[f"label {label}" for label in labels[:3]]
                   + ([f"... {len(labels) - 3} more"]
                      if len(labels) > 3 else [])
                   + [f"depth {depth}"],
                   pages=sorted(group["pages"]))


def _shared_name_entry(group: Dict[str, Any], who: str, name: str,
                       prints: str) -> QAEntry:
    """The QA entry for a label naming a sample whose name several of the
    hole's samples carry, none of which the printed depth settles
    (:func:`_one_name`): a ``conflict`` where the sheet's depth and the
    log's disagree, a ``partial`` where either gives no depth to compare."""
    hole, sample = group["hole"], group["named"][0]
    unit = getattr(group["depth"], "unit", "")
    logged = [_depth_in(top, unit) for top in group["shared"]
              if depth_m(top) is not None]
    head = (f"{who} {name} sample {sample} of {hole}, a name its log gives "
            f"{len(group['shared'])} samples"
            + (f" (at {', '.join(logged)})" if logged else ""))
    depth = _show(group["depth"])
    if depth_m(group["depth"]) is not None and logged:
        kind = "conflict"
        detail = (f"{head}, and {prints} a depth of {depth}; none of them is "
                  f"within {DEPTH_TOL_M} m of the printed interval. Linked "
                  f"to {hole} and to no sample")
    else:
        kind = "partial"
        why = (f"{prints} no depth to tell them apart"
               if group["depth"] is None else
               f"{prints} a depth of {depth} in a unit that does not "
               f"convert to compare" if depth_m(group["depth"]) is None else
               f"{prints} a depth of {depth}, and its log gives none of them "
               f"a depth")
        detail = f"{head}, and {why}. Linked to {hole} and to no sample"
    labels = group["labels"]
    return QAEntry(kind=kind, where=f"{hole}.lab_tests", detail=detail,
                   values=[f"label {label}" for label in labels[:3]]
                   + ([f"... {len(labels) - 3} more"]
                      if len(labels) > 3 else [])
                   + ([f"depth {_show(group['depth'])}"]
                      if group["depth"] is not None else []),
                   pages=sorted(group["pages"]))


def _depth_in(value: Optional[Quantity], unit: str) -> str:
    """A log's depth for a QA line, in ``unit`` (the printed depth's, so the
    line reads in one unit), to the nearest centimetre or the nearest step
    of that unit above it (0.1 ft): a depth read off a drawn log carries
    more figures than the log was drawn to. In the log's own unit where
    ``unit`` does not convert."""
    metres = depth_m(value)
    if metres is None or value is None:
        return _show(value)
    per_unit = depth_m(Quantity(value=1.0, unit=unit)) if unit else None
    if not per_unit:
        unit = value.unit
        per_unit = depth_m(Quantity(value=1.0, unit=unit)) if unit else None
    if not per_unit:
        return _show(value)
    places = max(0, round(math.log10(per_unit / 0.01)))
    return f"{round(metres / per_unit, places):g} {unit}"


def _at_depth(sample: Sample, test: Any) -> Optional[float]:
    """How far the sheet's depth sits from ``sample``, in metres, when it
    is within :data:`DEPTH_TOL_M` of it; else None."""
    depth, top = depth_m(test.depth_top), depth_m(sample.top)
    if depth is None or top is None:
        return None
    bottom = depth_m(sample.bottom)
    if bottom is not None and top - DEPTH_TOL_M <= depth <= \
            bottom + DEPTH_TOL_M:
        delta = 0.0 if top <= depth <= bottom else min(
            abs(depth - top), abs(depth - bottom))
    else:
        delta = abs(depth - top)
    return delta if delta <= DEPTH_TOL_M else None


def _interval_gap(sample: Sample, test: Any) -> Optional[float]:
    """How far the sheet's printed interval sits from the log's interval
    for ``sample``, in metres, when the printed one, widened by
    :data:`DEPTH_TOL_M` each side, meets it; else None.

    A point stands for an interval of no length: a sheet that prints only a
    top, or a log sample with no bottom. Used for the label check only; a
    link by depth alone stays with :func:`_at_depth`.
    """
    top, s_top = depth_m(test.depth_top), depth_m(sample.top)
    if top is None or s_top is None:
        return None
    bottom = depth_m(getattr(test, "depth_bottom", None))
    if bottom is None or bottom < top:
        bottom = top
    s_bottom = depth_m(sample.bottom)
    if s_bottom is None or s_bottom < s_top:
        s_bottom = s_top
    gap = max(0.0, s_top - bottom, top - s_bottom)
    return gap if gap <= DEPTH_TOL_M else None


def _sample_at_depth(inv: Investigation, test: Any
                     ) -> Tuple[Optional[Sample], Optional[float]]:
    """The sample of ``inv`` at the sheet's depth, by depth alone."""
    best: Optional[Sample] = None
    best_delta: Optional[float] = None
    for sample in inv.samples:
        delta = _at_depth(sample, test)
        if delta is not None and (best_delta is None or delta < best_delta):
            best, best_delta = sample, delta
    return best, best_delta


def _delta(sample: Sample, test: Any) -> Optional[float]:
    top, depth = depth_m(sample.top), depth_m(test.depth_top)
    if top is None or depth is None:
        return None
    return abs(depth - top)


def _link_summary_rows(test: LabTest, by_id: Dict[str, Investigation],
                       samples: Dict[str, List[Tuple[Investigation, Sample]]],
                       by_fold: Dict[str, Investigation],
                       from_labels: List[Tuple[str, Investigation, List[int],
                                               bool]],
                       clashes: Dict[Tuple[Any, ...], Dict[str, Any]],
                       qa: List[QAEntry], unit: str = "",
                       read_depths: Optional[List[Tuple[str, bool,
                                                        List[int]]]] = None,
                       among_twins: Any = None,
                       log_pages: Any = None) -> None:
    """Each row of a summary table linked as a sheet is, from its own columns.

    A row prints its own boring, sample and depth, and is linked by the
    sheet's rules (:func:`_hole_for`, :func:`_sample_for`) into its own link
    fields (RI-56); the printed columns never change. The holes the table
    names that no log holds are one entry for the table; the DIGGS writer
    gives each such name the hole a sheet printing it would get.

    A row that prints no boring of its own may name its hole in another
    column, headed for one ("Site ref.", "Location"; kept in ``other``):
    that cell is read as a boring column's would be (:func:`_hole_in_column`,
    RI-143). A table printing such a column once for a run of rows (a
    merged cell) leaves the rows below it blank there: a row printing no
    hole cell at all, under a row holed so, deeper than it, takes its hole
    where that hole's log has a sample at the row's depth -- unless the run
    of such rows ends at a row naming another hole, deeper still, with no
    return to a shallower depth between, whose log has a sample at the
    row's depth too: the cell may be printed on the middle row of the group
    it spans, so such rows (those after the last that hole's log has no
    sample at) may as well be that hole's, and are left unplaced. All are
    said in one note for the table. A depth printed with no unit, or only
    in the sample label, is read as a sheet's is (:func:`_depth_view`).

    A row naming a name two or more holes are printed under is linked to
    one of them only where it prints what tells them apart
    (``among_twins``, RI-135); a row below it that prints its hole cell
    blank takes the hole that row was linked to.
    """
    result = test.result
    rows = list(getattr(result, "rows", ()) or ())
    refused: Set[int] = set()            # id() of rows linked to neither twin

    def twin(inv: Optional[Investigation], original: Any, number: int,
             view: Any) -> Optional[Investigation]:
        if inv is None or among_twins is None:
            return inv
        got = among_twins(inv, original, list(test.pages),
                          f"summary row {number}", view)
        if got is None:
            refused.add(id(original))
        return got

    missing: List[str] = []
    # (the hole, the top in metres) of the row above, holed by a column of
    # its own or carried down from one.
    above: Optional[Tuple[Investigation, float]] = None
    by_column: List[str] = []
    # Each row as decided: [the row, its view, how its depth was read, its
    # hole, where the hole was read, whether it was carried down, its top].
    decided: List[List[Any]] = []
    run: List[int] = []                  # rows carried down since a holed one
    left = 0
    for number, original in enumerate(rows, 1):
        row, how = _depth_view(original, unit, need_hole=False)
        inv, through = _hole_for(row, by_id, samples, by_fold)
        top = depth_m(row.depth_top)
        inv = twin(inv, original, number, row)
        names = inv is not None or bool(fold_id(row.investigation_id)) \
            or id(original) in refused
        down = False
        if inv is None and not fold_id(row.investigation_id) \
                and id(original) not in refused:
            inv, heading, cell, printed = _hole_in_column(
                row, by_id, samples, by_fold)
            inv = twin(inv, original, number, row)
            names = printed
            if inv is not None:
                by_column.append(f"{heading!r} {cell!r} -> "
                                 f"{inv.investigation_id}")
            elif above is not None and not printed and top is not None \
                    and top > above[1] and \
                    _sample_at_depth(above[0], row)[0] is not None:
                inv, down = above[0], True
            above = (inv, top) if inv is not None and top is not None \
                else None
        else:
            above = None
        if run and not down:
            last = decided[run[-1]]
            if names and inv is not last[3] and top is not None \
                    and top > last[6]:
                # The cell may be this row's group's, printed on its middle
                # row: the carried rows this row's hole has a sample at too,
                # after the last it has none at (a group is one run of rows),
                # may as well be its own.
                either: List[int] = []
                for i in run:
                    if inv is not None and \
                            _sample_at_depth(inv, decided[i][1])[0] \
                            is not None:
                        either.append(i)
                    else:
                        either = []
                for i in either:
                    decided[i][3] = None
                left += len(either)
            run = []
        if down:
            run.append(len(decided))
        decided.append([original, row, how, inv, through, down, top])
    carried = sum(1 for d in decided if d[5] and d[3] is not None)
    for original, row, how, inv, through, _down, _top in decided:
        if inv is None:
            if fold_id(row.investigation_id) and \
                    id(original) not in refused and \
                    row.investigation_id not in missing:
                missing.append(row.investigation_id)
            continue
        if through:
            from_labels.append((_label_read(original, through), inv,
                                list(test.pages), True))
        original.linked_investigation_id = inv.investigation_id
        original.linked_qualifier = inv.qualifier
        original.linked_log_pages = log_pages(inv) if log_pages else ""
        if how and read_depths is not None:
            read_depths.append((how, True, list(test.pages)))
        sample, _delta_m, clash = _sample_for(inv, row, by_fold)
        if clash is not None:
            _gather(clashes, clash, test.kind, test.pages)
        if sample is not None:
            original.linked_sample_id = sample.sample_id
    if missing:
        qa.append(QAEntry(
            kind="partial", where="lab_tests.summary_table",
            detail=f"{len(missing)} exploration(s) named in the summary "
                   f"table are not among those read from this report",
            values=missing, pages=list(test.pages)))
    if by_column:
        shown = list(dict.fromkeys(by_column))
        qa.append(QAEntry(
            kind="note", where="lab_tests.summary_table",
            detail=(f"{len(by_column)} summary row(s) print no boring of "
                    f"their own and name their hole in another column; "
                    f"linked to the hole that column names"
                    + (f", and {carried} row(s) below them that print that "
                       f"column blank (a cell spanning the rows) to the hole "
                       f"of the row above, where its log has a sample at the "
                       f"row's depth" if carried else "")
                    + (f"; {left} row(s) printing it blank above a row of "
                       f"another hole deeper down, whose log has a sample at "
                       f"their depths too, are left unplaced (a spanning "
                       f"cell may be printed on the middle row of its group)"
                       if left else "")),
            values=shown[:12] + ([f"... {len(shown) - 12} more"]
                                 if len(shown) > 12 else []),
            pages=list(test.pages)))


# -- the counts -------------------------------------------------------------

# -- the calculations against the ground and against the narrative ----------

#: What a calculation's printed label has to say for its value to be the
#: thing the narrative recommends. Deliberately narrow: a printout states
#: dozens of pressures and only the one it CALLS a bearing pressure is the
#: recommendation, so a loose rule would raise a conflict on every trial
#: value on the page.
_BEARING_WORDS = ("bearing pressure", "bearing capacity", "allowable bearing",
                  "net allowable", "design bearing", "qall", "q_all")
_SETTLEMENT_WORDS = ("total settlement", "cumulative settlement",
                     "maximum settlement", "estimated settlement")
_SITE_CLASS_WORDS = ("site class", "seismic site class")
#: A settlement the narrative states in prose: the word, then a number, then
#: a unit of length. Nothing is inferred from a sentence without all three.
_PROSE_SETTLEMENT = re.compile(
    r"settlements?\b[^.]{0,80}?(\d+(?:\.\d+)?)\s*"
    r"(inch(?:es)?|in\.?|mm|millimet(?:er|re)s?|cm|m\b|ft\b|feet)", re.I)
#: The identifier a calculation's subject may name: a hole, a pit, a
#: sounding. The shape every log in this corpus uses, and nothing else --
#: a subject naming "the north wing" names no exploration and should not be
#: forced to name one.
_SUBJECT_ID = re.compile(r"\b([A-Za-z]{1,4}[- ]?\d{1,3}[A-Za-z]?)\b")


def _labelled(calc: Any, words: Sequence[str]) -> List[Any]:
    """The calculation's results whose printed label says one of ``words``."""
    out = []
    for row in calc.results:
        text = str(row.name or "").lower()
        if any(word in text for word in words):
            out.append(row)
    return out


def _link_calculations(record: ReportRecord, qa: List[QAEntry]) -> None:
    """A calculation whose subject names a boring is linked to that boring.

    A settlement worked for B-4 and the log of B-4 belong together, and a
    reviewer asking "what did they do with this hole" should get both. The
    subject stays the page's own words; the link goes in a field of its own
    and stays empty when the subject names no exploration, which is the
    ordinary case -- most calculations are for a structure.
    """
    by_id: Dict[str, Investigation] = {}
    by_key: Dict[str, List[Investigation]] = {}
    for inv in record.investigations:
        key = fold_id(inv.investigation_id)
        if key:
            by_id.setdefault(key, inv)
            by_key.setdefault(hole_key(inv.investigation_id), []).append(inv)
    if not by_id:
        return
    # The zero fold names a hole only when one logged spelling has its key.
    unique = {key: invs[0] for key, invs in by_key.items()
              if len({fold_id(i.investigation_id) for i in invs}) == 1}
    for calc in record.calculations:
        for token in _SUBJECT_ID.findall(calc.subject or ""):
            inv = by_id.get(fold_id(token)) or unique.get(hole_key(token))
            if inv is not None:
                calc.linked_investigation_id = inv.investigation_id
                break


def _check_calculations(record: ReportRecord, qa: List[QAEntry]) -> None:
    """The calculations against what the narrative says the report concluded.

    The narrative recommends a bearing pressure, states a settlement and
    names a site class; the appendix works them out. Where the two differ
    the record says so and settles nothing -- a report whose text recommends
    3 ksf over an appendix that computed 2 is telling a reviewer something,
    and picking one would destroy the only evidence there is to look at.
    """
    for calc in record.calculations:
        if calc.kind == "shallow_foundation_bearing":
            _check_against_bearing(record, calc, qa)
        elif calc.kind == "settlement":
            _check_against_settlement(record, calc, qa)
        elif calc.kind == "site_response":
            _check_against_site_class(record, calc, qa)


def _conflict(qa: List[QAEntry], calc: Any, row: Any, stated: str,
              said: str, where: str) -> None:
    qa.append(QAEntry(
        kind="disagreement", where=where,
        detail=f"the calculation on page(s) "
               f"{', '.join(str(p) for p in calc.pages)} prints "
               f"{row.name!r} as {said}, and the narrative says {stated}; "
               f"both are in the record and neither was changed",
        values=[f"calculation {said}", f"narrative {stated}"],
        pages=list(calc.pages)))


def _check_against_bearing(record: ReportRecord, calc: Any,
                           qa: List[QAEntry]) -> None:
    wanted = record.general.bearingCapacityValues
    if not wanted:
        return
    for row in _labelled(calc, _BEARING_WORDS):
        if row.value is None:
            continue
        verdicts = [_same(row.value, one.value) for one in wanted]
        if any(v is True for v in verdicts) or not any(
                v is False for v in verdicts):
            continue
        _conflict(qa, calc, row, "; ".join(_show(one.value) for one in wanted),
                  _show(row.value), f"calculations.{calc.kind}")


def _check_against_settlement(record: ReportRecord, calc: Any,
                              qa: List[QAEntry]) -> None:
    prose = " ".join(str(x) for x in
                     (record.general.bearingCapacity or [])
                     + (record.general.recommendedFoundations or []))
    match = _PROSE_SETTLEMENT.search(prose)
    if match is None:
        return
    stated = Quantity(value=float(match.group(1)),
                      unit=_SETTLEMENT_UNITS.get(match.group(2).lower().strip(
                          "."), match.group(2)))
    rows = _labelled(calc, _SETTLEMENT_WORDS)
    verdicts = [(row, _same(row.value, stated)) for row in rows
                if row.value is not None]
    if not verdicts or any(v is True for _row, v in verdicts):
        return
    for row, verdict in verdicts:
        if verdict is False:
            _conflict(qa, calc, row, _show(stated), _show(row.value),
                      f"calculations.{calc.kind}")


#: What the prose's own spelling of a length means to the record.
_SETTLEMENT_UNITS: Dict[str, str] = {
    "inch": "in", "inches": "in", "in": "in", "mm": "mm",
    "millimeter": "mm", "millimeters": "mm", "millimetre": "mm",
    "millimetres": "mm", "cm": "cm", "m": "m", "ft": "ft", "feet": "ft",
}


#: The letter a site class IS. A standalone A to F, because "Site Class D"
#: is a D and the letters of the words "Site Class" are not site classes --
#: a set of every A-to-F character in the string would call every answer a
#: match for every other.
_SITE_CLASS_LETTER = re.compile(r"\b([A-F])\b")


def _check_against_site_class(record: ReportRecord, calc: Any,
                              qa: List[QAEntry]) -> None:
    stated = str(record.natural_hazards.siteClass or "").strip()
    letters = set(_SITE_CLASS_LETTER.findall(stated.upper()))
    if not letters:
        return
    for row in _labelled(calc, _SITE_CLASS_WORDS):
        said = str(row.text or "").strip()
        got = set(_SITE_CLASS_LETTER.findall(said.upper()))
        if not got or got & letters:
            continue
        _conflict(qa, calc, row, stated, said,
                  f"calculations.{calc.kind}")


def _found_by_kind(record: ReportRecord) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for inv in record.investigations:
        out.setdefault(inv.kind, []).append(inv.investigation_id)
    return out


def _check_counts(record: ReportRecord, qa: List[QAEntry]) -> None:
    found = _found_by_kind(record)
    for field, (kinds, plural) in _COUNT_KINDS.items():
        stated = getattr(record.general, field, None)
        if stated is None:
            continue
        got = sum(len(found.get(kind, ())) for kind in kinds)
        if got == stated:
            continue
        qa.append(QAEntry(
            kind="count_mismatch", where=f"general.{field}",
            detail=f"the narrative states {stated} {plural}; {got} were read "
                   f"from the appendix",
            values=[f"narrative {stated}", f"found {got}"]))


def _check_dictionaries(record: ReportRecord, qa: List[QAEntry]) -> None:
    found = _found_by_kind(record)
    pairs = (("boringDictionary", ("boring",), "boring"),
             ("testPitDictionary", ("test_pit",), "test pit"))
    for field, kinds, name in pairs:
        stated = getattr(record.general, field, None)
        if not stated:
            continue
        ids: List[str] = []
        for kind in kinds:
            ids.extend(found.get(kind, ()))
        # hole_key, not fold_id: a narrative's "B-1" is a log's "B-01".
        folded = {hole_key(x) for x in ids}
        named = {hole_key(x) for x in stated}
        missing = [x for x in stated if hole_key(x) not in folded]
        extra = [x for x in ids if hole_key(x) not in named]
        if missing:
            qa.append(QAEntry(
                kind="count_mismatch", where=f"general.{field}",
                detail=f"the narrative names {len(missing)} {name}(s) that no "
                       f"log in this report carries",
                values=missing))
        if extra:
            qa.append(QAEntry(
                kind="count_mismatch", where=f"general.{field}",
                detail=f"{len(extra)} {name} log(s) were read that the "
                       f"narrative does not name",
                values=extra))


def _fill_found_counts(record: ReportRecord) -> None:
    found = _found_by_kind(record)
    record.narrative.found_ids = {k: list(v) for k, v in found.items()}
    counts = {f"{k}s": len(v) for k, v in found.items()}
    counts["lab_tests"] = len(record.lab_tests)
    record.narrative.found_counts = counts


# -- the summary table against the sheets -----------------------------------

def _result_value(test: LabTest, names: Sequence[str]) -> Any:
    result = test.result
    if result is None:
        return None
    for name in names:
        value = getattr(result, name, None)
        if value is not None:
            return value
    return None


def _cross_check_summary_table(record: ReportRecord,
                               qa: List[QAEntry]) -> None:
    """Every value the table and a sheet both carry, compared.

    A disagreement is recorded with BOTH values and BOTH pages and is never
    resolved: the table is a transcription and the sheet is the measurement,
    but which of the two was mis-transcribed is a question for the pages.
    A row and a sheet are compared only where :func:`_summary_matches`
    joins them; where it cannot tell two specimens apart it joins neither
    and says so in a note (RI-140).
    """
    tables = [t for t in record.lab_tests if t.kind == "summary_table"]
    sheets = [t for t in record.lab_tests if t.kind != "summary_table"]
    if not tables or not sheets:
        return
    unit = depth_unit(record)
    for table in tables:
        rows = list(getattr(table.result, "rows", ()) or ())
        joined, ties = _summary_matches(rows, sheets, unit=unit)
        for row, got in zip(rows, joined):
            for sheet in got:
                for column, names in CROSS_CHECKS.items():
                    ours = getattr(row, column, None)
                    theirs = _result_value(sheet, names)
                    if _same(ours, theirs) is False:
                        where = (f"{row.investigation_id or (lab_hole(row) + ' (linked)' if lab_hole(row) else '?')} "
                                 f"{_show(row.depth_top) if row.depth_top else ''}"
                                 ).strip()
                        qa.append(QAEntry(
                            kind="conflict",
                            where=f"lab_tests.{column} at {where}",
                            detail=f"the summary table and the {sheet.kind} "
                                   f"sheet give different values for "
                                   f"{column}; both are recorded",
                            values=[f"summary table {_show(ours)}",
                                    f"{sheet.kind} sheet {_show(theirs)}"],
                            pages=sorted(set(list(table.pages)
                                             + list(sheet.pages)))))
        _note_ties(table, ties, qa)


# -- which sheet a summary row is (RI-140) -----------------------------------
#
# A summary row is one specimen's results gathered from several tests, and a
# sheet is one test on one specimen. Matching them by hole and depth window
# alone joined an index-test row to another sample's chemistry sheet in the
# same hole a few centimetres off (RI-115's review, on a saved record): the
# cross-check then compared nothing, and the checklist took the row's
# missing Atterberg sheet for found. So a row and a sheet are joined only
# when they share a test, and each picks the closest of its rivals.

#: Two depths this close, in metres, are one depth: a depth printed in feet
#: on one side and in metres to the centimetre on the other, or 1.5 ft
#: beside 0.45 m. An interval matches EXACTLY when its top does and its
#: bottom does too (or neither side prints a bottom); anything else inside
#: :data:`DEPTH_TOL_M` is a window match.
EXACT_TOL_M = 0.01

#: ``summary-table column -> the kinds of sheet that run that test``. A
#: column names the test: a row reporting a liquid limit came off an
#: Atterberg sheet, whatever that sheet's own result came back holding (a
#: non-plastic soil prints no limits); a qu may be a UU triaxial's, which
#: reports su; a USCS symbol is read off the limits and the grading. A
#: value both sides carry (:data:`SHARED_VALUES`) joins them as well: the
#: water content an Atterberg sheet prints beside its own results.
COLUMN_KINDS: Dict[str, Tuple[str, ...]] = {
    "ll": ("atterberg",), "pl": ("atterberg",), "pi": ("atterberg",),
    "wc": ("moisture_content",),
    "uscs": ("atterberg", "gradation"),
    "percent_passing": ("gradation",), "fines_percent": ("gradation",),
    "sand_percent": ("gradation",), "gravel_percent": ("gradation",),
    "silt_clay_percent": ("gradation",),
    "wet_density": ("density",), "dry_density": ("density",),
    "max_dry_density": ("compaction",), "optimum_wc": ("compaction",),
    "qu": ("unconfined", "unconfined_rock", "triaxial"),
    "su": ("unconfined", "triaxial"),
    "c": ("triaxial", "direct_shear"),
    "phi_deg": ("triaxial", "direct_shear"),
    "swell_percent": ("swell_consolidation",),
    "organic_percent": ("organic_content",),
    "pH": ("chemical",), "resistivity": ("chemical",),
    "sulfate": ("chemical",), "chloride": ("chemical",),
    "sulfides": ("chemical",), "redox": ("chemical",),
}

#: ``a word of a column heading the row keeps in "other" -> the kinds of
#: sheet that run that test``: the columns a summary table carries that the
#: row has no field for. Words are folded (:func:`fold_id`, primes dropped,
#: so ``p'c`` is PC, and the dots of an initialism, so ``C.B.R.`` is CBR); a
#: run of words is matched as one ("specific gravity"). A swell is a
#: consolidometer's or a CBR's (the swell over the soak).
_HEADING_KINDS: Dict[str, Tuple[str, ...]] = {
    "CBR": ("cbr",),
    "GS": ("specific_gravity",), "SPECIFICGRAVITY": ("specific_gravity",),
    "PERMEABILITY": ("permeability",), "K": ("permeability",),
    "CC": ("swell_consolidation",), "CR": ("swell_consolidation",),
    "PC": ("swell_consolidation",),
    "CONSOLIDATION": ("swell_consolidation",),
    "SWELL": ("swell_consolidation", "cbr"),
}

#: ``summary-table column -> the fields a sheet's result may call it``, for
#: JOINING a row to a sheet: :data:`CROSS_CHECKS`, plus the names a result
#: gives a value that are not compared (a consolidation sheet's dry unit
#: weight beside the row's dry density).
SHARED_VALUES: Dict[str, Tuple[str, ...]] = {
    column: names + (("dry_unit_weight",) if column == "dry_density" else ())
    for column, names in CROSS_CHECKS.items()}

#: Words a cell prints for "nothing here", folded (:func:`fold_id`): a
#: dash folds to nothing; these are "n/a", "not tested" and the like,
#: abbreviated and spelled out. "NP", "ND" and "<10" are results and stay
#: readings.
_EMPTY_CELL_WORDS: Tuple[str, ...] = (
    "NA", "NT", "NOTTESTED", "NOTRUN", "NOTPERFORMED", "NOTDONE",
    "NOTMEASURED", "NOTAPPLICABLE", "NOTAVAILABLE")


def _reading(value: Any) -> bool:
    """Whether a cell holds a reading: a number, a quantity, a list of
    points, or words such as ``NP`` or ``<10`` -- not a dash or n/a."""
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return bool(value)
    if isinstance(value, str):
        folded = fold_id(value)
        return bool(folded) and folded not in _EMPTY_CELL_WORDS
    return True


#: A bracket in a heading: ``(k, m/s)``, ``[CBR %]``, ``(kΩ - cm)``.
_BRACKET = re.compile(r"[(\[]([^()\[\]]*)[)\]]")
#: Unit words a bracket prints beside a test's symbol, folded to lower
#: case: lengths, times, forces, pressures, masses and the like. A letter
#: alone that is a symbol (``k``, ``Gs``, ``Cc``) is no unit.
_UNIT_WORDS = frozenset(
    "m mm cm km s sec secs min mins h hr hrs d day days yr ft in yd n kn "
    "pa kpa mpa psi psf ksf tsf pcf g kg mg lb lbs kip kips t ohm ohms ppm "
    "ppb ml l percent pct meq".split())
#: A concentration (``mg/kg``, ``mg/L``, ``ppm``): the bracket names an
#: analyte (``Potassium (K, mg/L)``), not a test's symbol.
_CONCENTRATION = re.compile(
    r"\b(?:mg|µg|ug|g)\s*/\s*(?:kg|l|dm3)\b|\bpp[mb]\b|\bmeq\b", re.I)
#: An electrical resistance with its prefix (``kΩ``, ``k ohm``, ``k-ohm``,
#: ``MΩ``): a ``k`` there is kilo, not a permeability.
_PREFIXED_OHM = re.compile(
    r"(?<![A-Za-z])[kKmMµu]?[\s\-–.·]*(?:Ω|ohms?\b)", re.I)
#: ``element -> its chemical symbol``, for the elements and the ions named
#: after them that a soil or water chemistry sheet reports. A heading
#: naming one prints its symbol as the element's, not a test's
#: ("Potassium (K)" is no permeability, "Chromium (Cr)" no recompression
#: index).
_ELEMENT_SYMBOLS: Dict[str, str] = {
    "aluminium": "Al", "aluminum": "Al", "antimony": "Sb", "arsenic": "As",
    "barium": "Ba", "beryllium": "Be", "boron": "B", "cadmium": "Cd",
    "calcium": "Ca", "carbon": "C", "chlorine": "Cl", "chloride": "Cl",
    "chromium": "Cr", "cobalt": "Co", "copper": "Cu", "iron": "Fe",
    "lead": "Pb", "lithium": "Li", "magnesium": "Mg", "manganese": "Mn",
    "mercury": "Hg", "molybdenum": "Mo", "nickel": "Ni", "nitrogen": "N",
    "phosphorus": "P", "potassium": "K", "selenium": "Se",
    "silicon": "Si", "silver": "Ag", "sodium": "Na", "strontium": "Sr",
    "sulfur": "S", "sulphur": "S", "thallium": "Tl", "tin": "Sn",
    "titanium": "Ti", "uranium": "U", "vanadium": "V", "zinc": "Zn",
}


def _bracket_names(match: Any) -> str:
    """What a heading's bracket names (RI-143): its letter-only tokens that
    are no unit -- ``(k, m/s)`` is ``k``, ``(CBR, %)`` is ``CBR``,
    ``(Gs, Mg/m3)`` is ``Gs`` -- and nothing from a bracket of units
    alone (``(kN/m2)``, ``(%)``, ``(kΩ - cm)``) or of a concentration."""
    inner = match.group(1)
    if _CONCENTRATION.search(inner):
        return " "
    inner = _PREFIXED_OHM.sub(" ", inner)
    kept = [token.rstrip(".") for token in re.split(r"[\s,;:/·*^=+\-–]+",
                                                    inner)
            if re.fullmatch(r"[A-Za-z]+\.?", token)
            and token.rstrip(".").lower() not in _UNIT_WORDS]
    return " " + " ".join(kept) + " "


def _heading_kinds(heading: Any) -> Set[str]:
    """The kinds of sheet a column heading names (:data:`_HEADING_KINDS`),
    its brackets read for the symbols they print and their units left out
    (:func:`_bracket_names`), and the symbol of an element it names left
    out with the element (:data:`_ELEMENT_SYMBOLS`)."""
    text = str(heading or "")
    for element, symbol in _ELEMENT_SYMBOLS.items():
        if re.search(rf"(?<![A-Za-z]){element}(?![A-Za-z])", text, re.I):
            text = re.sub(rf"(?<![A-Za-z]){symbol}(?![A-Za-z])", " ", text,
                          flags=re.I)
    text = _BRACKET.sub(_bracket_names, text)
    text = re.sub(r"['’′]", "", text)
    text = re.sub(r"(?<![A-Za-z])(?:[A-Za-z]\.){2,}",
                  lambda m: m.group(0).replace(".", ""), text)
    words = [fold_id(w) for w in re.split(r"[^0-9A-Za-z]+", text) if w]
    out: Set[str] = set()
    for key, kinds in _HEADING_KINDS.items():
        if key in words or (len(key) > 3 and key in "".join(words)):
            out.update(kinds)
    return out


def _tests_named(row: Any) -> List[Tuple[str, Tuple[str, ...]]]:
    """The tests a summary row reports, one per column holding a reading:
    ``(column, the kinds of sheet that run its test)`` for every column of
    :data:`COLUMN_KINDS`, and ``(heading, kinds)`` for every heading in
    ``other`` whose cell holds one and names a test
    (:data:`_HEADING_KINDS`). A column naming several kinds is one test any
    of them runs (a qu off an unconfined test or a UU triaxial). The
    checklist asks each of these for a sheet (RI-142)."""
    tests: List[Tuple[str, Tuple[str, ...]]] = []
    for column, of in COLUMN_KINDS.items():
        if _reading(getattr(row, column, None)):
            tests.append((column, of))
    for heading, cell in getattr(row, "other", None) or ():
        if _reading(cell):
            kinds = _heading_kinds(heading)
            if kinds:
                tests.append((str(heading), tuple(sorted(kinds))))
    return tests


def _kinds_named(row: Any) -> Set[str]:
    """The kinds of sheet a summary row's columns name: those of every
    column holding a reading (:data:`COLUMN_KINDS`), and of every heading
    in ``other`` whose cell holds one (:func:`_tests_named`)."""
    return {kind for _column, of in _tests_named(row) for kind in of}


def _shares_a_value(row: Any, sheet: LabTest) -> bool:
    """Whether the row and the sheet both carry a value for one column
    (:data:`SHARED_VALUES`)."""
    result = sheet.result
    if result is None:
        return False
    for column, names in SHARED_VALUES.items():
        if _reading(getattr(row, column, None)) and any(
                _reading(getattr(result, name, None)) for name in names):
            return True
    return False


def _shares_a_test(row: Any, sheet: LabTest) -> bool:
    """A test this row reports and this sheet runs: the sheet's kind is one
    the row's columns name (:func:`_kinds_named`), or both carry a value
    for one column."""
    return sheet.kind in _kinds_named(row) or _shares_a_value(row, sheet)


def _sample_keys(test: Any) -> Set[str]:
    """The names a row or a sheet gives its sample, folded for matching:
    the one the reconciler linked it to and the one it printed (a sheet
    linked to the log's ``5`` still printed ``S-5``), each with a hole
    printed at the head of the label dropped (``"SB-1, S-2"`` in SB-1 is
    S-2) and leading zeros folded (S-02 is S-2). Empty when it names none.
    Two samples agree when they share a name."""
    hole = hole_key(lab_hole(test))
    out: Set[str] = set()
    for text in (getattr(test, "linked_sample_id", ""),
                 getattr(test, "sample_id", "")):
        text = str(text or "")
        if hole:
            head, rest = _leading_hole(text, {hole})
            if head and hole_key(rest):
                text = rest
        if hole_key(text):
            out.add(hole_key(text))
    return out


def _sample_number(test: Any) -> Optional[int]:
    """The sample a row or a sheet prints, where it prints a bare number
    ("8812", "07"; its hole at the head dropped, "QX-2 8812"); else None. A
    name with a letter or a separator in it ("S-5", "0-1.7") is not one."""
    text = str(getattr(test, "sample_id", "") or "")
    hole = hole_key(lab_hole(test))
    if hole:
        head, rest = _leading_hole(text, {hole})
        if head and hole_key(rest):
            text = rest
    text = text.strip(" ,;:")
    return int(text) if re.fullmatch(r"\d{1,9}", text) else None


def _one_numbering(rows: Sequence[Any], sheets: Sequence[Any]) -> bool:
    """Whether a table's rows and the sheets number their samples alike:
    a row and a sheet of one hole, at one depth, print the same bare number
    as their sample (:func:`_sample_number`); no row prints a number a
    sheet of another hole, or of its own hole at another depth, prints --
    the same number on another specimen is the other numbering's; and the
    sheets' numbers run on through the report -- no number is printed by
    two holes' sheets. A table numbering its specimens 1 to 6 down two
    holes, beside sheets numbering each hole's 1 to 3, is two numberings,
    though the first hole's agree: the sheets start again in each hole. So
    is one whose rows and sheets agree once and print each other's numbers
    on other specimens elsewhere: the one pair agrees by chance."""
    holes_of: Dict[int, Set[str]] = {}
    for sheet in sheets:
        number, hole = _sample_number(sheet), hole_key(lab_hole(sheet))
        if number is not None and hole:
            holes_of.setdefault(number, set()).add(hole)
    if any(len(holes) > 1 for holes in holes_of.values()):
        return False
    agree = False
    for row in rows:
        number, depth = _sample_number(row), depth_m(row.depth_top)
        hole = hole_key(lab_hole(row))
        if number is None or depth is None or not hole:
            continue
        for sheet in sheets:
            got = depth_m(getattr(sheet, "depth_top", None))
            other = hole_key(lab_hole(sheet))
            if got is None or not other or _sample_number(sheet) != number:
                continue
            if other != hole or abs(got - depth) > DEPTH_TOL_M:
                return False             # the number on another specimen
            agree = True
    return agree


def _numbers_apart(a: Any, b: Any) -> bool:
    """Whether a row and a sheet print two different bare numbers as their
    samples (BACKLOG RI-143). Where the table and the sheets number their
    samples alike (:func:`_one_numbering`), two numbers ("8812" and
    "8805") are two specimens, though both lie at one depth of one hole and
    so were linked to one sample of its log -- a bulk bag and a tube taken
    beside it. A name with a letter in it ("S-5" beside "5") is another
    spelling, which the links settle."""
    na, nb = _sample_number(a), _sample_number(b)
    return na is not None and nb is not None and na != nb


def _names_another(a: Any, b: Any) -> bool:
    """Whether two rows or two sheets name different specimens: each prints
    a hole, a sample, a top or a bottom, and they differ."""
    ha, hb = hole_key(lab_hole(a)), hole_key(lab_hole(b))
    if ha and hb and ha != hb:
        return True
    sa, sb = _sample_keys(a), _sample_keys(b)
    if sa and sb and not sa & sb:
        return True
    for field in ("depth_top", "depth_bottom"):
        da = depth_m(getattr(a, field, None))
        db = depth_m(getattr(b, field, None))
        if da is not None and db is not None and abs(da - db) > EXACT_TOL_M:
            return True
    return False


def _closeness(a: Any, b: Any) -> Tuple[float, int]:
    """How far a row's interval is from a sheet's: the gap between their
    tops in metres (0 within :data:`EXACT_TOL_M`), then 0 where their
    bottoms agree or neither prints one, 1 otherwise. ``(0, 0)`` is an
    exact interval match; where either prints no top nothing is known,
    ``(inf, 0)`` for every rival alike."""
    top_a, top_b = depth_m(a.depth_top), depth_m(b.depth_top)
    if top_a is None or top_b is None:
        return math.inf, 0
    gap = abs(top_a - top_b)
    if gap <= EXACT_TOL_M:
        gap = 0.0
    bot_a, bot_b = depth_m(a.depth_bottom), depth_m(b.depth_bottom)
    if bot_a is None and bot_b is None:
        same_bottom = True
    else:
        same_bottom = (bot_a is not None and bot_b is not None
                       and abs(bot_a - bot_b) <= EXACT_TOL_M)
    return gap, 0 if same_bottom else 1


def _as_close(key: Tuple[float, int], best: Tuple[float, int]) -> bool:
    if key[1] != best[1]:
        return False
    if math.isinf(best[0]):
        return math.isinf(key[0])
    return key[0] - best[0] <= EXACT_TOL_M


def _closest(to: Any, rivals: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    """``(won, tied)``: the rivals closest to ``to`` (:func:`_closeness`),
    with every other rival that names no other specimen than they do (a
    sample's sieve sheet printing 12.0-13.5 ft and its hydrometer sheet
    printing 12.0 ft only); or, where the closest name different specimens
    and a sample both name does not pick one out, none -- and those rivals
    as the tie."""
    if not rivals:
        return [], []
    keyed = [(_closeness(to, r), r) for r in rivals]
    best = min(key for key, _r in keyed)
    near = [r for key, r in keyed if _as_close(key, best)]
    if _apart(near):
        mine = _sample_keys(to)
        near = [r for r in near if mine & _sample_keys(r)]
        if not near or _apart(near):
            return [], [r for key, r in keyed if _as_close(key, best)]
    return [r for r in rivals
            if not any(_names_another(r, w) for w in near)], []


def _apart(group: Sequence[Any]) -> bool:
    return any(_names_another(a, b)
               for i, a in enumerate(group) for b in group[i + 1:])


def _may_join(row: Any, sheet: LabTest) -> bool:
    """Whether ``sheet`` may report on ``row``'s specimen: in the same hole
    (:func:`~report_ingest.model.lab_hole` on both: the link this pass
    made, else the boring as printed, RI-115), its top within
    :data:`DEPTH_TOL_M` of the row's (or, for a row with no depth, the same
    sample where the row prints one), and one of:

    - the sheet's kind is one the row's columns name (:func:`_kinds_named`);
    - a sample both name alike;
    - a value both carry (:data:`SHARED_VALUES`), unless both print samples
      and they differ: then the value is another sample's, printed beside
      that sample's own test (a chemistry sheet's water content);
    - for a row whose columns name no test at all, the sheet's interval is
      exactly the row's.
    """
    hole = hole_key(lab_hole(row))
    if hole and hole_key(lab_hole(sheet)) != hole:
        return False
    # Two holes printed under one name are two holes (RI-135): a sheet
    # linked to one of them reports on no row linked to the other, nor on a
    # row linked to neither (and the other way round).
    ours, yours = (getattr(row, "linked_qualifier", ""),
                   getattr(sheet, "linked_qualifier", ""))
    if (ours or yours) and ours != yours:
        return False
    sample, theirs = _sample_keys(row), _sample_keys(sheet)
    depth = depth_m(row.depth_top)
    if depth is not None:
        got = depth_m(sheet.depth_top)
        if got is None or abs(got - depth) > DEPTH_TOL_M:
            return False
    elif sample and not sample & theirs:
        return False
    kinds = _kinds_named(row)
    if sheet.kind in kinds or sample & theirs:
        return True
    if _shares_a_value(row, sheet):
        return not (sample and theirs)
    return not kinds and _closeness(row, sheet) == (0.0, 0)


def _its_sample_first(row: Any, mine: List[LabTest]) -> List[LabTest]:
    """``mine`` less the sheets naming another sample, where some name the
    row's own: a row printing ``S-3`` whose grading sheet says S-3 is S-3,
    and an Atterberg sheet of S-4 at that depth is another specimen. Where
    none names the row's sample (the sheets number samples their own way),
    all stay."""
    keys = _sample_keys(row)
    if not keys or not any(keys & _sample_keys(s) for s in mine):
        return mine
    return [s for s in mine
            if not _sample_keys(s) or keys & _sample_keys(s)]


def _names_no_specimen(row: Any) -> bool:
    """Whether a summary row prints no hole, no sample and no depth: a
    line of the table as read (a heading, a note, a row's second line),
    with nothing to find its sheet by. It is matched to none."""
    return (not hole_key(lab_hole(row)) and depth_m(row.depth_top) is None
            and not _sample_keys(row))


def _placed(row: Any, mine: List[LabTest]
            ) -> Tuple[List[LabTest], List[LabTest]]:
    """``(mine, [])``, or ``([], mine)`` where the row names too little of
    its specimen to choose among them. A row printing no hole is looked
    for in every hole, so where its sheets lie in more than one, only those
    naming its own sample are its own (none: it cannot choose). A row
    printing neither a depth nor a sample is placed by its hole alone, so
    its sheets must name one specimen."""
    keys = _sample_keys(row)
    if not hole_key(lab_hole(row)):
        holes = {hole_key(lab_hole(s)) for s in mine} - {""}
        if len(holes) > 1:
            agree = [s for s in mine if keys & _sample_keys(s)]
            if not agree:
                return [], mine
            mine = agree
    if depth_m(row.depth_top) is None and not keys and _apart(mine):
        return [], mine
    return mine, []


def _name_after_word(printed: Any) -> str:
    """The hole's own name a printed boring gives after a word for an
    exploration (RI-143): ``"BORE HOLE NO. (KW-12)"`` and ``"BH. KW - 12"``
    print KW-12. ``""`` where no such word heads it, or what follows is not
    one name with letters and a number (``"BH 83"``: a number alone may be
    any kind of hole's; ``"BH47, U9"``: a hole and a sample)."""
    text = str(printed or "")
    head = _EXPLORATION_HEAD.match(text)
    if not head or not head.end():
        return ""
    rest = text[head.end():].strip().rstrip(")]").strip()
    return rest if re.fullmatch(r"[A-Za-z]{1,4}\s*[-_.]?\s*\d{1,4}[A-Za-z]?",
                                rest) else ""


def _match_view(test: Any, unit: str) -> Any:
    """A row or a sheet as :func:`_summary_matches` matches it: its depth
    as :func:`_depth_view` reads it, and, where no hole of the logs is
    linked to it, the boring it prints by the hole's own name
    (:func:`_name_after_word`), so a summary row's ``KW - 12`` finds the
    sheets of ``BORE HOLE NO. (KW-12)`` though no log of KW-12 was read."""
    view, _how = _depth_view(test, unit)
    if getattr(view, "linked_investigation_id", ""):
        return view
    name = _name_after_word(getattr(view, "investigation_id", ""))
    return view.model_copy(update={"investigation_id": name}) if name \
        else view


def _summary_matches(rows: Sequence[Any], sheets: Sequence[LabTest],
                     unit: str = ""
                     ) -> Tuple[List[List[LabTest]],
                                List[Tuple[List[Any], List[LabTest]]]]:
    """Which sheets each row of ONE summary table reports on (BACKLOG
    RI-140), and the ties: ``(joined, ties)``, ``joined[i]`` the sheets of
    ``rows[i]``, each tie ``(rows, sheets)`` that could not be told apart.

    Each row and sheet is matched by its depth as :func:`_depth_view`
    reads it (RI-143): one printed with no unit in ``unit``, the record's
    one depth unit (:func:`depth_unit`; none given, it stays unread), and
    one printed only in the sample label read from there, for a row or
    sheet placed in a hole; and a hole no log holds by its own name after
    a word for an exploration (:func:`_match_view`). What comes back are
    the rows and sheets themselves.
    """
    views = [_match_view(x, unit) for x in list(rows) + list(sheets)]
    back = {id(v): x for v, x in zip(views, list(rows) + list(sheets))}
    joined, ties = _matches(views[:len(rows)], views[len(rows):])
    return ([[back[id(s)] for s in got] for got in joined],
            [([back[id(r)] for r in t_rows], [back[id(s)] for s in t_sheets])
             for t_rows, t_sheets in ties])


def _matches(rows: Sequence[Any], sheets: Sequence[LabTest]
             ) -> Tuple[List[List[LabTest]],
                        List[Tuple[List[Any], List[LabTest]]]]:
    """:func:`_summary_matches` over rows and sheets as their depths are
    read.

    A row printing no hole, sample or depth joins nothing
    (:func:`_names_no_specimen`). A row may join a sheet that
    :func:`_may_join` it; where some of those
    name the row's own sample, those naming another sample drop out
    (:func:`_its_sample_first`); where the table and the sheets number
    their samples alike (:func:`_one_numbering`), a sheet printing another
    number than a placed row's never joins it (:func:`_numbers_apart`).
    Of the sheets of one
    kind left, the row
    takes the closest (:func:`_closeness`: an exact interval before a
    window match, then the nearer top); a row is one specimen and reports
    each test once, so a second specimen's sheet of the same kind a few
    centimetres off is not its own. Sheets of other kinds -- its Atterberg
    and its grading sheet -- each join it on their own. A row that names
    too little of its specimen to choose (:func:`_placed`: no hole and
    sheets in several holes none of which names its sample; neither a
    depth nor a sample and sheets naming several specimens) joins none of
    them, a tie. A sheet taken by several rows of the table stays with the
    rows that share a test with it, if any do, and of those with the
    closest: the table prints each specimen once, and a row that took it
    only by its sample (its limits row, beside the row tabulating the
    grading) is not its row. Where the closest rivals name different
    specimens (a hole, a sample or a depth that differs) and no sample
    both name picks one out, nothing is joined: the tie is returned to be
    said, not guessed. Rivals that name no other specimen than the closest
    -- one test run three times on a sheet, a test read as one per page, a
    hydrometer sheet printing only the top of its sieve sheet's interval --
    join with it.
    """
    picked: List[List[LabTest]] = []
    ties: List[Tuple[List[Any], List[LabTest]]] = []
    numbered = _one_numbering(rows, sheets)
    for row in rows:
        if _names_no_specimen(row):
            picked.append([])
            continue
        mine, unplaced = _placed(row, _its_sample_first(
            row, [s for s in sheets if _may_join(row, s)]))
        if unplaced:
            ties.append(([row], unplaced))
            picked.append([])
            continue
        if numbered:
            # Only once the row is placed: a row printing no hole is not
            # placed by the sheets its number rules out (RI-143).
            mine = [s for s in mine if not _numbers_apart(row, s)]
        if depth_m(row.depth_top) is None and not _sample_keys(row):
            picked.append(mine)                 # one specimen (_placed)
            continue
        chosen: List[LabTest] = []
        for kind in dict.fromkeys(s.kind for s in mine):
            won, tied = _closest(row, [s for s in mine if s.kind == kind])
            chosen.extend(won)
            if tied:
                ties.append(([row], list(tied)))
        picked.append(chosen)
    joined: List[List[LabTest]] = [[] for _ in rows]
    for sheet in sheets:
        takers = [i for i, chosen in enumerate(picked)
                  if any(s is sheet for s in chosen)]
        if not takers:
            continue
        testing = [i for i in takers if _shares_a_test(rows[i], sheet)]
        takers = testing or takers
        won, tied = _closest(sheet, [rows[i] for i in takers])
        if tied:
            ties.append((list(tied), [sheet]))
            continue
        for i in takers:
            if any(r is rows[i] for r in won):
                joined[i].append(sheet)
    return joined, ties


def _sheets_for(row: Any, sheets: Sequence[LabTest],
                rows: Sequence[Any] = (), unit: str = "") -> List[LabTest]:
    """The sheets that report on the same specimen as this row: the row's
    share of :func:`_summary_matches` over ``rows``, the other rows of its
    table (none given: the row alone)."""
    table = list(rows)
    if not any(r is row for r in table):
        table.append(row)
    joined, _ties = _summary_matches(table, sheets, unit=unit)
    return next(got for r, got in zip(table, joined) if r is row)


def _specimen(test: Any, at: str = " at ") -> str:
    """A row or a sheet as a reviewer finds it: hole, sample, depth (after
    ``at``; a QA ``where`` has its own "at", so it passes a space)."""
    depth = f"{at}{_show(test.depth_top)}" if test.depth_top else ""
    return (f"{lab_hole(test) or '?'} {test.sample_id or ''}".strip()
            + depth)


def _note_ties(table: LabTest, ties: Sequence[Tuple[List[Any],
                                                    List[LabTest]]],
               qa: List[QAEntry]) -> None:
    """One note per row that could not tell its sheets apart, and one per
    sheet that could not tell its rows apart, ranked open."""
    by_row: Dict[int, Tuple[Any, List[LabTest]]] = {}
    for rows, sheets in ties:
        if len(rows) == 1:
            by_row.setdefault(id(rows[0]), (rows[0], []))[1].extend(sheets)
            continue
        (sheet,) = sheets
        qa.append(QAEntry(
            kind="note", rank="open",
            where=f"lab_tests.summary_table at {_specimen(sheet, ' ')}",
            detail=f"the {sheet.kind} sheet is as close to {len(rows)} rows "
                   f"of the summary table that name different specimens; "
                   f"it is cross-checked against none of them",
            values=[f"summary row {_specimen(r)}" for r in rows[:8]],
            pages=sorted(set(list(table.pages) + list(sheet.pages)))))
    for row, sheets in by_row.values():
        kinds = ", ".join(f"{n} {kind}" for kind, n in
                          _count_kinds(sheets).items())
        qa.append(QAEntry(
            kind="note", rank="open",
            where=f"lab_tests.summary_table at {_specimen(row, ' ')}",
            detail=f"the summary row is as close to sheets that name "
                   f"different specimens ({kinds}); it is cross-checked "
                   f"against none of them",
            values=[f"{s.kind} sheet {_specimen(s)}" for s in sheets[:8]],
            pages=sorted(set(list(table.pages)
                             + [p for s in sheets for p in s.pages]))))


def _count_kinds(sheets: Sequence[LabTest]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for sheet in sheets:
        out[sheet.kind] = out.get(sheet.kind, 0) + 1
    return out


# -- a log's own drives and depths ------------------------------------------
#
# Field session 2026-10-14 (BACKLOG RI-28): a refusal printed `60/15"` beside
# its interval 3.5-3.65 m was retyped `60/15in` and written as a 0.381 m drive
# inside a 0.15 m sample; a drive record lost a drive (3 / 3 / 4 given as
# [3, 4]); a boring's layers ran to 21.1 m under a 20.1 m total depth. Each
# was in the file with no cross-check entry. These look at one log against
# itself, read the drives the way the DIGGS writer will write them, and say
# what disagrees; they change nothing.

#: Drives add up to their sample's interval within this many metres, or this
#: fraction of the interval, whichever is larger: an 18 in drive (0.457 m)
#: in a 1.05-1.50 m interval is 7 mm out and agrees, and so do three 15 cm
#: drives under a remark that rounds their interval to "0 to 0.5 m", as
#: many logs' remarks do. A dropped drive, or a
#: refusal's distance read in the wrong unit, is a third of it or more.
DRIVE_TOL_M = 0.025
DRIVE_TOL_FRAC = 0.10
#: A refusal's drives may fall short of its interval; they may run past it
#: by no more than this (a `60/15"` read as inches in a 0.15 m sample runs
#: 0.23 m past it).
REFUSAL_TOL_M = 0.05
REFUSAL_TOL_FRAC = 0.25

#: A record needs this many complete drive records beside it, and this share
#: of them agreeing, before their count is taken as the form's.
DRIVE_COUNT_MIN_RECORDS = 3
DRIVE_COUNT_SHARE = 0.6

#: A depth this far below the total depth, in metres (or this fraction of
#: it), is below it: a header rounds the total depth it prints.
BELOW_TD_TOL_M = 0.05
BELOW_TD_TOL_FRAC = 0.005


def _record_pages(inv: Investigation, *provs: Any) -> List[int]:
    pages = [p.page for p in provs if p is not None]
    return sorted(set(pages)) or list(inv.pages)


def _check_drives(record: ReportRecord, qa: List[QAEntry]) -> None:
    """Each driven record's drives against its interval and its log's form."""
    from report_ingest.diggs_writer import _drive_sets
    from report_ingest.log_floor import shows_halt
    for inv in record.investigations:
        name = inv.investigation_id or "an exploration"
        samples = {fold_id(s.sample_id): s for s in inv.samples
                   if fold_id(s.sample_id)}
        complete: List[Tuple[int, Any]] = []        # (drive count, record)
        off: List[Dict[str, Any]] = []              # drives vs interval
        for spt in inv.spt:
            drives = _drive_sets(spt)
            top = depth_m(spt.depth_top)
            bottom = depth_m(spt.depth_bottom)
            interval_from = "its own depths"
            sample = samples.get(fold_id(spt.sample_id))
            if bottom is None and sample is not None \
                    and depth_m(sample.bottom) is not None:
                top, bottom = depth_m(sample.top), depth_m(sample.bottom)
                interval_from = f"sample {sample.sample_id}"
            at = (f"{name} driven record at {top:.2f} m" if top is not None
                  else f"{name} driven record")
            if spt.sample_id:
                at += f" ({spt.sample_id})"
            pages = _record_pages(inv, spt.prov)
            # A record that shows plainly that it ends in a halt is checked
            # as a refusal, whether its drives show the halt or its own drive
            # lengths do (16-29-50 over 6, 6 and 2.5 in; RI-50): it stops
            # short of its interval. A lone whole number over 50 is not
            # (shows_halt): it may be the N filed as a drive, and the drive
            # sum is what catches that.
            refusal = (spt.refusal or shows_halt(spt)
                       or any(isinstance(b, str) and "/" in b
                              for b in spt.blows))
            whole = len(drives) == len(spt.blows) and bool(drives)
            if whole and not refusal:
                complete.append((len(drives), spt))
            if top is None or bottom is None or not whole:
                continue
            interval = bottom - top
            if interval <= 0:
                qa.append(QAEntry(
                    kind="out_of_range", where=f"{name}.spt",
                    detail=f"{at}: its bottom ({bottom:.2f} m) is not below "
                           f"its top",
                    values=[f"{top:.2f} m", f"{bottom:.2f} m"], pages=pages))
                continue
            total = sum(pen for _count, pen in drives)
            if refusal:
                # A refusal stops short of its interval by definition, and
                # its box is often drawn to the recovery or a round depth:
                # only drives LONGER than the interval are wrong.
                if total - interval <= max(REFUSAL_TOL_M,
                                           REFUSAL_TOL_FRAC * interval):
                    continue
            elif abs(total - interval) <= max(DRIVE_TOL_M,
                                              DRIVE_TOL_FRAC * interval):
                continue
            each = ", ".join(f"{str(b)!r} = {pen:.3f} m"
                             for b, (_c, pen) in zip(spt.blows, drives))
            assumed = ("" if spt.increment is not None or spt.increments
                       or not any(isinstance(b, int) or str(b).isdigit()
                                  for b in spt.blows)
                       else "; no increment was given, so each plain count "
                            "was taken as 0.1524 m (6 in)")
            off.append({"at": at, "top": top, "bottom": bottom,
                        "total": total, "interval": interval, "each": each,
                        "assumed": assumed, "from": interval_from,
                        "pages": pages})
        _report_drive_sums(name, off, qa)
        _check_drive_counts(inv, name, complete, qa)


def _report_drive_sums(name: str, off: Sequence[Dict[str, Any]],
                       qa: List[QAEntry]) -> None:
    """One entry per disagreement; records of one hole that disagree the
    same way (every drive taken at the same assumed length against the same
    interval) are one entry naming each depth, so one wrong assumption does
    not bury the rest of the list."""
    groups: Dict[Tuple[float, float, str], List[Dict[str, Any]]] = {}
    for f in off:
        key = (round(f["total"], 3), round(f["interval"], 3), f["assumed"])
        groups.setdefault(key, []).append(f)
    for (total, interval, assumed), fs in groups.items():
        pages = sorted({p for f in fs for p in f["pages"]})
        fix = (" - a drive length is wrong (the increment the form counts "
               "over, or the distance printed in a refusal) or the interval "
               "is")
        if len(fs) == 1:
            f = fs[0]
            qa.append(QAEntry(
                kind="conflict", where=f"{name}.spt",
                detail=f"{f['at']}: its drives add up to {total:.3f} m "
                       f"({f['each']}{assumed}) but the interval "
                       f"({f['from']}) is {interval:.3f} m" + fix,
                values=[f"drives {total:.3f} m",
                        f"interval {f['top']:.2f}-{f['bottom']:.2f} m"],
                pages=pages))
            continue
        tops = ", ".join(f"{f['top']:.2f}" for f in fs[:12]) + (
            ", ..." if len(fs) > 12 else "")
        qa.append(QAEntry(
            kind="conflict", where=f"{name}.spt",
            detail=f"{name}: {len(fs)} driven records (at {tops} m) have "
                   f"drives adding up to {total:.3f} m{assumed} against "
                   f"intervals of {interval:.3f} m" + fix,
            values=[f"drives {total:.3f} m", f"interval {interval:.3f} m",
                    f"{len(fs)} records"],
            pages=pages))


def _check_drive_counts(inv: Investigation, name: str,
                        complete: Sequence[Tuple[int, Any]],
                        qa: List[QAEntry]) -> None:
    """A complete drive record with a different number of drives from the
    log's other complete records: one was dropped or added."""
    if len(complete) < DRIVE_COUNT_MIN_RECORDS:
        return
    tally: Dict[int, int] = {}
    for n, _spt in complete:
        tally[n] = tally.get(n, 0) + 1
    form, votes = max(tally.items(), key=lambda kv: (kv[1], kv[0]))
    if votes < DRIVE_COUNT_SHARE * len(complete):
        return
    for n, spt in complete:
        if n == form:
            continue
        top = depth_m(spt.depth_top)
        at = (f"{name} driven record at {top:.2f} m" if top is not None
              else f"{name} driven record")
        if spt.sample_id:
            at += f" ({spt.sample_id})"
        qa.append(QAEntry(
            kind="count_mismatch", where=f"{name}.spt",
            detail=f"{at}: {n} drive(s) given ({list(spt.blows)}) where "
                   f"{votes} of the log's {len(complete)} complete records "
                   f"give {form} - a drive may have been dropped or added; "
                   f"check the form",
            values=[f"{n} drives", f"form {form}"],
            pages=_record_pages(inv, spt.prov)))


def _check_below_total_depth(record: ReportRecord,
                             qa: List[QAEntry]) -> None:
    """Any layer, sample, driven record or water level below the hole's own
    total depth, said once per hole with every value."""
    for inv in record.investigations:
        td = depth_m(inv.total_depth)
        if td is None:
            continue
        limit = td + max(BELOW_TD_TOL_M, BELOW_TD_TOL_FRAC * td)
        name = inv.investigation_id or "an exploration"
        found: List[str] = []
        provs: List[Any] = []

        def look(what: str, q: Optional[Quantity], prov: Any) -> None:
            d = depth_m(q)
            if d is not None and d > limit:
                found.append(f"{what} {d:.2f} m")
                provs.append(prov)

        for i, ly in enumerate(inv.layers):
            look(f"layer {i + 1} top", ly.top, ly.prov)
            look(f"layer {i + 1} bottom", ly.bottom, ly.prov)
        for s in inv.samples:
            label = f"sample {s.sample_id}" if s.sample_id else "a sample"
            look(f"{label} top", s.top, s.prov)
            look(f"{label} bottom", s.bottom, s.prov)
        for spt in inv.spt:
            look("SPT top", spt.depth_top, spt.prov)
            look("SPT bottom", spt.depth_bottom, spt.prov)
        for w in inv.water:
            look(f"water ({w.when})", w.depth, w.prov)
        if not found:
            continue
        qa.append(QAEntry(
            kind="out_of_range", where=f"{name}.total_depth",
            detail=f"{name}: {len(found)} value(s) below its total depth of "
                   f"{td:.2f} m: {', '.join(found[:8])}"
                   + (", ..." if len(found) > 8 else "")
                   + " - the total depth or these depths were given wrong",
            values=[f"total depth {td:.2f} m"] + found[:8],
            pages=_record_pages(inv, *provs)))


# -- units, pages and what the readers could not settle ---------------------

def _check_units(record: ReportRecord, qa: List[QAEntry]) -> None:
    """Every unit with no conversion, named once with its fields.

    Two different things, named differently. A unit that WAS printed and has
    no factor in the table is one entry per unit. A value read with NO unit
    printed beside it is one entry for all of them: its unit is unknown, so
    it is not converted -- a depth of "4" is not 4 m -- unless its field is
    one whose kind has no unit (a pH; :data:`UNITLESS_FIELDS`). Calculations
    are not walked: the calculation reader names its own units, and an empty
    one there is a count or a factor of safety by that reader's schema.
    """
    seen: Dict[str, List[str]] = {}
    blank: List[str] = []
    for where, holder in ([("investigations", inv)
                           for inv in record.investigations]
                          + [("lab_tests", test) for test in record.lab_tests]
                          + [("general", record.general)]):
        for path, value in _quantities(holder, where):
            names = re.findall(r"[A-Za-z_]\w*", path)
            kind = UNITLESS_FIELDS.get(names[-1]) if names else None
            if value.to_si(kind=kind) is not None:
                continue
            if unit_printed(value.unit):
                seen.setdefault(value.unit, []).append(path)
            else:
                blank.append(path)
    for unit, paths in sorted(seen.items()):
        qa.append(QAEntry(
            kind="unconverted", where="units",
            detail=f"{len(paths)} value(s) are printed in {unit!r}, which has "
                   f"no conversion in the record's unit table, so they are "
                   f"stored as printed and are absent from the SI view",
            values=paths[:10]))
    if blank:
        qa.append(QAEntry(
            kind="unconverted", where="units",
            detail=f"{len(blank)} value(s) were read with no unit printed "
                   f"beside them, so their unit is unknown: they are stored "
                   f"as printed, not converted, and absent from the SI view",
            values=blank[:10]))


def _quantities(value: Any, path: str = "") -> List[Tuple[str, Quantity]]:
    out: List[Tuple[str, Quantity]] = []
    if isinstance(value, Quantity):
        return [(path, value)]
    if value is None or isinstance(value, (str, bool, int, float)):
        return out
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            out.extend(_quantities(item, f"{path}[{index}]"))
        return out
    fields = getattr(type(value), "model_fields", None)
    if fields:
        for name in fields:
            out.extend(_quantities(getattr(value, name, None),
                                   f"{path}.{name}" if path else name))
    return out


def _check_pages(record: ReportRecord, qa: List[QAEntry],
                 labels: Optional[Dict[int, str]],
                 items: Optional[Sequence[Any]],
                 no_text_pages: Optional[Sequence[int]],
                 di_pages: Optional[Sequence[int]]) -> None:
    if labels and items is not None:
        covered = {int(page) for item in items for page in item.pages}
        missed: Dict[str, List[int]] = {}
        for page, role in sorted(labels.items()):
            if role in ITEM_ROLES and int(page) not in covered:
                missed.setdefault(role, []).append(int(page))
        for role, pages in sorted(missed.items()):
            qa.append(QAEntry(
                kind="skipped", where=f"pages.{role}",
                detail=f"{len(pages)} page(s) labelled {role} belong to no "
                       f"work item, so nothing read them",
                pages=pages))

    if no_text_pages:
        covered = set(int(p) for p in (di_pages or ()))
        blind = [int(p) for p in no_text_pages if int(p) not in covered]
        if blind:
            qa.append(QAEntry(
                kind="unreadable", where="pages.text",
                detail=f"{len(blind)} page(s) have no reliable text layer and "
                       f"no Azure Document Intelligence result; anything on "
                       f"them was read from the picture or not at all",
                pages=blind))


def _carry_unresolved(qa: List[QAEntry],
                      rows: Optional[Sequence[Dict[str, Any]]]) -> None:
    for row in rows or ():
        pages = []
        page = row.get("page")
        if row.get("pages"):
            # A group of pages no answer came back for (model-first ingest's
            # ``skipped`` rows): every page of it, not only the first.
            pages = sorted({int(p) for p in row["pages"]})[:200]
        elif page is not None:
            pages = [int(page)]
        what = str(row.get("what") or row.get("field") or "reader")
        why = str(row.get("why") or row.get("detail") or "")
        values = [str(row["value"])] if row.get("value") is not None else []
        if row.get("read"):
            # An object kept out of the record whole, as the model read it
            # (the builders' flag mode, report_ingest.checks.as_read).
            values.append(str(row["read"]))
        # What a reader could not settle is ``partial``; what PYTHON refused
        # -- a depth off the sheet's ruler, a percentage past 100 -- is
        # ``out_of_range``, so the two are countable apart: one is a gap in
        # the reading, the other a value that was read and thrown out.
        refused_by = str(row.get("refused_by") or "")
        kind = ("out_of_range" if refused_by == "python"
                else "skipped" if refused_by == "budget" else "partial")
        qa.append(QAEntry(kind=kind, where=what, detail=why,
                          values=values, pages=pages))


# -- the optional comment ---------------------------------------------------

def _comment_on_conflicts(qa: List[QAEntry], engine: Any,
                          limit: int = 20) -> None:
    conflicts = [entry for entry in qa if entry.kind == "conflict"][:limit]
    if not conflicts:
        return
    from report_ingest.engine import text_block, user

    lines = []
    for index, entry in enumerate(conflicts, start=1):
        lines.append(f"{index}. {entry.where}: {entry.detail} -- "
                     + "; ".join(entry.values)
                     + (f" (pages {entry.pages})" if entry.pages else ""))
    reply = engine.complete(
        [user(text_block("THE DISAGREEMENTS\n" + "\n".join(lines)))],
        system=_CONFLICT_SYSTEM, output_format=_ConflictComments)
    found = reply.parsed
    if found is None:
        return
    for comment in found.comments:
        index = int(comment.index) - 1
        if 0 <= index < len(conflicts) and comment.comment.strip():
            conflicts[index].detail += (
                " Reviewer note: " + " ".join(comment.comment.split()))

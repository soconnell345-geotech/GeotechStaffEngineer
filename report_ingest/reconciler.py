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
compared, and only the disagreements are recorded.

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

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from report_ingest.model import (
    UNITLESS_FIELDS, Investigation, LabTest, QAEntry, Quantity, ReportRecord,
    Sample, si_numbers, unit_printed,
)
from report_ingest.scoring import KEY_CONTENT

__all__ = [
    "reconcile", "si_view", "DEPTH_TOL_M", "CROSS_CHECKS", "ITEM_ROLES",
    "fold_id", "hole_key", "depth_m",
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


def _leading_hole(label: Any, keys: Any) -> Tuple[str, str]:
    """The longest of ``keys`` that BEGINS ``label``, and what follows it;
    ``("", "")`` when none does or the label names more than one.

    A sheet that prints no boring heading names the specimen by its hole --
    ``"BH-7 U3"``, ``"BH1, U2"``, a hole, a sample and a depth run together
    -- and a lab test carrying that label as its boring, or as its sample
    with no boring at all, names that hole first. Only holes the report's
    own logs hold are looked for, and only ones with a letter in them: a
    bare ``"2"`` is as likely a sample number as a hole. The hole has to
    end where a word of the label ends -- at a space, a comma, a slash or
    the end -- so ``"B-1"`` is not found at the head of ``"B-12"`` or of
    ``"B-1A"``; and the search stops at the first number that follows
    another across a separator, so ``"B-1 2.0 ft"`` and ``"B-1-2"`` are
    B-1 and never B-12, whose folded spelling they share.

    A label naming a second logged hole of the same series further on --
    ``"B-1/B-2"``, ``"B-1 to B-4 composite"`` -- names no one hole, and
    nothing is returned for it.
    """
    text = str(label or "")
    best, end = _head_hole(text, keys)
    if not best:
        return "", ""
    series = re.match(r"[A-Z]*", best).group(0)
    for start in range(end, len(text)):
        if text[start].isalnum() and not text[start - 1].isalnum():
            other, _end = _head_hole(text[start:], keys)
            if other and other != best and \
                    re.match(r"[A-Z]*", other).group(0) == series:
                return "", ""
    return best, text[end:]


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
              engine: Any = None) -> ReportRecord:
    """Join the record up and record everything that does not join.

    Pure Python. ``labels`` and ``items`` are the final page labels and the
    work items -- the graph has both and passes them; without them the page
    checks are skipped rather than guessed. ``reader_unresolved`` is every
    line the readers could not settle, each already carrying its own ``what``
    and ``why``. ``engine`` is optional and buys ONE call, which comments on
    the conflicts and changes nothing.

    The record is modified in place and returned.
    """
    qa: List[QAEntry] = []

    _fold_continuations(record, qa)
    _link_lab_tests(record, qa)
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


def _hole_for(test: LabTest, by_id: Dict[str, Investigation],
              samples: Optional[Dict[str, List[Tuple[Investigation,
                                                     Sample]]]] = None,
              by_fold: Optional[Dict[str, Investigation]] = None
              ) -> Tuple[Optional[Investigation], str]:
    """The investigation a lab test names, and where the name was found.

    The printed boring first, matched whole: exactly as the logs spell it
    (``fold_id``), then with leading zeros folded (``hole_key``, used only
    when one logged hole has that key -- "B-07" and "B-7" logged as two
    holes leave a sheet's "B-007" unlinked). Then a hole the report's logs
    hold at the HEAD of that printed text (``"BH-7 U3"``); then, only where
    the sheet printed no boring at all, a hole at the head of the sample
    label (``"BH-03 S-2 1.5-1.95 m"``), which is how many sheets name a
    specimen. The second string says which, for the QA note: empty for a
    whole match.

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
        return None, ""
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


def _link_lab_tests(record: ReportRecord, qa: List[QAEntry]) -> None:
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

    from_labels: List[Tuple[LabTest, str, Investigation]] = []
    for test in record.lab_tests:
        if test.kind == "summary_table":
            _link_summary_rows(test, by_fold, qa)
            continue
        inv, through = _hole_for(test, by_id, sample_names, by_fold)
        if inv is not None and through:
            from_labels.append((test, through, inv))
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
        test.linked_investigation_id = inv.investigation_id
        sample, delta = _sample_for(inv, test)
        if sample is None:
            wanted = depth_m(test.depth_top)
            qa.append(QAEntry(
                kind="partial", where=f"lab_tests.{test.kind}",
                detail=(f"linked to {inv.investigation_id} but no sample of "
                        f"it sits within {DEPTH_TOL_M} m of the sheet's "
                        f"depth" if wanted is not None else
                        f"linked to {inv.investigation_id}; the sheet prints "
                        f"no depth, so no sample could be matched"),
                values=[_show(test.depth_top)] if test.depth_top else [],
                pages=list(test.pages)))
            continue
        test.linked_sample_id = sample.sample_id
        test.linked_depth_delta_m = None if delta is None else round(delta, 3)

    if from_labels:
        # One note for the report, not one a test: a reviewer wants to know
        # that these links were read out of a longer label, and which.
        shown: List[str] = []
        for test, through, inv in from_labels:
            label = test.investigation_id if through == "the boring as " \
                "printed" else test.sample_id
            line = f"{label!r} -> {inv.investigation_id}"
            if line not in shown:
                shown.append(line)
        qa.append(QAEntry(
            kind="note", where="lab_tests",
            detail=(f"{len(from_labels)} lab test(s) were linked to the hole "
                    f"at the head of a longer label -- the boring as printed "
                    f"(a hole, sample and depth run together), or the sample "
                    f"label where the sheet printed no boring; only holes "
                    f"read from this report's logs were looked for"),
            values=shown[:12] + ([f"... {len(shown) - 12} more"]
                                 if len(shown) > 12 else []),
            pages=sorted({int(p) for test, _t, _i in from_labels
                          for p in test.pages})))


def _sample_for(inv: Investigation,
                test: LabTest) -> Tuple[Optional[Sample], Optional[float]]:
    """The sample this sheet's specimen came from, and how far off it was.

    The sample label matched whole first, as it always was. Then a word of
    a longer label that IS one of this hole's sample names ("BH-7 U3" ->
    U3), taken only when that sample sits at the sheet's depth: the word
    may be a sample number of another kind, and the depth is what says it
    is this one (RI-07). Then the sample at the sheet's depth.
    """
    if test.sample_id:
        wanted = fold_id(test.sample_id)
        for sample in inv.samples:
            if fold_id(sample.sample_id) == wanted:
                return sample, _delta(sample, test)
    words = {fold_id(w) for w in re.split(
        r"[^0-9A-Za-z-]+", f"{test.investigation_id} {test.sample_id}")
        if fold_id(w)}
    own = fold_id(inv.investigation_id)
    for sample in inv.samples:
        name = fold_id(sample.sample_id)
        if name and name in words and name != own and not name.isdigit():
            delta = _at_depth(sample, test)
            if delta is not None:
                return sample, delta
    return _sample_at_depth(inv, test)


def _at_depth(sample: Sample, test: LabTest) -> Optional[float]:
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


def _sample_at_depth(inv: Investigation, test: LabTest
                     ) -> Tuple[Optional[Sample], Optional[float]]:
    """The sample of ``inv`` at the sheet's depth, by depth alone."""
    best: Optional[Sample] = None
    best_delta: Optional[float] = None
    for sample in inv.samples:
        delta = _at_depth(sample, test)
        if delta is not None and (best_delta is None or delta < best_delta):
            best, best_delta = sample, delta
    return best, best_delta


def _delta(sample: Sample, test: LabTest) -> Optional[float]:
    top, depth = depth_m(sample.top), depth_m(test.depth_top)
    if top is None or depth is None:
        return None
    return abs(depth - top)


def _link_summary_rows(test: LabTest, by_id: Dict[str, Investigation],
                       qa: List[QAEntry]) -> None:
    """A summary table names many holes; report the ones that are not here.

    Matched exactly as printed (``fold_id``), not with the lab link's zero
    fold or label head: a summary row carries no link of its own, so the
    DIGGS writer still places it by the printed words, and a row the logs
    spell differently is still a hole "not here" until rows are linked.
    """
    result = test.result
    rows = list(getattr(result, "rows", ()) or ())
    missing: List[str] = []
    for row in rows:
        printed = fold_id(row.investigation_id)
        if printed and printed not in by_id and \
                row.investigation_id not in missing:
            missing.append(row.investigation_id)
    if missing:
        qa.append(QAEntry(
            kind="partial", where="lab_tests.summary_table",
            detail=f"{len(missing)} exploration(s) named in the summary "
                   f"table are not among those read from this report",
            values=missing, pages=list(test.pages)))


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
    for inv in record.investigations:
        key = fold_id(inv.investigation_id)
        if key:
            by_id.setdefault(key, inv)
    if not by_id:
        return
    for calc in record.calculations:
        for token in _SUBJECT_ID.findall(calc.subject or ""):
            inv = by_id.get(fold_id(token))
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
        folded = {fold_id(x) for x in ids}
        named = {fold_id(x) for x in stated}
        missing = [x for x in stated if fold_id(x) not in folded]
        extra = [x for x in ids if fold_id(x) not in named]
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
    """
    tables = [t for t in record.lab_tests if t.kind == "summary_table"]
    sheets = [t for t in record.lab_tests if t.kind != "summary_table"]
    if not tables or not sheets:
        return
    for table in tables:
        for row in list(getattr(table.result, "rows", ()) or ()):
            for sheet in _sheets_for(row, sheets):
                for column, names in CROSS_CHECKS.items():
                    ours = getattr(row, column, None)
                    theirs = _result_value(sheet, names)
                    if _same(ours, theirs) is False:
                        where = (f"{row.investigation_id or '?'} "
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


def _sheets_for(row: Any, sheets: Sequence[LabTest]) -> List[LabTest]:
    """The sheets that report on the same specimen as this row."""
    out: List[LabTest] = []
    printed = fold_id(row.investigation_id)
    depth = depth_m(row.depth_top)
    for sheet in sheets:
        if printed and fold_id(sheet.investigation_id) != printed:
            continue
        if depth is not None:
            got = depth_m(sheet.depth_top)
            if got is None or abs(got - depth) > DEPTH_TOL_M:
                continue
        elif fold_id(row.sample_id) and \
                fold_id(sheet.sample_id) != fold_id(row.sample_id):
            continue
        out.append(sheet)
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
            refusal = spt.refusal or any(isinstance(b, str) and "/" in b
                                         for b in spt.blows)
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
        if page is not None:
            pages = [int(page)]
        what = str(row.get("what") or row.get("field") or "reader")
        why = str(row.get("why") or row.get("detail") or "")
        values = [str(row["value"])] if row.get("value") is not None else []
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

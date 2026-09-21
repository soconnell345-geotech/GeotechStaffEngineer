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
0.15 m and whatever units each of them printed. The printed values are never
touched -- ``LabTest.investigation_id`` stays the sheet's own words -- and the
link goes in fields of its own, empty when nothing matched.

**Counts what is there.** The narrative's stated counts and named identifiers
against the investigations the appendix actually yielded.

**Cross-checks the summary table.** Most reports print a "Summary of
Laboratory Tests" table; every value in it that also appears on a sheet is
compared, and only the disagreements are recorded.

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
    Investigation, LabTest, QAEntry, Quantity, ReportRecord, Sample,
    si_numbers,
)
from report_ingest.scoring import KEY_CONTENT

__all__ = [
    "reconcile", "si_view", "DEPTH_TOL_M", "CROSS_CHECKS", "ITEM_ROLES",
    "fold_id", "depth_m",
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

    _link_lab_tests(record, qa)
    _link_calculations(record, qa)
    _check_calculations(record, qa)
    _check_counts(record, qa)
    _check_dictionaries(record, qa)
    _cross_check_summary_table(record, qa)
    _check_units(record, qa)
    _check_pages(record, qa, labels, items, no_text_pages, di_pages)
    _carry_unresolved(qa, reader_unresolved)

    if engine is not None:
        _comment_on_conflicts(qa, engine)

    record.qa.extend(qa)
    _fill_found_counts(record)
    return record


# -- the lab-to-ground link -------------------------------------------------

def _link_lab_tests(record: ReportRecord, qa: List[QAEntry]) -> None:
    by_id: Dict[str, Investigation] = {}
    for inv in record.investigations:
        key = fold_id(inv.investigation_id)
        if key:
            by_id.setdefault(key, inv)

    for test in record.lab_tests:
        if test.kind == "summary_table":
            _link_summary_rows(test, by_id, qa)
            continue
        printed = fold_id(test.investigation_id)
        if not printed:
            qa.append(QAEntry(
                kind="partial", where=f"lab_tests.{test.kind}",
                detail="the sheet names no exploration, so this test could "
                       "not be linked to the ground",
                pages=list(test.pages)))
            continue
        inv = by_id.get(printed)
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


def _sample_for(inv: Investigation,
                test: LabTest) -> Tuple[Optional[Sample], Optional[float]]:
    """The sample this sheet's specimen came from, and how far off it was."""
    if test.sample_id:
        wanted = fold_id(test.sample_id)
        for sample in inv.samples:
            if fold_id(sample.sample_id) == wanted:
                return sample, _delta(sample, test)
    depth = depth_m(test.depth_top)
    if depth is None:
        return None, None
    best: Optional[Sample] = None
    best_delta: Optional[float] = None
    for sample in inv.samples:
        top = depth_m(sample.top)
        if top is None:
            continue
        bottom = depth_m(sample.bottom)
        if bottom is not None and top - DEPTH_TOL_M <= depth <= \
                bottom + DEPTH_TOL_M:
            delta = 0.0 if top <= depth <= bottom else min(
                abs(depth - top), abs(depth - bottom))
        else:
            delta = abs(depth - top)
        if delta <= DEPTH_TOL_M and (best_delta is None or delta < best_delta):
            best, best_delta = sample, delta
    return best, best_delta


def _delta(sample: Sample, test: LabTest) -> Optional[float]:
    top, depth = depth_m(sample.top), depth_m(test.depth_top)
    if top is None or depth is None:
        return None
    return abs(depth - top)


def _link_summary_rows(test: LabTest, by_id: Dict[str, Investigation],
                       qa: List[QAEntry]) -> None:
    """A summary table names many holes; report the ones that are not here."""
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


# -- units, pages and what the readers could not settle ---------------------

def _check_units(record: ReportRecord, qa: List[QAEntry]) -> None:
    """Every printed unit with no conversion, named once with its fields."""
    seen: Dict[str, List[str]] = {}
    for where, holder in ([("investigations", inv)
                           for inv in record.investigations]
                          + [("lab_tests", test) for test in record.lab_tests]
                          + [("general", record.general)]):
        for path, value in _quantities(holder, where):
            if value.to_si() is None:
                seen.setdefault(value.unit, []).append(path)
    for unit, paths in sorted(seen.items()):
        qa.append(QAEntry(
            kind="unconverted", where="units",
            detail=f"{len(paths)} value(s) are printed in {unit!r}, which has "
                   f"no conversion in the record's unit table, so they are "
                   f"stored as printed and are absent from the SI view",
            values=paths[:10]))


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

"""Scoring one calculation: what the floor alone gave, and what the reader did.

TWO NUMBERS, NOT ONE, for the same reason the log and laboratory scorers
report two. The pattern that reads ``label -> value`` off a page's tables and
lines is free and deterministic, and on a spreadsheet printed to PDF it
already holds most of the numbers. Reporting only the reader's score would
credit the model with everything the pattern had; so every metric that CAN be
scored without meaning is computed twice -- BEFORE, over the floor alone
(:func:`report_ingest.calc_reader.floor_from_pages`), and AFTER, over the
record the reader built.

WHAT THE FLOOR CANNOT BE SCORED ON, and why that is the honest line. A
pattern over a page does not know that a printout is a settlement
calculation, that the method is Schmertmann's, or that the sheet is for the
north wing's mat. It also cannot tell an input from a result -- both are a
label with a number beside it -- so the floor puts every value it finds in
``inputs`` and the scorer searches BOTH lists for every truth value, counting
separately how many landed in the wrong one. ``kind``, ``method`` and
``subject`` therefore have no BEFORE column at all.

THE METRICS, and the tolerance each is judged at:

``kind``      what the calculation works out. Right or wrong.
``program``   the program and version as printed, fuzzy at
              :data:`PROGRAM_RATIO`. A truth file whose program is ``null``
              scores the reader for saying nothing: naming a program a
              spreadsheet does not name is a wrong answer, not a blank.
``method``    the method or standard named, fuzzy at :data:`NAME_RATIO`.
``subject``   what it is for, fuzzy at :data:`NAME_RATIO`.
``inputs``    every value the hand recorded as given: the LABEL matched
              fuzzily and the VALUE within :data:`VALUE_TOL` or the printed
              precision, whichever is looser, compared in SI wherever both
              units convert.
``results``   the same, for every value the hand recorded as worked out.

THE TRUTH FILES' OWN SHAPE is read here rather than in the harness, because
this module ships in the wheel and runs on the cluster while the measurement
script lives in the repo. A truth file names its values the way the PAGE
names them and carries the unit as a field of its own -- ``{"name": "Footing
Width B (ft)", "value": 25.8, "unit": "ft", "page": 43}`` -- because a
calculation's label is the page's own words and splitting a unit out of a key
the way the laboratory truth does would lose them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from report_ingest.model import Quantity

__all__ = [
    "METRICS", "MODEL_ONLY", "NAME_RATIO", "PROGRAM_RATIO", "VALUE_TOL",
    "Score", "CalcScore", "Expected", "expectations_for", "pages_of",
    "report_of", "score_floor", "score_record", "score_one_calc",
    "same_value", "same_name", "same_text",
]

#: Two printed labels are the same label at this partial ratio. The reader's
#: own figure (:data:`report_ingest.calc_reader.NAME_RATIO`), so the merge and
#: the scorecard draw the line in the same place.
NAME_RATIO = 80.0
#: A program name matches at this partial ratio. Tighter than a label,
#: because a program name is short and a loose match would let any four
#: letters in common pass.
PROGRAM_RATIO = 85.0
#: A value is the truth's value within two per cent, or within the last
#: printed digit, whichever is LOOSER. The relative part carries a
#: conversion; the precision part carries a page that printed 0.74.
VALUE_TOL = 0.02

#: The metrics, in the order a scorecard prints them.
METRICS: Tuple[str, ...] = ("kind", "program", "method", "subject", "inputs",
                            "results")
#: The three a pattern over the page cannot answer, so they have no BEFORE
#: column. ``program`` is NOT among them: a banner is a pattern.
MODEL_ONLY: Tuple[str, ...] = ("kind", "method", "subject")

_ID_PAGE = re.compile(r"_p(\d+)$")
_ID_REPORT = re.compile(r"(R\d{2})")


# ---------------------------------------------------------------------------
# reading a truth file
# ---------------------------------------------------------------------------

def report_of(truth: Dict[str, Any]) -> str:
    """The corpus report ID a truth file is about."""
    said = str(truth.get("report") or "").strip()
    if said:
        return said
    match = _ID_REPORT.search(str(truth.get("id") or ""))
    return match.group(1) if match else ""


def pages_of(truth: Dict[str, Any]) -> List[int]:
    """The pages of the run, from the file or from its own id.

    A calculation is a RUN of pages, so a truth file states them. The id's
    trailing ``_p<n>`` is the first page and the fallback for a file that
    states nothing, because a run of one page is a run.
    """
    pages = truth.get("pages")
    if isinstance(pages, (list, tuple)) and pages:
        return [int(p) for p in pages]
    if isinstance(pages, str) and "-" in pages:
        first, last = pages.split("-", 1)
        return list(range(int(first), int(last) + 1))
    first = truth.get("first_page")
    n = truth.get("n_pages")
    if first is not None and n:
        return list(range(int(first), int(first) + int(n)))
    match = _ID_PAGE.search(str(truth.get("id") or ""))
    if match and first is None:
        return [int(match.group(1))]
    return [int(first)] if first is not None else []


@dataclass
class Expected:
    """One value the hand recorded, with the page it was printed on."""

    name: str
    value: Optional[float] = None
    unit: str = ""
    text: str = ""
    page: Optional[int] = None
    #: Other names or values the hand judged defensible for the same thing.
    alternates: List[str] = field(default_factory=list)

    @property
    def quantity(self) -> Optional[Quantity]:
        if self.value is None:
            return None
        return Quantity(value=float(self.value), unit=self.unit)

    @property
    def shown(self) -> str:
        if self.value is None:
            return self.text
        return f"{self.value:g} {self.unit}".strip()


def _expected(rows: Any) -> List[Expected]:
    out: List[Expected] = []
    for row in (rows or ()):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        value = row.get("value")
        out.append(Expected(
            name=name,
            value=None if value is None else float(value),
            unit=str(row.get("unit") or "").strip(),
            text=str(row.get("text") or "").strip(),
            page=None if row.get("page") is None else int(row["page"]),
            alternates=[str(x) for x in (row.get("_alternates") or ())]))
    return out


def expectations_for(truth: Dict[str, Any]
                     ) -> Tuple[List[Expected], List[Expected]]:
    """``(inputs, results)`` the hand recorded for one calculation."""
    return _expected(truth.get("inputs")), _expected(truth.get("results"))


# ---------------------------------------------------------------------------
# comparing
# ---------------------------------------------------------------------------

def _fold(text: Any) -> str:
    return "".join(ch.lower() for ch in str(text or "") if ch.isalnum())


def _ratio(left: str, right: str) -> float:
    """rapidfuzz's partial ratio, or containment without it.

    rapidfuzz arrives with planlens, so it is normally here. The fallback is
    blunt rather than clever: a missing package must show up as a different
    NUMBER, never as a quietly different definition of "the same label".
    """
    if not left or not right:
        return 0.0
    try:
        from rapidfuzz import fuzz
    except ImportError:                      # pragma: no cover - fallback
        return 100.0 if left in right or right in left else 0.0
    return float(fuzz.partial_ratio(left, right))


def same_name(left: str, right: str, ratio: float = NAME_RATIO) -> bool:
    """Do two printed labels name the same thing?"""
    a, b = _fold(left), _fold(right)
    if not a or not b:
        return False
    return a == b or _ratio(a, b) >= ratio


def _precision(value: float) -> float:
    text = repr(float(value))
    if "e" in text or "E" in text:
        return abs(value) * 1e-6
    return 0.5 * (10.0 ** -(len(text.split(".")[1]) if "." in text else 0))


def same_text(said: str, wanted: str) -> bool:
    """Is a value printed as WORDS the hand's value?

    A partial ratio is the wrong tool for a one-letter answer: a seismic site
    class of ``E`` scores 100 against ``Site Class C``, because the single
    letter is somewhere inside it. So a short answer has to be the OTHER'S
    TAIL -- ``C`` matches ``Site Class C`` and ``E`` does not -- and only a
    longer one is compared fuzzily.
    """
    a, b = _fold(said), _fold(wanted)
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) <= 3:
        return a.endswith(b) or b.endswith(a)
    return _ratio(a, b) >= NAME_RATIO


def same_value(expect: Expected, got: Any) -> bool:
    """Is a record's value the hand's value?

    Compared in SI where both units convert and both are stated, and as
    printed otherwise: a record in kips beside a truth in kN is one number,
    and a record in a unit neither side can convert is compared to itself.
    """
    if expect.value is None:
        said = str(getattr(got, "text", "") or "").strip()
        if not said and getattr(got, "value", None) is not None:
            said = f"{got.value.value:g}"
        return bool(said) and same_text(said, expect.text)
    quantity = getattr(got, "value", None)
    if quantity is None:
        number = _number_in(str(getattr(got, "text", "") or ""))
        if number is None:
            return False
        quantity = Quantity(value=number, unit=expect.unit)
    wanted = expect.quantity
    left, right = quantity.value, wanted.value
    if expect.unit and quantity.unit and \
            _fold(quantity.unit) != _fold(expect.unit):
        si_got, si_want = quantity.si_value, wanted.si_value
        if si_got is None or si_want is None:
            return False
        left, right = si_got, si_want
    tol = max(abs(right) * VALUE_TOL, _precision(right))
    return abs(left - right) <= max(tol, abs(right) * 1e-9)


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _number_in(text: str) -> Optional[float]:
    match = _NUM.search(str(text or "").replace(",", ""))
    return float(match.group(0)) if match else None


def _find(expect: Expected, rows: Sequence[Any]) -> Optional[Any]:
    """The row of a record that answers one expectation, or None."""
    names = [expect.name] + list(expect.alternates)
    for row in rows:
        if any(same_name(name, row.name) for name in names) \
                and same_value(expect, row):
            return row
    return None


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------

@dataclass
class Score:
    found: int = 0
    total: int = 0
    misses: List[str] = field(default_factory=list)

    def add(self, ok: bool, what: str = "") -> None:
        self.total += 1
        if ok:
            self.found += 1
        elif what:
            self.misses.append(what)

    @property
    def rate(self) -> Optional[float]:
        return self.found / self.total if self.total else None

    def __iadd__(self, other: "Score") -> "Score":
        self.found += other.found
        self.total += other.total
        self.misses.extend(other.misses)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"found": self.found, "total": self.total, "rate": self.rate}


@dataclass
class CalcScore:
    """One calculation, scored. ``stage`` is ``floor`` or ``record``."""

    calc_id: str
    report: str
    kind: str
    stage: str
    scores: Dict[str, Score] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    model_calls: int = 0
    tool_calls: int = 0
    unresolved: int = 0
    changes: int = 0
    n_pages: int = 0
    kind_read: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    #: A truth value found in the record's OTHER list -- a result the reader
    #: filed as an input, or the reverse. Counted as FOUND, because the
    #: number and its label were recovered, and counted here as well so a
    #: scorecard can say how often the line between them was drawn wrong.
    misplaced: int = 0
    #: The model's answer scored ALONE, before the merge with the floor.
    model_alone: Optional[Dict[str, Any]] = None
    #: The merge, as the lab scorer reports it.
    floor_values: int = 0
    disagreements: int = 0
    kept: int = 0
    added: int = 0
    reconciled: int = 0

    def score(self, name: str) -> Score:
        return self.scores.setdefault(name, Score())

    @property
    def total(self) -> Score:
        out = Score()
        for name in METRICS:
            got = self.scores.get(name)
            if got is not None:
                out += got
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calc_id": self.calc_id, "report": self.report,
            "kind": self.kind, "stage": self.stage,
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
            "overall": self.total.to_dict(),
            "cost": dict(self.cost), "model_calls": self.model_calls,
            "tool_calls": self.tool_calls, "unresolved": self.unresolved,
            "changes": self.changes, "n_pages": self.n_pages,
            "kind_read": self.kind_read, "error": self.error,
            "misplaced": self.misplaced,
            "model_alone": self.model_alone,
            "floor_values": self.floor_values,
            "disagreements": self.disagreements, "kept": self.kept,
            "added": self.added, "reconciled": self.reconciled,
        }


def _blank(calc_id: str, kind: str, stage: str,
           error: Optional[str] = None) -> CalcScore:
    return CalcScore(calc_id=calc_id, report="", kind=kind, stage=stage,
                     error=error)


def _score_values(out: CalcScore, truth: Dict[str, Any],
                  calc: Any) -> None:
    """The inputs and the results, each searched in both lists."""
    inputs, results = expectations_for(truth)
    for metric, wanted, mine, theirs in (
            ("inputs", inputs, calc.inputs, calc.results),
            ("results", results, calc.results, calc.inputs)):
        for expect in wanted:
            hit = _find(expect, mine)
            where = ""
            if hit is None:
                hit = _find(expect, theirs)
                if hit is not None:
                    out.misplaced += 1
                    where = " (found in the other list)"
            out.score(metric).add(
                hit is not None,
                f"{expect.name} = {expect.shown}"
                + (f" [p{expect.page}]" if expect.page is not None else "")
                + where)


def score_floor(truth: Dict[str, Any], floor: Any) -> CalcScore:
    """Score the floor alone against the truth, as the baseline.

    ``kind``, ``method`` and ``subject`` are not asked: a pattern over a page
    cannot answer them and crediting it with any of the three would flatter
    the baseline into meaninglessness. ``program`` IS asked, because a
    banner is a pattern.
    """
    calc_id = str(truth.get("id") or "")
    out = CalcScore(calc_id=calc_id, report=report_of(truth),
                    kind=str(truth.get("kind") or ""), stage="floor")
    out.floor_values = len(floor.inputs) + len(floor.results)
    _score_program(out, truth, floor)
    _score_values(out, truth, floor)
    return out


def _score_program(out: CalcScore, truth: Dict[str, Any], calc: Any) -> None:
    wanted = truth.get("program")
    said = str(calc.program or "").strip()
    if wanted is None:
        out.score("program").add(
            not said, f"program: the pages name none and the reader said "
                      f"{said!r}")
        return
    out.score("program").add(
        bool(said) and same_name(str(wanted), said, PROGRAM_RATIO),
        f"program {wanted!r} -> {said or 'nothing'!r}")


def score_record(truth: Dict[str, Any], calc: Any) -> CalcScore:
    """Score the reader's calculation against the truth, metric by metric."""
    calc_id = str(truth.get("id") or "")
    out = CalcScore(calc_id=calc_id, report=report_of(truth),
                    kind=str(truth.get("kind") or ""), stage="record")
    out.kind_read = str(calc.kind or "")
    out.score("kind").add(out.kind_read == out.kind,
                          f"kind {out.kind} -> {out.kind_read or 'nothing'}")
    _score_program(out, truth, calc)
    for metric, wanted in (("method", truth.get("method")),
                           ("subject", truth.get("subject"))):
        if not str(wanted or "").strip():
            continue
        said = str(getattr(calc, metric, "") or "")
        alternates = [str(x) for x in
                      (truth.get("_alternates", {}) or {}).get(metric, ())]
        out.score(metric).add(
            any(same_name(one, said) for one in [str(wanted)] + alternates),
            f"{metric} {wanted!r} -> {said or 'nothing'!r}")
    _score_values(out, truth, calc)
    return out


# ---------------------------------------------------------------------------
# both, on one calculation
# ---------------------------------------------------------------------------

def score_one_calc(truth: Dict[str, Any], doc: Any, engine: Any, *,
                   budget: int = 2, report_id: str = ""
                   ) -> Tuple[CalcScore, CalcScore]:
    """``(before, after)`` for one calculation: the floor, then the reader.

    The floor is built from the same open document the reader is handed, so
    the difference between the two columns is the model and nothing else.
    """
    from report_ingest.calc_reader import floor_from_pages, read_calculation

    calc_id = str(truth.get("id") or "")
    kind = str(truth.get("kind") or "")
    pages = pages_of(truth)
    report = report_id or report_of(truth)
    if not pages:
        error = "the truth file names no page"
        return (_blank(calc_id, kind, "floor", error),
                _blank(calc_id, kind, "record", error))
    try:
        floor = floor_from_pages(doc, pages, report)
        before = score_floor(truth, floor.as_calculation(report))
        before.n_pages = len(pages)
    except Exception as exc:                      # a page that will not read
        error = f"{type(exc).__name__}: {exc}"
        return (_blank(calc_id, kind, "floor", error),
                _blank(calc_id, kind, "record", error))
    if engine is None:
        return before, _blank(calc_id, kind, "record",
                              "no engine: the floor alone was scored")
    try:
        result = read_calculation(doc, pages, engine, budget=budget,
                                  report_id=report)
    except Exception as exc:                      # a model call that failed
        return before, _blank(calc_id, kind, "record",
                              f"{type(exc).__name__}: {exc}")
    after = score_record(truth, result.calculation)
    after.cost = dict(result.cost)
    after.model_calls = result.model_calls
    after.tool_calls = result.tool_calls
    after.unresolved = len(result.unresolved)
    after.changes = len(result.changes)
    after.n_pages = len(pages)
    after.warnings = list(result.warnings)
    after.floor_values = int(result.floor_values)
    if result.model_calculation is not None:
        alone = score_record(truth, result.model_calculation)
        after.model_alone = {
            "scores": {k: v.to_dict() for k, v in alone.scores.items()},
            "overall": alone.total.to_dict()}
    after.disagreements = len(result.disagreements)
    after.kept = len(result.kept)
    after.added = len(result.added)
    after.reconciled = int(result.reconciled)
    return before, after

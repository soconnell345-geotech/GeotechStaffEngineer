"""Scoring a test pit, a cone sounding and a dynamic probe.

TWO NUMBERS, NOT ONE, for the same reason every other scorer in this package
reports two. The deterministic pass -- the log grid on a pit, the sheet's own
tabulated table or its printed axes on a sounding -- is free, and reporting
only the reader's score would credit the model with what geometry already
had. So every metric is computed BEFORE (the floor alone) and AFTER (the
record the reader built), over the same pages.

WHAT A TEST PIT IS SCORED ON. A pit is a log, so it is scored by the log
scorer's own metrics and its own tolerances -- layer tops within 0.30 m,
samples and water within 0.15 m, header fields by the same table -- plus the
one thing a pit has and a hole does not: its plan DIMENSIONS. Restating the
log scorer's rules here would let the two drift apart, so this module CALLS
it; what it adds is the ``dimensions`` metric and the file format that
carries them.

WHAT A SOUNDING IS SCORED ON, AND WHY THE TOLERANCE IS WHAT IT IS. A truth
file for a sounding is a SERIES: the values a human read at a stated depth
step. So the unit of scoring is a depth. For each truth depth, the reader's
value at that depth (within :data:`DEPTH_TOL`) is looked up, and it counts as
FOUND when it is within:

* ``max(5 %, one axis tick)`` for tip resistance and sleeve friction. Five
  per cent because a value read off a plotted trace against a printed scale
  is not better than that and a scorer that demanded more would be measuring
  the eye rather than the reader; one tick because on a coarse axis five per
  cent is finer than the paper can state;
* ``10 %`` for pore pressure, which is plotted on the coarsest axis of the
  three and is the channel a sheet most often omits;
* EXACTLY for a dynamic probe's blow count, which is a COUNT. There is no
  tolerance on counting to nineteen.

A truth depth the reader has no point at is a MISS, and a reader point at a
depth the truth does not carry is not counted either way -- the truth is a
sample of the sounding at the hand's own step and the reader may legitimately
carry more.

THE "BEFORE" COLUMN IS THE FLOOR ALONE. On a TABULATED sheet that is the
table, and it should score close to perfect -- the numbers are printed. On a
PLOTTED sheet the floor holds NO points at all, so before is zero on every
series metric and the whole of the score is the digitising. Those are two
different measurements and the scorecard prints the split, because averaging
them would hide the only number anyone wants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DEPTH_TOL", "CHANNEL_TOL", "U2_TOL", "METRICS", "PIT_METRICS",
    "SERIES_METRICS", "Score", "SoundingScore",
    "kind_of", "pages_of", "report_of", "truth_series",
    "score_pit", "score_sounding", "score_floor", "score_record",
    "score_one_sounding",
]

#: How close a reader's point has to be to a truth depth to be THAT point,
#: in metres. Tighter than the log scorer's 0.15 m sample window on purpose:
#: a sounding is read at a stated step and a point half a step away is a
#: different reading, not the same one read imprecisely.
DEPTH_TOL = 0.05
#: The fraction a tip resistance or a sleeve friction may be out by. One
#: axis tick wins where it is coarser; the truth file states the tick.
CHANNEL_TOL = 0.05
#: The same for pore pressure, which is plotted coarsest.
U2_TOL = 0.10
#: Layer tops and the pit's own dimensions, in metres. The log scorer's own
#: layer figure, because a pit's dimensions are read off the same header.
DIMENSION_TOL_M = 0.15
FT_PER_M = 3.280839895

#: The metrics of a TEST PIT: the log scorer's, plus the pit's size.
PIT_METRICS: Tuple[str, ...] = (
    "layer_top", "uscs", "sample_depth", "water", "index", "fields",
    "dimensions",
)
#: The metrics of a SOUNDING. ``header`` is the identifier, the units, the
#: cone or the hammer -- everything that is not the series.
SERIES_METRICS: Tuple[str, ...] = ("depth", "qc", "fs", "u2", "blows",
                                   "index", "header")
#: Every metric this module can print, in the order a scorecard prints them.
METRICS: Tuple[str, ...] = tuple(
    list(SERIES_METRICS) + [m for m in PIT_METRICS if m not in
                            SERIES_METRICS])


# ---------------------------------------------------------------------------
# reading a truth file
# ---------------------------------------------------------------------------

def report_of(truth: Dict[str, Any]) -> str:
    """The corpus report ID a truth file is about."""
    said = str(truth.get("report") or "").strip()
    if said:
        return said
    import re

    match = re.search(r"(R\d{2})", str(truth.get("id") or ""))
    return match.group(1) if match else ""


def kind_of(truth: Dict[str, Any]) -> str:
    """``test_pit``, ``cpt`` or ``dcp``, from the file or from its own id."""
    said = str(truth.get("kind") or "").strip()
    if said in ("test_pit", "cpt", "dcp"):
        return said
    name = str(truth.get("id") or "")
    for kind in ("test_pit", "cpt", "dcp"):
        if name.startswith(kind + "__"):
            return kind
    return "cpt"


def pages_of(truth: Dict[str, Any]) -> List[int]:
    """The pages of the sheet, from the file or from its own id."""
    pages = truth.get("pages")
    if isinstance(pages, (list, tuple)) and pages:
        return [int(p) for p in pages]
    import re

    match = re.search(r"_p(\d+)$", str(truth.get("id") or ""))
    return [int(match.group(1))] if match else []


def truth_series(truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The series a truth file carries, one entry per depth."""
    rows = truth.get("series")
    return [dict(r) for r in rows] if isinstance(rows, (list, tuple)) else []


def _to_m(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    """A depth in metres, so one tolerance means one thing on every sheet."""
    if value is None:
        return None
    key = str(unit or "").strip().lower()
    if key in ("ft", "feet", "foot", "'"):
        return float(value) / FT_PER_M
    if key == "cm":
        return float(value) / 100.0
    if key == "mm":
        return float(value) / 1000.0
    return float(value)


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
class SoundingScore:
    """One sheet, scored. ``stage`` is ``floor`` or ``record``."""

    sounding_id: str
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
    #: Whether the SHEET tabulated its series. The one fact that decides
    #: what the floor could have done, and therefore how to read `before`.
    tabulated: bool = False
    floor_points: int = 0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    #: The model's answer scored ALONE, before the merge with the floor.
    model_alone: Optional[Dict[str, Any]] = None
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
            "sounding_id": self.sounding_id, "report": self.report,
            "kind": self.kind, "stage": self.stage,
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
            "overall": self.total.to_dict(),
            "cost": dict(self.cost), "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "unresolved": self.unresolved, "changes": self.changes,
            "n_pages": self.n_pages, "tabulated": self.tabulated,
            "floor_points": self.floor_points,
            "error": self.error, "model_alone": self.model_alone,
            "disagreements": self.disagreements, "kept": self.kept,
            "added": self.added, "reconciled": self.reconciled,
        }


def _blank(truth: Dict[str, Any], stage: str,
           error: str) -> SoundingScore:
    return SoundingScore(
        sounding_id=str(truth.get("id") or ""), report=report_of(truth),
        kind=kind_of(truth), stage=stage, error=error)


# ---------------------------------------------------------------------------
# the series
# ---------------------------------------------------------------------------

def _q_m(quantity: Any) -> Optional[float]:
    """A quantity in metres, or None."""
    if quantity is None:
        return None
    converted = quantity.to_si()
    return None if converted is None else converted.value


def _si(quantity: Any) -> Optional[float]:
    """A reading's number in SI, or its PRINTED number where nothing converts.

    A dynamic resistance in daN/cm2 and a penetration index in mm/blow are
    real units that the record's conversion table has no entry for. Falling
    back to the printed number is right because :func:`_si_of` falls back
    the same way on the truth's side, so the two are compared in the same
    unit -- the sheet's own. Returning None instead would mark every such
    reading a miss however exactly the reader read it.
    """
    if quantity is None:
        return None
    if not hasattr(quantity, "to_si"):
        return float(quantity)
    converted = quantity.to_si()
    return float(quantity.value) if converted is None else converted.value


def _near(value: float, wanted: float, fraction: float,
          tick: Optional[float]) -> bool:
    """Is a reading the truth's reading, to a fraction or to one tick?"""
    slack = fraction * max(abs(wanted), 1e-9)
    if tick:
        slack = max(slack, abs(float(tick)))
    return abs(value - wanted) <= slack


def _point_at(points: Sequence[Any], depth_m: float) -> Optional[Any]:
    """The reader's point at that depth, or None."""
    best, gap = None, DEPTH_TOL
    for point in points:
        here = _q_m(point.depth)
        if here is None:
            continue
        if abs(here - depth_m) <= gap:
            best, gap = point, abs(here - depth_m)
    return best


def score_sounding(truth: Dict[str, Any], investigations: Sequence[Any],
                   stage: str) -> SoundingScore:
    """Score a cone sounding or a dynamic probe against the hand's series."""
    kind = kind_of(truth)
    out = SoundingScore(sounding_id=str(truth.get("id") or ""),
                        report=report_of(truth), kind=kind, stage=stage)
    unit = truth.get("depth_unit")
    ticks = dict(truth.get("axis_ticks") or {})
    invs = list(investigations)
    data = None
    for inv in invs:
        data = (inv.cpt if kind == "cpt" else inv.dcp) or data
    points = list(data.points) if data is not None else []

    for row in truth_series(truth):
        depth = _to_m(row.get("depth"), unit)
        if depth is None:
            continue
        point = _point_at(points, depth)
        out.score("depth").add(point is not None,
                               f"no reading at depth {row.get('depth')}")
        if point is None:
            # Every channel the hand read at this depth is a miss too: the
            # reader has nothing there at all.
            for name in (("qc", "fs", "u2") if kind == "cpt"
                         else ("blows", "index")):
                if row.get(name) is not None:
                    out.score(name).add(
                        False, f"{name} at {row.get('depth')}: no point")
            continue
        if kind == "cpt":
            for name, tol in (("qc", CHANNEL_TOL), ("fs", CHANNEL_TOL),
                              ("u2", U2_TOL)):
                wanted = row.get(name)
                if wanted is None:
                    continue
                got = _si(getattr(point, name, None))
                want_si = _si_of(wanted, truth.get(f"{name}_unit"))
                ok = (got is not None and want_si is not None
                      and _near(got, want_si, tol,
                                _tick_si(ticks.get(name),
                                         truth.get(f"{name}_unit"))))
                out.score(name).add(
                    ok, f"{name} at {row.get('depth')}: wanted {wanted}, "
                        f"got {got}")
        else:
            wanted = row.get("blows")
            if wanted is not None:
                got = getattr(point, "blows", None)
                # A blow count is a COUNT. Nineteen is not twenty.
                out.score("blows").add(
                    got is not None and abs(float(got) - float(wanted)) < 1e-9,
                    f"blows at {row.get('depth')}: wanted {wanted}, "
                    f"got {got}")
            wanted = row.get("index")
            if wanted is not None:
                got = _si(getattr(point, "index", None))
                want_si = _si_of(wanted, truth.get("index_unit"))
                out.score("index").add(
                    got is not None and want_si is not None
                    and _near(got, want_si, CHANNEL_TOL,
                              _tick_si(ticks.get("index"),
                                       truth.get("index_unit"))),
                    f"index at {row.get('depth')}: wanted {wanted}, "
                    f"got {got}")
    _score_header(out, truth, invs, data)
    return out


def _si_of(value: Any, unit: Any) -> Optional[float]:
    """A truth value in SI, using the unit the truth file states."""
    if value is None:
        return None
    text = str(unit or "").strip()
    if not text:
        return float(value)
    from report_ingest.model import to_si

    got = to_si(float(value), text)
    return got[0] if got else float(value)


def _tick_si(tick: Any, unit: Any) -> Optional[float]:
    if tick is None:
        return None
    return _si_of(tick, unit)


#: The header facts a sounding's truth file states and the record's own
#: field for each. A fact the truth file does not state is not asked.
_HEADER_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("investigation_id", "investigation_id"),
    ("depth_unit", "depth_unit"),
    ("date", "date_started"),
    ("cone_type", "__cone_type__"),
    ("standard", "__standard__"),
    ("test_type", "__test_type__"),
    ("index_name", "__index_name__"),
)


def _fold(text: Any) -> str:
    return "".join(ch.lower() for ch in str(text or "") if ch.isalnum())


def _score_header(out: SoundingScore, truth: Dict[str, Any],
                  invs: Sequence[Any], data: Any) -> None:
    """The identifier, the unit, the instrument: one metric for all of them."""
    inv = invs[0] if invs else None
    for key, attribute in _HEADER_FIELDS:
        wanted = truth.get(key)
        if wanted is None or not str(wanted).strip():
            continue
        if attribute.startswith("__"):
            got = getattr(data, attribute.strip("_"), "") if data is not None \
                else ""
        else:
            got = getattr(inv, attribute, "") if inv is not None else ""
        left, right = _fold(wanted), _fold(got)
        out.score("header").add(
            bool(right) and (left == right or left in right or right in left),
            f"{key}: wanted {wanted!r}, got {got!r}")
    for key, attribute in (("ground_level", "elevation"),
                           ("total_depth", "total_depth")):
        wanted = truth.get(key)
        if wanted is None:
            continue
        got = _q_m(getattr(inv, attribute, None)) if inv is not None else None
        want_m = _to_m(wanted, truth.get("depth_unit"))
        out.score("header").add(
            got is not None and want_m is not None
            and abs(got - want_m) <= DIMENSION_TOL_M,
            f"{key}: wanted {wanted}, got {got}")


# ---------------------------------------------------------------------------
# a test pit
# ---------------------------------------------------------------------------

def score_pit(truth: Dict[str, Any], investigations: Sequence[Any],
              stage: str) -> SoundingScore:
    """Score a test pit: the log scorer's metrics, plus its dimensions.

    The log scorer is CALLED rather than copied, so a pit and a boring are
    judged by one set of rules and one set of tolerances. What is added here
    is the one thing a pit has that a hole does not.
    """
    from report_ingest.log_scoring import score_record as score_log

    inner = score_log(truth, investigations)
    out = SoundingScore(sounding_id=str(truth.get("id") or ""),
                        report=report_of(truth), kind="test_pit", stage=stage)
    for name, score in inner.scores.items():
        if name not in PIT_METRICS:
            continue
        here = out.score(name)
        here.found += score.found
        here.total += score.total
        here.misses.extend(score.misses)
    wanted = truth.get("dimensions") or {}
    if wanted:
        pit = None
        for inv in investigations:
            pit = inv.pit or pit
        unit = wanted.get("unit") or truth.get("depth_unit")
        for name in ("length", "width", "depth"):
            value = wanted.get(name)
            if value is None:
                continue
            got = _q_m(getattr(pit, name, None)) if pit is not None else None
            want_m = _to_m(value, unit)
            out.score("dimensions").add(
                got is not None and want_m is not None
                and abs(got - want_m) <= DIMENSION_TOL_M,
                f"pit {name}: wanted {value} {unit}, got {got}")
    return out


# ---------------------------------------------------------------------------
# before and after
# ---------------------------------------------------------------------------

def score_floor(truth: Dict[str, Any], investigations: Sequence[Any]
                ) -> SoundingScore:
    """The floor alone, on whatever kind of sheet this is."""
    kind = kind_of(truth)
    if kind == "test_pit":
        return score_pit(truth, investigations, "floor")
    return score_sounding(truth, investigations, "floor")


def score_record(truth: Dict[str, Any], investigations: Sequence[Any]
                 ) -> SoundingScore:
    """The reader's record, on whatever kind of sheet this is."""
    kind = kind_of(truth)
    if kind == "test_pit":
        return score_pit(truth, investigations, "record")
    return score_sounding(truth, investigations, "record")


def score_one_sounding(truth: Dict[str, Any], doc: Any, engine: Any, *,
                       budget: int = 2, report_id: str = ""
                       ) -> Tuple[SoundingScore, SoundingScore]:
    """``(before, after)`` for one sheet: the floor, then the reader.

    The floor is built from the same open document the reader is handed, so
    the difference between the two columns is the model and nothing else.
    ``engine`` may be None, which scores the floor alone -- no model, no
    network, and the honest baseline.
    """
    kind = kind_of(truth)
    pages = pages_of(truth)
    report = report_id or report_of(truth)
    if not pages:
        error = "the truth file names no page"
        return _blank(truth, "floor", error), _blank(truth, "record", error)

    if kind == "test_pit":
        return _score_one_pit(truth, doc, engine, pages, report, budget)
    return _score_one_sounding(truth, doc, engine, pages, report, budget)


def _score_one_pit(truth: Dict[str, Any], doc: Any, engine: Any,
                   pages: Sequence[int], report: str, budget: int
                   ) -> Tuple[SoundingScore, SoundingScore]:
    from planlens.document.loggrid import log_grid

    from report_ingest.log_floor import seed_from_grid

    try:
        grid = log_grid(doc, pages)
        lines: List[Any] = []
        for page in pages:
            try:
                lines.extend(doc.page(page).lines)
            except Exception:
                continue
        floor = seed_from_grid(grid, pages, report, lines=lines)
    except Exception as exc:                      # a page that will not read
        error = f"{type(exc).__name__}: {exc}"
        return _blank(truth, "floor", error), _blank(truth, "record", error)
    before = score_pit(truth, [floor], "floor")
    before.n_pages = len(pages)
    if engine is None:
        return before, _blank(truth, "record",
                              "no engine: the floor alone was scored")
    from report_ingest.log_reader import read_log

    try:
        result = read_log(doc, pages, engine, budget=max(1, budget), grid=grid,
                          report_id=report)
    except Exception as exc:                      # a model call that failed
        return before, _blank(truth, "record", f"{type(exc).__name__}: {exc}")
    after = score_pit(truth, [result.investigation], "record")
    after.n_pages = len(pages)
    after.cost = dict(result.cost)
    after.model_calls = result.model_calls
    after.unresolved = len(result.unresolved)
    after.changes = len(result.changes)
    after.warnings = list(result.warnings)
    if result.model_investigation is not None:
        alone = score_pit(truth, [result.model_investigation], "record")
        after.model_alone = {
            "scores": {k: v.to_dict() for k, v in alone.scores.items()},
            "overall": alone.total.to_dict()}
    after.disagreements = len(result.disagreements)
    after.kept = len(result.kept)
    after.added = len(result.added)
    after.reconciled = int(result.reconciled)
    return before, after


def _score_one_sounding(truth: Dict[str, Any], doc: Any, engine: Any,
                        pages: Sequence[int], report: str, budget: int
                        ) -> Tuple[SoundingScore, SoundingScore]:
    from report_ingest.sounding_reader import read_sounding, sounding_floor

    kind = kind_of(truth)
    try:
        floor = sounding_floor(doc, pages, kind, report)
    except Exception as exc:                      # a page that will not read
        error = f"{type(exc).__name__}: {exc}"
        return _blank(truth, "floor", error), _blank(truth, "record", error)
    seed = [floor.investigation] if floor.investigation is not None else []
    before = score_sounding(truth, seed, "floor")
    before.n_pages = len(pages)
    before.tabulated = floor.tabulated
    before.floor_points = floor.n_points
    before.warnings = list(floor.warnings)
    if engine is None:
        return before, _blank(truth, "record",
                              "no engine: the floor alone was scored")
    try:
        result = read_sounding(doc, pages, engine, kind=kind,
                               budget=max(1, budget), report_id=report)
    except Exception as exc:                      # a model call that failed
        return before, _blank(truth, "record", f"{type(exc).__name__}: {exc}")
    after = score_sounding(truth, [result.investigation], "record")
    after.n_pages = len(pages)
    after.tabulated = result.tabulated
    after.floor_points = result.floor_points
    after.cost = dict(result.cost)
    after.model_calls = result.model_calls
    after.tool_calls = result.tool_calls
    after.unresolved = len(result.unresolved)
    after.changes = len(result.changes)
    after.warnings = list(result.warnings)
    if result.model_investigation is not None:
        alone = score_sounding(truth, [result.model_investigation], "record")
        after.model_alone = {
            "scores": {k: v.to_dict() for k, v in alone.scores.items()},
            "overall": alone.total.to_dict()}
    after.disagreements = len(result.disagreements)
    after.kept = len(result.kept)
    after.added = len(result.added)
    after.reconciled = int(result.reconciled)
    return before, after

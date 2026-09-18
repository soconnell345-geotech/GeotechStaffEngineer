"""Scoring one log: what the grid alone recovered, and what the reader did.

TWO NUMBERS, NOT ONE. ``log_grid`` places text at a column and a depth and
asserts no meaning; the reader turns that into an Investigation. Reporting
only the reader's score would credit it with everything the geometry already
had, so every metric here is computed twice -- BEFORE, from the grid's cells
alone, and AFTER, from the record the reader built. The gap between them is
what the model bought.

THE MATCHING RULES ARE THE WP2a ONES, because the two measurements have to
mean the same thing. They are restated here rather than imported, because
this module ships in the wheel and runs on the cluster while the WP2a script
lives in the repo's harness folder. The rules, and why each is what it is:

* depths are compared in METRES whatever the log prints, so one tolerance
  means the same thing on a metric and an imperial log;
* a sample is an INTERVAL, not a point, and half the templates print a blow
  record against the middle of it rather than its top -- so the window is the
  sample interval widened by the tolerance, and where the truth states only a
  top it is the top plus a 0.46 m drive;
* a blow record counts as found when its drives appear IN ORDER among the
  numbers standing in that window, because some forms print ``5-9-12`` in one
  cell and some print each drive on its own line;
* an N value counts as found when it is PRINTED, or when the drives that
  define it -- the second and third six inches -- stand at that depth. Half
  the templates print only the drives, and neither the grid nor the reader
  does arithmetic by design, so counting that as a miss would score a
  decision rather than a defect;
* a column counts as the right one when it carries the value's canonical name
  or one of its family, because a form that heads one column "SAMPLING DATA"
  and prints the id, the drives and the recovery inside it is not wrong.

WHAT THE RECORD ADDS, which the grid cannot be scored on the same way: a USCS
symbol bound to the layer that carries it, a water reading bound to its
timing, a recovery bound to its sample. Those are scored before as "is the
value anywhere in the right place on the page" and after as "is it on the
right object", which is the honest comparison -- the grid never claimed the
binding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SAMPLE_TOL_M", "LAYER_TOL_M", "WATER_TOL_M", "DRIVEN_LENGTH_M",
    "COLUMN_FAMILY", "FIELD_MAP", "INDEX_KEYS", "METRICS",
    "Score", "LogScore", "score_grid", "score_record", "score_one_log",
    "truth_investigations",
]

#: From the plan. A sample or index value has to land within 0.15 m and a
#: layer top within 0.3 m; a water level is a sample-grade reading.
SAMPLE_TOL_M = 0.15
LAYER_TOL_M = 0.30
WATER_TOL_M = 0.15
FT_PER_M = 3.280839895
#: What a driven sample is taken to be where the truth states only its top.
#: An 18 in drive is 0.46 m and a metric one is 0.45, so one number covers
#: both.
DRIVEN_LENGTH_M = 0.46

#: Which canonical grid-column names count as the right home for each truth
#: value. Copied from the WP2a scorer so before and after ask the same thing.
COLUMN_FAMILY: Dict[str, Tuple[str, ...]] = {
    "blows": ("blows", "n_value", "sample_id", "tests"),
    "n": ("n_value", "blows", "sample_id", "tests"),
    "wc": ("water_content", "tests"),
    "duw": ("dry_unit_weight", "tests"),
    "ll": ("liquid_limit", "plasticity_index", "tests"),
    "pl": ("plastic_limit", "plasticity_index", "tests"),
    "pi": ("plasticity_index", "tests"),
    "fines": ("fines", "tests"),
    "qu": ("qu", "tests"),
    "rqd": ("rqd", "recovery", "tests"),
    "pp_kpa": ("pocket_pen", "qu", "tests"),
    "recovery": ("recovery", "sample_id", "tests"),
}

#: The per-sample values scored as index properties, in the truth's spelling.
INDEX_KEYS: Tuple[str, ...] = ("wc", "duw", "ll", "pl", "pi", "fines", "qu",
                               "rqd", "pp_kpa")

#: Truth header-field key -> the grid's canonical keys. The WP2a table.
FIELD_MAP: Dict[str, Tuple[str, ...]] = {
    "boring_id": ("boring_id", "test_pit_id"),
    "test_pit_id": ("test_pit_id", "boring_id"),
    "project": ("project",),
    "contract_number": ("project_number",),
    "project_number": ("project_number",),
    "client": ("client",),
    "hammer": ("hammer_type",),
    "hammer_type": ("hammer_type",),
    "advancement_method": ("drilling_method",),
    "method": ("drilling_method",),
    "drilling_method": ("drilling_method",),
    "equipment": ("drilling_equipment",),
    "drill_rig": ("drilling_equipment",),
    "driller": ("driller", "contractor"),
    "contractor": ("contractor", "driller"),
    "foreman": ("foreman", "driller"),
    "representative": ("logged_by",),
    "logged_by": ("logged_by",),
    "date_started": ("date_started",),
    "date_completed": ("date_finished",),
    "date_finished": ("date_finished",),
    "ground_surface_elevation": ("ground_surface_elevation",),
    "total_depth": ("total_depth",),
    "sheet": ("sheet",),
    "page_of": ("sheet",),
    "water_observations": ("groundwater",),
}

#: The metrics, in the order a scorecard prints them.
METRICS: Tuple[str, ...] = ("n_value", "blows", "sample_depth", "layer_top",
                            "uscs", "water", "recovery", "index", "fields")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _to_m(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / FT_PER_M if unit == "ft" else float(value)


def _numbers(text: str) -> Tuple[float, ...]:
    from planlens.document.loggrid import numbers_in
    return numbers_in(str(text or ""))


def _close(a: float, b: float) -> bool:
    """Is a printed number the truth's number? Two per cent or 0.5, whichever
    is larger -- a log prints 12 for 11.8 and 116 for 115.6."""
    return abs(a - b) <= max(0.5, 0.02 * abs(b))


def _wanted_numbers(value: Any) -> List[float]:
    """The numbers in a truth value, however the hand wrote it."""
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    return list(_numbers(str(value)))


def _blow_numbers(blows: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for entry in blows:
        if isinstance(entry, (int, float)):
            out.append(float(entry))
        else:
            out.extend(_numbers(str(entry)))
    return out


def _in_order(got: Sequence[float], want: Sequence[float]) -> bool:
    """Does ``got`` contain ``want`` as a run, in order?"""
    if not want or len(want) > len(got):
        return False
    for i in range(len(got) - len(want) + 1):
        if all(_close(got[i + j], want[j]) for j in range(len(want))):
            return True
    return False


def _as_multiset(got: Sequence[float], want: Sequence[float]) -> bool:
    """Are all of ``want`` present in ``got``, order not asked?

    Spread one drive per cell, the grid asserts no order -- each cell is an
    independent value with its own box -- so a stack is checked this way and
    a single cell is checked in order.
    """
    pool = list(got)
    for value in want:
        hit = next((n for n in pool if _close(n, value)), None)
        if hit is None:
            return False
        pool.remove(hit)
    return True


def _same_text(a: str, b: str) -> bool:
    left = "".join(ch.lower() for ch in str(a or "") if ch.isalnum())
    right = "".join(ch.lower() for ch in str(b or "") if ch.isalnum())
    return bool(left) and (left in right or right in left)


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
        return {"found": self.found, "total": self.total,
                "rate": self.rate}


@dataclass
class LogScore:
    """One log, scored. ``stage`` is ``grid`` or ``record``."""

    log_id: str
    report: str
    stage: str
    scores: Dict[str, Score] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    model_calls: int = 0
    unresolved: int = 0
    changes: int = 0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

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
            "log_id": self.log_id, "report": self.report, "stage": self.stage,
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
            "overall": self.total.to_dict(),
            "cost": dict(self.cost), "model_calls": self.model_calls,
            "unresolved": self.unresolved, "changes": self.changes,
            "error": self.error,
        }


def truth_investigations(truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The blocks of one truth file: usually one, several on a tabular sheet."""
    blocks = list(truth.get("investigations") or [])
    return blocks if blocks else [truth]


def _flat(truth: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for block in truth_investigations(truth):
        out.extend(block.get(key) or [])
    return out


def _window(sample: Dict[str, Any], unit: Optional[str], tol: float
            ) -> Optional[Tuple[float, float]]:
    top = sample.get("top")
    if top is None:
        return None
    lo = _to_m(top, unit)
    bottom = sample.get("bottom")
    hi = _to_m(bottom, unit) if bottom is not None else lo + DRIVEN_LENGTH_M
    return lo - tol, hi + tol


# ---------------------------------------------------------------------------
# BEFORE: what the grid alone recovered
# ---------------------------------------------------------------------------

def score_grid(truth: Dict[str, Any], grid: Any) -> LogScore:
    """Score ``log_grid``'s own cells against the truth, the WP2a way."""
    log_id = truth["id"]
    out = LogScore(log_id=log_id, report=log_id.split("_")[0], stage="grid")
    out.warnings = list(getattr(grid, "warnings", ()) or ())
    unit = truth.get("depth_unit")
    samples = _flat(truth, "samples")
    layers = _flat(truth, "layers")
    waters = _flat(truth, "water")
    cells = [c for c in grid.rows if c.depth is not None]

    def cell_m(value):
        return _to_m(value, grid.unit)

    def in_window(cell, window) -> bool:
        lo, hi = window
        depth = cell_m(cell.depth)
        if depth is not None and lo <= depth <= hi:
            return True
        top, bottom = cell_m(cell.depth_top), cell_m(cell.depth_bottom)
        return (top is not None and bottom is not None
                and top <= hi and bottom >= lo)

    def family(cell, names: Sequence[str]) -> bool:
        column = grid.column(cell.column_id)
        return bool(names) and bool(set(column.names if column else ())
                                    & set(names))

    def numbers_in_window(window, names: Sequence[str]) -> List[float]:
        run: List[Tuple[float, float, int, float]] = []
        for cell in cells:
            if not family(cell, names) or not in_window(cell, window):
                continue
            for order, number in enumerate(cell.numbers):
                run.append((cell.depth or 0.0, cell.bbox[0], order, number))
        run.sort()
        return [n for *_rest, n in run]

    def one_cell_records(window, names, want) -> bool:
        for cell in cells:
            if not family(cell, names) or not in_window(cell, window):
                continue
            if _in_order(list(cell.numbers), want):
                return True
        return False

    for sample in samples:
        window = _window(sample, unit, SAMPLE_TOL_M)
        if window is None:
            continue
        depth_text = f"{sample.get('top')} {unit}"
        blows = sample.get("blows") or []
        want = _blow_numbers(blows)
        if blows:
            tight = numbers_in_window(window, ("blows", "n_value"))
            wide = numbers_in_window(window, COLUMN_FAMILY["blows"])
            ok = (one_cell_records(window, ("blows", "n_value"), want)
                  or one_cell_records(window, COLUMN_FAMILY["blows"], want)
                  or _as_multiset(tight, want) or _as_multiset(wide, want))
            out.score("blows").add(ok, f"blows {blows} at {depth_text}")
        if sample.get("n") is not None:
            n = float(sample["n"])
            pool = numbers_in_window(window, COLUMN_FAMILY["n"])
            ok = any(_close(v, n) for v in pool)
            if not ok and len(want) >= 3 and _close(want[1] + want[2], n):
                ok = _as_multiset(
                    numbers_in_window(window, ("blows", "n_value")), want)
            out.score("n_value").add(ok, f"N={sample['n']} at {depth_text}")
        # A sample DEPTH is recovered when anything belonging to that sample
        # stands in its window: the grid has no sample object to bind to.
        placed = bool(numbers_in_window(window, COLUMN_FAMILY["blows"])
                      or numbers_in_window(window, COLUMN_FAMILY["recovery"]))
        out.score("sample_depth").add(placed, f"sample at {depth_text}")
        for key in ("recovery", "rqd"):
            value = sample.get(key)
            if value is None:
                continue
            want_values = _wanted_numbers(value)
            pool = numbers_in_window(window, COLUMN_FAMILY[key])
            out.score("recovery").add(
                any(_close(v, w) for v in pool for w in want_values),
                f"{key}={value} at {depth_text}")
        for key in INDEX_KEYS:
            if key in ("rqd",):
                continue
            value = sample.get(key)
            if value is None:
                continue
            want_values = _wanted_numbers(value)
            if not want_values:
                continue
            pool = numbers_in_window(window, COLUMN_FAMILY[key])
            out.score("index").add(
                any(_close(v, w) for v in pool for w in want_values),
                f"{key}={value} at {depth_text}")

    tops = [_to_m(ly.top, grid.unit) for ly in grid.layers
            if ly.top is not None]
    for layer in layers:
        top = layer.get("top")
        if top is None:
            continue
        want = _to_m(top, unit)
        out.score("layer_top").add(
            any(t is not None and abs(t - want) <= LAYER_TOL_M for t in tops),
            f"layer top {top} {unit}")
        uscs = layer.get("uscs")
        if uscs:
            # The grid binds no symbol to a layer. It is credited when the
            # symbol is in the description it did bind at that top.
            ok = any(ly.top is not None
                     and abs((_to_m(ly.top, grid.unit) or 1e9) - want)
                     <= LAYER_TOL_M
                     and str(uscs).upper() in (ly.description or "").upper()
                     for ly in grid.layers)
            out.score("uscs").add(ok, f"USCS {uscs} at {top} {unit}")

    # Water: the grid has no water object, so it is credited when the depth
    # is printed anywhere it placed or in the groundwater header field.
    everywhere = [n for cell in grid.rows for n in cell.numbers]
    everywhere += list(_numbers(grid.fields.get("groundwater", "")))
    for water in waters:
        depth = water.get("depth")
        if depth is None:
            continue
        want = _to_m(depth, unit)
        got = [_to_m(v, grid.unit) for v in everywhere]
        out.score("water").add(
            any(v is not None and abs(v - want) <= WATER_TOL_M for v in got),
            f"water at {depth} {unit}")

    fields = dict(truth.get("fields") or {})
    for key, value in fields.items():
        names = FIELD_MAP.get(key)
        if not names:
            continue
        got = ""
        for name in names:
            if grid.fields.get(name):
                got = grid.fields[name]
                break
        ok = False
        if got:
            if isinstance(value, (int, float)):
                ok = any(_close(n, float(value)) for n in _numbers(got))
            else:
                ok = _same_text(value, got)
        out.score("fields").add(ok, f"{key}={value!r} -> {got!r}")
    return out


# ---------------------------------------------------------------------------
# AFTER: what the reader's record holds
# ---------------------------------------------------------------------------

def _record_fields(inv: Any) -> Dict[str, str]:
    """Everything the record knows that a truth field could be checked against.

    The record puts the hammer on ``drilling`` and the dates on the
    investigation, and keeps whatever else the header printed in ``fields``.
    All three are one namespace for scoring.
    """
    out: Dict[str, str] = {}

    def put(key: str, value: Any) -> None:
        """Set a key only when there is something in it.

        Writing an EMPTY string would be worse than not writing at all: the
        ``setdefault`` below would then find the key taken and never reach
        the log's own value for it, and the metric would count a miss the
        reader did not make.
        """
        if str(value or "").strip():
            out[key] = str(value)

    put("boring_id", inv.investigation_id)
    put("test_pit_id", inv.investigation_id)
    put("hammer_type", inv.drilling.hammer_type)
    put("drilling_method", inv.drilling.method)
    put("drilling_equipment", inv.drilling.equipment)
    put("driller", inv.drilling.driller)
    put("contractor", inv.drilling.contractor or inv.drilling.driller)
    put("foreman", inv.drilling.driller)
    put("logged_by", inv.drilling.logged_by)
    put("date_started", inv.date_started)
    put("date_finished", inv.date_finished)
    put("sheet", inv.sheet)
    if inv.total_depth is not None:
        put("total_depth", f"{inv.total_depth.value:g}")
    if inv.elevation is not None:
        put("ground_surface_elevation", f"{inv.elevation.value:g}")
    if inv.water:
        put("groundwater", " ".join(
            (f"{w.depth.value:g}" if w.depth is not None else "") +
            f" {w.when} {w.note}" for w in inv.water))
    for key, value in inv.fields.items():
        out.setdefault(key, value)
        # The grid's canonical names are what FIELD_MAP points at, and the
        # record keeps the log's own keys, so both spellings are offered.
        for canonical in FIELD_MAP.get(key, ()):
            out.setdefault(canonical, value)
    return {k: v for k, v in out.items() if str(v or "").strip()}


def score_record(truth: Dict[str, Any], investigations: Sequence[Any]
                 ) -> LogScore:
    """Score the reader's record against the truth, metric by metric."""
    log_id = truth["id"]
    out = LogScore(log_id=log_id, report=log_id.split("_")[0], stage="record")
    unit = truth.get("depth_unit")
    invs = list(investigations)
    if not invs:
        # Nothing came back. Every truth value is a miss, which is the right
        # score: a reader that returns nothing has recovered nothing.
        invs = []

    def q_m(quantity) -> Optional[float]:
        if quantity is None:
            return None
        converted = quantity.to_si()
        return None if converted is None else converted.value

    record_samples = [(s, q_m(s.top), q_m(s.bottom))
                      for inv in invs for s in inv.samples]
    record_spt = [(r, q_m(r.depth_top)) for inv in invs for r in inv.spt]
    record_layers = [(ly, q_m(ly.top)) for inv in invs for ly in inv.layers]
    record_water = [q_m(w.depth) for inv in invs for w in inv.water]
    record_fields: Dict[str, str] = {}
    for inv in invs:
        for key, value in _record_fields(inv).items():
            record_fields.setdefault(key, value)

    for sample in _flat(truth, "samples"):
        window = _window(sample, unit, SAMPLE_TOL_M)
        if window is None:
            continue
        lo, hi = window
        depth_text = f"{sample.get('top')} {unit}"
        near_samples = [s for s, top, _b in record_samples
                        if top is not None and lo <= top <= hi]
        near_spt = [r for r, top in record_spt
                    if top is not None and lo <= top <= hi]
        out.score("sample_depth").add(
            bool(near_samples or near_spt), f"sample at {depth_text}")

        blows = sample.get("blows") or []
        want = _blow_numbers(blows)
        if blows:
            ok = any(_in_order(_blow_numbers(r.blows), want)
                     or _as_multiset(_blow_numbers(r.blows), want)
                     for r in near_spt)
            out.score("blows").add(ok, f"blows {blows} at {depth_text}")
        if sample.get("n") is not None:
            n = float(sample["n"])
            ok = any(r.n is not None and _close(float(r.n), n)
                     for r in near_spt)
            if not ok and len(want) >= 3 and _close(want[1] + want[2], n):
                # The log printed only the drives, and neither the grid nor
                # the reader does arithmetic. The drives standing at the
                # right depth ARE the N being present.
                ok = any(r.n is None
                         and (_in_order(_blow_numbers(r.blows), want)
                              or _as_multiset(_blow_numbers(r.blows), want))
                         for r in near_spt)
            out.score("n_value").add(ok, f"N={sample['n']} at {depth_text}")

        for key, attribute, ceiling in (("recovery", "recovery_percent", 100),
                                        ("rqd", "rqd_percent", 100)):
            value = sample.get(key)
            if value is None:
                continue
            want_values = _wanted_numbers(value)
            ok = any(getattr(s, attribute) is not None
                     and any(_close(getattr(s, attribute), w)
                             for w in want_values)
                     for s in near_samples)
            out.score("recovery").add(ok, f"{key}={value} at {depth_text}")

        for key, attribute in (("wc", "water_content"),
                               ("duw", "dry_unit_weight"),
                               ("ll", "liquid_limit"),
                               ("pl", "plastic_limit"),
                               ("pi", "plasticity_index"),
                               ("fines", "fines_percent"),
                               ("qu", "qu"), ("pp_kpa", "pocket_pen")):
            value = sample.get(key)
            if value is None:
                continue
            want_values = _wanted_numbers(value)
            if not want_values:
                continue
            ok = False
            for s in near_samples:
                got = getattr(s, attribute)
                if got is None:
                    continue
                number = got.value if hasattr(got, "value") else float(got)
                if any(_close(number, w) for w in want_values):
                    ok = True
                    break
            out.score("index").add(ok, f"{key}={value} at {depth_text}")

    for layer in _flat(truth, "layers"):
        top = layer.get("top")
        if top is None:
            continue
        want = _to_m(top, unit)
        near = [ly for ly, ly_top in record_layers
                if ly_top is not None and abs(ly_top - want) <= LAYER_TOL_M]
        out.score("layer_top").add(bool(near), f"layer top {top} {unit}")
        uscs = layer.get("uscs")
        if uscs:
            out.score("uscs").add(
                any(_same_text(ly.uscs, str(uscs)) for ly in near),
                f"USCS {uscs} at {top} {unit}")

    for water in _flat(truth, "water"):
        depth = water.get("depth")
        if depth is None:
            continue
        want = _to_m(depth, unit)
        out.score("water").add(
            any(d is not None and abs(d - want) <= WATER_TOL_M
                for d in record_water),
            f"water at {depth} {unit}")

    for key, value in (truth.get("fields") or {}).items():
        if key not in FIELD_MAP:
            continue
        got = record_fields.get(key, "")
        if not got:
            for canonical in FIELD_MAP[key]:
                if record_fields.get(canonical):
                    got = record_fields[canonical]
                    break
        ok = False
        if got:
            if isinstance(value, (int, float)):
                ok = any(_close(n, float(value)) for n in _numbers(got))
            else:
                ok = _same_text(value, got)
        out.score("fields").add(ok, f"{key}={value!r} -> {got!r}")
    return out


# ---------------------------------------------------------------------------
# both, on one log
# ---------------------------------------------------------------------------

def score_one_log(truth: Dict[str, Any], doc: Any, engine: Any, *,
                  budget: int = 6, report_id: str = ""
                  ) -> Tuple[LogScore, LogScore]:
    """``(before, after)`` for one log: run the grid, then the reader.

    The grid runs ONCE and is handed to the reader, so the two scores are
    computed over exactly the same geometry and the difference between them
    is the model and nothing else.
    """
    from planlens.document.loggrid import log_grid

    from report_ingest.log_reader import read_log

    log_id = truth["id"]
    pages = [int(p) for p in truth.get("pages") or []]
    report = report_id or log_id.split("_")[0]
    try:
        grid = log_grid(doc, pages)
    except Exception as exc:                        # a page that will not read
        error = f"{type(exc).__name__}: {exc}"
        before = LogScore(log_id=log_id, report=report, stage="grid",
                          error=error)
        after = LogScore(log_id=log_id, report=report, stage="record",
                         error=error)
        return before, after

    before = score_grid(truth, grid)
    try:
        result = read_log(doc, pages, engine, budget=budget, grid=grid,
                          report_id=report)
    except Exception as exc:                        # a model call that failed
        after = LogScore(log_id=log_id, report=report, stage="record",
                         error=f"{type(exc).__name__}: {exc}")
        return before, after

    after = score_record(truth, [result.investigation])
    after.cost = dict(result.cost)
    after.model_calls = result.model_calls
    after.unresolved = len(result.unresolved)
    after.changes = len(result.changes)
    after.warnings = list(result.warnings)
    return before, after

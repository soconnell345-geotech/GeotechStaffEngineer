"""Scoring one lab sheet: what the tables alone gave, and what the reader did.

TWO NUMBERS, NOT ONE, for the same reason the log scorer reports two. A page's
detected TABLES are free and deterministic, and on a tabulated sheet they
already hold most of the numbers. Reporting only the reader's score would
credit the model with everything the table extractor had; so every metric that
CAN be scored without meaning is computed twice -- BEFORE, over the numbers in
the page's tables, and AFTER, over the records the reader built.

WHAT THE TABLES CANNOT BE SCORED ON, and why that is the honest line. A table
is a grid of numbers. It does not know that 31 is a liquid limit, that this
sheet is a direct shear test, or that the specimen came from B-1 at 2.5 ft --
and a scorer that gave it credit for those would be scoring the reader's job
against the extractor's output. So ``kind`` and ``link`` have no BEFORE column
at all, and the index, series and curve metrics ask of the tables only "is
this number on the page", which is the most a table can answer.

THE METRICS, and the tolerance each is judged at:

``kind``      the sheet's test kind, from the sheet's own title. Right or wrong.
``link``      the boring identifier AND the depth printed on the sheet, the
              depth within 0.15 m, compared in metres whatever the sheet
              prints.
``index``     every scalar the sheet printed -- limits, water content,
              densities, fractions, strengths, pH, resistivity -- EXACT, to
              within a rounding of the last printed digit.
``series``    a grading curve's percent-passing values, each within 1 percent.
``curve``     a plotted curve's points, within the tolerance the truth file
              states for that sheet, because a curve read off a plot is worth
              what the plot's own resolution is worth and no more.

Everything is compared in SI, so one tolerance means the same thing on a sheet
printing psf and a sheet printing kPa.

THE TRUTH FILES' OWN SHAPE is read here rather than in the harness, because
this module ships in the wheel and runs on the cluster while the measurement
script lives in the repo. The hand wrote each sheet's unit into the KEY --
``c_psf``, ``ucs_MPa``, ``resistivity_ohm_cm`` -- so a key is split into a
stem and a unit suffix and values are looked up by stem. That is why a truth
sheet nobody has written a line of code for still reads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from report_ingest.model import Quantity, si_numbers, to_si

__all__ = [
    "UNIT_SUFFIX", "SIEVE_MM", "DEPTH_TOL_M", "EXACT_TOL", "PASSING_TOL",
    "METRICS", "Score", "LabScore", "Expectation",
    "find", "number", "quantity", "reported", "depth_unit_of", "report_of",
    "pages_of", "sieve_points", "curve_in", "expectations_for",
    "score_tables", "score_record", "score_one_sheet", "table_numbers",
]

#: A specimen is linked when its depth lands within this, in metres. The log
#: scorer's sample tolerance, for the same reason: a sheet prints the depth
#: the sampler was at and a log prints the interval it came from.
DEPTH_TOL_M = 0.15
#: An index value is EXACT when it agrees to a rounding of the last digit a
#: sheet prints. Not a measurement tolerance: a liquid limit of 31 that comes
#: back as 32 is a misreading, not a disagreement.
EXACT_TOL = 0.01
#: The plan's tolerance for a grading: each percent-passing value within one
#: percentage point.
PASSING_TOL = 1.0
#: What a curve is judged at when its truth file states no tolerance of its
#: own: 2 % of the values' own span, which is the plan's figure for a point
#: digitised off a plot.
CURVE_SPAN_FRACTION = 0.02

#: The metrics, in the order a scorecard prints them.
METRICS: Tuple[str, ...] = ("kind", "link", "index", "series", "curve")
#: The two a page's tables cannot answer, so they have no BEFORE column.
MODEL_ONLY: Tuple[str, ...] = ("kind", "link")

#: The unit a truth key's suffix means. The hand wrote each sheet's own unit
#: into its key, so this table is the whole of the unit handling.
UNIT_SUFFIX: Dict[str, str] = {
    "pct": "%", "percent": "%",
    "psf": "psf", "psi": "psi", "tsf": "tsf", "ksf": "ksf",
    "ton_ft2": "tsf", "tons_ft2": "tsf", "kg_cm2": "kg/cm2",
    "kpa": "kPa", "mpa": "MPa", "bar": "bar",
    "pcf": "pcf", "g_cm3": "g/cm3", "mg_m3": "Mg/m3", "kn_m3": "kN/m3",
    "t_m3": "t/m3",
    "mm": "mm", "cm": "cm", "m": "m", "in": "in", "ft": "ft",
    "ohm_cm": "ohm-cm", "kohm_cm": "kohm-cm", "ohm_m": "ohm.m",
    "mg_kg": "mg/kg", "ppm": "ppm", "mv": "mV",
    "c": "degC", "deg": "deg", "degrees": "deg",
    "cm3": "cm3", "ml": "ml", "g": "g", "kg": "kg", "kn": "kN",
    "m_s": "m/s", "cm_s": "cm/s",
}

#: The opening of a sieve the trade names by number or by fraction, in
#: millimetres. Only what these sheets actually print; a designation not here
#: keeps its name and gets no size, because the record would rather say it
#: does not know the opening than invent one.
SIEVE_MM: Dict[str, float] = {
    "3": 75.0, "3 in": 75.0, '3"': 75.0,
    "2": 50.8, '2"': 50.8,
    "1 1/2": 38.1, '1 1/2"': 38.1, '1-1/2"': 38.1,
    "1": 25.4, '1"': 25.4,
    "3/4": 19.0, '3/4"': 19.0,
    "1/2": 12.7, '1/2"': 12.7,
    "3/8": 9.5, '3/8"': 9.5,
    "no. 4": 4.75, "4": 4.75,
    "no. 8": 2.36, "8": 2.36,
    "no. 10": 2.0, "10": 2.0,
    "no. 16": 1.18, "16": 1.18,
    "no. 20": 0.85, "20": 0.85,
    "no. 30": 0.6, "30": 0.6,
    "no. 40": 0.425, "40": 0.425,
    "no. 50": 0.3, "50": 0.3,
    "no. 60": 0.25, "60": 0.25,
    "no. 100": 0.15, "100": 0.15,
    "no. 200": 0.075, "200": 0.075,
}

#: Truth keys that say WHERE a specimen is or WHAT it is called rather than
#: what was measured. They are scored as the link, not as index values.
_NOT_INDEX = {
    "investigation_id", "sample_id", "depth_top", "depth_bottom", "n",
    "specimen", "stage", "trial", "trialno", "readings_count", "page",
    "figure", "job_no", "job_number", "test_id", "lab_sample_id",
    "note", "curve_digitised", "curves_digitised", "bracketed",
}

#: A truth key whose value is a number the sheet prints with no unit in the
#: key, but whose unit the trade fixes anyway. Only the grading's D-values:
#: every laboratory on earth prints them in millimetres and none of them
#: writes the unit into the column head.
_UNITLESS_BY_CONVENTION: Dict[str, str] = {
    "d10": "mm", "d15": "mm", "d30": "mm", "d50": "mm", "d60": "mm",
    "d85": "mm", "d90": "mm", "d100": "mm",
}

#: What a key has to look like for its pairs to be scored as a CURVE. A truth
#: file carries lists of pairs that are not results -- a compaction sheet's
#: raw weighing rows, a shear box's dial readings -- and scoring those would
#: mark a reader down for not recording numbers the record has no shape for.
#: These are the families the record does have a shape for.
_CURVE_FAMILY: Tuple[str, ...] = (
    "curve", "points", "passing", "finer", "rebound", "envelope",
    "stress_strain", "flow", "compaction", "consolidation",
)


# ---------------------------------------------------------------------------
# reading a truth file's own spelling
# ---------------------------------------------------------------------------

def _number(value: Any) -> Optional[float]:
    """The first number in a truth value, however the hand wrote it."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _split_key(key: str, stem: str) -> Optional[str]:
    """The unit a key means for this stem, or None when it is not this key."""
    if key == stem:
        return ""
    if not key.startswith(stem + "_"):
        return None
    return UNIT_SUFFIX.get(key[len(stem) + 1:].lower())


def find(block: Dict[str, Any], *stems: str) -> Tuple[Any, str, str]:
    """``(value, unit, key)`` for the first stem this block carries.

    The stems are tried in order and, for each, every key of the block, so
    ``find(t, "dry_unit_weight", "dry_density")`` answers a sheet that wrote
    ``dry_unit_weight_pcf`` and one that wrote ``dry_density_g_cm3`` with the
    same call and the right unit for each.
    """
    for stem in stems:
        for key, value in block.items():
            unit = _split_key(key, stem)
            if unit is None or value is None:
                continue
            return value, unit, key
    return None, "", ""


def number(block: Dict[str, Any], *stems: str) -> Optional[float]:
    """A plain number: a percentage, a ratio, a count."""
    value, _unit, _key = find(block, *stems)
    return _number(value)


def quantity(block: Dict[str, Any], *stems: str) -> Optional[Quantity]:
    """A value with the unit its key names, or None."""
    value, unit, _key = find(block, *stems)
    got = _number(value)
    if got is None or not unit:
        return None
    return Quantity(value=got, unit=unit)


def reported(block: Dict[str, Any], *stems: str) -> Any:
    """A number, a number with a unit, or the words the sheet printed.

    ``"<10"`` comes back as the string. It is not the number ten, and this is
    the one place a reader of these files could quietly make it one.
    """
    value, unit, _key = find(block, *stems)
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
        if value is None:
            return None
    if isinstance(value, str):
        text = value.strip()
        try:
            return (Quantity(value=float(text), unit=unit) if unit
                    else float(text))
        except ValueError:
            return text
    if isinstance(value, (int, float)):
        return (Quantity(value=float(value), unit=unit) if unit
                else float(value))
    return str(value)


def depth_unit_of(truth: Dict[str, Any]) -> str:
    """``ft`` or ``m``, out of whatever the hand wrote in ``depth_unit``."""
    raw = str(truth.get("depth_unit") or "").strip().lower()
    for candidate in ("ft", "m"):
        if raw == candidate or raw.startswith(candidate + " "):
            return candidate
    return ""


def report_of(truth: Dict[str, Any]) -> str:
    """``R36`` out of ``atterberg__R36_p52``."""
    parts = str(truth.get("id") or "").split("__")
    return parts[-1].split("_")[0] if parts else ""


def pages_of(truth: Dict[str, Any]) -> List[int]:
    """The 0-based page the sheet is on, out of its id."""
    match = re.search(r"_p(\d+)$", str(truth.get("id") or ""))
    return [int(match.group(1))] if match else []


def sieve_points(mapping: Dict[str, Any], keyed_mm: bool
                 ) -> List[Tuple[Optional[float], float, str]]:
    """``(size in mm, percent passing, the sieve as named)`` for a grading.

    ``keyed_mm`` says the keys are openings in millimetres rather than sieve
    designations. A designation gets its opening from :data:`SIEVE_MM` when
    the table knows it, and none at all when it does not.
    """
    out: List[Tuple[Optional[float], float, str]] = []
    for key, value in mapping.items():
        percent = _number(value)
        if percent is None:
            continue
        if keyed_mm:
            size = _number(key)
        else:
            size = SIEVE_MM.get(str(key).strip().lower())
        out.append((size, max(0.0, min(100.0, percent)), str(key)))
    out.sort(key=lambda p: -(p[0] if p[0] is not None else 0.0))
    return out


def grading_in(block: Dict[str, Any]
               ) -> List[Tuple[Optional[float], float, str]]:
    """The grading curve a block carries, under whichever key it used."""
    for key in ("percent_passing_mm", "passing_mm"):
        if isinstance(block.get(key), dict):
            return sieve_points(block[key], keyed_mm=True)
    for key in ("percent_finer", "percent_passing", "passing", "sieve"):
        if isinstance(block.get(key), dict):
            return sieve_points(block[key], keyed_mm=False)
    return []


def _trailing_unit(text: str) -> str:
    """The unit at the end of a key fragment, or ``''``."""
    parts = text.split("_")
    for n in (3, 2, 1):
        if len(parts) >= n:
            candidate = "_".join(parts[-n:]).lower()
            if candidate in UNIT_SUFFIX:
                return UNIT_SUFFIX[candidate]
    return ""


def curve_in(block: Dict[str, Any], *stems: str
             ) -> Tuple[List[Tuple[float, float]], str, str, str]:
    """``(points, x unit, y unit, the key)`` for a curve this block carries.

    The hand wrote both units into the one key, either side of ``_vs_``
    (``curve_pressure_psf_vs_axial_strain_pct``), so both are read off it.
    """
    for key, value in block.items():
        if not isinstance(value, list) or not value:
            continue
        lowered = key.lower()
        if stems and not any(stem in lowered for stem in stems):
            continue
        if not all(isinstance(item, (list, tuple)) and len(item) >= 2
                   for item in value):
            continue
        left, _, right = lowered.partition("_vs_")
        pairs = [(float(item[0]), float(item[1])) for item in value]
        # A key that names one unit means it for BOTH axes: a shear box's
        # points_normal_ton_ft2_vs_shear is tons per square foot each way,
        # and reading the y as dimensionless would put the envelope out by
        # the size of the unit.
        x_unit = _trailing_unit(left)
        return pairs, x_unit, _trailing_unit(right) or x_unit, key
    return [], "", "", ""


def _tolerance_of(block: Dict[str, Any], truth: Dict[str, Any],
                  values: Sequence[float]) -> float:
    """What a curve on this sheet is judged at, in the curve's own unit.

    The truth file states it where the hand thought the plot deserved one
    (``tolerance_strain_pct``, ``tolerance_kPa``, ``tolerance_psf``); where it
    does not, two per cent of the values' own span, which is the plan's
    figure for a digitised point.
    """
    for source in (block, truth):
        for key, value in source.items():
            if not key.startswith("tolerance"):
                continue
            got = _number(value)
            if got is not None:
                return abs(got)
    if not values:
        return 0.0
    span = max(values) - min(values)
    return max(abs(span) * CURVE_SPAN_FRACTION, 1e-9)


# ---------------------------------------------------------------------------
# what a sheet should have produced
# ---------------------------------------------------------------------------

@dataclass
class Expectation:
    """One specimen's worth of truth, in the shape a score is computed from."""

    label: str
    kind: str
    investigation_id: str
    depth_m: Optional[float]
    depth_printed: str = ""
    #: ``name -> (value in SI, the SI unit, the value as printed)``.
    index: Dict[str, Tuple[float, str, float]] = field(default_factory=dict)
    #: ``(size in mm or None, percent passing, the sieve as named)``.
    series: List[Tuple[Optional[float], float, str]] = field(
        default_factory=list)
    #: ``(name, [(x, y) in SI], tolerance on y in SI, x unit, y unit)``.
    curves: List[Tuple[str, List[Tuple[float, float]], float, str, str]] = \
        field(default_factory=list)

    @property
    def n_values(self) -> int:
        return (len(self.index) + len(self.series)
                + sum(len(points) for _n, points, _t, _x, _y in self.curves))


def _si(value: Any, unit: str) -> Optional[Tuple[float, str]]:
    got = to_si(value, unit)
    return got if got is not None else None


def split_unit(key: str) -> Tuple[str, str]:
    """``(stem, unit)`` for a truth key, taking the LONGEST unit that fits.

    ``bulk_density_g_cm3`` is a density in grams per cubic centimetre, not a
    ``bulk_density_g`` in cubic centimetres, and ``resistivity_kohm_cm`` is
    kilohm-centimetres and not centimetres. Splitting on the last underscore
    gets both wrong and gets them wrong QUIETLY -- as a value a thousand
    times out -- so the longest suffix that names a unit wins.
    """
    parts = key.split("_")
    for n in (3, 2, 1):
        if len(parts) > n:
            unit = UNIT_SUFFIX.get("_".join(parts[-n:]).lower())
            if unit is not None:
                return "_".join(parts[:-n]), unit
    convention = _UNITLESS_BY_CONVENTION.get(key.lower())
    if convention is not None:
        return key, convention
    return key, ""


def _printed_number(value: Any) -> Optional[float]:
    """A value that IS a number, rather than one that contains one.

    ``0.32`` is a number. ``"8/6/2021"`` is a date, ``"Estrato No. 7-B"`` is a
    description and ``"1 end"`` is a note, and a scorer that took the first
    digits out of each would demand that a reader record the eight, the seven
    and the one as measurements.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lstrip("<>").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _flatten(block: Dict[str, Any], prefix: str = "",
             depth: int = 0) -> Dict[str, Any]:
    """A truth block's scalars, nested ones included, under joined names.

    A triaxial prints its specimens as a LIST of blocks and a 1991 sheet
    prints four nested stages; a scorer that looked only at the top level
    would silently ask nothing of either, and a sheet whose every value is
    nested would score a tidy zero out of zero. Everything is flattened to
    ``specimens.1.peak_deviator_kPa`` so every printed number is asked for
    exactly once.
    """
    out: Dict[str, Any] = {}
    if depth > 3:
        return out
    for key, value in block.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}.", depth + 1))
        elif isinstance(value, list):
            if all(isinstance(item, dict) for item in value) and value:
                for i, item in enumerate(value, start=1):
                    out.update(_flatten(item, f"{name}.{i}.", depth + 1))
        else:
            out[name] = value
    return out


def _index_of(block: Dict[str, Any]) -> Dict[str, Tuple[float, str, float]]:
    """``stem -> (value in SI, the SI unit, the value as printed)``.

    Generic on purpose. The hand wrote each sheet's unit into its key, so a
    value converts without anyone knowing what the sheet was -- and a scorer
    that had to list the fields of each kind would quietly stop scoring a
    field nobody remembered to add.

    The printed value is carried alongside the converted one because a reader
    may keep a value the record has no typed field for, as printed, in the
    test's ``fields``. That is a recovery and is scored as one.
    """
    out: Dict[str, Tuple[float, str, float]] = {}
    for key, value in _flatten(block).items():
        if value is None:
            continue
        stem, unit = split_unit(key)
        leaf = stem.rsplit(".", 1)[-1].lower()
        if leaf in _NOT_INDEX or key.rsplit(".", 1)[-1].lower() in _NOT_INDEX:
            continue
        if key.lower().startswith("tolerance") or "date" in key.lower():
            continue
        got = _printed_number(value)
        if got is None:
            continue                 # words, not a number: not an index value
        converted = _si(got, unit)
        if converted is None:
            continue                 # a unit nothing can convert
        out[stem] = (converted[0], converted[1], got)
    return out


def expectations_for(truth: Dict[str, Any]) -> List[Expectation]:
    """One :class:`Expectation` per specimen the truth sheet describes."""
    unit = depth_unit_of(truth)
    kind = str(truth.get("kind") or "other")
    blocks: List[Tuple[str, Dict[str, Any]]] = []
    if truth.get("rows"):
        blocks = [(f"row {i + 1}", row)
                  for i, row in enumerate(truth["rows"])]
    else:
        blocks = [(f"test {i + 1}", block)
                  for i, block in enumerate(truth.get("tests") or [])]
    out: List[Expectation] = []
    for label, block in blocks:
        depth = _number(block.get("depth_top"))
        in_metres = None
        if depth is not None and unit:
            converted = _si(depth, unit)
            in_metres = converted[0] if converted else None
        curves: List[Tuple[str, List[Tuple[float, float]], float, str,
                           str]] = []
        for key, value in block.items():
            if not isinstance(value, list) or not value:
                continue
            # A curve is a list of PAIRS under a name the record has a shape
            # for. A list of nine-column weighing rows is not a curve, and a
            # box of dial readings is not a result.
            if not all(isinstance(item, (list, tuple)) and len(item) == 2
                       for item in value):
                continue
            if not any(family in key.lower() for family in _CURVE_FAMILY):
                continue
            points, x_unit, y_unit, _key = curve_in({key: value})
            if not points:
                continue
            xs = [_si(x, x_unit) for x, _y in points]
            ys = [_si(y, y_unit) for _x, y in points]
            if any(v is None for v in xs) or any(v is None for v in ys):
                continue
            si_points = [(xs[i][0], ys[i][0]) for i in range(len(points))]
            tolerance = _tolerance_of(block, truth,
                                      [y for _x, y in points])
            scale = _si(tolerance, y_unit)
            curves.append((key, si_points,
                           abs(scale[0]) if scale else tolerance,
                           xs[0][1] if xs else "", ys[0][1] if ys else ""))
        out.append(Expectation(
            label=label, kind=kind,
            investigation_id=str(block.get("investigation_id") or "").strip(),
            depth_m=in_metres,
            depth_printed=f"{depth} {unit}" if depth is not None else "",
            index=_index_of(block),
            series=grading_in(block),
            curves=curves))
    return out


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
class LabScore:
    """One sheet, scored. ``stage`` is ``tables`` or ``record``."""

    sheet_id: str
    report: str
    kind: str
    stage: str
    scores: Dict[str, Score] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    model_calls: int = 0
    tool_calls: int = 0
    unresolved: int = 0
    changes: int = 0
    kinds_read: List[str] = field(default_factory=list)
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
            "sheet_id": self.sheet_id, "report": self.report,
            "kind": self.kind, "stage": self.stage,
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
            "overall": self.total.to_dict(),
            "cost": dict(self.cost), "model_calls": self.model_calls,
            "tool_calls": self.tool_calls, "unresolved": self.unresolved,
            "changes": self.changes, "kinds_read": list(self.kinds_read),
            "error": self.error,
        }


def _close(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= max(tol, abs(float(b)) * 1e-9)


def _blank(sheet_id: str, kind: str, stage: str,
           error: Optional[str] = None) -> LabScore:
    return LabScore(sheet_id=sheet_id, report=sheet_id.split("__")[-1]
                    .split("_")[0], kind=kind, stage=stage, error=error)


# ---------------------------------------------------------------------------
# BEFORE: what the page's own tables hold
# ---------------------------------------------------------------------------

def table_numbers(doc: Any, pages: Sequence[int]) -> List[float]:
    """Every number in every table planlens detected on these pages.

    The deterministic baseline. No meaning is attached to any of them; the
    question a table can answer is only whether the number is on the page.
    """
    out: List[float] = []
    for page in pages:
        try:
            content = doc.page(page)
        except Exception:                      # a page that will not read
            continue
        for table in content.tables:
            grid = ([table.header] if table.header else []) + table.rows
            for row in grid:
                for cell in row or ():
                    for token in re.findall(r"-?\d+(?:\.\d+)?",
                                            str(cell or "")):
                        out.append(float(token))
    return out


def score_tables(truth: Dict[str, Any], doc: Any,
                 pages: Sequence[int]) -> LabScore:
    """Score the page's detected tables against the truth, as a baseline.

    Only ``index``, ``series`` and ``curve`` are asked, and of each only "is
    this number on the page". ``kind`` and ``link`` are not scored at all: a
    table has no idea what test it is or which boring it belongs to, and
    crediting it with either would flatter the baseline into meaninglessness.
    """
    sheet_id = str(truth.get("id") or "")
    out = _blank(sheet_id, str(truth.get("kind") or ""), "tables")
    pool = table_numbers(doc, pages)
    for expect in expectations_for(truth):
        where = f"{expect.label}"
        for name, (value, _unit, printed) in sorted(expect.index.items()):
            out.score("index").add(
                any(_close(v, value, EXACT_TOL)
                    or _close(v, printed, EXACT_TOL) for v in pool),
                f"{where} {name}={value:.6g}")
        for _size, percent, sieve in expect.series:
            out.score("series").add(
                any(_close(v, percent, PASSING_TOL) for v in pool),
                f"{where} passing {sieve}={percent:g}")
        for name, points, tolerance, _xu, _yu in expect.curves:
            for x, y in points:
                out.score("curve").add(
                    any(_close(v, y, tolerance) for v in pool),
                    f"{where} {name} y={y:.6g}")
    return out


# ---------------------------------------------------------------------------
# AFTER: what the reader's records hold
# ---------------------------------------------------------------------------

def _tests_near(tests: Sequence[Any], expect: Expectation) -> List[Any]:
    """The records that belong to this specimen: same boring, same depth."""
    out = []
    for test in tests:
        if expect.investigation_id and \
                test.investigation_id.strip() != expect.investigation_id:
            continue
        if expect.depth_m is not None:
            got = test.depth_top.si_value if test.depth_top else None
            if got is None or abs(got - expect.depth_m) > DEPTH_TOL_M:
                continue
        out.append(test)
    return out


def _rows_near(tests: Sequence[Any], expect: Expectation) -> List[Any]:
    """The summary-table rows that belong to this specimen."""
    from report_ingest.model import SummaryTableResult

    out = []
    for test in tests:
        if not isinstance(test.result, SummaryTableResult):
            continue
        for row in test.result.rows:
            if expect.investigation_id and \
                    row.investigation_id.strip() != expect.investigation_id:
                continue
            if expect.depth_m is not None:
                got = row.depth_top.si_value if row.depth_top else None
                if got is None or abs(got - expect.depth_m) > DEPTH_TOL_M:
                    continue
            out.append(row)
    return out


def _pool_for(tests: Sequence[Any], expect: Expectation) -> List[float]:
    """Every number the record holds for this specimen.

    Both the CONVERTED numbers of its typed result and the numbers a reader
    kept AS PRINTED in the test's own ``fields`` or a row's ``other`` -- the
    slots that exist so a value with no typed home is not lost. A value found
    in either has been recovered, and the scorecard says so.
    """
    rows = _rows_near(tests, expect)
    near = _tests_near(tests, expect)
    holders: List[Any] = list(rows)
    if not holders:
        holders = [test.result for test in near if test.result is not None]
    pool: List[float] = []
    for holder in holders:
        pool.extend(value for _path, value, _unit in si_numbers(holder))
    texts: List[str] = []
    for test in near:
        texts.extend(str(v) for v in test.fields.values())
    for row in rows:
        texts.extend(str(v) for _name, v in row.other)
    for text in texts:
        for token in re.findall(r"-?\d+(?:\.\d+)?", text):
            pool.append(float(token))
    return pool


def _sieves_for(tests: Sequence[Any], expect: Expectation
                ) -> List[Tuple[Optional[float], float, str]]:
    """Every grading point the record holds for this specimen."""
    from report_ingest.model import GradationResult, SummaryTableResult

    out: List[Tuple[Optional[float], float, str]] = []
    for row in _rows_near(tests, expect):
        out.extend((p.size.si_value * 1000.0 if p.size is not None
                    and p.size.si_value is not None else None,
                    p.percent_passing, p.sieve) for p in row.percent_passing)
    for test in _tests_near(tests, expect):
        result = test.result
        if isinstance(result, GradationResult):
            out.extend((p.size.si_value * 1000.0 if p.size is not None
                        and p.size.si_value is not None else None,
                        p.percent_passing, p.sieve)
                       for p in result.percent_passing)
        elif isinstance(result, SummaryTableResult):
            continue
    return out


def score_record(truth: Dict[str, Any], tests: Sequence[Any]) -> LabScore:
    """Score the reader's records against the truth, metric by metric."""
    sheet_id = str(truth.get("id") or "")
    kind = str(truth.get("kind") or "")
    out = _blank(sheet_id, kind, "record")
    out.kinds_read = sorted({t.kind for t in tests})
    got_kinds = {t.kind for t in tests}
    for expect in expectations_for(truth):
        where = expect.label
        out.score("kind").add(expect.kind in got_kinds,
                              f"{where} kind {expect.kind} -> "
                              f"{sorted(got_kinds) or 'nothing'}")
        if expect.investigation_id or expect.depth_m is not None:
            linked = bool(_tests_near(tests, expect)
                          or _rows_near(tests, expect))
            out.score("link").add(
                linked,
                f"{where} {expect.investigation_id or '(no boring)'} at "
                f"{expect.depth_printed or '(no depth)'}")
        pool = _pool_for(tests, expect)
        for name, (value, _unit, printed) in sorted(expect.index.items()):
            out.score("index").add(
                any(_close(v, value, EXACT_TOL)
                    or _close(v, printed, EXACT_TOL) for v in pool),
                f"{where} {name}={value:.6g}")
        record_sieves = _sieves_for(tests, expect)
        for size, percent, sieve in expect.series:
            ok = any(_close(p, percent, PASSING_TOL)
                     and (size is None or s is None
                          or _close(s, size, max(0.01, size * 0.02)))
                     for s, p, _name in record_sieves)
            out.score("series").add(ok,
                                    f"{where} passing {sieve}={percent:g}")
        for name, points, tolerance, _xu, _yu in expect.curves:
            for x, y in points:
                out.score("curve").add(
                    any(_close(v, y, tolerance) for v in pool),
                    f"{where} {name} at x={x:.6g}: y={y:.6g}")
    return out


# ---------------------------------------------------------------------------
# both, on one sheet
# ---------------------------------------------------------------------------

def score_one_sheet(truth: Dict[str, Any], doc: Any, engine: Any, *,
                    budget: int = 4, report_id: str = "",
                    hint_kind: Optional[str] = None
                    ) -> Tuple[LabScore, LabScore]:
    """``(before, after)`` for one sheet: the tables, then the reader.

    The tables are scored from the same open document the reader is handed,
    so the difference between the two columns is the model and nothing else.
    """
    from report_ingest.lab_reader import read_lab_sheet

    sheet_id = str(truth.get("id") or "")
    kind = str(truth.get("kind") or "")
    pages = pages_of(truth)
    report = report_id or report_of(truth)
    if not pages:
        error = "the truth file's id does not name a page"
        return (_blank(sheet_id, kind, "tables", error),
                _blank(sheet_id, kind, "record", error))
    try:
        before = score_tables(truth, doc, pages)
    except Exception as exc:                      # a page that will not read
        error = f"{type(exc).__name__}: {exc}"
        return (_blank(sheet_id, kind, "tables", error),
                _blank(sheet_id, kind, "record", error))
    try:
        result = read_lab_sheet(doc, pages, engine, budget=budget,
                                hint_kind=hint_kind, report_id=report)
    except Exception as exc:                      # a model call that failed
        return before, _blank(sheet_id, kind, "record",
                              f"{type(exc).__name__}: {exc}")
    after = score_record(truth, result.tests)
    after.cost = dict(result.cost)
    after.model_calls = result.model_calls
    after.tool_calls = result.tool_calls
    after.unresolved = len(result.unresolved)
    after.changes = len(result.changes)
    after.warnings = list(result.warnings)
    return before, after

"""WP2a: score ``planlens.document.loggrid.log_grid`` against hand-truthed logs.

The truth lives in the gitignored ``raw/truth/logs/<ID>_p<page>.json``, one
file per log, transcribed by hand from the rendered page. This script runs
``log_grid`` on the pages each truth file names and scores five things:

a. **the ruler** — found at all, and in the unit the truth states;
b. **samples** — every truth sample's blow record and N value found as a cell
   in a blows-family column within 0.15 m of the truth depth;
c. **layers** — every truth layer top within 0.3 m of a grid layer top;
d. **index values** — water content, dry unit weight, Atterberg limits and
   fines found as a cell within 0.15 m of the truth depth;
e. **fields** — the header key-values recovered.

Plus a precision proxy: how many of the cells the grid emitted match nothing
in the truth at all.

PRIVACY. Reports are IDs. Nothing here prints a project name, a firm or a
file name, and the ledger it appends to is inside the gitignored raw folder's
sibling — ``MEASUREMENTS.md`` — which IS committed, so only IDs and numbers
go into it.

Run::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp2a_loggrid
    ... --append          # also write the scorecard into MEASUREMENTS.md
    ... --only R36,R37    # a subset
    ... --detail          # list every miss on the OPEN logs
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import corpus as C

TRUTH_DIR = C.RAW_DIR / "truth" / "logs"
OPEN_FILE = TRUTH_DIR / "OPEN.txt"
LEDGER = (C.RAW_DIR.parent / "MEASUREMENTS.md")

#: The open set: the logs the rules may be tuned on. Everything else is
#: blind — scored, never looked at.
DEFAULT_OPEN = ("R36", "R37", "R06", "R07", "R15", "R28")

#: Tolerances, from the plan. Depths are compared in metres whatever the log
#: prints, so one number means the same thing on a metric and an imperial log.
SAMPLE_TOL_M = 0.15
LAYER_TOL_M = 0.30
FT_PER_M = 3.280839895

#: How long a driven sample is taken to be when the truth states only where
#: it started. A split-spoon drive is 18 in (0.46 m) on an imperial log and
#: 0.45 m on a metric one, so one number covers both.
DRIVEN_LENGTH_M = 0.46

#: Which canonical column names count as the right home for each truth value.
#: A form that heads one column "SAMPLING DATA" or "FIELD TEST RESULTS" and
#: prints the blow record, the N value and the recovery inside it is not
#: wrong, so the blows family is a family.
COLUMN_FAMILY = {
    "blows": ("blows", "n_value", "sample_id", "tests"),
    "n": ("n_value", "blows", "sample_id", "tests"),
    "wc": ("water_content", "tests"),
    "duw": ("dry_unit_weight", "tests"),
    "ll": ("liquid_limit", "plasticity_index", "tests"),
    "pl": ("plastic_limit", "plasticity_index", "tests"),
    "pi": ("plasticity_index", "tests"),
    "fines": ("fines", "tests"),
    "qu": ("qu", "tests"),
    "recovery": ("recovery", "sample_id", "tests"),
}

#: Truth field keys that the grid's canonical keys answer. The truth was
#: written to the record's vocabulary, the grid to the page's; this is the
#: join, and it is a table on purpose so it is easy to argue with.
FIELD_MAP = {
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_m(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if unit == "ft":
        return float(value) / FT_PER_M
    return float(value)


def _numbers(text: str) -> Tuple[float, ...]:
    from planlens.document.loggrid import numbers_in
    return numbers_in(text)


def _close(a: float, b: float) -> bool:
    """Is a printed number the truth's number? Two per cent or 0.5, whichever
    is larger — a log prints 12 for 11.8 and 116 for 115.6."""
    return abs(a - b) <= max(0.5, 0.02 * abs(b))


def _blows_match(cell_numbers: Sequence[Any],
                 blows: Sequence[Any]) -> bool:
    """Do the cell's numbers contain the truth's blow record, in order?

    A refusal is printed as text ("50/4\"") and transcribed as text; both
    sides are reduced to the numbers they contain, so 50 and 4 have to be
    there in that order and nothing is asked about the notation.
    """
    want: List[float] = []
    for b in blows:
        if isinstance(b, (int, float)):
            want.append(float(b))
        else:
            want.extend(_numbers(str(b)))
    if not want:
        return False
    got = list(cell_numbers)
    if len(want) > len(got):
        return False
    for i in range(len(got) - len(want) + 1):
        if all(_close(got[i + j], want[j]) for j in range(len(want))):
            return True
    return False


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


@dataclass
class LogScore:
    log_id: str
    report: str
    ruler: Score = field(default_factory=Score)
    unit: Score = field(default_factory=Score)
    samples: Score = field(default_factory=Score)
    layers: Score = field(default_factory=Score)
    index: Score = field(default_factory=Score)
    fields: Score = field(default_factory=Score)
    n_cells: int = 0
    n_unmatched_cells: int = 0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# scoring one log
# ---------------------------------------------------------------------------

def score_one(truth: Dict[str, Any], di: str = "auto") -> LogScore:
    from planlens.document.loggrid import log_grid

    log_id = truth["id"]
    report = log_id.split("_")[0]
    out = LogScore(log_id=log_id, report=report)
    try:
        doc = C.open_report(report, di=di, warn=False)
        grid = log_grid(doc, list(truth["pages"]))
    except Exception as exc:                      # pragma: no cover - harness
        out.error = f"{type(exc).__name__}: {exc}"
        return out

    out.warnings = list(grid.warnings)
    truth_unit = truth.get("depth_unit")

    def truth_m(value):
        return _to_m(value, truth_unit)

    def cell_m(value):
        return _to_m(value, grid.unit)

    # (a) the ruler, and its unit
    out.ruler.add(bool(grid.rulers), "no ruler on any page")
    out.unit.add(grid.unit == truth_unit,
                 f"unit {grid.unit!r} not {truth_unit!r}")

    # index the cells once
    cells = [c for c in grid.rows if c.depth is not None]
    out.n_cells = len(grid.rows)
    matched = set()

    def find(value, window, family, blows=None) -> Optional[int]:
        """The first cell of the right family standing in a depth window.

        A sample is an INTERVAL, not a point, and a form prints its blow
        record against the middle of that interval as often as against its
        top. So the window is the sample interval widened by the tolerance,
        and a cell counts when its box falls inside it. Where the truth
        states only the top, the window is the top plus a driven sample.
        """
        lo, hi = window
        best = None
        run: List[Tuple[float, float, int, int, float]] = []
        for i, cell in enumerate(cells):
            col = grid.column(cell.column_id)
            names = col.names if col else ()
            if family and not (set(names) & set(family)):
                continue
            dm = cell_m(cell.depth)
            top = cell_m(cell.depth_top)
            bottom = cell_m(cell.depth_bottom)
            near = dm is not None and lo <= dm <= hi
            if not near and top is not None and bottom is not None:
                near = top <= hi and bottom >= lo
            if not near:
                continue
            if blows is not None:
                for order, n in enumerate(cell.numbers):
                    # order keeps the drives in the order the cell prints
                    # them: sorting on the value would turn 4+11+10+10 into
                    # 4, 10, 10, 11 and no record would ever match.
                    run.append((cell.depth or 0.0, cell.bbox[0], order, i, n))
                continue
            if any(_close(n, value) for n in cell.numbers):
                if best is None:
                    best = i
        if blows is not None:
            # Some forms print a blow record as one cell ("5-9-12") and some
            # print each drive on its own line down the sampler column. Both
            # are the same record at the same depth, so the numbers standing
            # in the window are read in page order and the record is looked
            # for in THEM, not inside any one cell.
            run.sort()
            if _blows_match([n for *_rest, n in run], blows):
                return run[0][3] if run else None
            return None
        return best

    def window_of(sample):
        top = sample.get("top")
        if top is None:
            return None
        bottom = sample.get("bottom")
        lo = truth_m(top)
        hi = truth_m(bottom) if bottom is not None else lo + DRIVEN_LENGTH_M
        return (lo - SAMPLE_TOL_M, hi + SAMPLE_TOL_M)

    # (b) samples: the blow record and the N value
    for sample in truth.get("samples") or ():
        depth = sample.get("top")
        window = window_of(sample)
        if window is None:
            continue
        blows = sample.get("blows")
        if blows:
            hit = find(None, window, COLUMN_FAMILY["blows"], blows=blows)
            out.samples.add(hit is not None,
                            f"blows {blows} at {depth} {truth_unit}")
            if hit is not None:
                matched.add(hit)
        if sample.get("n") is not None:
            hit = find(sample["n"], window, COLUMN_FAMILY["n"])
            if hit is None and blows:
                # Half the templates print the N value and half print only
                # the drives that define it (N is the second and third six
                # inches). The grid does no arithmetic by design, so the
                # drives standing at the right depth ARE the N value being
                # present, and the scorer says so rather than counting a
                # miss the reader could not have avoided.
                hit = find(None, window, COLUMN_FAMILY["blows"], blows=blows)
                drives = [b for b in blows if isinstance(b, (int, float))]
                if not (hit is not None and len(drives) >= 3
                        and _close(drives[1] + drives[2],
                                   float(sample["n"]))):
                    hit = None
            out.samples.add(hit is not None,
                            f"N={sample['n']} at {depth} {truth_unit}")
            if hit is not None:
                matched.add(hit)

    # (d) index values on the log face
    for sample in truth.get("samples") or ():
        depth = sample.get("top")
        window = window_of(sample)
        if window is None:
            continue
        for key in ("wc", "duw", "ll", "pl", "pi", "fines", "qu"):
            value = sample.get(key)
            if value is None:
                continue
            hit = find(value, window, COLUMN_FAMILY[key])
            out.index.add(hit is not None,
                          f"{key}={value} at {depth} {truth_unit}")
            if hit is not None:
                matched.add(hit)

    # (c) layer tops
    tops = [ly.top for ly in grid.layers if ly.top is not None]
    tops_m = [cell_m(t) for t in tops]
    for layer in truth.get("layers") or ():
        top = layer.get("top")
        if top is None:
            continue
        want = truth_m(top)
        ok = any(t is not None and abs(t - want) <= LAYER_TOL_M
                 for t in tops_m)
        out.layers.add(ok, f"layer top {top} {truth_unit}")

    # (e) header fields
    for key, value in (truth.get("fields") or {}).items():
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
                a = "".join(ch.lower() for ch in str(value) if ch.isalnum())
                b = "".join(ch.lower() for ch in got if ch.isalnum())
                ok = bool(a) and (a in b or b in a)
        out.fields.add(ok, f"{key}={value!r} -> {got!r}")

    # precision proxy: cells that match nothing the truth states
    out.n_unmatched_cells = len(cells) - len(matched)
    return out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def open_reports() -> Tuple[str, ...]:
    if OPEN_FILE.is_file():
        names = [ln.strip() for ln in
                 OPEN_FILE.read_text(encoding="utf-8").splitlines()]
        found = tuple(n for n in names if n)
        if found:
            return found
    return DEFAULT_OPEN


def load_truth(only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    if not TRUTH_DIR.is_dir():
        raise FileNotFoundError(
            f"no hand-truthed logs yet ({TRUTH_DIR} is gitignored and exists "
            f"only where the owner put it)")
    out = []
    for path in sorted(TRUTH_DIR.glob("*.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        if only and truth["id"].split("_")[0] not in only:
            continue
        out.append(truth)
    return out


def _pct(score: Score) -> str:
    if not score.total:
        return "     -"
    return f"{score.rate:5.0%} {score.found}/{score.total}"


def report(scores: Sequence[LogScore], openset: Sequence[str],
           detail: bool = False) -> str:
    lines: List[str] = []
    header = (f"{'log':<12}{'set':<7}{'ruler':>7}{'unit':>7}"
              f"{'samples':>13}{'layers':>13}{'index':>13}{'fields':>13}"
              f"{'cells':>7}{'unmatched':>10}")
    lines.append(header)
    lines.append("-" * len(header))
    groups: Dict[str, List[LogScore]] = {"open": [], "blind": []}
    for s in scores:
        groups["open" if s.report in openset else "blind"].append(s)
    for name in ("open", "blind"):
        for s in groups[name]:
            if s.error:
                lines.append(f"{s.log_id:<12}{name:<7}ERROR {s.error[:60]}")
                continue
            lines.append(
                f"{s.log_id:<12}{name:<7}"
                f"{'yes' if s.ruler.found else 'NO':>7}"
                f"{'yes' if s.unit.found else 'NO':>7}"
                f"{_pct(s.samples):>13}{_pct(s.layers):>13}"
                f"{_pct(s.index):>13}{_pct(s.fields):>13}"
                f"{s.n_cells:>7}{s.n_unmatched_cells:>10}")
        if groups[name]:
            total = LogScore(log_id=f"ALL {name}", report="")
            for s in groups[name]:
                if s.error:
                    continue
                total.ruler += s.ruler
                total.unit += s.unit
                total.samples += s.samples
                total.layers += s.layers
                total.index += s.index
                total.fields += s.fields
                total.n_cells += s.n_cells
                total.n_unmatched_cells += s.n_unmatched_cells
            lines.append(
                f"{'ALL':<12}{name:<7}"
                f"{_pct(total.ruler):>7}{_pct(total.unit):>7}"
                f"{_pct(total.samples):>13}{_pct(total.layers):>13}"
                f"{_pct(total.index):>13}{_pct(total.fields):>13}"
                f"{total.n_cells:>7}{total.n_unmatched_cells:>10}"
                .replace("  ", " ", 0))
            lines.append("")
    warned = [s for s in scores if s.warnings]
    if warned:
        lines.append("warnings seen:")
        for s in warned:
            for w in s.warnings:
                lines.append(f"  {s.log_id}: {w}")
        lines.append("")
    if detail:
        lines.append("misses on the OPEN logs (the blind ones are not listed):")
        for s in scores:
            if s.report not in openset:
                continue
            for label, sc in (("sample", s.samples), ("layer", s.layers),
                              ("index", s.index), ("field", s.fields)):
                for miss in sc.misses:
                    lines.append(f"  {s.log_id} {label}: {miss}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated report IDs")
    ap.add_argument("--di", default="auto", choices=("auto", "all", "none"))
    ap.add_argument("--detail", action="store_true",
                    help="list every miss on the OPEN logs")
    ap.add_argument("--append", action="store_true",
                    help="append the scorecard to MEASUREMENTS.md")
    args = ap.parse_args(argv)

    only = tuple(x.strip() for x in args.only.split(",") if x.strip()) or None
    openset = open_reports()
    truths = load_truth(only)
    if not truths:
        print("no truth files matched")
        return 1
    scores = [score_one(t, di=args.di) for t in truths]
    text = report(scores, openset, detail=args.detail)
    print(text)
    if args.append:
        import datetime
        stamp = datetime.date.today().isoformat()
        block = (f"\n### log_grid scorecard, {stamp}\n\n"
                 f"Tolerances: sample and index values {SAMPLE_TOL_M} m, "
                 f"layer tops {LAYER_TOL_M} m; depths compared in metres "
                 f"whatever the log prints. Open set = "
                 f"{', '.join(openset)}.\n\n```\n{text}\n```\n")
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(block)
        print(f"\nappended to {LEDGER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

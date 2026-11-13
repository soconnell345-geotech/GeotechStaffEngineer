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
  decision rather than a defect. That credit is given only for a record of
  as many increments as a printed rule is held for (``N_RULES``: three,
  and four, the three and the occasional fourth).
  For any other number the scorer works out nothing: where the drives stand
  at that depth with no N printed, the answer's N is SET ASIDE
  (``LogScore.set_aside``, group ``n_no_rule``), counted and listed but in
  no total (BACKLOG RI-79);
* a blow record is compared on its COUNTS: several increments joined in one
  string are each read, and the penetration of a drive stopped short
  (``50/5"``, ``50/20``) is kept out of them and compared only where both
  sides state one;
* the blows of a driven sampler that is not a split spoon (the answer
  sheet's ``blows_kind: "sampler"``) are found on a drive record of their
  own at that sample's depth -- the record keeps every driven record in
  ``spt``, N or no N -- never among another test's increments;
* a header field the answer sheet strikes (``fields_struck``) is left out
  of the total;
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

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SAMPLE_TOL_M", "LAYER_TOL_M", "WATER_TOL_M", "DRIVEN_LENGTH_M",
    "COLUMN_FAMILY", "FIELD_MAP", "INDEX_KEYS", "METRICS",
    "INTERVAL_METRICS",
    "Score", "LogScore", "score_grid", "score_record", "score_one_log",
    "truth_investigations", "rescore_saved", "dump_investigations",
    "N_RULES", "SET_ASIDE", "rescore_run",
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

#: The interval groups (BACKLOG RI-103), scored only where an answer sheet's
#: layer carries their keys and NOT part of :data:`METRICS`, so
#: ``LogScore.total`` and every total built on it stay what they were:
#:
#: * ``interval_note``: each of a layer's ``notes`` (``{"depth": 7, "text":
#:   "wet below 7 ft"}``, ``depth`` left out where the log writes none) is
#:   found when the record's interval at that top holds a note with those
#:   words (:func:`_same_words`: equal after case, spacing and punctuation
#:   are set aside) and, where the sheet gives a depth, a depth within
#:   :data:`LAYER_TOL_M`; a sheet note with no depth is found only in a
#:   record note with none, as a depth the log never wrote is not a reading;
#: * ``line_kind``: ``line`` (the kind of the interval's top line as the
#:   legend prints it; ``""`` says the log says nothing, and then the record
#:   must say nothing) against ``Layer.top_line``, exact after whitespace
#:   and case;
#: * ``line_width``: ``line_full_width`` (true or false) against
#:   ``Layer.top_line_full_width``;
#: * ``stratum_label``: ``label`` (as printed; ``""`` says the log prints
#:   none for this interval) against ``Layer.stratum_label``, exact after
#:   whitespace and case, whatever the carried marks say;
#: * ``label_source``: ``label_source`` against
#:   ``Layer.stratum_label_source`` (:func:`_same_text`), where the label
#:   was found;
#: * ``label_carried``: ``label_carried`` (absent is false: printed against
#:   this interval) against ``Layer.stratum_label_carried``, where the label
#:   was found and is not empty: a label the record carries down that the
#:   sheet marks as printed, or the reverse, misses here and not on the
#:   label.
#:
#: The record's interval "at that top" is the one whose top is nearest the
#: sheet's, within :data:`LAYER_TOL_M`.
INTERVAL_METRICS: Tuple[str, ...] = (
    "interval_note", "line_kind", "line_width", "stratum_label",
    "label_source", "label_carried")


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


#: A length unit after a penetration, folded to one spelling: ``5"``,
#: ``5 in`` and ``5 inches`` are one unit.
_PEN_UNITS: Dict[str, str] = {
    "mm": "mm", "cm": "cm", "m": "m", "in": "in", "in.": "in", "ins": "in",
    "inch": "in", "inches": "in", '"': "in", "\u201d": "in", "\u2033": "in",
    "''": "in", "ft": "ft", "ft.": "ft", "'": "ft", "\u2032": "ft",
}
_PEN_UNIT_RE = (r"(?:mm|cm|m|inches|inch|ins|in\.?|ft\.?|\"|\u201d|\u2033|''"
                r"|'|\u2032)")
#: One increment inside a blow record: a count, and where the record says
#: so the penetration it was counted over -- ``21 for 60mm``, ``50/5"``,
#: ``50 blows for 20 mm``. A slash with NO unit after it is left to the
#: caller: ``3/6/8`` is three increments, a lone ``50/20`` a count over a
#: penetration.
_INCREMENT_RE = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)(?:\s*blows?)?"
    r"(?:\s*for\s*(\d+(?:\.\d+)?)\s*(" + _PEN_UNIT_RE + r")?(?![A-Za-z])"
    r"|\s*/\s*(\d+(?:\.\d+)?)\s*(" + _PEN_UNIT_RE + r")(?![A-Za-z]))?",
    re.I)
#: A whole entry that is one count over one penetration with no unit:
#: ``50/20``.
_LONE_OVER_RE = re.compile(
    r"\(?\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*\)?")


def _blow_entries(blows: Sequence[Any]
                  ) -> List[Tuple[float, Optional[float], str]]:
    """A blow record as ``(count, penetration, unit)`` per increment.

    An entry is a number, or a string as a log prints it (BACKLOG RI-79):

    * several increments in one string, however joined -- ``3,6``,
      ``3, 6, 8``, ``3/6/8``, ``3-6-8``, ``1,2/2,3,3,4`` -- are each an
      increment. (Read as one number, ``3,6`` gave its first count only.)
    * a count over a penetration -- ``50/5"``, ``50/20mm``, ``21 for
      60mm``, and a lone ``50/20`` with no unit -- is ONE increment whose
      count is the count. The penetration is kept beside it, never among
      the counts: ``report_ingest.model.SPT`` keeps such an entry as the
      string the log printed, "50 blows over 5 in, a drive stopped short of
      its increment, not the number fifty".

    The penetration is None, and the unit empty, where the entry gives
    none.
    """
    out: List[Tuple[float, Optional[float], str]] = []
    for entry in blows:
        if isinstance(entry, bool):
            continue
        if isinstance(entry, (int, float)):
            out.append((float(entry), None, ""))
            continue
        text = str(entry or "").strip()
        lone = _LONE_OVER_RE.fullmatch(text)
        if lone:
            out.append((float(lone.group(1)), float(lone.group(2)), ""))
            continue
        for found in _INCREMENT_RE.finditer(text):
            pen = found.group(2) or found.group(4)
            unit = (found.group(3) or found.group(5) or "").lower()
            out.append((float(found.group(1)),
                        float(pen) if pen is not None else None,
                        _PEN_UNITS.get(unit, unit)))
    return out


def _blow_numbers(blows: Sequence[Any]) -> List[float]:
    """The blow COUNT of every increment of a record, in order: the
    penetration of a drive stopped short is not a count
    (:func:`_blow_entries`)."""
    return [count for count, _pen, _unit in _blow_entries(blows)]


def _penetrations_agree(got: Sequence[Any], want: Sequence[Any]) -> bool:
    """Do two blow records agree on the penetrations they both state?

    Only where BOTH state one in the same unit (or both with none): the
    answer's ``50/5"`` against a record's ``50/3"`` is a different drive.
    A side that states no penetration (an answer that carries counts only)
    is not held against the other.
    """
    mine = [(p, u) for _c, p, u in _blow_entries(got) if p is not None]
    theirs = [(p, u) for _c, p, u in _blow_entries(want) if p is not None]
    if not mine or not theirs:
        return True
    if {u for _p, u in mine} != {u for _p, u in theirs}:
        return True                     # different units: not comparable
    return _as_multiset([p for p, _u in mine], [p for p, _u in theirs])


#: How many increments a driven record must have for its N to be worked out
#: from them, and which of them are summed (0-based). ONE rule is held in
#: print: three successive 6 in (150 mm) increments, "the first recorded
#: increment is considered as a 'seating' penetration, while the number of
#: blows to advance the second and third increments are summed to give the
#: N-value" (FHWA NHI-06-088, Soils and Foundations Reference Manual
#: Vol. I, Sec. 3.7, p. 3-51; ``geotech_references/fhwa_soils_foundations``
#: chapter 3). A record of four drives keeps that rule: Fig. 3-20 (FHWA
#: NHI-06-088 Vol. I, Sec. 3.7, p. 3-52) draws "First Increment", "Second
#: Increment", "Third Increment" and "Note: Occasional Fourth Increment Used
#: to provide additional soil material", so the N is still the 2nd + 3rd.
#: That is the record's own rule for a fourth drive, cited from the same
#: page ("Fig. 3-20, FHWA NHI-06-088 Vol. I, p. 3-52",
#: ``log_floor.FOURTH_INCREMENT_SOURCE``, RI-70). The figure also prints
#: "the last 300 mm"; the lead's call of 2026-11-13 (RI-79 review) keeps
#: four drives on the record's reading. No other count has an entry,
#: because no page held says how its increments are summed: a record driven
#: in six increments follows a different procedure (EN 1997-2:2007 Sec.
#: 4.6.2, p. 49, sends the test to EN ISO 22476-3, which the library does
#: not hold; Sec. 4.6.1 (2)P gives N as the blows "for 300 mm of sampler
#: penetration ... below a seating drive" and prints no increments), and
#: the figure prints one fourth increment, so five or more is not its case.
N_RULES: Dict[int, Tuple[int, ...]] = {3: (1, 2), 4: (1, 2)}

#: The set-aside groups a score can carry (``LogScore.set_aside``): counted
#: and listed, never in a metric's total.
SET_ASIDE: Dict[str, str] = {
    "n_no_rule": "N not printed, no rule held for this number of "
                 "increments",
    "fields_struck": "header fields the answer sheet strikes as not "
                     "printed on the form",
}


def _n_by_rule(counts: Sequence[float]) -> Optional[float]:
    """The N a record's increments define, where a printed rule is held
    for that number of increments (:data:`N_RULES`); else None."""
    rule = N_RULES.get(len(counts))
    return None if rule is None else sum(counts[i] for i in rule)


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


def _same_words(a: str, b: str) -> bool:
    """Equal after the normalising :func:`_same_text` applies (case,
    spacing, punctuation): one holding the other is not enough, so "wet"
    is not "wet below 7 ft" (a note's words, RI-103)."""
    left = "".join(ch.lower() for ch in str(a or "") if ch.isalnum())
    right = "".join(ch.lower() for ch in str(b or "") if ch.isalnum())
    return bool(left) and left == right


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
        # The misses travel with the counts (the first 25): a run file that
        # says only "recovery 0/4" cannot say WHICH values were missed, and
        # the Foundry run of 2026-10-02 could not be diagnosed from its files.
        return {"found": self.found, "total": self.total,
                "rate": self.rate, "misses": list(self.misses[:25])}


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
    #: Since 5.23.0 the reader's record is the MERGE of the grid's floor and
    #: the model's answer, so ``after`` is "floor + model". The model's
    #: answer scored ALONE is kept here, so a scorecard can say what the
    #: model would have scored without the floor -- which on the first
    #: cluster run was less than the grid. The counts are the merge's own.
    model_alone: Optional[Dict[str, Any]] = None
    #: The grid's floor RECORD scored alone -- what the merge started from.
    #: ``before`` scores the grid's cells, which can credit a value the
    #: floor's record never held (an index value printed a line off its
    #: sample's row); this is the record's own score, printed beside before,
    #: model and after (Foundry brief 4: the blind index fell 44 % -> 0 %
    #: and the RESULTS could not say whether the floor ever had them).
    floor_alone: Optional[Dict[str, Any]] = None
    disagreements: int = 0
    kept: int = 0
    added: int = 0
    reconciled: int = 0
    #: Slots where the model returned the floor's own unrounded depth: one
    #: reading, the floor's, and not counted in ``reconciled`` (RI-131); a
    #: run before RI-131 counted them there. And blow records refused for
    #: more counts than a driven record has drives (RI-130).
    echoed: int = 0
    refused: int = 0
    #: What the reader recorded for this log -- the merged investigation,
    #: the model's own and the grid's floor -- so a check fixed after a run
    #: can be re-applied to the run file with no model call
    #: (:func:`rescore_saved`), and a run file can answer "did the FLOOR have
    #: that value" rather than only "did the grid's cells". ``None`` on a
    #: score that has no record (the grid stage, an error).
    record: Optional[List[Dict[str, Any]]] = None
    model_record: Optional[List[Dict[str, Any]]] = None
    floor_record: Optional[List[Dict[str, Any]]] = None
    #: What the visual-scales voters did on this log
    #: (:func:`report_ingest.visual_scales.log_grid_for`): the label-reading
    #: calls, the layer tops found in a scan's pixels and their votes
    #: against the record. Empty with the setting off (BACKLOG RI-05).
    visual_scales: Dict[str, Any] = field(default_factory=dict)
    #: Answer values that are in NO metric's total, by group
    #: (:data:`SET_ASIDE`), each a line saying which value: an N the record
    #: could not be expected to hold, a header field the sheet strikes.
    #: Empty for an answer sheet with none, and then not in the run file.
    set_aside: Dict[str, List[str]] = field(default_factory=dict)

    def aside(self, group: str, what: str) -> None:
        self.set_aside.setdefault(group, []).append(what)

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
            **({"visual_scales": dict(self.visual_scales)}
               if self.visual_scales else {}),
            **({"set_aside": _aside_dict(self.set_aside)}
               if self.set_aside else {}),
            "log_id": self.log_id, "report": self.report, "stage": self.stage,
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
            "overall": self.total.to_dict(),
            "cost": dict(self.cost), "model_calls": self.model_calls,
            "unresolved": self.unresolved, "changes": self.changes,
            "error": self.error,
            "model_alone": self.model_alone,
            "floor_alone": self.floor_alone,
            "disagreements": self.disagreements, "kept": self.kept,
            "added": self.added, "reconciled": self.reconciled,
            "echoed": self.echoed, "refused": self.refused,
            "record": self.record, "model_record": self.model_record,
            "floor_record": self.floor_record,
        }


def _aside_dict(groups: Dict[str, List[str]]) -> Dict[str, Any]:
    return {group: {"count": len(lines), "what": SET_ASIDE.get(group, ""),
                    "items": list(lines[:25])}
            for group, lines in groups.items()}


def _struck(truth: Dict[str, Any]) -> set:
    """The header fields an answer sheet strikes: a reviewer who opened the
    pages found the form does not print them (an AGS4 answer sheet fills
    its header from the data, whatever the form prints)."""
    return {str(key) for key in truth.get("fields_struck") or ()}


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
            standing = (not ok and bool(want) and _as_multiset(
                numbers_in_window(window, ("blows", "n_value")), want))
            ruled = _n_by_rule(want)
            if standing and ruled is None:
                out.aside("n_no_rule", f"N={sample['n']} at {depth_text} "
                                       f"({len(want)} increments)")
            else:
                ok = ok or (standing and _close(ruled, n))
                out.score("n_value").add(
                    ok, f"N={sample['n']} at {depth_text}")
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
    struck = _struck(truth)
    for key, value in fields.items():
        names = FIELD_MAP.get(key)
        if not names:
            continue
        if key in struck:
            out.aside("fields_struck", f"{key}={value!r}")
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
        # record keeps the log's own keys, so both spellings are offered. A
        # key kept as the log printed its label ("Foreman", "Drill Rig") is
        # offered in the truth's snake case too.
        snake = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
        out.setdefault(snake, value)
        for name in (key, snake):
            for canonical in FIELD_MAP.get(name, ()):
                out.setdefault(canonical, value)
    return {k: v for k, v in out.items() if str(v or "").strip()}


def _recovery_numbers(sample: Any, attribute: str) -> List[float]:
    """What a sample says its recovery (or RQD) was, as printed.

    A log prints recovery as a PERCENTAGE or as a LENGTH -- ``45`` under
    ``REC (cm)``, ``14`` under ``RECOVERY (in)`` -- and the record keeps the
    two apart, in ``recovery_percent`` and in ``recovery``. The truth writes
    whichever the log printed (``45``, ``"29cm 64%"``), so both are offered.
    Until 2026-10-04 only the percentage was looked at, so a recovery the
    grid's floor read off a length column -- and the merge then KEPT, because
    the model left it out -- scored as missed on every log that prints one.
    """
    out: List[float] = []
    got = getattr(sample, attribute, None)
    if got is not None:
        out.append(float(got))
    if attribute == "recovery_percent":
        length = getattr(sample, "recovery", None)
        if length is not None:
            out.append(float(length.value))
    return out


def _as_numbers(got: Any) -> List[float]:
    """An index value's numbers: as printed and, for a quantity, in SI.

    A pocket penetrometer the log prints in tsf is the truth's ``pp_kpa``
    once converted; comparing only the printed number would mark it missed
    however exactly it was read.
    """
    if not hasattr(got, "value"):
        return [float(got)]
    out = [float(got.value)]
    converted = got.si_value if hasattr(got, "si_value") else None
    if converted is not None:
        out.append(float(converted))
    return out


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
        def holds(r) -> bool:
            got = _blow_numbers(r.blows)
            return ((_in_order(got, want) or _as_multiset(got, want))
                    and _penetrations_agree(r.blows, blows))

        if blows and sample.get("blows_kind") == "sampler":
            # A driven sampler that is not a split spoon (a tube, a ring):
            # the record keeps its blows as a drive record of its own at
            # the sample's depth (``model.SPT``: "one driven penetration
            # record", its N None), so that is where they are found -- the
            # whole record, never a count among another test's increments.
            ok = any(len(_blow_numbers(r.blows)) == len(want) and holds(r)
                     for r in near_spt)
            out.score("blows").add(
                ok, f"sampler blows {blows} at {depth_text}")
        elif blows:
            ok = any(holds(r) for r in near_spt)
            out.score("blows").add(ok, f"blows {blows} at {depth_text}")
        if sample.get("n") is not None:
            n = float(sample["n"])
            ok = any(r.n is not None and _close(float(r.n), n)
                     for r in near_spt)
            # The log printed only the drives, and neither the grid nor the
            # reader does arithmetic. Where a printed rule is held for that
            # number of increments (N_RULES) and gives the answer's N, the
            # drives standing at the right depth ARE the N being present.
            # For any other number nothing is worked out: the N is set
            # aside, counted and listed, and is in no total.
            standing = (not ok and bool(want)
                        and any(r.n is None and holds(r) for r in near_spt))
            ruled = _n_by_rule(want)
            if standing and ruled is None:
                out.aside("n_no_rule", f"N={sample['n']} at {depth_text} "
                                       f"({len(want)} increments)")
            else:
                ok = ok or (standing and _close(ruled, n))
                out.score("n_value").add(
                    ok, f"N={sample['n']} at {depth_text}")

        for key, attribute in (("recovery", "recovery_percent"),
                               ("rqd", "rqd_percent")):
            value = sample.get(key)
            if value is None:
                continue
            want_values = _wanted_numbers(value)
            ok = any(_close(got, w)
                     for s in near_samples
                     for got in _recovery_numbers(s, attribute)
                     for w in want_values)
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
                if any(_close(number, w) for number in _as_numbers(got)
                       for w in want_values):
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
        _score_interval(out, layer, record_layers, want, f"{top} {unit}",
                        unit, q_m)

    for water in _flat(truth, "water"):
        depth = water.get("depth")
        if depth is None:
            continue
        want = _to_m(depth, unit)
        out.score("water").add(
            any(d is not None and abs(d - want) <= WATER_TOL_M
                for d in record_water),
            f"water at {depth} {unit}")

    struck = _struck(truth)
    for key, value in (truth.get("fields") or {}).items():
        if key not in FIELD_MAP:
            continue
        if key in struck:
            out.aside("fields_struck", f"{key}={value!r}")
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


def _label_text(text: Any) -> str:
    """A label to compare as printed: exact after whitespace and case."""
    return " ".join(str(text or "").split()).casefold()


def _score_interval(out: LogScore, layer: Dict[str, Any],
                    record_layers: Sequence[Tuple[Any, Optional[float]]],
                    want: float, where: str, unit: Optional[str],
                    q_m: Any) -> None:
    """The interval groups of one answer-sheet layer (:data:`
    INTERVAL_METRICS`), each scored only where the sheet's layer carries its
    key: a sheet without the keys adds nothing to ``out``."""
    keys = ("notes", "line", "line_full_width", "label")
    if not any(key in layer for key in keys):
        return
    near = sorted(((abs(top - want), n) for n, (_ly, top)
                   in enumerate(record_layers)
                   if top is not None and abs(top - want) <= LAYER_TOL_M))
    found = record_layers[near[0][1]][0] if near else None

    for note in layer.get("notes") or []:
        words = str(note.get("text") or "")
        depth = note.get("depth")
        ok = False
        for got in (getattr(found, "notes", None) or []):
            if not _same_words(got.text, words):
                continue
            got_m = q_m(got.depth)
            if depth is None:
                ok = got.depth is None
            else:
                ok = (got_m is not None and
                      abs(got_m - _to_m(depth, unit)) <= LAYER_TOL_M)
            if ok:
                break
        at = f" at {depth} {unit}" if depth is not None else ""
        out.score("interval_note").add(
            ok, f"note {words!r}{at} in the interval at {where}")

    if "line" in layer:
        line = layer.get("line")
        out.score("line_kind").add(
            found is not None
            and _label_text(found.top_line) == _label_text(line),
            f"line {line!r} at {where}")
    if layer.get("line_full_width") is not None:
        width = bool(layer["line_full_width"])
        out.score("line_width").add(
            found is not None and found.top_line_full_width is width,
            f"line full width {width} at {where}")

    if "label" in layer:
        label = layer.get("label")
        ok = (found is not None
              and _label_text(found.stratum_label) == _label_text(label))
        out.score("stratum_label").add(ok, f"label {label!r} at {where}")
        if ok and _label_text(label):
            if layer.get("label_source"):
                source = str(layer["label_source"])
                out.score("label_source").add(
                    _same_text(found.stratum_label_source, source),
                    f"label source {source!r} at {where}")
            carried = bool(layer.get("label_carried"))
            out.score("label_carried").add(
                bool(found.stratum_label_carried) is carried,
                f"label {label!r} at {where} "
                + ("carried down" if carried else "printed against it"))


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
    from report_ingest import visual_scales
    from report_ingest.log_reader import read_log

    log_id = truth["id"]
    pages = [int(p) for p in truth.get("pages") or []]
    report = report_id or log_id.split("_")[0]
    # The grid as the reader would build it (report_ingest.visual_scales:
    # the setting decides whether pixel-found layer tops count), so "grid
    # alone" and the reader's floor are the same grid.
    extra, charge = visual_scales.meter()
    try:
        grid, scales_info = visual_scales.log_grid_for(doc, pages, engine,
                                                       charge)
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
        visual_scales.add_cost(result.cost, extra)
        if scales_info and not result.visual_scales:
            result.visual_scales = scales_info
    except Exception as exc:                        # a model call that failed
        after = LogScore(log_id=log_id, report=report, stage="record",
                         error=f"{type(exc).__name__}: {exc}")
        return before, after

    after = score_record(truth, [result.investigation])
    after.record = dump_investigations([result.investigation])
    after.cost = dict(result.cost)
    after.model_calls = result.model_calls
    after.unresolved = len(result.unresolved)
    after.changes = len(result.changes)
    after.warnings = list(result.warnings)
    # The model's answer alone, before the merge with the grid's floor: the
    # third number, so the scorecard can print floor / model / floor+model.
    model_alone = getattr(result, "model_investigation", None)
    if model_alone is not None:
        after.model_record = dump_investigations([model_alone])
        alone = score_record(truth, [model_alone])
        after.model_alone = {
            "scores": {k: v.to_dict() for k, v in alone.scores.items()},
            "overall": alone.total.to_dict()}
    floor = getattr(result, "floor", None)
    if floor is not None:
        after.floor_record = dump_investigations([floor])
        after.floor_alone = _scored(score_record(truth, [floor]))
    after.disagreements = len(getattr(result, "disagreements", ()) or ())
    after.kept = len(getattr(result, "kept", ()) or ())
    after.added = len(getattr(result, "added", ()) or ())
    after.reconciled = int(getattr(result, "reconciled", 0) or 0)
    after.echoed = len(getattr(result, "echoed", ()) or ())
    after.refused = len(getattr(result, "refused", ()) or ())
    # The grid's own (label calls, pixel layer tops) and the reader's (the
    # votes), together: a reader handed the grid holds only its votes.
    after.visual_scales = {**(scales_info or {}),
                           **(getattr(result, "visual_scales", None) or {})}
    return before, after


# ---------------------------------------------------------------------------
# a saved run, re-scored
# ---------------------------------------------------------------------------

def dump_investigations(investigations: Optional[Sequence[Any]]
                        ) -> Optional[List[Dict[str, Any]]]:
    """Investigations as JSON-ready dicts, for a run file. ``None`` stays
    ``None``; a record that will not dump is left out rather than failing
    the run that produced it."""
    if investigations is None:
        return None
    out: List[Dict[str, Any]] = []
    for inv in investigations:
        if inv is None:
            continue
        try:
            out.append(inv.model_dump(mode="json"))
        except Exception:                        # a record that will not dump
            continue
    return out


def _load_investigations(blobs: Sequence[Dict[str, Any]]) -> List[Any]:
    from report_ingest.model import Investigation

    return [Investigation.model_validate(b) for b in blobs]


def _scored(score: LogScore) -> Dict[str, Any]:
    return {"scores": {k: v.to_dict() for k, v in score.scores.items()},
            "overall": score.total.to_dict(),
            **({"set_aside": _aside_dict(score.set_aside)}
               if score.set_aside else {})}


def rescore_saved(truth: Dict[str, Any], after: Dict[str, Any]
                  ) -> Optional[Dict[str, Any]]:
    """Re-score a saved run file's ``after`` blob against ``truth`` with the
    CURRENT checks -- no model call, no PDF.

    The merged record replaces ``scores`` and ``overall``; the model's own
    record, where kept, replaces ``model_alone``; the grid's floor, where
    kept, is scored into ``floor_alone`` -- which is what says whether a
    value the grid's CELLS were credited for ever reached the floor's
    RECORD. ``None`` for a run file written before the records were kept
    (before 2026-10-04).
    """
    if not after or after.get("record") is None:
        return None
    out = dict(after)
    out.pop("set_aside", None)          # the current checks decide it anew
    out.update(_scored(score_record(
        truth, _load_investigations(after["record"]))))
    if after.get("model_record") is not None:
        out["model_alone"] = _scored(score_record(
            truth, _load_investigations(after["model_record"])))
    if after.get("floor_record") is not None:
        out["floor_alone"] = _scored(score_record(
            truth, _load_investigations(after["floor_record"])))
    return out


def rescore_run(logs_dir: Any, truth_dir: Any, *, write: bool = False
                ) -> Dict[str, Any]:
    """Re-score every run file of a saved logs run with the CURRENT checks
    -- no model call, no PDF (BACKLOG RI-79).

    ``logs_dir`` holds the run files (``<out_dir>/logs/<log id>.json``, as
    the logs stage writes them) and ``truth_dir`` the answer sheets
    (``<log id>.json``). Returns, per metric, found and total as SAVED and
    as RE-SCORED, the set-aside counts, and the run files that could not be
    re-scored (no answer sheet, or written before the records were kept).
    With ``write`` each run file's ``after`` is replaced by the re-scored
    one and what it held is kept beside it as ``after_as_run``, once.
    Counts only: no value of a record or an answer sheet is returned.
    """
    import json
    from pathlib import Path

    def tally(into: Dict[str, List[int]], blob: Dict[str, Any]) -> None:
        for name, score in (blob.get("scores") or {}).items():
            cell = into.setdefault(name, [0, 0])
            cell[0] += int(score.get("found") or 0)
            cell[1] += int(score.get("total") or 0)

    saved: Dict[str, List[int]] = {}
    again: Dict[str, List[int]] = {}
    aside: Dict[str, int] = {}
    skipped: Dict[str, str] = {}
    done = 0
    for path in sorted(Path(logs_dir).glob("*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        after = blob.get("after") or {}
        sheet = Path(truth_dir) / f"{blob.get('log_id') or path.stem}.json"
        if not sheet.is_file():
            skipped[path.stem] = "no answer sheet"
            continue
        truth = json.loads(sheet.read_text(encoding="utf-8"))
        new = rescore_saved(truth, after)
        if new is None:
            skipped[path.stem] = "the run file keeps no record"
            continue
        done += 1
        tally(saved, blob.get("after_as_run") or after)
        tally(again, new)
        for group, entry in (new.get("set_aside") or {}).items():
            aside[group] = aside.get(group, 0) + int(entry.get("count") or 0)
        if write:
            blob.setdefault("after_as_run", after)
            blob["after"] = new
            path.write_text(json.dumps(blob, indent=1, ensure_ascii=False),
                            encoding="utf-8")

    def total(cells: Dict[str, List[int]]) -> List[int]:
        return [sum(c[0] for k, c in cells.items() if k in METRICS),
                sum(c[1] for k, c in cells.items() if k in METRICS)]

    return {"run_files": done, "skipped": skipped,
            "saved": {**saved, "overall": total(saved)},
            "rescored": {**again, "overall": total(again)},
            "set_aside": aside}

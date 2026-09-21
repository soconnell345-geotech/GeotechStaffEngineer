"""The arithmetic of two voters on one value: the floor and the model.

THE PRINCIPLE, from the first full cluster run (2026-09-18). The log reader
was handed the grid's rows and re-emitted the whole record from the model's
answer: seven of ten blind logs came back with FEWER values than the grid
alone had placed. The lab reader did the same on six sheets. The deterministic
pass -- the log grid for a log, the detected tables for a sheet -- is the
FIRST voter, and what it found is the floor. The model is the second voter:
it may ADD a value the floor lacks, it may CORRECT a floor value only with
evidence, and it may never silently drop one. The owner's standing direction
is the other half: *"if multiple methods say different things, it could
trigger an extra review ... would be good to have confidence values"* -- so a
disagreement is kept in the record with both values and a QA entry, never
settled in silence.

WHAT THIS MODULE HOLDS is the part both readers share: what counts as the
same number (the scorer's own tolerances, so a difference the scorecard would
never see is not a disagreement), what counts as evidence, the
:class:`MergeLog` a merge writes as it goes, and the one function that settles
a slot. The seeding of a floor from a grid or from tables, and the walk over
one reader's record, live beside each reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from report_ingest.model import Alternative, Provenance, Quantity

__all__ = [
    "DEPTH_TOL_M", "LAYER_TOL_M", "EXACT_TOL", "MergeLog", "Settled",
    "depth_m", "same_depth", "same_number", "same_text", "has_evidence",
    "settle", "show", "fold",
]

#: The scorer's own tolerances (``log_scoring``, ``lab_scoring``): a sample,
#: a water level or an index value within 0.15 m is the same reading, a
#: layer top within 0.30 m, and a printed number is the truth's number to two
#: per cent or half a unit. A difference the scorecard would never notice is
#: not a disagreement, so these are the lines drawn here too.
DEPTH_TOL_M = 0.15
LAYER_TOL_M = 0.30
#: A laboratory index value is EXACT to a rounding of the last printed digit
#: (``lab_scoring.EXACT_TOL``). A liquid limit of 31 that comes back as 32 is
#: a misreading, and the record says so.
EXACT_TOL = 0.01


def depth_m(value: Optional[Quantity]) -> Optional[float]:
    """A depth in metres, or the raw number when its unit is unknown.

    A log whose unit could not be read carries the ruler's own numbers, and
    two readings of that log are compared as those numbers: the tolerance is
    then loose or tight by the unit's factor, which is the honest state of
    knowledge about that log.
    """
    if value is None:
        return None
    converted = value.si_value
    return float(value.value) if converted is None else converted


def same_depth(a: Optional[Quantity], b: Optional[Quantity],
               tol: float = DEPTH_TOL_M) -> Optional[bool]:
    """Are two depths one reading? ``None`` when either is missing."""
    left, right = depth_m(a), depth_m(b)
    if left is None or right is None:
        return None
    return abs(left - right) <= tol


def same_number(a: Optional[float], b: Optional[float],
                tol: Optional[float] = None) -> Optional[bool]:
    """Are two printed numbers the same one? ``None`` when either is missing.

    With no ``tol`` the log scorer's rule applies: two per cent or half a
    unit, whichever is larger, because a log prints 12 for 11.8 and 116 for
    115.6. A caller scoring a laboratory value passes :data:`EXACT_TOL`.
    """
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    if tol is None:
        tol = max(0.5, 0.02 * abs(b))
    return abs(a - b) <= max(tol, abs(b) * 1e-9)


def fold(text: Any) -> str:
    return "".join(ch.lower() for ch in str(text or "") if ch.isalnum())


def same_text(a: Any, b: Any) -> Optional[bool]:
    """Do two printed strings say the same thing? ``None`` when either is
    empty. One containing the other counts: the grid reads "SANDY LEAN CLAY
    (CL) dark brown" and the model reads "SANDY LEAN CLAY (CL), dark brown,
    very stiff", and those are one description."""
    left, right = fold(a), fold(b)
    if not left or not right:
        return None
    return left == right or left in right or right in left


def show(value: Any) -> str:
    """A value as the record prints it, for a QA line."""
    if value is None:
        return ""
    if isinstance(value, Quantity):
        return f"{value.value:g} {value.unit}".strip()
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "-".join(show(v) for v in value)
    return str(value)


def has_evidence(prov: Optional[Provenance],
                 floor_bbox: Optional[Sequence[float]] = None) -> bool:
    """Did the model earn the right to overrule the floor on this object?

    The rule in the brief: the model may correct a floor value only with
    evidence, which means it named the page and the box and said what it
    read. A provenance that copies a box off the rows AND carries a note is
    that; so is a value read off the PICTURE with a note saying what the
    picture shows, because a symbol has no row to copy a box from and the
    note is the whole of what can be said. A provenance with no box and no
    note, or a box that is the floor's own cell with nothing said about it,
    is a re-emission and does not overrule anything.
    """
    if prov is None:
        return False
    if not str(prov.note or "").strip():
        return False
    if prov.bbox is None:
        return prov.method in ("model_from_picture", "vision")
    if floor_bbox is not None:
        try:
            same_box = all(abs(float(a) - float(b)) < 0.5
                           for a, b in zip(prov.bbox, floor_bbox))
        except (TypeError, ValueError):
            same_box = False
        if same_box:
            return True     # the same cell, read again and explained: fine
    return True


@dataclass
class MergeLog:
    """What one merge did, for the result and for the QA section.

    ``disagreements`` is the list the owner asked for: every slot where the
    two voters said different things, with both values, which one the record
    carries and why. ``kept`` is every floor value the model did not return
    -- kept, with a note, never dropped. ``added`` is every value the model
    brought that the floor lacked. ``reconciled`` counts the slots both gave
    the same answer to, which is the confidence claim.
    """

    disagreements: List[Dict[str, Any]] = field(default_factory=list)
    kept: List[Dict[str, Any]] = field(default_factory=list)
    added: List[Dict[str, Any]] = field(default_factory=list)
    reconciled: int = 0

    def keep(self, where: str, what: str, page: Optional[int] = None,
             value: Any = None) -> None:
        row: Dict[str, Any] = {
            "where": where, "what": what,
            "why": "kept from the floor; the reader did not return it"}
        if page is not None:
            row["page"] = int(page)
        if value is not None:
            row["value"] = show(value)
        self.kept.append(row)

    def add(self, where: str, what: str, page: Optional[int] = None,
            value: Any = None, method: str = "model") -> None:
        row: Dict[str, Any] = {"where": where, "what": what,
                               "method": method}
        if page is not None:
            row["page"] = int(page)
        if value is not None:
            row["value"] = show(value)
        self.added.append(row)

    def disagree(self, where: str, what: str, floor_value: Any,
                 model_value: Any, kept: str, why: str,
                 page: Optional[int] = None,
                 floor_method: str = "grid",
                 floor_confidence: float = 0.5,
                 model_confidence: float = 0.5,
                 model_method: str = "model") -> None:
        row: Dict[str, Any] = {
            "where": where, "what": what,
            "floor": show(floor_value), "model": show(model_value),
            "floor_method": floor_method, "model_method": model_method,
            "kept": kept, "why": why,
            "confidence": {"floor": round(float(floor_confidence), 2),
                           "model": round(float(model_confidence), 2)},
        }
        if page is not None:
            row["page"] = int(page)
        self.disagreements.append(row)

    def to_dict(self) -> Dict[str, Any]:
        return {"disagreements": [dict(d) for d in self.disagreements],
                "kept": [dict(k) for k in self.kept],
                "added": [dict(a) for a in self.added],
                "reconciled": self.reconciled}


@dataclass
class Settled:
    """One slot after both voters have spoken."""

    value: Any
    method: str
    confidence: float
    alternative: Optional[Alternative] = None
    verdict: str = ""        # floor_only, model_only, reconciled, floor_wins,
    #                          model_wins, empty


def settle(field_name: str, floor_value: Any, model_value: Any,
           same: Optional[bool], *, floor_method: str = "grid",
           floor_confidence: float = 0.5, model_method: str = "model",
           model_confidence: float = 0.9, evidence: bool = False,
           floor_unit: str = "", model_unit: str = "",
           model_note: str = "") -> Settled:
    """Settle one slot between the floor and the model.

    ``same`` is the caller's verdict on whether the two values are one
    reading (None when either side is missing). The rules, in order:

    * only the floor has it -> the floor's value, kept;
    * only the model has it -> the model's value, added;
    * both, the same -> the MODEL'S value (it read the printed number; the
      floor's may be interpolated off a ruler) as ``reconciled``, at the
      higher of the two confidences;
    * both, different, with evidence -> the model's value, the floor's as the
      alternative;
    * both, different, no evidence -> the floor's value, the model's as the
      alternative.

    Every difference is a disagreement whichever side wins; the caller logs
    it. Nothing here drops a value.
    """
    if floor_value is None and model_value is None:
        return Settled(None, floor_method, 0.0, verdict="empty")
    if model_value is None:
        return Settled(floor_value, floor_method, floor_confidence,
                       verdict="floor_only")
    if floor_value is None:
        return Settled(model_value, model_method, model_confidence,
                       verdict="model_only")
    if same:
        return Settled(model_value, "reconciled",
                       min(1.0, max(floor_confidence, model_confidence)),
                       verdict="reconciled")
    if evidence:
        alt = Alternative(field=field_name, value=show(floor_value),
                          unit=floor_unit, method=floor_method,  # type: ignore[arg-type]
                          confidence=floor_confidence,
                          note="the floor's value; the model overruled it "
                               "with a box and a note")
        return Settled(model_value, model_method, model_confidence,
                       alternative=alt, verdict="model_wins")
    alt = Alternative(field=field_name, value=show(model_value),
                      unit=model_unit, method=model_method,  # type: ignore[arg-type]
                      confidence=model_confidence,
                      note=(model_note or "the model's value; it named no "
                                          "box or note to overrule the floor"))
    return Settled(floor_value, floor_method, floor_confidence,
                   alternative=alt, verdict="floor_wins")


def unit_of(value: Any) -> str:
    return value.unit if isinstance(value, Quantity) else ""


def carry(prov: Optional[Provenance], settled: Settled,
          note: str = "") -> Optional[Provenance]:
    """A provenance updated with what the vote decided for one slot.

    The object-level provenance keeps its page and box; the method becomes
    the vote's, the confidence the vote's, and an alternative is appended
    when one voter lost. Called once per settled slot on the same object,
    so the LAST slot's method stands -- callers settle the object's own
    depth or value last for that reason.
    """
    if prov is None:
        return None
    updated = prov.model_copy(deep=True)
    updated.method = settled.method  # type: ignore[assignment]
    updated.confidence = max(0.0, min(1.0, float(settled.confidence)))
    if settled.alternative is not None:
        updated.alternatives.append(settled.alternative)
    if note and note not in (updated.note or ""):
        updated.note = f"{updated.note}; {note}".strip("; ") if updated.note \
            else note
    return updated

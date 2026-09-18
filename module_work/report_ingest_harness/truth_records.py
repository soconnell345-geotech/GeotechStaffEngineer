"""The hand-truthed logs as :class:`report_ingest.model.Investigation` records.

The truth files in the gitignored ``raw/truth/logs/`` were written by hand off
the rendered pages, in a shape close to the record but not the record: the
sample types are the trade's words rather than the model's vocabulary, a
recovery is sometimes a percentage and sometimes ``"33cm 73%"``, a water
reading's timing is whatever the log printed, and one sheet carries three
borings in an ``investigations`` list.

This turns them into records. It exists for two reasons:

* the DIGGS writer can then be gated on FIFTEEN REAL LOGS with no model in the
  loop at all -- every truth file becomes a record, is written to DIGGS,
  validated against the schema and read back, and every depth, N value, water
  level and index value has to come back;
* the log reader's score can be computed against the same records the writer
  is gated on, so the two measurements mean the same thing.

PRIVACY. Everything read is under the gitignored raw folder. Nothing here
prints, returns or records a project name, a firm or a file name -- only the
report ID and the log's own identifier, which is a boring number.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import corpus as C
from report_ingest.model import (
    DrillingDetails, Investigation, Layer, Quantity, SPT, Sample, WaterLevel,
)

__all__ = [
    "TRUTH_DIR", "truth_files", "load_truth", "investigations_from_truth",
    "SAMPLE_KIND", "WATER_WHEN", "truth_available",
]

TRUTH_DIR = C.RAW_DIR / "truth" / "logs"

#: The truth's word for a sampler -> the record's vocabulary. Anything not
#: here becomes ``other``, which is an answer: the record would rather say it
#: does not know than call a sampler something it was not.
SAMPLE_KIND: Dict[str, str] = {
    "spt": "spt", "ss": "spt", "split spoon": "spt",
    "ring": "ring", "california": "ring", "modified california": "ring",
    "shelby": "shelby", "st": "shelby", "thin wall": "shelby",
    "bulk": "bulk", "bag": "bulk",
    "grab": "grab",
    "core": "core", "rock core": "core", "nq": "core", "hq": "core",
    "cuttings": "cuttings", "wash": "cuttings",
}

#: What the log printed about WHEN the water was read -> the record's five
#: states. Matched on the words in the phrase, longest first, so "at end of
#: drilling" lands on completion and "24 hrs after drilling" on after_hours.
WATER_WHEN: Sequence[Tuple[str, str]] = (
    ("not encountered", "not_encountered"),
    ("no groundwater", "not_encountered"),
    ("dry", "not_encountered"),
    ("after drilling", "after_hours"),
    ("hrs", "after_hours"),
    ("hours", "after_hours"),
    ("stabilized", "after_hours"),
    ("temporary well", "after_hours"),
    ("start of day", "after_hours"),
    ("end of drilling", "at_completion"),
    ("completion", "at_completion"),
    ("end of day", "at_completion"),
    ("casing pulled", "at_completion"),
    ("time of drilling", "while_drilling"),
    ("while drilling", "while_drilling"),
    ("encountered", "while_drilling"),
    ("first", "while_drilling"),
    ("nivel freatico", "while_drilling"),
)

#: Which header-field keys of the truth fill which record field. The truth was
#: written in the log's own words -- and in Spanish and French on two of them
#: -- so this is deliberately a table rather than a guess.
_FIELD_MAP: Dict[str, str] = {
    "boring_id": "investigation_id", "test_pit_id": "investigation_id",
    "hammer": "hammer_type", "hammer_type": "hammer_type",
    "sampler_hammer": "hammer_type",
    "method": "method", "drilling_method": "method",
    "advancement_method": "method",
    "equipment": "equipment", "drill_rig": "equipment",
    "drilling_equipment": "equipment",
    "driller": "driller", "foreman": "driller", "drilling_foreman": "driller",
    "contractor": "contractor", "drilling_contractor": "contractor",
    "drilling_company": "contractor", "boring_contractor": "contractor",
    "excavation_contractor": "contractor",
    "representative": "logged_by", "logged_by": "logged_by",
    "field_engineer": "logged_by", "sampled_by": "logged_by",
    "sampler": "sampler", "bit": "sampler",
}

#: The unit a dry unit weight is printed in, by the log's depth unit. Every
#: truth log that states one is imperial and heads the column pcf; a metric
#: log heads it kN/m3.
_DUW_UNIT = {"ft": "pcf", "m": "kN/m3"}


def truth_available() -> bool:
    """Is the private truth folder here? It is gitignored, so usually not."""
    return TRUTH_DIR.is_dir() and any(TRUTH_DIR.glob("*.json"))


def truth_files() -> List[Path]:
    return sorted(TRUTH_DIR.glob("*.json"))


def load_truth(only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Every truth file, or the ones whose report is in ``only``."""
    if not TRUTH_DIR.is_dir():
        raise FileNotFoundError(
            f"no hand-truthed logs ({TRUTH_DIR} is gitignored and exists only "
            f"where the owner put it)")
    out: List[Dict[str, Any]] = []
    for path in truth_files():
        truth = json.loads(path.read_text(encoding="utf-8"))
        if only and truth["id"].split("_")[0] not in only:
            continue
        out.append(truth)
    return out


# ---------------------------------------------------------------------------
# the conversion
# ---------------------------------------------------------------------------

def _q(value: Any, unit: str) -> Optional[Quantity]:
    if value is None or unit is None:
        return None
    try:
        return Quantity(value=float(value), unit=unit)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> Optional[float]:
    """The first number in a truth value, however the hand wrote it.

    The truth writes an index value as a number most of the time and as a
    string where the log printed one -- ``"<1"``, ``"25cm 56%"``, ``"NP"``.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _recovery(value: Any) -> Tuple[Optional[float], Optional[str]]:
    """``(percent, length text)`` from a truth recovery.

    ``45`` is 45 %; ``"33cm 73%"`` is both, and the percentage is the one the
    record carries as a number.
    """
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        return max(0.0, min(100.0, float(value))), None
    text = str(value)
    percent = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    length = re.search(r"(\d+(?:\.\d+)?)\s*(cm|mm|m|in)\b", text)
    return ((max(0.0, min(100.0, float(percent.group(1)))) if percent
             else None),
            (f"{length.group(1)} {length.group(2)}" if length else None))


def _sample_kind(value: Any) -> str:
    text = str(value or "").strip().lower()
    return SAMPLE_KIND.get(text, "other")


def _water_when(value: Any) -> str:
    text = str(value or "").strip().lower()
    for needle, when in WATER_WHEN:
        if needle in text:
            return when
    return "unknown"


def _hours(value: Any) -> Optional[float]:
    """Hours out of a phrase like ``24 hrs after drilling``."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hr|hrs|hour|hours)\b",
                      str(value or "").lower())
    return float(match.group(1)) if match else None


def _layers(rows: Sequence[dict], unit: str) -> List[Layer]:
    out: List[Layer] = []
    for row in rows:
        top = _q(row.get("top"), unit)
        if top is None:
            continue
        out.append(Layer(
            top=top, bottom=_q(row.get("bottom"), unit),
            description=str(row.get("description") or ""),
            uscs=str(row.get("uscs") or "")))
    return out


def _samples(rows: Sequence[dict], unit: str) -> Tuple[List[Sample], List[SPT]]:
    """The truth's samples as both a sample list and a driven-record list.

    One truth sample carries both: what was taken (the sampler, the index
    values) and how it was driven (the blows and the N). The record keeps them
    apart, because a ring sample's blows are not an SPT N value and a record
    that mixed them would let one be read as the other.
    """
    samples: List[Sample] = []
    driven: List[SPT] = []
    for row in rows:
        top = _q(row.get("top"), unit)
        if top is None:
            continue
        bottom = _q(row.get("bottom"), unit)
        percent, _length = _recovery(row.get("recovery"))
        duw = _DUW_UNIT.get(unit)
        samples.append(Sample(
            sample_id=str(row.get("id") or ""),
            top=top, bottom=bottom, kind=_sample_kind(row.get("type")),
            recovery_percent=percent,
            rqd_percent=_clamp(_number(row.get("rqd"))),
            water_content=_number(row.get("wc")),
            dry_unit_weight=_q(_number(row.get("duw")), duw),
            liquid_limit=_number(row.get("ll")),
            plastic_limit=_number(row.get("pl")),
            plasticity_index=_number(row.get("pi")),
            fines_percent=_clamp(_number(row.get("fines"))),
            pocket_pen=_q(_number(row.get("pp_kpa")), "kPa"),
            uscs=str(row.get("uscs") or ""),
            note=str(row.get("note") or "")))
        blows = row.get("blows")
        if not blows:
            continue
        driven.append(SPT(
            depth_top=top, depth_bottom=bottom,
            blows=[b if isinstance(b, int) else str(b) for b in blows],
            n=(int(row["n"]) if row.get("n") is not None else None),
            refusal=bool(row.get("refusal")
                         or any(isinstance(b, str) and "/" in b
                                for b in blows)),
            sample_id=str(row.get("id") or "")))
    return samples, driven


def _clamp(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(100.0, value))


def _water(rows: Sequence[dict], unit: str) -> List[WaterLevel]:
    out: List[WaterLevel] = []
    for row in rows:
        when = _water_when(row.get("when"))
        depth = _q(_number(row.get("depth")), unit)
        if depth is None and when != "not_encountered":
            continue
        out.append(WaterLevel(
            depth=depth, when=when,
            hours=_hours(row.get("when")) or _hours(row.get("time")),
            date=str(row.get("date") or ""),
            casing_depth=_q(_number(row.get("casing")), unit),
            caved_depth=_q(_number(row.get("caved")), unit),
            note=str(row.get("note") or row.get("when") or "")))
    return out


def _drilling(fields: Dict[str, Any]) -> DrillingDetails:
    kwargs: Dict[str, str] = {}
    for key, value in fields.items():
        target = _FIELD_MAP.get(key)
        if target and target != "investigation_id" and str(value or "").strip():
            kwargs.setdefault(target, str(value).strip())
    return DrillingDetails(**kwargs)


def _one(truth: Dict[str, Any], block: Dict[str, Any],
         report: str) -> Investigation:
    """One investigation out of a truth file, or out of one of its borings."""
    unit = truth.get("depth_unit") or ""
    fields = {**(truth.get("fields") or {}), **(block.get("fields") or {})}
    kind = "test_pit" if "test_pit" in str(truth.get("kind") or "") else "boring"
    samples, driven = _samples(block.get("samples") or [], unit)
    elevation = _number(block.get("ground_elevation")
                        or fields.get("ground_surface_elevation")
                        or fields.get("ground_elevation")
                        or fields.get("elevation"))
    return Investigation(
        investigation_id=str(block.get("investigation_id")
                             or truth.get("investigation_id") or ""),
        kind=kind,
        depth_unit=unit,
        units_known=bool(unit),
        x=_number(fields.get("longitude")),
        y=_number(fields.get("latitude")),
        coordinate_system=str(fields.get("coordinate_system") or ""),
        elevation=_q(elevation, unit),
        total_depth=_q(_number(fields.get("total_depth")
                               or fields.get("completion_depth")
                               or fields.get("depth")), unit),
        date_started=str(fields.get("date_started") or fields.get("date")
                         or fields.get("date_drilled") or ""),
        date_finished=str(fields.get("date_completed")
                          or fields.get("date_finished") or ""),
        drilling=_drilling(fields),
        layers=_layers(block.get("layers") or [], unit),
        samples=samples, spt=driven,
        water=_water(block.get("water") or [], unit),
        remarks=str(truth.get("remarks") or ""),
        pages=[int(p) for p in (truth.get("pages") or [])],
        source_report=report,
        sheet=str(truth.get("sheet") or ""),
        fields={k: str(v) for k, v in fields.items()
                if str(v or "").strip()})


def investigations_from_truth(truth: Dict[str, Any]) -> List[Investigation]:
    """Every investigation one truth file describes.

    Usually one. A tabular sheet listing several borings carries them in an
    ``investigations`` list, and each becomes its own record -- which is what
    the DIGGS file needs, since each is its own sampling feature.
    """
    report = str(truth["id"]).split("_")[0]
    blocks = list(truth.get("investigations") or [])
    if not blocks:
        blocks = [truth]
    out = [_one(truth, block, report) for block in blocks]
    # A boring with no identifier cannot be pointed at by an xlink, so it is
    # given the truth file's own handle rather than left nameless.
    for n, inv in enumerate(out, start=1):
        if not inv.investigation_id:
            inv.investigation_id = (f"{truth['id']}-{n}" if len(out) > 1
                                    else str(truth["id"]))
    return out

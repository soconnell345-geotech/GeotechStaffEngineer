"""The hand-truthed lab sheets as :class:`report_ingest.model.LabTest` records.

The truth files in the gitignored ``raw/truth/lab/`` were written by hand off
the rendered pages, in the sheets' OWN words. That is what makes them worth
having and what makes this module necessary: a French sieve sheet keys its
curve by millimetres and an American one by sieve number, a density is
``dry_unit_weight_pcf`` on one page and ``dry_density_g_cm3`` on the next, a
chloride is ``25`` here and ``"<10.0"`` there, and a triaxial prints four
nested stages where another prints three lines.

This turns all of that into records, for two reasons:

* the DIGGS lab writer can then be gated on THIRTY-ONE REAL SHEETS with no
  model in the loop at all -- every truth file becomes records, is written to
  DIGGS, validated against the bundled schema and read back, and every number
  has to come back;
* the lab reader's score is computed against the same records, so the two
  measurements mean the same thing.

HOW THE UNITS ARE READ. The hand wrote the unit into the KEY -- ``c_psf``,
``ucs_MPa``, ``resistivity_ohm_cm``, ``bulk_density_Mg_m3`` -- so a key is
split into a stem and a unit suffix and values are looked up by stem. That is
why it copes with sheets it has never seen: a new sheet that writes ``qu_tsf``
needs no new line here.

THE READING OF THOSE KEYS LIVES IN ``report_ingest.lab_scoring``, not here,
and is imported. The scorer ships in the wheel and runs on the cluster, this
converter is repo-only tooling, and the two must read a truth file the same
way or the gate and the scorecard would be measuring different files.

PRIVACY. Everything read is under the gitignored raw folder. Nothing here
prints, returns or records a project name, a firm or a file name -- only the
report ID and the identifiers a lab sheet printed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import corpus as C
from report_ingest.lab_scoring import (  # the truth files' own shape
    SIEVE_MM, UNIT_SUFFIX, curve_in as _points_in, depth_unit_of, grading_in,
    number, pages_of, quantity, report_of, reported, sieve_points,
)
from report_ingest.model import (
    AtterbergResult, CBRResult, ChemicalResult, CompactionPoint,
    CompactionResult, ConsolidationPoint, ConsolidationResult,
    GradationResult, LabTest, MoistureDensityResult, OtherResult, Quantity,
    ShearPoint, SievePoint, StrengthResult, StrengthSpecimen, SummaryRow,
    SummaryTableResult,
)

__all__ = [
    "TRUTH_DIR", "truth_available", "truth_files", "load_truth",
    "lab_tests_from_truth", "open_reports", "report_of", "pages_of",
]

TRUTH_DIR = C.RAW_DIR / "truth" / "lab"


def truth_available() -> bool:
    """Is the private truth folder here? It is gitignored, so usually not."""
    return TRUTH_DIR.is_dir() and any(TRUTH_DIR.glob("*.json"))


def truth_files() -> List[Path]:
    return sorted(TRUTH_DIR.glob("*.json"))


def open_reports(default: Sequence[str] = ()) -> Tuple[str, ...]:
    """The reports whose lab pages the builder was allowed to look at."""
    path = TRUTH_DIR / "OPEN.txt"
    if path.is_file():
        names = tuple(line.strip() for line
                      in path.read_text(encoding="utf-8").splitlines()
                      if line.strip())
        if names:
            return names
    return tuple(default)


def load_truth(only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Every truth sheet, or the ones whose report is in ``only``."""
    if not TRUTH_DIR.is_dir():
        raise FileNotFoundError(
            f"no hand-truthed lab sheets ({TRUTH_DIR} is gitignored and "
            f"exists only where the owner put it)")
    out: List[Dict[str, Any]] = []
    for path in truth_files():
        truth = json.loads(path.read_text(encoding="utf-8"))
        if only and report_of(truth) not in only:
            continue
        out.append(truth)
    return out


# ---------------------------------------------------------------------------
# what the record needs that the shared readers do not give
# ---------------------------------------------------------------------------

def _number(value: Any) -> Optional[float]:
    """The first number in a truth value, however the hand wrote it."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _depth(value: Any, unit: str) -> Optional[Quantity]:
    got = _number(value)
    if got is None or not unit:
        return None
    return Quantity(value=got, unit=unit)


def _sieve_points(mapping: Dict[str, Any], keyed_mm: bool
                  ) -> List[SievePoint]:
    """A grading curve as the record holds it."""
    return [SievePoint(percent_passing=percent,
                       size=(Quantity(value=size, unit="mm")
                             if size is not None else None),
                       sieve=name)
            for size, percent, name in sieve_points(mapping, keyed_mm)]


def _grading_of(block: Dict[str, Any]) -> List[SievePoint]:
    return [SievePoint(percent_passing=percent,
                       size=(Quantity(value=size, unit="mm")
                             if size is not None else None),
                       sieve=name)
            for size, percent, name in grading_in(block)]


def _points(block: Dict[str, Any], *stems: str
            ) -> Tuple[List[Tuple[float, float]], str, str]:
    points, x_unit, y_unit, _key = _points_in(block, *stems)
    return points, x_unit, y_unit


def _fields(block: Dict[str, Any], used: Sequence[str],
            prefix: str = "", depth: int = 0) -> Dict[str, str]:
    """Everything the sheet printed that no field of the record took.

    Nested blocks are flattened rather than dropped. A 1991 triaxial prints
    its saturation, consolidation and compression stages as nested groups and
    the record models three of their values; the rest are still numbers a
    laboratory printed, and ``fields`` is where the record keeps what it has
    no typed home for.
    """
    skip = set(used)
    out: Dict[str, str] = {}
    if depth > 3:
        return out
    for key, value in block.items():
        if key in skip or value is None:
            continue
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_fields(value, used, f"{name}.", depth + 1))
        elif isinstance(value, list):
            if all(isinstance(item, dict) for item in value) and value:
                for i, item in enumerate(value, start=1):
                    out.update(_fields(item, used, f"{name}.{i}.", depth + 1))
        else:
            out[name] = str(value)
    return out


# ---------------------------------------------------------------------------
# one truth sheet -> records
# ---------------------------------------------------------------------------

_COMMON = ("investigation_id", "sample_id", "depth_top", "depth_bottom",
           "description", "uscs", "lab", "date", "standard")


def _base(truth: Dict[str, Any], block: Dict[str, Any], kind: str,
          unit: str) -> Dict[str, Any]:
    """The identity every LabTest off this sheet shares."""
    return {
        "kind": kind,
        "investigation_id": str(block.get("investigation_id") or "").strip(),
        "sample_id": str(block.get("sample_id") or "").strip(),
        "depth_top": _depth(block.get("depth_top"), unit),
        "depth_bottom": _depth(block.get("depth_bottom"), unit),
        "standard": str(block.get("standard") or truth.get("standard") or ""),
        "lab": str(block.get("lab") or (truth.get("fields") or {}).get("lab")
                   or ""),
        "date": str(block.get("date") or ""),
        "language": str(truth.get("language") or "")[:2],
        "pages": pages_of(truth),
        "source_report": report_of(truth),
        "curves_digitised": bool(block.get("curve_digitised")
                                 or block.get("curves_digitised")),
    }


def _atterberg(truth, block, unit) -> List[LabTest]:
    flow = [(int(round(float(a))), float(b))
            for a, b in (block.get("flow_curve") or [])]
    result = AtterbergResult(
        ll=number(block, "ll", "liquid_limit"),
        pl=number(block, "pl", "plastic_limit"),
        pi=number(block, "pi", "plasticity_index"),
        non_plastic=str(block.get("pl") or "").upper().startswith("N.P")
        or str(block.get("pl") or "").upper() == "NP",
        shrinkage_limit=number(block, "sl", "shrinkage_limit"),
        flow_curve=flow,
        pl_trials=[float(v) for v in (block.get("pl_trials") or [])],
        water_content=number(block, "wc", "water_content", "moisture"),
        uscs=str(block.get("uscs") or ""),
        description=str(block.get("description") or ""))
    out = [LabTest(**_base(truth, block, "atterberg", unit), result=result,
                   fields=_fields(block, _COMMON + (
                       "ll", "pl", "pi", "flow_curve", "pl_trials", "wc_pct",
                       "fines")))]
    # An Atterberg sheet that also prints a fines percentage has reported a
    # gradation on the same specimen. Two tests, which is what the reader is
    # told to do with a sheet that carries two.
    fines = number(block, "fines", "fines_percent", "percent_fines")
    if fines is not None:
        out.append(LabTest(**_base(truth, block, "gradation", unit),
                           result=GradationResult(
                               fines_percent=fines,
                               uscs=str(block.get("uscs") or ""),
                               description=str(block.get("description") or ""))))
    return out


def _gradation(truth, block, unit) -> List[LabTest]:
    result = GradationResult(
        percent_passing=_grading_of(block),
        d10=quantity(block, "d10"), d30=quantity(block, "d30"),
        d50=quantity(block, "d50"), d60=quantity(block, "d60"),
        d85=quantity(block, "d85"), d90=quantity(block, "d90"),
        d100=quantity(block, "d100"),
        cu=number(block, "cu"), cc=number(block, "cc"),
        cobbles_percent=number(block, "cobbles"),
        gravel_percent=number(block, "gravel"),
        sand_percent=number(block, "sand"),
        silt_percent=number(block, "silt"),
        clay_percent=number(block, "clay"),
        fines_percent=number(block, "fines"),
        hydrometer=bool(block.get("hydrometer")),
        water_content=number(block, "wc", "water_content"),
        uscs=str(block.get("uscs") or ""),
        description=str(block.get("description") or ""))
    # The d-values on these sheets are millimetres whether or not the hand
    # wrote a unit into the key, because a grading is always millimetres.
    for name in ("d10", "d30", "d50", "d60", "d85", "d90", "d100"):
        if getattr(result, name) is None:
            raw = number(block, name)
            if raw is not None:
                setattr(result, name, Quantity(value=raw, unit="mm"))
    out = [LabTest(**_base(truth, block, "gradation", unit), result=result,
                   fields=_fields(block, _COMMON + (
                       "percent_finer", "percent_passing_mm", "ll", "pl",
                       "pi")))]
    ll = number(block, "ll")
    pl = number(block, "pl")
    pi = number(block, "pi")
    if ll is not None or pl is not None or pi is not None:
        out.append(LabTest(**_base(truth, block, "atterberg", unit),
                           result=AtterbergResult(
                               ll=ll, pl=pl, pi=pi,
                               uscs=str(block.get("uscs") or ""),
                               description=str(block.get("description") or ""))))
    return out


def _consolidation(truth, block, unit) -> List[LabTest]:
    pairs, x_unit, y_unit = _points(block, "curve_pressure", "consolidation")
    points = [ConsolidationPoint(
        stress=Quantity(value=x, unit=x_unit or "kPa"),
        strain_percent=(y if y_unit == "%" else None),
        void_ratio=(None if y_unit == "%" else y),
        stage="load") for x, y in pairs]
    rebound, r_x, r_y = _points(block, "rebound", "unload")
    points += [ConsolidationPoint(
        stress=Quantity(value=x, unit=r_x or x_unit or "kPa"),
        strain_percent=(y if (r_y or y_unit) == "%" else None),
        void_ratio=(None if (r_y or y_unit) == "%" else y),
        stage="rebound") for x, y in rebound]
    swell_at = None
    for key in block:
        match = re.match(r"swell_pct_at_(\d+(?:\.\d+)?)([a-z0-9_]*)$", key)
        if match:
            swell_at = Quantity(
                value=float(match.group(1)),
                unit=UNIT_SUFFIX.get(match.group(2), "") or "psf")
            break
    result = ConsolidationResult(
        test_type=str(block.get("test_type") or "swell"),
        points=points,
        swell_percent=number(block, "swell", "swell_pct")
        or _swell_at_percent(block),
        swell_at=swell_at,
        swell_pressure=quantity(block, "swell_pressure"),
        pc=quantity(block, "pc", "preconsolidation"),
        cc=number(block, "cc", "compression_index"),
        cr=number(block, "cr", "recompression_index"),
        e0=number(block, "e0", "void_ratio"),
        dry_unit_weight=quantity(block, "dry_unit_weight", "dry_density"),
        wc=number(block, "wc", "water_content", "moisture"),
        uscs=str(block.get("uscs") or ""),
        description=str(block.get("description") or ""))
    return [LabTest(**_base(truth, block, "swell_consolidation", unit),
                    result=result,
                    fields=_fields(block, _COMMON + ("test_type",)))]


def _swell_at_percent(block: Dict[str, Any]) -> Optional[float]:
    for key, value in block.items():
        if key.startswith("swell_pct_at_"):
            return _number(value)
    return None


def _strength(truth, block, unit, kind: str) -> List[LabTest]:
    specimens: List[StrengthSpecimen] = []
    for n, raw in enumerate(block.get("specimens") or [], start=1):
        specimens.append(StrengthSpecimen(
            specimen_id=str(raw.get("n") or raw.get("specimen") or n),
            confining=quantity(raw, "confining", "effective_confining",
                               "cell_pressure", "normal"),
            peak_deviator=quantity(raw, "peak_deviator", "peak_shear",
                                   "max_deviator", "q_f"),
            strain_at_peak_percent=number(raw, "strain_at_peak",
                                          "strain_at_failure"),
            pore_pressure=quantity(raw, "max_pore_pressure_change",
                                   "pore_pressure", "pwp_at_failure"),
            stress_ratio=number(raw, "max_effective_stress_ratio",
                                "stress_ratio", "max_stress_ratio"),
            wc=number(raw, "wc", "moisture"),
            dry_density=quantity(raw, "dry_density"),
            wet_density=quantity(raw, "bulk_density", "wet_density"),
            height=quantity(raw, "height"),
            diameter=quantity(raw, "diameter")))
    # A sheet that reports one specimen as a set of nested stages rather than
    # as a specimens list: take what each stage states and leave the rest in
    # the fields, which is where everything unmodelled goes.
    if not specimens:
        flat: Dict[str, Any] = {}
        for key in ("before", "after", "saturation", "consolidation",
                    "compression", "first_row", "last_row_approx", "specimen"):
            nested = block.get(key)
            if isinstance(nested, dict):
                flat.update({f"{k}": v for k, v in nested.items()
                             if k not in flat})
        flat.update({k: v for k, v in block.items()
                     if not isinstance(v, (dict, list))})
        specimen = StrengthSpecimen(
            specimen_id=str(block.get("specimen") or block.get("stage") or ""),
            confining=quantity(flat, "effective_confining", "cell_pressure",
                               "confining"),
            peak_deviator=quantity(flat, "q_f", "max_deviator", "dev",
                                   "peak_deviator", "load_at_failure"),
            strain_at_peak_percent=number(flat, "strain_at_failure",
                                          "strain_at_peak", "strain"),
            pore_pressure=quantity(flat, "pwp_at_failure", "final_pwp",
                                   "pore_pressure"),
            stress_ratio=number(flat, "max_stress_ratio", "ratio"),
            wc=number(flat, "moisture", "wc"),
            dry_density=quantity(flat, "dry_density"),
            wet_density=quantity(flat, "bulk_density", "wet_density"),
            height=quantity(flat, "height", "length"),
            diameter=quantity(flat, "diameter"))
        if any(getattr(specimen, name) is not None for name in
               ("confining", "peak_deviator", "strain_at_peak_percent",
                "pore_pressure", "stress_ratio", "wc", "dry_density",
                "wet_density", "height", "diameter")):
            specimens = [specimen]
    pairs, x_unit, y_unit = _points(block, "points_normal", "envelope")
    points = [ShearPoint(x=Quantity(value=x, unit=x_unit or "kPa"),
                         y=Quantity(value=y, unit=y_unit or x_unit or "kPa"))
              for x, y in pairs]
    phi = number(block, "phi_deg") or number(block, "phi")
    result = StrengthResult(
        kind=kind,
        test_type=str(block.get("test_type") or truth.get("test_type") or ""),
        specimens=specimens,
        c=quantity(block, "c", "cohesion"),
        phi_deg=phi,
        qu=quantity(block, "ucs", "qu", "unconfined"),
        su=quantity(block, "su", "cu"),
        strain_at_failure_percent=number(block, "strain_at_failure"),
        points=points,
        wc=number(block, "wc", "moisture"),
        dry_density=quantity(block, "dry_density", "dry_unit_weight"),
        wet_density=quantity(block, "wet_density", "bulk_density"),
        rock_type=str(block.get("rock_type") or ""),
        weathering=str(block.get("weathering_grade")
                       or block.get("weathering") or ""),
        uscs=str(block.get("uscs") or ""),
        description=str(block.get("description") or ""))
    return [LabTest(**_base(truth, block, kind, unit), result=result,
                    fields=_fields(block, _COMMON + ("test_type",)))]


def _compaction(truth, block, unit) -> List[LabTest]:
    pairs, x_unit, y_unit = _points(block, "points_wc", "compaction")
    points = [CompactionPoint(water_content=x,
                              dry_density=Quantity(value=y,
                                                   unit=y_unit or "g/cm3"))
              for x, y in pairs]
    result = CompactionResult(
        points=points,
        max_dry_density=quantity(block, "max_dry_density"),
        optimum_wc=number(block, "optimum_wc", "optimum"),
        method=str(block.get("method") or block.get("test_type") or ""),
        mould_volume=quantity(block, "mould_volume"),
        blows_per_layer=(int(x) if (x := number(block, "blows_per_layer"))
                         is not None else None),
        layers=(int(x) if (x := number(block, "layers")) is not None else None),
        rammer_mass=quantity(block, "rammer_mass", "mould_mass"),
        description=str(block.get("description") or ""))
    return [LabTest(**_base(truth, block, "compaction", unit), result=result,
                    fields=_fields(block, _COMMON + ("raw_rows",)))]


def _cbr(truth, block, unit) -> List[LabTest]:
    result = CBRResult(
        cbr_percent=number(block, "cbr"),
        cbr_at_0_1in=number(block, "cbr_0_1", "cbr_at_0_1in"),
        cbr_at_0_2in=number(block, "cbr_0_2", "cbr_at_0_2in"),
        swell_percent=number(block, "swell"),
        soaked=block.get("soaked"),
        surcharge=quantity(block, "surcharge"),
        dry_density=quantity(block, "dry_density"),
        wc=number(block, "wc", "moisture"),
        compaction_percent=number(block, "compaction"),
        description=str(block.get("description") or ""))
    return [LabTest(**_base(truth, block, "cbr", unit), result=result,
                    fields=_fields(block, _COMMON))]


def _moisture(truth, block, unit, kind: str) -> List[LabTest]:
    each = [float(v) for v in (block.get("wc_pct_each") or [])]
    result = MoistureDensityResult(
        kind=kind,
        wc=number(block, "wc_pct_avg") or number(block, "wc", "moisture",
                                                 "water_content"),
        water_contents=each,
        wet_density=quantity(block, "bulk_density", "wet_density"),
        dry_density=quantity(block, "dry_density", "dry_unit_weight"),
        specific_gravity=number(block, "gs", "specific_gravity"),
        void_ratio=number(block, "e0", "void_ratio"),
        saturation_percent=number(block, "saturation"),
        ash_percent=number(block, "ash"),
        organic_percent=number(block, "organic"),
        uscs=str(block.get("uscs") or ""),
        description=str(block.get("description") or ""))
    return [LabTest(**_base(truth, block, kind, unit), result=result,
                    fields=_fields(block, _COMMON + ("wc_pct_each",)))]


def _chemical(truth, block, unit) -> List[LabTest]:
    redox = block.get("redox_avg_mV")
    result = ChemicalResult(
        pH=reported(block, "ph", "pH"),
        resistivity=reported(block, "resistivity",
                             "resistivity_as_received"),
        resistivity_minimum=reported(block, "min_resistivity",
                                     "resistivity_min"),
        sulfate=reported(block, "sulfate"),
        chloride=reported(block, "chloride"),
        sulfides=reported(block, "sulfides", "sulfide"),
        redox=(Quantity(value=float(redox), unit="mV") if redox is not None
               else reported(block, "redox")),
        total_salts=reported(block, "total_salts"),
        conductivity=reported(block, "conductivity"),
        temperature=reported(block, "temperature"),
        wc=reported(block, "wc", "wc_as_received", "moisture"),
        reporting_limit=reported(block, "reporting_limit"),
        lab_sample_id=str(block.get("lab_sample_id") or ""),
        description=str(block.get("description") or ""))
    return [LabTest(**_base(truth, block, "chemical", unit), result=result,
                    fields=_fields(block, _COMMON + ("redox_mV",)))]


def _summary(truth, unit) -> List[LabTest]:
    rows: List[SummaryRow] = []
    for raw in truth.get("rows") or []:
        pl_value: Any = raw.get("pl")
        pl: Any = None
        if pl_value is not None:
            pl = _number(pl_value)
            if pl is None:
                pl = str(pl_value)
        rows.append(SummaryRow(
            investigation_id=str(raw.get("investigation_id") or ""),
            sample_id=str(raw.get("sample_id") or ""),
            depth_top=_depth(raw.get("depth_top"), unit),
            depth_bottom=_depth(raw.get("depth_bottom"), unit),
            elevation_top=_depth(raw.get("elev_top"), unit),
            sample_type=str(raw.get("sample_type") or ""),
            description=str(raw.get("description") or ""),
            uscs=str(raw.get("uscs") or ""),
            stratum=str(raw.get("stratum") or ""),
            lab=str(raw.get("lab") or ""),
            wc=number(raw, "wc", "moisture"),
            ll=number(raw, "ll"),
            pl=pl,
            pi=number(raw, "pi"),
            percent_passing=_summary_sieves(raw),
            fines_percent=number(raw, "fines"),
            sand_percent=number(raw, "sand"),
            gravel_percent=number(raw, "gravel"),
            silt_clay_percent=number(raw, "clay_silt", "silt_clay"),
            wet_density=quantity(raw, "bulk_density", "wet_density"),
            dry_density=quantity(raw, "dry_density"),
            max_dry_density=quantity(raw, "max_dry_density"),
            optimum_wc=number(raw, "optimum_wc"),
            qu=quantity(raw, "qu", "ucs"),
            su=quantity(raw, "su", "cu"),
            c=quantity(raw, "c", "cohesion"),
            phi_deg=number(raw, "phi_deg", "phi"),
            swell_percent=number(raw, "swell"),
            organic_percent=number(raw, "organic"),
            pH=reported(raw, "ph", "pH"),
            resistivity=reported(raw, "resistivity"),
            sulfate=reported(raw, "sulfate"),
            chloride=reported(raw, "chloride"),
            sulfides=reported(raw, "sulfides", "sulfide"),
            redox=reported(raw, "redox"),
            other=_row_other(raw)))
    block = {"investigation_id": "", "description": truth.get("note") or ""}
    return [LabTest(**_base(truth, block, "summary_table", unit),
                    result=SummaryTableResult(
                        rows=rows, title=str(truth.get("note") or ""),
                        sheet=str((truth.get("fields") or {}).get("sheet")
                                  or "")),
                    fields={k: str(v) for k, v in
                            (truth.get("fields") or {}).items()})]


#: Keys a summary row carries that a :class:`SummaryRow` field already holds.
#: Everything else the row printed goes into ``other``, so a column the model
#: has no field for is kept rather than dropped.
_ROW_TAKEN = {
    "investigation_id", "sample_id", "depth_top", "depth_bottom", "elev_top",
    "sample_type", "description", "uscs", "stratum", "lab",
}


def _row_other(raw: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Every column of a summary row that no field of the record took."""
    out: List[Tuple[str, str]] = []
    for key, value in raw.items():
        if key in _ROW_TAKEN or value is None:
            continue
        if isinstance(value, dict):
            out.extend((f"{key}.{k}", str(v)) for k, v in value.items())
            continue
        if isinstance(value, list):
            continue
        out.append((key, str(value)))
    return out


def _summary_sieves(raw: Dict[str, Any]) -> List[SievePoint]:
    """The sieve columns a summary row carries, however they are spelled."""
    if isinstance(raw.get("passing"), dict):
        return _sieve_points(raw["passing"], keyed_mm=True)
    out: List[SievePoint] = []
    for key, value in raw.items():
        match = re.match(r"pass(?:ing)?_(\d+)$", key)
        if not match:
            continue
        percent = _number(value)
        if percent is None:
            continue
        sieve = f"No. {match.group(1)}"
        millimetres = SIEVE_MM.get(sieve.lower())
        out.append(SievePoint(
            percent_passing=max(0.0, min(100.0, percent)),
            size=(Quantity(value=millimetres, unit="mm")
                  if millimetres is not None else None),
            sieve=sieve))
    out.sort(key=lambda p: -(p.size.value if p.size is not None else 0.0))
    return out


#: ``truth kind -> how to build it``. The kinds are the vocabulary of the
#: truth README, and every one of them is here.
_BUILDERS = {
    "atterberg": lambda t, b, u: _atterberg(t, b, u),
    "gradation": lambda t, b, u: _gradation(t, b, u),
    "swell_consolidation": lambda t, b, u: _consolidation(t, b, u),
    "consolidation": lambda t, b, u: _consolidation(t, b, u),
    "triaxial": lambda t, b, u: _strength(t, b, u, "triaxial"),
    "direct_shear": lambda t, b, u: _strength(t, b, u, "direct_shear"),
    "unconfined": lambda t, b, u: _strength(t, b, u, "unconfined"),
    "unconfined_rock": lambda t, b, u: _strength(t, b, u, "unconfined_rock"),
    "compaction": lambda t, b, u: _compaction(t, b, u),
    "cbr": lambda t, b, u: _cbr(t, b, u),
    "moisture_content": lambda t, b, u: _moisture(t, b, u, "moisture_content"),
    "density": lambda t, b, u: _moisture(t, b, u, "density"),
    "organic_content": lambda t, b, u: _moisture(t, b, u, "organic_content"),
    "chemical": lambda t, b, u: _chemical(t, b, u),
}


def lab_tests_from_truth(truth: Dict[str, Any]) -> List[LabTest]:
    """Every laboratory test one truth sheet describes.

    A sheet with several specimens becomes several tests; a sheet carrying
    both a grading and Atterberg limits becomes two tests on the same
    specimen; a summary table becomes ONE test holding a row per specimen,
    which is what the record says a summary table is.
    """
    kind = str(truth.get("kind") or "other")
    unit = depth_unit_of(truth)
    if kind == "summary_table":
        return _summary(truth, unit)
    builder = _BUILDERS.get(kind)
    out: List[LabTest] = []
    for block in (truth.get("tests") or []):
        if builder is None:
            out.append(LabTest(
                **_base(truth, block, "other", unit),
                result=OtherResult(fields=_fields(block, ())),
                fields=_fields(block, _COMMON)))
            continue
        out.extend(builder(truth, block, unit))
    return out

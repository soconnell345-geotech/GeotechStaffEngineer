"""Shared input-resolution helpers for the pavement_design module.

Every helper resolves one design input to a single value and returns the
value plus the provenance (reference string and any note about a defaulted
or midpoint-selected quantity), so the orchestrators can carry a complete
audit trail. All physics comes from ``geotech_references.aashto_1993`` --
this module never re-derives digitized math.

UNITS: US customary (psi, pci, inches, kips) -- documented exception to the
repo SI rule; see DESIGN.md.
"""

import math

from geotech_references.aashto_1993 import equations as _eq
from geotech_references.aashto_1993 import tables as _tb

# So defaults: the AASHO Road Test performance prediction error (0.25 rigid,
# 0.35 flexible) "corresponds to a total standard deviation for traffic of
# 0.35 and 0.45 for rigid and flexible pavements" (Part II Section 2.1.3
# step 3, printed II-9/10); the guide's ranges are flexible 0.40-0.50 and
# rigid 0.30-0.40 (Part I Section 4.3, printed I-62).
SO_DEFAULT = {"flexible": 0.45, "rigid": 0.35}
_SO_BASIS = ("the guide's total standard deviation with traffic for "
             "{pt} pavements (Section 2.1.3 step 3, printed II-9/10)")
PO_DEFAULT = {"flexible": 4.2, "rigid": 4.5}


def round_up(x, increment):
    """Round x UP to the next multiple of increment (guide practice:
    layer thicknesses to the nearest half inch, never rounded down)."""
    if increment <= 0:
        raise ValueError(f"increment must be > 0, got {increment}")
    n = math.ceil(x / increment - 1e-9)
    return round(max(n, 0) * increment, 6)


def resolve_reliability(reliability_pct=None, zr=None):
    """ZR from a reliability level (Table 4.1), or accept a direct ZR.

    Returns (reliability_pct, zr, reference, notes).
    """
    notes = []
    if zr is not None:
        if reliability_pct is not None:
            notes.append(
                "Both reliability_pct and zr supplied; the direct zr governs."
            )
        return reliability_pct, float(zr), None, notes
    if reliability_pct is None:
        raise ValueError("Provide reliability_pct (50-99.99) or a direct zr.")
    r = _tb.standard_normal_deviate_zr(reliability_pct)
    return reliability_pct, r["zr"], r["reference"], notes


def resolve_so(pavement_type, so=None):
    """Overall standard deviation So; defaults to the guide's total
    standard deviation with traffic for the pavement type (0.45 flexible,
    0.35 rigid; printed II-9/10), the middle of its recommended range.

    Returns (so, reference, notes).
    """
    rng = _tb.overall_standard_deviation_range(pavement_type)
    notes = []
    if so is None:
        so = SO_DEFAULT[pavement_type]
        notes.append(
            f"So defaulted to {so}: "
            + _SO_BASIS.format(pt=pavement_type)
            + f"; the recommended range is {rng['so_min']}-{rng['so_max']} "
            "(printed I-62)."
        )
    elif not (rng["so_min"] - 0.05 <= so <= rng["so_max"] + 0.05):
        notes.append(
            f"So = {so} is outside the guide's recommended "
            f"{rng['so_min']}-{rng['so_max']} range for {pavement_type}."
        )
    return float(so), rng["reference"], notes


def resolve_delta_psi(pavement_type, delta_psi=None, po=None, pt=2.5):
    """Design serviceability loss dPSI = po - pt (Section 2.2.1).

    Returns (delta_psi, po, pt, reference, notes).
    """
    notes = []
    if po is None:
        po = PO_DEFAULT[pavement_type]
    if delta_psi is not None:
        return float(delta_psi), po, pt, None, notes
    r = _eq.design_serviceability_loss(po, pt)
    return r["delta_psi"], po, pt, r["reference"], notes


def resolve_effective_mr(mr_psi=None, monthly_mr_psi=None):
    """Effective roadbed resilient modulus: direct value or the seasonal
    relative-damage average (Figure 2.3/2.4).

    Returns (effective_mr_psi, reference, notes, detail_or_None).
    """
    notes = []
    if monthly_mr_psi:
        r = _eq.effective_roadbed_resilient_modulus(monthly_mr_psi)
        if mr_psi is not None:
            notes.append(
                "Both mr_psi and monthly_mr_psi supplied; the seasonal "
                "effective MR governs."
            )
        return r["effective_mr_psi"], r["reference"], notes, r
    if mr_psi is None:
        raise ValueError(
            "Provide mr_psi (effective roadbed resilient modulus, psi) or "
            "monthly_mr_psi (seasonal list for the Figure 2.3/2.4 average)."
        )
    if mr_psi <= 0:
        raise ValueError(f"mr_psi must be > 0, got {mr_psi}")
    return float(mr_psi), None, notes, None


def midpoint_range(lo, hi):
    return round(0.5 * (lo + hi), 4)


def add_ref(references, ref):
    """Append a reference string if new (order-preserving dedup)."""
    if ref and ref not in references:
        references.append(ref)

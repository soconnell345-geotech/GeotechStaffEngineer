"""AASHTO 1993 flexible pavement design (Part II, Ch 3; Figures 3.1/3.2).

Orchestrates the digitized guide equations in
``geotech_references.aashto_1993`` into a complete flexible design:
reliability (ZR/So) -> serviceability loss -> effective roadbed MR ->
required SN (Figure 3.1 solve) -> layered split D1/D2/D3 (Figure 3.2
cascade with re-solves at each foundation modulus) -> practical rounding
and Section 3.1.4 minimum thicknesses -> forward adequacy check.

UNITS: US customary (psi, inches, kips, ESALs) -- documented exception to
the repo SI rule; see DESIGN.md.
"""

from dataclasses import dataclass

from geotech_references.aashto_1993 import equations as _eq
from geotech_references.aashto_1993 import tables as _tb

from .common import (add_ref, midpoint_range, resolve_delta_psi,
                     resolve_effective_mr, resolve_reliability, resolve_so,
                     round_up)
from .results import FlexiblePavementResult

LAYER_TYPES = ("asphalt", "granular_base", "granular_subbase",
               "cement_treated_base", "bituminous_treated_base")
_UNBOUND = ("granular_base", "granular_subbase")


@dataclass
class PavementLayer:
    """One course of the flexible section, top-down.

    Give the material property matching the layer type -- ``modulus_psi``
    for asphalt (EAC at 68 F) and granular layers (EBS/ESB), or
    ``ucs_7day_psi`` / ``marshall_stability_lb`` for cement- /
    bituminous-treated bases -- or override the layer coefficient ``a``
    directly. Drainage ``m`` applies to UNBOUND (granular) layers only
    (Table 2.4); give it directly or via ``drainage_quality`` +
    ``pct_saturation_time`` (midpoint of the table range is used).
    ``thickness_in`` set on every layer switches the analysis to check
    mode. In design mode, non-surface layers also need ``modulus_psi``
    (used as the foundation modulus for the Figure 3.2 re-solve above
    that layer).
    """

    layer_type: str
    modulus_psi: float = None
    ucs_7day_psi: float = None
    marshall_stability_lb: float = None
    a: float = None
    m: float = None
    drainage_quality: str = None
    pct_saturation_time: str = None
    thickness_in: float = None
    min_thickness_in: float = None


def _resolve_a(layer: PavementLayer, references, warnings):
    """Layer coefficient for one course; returns (a, basis string)."""
    lt = layer.layer_type
    if layer.a is not None:
        if layer.a <= 0:
            raise ValueError(f"Layer coefficient a must be > 0, got {layer.a}")
        return float(layer.a), "user-specified"
    if lt == "asphalt":
        if layer.modulus_psi is None:
            raise ValueError(
                "asphalt layer needs modulus_psi (EAC at 68 F) or a direct "
                "'a' coefficient."
            )
        r = _tb.layer_coefficient_a1_asphalt(layer.modulus_psi)
        add_ref(references, r["reference"])
        if "note" in r:
            warnings.append(f"a1 (asphalt): {r['note']}")
        return r["a1"], "Figure 2.5 chart read (EAC)"
    if lt == "granular_base":
        if layer.modulus_psi is None:
            raise ValueError(
                "granular_base layer needs modulus_psi (EBS) or a direct 'a'."
            )
        r = _eq.layer_coefficient_a2_granular_base(layer.modulus_psi)
        add_ref(references, r["reference"])
        return r["a2"], "printed regression (EBS)"
    if lt == "granular_subbase":
        if layer.modulus_psi is None:
            raise ValueError(
                "granular_subbase layer needs modulus_psi (ESB) or a direct 'a'."
            )
        r = _eq.layer_coefficient_a3_granular_subbase(layer.modulus_psi)
        add_ref(references, r["reference"])
        return r["a3"], "printed regression (ESB)"
    if lt == "cement_treated_base":
        if layer.ucs_7day_psi is None:
            raise ValueError(
                "cement_treated_base layer needs ucs_7day_psi (7-day UCS) "
                "or a direct 'a'."
            )
        r = _tb.layer_coefficient_a2_cement_treated(layer.ucs_7day_psi)
        add_ref(references, r["reference"])
        warnings.append(
            "a2 (cement-treated) is a sparse 3-point chart read (Figure 2.8)."
        )
        return r["a2"], "Figure 2.8 chart read (UCS)"
    if lt == "bituminous_treated_base":
        if layer.marshall_stability_lb is None:
            raise ValueError(
                "bituminous_treated_base layer needs marshall_stability_lb "
                "or a direct 'a'."
            )
        r = _tb.layer_coefficient_a2_bituminous_treated(
            layer.marshall_stability_lb)
        add_ref(references, r["reference"])
        warnings.append(
            "a2 (bituminous-treated) is a sparse 3-point chart read "
            "(Figure 2.9)."
        )
        return r["a2"], "Figure 2.9 chart read (Marshall)"
    raise ValueError(
        f"Unknown layer_type '{lt}'. Use one of: {', '.join(LAYER_TYPES)}"
    )


def _resolve_m(layer: PavementLayer, references, notes):
    """Drainage coefficient for one course; returns (m, basis string)."""
    if layer.layer_type not in _UNBOUND:
        if layer.m is not None and abs(layer.m - 1.0) > 1e-9:
            notes.append(
                f"Drainage m on '{layer.layer_type}' ignored (Table 2.4 "
                "applies to UNTREATED base/subbase only); m = 1.0 used."
            )
        return 1.0, "not applicable (bound layer)"
    if layer.m is not None:
        if layer.m <= 0:
            raise ValueError(f"Drainage m must be > 0, got {layer.m}")
        return float(layer.m), "user-specified"
    if layer.drainage_quality is not None:
        pct = layer.pct_saturation_time or "1-5%"
        if layer.pct_saturation_time is None:
            notes.append(
                f"pct_saturation_time defaulted to '1-5%' for the "
                f"{layer.layer_type} drainage m lookup."
            )
        r = _tb.drainage_mi_flexible(layer.drainage_quality, pct)
        add_ref(references, r["reference"])
        m = midpoint_range(r["mi_min"], r["mi_max"])
        return m, (f"Table 2.4 midpoint ({r['mi_min']}-{r['mi_max']}, "
                   f"quality={r['quality']}, {pct})")
    notes.append(
        f"Drainage m defaulted to 1.0 for the {layer.layer_type} (AASHO "
        "Road Test conditions, Table 2.4 'fair')."
    )
    return 1.0, "default 1.0 (AASHO Road Test conditions)"


def design_flexible_pavement(
    w18,
    reliability_pct=None,
    zr=None,
    so=None,
    mr_psi=None,
    monthly_mr_psi=None,
    pt=2.5,
    po=None,
    delta_psi=None,
    layers=None,
    thickness_increment_in=0.5,
    enforce_minimums=True,
) -> FlexiblePavementResult:
    """Complete AASHTO 1993 flexible pavement design or adequacy check.

    Modes (inferred from the layers):

    - **design** -- no layer has ``thickness_in``: solves the required SN
      over the roadbed (Figure 3.1), re-solves over each intermediate
      foundation (base/subbase modulus) per Figure 3.2, splits thicknesses
      top-down with round-as-you-go practical rounding, and enforces the
      Section 3.1.4 traffic-based minimums.
    - **check** -- every layer has ``thickness_in``: computes the provided
      SN and compares to the requirement; no thicknesses are changed.

    Parameters
    ----------
    w18 : float
        Design-lane cumulative 18-kip ESALs (see
        ``traffic.compute_design_esals``).
    reliability_pct : float, optional
        Design reliability R percent (Table 4.1 -> ZR); or give ``zr``.
    zr : float, optional
        Standard normal deviate directly (overrides reliability_pct).
    so : float, optional
        Overall standard deviation; defaults to 0.45 (midpoint of the
        guide's flexible range 0.40-0.50).
    mr_psi : float, optional
        Effective roadbed resilient modulus, psi.
    monthly_mr_psi : list, optional
        Seasonal roadbed MR values -> effective MR via the Figure 2.3/2.4
        relative-damage average (supersedes mr_psi if both given).
    pt : float, optional
        Terminal serviceability (default 2.5); po defaults to 4.2.
    po, delta_psi : float, optional
        Override the initial serviceability or give dPSI directly.
    layers : list of PavementLayer (or dicts)
        Section top-down; first layer must be 'asphalt'. 1-3 layers
        (full-depth AC / AC+base / AC+base+subbase).
    thickness_increment_in : float, optional
        Practical rounding increment for design mode (default 0.5 in,
        rounded UP, guide practice).
    enforce_minimums : bool, optional
        Apply the Section 3.1.4 minimum AC/base thicknesses (design mode).

    Returns
    -------
    FlexiblePavementResult
    """
    if w18 is None or w18 <= 0:
        raise ValueError(f"w18 must be > 0, got {w18}")
    if not layers:
        raise ValueError("Provide at least one PavementLayer (asphalt surface).")
    layers = [PavementLayer(**lay) if isinstance(lay, dict) else lay
              for lay in layers]
    if len(layers) > 3:
        raise ValueError(
            "At most 3 layers (surface, base, subbase) are supported by the "
            "Figure 3.2 procedure."
        )
    if layers[0].layer_type != "asphalt":
        raise ValueError(
            f"The first (top) layer must be 'asphalt', got "
            f"'{layers[0].layer_type}'."
        )
    for lay in layers[1:]:
        if lay.layer_type == "asphalt":
            raise ValueError("Only the top layer may be 'asphalt'.")
    if layers[-1].layer_type == "granular_subbase" and len(layers) == 2:
        raise ValueError(
            "A subbase requires a base course above it (asphalt directly on "
            "subbase is not a Figure 3.2 section); use layer_type "
            "'granular_base' for a 2-layer section."
        )

    references = []
    notes = []
    warnings = []

    reliability_pct, zr_val, ref, n = resolve_reliability(reliability_pct, zr)
    add_ref(references, ref)
    notes.extend(n)
    so_val, ref, n = resolve_so("flexible", so)
    add_ref(references, ref)
    notes.extend(n)
    dpsi, po_val, pt_val, ref, n = resolve_delta_psi(
        "flexible", delta_psi, po, pt)
    add_ref(references, ref)
    notes.extend(n)
    mr_eff, ref, n, _detail = resolve_effective_mr(mr_psi, monthly_mr_psi)
    add_ref(references, ref)
    notes.extend(n)

    # Per-layer coefficients.
    resolved = []
    for lay in layers:
        a, a_basis = _resolve_a(lay, references, warnings)
        m, m_basis = _resolve_m(lay, references, notes)
        resolved.append({
            "layer_type": lay.layer_type, "a": a, "a_basis": a_basis,
            "m": m, "m_basis": m_basis, "layer": lay,
        })

    thicknesses_given = [lay.thickness_in is not None for lay in layers]
    if all(thicknesses_given):
        mode = "check"
    elif not any(thicknesses_given):
        mode = "design"
    else:
        raise ValueError(
            "Give thickness_in on EVERY layer (check mode) or on none "
            "(design mode); a mixed section is ambiguous."
        )

    # Required SN over each successive foundation (Figure 3.2): the SN over
    # layer i+1 uses that layer's modulus as the design "roadbed"; the last
    # solve uses the actual roadbed effective MR. Design mode requires the
    # full stack; in check mode intermediate solves are informational and
    # included only where a modulus is available.
    def _solve_sn(foundation_mr):
        r = _eq.flexible_sn_from_w18(w18, zr_val, so_val, dpsi, foundation_mr)
        add_ref(references, r["reference"])
        return r["sn"]

    sn_stack = []
    for i, entry in enumerate(resolved[:-1]):
        below = resolved[i + 1]
        found_mr = below["layer"].modulus_psi
        if found_mr is None:
            if mode == "design":
                raise ValueError(
                    f"Design mode needs modulus_psi on the "
                    f"'{below['layer_type']}' layer (used as the foundation "
                    "modulus for the Figure 3.2 re-solve above it); give "
                    "modulus_psi, or supply thickness_in on every layer for "
                    "a check instead."
                )
            continue
        sn_stack.append({
            "over": below["layer_type"], "foundation_mr_psi": found_mr,
            "sn_required": _solve_sn(found_mr),
        })
    sn_roadbed = _solve_sn(mr_eff)
    sn_stack.append({
        "over": "roadbed", "foundation_mr_psi": mr_eff,
        "sn_required": sn_roadbed,
    })

    minimums_applied = {}
    if mode == "design":
        # Section 3.1.4 practical minimums by traffic level.
        min_ac = min_base = 0.0
        if enforce_minimums:
            mt = _tb.minimum_layer_thickness(w18)
            add_ref(references, mt["reference"])
            min_ac = mt["asphalt_concrete_min_in"]
            min_base = mt["aggregate_base_min_in"]
            if "note" in mt:
                notes.append(f"Section 3.1.4 minimum AC: {mt['note']}.")

        # Round-as-you-go top-down split (Figure 3.2 with practical
        # rounding: each layer is rounded UP before the next is sized, so
        # rounding surplus carries down as extra capacity).
        sn_carried = 0.0
        for i, entry in enumerate(resolved):
            sn_req_here = sn_stack[i]["sn_required"]
            need = max(sn_req_here - sn_carried, 0.0)
            d_raw = need / (entry["a"] * entry["m"])
            d = round_up(d_raw, thickness_increment_in)
            floor = 0.0
            if entry["layer_type"] == "asphalt":
                floor = max(min_ac, entry["layer"].min_thickness_in or 0.0)
            elif entry["layer_type"] == "granular_base":
                floor = max(min_base, entry["layer"].min_thickness_in or 0.0)
            elif entry["layer"].min_thickness_in:
                floor = entry["layer"].min_thickness_in
            if floor > d:
                d = round_up(floor, thickness_increment_in)
                minimums_applied[entry["layer_type"]] = floor
            if d <= 0 and need > 0:
                d = thickness_increment_in
            entry["thickness_in"] = d
            entry["d_unrounded_in"] = round(d_raw, 3)
            sn_carried += entry["a"] * d * entry["m"]
            entry["sn_cumulative"] = round(sn_carried, 3)
        add_ref(references,
                "AASHTO 1993 Guide, Figure 3.2 (pdf_page 125, printed II-36)")
        if minimums_applied:
            notes.append(
                f"Section 3.1.4 minimum thickness governed: "
                f"{minimums_applied}."
            )
    else:
        sn_carried = 0.0
        for entry in resolved:
            entry["thickness_in"] = float(entry["layer"].thickness_in)
            if entry["thickness_in"] < 0:
                raise ValueError("thickness_in must be >= 0")
            sn_carried += entry["a"] * entry["thickness_in"] * entry["m"]
            entry["sn_cumulative"] = round(sn_carried, 3)

    # Provided SN (Section 3.1.4 composition) and forward adequacy check.
    kw = {"a1": resolved[0]["a"], "d1": resolved[0]["thickness_in"]}
    if len(resolved) > 1:
        kw.update(a2=resolved[1]["a"], d2=resolved[1]["thickness_in"],
                  m2=resolved[1]["m"])
    if len(resolved) > 2:
        kw.update(a3=resolved[2]["a"], d3=resolved[2]["thickness_in"],
                  m3=resolved[2]["m"])
    sn_comp = _eq.structural_number(**kw)
    add_ref(references, sn_comp["reference"])
    sn_provided = sn_comp["sn"]

    fwd = _eq.flexible_w18_from_sn(sn_provided, zr_val, so_val, dpsi, mr_eff)
    add_ref(references, fwd["reference"])
    w18_capacity = fwd["w18"]
    adequate = (sn_provided >= sn_roadbed - 1e-9) and (w18_capacity >= w18)

    layer_rows = [{k: v for k, v in entry.items() if k != "layer"}
                  for entry in resolved]
    return FlexiblePavementResult(
        mode=mode,
        w18=w18,
        reliability_pct=reliability_pct,
        zr=zr_val,
        so=so_val,
        po=po_val,
        pt=pt_val,
        delta_psi=dpsi,
        effective_mr_psi=mr_eff,
        sn_required=sn_roadbed,
        sn_provided=sn_provided,
        layers=layer_rows,
        sn_stack=sn_stack,
        w18_capacity=w18_capacity,
        adequate=adequate,
        minimums_applied=minimums_applied,
        notes=notes,
        warnings=warnings,
        references=references,
    )

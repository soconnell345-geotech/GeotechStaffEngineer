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
_TREATED = ("cement_treated_base", "bituminous_treated_base")

# Figures 2.8 and 2.9 are parallel scales: a2 on the left, the strength
# scale beside it, read straight across. Read off the printed pages
# (GM-298) at each labelled strength tick; linear between ticks. The
# reference layer's three-point tables put a2 = 0.10 at 200 psi / 200 lb,
# where the print gives about 0.126 (0.10 sits near 30 psi / 120 lb), and
# hold 0.22 above 800 psi where the print rises to 0.25 at 1,000 psi.
_A2_CEMENT_SCALE = ((200, 0.126), (400, 0.155), (600, 0.188), (800, 0.220),
                    (1000, 0.250))
_A2_CEMENT_REF = "AASHTO 1993 Guide, Figure 2.8 (pdf_page 112, printed II-23)"
_A2_BITUMINOUS_SCALE = ((200, 0.126), (400, 0.161), (600, 0.180),
                        (800, 0.204), (1000, 0.230), (1200, 0.250),
                        (1400, 0.275), (1600, 0.300), (1800, 0.320))
_A2_BITUMINOUS_REF = ("AASHTO 1993 Guide, Figure 2.9 (pdf_page 113, "
                      "printed II-24)")
_HIGH_MODULUS_REF = ("AASHTO 1993 Guide, Section 3.1.5 (pdf_page 126, "
                     "printed II-37)")

# Section 3.1.5 (printed II-37): the Figure 3.2 procedure "should not be
# applied to determine the SN required above subbase or base materials
# having a modulus greater than 40,000 psi".
HIGH_MODULUS_PSI = 40_000.0


def _a2_from_scale(value, scale, what, unit, warnings):
    """a2 read straight across a printed parallel scale (Figs 2.8/2.9)."""
    lo, hi = scale[0][0], scale[-1][0]
    if value < lo:
        raise ValueError(
            f"{what} {value} {unit} is below the lowest labelled tick of the "
            f"printed scale ({lo} {unit}); give the layer coefficient 'a' "
            "directly."
        )
    if value > hi:
        warnings.append(
            f"{what} {value} {unit} is above the printed scale's top tick "
            f"({hi} {unit}); a2 = {scale[-1][1]} at {hi} {unit} used."
        )
        return scale[-1][1]
    for (x0, a0), (x1, a1) in zip(scale, scale[1:]):
        if x0 <= value <= x1:
            return round(a0 + (a1 - a0) * (value - x0) / (x1 - x0), 4)
    return scale[-1][1]  # pragma: no cover


def _is_high_modulus(layer):
    """True when the layer is a base/subbase stiffer than 40,000 psi: its
    modulus says so, or, with no modulus given, it is a cement- or
    bituminous-treated base (every modulus on the Figure 2.8 and 2.9
    scales is above 40,000 psi)."""
    if layer.modulus_psi is not None:
        return layer.modulus_psi > HIGH_MODULUS_PSI
    return layer.layer_type in _TREATED


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
    mode. In design mode, granular layers below the surface also need
    ``modulus_psi`` (used as the foundation modulus for the Figure 3.2
    re-solve above that layer). Above a layer stiffer than 40,000 psi (a
    treated base with no ``modulus_psi`` counts as one) no SN is solved
    (Section 3.1.5, printed II-37): the layer above takes its minimum
    practical thickness, ``min_thickness_in`` or the Section 3.1.4
    minimum, whichever is larger.
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
        a2 = _a2_from_scale(layer.ucs_7day_psi, _A2_CEMENT_SCALE,
                            "7-day UCS", "psi", warnings)
        add_ref(references, _A2_CEMENT_REF)
        return a2, "Figure 2.8 read across from the 7-day UCS scale"
    if lt == "bituminous_treated_base":
        if layer.marshall_stability_lb is None:
            raise ValueError(
                "bituminous_treated_base layer needs marshall_stability_lb "
                "or a direct 'a'."
            )
        a2 = _a2_from_scale(layer.marshall_stability_lb,
                            _A2_BITUMINOUS_SCALE, "Marshall stability", "lb",
                            warnings)
        add_ref(references, _A2_BITUMINOUS_REF)
        return a2, "Figure 2.9 read across from the Marshall stability scale"
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
        f"Drainage m defaulted to 1.0 for the {layer.layer_type}: the value "
        "for conditions at the AASHO Road Test, whatever the material "
        "(Section 2.4.1, printed II-25)."
    )
    return 1.0, "default 1.0 (AASHO Road Test conditions, II-25)"


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
    swelling=None,
    frost=None,
    design_period_yr=None,
) -> FlexiblePavementResult:
    """Complete AASHTO 1993 flexible pavement design or adequacy check.

    Modes (inferred from the layers):

    - **design** -- no layer has ``thickness_in``: solves the required SN
      over the roadbed (Figure 3.1), re-solves over each intermediate
      foundation (base/subbase modulus) per Figure 3.2, splits thicknesses
      top-down with round-as-you-go practical rounding, and enforces the
      Section 3.1.4 traffic-based minimums. Over a base or subbase stiffer
      than 40,000 psi (a treated base with no modulus counts as one) no SN
      is solved (Section 3.1.5, printed II-37): the layers above it take
      their minimum practical thickness.
    - **check** -- every layer has ``thickness_in``: computes the provided
      SN and compares to the requirement over the roadbed and, where a
      layer's modulus (at most 40,000 psi) is given, over that layer too
      (Figure 3.2: SN1* >= SN1, SN1* + SN2* >= SN2); no thicknesses are
      changed.

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
        Overall standard deviation; defaults to 0.45, the guide's value for
        flexible pavements with traffic variance (Section 2.1.3, printed
        II-9/10; range 0.40-0.50, printed I-62).
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
        Practical rounding increment for design mode (default 0.5 in). The
        guide rounds flexible layers "to the nearest 1/2 inch" (Section
        3.1.4, printed II-34); the module rounds UP, since Figure 3.2 asks
        the value used to be at least the value required.
    enforce_minimums : bool, optional
        Apply the Section 3.1.4 minimum AC/base thicknesses (design mode).
    swelling, frost : dict, optional
        Roadbed swelling {vr_in, ps_pct, theta} (Figure G.4) and/or frost
        heave {phi_mm_day, pf_pct, delta_psi_max} (Figure G.8) specs. The
        environmental serviceability loss at ``design_period_yr`` is
        subtracted from the design dPSI before the SN solves (Table 3.1
        Step 4), so the section is designed on the traffic-available
        budget. See ``performance.estimate_performance_period`` for the
        full printed performance-period iteration on the designed section.
    design_period_yr : float, optional
        Analysis period, years -- required when swelling/frost are given.

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
    from .performance import resolve_environmental_loss
    environmental = resolve_environmental_loss(design_period_yr, swelling,
                                               frost, references, notes)
    if environmental is not None:
        dpsi_env = environmental["delta_psi_total"]
        if dpsi_env >= dpsi:
            raise ValueError(
                f"Environmental serviceability loss ({dpsi_env}) consumes "
                f"the entire design dPSI ({dpsi}) at {design_period_yr} yr "
                "-- shorten the analysis period, mitigate the roadbed "
                "(see the guide's Appendix G options), or raise po/lower pt."
            )
        if dpsi_env > 0.5 * dpsi:
            warnings.append(
                f"Environmental loss ({dpsi_env}) exceeds half the design "
                f"dPSI ({dpsi}); the guide recommends considering roadbed "
                "mitigation before designing around swelling/frost of this "
                "magnitude."
            )
        dpsi_traffic = round(dpsi - dpsi_env, 4)
        environmental = dict(environmental, delta_psi_design=dpsi,
                             delta_psi_traffic=dpsi_traffic)
        dpsi = dpsi_traffic
    mr_eff, ref, n, mr_detail = resolve_effective_mr(mr_psi, monthly_mr_psi)
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
    # solve uses the actual roadbed effective MR. Over a base or subbase
    # stiffer than 40,000 psi no SN is solved (Section 3.1.5, printed
    # II-37); in design mode the layers above it take their minimum
    # practical thickness. In check mode an intermediate solve needs that
    # layer's modulus.
    def _solve_sn(foundation_mr):
        r = _eq.flexible_sn_from_w18(w18, zr_val, so_val, dpsi, foundation_mr)
        add_ref(references, r["reference"])
        return r["sn"]

    sn_over = []      # per layer: the SN required over the layer below it
    high_below = []   # layers stiffer than 40,000 psi with layers above
    for i, entry in enumerate(resolved[:-1]):
        below = resolved[i + 1]
        found_mr = below["layer"].modulus_psi
        if _is_high_modulus(below["layer"]):
            sn_over.append(None)
            high_below.append(below["layer_type"])
            continue
        if found_mr is None:
            if mode == "design":
                raise ValueError(
                    f"Design mode needs modulus_psi on the "
                    f"'{below['layer_type']}' layer (used as the foundation "
                    "modulus for the Figure 3.2 re-solve above it); give "
                    "modulus_psi, or supply thickness_in on every layer for "
                    "a check instead."
                )
            sn_over.append(None)
            continue
        sn_over.append({
            "over": below["layer_type"], "foundation_mr_psi": found_mr,
            "sn_required": _solve_sn(found_mr),
        })
    sn_roadbed = _solve_sn(mr_eff)
    sn_over.append({
        "over": "roadbed", "foundation_mr_psi": mr_eff,
        "sn_required": sn_roadbed,
    })
    if high_below:
        add_ref(references, _HIGH_MODULUS_REF)
        notes.append(
            f"No SN solved over the {' / '.join(high_below)}: modulus above "
            "40,000 psi (a treated base with no modulus given counts as "
            "one; every modulus on Figures 2.8 and 2.9 is higher). The "
            "guide: the Figure 3.2 procedure 'should not be applied to "
            "determine the SN required above subbase or base materials "
            "having a modulus greater than 40,000 psi' (Section 3.1.5, "
            "printed II-37). "
            + ("The layer above is set at its minimum practical thickness "
               "(Section 3.1.4 minimum or min_thickness_in, the larger); "
               "the guide leaves it to cost effectiveness, so give "
               "min_thickness_in for a thicker one."
               if mode == "design" else
               "Only the SN over the roadbed and over the layers at or "
               "below 40,000 psi is checked.")
        )

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
            floor = 0.0
            if entry["layer_type"] == "asphalt":
                floor = max(min_ac, entry["layer"].min_thickness_in or 0.0)
            elif entry["layer_type"] == "granular_base":
                floor = max(min_base, entry["layer"].min_thickness_in or 0.0)
            elif entry["layer"].min_thickness_in:
                floor = entry["layer"].min_thickness_in
            if sn_over[i] is None:
                # Above a layer stiffer than 40,000 psi (II-37).
                if floor <= 0:
                    raise ValueError(
                        f"The {entry['layer_type']} sits on a layer stiffer "
                        "than 40,000 psi, so the guide sets its thickness by "
                        "cost effectiveness and minimum practical thickness, "
                        "not by an SN solve (Section 3.1.5, printed II-37). "
                        "Give min_thickness_in on it"
                        + (" or keep enforce_minimums on"
                           if entry["layer_type"] in ("asphalt",
                                                      "granular_base")
                           else "") + "."
                    )
                d_raw = floor
                d = round_up(floor, thickness_increment_in)
                entry["thickness_basis"] = (
                    "minimum practical thickness over a layer stiffer than "
                    "40,000 psi (II-37)")
            else:
                need = max(sn_over[i]["sn_required"] - sn_carried, 0.0)
                d_raw = need / (entry["a"] * entry["m"])
                d = round_up(d_raw, thickness_increment_in)
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

    # Figure 3.2 note 2 (printed II-36): the SN provided above each
    # foundation must be at least the SN required over it (SN1* >= SN1,
    # SN1* + SN2* >= SN2, ...).
    sn_stack = []
    sn_above = 0.0
    for i, entry in enumerate(resolved):
        sn_above += entry["a"] * entry["thickness_in"] * entry["m"]
        row = sn_over[i]
        if row is None:
            continue
        ok = sn_above >= row["sn_required"] - 1e-9
        sn_stack.append(dict(row, sn_provided_above=round(sn_above, 3),
                             adequate=ok))
        if not ok and row["over"] != "roadbed":
            warnings.append(
                f"SN provided above the {row['over']} is "
                f"{sn_above:.2f}, below the {row['sn_required']:.2f} "
                f"required over it (Figure 3.2, MR = "
                f"{row['foundation_mr_psi']:,.0f} psi): the layers above it "
                "are too thin for its modulus."
            )

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
    adequate = ((sn_provided >= sn_roadbed - 1e-9)
                and w18_capacity >= w18
                and all(row["adequate"] for row in sn_stack))

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
        environmental=environmental,
        effective_mr_detail=mr_detail,
        minimums_applied=minimums_applied,
        notes=notes,
        warnings=warnings,
        references=references,
    )

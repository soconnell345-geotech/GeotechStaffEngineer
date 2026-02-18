"""
Engineering judgment / QA module for geotechnical analysis results.

Reviews analysis results and flags concerns using rules-of-thumb and
typical engineering ranges.  Each check function returns List[str] —
empty list means no concerns, populated list contains plain-English
warnings the LLM agent can include in its response.

Warnings are prefixed with severity:
    INFO:     Noteworthy observation, not necessarily a problem
    WARNING:  Unusual result that should be verified
    CRITICAL: Likely error or unsafe condition — do not proceed without review

References:
    - FHWA GEC-12: Design and Construction of Driven Pile Foundations
    - FHWA GEC-10: Drilled Shafts
    - AASHTO LRFD Bridge Design Specifications
    - Terzaghi, Peck & Mesri (1996)
    - Das, B.M. (2019) - Principles of Foundation Engineering
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Bearing Capacity Checks
# ---------------------------------------------------------------------------

def check_bearing_capacity(
    qult_kPa: float,
    q_allowable_kPa: float,
    soil_type: str = "unknown",
    footing_width_m: float = 0.0,
    footing_depth_m: float = 0.0,
    applied_stress_kPa: Optional[float] = None,
    factor_of_safety: float = 3.0,
) -> List[str]:
    """Flag if bearing capacity results are outside typical ranges.

    Parameters
    ----------
    qult_kPa : float
        Ultimate bearing capacity (kPa).
    q_allowable_kPa : float
        Allowable bearing capacity (kPa).
    soil_type : str
        Soil description: "soft clay", "stiff clay", "loose sand",
        "dense sand", "gravel", "rock", or "unknown".
    footing_width_m : float
        Footing width (m).
    footing_depth_m : float
        Footing embedment depth (m).
    applied_stress_kPa : float, optional
        Applied bearing pressure for demand/capacity check.
    factor_of_safety : float
        Factor of safety used.

    Returns
    -------
    List[str]
        Warning strings.
    """
    warnings: List[str] = []
    soil = soil_type.lower().strip()

    # Typical ranges (kPa) by soil type
    _RANGES = {
        "soft clay":   (50, 150),
        "stiff clay":  (150, 500),
        "loose sand":  (100, 300),
        "dense sand":  (300, 800),
        "gravel":      (400, 1200),
        "rock":        (500, 10000),
    }

    if soil in _RANGES:
        lo, hi = _RANGES[soil]
        if qult_kPa < lo:
            warnings.append(
                f"WARNING: qult = {qult_kPa:.0f} kPa is below typical range for "
                f"{soil_type} ({lo}-{hi} kPa) — verify soil parameters"
            )
        elif qult_kPa > hi:
            warnings.append(
                f"WARNING: qult = {qult_kPa:.0f} kPa is above typical range for "
                f"{soil_type} ({lo}-{hi} kPa) — verify soil parameters"
            )

    # Absolute thresholds
    if qult_kPa < 75 and "rock" not in soil:
        warnings.append(
            f"WARNING: qult = {qult_kPa:.0f} kPa is very low — verify soft soil "
            "conditions and check for undrained loading"
        )

    if qult_kPa > 2000 and "rock" not in soil:
        warnings.append(
            f"WARNING: qult = {qult_kPa:.0f} kPa is unusually high for soil — "
            "verify friction angle and cohesion inputs"
        )

    # Demand vs capacity
    if applied_stress_kPa is not None:
        if applied_stress_kPa > q_allowable_kPa:
            warnings.append(
                f"CRITICAL: Applied stress ({applied_stress_kPa:.0f} kPa) exceeds "
                f"allowable bearing capacity ({q_allowable_kPa:.0f} kPa) — "
                "foundation is inadequate"
            )
        elif applied_stress_kPa > 0.8 * q_allowable_kPa:
            warnings.append(
                f"WARNING: Applied stress ({applied_stress_kPa:.0f} kPa) is >{80}% "
                f"of allowable ({q_allowable_kPa:.0f} kPa) — low margin"
            )

    # Over-design check
    if applied_stress_kPa is not None and applied_stress_kPa > 0:
        actual_fos = qult_kPa / applied_stress_kPa
        if actual_fos > 6.0:
            warnings.append(
                f"INFO: Actual FOS = {actual_fos:.1f} — foundation may be overdesigned, "
                "consider reducing footing size"
            )

    # FOS check
    if factor_of_safety < 2.0:
        warnings.append(
            f"WARNING: FOS = {factor_of_safety:.1f} is below the typical minimum "
            "of 2.5-3.0 for bearing capacity"
        )

    # Depth/width ratio
    if footing_width_m > 0 and footing_depth_m > 0:
        dw_ratio = footing_depth_m / footing_width_m
        if dw_ratio > 3.0:
            warnings.append(
                f"INFO: Embedment depth/width ratio = {dw_ratio:.1f} — "
                "deep foundation behavior may govern; verify depth factors"
            )

    return warnings


# ---------------------------------------------------------------------------
# Settlement Checks
# ---------------------------------------------------------------------------

def check_settlement(
    total_settlement_mm: float,
    structure_type: str = "building",
    differential_settlement_mm: Optional[float] = None,
    span_m: Optional[float] = None,
    consolidation_time_years: Optional[float] = None,
    secondary_fraction: Optional[float] = None,
    immediate_mm: float = 0.0,
    consolidation_mm: float = 0.0,
    secondary_mm: float = 0.0,
) -> List[str]:
    """Compare settlement to typical tolerances.

    Parameters
    ----------
    total_settlement_mm : float
        Total settlement (mm).
    structure_type : str
        "bridge", "building_steel", "building_concrete", "industrial",
        "mse_wall", or "building" (generic).
    differential_settlement_mm : float, optional
        Differential settlement between adjacent footings (mm).
    span_m : float, optional
        Span between footings for angular distortion check (m).
    consolidation_time_years : float, optional
        Time for primary consolidation (years).
    secondary_fraction : float, optional
        Fraction of total settlement from secondary compression (0-1).
    immediate_mm, consolidation_mm, secondary_mm : float
        Settlement component breakdown.

    Returns
    -------
    List[str]
        Warning strings.
    """
    warnings: List[str] = []

    # Typical total settlement limits (mm)
    _LIMITS = {
        "bridge":             50,
        "building_steel":     50,
        "building_concrete":  50,
        "building":           50,
        "industrial":         75,
        "mse_wall":           75,
    }

    # Angular distortion limits
    _ANGULAR_LIMITS = {
        "bridge":             1 / 500,
        "building_steel":     1 / 300,
        "building_concrete":  1 / 500,
        "building":           1 / 400,
        "industrial":         1 / 300,
        "mse_wall":           1 / 200,
    }

    stype = structure_type.lower().strip()
    limit = _LIMITS.get(stype, 50)
    angular_limit = _ANGULAR_LIMITS.get(stype, 1 / 400)

    if total_settlement_mm > limit:
        warnings.append(
            f"CRITICAL: Total settlement = {total_settlement_mm:.0f} mm exceeds "
            f"typical limit of {limit} mm for {structure_type}"
        )
    elif total_settlement_mm > 0.8 * limit:
        warnings.append(
            f"WARNING: Total settlement = {total_settlement_mm:.0f} mm is >{80}% "
            f"of typical limit ({limit} mm) for {structure_type} — marginal"
        )

    # Angular distortion
    if differential_settlement_mm is not None and span_m is not None and span_m > 0:
        angular_distortion = (differential_settlement_mm / 1000.0) / span_m
        if angular_distortion > angular_limit:
            warnings.append(
                f"CRITICAL: Angular distortion = 1/{1/angular_distortion:.0f} "
                f"exceeds limit of 1/{1/angular_limit:.0f} for {structure_type}"
            )
        elif angular_distortion > 0.8 * angular_limit:
            warnings.append(
                f"WARNING: Angular distortion = 1/{1/angular_distortion:.0f} "
                f"is close to limit of 1/{1/angular_limit:.0f} — marginal"
            )

    # Consolidation time
    if consolidation_time_years is not None:
        if consolidation_time_years > 10:
            warnings.append(
                f"WARNING: Primary consolidation time = {consolidation_time_years:.1f} years — "
                "consider wick drains or surcharge preloading"
            )
        elif consolidation_time_years > 5:
            warnings.append(
                f"INFO: Primary consolidation time = {consolidation_time_years:.1f} years — "
                "wick drains may be economical"
            )

    # Secondary compression
    if secondary_fraction is not None and secondary_fraction > 0.2:
        warnings.append(
            f"WARNING: Secondary compression is {secondary_fraction*100:.0f}% "
            "of total settlement — long-term creep is significant"
        )

    # Zero or negative settlement
    if total_settlement_mm < 0:
        warnings.append(
            f"CRITICAL: Negative settlement ({total_settlement_mm:.1f} mm) — "
            "check input signs and loading conditions"
        )

    # Very large settlement
    if total_settlement_mm > 200:
        warnings.append(
            f"WARNING: Settlement = {total_settlement_mm:.0f} mm is very large — "
            "ground improvement or deep foundations likely needed"
        )

    return warnings


# ---------------------------------------------------------------------------
# Axial Pile Capacity Checks
# ---------------------------------------------------------------------------

def check_pile_capacity(
    capacity_kN: float,
    pile_type: str = "steel_pipe",
    pile_diameter_m: float = 0.3,
    pile_length_m: float = 15.0,
    Q_skin_kN: float = 0.0,
    Q_tip_kN: float = 0.0,
    factor_of_safety: float = 2.5,
    applied_load_kN: Optional[float] = None,
    soil_description: str = "unknown",
) -> List[str]:
    """Flag unusual axial pile capacity results.

    Parameters
    ----------
    capacity_kN : float
        Ultimate axial capacity (kN).
    pile_type : str
        "h_pile", "steel_pipe", "concrete", "timber", "drilled_shaft".
    pile_diameter_m : float
        Pile diameter or width (m).
    pile_length_m : float
        Pile embedment length (m).
    Q_skin_kN : float
        Skin friction component (kN).
    Q_tip_kN : float
        End bearing component (kN).
    factor_of_safety : float
        Factor of safety applied.
    applied_load_kN : float, optional
        Design load for utilization check.
    soil_description : str
        General soil description.

    Returns
    -------
    List[str]
        Warning strings.
    """
    warnings: List[str] = []
    ptype = pile_type.lower().strip()

    # Typical capacity ranges by pile type (kN)
    _RANGES = {
        "h_pile":        (400, 1500),
        "steel_pipe":    (300, 1200),
        "concrete":      (500, 2000),
        "timber":        (150, 600),
        "drilled_shaft": (1000, 5000),
    }

    if ptype in _RANGES:
        lo, hi = _RANGES[ptype]
        if capacity_kN < lo:
            warnings.append(
                f"INFO: Capacity = {capacity_kN:.0f} kN is below typical range for "
                f"{pile_type} ({lo}-{hi} kN) — short pile or soft soil?"
            )
        elif capacity_kN > hi * 1.5:
            warnings.append(
                f"WARNING: Capacity = {capacity_kN:.0f} kN is well above typical range for "
                f"{pile_type} ({lo}-{hi} kN) — verify soil parameters"
            )

    # L/D ratio
    if pile_diameter_m > 0:
        ld_ratio = pile_length_m / pile_diameter_m
        if ld_ratio > 60:
            warnings.append(
                f"WARNING: L/D = {ld_ratio:.0f} is very high (>60) — "
                "check for buckling and installation difficulties"
            )
        if ld_ratio < 10 and capacity_kN > 0:
            warnings.append(
                f"INFO: L/D = {ld_ratio:.0f} is low (<10) — pile may behave "
                "as a short rigid pile; verify p-y or lateral analysis"
            )

    # Skin vs tip concentration
    total = Q_skin_kN + Q_tip_kN
    if total > 0:
        tip_fraction = Q_tip_kN / total
        skin_fraction = Q_skin_kN / total
        if tip_fraction > 0.8:
            warnings.append(
                f"INFO: End bearing provides {tip_fraction*100:.0f}% of capacity — "
                "verify tip conditions; consider setup/relaxation effects"
            )
        if skin_fraction > 0.8:
            warnings.append(
                f"INFO: Skin friction provides {skin_fraction*100:.0f}% of capacity — "
                "friction pile; consider negative skin friction if fill is placed"
            )

    # Demand vs capacity
    if applied_load_kN is not None and capacity_kN > 0:
        allowable = capacity_kN / factor_of_safety
        utilization = applied_load_kN / allowable
        if utilization > 1.0:
            warnings.append(
                f"CRITICAL: Applied load ({applied_load_kN:.0f} kN) exceeds "
                f"allowable capacity ({allowable:.0f} kN, FOS={factor_of_safety:.1f})"
            )
        elif utilization > 0.9:
            warnings.append(
                f"WARNING: Pile utilization = {utilization*100:.0f}% — low margin"
            )

    # FOS
    if factor_of_safety < 2.0:
        warnings.append(
            f"WARNING: FOS = {factor_of_safety:.1f} is below typical minimum of 2.0-2.5 "
            "for pile capacity"
        )

    # Very low capacity
    if capacity_kN < 100:
        warnings.append(
            f"WARNING: Capacity = {capacity_kN:.0f} kN is very low — "
            "verify soil parameters and pile installation method"
        )

    return warnings


# ---------------------------------------------------------------------------
# Lateral Pile Checks
# ---------------------------------------------------------------------------

def check_lateral_pile(
    deflection_mm: float,
    max_moment_kNm: float,
    pile_diameter_m: float = 0.3,
    pile_length_m: float = 15.0,
    max_moment_depth_m: float = 0.0,
    service_or_ultimate: str = "service",
    structure_type: str = "bridge",
    converged: bool = True,
    head_condition: str = "free",
    moment_at_head_kNm: float = 0.0,
) -> List[str]:
    """Flag unusual lateral pile results.

    Parameters
    ----------
    deflection_mm : float
        Pile head deflection (mm).
    max_moment_kNm : float
        Maximum bending moment (kN-m).
    pile_diameter_m : float
        Pile diameter (m).
    pile_length_m : float
        Pile embedded length (m).
    max_moment_depth_m : float
        Depth to maximum moment (m).
    service_or_ultimate : str
        "service" or "ultimate".
    structure_type : str
        "bridge", "sign_structure", "sound_wall", "building".
    converged : bool
        Whether the solver converged.
    head_condition : str
        "free" or "fixed".
    moment_at_head_kNm : float
        Moment at pile head (kN-m).

    Returns
    -------
    List[str]
        Warning strings.
    """
    warnings: List[str] = []

    # Solver convergence
    if not converged:
        warnings.append(
            "CRITICAL: Lateral pile solver did not converge — results are unreliable. "
            "Try reducing load or increasing mesh density."
        )

    # Service deflection limits (mm)
    _DEFLECTION_LIMITS = {
        "bridge":         25,
        "building":       25,
        "sign_structure": 50,
        "sound_wall":     50,
    }
    stype = structure_type.lower().strip()

    if service_or_ultimate.lower() == "service":
        limit = _DEFLECTION_LIMITS.get(stype, 25)
        if deflection_mm > limit:
            warnings.append(
                f"WARNING: Service deflection = {deflection_mm:.1f} mm exceeds "
                f"typical limit of {limit} mm for {structure_type}"
            )
        elif deflection_mm > 0.8 * limit:
            warnings.append(
                f"INFO: Service deflection = {deflection_mm:.1f} mm is close to "
                f"typical limit ({limit} mm)"
            )

    # Max moment depth
    if pile_diameter_m > 0 and max_moment_depth_m > 0:
        depth_ratio = max_moment_depth_m / pile_diameter_m
        if depth_ratio > 10:
            warnings.append(
                f"INFO: Max moment at depth {max_moment_depth_m:.1f}m "
                f"({depth_ratio:.0f} diameters) — unusually deep, verify p-y curves"
            )

    # Pile length adequacy — zero crossing check
    if pile_diameter_m > 0:
        min_length = 15 * pile_diameter_m
        if pile_length_m < min_length:
            warnings.append(
                f"WARNING: Pile length ({pile_length_m:.1f}m) may be insufficient — "
                f"recommend >= {min_length:.1f}m (15D) for full lateral fixity"
            )

    # Free head with non-zero moment at head (possible solver issue)
    if head_condition.lower() == "free" and abs(moment_at_head_kNm) > 0.01 * max_moment_kNm:
        if max_moment_kNm > 0:
            warnings.append(
                f"INFO: Non-zero moment at pile head ({moment_at_head_kNm:.1f} kN-m) "
                "with free-head condition — verify applied moment input"
            )

    # Very large deflection
    if deflection_mm > 100:
        warnings.append(
            f"WARNING: Deflection = {deflection_mm:.0f} mm is very large — "
            "pile may have failed; check for plastic hinge formation"
        )

    # Zero deflection with applied load
    if deflection_mm < 0.01:
        warnings.append(
            "INFO: Near-zero deflection — verify that load was applied correctly"
        )

    return warnings


# ---------------------------------------------------------------------------
# Sheet Pile Checks
# ---------------------------------------------------------------------------

def check_sheet_pile(
    embedment_m: float,
    retained_height_m: float,
    max_moment_kNm_per_m: float,
    wall_type: str = "cantilever",
    soil_type: str = "sand",
    FOS_passive: float = 1.5,
    anchor_force_kN_per_m: Optional[float] = None,
) -> List[str]:
    """Flag unusual sheet pile wall results.

    Parameters
    ----------
    embedment_m : float
        Required embedment depth (m).
    retained_height_m : float
        Height of retained soil / excavation depth (m).
    max_moment_kNm_per_m : float
        Maximum bending moment (kN-m/m of wall).
    wall_type : str
        "cantilever" or "anchored".
    soil_type : str
        "sand", "clay", or "mixed".
    FOS_passive : float
        Factor of safety on passive resistance.
    anchor_force_kN_per_m : float, optional
        Anchor force per meter of wall (kN/m).

    Returns
    -------
    List[str]
        Warning strings.
    """
    warnings: List[str] = []
    wtype = wall_type.lower().strip()
    stype = soil_type.lower().strip()

    # Embedment ratio checks
    embed_ratio = embedment_m / retained_height_m if retained_height_m > 0 else 0

    if wtype == "cantilever":
        if stype == "sand":
            if embed_ratio < 1.2:
                warnings.append(
                    f"WARNING: Embedment ratio = {embed_ratio:.2f} is low for "
                    "cantilever in sand (typical 1.5-2.5)"
                )
            elif embed_ratio > 3.0:
                warnings.append(
                    f"INFO: Embedment ratio = {embed_ratio:.2f} — may be overdesigned "
                    "for cantilever in sand (typical 1.5-2.5)"
                )
        elif stype == "clay":
            if embed_ratio < 0.8:
                warnings.append(
                    f"WARNING: Embedment ratio = {embed_ratio:.2f} is low for "
                    "cantilever in clay (typical 1.0-2.0)"
                )

        # Cantilever height limit
        if retained_height_m > 6:
            warnings.append(
                f"WARNING: Cantilever wall retaining {retained_height_m:.1f}m — "
                "cantilever walls are rarely practical above 5-6m; "
                "consider anchored or braced wall"
            )

    elif wtype == "anchored":
        if embed_ratio < 0.3:
            warnings.append(
                f"WARNING: Embedment ratio = {embed_ratio:.2f} is very low for "
                "anchored wall (typical 0.5-1.5)"
            )
        elif embed_ratio > 2.0:
            warnings.append(
                f"INFO: Embedment ratio = {embed_ratio:.2f} — may be overdesigned "
                "for anchored wall (typical 0.5-1.5)"
            )

    # FOS on passive
    if FOS_passive < 1.5:
        warnings.append(
            f"WARNING: FOS on passive resistance = {FOS_passive:.2f} — "
            "typical minimum is 1.5 for temporary, 2.0 for permanent"
        )
    if FOS_passive < 1.2:
        warnings.append(
            f"CRITICAL: FOS on passive = {FOS_passive:.2f} — inadequate safety margin"
        )

    # Very high moment
    if max_moment_kNm_per_m > 500:
        warnings.append(
            f"INFO: Max moment = {max_moment_kNm_per_m:.0f} kN-m/m — "
            "verify section modulus of selected sheet pile is adequate"
        )

    return warnings


# ---------------------------------------------------------------------------
# Wave Equation Checks
# ---------------------------------------------------------------------------

def check_wave_equation(
    blow_count: float,
    max_comp_stress_kPa: float,
    max_tension_stress_kPa: float,
    pile_type: str = "steel",
    fy_kPa: float = 248_000.0,
    fc_prime_kPa: float = 35_000.0,
    capacity_kN: float = 0.0,
) -> List[str]:
    """Flag driving stress concerns.

    Parameters
    ----------
    blow_count : float
        Blow count (blows/0.3m, i.e. blows per foot).
    max_comp_stress_kPa : float
        Maximum compression stress in pile (kPa).
    max_tension_stress_kPa : float
        Maximum tension stress in pile (kPa).
    pile_type : str
        "steel", "concrete", "timber".
    fy_kPa : float
        Steel yield stress (kPa). Default 248 MPa = 248,000 kPa.
    fc_prime_kPa : float
        Concrete compressive strength (kPa). Default 35 MPa.
    capacity_kN : float
        Target pile capacity for context.

    Returns
    -------
    List[str]
        Warning strings.
    """
    warnings: List[str] = []
    ptype = pile_type.lower().strip()

    # Stress limits
    if ptype == "steel":
        comp_limit = 0.9 * fy_kPa
        tens_limit = 0.9 * fy_kPa
        if max_comp_stress_kPa > comp_limit:
            warnings.append(
                f"CRITICAL: Compression stress = {max_comp_stress_kPa/1000:.0f} MPa "
                f"exceeds 0.9*fy = {comp_limit/1000:.0f} MPa — pile damage likely"
            )
        elif max_comp_stress_kPa > 0.8 * comp_limit:
            warnings.append(
                f"WARNING: Compression stress = {max_comp_stress_kPa/1000:.0f} MPa "
                f"is close to limit ({comp_limit/1000:.0f} MPa)"
            )
        if max_tension_stress_kPa > tens_limit:
            warnings.append(
                f"CRITICAL: Tension stress = {max_tension_stress_kPa/1000:.0f} MPa "
                f"exceeds 0.9*fy = {tens_limit/1000:.0f} MPa"
            )

    elif ptype == "concrete":
        comp_limit = 0.85 * fc_prime_kPa
        tens_limit = 0.7 * (fc_prime_kPa ** 0.5) * 1000  # sqrt(f'c in kPa) * factor
        # Correct: 0.7 * sqrt(f'c in MPa) in MPa, convert to kPa
        fc_mpa = fc_prime_kPa / 1000.0
        tens_limit_kPa = 0.7 * (fc_mpa ** 0.5) * 1000.0
        if max_comp_stress_kPa > comp_limit:
            warnings.append(
                f"CRITICAL: Compression stress = {max_comp_stress_kPa/1000:.0f} MPa "
                f"exceeds 0.85*f'c = {comp_limit/1000:.1f} MPa"
            )
        if max_tension_stress_kPa > tens_limit_kPa:
            warnings.append(
                f"CRITICAL: Tension stress = {max_tension_stress_kPa/1000:.1f} MPa "
                f"exceeds 0.7*sqrt(f'c) = {tens_limit_kPa/1000:.1f} MPa — "
                "cracking expected"
            )

    elif ptype == "timber":
        # Typical timber allowable static ~ 8-12 MPa, driving limit 3x
        timber_limit_kPa = 30_000.0  # ~30 MPa
        if max_comp_stress_kPa > timber_limit_kPa:
            warnings.append(
                f"CRITICAL: Compression stress = {max_comp_stress_kPa/1000:.0f} MPa "
                f"exceeds timber driving limit (~{timber_limit_kPa/1000:.0f} MPa)"
            )

    # Blow count checks (in blows per foot / 0.3m)
    if blow_count > 240:
        warnings.append(
            f"CRITICAL: Blow count = {blow_count:.0f} bl/ft — practical refusal. "
            "Pile cannot be driven further with this hammer."
        )
    elif blow_count > 120:
        warnings.append(
            f"WARNING: Blow count = {blow_count:.0f} bl/ft — hard driving. "
            "Verify hammer energy is sufficient."
        )

    if blow_count < 10 and capacity_kN > 0:
        warnings.append(
            f"INFO: Blow count = {blow_count:.0f} bl/ft at target capacity — "
            "very easy driving. Consider setup/freeze effects; "
            "restrike testing recommended."
        )

    return warnings


# ---------------------------------------------------------------------------
# Pile Group Checks
# ---------------------------------------------------------------------------

def check_pile_group(
    max_compression_kN: float,
    max_tension_kN: float,
    max_utilization: float,
    n_piles: int,
    spacing_m: float = 0.0,
    pile_diameter_m: float = 0.3,
    pile_forces: Optional[List[Dict[str, Any]]] = None,
    design_for_tension: bool = False,
) -> List[str]:
    """Flag pile group concerns.

    Parameters
    ----------
    max_compression_kN : float
        Maximum compression in any pile (kN).
    max_tension_kN : float
        Maximum tension in any pile (kN, positive = tension).
    max_utilization : float
        Maximum utilization ratio (demand/capacity).
    n_piles : int
        Number of piles in group.
    spacing_m : float
        Center-to-center pile spacing (m).
    pile_diameter_m : float
        Pile diameter (m).
    pile_forces : list, optional
        Per-pile force breakdown for distribution analysis.
    design_for_tension : bool
        Whether piles are designed for tension.

    Returns
    -------
    List[str]
        Warning strings.
    """
    warnings: List[str] = []

    # Tension check
    if max_tension_kN > 0 and not design_for_tension:
        warnings.append(
            f"WARNING: Pile(s) in tension (max = {max_tension_kN:.0f} kN) — "
            "piles are not designed for uplift. Revise layout or add piles."
        )

    # Utilization
    if max_utilization > 1.0:
        warnings.append(
            f"CRITICAL: Maximum utilization = {max_utilization:.2f} — "
            "pile capacity exceeded"
        )
    elif max_utilization > 0.9:
        warnings.append(
            f"WARNING: Maximum utilization = {max_utilization:.2f} — low margin"
        )

    # Load distribution
    if pile_forces and len(pile_forces) >= 2:
        axial_loads = [abs(p.get("axial_kN", 0)) for p in pile_forces]
        max_load = max(axial_loads)
        min_load = min(axial_loads) if min(axial_loads) > 0 else 1.0
        ratio = max_load / min_load if min_load > 0 else float('inf')
        if ratio > 2.0:
            warnings.append(
                f"WARNING: Load distribution ratio = {ratio:.1f}:1 — "
                "uneven loading, consider adjusting pile layout"
            )

    # Spacing checks
    if spacing_m > 0 and pile_diameter_m > 0:
        sd_ratio = spacing_m / pile_diameter_m
        if sd_ratio < 3.0:
            warnings.append(
                f"WARNING: Spacing/diameter = {sd_ratio:.1f} (<3.0) — "
                "significant group effects; verify p-multipliers are applied"
            )
        elif sd_ratio > 8.0:
            warnings.append(
                f"INFO: Spacing/diameter = {sd_ratio:.1f} (>8.0) — "
                "wide spacing, check pile cap spanning and rigidity assumption"
            )

    return warnings


# ---------------------------------------------------------------------------
# Cross-Module: Foundation Selection
# ---------------------------------------------------------------------------

def check_foundation_selection(
    shallow_qallowable_kPa: Optional[float] = None,
    shallow_settlement_mm: Optional[float] = None,
    shallow_fos: Optional[float] = None,
    pile_length_m: Optional[float] = None,
    pile_capacity_kN: Optional[float] = None,
    settlement_limit_mm: float = 50.0,
    has_soft_layer_below_footing: bool = False,
    has_liquefiable_layer: bool = False,
    applied_stress_kPa: Optional[float] = None,
) -> List[str]:
    """Help decide between shallow and deep foundations.

    Parameters
    ----------
    shallow_qallowable_kPa : float, optional
        Allowable bearing capacity of shallow foundation (kPa).
    shallow_settlement_mm : float, optional
        Predicted settlement of shallow foundation (mm).
    shallow_fos : float, optional
        Factor of safety for shallow foundation.
    pile_length_m : float, optional
        Required pile length (m).
    pile_capacity_kN : float, optional
        Axial pile capacity (kN).
    settlement_limit_mm : float
        Settlement limit for the structure (mm).
    has_soft_layer_below_footing : bool
        True if a soft compressible layer exists within 2B below footing.
    has_liquefiable_layer : bool
        True if a liquefiable layer exists in the profile.
    applied_stress_kPa : float, optional
        Applied bearing pressure (kPa).

    Returns
    -------
    List[str]
        Advisory strings.
    """
    warnings: List[str] = []

    # Shallow foundation feasibility
    if shallow_settlement_mm is not None:
        if shallow_settlement_mm > 0.8 * settlement_limit_mm:
            warnings.append(
                f"WARNING: Shallow foundation settlement ({shallow_settlement_mm:.0f} mm) "
                f"is >{80}% of limit ({settlement_limit_mm:.0f} mm) — marginal. "
                "Consider ground improvement or deep foundations."
            )

    if shallow_fos is not None and shallow_fos > 5.0:
        warnings.append(
            f"INFO: Shallow foundation FOS = {shallow_fos:.1f} — overdesigned. "
            "Consider reducing footing size."
        )

    # Deep foundation check
    if pile_length_m is not None and pile_length_m < 5.0:
        warnings.append(
            f"INFO: Required pile length = {pile_length_m:.1f}m — very short. "
            "Consider shallow foundation as a more economical alternative."
        )

    # Soft layer warning
    if has_soft_layer_below_footing:
        warnings.append(
            "WARNING: Soft compressible layer within 2B below footing — "
            "check for punch-through failure and excessive settlement"
        )

    # Liquefaction
    if has_liquefiable_layer:
        warnings.append(
            "CRITICAL: Liquefiable layer identified in profile — "
            "deep foundations typically required. If shallow foundations are used, "
            "ground improvement is necessary."
        )

    return warnings


# ---------------------------------------------------------------------------
# Cross-Module: Parameter Consistency
# ---------------------------------------------------------------------------

def check_parameter_consistency(
    layers: Optional[list] = None,
    profile: Optional[Any] = None,
) -> List[str]:
    """Check that soil parameters are internally consistent.

    Accepts either a list of SoilLayer objects or a SoilProfile.
    If a SoilProfile is provided, uses its layers.

    Parameters
    ----------
    layers : list, optional
        List of SoilLayer objects.
    profile : SoilProfile, optional
        Complete soil profile.

    Returns
    -------
    List[str]
        Warning strings.
    """
    warnings: List[str] = []

    if profile is not None:
        layers = profile.layers
    if layers is None:
        return warnings

    for i, layer in enumerate(layers):
        prefix = f"Layer {i} ('{getattr(layer, 'description', 'unknown')}')"

        cu = getattr(layer, 'cu', None)
        phi = getattr(layer, 'phi', None)
        N_spt = getattr(layer, 'N_spt', None)
        gamma = getattr(layer, 'gamma', None)
        gamma_sat = getattr(layer, 'gamma_sat', None)
        Cc = getattr(layer, 'Cc', None)
        sigma_p = getattr(layer, 'sigma_p', None)
        is_cohesive = getattr(layer, 'is_cohesive', None)

        # cu and phi both specified
        if cu is not None and cu > 0 and phi is not None and phi > 0:
            warnings.append(
                f"INFO: {prefix}: both cu={cu:.0f} kPa and phi={phi:.0f}deg specified — "
                "ensure correct analysis type (total stress vs effective stress)"
            )

        # N_spt vs cu inconsistency
        if N_spt is not None and cu is not None and N_spt > 0 and cu > 0:
            expected_cu = 6.25 * N_spt
            if cu > 3 * expected_cu:
                warnings.append(
                    f"WARNING: {prefix}: cu={cu:.0f} kPa >> N_spt={N_spt:.0f} "
                    f"(expected ~{expected_cu:.0f} kPa from Terzaghi-Peck) — "
                    "parameters may be inconsistent"
                )
            if cu < expected_cu / 3:
                warnings.append(
                    f"WARNING: {prefix}: cu={cu:.0f} kPa << N_spt={N_spt:.0f} "
                    f"(expected ~{expected_cu:.0f} kPa from Terzaghi-Peck) — "
                    "parameters may be inconsistent"
                )

        # phi vs N_spt inconsistency
        if N_spt is not None and phi is not None and N_spt > 0 and phi > 0:
            # Rough check: N=5 -> phi~28, N=30 -> phi~36, N=50 -> phi~42
            if phi > 40 and N_spt < 20:
                warnings.append(
                    f"WARNING: {prefix}: phi={phi:.0f}deg but N_spt={N_spt:.0f} — "
                    "high friction angle with low blow count, verify"
                )
            if phi < 28 and N_spt > 30:
                warnings.append(
                    f"WARNING: {prefix}: phi={phi:.0f}deg but N_spt={N_spt:.0f} — "
                    "low friction angle with high blow count, verify"
                )

        # gamma_sat < gamma
        if gamma is not None and gamma_sat is not None:
            if gamma_sat < gamma:
                warnings.append(
                    f"CRITICAL: {prefix}: gamma_sat={gamma_sat:.1f} < gamma={gamma:.1f} — "
                    "impossible; saturated weight must be >= total weight"
                )

        # Cc range
        if Cc is not None and Cc > 1.0:
            warnings.append(
                f"WARNING: {prefix}: Cc={Cc:.2f} is very high — "
                "typical for organic soils only (peat, organic clay)"
            )

        # OCR from sigma_p
        if sigma_p is not None and gamma is not None:
            # Rough mid-layer effective stress
            mid_depth = getattr(layer, 'mid_depth', None)
            if mid_depth is not None and mid_depth > 0:
                # Approximate sigma_v' (very rough: gamma * depth * 0.7 for typical GWT)
                approx_sigma_v = gamma * mid_depth * 0.6  # rough average
                if approx_sigma_v > 0:
                    ocr = sigma_p / approx_sigma_v
                    if ocr > 10:
                        warnings.append(
                            f"INFO: {prefix}: estimated OCR ≈ {ocr:.0f} — very high, verify"
                        )

    return warnings


# ── Slope Stability Checks ────────────────────────────────────────

def check_slope_stability(
    FOS: float,
    is_stable: bool,
    FOS_required: float = 1.5,
    method: str = "bishop",
    has_seismic: bool = False,
    n_slices: int = 30,
    FOS_fellenius: float = None,
    FOS_bishop: float = None,
    kh: float = 0.0,
) -> List[str]:
    """Check slope stability analysis results for issues.

    Parameters
    ----------
    FOS : float
        Computed factor of safety.
    is_stable : bool
        Whether FOS >= FOS_required.
    FOS_required : float
        Minimum required FOS (default 1.5).
    method : str
        Analysis method used ('fellenius', 'bishop', 'spencer').
    has_seismic : bool
        Whether seismic loading was applied.
    n_slices : int
        Number of slices used in analysis.
    FOS_fellenius : float, optional
        Fellenius FOS if comparison was run.
    FOS_bishop : float, optional
        Bishop FOS if comparison was run.
    kh : float
        Horizontal seismic coefficient.

    Returns
    -------
    List[str]
        Warnings prefixed with CRITICAL / WARNING / INFO.
    """
    warnings: List[str] = []

    # 1. FOS < 1.0 — failure
    if FOS < 1.0:
        warnings.append(
            f"CRITICAL: FOS = {FOS:.3f} — slope failure predicted (FOS < 1.0)"
        )
    # 2. FOS marginal (>= 1.0 but below required)
    elif FOS < FOS_required:
        warnings.append(
            f"WARNING: FOS = {FOS:.3f} is below required FOS of {FOS_required:.2f}"
            " — marginal stability"
        )

    # 3. Overdesigned
    if FOS > 3.0 * FOS_required:
        warnings.append(
            f"INFO: FOS = {FOS:.3f} is much higher than required ({FOS_required:.2f})"
            " — slope may be overdesigned or slip surface is not critical"
        )

    # 4. Seismic with low FOS
    if has_seismic and FOS < 1.1:
        warnings.append(
            f"CRITICAL: Seismic FOS = {FOS:.3f} (kh={kh:.3f}) is below 1.1"
            " — seismic slope failure risk"
        )

    # 5. Low slice count
    if n_slices < 20:
        warnings.append(
            f"INFO: Only {n_slices} slices used"
            " — consider using >= 20-30 for accuracy"
        )

    # 6. Fellenius alone
    if method.lower() == "fellenius" and FOS_bishop is None:
        warnings.append(
            "INFO: Fellenius method alone is less accurate"
            " — recommend Bishop or Spencer for comparison"
        )

    # 7. Fellenius vs Bishop divergence
    if (FOS_fellenius is not None and FOS_bishop is not None
            and FOS_bishop > 0):
        divergence = abs(FOS_fellenius - FOS_bishop) / FOS_bishop
        if divergence > 0.15:
            warnings.append(
                f"INFO: Fellenius FOS ({FOS_fellenius:.3f}) differs from"
                f" Bishop ({FOS_bishop:.3f}) by {divergence:.0%}"
                " — large divergence may indicate complex geometry"
            )

    return warnings

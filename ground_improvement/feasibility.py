"""
Ground improvement feasibility evaluation and decision support.

Takes extracted soil parameters and design requirements and recommends
which ground improvement methods are applicable, with preliminary sizing.

This module takes plain floats/strings (NOT SoilProfile objects) to
avoid circular imports.

References:
    FHWA GEC-13: Ground Modification Methods Reference Manual
    FHWA NHI-06-019/020: Ground Improvement Methods
"""

from typing import Any, Dict, List, Optional

from ground_improvement.results import FeasibilityResult


def evaluate_feasibility(
    soil_type: str,
    fines_content: float = None,
    N_spt: float = None,
    cu_kPa: float = None,
    thickness_m: float = 0.0,
    depth_to_top_m: float = 0.0,
    required_bearing_kPa: float = 0.0,
    current_bearing_kPa: float = 0.0,
    predicted_settlement_mm: float = 0.0,
    allowable_settlement_mm: float = 50.0,
    time_constraint_months: float = 0.0,
    cv_m2_per_year: float = None,
    Hdr_m: float = None,
    gwt_depth_m: float = None,
) -> FeasibilityResult:
    """Evaluate which ground improvement methods are applicable.

    Parameters
    ----------
    soil_type : str
        One of 'soft_clay', 'loose_sand', 'mixed', 'organic'.
    fines_content : float, optional
        Percent passing #200 sieve. Required for vibro assessment.
    N_spt : float, optional
        Average SPT N in the treatment zone.
    cu_kPa : float, optional
        Average undrained shear strength (kPa).
    thickness_m : float
        Thickness of the compressible / treatable layer (m).
    depth_to_top_m : float
        Depth to top of the treatable layer (m).
    required_bearing_kPa : float
        Required bearing capacity (kPa). 0 = not a bearing problem.
    current_bearing_kPa : float
        Current (unreinforced) bearing capacity (kPa).
    predicted_settlement_mm : float
        Predicted settlement without improvement (mm).
    allowable_settlement_mm : float
        Allowable settlement limit (mm). Default 50 mm.
    time_constraint_months : float
        Time available for improvement (months). 0 = no constraint.
    cv_m2_per_year : float, optional
        Vertical coefficient of consolidation.
    Hdr_m : float, optional
        Vertical drainage path length (m).
    gwt_depth_m : float, optional
        Depth to groundwater table (m).

    Returns
    -------
    FeasibilityResult
        Applicable methods, exclusions, recommendations, and preliminary sizing.
    """
    valid_types = ("soft_clay", "loose_sand", "mixed", "organic")
    if soil_type not in valid_types:
        raise ValueError(f"soil_type must be one of {valid_types}, got '{soil_type}'")

    applicable = []
    not_applicable = []
    recommendations = []
    sizing = {}

    is_clay = soil_type in ("soft_clay", "organic", "mixed")
    is_sand = soil_type in ("loose_sand", "mixed")
    is_organic = soil_type == "organic"
    has_time_constraint = time_constraint_months > 0
    settlement_problem = predicted_settlement_mm > allowable_settlement_mm
    bearing_problem = (required_bearing_kPa > 0
                       and current_bearing_kPa > 0
                       and required_bearing_kPa > current_bearing_kPa)

    # Build description strings
    soil_desc = f"{soil_type}"
    if cu_kPa is not None:
        soil_desc += f", cu={cu_kPa:.0f} kPa"
    if N_spt is not None:
        soil_desc += f", N={N_spt:.0f}"
    if thickness_m > 0:
        soil_desc += f", {thickness_m:.1f}m thick"

    problem_parts = []
    if settlement_problem:
        problem_parts.append(
            f"settlement {predicted_settlement_mm:.0f}mm > {allowable_settlement_mm:.0f}mm allowable"
        )
    if bearing_problem:
        problem_parts.append(
            f"bearing {current_bearing_kPa:.0f} kPa < {required_bearing_kPa:.0f} kPa required"
        )
    if has_time_constraint:
        problem_parts.append(f"time constraint: {time_constraint_months:.0f} months")
    design_desc = "; ".join(problem_parts) if problem_parts else "general improvement"

    # ---- Aggregate Piers ----
    _eval_aggregate_piers(
        is_clay, is_organic, cu_kPa, N_spt, thickness_m,
        settlement_problem, predicted_settlement_mm, allowable_settlement_mm,
        bearing_problem, required_bearing_kPa, current_bearing_kPa,
        applicable, not_applicable, recommendations, sizing,
    )

    # ---- Wick Drains ----
    _eval_wick_drains(
        is_clay, is_sand, soil_type, has_time_constraint, time_constraint_months,
        cv_m2_per_year, Hdr_m, settlement_problem,
        applicable, not_applicable, recommendations, sizing,
    )

    # ---- Surcharge Preloading ----
    _eval_surcharge(
        is_clay, is_sand, soil_type, has_time_constraint, time_constraint_months,
        settlement_problem, cv_m2_per_year, Hdr_m,
        applicable, not_applicable, recommendations, sizing,
    )

    # ---- Vibro-Compaction ----
    _eval_vibro(
        is_sand, soil_type, fines_content, N_spt, gwt_depth_m,
        applicable, not_applicable, recommendations, sizing,
    )

    # Final ranking / top recommendation
    if not applicable:
        recommendations.insert(
            0, "No standard ground improvement methods appear feasible — "
               "consider deep foundations"
        )

    return FeasibilityResult(
        applicable_methods=applicable,
        not_applicable=not_applicable,
        recommendations=recommendations,
        preliminary_sizing=sizing,
        soil_description=soil_desc,
        design_problem=design_desc,
    )


def _eval_aggregate_piers(
    is_clay, is_organic, cu_kPa, N_spt, thickness_m,
    settlement_problem, predicted_settlement_mm, allowable_settlement_mm,
    bearing_problem, required_bearing_kPa, current_bearing_kPa,
    applicable, not_applicable, recommendations, sizing,
):
    """Evaluate aggregate pier feasibility."""
    method = "Aggregate Piers"

    # Exclusions
    if is_organic and cu_kPa is not None and cu_kPa < 10:
        not_applicable.append({
            "method": method,
            "reason": f"Very soft organic soil (cu={cu_kPa:.0f} kPa) — "
                      "too weak to support pier installation"
        })
        return

    if thickness_m > 12:
        not_applicable.append({
            "method": method,
            "reason": f"Treatment depth ({thickness_m:.1f}m) exceeds practical "
                      "limit (~10-12 m) for aggregate piers"
        })
        return

    # Applicable conditions
    pier_ok = False
    if is_clay and cu_kPa is not None and 10 <= cu_kPa <= 100:
        pier_ok = True
    elif N_spt is not None and N_spt < 15:
        pier_ok = True
    elif is_clay and cu_kPa is None:
        pier_ok = True  # assume soft enough if clay with no cu data

    if not pier_ok:
        not_applicable.append({
            "method": method,
            "reason": "Soil is too stiff/dense for significant improvement"
        })
        return

    applicable.append(method)

    # Preliminary sizing
    pier_sizing = {
        "typical_column_diameter": "0.45 - 0.9 m",
        "typical_spacing": "1.5 - 3.0 m (triangular)",
        "typical_n": "3-8 (stress concentration ratio)",
    }

    if settlement_problem and predicted_settlement_mm > 0:
        required_SRF = allowable_settlement_mm / predicted_settlement_mm
        pier_sizing["required_SRF"] = f"{required_SRF:.2f}"
        if required_SRF < 0.2:
            recommendations.append(
                f"Aggregate piers: required SRF ({required_SRF:.2f}) is very aggressive — "
                "may need very tight spacing or combined approach"
            )
        else:
            recommendations.append(
                f"Aggregate piers can reduce settlement by "
                f"SRF={required_SRF:.2f} with appropriate spacing"
            )

    if bearing_problem:
        improvement_needed = required_bearing_kPa / current_bearing_kPa
        pier_sizing["bearing_improvement_needed"] = f"{improvement_needed:.1f}x"
        recommendations.append(
            f"Aggregate piers: need {improvement_needed:.1f}x bearing improvement"
        )

    sizing["aggregate_piers"] = pier_sizing


def _eval_wick_drains(
    is_clay, is_sand, soil_type, has_time_constraint, time_constraint_months,
    cv_m2_per_year, Hdr_m, settlement_problem,
    applicable, not_applicable, recommendations, sizing,
):
    """Evaluate wick drain feasibility."""
    method = "Wick Drains (PVD)"

    if soil_type == "loose_sand":
        not_applicable.append({
            "method": method,
            "reason": "Sand drains naturally — PVDs provide no benefit"
        })
        return

    if not is_clay:
        # mixed soils may still benefit if clay-dominant
        pass

    if not settlement_problem and not has_time_constraint:
        not_applicable.append({
            "method": method,
            "reason": "No settlement problem or time constraint — drains not needed"
        })
        return

    applicable.append(method)

    drain_sizing = {
        "typical_spacing": "1.0 - 2.5 m",
        "typical_pattern": "triangular",
        "typical_dw": "0.05 - 0.07 m",
    }

    if cv_m2_per_year is not None and Hdr_m is not None and has_time_constraint:
        # Estimate time without drains for 90% U
        from settlement.time_rate import time_for_consolidation
        t_no_drains = time_for_consolidation(90.0, cv_m2_per_year, Hdr_m)
        t_no_drains_months = t_no_drains * 12.0
        drain_sizing["time_without_drains_months"] = f"{t_no_drains_months:.0f}"

        if t_no_drains_months > time_constraint_months:
            recommendations.append(
                f"Wick drains recommended: 90% consolidation takes "
                f"{t_no_drains_months:.0f} months without drains vs "
                f"{time_constraint_months:.0f} month constraint"
            )
        else:
            recommendations.append(
                f"Consolidation may complete in {t_no_drains_months:.0f} months "
                "without drains — drains may not be necessary"
            )
    elif has_time_constraint:
        recommendations.append(
            "Wick drains can accelerate consolidation — "
            "provide cv and Hdr for preliminary spacing estimate"
        )

    sizing["wick_drains"] = drain_sizing


def _eval_surcharge(
    is_clay, is_sand, soil_type, has_time_constraint, time_constraint_months,
    settlement_problem, cv_m2_per_year, Hdr_m,
    applicable, not_applicable, recommendations, sizing,
):
    """Evaluate surcharge preloading feasibility."""
    method = "Surcharge Preloading"

    if soil_type == "loose_sand":
        not_applicable.append({
            "method": method,
            "reason": "Surcharge preloading is for compressible soils, "
                      "not loose sands — consider vibro-compaction instead"
        })
        return

    if not settlement_problem:
        not_applicable.append({
            "method": method,
            "reason": "No settlement problem — surcharge not needed"
        })
        return

    # Time check: surcharge alone is slow for thick clay layers
    tight_schedule = has_time_constraint and time_constraint_months < 6

    applicable.append(method)

    surcharge_sizing = {
        "typical_surcharge_height": "1 - 5 m of fill",
        "typical_fill_unit_weight": "18 - 20 kN/m³",
    }

    if tight_schedule:
        recommendations.append(
            "Surcharge alone may be too slow with tight schedule — "
            "combine with wick drains for faster consolidation"
        )
        surcharge_sizing["note"] = "Consider combining with wick drains"
    else:
        recommendations.append(
            "Surcharge preloading is viable — allow sufficient time "
            "for consolidation monitoring"
        )

    sizing["surcharge"] = surcharge_sizing


def _eval_vibro(
    is_sand, soil_type, fines_content, N_spt, gwt_depth_m,
    applicable, not_applicable, recommendations, sizing,
):
    """Evaluate vibro-compaction feasibility."""
    method = "Vibro-Compaction"

    if soil_type in ("soft_clay", "organic"):
        not_applicable.append({
            "method": method,
            "reason": "Not applicable to cohesive/organic soils"
        })
        return

    if fines_content is not None and fines_content > 20:
        not_applicable.append({
            "method": method,
            "reason": f"Fines content ({fines_content:.0f}%) too high (> 20%)"
        })
        return

    if N_spt is not None and N_spt > 25:
        not_applicable.append({
            "method": method,
            "reason": f"N_spt ({N_spt:.0f}) indicates soil is already dense"
        })
        return

    if not is_sand and fines_content is None:
        not_applicable.append({
            "method": method,
            "reason": "Soil type is not clearly sandy — "
                      "provide fines content for assessment"
        })
        return

    # Marginal check
    marginal = False
    if fines_content is not None and fines_content > 15:
        marginal = True

    applicable.append(method)

    vibro_sizing = {
        "typical_spacing": "1.5 - 3.5 m (triangular)",
        "target_N_spt": "20 - 30",
    }

    if marginal:
        recommendations.append(
            f"Vibro-compaction feasible but marginal (fines={fines_content:.0f}%) — "
            "field trial recommended"
        )
    else:
        fc_str = f" (fines={fines_content:.0f}%)" if fines_content is not None else ""
        recommendations.append(
            f"Vibro-compaction is well-suited for this loose sand{fc_str}"
        )

    if N_spt is not None:
        vibro_sizing["current_N_spt"] = f"{N_spt:.0f}"

    sizing["vibro_compaction"] = vibro_sizing

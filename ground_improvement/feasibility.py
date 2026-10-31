"""
Ground improvement feasibility evaluation and decision support.

Takes extracted soil parameters and design requirements and recommends
which ground improvement methods are applicable, with preliminary sizing.

This module takes plain floats/strings (NOT SoilProfile objects) to
avoid circular imports.

References:
    FHWA GEC-13 (FHWA-NHI-16-027): Ground Modification Methods Reference
    Manual, Vols. I and II (screens and typical sizes checked against the
    print, GM-291)
"""

from typing import Any, Dict, List, Optional

from ground_improvement.results import FeasibilityResult
from ground_improvement.vibro import (
    FIG_4_34, FINES_LIMIT_PERCENT, TARGETS_GEC13, density_from_N,
)

_FT = 0.3048
#: GEC-13 (FHWA-NHI-16-027) p. 5-9, item 7: "Typically, the maximum
#: practical depth of stone columns and rammed aggregate piers is 100 feet
#: and 35 feet respectively."
STONE_COLUMN_MAX_DEPTH_M = 100 * _FT
RAMMED_PIER_MAX_DEPTH_M = 35 * _FT
#: GEC-13 p. 4-65: vibro-compaction "allows maximum improvement of granular
#: soils to a depth of about 165 feet, with generally recommended depth of
#: about 100 feet."
VIBRO_MAX_DEPTH_M = 165 * _FT
VIBRO_RECOMMENDED_DEPTH_M = 100 * _FT
_PSF = 0.04788026  # kPa
#: GEC-13 p. 5-9, item 2: "The most significant improvement is likely to be
#: obtained in compressible silts and clays ranging in shear strength from
#: 300 to 1000 psf."
AGG_COLUMN_SU_BEST_KPA = (300 * _PSF, 1000 * _PSF)
#: GEC-13 p. 5-9, item 5: "a practical upper limit is in the range of an
#: undrained strength of 1000 to 2000 psf for stone columns. If stone
#: columns are used in these stiff soils or through stiff lenses, the column
#: hole is commonly pre-bored".
AGG_COLUMN_SU_UPPER_KPA = (1000 * _PSF, 2000 * _PSF)


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
        Depth to top of the treatable layer (m). The treatment reaches
        depth_to_top_m + thickness_m, which is held against GEC-13's
        printed depths (GM-179): aggregate columns to 100 ft (30.5 m) as
        stone columns, 35 ft (10.7 m) as rammed aggregate piers (p. 5-9);
        vibro-compaction to about 165 ft (50 m), about 100 ft (30.5 m)
        generally recommended (p. 4-65).
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
        Depth to groundwater table (m). Not used by the screening (GM-179):
        GEC-13 finds vibro-compaction "effective both above and below the
        natural groundwater level" (p. 4-65), and no other screen here
        reads it. Kept so that older calls still run.

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
        is_clay, is_organic, cu_kPa, N_spt, thickness_m, depth_to_top_m,
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
        is_sand, soil_type, fines_content, N_spt, thickness_m, depth_to_top_m,
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


def _treatment_depth(depth_to_top_m, thickness_m):
    """The depth the treatment reaches, and how it was made up."""
    bottom = depth_to_top_m + thickness_m
    return bottom, (f"{bottom:.1f} m ({depth_to_top_m:.1f} m to the top of "
                    f"the layer + {thickness_m:.1f} m thick)")


def _eval_aggregate_piers(
    is_clay, is_organic, cu_kPa, N_spt, thickness_m, depth_to_top_m,
    settlement_problem, predicted_settlement_mm, allowable_settlement_mm,
    bearing_problem, required_bearing_kPa, current_bearing_kPa,
    applicable, not_applicable, recommendations, sizing,
):
    """Evaluate aggregate pier feasibility."""
    method = "Aggregate Piers"

    su_lo, su_hi = AGG_COLUMN_SU_UPPER_KPA
    bottom, reaches = _treatment_depth(depth_to_top_m, thickness_m)
    if bottom > STONE_COLUMN_MAX_DEPTH_M:
        not_applicable.append({
            "method": method,
            "reason": f"Treatment to {reaches} is beyond the typical "
                      "maximum practical depth of stone columns, 100 ft "
                      "(30.5 m), and of rammed aggregate piers, 35 ft "
                      "(10.7 m) (GEC-13 p. 5-9: 'Typically, the maximum "
                      "practical depth of stone columns and rammed "
                      "aggregate piers is 100 feet and 35 feet "
                      "respectively')"
        })
        return

    # Applicable conditions (GM-291: the undrained strength screen as GEC-13
    # p. 5-9 prints it; until then 10 to 100 kPa, an organic soil under 10
    # kPa refused, printed nowhere). The N_spt < 15 screen of a granular
    # soil is not printed in GEC-13 Ch. 5 either; kept, said in CHECK.md.
    pier_ok = False
    if is_clay and cu_kPa is not None and cu_kPa <= su_hi:
        pier_ok = True
    elif N_spt is not None and N_spt < 15:
        pier_ok = True
    elif is_clay and cu_kPa is None:
        pier_ok = True  # assume soft enough if clay with no cu data

    if not pier_ok:
        reason = "Soil is too stiff/dense for significant improvement"
        if is_clay and cu_kPa is not None:
            reason = (f"cu={cu_kPa:.0f} kPa is above GEC-13's practical "
                      "upper limit for stone columns, 'in the range of an "
                      "undrained strength of 1000 to 2000 psf' (48 to 96 "
                      "kPa), beyond which 'the column hole is commonly "
                      "pre-bored' (p. 5-9)")
        not_applicable.append({"method": method, "reason": reason})
        return

    applicable.append(method)
    if is_clay and cu_kPa is not None:
        best_lo, best_hi = AGG_COLUMN_SU_BEST_KPA
        if cu_kPa > su_lo:
            recommendations.append(
                f"Aggregate piers: cu={cu_kPa:.0f} kPa is within GEC-13's "
                "practical upper limit for stone columns, 1000 to 2000 psf "
                "(48 to 96 kPa); in such stiff soils 'the column hole is "
                "commonly pre-bored' (p. 5-9)")
        elif cu_kPa < best_lo:
            recommendations.append(
                f"Aggregate piers: cu={cu_kPa:.0f} kPa is below the 300 to "
                "1000 psf (14 to 48 kPa) in which GEC-13 finds 'the most "
                "significant improvement' (p. 5-9)"
                + ("; 'Special care must be taken when using aggregate "
                   "columns in soils containing organics and peat lenses or "
                   "layers with undrained shear strength of less than 300 "
                   "psf' (p. 5-9)" if is_organic else ""))
        if is_organic:
            recommendations.append(
                "Aggregate piers: GEC-13 p. 5-9, 'When the thickness of the "
                "organic layer is greater than one to two aggregate column "
                "diameters, the ability to develop consistent column "
                "diameters becomes questionable'")

    # Preliminary sizing: GEC-13 as printed (GM-291)
    pier_sizing = {
        "typical_column_diameter": (
            "1.5 to 4 ft (0.46 to 1.22 m), typically 3.0 to 3.6 ft for the "
            "dry method (GEC-13 p. 5-31)"),
        "typical_spacing": "5 to 12 ft (1.5 to 3.7 m) (GEC-13 p. 5-31)",
        "typical_n": (
            "stress concentration ratio 2.0 to 5.0 as measured, 2.5 often "
            "used for preliminary design (GEC-13 pp. 5-31, 5-32); rammed "
            "aggregate piers 5 to 10 under embankments, 10 under rigid "
            "footings (p. 5-43)"),
    }
    if bottom > RAMMED_PIER_MAX_DEPTH_M:
        pier_sizing["column_type"] = "stone columns"
        recommendations.append(
            f"Aggregate piers: treatment to {reaches} is beyond the typical "
            "maximum practical depth of rammed aggregate piers, 35 ft "
            "(10.7 m), and within that of stone columns, 100 ft (30.5 m) "
            "(GEC-13 p. 5-9): of the aggregate columns, stone columns"
        )

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
        "typical_spacing": (
            "3 to 5 ft (0.9 to 1.5 m) in homogeneous clays, 4 to 6 ft in "
            "silty clays, 5 to 6.5 ft in coarser soils, for about 90 "
            "percent consolidation in 3 to 4 months (GEC-13 p. 2-33)"),
        "typical_pattern": "triangular",
        "typical_dw": (
            "1.5 to 5.5 in (0.04 to 0.14 m), most commonly 2.5 in "
            "(0.064 m) (GEC-13 p. 2-35)"),
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

    # GM-291: GEC-13 prints no typical surcharge height or fill unit
    # weight (until then "1 - 5 m of fill", "18 - 20 kN/m³")
    surcharge_sizing = {
        "basis": (
            "GEC-13 prints no typical surcharge height: 'There are three "
            "basic variables that can be manipulated ... time, PVD "
            "spacing, and surcharge' (p. 2-36); the surcharge_preloading "
            "tool works the time for a given surcharge"),
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
    is_sand, soil_type, fines_content, N_spt, thickness_m, depth_to_top_m,
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

    fines_lo, fines_hi = FINES_LIMIT_PERCENT
    if fines_content is not None and fines_content > fines_hi:
        not_applicable.append({
            "method": method,
            "reason": f"Fines content ({fines_content:.0f}%) is above "
                      "GEC-13's 12 to 15 percent: 'proper densification "
                      "generally cannot be achieved when the granular soil "
                      "contains more than 12 to 15 percent silt OR more "
                      "than 2 percent clay' (p. 4-65)"
        })
        return

    if N_spt is not None and density_from_N(N_spt)[1][0] >= 85.0:
        not_applicable.append({
            "method": method,
            "reason": f"N_spt ({N_spt:.0f}) is very dense by GEC-13 Table "
                      "4-8 (relative density 85 to 100 percent); p. 4-75: "
                      "'It is unlikely, even for heavy loading, that it "
                      "will be necessary to achieve a relative density "
                      "above 85 percent'"
        })
        return

    if not is_sand and fines_content is None:
        not_applicable.append({
            "method": method,
            "reason": "Soil type is not clearly sandy — "
                      "provide fines content for assessment"
        })
        return

    bottom, reaches = _treatment_depth(depth_to_top_m, thickness_m)
    if bottom > VIBRO_MAX_DEPTH_M:
        not_applicable.append({
            "method": method,
            "reason": f"Treatment to {reaches} is beyond the depth to which "
                      "vibro-compaction improves granular soils, about "
                      "165 ft (50 m) (GEC-13 p. 4-65)"
        })
        return

    # Marginal check: within GEC-13's 12 to 15 percent (p. 4-65)
    marginal = False
    if fines_content is not None and fines_content > fines_lo:
        marginal = True

    applicable.append(method)

    vibro_sizing = {
        "typical_spacing": (
            f"{FIG_4_34[max(FIG_4_34)]:.2f} to {FIG_4_34[min(FIG_4_34)]:.2f} "
            f"m for a target relative density of {max(FIG_4_34):.0f} to "
            f"{min(FIG_4_34):.0f} percent, silty sands (GEC-13 Fig. 4-34, "
            "p. 4-77)"),
        "target_relative_density": TARGETS_GEC13,
    }

    if marginal:
        recommendations.append(
            f"Vibro-compaction feasible but marginal (fines={fines_content:.0f}%, "
            "within GEC-13's 12 to 15 percent, p. 4-65) — "
            "field trial recommended"
        )
    else:
        fc_str = f" (fines={fines_content:.0f}%)" if fines_content is not None else ""
        recommendations.append(
            f"Vibro-compaction is well-suited for this loose sand{fc_str}"
        )

    if bottom > VIBRO_RECOMMENDED_DEPTH_M:
        recommendations.append(
            f"Vibro-compaction: treatment to {reaches} is beyond the "
            "generally recommended depth of about 100 ft (30.5 m), within "
            "the maximum of about 165 ft (50 m) (GEC-13 p. 4-65)"
        )

    if N_spt is not None:
        vibro_sizing["current_N_spt"] = f"{N_spt:.0f}"

    sizing["vibro_compaction"] = vibro_sizing

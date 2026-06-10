"""Ground improvement adapter — aggregate piers, wick drains, vibro, surcharge."""

import inspect
import math

from funhouse_agent.adapters import (
    apply_aliases, clean_result, reject_unknown_params, require_params,
)
from ground_improvement import (
    analyze_aggregate_piers, analyze_wick_drains, design_drain_spacing,
    analyze_surcharge_preloading, analyze_vibro_compaction, evaluate_feasibility,
)


def _valid_kwargs(func):
    return set(inspect.signature(func).parameters)


def _call(func, p, *, method, aliases=None, required=()):
    """Alias-map, validate, and call a module function with **p."""
    if aliases:
        p = apply_aliases(p, aliases)
    valid = _valid_kwargs(func)
    reject_unknown_params(p, valid, method=method)
    require_params(p, required, method=method, valid=valid)
    return clean_result(func(**p).to_dict())


# Aliases: names the agent reaches for (and that older METHOD_INFO advertised)
# mapped onto the module's actual keyword names.
_AGG_ALIASES = {"column_modulus": "E_column", "soil_modulus": "E_soil",
                "stress_concentration_ratio": "n", "diameter": "column_diameter"}
_WICK_ALIASES = {"drain_spacing": "spacing", "time_years": "time",
                 "drain_diameter": "dw"}
_DESIGN_ALIASES = {"time_years": "target_time", "time": "target_time",
                   "drain_diameter": "dw"}
_SURCHARGE_ALIASES = {"surcharge_pressure": "surcharge_kPa",
                      "surcharge": "surcharge_kPa",
                      "S_ult": "S_ultimate", "settlement_m": "S_ultimate"}
_VIBRO_ALIASES = {"D50_mm": "D50", "initial_N": "initial_N_spt",
                  "target_N": "target_N_spt"}
_FEAS_ALIASES = {"treatment_depth": "thickness_m", "thickness": "thickness_m",
                 "N_spt_avg": "N_spt"}


def _run_aggregate_piers(p):
    p = apply_aliases(p, _AGG_ALIASES)
    # Accept the commonly-guessed area_replacement_ratio (As/A) and convert it
    # to a center-to-center spacing for the given diameter + pattern.
    if "area_replacement_ratio" in p:
        p = dict(p)
        ar = p.pop("area_replacement_ratio")
        if "spacing" not in p:
            require_params(p, ["column_diameter"], method="aggregate_piers",
                           valid=_valid_kwargs(analyze_aggregate_piers))
            d = p["column_diameter"]
            A_col = math.pi * d ** 2 / 4.0
            trib = 0.866 if p.get("pattern", "triangular") == "triangular" else 1.0
            p["spacing"] = math.sqrt(A_col / (ar * trib))
    return _call(analyze_aggregate_piers, p, method="aggregate_piers",
                 required=["column_diameter", "spacing"])


def _run_wick_drains(p):
    return _call(analyze_wick_drains, p, method="wick_drains",
                 aliases=_WICK_ALIASES,
                 required=["spacing", "ch", "cv", "Hdr", "time"])


def _run_design_drain_spacing(p):
    return _call(design_drain_spacing, p, method="design_drain_spacing",
                 aliases=_DESIGN_ALIASES,
                 required=["target_U", "target_time", "ch", "cv", "Hdr"])


def _run_surcharge_preloading(p):
    return _call(analyze_surcharge_preloading, p, method="surcharge_preloading",
                 aliases=_SURCHARGE_ALIASES,
                 required=["S_ultimate", "surcharge_kPa", "cv", "Hdr"])


def _run_vibro_compaction(p):
    return _call(analyze_vibro_compaction, p, method="vibro_compaction",
                 aliases=_VIBRO_ALIASES,
                 required=["fines_content", "initial_N_spt"])


def _run_feasibility(p):
    return _call(evaluate_feasibility, p, method="feasibility",
                 aliases=_FEAS_ALIASES, required=["soil_type"])


_PRIEBE_ALIASES = {"area_replacement_ratio": "as_ratio", "as": "as_ratio",
                   "ar": "as_ratio", "phi_c": "phi_column",
                   "friction_angle_column": "phi_column", "nu": "nu_soil"}


def _run_priebe_improvement_factor(p):
    from ground_improvement.aggregate_piers import priebe_basic_improvement_factor
    p = apply_aliases(p, _PRIEBE_ALIASES)
    valid = _valid_kwargs(priebe_basic_improvement_factor)
    reject_unknown_params(p, valid, method="priebe_improvement_factor")
    require_params(p, ["as_ratio"], method="priebe_improvement_factor", valid=valid)
    n0 = priebe_basic_improvement_factor(**p)
    return {
        "n0": round(n0, 3),
        "as_ratio": p["as_ratio"],
        "phi_column_deg": p.get("phi_column", 42.5),
        "nu_soil": p.get("nu_soil", round(1.0 / 3.0, 6)),
        "note": "Priebe (1995) basic improvement factor: settlement_unimproved / settlement_improved.",
    }


METHOD_REGISTRY = {
    "aggregate_piers": _run_aggregate_piers,
    "wick_drains": _run_wick_drains,
    "design_drain_spacing": _run_design_drain_spacing,
    "surcharge_preloading": _run_surcharge_preloading,
    "vibro_compaction": _run_vibro_compaction,
    "feasibility": _run_feasibility,
    "priebe_improvement_factor": _run_priebe_improvement_factor,
}

_PATTERN = {"type": "str", "required": False, "default": "triangular",
            "allowed_values": ["triangular", "square"],
            "description": "Installation pattern."}

METHOD_INFO = {
    "aggregate_piers": {
        "category": "Aggregate Piers",
        "brief": "Aggregate pier (stone column) design: settlement reduction, capacity increase.",
        "parameters": {
            "column_diameter": {"type": "float", "required": True, "description": "Pier column diameter (m)."},
            "spacing": {"type": "float", "required": True, "description": "Center-to-center spacing (m). Alternatively give area_replacement_ratio (As/A, 0.1-0.35 typical) and spacing is back-calculated."},
            "area_replacement_ratio": {"type": "float", "required": False, "description": "Area replacement ratio As/A (0.1-0.35 typical). Used to compute spacing when spacing is not given."},
            "pattern": _PATTERN,
            "E_column": {"type": "float", "required": False, "default": 80000.0, "description": "Aggregate column modulus (kPa)."},
            "E_soil": {"type": "float", "required": False, "default": 5000.0, "description": "Surrounding soil modulus (kPa)."},
            "n": {"type": "float", "required": False, "default": 5.0, "description": "Stress concentration ratio."},
            "q_unreinforced": {"type": "float", "required": False, "default": 0.0, "description": "Unreinforced bearing capacity (kPa). 0 = not computed."},
            "S_unreinforced": {"type": "float", "required": False, "default": 0.0, "description": "Unreinforced settlement (mm). 0 = not computed."},
        },
        "returns": {"area_replacement_ratio": "As/A achieved.", "settlement_improvement_factor": "Settlement reduction factor."},
    },
    "priebe_improvement_factor": {
        "category": "Aggregate Piers",
        "brief": "Priebe (1995) basic improvement factor n0 for vibro replacement / stone columns.",
        "parameters": {
            "as_ratio": {"type": "float", "required": True, "description": "Area replacement ratio as = Ac/A, 0 < as < 1 (0.1-0.35 typical). Alias: area_replacement_ratio."},
            "phi_column": {"type": "float", "required": False, "default": 42.5, "description": "Friction angle of the column material (degrees)."},
            "nu_soil": {"type": "float", "required": False, "default": 0.333, "description": "Poisson's ratio of the surrounding soil."},
        },
        "returns": {"n0": "Basic improvement factor (settlement reduction ratio, >= 1)."},
    },
    "wick_drains": {
        "category": "Wick Drains",
        "brief": "Prefabricated vertical drain (PVD) consolidation analysis.",
        "parameters": {
            "spacing": {"type": "float", "required": True, "description": "Center-to-center drain spacing (m)."},
            "ch": {"type": "float", "required": True, "description": "Horizontal coefficient of consolidation (m2/yr)."},
            "cv": {"type": "float", "required": True, "description": "Vertical coefficient of consolidation (m2/yr)."},
            "Hdr": {"type": "float", "required": True, "description": "Vertical drainage path (m)."},
            "time": {"type": "float", "required": True, "description": "Analysis time (years)."},
            "dw": {"type": "float", "required": False, "default": 0.066, "description": "Equivalent drain diameter (m)."},
            "pattern": _PATTERN,
            "smear_ratio": {"type": "float", "required": False, "default": 2.0, "description": "Smear zone ratio ds/dw."},
            "kh_ks_ratio": {"type": "float", "required": False, "default": 2.0, "description": "Permeability ratio kh/ks."},
        },
        "returns": {"U_total_percent": "Total degree of consolidation (%)."},
    },
    "design_drain_spacing": {
        "category": "Wick Drains",
        "brief": "Find required drain spacing for target consolidation in target time.",
        "parameters": {
            "target_U": {"type": "float", "required": True, "description": "Target combined degree of consolidation (percent, e.g. 90)."},
            "target_time": {"type": "float", "required": True, "description": "Time available to reach target_U (years)."},
            "ch": {"type": "float", "required": True, "description": "Horizontal coefficient of consolidation (m2/yr)."},
            "cv": {"type": "float", "required": True, "description": "Vertical coefficient of consolidation (m2/yr)."},
            "Hdr": {"type": "float", "required": True, "description": "Vertical drainage path (m)."},
            "dw": {"type": "float", "required": False, "default": 0.066, "description": "Equivalent drain diameter (m)."},
            "pattern": _PATTERN,
        },
        "returns": {"spacing_m": "Designed drain spacing.", "U_total_percent": "Consolidation achieved at target time."},
    },
    "surcharge_preloading": {
        "category": "Surcharge",
        "brief": "Surcharge preloading settlement and time analysis (optionally with wick drains).",
        "parameters": {
            "S_ultimate": {"type": "float", "required": True, "description": "Ultimate consolidation settlement under the surcharge (m). Compute first via the settlement module if unknown."},
            "surcharge_kPa": {"type": "float", "required": True, "description": "Applied surcharge pressure (kPa)."},
            "cv": {"type": "float", "required": True, "description": "Vertical coefficient of consolidation (m2/yr)."},
            "Hdr": {"type": "float", "required": True, "description": "Vertical drainage path (m)."},
            "target_U": {"type": "float", "required": False, "default": 90.0, "description": "Target degree of consolidation (percent)."},
            "ch": {"type": "float", "required": False, "description": "Horizontal coefficient of consolidation (m2/yr). Give with drain_spacing to include wick drains."},
            "drain_spacing": {"type": "float", "required": False, "description": "Wick drain spacing (m), if drains are used."},
            "pattern": _PATTERN,
            "sigma_v0": {"type": "float", "required": False, "default": 0.0, "description": "Current effective stress at layer center (kPa)."},
        },
        "returns": {"time_to_target_years": "Time to achieve target consolidation."},
    },
    "vibro_compaction": {
        "category": "Vibro Compaction",
        "brief": "Vibro compaction (vibroflotation) feasibility and densification.",
        "parameters": {
            "fines_content": {"type": "float", "required": True, "description": "Percent passing #200 sieve (0-100)."},
            "initial_N_spt": {"type": "float", "required": True, "description": "Current SPT blow count."},
            "target_N_spt": {"type": "float", "required": False, "default": 25.0, "description": "Desired SPT blow count."},
            "D50": {"type": "float", "required": False, "description": "Median grain size (mm)."},
            "pattern": _PATTERN,
        },
        "returns": {"is_feasible": "Whether vibro compaction is suitable."},
    },
    "feasibility": {
        "category": "General",
        "brief": "Ground improvement method feasibility screening.",
        "parameters": {
            "soil_type": {"type": "str", "required": True, "allowed_values": ["soft_clay", "loose_sand", "mixed", "organic"], "description": "Soil type to improve."},
            "fines_content": {"type": "float", "required": False, "description": "Percent passing #200 sieve. Required for vibro assessment."},
            "N_spt": {"type": "float", "required": False, "description": "Average SPT N in the treatment zone."},
            "cu_kPa": {"type": "float", "required": False, "description": "Average undrained shear strength (kPa)."},
            "thickness_m": {"type": "float", "required": False, "default": 0.0, "description": "Thickness of the compressible/treatable layer (m)."},
            "depth_to_top_m": {"type": "float", "required": False, "default": 0.0, "description": "Depth to top of the treatable layer (m)."},
            "required_bearing_kPa": {"type": "float", "required": False, "default": 0.0, "description": "Required bearing capacity (kPa). 0 = not a bearing problem."},
            "predicted_settlement_mm": {"type": "float", "required": False, "default": 0.0, "description": "Predicted settlement without improvement (mm)."},
            "allowable_settlement_mm": {"type": "float", "required": False, "default": 50.0, "description": "Allowable settlement limit (mm)."},
            "time_constraint_months": {"type": "float", "required": False, "default": 0.0, "description": "Time available (months). 0 = no constraint."},
        },
        "returns": {"recommended_methods": "List of feasible methods.", "is_feasible": "Overall feasibility."},
    },
}

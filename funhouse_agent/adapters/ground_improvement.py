"""Ground improvement adapter — aggregate piers, wick drains, vibro, surcharge."""

import inspect
import math

from funhouse_agent.adapters import (
    apply_aliases, clean_result, declare_aliases, declare_applies_when,
    reject_unknown_params, require_params,
)
from ground_improvement import (
    analyze_aggregate_piers, analyze_wick_drains, design_drain_spacing,
    analyze_surcharge_preloading, analyze_vibro_compaction, evaluate_feasibility,
)


def _valid_kwargs(func):
    return set(inspect.signature(func).parameters)


def _run(func, p, *, method, aliases=None, required=(), refused=()):
    """Alias-map, validate, and call a module function with **p: (result,
    inputs in force), each of the function's inputs as given or its
    default. ``refused`` names inputs the function takes and the tool
    does not (GM-179): they change no result, so they are refused, not
    passed."""
    if aliases:
        p = apply_aliases(p, aliases)
    valid = _valid_kwargs(func) - set(refused)
    reject_unknown_params(p, valid, method=method)
    require_params(p, required, method=method, valid=valid)
    used = {name: p.get(name, prm.default)
            for name, prm in inspect.signature(func).parameters.items()
            if name in valid}
    return func(**p), used


def _call(func, p, *, method, aliases=None, required=(), refused=()):
    """Alias-map, validate, and call a module function with **p."""
    result, _ = _run(func, p, method=method, aliases=aliases,
                     required=required, refused=refused)
    return clean_result(result.to_dict())


# Aliases: names the agent reaches for (and that older METHOD_INFO advertised)
# mapped onto the module's actual keyword names.
_AGG_ALIASES = {"soil_modulus": "E_soil",
                "stress_concentration_ratio": "n", "diameter": "column_diameter",
                "column_modulus": "E_column"}
# GM-179: with the stress-ratio settlement method the column modulus
# changes no result (only a warning below E_soil): the stress split is n,
# given. Refused there by each of its names; GEC-13 Fig. 5-21's methods
# take it (GM-220).
_BY_MODULI = ("weighted_modulus", "balaam_rigid", "balaam_flexible")
_NO_E_COLUMN = (
    "the stress-ratio settlement method (settlement_method "
    "'stress_ratio', the default) does not use the column's modulus: the "
    "stress split is the stress concentration ratio n, taken as given, "
    "not worked out from the moduli. Give n (GEC-13: 'between 2.0 and "
    "5.0' for stone columns, Sec. 4.1.2.4; for rammed aggregate piers '4 "
    "to 45 for rigid footings' and 5 to 10 under embankments, p. 5-45), "
    "or a settlement_method that takes E_c / E_s off GEC-13 Fig. 5-21: "
    "'weighted_modulus', 'balaam_rigid' or 'balaam_flexible'")
# GM-179: no screen reads the water table.
_NO_GWT = (
    "the feasibility screening does not use the water table: GEC-13 finds "
    "vibro-compaction 'effective both above and below the natural "
    "groundwater level' (p. 4-65), and no other screen here reads it. "
    "Give the layer's depth_to_top_m and thickness_m, which the screening "
    "holds against GEC-13's depth limits")
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


# The tools below as (result, inputs in force): their own code path, also
# run by ``calc_package.ground_improvement_package`` (GM-141). ``method``
# names the caller in error messages.

def run_aggregate_piers(p, *, method="aggregate_piers"):
    by_moduli = p.get("settlement_method") in _BY_MODULI
    column = [k for k in ("E_column", "column_modulus") if k in p]
    if column and not by_moduli:
        raise ValueError(f"{method} does not take {column}: {_NO_E_COLUMN}.")
    p = apply_aliases(p, _AGG_ALIASES)
    # Accept the commonly-guessed area_replacement_ratio (As/A) and convert it
    # to a center-to-center spacing for the given diameter + pattern.
    if "area_replacement_ratio" in p:
        p = dict(p)
        ar = p.pop("area_replacement_ratio")
        if "spacing" not in p:
            require_params(p, ["column_diameter"], method=method,
                           valid=_valid_kwargs(analyze_aggregate_piers))
            d = p["column_diameter"]
            A_col = math.pi * d ** 2 / 4.0
            trib = 0.866 if p.get("pattern", "triangular") == "triangular" else 1.0
            p["spacing"] = math.sqrt(A_col / (ar * trib))
    # GEC-13 Fig. 5-21's methods are by E_c / E_s: both moduli are given
    return _run(analyze_aggregate_piers, p, method=method,
                required=["column_diameter", "spacing"]
                + (["E_column", "E_soil"] if by_moduli else []),
                refused=() if by_moduli else ("E_column",))


def run_wick_drains(p, *, method="wick_drains"):
    return _run(analyze_wick_drains, p, method=method,
                aliases=_WICK_ALIASES,
                required=["spacing", "ch", "cv", "Hdr", "time"])


def run_surcharge_preloading(p, *, method="surcharge_preloading"):
    return _run(analyze_surcharge_preloading, p, method=method,
                aliases=_SURCHARGE_ALIASES,
                required=["S_ultimate", "surcharge_kPa", "cv", "Hdr"])


def run_vibro_compaction(p, *, method="vibro_compaction"):
    return _run(analyze_vibro_compaction, p, method=method,
                aliases=_VIBRO_ALIASES,
                required=["fines_content", "initial_N_spt"])


def _run_aggregate_piers(p):
    return clean_result(run_aggregate_piers(p)[0].to_dict())


def _run_wick_drains(p):
    return clean_result(run_wick_drains(p)[0].to_dict())


def _run_design_drain_spacing(p):
    return _call(design_drain_spacing, p, method="design_drain_spacing",
                 aliases=_DESIGN_ALIASES,
                 required=["target_U", "target_time", "ch", "cv", "Hdr"])


def _run_surcharge_preloading(p):
    return clean_result(run_surcharge_preloading(p)[0].to_dict())


def _run_vibro_compaction(p):
    return clean_result(run_vibro_compaction(p)[0].to_dict())


def _run_feasibility(p):
    return _call(evaluate_feasibility, p, method="feasibility",
                 aliases=_FEAS_ALIASES, required=["soil_type"],
                 refused=("gwt_depth_m",))


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
        "nu_soil": p.get("nu_soil", 1.0 / 3.0),
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
            "E_soil": {"type": "float", "required": False, "default": 5000.0, "description": "Surrounding soil modulus (kPa). With settlement_method 'stress_ratio', used only for the composite modulus E_comp = E_soil*[1 + as*(n-1)]; the GEC-13 Fig. 5-21 methods need it given, for E_c/E_s."},
            "n": {"type": "float", "required": False, "default": 5.0, "description": "Stress concentration ratio sigma_c/sigma_s; sets the improved bearing and, with settlement_method 'stress_ratio', the settlement reduction factor. GEC-13: 'between 2.0 and 5.0' for stone columns (Sec. 4.1.2.4); for rammed aggregate piers '4 to 45 for rigid footings' and 5 to 10 under embankments (p. 5-45)."},
            "settlement_method": {"type": "str", "required": False, "default": "stress_ratio", "allowed_values": ["stress_ratio", *_BY_MODULI], "description": "How the settlement over the columns is worked out. 'stress_ratio' (default): SRF = 1/[1 + as*(n-1)] from n. The others give the settlement ratio n' (settlement without columns / with them) by E_c/E_s off GEC-13 Fig. 5-21 (p. 5-36), and need E_column and E_soil: 'weighted_modulus' (curve 2, n' = 1 + as*(E_c/E_s - 1), any E_c/E_s); 'balaam_rigid' (curves 5b, 5c: E_c/E_s 5 to 10, A/A_c 1.75 to 9); 'balaam_flexible' (curve 5a: E_c/E_s = 10, A/A_c 1.5 to 8.5). GEC-13 p. 5-37: the FHWA (1983) Equilibrium Method, 'roughly equivalent to the Balaam relationships', 'represents an average or lower bound estimate suitable for preliminary analyses'; the Priebe curve (priebe_improvement_factor) 'the upper bound effectiveness'."},
            "E_column": {"type": "float", "required": False, "description": "Aggregate column modulus E_c (kPa), with E_soil, for the settlement_method 'weighted_modulus', 'balaam_rigid' or 'balaam_flexible' (E_c/E_s, GEC-13 Fig. 5-21). Refused with 'stress_ratio', which takes n instead. Alias: column_modulus."},
            "q_unreinforced": {"type": "float", "required": False, "default": 0.0, "description": "Unreinforced bearing capacity (kPa). 0 = not computed."},
            "S_unreinforced": {"type": "float", "required": False, "default": 0.0, "description": "Unreinforced settlement (mm). 0 = not computed."},
        },
        "returns": {"area_replacement_ratio": "As/A achieved.", "settlement_reduction_factor": "SRF: settlement with columns / without (1/n' with the Fig. 5-21 methods).", "settlement_ratio": "Fig. 5-21 methods only: n' = settlement without columns / with them.", "modular_ratio": "Fig. 5-21 methods only: E_c/E_s."},
    },
    "priebe_improvement_factor": {
        "category": "Aggregate Piers",
        "brief": "Priebe (1995) basic improvement factor n0 for vibro replacement / stone columns.",
        "parameters": {
            "as_ratio": {"type": "float", "required": True, "description": "Area replacement ratio as = Ac/A, 0 < as < 1 (0.1-0.35 typical). Alias: area_replacement_ratio."},
            "phi_column": {"type": "float", "required": False, "default": 42.5, "description": "Friction angle of the column material (degrees)."},
            "nu_soil": {"type": "float", "required": False, "default": 1.0 / 3.0, "description": "Poisson's ratio of the surrounding soil (default 1/3)."},
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
            "n_time_points": {"type": "int", "required": False, "default": 50, "description": "Points in the time-settlement curve."},
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
            "smear_ratio": {"type": "float", "required": False, "default": 2.0, "description": "Smear zone ratio ds/dw."},
            "kh_ks_ratio": {"type": "float", "required": False, "default": 2.0, "description": "Permeability ratio kh/ks."},
            "spacing_range": {"type": "array", "required": False, "default": [1.0, 3.5], "description": "Spacing search range [min, max] (m)."},
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
            "dw": {"type": "float", "required": False, "default": 0.066, "description": "Equivalent drain diameter (m), with drains."},
            "smear_ratio": {"type": "float", "required": False, "default": 2.0, "description": "Smear zone ratio ds/dw, with drains."},
            "kh_ks_ratio": {"type": "float", "required": False, "default": 2.0, "description": "Permeability ratio kh/ks, with drains."},
            "sigma_v0": {"type": "float", "required": False, "default": 0.0, "description": "Current effective stress at layer center (kPa)."},
            "n_time_points": {"type": "int", "required": False, "default": 50, "description": "Points in the time-settlement curve."},
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
            "thickness_m": {"type": "float", "required": False, "default": 0.0, "description": "Thickness of the compressible/treatable layer (m). The treatment reaches depth_to_top_m + thickness_m (see depth_to_top_m)."},
            "depth_to_top_m": {"type": "float", "required": False, "default": 0.0, "description": "Depth to the top of the treatable layer (m). The treatment reaches depth_to_top_m + thickness_m, held against GEC-13's printed depths: aggregate piers not applicable beyond 100 ft (30.5 m), the typical maximum practical depth of stone columns, and stone columns only beyond 35 ft (10.7 m), that of rammed aggregate piers (p. 5-9: 'Typically, the maximum practical depth of stone columns and rammed aggregate piers is 100 feet and 35 feet respectively'); vibro-compaction not applicable beyond about 165 ft (50 m) and noted beyond the generally recommended about 100 ft (30.5 m) (p. 4-65)."},
            "required_bearing_kPa": {"type": "float", "required": False, "default": 0.0, "description": "Required bearing capacity (kPa). 0 = not a bearing problem."},
            "current_bearing_kPa": {"type": "float", "required": False, "default": 0.0, "description": "Current (unimproved) bearing capacity (kPa)."},
            "predicted_settlement_mm": {"type": "float", "required": False, "default": 0.0, "description": "Predicted settlement without improvement (mm)."},
            "allowable_settlement_mm": {"type": "float", "required": False, "default": 50.0, "description": "Allowable settlement limit (mm)."},
            "time_constraint_months": {"type": "float", "required": False, "default": 0.0, "description": "Time available (months). 0 = no constraint."},
            "cv_m2_per_year": {"type": "float", "required": False, "description": "Vertical coefficient of consolidation (m2/yr)."},
            "Hdr_m": {"type": "float", "required": False, "description": "Vertical drainage path length (m)."},
        },
        "returns": {"recommended_methods": "List of feasible methods.", "is_feasible": "Overall feasibility."},
    },
}

declare_aliases(METHOD_INFO, {
    "aggregate_piers": _AGG_ALIASES,
    "wick_drains": _WICK_ALIASES,
    "design_drain_spacing": _DESIGN_ALIASES,
    "surcharge_preloading": _SURCHARGE_ALIASES,
    "vibro_compaction": _VIBRO_ALIASES,
    "feasibility": _FEAS_ALIASES,
    "priebe_improvement_factor": _PRIEBE_ALIASES,
})

# When each default applies (GM-119): the drain inputs of a preload to a
# preload with drains.
declare_applies_when(METHOD_INFO, {
    "surcharge_preloading": {
        name: {"drain_spacing": "set"}
        for name in ("pattern", "dw", "smear_ratio", "kh_ks_ratio")},
    "aggregate_piers": {
        "E_column": {"settlement_method": list(_BY_MODULI)},
        # with a Fig. 5-21 method n sets only the improved bearing
        "n": [{"settlement_method": ["stress_ratio"]},
              {"q_unreinforced": "set"}]}})

# What a left-out input with no default stands for (BACKLOG GM-165): the
# screening's consolidation time needs cv and Hdr.
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
_NO_TIME = ("consolidation time not estimated",
            "the time to 90 % consolidation without drains needs both cv "
            "and Hdr; the result asks for them")
declare_when_absent(METHOD_INFO, {
    "feasibility": {"cv_m2_per_year": _NO_TIME, "Hdr_m": _NO_TIME}})
for _name in ("cv_m2_per_year", "Hdr_m"):
    METHOD_INFO["feasibility"]["parameters"][_name]["applies_when"] = {
        "time_constraint_months": "set"}
# Taken and not used before (GM-184), refused since GM-179: the water
# table. The depth to the treatable layer is used since GM-179. The column
# modulus is refused by run_aggregate_piers with the stress-ratio method
# only: GEC-13 Fig. 5-21's methods take it (GM-220).
METHOD_INFO["feasibility"]["refused_parameters"] = {"gwt_depth_m": _NO_GWT}

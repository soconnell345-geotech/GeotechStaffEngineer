"""Drilled shaft adapter — GEC-10 alpha/beta/rock socket capacity + LRFD."""

from funhouse_agent.adapters import (
    apply_aliases, check_nested_list, declare_aliases,
    declare_applies_when, field_keys,
    reject_unknown_params, require_keys, require_params,
)
from drilled_shaft import DrillShaft, ShaftSoilLayer, ShaftSoilProfile, DrillShaftAnalysis
from drilled_shaft.lrfd import apply_lrfd, RESISTANCE_FACTORS

# Shaft + soil params shared by every method below.
_SHAFT_PARAMS = ("diameter", "shaft_length", "socket_diameter",
                 "socket_length", "bell_diameter", "casing_depth")
_SOIL_PARAMS = ("layers", "gwt_depth")
# GEC-10 rational side-resistance method selectors (opt-in; defaults preserve
# the AASHTO / depth-based behavior).
_RATIONAL_PARAMS = ("beta_method", "alpha_method", "su_test_type", "pa")
# 'length' is a common guess for the shaft length.
_SHAFT_ALIASES = {"length": "shaft_length", "pile_length": "shaft_length"}

# Keys a layer dict may carry: ShaftSoilLayer's fields plus the aliases read
# for the rational-beta inputs. Anything else was read with .get and silently
# left out, e.g. 'friction_angle' ran the layer with phi = 0 (GM-58).
_LAYER_KEYS = field_keys(ShaftSoilLayer, "N160", "sigma_v_preload")
# Words the near-miss matcher cannot place. Suggestions only, never renamed.
_LAYER_KEY_HINTS = {
    "frictionangle": ("phi",), "c": ("cu",), "cohesion": ("cu",),
    "su": ("cu",), "gamma": ("unit_weight",),
}


def _build_shaft(params):
    require_params(params, ["diameter", "shaft_length"],
                   method="drilled_shaft",
                   valid=_SHAFT_PARAMS + _SOIL_PARAMS)
    return DrillShaft(diameter=params["diameter"], length=params["shaft_length"],
                       socket_diameter=params.get("socket_diameter"),
                       socket_length=params.get("socket_length", 0.0),
                       bell_diameter=params.get("bell_diameter"),
                       casing_depth=params.get("casing_depth", 0.0))


def _build_soil_profile(params):
    require_params(params, ["layers"], method="drilled_shaft")
    check_nested_list(params["layers"], _LAYER_KEYS, method="drilled_shaft",
                      label="layers", hints=_LAYER_KEY_HINTS)
    layers = []
    for l in params["layers"]:
        require_keys(l, ["thickness", "soil_type", "unit_weight"],
                     method="drilled_shaft", item_label="layers[]")
        # Optional rational-beta inputs (used only when beta_method="rational").
        n1_60 = l.get("N1_60", l.get("N160", 0.0))
        sigma_v_ref = l.get("sigma_v_ref", l.get("sigma_v_preload", 0.0))
        layers.append(ShaftSoilLayer(
            thickness=l["thickness"], soil_type=l["soil_type"], unit_weight=l["unit_weight"],
            cu=l.get("cu", 0.0), phi=l.get("phi", 0.0), N60=l.get("N60", 0.0),
            qu=l.get("qu", 0.0), RQD=l.get("RQD", 100.0),
            N1_60=n1_60 or 0.0, sigma_v_ref=sigma_v_ref or 0.0, OCR=l.get("OCR", 0.0),
            description=l.get("description", ""),
        ))
    return ShaftSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth"))


def _rational_kwargs(params):
    """Collect the opt-in GEC-10 rational method selectors for the analysis."""
    kw = {}
    if "beta_method" in params:
        kw["beta_method"] = params["beta_method"]
    if "alpha_method" in params:
        kw["alpha_method"] = params["alpha_method"]
    if "su_test_type" in params:
        kw["su_test_type"] = params["su_test_type"]
    if "pa" in params:
        kw["pa"] = params["pa"]
    return kw


def run_drilled_shaft_capacity(params, *, method="drilled_shaft_capacity"):
    """(result, analysis) of ``drilled_shaft_capacity``: the tool's own code
    path, also run by ``calc_package.drilled_shaft_package`` (GM-141).
    ``method`` names the caller in error messages."""
    params = apply_aliases(params, _SHAFT_ALIASES)
    reject_unknown_params(params, _SHAFT_PARAMS + _SOIL_PARAMS + _RATIONAL_PARAMS + ("factor_of_safety",),
                          method=method)
    shaft = _build_shaft(params)
    soil = _build_soil_profile(params)
    analysis = DrillShaftAnalysis(shaft=shaft, soil=soil,
                                  factor_of_safety=params.get("factor_of_safety", 2.5),
                                  **_rational_kwargs(params))
    return analysis.compute(), analysis


def _run_drilled_shaft_capacity(params):
    return run_drilled_shaft_capacity(params)[0].to_dict()


def _run_capacity_vs_depth(params):
    params = apply_aliases(params, _SHAFT_ALIASES)
    reject_unknown_params(
        params,
        _SHAFT_PARAMS + _SOIL_PARAMS + _RATIONAL_PARAMS + ("factor_of_safety", "depth_min",
                                        "depth_max", "n_points"),
        method="capacity_vs_depth")
    shaft = _build_shaft(params)
    soil = _build_soil_profile(params)
    analysis = DrillShaftAnalysis(shaft=shaft, soil=soil,
                                  factor_of_safety=params.get("factor_of_safety", 2.5),
                                  **_rational_kwargs(params))
    curve = analysis.capacity_vs_depth(depth_min=params.get("depth_min", 3.0),
                                        depth_max=params.get("depth_max"), n_points=params.get("n_points", 20))
    return {"capacity_vs_depth": curve}


def _run_lrfd_capacity(params):
    params = apply_aliases(params, _SHAFT_ALIASES)
    reject_unknown_params(params, _SHAFT_PARAMS + _SOIL_PARAMS + _RATIONAL_PARAMS + ("tip_soil_type",),
                          method="lrfd_capacity")
    shaft = _build_shaft(params)
    soil = _build_soil_profile(params)
    analysis = DrillShaftAnalysis(shaft=shaft, soil=soil, factor_of_safety=1.0,
                                  **_rational_kwargs(params))
    result = analysis.compute()
    tip_soil_type = params.get("tip_soil_type", "cohesive")
    if tip_soil_type not in ("cohesive", "cohesionless", "rock"):
        raise ValueError(
            f"lrfd_capacity: unknown tip_soil_type '{tip_soil_type}'. "
            "Allowed: ['cohesive', 'cohesionless', 'rock'].")
    lrfd = apply_lrfd(result, tip_soil_type)
    output = result.to_dict()
    output["lrfd"] = lrfd
    return output


def _run_get_resistance_factors(params):
    return {"resistance_factors": RESISTANCE_FACTORS}


METHOD_REGISTRY = {
    "drilled_shaft_capacity": _run_drilled_shaft_capacity,
    "capacity_vs_depth": _run_capacity_vs_depth,
    "lrfd_capacity": _run_lrfd_capacity,
    "get_resistance_factors": _run_get_resistance_factors,
}

# Shared METHOD_INFO parameter blocks — every method that accepts these params
# documents them identically (with allowed_values), so capacity_vs_depth and
# lrfd_capacity stay in lockstep with drilled_shaft_capacity instead of drifting.
_INFO_DIAMETER = {"type": "float", "required": True, "description": "Shaft diameter (m)."}
_INFO_LAYERS = {"type": "array", "required": True, "description": "Array of {thickness, soil_type, unit_weight, cu, phi, N60, qu, RQD} dicts. soil_type: cohesive/cohesionless/rock. A cohesionless layer at the shaft tip needs its N60 (end bearing q_b = 57.5*N60 kPa, N60 capped at 50); without it the call is refused. For beta_method='rational' each cohesionless layer also takes N1_60 (overburden-corrected SPT, seeds phi'=27.5+9.2*log10[(N1)60]) and optionally sigma_v_ref (kPa, pre-scour effective stress for OCR; default = current sigma'v) or a direct OCR."}
_INFO_GWT = {"type": "float", "required": False, "description": "Groundwater depth (m)."}
_INFO_FOS = {"type": "float", "required": False, "default": 2.5, "description": "Factor of safety."}
# GEC-10 rational side-resistance selectors (opt-in; defaults preserve AASHTO/depth).
_INFO_RATIONAL = {
    "beta_method": {"type": "str", "required": False, "default": "depth", "allowed_values": ["depth", "rational"], "description": "Cohesionless side resistance. 'depth' = O'Neill & Reese 1.5-0.245*sqrt(z_ft) (default). 'rational' = GEC-10 Appendix A OCR/Ko chain: beta=Ko*tan(phi'), Ko=(1-sin phi')*OCR^(sin phi'), OCR=sigma'p/sigma_v_ref, sigma'p=0.47*pa*N60^0.6 (needs per-layer N60/N1_60)."},
    "alpha_method": {"type": "str", "required": False, "default": "aashto", "allowed_values": ["aashto", "rational"], "description": "Cohesive side resistance. 'aashto' = piecewise 0.55 (default). 'rational' = GEC-10 Chen-2011 alpha=0.30+0.17/(su_CIUC/pa), applied to the CIUC-equivalent strength (fs=alpha*su_CIUC)."},
    "su_test_type": {"type": "str", "required": False, "default": "ciuc", "allowed_values": ["ciuc", "uc", "uu"], "description": "For alpha_method='rational': the lab test the layer cu represents, for the UU/UC->CIUC transform. 'ciuc' = no transform (default); 'uc'/'uu' = Chen & Kulhawy (1993) transform."},
    "pa": {"type": "float", "required": False, "default": 101.325, "description": "Atmospheric pressure (kPa) for the rational chains."},
}
_INFO_SHAFT_GEOM = {
    "socket_diameter": {"type": "float", "required": False, "description": "Rock socket diameter (m). Defaults to shaft diameter."},
    "socket_length": {"type": "float", "required": False, "default": 0.0, "description": "Rock socket length (m)."},
    "bell_diameter": {"type": "float", "required": False, "description": "Belled base diameter (m)."},
    "casing_depth": {"type": "float", "required": False, "default": 0.0, "description": "Permanent casing depth (m, no side resistance)."},
}

METHOD_INFO = {
    "drilled_shaft_capacity": {
        "category": "Drilled Shaft",
        "brief": "Full drilled shaft capacity (GEC-10 alpha/beta/rock socket).",
        "parameters": {
            "diameter": _INFO_DIAMETER,
            "shaft_length": {"type": "float", "required": True, "description": "Shaft length (m). Alias: length."},
            "layers": _INFO_LAYERS,
            "gwt_depth": _INFO_GWT,
            "factor_of_safety": _INFO_FOS,
            **_INFO_RATIONAL,
            **_INFO_SHAFT_GEOM,
        },
        "returns": {"Q_ultimate_kN": "Ultimate capacity.", "Q_skin_kN": "Side resistance.", "Q_tip_kN": "Tip resistance."},
    },
    "capacity_vs_depth": {
        "category": "Drilled Shaft",
        "brief": "Capacity vs depth curve for length optimization.",
        "parameters": {
            "diameter": _INFO_DIAMETER,
            "shaft_length": {"type": "float", "required": True, "description": "Maximum shaft length (m); upper bound of the depth sweep. Alias: length."},
            "layers": _INFO_LAYERS,
            "gwt_depth": _INFO_GWT,
            "factor_of_safety": _INFO_FOS,
            **_INFO_RATIONAL,
            **_INFO_SHAFT_GEOM,
            "depth_min": {"type": "float", "required": False, "default": 3.0, "description": "Minimum trial shaft length (m) in the sweep."},
            "depth_max": {"type": "float", "required": False, "description": "Maximum trial shaft length (m). Default: min(shaft_length, soil profile depth)."},
            "n_points": {"type": "int", "required": False, "default": 20, "description": "Number of trial lengths in the sweep."},
        },
        "returns": {"capacity_vs_depth": "List of {depth_m, Q_ultimate_kN, Q_skin_kN, Q_tip_kN, Q_allowable_kN} dicts (Q_allowable = Q_ultimate / factor_of_safety)."},
    },
    "lrfd_capacity": {
        "category": "Drilled Shaft",
        "brief": "Drilled shaft capacity with AASHTO LRFD resistance factors.",
        "parameters": {
            "diameter": _INFO_DIAMETER,
            "shaft_length": {"type": "float", "required": True, "description": "Shaft length (m). Alias: length."},
            "layers": _INFO_LAYERS,
            "gwt_depth": _INFO_GWT,
            "tip_soil_type": {"type": "str", "required": False, "default": "cohesive", "allowed_values": ["cohesive", "cohesionless", "rock"], "description": "Soil type at shaft tip."},
            **_INFO_RATIONAL,
            **_INFO_SHAFT_GEOM,
        },
        "returns": {"Q_ultimate_kN": "Unfactored ultimate capacity.", "lrfd": "Factored resistances by component."},
    },
    "get_resistance_factors": {
        "category": "Drilled Shaft",
        "brief": "AASHTO LRFD resistance factors for drilled shaft design.",
        "parameters": {},
        "returns": {"resistance_factors": "Dict of component → phi factor."},
    },
}

# The concrete strength was declared (default 28 MPa) and changed nothing:
# GEC-10's geotechnical resistances do not use it, and no structural check
# of the shaft is made here (GM-160).
_NO_FC = ("the geotechnical capacity (GEC-10 side and base resistance) does "
          "not use the concrete strength, and these tools make no "
          "structural check of the shaft")
for _m in ("drilled_shaft_capacity", "capacity_vs_depth", "lrfd_capacity"):
    METHOD_INFO[_m]["refused_parameters"] = {"concrete_fc": _NO_FC}

declare_aliases(METHOD_INFO, {
    m: _SHAFT_ALIASES for m in ("drilled_shaft_capacity", "capacity_vs_depth",
                                "lrfd_capacity")})

# When each default applies (GM-119): the CIUC transform and the
# atmospheric pressure go with GEC-10's rational chains only.
declare_applies_when(METHOD_INFO, {
    m: {"su_test_type": {"alpha_method": ["rational"]},
        "pa": [{"alpha_method": ["rational"]},
               {"beta_method": ["rational"]}]}
    for m in ("drilled_shaft_capacity", "capacity_vs_depth",
              "lrfd_capacity")})

"""Drilled shaft adapter — GEC-10 alpha/beta/rock socket capacity + LRFD."""

from funhouse_agent.adapters import (
    apply_aliases, check_nested_list, declare_aliases,
    declare_applies_when, field_keys,
    reject_unknown_params, require_keys, require_params,
)
from drilled_shaft import DrillShaft, ShaftSoilLayer, ShaftSoilProfile, DrillShaftAnalysis
from drilled_shaft.lrfd import apply_lrfd, RESISTANCE_FACTORS
from drilled_shaft.capacity import skipped_depths_note

# Shaft + soil params shared by every method below.
_SHAFT_PARAMS = ("diameter", "shaft_length", "socket_diameter",
                 "bell_diameter", "casing_depth", "fractured_rock_thickness")
_SOIL_PARAMS = ("layers", "gwt_depth")
# Side-resistance method selectors and settings: the beta in sand (GEC-10
# 2018 Eq. 10-13 by default since GM-241) and its K/Ko, the alpha in clay,
# the rock side resistance's C, and the top zone with no side resistance.
_RATIONAL_PARAMS = ("beta_method", "alpha_method", "su_test_type", "pa",
                    "rock_side_C", "K_over_Ko", "top_exclusion")
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
                       bell_diameter=params.get("bell_diameter"),
                       casing_depth=params.get("casing_depth", 0.0),
                       fractured_rock_thickness=params.get(
                           "fractured_rock_thickness", 0.0))


def _build_soil_profile(params):
    require_params(params, ["layers"], method="drilled_shaft")
    check_nested_list(params["layers"], _LAYER_KEYS, method="drilled_shaft",
                      label="layers", hints=_LAYER_KEY_HINTS)
    layers = []
    for l in params["layers"]:
        require_keys(l, ["thickness", "soil_type", "unit_weight"],
                     method="drilled_shaft", item_label="layers[]")
        # Inputs of the GEC-10 beta (beta_method 'rational', the default).
        n1_60 = l.get("N1_60", l.get("N160", 0.0))
        sigma_v_ref = l.get("sigma_v_ref", l.get("sigma_v_preload", 0.0))
        layers.append(ShaftSoilLayer(
            thickness=l["thickness"], soil_type=l["soil_type"], unit_weight=l["unit_weight"],
            cu=l.get("cu", 0.0), phi=l.get("phi", 0.0), N60=l.get("N60", 0.0),
            qu=l.get("qu", 0.0), RQD=l.get("RQD", 100.0),
            N1_60=n1_60 or 0.0, sigma_v_ref=sigma_v_ref or 0.0, OCR=l.get("OCR", 0.0),
            sigma_p_correlation=l.get("sigma_p_correlation", "clean_sand"),
            description=l.get("description", ""),
        ))
    return ShaftSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth"))


def _rational_kwargs(params):
    """The side-resistance selectors and settings the call gave, for the
    analysis: GEC-10 Eq. 10-21's C for the rock side resistance (GM-221);
    K/Ko of the GEC-10 beta and the top zone (GM-241)."""
    return {k: params[k] for k in _RATIONAL_PARAMS if k in params}


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
    skipped = []
    n_points = params.get("n_points", 20)
    curve = analysis.capacity_vs_depth(depth_min=params.get("depth_min", 3.0),
                                        depth_max=params.get("depth_max"), n_points=n_points,
                                        skipped=skipped)
    out = {"capacity_vs_depth": curve}
    if skipped:
        # GM-203: a refused trial length was dropped without a word
        out["skipped_depths"] = skipped
        out["warning"] = skipped_depths_note(skipped, n_points)
    return out


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
_INFO_LAYERS = {"type": "array", "required": True, "description": "Array of {thickness, soil_type, unit_weight, cu, phi, N60, qu, RQD} dicts. soil_type: cohesive/cohesionless/rock. A cohesionless layer at the shaft tip needs its N60 (end bearing q_b = 57.5*N60 kPa, N60 capped at 50); without it the call is refused. Under beta_method 'rational' (the default, GEC-10 2018 Eq. 10-13) each cohesionless layer the shaft passes through needs its N60 (sigma'p by Eq. 10-11 or 10-12) or a direct OCR, else the call is refused; it also takes N1_60 (overburden-corrected SPT; phi' = 27.5+9.2*log10[(N1)60], GEC-10 Eq. A-2, in place of phi), sigma_p_correlation ('clean_sand' default: Eq. 10-11 with m = 0.6, clean quartzitic sands; 'silty_sand': m = 0.8, silty sands to sandy silts; 'gravel': Eq. 10-12, sigma'p/pa = 0.15*N60, gravelly soils; GEC-10 p. 10-11) and sigma_v_ref (kPa, pre-scour effective stress for the OCR; default the layer's own sigma'v)."}
_INFO_GWT = {"type": "float", "required": False, "description": "Groundwater depth (m)."}
_INFO_FOS = {"type": "float", "required": False, "default": 2.5, "description": "Factor of safety."}
# Side-resistance selectors and settings (GEC-10 2018 beta by default since GM-241).
_INFO_RATIONAL = {
    "beta_method": {"type": "str", "required": False, "default": "rational", "allowed_values": ["rational", "depth"], "description": "Cohesionless side resistance. 'rational' (default) = GEC-10 (2018) Sec. 10.3.5.1, Eqs. 10-5 to 10-13 (pp. 10-10 to 10-12): f_SN = beta*sigma'v on each layer's average sigma'v, beta = (K/Ko)*Ko*tan(phi'), Ko = (1-sin phi')*OCR^(sin phi') <= Kp, OCR = sigma'p/sigma'v, sigma'p from the layer's N60 (Eq. 10-11 or 10-12), beta limited to its value at 7.5 ft, no limit on f_SN; needs each cohesionless layer's N60 or OCR. 'depth' = O'Neill & Reese (1999) beta = 1.5-0.135*sqrt(z, ft) = 1.5-0.245*sqrt(z, m), 0.25-1.2, x N60/15 where N60 < 15, f_s <= 200 kPa (FHWA NHI-06-089 Eqs. 9-41 to 9-43), which GEC-10 (2018) does not print; the default until GM-241."},
    "K_over_Ko": {"type": "float", "required": False, "default": 1.0, "description": "K/Ko in the GEC-10 beta (beta_method 'rational'): 1 = 'a rational first-order approximation', no stress change from construction (GEC-10 p. 10-11); 2/3 or lower 'when these aspects of construction quality are not controlled properly' or soil caving is allowed (p. 10-12). 0 < K_over_Ko <= 1."},
    "top_exclusion": {"type": "float", "required": False, "default": 1.5, "description": "Side resistance left out over the top of the shaft (m), in every layer. Default 1.5 = GEC-10's 'top 5 ft' (p. 10-17), 'or to the depth of seasonal moisture change', 'subject to modification based on local experience and judgment of the designer'; it is printed for cohesive soils, and Example 10-1 (sand) takes side resistance from the ground surface (0). casing_depth, where deeper, takes its place."},
    "alpha_method": {"type": "str", "required": False, "default": "aashto", "allowed_values": ["aashto", "rational"], "description": "Cohesive side resistance. 'aashto' = piecewise 0.55 (default). 'rational' = GEC-10 Chen-2011 alpha=0.30+0.17/(su_CIUC/pa), applied to the CIUC-equivalent strength (fs=alpha*su_CIUC)."},
    "su_test_type": {"type": "str", "required": False, "default": "ciuc", "allowed_values": ["ciuc", "uc", "uu"], "description": "For alpha_method='rational': the lab test the layer cu represents, for the UU/UC->CIUC transform. 'ciuc' = no transform (default); 'uc'/'uu' = Chen & Kulhawy (1993) transform."},
    "pa": {"type": "float", "required": False, "default": 101.325, "description": "Atmospheric pressure (kPa) for the rational chains."},
    "rock_side_C": {"type": "float", "required": False, "default": 0.65, "description": "C in the rock side resistance f_s = C*pa*sqrt(qu/pa) (GEC-10 Eq. 10-21, p. 10-20). Default 0.65 = Eq. 10-22 (O'Neill and Reese 1999, for sockets where caving is anticipated) with alpha_E = 1.0. GEC-10 recommends C = 1.0 for routine design of a 'normal' socket (p. 10-21; Example 10-4); 0.65*alpha_E with alpha_E from Table 10-3 where caving is anticipated; 1.9 for roughened sockets, verified by load tests or local experience (p. 10-22). Needs a rock layer."},
}
_INFO_SHAFT_GEOM = {
    "socket_diameter": {"type": "float", "required": False, "description": "Rock socket diameter (m). Defaults to shaft diameter. One diameter for every rock layer: GEC-10 sizes a socket by one diameter B (Eq. 10-21, Example 10-4) and gives none per rock layer. The rock side resistance and, with the tip in rock, the base (pi/4*B^2, GEC-10 Example 10-4, p. 10-28) take it."},
    "bell_diameter": {"type": "float", "required": False, "description": "Belled base diameter (m)."},
    "casing_depth": {"type": "float", "required": False, "default": 0.0, "description": "Permanent casing depth (m, no side resistance)."},
    "fractured_rock_thickness": {"type": "float", "required": False, "default": 0.0, "description": "Decomposed or fractured rock (m) below the top of the first rock layer that is 'not considered part of the socket for design purposes' (GEC-10 p. 8-15): no rock side resistance over it, while the soil above keeps its own (casing_depth takes both). 0 = all rock is socket. Needs a rock layer."},
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
        "returns": {"Q_ultimate_kN": "Ultimate capacity.", "Q_skin_kN": "Side resistance.", "Q_tip_kN": "Tip resistance.", "base_diameter_m": "Diameter of the base the tip resistance acts on: the bell's, the rock socket's with the tip in rock, else the shaft's."},
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
        "returns": {"capacity_vs_depth": "List of {depth_m, Q_ultimate_kN, Q_skin_kN, Q_tip_kN, Q_allowable_kN} dicts; Q_allowable_kN = Q_ultimate_kN / factor_of_safety.",
                    "skipped_depths": "Only when the analysis refused some trial lengths: {depth_m, reason} for each, left out of the curve.",
                    "warning": "Only with skipped_depths: one sentence naming them. Tell the user."},
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
# The socket length was taken and printed and changed nothing (GM-184);
# refused since GM-179: the socket is the shaft within the rock layers.
_NO_SOCKET_LENGTH = (
    "the capacity does not take a socket length: rock side resistance "
    "(GEC-10) is taken over the shaft's length within the rock layers, so "
    "the socket is what layers and shaft_length make it. Give the rock as "
    "layers (soil_type 'rock' with qu) and the shaft_length that ends the "
    "shaft that far into it. A permanent casing into the rock is "
    "casing_depth: no side resistance, soil or rock, above it")
for _m in ("drilled_shaft_capacity", "capacity_vs_depth", "lrfd_capacity"):
    METHOD_INFO[_m]["refused_parameters"] = {
        "concrete_fc": _NO_FC, "socket_length": _NO_SOCKET_LENGTH}

declare_aliases(METHOD_INFO, {
    m: _SHAFT_ALIASES for m in ("drilled_shaft_capacity", "capacity_vs_depth",
                                "lrfd_capacity")})

# When each default applies (GM-119): the CIUC transform and the
# atmospheric pressure go with GEC-10's rational chains only, and K/Ko
# with its beta (GM-241).
declare_applies_when(METHOD_INFO, {
    m: {"su_test_type": {"alpha_method": ["rational"]},
        "pa": [{"alpha_method": ["rational"]},
               {"beta_method": ["rational"]}],
        "K_over_Ko": {"beta_method": ["rational"]}}
    for m in ("drilled_shaft_capacity", "capacity_vs_depth",
              "lrfd_capacity")})


# What a left-out input with no default stands for (BACKLOG GM-165):
# the water table and the rock socket's diameter.
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
_NO_WATER = ("no water table",
             "total unit weight throughout: effective stress equals total "
             "stress")
declare_when_absent(METHOD_INFO, {
    "drilled_shaft_capacity": {"gwt_depth": _NO_WATER,
                               "socket_diameter": "diameter"},
    "capacity_vs_depth": {
        "gwt_depth": _NO_WATER, "socket_diameter": "diameter",
        "depth_max": ("the shaft length or the soil profile depth, "
                      "whichever is less", "the deepest trial shaft")},
    "lrfd_capacity": {"gwt_depth": _NO_WATER, "socket_diameter": "diameter"},
})

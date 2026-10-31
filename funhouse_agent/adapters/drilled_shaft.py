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
                 "bell_diameter", "casing_depth", "fractured_rock_thickness",
                 "concrete_fc")
_SOIL_PARAMS = ("layers", "gwt_depth")
# Method selectors and settings: the beta in sand (GEC-10 2018 Eq. 10-13
# by default since GM-241) and its K/Ko, the alpha in clay (Eq. 10-18 by
# default since GM-244), the rock side resistance's C, the top and bottom
# zones with no side resistance, and the base: GEC-10 (2018) as printed by
# default, the older methods as named options (GM-244); Eq. 10-22 with
# Table 10-3 for caving rock (GM-284).
_RATIONAL_PARAMS = ("beta_method", "alpha_method", "su_test_type", "pa",
                    "rock_side_C", "K_over_Ko", "top_exclusion",
                    "bottom_exclusion", "clay_base_method",
                    "rock_base_method", "rock_base_conditions_met",
                    "large_base_reduction", "rock_side_method",
                    "rock_joints")
# 'length' is a common guess for the shaft length.
_SHAFT_ALIASES = {"length": "shaft_length", "pile_length": "shaft_length"}

# Keys a layer dict may carry: ShaftSoilLayer's fields plus the aliases read
# for the rational-beta inputs. Anything else was read with .get and silently
# left out, e.g. 'friction_angle' ran the layer with phi = 0 (GM-58).
_LAYER_KEYS = field_keys(ShaftSoilLayer, "N160", "sigma_v_preload")
# Words the near-miss matcher cannot place. Suggestions only, never renamed.
_LAYER_KEY_HINTS = {
    "frictionangle": ("phi",), "c": ("cu",), "cohesion": ("cu",),
    "su": ("cu",), "gamma": ("unit_weight",), "mi": ("m_i",),
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
                           "fractured_rock_thickness", 0.0),
                       concrete_fc=params.get("concrete_fc"))


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
            GSI=l.get("GSI"), m_i=l.get("m_i"),
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
    tip_soil_type = params.get("tip_soil_type")
    if tip_soil_type not in (None, "cohesive", "cohesionless", "rock"):
        raise ValueError(
            f"lrfd_capacity: unknown tip_soil_type '{tip_soil_type}'. "
            "Allowed: ['cohesive', 'cohesionless', 'rock'].")
    lrfd = apply_lrfd(result, tip_soil_type)
    output = result.to_dict()
    output["lrfd"] = lrfd
    extra = []
    if (result.base_steps or {}).get("method") == "rqd":
        extra.append(_rqd_tip_factor_choice(lrfd))
    elif (result.base_steps or {}).get("equation") == "10-29":
        extra.append(_hoek_brown_tip_factor_choice(lrfd))
    if extra:
        output["choices"] = list(output.get("choices") or []) + extra
    return output


def _rqd_tip_factor_choice(lrfd) -> dict:
    """The resistance factor taken for the base in rock under
    rock_base_method 'rqd' (GM-244 review): Table 8-4's 0.50 is for Eq.
    10-23 or Eq. 10-29, not for that reduction."""
    from geotech_common.choices import MODULE, choice
    return choice(
        "resistance factor for base resistance in rock", lrfd["phi_tip"],
        MODULE,
        basis=("GEC-10 (FHWA-NHI-18-024) Table 8-4 (p. 8-14) prints 0.50 "
               "for base resistance in rock by Eq. 10-23 or Eq. 10-29. "
               "rock_base_method 'rqd' (q_b/q_u reduced by RQD) is neither, "
               "and GEC-10 prints no factor calibrated for it: 0.50 is "
               "taken, the engineer's judgment"),
        alternatives=["rock_base_method 'gec10', Eq. 10-23 or the "
                      "Hoek-Brown base, Eq. 10-29, the methods Table 8-4's "
                      "0.50 is printed for"])


def _hoek_brown_tip_factor_choice(lrfd) -> dict:
    """The resistance factor taken for the Hoek-Brown base of a fractured
    rock mass, GEC-10 Eq. 10-29 (GM-251): the text and Table 8-4 agree."""
    from geotech_common.choices import MODULE, choice
    return choice(
        "resistance factor for base resistance in rock", lrfd["phi_tip"],
        MODULE,
        basis=("GEC-10 (FHWA-NHI-18-024) p. 10-24: 'The recommend "
               "resistance factor for application to Equation 10-29 is phi "
               "= 0.50, based on fitting to an ASD factor of safety FS = "
               "2.5'; Table 8-4 (p. 8-14) prints 0.50 for base resistance in "
               "rock by 'Eq. 10-23 or Eq. 10-29'. The two agree"))


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
_INFO_LAYERS = {"type": "array", "required": True, "description": "Array of {thickness, soil_type, unit_weight, cu, phi, N60, qu, RQD} dicts. soil_type: cohesive/cohesionless/rock. A cohesionless layer at the shaft tip needs its N60 (end bearing q_b = 57.5*N60 kPa, GEC-10 Eq. 10-14, N60 the mean of the cohesionless layers' N60 between the base and two base diameters below it, by thickness, limited to 50, p. 10-15; a layer there with no N60, or of cohesive soil or rock, is left out); without it the call is refused. Under beta_method 'rational' (the default, GEC-10 2018 Eq. 10-13) each cohesionless layer the shaft passes through needs its N60 (sigma'p by Eq. 10-11 or 10-12) or a direct OCR, else the call is refused; it also takes N1_60 (overburden-corrected SPT; phi' = 27.5+9.2*log10[(N1)60], GEC-10 Example 10-1 and Eq. A-2, where the layer gives no phi; a phi given is used and the result's choices give the correlation's value beside it), sigma_p_correlation ('clean_sand' default: Eq. 10-11 with m = 0.6, clean quartzitic sands; 'silty_sand': m = 0.8, silty sands to sandy silts; 'gravel': Eq. 10-12, sigma'p/pa = 0.15*N60, gravelly soils; GEC-10 p. 10-11) and sigma_v_ref (kPa, pre-scour effective stress for the OCR; default the layer's own sigma'v). A rock layer's RQD (default 100) sets alpha_E of the side resistance under rock_side_method 'caving' (GEC-10 Table 10-3) and the base under rock_base_method 'rqd'; under the default 'gec10' an RQD below 100 within one base diameter below the base needs rock_base_conditions_met, or the tip rock layer's GSI and m_i. A rock layer also takes GSI (Geological Strength Index of the rock mass, 0 to 100, from the core descriptions; GEC-10 p. 10-23 names GSI, not RMR) and m_i (Hoek-Brown constant of the intact rock, by rock type, GEC 5 Table 9-10, as GEC-10 p. 10-24 points to): with the base in that layer they give GEC-10's base for a fractured rock mass, the Hoek-Brown base, Eqs. 10-24 to 10-29, taken where the rock is not taken to meet the conditions of Eq. 10-23 and shown beside 2.5*qu where it is (see rock_base_conditions_met); a rock layer within one base diameter below the base that gives them has its own Hoek-Brown base among the result's alternatives. GSI or m_i on any other layer (above the base, further below it, or with the base in soil) is refused: nothing would use them; capacity_vs_depth takes them on any layer a trial base is in."}
_INFO_GWT = {"type": "float", "required": False, "description": "Groundwater depth (m)."}
_INFO_FOS = {"type": "float", "required": False, "default": 2.5, "description": "Factor of safety."}
# Side-resistance selectors and settings (GEC-10 2018 beta by default since GM-241).
_INFO_RATIONAL = {
    "beta_method": {"type": "str", "required": False, "default": "rational", "allowed_values": ["rational", "depth"], "description": "Cohesionless side resistance. 'rational' (default) = GEC-10 (2018) Sec. 10.3.5.1, Eqs. 10-5 to 10-13 (pp. 10-10 to 10-12): f_SN = beta*sigma'v on each layer's average sigma'v, beta = (K/Ko)*Ko*tan(phi'), Ko = (1-sin phi')*OCR^(sin phi') <= Kp, OCR = sigma'p/sigma'v, sigma'p from the layer's N60 (Eq. 10-11 or 10-12), beta limited to its value at 7.5 ft, no limit on f_SN; needs each cohesionless layer's N60 or OCR. 'depth' = O'Neill & Reese (1999) beta = 1.5-0.135*sqrt(z, ft) = 1.5-0.245*sqrt(z, m), 0.25-1.2, x N60/15 where N60 < 15, f_s <= 4 ksf = 191.5 kPa (FHWA NHI-06-089 Eqs. 9-41 to 9-43, the limit p. 9-133), which GEC-10 (2018) does not print."},
    "K_over_Ko": {"type": "float", "required": False, "default": 1.0, "description": "K/Ko in the GEC-10 beta (beta_method 'rational'): 1 = 'a rational first-order approximation', no stress change from construction (GEC-10 p. 10-11); 2/3 or lower 'when these aspects of construction quality are not controlled properly' or soil caving is allowed (p. 10-12). 0 < K_over_Ko <= 1."},
    "top_exclusion": {"type": "float", "required": False, "description": "Side resistance left out over the top of the shaft (m), in every layer, when given. Left out: none over the top 1.5 m in cohesive soil, GEC-10's 'top 5 ft' (p. 10-17, printed in Sec. 10.3.5.2 for cohesive soils, 'or to the depth of seasonal moisture change', 'subject to modification based on local experience and judgment of the designer'), and side resistance from the top in cohesionless soil and rock (Secs. 10.3.5.1 and 10.3.5.3 print no such zone; Examples 10-1 and 10-5 take the sand's from the surface, Example 10-4 the socket's over its whole length). Give 1.5 to leave the top 5 ft out of sand and rock too (frost, seasonal moisture, gapping from cyclic lateral load), or the depth of seasonal moisture change. casing_depth, where deeper, takes its place."},
    "bottom_exclusion": {"type": "float", "required": False, "default": 0.0, "description": "Side resistance left out over the bottom of the shaft (m) in cohesive soil. Default 0: GEC-10 (2018) prints no such zone, and Example 10-2 (p. 10-18) takes the clay's side resistance to the base. FHWA NHI-06-089 p. 9-129 (O'Neill & Reese 1999) leaves out the bottom 5 ft (1.524 m), its Fig. 9-49 the bottom one diameter (give the diameter for that)."},
    "alpha_method": {"type": "str", "required": False, "default": "rational", "allowed_values": ["rational", "aashto"], "description": "Cohesive side resistance. 'rational' (default) = GEC-10 (2018) Sec. 10.3.5.2, Eq. 10-18 (p. 10-16): alpha = 0.30+0.17/(su(CIUC)/pa), f_SN = alpha*su(CIUC), su(CIUC) from cu by su_test_type. 'aashto' = O'Neill & Reese (1999) as FHWA NHI-06-089 Eqs. 9-37a/b print it (p. 9-129): alpha = 0.55 for cu/pa <= 1.5, 0.55-0.1*(cu/pa-1.5) to 2.5, on cu; GEC-10 recommends Eq. 10-18 'as a revision' to it."},
    "su_test_type": {"type": "str", "required": False, "default": "ciuc", "allowed_values": ["ciuc", "uc", "uu"], "description": "For alpha_method 'rational': the lab test each layer's cu came from. 'ciuc' (default) = cu taken as su(CIUC), no transform; 'uc' / 'uu' = transformed to su(CIUC) on the layer's sigma'vo by GEC-10 Eq. 10-16 / 10-17 (Chen & Kulhawy 1993, p. 10-16; Example 10-2 takes a UU strength). The base in clay takes the same su(CIUC) (Example 10-3)."},
    "clay_base_method": {"type": "str", "required": False, "default": "gec10", "allowed_values": ["gec10", "oneill_reese"], "description": "Base resistance with the tip in cohesive soil. 'gec10' (default) = GEC-10 (2018) p. 10-19: q_BN = N*c*su (Eq. 10-19), su the mean of the cohesive soil over 2B below the base, by thickness (p. 10-19; the tip layer at its su(CIUC) as its side resistance took it, each clay layer below on its own cu, transformed alike; cohesionless or rock layers within 2B left out; where the profile ends short of 2B, the mean of what it gives, said in the choices), N*c from Table 10-2 by su (6.5 at 500 psf, 8.0 at 1,000 psf, 9.0 at 2,000 psf or more; held at 6.5 below 500 psf), (2/3)[1 + D/(6B)]*N*c*su where the depth D < 3B (Eq. 10-20), no upper limit. 'oneill_reese' = FHWA NHI-06-089 Eqs. 9-39, 9-40 (p. 9-131): Nc = 6[1 + 0.2 z/D] <= 9, x 0.67 where su < 0.5 ksf, q_t <= 80 ksf, on the tip layer's cu."},
    "rock_base_method": {"type": "str", "required": False, "default": "gec10", "allowed_values": ["gec10", "rqd"], "description": "Base resistance with the tip in rock. 'gec10' (default) = GEC-10 (2018) Eq. 10-23 (p. 10-22): q_BN = 2.5*qu, for rock massive or tightly jointed to one diameter below the base, no cavities beneath, a clean base (Example 10-4 takes it with RQD 37 to 100); for a fractured rock mass the Hoek-Brown base, Eqs. 10-24 to 10-29 (pp. 10-23, 10-24; Turner and Ramey 2010): m_b = m_i*exp((GSI-100)/28), s = exp((GSI-100)/9), a = 1/2+(1/6)(exp(-GSI/15)-exp(-20/3)), A = sigma'vb + qu*(m_b*sigma'vb/qu + s)^a, q_BN = A + qu*(m_b*A/qu + s)^a, q_BN/qu limited to 2.5 (p. 10-24), on the tip rock layer's qu, GSI and m_i and the vertical effective stress at the base; resistance factor 0.50 (p. 10-24, Table 8-4); which of the two by rock_base_conditions_met. 'rqd' = q_b/q_u reduced by the tip rock's RQD below 70 (1.5 at 50, 0.8 at 25, 0.4 at 0), cited to GEC-10's 2010 edition, not in the reference library, and not a method GEC-10 Table 8-4 gives a resistance factor for."},
    "rock_base_conditions_met": {"type": "bool", "required": False, "description": "For rock_base_method 'gec10' with the tip in rock: whether the rock below the base meets the three conditions GEC-10 p. 10-22 sets on q_BN = 2.5*qu (massive or tightly jointed, no compressible seams or joints, to at least one diameter below the base; no solution cavities or voids beneath it; a clean base achieved and verified). The engineer's judgment: Example 10-4 takes 2.5 with RQD 37 to 100 because 'Rock coring and local experience indicate the rock mass below the socket tip is competent'. true = 2.5*qu (Eq. 10-23) as stated. false = GEC-10's base for a fractured rock mass, the Hoek-Brown base (Eqs. 10-24 to 10-29), on the tip rock layer's GSI and m_i; refused, naming them, where that layer does not give them. Left out = taken as met, said in the result's choices, where every layer within one base diameter below the base is rock with RQD 100 (the layer default), with the Hoek-Brown base on the tip rock layer's GSI and m_i beside it where they are given; where a layer there has an RQD below 100 or is soil, the Hoek-Brown base on the tip rock layer's GSI and m_i (2.5*qu bounds it above), and the call is refused, naming them, where that layer does not give them."},
    "large_base_reduction": {"type": "bool", "required": False, "default": False, "description": "O'Neill & Reese's (1999) reduction of base resistance for a large base: 1.27/B in cohesionless soil for a base wider than 1.27 m, F_r = 2.5/(a*B + 2.5*b) <= 1 in cohesive soil for a base wider than 1.90 m (B the bell's where there is one). Default false: GEC-10 (2018) prints neither (Eqs. 10-14, 10-19, 10-20); true applies it."},
    "pa": {"type": "float", "required": False, "default": 101.325, "description": "Atmospheric pressure (kPa) for the rational chains."},
    "rock_side_C": {"type": "float", "required": False, "default": 1.0, "description": "C in the rock side resistance f_s = C*pa*sqrt(qu/pa) (GEC-10 Eq. 10-21, p. 10-20). Default 1.0, 'recommended for design of \"normal\" rock sockets' (pp. 10-20, 10-21; Example 10-4): conventional equipment, nominally clean sidewalls, no caving, no smearing. 0.65 = the value of Eq. 10-22 with alpha_E = 1.0 (rock_side_method 'caving' takes alpha_E from Table 10-3 by RQD). 1.9 for roughened sockets, verified by load tests or local experience (p. 10-22). For rock_side_method 'normal'; refused with 'caving', which takes no C. Needs a rock layer."},
    "rock_side_method": {"type": "str", "required": False, "default": "normal", "allowed_values": ["normal", "caving"], "description": "Rock side resistance. 'normal' (default) = GEC-10 Eq. 10-21, f_s = C*pa*sqrt(qu/pa) with rock_side_C, for a 'normal' socket: conventional equipment, nominally clean sidewalls (pp. 10-20, 10-21). 'caving' = Eq. 10-22 (O'Neill and Reese 1999), f_s = 0.65*alpha_E*pa*sqrt(qu/pa), alpha_E from Table 10-3 (p. 10-21) by each rock layer's RQD and rock_joints (closed joints 1.00, 0.85, 0.60, 0.50, 0.45; open or gouge-filled 0.85, 0.55, 0.55, 0.50, 0.45 at RQD 100, 70, 50, 30, 20 %; linear between the rows, held at 0.45 below RQD 20), for highly fractured or weathered rock that cannot be drilled without artificial support (casing, plug-ahead) or where caving is anticipated. Needs a rock layer."},
    "rock_joints": {"type": "str", "required": False, "default": "closed", "allowed_values": ["closed", "open"], "description": "For rock_side_method 'caving': the joint condition of GEC-10 Table 10-3, 'closed' (default) or 'open' (open or gouge-filled joints), from the cores."},
}
_INFO_SHAFT_GEOM = {
    "socket_diameter": {"type": "float", "required": False, "description": "Rock socket diameter (m). Defaults to shaft diameter. One diameter for every rock layer: GEC-10 sizes a socket by one diameter B (Eq. 10-21, Example 10-4) and gives none per rock layer. The rock side resistance and, with the tip in rock, the base (pi/4*B^2, GEC-10 Example 10-4, p. 10-28) take it."},
    "bell_diameter": {"type": "float", "required": False, "description": "Belled base diameter (m)."},
    "casing_depth": {"type": "float", "required": False, "default": 0.0, "description": "Permanent casing depth (m, no side resistance)."},
    "fractured_rock_thickness": {"type": "float", "required": False, "default": 0.0, "description": "Decomposed or fractured rock (m) below the top of the first rock layer that is 'not considered part of the socket for design purposes' (GEC-10 p. 8-15): no rock side resistance over it, while the soil above keeps its own (casing_depth takes both). 0 = all rock is socket. Needs a rock layer."},
    "concrete_fc": {"type": "float", "required": False, "description": "Compressive strength f'c of the shaft concrete (kPa; 28 MPa = 28000, at least 1000). The rock side resistance takes qu no higher than it: 'For design, the value of qu used in Equation 10-21 should not exceed the compressive strength of the drilled shaft concrete' (GEC-10 p. 10-21; Example 10-4 checks qu 360 ksf against f'c 576 ksf), in Eq. 10-22 alike. Left out, that limit is not checked and the result's choices say so. No structural check of the shaft is made. Needs a rock layer."},
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
        "brief": "Drilled shaft capacity with the LRFD resistance factors of GEC-10 Table 8-4.",
        "parameters": {
            "diameter": _INFO_DIAMETER,
            "shaft_length": {"type": "float", "required": True, "description": "Shaft length (m). Alias: length."},
            "layers": _INFO_LAYERS,
            "gwt_depth": _INFO_GWT,
            "tip_soil_type": {"type": "str", "required": False, "allowed_values": ["cohesive", "cohesionless", "rock"], "description": "Soil type at the shaft tip, for the base resistance factor (GEC-10 Table 8-4: 0.40 cohesive, 0.50 cohesionless and rock). Left out, the soil of the layer the base is in; given, it must be that soil, else the call is refused."},
            **_INFO_RATIONAL,
            **_INFO_SHAFT_GEOM,
        },
        "returns": {"Q_ultimate_kN": "Unfactored ultimate capacity.", "lrfd": "Factored resistances by component (GEC-10 Table 8-4, p. 8-14: side 0.45 clay, 0.55 sand, 0.50 rock; base 0.40 clay, 0.50 sand and rock), the base factor by the soil the base is in."},
    },
    "get_resistance_factors": {
        "category": "Drilled Shaft",
        "brief": "AASHTO LRFD resistance factors for drilled shaft design.",
        "parameters": {},
        "returns": {"resistance_factors": "Dict of component → phi factor."},
    },
}

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
        "socket_length": _NO_SOCKET_LENGTH}

declare_aliases(METHOD_INFO, {
    m: _SHAFT_ALIASES for m in ("drilled_shaft_capacity", "capacity_vs_depth",
                                "lrfd_capacity")})

# When each default applies (GM-119): the CIUC transform and the
# atmospheric pressure go with GEC-10's rational chains only, and K/Ko
# with its beta (GM-241). The base methods, the large-base reduction and
# the bottom zone apply by the soil at the base and above it, which no
# other parameter says: listed when the result's choices report them as
# defaults (GM-244).
_DS_APPLIES = {"su_test_type": {"alpha_method": ["rational"]},
               "pa": [{"alpha_method": ["rational"]},
                      {"beta_method": ["rational"]}],
               "K_over_Ko": {"beta_method": ["rational"]},
               "clay_base_method": False, "rock_base_method": False,
               "rock_base_conditions_met": False,
               "large_base_reduction": False, "bottom_exclusion": False,
               "rock_joints": {"rock_side_method": ["caving"]},
               "concrete_fc": False}
# The side resistance settings by soil (sand, clay, rock socket) apply by
# the layers, which no other parameter says: listed where the run's
# choices report them (capacity_vs_depth gives no choices, so keeps them).
_BY_SOIL = {name: False for name in (
    "beta_method", "K_over_Ko", "alpha_method", "su_test_type",
    "rock_side_C", "fractured_rock_thickness", "rock_side_method",
    "rock_joints")}
declare_applies_when(METHOD_INFO, {
    "drilled_shaft_capacity": {**_DS_APPLIES, **_BY_SOIL},
    "capacity_vs_depth": _DS_APPLIES,
    "lrfd_capacity": {**_DS_APPLIES, **_BY_SOIL}})


# What a left-out input with no default stands for (BACKLOG GM-165):
# the water table, the rock socket's diameter and the top zone by soil
# type (GM-241).
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
_NO_WATER = ("no water table",
             "total unit weight throughout: effective stress equals total "
             "stress")
_TOP_ZONE = ("1.5 m in cohesive soil, 0 in cohesionless soil and rock",
             "GEC-10 (FHWA-NHI-18-024) p. 10-17 prints the top 5 ft in "
             "Sec. 10.3.5.2, Cohesive Soils; Secs. 10.3.5.1 and 10.3.5.3 "
             "print no such zone in cohesionless soil or rock: Examples "
             "10-1 and 10-5 take the sand's side resistance from the "
             "ground surface, Example 10-4 the socket's over its whole "
             "length")
_ROCK_CONDITIONS = (
    "the three conditions of Eq. 10-23 taken as met where the layers "
    "within one base diameter below the base show no RQD below 100 and no "
    "soil (the Hoek-Brown base beside it where the tip rock layer gives GSI "
    "and m_i); where they show either, the Hoek-Brown base of a fractured "
    "rock mass (Eqs. 10-24 to 10-29) on the tip rock layer's GSI and m_i, "
    "refused, naming them, where it gives none",
    "GEC-10 (FHWA-NHI-18-024) p. 10-22: N*cr = 2.5 is recommended where "
    "the rock below the base is massive or tightly jointed to at least one "
    "diameter below it, with no solution cavities or voids and a clean "
    "base; whether it is, is the engineer's judgment")
# The concrete strength and the rock side resistance (GM-284).
_NO_FC = ("not checked: qu as given in the rock side resistance",
          "GEC-10 (FHWA-NHI-18-024) p. 10-21: the qu used in Eq. 10-21 "
          "should not exceed the compressive strength of the drilled shaft "
          "concrete; without f'c the limit cannot be checked")
declare_when_absent(METHOD_INFO, {
    "drilled_shaft_capacity": {"gwt_depth": _NO_WATER,
                               "socket_diameter": "diameter",
                               "top_exclusion": _TOP_ZONE,
                               "rock_base_conditions_met": _ROCK_CONDITIONS,
                               "concrete_fc": _NO_FC},
    "capacity_vs_depth": {
        "gwt_depth": _NO_WATER, "socket_diameter": "diameter",
        "top_exclusion": _TOP_ZONE,
        "rock_base_conditions_met": _ROCK_CONDITIONS,
        "concrete_fc": _NO_FC,
        "depth_max": ("the shaft length or the soil profile depth, "
                      "whichever is less", "the deepest trial shaft")},
    "lrfd_capacity": {"gwt_depth": _NO_WATER, "socket_diameter": "diameter",
                      "top_exclusion": _TOP_ZONE,
                      "rock_base_conditions_met": _ROCK_CONDITIONS,
                      "concrete_fc": _NO_FC,
                      "tip_soil_type": (
                          "the soil of the layer the base is in",
                          "GEC-10 (FHWA-NHI-18-024) Table 8-4 (p. 8-14) "
                          "gives the base resistance factor by the soil and "
                          "method of the base")},
})

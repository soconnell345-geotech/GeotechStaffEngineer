"""Downdrag adapter — Fellenius neutral plane (UFC 3-220-20) plus the
CGPR #56 method family (Greenfield & Filz 2009): Endo, Poulos, the report's
Fellenius formulation, PILENEG, pile-group methods, and a comparison runner;
and post-liquefaction downdrag (FHWA GEC-10 §10.7)."""

from downdrag import (
    DowndragSoilLayer, DowndragSoilProfile, DowndragAnalysis,
    endo_method, poulos_method, fellenius_method_cgpr56, pileneg_procedure,
    rigid_block_method, drag_load_reduction_method,
    downdrag_method_comparison, post_liquefaction_downdrag,
    LIQUEFIED_SIDE_RESISTANCE,
)
from downdrag.liquefaction import LIQUEFIED_LAYER_KEYS
from funhouse_agent.adapters import (
    check_nested_keys, check_nested_list, clean_result,
    declare_applies_when, field_keys, reject_unknown_params, require_keys,
    require_params,
)

_SOIL_TYPES = ("cohesionless", "cohesive")

# Keys a layers[] dict may carry: DowndragSoilLayer's fields. Anything else
# was read with .get and a default and silently left out (GM-58): a settling
# clay's 'pc' ran it normally consolidated and 'Es' left a settling sand
# without its modulus.
_LAYER_KEYS = field_keys(DowndragSoilLayer)
# The CGPR #56 Fellenius recompute's 'consolidation' dict and its layers, as
# downdrag.cgpr56.consolidation_settlement_profile reads them (C_ec and p_c
# with .get: a misspelt one ran the layer as overconsolidated throughout).
_CONSOLIDATION_KEYS = ("layers", "p0_profile", "dp_profile",
                       "sublayer_thickness")
_CONSOLIDATION_LAYER_KEYS = ("z_top", "z_bot", "C_er", "C_ec", "p_c")
# Words the near-miss matcher cannot place; only keys valid for the item are
# offered. Suggestions only, never renamed.
_LAYER_KEY_HINTS = {
    "frictionangle": ("phi",), "c": ("cu",), "cohesion": ("cu",),
    "su": ("cu",), "gamma": ("unit_weight",),
    "pc": ("sigma_p", "p_c"), "preconsolidation": ("sigma_p", "p_c"),
    "sigmap": ("p_c",), "cr": ("Cr", "C_er"), "cc": ("Cc", "C_ec"),
    "modulus": ("E_s",), "nu": ("nu_s",), "poisson": ("nu_s",),
    "settles": ("settling",), "top": ("z_top",), "bottom": ("z_bot",),
}
_LAYER_KEY_NOTES = {
    ("C_er",): ("(C_er is the modified recompression index Cr/(1+e0), not "
                "Cr itself.)"),
    ("C_ec",): ("(C_ec is the modified compression index Cc/(1+e0), not "
                "Cc itself.)"),
}

# Every top-level parameter _run_downdrag_analysis consumes.
_VALID_PARAMS = (
    "layers", "gwt_depth", "pile_length", "pile_diameter", "pile_perimeter",
    "pile_area", "pile_E", "pile_unit_weight", "Q_dead", "structural_capacity",
    "allowable_settlement", "fill_thickness", "fill_unit_weight", "gw_drawdown", "Nt", "n_sublayers",
    "neutral_plane_method", "toe_mobilization", "endo_bearing_condition",
    "skin_friction_stress", "include_pile_weight",
    "influence_depth_criterion",
)


#: What each kind of layer needs; appended to a layer's validation error.
_LAYER_FIX = (
    "Each layer needs thickness (m), soil_type, unit_weight (kN/m3) and phi "
    "(deg, cohesionless) or cu (kPa, cohesive). A layer with settling=true "
    "also needs what its settlement is computed from: cohesionless -> E_s "
    "(elastic modulus, kPa, > 0; nu_s optional); cohesive -> Cc and e0 (Cr, "
    "sigma_p optional) or C_ec. If the layer does not settle, set "
    "settling=false.")


def run_downdrag_analysis(params, *, method="downdrag_analysis"):
    """(result, analysis) of ``downdrag_analysis``: the tool's own code
    path, also run by ``calc_package.downdrag_package`` (GM-141).
    ``method`` names the caller in error messages."""
    reject_unknown_params(params, _VALID_PARAMS, method=method)
    analysis = _build_analysis(params, method, _VALID_PARAMS)
    return analysis.compute(), analysis


def _build_analysis(params, method, valid):
    """The ``DowndragAnalysis`` the static parameters describe."""
    require_params(params, ["layers", "pile_length", "pile_diameter"],
                   method=method, valid=valid)
    check_nested_list(params["layers"], _LAYER_KEYS,
                      method=method, label="layers",
                      hints=_LAYER_KEY_HINTS, notes=_LAYER_KEY_NOTES)
    layers = []
    for i, l in enumerate(params["layers"]):
        require_keys(l, ["thickness", "soil_type", "unit_weight"], method=method)
        if l["soil_type"] not in _SOIL_TYPES:
            raise ValueError(
                f"{method}: layer soil_type must be one of {list(_SOIL_TYPES)} "
                f"(got '{l['soil_type']}'). Mark settling layers with settling=True."
            )
        try:
            layers.append(DowndragSoilLayer(
                thickness=l["thickness"], soil_type=l["soil_type"], unit_weight=l["unit_weight"],
                phi=l.get("phi", 0.0), cu=l.get("cu", 0.0), beta=l.get("beta"), alpha=l.get("alpha"),
                Cc=l.get("Cc", 0.0), Cr=l.get("Cr", 0.0), e0=l.get("e0", 0.0),
                C_ec=l.get("C_ec"), C_er=l.get("C_er"), sigma_p=l.get("sigma_p"),
                E_s=l.get("E_s"), nu_s=l.get("nu_s", 0.3),
                settling=l.get("settling", False), description=l.get("description", ""),
            ))
        except ValueError as exc:
            raise ValueError(
                f"{method}: layers[{i}] ({l.get('description') or l['soil_type']}): "
                f"{exc}. {_LAYER_FIX}") from None
    soil = DowndragSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth", 0.0))
    analysis = DowndragAnalysis(
        soil=soil, pile_length=params["pile_length"], pile_diameter=params["pile_diameter"],
        pile_perimeter=params.get("pile_perimeter"), pile_area=params.get("pile_area"),
        pile_E=params.get("pile_E", 200e6), pile_unit_weight=params.get("pile_unit_weight", 78.5),
        include_pile_weight=params.get("include_pile_weight", False),
        Q_dead=params.get("Q_dead", 0.0), structural_capacity=params.get("structural_capacity"),
        allowable_settlement=params.get("allowable_settlement"),
        fill_thickness=params.get("fill_thickness", 0.0), fill_unit_weight=params.get("fill_unit_weight", 19.0),
        gw_drawdown=params.get("gw_drawdown", 0.0), Nt=params.get("Nt"), n_sublayers=params.get("n_sublayers", 10),
        neutral_plane_method=params.get("neutral_plane_method", "auto"),
        toe_mobilization=params.get("toe_mobilization", 1.0),
        endo_bearing_condition=params.get("endo_bearing_condition"),
        skin_friction_stress=params.get("skin_friction_stress", "final"),
        influence_depth_criterion=params.get("influence_depth_criterion",
                                             "bearing_stress"),
    )
    return analysis


def _run_downdrag_analysis(params):
    return _basis_first(run_downdrag_analysis(params)[0].to_dict())


#: Result keys that say how far to trust the numbers; put ahead of the long
#: depth profiles so a size-capped tool result can never cut them off.
_BASIS_KEYS = ("neutral_plane_depth_m", "neutral_plane_method",
               "headline_method", "neutral_plane_basis", "methods_range",
               "warnings", "neutral_plane_comparison", "pile_weight_basis",
               "below_toe")


def _basis_first(d: dict) -> dict:
    """The result with how the neutral plane was located, the spread of the
    methods and the warnings first (live smoke G8: an untold toe default put
    the plane at the toe; the warnings must reach the model)."""
    out = {k: d[k] for k in _BASIS_KEYS if k in d}
    out.update((k, v) for k, v in d.items() if k not in out)
    return clean_result(out)


# ── Post-liquefaction downdrag (FHWA GEC-10 §10.7; GM-03) ────────────────

_POST_LIQUEFACTION_PARAMS = _VALID_PARAMS + (
    "liquefied_layers", "liquefaction_result", "volumetric_strain",
    "fs_liq_threshold", "liquefied_side_resistance",
    "liquefied_side_resistance_ratio",
)
_LIQUEFIED_KEY_HINTS = {
    "strain": ("volumetric_strain",), "ev": ("volumetric_strain",),
    "epsv": ("volumetric_strain",), "epsilonv": ("volumetric_strain",),
    "settle": ("settlement",), "settlementm": ("settlement",),
    "fos": ("fs_liq",), "fosliq": ("fs_liq",), "fs": ("fs_liq",),
    "n160": ("N1_60",), "sr": ("su_r",), "residualstrength": ("su_r",),
    "z_top": ("top",), "ztop": ("top",), "zbot": ("bottom",),
    "zbottom": ("bottom",),
}


def _run_post_liquefaction(params):
    method = "post_liquefaction_downdrag"
    reject_unknown_params(params, _POST_LIQUEFACTION_PARAMS, method=method)
    if params.get("liquefied_layers") is not None:
        check_nested_list(params["liquefied_layers"], LIQUEFIED_LAYER_KEYS,
                          method=method, label="liquefied_layers",
                          hints=_LIQUEFIED_KEY_HINTS)
    static = {k: v for k, v in params.items() if k in _VALID_PARAMS}
    analysis = _build_analysis(static, method, _POST_LIQUEFACTION_PARAMS)
    try:
        result = post_liquefaction_downdrag(
            analysis,
            liquefied_layers=params.get("liquefied_layers"),
            liquefaction_result=params.get("liquefaction_result"),
            volumetric_strain=params.get("volumetric_strain"),
            fs_liq_threshold=params.get("fs_liq_threshold", 1.0),
            liquefied_side_resistance=params.get(
                "liquefied_side_resistance", "half_pre_earthquake"),
            liquefied_side_resistance_ratio=params.get(
                "liquefied_side_resistance_ratio"),
        )
    except ValueError as exc:
        raise ValueError(f"{method}: {exc}") from None
    return clean_result(result.to_dict())


# ── CGPR #56 method family (Greenfield & Filz 2009) ──────────────────────

def _profile(params, key, method, required=True):
    """Convert a JSON [[depth, value], ...] array to a list of tuples."""
    raw = params.get(key)
    if raw is None:
        if required:
            raise ValueError(f"{method}: '{key}' is required — an array of "
                             f"[depth_m, value] pairs.")
        return None
    try:
        pts = [(float(p[0]), float(p[1])) for p in raw]
    except (TypeError, ValueError, IndexError):
        raise ValueError(
            f"{method}: '{key}' must be an array of [depth, value] pairs "
            f"(e.g. [[0, 0], [15, 40]]); a repeated depth encodes a jump."
        )
    return pts


def _consolidation(params, method):
    """Validate the optional consolidation dict for the Fellenius/comparison
    settlement recompute."""
    cons = params.get("consolidation")
    if cons is None:
        return None
    check_nested_keys(cons, _CONSOLIDATION_KEYS, method=method,
                      label="consolidation", hints=_LAYER_KEY_HINTS)
    require_keys(cons, ["layers", "p0_profile", "dp_profile"], method=method,
                 item_label="consolidation")
    check_nested_list(cons["layers"], _CONSOLIDATION_LAYER_KEYS,
                      method=method, label="consolidation.layers",
                      hints=_LAYER_KEY_HINTS, notes=_LAYER_KEY_NOTES)
    for lay in cons["layers"]:
        require_keys(lay, ["z_top", "z_bot", "C_er"], method=method,
                     item_label="consolidation layers[]")
    return cons


_ENDO_PARAMS = (
    "Q_static", "pile_length", "pile_perimeter", "skin_friction_profile",
    "bearing_condition", "neutral_plane_ratio", "neutral_plane_depth",
)


def _run_endo(params):
    reject_unknown_params(params, _ENDO_PARAMS, method="endo_downdrag")
    require_params(params, ["Q_static", "pile_length", "pile_perimeter"],
                   method="endo_downdrag", valid=_ENDO_PARAMS)
    return clean_result(endo_method(
        Q_static=params["Q_static"], pile_length=params["pile_length"],
        pile_perimeter=params["pile_perimeter"],
        skin_friction_profile=_profile(params, "skin_friction_profile",
                                       "endo_downdrag"),
        bearing_condition=params.get("bearing_condition"),
        neutral_plane_ratio=params.get("neutral_plane_ratio"),
        neutral_plane_depth=params.get("neutral_plane_depth"),
    ).to_dict())


_POULOS_PARAMS = (
    "Q_static", "pile_length", "pile_perimeter", "pile_area", "pile_E",
    "depth_to_bearing_layer", "toe_bearing_capacity", "toe_area",
    "skin_friction_profile", "fs_consolidating", "fs_bearing",
    "settlement_profile", "s_equivalent",
)


def _run_poulos(params):
    reject_unknown_params(params, _POULOS_PARAMS, method="poulos_downdrag")
    require_params(
        params,
        ["Q_static", "pile_length", "pile_perimeter", "pile_area", "pile_E",
         "depth_to_bearing_layer", "toe_bearing_capacity"],
        method="poulos_downdrag", valid=_POULOS_PARAMS)
    return clean_result(poulos_method(
        Q_static=params["Q_static"], pile_length=params["pile_length"],
        pile_perimeter=params["pile_perimeter"], pile_area=params["pile_area"],
        pile_E=params["pile_E"],
        depth_to_bearing_layer=params["depth_to_bearing_layer"],
        toe_bearing_capacity=params["toe_bearing_capacity"],
        skin_friction_profile=_profile(params, "skin_friction_profile",
                                       "poulos_downdrag", required=False),
        toe_area=params.get("toe_area"),
        fs_consolidating=params.get("fs_consolidating"),
        fs_bearing=params.get("fs_bearing"),
        settlement_profile=_profile(params, "settlement_profile",
                                    "poulos_downdrag", required=False),
        s_equivalent=params.get("s_equivalent"),
    ).to_dict())


_FELLENIUS_PARAMS = (
    "Q_static", "pile_length", "pile_perimeter", "pile_area", "pile_E",
    "toe_bearing_capacity", "toe_area", "skin_friction_profile",
    "settlement_profile", "consolidation", "eq_footing_width",
    "eq_footing_length", "eq_footing_load",
)


def _run_fellenius_cgpr56(params):
    reject_unknown_params(params, _FELLENIUS_PARAMS, method="fellenius_cgpr56")
    require_params(
        params,
        ["Q_static", "pile_length", "pile_perimeter", "pile_area", "pile_E",
         "toe_bearing_capacity"],
        method="fellenius_cgpr56", valid=_FELLENIUS_PARAMS)
    return clean_result(fellenius_method_cgpr56(
        Q_static=params["Q_static"], pile_length=params["pile_length"],
        pile_perimeter=params["pile_perimeter"], pile_area=params["pile_area"],
        pile_E=params["pile_E"],
        toe_bearing_capacity=params["toe_bearing_capacity"],
        skin_friction_profile=_profile(params, "skin_friction_profile",
                                       "fellenius_cgpr56"),
        toe_area=params.get("toe_area"),
        settlement_profile=_profile(params, "settlement_profile",
                                    "fellenius_cgpr56", required=False),
        consolidation=_consolidation(params, "fellenius_cgpr56"),
        eq_footing_width=params.get("eq_footing_width"),
        eq_footing_length=params.get("eq_footing_length"),
        eq_footing_load=params.get("eq_footing_load"),
    ).to_dict())


_PILENEG_PARAMS = (
    "Q_static", "pile_length", "pile_perimeter", "pile_area", "pile_E",
    "toe_bearing_capacity", "toe_area", "pile_width", "bearing_E",
    "bearing_nu", "skin_friction_profile", "settlement_profile",
    "trial_depths",
)


def _run_pileneg(params):
    reject_unknown_params(params, _PILENEG_PARAMS, method="pileneg_downdrag")
    require_params(
        params,
        ["Q_static", "pile_length", "pile_perimeter", "pile_area", "pile_E",
         "toe_bearing_capacity", "pile_width", "bearing_E"],
        method="pileneg_downdrag", valid=_PILENEG_PARAMS)
    return clean_result(pileneg_procedure(
        Q_static=params["Q_static"], pile_length=params["pile_length"],
        pile_perimeter=params["pile_perimeter"], pile_area=params["pile_area"],
        pile_E=params["pile_E"],
        toe_bearing_capacity=params["toe_bearing_capacity"],
        pile_width=params["pile_width"], bearing_E=params["bearing_E"],
        skin_friction_profile=_profile(params, "skin_friction_profile",
                                       "pileneg_downdrag"),
        settlement_profile=_profile(params, "settlement_profile",
                                    "pileneg_downdrag"),
        bearing_nu=params.get("bearing_nu", 0.3),
        toe_area=params.get("toe_area"),
        trial_depths=params.get("trial_depths"),
    ).to_dict())


_COMPARISON_PARAMS = (
    "Q_static", "pile_length", "pile_perimeter", "pile_area", "pile_E",
    "toe_bearing_capacity", "toe_area", "skin_friction_profile",
    "settlement_profile", "consolidation", "endo_bearing_condition",
    "endo_neutral_plane_ratio", "depth_to_bearing_layer",
    "poulos_s_equivalent", "pile_width", "bearing_E", "bearing_nu",
    "eq_footing_width", "eq_footing_length", "eq_footing_load",
)


def _run_comparison(params):
    reject_unknown_params(params, _COMPARISON_PARAMS,
                          method="downdrag_method_comparison")
    require_params(
        params,
        ["Q_static", "pile_length", "pile_perimeter", "pile_area", "pile_E",
         "toe_bearing_capacity"],
        method="downdrag_method_comparison", valid=_COMPARISON_PARAMS)
    result = downdrag_method_comparison(
        Q_static=params["Q_static"], pile_length=params["pile_length"],
        pile_perimeter=params["pile_perimeter"], pile_area=params["pile_area"],
        pile_E=params["pile_E"],
        toe_bearing_capacity=params["toe_bearing_capacity"],
        skin_friction_profile=_profile(params, "skin_friction_profile",
                                       "downdrag_method_comparison"),
        toe_area=params.get("toe_area"),
        settlement_profile=_profile(params, "settlement_profile",
                                    "downdrag_method_comparison",
                                    required=False),
        consolidation=_consolidation(params, "downdrag_method_comparison"),
        endo_bearing_condition=params.get("endo_bearing_condition"),
        endo_neutral_plane_ratio=params.get("endo_neutral_plane_ratio"),
        depth_to_bearing_layer=params.get("depth_to_bearing_layer"),
        poulos_s_equivalent=params.get("poulos_s_equivalent"),
        pile_width=params.get("pile_width"),
        bearing_E=params.get("bearing_E"),
        bearing_nu=params.get("bearing_nu", 0.3),
        eq_footing_width=params.get("eq_footing_width"),
        eq_footing_length=params.get("eq_footing_length"),
        eq_footing_load=params.get("eq_footing_load"),
    )
    out = clean_result(result.to_dict())
    out["summary_table"] = result.summary()
    return out


_RIGID_BLOCK_PARAMS = (
    "Q_static_group", "n_piles", "spacing", "neutral_plane_depth",
    "cu_average", "delta_q",
)


def _run_rigid_block(params):
    reject_unknown_params(params, _RIGID_BLOCK_PARAMS,
                          method="group_rigid_block")
    require_params(params, list(_RIGID_BLOCK_PARAMS),
                   method="group_rigid_block", valid=_RIGID_BLOCK_PARAMS)
    return clean_result(rigid_block_method(
        Q_static_group=params["Q_static_group"], n_piles=params["n_piles"],
        spacing=params["spacing"],
        neutral_plane_depth=params["neutral_plane_depth"],
        cu_average=params["cu_average"], delta_q=params["delta_q"],
    ).to_dict())


_REDUCTION_PARAMS = (
    "F_max_single", "Q_static_group", "n_piles", "location", "s_over_d",
    "spacing", "pile_diameter",
)


def _run_group_reduction(params):
    reject_unknown_params(params, _REDUCTION_PARAMS,
                          method="group_drag_reduction")
    require_params(params, ["F_max_single", "Q_static_group", "n_piles",
                            "location"],
                   method="group_drag_reduction", valid=_REDUCTION_PARAMS)
    return clean_result(drag_load_reduction_method(
        F_max_single=params["F_max_single"],
        Q_static_group=params["Q_static_group"], n_piles=params["n_piles"],
        location=params["location"], s_over_d=params.get("s_over_d"),
        spacing=params.get("spacing"),
        pile_diameter=params.get("pile_diameter"),
    ).to_dict())


METHOD_REGISTRY = {
    "downdrag_analysis": _run_downdrag_analysis,
    "endo_downdrag": _run_endo,
    "poulos_downdrag": _run_poulos,
    "fellenius_cgpr56": _run_fellenius_cgpr56,
    "pileneg_downdrag": _run_pileneg,
    "downdrag_method_comparison": _run_comparison,
    "group_rigid_block": _run_rigid_block,
    "group_drag_reduction": _run_group_reduction,
    "post_liquefaction_downdrag": _run_post_liquefaction,
}

METHOD_INFO = {
    "downdrag_analysis": {
        "category": "Downdrag",
        "brief": "Downdrag (negative skin friction), neutral plane method: every method's neutral plane and drag force side by side.",
        "parameters": {
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_diameter": {"type": "float", "required": True, "description": "Pile diameter (m)."},
            "layers": {"type": "array", "required": True, "description": "Array of {thickness, soil_type, unit_weight, phi, cu, beta, alpha, settling, description} dicts. soil_type must be 'cohesionless' or 'cohesive' (NOT 'sand'/'clay'/'settling_fill'). Mark settling layers with settling=True; a settling layer also needs its compressibility: cohesionless -> E_s (elastic modulus, kPa, > 0) and optional nu_s; cohesive -> Cc and e0 (optional Cr, sigma_p kPa) or C_ec/C_er."},
            "Q_dead": {"type": "float", "required": False, "default": 0.0, "description": "Dead load at pile top (kN)."},
            "fill_thickness": {"type": "float", "required": False, "default": 0.0, "description": "Fill thickness causing downdrag (m)."},
            "fill_unit_weight": {"type": "float", "required": False, "default": 19.0, "description": "Fill unit weight (kN/m3)."},
            "gw_drawdown": {"type": "float", "required": False, "default": 0.0, "description": "Groundwater drawdown (m)."},
            "gwt_depth": {"type": "float", "required": False, "default": 0.0, "description": "Groundwater depth (m)."},
            "pile_E": {"type": "float", "required": False, "default": 200e6, "description": "Pile elastic modulus (kPa). Default is steel."},
            "pile_perimeter": {"type": "float", "required": False, "description": "Pile perimeter (m). Computed from diameter if omitted."},
            "pile_area": {"type": "float", "required": False, "description": "Pile cross-section area (m2). Computed from diameter if omitted."},
            "structural_capacity": {"type": "float", "required": False, "description": "Pile structural capacity (kN) for the max-load check."},
            "allowable_settlement": {"type": "float", "required": False, "description": "Allowable pile settlement (m) for the settlement check."},
            "Nt": {"type": "float", "required": False, "description": "Toe bearing capacity factor: R_toe = Nt x sigma'v(toe) x A_toe in cohesionless soil, Nt x cu x A_toe in cohesive. Give the MOBILIZED value -- the toe resistance actually developed at the pile's settlement -- not the ultimate. Omitted, Nt is the ULTIMATE value from the toe layer's phi (100-150 in dense sand; 9 in clay); then the load and resistance curves often do not intersect and the neutral plane is where the pile and soil settle equally, the toe force compatible with the toe penetration (UFC 3-220-20 6-5.8.4.3; see neutral_plane_method). UFC 3-220-20 6-5.8.4.2: check 0 %, 50 % and 100 % toe mobilization."},
            "pile_unit_weight": {"type": "float", "required": False, "default": 78.5, "description": "Pile material unit weight (kN/m3), times pile_area for the pile's own weight; used only when include_pile_weight is true. Default 78.5 = steel, the same material as the pile_E default (concrete is about 24). For an H or pipe pile give pile_area as the steel area (pile_area defaults to the solid circle)."},
            "include_pile_weight": {"type": "bool", "required": False, "default": False, "description": "Whether the load curve carries the pile's own weight. Default false: left out, as UFC 3-220-20 Fig 6-26 and FHWA GEC-12 7.3.6.1 (and their worked examples) do. True adds the weight of the pile above the neutral plane to Q_np and the drag force."},
            "n_sublayers": {"type": "int", "required": False, "default": 10, "description": "Sublayers per soil layer for the depth discretization."},
            "neutral_plane_method": {"type": "string", "required": False, "default": "auto", "description": "Which method gives the headline numbers; every method is evaluated and returned in neutral_plane_comparison whatever is chosen. 'auto' (default) = the UFC 3-220-20 6-7.4 procedure: the intersection of the load and resistance curves with the toe at toe_mobilization, or, when they do not intersect, where the pile and soil settle equally with the toe force compatible with the toe penetration (UFC 6-5.8.4.3, Fig 6-19). 'force_equilibrium' (the curves' intersection only; error if they do not intersect); 'settlement_compatibility' (pile and soil settle equally, toe force compatible with the toe penetration); 'bearing_layer_top' (top of the bearing layer, end-bearing pile, UFC 6-7.4 step 5); 'pile_toe' (whole shaft in negative skin friction, upper bound); 'endo' (Endo et al. 1969, CGPR #56 Table 3.1 fraction of the embedment; needs endo_bearing_condition).", "allowed_values": ["auto", "force_equilibrium", "settlement_compatibility", "bearing_layer_top", "pile_toe", "endo"]},
            "toe_mobilization": {"type": "float", "required": False, "default": 1.0, "description": "Fraction (0-1) of the nominal toe resistance mobilized where the load and resistance curves intersect. 1.0 = UFC 6-7.4 step 3 full mobilization ('a conservative approach for evaluating the drag force'); 0.5 and 0 = the other UFC 6-5.8.4.2 / GEC-12 7.3.6.1 cases for settlement (all three always reported in the comparison)."},
            "endo_bearing_condition": {"type": "string", "required": False, "description": "CGPR #56 Table 3.1 condition for Endo et al. (1969).", "allowed_values": ["floating", "stiff_flexible", "end_bearing"]},
            "skin_friction_stress": {"type": "string", "required": False, "default": "final", "description": "Effective stress for beta-method skin friction and the sand toe: 'final' (default; sigma'v0 + fill/drawdown change, as UFC 3-220-20 6-7.4 step 2 requires) or 'initial' (sigma'v0).", "allowed_values": ["final", "initial"]},
            "influence_depth_criterion": {"type": "string", "required": False, "default": "bearing_stress", "description": "How deep below the toe the toe load is counted in the settlement (toe force spread 2V:1H from an equivalent footing as wide as the pile): to where its stress increase falls below 'bearing_stress' (default) = 10% of the bearing stress of the equivalent footing (UFC 3-220-20 6-5.8.4.3; 2.16 pile widths for a single pile) or 'effective_stress' = 10% of the existing vertical effective stress (FHWA GEC-12 7.3.5.3; deeper for a loaded toe in soft soil). Either way the fill / drawdown, which does not decay with depth, compresses the settling layers below the toe down to the bottom of the soil profile; give the profile down to the base of the compressible soil.", "allowed_values": ["bearing_stress", "effective_stress"]},
        },
        "returns": {
            "neutral_plane_depth_m": "Neutral plane depth (m) of the headline method.",
            "neutral_plane_method": "neutral_plane_method value of the headline method: 'force_equilibrium', 'settlement_compatibility', 'bearing_layer_top', 'pile_toe', 'endo', or 'none' (dead load exceeds the total resistance: no neutral plane).",
            "headline_method": "Name of the headline method as its reference words it (e.g. 'Neutral plane at 100% toe mobilization').",
            "neutral_plane_basis": "How the headline neutral plane was located, with its source (starts 'UFC 3-220-20 6-7.4 procedure' by default). Report it with the result.",
            "methods_range": "One sentence giving the spread of neutral plane and drag force over the methods that apply. Report it.",
            "neutral_plane_comparison": "Every method on the same inputs, one row each, named as its reference names it (neutral plane at 100/50/0% toe mobilization, toe resistance compatible with the toe penetration, neutral plane at the bearing layer, at the pile toe, Endo et al. 1969): source, assumptions, applies, used (the headline row), reason, neutral_plane_depth_m, max_pile_load_kN, dragload_kN, drag_force_kN, toe_force_kN, soil/pile settlement at the plane (mm), elastic_compression_above_np_mm, pile_settlement_mm (head). Present it as a table.",
            "warnings": "What the engineer must know about the result (e.g. curves that do not intersect, the upper-bound drag, an overloaded pile). Report every one.",
            "dragload_kN": "Drag load: negative skin friction above the neutral plane (kN; UFC 3-220-20 6-7.4 step 4).",
            "drag_force_kN": "Drag force DF = Q_np - Q_dead (kN; UFC 3-220-20 Eq 6-80, GEC-12 Eq 7-70); includes the pile weight above the plane only when include_pile_weight is true.",
            "include_pile_weight": "Whether the load curve carried the pile's own weight (false by default, as UFC 3-220-20 Fig 6-26 and GEC-12 7.3.6.1).",
            "pile_weight_basis": "One sentence saying whether the pile's own weight is in the load curve, naming the references. Report it.",
            "max_pile_load_kN": "Maximum axial load, at the neutral plane (kN).",
            "toe_resistance_kN": "Toe resistance from Nt (kN; ultimate unless Nt was given as mobilized).",
            "toe_force_mobilized_kN": "Toe force actually carried at the neutral plane (kN): the full toe resistance where the curves intersect with the toe fully mobilized, less otherwise.",
            "positive_skin_friction_kN": "Positive shaft resistance below the neutral plane (kN).",
            "total_resistance_kN": "Toe + positive shaft resistance (kN).",
            "pile_settlement_m": "Pile head settlement (m): settlement at the neutral plane + elastic compression above it (UFC 3-220-20 Fig 6-19).",
            "below_toe": "How deep the settlement below the toe was counted and why it stopped: the toe load to toe_load_counted_to_m (toe_load_stopped_by 'criterion', 'profile_end' or 'no_toe_load'), the fill / drawdown to fill_counted_to_m (the bottom of the soil profile), the settlement of each (mm), what the other influence_depth_criterion gives (other_criterion), and basis, one paragraph saying it. Report the basis.",
            "soil_settlement_at_np_m": "Soil settlement at the neutral plane (m), from the original ground.",
            "geotechnical_ok": "Q_dead <= total resistance (when checked).",
            "structural_ok": "LRFD demand <= structural_capacity (when given).",
            "z_m": "Depth profile (m); with axial_load_kN, soil_settlement_mm and unit_skin_friction_kPa at the same depths.",
        },
    },
    "endo_downdrag": {
        "category": "Downdrag",
        "brief": "Endo drag load estimate: assumed neutral plane at a fraction of embedment (CGPR #56 3.2.1).",
        "parameters": {
            "Q_static": {"type": "float", "required": True, "description": "Static (dead) load at pile head (kN)."},
            "pile_length": {"type": "float", "required": True, "description": "Embedded pile length (m)."},
            "pile_perimeter": {"type": "float", "required": True, "description": "Pile perimeter (m)."},
            "skin_friction_profile": {"type": "array", "required": True, "description": "Fully mobilized skin friction vs depth: [[depth_m, fs_kPa], ...] piecewise linear; repeat a depth to encode a jump."},
            "bearing_condition": {"type": "string", "required": False, "description": "Table 3.1 neutral-plane assumption: 'floating' (0.67L), 'stiff_flexible' (0.75L), 'end_bearing' (1.0L). Give exactly one of bearing_condition, neutral_plane_ratio, neutral_plane_depth.", "allowed_values": ["floating", "stiff_flexible", "end_bearing"]},
            "neutral_plane_ratio": {"type": "float", "required": False, "description": "Assumed neutral plane as a fraction of pile length (0-1)."},
            "neutral_plane_depth": {"type": "float", "required": False, "description": "Assumed neutral plane depth (m)."},
        },
        "returns": {"neutral_plane_depth": "Assumed NP depth (m).", "drag_load": "Drag load (kN).", "max_force": "Max pile force (kN). No settlement (method limitation)."},
    },
    "poulos_downdrag": {
        "category": "Downdrag",
        "brief": "Poulos hand approximation: two-layer closed-form neutral plane + settlement (CGPR #56 3.2.2).",
        "parameters": {
            "Q_static": {"type": "float", "required": True, "description": "Static load at pile head (kN)."},
            "pile_length": {"type": "float", "required": True, "description": "Embedded pile length (m)."},
            "pile_perimeter": {"type": "float", "required": True, "description": "Pile perimeter (m)."},
            "pile_area": {"type": "float", "required": True, "description": "Pile cross-section area (m2)."},
            "pile_E": {"type": "float", "required": True, "description": "Pile elastic modulus (kPa)."},
            "depth_to_bearing_layer": {"type": "float", "required": True, "description": "Depth to top of the incompressible bearing layer L1 (m)."},
            "toe_bearing_capacity": {"type": "float", "required": True, "description": "Toe bearing capacity qb (kPa)."},
            "skin_friction_profile": {"type": "array", "required": False, "description": "[[depth_m, fs_kPa], ...]; averages fs over each layer. Or give fs_consolidating/fs_bearing directly."},
            "fs_consolidating": {"type": "float", "required": False, "description": "Average skin friction of the consolidating layer (kPa)."},
            "fs_bearing": {"type": "float", "required": False, "description": "Average skin friction of the bearing layer (kPa)."},
            "settlement_profile": {"type": "array", "required": False, "description": "Free-field soil settlement vs depth [[depth_m, settlement_m], ...] (needed for settlement when NP is above the bearing layer)."},
            "s_equivalent": {"type": "float", "required": False, "description": "Equivalent-pile settlement in the bearing layer (m); needed only when the bearing layer is fully mobilized (z_max > L1)."},
            "toe_area": {"type": "float", "required": False, "description": "Toe bearing area (m2). Default pile_area."},
        },
        "returns": {"neutral_plane_depth": "NP depth (m).", "max_force": "Max pile force (kN).", "pile_settlement": "Pile head settlement (m), when inputs allow."},
    },
    "fellenius_cgpr56": {
        "category": "Downdrag",
        "brief": "Fellenius neutral plane per the CGPR #56 formulation (load/resistance curve intersection, user fs profile; 3.2.3). For the parameter-driven UFC version use downdrag_analysis.",
        "parameters": {
            "Q_static": {"type": "float", "required": True, "description": "Static load at pile head (kN)."},
            "pile_length": {"type": "float", "required": True, "description": "Embedded pile length (m)."},
            "pile_perimeter": {"type": "float", "required": True, "description": "Pile perimeter (m)."},
            "pile_area": {"type": "float", "required": True, "description": "Pile cross-section area (m2)."},
            "pile_E": {"type": "float", "required": True, "description": "Pile elastic modulus (kPa)."},
            "toe_bearing_capacity": {"type": "float", "required": True, "description": "Toe bearing capacity qb (kPa)."},
            "skin_friction_profile": {"type": "array", "required": True, "description": "Fully mobilized skin friction vs depth [[depth_m, fs_kPa], ...]; repeat a depth for a jump."},
            "settlement_profile": {"type": "array", "required": False, "description": "Free-field settlement [[depth_m, settlement_m], ...]. Used if 'consolidation' not given (pile-load-transfer settlement then NOT included)."},
            "consolidation": {"type": "object", "required": False, "description": "Recompute settlement with a 2:1 equivalent footing at the NP (report Table 3.5): {layers: [{z_top, z_bot, C_er, C_ec?, p_c?}], p0_profile: [[z, kPa]], dp_profile: kPa or [[z, kPa]], sublayer_thickness?}."},
            "eq_footing_width": {"type": "float", "required": False, "description": "Equivalent footing width at the NP (m); pile width, or group width for a group. Required with 'consolidation'."},
            "eq_footing_length": {"type": "float", "required": False, "description": "Equivalent footing length (m). Default width."},
            "eq_footing_load": {"type": "float", "required": False, "description": "Load on the equivalent footing (kN). Default Q_static; use the total group load for a group."},
            "toe_area": {"type": "float", "required": False, "description": "Toe bearing area (m2). Default pile_area."},
        },
        "returns": {"neutral_plane_depth": "NP depth (m).", "max_force": "Max pile force (kN).", "pile_settlement": "Pile head settlement (m).", "pile_in_failure": "True if load exceeds resistance everywhere."},
    },
    "pileneg_downdrag": {
        "category": "Downdrag",
        "brief": "PILENEG procedure: partially mobilized toe via elastic pile movement envelope, as documented in CGPR #56 3.2.4 (Briaud & Tucker 1997).",
        "parameters": {
            "Q_static": {"type": "float", "required": True, "description": "Static load at pile head (kN)."},
            "pile_length": {"type": "float", "required": True, "description": "Embedded pile length (m)."},
            "pile_perimeter": {"type": "float", "required": True, "description": "Pile perimeter (m)."},
            "pile_area": {"type": "float", "required": True, "description": "Pile cross-section area (m2)."},
            "pile_E": {"type": "float", "required": True, "description": "Pile elastic modulus (kPa)."},
            "toe_bearing_capacity": {"type": "float", "required": True, "description": "Toe bearing capacity qb (kPa)."},
            "pile_width": {"type": "float", "required": True, "description": "Pile width/diameter D (m) for the elastic toe penetration."},
            "bearing_E": {"type": "float", "required": True, "description": "Bearing layer elastic modulus Es (kPa)."},
            "bearing_nu": {"type": "float", "required": False, "default": 0.3, "description": "Bearing layer Poisson's ratio."},
            "skin_friction_profile": {"type": "array", "required": True, "description": "Fully mobilized skin friction vs depth [[depth_m, fs_kPa], ...]."},
            "settlement_profile": {"type": "array", "required": True, "description": "Free-field settlement [[depth_m, settlement_m], ...]."},
            "trial_depths": {"type": "array", "required": False, "description": "Trial NP depths (m) for a report-style coarse envelope. Default: dense grid."},
            "toe_area": {"type": "float", "required": False, "description": "Toe bearing area (m2). Default pile_area."},
        },
        "returns": {"neutral_plane_depth": "NP depth (m).", "max_force": "Max pile force (kN).", "toe_load": "Mobilized toe load (kN).", "toe_fully_mobilized": "If True, use fellenius_cgpr56 instead (report guidance).", "pile_settlement": "Pile head settlement (m)."},
    },
    "downdrag_method_comparison": {
        "category": "Downdrag",
        "brief": "Run all applicable CGPR #56 single-pile downdrag methods (Endo/Poulos/Fellenius/PILENEG) on one input set and tabulate.",
        "parameters": {
            "Q_static": {"type": "float", "required": True, "description": "Static load at pile head (kN)."},
            "pile_length": {"type": "float", "required": True, "description": "Embedded pile length (m)."},
            "pile_perimeter": {"type": "float", "required": True, "description": "Pile perimeter (m)."},
            "pile_area": {"type": "float", "required": True, "description": "Pile cross-section area (m2)."},
            "pile_E": {"type": "float", "required": True, "description": "Pile elastic modulus (kPa)."},
            "toe_bearing_capacity": {"type": "float", "required": True, "description": "Toe bearing capacity qb (kPa)."},
            "skin_friction_profile": {"type": "array", "required": True, "description": "Fully mobilized skin friction vs depth [[depth_m, fs_kPa], ...]; repeat a depth for a jump."},
            "settlement_profile": {"type": "array", "required": False, "description": "Free-field settlement [[depth_m, settlement_m], ...] (enables settlement outputs + PILENEG)."},
            "consolidation": {"type": "object", "required": False, "description": "Consolidation params for the Fellenius settlement recompute and (if no settlement_profile) the free-field profile — see fellenius_cgpr56."},
            "endo_bearing_condition": {"type": "string", "required": False, "description": "Enables Endo: 'floating'|'stiff_flexible'|'end_bearing' (Table 3.1).", "allowed_values": ["floating", "stiff_flexible", "end_bearing"]},
            "endo_neutral_plane_ratio": {"type": "float", "required": False, "description": "Alternative Endo NP assumption as a fraction of length."},
            "depth_to_bearing_layer": {"type": "float", "required": False, "description": "Enables Poulos: depth to top of bearing layer (m)."},
            "poulos_s_equivalent": {"type": "float", "required": False, "description": "Poulos equivalent-pile settlement (m) for the fully mobilized branch."},
            "pile_width": {"type": "float", "required": False, "description": "Enables PILENEG (with bearing_E): pile width D (m)."},
            "bearing_E": {"type": "float", "required": False, "description": "Enables PILENEG: bearing layer modulus Es (kPa)."},
            "bearing_nu": {"type": "float", "required": False, "default": 0.3, "description": "Bearing layer Poisson's ratio."},
            "eq_footing_width": {"type": "float", "required": False, "description": "Fellenius equivalent footing width (m); required with 'consolidation'."},
            "eq_footing_length": {"type": "float", "required": False, "description": "Fellenius equivalent footing length (m)."},
            "eq_footing_load": {"type": "float", "required": False, "description": "Fellenius equivalent footing load (kN). Default Q_static."},
            "toe_area": {"type": "float", "required": False, "description": "Toe bearing area (m2). Default pile_area."},
        },
        "returns": {"comparison_table": "Per-method rows: neutral_plane_depth, max_force, drag_load, pile_settlement (or a skip reason).", "results": "Full per-method results.", "summary_table": "Formatted text table."},
    },
    "group_rigid_block": {
        "category": "Downdrag - pile groups",
        "brief": "Rigid block pile-group drag loads: perimeter vs interior piles (CGPR #56 4.3.1; Terzaghi & Peck / Broms).",
        "parameters": {
            "Q_static_group": {"type": "float", "required": True, "description": "Total static load on the group (kN)."},
            "n_piles": {"type": "int", "required": True, "description": "Number of piles in the group."},
            "spacing": {"type": "float", "required": True, "description": "Center-to-center pile spacing (m)."},
            "neutral_plane_depth": {"type": "float", "required": True, "description": "Neutral plane depth from a single-pile analysis (m)."},
            "cu_average": {"type": "float", "required": True, "description": "Average undrained strength, surface to NP (kPa)."},
            "delta_q": {"type": "float", "required": True, "description": "Increased vertical effective stress from fill/water table change (kPa)."},
        },
        "returns": {"perimeter_pile_max_force": "Max force, perimeter pile (kN).", "interior_pile_max_force": "Max force, interior pile (kN)."},
    },
    "group_drag_reduction": {
        "category": "Downdrag - pile groups",
        "brief": "Group drag load via Jeong & Briaud reduction factor A on a single-pile result (CGPR #56 4.3.2, Table 4.1).",
        "parameters": {
            "F_max_single": {"type": "float", "required": True, "description": "Max pile force from a single-pile method (kN)."},
            "Q_static_group": {"type": "float", "required": True, "description": "Total static load on the group (kN)."},
            "n_piles": {"type": "int", "required": True, "description": "Number of piles (reduction valid ~9-25 piles)."},
            "location": {"type": "string", "required": True, "description": "Pile position in the group.", "allowed_values": ["interior", "side", "corner"]},
            "s_over_d": {"type": "float", "required": False, "description": "Spacing/diameter ratio. Or give spacing + pile_diameter."},
            "spacing": {"type": "float", "required": False, "description": "Center-to-center spacing (m)."},
            "pile_diameter": {"type": "float", "required": False, "description": "Pile diameter/width (m)."},
        },
        "returns": {"reduction_factor": "A (interpolated 2.5-5 diameters; 1.0 beyond 5).", "max_force_group_pile": "Reduced max force for the group pile (kN)."},
    },
}

# Post-liquefaction downdrag takes every input of the static analysis (the
# state before the earthquake) and the liquefied layers.
METHOD_INFO["post_liquefaction_downdrag"] = {
    "category": "Downdrag",
    "brief": "Post-liquefaction downdrag (FHWA GEC-10 10.7): neutral plane, drag force and settlement before the earthquake and after the liquefied layers reconsolidate, every neutral-plane method and every cited side resistance for the liquefied layers side by side.",
    "parameters": {
        **{name: dict(spec) for name, spec in
           METHOD_INFO["downdrag_analysis"]["parameters"].items()},
        "liquefied_layers": {"type": "array", "required": False, "description": "The layers that liquefy, instead of liquefaction_result: array of {top, bottom (m below the pile head), volumetric_strain (post-liquefaction reconsolidation strain, decimal: 0.03 = 3 %) or settlement (m, the layer's reconsolidation settlement), su_r (kPa, residual undrained strength; optional), N1_60 (optional; su_r by GEC-10 Eq 9-13 when su_r is not given), fs_liq (optional, reported)}. The other inputs (layers, pile, Q_dead, fill, neutral_plane_method, ...) describe the state before the earthquake, as for downdrag_analysis."},
        "liquefaction_result": {"type": "object", "required": False, "description": "The liquefaction tool's result, instead of liquefied_layers: its layer_results rows (depth_m, FOS_liq, N1_60; the SPT routes), or a profile with depth, factor_of_safety and volumetric_strain arrays. A reading liquefies where FOS_liq <= fs_liq_threshold and stands for the depth halfway to its neighbours, below gwt_depth. The tool's CPT route returns only LPI/LSN/LDI, so for a CPT give liquefied_layers."},
        "volumetric_strain": {"type": "float", "required": False, "description": "Reconsolidation volumetric strain (decimal) for liquefaction_result's readings: one value for every liquefied reading, or a list with one value per reading in the result's order. Required with liquefaction_result when the result carries no strains (the liquefaction tool's SPT routes compute none). GEC-10 10.7 estimates it by Ishihara and Yoshimine (1992) as described in Idriss and Boulanger (2008)."},
        "fs_liq_threshold": {"type": "float", "required": False, "default": 1.0, "description": "A reading of liquefaction_result liquefies where FOS_liq <= this. 1.0 = GEC-10 9.4.2.2 ('layers predicted to undergo liquefaction (FSliq <= 1.0)'); 1.0 < FSliq <= about 1.5 generates excess pore pressure."},
        "liquefied_side_resistance": {"type": "string", "required": False, "default": "half_pre_earthquake", "description": "Unit side resistance of the liquefied layers after reconsolidation, as drag above the neutral plane and resistance below it, for the headline numbers; every option that can be evaluated is reported in side_resistance_comparison whatever is chosen. 'half_pre_earthquake' (default) = 50 percent of the pre-earthquake (static) side resistance, GEC-10 10.7 p. 10-52 (Rollins et al. 2018); 'zero' = none, GEC-10 p. 10-51 and UFC 3-220-20 Table 6-49 (the state while liquefied); 'residual_strength' = su_r, the lower bound of GEC-10 p. 10-51 and GEC-12 7.4.2.2 (needs su_r or N1_60 for each layer; never more than the static value); 'ratio' = liquefied_side_resistance_ratio x the static value. The other layers keep their static side resistance.", "allowed_values": list(LIQUEFIED_SIDE_RESISTANCE)},
        "liquefied_side_resistance_ratio": {"type": "float", "required": False, "description": "Fraction (0-1) of the pre-earthquake unit side resistance the liquefied layers carry. Required when liquefied_side_resistance='ratio'; given with another option, it adds a 'ratio' row to side_resistance_comparison."},
    },
    "returns": {
        "post_liquefaction_basis": "What was done and on what basis (GEC-10 10.7, GEC-12 7.4.2.2 quoted). Report it.",
        "warnings": "What the engineer must know: a toe in or on a liquefied layer, a liquefied zone below the neutral plane before the earthquake with the methods whose drag force rises, readings left out. Report every one.",
        "methods_range_after": "The spread of neutral plane and drag force over the neutral-plane methods after liquefaction. Report it.",
        "neutral_plane_before_m / neutral_plane_after_m": "Neutral plane (m) of the headline method (neutral_plane_method) before and after.",
        "drag_force_before_kN / drag_force_after_kN": "Drag force Q_np - Q_dead (kN; UFC 3-220-20 Eq 6-80) before and after.",
        "max_pile_load_before_kN / max_pile_load_after_kN": "Q_np, the maximum axial load (kN).",
        "pile_settlement_before_mm / pile_settlement_after_mm": "Pile head settlement (mm).",
        "reconsolidation_settlement_mm": "Reconsolidation settlement of all the liquefied layers (mm): what the ground surface settles by.",
        "liquefied_zones": "Each contiguous liquefied zone: top, bottom, strain, settlement, side resistance before and after (kN), its position against the headline neutral plane before the earthquake, and below_the_plane_before_of (the methods whose plane before lies above it). Present it as a table.",
        "side_resistance_comparison": "Each side-resistance option (headline method): name, source, assumptions, applies, used, neutral plane, drag force, pile settlement. Present it as a table.",
        "by_method": "Every neutral-plane method before and after: neutral plane, drag force, pile settlement in each state. Present it as a table.",
        "limit_states_note": "The limit states GEC-10 10.7 checks the drag at (structural strength and service; AASHTO's Extreme Event I with load factor 1.0 noted). Report it.",
        "structural_ok / structural_demand_kN": "1.25 Q_dead + 1.10 DF after liquefaction <= structural_capacity (when given).",
        "choices": "Each choice the run made, given or default, with its basis.",
        "before": "The static analysis's headline: neutral plane, method, basis, drag force, warnings.",
        "after": "The full downdrag_analysis result after liquefaction (neutral_plane_comparison, profiles z_m / axial_load_kN / soil_settlement_mm / unit_skin_friction_kPa, below_toe).",
    },
}

# When each default applies (GM-119): the pile's unit weight only when its
# own weight is in the load; the FS_liq threshold only for a liquefaction
# result.
declare_applies_when(METHOD_INFO, {
    "downdrag_analysis": {
        "pile_unit_weight": {"include_pile_weight": [True]}},
    "post_liquefaction_downdrag": {
        "pile_unit_weight": {"include_pile_weight": [True]},
        "fs_liq_threshold": {"liquefaction_result": "set"}}})


# What a left-out input with no default stands for (BACKLOG GM-165):
# the pile's section from its diameter, the toe factor and area, and
# the CGPR equivalent footing.
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
_SECTION_CASES = {
    "pile_perimeter": ("pi D, from pile_diameter",
                       "the perimeter of a circular section"),
    "pile_area": ("pi D^2 / 4, from pile_diameter",
                  "the solid circle; give the steel area for an H or "
                  "pipe pile"),
    "Nt": ("the ultimate value from the toe layer",
           "Nt from the toe layer's friction angle (Fellenius 1991; "
           "GEC-12 Table 7-9), 9 in clay: the ultimate, not the "
           "mobilized, toe resistance")}
declare_when_absent(METHOD_INFO, {
    "downdrag_analysis": dict(_SECTION_CASES),
    "post_liquefaction_downdrag": dict(_SECTION_CASES),
    "poulos_downdrag": {"toe_area": "pile_area"},
    "fellenius_cgpr56": {"toe_area": "pile_area",
                         "eq_footing_length": "eq_footing_width",
                         "eq_footing_load": "Q_static"},
    "pileneg_downdrag": {
        "toe_area": "pile_area",
        "trial_depths": ("a dense grid over the pile length",
                         "neutral-plane trial depths evenly spaced from the "
                         "head to the toe")},
    "downdrag_method_comparison": {"toe_area": "pile_area",
                                   "eq_footing_length": "eq_footing_width",
                                   "eq_footing_load": "Q_static"},
})

"""Downdrag adapter — Fellenius neutral plane (UFC 3-220-20) plus the
CGPR #56 method family (Greenfield & Filz 2009): Endo, Poulos, the report's
Fellenius formulation, PILENEG, pile-group methods, and a comparison runner."""

from downdrag import (
    DowndragSoilLayer, DowndragSoilProfile, DowndragAnalysis,
    endo_method, poulos_method, fellenius_method_cgpr56, pileneg_procedure,
    rigid_block_method, drag_load_reduction_method,
    downdrag_method_comparison,
)
from funhouse_agent.adapters import (
    clean_result, reject_unknown_params, require_keys, require_params,
)

_SOIL_TYPES = ("cohesionless", "cohesive")

# Every top-level parameter _run_downdrag_analysis consumes.
_VALID_PARAMS = (
    "layers", "gwt_depth", "pile_length", "pile_diameter", "pile_perimeter",
    "pile_area", "pile_E", "pile_unit_weight", "Q_dead", "structural_capacity",
    "fill_thickness", "fill_unit_weight", "gw_drawdown", "Nt", "n_sublayers",
)


def _run_downdrag_analysis(params):
    reject_unknown_params(params, _VALID_PARAMS, method="downdrag_analysis")
    require_params(params, ["layers", "pile_length", "pile_diameter"],
                   method="downdrag_analysis", valid=_VALID_PARAMS)
    layers = []
    for l in params["layers"]:
        require_keys(l, ["thickness", "soil_type", "unit_weight"], method="downdrag_analysis")
        if l["soil_type"] not in _SOIL_TYPES:
            raise ValueError(
                f"downdrag_analysis: layer soil_type must be one of {list(_SOIL_TYPES)} "
                f"(got '{l['soil_type']}'). Mark settling layers with settling=True."
            )
        layers.append(DowndragSoilLayer(
            thickness=l["thickness"], soil_type=l["soil_type"], unit_weight=l["unit_weight"],
            phi=l.get("phi", 0.0), cu=l.get("cu", 0.0), beta=l.get("beta"), alpha=l.get("alpha"),
            Cc=l.get("Cc", 0.0), Cr=l.get("Cr", 0.0), e0=l.get("e0", 0.0),
            C_ec=l.get("C_ec"), C_er=l.get("C_er"), sigma_p=l.get("sigma_p"),
            E_s=l.get("E_s"), nu_s=l.get("nu_s", 0.3),
            settling=l.get("settling", False), description=l.get("description", ""),
        ))
    soil = DowndragSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth", 0.0))
    analysis = DowndragAnalysis(
        soil=soil, pile_length=params["pile_length"], pile_diameter=params["pile_diameter"],
        pile_perimeter=params.get("pile_perimeter"), pile_area=params.get("pile_area"),
        pile_E=params.get("pile_E", 200e6), pile_unit_weight=params.get("pile_unit_weight", 24.0),
        Q_dead=params.get("Q_dead", 0.0), structural_capacity=params.get("structural_capacity"),
        fill_thickness=params.get("fill_thickness", 0.0), fill_unit_weight=params.get("fill_unit_weight", 19.0),
        gw_drawdown=params.get("gw_drawdown", 0.0), Nt=params.get("Nt"), n_sublayers=params.get("n_sublayers", 10),
    )
    return analysis.compute().to_dict()


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
    require_keys(cons, ["layers", "p0_profile", "dp_profile"], method=method,
                 item_label="consolidation")
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
}

METHOD_INFO = {
    "downdrag_analysis": {
        "category": "Downdrag",
        "brief": "Full downdrag (negative skin friction) analysis via Fellenius neutral plane.",
        "parameters": {
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_diameter": {"type": "float", "required": True, "description": "Pile diameter (m)."},
            "layers": {"type": "array", "required": True, "description": "Array of {thickness, soil_type, unit_weight, phi, cu, beta, Cc, e0, settling} dicts. soil_type must be 'cohesionless' or 'cohesive' (NOT 'sand'/'clay'/'settling_fill'). Mark settling layers with settling=True."},
            "Q_dead": {"type": "float", "required": False, "default": 0.0, "description": "Dead load at pile top (kN)."},
            "fill_thickness": {"type": "float", "required": False, "description": "Fill thickness causing downdrag (m)."},
            "fill_unit_weight": {"type": "float", "required": False, "default": 19.0, "description": "Fill unit weight (kN/m3)."},
            "gw_drawdown": {"type": "float", "required": False, "description": "Groundwater drawdown (m)."},
            "gwt_depth": {"type": "float", "required": False, "default": 0.0, "description": "Groundwater depth (m)."},
            "pile_E": {"type": "float", "required": False, "default": 200e6, "description": "Pile elastic modulus (kPa). Default is steel."},
            "pile_perimeter": {"type": "float", "required": False, "description": "Pile perimeter (m). Computed from diameter if omitted."},
            "pile_area": {"type": "float", "required": False, "description": "Pile cross-section area (m2). Computed from diameter if omitted."},
            "structural_capacity": {"type": "float", "required": False, "description": "Pile structural capacity (kN) for the max-load check."},
            "Nt": {"type": "float", "required": False, "description": "Toe bearing capacity coefficient override."},
        },
        "returns": {"neutral_plane_depth_m": "Neutral plane depth.", "dragload_kN": "Downdrag force on pile."},
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

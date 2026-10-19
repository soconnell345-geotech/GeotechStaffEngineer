"""Axial pile adapter — driven pile capacity (Nordlund/Tomlinson/Beta)."""

from dataclasses import replace

from axial_pile import PileSection, make_pipe_pile, make_concrete_pile, make_h_pile
from axial_pile import AxialSoilLayer, AxialSoilProfile, AxialPileAnalysis
from axial_pile.capacity import (
    COHESIVE_NT, INSIDE_FRICTION_RATIO, OPEN_PIPE_TOE_LIMIT,
    UPLIFT_SKIN_FRACTION, _TOE_STRESS_LIMIT,
)
from axial_pile.soil_profile import FIG_7_18_LAYER_CASES
from funhouse_agent.adapters import (
    apply_aliases, check_nested_list, declare_aliases, declare_applies_when,
    field_keys, reject_unknown_params, require_keys, require_params,
)

_PILE_TYPES = ("pipe_closed", "pipe_open", "concrete_square", "concrete_circular", "h_pile",
               "timber")

# A section given by its own properties (GM-144; the calc package took them
# before GM-141): area, perimeter and width, with the toe area where the
# methods need one the area does not give.
_CUSTOM_SECTION = ("pile_area", "pile_perimeter", "pile_tip_area",
                   "pile_width", "pile_depth", "pile_name")
# Pile-geometry params (incl. the accepted aliases width/thickness/diameter).
_PILE_PARAMS = ("pile_type", "diameter", "wall_thickness", "width",
                "thickness", "designation", "E") + _CUSTOM_SECTION
_SOIL_PARAMS = ("layers", "gwt_depth")
# The GEC-12 limits and judgments AxialPileAnalysis takes (GM-121), each
# with the module's own default.
_GEC12_OPTIONS = {
    "toe_stress_limit": _TOE_STRESS_LIMIT,
    "inside_friction_ratio": INSIDE_FRICTION_RATIO,
    "open_pipe_toe_limit": OPEN_PIPE_TOE_LIMIT,
    "open_pipe_method": "auto",
    "cohesive_Nt": COHESIVE_NT,
    "h_pile_toe_area": "box",
    "adhesion_curve": "auto",
    "adhesion_beyond_chart": "chart_end",
    "displaced_volume": None,
    # GM-144: the rest of GEC-12 chapter 7
    "adhesion_chart": "fig_7_17",
    "shrink_swell_depth": None,
    "beta_coefficients": "from_phi",
    "toe_phi_zone": "toe_layer",
}

# Keys a layer dict may carry: AxialSoilLayer's fields plus the 'toe_phi'
# alias. Anything else was read with .get and silently left out (GM-58).
_LAYER_KEYS = field_keys(AxialSoilLayer, "toe_phi")
# Words the near-miss matcher cannot place. Suggestions only, never renamed.
_LAYER_KEY_HINTS = {
    "phi": ("friction_angle",), "phiprime": ("friction_angle",),
    "c": ("cohesion",), "cu": ("cohesion",), "su": ("cohesion",),
    "gamma": ("unit_weight",),
}


def _build_pile(params):
    pile_type = params.get("pile_type", "pipe_closed")
    if any(params.get(k) is not None for k in _CUSTOM_SECTION):
        return _custom_pile(params, pile_type)
    if pile_type in ("pipe_closed", "pipe_open"):
        # 'width' is a common guess for the pipe diameter; 'thickness' for wall_thickness.
        p = apply_aliases(params, {"width": "diameter", "thickness": "wall_thickness"})
        require_params(p, ["diameter", "wall_thickness"],
                       method=f"pile_type '{pile_type}'",
                       valid=["diameter", "wall_thickness", "E"])
        return make_pipe_pile(diameter=p["diameter"], thickness=p["wall_thickness"],
                              closed_end=(pile_type == "pipe_closed"), E=p.get("E", 200e6))
    elif pile_type in ("concrete_square", "concrete_circular"):
        # 'diameter' is the natural name for a circular pile's width.
        p = apply_aliases(params, {"diameter": "width"})
        require_params(p, ["width"], method=f"pile_type '{pile_type}'",
                       valid=["width", "E"])
        shape = "square" if "square" in pile_type else "circular"
        return make_concrete_pile(width=p["width"], shape=shape, E=p.get("E", 25e6))
    elif pile_type == "h_pile":
        require_params(params, ["designation"], method="pile_type 'h_pile'",
                       valid=["designation", "E"])
        return make_h_pile(designation=params["designation"], E=params.get("E", 200e6))
    elif pile_type == "timber":
        raise ValueError(
            "pile_type 'timber' takes a custom section: pile_area, "
            "pile_perimeter and pile_width (and pile_tip_area if the toe "
            "differs from the section).")
    else:
        raise ValueError(f"Unknown pile_type '{pile_type}'. Allowed: {list(_PILE_TYPES)}.")


#: The section each pile_type's custom section is, for the GEC-12 charts.
_CUSTOM_TYPES = {"h_pile": "h_pile", "pipe_closed": "pipe_closed",
                 "concrete_square": "concrete",
                 "concrete_circular": "concrete", "timber": "timber"}


def _custom_pile(params, pile_type):
    """A section from its own properties (GM-144): pile_area (the section's
    area: an H-pile's or a closed pipe's steel, a solid pile's gross area),
    pile_perimeter (for the side resistance; the box perimeter of an
    H-pile), pile_width (b, for D/b and the width notes) and pile_tip_area
    (the toe area: the box area of an H-pile, the gross area of a closed
    pipe; a solid pile's is its area). pile_type still picks the GEC-12
    curves."""
    method = f"pile_type '{pile_type}' with a custom section"
    if pile_type == "pipe_open":
        raise ValueError(
            "An open-end pipe's section comes from diameter and "
            "wall_thickness (its plugged area and inner perimeter follow "
            "from them); a custom section (pile_area, pile_perimeter, ...) "
            "is for pipe_closed, concrete, timber and H-piles.")
    if pile_type not in _CUSTOM_TYPES:
        raise ValueError(f"Unknown pile_type '{pile_type}'. Allowed: {list(_PILE_TYPES)}.")
    named = [k for k in ("diameter", "wall_thickness", "width", "thickness",
                         "designation") if params.get(k) is not None]
    if named:
        raise ValueError(
            f"{method}: give the section by its properties (pile_area, "
            f"pile_perimeter, pile_width, pile_tip_area) or by {named}, not "
            f"both.")
    solid = pile_type in ("concrete_square", "concrete_circular", "timber")
    needed = ["pile_area", "pile_perimeter", "pile_width"]
    if not solid:
        # the toe area an H-pile's steel or a closed pipe's steel cannot give
        needed.append("pile_tip_area")
    require_params(params, needed, method=method,
                   valid=list(_CUSTOM_SECTION) + ["E"])
    tip_area = params.get("pile_tip_area")
    if tip_area is None:
        tip_area = params["pile_area"]
    for k, v in (("pile_area", params["pile_area"]),
                 ("pile_perimeter", params["pile_perimeter"]),
                 ("pile_width", params["pile_width"]),
                 ("pile_tip_area", tip_area)):
        if not v > 0:
            raise ValueError(f"{method}: {k} must be > 0, got {v}")
    kind = _CUSTOM_TYPES[pile_type]
    E = params.get("E")
    if E is None and kind != "timber":
        E = 25e6 if kind == "concrete" else 200e6
    return PileSection(
        name=params.get("pile_name") or f"Custom {kind.replace('_', ' ')}",
        pile_type=kind, area=params["pile_area"],
        perimeter=params["pile_perimeter"], tip_area=tip_area,
        width=params["pile_width"], depth=params.get("pile_depth"), E=E,
        outer_diameter=(params["pile_width"] if kind == "pipe_closed"
                        else None))


def _build_soil_profile(params, *, method):
    require_params(params, ["layers"], method=method)
    check_nested_list(params["layers"], _LAYER_KEYS, method=method,
                      label="layers", hints=_LAYER_KEY_HINTS)
    layers = []
    for l in params["layers"]:
        require_keys(l, ["thickness", "unit_weight"], method=method)
        # 'toe_friction_angle' (or its alias 'toe_phi') is an OPTIONAL separate
        # end-bearing phi for a layer; left unset it falls back to friction_angle.
        toe_phi = l.get("toe_friction_angle", l.get("toe_phi"))
        layers.append(AxialSoilLayer(
            thickness=l["thickness"], soil_type=l.get("soil_type", "cohesionless"),
            unit_weight=l["unit_weight"], friction_angle=l.get("friction_angle"),
            cohesion=l.get("cohesion"), delta_phi_ratio=l.get("delta_phi_ratio"),
            toe_friction_angle=toe_phi,
            description=l.get("description", ""),
            beta=l.get("beta"), Nt=l.get("Nt"),
            fig_7_18_case=l.get("fig_7_18_case"),
            soil_class=l.get("soil_class"),
        ))
    return AxialSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth"))


def run_axial_pile_capacity(params, *, method="axial_pile_capacity"):
    """(result, analysis) of ``axial_pile_capacity``: the tool's own code
    path, also run by ``calc_package.axial_pile_package`` (GM-141).
    ``method`` names the caller in error messages."""
    reject_unknown_params(
        params,
        _PILE_PARAMS + _SOIL_PARAMS + (
            "pile_length", "method", "factor_of_safety", "include_uplift",
            "cohesive_phi", "uplift_skin_fraction", "pile_weight", "head_depth")
        + tuple(_GEC12_OPTIONS),
        method=method)
    require_params(params, ["pile_type"], method=method)
    pile = _build_pile(params)
    soil = _build_soil_profile(params, method=method)
    require_params(params, ["pile_length"], method=method)
    analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=params["pile_length"],
                                  method=params.get("method", "auto"),
                                  factor_of_safety=params.get("factor_of_safety", 2.5),
                                  include_uplift=params.get("include_uplift", False),
                                  cohesive_phi=params.get("cohesive_phi", 25.0),
                                  uplift_skin_fraction=params.get(
                                      "uplift_skin_fraction", UPLIFT_SKIN_FRACTION),
                                  pile_weight=params.get("pile_weight"),
                                  head_depth=params.get("head_depth", 0.0),
                                  **{k: params.get(k, v)
                                     for k, v in _GEC12_OPTIONS.items()})
    result = analysis.compute()
    return result, analysis


def _run_axial_pile_capacity(params):
    return run_axial_pile_capacity(params)[0].to_dict()


def _run_capacity_vs_depth(params):
    reject_unknown_params(
        params,
        _PILE_PARAMS + _SOIL_PARAMS + (
            "pile_length", "method", "depth_min", "depth_max", "n_points",
            "head_depth", "cohesive_phi") + tuple(_GEC12_OPTIONS),
        method="capacity_vs_depth")
    require_params(params, ["pile_type"], method="capacity_vs_depth")
    pile = _build_pile(params)
    soil = _build_soil_profile(params, method="capacity_vs_depth")
    # GM-144: the same GEC-12 options as axial_pile_capacity
    analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=params.get("pile_length", soil.total_thickness),
                                  method=params.get("method", "auto"),
                                  head_depth=params.get("head_depth", 0.0),
                                  cohesive_phi=params.get("cohesive_phi", 25.0),
                                  **{k: params.get(k, v)
                                     for k, v in _GEC12_OPTIONS.items()})
    curve = analysis.capacity_vs_depth(depth_min=params.get("depth_min", 3.0),
                                        depth_max=params.get("depth_max"), n_points=params.get("n_points", 20))
    out = {"capacity_vs_depth": curve}
    if curve:
        # what the deepest run chose (the toe's choices change with depth;
        # each depth's warnings ride on its row)
        deepest = params.get("depth_max")
        if deepest is None:
            deepest = min(analysis.pile_length, soil.total_thickness)
        out["choices"] = replace(analysis,
                                 pile_length=float(deepest)).compute().choices
    return out


def _run_make_pile_section(params):
    reject_unknown_params(params, _PILE_PARAMS, method="make_pile_section")
    pile = _build_pile(params)
    return {
        "name": pile.name,
        "pile_type": pile.pile_type,
        "area_m2": round(pile.area, 6),
        "perimeter_m": round(pile.perimeter, 4),
        "tip_area_m2": round(pile.tip_area, 6),
        "width_m": round(pile.width, 4),
    }


METHOD_REGISTRY = {
    "axial_pile_capacity": _run_axial_pile_capacity,
    "capacity_vs_depth": _run_capacity_vs_depth,
    "make_pile_section": _run_make_pile_section,
}

# The custom section's inputs (GM-144), the same on the three methods.
_SECTION_INFO = {
    "pile_area": {"type": "float", "required": False, "description": "Custom section (m2): the cross-section area, the steel of an H-pile or a closed pipe, the gross area of a concrete or timber pile (it gives the Nordlund displaced volume of an H-pile and of a solid pile). With pile_perimeter and pile_width it replaces diameter, wall_thickness, width or designation; pile_type still picks the GEC-12 curves (not for pipe_open: its section comes from diameter and wall_thickness)."},
    "pile_perimeter": {"type": "float", "required": False, "description": "Custom section: the perimeter for the side resistance (m); the box perimeter of an H-pile (GEC-12 Steps 6 and p. 247)."},
    "pile_tip_area": {"type": "float", "required": False, "description": "Custom section: the toe area (m2). Required for h_pile (its box area; h_pile_toe_area 'steel' takes pile_area) and pipe_closed (the gross area); a concrete or timber pile's is its pile_area when not given."},
    "pile_width": {"type": "float", "required": False, "description": "Custom section: the width or diameter b (m), an H-pile's flange width; for D/b and the width notes."},
    "pile_depth": {"type": "float", "required": False, "description": "Custom section: the section's depth (m), an H-pile's; reported only."},
    "pile_name": {"type": "str", "required": False, "description": "Custom section: its name, e.g. 'HP16x101'."},
}

# The GEC-12 options GM-144 added, the same on both capacity methods.
_GM144_INFO = {
    "adhesion_chart": {"type": "str", "required": False, "default": "fig_7_17", "allowed_values": ["fig_7_17", "fig_7_18", "smaller"], "description": "alpha-method chart (GEC-12 p. 246): 'fig_7_17' (Tomlinson's Ca, recommended generally), 'fig_7_18' (Tomlinson's alpha for each clay layer's stratigraphy case: sand or sandy gravel over stiff clay, soft clay over stiff clay, stiff clay alone) or 'smaller' (the smaller Ca of the two in each layer, which p. 246 advises the inexperienced user to take). A layer's case is its fig_7_18_case, else from the profile: cohesive layers under a cohesionless layer the sand case, from the ground surface the stiff-clay case; the soft-clay case only where a layer gives it."},
    "shrink_swell_depth": {"type": "float", "required": False, "description": "Depth (m below the ground surface) of a shrink-swell zone: no shaft resistance in cohesive soil above it (GEC-12 p. 247). Default none."},
    "beta_coefficients": {"type": "str", "required": False, "default": "from_phi", "allowed_values": ["from_phi", "table_7_9"], "description": "method='beta': beta and Nt where a layer gives none. 'from_phi': beta = (1 - sin phi') tan phi' (Fellenius 1991, about 0.27-0.30), Nt within Table 7-9's ranges by phi'; 'table_7_9': GEC-12 Table 7-9 by the layer's soil_class (clay 0.15-0.35, silt 0.25-0.50, sand 0.30-0.90, gravel 0.35-0.80), linear in phi' across the class's phi' range (Steps 2b/5b). A clay toe keeps cohesive_Nt."},
    "toe_phi_zone": {"type": "str", "required": False, "default": "toe_layer", "allowed_values": ["toe_layer", "average_3b"], "description": "Nordlund toe: the phi near the toe. 'toe_layer' (the toe layer's toe_friction_angle or friction_angle) or 'average_3b' (averaged over the cohesionless soil from the toe to 3b below, GEC-12 Step 7c's zone)."},
}

_LAYERS_DOC = ("Array of soil-layer dicts (measured from the GROUND SURFACE): {thickness (m), soil_type, unit_weight (kN/m3), friction_angle (deg, for cohesionless), cohesion (kPa, for cohesive), toe_friction_angle (deg, OPTIONAL: separate end-bearing phi if the pile tip is in this layer; defaults to friction_angle), delta_phi_ratio (OPTIONAL Nordlund delta/phi; default from GEC-12 Fig. 7-9 by pile type and displaced volume), beta and Nt (OPTIONAL, beta method: the layer's beta and toe Nt from local experience, GEC-12 Steps 2a/5a), fig_7_18_case (OPTIONAL, a cohesive layer: "
               + ", ".join(f"'{c}'" for c in FIG_7_18_LAYER_CASES)
               + "; GEC-12 Fig. 7-18's case, which starts a clay stratum at the layer's top), soil_class (OPTIONAL: 'clay', 'silt', 'sand' or 'gravel', GEC-12 Table 7-9's soil type for beta_coefficients 'table_7_9'; default clay or sand)}. soil_type must be 'cohesionless' or 'cohesive' (NOT 'sand'/'clay').")

METHOD_INFO = {
    "axial_pile_capacity": {
        "category": "Axial Pile",
        "brief": "Full driven pile axial capacity analysis (Nordlund/Tomlinson/Beta).",
        "parameters": {
            "pile_type": {"type": "str", "required": True, "allowed_values": list(_PILE_TYPES), "description": "Pile type. pipe_closed/pipe_open require diameter + wall_thickness; concrete_square/concrete_circular require width (side dimension OR diameter); h_pile requires designation; or a custom section (pile_area, pile_perimeter, pile_width, pile_tip_area) for any but pipe_open, timber always."},
            "diameter": {"type": "float", "required": False, "description": "Pile diameter (m) for pipe piles."},
            "wall_thickness": {"type": "float", "required": False, "description": "Pipe wall thickness (m)."},
            "width": {"type": "float", "required": False, "description": "Side dimension (m) for concrete piles."},
            "designation": {"type": "str", "required": False, "description": "H-pile designation (e.g., 'HP14x117')."},
            "E": {"type": "float", "required": False, "description": "Pile elastic modulus (kPa). Defaults: steel 200e6, concrete 25e6."},
            "pile_length": {"type": "float", "required": True, "description": "Embedded pile length from the pile head to the tip (m)."},
            "layers": {"type": "array", "required": True, "description": _LAYERS_DOC},

            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth below ground surface (m)."},
            "head_depth": {"type": "float", "required": False, "default": 0.0, "description": "Depth of the pile head below the ground surface (m), e.g. the bottom of a footing 1.5 m down. Layers above the head contribute no skin friction; overburden above the head still raises effective stress at the tip. Default 0 = head at the ground surface."},
            "factor_of_safety": {"type": "float", "required": False, "default": 2.5, "description": "Factor of safety."},
            "method": {"type": "str", "required": False, "default": "auto", "allowed_values": ["auto", "beta"], "description": "auto = Nordlund (sand) / Tomlinson (clay); beta = effective-stress method for all layers."},
            "include_uplift": {"type": "bool", "required": False, "default": False, "description": "Also report uplift (tension) capacity."},
            "cohesive_phi": {"type": "float", "required": False, "default": 25.0, "description": "FALLBACK drained phi' (deg) for cohesive layers under method='beta' (skin AND tip); a per-layer friction_angle on a cohesive layer wins over this global."},
            "uplift_skin_fraction": {"type": "float", "required": False, "default": UPLIFT_SKIN_FRACTION, "description": "Fraction of the outside side resistance taken in uplift (with include_uplift). Default 1.0: GEC-12 Eq. 7-38 takes the whole shaft resistance and puts the uplift reduction in the resistance factor; 0.7 is the customary share in free-draining soils GEC-12 cites."},
            "pile_weight": {"type": "float", "required": False, "description": "Pile self-weight (kN) added to uplift capacity (with include_uplift)."},
            "toe_stress_limit": {"type": "float", "required": False, "default": _TOE_STRESS_LIMIT, "description": "Limit on sigma'p at the toe for the Nordlund toe resistance (kPa). Default 143.6 kPa = 3 ksf, GEC-12 Step 8; null for no limit (GEC-12's worked example did not limit it)."},
            "inside_friction_ratio": {"type": "float", "required": False, "default": INSIDE_FRICTION_RATIO, "description": "Open-end pipe, unplugged case: interior over exterior unit side resistance. Default 1/3, the low end of GEC-12's 'typically on the order of 1/3 to 1/2' (Sec. 7.10.7)."},
            "open_pipe_toe_limit": {"type": "float", "required": False, "default": OPEN_PIPE_TOE_LIMIT, "description": "Open-end pipe in cohesionless soil: limiting unit toe resistance (kPa). Default 5027 kPa = 105 ksf (Tomlinson, GEC-12 Sec. 7.10.7); null for none."},
            "open_pipe_method": {"type": "str", "required": False, "default": "auto", "allowed_values": ["auto", "paikowsky_whitman", "tomlinson"], "description": "Open-end pipe: 'auto' takes Tomlinson's Eq. 7-90 for a pipe under 24 in with the toe in clay and over half its length in clay (GEC-12 p. 407), else the lesser of the plugged and unplugged cases (Eqs. 7-88/7-89, 'paikowsky_whitman')."},
            "cohesive_Nt": {"type": "float", "required": False, "default": COHESIVE_NT, "description": "method='beta': Nt of a cohesive toe layer. Default 3 (GEC-12: a relatively low Nt in clays unless local correlations suggest higher; Table 7-9 clay 3-30). A layer's own Nt wins."},
            "h_pile_toe_area": {"type": "str", "required": False, "default": "box", "allowed_values": ["box", "steel"], "description": "H-pile toe area: 'box' (depth x flange width, GEC-12 Sec. 7.10.7) or 'steel'. GEC-12 Step 9 leaves it to past experience and local load tests."},
            "adhesion_curve": {"type": "str", "required": False, "default": "auto", "allowed_values": ["auto", "concrete_timber_corrugated", "smooth_steel"], "description": "alpha-method: GEC-12 Fig. 7-17 curves. 'auto' = smooth steel for pipe piles, the concrete/timber/corrugated curve for H-piles (GEC-12 p. 247) and concrete or timber piles."},
            "adhesion_beyond_chart": {"type": "str", "required": False, "default": "chart_end", "allowed_values": ["chart_end", "api"], "description": "alpha-method where su exceeds Fig. 7-17: 'chart_end' (Ca held at the curve's end, as GEC-12's worked example, Vol 3 Table D-100) or 'api' (alpha from API RP 2A, which GEC-12 p. 247 recommends for highly overconsolidated clays; Ca jumps at the chart's end). Within the chart Ca is at most su."},
            "displaced_volume": {"type": "float", "required": False, "description": "Nordlund displaced volume V per unit length (m3/m). Default: the steel area of an H-pile, the gross area of other piles (an open pipe as a closed pipe; its steel area, if it drives unplugged, is below GEC-12 Fig. 7-9)."},
        },
        "returns": {"Q_ultimate_kN": "Ultimate capacity.", "Q_skin_kN": "Skin friction.", "Q_tip_kN": "Tip resistance.", "Q_allowable_kN": "Allowable capacity.", "choices": "What the run chose and how: what, chosen, how (given, default or module), basis, alternatives."},
    },
    "capacity_vs_depth": {
        "category": "Axial Pile",
        "brief": "Capacity vs depth curve for pile length optimization.",
        "parameters": {
            "pile_type": {"type": "str", "required": True, "allowed_values": list(_PILE_TYPES), "description": "Pile type (see axial_pile_capacity for geometry params per type, and a custom section)."},
            "diameter": {"type": "float", "required": False, "description": "Pile diameter (m) for pipe piles."},
            "wall_thickness": {"type": "float", "required": False, "description": "Pipe wall thickness (m)."},
            "width": {"type": "float", "required": False, "description": "Side dimension (m) for concrete piles."},
            "designation": {"type": "str", "required": False, "description": "H-pile designation (e.g., 'HP14x117')."},
            "E": {"type": "float", "required": False, "description": "Pile elastic modulus (kPa). Defaults: steel 200e6, concrete 25e6."},
            "layers": {"type": "array", "required": True, "description": _LAYERS_DOC},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m)."},
            "head_depth": {"type": "float", "required": False, "default": 0.0, "description": "Depth of the pile head below the ground surface (m). Default 0 = head at the ground surface (see axial_pile_capacity)."},
            "pile_length": {"type": "float", "required": False, "description": "Reference pile length (m). Defaults to total soil thickness."},
            "method": {"type": "str", "required": False, "default": "auto", "allowed_values": ["auto", "beta"], "description": "Capacity method (see axial_pile_capacity)."},
            "depth_min": {"type": "float", "required": False, "default": 3.0, "description": "Min depth (m)."},
            "depth_max": {"type": "float", "required": False, "description": "Max depth (m)."},
            "n_points": {"type": "int", "required": False, "default": 20, "description": "Points on the curve."},
        },
        "returns": {"capacity_vs_depth": "List of {depth, Q_ult, Q_skin, Q_tip} dicts.", "choices": "What the run at the deepest point chose and how (as axial_pile_capacity's choices)."},
    },
    "make_pile_section": {
        "category": "Axial Pile",
        "brief": "Create a pile section and return its geometric properties.",
        "parameters": {
            "pile_type": {"type": "str", "required": True, "allowed_values": list(_PILE_TYPES), "description": "Pile type. pipe_closed/pipe_open require diameter + wall_thickness; concrete_square/concrete_circular require width (side dimension OR diameter); h_pile requires designation; or a custom section (pile_area, pile_perimeter, pile_width, pile_tip_area) for any but pipe_open, timber always."},
            "diameter": {"type": "float", "required": False, "description": "Pile diameter (m) for pipe piles."},
            "wall_thickness": {"type": "float", "required": False, "description": "Pipe wall thickness (m)."},
            "width": {"type": "float", "required": False, "description": "Side dimension (m) for concrete piles."},
            "designation": {"type": "str", "required": False, "description": "H-pile designation (e.g., 'HP14x117')."},
            "E": {"type": "float", "required": False, "description": "Pile elastic modulus (kPa). Defaults: steel 200e6, concrete 25e6."},
        },
        "returns": {"name": "Pile section name.", "area_m2": "Cross-sectional area.", "perimeter_m": "Perimeter.", "tip_area_m2": "Tip area.", "width_m": "Width."},
    },
}

# GM-144: every method takes a custom section; both capacity methods take
# the GEC-12 options, capacity_vs_depth the ones axial_pile_capacity has
# had since GM-121 (and the beta method's cohesive_phi) as well.
for _m in METHOD_INFO:
    METHOD_INFO[_m]["parameters"].update(_SECTION_INFO)
METHOD_INFO["axial_pile_capacity"]["parameters"].update(_GM144_INFO)
METHOD_INFO["capacity_vs_depth"]["parameters"].update({
    k: METHOD_INFO["axial_pile_capacity"]["parameters"][k]
    for k in ("cohesive_phi",) + tuple(_GEC12_OPTIONS)})

# The pile geometry's other spellings _build_pile reads (GM-110).
declare_aliases(METHOD_INFO, {
    m: {"thickness": "wall_thickness"} for m in METHOD_INFO})

# When each default applies (GM-119): the fallback phi' of clay to the beta
# method, the uplift fraction to an uplift run.
_APPLIES_WHEN = {
    "cohesive_phi": {"method": ["beta"]},
    "inside_friction_ratio": {"pile_type": ["pipe_open"]},
    "open_pipe_method": {"pile_type": ["pipe_open"]},
    "h_pile_toe_area": {"pile_type": ["h_pile"]},
    "beta_coefficients": {"method": ["beta"]},
    # These apply by the soil (a sand or clay toe, a clay layer, su past
    # Fig. 7-17), which no other parameter says: listed when the result's
    # choices report them as defaults.
    "cohesive_Nt": False,
    "toe_stress_limit": False,
    "adhesion_curve": False,
    "adhesion_beyond_chart": False,
    "open_pipe_toe_limit": False,
    "adhesion_chart": False,
    "toe_phi_zone": False,
}
declare_applies_when(METHOD_INFO, {
    "axial_pile_capacity": {
        **_APPLIES_WHEN,
        "uplift_skin_fraction": {"include_uplift": [True]}},
    "capacity_vs_depth": _APPLIES_WHEN,
})

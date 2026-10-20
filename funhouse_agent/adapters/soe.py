"""Support of excavation adapter — braced/cantilever, stability, anchors."""

from funhouse_agent.adapters import (
    check_nested_list, declare_applies_when, field_keys,
    reject_unknown_params, require_keys, require_params,
)
from soe import (
    ExcavationGeometry, SOEWallLayer, SupportLevel,
    select_apparent_pressure, analyze_braced_excavation, analyze_cantilever_excavation,
    select_hp_section, select_sheet_pile, select_w_section,
    check_basal_heave_terzaghi, check_basal_heave_bjerrum_eide,
    check_basal_heave_caltrans, check_bottom_blowout, check_piping,
    design_ground_anchor, fhwa_apparent_pressure_anchored_wall,
)
from soe.anchor_design import GROUNDS, ROCK_TYPES, list_bond_stress_types


# Keys a layer / support dict may carry: the dataclass fields they build.
# Anything else was read with .get and silently left out, e.g. a layer's
# 'phi' ran at the 30 degree default (GM-58).
_LAYER_KEYS = field_keys(SOEWallLayer)
_SUPPORT_KEYS = field_keys(SupportLevel)
# Words the near-miss matcher cannot place. Suggestions only, never renamed.
_LAYER_KEY_HINTS = {
    "phi": ("friction_angle",), "phiprime": ("friction_angle",),
    "c": ("cohesion",), "cu": ("cohesion",), "su": ("cohesion",),
    "gamma": ("unit_weight",),
}
_SUPPORT_KEY_HINTS = {
    "type": ("support_type",), "angle": ("angle_deg",),
    "inclination": ("angle_deg",), "preload": ("preload_kN_per_m",),
}


def _soil_type(layer, H):
    """Accept 'clay' (field note 2026-09-15: the brief used to offer it and the
    module rejected it) by the Terzaghi-Peck stability number N = gamma*H/cu:
    N > 4 soft_clay, otherwise stiff_clay. A layer the analysis does not read
    it on (a cantilever's, or one below the excavation base) and that gives
    none is built as 'sand', which nothing reads."""
    t = str(layer.get("soil_type", "sand")).strip().lower()
    if t == "clay":
        cu = float(layer.get("cohesion", 0.0) or 0.0)
        gamma = float(layer["unit_weight"])
        return "soft_clay" if cu <= 0 or gamma * float(H) / cu > 4.0 else "stiff_clay"
    return t


def _require_soil_types(layers, H, *, method):
    """Refuse a layer above the excavation base with no soil_type (GM-182):
    it picks the apparent-pressure envelope, sand or clay, by the thickness
    of each over the excavation depth, and it used to default to sand
    unstated."""
    top = 0.0
    for i, l in enumerate(layers):
        if top >= float(H):
            return
        if l.get("soil_type") in (None, ""):
            raise ValueError(
                f"{method}: layers[{i}] (top {top:g} m) has no soil_type. "
                f"Each layer above the excavation base ({float(H):g} m) "
                "needs one: it picks the Terzaghi and Peck (1967) apparent "
                "pressure envelope, the one for sands (0.65 Ka gamma H) "
                "where sand is the thicker over the excavation depth, else "
                "the one for clays by N = gamma H / cu (soft to medium clay "
                "N > 4, stiff fissured clay N <= 4). Give 'sand', or "
                "'clay' ('soft_clay' or 'stiff_clay').")
        top += float(l.get("thickness") or 0.0)


def _build_layers(params, *, method, envelope=False):
    """The layers of a wall analysis. ``envelope``: the analysis picks the
    apparent-pressure envelope from them, so each layer above the
    excavation base must give its soil_type (GM-182)."""
    require_params(params, ["layers"], method=method)
    check_nested_list(params["layers"], _LAYER_KEYS, method=method,
                      label="layers", hints=_LAYER_KEY_HINTS)
    H = params.get("excavation_depth", 0.0) or 0.0
    layers = []
    for l in params["layers"]:
        require_keys(l, ["thickness", "unit_weight"], method=method)
    if envelope:
        _require_soil_types(params["layers"], H, method=method)
    for l in params["layers"]:
        layers.append(SOEWallLayer(
            thickness=l["thickness"], unit_weight=l["unit_weight"],
            friction_angle=l.get("friction_angle", 30.0), cohesion=l.get("cohesion", 0.0),
            soil_type=_soil_type(l, H), description=l.get("description", ""),
        ))
    return layers


def _build_geometry(params, *, method, envelope=False):
    require_params(params, ["excavation_depth"], method=method)
    layers = _build_layers(params, method=method, envelope=envelope)
    supports = []
    check_nested_list(params.get("supports"), _SUPPORT_KEYS, method=method,
                      label="supports", hints=_SUPPORT_KEY_HINTS)
    for s in params.get("supports", []):
        require_keys(s, ["depth"], method=method, item_label="supports[]")
        supports.append(SupportLevel(
            depth=s["depth"], support_type=s.get("support_type", "strut"),
            spacing=s.get("spacing", 3.0), angle_deg=s.get("angle_deg", 0.0),
            preload_kN_per_m=s.get("preload_kN_per_m", 0.0),
        ))
    return ExcavationGeometry(
        excavation_depth=params["excavation_depth"], soil_layers=layers,
        support_levels=supports, surcharge=params.get("surcharge", 10.0),
        gwt_depth=params.get("gwt_depth"),
        gwt_depth_excavation=params.get("gwt_depth_excavation"),
    )


# Geometry params consumed by _build_geometry (shared by both wall analyses).
# No excavation_width: nothing in soe/ reads it, so it is refused, not
# dropped (GM-110).
_PASSIVE_THEORY = {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "log_spiral", "caquot_kerisel"], "description": "The passive coefficient below the excavation: 'rankine' (default) = Rankine's Kp, no wall friction, as the Caltrans Trenching and Shoring Manual's examples take it where the wall friction is zero (pp. 7-18, 7-27), with its note that Rankine 'tends to underestimate the passive earth pressure' and Sec. 4-6 (p. 4-28): Rankine 'should not be used ... because it does not account for wall friction'; 'log_spiral' (or 'caquot_kerisel', the same) = the log-spiral Kp of log_spiral_source at passive_wall_friction_deg in each layer below the excavation. Choices give the other theory's Kp and embedment."}
_LOG_SPIRAL_SOURCE = {"type": "str", "required": False, "default": "ufc_3_220_20", "allowed_values": ["ufc_3_220_20", "caquot_kerisel_1948", "dm7_2_eq_4_12", "dm7_2_fig_4_12", "dm7_2_fig_4_13"], "description": "With passive_theory 'log_spiral', the printed source of Kp, as earth_pressure_coefficient takes it: 'ufc_3_220_20' (default) = UFC 3-220-20 Eq. 4-12 (p. 191), its log-spiral Kp for a vertical face and level ground at any wall friction; 'caquot_kerisel_1948' = the Caltrans Trenching and Shoring Manual Fig. 4-20 chart (its Example 8-1); 'dm7_2_fig_4_12' / 'dm7_2_fig_4_13' = UFC's charts, drawn for delta/phi' 0.66 only."}
_PASSIVE_WALL_FRICTION = {"type": "float", "required": False, "description": "With passive_theory 'log_spiral', the wall friction delta (deg) in every layer below the excavation, at most its phi': the engineer's call (UFC 3-220-20 Table 4-1, p. 184; Caltrans Example 8-1 takes 15 deg on phi 30). Left out: the source's own (delta/phi' = 0.66, UFC p. 191, or delta = phi' with 'caquot_kerisel_1948') in a layer without cohesion, none in a layer with cohesion (UFC p. 238: wall friction 'only considered for passive pressure in granular soils')."}
_PASSIVE_CHOICES = "How the passive pressure below the excavation was taken: passive_theory with each layer's Kp and the embedment, the other theory's as the alternative, the Caltrans basis; with 'log_spiral' its source (with the other sources' Kp) and each layer's wall friction. Tell the user."

_GEOM_PARAMS = ("excavation_depth", "layers", "supports", "surcharge",
                "gwt_depth", "gwt_depth_excavation")

# The passive coefficient below the excavation (GM-242), shared by both wall
# analyses; 'caquot_kerisel' is what earth_pressure_coefficient calls the
# log-spiral theory.
_PASSIVE_PARAMS = ("passive_theory", "log_spiral_source",
                   "passive_wall_friction_deg")
_PASSIVE_THEORY_ALIASES = {"caquot_kerisel": "log_spiral",
                           "logspiral": "log_spiral",
                           "log-spiral": "log_spiral"}


def _passive(p):
    """The passive theory's inputs as the analyses take them (GM-242)."""
    theory = p.get("passive_theory", "rankine")
    return {"passive_theory": _PASSIVE_THEORY_ALIASES.get(theory, theory),
            "log_spiral_source": p.get("log_spiral_source"),
            "passive_wall_friction_deg": p.get("passive_wall_friction_deg")}


def run_braced_excavation(p, *, method="braced_excavation"):
    """The braced_excavation tool's run: (result, the inputs it used)."""
    reject_unknown_params(p, _GEOM_PARAMS + ("Fy", "FOS_passive", "embedment_increase")
                          + _PASSIVE_PARAMS, method=method)
    geometry = _build_geometry(p, method=method, envelope=True)
    used = {"geometry": geometry, "Fy": p.get("Fy", 345.0),
            "FOS_passive": p.get("FOS_passive", 1.5),
            "embedment_increase": p.get("embedment_increase", 1.0),
            **_passive(p)}
    result = analyze_braced_excavation(
        geometry, Fy=used["Fy"], FOS_passive=used["FOS_passive"],
        embedment_increase=used["embedment_increase"],
        **{k: used[k] for k in _PASSIVE_PARAMS})
    return result, used


def _run_braced_excavation(p):
    return run_braced_excavation(p)[0].to_dict()


def run_cantilever_excavation(p, *, method="cantilever_excavation"):
    """The cantilever_excavation tool's run: (result, the inputs it used)."""
    # No supports: a cantilever has none (GM-110).
    reject_unknown_params(
        p, tuple(k for k in _GEOM_PARAMS if k != "supports")
        + ("FOS_passive", "Fy", "embedment_increase") + _PASSIVE_PARAMS,
        method=method)
    geometry = _build_geometry(p, method=method)
    used = {"geometry": geometry, "FOS_passive": p.get("FOS_passive", 1.5),
            "Fy": p.get("Fy", 345.0),
            "embedment_increase": p.get("embedment_increase", 1.2),
            **_passive(p)}
    result = analyze_cantilever_excavation(
        geometry, FOS_passive=used["FOS_passive"], Fy=used["Fy"],
        embedment_increase=used["embedment_increase"],
        **{k: used[k] for k in _PASSIVE_PARAMS})
    return result, used


def _run_cantilever_excavation(p):
    return run_cantilever_excavation(p)[0].to_dict()


def _run_apparent_pressure(params):
    reject_unknown_params(params, ("excavation_depth", "layers", "surcharge", "gwt_depth"),
                          method="apparent_pressure")
    require_params(params, ["excavation_depth"], method="apparent_pressure")
    layers = _build_layers(params, method="apparent_pressure", envelope=True)
    return select_apparent_pressure(layers, params["excavation_depth"], params.get("surcharge", 0.0),
                                    gwt_depth=params.get("gwt_depth"))


def _run_select_wall_section(params):
    reject_unknown_params(params, ("required_Sx_cm3", "section_type"),
                          method="select_wall_section")
    require_params(params, ["required_Sx_cm3"], method="select_wall_section",
                   valid=["required_Sx_cm3", "section_type"])
    t = params.get("section_type", "hp")
    sx = params["required_Sx_cm3"]
    if t == "hp": r = select_hp_section(sx)
    elif t == "sheet_pile": r = select_sheet_pile(sx)
    elif t == "w": r = select_w_section(sx)
    else: return {"error": f"Unknown section_type '{t}'. Allowed: ['hp', 'sheet_pile', 'w']."}
    return r if r else {"error": f"No {t} section for Sx = {sx} cm3"}


def _run_check_basal_heave(params):
    m = params.get("method", "terzaghi")
    reject_unknown_params(params, ("H", "cu", "gamma", "method", "q_surcharge",
                                   "B", "Be", "Le", "L", "Nc", "width_factor",
                                   "FOS_required"),
                          method="check_basal_heave")
    require_params(params, ["H", "cu", "gamma"], method="check_basal_heave",
                   valid=["H", "cu", "gamma", "method", "q_surcharge", "B",
                          "Be", "Le", "L", "Nc", "width_factor", "FOS_required"])
    if m == "bjerrum_eide":
        require_params(params, ["Be", "Le"], method="check_basal_heave (bjerrum_eide)",
                       valid=["H", "cu", "gamma", "Be", "Le", "q_surcharge", "FOS_required"])
    if m == "caltrans":
        require_params(params, ["B"], method="check_basal_heave (caltrans)",
                       valid=["H", "cu", "gamma", "B", "L", "Nc", "q_surcharge",
                              "width_factor", "FOS_required"])
    if m == "terzaghi":
        return check_basal_heave_terzaghi(H=params["H"], cu=params["cu"], gamma=params["gamma"],
                                           q_surcharge=params.get("q_surcharge", 0.0), B=params.get("B", 0.0),
                                           FOS_required=params.get("FOS_required", 1.5)).to_dict()
    elif m == "bjerrum_eide":
        return check_basal_heave_bjerrum_eide(H=params["H"], cu=params["cu"], gamma=params["gamma"],
                                               Be=params["Be"], Le=params["Le"],
                                               q_surcharge=params.get("q_surcharge", 0.0),
                                               FOS_required=params.get("FOS_required", 1.5)).to_dict()
    elif m == "caltrans":
        return check_basal_heave_caltrans(H=params["H"], cu=params["cu"], gamma=params["gamma"],
                                          B=params["B"], L=params.get("L"), Nc=params.get("Nc"),
                                          q_surcharge=params.get("q_surcharge", 0.0),
                                          width_factor=params.get("width_factor", 0.7),
                                          FOS_required=params.get("FOS_required", 1.5)).to_dict()
    return {"error": f"Unknown method '{m}'. Allowed: ['terzaghi', 'bjerrum_eide', 'caltrans']."}


#: The FHWA anchored-wall inputs the single-anchor route alone reads.
_SINGLE_ANCHOR_ONLY = ("FOS_embedment", "Kp")


def run_fhwa_apparent_pressure_anchored_wall(
        p, *, method="fhwa_apparent_pressure_anchored_wall"):
    """The FHWA anchored-wall tool's run: (result dict, the inputs it used)."""
    reject_unknown_params(
        p, ("H", "anchor_depths", "gamma", "phi", "surcharge", "spacing",
            "inclination_deg", "Ka", "Kp", "FOS_embedment"),
        method=method)
    require_params(p, ["H", "anchor_depths", "gamma", "phi"], method=method)
    depths = p["anchor_depths"]
    single = [k for k in _SINGLE_ANCHOR_ONLY if k in p]
    if single and isinstance(depths, (list, tuple)) and len(depths) > 1:
        # GM-182: the tributary area method does not read them
        raise ValueError(
            f"{method}: {' and '.join(single)} apply to a single anchor "
            "only (free earth support about the anchor, Caltrans Trenching "
            "and Shoring Manual Sec. 8-4). With two or more anchor levels "
            "the tributary area method (GEC-4 Figure 39) carries the load "
            "below the lowest anchor as the subgrade reaction R and works "
            "out no embedment; leave them out. For the embedment below the "
            "lowest anchor, run braced_excavation with the anchors as "
            "supports (FOS_passive about the lowest support).")
    used = {"H": p["H"], "anchor_depths": p["anchor_depths"],
            "gamma": p["gamma"], "phi": p["phi"],
            "surcharge": p.get("surcharge", 0.0),
            "spacing": p.get("spacing", 1.0),
            "inclination_deg": p.get("inclination_deg", 0.0),
            "Ka": p.get("Ka"), "Kp": p.get("Kp"),
            "FOS_embedment": p.get("FOS_embedment", 1.3)}
    return fhwa_apparent_pressure_anchored_wall(**used), used


def _run_fhwa_apparent_pressure(p):
    return run_fhwa_apparent_pressure_anchored_wall(p)[0]


def _run_check_bottom_blowout(p):
    reject_unknown_params(p, ("D_embed", "hw_excess", "gamma_soil", "gamma_w",
                              "FOS_required"),
                          method="check_bottom_blowout")
    require_params(p, ["D_embed", "hw_excess", "gamma_soil"], method="check_bottom_blowout",
                   valid=["D_embed", "hw_excess", "gamma_soil", "gamma_w", "FOS_required"])
    return check_bottom_blowout(D_embed=p["D_embed"], hw_excess=p["hw_excess"], gamma_soil=p["gamma_soil"],
                                 gamma_w=p.get("gamma_w", 9.81), FOS_required=p.get("FOS_required", 1.5)).to_dict()


def _run_check_piping(p):
    reject_unknown_params(p, ("delta_h", "flow_path", "Gs", "void_ratio",
                              "FOS_required"),
                          method="check_piping")
    require_params(p, ["delta_h", "flow_path"], method="check_piping",
                   valid=["delta_h", "flow_path", "Gs", "void_ratio", "FOS_required"])
    return check_piping(delta_h=p["delta_h"], flow_path=p["flow_path"], Gs=p.get("Gs", 2.65),
                         void_ratio=p.get("void_ratio", 0.65), FOS_required=p.get("FOS_required", 2.0)).to_dict()


_ANCHOR_PARAMS = (
    "design_load_kN", "anchor_depth", "excavation_depth", "phi_deg",
    "soil_type", "anchor_angle_deg", "drill_diameter_mm", "tendon_type",
    "FOS_bond", "lock_off_pct", "max_load_pct", "bond_stress_kPa",
    "test_load_factor", "load_transfer_kN_per_m")


def _run_design_ground_anchor(p):
    reject_unknown_params(p, _ANCHOR_PARAMS, method="design_ground_anchor")
    require_params(p, ["design_load_kN", "anchor_depth", "excavation_depth", "phi_deg"],
                   method="design_ground_anchor", valid=list(_ANCHOR_PARAMS))
    return design_ground_anchor(
        design_load_kN=p["design_load_kN"], anchor_depth=p["anchor_depth"],
        excavation_depth=p["excavation_depth"], phi_deg=p["phi_deg"],
        soil_type=p.get("soil_type"), anchor_angle_deg=p.get("anchor_angle_deg", 15.0),
        drill_diameter_mm=p.get("drill_diameter_mm"), tendon_type=p.get("tendon_type", "strand_15mm"),
        FOS_bond=p.get("FOS_bond"), lock_off_pct=p.get("lock_off_pct", 0.80),
        max_load_pct=p.get("max_load_pct", 0.60), bond_stress_kPa=p.get("bond_stress_kPa"),
        test_load_factor=p.get("test_load_factor", 1.33),
        load_transfer_kN_per_m=p.get("load_transfer_kN_per_m"),
    ).to_dict()


METHOD_REGISTRY = {
    "braced_excavation": _run_braced_excavation,
    "cantilever_excavation": _run_cantilever_excavation,
    "apparent_pressure": _run_apparent_pressure,
    "select_wall_section": _run_select_wall_section,
    "check_basal_heave": _run_check_basal_heave,
    "check_bottom_blowout": _run_check_bottom_blowout,
    "check_piping": _run_check_piping,
    "design_ground_anchor": _run_design_ground_anchor,
    "fhwa_apparent_pressure_anchored_wall": _run_fhwa_apparent_pressure,
}

_LAYERS = {"type": "array", "required": True, "description": "Array of soil-layer dicts, top down: {thickness (m, required), unit_weight (kN/m3, required), friction_angle (deg, default 30), cohesion (kPa, default 0), soil_type ('sand' or 'clay', also 'soft_clay' or 'stiff_clay'; REQUIRED on each layer above the excavation base, refused if missing)}. soil_type picks the Terzaghi-Peck apparent-pressure envelope: the one for sands where sand is the thicker over the excavation depth, else the one for clays, soft to medium or stiff fissured by the average N = gamma*H/cu over the excavation depth (> 4 soft). The embedment below the excavation uses each layer's own phi and c. The last layer continues below its thickness."}
# The cantilever reads no envelope: soil_type is accepted, not read (GM-182).
_LAYERS_CANTILEVER = {"type": "array", "required": True, "description": "Array of soil-layer dicts, top down: {thickness (m, required), unit_weight (kN/m3, required), friction_angle (deg, default 30), cohesion (kPa, default 0), soil_type (optional and not read here: the cantilever analysis uses each layer's own phi and c; accepted so the same layers serve braced_excavation)}. The last layer continues below its thickness."}

_GWT_EXC = {"type": "float", "required": False, "description": "Groundwater depth on the excavation side (m from top of wall), never above the excavation base. Default: same as gwt_depth."}

# The ground anchor (GM-185, GM-191/192): GEC-4's printed load transfers,
# factors of safety and minimum lengths.
_ANCHOR_INFO = {
    "category": "Anchors",
    "brief": "Ground anchor design per GEC-4 (FHWA-IF-99-015) Sec. 5.3: "
             "unbonded and bond lengths, tendon, test load.",
    "parameters": {
        "design_load_kN": {"type": "float", "required": True, "description": "Design anchor load (kN)."},
        "anchor_depth": {"type": "float", "required": True, "description": "Anchor depth (m)."},
        "excavation_depth": {"type": "float", "required": True, "description": "Excavation depth H (m); the unbonded length extends the greater of 1.5 m and H/5 behind the critical potential failure surface (GEC-4 App. E, p. E-15)."},
        "phi_deg": {"type": "float", "required": True, "description": "Soil friction angle (degrees): the critical potential failure surface rises from the corner of the excavation at 45 + phi/2 (GEC-4 Sec. 5.3.2, p. 65, walls in cohesionless soils)."},
        "soil_type": {"type": "str", "required": False, "default": "sand_medium", "allowed_values": list(list_bond_stress_types()) + list(GROUNDS), "description": "Ground of the bond zone, a printed row: soil from GEC-4 Table 6 (p. 71; small diameter straight shaft gravity-grouted anchors; SPT corrected for overburden): sand_gravel_loose 145, sand_gravel_medium 220, sand_gravel_dense 290, sand_loose 100, sand_medium 145, sand_dense 190, sand_silt_loose 70, sand_silt_medium 100, sand_silt_dense 130, silt_clay_stiff 30, silt_clay_hard 60 kN/m (silt-clay mixtures with low plasticity); rock from Table 8 (p. 74): granite_basalt 730, dolomitic_limestone 580, soft_limestone 440, sandstone 440, slates_hard_shales 360, soft_shales 150 kN/m. The ultimate load transfer is per metre of bond length (the hole diameter does not enter it). Also sets FOS_bond's default and the minimum bond length (soil or rock). With bond_stress_kPa or load_transfer_kN_per_m it is required, and may be the ground alone, 'soil' or 'rock': the value does not say the ground (GEC-4 Table 7, p. 73: 0.15 to 3.1 MPa in rock, 0.03 to 1.38 MPa in soil)."},
        "tendon_type": {"type": "str", "required": False, "default": "strand_15mm", "allowed_values": ["strand_13mm", "strand_15mm", "bar_26mm", "bar_32mm", "bar_36mm", "bar_44mm"], "description": "Tendon type. Bar or strand sets the minimum unbonded length (GEC-4 p. 67: 3 m bar, 4.5 m strand) and, in rock, the minimum bond length (p. E-15: 3 m bar, 4.5 m strand; 4.5 m for both in soil)."},
        "anchor_angle_deg": {"type": "float", "required": False, "default": 15.0, "description": "Anchor inclination below horizontal (deg)."},
        "load_transfer_kN_per_m": {"type": "float", "required": False, "description": "Ultimate load transfer per metre of bond length (kN/m), in place of soil_type's, e.g. as verified by preproduction performance tests (GEC-4 p. 75). Not with bond_stress_kPa. Give soil_type with it (a row, or 'soil' or 'rock')."},
        "bond_stress_kPa": {"type": "float", "required": False, "description": "Average ultimate bond stress at the grout/ground interface (kPa), e.g. from GEC-4 Table 7 (p. 73, after PTI 1996), in place of soil_type's load transfer: multiplied by the perimeter of drill_diameter_mm. Not with load_transfer_kN_per_m. Give soil_type with it (a row, or 'soil' or 'rock')."},
        "drill_diameter_mm": {"type": "float", "required": False, "default": 150.0, "description": "Drill hole diameter (mm): the perimeter a given bond_stress_kPa acts on (GEC-4 p. 72: capacity estimates on the as-drilled hole diameter; p. 70: generally less than 150 mm, hollow stem augered anchors about 300 mm). Not used with a load transfer per metre."},
        "FOS_bond": {"type": "float", "required": False, "default_depends_on": {"parameter": "soil_type", "values": {k: 3.0 for k in ROCK_TYPES + ("rock",)}, "otherwise": 2.0}, "description": "FOS on the ultimate load transfer. Default depends on soil_type: 3.0 for the Table 8 rock rows and 'rock' (competent rock, GEC-4 p. 74), else 2.0 (soil, p. 71). GEC-4 also gives 2.0 for weak rocks such as clay shales, intermediate geomaterials of unconfined compressive strength 0.5 to 5.0 MPa (p. 74)."},
        "lock_off_pct": {"type": "float", "required": False, "default": 0.80, "description": "Lock-off load fraction of design load. Reported as tendon.lock_off_kN only; it does not size the tendon."},
        "max_load_pct": {"type": "float", "required": False, "default": 0.60, "description": "Design-load limit as a fraction of the tendon's SMTS (GEC-4 Sec. 5.3.8, p. 77: 0.60). The tendon is also sized so the test load (test_load_factor x design load) is at most 0.80 SMTS, which allows 0.80 / test_load_factor SMTS (0.60 at 1.33, 0.53 at 1.50); tendon.governed_by names the limit that sized it."},
        "test_load_factor": {"type": "float", "required": False, "default": 1.33, "description": "Test load as a multiple of the design load: the maximum load of the proof and performance tests, carried by the tendon at no more than 0.80 SMTS. GEC-4 Sec. 7.4.2.2 (p. 143): 1.20 to 1.50, 1.33 commonly used for permanent and 1.20 for temporary applications, 1.50 for potentially creeping soils; Tables 21 and 22 (pp. 144, 147) test to 1.33, which may be set at 1.20 for a temporary support of excavation. Default 1.33, the stricter; at max_load_pct 0.60, 1.20 gives the same tendon."},
    },
    "returns": {
        "unbonded_length_m": "Unbonded length: to the greater of 1.5 m and H/5 behind the critical potential failure surface, at least 4.5 m (strand) or 3 m (bar).",
        "bond_length_m": "Bond length FOS_bond x DL / load_transfer_kN_per_m, at least 4.5 m (3 m for a bar in rock).",
        "total_length_m": "Total anchor length.",
        "load_transfer_kN_per_m": "Ultimate load transfer per metre of bond length used.",
        "FOS_bond": "Factor of safety on bond used.",
        "ground": "Ground of the bond zone (soil or rock): set FOS_bond's default and the minimum bond length.",
        "tendon": "Tendon size and governed_by, the SMTS limit that sized it.",
        "test_load_kN": "The test load (test_load_factor x design load).",
        "choices": "Method, test load, SMTS limit, load transfer, FOS_bond, distance behind the failure surface, minimum unbonded and bond lengths: given, default or the module's, with basis and alternatives.",
    },
}

METHOD_INFO = {
    "braced_excavation": {"category": "Braced Excavation", "brief": "Multi-level braced excavation analysis (Terzaghi-Peck).",
        "parameters": {"excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."}, "layers": _LAYERS, "supports": {"type": "array", "required": False, "description": "Support levels [{depth (m), support_type ('strut' or 'anchor'), spacing (m), angle_deg}]."}, "surcharge": {"type": "float", "required": False, "default": 10.0, "description": "Surface surcharge (kPa); added to the envelope as Ka*q (sand) or q (clay)."}, "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth behind the wall (m); water pressure is added for sand envelopes."}, "gwt_depth_excavation": _GWT_EXC, "FOS_passive": {"type": "float", "required": False, "default": 1.5, "description": "Embedment FS, MR = FS*MD about the lowest support (Caltrans uses 1.3)."}, "embedment_increase": {"type": "float", "required": False, "default": 1.0, "description": "Multiplier on the free-earth-support embedment."}, "passive_theory": _PASSIVE_THEORY, "log_spiral_source": _LOG_SPIRAL_SOURCE, "passive_wall_friction_deg": _PASSIVE_WALL_FRICTION, "Fy": {"type": "float", "required": False, "default": 345.0, "description": "Steel yield strength (MPa) for section sizing."}},
        "returns": {"support_reactions": "Support loads (kN/m of wall).", "max_moment_kNm_per_m": "Max wall moment (spans and embedded portion).", "required_embedment_m": "Free-earth-support embedment below the excavation; null when no moment balance within 6H.", "free_earth": "Embedment detail: D at FS and at 1.0, lowest-support load from below.", "converged": "False when free earth support found no moment balance: the depths (and what follows from them) are null and the first note says why.", "notes": "What pressure was applied and how.", "choices": _PASSIVE_CHOICES}},
    "cantilever_excavation": {"category": "Cantilever Excavation", "brief": "Cantilever (unbraced) excavation wall analysis.",
        "parameters": {"excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."}, "layers": _LAYERS_CANTILEVER, "surcharge": {"type": "float", "required": False, "default": 10.0, "description": "Surface surcharge (kPa)."}, "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth behind the wall (m)."}, "gwt_depth_excavation": _GWT_EXC, "FOS_passive": {"type": "float", "required": False, "default": 1.5, "description": "FOS on passive resistance."}, "embedment_increase": {"type": "float", "required": False, "default": 1.2, "description": "D / D0 (Caltrans/AASHTO 1.2; use 1.0 to carry all safety on FOS_passive)."}, "passive_theory": _PASSIVE_THEORY, "log_spiral_source": _LOG_SPIRAL_SOURCE, "passive_wall_friction_deg": _PASSIVE_WALL_FRICTION, "Fy": {"type": "float", "required": False, "default": 345.0, "description": "Steel yield strength (MPa) for section sizing."}},
        "returns": {"required_embedment_m": "Embedment D = embedment_increase x D0 (Caltrans T&S Simplified Method, layered Rankine pressures with water; the log-spiral passive Kp with passive_theory 'log_spiral').", "Kp": "The passive coefficient of the layer the wall is embedded in.", "choices": _PASSIVE_CHOICES, "embedment_converged_m": "D0.", "max_moment_kNm_per_m": "Maximum moment (below the excavation).", "max_moment_depth_m": "Depth of the maximum moment.", "converged": "False when no D0 within 6H balances the moments: the wall cannot stand as a cantilever, embedment/moment/shear are null and the first note says why.", "notes": "Applicability notes."}},
    "apparent_pressure": {"category": "Earth Pressure", "brief": "Terzaghi-Peck apparent earth pressure envelope.",
        "parameters": {"excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."}, "layers": _LAYERS, "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge (kPa); not part of the envelope (braced_excavation adds it)."}, "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m); the sand envelope uses the effective unit weight below it."}},
        "returns": {"type": "Envelope type.", "max_pressure_kPa": "Maximum pressure."}},
    "select_wall_section": {"category": "Section Selection", "brief": "Select lightest HP/sheet/W section for demand.",
        "parameters": {"required_Sx_cm3": {"type": "float", "required": True, "description": "Required section modulus (cm3)."}, "section_type": {"type": "str", "required": False, "default": "hp", "allowed_values": ["hp", "sheet_pile", "w"], "description": "Section family."}},
        "returns": {"designation": "Section designation.", "Sx_cm3": "Section modulus."}},
    "check_basal_heave": {"category": "Stability", "brief": "Basal heave stability check (Terzaghi, Bjerrum-Eide, or Caltrans force-balance).",
        "parameters": {"H": {"type": "float", "required": True, "description": "Excavation depth (m)."}, "cu": {"type": "float", "required": True, "description": "Undrained shear strength (kPa)."}, "gamma": {"type": "float", "required": True, "description": "Soil unit weight (kN/m3)."}, "method": {"type": "str", "required": False, "default": "terzaghi", "allowed_values": ["terzaghi", "bjerrum_eide", "caltrans"], "description": "Check method. bjerrum_eide requires Be+Le (inverted-footing bearing ratio, NO side shear). caltrans requires B (Caltrans/Terzaghi force-balance with sidewall shear S=cu*H and the 0.7B block; optional L for the rectangular Nc correction)."}, "B": {"type": "float", "required": False, "default": 0.0, "description": "Excavation width (m); terzaghi and caltrans methods."}, "L": {"type": "float", "required": False, "description": "Excavation length (m) for the caltrans rectangular Nc correction (default square)."}, "Nc": {"type": "float", "required": False, "description": "Explicit chart-read bearing factor Nc (caltrans); default Skempton's (1951) Nc = 5.14 (1 + 0.2 B/L)(1 + 0.2 H/B), H/B capped at 2.5."}, "Be": {"type": "float", "required": False, "description": "Excavation width (m). REQUIRED for bjerrum_eide."}, "Le": {"type": "float", "required": False, "description": "Excavation length (m). REQUIRED for bjerrum_eide."}, "width_factor": {"type": "float", "required": False, "default": 0.7, "description": "Failure-block width as a fraction of B (caltrans). Default 0.7 (Terzaghi)."}, "q_surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge (kPa)."}, "FOS_required": {"type": "float", "required": False, "default": 1.5, "description": "Required FOS."}},
        "returns": {"FOS": "Factor of safety.", "is_stable": "Pass/fail."}},
    "fhwa_apparent_pressure_anchored_wall": {"category": "Earth Pressure", "brief": "FHWA/GEC-4 apparent-pressure anchored-wall design (tributary method): apparent envelope pe + tributary anchor loads + subgrade reaction + hinge moments + anchor design loads.",
        "parameters": {"H": {"type": "float", "required": True, "description": "Total wall height (m)."}, "anchor_depths": {"type": "array", "required": True, "description": "Depths of the anchor levels below the top of wall (m), e.g. [2.5, 6.25]."}, "gamma": {"type": "float", "required": True, "description": "Retained-soil unit weight (kN/m3)."}, "phi": {"type": "float", "required": True, "description": "Retained-soil friction angle (deg)."}, "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Uniform surface surcharge q (kPa); adds ps = Ka*q."}, "spacing": {"type": "float", "required": False, "default": 1.0, "description": "Horizontal anchor spacing (m) for the design loads."}, "inclination_deg": {"type": "float", "required": False, "default": 0.0, "description": "Anchor inclination from horizontal (deg)."}, "Ka": {"type": "float", "required": False, "description": "Active coefficient override (default Rankine Ka)."}, "Kp": {"type": "float", "required": False, "description": "Single anchor only (refused with two or more anchors): passive coefficient below the excavation (default Rankine; pass a log-spiral value)."}, "FOS_embedment": {"type": "float", "required": False, "default": 1.3, "description": "Single anchor only (refused with two or more anchors): embedment FS, MR = FS*MD about the anchor (Caltrans T&S Manual Sec. 8-4)."}},
        "returns": {"pe_kPa": "Max apparent-pressure ordinate.", "anchors": "Per-anchor {depth, TH (horizontal load), design_load}.", "subgrade_reaction_kN_per_m": "Base reaction R (two or more anchors; None for one).", "max_moment_kN_m_per_m": "Governing wall moment.", "embedment_D_m": "Single anchor: free-earth-support embedment (Caltrans Ex 8-1 method); null with no moment balance (converged false, see notes)."}},
    "check_bottom_blowout": {"category": "Stability", "brief": "Bottom blowout (hydraulic uplift) check.",
        "parameters": {"D_embed": {"type": "float", "required": True, "description": "Wall embedment below excavation (m)."}, "hw_excess": {"type": "float", "required": True, "description": "Excess water head (m)."}, "gamma_soil": {"type": "float", "required": True, "description": "Soil unit weight (kN/m3)."}, "gamma_w": {"type": "float", "required": False, "default": 9.81, "description": "Water unit weight (kN/m3)."}, "FOS_required": {"type": "float", "required": False, "default": 1.5, "description": "Required FOS."}},
        "returns": {"FOS": "Factor of safety."}},
    "check_piping": {"category": "Stability", "brief": "Piping (internal erosion) check.",
        "parameters": {"delta_h": {"type": "float", "required": True, "description": "Head difference (m)."}, "flow_path": {"type": "float", "required": True, "description": "Seepage path length (m)."}, "Gs": {"type": "float", "required": False, "default": 2.65, "description": "Specific gravity of solids."}, "void_ratio": {"type": "float", "required": False, "default": 0.65, "description": "Void ratio."}, "FOS_required": {"type": "float", "required": False, "default": 2.0, "description": "Required FOS."}},
        "returns": {"FOS": "Factor of safety.", "i_exit": "Exit gradient."}},
    "design_ground_anchor": _ANCHOR_INFO,
}

# Names the wall analyses do not take, with the reason dispatch gives
# (GM-110).
_NO_WIDTH = ("the wall analysis does not use the excavation width; for "
             "basal heave give it to check_basal_heave (B, or Be and Le)")
METHOD_INFO["braced_excavation"]["refused_parameters"] = {
    "excavation_width": _NO_WIDTH}
METHOD_INFO["cantilever_excavation"]["refused_parameters"] = {
    "excavation_width": _NO_WIDTH,
    "supports": ("a cantilever wall has no supports; a wall with struts or "
                 "anchors is braced_excavation")}

_LOG_SPIRAL = ["log_spiral", "caquot_kerisel"]

# When each default applies (GM-119): the failure-block width to the
# Caltrans check; the apparent-pressure envelope leaves the surcharge out
# (braced_excavation adds it).
declare_applies_when(METHOD_INFO, {
    "check_basal_heave": {"width_factor": {"method": ["caltrans"]}},
    # GM-191: the hole's perimeter carries a given bond stress only; a
    # load transfer per metre does not depend on it. GM-199: soil_type's
    # default applies only with neither given; beside a given bond stress
    # or load transfer the call names the ground (soil or rock) itself.
    "design_ground_anchor": {
        "drill_diameter_mm": {"bond_stress_kPa": "set"},
        "soil_type": {"bond_stress_kPa": "unset",
                      "load_transfer_kN_per_m": "unset"}},
    "apparent_pressure": {"surcharge": False},
    # GM-242: the log-spiral passive Kp's inputs
    **{name: {"log_spiral_source": {"passive_theory": _LOG_SPIRAL},
              "passive_wall_friction_deg": {"passive_theory": _LOG_SPIRAL}}
       for name in ("braced_excavation", "cantilever_excavation")},
    # GM-182: a single anchor's alone; that run's choices call them
    # "default" when left out, so defaults_used lists them there only
    "fhwa_apparent_pressure_anchored_wall": {
        name: False for name in _SINGLE_ANCHOR_ONLY}})


# What a left-out input with no default stands for (BACKLOG GM-165):
# the water table on each side, the Caltrans basal-heave length and Nc,
# and the FHWA wall's earth pressure coefficients.
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
_NO_WATER = ("no water table",
             "no water pressure on the wall; total unit weight throughout")
_LEFT_OUT_FRICTION = (
    "by layer: the log-spiral source's own, none with cohesion",
    "delta/phi' = 0.66 (UFC 3-220-20 p. 191), or delta = phi' with "
    "'caquot_kerisel_1948', in a layer below the excavation without "
    "cohesion; none in one with cohesion (UFC p. 238); the choices give "
    "each layer's")
declare_when_absent(METHOD_INFO, {
    "braced_excavation": {"gwt_depth": _NO_WATER,
                          "gwt_depth_excavation": "gwt_depth",
                          "passive_wall_friction_deg": _LEFT_OUT_FRICTION},
    "cantilever_excavation": {"gwt_depth": _NO_WATER,
                              "gwt_depth_excavation": "gwt_depth",
                              "passive_wall_friction_deg":
                                  _LEFT_OUT_FRICTION},
    "apparent_pressure": {"gwt_depth": _NO_WATER},
    "check_basal_heave": {
        "L": "B",
        "Nc": ("Skempton's (1951) Nc from H/B and B/L",
               "Nc = 5.14 (1 + 0.2 B/L)(1 + 0.2 H/B), H/B capped at 2.5, "
               "the chart the Caltrans heave method reads; give Nc for "
               "another chart reading")},
    "fhwa_apparent_pressure_anchored_wall": {
        "Ka": ("Rankine Ka from phi",
               "a vertical wall and level ground; give Ka for another case"),
        "Kp": ("Rankine Kp from phi",
               "no wall friction; give a log-spiral value for a wall with "
               "friction")},
})
for _name in ("L", "Nc"):
    METHOD_INFO["check_basal_heave"]["parameters"][_name][
        "applies_when"] = {"method": ["caltrans"]}

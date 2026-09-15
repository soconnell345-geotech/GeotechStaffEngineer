"""Support of excavation adapter — braced/cantilever, stability, anchors."""

from funhouse_agent.adapters import reject_unknown_params, require_keys, require_params
from soe import (
    ExcavationGeometry, SOEWallLayer, SupportLevel,
    select_apparent_pressure, analyze_braced_excavation, analyze_cantilever_excavation,
    select_hp_section, select_sheet_pile, select_w_section,
    check_basal_heave_terzaghi, check_basal_heave_bjerrum_eide,
    check_basal_heave_caltrans, check_bottom_blowout, check_piping,
    design_ground_anchor, fhwa_apparent_pressure_anchored_wall,
)


def _soil_type(layer, H):
    """Accept 'clay' (field note 2026-09-15: the brief used to offer it and the
    module rejected it) by the Terzaghi-Peck stability number N = gamma*H/cu:
    N > 4 soft_clay, otherwise stiff_clay."""
    t = str(layer.get("soil_type", "sand")).strip().lower()
    if t == "clay":
        cu = float(layer.get("cohesion", 0.0) or 0.0)
        gamma = float(layer["unit_weight"])
        return "soft_clay" if cu <= 0 or gamma * float(H) / cu > 4.0 else "stiff_clay"
    return t


def _build_layers(params, *, method):
    require_params(params, ["layers"], method=method)
    H = params.get("excavation_depth", 0.0) or 0.0
    layers = []
    for l in params["layers"]:
        require_keys(l, ["thickness", "unit_weight"], method=method)
        layers.append(SOEWallLayer(
            thickness=l["thickness"], unit_weight=l["unit_weight"],
            friction_angle=l.get("friction_angle", 30.0), cohesion=l.get("cohesion", 0.0),
            soil_type=_soil_type(l, H), description=l.get("description", ""),
        ))
    return layers


def _build_geometry(params, *, method):
    require_params(params, ["excavation_depth"], method=method)
    layers = _build_layers(params, method=method)
    supports = []
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
        gwt_depth=params.get("gwt_depth"), excavation_width=params.get("excavation_width", 0.0),
        gwt_depth_excavation=params.get("gwt_depth_excavation"),
    )


# Geometry params consumed by _build_geometry (shared by both wall analyses).
_GEOM_PARAMS = ("excavation_depth", "layers", "supports", "surcharge",
                "gwt_depth", "excavation_width", "gwt_depth_excavation")


def _run_braced_excavation(p):
    reject_unknown_params(p, _GEOM_PARAMS + ("Fy", "FOS_passive", "embedment_increase"),
                          method="braced_excavation")
    return analyze_braced_excavation(
        _build_geometry(p, method="braced_excavation"), Fy=p.get("Fy", 345.0),
        FOS_passive=p.get("FOS_passive", 1.5),
        embedment_increase=p.get("embedment_increase", 1.0)).to_dict()


def _run_cantilever_excavation(p):
    reject_unknown_params(p, _GEOM_PARAMS + ("FOS_passive", "Fy", "embedment_increase"),
                          method="cantilever_excavation")
    return analyze_cantilever_excavation(
        _build_geometry(p, method="cantilever_excavation"),
        FOS_passive=p.get("FOS_passive", 1.5), Fy=p.get("Fy", 345.0),
        embedment_increase=p.get("embedment_increase", 1.2)).to_dict()


def _run_apparent_pressure(params):
    reject_unknown_params(params, ("excavation_depth", "layers", "surcharge", "gwt_depth"),
                          method="apparent_pressure")
    require_params(params, ["excavation_depth"], method="apparent_pressure")
    layers = _build_layers(params, method="apparent_pressure")
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


def _run_fhwa_apparent_pressure(p):
    reject_unknown_params(
        p, ("H", "anchor_depths", "gamma", "phi", "surcharge", "spacing",
            "inclination_deg", "Ka", "Kp", "FOS_embedment"),
        method="fhwa_apparent_pressure_anchored_wall")
    require_params(p, ["H", "anchor_depths", "gamma", "phi"],
                   method="fhwa_apparent_pressure_anchored_wall")
    return fhwa_apparent_pressure_anchored_wall(
        H=p["H"], anchor_depths=p["anchor_depths"], gamma=p["gamma"], phi=p["phi"],
        surcharge=p.get("surcharge", 0.0), spacing=p.get("spacing", 1.0),
        inclination_deg=p.get("inclination_deg", 0.0), Ka=p.get("Ka"),
        Kp=p.get("Kp"), FOS_embedment=p.get("FOS_embedment", 1.3))


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


def _run_design_ground_anchor(p):
    reject_unknown_params(
        p,
        ("design_load_kN", "anchor_depth", "excavation_depth", "phi_deg",
         "soil_type", "anchor_angle_deg", "drill_diameter_mm", "tendon_type",
         "FOS_bond", "lock_off_pct", "max_load_pct", "bond_stress_kPa"),
        method="design_ground_anchor")
    require_params(p, ["design_load_kN", "anchor_depth", "excavation_depth", "phi_deg"],
                   method="design_ground_anchor",
                   valid=["design_load_kN", "anchor_depth", "excavation_depth", "phi_deg",
                          "soil_type", "anchor_angle_deg", "drill_diameter_mm", "tendon_type",
                          "FOS_bond", "lock_off_pct", "max_load_pct", "bond_stress_kPa"])
    return design_ground_anchor(
        design_load_kN=p["design_load_kN"], anchor_depth=p["anchor_depth"],
        excavation_depth=p["excavation_depth"], phi_deg=p["phi_deg"],
        soil_type=p.get("soil_type", "sand_medium"), anchor_angle_deg=p.get("anchor_angle_deg", 15.0),
        drill_diameter_mm=p.get("drill_diameter_mm", 150.0), tendon_type=p.get("tendon_type", "strand_15mm"),
        FOS_bond=p.get("FOS_bond", 2.0), lock_off_pct=p.get("lock_off_pct", 0.80),
        max_load_pct=p.get("max_load_pct", 0.60), bond_stress_kPa=p.get("bond_stress_kPa"),
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

_LAYERS = {"type": "array", "required": True, "description": "Array of soil-layer dicts, top down: {thickness (m, required), unit_weight (kN/m3, required), friction_angle (deg, default 30), cohesion (kPa, default 0), soil_type ('sand', 'soft_clay' or 'stiff_clay'; 'clay' is mapped by N = gamma*H/cu > 4 -> soft; default 'sand')}. soil_type picks the apparent-pressure envelope; embedment and cantilever calcs use each layer's own phi and c. The last layer continues below its thickness."}

_GWT_EXC = {"type": "float", "required": False, "description": "Groundwater depth on the excavation side (m from top of wall), never above the excavation base. Default: same as gwt_depth."}

METHOD_INFO = {
    "braced_excavation": {"category": "Braced Excavation", "brief": "Multi-level braced excavation analysis (Terzaghi-Peck).",
        "parameters": {"excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."}, "layers": _LAYERS, "supports": {"type": "array", "required": False, "description": "Support levels [{depth (m), support_type ('strut' or 'anchor'), spacing (m), angle_deg}]."}, "surcharge": {"type": "float", "required": False, "default": 10.0, "description": "Surface surcharge (kPa); added to the envelope as Ka*q (sand) or q (clay)."}, "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth behind the wall (m); water pressure is added for sand envelopes."}, "gwt_depth_excavation": _GWT_EXC, "excavation_width": {"type": "float", "required": False, "default": 0.0, "description": "Excavation width (m), for stability checks."}, "FOS_passive": {"type": "float", "required": False, "default": 1.5, "description": "Embedment FS, MR = FS*MD about the lowest support (Caltrans uses 1.3)."}, "embedment_increase": {"type": "float", "required": False, "default": 1.0, "description": "Multiplier on the free-earth-support embedment."}, "Fy": {"type": "float", "required": False, "default": 345.0, "description": "Steel yield strength (MPa) for section sizing."}},
        "returns": {"support_reactions": "Support loads (kN/m of wall).", "max_moment_kNm_per_m": "Max wall moment (spans and embedded portion).", "required_embedment_m": "Free-earth-support embedment below the excavation.", "free_earth": "Embedment detail: D at FS and at 1.0, lowest-support load from below.", "notes": "What pressure was applied and how."}},
    "cantilever_excavation": {"category": "Cantilever Excavation", "brief": "Cantilever (unbraced) excavation wall analysis.",
        "parameters": {"excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."}, "layers": _LAYERS, "surcharge": {"type": "float", "required": False, "default": 10.0, "description": "Surface surcharge (kPa)."}, "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth behind the wall (m)."}, "gwt_depth_excavation": _GWT_EXC, "FOS_passive": {"type": "float", "required": False, "default": 1.5, "description": "FOS on passive resistance."}, "embedment_increase": {"type": "float", "required": False, "default": 1.2, "description": "D / D0 (Caltrans/AASHTO 1.2; use 1.0 to carry all safety on FOS_passive)."}, "Fy": {"type": "float", "required": False, "default": 345.0, "description": "Steel yield strength (MPa) for section sizing."}},
        "returns": {"required_embedment_m": "Embedment D = embedment_increase x D0 (Caltrans T&S Simplified Method, layered Rankine pressures with water).", "embedment_converged_m": "D0.", "max_moment_kNm_per_m": "Maximum moment (below the excavation).", "max_moment_depth_m": "Depth of the maximum moment.", "notes": "Applicability notes."}},
    "apparent_pressure": {"category": "Earth Pressure", "brief": "Terzaghi-Peck apparent earth pressure envelope.",
        "parameters": {"excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."}, "layers": _LAYERS, "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge (kPa); not part of the envelope (braced_excavation adds it)."}, "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m); the sand envelope uses the effective unit weight below it."}},
        "returns": {"type": "Envelope type.", "max_pressure_kPa": "Maximum pressure."}},
    "select_wall_section": {"category": "Section Selection", "brief": "Select lightest HP/sheet/W section for demand.",
        "parameters": {"required_Sx_cm3": {"type": "float", "required": True, "description": "Required section modulus (cm3)."}, "section_type": {"type": "str", "required": False, "default": "hp", "allowed_values": ["hp", "sheet_pile", "w"], "description": "Section family."}},
        "returns": {"designation": "Section designation.", "Sx_cm3": "Section modulus."}},
    "check_basal_heave": {"category": "Stability", "brief": "Basal heave stability check (Terzaghi, Bjerrum-Eide, or Caltrans force-balance).",
        "parameters": {"H": {"type": "float", "required": True, "description": "Excavation depth (m)."}, "cu": {"type": "float", "required": True, "description": "Undrained shear strength (kPa)."}, "gamma": {"type": "float", "required": True, "description": "Soil unit weight (kN/m3)."}, "method": {"type": "str", "required": False, "default": "terzaghi", "allowed_values": ["terzaghi", "bjerrum_eide", "caltrans"], "description": "Check method. bjerrum_eide requires Be+Le (inverted-footing bearing ratio, NO side shear). caltrans requires B (Caltrans/Terzaghi force-balance with sidewall shear S=cu*H and the 0.7B block; optional L for the rectangular Nc correction)."}, "B": {"type": "float", "required": False, "default": 0.0, "description": "Excavation width (m); terzaghi and caltrans methods."}, "L": {"type": "float", "required": False, "description": "Excavation length (m) for the caltrans rectangular Nc correction (default square)."}, "Nc": {"type": "float", "required": False, "description": "Explicit chart-read bearing factor Nc (caltrans); default from the Skempton/Bjerrum-Eide form."}, "Be": {"type": "float", "required": False, "description": "Excavation width (m). REQUIRED for bjerrum_eide."}, "Le": {"type": "float", "required": False, "description": "Excavation length (m). REQUIRED for bjerrum_eide."}, "width_factor": {"type": "float", "required": False, "default": 0.7, "description": "Failure-block width as a fraction of B (caltrans). Default 0.7 (Terzaghi)."}, "q_surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge (kPa)."}},
        "returns": {"FOS": "Factor of safety.", "is_stable": "Pass/fail."}},
    "fhwa_apparent_pressure_anchored_wall": {"category": "Earth Pressure", "brief": "FHWA/GEC-4 apparent-pressure anchored-wall design (tributary method): apparent envelope pe + tributary anchor loads + subgrade reaction + hinge moments + anchor design loads.",
        "parameters": {"H": {"type": "float", "required": True, "description": "Total wall height (m)."}, "anchor_depths": {"type": "array", "required": True, "description": "Depths of the anchor levels below the top of wall (m), e.g. [2.5, 6.25]."}, "gamma": {"type": "float", "required": True, "description": "Retained-soil unit weight (kN/m3)."}, "phi": {"type": "float", "required": True, "description": "Retained-soil friction angle (deg)."}, "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Uniform surface surcharge q (kPa); adds ps = Ka*q."}, "spacing": {"type": "float", "required": False, "default": 1.0, "description": "Horizontal anchor spacing (m) for the design loads."}, "inclination_deg": {"type": "float", "required": False, "default": 0.0, "description": "Anchor inclination from horizontal (deg)."}, "Ka": {"type": "float", "required": False, "description": "Active coefficient override (default Rankine Ka)."}, "Kp": {"type": "float", "required": False, "description": "Single anchor: passive coefficient below the excavation (default Rankine; pass a log-spiral value)."}, "FOS_embedment": {"type": "float", "required": False, "default": 1.3, "description": "Single anchor: embedment FS, MR = FS*MD (Caltrans)."}},
        "returns": {"pe_kPa": "Max apparent-pressure ordinate.", "anchors": "Per-anchor {depth, TH (horizontal load), design_load}.", "subgrade_reaction_kN_per_m": "Base reaction R (two or more anchors; None for one).", "max_moment_kN_m_per_m": "Governing wall moment.", "embedment_D_m": "Single anchor: free-earth-support embedment (Caltrans Ex 8-1 method)."}},
    "check_bottom_blowout": {"category": "Stability", "brief": "Bottom blowout (hydraulic uplift) check.",
        "parameters": {"D_embed": {"type": "float", "required": True, "description": "Wall embedment below excavation (m)."}, "hw_excess": {"type": "float", "required": True, "description": "Excess water head (m)."}, "gamma_soil": {"type": "float", "required": True, "description": "Soil unit weight (kN/m3)."}, "gamma_w": {"type": "float", "required": False, "default": 9.81, "description": "Water unit weight (kN/m3)."}, "FOS_required": {"type": "float", "required": False, "default": 1.5, "description": "Required FOS."}},
        "returns": {"FOS": "Factor of safety."}},
    "check_piping": {"category": "Stability", "brief": "Piping (internal erosion) check.",
        "parameters": {"delta_h": {"type": "float", "required": True, "description": "Head difference (m)."}, "flow_path": {"type": "float", "required": True, "description": "Seepage path length (m)."}, "Gs": {"type": "float", "required": False, "default": 2.65, "description": "Specific gravity of solids."}, "void_ratio": {"type": "float", "required": False, "default": 0.65, "description": "Void ratio."}, "FOS_required": {"type": "float", "required": False, "default": 2.0, "description": "Required FOS."}},
        "returns": {"FOS": "Factor of safety.", "i_exit": "Exit gradient."}},
    "design_ground_anchor": {"category": "Anchors", "brief": "Ground anchor design per GEC-4/PTI.",
        "parameters": {"design_load_kN": {"type": "float", "required": True, "description": "Design anchor load (kN)."}, "anchor_depth": {"type": "float", "required": True, "description": "Anchor depth (m)."}, "excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."}, "phi_deg": {"type": "float", "required": True, "description": "Soil friction angle (degrees)."}, "soil_type": {"type": "str", "required": False, "default": "sand_medium", "allowed_values": ["sand_loose", "sand_medium", "sand_dense", "gravel", "clay_stiff", "clay_hard", "rock_soft", "rock_medium", "rock_hard"], "description": "Soil/rock type for bond stress lookup (or give bond_stress_kPa directly)."}, "tendon_type": {"type": "str", "required": False, "default": "strand_15mm", "allowed_values": ["strand_13mm", "strand_15mm", "bar_26mm", "bar_32mm", "bar_36mm", "bar_44mm"], "description": "Tendon type."}, "anchor_angle_deg": {"type": "float", "required": False, "default": 15.0, "description": "Anchor inclination below horizontal (deg)."}, "bond_stress_kPa": {"type": "float", "required": False, "description": "Override ultimate bond stress (kPa)."}, "drill_diameter_mm": {"type": "float", "required": False, "default": 150.0, "description": "Drill hole diameter (mm)."}, "FOS_bond": {"type": "float", "required": False, "default": 2.0, "description": "FOS on bond stress."}, "lock_off_pct": {"type": "float", "required": False, "default": 0.80, "description": "Lock-off load fraction of design load."}, "max_load_pct": {"type": "float", "required": False, "default": 0.60, "description": "Max tendon load fraction of ultimate strength."}},
        "returns": {"unbonded_length_m": "Unbonded length.", "bond_length_m": "Bond length.", "total_length_m": "Total anchor length."}},
}

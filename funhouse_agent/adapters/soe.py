"""Support of excavation adapter — braced/cantilever, stability, anchors."""

from soe import (
    ExcavationGeometry, SOEWallLayer, SupportLevel,
    select_apparent_pressure, analyze_braced_excavation, analyze_cantilever_excavation,
    select_hp_section, select_sheet_pile, select_w_section,
    check_basal_heave_terzaghi, check_basal_heave_bjerrum_eide,
    check_bottom_blowout, check_piping, design_ground_anchor,
)


def _build_geometry(params):
    layers = [SOEWallLayer(
        thickness=l["thickness"], unit_weight=l["unit_weight"],
        friction_angle=l.get("friction_angle", 30.0), cohesion=l.get("cohesion", 0.0),
        soil_type=l.get("soil_type", "sand"), description=l.get("description", ""),
    ) for l in params["layers"]]
    supports = [SupportLevel(
        depth=s["depth"], support_type=s.get("support_type", "strut"),
        spacing=s.get("spacing", 3.0), angle_deg=s.get("angle_deg", 0.0),
        preload_kN_per_m=s.get("preload_kN_per_m", 0.0),
    ) for s in params.get("supports", [])]
    return ExcavationGeometry(
        excavation_depth=params["excavation_depth"], soil_layers=layers,
        support_levels=supports, surcharge=params.get("surcharge", 10.0),
        gwt_depth=params.get("gwt_depth"), excavation_width=params.get("excavation_width", 0.0),
    )


def _run_braced_excavation(p): return analyze_braced_excavation(_build_geometry(p), Fy=p.get("Fy", 345.0)).to_dict()
def _run_cantilever_excavation(p): return analyze_cantilever_excavation(_build_geometry(p), FOS_passive=p.get("FOS_passive", 1.5), Fy=p.get("Fy", 345.0)).to_dict()


def _run_apparent_pressure(params):
    layers = [SOEWallLayer(thickness=l["thickness"], unit_weight=l["unit_weight"],
                            friction_angle=l.get("friction_angle", 30.0), cohesion=l.get("cohesion", 0.0),
                            soil_type=l.get("soil_type", "sand")) for l in params["layers"]]
    return select_apparent_pressure(layers, params["excavation_depth"], params.get("surcharge", 0.0))


def _run_select_wall_section(params):
    t = params.get("section_type", "hp")
    sx = params["required_Sx_cm3"]
    if t == "hp": r = select_hp_section(sx)
    elif t == "sheet_pile": r = select_sheet_pile(sx)
    elif t == "w": r = select_w_section(sx)
    else: return {"error": f"Unknown section_type '{t}'."}
    return r if r else {"error": f"No {t} section for Sx = {sx} cm3"}


def _run_check_basal_heave(params):
    m = params.get("method", "terzaghi")
    if m == "terzaghi":
        return check_basal_heave_terzaghi(H=params["H"], cu=params["cu"], gamma=params["gamma"],
                                           q_surcharge=params.get("q_surcharge", 0.0), B=params.get("B", 0.0),
                                           FOS_required=params.get("FOS_required", 1.5)).to_dict()
    elif m == "bjerrum_eide":
        return check_basal_heave_bjerrum_eide(H=params["H"], cu=params["cu"], gamma=params["gamma"],
                                               Be=params["Be"], Le=params["Le"],
                                               q_surcharge=params.get("q_surcharge", 0.0),
                                               FOS_required=params.get("FOS_required", 1.5)).to_dict()
    return {"error": f"Unknown method '{m}'."}


def _run_check_bottom_blowout(p):
    return check_bottom_blowout(D_embed=p["D_embed"], hw_excess=p["hw_excess"], gamma_soil=p["gamma_soil"],
                                 gamma_w=p.get("gamma_w", 9.81), FOS_required=p.get("FOS_required", 1.5)).to_dict()


def _run_check_piping(p):
    return check_piping(delta_h=p["delta_h"], flow_path=p["flow_path"], Gs=p.get("Gs", 2.65),
                         void_ratio=p.get("void_ratio", 0.65), FOS_required=p.get("FOS_required", 2.0)).to_dict()


def _run_design_ground_anchor(p):
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
}

METHOD_INFO = {
    "braced_excavation": {"category": "Braced Excavation", "brief": "Multi-level braced excavation analysis (Terzaghi-Peck).",
        "parameters": {"excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."}, "layers": {"type": "array", "required": True, "description": "Soil layers."}, "supports": {"type": "array", "required": False, "description": "Support levels [{depth, support_type, spacing}]."}},
        "returns": {"n_support_levels": "Number of supports.", "support_reactions": "Support forces.", "max_moment_kNm_per_m": "Max wall moment."}},
    "cantilever_excavation": {"category": "Cantilever Excavation", "brief": "Cantilever (unbraced) excavation wall analysis.",
        "parameters": {"excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."}, "layers": {"type": "array", "required": True, "description": "Soil layers."}},
        "returns": {"embedment_depth_m": "Required embedment.", "max_moment_kNm_per_m": "Maximum moment."}},
    "apparent_pressure": {"category": "Earth Pressure", "brief": "Terzaghi-Peck apparent earth pressure envelope.",
        "parameters": {"excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."}, "layers": {"type": "array", "required": True, "description": "Soil layers."}},
        "returns": {"type": "Envelope type.", "max_pressure_kPa": "Maximum pressure."}},
    "select_wall_section": {"category": "Section Selection", "brief": "Select lightest HP/sheet/W section for demand.",
        "parameters": {"required_Sx_cm3": {"type": "float", "required": True, "description": "Required section modulus (cm3)."}, "section_type": {"type": "str", "required": False, "default": "hp", "description": "hp/sheet_pile/w."}},
        "returns": {"designation": "Section designation.", "Sx_cm3": "Section modulus."}},
    "check_basal_heave": {"category": "Stability", "brief": "Basal heave stability check (Terzaghi or Bjerrum-Eide).",
        "parameters": {"H": {"type": "float", "required": True, "description": "Excavation depth (m)."}, "cu": {"type": "float", "required": True, "description": "Undrained shear strength (kPa)."}, "gamma": {"type": "float", "required": True, "description": "Soil unit weight (kN/m3)."}, "method": {"type": "str", "required": False, "default": "terzaghi", "description": "terzaghi or bjerrum_eide."}},
        "returns": {"FOS": "Factor of safety.", "is_stable": "Pass/fail."}},
    "check_bottom_blowout": {"category": "Stability", "brief": "Bottom blowout (hydraulic uplift) check.",
        "parameters": {"D_embed": {"type": "float", "required": True, "description": "Wall embedment below excavation (m)."}, "hw_excess": {"type": "float", "required": True, "description": "Excess water head (m)."}, "gamma_soil": {"type": "float", "required": True, "description": "Soil unit weight (kN/m3)."}},
        "returns": {"FOS": "Factor of safety."}},
    "check_piping": {"category": "Stability", "brief": "Piping (internal erosion) check.",
        "parameters": {"delta_h": {"type": "float", "required": True, "description": "Head difference (m)."}, "flow_path": {"type": "float", "required": True, "description": "Seepage path length (m)."}},
        "returns": {"FOS": "Factor of safety.", "i_exit": "Exit gradient."}},
    "design_ground_anchor": {"category": "Anchors", "brief": "Ground anchor design per GEC-4/PTI.",
        "parameters": {"design_load_kN": {"type": "float", "required": True, "description": "Design anchor load (kN)."}, "anchor_depth": {"type": "float", "required": True, "description": "Anchor depth (m)."}, "excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."}, "phi_deg": {"type": "float", "required": True, "description": "Soil friction angle (degrees)."}},
        "returns": {"unbonded_length_m": "Unbonded length.", "bond_length_m": "Bond length.", "total_length_m": "Total anchor length."}},
}

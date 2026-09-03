"""Section properties adapter — flat dict -> section_props_agent -> dict."""

from funhouse_agent.adapters import (
    clean_result, reject_unknown_params, require_params,
)

_SHAPES = ["rectangle", "circle", "chs", "rhs", "i_section"]

_ERR = ("sectionproperties is not installed. "
        "Install with: pip install sectionproperties")


def _run_section_properties(params: dict) -> dict:
    from section_props_agent import analyze_section, has_sectionproperties
    if not has_sectionproperties():
        return {"error": _ERR}
    reject_unknown_params(
        params,
        ("shape", "d", "b", "t", "t_f", "t_w", "r", "r_out", "mesh_size",
         "warping"),
        method="section_properties")
    require_params(params, ["shape"], method="section_properties")
    dims = {k: params[k] for k in ("d", "b", "t", "t_f", "t_w", "r", "r_out")
            if k in params}
    result = analyze_section(
        params["shape"], mesh_size=params.get("mesh_size"),
        warping=params.get("warping", True), **dims)
    return clean_result(result.to_dict())


def _run_polygon_section(params: dict) -> dict:
    from section_props_agent import (
        analyze_polygon_section, has_sectionproperties)
    if not has_sectionproperties():
        return {"error": _ERR}
    reject_unknown_params(
        params, ("points", "mesh_size", "warping"), method="polygon_section")
    require_params(params, ["points"], method="polygon_section")
    result = analyze_polygon_section(
        params["points"], mesh_size=params.get("mesh_size"),
        warping=params.get("warping", True))
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "section_properties": _run_section_properties,
    "polygon_section": _run_polygon_section,
}

METHOD_INFO = {
    "section_properties": {
        "category": "Cross-Section Analysis",
        "brief": "Properties of parametric shapes (rectangle/circle/CHS/RHS/I): A, I, Z, S, r, J, warping. Dimensions in mm.",
        "parameters": {
            "shape": {"type": "str", "required": True, "allowed_values": _SHAPES, "description": "Section shape."},
            "d": {"type": "float", "required": False, "description": "Depth or diameter (mm). Required for all shapes."},
            "b": {"type": "float", "required": False, "description": "Width (mm) — rectangle, rhs, i_section."},
            "t": {"type": "float", "required": False, "description": "Wall thickness (mm) — chs, rhs."},
            "t_f": {"type": "float", "required": False, "description": "Flange thickness (mm) — i_section."},
            "t_w": {"type": "float", "required": False, "description": "Web thickness (mm) — i_section."},
            "r": {"type": "float", "required": False, "default": 0.0, "description": "Root radius (mm) — i_section."},
            "r_out": {"type": "float", "required": False, "description": "Outer corner radius (mm) — rhs. Default 2t."},
            "mesh_size": {"type": "float", "required": False, "description": "FE mesh element area target (mm^2). Auto if omitted."},
            "warping": {"type": "bool", "required": False, "default": True, "description": "Compute J and warping constant (slower)."},
        },
        "returns": {"area_mm2": "Area.", "ixx_mm4": "Centroidal Ixx.",
                    "zxx_plus_mm3": "Elastic modulus.", "sxx_mm3": "Plastic modulus.",
                    "j_mm4": "Torsion constant."},
    },
    "polygon_section": {
        "category": "Cross-Section Analysis",
        "brief": "Properties of an arbitrary closed polygon section from vertex coordinates (mm).",
        "parameters": {
            "points": {"type": "array", "required": True, "description": "Vertices [[x, y], ...] in mm, in order."},
            "mesh_size": {"type": "float", "required": False, "description": "FE mesh element area target (mm^2)."},
            "warping": {"type": "bool", "required": False, "default": True, "description": "Compute J and warping constant."},
        },
        "returns": {"area_mm2": "Area.", "ixx_mm4": "Centroidal Ixx.",
                    "i11_mm4": "Principal I11.", "phi_deg": "Principal axis angle."},
    },
}

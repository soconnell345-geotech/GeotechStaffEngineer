"""PyNite adapter — flat dict -> pynite_agent -> dict."""

from funhouse_agent.adapters import (
    clean_result, reject_unknown_params, require_params,
)

_ERR = "PyNiteFEA is not installed. Install with: pip install PyNiteFEA"

_SUPPORT_TYPES = ["fixed", "pinned", "roller_y", "roller_x", "free"]


def _run_frame(params: dict) -> dict:
    from pynite_agent import analyze_frame, has_pynite
    if not has_pynite():
        return {"error": _ERR}
    reject_unknown_params(
        params,
        ("nodes", "members", "supports", "nodal_loads",
         "member_dist_loads", "member_point_loads", "auto_stabilize_2d"),
        method="frame_analysis")
    require_params(params, ["nodes", "members", "supports"],
                   method="frame_analysis")
    result = analyze_frame(
        nodes=params["nodes"], members=params["members"],
        supports=params["supports"],
        nodal_loads=params.get("nodal_loads"),
        member_dist_loads=params.get("member_dist_loads"),
        member_point_loads=params.get("member_point_loads"),
        auto_stabilize_2d=params.get("auto_stabilize_2d", True),
    )
    return clean_result(result.to_dict())


def _run_continuous_beam(params: dict) -> dict:
    from pynite_agent import analyze_continuous_beam, has_pynite
    if not has_pynite():
        return {"error": _ERR}
    reject_unknown_params(
        params,
        ("span_lengths", "E", "I", "A", "udl", "span_udls", "point_loads",
         "support_types"),
        method="continuous_beam")
    require_params(params, ["span_lengths", "E", "I"],
                   method="continuous_beam")
    result = analyze_continuous_beam(
        span_lengths_m=params["span_lengths"],
        E_kPa=params["E"], I_m4=params["I"],
        A_m2=params.get("A", 0.01),
        udl_kN_m=params.get("udl", 0.0),
        span_udls_kN_m=params.get("span_udls"),
        point_loads=params.get("point_loads"),
        support_types=params.get("support_types"),
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "frame_analysis": _run_frame,
    "continuous_beam": _run_continuous_beam,
}

METHOD_INFO = {
    "frame_analysis": {
        "category": "Frame Analysis",
        "brief": "Linear-elastic 2D/3D frame: nodes/members/supports/loads -> reactions + member force/deflection envelopes. SI: m, kN, kPa.",
        "parameters": {
            "nodes": {"type": "array", "required": True, "description": "[{name, x, y[, z]}] coordinates in m."},
            "members": {"type": "array", "required": True, "description": "[{name, i, j, E (kPa), A (m2), Iz (m4)[, Iy, J, G, nu]}]."},
            "supports": {"type": "array", "required": True, "description": "[{node, type: fixed|pinned|roller_y|roller_x}] or explicit dx..rz booleans."},
            "nodal_loads": {"type": "array", "required": False, "description": "[{node, direction: FX|FY|FZ|MX|MY|MZ, value (kN or kN*m)}]. -FY = downward."},
            "member_dist_loads": {"type": "array", "required": False, "description": "[{member, direction (default FY), w (kN/m) or w1/w2[, x1, x2]}]. Negative FY = downward."},
            "member_point_loads": {"type": "array", "required": False, "description": "[{member, direction, value (kN), x (m from i-node)}]."},
            "auto_stabilize_2d": {"type": "bool", "required": False, "default": True, "description": "Restrain out-of-plane DOFs for planar (z=0) models."},
        },
        "returns": {
            "reactions": "Per supported node: FX/FY/FZ (kN), MX/MY/MZ (kN*m).",
            "members": "Per member: moment/shear/axial envelopes + deflection.",
        },
    },
    "continuous_beam": {
        "category": "Frame Analysis",
        "brief": "Multi-span continuous beam: reactions, support (hogging) and span (sagging) moments, deflections. Sagging positive.",
        "parameters": {
            "span_lengths": {"type": "array", "required": True, "description": "Span lengths (m), left to right."},
            "E": {"type": "float", "required": True, "description": "Elastic modulus (kPa; steel 200e6)."},
            "I": {"type": "float", "required": True, "description": "Second moment of area (m^4)."},
            "A": {"type": "float", "required": False, "default": 0.01, "description": "Section area (m^2)."},
            "udl": {"type": "float", "required": False, "default": 0.0, "description": "UDL on all spans, downward positive (kN/m)."},
            "span_udls": {"type": "array", "required": False, "description": "Per-span UDLs (kN/m, downward positive); overrides udl."},
            "point_loads": {"type": "array", "required": False, "description": "[{span (1-based), x (m from span start), P (kN downward)}]."},
            "support_types": {"type": "array", "required": False, "allowed_values": _SUPPORT_TYPES, "description": "One per support; default pinned + rollers."},
        },
        "returns": {
            "support_reactions_kN": "Vertical reactions at each support.",
            "support_moments_kNm": "Moments over supports (hogging negative).",
            "span_max_sagging_kNm": "Max sagging moment per span.",
            "max_deflection_mm": "Largest deflection.",
        },
    },
}

"""Pile group adapter — rigid cap analysis, 6-DOF, efficiency."""

from funhouse_agent.adapters import require_keys, require_params
from pile_group import (
    GroupPile, create_rectangular_layout, converse_labarre,
    block_failure_capacity, p_multiplier, GroupLoad,
    analyze_vertical_group_simple, analyze_group_6dof,
)


def _build_piles(params):
    if "n_rows" in params and "n_cols" in params:
        # Accept a single 'spacing' for both directions.
        sx = params.get("spacing_x", params.get("spacing"))
        sy = params.get("spacing_y", params.get("spacing"))
        if sx is None or sy is None:
            raise ValueError(
                "Rectangular layout requires spacing_x and spacing_y (m), "
                "or a single 'spacing' applied to both directions."
            )
        piles = create_rectangular_layout(
            n_rows=params["n_rows"], n_cols=params["n_cols"],
            spacing_x=sx, spacing_y=sy,
            axial_stiffness=params.get("axial_stiffness"), lateral_stiffness=params.get("lateral_stiffness"),
        )
        for p in piles:
            if params.get("axial_capacity_compression"):
                p.axial_capacity_compression = params["axial_capacity_compression"]
            if params.get("axial_capacity_tension"):
                p.axial_capacity_tension = params["axial_capacity_tension"]
        return piles
    elif "piles" in params:
        for pd in params["piles"]:
            require_keys(pd, ["x", "y"], method="pile_group", item_label="piles[]")
        return [GroupPile(
            x=pd["x"], y=pd["y"], batter_x=pd.get("batter_x", 0.0), batter_y=pd.get("batter_y", 0.0),
            axial_stiffness=pd.get("axial_stiffness"), lateral_stiffness=pd.get("lateral_stiffness"),
            axial_capacity_compression=pd.get("axial_capacity_compression"),
            axial_capacity_tension=pd.get("axial_capacity_tension"), label=pd.get("label", ""),
        ) for pd in params["piles"]]
    raise ValueError("Provide n_rows/n_cols for rectangular layout or 'piles' array.")


def _build_load(params):
    return GroupLoad(Vx=params.get("Vx", 0.0), Vy=params.get("Vy", 0.0), Vz=params.get("Vz", 0.0),
                     Mx=params.get("Mx", 0.0), My=params.get("My", 0.0), Mz=params.get("Mz", 0.0))


def _run_pile_group_simple(params):
    return analyze_vertical_group_simple(_build_piles(params), _build_load(params)).to_dict()


def _run_pile_group_6dof(params):
    return analyze_group_6dof(_build_piles(params), _build_load(params)).to_dict()


def _run_group_efficiency(params):
    results = {}
    if "n_rows" in params and "n_cols" in params and "pile_diameter" in params:
        s = params.get("spacing", params.get("spacing_x"))
        if s is None:
            raise ValueError(
                "group_efficiency: missing 'spacing' (center-to-center, m). "
                "Required with n_rows/n_cols/pile_diameter for Converse-Labarre."
            )
        results["converse_labarre_Eg"] = round(converse_labarre(
            params["n_rows"], params["n_cols"], params["pile_diameter"], s), 4)
    if "pile_length" in params and "cu" in params:
        results["block_failure_kN"] = round(block_failure_capacity(
            params["n_rows"], params["n_cols"], params.get("spacing_x", params.get("spacing", 0)),
            params.get("spacing_y", params.get("spacing", 0)), params["pile_length"],
            params["cu"], params["pile_diameter"]), 1)
    if "row_position" in params:
        sd = params.get("spacing_diameter_ratio", params.get("spacing", 3.0) / params.get("pile_diameter", 0.3))
        results["p_multiplier"] = round(p_multiplier(params["row_position"], sd), 3)
    return results


METHOD_REGISTRY = {
    "pile_group_simple": _run_pile_group_simple,
    "pile_group_6dof": _run_pile_group_6dof,
    "group_efficiency": _run_group_efficiency,
}

METHOD_INFO = {
    "pile_group_simple": {
        "category": "Pile Group",
        "brief": "Simplified elastic vertical group analysis.",
        "parameters": {
            "n_rows": {"type": "int", "required": False, "description": "Rows for rectangular layout."},
            "n_cols": {"type": "int", "required": False, "description": "Columns for rectangular layout."},
            "spacing_x": {"type": "float", "required": False, "description": "X spacing (m). Or give a single 'spacing' for both directions."},
            "spacing_y": {"type": "float", "required": False, "description": "Y spacing (m). Or give a single 'spacing' for both directions."},
            "spacing": {"type": "float", "required": False, "description": "Uniform center-to-center spacing (m), used for both x and y."},
            "piles": {"type": "array", "required": False, "description": "Explicit pile list [{x, y, batter_x, batter_y, ...}] instead of n_rows/n_cols."},
            "Vz": {"type": "float", "required": True, "description": "Vertical load (kN)."},
            "Mx": {"type": "float", "required": False, "description": "Moment about x (kN-m)."},
            "My": {"type": "float", "required": False, "description": "Moment about y (kN-m)."},
        },
        "returns": {"pile_forces": "Per-pile axial forces.", "max_compression_kN": "Max compression."},
    },
    "pile_group_6dof": {
        "category": "Pile Group",
        "brief": "General 6-DOF rigid cap analysis (axial + lateral + moments).",
        "parameters": {
            "n_rows": {"type": "int", "required": False, "description": "Rows for rectangular layout."},
            "n_cols": {"type": "int", "required": False, "description": "Columns."},
            "spacing_x": {"type": "float", "required": False, "description": "X spacing (m). Or give a single 'spacing' for both directions."},
            "spacing_y": {"type": "float", "required": False, "description": "Y spacing (m). Or give a single 'spacing' for both directions."},
            "spacing": {"type": "float", "required": False, "description": "Uniform center-to-center spacing (m), used for both x and y."},
            "piles": {"type": "array", "required": False, "description": "Explicit pile list [{x, y, batter_x, batter_y, ...}] instead of n_rows/n_cols."},
            "Vx": {"type": "float", "required": False, "description": "Lateral load x (kN)."},
            "Vy": {"type": "float", "required": False, "description": "Lateral load y (kN)."},
            "Vz": {"type": "float", "required": False, "description": "Vertical load (kN)."},
        },
        "returns": {"pile_forces": "Per-pile forces.", "cap_displacement": "Cap displacement."},
    },
    "group_efficiency": {
        "category": "Pile Group",
        "brief": "Group efficiency: Converse-Labarre, block failure, p-multiplier.",
        "parameters": {
            "n_rows": {"type": "int", "required": True, "description": "Number of rows."},
            "n_cols": {"type": "int", "required": True, "description": "Number of columns."},
            "pile_diameter": {"type": "float", "required": True, "description": "Pile diameter (m)."},
            "spacing": {"type": "float", "required": True, "description": "Center-to-center spacing (m)."},
        },
        "returns": {"converse_labarre_Eg": "Group efficiency factor."},
    },
}

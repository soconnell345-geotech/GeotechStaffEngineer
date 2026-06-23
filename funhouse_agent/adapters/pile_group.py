"""Pile group adapter — rigid cap analysis, 6-DOF, efficiency."""

from funhouse_agent.adapters import (
    apply_aliases, reject_unknown_params, require_keys, require_params,
)
from pile_group import (
    GroupPile, create_rectangular_layout, converse_labarre,
    block_failure_capacity, p_multiplier, GroupLoad,
    analyze_vertical_group_simple, analyze_group_6dof,
    meyerhof_group_settlement,
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


# Every top-level parameter the group-analysis methods consume.
_GROUP_VALID_PARAMS = (
    "n_rows", "n_cols", "spacing_x", "spacing_y", "spacing", "piles",
    "axial_stiffness", "lateral_stiffness", "axial_capacity_compression",
    "axial_capacity_tension", "Vx", "Vy", "Vz", "Mx", "My", "Mz",
)


def _run_pile_group_simple(params):
    reject_unknown_params(params, _GROUP_VALID_PARAMS, method="pile_group_simple")
    return analyze_vertical_group_simple(_build_piles(params), _build_load(params)).to_dict()


def _run_pile_group_6dof(params):
    reject_unknown_params(params, _GROUP_VALID_PARAMS, method="pile_group_6dof")
    return analyze_group_6dof(_build_piles(params), _build_load(params)).to_dict()


def _run_group_efficiency(params):
    reject_unknown_params(
        params,
        ("n_rows", "n_cols", "pile_diameter", "spacing", "spacing_x",
         "spacing_y", "pile_length", "cu", "row_position",
         "spacing_diameter_ratio"),
        method="group_efficiency")
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
        require_params(params, ["n_rows", "n_cols", "pile_diameter"],
                       method="group_efficiency (block failure)",
                       valid=["n_rows", "n_cols", "pile_diameter", "spacing",
                              "spacing_x", "spacing_y", "pile_length", "cu"])
        results["block_failure_kN"] = round(block_failure_capacity(
            params["n_rows"], params["n_cols"], params.get("spacing_x", params.get("spacing", 0)),
            params.get("spacing_y", params.get("spacing", 0)), params["pile_length"],
            params["cu"], params["pile_diameter"]), 1)
    if "row_position" in params:
        sd = params.get("spacing_diameter_ratio", params.get("spacing", 3.0) / params.get("pile_diameter", 0.3))
        results["p_multiplier"] = round(p_multiplier(params["row_position"], sd), 3)
    return results


# Meyerhof (1976) SPT group-settlement aliases (accept a few common synonyms
# for the SI plan dimensions / pressure / SPT count without silently dropping).
_MEYERHOF_ALIASES = {
    "B": "group_width", "width": "group_width",
    "L": "group_length", "Z": "group_length", "length": "group_length",
    "Q": "load_kN", "load": "load_kN", "load_permanent_kN": "load_kN",
    "pf": "pf_kPa", "net_pressure_kPa": "pf_kPa",
    "DB": "embedment_DB", "embedment": "embedment_DB", "embedment_m": "embedment_DB",
    "N": "N160", "N1_60": "N160", "N1_60_avg": "N160",
}

_MEYERHOF_VALID_PARAMS = (
    "group_width", "group_length", "N160", "load_kN", "pf_kPa", "embedment_DB",
)


def _run_meyerhof_group_settlement(params):
    params = apply_aliases(params, _MEYERHOF_ALIASES)
    reject_unknown_params(params, _MEYERHOF_VALID_PARAMS,
                          method="meyerhof_group_settlement")
    require_params(params, ["group_width", "group_length", "N160"],
                   method="meyerhof_group_settlement",
                   valid=_MEYERHOF_VALID_PARAMS)
    if params.get("load_kN") is None and params.get("pf_kPa") is None:
        raise ValueError(
            "meyerhof_group_settlement: provide either load_kN (total "
            "permanent load, kN) or pf_kPa (net foundation pressure, kPa)."
        )
    return meyerhof_group_settlement(
        group_width=params["group_width"],
        group_length=params["group_length"],
        N160=params["N160"],
        load_kN=params.get("load_kN"),
        pf_kPa=params.get("pf_kPa"),
        embedment_DB=params.get("embedment_DB", 0.0),
    )


METHOD_REGISTRY = {
    "pile_group_simple": _run_pile_group_simple,
    "pile_group_6dof": _run_pile_group_6dof,
    "group_efficiency": _run_group_efficiency,
    "meyerhof_group_settlement": _run_meyerhof_group_settlement,
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
            "axial_capacity_compression": {"type": "float", "required": False, "description": "Per-pile compression capacity (kN) for utilization checks (rectangular layout)."},
            "axial_capacity_tension": {"type": "float", "required": False, "description": "Per-pile tension capacity (kN) (rectangular layout)."},
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
            "Mx": {"type": "float", "required": False, "description": "Moment about x (kN-m)."},
            "My": {"type": "float", "required": False, "description": "Moment about y (kN-m)."},
            "Mz": {"type": "float", "required": False, "description": "Torsion about z (kN-m)."},
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
            "pile_length": {"type": "float", "required": False, "description": "Pile length (m). With cu, also computes block failure capacity."},
            "cu": {"type": "float", "required": False, "description": "Undrained shear strength (kPa) for block failure."},
            "row_position": {"type": "int", "required": False, "description": "Row number (1 = lead) — returns the lateral p-multiplier."},
            "spacing_diameter_ratio": {"type": "float", "required": False, "description": "s/D ratio for the p-multiplier (computed from spacing/pile_diameter if omitted)."},
        },
        "returns": {"converse_labarre_Eg": "Group efficiency factor.", "block_failure_kN": "Block failure capacity (with pile_length + cu).", "p_multiplier": "Lateral p-multiplier (with row_position)."},
    },
    "meyerhof_group_settlement": {
        "category": "Pile Group",
        "brief": "Meyerhof (1976) SPT pile-group settlement S=4·pf·If·√B/N160 (GEC-12). SI in, mm out.",
        "parameters": {
            "group_width": {"type": "float", "required": True, "description": "Group plan width B, the SMALLER plan dimension (m)."},
            "group_length": {"type": "float", "required": True, "description": "Group plan length Z, the larger plan dimension (m). Used with group_width to form the plan area when load_kN is given."},
            "N160": {"type": "float", "required": True, "description": "Average corrected SPT (N1)60 within ~B below the pile-group toe (dimensionless)."},
            "load_kN": {"type": "float", "required": False, "description": "Total unfactored permanent load Q (kN); net pressure pf = Q/(B·Z). Provide this OR pf_kPa."},
            "pf_kPa": {"type": "float", "required": False, "description": "Net foundation pressure pf (kPa), supplied directly. Takes precedence over load_kN. Provide this OR load_kN."},
            "embedment_DB": {"type": "float", "required": False, "description": "Embedment depth of the group into the bearing stratum DB (m); If = 1 − (2/3·DB)/(8·B), floored at 0.5. Default 0.0 (If = 1.0)."},
        },
        "returns": {"settlement_mm": "Group settlement (mm).", "settlement_in": "Group settlement (inches).", "influence_factor": "Embedment factor If.", "pf_kPa": "Net foundation pressure used (kPa)."},
    },
}

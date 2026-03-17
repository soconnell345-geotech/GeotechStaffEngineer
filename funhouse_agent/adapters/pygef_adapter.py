"""pygef adapter — CPT and borehole file parsing via pygef."""

from funhouse_agent.adapters import clean_result


def _run_parse_cpt(params):
    from pygef_agent import has_pygef, parse_cpt_file

    if not has_pygef():
        return {"error": "pygef is not installed. Install with: pip install pygef"}

    file_path = params.get("file_path")
    engine = params.get("engine", "auto")
    index = params.get("index", 0)

    result = parse_cpt_file(
        file_path=file_path,
        engine=engine,
        index=index,
    )
    return clean_result(result.to_dict())


def _run_parse_borehole(params):
    from pygef_agent import has_pygef, parse_bore_file

    if not has_pygef():
        return {"error": "pygef is not installed. Install with: pip install pygef"}

    file_path = params.get("file_path")
    engine = params.get("engine", "auto")
    index = params.get("index", 0)

    result = parse_bore_file(
        file_path=file_path,
        engine=engine,
        index=index,
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "parse_cpt": _run_parse_cpt,
    "parse_borehole": _run_parse_borehole,
}

METHOD_INFO = {
    "parse_cpt": {
        "category": "File Import",
        "brief": "Parse a CPT file (GEF or BRO-XML) into depth/qc/fs/u2 arrays (kPa).",
        "parameters": {
            "file_path": {"type": "str", "required": True, "description": "Path to CPT file (.gef or .xml)."},
            "engine": {"type": "str", "required": False, "default": "auto", "description": "Parser engine: 'auto', 'gef', or 'xml'."},
            "index": {"type": "int", "required": False, "default": 0, "description": "Record index for multi-record XML files."},
        },
        "returns": {
            "n_points": "Number of data points.",
            "alias": "Test ID or filename.",
            "final_depth_m": "Final penetration depth (m).",
            "gwl_m": "Groundwater level (m below surface) or null.",
            "depth_m": "Depth array (m).",
            "q_c_kPa": "Cone tip resistance array (kPa).",
            "f_s_kPa": "Sleeve friction array (kPa).",
            "u_2_kPa": "Pore pressure u2 array (kPa).",
            "Rf_pct": "Friction ratio array (%).",
        },
    },
    "parse_borehole": {
        "category": "File Import",
        "brief": "Parse a borehole file (GEF or BRO-XML) into layer descriptions.",
        "parameters": {
            "file_path": {"type": "str", "required": True, "description": "Path to borehole file (.gef or .xml)."},
            "engine": {"type": "str", "required": False, "default": "auto", "description": "Parser engine: 'auto', 'gef', or 'xml'."},
            "index": {"type": "int", "required": False, "default": 0, "description": "Record index for multi-record XML files."},
        },
        "returns": {
            "n_layers": "Number of soil layers.",
            "alias": "Borehole ID or filename.",
            "final_depth_m": "Total bore depth (m).",
            "gwl_m": "Groundwater level (m below surface) or null.",
            "layers": "List of layer dicts with top_m, bottom_m, soil_name, soil_code.",
        },
    },
}

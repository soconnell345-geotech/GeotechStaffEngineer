"""DXF import adapter — discover layers, parse geometry, build slope/FEM inputs."""

from funhouse_agent.adapters import clean_result


def _run_discover_layers(params):
    from dxf_import import discover_layers

    filepath = params.get("file_path")
    result = discover_layers(filepath=filepath)
    return clean_result(result.to_dict())


def _run_parse_dxf_geometry(params):
    from dxf_import import parse_dxf_geometry, LayerMapping

    filepath = params.get("file_path")
    units = params.get("units", "m")
    flip_y = params.get("flip_y", False)

    # Build LayerMapping from flat dict params
    mapping_dict = params.get("layer_mapping", {})
    mapping = LayerMapping(
        surface=mapping_dict.get("surface", ""),
        soil_boundaries=mapping_dict.get("soil_boundaries", {}),
        water_table=mapping_dict.get("water_table"),
        nails=mapping_dict.get("nails"),
        annotations=mapping_dict.get("annotations"),
    )

    result = parse_dxf_geometry(
        filepath=filepath,
        layer_mapping=mapping,
        units=units,
        flip_y=flip_y,
    )
    return clean_result(result.to_dict())


def _run_build_slope_geometry(params):
    from dxf_import import build_slope_geometry, SoilPropertyAssignment
    from dxf_import.results import DxfParseResult

    # Reconstruct DxfParseResult from dict
    pr = params.get("parse_result", {})
    surface_points = [
        (p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
        for p in pr.get("surface_points", [])
    ]
    boundary_profiles = {}
    for name, pts in pr.get("boundary_profiles", {}).items():
        boundary_profiles[name] = [
            (p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
            for p in pts
        ]
    gwt_points = None
    if pr.get("gwt_points"):
        gwt_points = [
            (p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
            for p in pr["gwt_points"]
        ]
    nail_lines = pr.get("nail_lines", [])
    text_annotations = pr.get("text_annotations", [])

    parse_result = DxfParseResult(
        surface_points=surface_points,
        boundary_profiles=boundary_profiles,
        gwt_points=gwt_points,
        nail_lines=nail_lines,
        text_annotations=text_annotations,
        units_used=pr.get("units_used", "m"),
        warnings=pr.get("warnings", []),
    )

    # Build SoilPropertyAssignment list
    soil_props = []
    for sp_dict in params.get("soil_properties", []):
        soil_props.append(SoilPropertyAssignment(
            name=sp_dict.get("name", ""),
            gamma=sp_dict.get("gamma", 18.0),
            gamma_sat=sp_dict.get("gamma_sat"),
            phi=sp_dict.get("phi", 0.0),
            c_prime=sp_dict.get("c_prime", 0.0),
            cu=sp_dict.get("cu", 0.0),
            analysis_mode=sp_dict.get("analysis_mode", "drained"),
        ))

    nail_defaults = params.get("nail_defaults")

    geom = build_slope_geometry(
        parse_result=parse_result,
        soil_properties=soil_props,
        nail_defaults=nail_defaults,
    )
    return clean_result(geom.to_dict())


def _run_build_fem_inputs(params):
    from dxf_import import build_fem_inputs, FEMSoilPropertyAssignment
    from dxf_import.results import DxfParseResult

    # Reconstruct DxfParseResult from dict
    pr = params.get("parse_result", {})
    surface_points = [
        (p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
        for p in pr.get("surface_points", [])
    ]
    boundary_profiles = {}
    for name, pts in pr.get("boundary_profiles", {}).items():
        boundary_profiles[name] = [
            (p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
            for p in pts
        ]
    gwt_points = None
    if pr.get("gwt_points"):
        gwt_points = [
            (p["x"], p["z"]) if isinstance(p, dict) else tuple(p)
            for p in pr["gwt_points"]
        ]

    parse_result = DxfParseResult(
        surface_points=surface_points,
        boundary_profiles=boundary_profiles,
        gwt_points=gwt_points,
        nail_lines=pr.get("nail_lines", []),
        text_annotations=pr.get("text_annotations", []),
        units_used=pr.get("units_used", "m"),
        warnings=pr.get("warnings", []),
    )

    # Build FEMSoilPropertyAssignment list
    soil_props = []
    for sp_dict in params.get("soil_properties", []):
        soil_props.append(FEMSoilPropertyAssignment(
            name=sp_dict.get("name", ""),
            gamma=sp_dict.get("gamma", 18.0),
            phi=sp_dict.get("phi", 0.0),
            c=sp_dict.get("c", 0.0),
            E=sp_dict.get("E", 30000.0),
            nu=sp_dict.get("nu", 0.3),
            psi=sp_dict.get("psi", 0.0),
            model=sp_dict.get("model", "mc"),
            hs_params=sp_dict.get("hs_params"),
        ))

    result = build_fem_inputs(
        parse_result=parse_result,
        soil_properties=soil_props,
    )
    return clean_result(result)


METHOD_REGISTRY = {
    "discover_layers": _run_discover_layers,
    "parse_geometry": _run_parse_dxf_geometry,
    "build_slope_geometry": _run_build_slope_geometry,
    "build_fem_inputs": _run_build_fem_inputs,
}

METHOD_INFO = {
    "discover_layers": {
        "category": "DXF Import",
        "brief": "Inventory all DXF layers with entity counts, types, and bounding boxes.",
        "parameters": {
            "file_path": {"type": "str", "required": True, "description": "Path to a DXF file."},
        },
        "returns": {
            "n_layers": "Number of layers with entities.",
            "n_total_entities": "Total entity count.",
            "units_hint": "Units detected from DXF header.",
            "layers": "List of layer info dicts (name, n_entities, entity_types, sample_texts, bbox).",
        },
    },
    "parse_geometry": {
        "category": "DXF Import",
        "brief": "Extract slope geometry coordinates from mapped DXF layers.",
        "parameters": {
            "file_path": {"type": "str", "required": True, "description": "Path to DXF file."},
            "layer_mapping": {"type": "dict", "required": True, "description": "Mapping with keys: surface (str), soil_boundaries ({layer: soil_name}), water_table (str), nails (str), annotations ([str])."},
            "units": {"type": "str", "required": False, "default": "m", "description": "Drawing units: 'm', 'mm', 'cm', 'ft', 'in'."},
            "flip_y": {"type": "bool", "required": False, "default": False, "description": "Negate Y values for downward-positive drawings."},
        },
        "returns": {
            "surface_points": "Ground surface as [{x, z}, ...].",
            "boundary_profiles": "Soil boundaries: {name: [{x, z}, ...]}.",
            "gwt_points": "GWT profile or null.",
            "nail_lines": "Nail geometry list.",
            "warnings": "Any parsing warnings.",
        },
    },
    "build_slope_geometry": {
        "category": "DXF Import",
        "brief": "Assemble SlopeGeometry from DXF parse result and soil properties.",
        "parameters": {
            "parse_result": {"type": "dict", "required": True, "description": "Output from parse_dxf_geometry (surface_points, boundary_profiles, etc.)."},
            "soil_properties": {"type": "list", "required": True, "description": "List of dicts: {name, gamma, gamma_sat, phi, c_prime, cu, analysis_mode}."},
            "nail_defaults": {"type": "dict", "required": False, "description": "Default nail params: {bar_diameter, drill_hole_diameter, fy, bond_stress, spacing_h}."},
        },
        "returns": {
            "surface_points": "Ground surface profile.",
            "soil_layers": "Assembled soil layers with geometry and properties.",
            "gwt_points": "GWT profile or null.",
        },
    },
    "build_fem_inputs": {
        "category": "DXF Import",
        "brief": "Convert DXF parse result to fem2d input format with FEM soil properties.",
        "parameters": {
            "parse_result": {"type": "dict", "required": True, "description": "Output from parse_dxf_geometry (surface_points, boundary_profiles, etc.)."},
            "soil_properties": {"type": "list", "required": True, "description": "List of dicts: {name, gamma, phi, c, E, nu, psi, model, hs_params}."},
        },
        "returns": {
            "surface_points": "Ground surface profile.",
            "soil_layers": "FEM layer dicts with geometry and material properties.",
            "gwt": "GWT array or null.",
            "boundary_polylines": "Boundary polylines for layer assignment.",
        },
    },
}

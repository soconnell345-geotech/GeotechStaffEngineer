"""DXF import adapter — discover layers, parse geometry, build slope/FEM inputs."""

from funhouse_agent.adapters import (
    check_nested_keys, check_nested_list, clean_result,
    reject_unknown_params, require_params,
)

# The keys this adapter reads from each nested dict: the LayerMapping
# fields, the nail_defaults build_slope_geometry reads (SoilNail's own
# names), and the soil_properties of a slope (SoilPropertyAssignment) and of
# an FEM model (FEMSoilPropertyAssignment). Anything else was read with .get
# and a default and silently left out (GM-58).
_MAPPING_KEYS = ("surface", "soil_boundaries", "water_table", "nails",
                 "annotations")
_NAIL_DEFAULT_KEYS = ("inclination", "bar_diameter", "drill_hole_diameter",
                      "fy", "bond_stress", "spacing_h")
_SOIL_PROPERTY_KEYS = {
    "slope": ("name", "gamma", "gamma_sat", "phi", "c_prime", "cu",
              "analysis_mode"),
    "fem": ("name", "gamma", "phi", "c", "E", "nu", "psi", "model",
            "hs_params"),
}


def _slope_params(geom) -> dict:
    """A SlopeGeometry as the slope_stability adapter's own parameters
    (surface_points, soil_layers, gwt_points and any nails), so the result is
    the next call's input: build_slope_geometry's own docstring says the
    geometry is "ready for analyze_slope() or search_critical_surface()".
    SlopeGeometry has no to_dict, so this adapter raised AttributeError on
    every real run; only mocked tests had called it."""
    import dataclasses

    def _pts(points):
        return [[float(x), float(z)] for x, z in points]

    layers = []
    for lyr in geom.soil_layers:
        d = {"name": lyr.name, "top_elevation": lyr.top_elevation,
             "bottom_elevation": lyr.bottom_elevation, "gamma": lyr.gamma,
             "phi": lyr.phi, "c_prime": lyr.c_prime, "cu": lyr.cu,
             "analysis_mode": lyr.analysis_mode}
        if lyr.gamma_sat is not None:
            d["gamma_sat"] = lyr.gamma_sat
        if lyr.bottom_boundary_points:
            d["bottom_boundary_points"] = _pts(lyr.bottom_boundary_points)
        layers.append(d)
    out = {"surface_points": _pts(geom.surface_points),
           "soil_layers": layers,
           "gwt_points": _pts(geom.gwt_points) if geom.gwt_points else None}
    if geom.nails:
        out["nails"] = [dataclasses.asdict(n) for n in geom.nails]
    return out


def check_soil_properties(items, kind, *, method):
    """Refuse a soil_properties[] key a ``kind`` ('slope' or 'fem')
    assignment does not take (GM-58): every one was read with .get and a
    default, so a slope layer's 'c' ran with c' = 0 and an FEM layer's
    'unit_weight' at gamma = 18. The did-you-means are the slope and fem2d
    adapters' own. Shared with the pdf_import adapter, which builds the same
    assignments."""
    if kind == "slope":
        from funhouse_agent.adapters.slope_stability import (
            _LAYER_KEY_HINTS as hints, _LAYER_KEY_NOTES as notes,
        )
    else:
        from funhouse_agent.adapters.fem2d_adapter import (
            _LAYER_KEY_HINTS as hints, _LAYER_KEY_NOTES as notes,
        )
    check_nested_list(items, _SOIL_PROPERTY_KEYS[kind], method=method,
                      label="soil_properties", hints=hints, notes=notes)


def _run_discover_layers(params):
    from dxf_import import discover_layers

    reject_unknown_params(params, ("file_path",), method="discover_layers")
    require_params(params, ["file_path"], method="discover_layers")
    filepath = params.get("file_path")
    result = discover_layers(filepath=filepath)
    return clean_result(result.to_dict())


def _run_parse_dxf_geometry(params):
    from dxf_import import parse_dxf_geometry, LayerMapping

    reject_unknown_params(params, ("file_path", "layer_mapping", "units",
                                   "flip_y"),
                          method="parse_geometry")
    require_params(params, ["file_path", "layer_mapping"],
                   method="parse_geometry",
                   valid=["file_path", "layer_mapping", "units", "flip_y"])
    filepath = params.get("file_path")
    units = params.get("units", "m")
    flip_y = params.get("flip_y", False)

    # Build LayerMapping from flat dict params
    mapping_dict = params.get("layer_mapping", {})
    check_nested_keys(mapping_dict, _MAPPING_KEYS,
                      method="parse_geometry", label="layer_mapping",
                      hints={"water": ("water_table",), "gwt": ("water_table",),
                             "groundwater": ("water_table",),
                             "ground": ("surface",),
                             "boundaries": ("soil_boundaries",),
                             "text": ("annotations",)})
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

    reject_unknown_params(params, ("parse_result", "soil_properties",
                                   "nail_defaults"),
                          method="build_slope_geometry")
    require_params(params, ["parse_result", "soil_properties"],
                   method="build_slope_geometry",
                   valid=["parse_result", "soil_properties", "nail_defaults"])
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
    check_soil_properties(params.get("soil_properties"), "slope",
                          method="build_slope_geometry")
    if params.get("nail_defaults") is not None:
        check_nested_keys(params["nail_defaults"], _NAIL_DEFAULT_KEYS,
                          method="build_slope_geometry",
                          label="nail_defaults")
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
    return clean_result(_slope_params(geom))


def _run_build_fem_inputs(params):
    from dxf_import import build_fem_inputs, FEMSoilPropertyAssignment
    from dxf_import.results import DxfParseResult

    reject_unknown_params(params, ("parse_result", "soil_properties"),
                          method="build_fem_inputs")
    require_params(params, ["parse_result", "soil_properties"],
                   method="build_fem_inputs",
                   valid=["parse_result", "soil_properties"])
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
    check_soil_properties(params.get("soil_properties"), "fem",
                          method="build_fem_inputs")
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
            "units": {"type": "str", "required": False, "default": "m", "allowed_values": ["m", "mm", "cm", "ft", "in"], "description": "Drawing units."},
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
            "surface_points": "Ground surface profile [[x, z], ...].",
            "soil_layers": "Assembled soil layers with geometry and properties, as slope_stability's soil_layers.",
            "gwt_points": "GWT profile [[x, z], ...] or null.",
            "nails": "Soil nails from the DXF nail lines, as slope_stability's nails (only when there are any). The whole result is slope_stability.analyze_slope / search_critical_surface input.",
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

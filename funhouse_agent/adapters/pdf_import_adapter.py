"""PDF import adapter — discover PDF content, extract vector geometry, build slope/FEM inputs."""

from funhouse_agent.adapters import clean_result


def _run_discover_pdf_content(params):
    from pdf_import import discover_pdf_content

    filepath = params.get("file_path")
    page = params.get("page", 0)

    result = discover_pdf_content(
        filepath=filepath,
        page=page,
    )
    return clean_result(result)


def _run_extract_vector_geometry(params):
    from pdf_import import extract_vector_geometry

    filepath = params.get("file_path")
    page = params.get("page", 0)
    scale = params.get("scale", 1.0)
    origin = params.get("origin", "bottom_left")
    role_mapping = params.get("role_mapping")

    result = extract_vector_geometry(
        filepath=filepath,
        page=page,
        scale=scale,
        origin=origin,
        role_mapping=role_mapping,
    )
    return clean_result(result.to_dict())


def _run_build_slope_geometry(params):
    from pdf_import import to_dxf_parse_result, PdfParseResult
    from dxf_import.converter import SoilPropertyAssignment, build_slope_geometry

    pr = params.get("parse_result", {})
    surface_points = [(p["x"], p["z"]) for p in pr.get("surface_points", [])]
    boundary_profiles = {
        name: [(p["x"], p["z"]) for p in pts]
        for name, pts in pr.get("boundary_profiles", {}).items()
    }
    gwt_points = None
    if pr.get("gwt_points"):
        gwt_points = [(p["x"], p["z"]) for p in pr["gwt_points"]]
    pdf_result = PdfParseResult(
        surface_points=surface_points,
        boundary_profiles=boundary_profiles,
        gwt_points=gwt_points,
    )
    dxf_result = to_dxf_parse_result(pdf_result)
    soil_props = [
        SoilPropertyAssignment(
            name=sp.get("name", ""), gamma=sp.get("gamma", 18.0),
            gamma_sat=sp.get("gamma_sat"), phi=sp.get("phi", 0.0),
            c_prime=sp.get("c_prime", 0.0), cu=sp.get("cu", 0.0),
            analysis_mode=sp.get("analysis_mode", "drained"),
        )
        for sp in params.get("soil_properties", [])
    ]
    geom = build_slope_geometry(dxf_result, soil_props)
    return clean_result({
        "surface_points": [{"x": round(x, 4), "z": round(z, 4)} for x, z in geom.surface_points],
        "n_layers": len(geom.soil_layers),
        "layers": [
            {"name": lyr.name, "top_elevation_m": round(lyr.top_elevation, 3),
             "bottom_elevation_m": round(lyr.bottom_elevation, 3),
             "gamma_kNm3": lyr.gamma, "phi_deg": lyr.phi, "c_prime_kPa": lyr.c_prime}
            for lyr in geom.soil_layers
        ],
        "has_gwt": geom.gwt_points is not None,
    })


def _run_build_fem_inputs(params):
    from pdf_import import to_dxf_parse_result, PdfParseResult
    from dxf_import.converter import FEMSoilPropertyAssignment, build_fem_inputs

    pr = params.get("parse_result", {})
    surface_points = [(p["x"], p["z"]) for p in pr.get("surface_points", [])]
    boundary_profiles = {
        name: [(p["x"], p["z"]) for p in pts]
        for name, pts in pr.get("boundary_profiles", {}).items()
    }
    gwt_points = None
    if pr.get("gwt_points"):
        gwt_points = [(p["x"], p["z"]) for p in pr["gwt_points"]]
    pdf_result = PdfParseResult(
        surface_points=surface_points,
        boundary_profiles=boundary_profiles,
        gwt_points=gwt_points,
    )
    dxf_result = to_dxf_parse_result(pdf_result)
    soil_props = [
        FEMSoilPropertyAssignment(
            name=sp.get("name", ""), gamma=sp.get("gamma", 18.0),
            phi=sp.get("phi", 0.0), c=sp.get("c", 0.0),
            E=sp.get("E", 30000.0), nu=sp.get("nu", 0.3),
            psi=sp.get("psi", 0.0), model=sp.get("model", "mc"),
            hs_params=sp.get("hs_params"),
        )
        for sp in params.get("soil_properties", [])
    ]
    result = build_fem_inputs(dxf_result, soil_props)
    return clean_result({
        "surface_points": [{"x": round(x, 4), "z": round(z, 4)} for x, z in result["surface_points"]],
        "n_layers": len(result["soil_layers"]),
        "layers": result["soil_layers"],
        "has_gwt": result["gwt"] is not None,
    })


METHOD_REGISTRY = {
    "discover_pdf_content": _run_discover_pdf_content,
    "extract_vector_geometry": _run_extract_vector_geometry,
    "build_slope_geometry": _run_build_slope_geometry,
    "build_fem_inputs": _run_build_fem_inputs,
}

METHOD_INFO = {
    "discover_pdf_content": {
        "category": "PDF Import",
        "brief": "Inventory a PDF page: vector path counts by color, text blocks, dimensions.",
        "parameters": {
            "file_path": {"type": "str", "required": True, "description": "Path to PDF file."},
            "page": {"type": "int", "required": False, "default": 0, "description": "Page number (0-indexed)."},
        },
        "returns": {
            "page_size": "Dict with width and height in points.",
            "n_drawings": "Total vector path count.",
            "colors": "Dict of hex_color to count.",
            "text_blocks": "List of text block dicts (text, x, y, size).",
            "has_images": "Whether page contains raster images.",
        },
    },
    "extract_vector_geometry": {
        "category": "PDF Import",
        "brief": "Extract cross-section geometry from PDF vector drawings via PyMuPDF.",
        "parameters": {
            "file_path": {"type": "str", "required": True, "description": "Path to PDF file."},
            "page": {"type": "int", "required": False, "default": 0, "description": "Page number (0-indexed)."},
            "scale": {"type": "float", "required": False, "default": 1.0, "description": "Scale factor: drawing_units * scale = meters."},
            "origin": {"type": "str", "required": False, "default": "bottom_left", "description": "Coordinate origin: 'bottom_left' or 'top_left'."},
            "role_mapping": {"type": "dict", "required": False, "description": "Maps hex colors to roles: {'#000000': 'surface', '#0000ff': 'gwt', '#808080': 'boundary_Clay'}."},
        },
        "returns": {
            "surface_points": "Ground surface as [{x, z}, ...].",
            "boundary_profiles": "Soil boundaries: {name: [{x, z}, ...]}.",
            "gwt_points": "GWT profile or null.",
            "text_annotations": "Text labels with coordinates.",
            "extraction_method": "Always 'vector'.",
            "confidence": "1.0 for vector extraction.",
        },
    },
    "build_slope_geometry": {
        "category": "PDF Import",
        "brief": "Convert extracted PDF geometry to SlopeGeometry for slope stability analysis.",
        "parameters": {
            "parse_result": {"type": "dict", "required": True, "description": "Output from extract_vector_geometry."},
            "soil_properties": {"type": "array", "required": True, "description": "List of {name, gamma, phi, c_prime, cu, analysis_mode} dicts."},
        },
        "returns": {"surface_points": "Surface profile.", "n_layers": "Number of soil layers.", "layers": "Layer details."},
    },
    "build_fem_inputs": {
        "category": "PDF Import",
        "brief": "Convert extracted PDF geometry to fem2d FEM input format.",
        "parameters": {
            "parse_result": {"type": "dict", "required": True, "description": "Output from extract_vector_geometry."},
            "soil_properties": {"type": "array", "required": True, "description": "List of {name, gamma, phi, c, E, nu, model} dicts."},
        },
        "returns": {"surface_points": "Surface profile.", "n_layers": "Number of soil layers.", "layers": "Layer details with stiffness."},
    },
}

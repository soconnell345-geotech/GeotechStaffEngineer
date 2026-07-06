"""PDF import adapter — discover PDF content, extract vector geometry, build slope/FEM inputs."""

from funhouse_agent.adapters import (
    clean_result, reject_unknown_params, require_params,
)


def _point_xz(p):
    """Accept a point as {x, z} dict or [x, z] pair."""
    return (p["x"], p["z"]) if isinstance(p, dict) else tuple(p)


def _run_discover_pdf_content(params):
    from pdf_import import discover_pdf_content

    reject_unknown_params(params, ("file_path", "page"),
                          method="discover_pdf_content")
    require_params(params, ["file_path"], method="discover_pdf_content",
                   valid=["file_path", "page"])
    filepath = params.get("file_path")
    page = params.get("page", 0)

    result = discover_pdf_content(
        filepath=filepath,
        page=page,
    )
    return clean_result(result)


def _run_extract_vector_geometry(params):
    from pdf_import import extract_vector_geometry

    _valid = ("file_path", "page", "scale", "origin", "role_mapping")
    reject_unknown_params(params, _valid, method="extract_vector_geometry")
    require_params(params, ["file_path"], method="extract_vector_geometry",
                   valid=_valid)
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


def _run_calibrate_scale(params):
    from pdf_import import calibrate_scale

    reject_unknown_params(params, ("p1", "p2", "distance_m"),
                          method="calibrate_scale")
    require_params(params, ["p1", "p2", "distance_m"], method="calibrate_scale",
                   valid=["p1", "p2", "distance_m"])
    sf = calibrate_scale(_point_xz(params["p1"]), _point_xz(params["p2"]),
                         params["distance_m"])
    return {"scale_factor": sf, "basis": "two_point_calibration", "applied": False,
            "note": "Pass scale_factor as extract_vector_geometry(scale=...)."}


def _run_propose_scale(params):
    from pdf_import import propose_scale

    reject_unknown_params(params, ("text_blocks", "calibration"),
                          method="propose_scale")
    require_params(params, ["text_blocks"], method="propose_scale",
                   valid=["text_blocks", "calibration"])
    return clean_result(propose_scale(params["text_blocks"],
                                      calibration=params.get("calibration")))


def _run_propose_role_mapping(params):
    from pdf_import import (
        propose_role_mapping, extract_colored_paths, discover_pdf_content,
    )

    _valid = ("file_path", "page", "origin", "regions", "text_blocks")
    reject_unknown_params(params, _valid, method="propose_role_mapping")
    regions = params.get("regions")
    text_blocks = params.get("text_blocks")
    if regions is None or text_blocks is None:
        require_params(params, ["file_path"], method="propose_role_mapping",
                       valid=_valid)
        page = params.get("page", 0)
        origin = params.get("origin", "bottom_left")
        if regions is None:
            regions = extract_colored_paths(filepath=params["file_path"],
                                            page=page, origin=origin)
        if text_blocks is None:
            text_blocks = discover_pdf_content(
                filepath=params["file_path"], page=page)["text_blocks"]
    # Normalize region points ({x,z}/[x,y]) to (x, y) tuples.
    norm = [{"color": r.get("color"),
             "points": [_point_xz(p) if isinstance(p, dict) else tuple(p)
                        for p in r.get("points", [])]} for r in regions]
    return clean_result(propose_role_mapping(norm, text_blocks))


def _run_cleanup_geometry(params):
    from pdf_import import cleanup_geometry

    _valid = ("parse_result", "tol", "snap_tol", "angle_tol_deg", "join")
    reject_unknown_params(params, _valid, method="cleanup_geometry")
    require_params(params, ["parse_result"], method="cleanup_geometry",
                   valid=_valid)
    pr = params["parse_result"]

    def _pts(seq):
        return [_point_xz(p) for p in (seq or [])]

    surface = _pts(pr.get("surface_points"))
    boundaries = {name: _pts(pts)
                  for name, pts in (pr.get("boundary_profiles") or {}).items()}
    gwt = _pts(pr.get("gwt_points")) if pr.get("gwt_points") is not None else None

    out = cleanup_geometry(
        surface_points=surface, boundary_profiles=boundaries, gwt_points=gwt,
        tol=params.get("tol", 1e-4), snap_tol=params.get("snap_tol", 1e-3),
        angle_tol_deg=params.get("angle_tol_deg", 1.0),
        join=params.get("join", False))

    def _xz(seq):
        return [{"x": round(x, 4), "z": round(z, 4)} for x, z in (seq or [])]

    return clean_result({
        "surface_points": _xz(out["surface_points"]),
        "boundary_profiles": {n: _xz(p) for n, p in out["boundary_profiles"].items()},
        "gwt_points": _xz(out["gwt_points"]) if out["gwt_points"] is not None else None,
        "report": out["report"],
    })


def _run_build_slope_geometry(params):
    from pdf_import import to_dxf_parse_result, PdfParseResult
    from dxf_import.converter import SoilPropertyAssignment, build_slope_geometry

    reject_unknown_params(params, ("parse_result", "soil_properties"),
                          method="build_slope_geometry")
    require_params(params, ["parse_result", "soil_properties"],
                   method="build_slope_geometry",
                   valid=["parse_result", "soil_properties"])
    pr = params.get("parse_result", {})
    surface_points = [_point_xz(p) for p in pr.get("surface_points", [])]
    boundary_profiles = {
        name: [_point_xz(p) for p in pts]
        for name, pts in pr.get("boundary_profiles", {}).items()
    }
    gwt_points = None
    if pr.get("gwt_points"):
        gwt_points = [_point_xz(p) for p in pr["gwt_points"]]
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

    reject_unknown_params(params, ("parse_result", "soil_properties"),
                          method="build_fem_inputs")
    require_params(params, ["parse_result", "soil_properties"],
                   method="build_fem_inputs",
                   valid=["parse_result", "soil_properties"])
    pr = params.get("parse_result", {})
    surface_points = [_point_xz(p) for p in pr.get("surface_points", [])]
    boundary_profiles = {
        name: [_point_xz(p) for p in pts]
        for name, pts in pr.get("boundary_profiles", {}).items()
    }
    gwt_points = None
    if pr.get("gwt_points"):
        gwt_points = [_point_xz(p) for p in pr["gwt_points"]]
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
    "calibrate_scale": _run_calibrate_scale,
    "propose_scale": _run_propose_scale,
    "propose_role_mapping": _run_propose_role_mapping,
    "cleanup_geometry": _run_cleanup_geometry,
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
            "origin": {"type": "str", "required": False, "default": "bottom_left", "allowed_values": ["bottom_left", "top_left"], "description": "Coordinate origin."},
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
    "calibrate_scale": {
        "category": "PDF Import",
        "brief": "Two-point scale calibration: meters-per-drawing-unit from two points + their known real-world distance (deterministic, for extract_vector_geometry scale=...).",
        "parameters": {
            "p1": {"type": "array", "required": True, "description": "First point in drawing units, [x, y] or {x, z}."},
            "p2": {"type": "array", "required": True, "description": "Second point in drawing units."},
            "distance_m": {"type": "float", "required": True, "description": "Known real-world distance between p1 and p2 (meters)."},
        },
        "returns": {"scale_factor": "Meters per drawing unit.", "applied": "Always false — pass it as scale=... yourself."},
    },
    "propose_scale": {
        "category": "PDF Import",
        "brief": "Parse scale annotations ('SCALE 1:100', '1\"=20 ft', '1 cm=2 m') from the page text_blocks into scale CANDIDATES with provenance/confidence. Proposals only — never silently applied; confirm one and pass it as scale=...",
        "parameters": {
            "text_blocks": {"type": "array", "required": True, "description": "Text blocks from discover_pdf_content (each has a 'text')."},
            "calibration": {"type": "dict", "required": False, "description": "Optional {p1, p2, distance_m} to add a deterministic two-point candidate (confidence 1.0)."},
        },
        "returns": {"candidates": "Scale candidates (highest confidence first), each {scale_factor, basis, provenance, confidence, applied:false}.", "recommended": "Best candidate or null."},
    },
    "propose_role_mapping": {
        "category": "PDF Import",
        "brief": "Propose a colour->role role_mapping by associating soil/GWT/surface text labels to the vector regions they sit in / are nearest to. Proposal only — confirm before passing to extract_vector_geometry(role_mapping=...).",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "PDF path — extracts coloured paths + text blocks internally. Provide this OR regions+text_blocks."},
            "page": {"type": "int", "required": False, "default": 0, "description": "Page number (0-indexed)."},
            "origin": {"type": "str", "required": False, "default": "bottom_left", "allowed_values": ["bottom_left", "top_left"], "description": "Coordinate origin."},
            "regions": {"type": "array", "required": False, "description": "Alternative to file_path: [{color, points:[[x,y],...]}, ...]."},
            "text_blocks": {"type": "array", "required": False, "description": "Alternative to file_path: [{text, x, y}, ...]."},
        },
        "returns": {"role_mapping": "Proposed {hex_color: role} (roles: surface/gwt/boundary_<Name>).", "associations": "Per-label {label, role, color, method, distance}.", "applied": "Always false."},
    },
    "cleanup_geometry": {
        "category": "PDF Import",
        "brief": "Clean extracted geometry before build_slope_geometry: dedupe points, thin collinear runs, snap near-coincident surface/boundary endpoints, optionally join broken polylines. Returns cleaned geometry + a point-count report.",
        "parameters": {
            "parse_result": {"type": "dict", "required": True, "description": "Extracted geometry {surface_points, boundary_profiles, gwt_points} (from extract_vector_geometry)."},
            "tol": {"type": "float", "required": False, "default": 0.0001, "description": "Dedupe distance tolerance (m)."},
            "snap_tol": {"type": "float", "required": False, "default": 0.001, "description": "Endpoint-snapping tolerance across surface + boundaries (m)."},
            "angle_tol_deg": {"type": "float", "required": False, "default": 1.0, "description": "Collinear-thinning bend tolerance (deg): interior vertices turning less than this are removed."},
            "join": {"type": "bool", "required": False, "default": False, "description": "Also join broken polyline segments within each boundary."},
        },
        "returns": {"surface_points": "Cleaned surface.", "boundary_profiles": "Cleaned boundaries.", "gwt_points": "Cleaned GWT or null.", "report": "Before/after point counts."},
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

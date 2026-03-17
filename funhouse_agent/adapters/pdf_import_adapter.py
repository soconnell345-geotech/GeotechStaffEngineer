"""PDF import adapter — discover PDF content and extract vector geometry."""

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


METHOD_REGISTRY = {
    "discover_pdf_content": _run_discover_pdf_content,
    "extract_vector_geometry": _run_extract_vector_geometry,
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
}

"""
PDF Import Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. pdf_import_agent           - Run a PDF import operation
  2. pdf_import_list_methods    - Browse available methods
  3. pdf_import_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - pip install geotech-staff-engineer[pdf] (PyPI)
  - These functions accept and return JSON strings for LLM compatibility
"""

import json
import base64

try:
    from functions.api import function
except ImportError:
    def function(fn):
        fn.__wrapped__ = fn
        return fn


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_discover_pdf_content(params):
    from pdf_import import discover_pdf_content
    filepath = params.get("filepath")
    content_b64 = params.get("content_base64")
    content = None
    if content_b64:
        content = base64.b64decode(content_b64)
    page = params.get("page", 0)
    return discover_pdf_content(filepath=filepath, content=content, page=page)


def _run_extract_vector_geometry(params):
    from pdf_import import extract_vector_geometry
    filepath = params.get("filepath")
    content_b64 = params.get("content_base64")
    content = None
    if content_b64:
        content = base64.b64decode(content_b64)
    result = extract_vector_geometry(
        filepath=filepath,
        content=content,
        page=params.get("page", 0),
        scale=params.get("scale", 1.0),
        origin=params.get("origin", "bottom_left"),
        role_mapping=params.get("role_mapping"),
    )
    return result.to_dict()


def _run_extract_geometry_vision(params):
    from pdf_import import extract_geometry_vision
    # Vision extraction requires an image_fn — not available in Foundry
    # This method returns an error directing users to use vector extraction
    # or supply geometry directly
    return {
        "error": (
            "Vision extraction requires an LLM vision function (image_fn) "
            "which is not available in Foundry agent context. Use "
            "extract_vector_geometry for programmatic extraction, or "
            "use the funhouse_agent for vision-based extraction."
        )
    }


def _run_build_slope_geometry(params):
    from pdf_import import to_dxf_parse_result, PdfParseResult
    from dxf_import.converter import SoilPropertyAssignment, build_slope_geometry
    # Reconstruct PdfParseResult from dict
    pr_dict = params.get("parse_result", {})
    surface_points = [
        (p["x"], p["z"]) for p in pr_dict.get("surface_points", [])
    ]
    boundary_profiles = {}
    for name, pts in pr_dict.get("boundary_profiles", {}).items():
        boundary_profiles[name] = [(p["x"], p["z"]) for p in pts]
    gwt_points = None
    if "gwt_points" in pr_dict and pr_dict["gwt_points"]:
        gwt_points = [(p["x"], p["z"]) for p in pr_dict["gwt_points"]]
    pdf_result = PdfParseResult(
        surface_points=surface_points,
        boundary_profiles=boundary_profiles,
        gwt_points=gwt_points,
    )
    dxf_result = to_dxf_parse_result(pdf_result)
    # Build soil properties
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
    geom = build_slope_geometry(dxf_result, soil_props)
    return {
        "surface_points": [
            {"x": round(x, 4), "z": round(z, 4)}
            for x, z in geom.surface_points
        ],
        "n_layers": len(geom.soil_layers),
        "layers": [
            {
                "name": lyr.name,
                "top_elevation_m": round(lyr.top_elevation, 3),
                "bottom_elevation_m": round(lyr.bottom_elevation, 3),
                "gamma_kNm3": lyr.gamma,
                "phi_deg": lyr.phi,
                "c_prime_kPa": lyr.c_prime,
            }
            for lyr in geom.soil_layers
        ],
        "has_gwt": geom.gwt_points is not None,
    }


def _run_build_fem_inputs(params):
    from pdf_import import to_dxf_parse_result, PdfParseResult
    from dxf_import.converter import FEMSoilPropertyAssignment, build_fem_inputs
    # Reconstruct PdfParseResult from dict
    pr_dict = params.get("parse_result", {})
    surface_points = [
        (p["x"], p["z"]) for p in pr_dict.get("surface_points", [])
    ]
    boundary_profiles = {}
    for name, pts in pr_dict.get("boundary_profiles", {}).items():
        boundary_profiles[name] = [(p["x"], p["z"]) for p in pts]
    gwt_points = None
    if "gwt_points" in pr_dict and pr_dict["gwt_points"]:
        gwt_points = [(p["x"], p["z"]) for p in pr_dict["gwt_points"]]
    pdf_result = PdfParseResult(
        surface_points=surface_points,
        boundary_profiles=boundary_profiles,
        gwt_points=gwt_points,
    )
    dxf_result = to_dxf_parse_result(pdf_result)
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
    result = build_fem_inputs(dxf_result, soil_props)
    return {
        "surface_points": [
            {"x": round(x, 4), "z": round(z, 4)}
            for x, z in result["surface_points"]
        ],
        "n_layers": len(result["soil_layers"]),
        "layers": result["soil_layers"],
        "has_gwt": result["gwt"] is not None,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "discover_pdf_content": _run_discover_pdf_content,
    "extract_vector_geometry": _run_extract_vector_geometry,
    "extract_geometry_vision": _run_extract_geometry_vision,
    "build_slope_geometry": _run_build_slope_geometry,
    "build_fem_inputs": _run_build_fem_inputs,
}

METHOD_INFO = {
    "discover_pdf_content": {
        "category": "Discovery",
        "brief": "Inventory PDF page: vector paths by color, text blocks, images.",
        "description": (
            "Reads a PDF page and catalogs vector drawing paths grouped by stroke "
            "color, text blocks with positions and sizes, and image presence. "
            "Use this first to understand the PDF structure before creating a role_mapping."
        ),
        "parameters": {
            "filepath": {"type": "str", "required": False,
                         "description": "Path to PDF file."},
            "content_base64": {"type": "str", "required": False,
                               "description": "Base64-encoded PDF content."},
            "page": {"type": "int", "required": False, "default": 0,
                     "description": "Page number (0-indexed)."},
        },
        "returns": {
            "page_size": "Width and height in points.",
            "n_drawings": "Total vector path count.",
            "colors": "Dict of hex_color → count.",
            "text_blocks": "List of {text, x, y, size}.",
            "has_images": "Whether page contains raster images.",
        },
    },
    "extract_vector_geometry": {
        "category": "Extraction",
        "brief": "Extract cross-section geometry from PDF vector drawings.",
        "description": (
            "Uses PyMuPDF to extract vector paths from a PDF page, groups them "
            "by stroke color, and assigns geometric roles (surface, boundaries, GWT) "
            "via role_mapping. Output is compatible with build_slope_geometry() and "
            "build_fem_inputs()."
        ),
        "parameters": {
            "filepath": {"type": "str", "required": False},
            "content_base64": {"type": "str", "required": False},
            "page": {"type": "int", "required": False, "default": 0},
            "scale": {"type": "float", "required": False, "default": 1.0,
                      "description": "Scale factor: drawing_units * scale = meters."},
            "origin": {"type": "str", "required": False, "default": "bottom_left",
                       "choices": ["bottom_left", "top_left"]},
            "role_mapping": {
                "type": "dict", "required": False,
                "description": (
                    "Maps hex colors to roles: "
                    '{"#000000": "surface", "#0000ff": "gwt", "#808080": "boundary_Clay"}'
                ),
            },
        },
        "returns": {
            "surface_points": "List of {x, z} in meters.",
            "boundary_profiles": "Dict of name → [{x, z}].",
            "gwt_points": "GWT profile or null.",
        },
    },
    "extract_geometry_vision": {
        "category": "Extraction",
        "brief": "Extract geometry using LLM vision (requires vision function).",
        "description": (
            "Uses LLM vision to analyze a PDF/image and extract geometry. "
            "Not available in Foundry context — use funhouse_agent instead."
        ),
        "parameters": {},
        "returns": {"error": "Vision not available in Foundry."},
    },
    "build_slope_geometry": {
        "category": "Conversion",
        "brief": "Convert PDF geometry to SlopeGeometry for slope stability.",
        "description": (
            "Takes extract_vector_geometry output and user soil properties "
            "to build a SlopeGeometry for slope_stability analysis."
        ),
        "parameters": {
            "parse_result": {"type": "dict", "required": True,
                             "description": "Output from extract_vector_geometry."},
            "soil_properties": {"type": "list", "required": True,
                                "description": "List of {name, gamma, phi, c_prime, cu, analysis_mode}."},
        },
        "returns": {
            "surface_points": "Surface profile.",
            "n_layers": "Number of soil layers.",
            "layers": "Layer details.",
        },
    },
    "build_fem_inputs": {
        "category": "Conversion",
        "brief": "Convert PDF geometry to fem2d FEM input format.",
        "description": (
            "Takes extract_vector_geometry output and user soil properties "
            "(with stiffness) to build fem2d input dicts."
        ),
        "parameters": {
            "parse_result": {"type": "dict", "required": True,
                             "description": "Output from extract_vector_geometry."},
            "soil_properties": {"type": "list", "required": True,
                                "description": "List of {name, gamma, phi, c, E, nu, model}."},
        },
        "returns": {
            "surface_points": "Surface profile.",
            "n_layers": "Number of soil layers.",
            "layers": "Layer details with stiffness.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def pdf_import_agent(method: str, parameters_json: str) -> str:
    """
    PDF Import — extract cross-section geometry from PDF drawings.

    Uses PyMuPDF vector path analysis or LLM vision to extract geometry
    from Plaxis/Slope-W/hand-drawn cross-sections. Output feeds into
    slope_stability or fem2d analysis.

    Parameters:
        method: Operation name (e.g. "discover_pdf_content").
        parameters_json: JSON string of parameters.

    Returns:
        JSON string with results or an error message.
    """
    try:
        params = json.loads(parameters_json)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid parameters_json: {str(e)}"})

    if method not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY.keys()))
        return json.dumps({
            "error": f"Unknown method '{method}'. Available: {available}"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(result, default=str)
    except ImportError as e:
        return json.dumps({"error": f"ImportError: {str(e)}"})
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def pdf_import_list_methods(category: str = "") -> str:
    """
    Lists available PDF import operations.

    Parameters:
        category: Optional filter ("Discovery", "Extraction", "Conversion").

    Returns:
        JSON string with method names and brief descriptions.
    """
    result = {}
    for method_name, info in METHOD_INFO.items():
        if category and info["category"].lower() != category.lower():
            continue
        cat = info["category"]
        if cat not in result:
            result[cat] = {}
        result[cat][method_name] = info["brief"]

    if not result:
        cats = sorted(set(i["category"] for i in METHOD_INFO.values()))
        return json.dumps({
            "error": f"No methods found for category '{category}'. "
                     f"Available: {', '.join(cats)}"
        })
    return json.dumps(result)


@function
def pdf_import_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a PDF import operation.

    Parameters:
        method: The method name (e.g. "discover_pdf_content").

    Returns:
        JSON string with parameters, types, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({
            "error": f"Unknown method '{method}'. Available: {available}"
        })
    return json.dumps(METHOD_INFO[method], default=str)

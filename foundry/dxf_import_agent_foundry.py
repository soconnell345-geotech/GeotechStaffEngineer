"""
DXF Import Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. dxf_import_agent           - Run a DXF import operation
  2. dxf_import_list_methods    - Browse available methods
  3. dxf_import_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - pip install geotech-staff-engineer[dxf] (PyPI)
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

def _run_discover_layers(params):
    from dxf_import import discover_layers
    filepath = params.get("filepath")
    content_b64 = params.get("content_base64")
    content = None
    if content_b64:
        content = base64.b64decode(content_b64)
    result = discover_layers(filepath=filepath, content=content)
    return result.to_dict()


def _run_parse_geometry(params):
    from dxf_import import parse_dxf_geometry, LayerMapping
    filepath = params.get("filepath")
    content_b64 = params.get("content_base64")
    content = None
    if content_b64:
        content = base64.b64decode(content_b64)
    mapping_dict = params.get("layer_mapping", {})
    mapping = LayerMapping(
        surface=mapping_dict.get("surface", ""),
        soil_boundaries=mapping_dict.get("soil_boundaries", {}),
        water_table=mapping_dict.get("water_table"),
        nails=mapping_dict.get("nails"),
        annotations=mapping_dict.get("annotations"),
    )
    units = params.get("units", "m")
    flip_y = params.get("flip_y", False)
    result = parse_dxf_geometry(
        filepath=filepath, content=content,
        layer_mapping=mapping, units=units, flip_y=flip_y,
    )
    return result.to_dict()


def _run_build_slope_geometry(params):
    from dxf_import import build_slope_geometry, SoilPropertyAssignment
    from dxf_import.results import DxfParseResult
    # Reconstruct DxfParseResult from dict
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
    nail_lines = pr_dict.get("nail_lines", [])
    parse_result = DxfParseResult(
        surface_points=surface_points,
        boundary_profiles=boundary_profiles,
        gwt_points=gwt_points,
        nail_lines=nail_lines,
        units_used=pr_dict.get("units_used", "m"),
    )
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
    nail_defaults = params.get("nail_defaults")
    geom = build_slope_geometry(parse_result, soil_props, nail_defaults)
    # Return SlopeGeometry as dict
    result = {
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
                "cu_kPa": lyr.cu,
                "analysis_mode": lyr.analysis_mode,
            }
            for lyr in geom.soil_layers
        ],
        "has_gwt": geom.gwt_points is not None,
        "n_nails": len(geom.nails) if geom.nails else 0,
    }
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "discover_layers": _run_discover_layers,
    "parse_geometry": _run_parse_geometry,
    "build_slope_geometry": _run_build_slope_geometry,
}

METHOD_INFO = {
    "discover_layers": {
        "category": "Discovery",
        "brief": "Inventory all DXF layers with entity counts, types, and sample text.",
        "description": (
            "Reads a DXF file and lists every layer that contains entities. "
            "Returns entity type counts (LWPOLYLINE, LINE, TEXT, etc.), sample "
            "text content, bounding boxes, and $INSUNITS hint. Use this first "
            "to understand the DXF structure before creating a layer mapping."
        ),
        "reference": "ezdxf documentation, DXF R2010+ format",
        "parameters": {
            "filepath": {
                "type": "str", "required": False,
                "description": "Path to DXF file on disk. Provide either filepath or content_base64.",
            },
            "content_base64": {
                "type": "str", "required": False,
                "description": "Base64-encoded DXF file content (for upload workflows).",
            },
        },
        "returns": {
            "n_layers": "Number of layers with entities.",
            "layers": "List of layer info dicts with name, n_entities, entity_types, sample_texts, bbox.",
            "units_hint": "Detected units from $INSUNITS header (or null).",
            "n_total_entities": "Total entity count.",
        },
        "related": {
            "parse_geometry": "Extract coordinates from specific layers after discovery.",
        },
        "typical_workflow": (
            "1. discover_layers — see what layers exist\n"
            "2. Create layer_mapping from discovered layer names\n"
            "3. parse_geometry — extract coordinates\n"
            "4. build_slope_geometry — add soil properties\n"
            "5. slope_stability_agent.analyze_slope — run analysis"
        ),
        "common_mistakes": [
            "Passing a .dwg file — only DXF is supported.",
            "Not checking $INSUNITS — coordinates may be in feet or mm.",
        ],
    },
    "parse_geometry": {
        "category": "Parsing",
        "brief": "Extract slope geometry coordinates from mapped DXF layers.",
        "description": (
            "Given a layer mapping (surface, boundaries, GWT, nails), extracts "
            "polyline/line coordinates from the DXF file, merges multi-polyline "
            "layers, converts units to meters, and returns sorted coordinate arrays "
            "ready for build_slope_geometry()."
        ),
        "reference": "ezdxf documentation",
        "parameters": {
            "filepath": {
                "type": "str", "required": False,
                "description": "Path to DXF file.",
            },
            "content_base64": {
                "type": "str", "required": False,
                "description": "Base64-encoded DXF content.",
            },
            "layer_mapping": {
                "type": "dict", "required": True,
                "description": (
                    "Dict with keys: surface (str, required), "
                    "soil_boundaries ({dxf_layer: soil_name}), "
                    "water_table (str), nails (str), annotations ([str])."
                ),
            },
            "units": {
                "type": "str", "required": False, "default": "m",
                "choices": ["m", "mm", "cm", "ft", "in"],
                "description": "Drawing units. Coordinates converted to meters.",
            },
            "flip_y": {
                "type": "bool", "required": False, "default": False,
                "description": "Negate Y values for downward-positive drawings.",
            },
        },
        "returns": {
            "surface_points": "List of {x, z} dicts in meters.",
            "boundary_profiles": "Dict of soil_name → [{x, z}] arrays.",
            "gwt_points": "GWT profile [{x, z}] or null.",
            "nail_lines": "List of {x_head, z_head, x_tip, z_tip} dicts.",
        },
    },
    "build_slope_geometry": {
        "category": "Conversion",
        "brief": "Assemble SlopeGeometry from DXF coordinates + soil properties.",
        "description": (
            "Takes a parse_geometry result and user-supplied soil properties "
            "(gamma, phi, c', cu) to create a SlopeGeometry object. The DXF "
            "provides geometry only — strength parameters must come from the user. "
            "Returns a structured dict ready for slope_stability_agent.analyze_slope()."
        ),
        "parameters": {
            "parse_result": {
                "type": "dict", "required": True,
                "description": "Output from parse_geometry (surface_points, boundary_profiles, etc.).",
            },
            "soil_properties": {
                "type": "list", "required": True,
                "description": (
                    "List of dicts with: name, gamma, gamma_sat, phi, c_prime, cu, "
                    "analysis_mode. Names must match boundary soil names."
                ),
            },
            "nail_defaults": {
                "type": "dict", "required": False,
                "description": (
                    "Default nail parameters: bar_diameter (mm), drill_hole_diameter (mm), "
                    "fy (MPa), bond_stress (kPa), spacing_h (m)."
                ),
            },
        },
        "returns": {
            "surface_points": "Surface profile [{x, z}].",
            "n_layers": "Number of soil layers.",
            "layers": "Layer details (name, elevations, properties).",
            "has_gwt": "Whether GWT is present.",
            "n_nails": "Number of nails converted.",
        },
        "related": {
            "slope_stability_agent.analyze_slope": "Run stability analysis on the result.",
            "slope_stability_agent.search_critical_surface": "Search for critical slip surface.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def dxf_import_agent(method: str, parameters_json: str) -> str:
    """
    DXF Import for Slope Stability — import CAD cross-sections.

    Reads DXF files (AutoCAD/Civil 3D) and converts cross-section geometry
    into SlopeGeometry for slope stability analysis. Three-step workflow:
    discover_layers → parse_geometry → build_slope_geometry.

    Call dxf_import_list_methods() first to see available operations,
    then dxf_import_describe_method() for parameter details.

    Parameters:
        method: Operation name (e.g. "discover_layers").
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
def dxf_import_list_methods(category: str = "") -> str:
    """
    Lists available DXF import operations.

    Parameters:
        category: Optional filter ("Discovery", "Parsing", "Conversion").
                  Leave empty for all.

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
def dxf_import_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a DXF import operation.

    Parameters:
        method: The method name (e.g. "discover_layers").

    Returns:
        JSON string with parameters, types, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({
            "error": f"Unknown method '{method}'. Available: {available}"
        })
    return json.dumps(METHOD_INFO[method], default=str)

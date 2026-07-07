"""Drawing IR adapter — digitize a drawing once, then query SLICES of it.

The LLM-facing surface for :mod:`drawing_ir`. Because a full IR can be large,
``digitize_drawing`` caches the IR server-side keyed by a short ``handle`` and
returns only a summary + stats; the agent then pulls slices with
``query_drawing`` (spatial/angle/text/layer queries) and exact coordinates for a
shortlist with ``get_entities``. This mirrors the north star: the deterministic
extractor owns coordinates; the LLM requests structured slices, never pixels.
"""

import os
import uuid

from funhouse_agent.adapters import (
    clean_result, reject_unknown_params, require_params,
)

# ---------------------------------------------------------------------------
# Server-side IR cache (keyed by handle). Bounded to avoid unbounded growth.
# ---------------------------------------------------------------------------
_IR_CACHE = {}
_IR_ORDER = []
_IR_CACHE_MAX = 32


def _store_ir(ir):
    handle = "dwg_" + uuid.uuid4().hex[:8]
    _IR_CACHE[handle] = ir
    _IR_ORDER.append(handle)
    while len(_IR_ORDER) > _IR_CACHE_MAX:
        old = _IR_ORDER.pop(0)
        _IR_CACHE.pop(old, None)
    return handle


def _get_ir(handle):
    ir = _IR_CACHE.get(handle)
    if ir is None:
        raise ValueError(
            f"Unknown drawing handle '{handle}'. Call digitize_drawing first; "
            f"active handles: {sorted(_IR_CACHE.keys()) or ['(none)']}.")
    return ir


_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def _auto_source(file_path):
    ext = os.path.splitext(str(file_path))[1].lower()
    if ext == ".dxf":
        return "dxf"
    if ext == ".pdf":
        return "pdf_vector"
    if ext in _RASTER_EXTS:
        return "raster"
    raise ValueError(
        f"Cannot auto-detect source for '{file_path}'. Pass source= one of "
        f"dxf/pdf_vector/raster.")


def _point_xy(p):
    return (p["x"], p["y"]) if isinstance(p, dict) else tuple(p)


# ---------------------------------------------------------------------------
# digitize_drawing
# ---------------------------------------------------------------------------

def _run_digitize_drawing(params):
    from drawing_ir import from_dxf, from_pdf_vector, from_raster, queries

    _valid = ("file_path", "source", "page", "scale", "units", "origin",
              "calibration", "detect_lines", "detect_circles",
              "detect_contours", "ocr")
    reject_unknown_params(params, _valid, method="digitize_drawing")
    require_params(params, ["file_path"], method="digitize_drawing",
                   valid=_valid)
    file_path = params["file_path"]
    source = params.get("source", "auto")
    if source == "auto":
        source = _auto_source(file_path)

    if source == "dxf":
        ir = from_dxf(filepath=file_path, units=params.get("units"))
    elif source == "pdf_vector":
        ir = from_pdf_vector(
            filepath=file_path, page=params.get("page", 0),
            scale=params.get("scale"), calibration=params.get("calibration"),
            origin=params.get("origin", "bottom_left"))
    elif source == "raster":
        ir = from_raster(
            filepath=file_path, scale=params.get("scale"),
            origin=params.get("origin", "bottom_left"),
            detect_lines=params.get("detect_lines", True),
            detect_circles=params.get("detect_circles", True),
            detect_contours=params.get("detect_contours", True),
            ocr=params.get("ocr", True))
    else:
        raise ValueError(
            f"Unknown source '{source}'. Use dxf/pdf_vector/raster/auto.")

    handle = _store_ir(ir)
    out = {"handle": handle, "source": ir.source}
    out.update(queries.summary_stats(ir))
    if ir.metadata.get("scale_candidates"):
        out["scale_candidates"] = ir.metadata["scale_candidates"]
    out["note"] = ("IR cached under 'handle'. Use query_drawing to request "
                   "slices and get_entities for exact coordinates of specific "
                   "ids. Confidence < 1.0 marks raster detections.")
    return clean_result(out)


# ---------------------------------------------------------------------------
# query_drawing
# ---------------------------------------------------------------------------

# name -> (function, required-param-names, all-param-names)
def _query_registry():
    from drawing_ir import queries as q
    return {
        "entities_in_bbox": (q.entities_in_bbox,
                             ["x_min", "y_min", "x_max", "y_max"],
                             ["x_min", "y_min", "x_max", "y_max", "mode",
                              "entity_type"]),
        "nearest_entity": (q.nearest_entity, ["x", "y"],
                           ["x", "y", "entity_type", "k"]),
        "lines_by_angle": (q.lines_by_angle, ["min_deg", "max_deg"],
                           ["min_deg", "max_deg"]),
        "horizontal_lines": (q.horizontal_lines, [], ["tol_deg"]),
        "vertical_lines": (q.vertical_lines, [], ["tol_deg"]),
        "polylines_longer_than": (q.polylines_longer_than, ["min_length"],
                                  ["min_length"]),
        "text_items": (q.text_items, [], ["pattern"]),
        "text_near": (q.text_near, ["entity_id", "radius"],
                      ["entity_id", "radius"]),
        "entities_on_layer": (q.entities_on_layer, ["layer"], ["layer"]),
        "entities_by_color": (q.entities_by_color, ["color"], ["color"]),
        "candidate_ground_surface": (q.candidate_ground_surface, [], []),
        "summary_stats": (q.summary_stats, [], []),
    }


QUERY_NAMES = sorted([
    "entities_in_bbox", "nearest_entity", "lines_by_angle", "horizontal_lines",
    "vertical_lines", "polylines_longer_than", "text_items", "text_near",
    "entities_on_layer", "entities_by_color", "candidate_ground_surface",
    "summary_stats",
])


def _run_query_drawing(params):
    reject_unknown_params(params, ("handle", "query", "params"),
                          method="query_drawing")
    require_params(params, ["handle", "query"], method="query_drawing",
                   valid=["handle", "query", "params"])
    ir = _get_ir(params["handle"])
    query = params["query"]
    registry = _query_registry()
    if query not in registry:
        raise ValueError(
            f"Unknown query '{query}'. Available: {QUERY_NAMES}.")
    func, required, allowed = registry[query]
    qparams = dict(params.get("params") or {})
    # Normalize any {x,y}/[x,y] point-ish params are passed through as-is;
    # the query functions take scalars, so no coercion needed here.
    missing = [k for k in required if k not in qparams]
    if missing:
        raise ValueError(
            f"query '{query}' missing required params {missing}. "
            f"Accepts: {allowed}.")
    unknown = [k for k in qparams if k not in allowed]
    if unknown:
        raise ValueError(
            f"query '{query}': unknown params {sorted(unknown)}. "
            f"Accepts: {allowed}.")
    result = func(ir, **qparams)
    payload = {"handle": params["handle"], "query": query, "result": result}
    if isinstance(result, list):
        payload["n_results"] = len(result)
    return clean_result(payload)


# ---------------------------------------------------------------------------
# get_entities
# ---------------------------------------------------------------------------

def _run_get_entities(params):
    from drawing_ir import queries

    reject_unknown_params(params, ("handle", "ids"), method="get_entities")
    require_params(params, ["handle", "ids"], method="get_entities",
                   valid=["handle", "ids"])
    ir = _get_ir(params["handle"])
    ids = params["ids"]
    if isinstance(ids, str):
        ids = [ids]
    return clean_result({"handle": params["handle"],
                         "entities": queries.get_entities(ir, list(ids))})


METHOD_REGISTRY = {
    "digitize_drawing": _run_digitize_drawing,
    "query_drawing": _run_query_drawing,
    "get_entities": _run_get_entities,
}

METHOD_INFO = {
    "digitize_drawing": {
        "category": "Drawing IR",
        "brief": ("Digitize a drawing (DXF / PDF-vector / raster image) into a "
                  "unified intermediate representation. Caches the IR "
                  "server-side under a 'handle' and returns a summary + stats "
                  "(counts by type/layer, page metadata, extent, scale). Then "
                  "use query_drawing / get_entities — the full IR is never "
                  "returned by default."),
        "parameters": {
            "file_path": {"type": "str", "required": True,
                          "description": "Path to the drawing file (.dxf, .pdf, or a raster image)."},
            "source": {"type": "str", "required": False, "default": "auto",
                       "allowed_values": ["auto", "dxf", "pdf_vector", "raster"],
                       "description": "Ingest leg. 'auto' picks by file extension."},
            "page": {"type": "int", "required": False, "default": 0,
                     "description": "PDF page (0-indexed); pdf_vector only."},
            "scale": {"type": "float", "required": False,
                      "description": "Model units per page/drawing unit (m per PDF point, or m per pixel for raster). Promotes coordinates to model space (meters). Omit to stay in page/pixel units."},
            "units": {"type": "str", "required": False,
                      "allowed_values": ["m", "mm", "cm", "ft", "in"],
                      "description": "DXF drawing units override (default: the DXF $INSUNITS header, else meters). dxf only."},
            "origin": {"type": "str", "required": False, "default": "bottom_left",
                       "allowed_values": ["bottom_left", "top_left"],
                       "description": "Y-orientation for PDF/raster (bottom_left = engineering up-positive)."},
            "calibration": {"type": "dict", "required": False,
                            "description": "Two-point PDF scale calibration {p1:[x,y], p2:[x,y], distance_m}. Alternative to scale."},
            "detect_lines": {"type": "bool", "required": False, "default": True,
                             "description": "Raster: detect straight segments (Hough)."},
            "detect_circles": {"type": "bool", "required": False, "default": True,
                               "description": "Raster: detect circles (Hough gradient)."},
            "detect_contours": {"type": "bool", "required": False, "default": True,
                                "description": "Raster: trace closed shapes (contours). May overlap Hough lines on line-work."},
            "ocr": {"type": "bool", "required": False, "default": True,
                    "description": "Raster: attempt OCR text (needs pytesseract + Tesseract; skipped with a warning if absent)."},
        },
        "returns": {
            "handle": "Cache handle for query_drawing / get_entities.",
            "source": "dxf | pdf_vector | raster_trace.",
            "counts_by_type": "Entity counts per type.",
            "counts_by_layer": "Entity counts per layer/group.",
            "page": "Page size, units, coordinate_space, origin.",
            "scale": "Applied model scale (or null for page space).",
            "bbox": "Overall extent [x_min,y_min,x_max,y_max].",
            "scale_candidates": "PDF-only: proposed scales from page text (proposals, not applied).",
        },
    },
    "query_drawing": {
        "category": "Drawing IR",
        "brief": ("Request a SLICE of a cached drawing IR by handle. Returns "
                  "compact entity references (id + small summary), not full "
                  "coordinate dumps — follow up with get_entities for exact "
                  "coordinates of a shortlist."),
        "parameters": {
            "handle": {"type": "str", "required": True,
                       "description": "Handle from digitize_drawing."},
            "query": {"type": "str", "required": True,
                      "allowed_values": QUERY_NAMES,
                      "description": "Which slice to compute."},
            "params": {"type": "dict", "required": False,
                       "description": ("Query params. entities_in_bbox: "
                                       "{x_min,y_min,x_max,y_max,mode?,entity_type?}; "
                                       "nearest_entity: {x,y,entity_type?,k?}; "
                                       "lines_by_angle: {min_deg,max_deg}; "
                                       "horizontal_lines/vertical_lines: {tol_deg?}; "
                                       "polylines_longer_than: {min_length}; "
                                       "text_items: {pattern?}; "
                                       "text_near: {entity_id,radius}; "
                                       "entities_on_layer: {layer}; "
                                       "entities_by_color: {color}; "
                                       "candidate_ground_surface/summary_stats: {}.")},
        },
        "returns": {
            "result": "Query output — a list of entity refs or a stats/proposal dict.",
            "n_results": "List length when the result is a list.",
        },
    },
    "get_entities": {
        "category": "Drawing IR",
        "brief": ("Fetch full, exact coordinates for specific entity ids from a "
                  "cached drawing IR (the deterministic coordinates the LLM "
                  "should trust over its own pixel reading)."),
        "parameters": {
            "handle": {"type": "str", "required": True,
                       "description": "Handle from digitize_drawing."},
            "ids": {"type": "array", "required": True,
                    "description": "Entity ids to retrieve (e.g. ['e0','e5'])."},
        },
        "returns": {"entities": "Full entity dicts with exact coordinates."},
    },
}

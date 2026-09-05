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

# Set-level digitization cache for search_drawing_set: repeated queries over
# the same drawing set (the chatbot norm: "how many X", then "where", then
# "show me") re-digitize nothing. Keyed by (path, mtime, page); bounded like
# the handle cache. Real dense sheets cost 0.1-13 s each to digitize.
_SET_IR_CACHE = {}
_SET_IR_ORDER = []
_SET_IR_CACHE_MAX = 64
# Cached IRs are shared, mutable objects (OCR augmentation extends them in
# place, guarded by ir.metadata["ocr"]) and OCR takes minutes — without a
# lock, two concurrent conversations hitting the same uncached sheet both
# pass the guard and permanently DOUBLE the OCR text items (verifier
# finding, 2026-09-05). One lock per cache key serializes digitize+augment;
# _SET_IR_LOCKS is only ever mutated under _SET_IR_LOCKS_GUARD.
import threading

_SET_IR_LOCKS = {}
_SET_IR_LOCKS_GUARD = threading.Lock()


def _set_ir_lock(key):
    with _SET_IR_LOCKS_GUARD:
        lock = _SET_IR_LOCKS.get(key)
        if lock is None:
            lock = _SET_IR_LOCKS[key] = threading.Lock()
        return lock


def _digitized_cached(fp, page, source):
    try:
        mtime = os.path.getmtime(fp)
    except OSError:
        mtime = None
    key = (os.path.abspath(str(fp)), mtime, page, source)
    with _set_ir_lock(key):
        ir = _SET_IR_CACHE.get(key)
        if ir is None:
            from planlens.ir import from_dxf, from_pdf_vector
            ir = (from_pdf_vector(filepath=fp, page=page)
                  if source == "pdf_vector" else from_dxf(filepath=fp))
            _SET_IR_CACHE[key] = ir
            _SET_IR_ORDER.append(key)
            while len(_SET_IR_ORDER) > _SET_IR_CACHE_MAX:
                _SET_IR_CACHE.pop(_SET_IR_ORDER.pop(0), None)
    return ir


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
    from planlens.ir import from_dxf, from_pdf_vector, from_raster, queries

    _valid = ("file_path", "source", "page", "scale", "units", "origin",
              "calibration", "detect_lines", "detect_circles",
              "detect_contours", "ocr", "ocr_text")
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

    ocr_out = None
    if params.get("ocr_text") and source == "pdf_vector":
        from planlens.ocr import augment_ir_with_ocr
        ocr_out = augment_ir_with_ocr(ir, filepath=file_path,
                                      page=params.get("page", 0))

    handle = _store_ir(ir)
    out = {"handle": handle, "source": ir.source}
    out.update(queries.summary_stats(ir))
    if ocr_out is not None:
        out["ocr"] = ocr_out
    if ir.metadata.get("scale_candidates"):
        out["scale_candidates"] = ir.metadata["scale_candidates"]
    out["note"] = ("IR cached under 'handle'. Use query_drawing to request "
                   "slices and get_entities for exact coordinates of specific "
                   "ids. Confidence < 1.0 marks raster/OCR detections.")
    if not out.get("has_text") and source == "pdf_vector" \
            and ocr_out is None:
        out["note"] += (
            " This sheet has NO extractable text layer (SHX/stroked "
            "lettering): re-digitize with ocr_text=true to read the "
            "lettering optically before running text queries.")
    return clean_result(out)


# ---------------------------------------------------------------------------
# query_drawing
# ---------------------------------------------------------------------------

# name -> (function, required-param-names, all-param-names)
def _query_registry():
    from planlens.ir import queries as q

    def _ending_near(ir, x, y, radius, entity_types=None):
        return q.entities_ending_near(ir, (x, y), radius,
                                      entity_types=entity_types)

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
        "entities_ending_near": (_ending_near, ["x", "y", "radius"],
                                 ["x", "y", "radius", "entity_types"]),
        "text_anchored_geometry": (q.text_anchored_geometry, ["pattern"],
                                   ["pattern", "radius", "entity_types"]),
        "find_leaders": (q.find_leaders, [],
                         ["max_arrowhead_size", "search_radius",
                          "text_radius", "min_confidence",
                          "exclude_dimensions", "dimension_confidence"]),
        "find_dimensions": (q.find_dimensions, [],
                            ["max_arrowhead_size", "search_radius",
                             "text_radius", "min_confidence"]),
        "find_title_block": (q.find_title_block, [],
                             ["edge_frac", "min_confidence"]),
        "find_bubble_callouts": (q.find_bubble_callouts, [],
                                 ["max_radius", "text_max_chars",
                                  "min_confidence"]),
        "find_revision_clouds": (q.find_revision_clouds, [],
                                 ["min_arcs", "min_confidence"]),
        "entities_on_layer": (q.entities_on_layer, ["layer"], ["layer"]),
        "entities_by_color": (q.entities_by_color, ["color"], ["color"]),
        "candidate_ground_surface": (q.candidate_ground_surface, [], []),
        "summary_stats": (q.summary_stats, [], []),
    }


QUERY_NAMES = sorted([
    "entities_in_bbox", "nearest_entity", "lines_by_angle", "horizontal_lines",
    "vertical_lines", "polylines_longer_than", "text_items", "text_near",
    "entities_ending_near", "text_anchored_geometry", "find_leaders",
    "find_dimensions", "find_title_block", "find_bubble_callouts",
    "find_revision_clouds",
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
    from planlens.ir import queries

    reject_unknown_params(params, ("handle", "ids"), method="get_entities")
    require_params(params, ["handle", "ids"], method="get_entities",
                   valid=["handle", "ids"])
    ir = _get_ir(params["handle"])
    ids = params["ids"]
    if isinstance(ids, str):
        ids = [ids]
    return clean_result({"handle": params["handle"],
                         "entities": queries.get_entities(ir, list(ids))})


# ---------------------------------------------------------------------------
# snip_region — save a zoomed crop of a PDF page to a PNG file
# ---------------------------------------------------------------------------

def _run_snip_region(params):
    _valid = ("file_path", "output_path", "page", "bbox", "frame", "scale",
              "dpi", "pad_frac", "marks")
    reject_unknown_params(params, _valid, method="snip_region")
    require_params(params, ["file_path", "output_path"], method="snip_region",
                   valid=_valid)
    file_path = params["file_path"]
    page = params.get("page", 0)
    bbox = params.get("bbox")
    frame = params.get("frame", "ir")
    scale = params.get("scale")
    marks = params.get("marks")
    if frame not in ("ir", "pdf"):
        raise ValueError("frame must be 'ir' (bottom-left, y up — the frame "
                         "digitize_drawing/query_drawing coordinates are in) "
                         "or 'pdf' (top-left points, y down).")

    if frame == "ir" and (bbox is not None or marks):
        import fitz
        with fitz.open(file_path) as doc:
            if page >= len(doc):
                raise ValueError(f"Page {page} out of range "
                                 f"(document has {len(doc)} pages)")
            height_pt = doc[page].rect.height

        def to_pdf(x, y):
            if scale:
                x, y = x / scale, y / scale
            return x, height_pt - y

        if bbox is not None:
            x0, y0 = to_pdf(bbox[0], bbox[1])
            x1, y1 = to_pdf(bbox[2], bbox[3])
            bbox = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        if marks:
            conv = []
            for m in marks:
                mx, my = to_pdf(m[0], m[1])
                conv.append((mx, my) + tuple(m[2:3]))
            marks = conv

    from funhouse_agent.vision_tools import render_region_to_file
    saved = render_region_to_file(
        params["output_path"], filepath=file_path, page=page,
        bbox=tuple(bbox) if bbox is not None else None,
        dpi=params.get("dpi", 300), pad_frac=params.get("pad_frac", 0.15),
        marks=[tuple(m) for m in marks] if marks else None)
    return clean_result({
        "saved": saved, "page": page,
        "note": ("PNG crop written. Analyze it with analyze_image (pass this "
                 "path), or use the render_region tool to render+analyze in "
                 "one step."),
    })


# ---------------------------------------------------------------------------
# search_drawing_set — iterate every page of one or more files (P2-C)
# ---------------------------------------------------------------------------

_CONSTRUCT_FINDERS = ("leaders", "dimensions", "title_block",
                      "bubble_callouts", "revision_clouds")
#: Compact per-page result cap so a busy set stays JSON-budget friendly.
_SET_PAGE_RESULT_CAP = 20


def _construct_summary(name, p):
    """One compact line per proposal (location + label + confidence)."""
    out = {"confidence": p.get("confidence")}
    for k in ("text", "kind"):
        if p.get(k) is not None:
            out[k] = p[k]
    for k in ("tip_xy", "midpoint_xy", "center_xy"):
        if p.get(k) is not None:
            out["at"] = p[k]
            break
    if "at" not in out and p.get("region_bbox"):
        out["at"] = p["region_bbox"]
    if "at" not in out and p.get("bbox"):
        out["at"] = p["bbox"]
    return out


def _run_search_drawing_set(params):
    from planlens.ir import from_dxf, from_pdf_vector, queries as q

    _valid = ("file_paths", "pattern", "construct", "pages",
              "min_confidence", "ocr_text")
    reject_unknown_params(params, _valid, method="search_drawing_set")
    require_params(params, ["file_paths"], method="search_drawing_set",
                   valid=_valid)
    file_paths = params["file_paths"]
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    pattern = params.get("pattern")
    construct = params.get("construct")
    ocr_text = bool(params.get("ocr_text", False))
    min_conf = params.get("min_confidence", 0.5)
    if construct is not None and construct not in _CONSTRUCT_FINDERS:
        raise ValueError(f"Unknown construct '{construct}'. "
                         f"Use one of {list(_CONSTRUCT_FINDERS)}.")
    if pattern is None and construct is None:
        raise ValueError("Give 'pattern' (text search), 'construct' "
                         "(annotation search), or both (construct results "
                         "filtered to those whose text matches pattern).")

    finders = {
        "leaders": lambda ir: q.find_leaders(
            ir, min_confidence=min_conf, exclude_dimensions=True),
        "dimensions": lambda ir: q.find_dimensions(
            ir, min_confidence=min_conf),
        "title_block": lambda ir: q.find_title_block(
            ir, min_confidence=min_conf),
        "bubble_callouts": lambda ir: q.find_bubble_callouts(
            ir, min_confidence=min_conf),
        "revision_clouds": lambda ir: q.find_revision_clouds(
            ir, min_confidence=min_conf),
    }

    files_out = []
    total = 0
    for fp in file_paths:
        source = _auto_source(fp)
        if source == "pdf_vector":
            import fitz
            with fitz.open(fp) as doc:
                n_pages = len(doc)
            pages = params.get("pages")
            page_list = (list(range(n_pages)) if pages is None
                         else [p for p in
                               ([pages] if isinstance(pages, int) else pages)
                               if 0 <= p < n_pages])
        elif source == "dxf":
            n_pages, page_list = 1, [0]
        else:
            raise ValueError(
                f"'{fp}': search_drawing_set handles PDF and DXF files; "
                "digitize a raster image with digitize_drawing instead.")

        pages_out = []
        file_total = 0
        for pg in page_list:
            ir = _digitized_cached(fp, pg, source)
            if ocr_text and source == "pdf_vector":
                # Read the lettering optically on no-text-layer pages so
                # pattern searches see it. Cached IRs are augmented once —
                # the check and the minutes-long augment share the cache
                # key's lock so concurrent callers can't double-augment.
                try:
                    mtime = os.path.getmtime(fp)
                except OSError:
                    mtime = None
                key = (os.path.abspath(str(fp)), mtime, pg, source)
                with _set_ir_lock(key):
                    if (not (ir.metadata or {}).get("ocr")
                            and not any(e.KIND == "text"
                                        for e in ir.entities)):
                        from planlens.ocr import augment_ir_with_ocr
                        augment_ir_with_ocr(ir, filepath=fp, page=pg)
            if construct is not None:
                props = finders[construct](ir)
                if pattern is not None:
                    low = str(pattern).lower()
                    # Proposals carry text under "text" (leaders/bubbles) or
                    # "texts" (title blocks: a list of contained text items);
                    # cloud proposals have no text and never match a pattern.
                    def _prop_text(p):
                        if p.get("text"):
                            return str(p["text"])
                        return " ".join(str(t.get("content", ""))
                                        for t in (p.get("texts") or []))
                    props = [p for p in props if low in _prop_text(p).lower()]
                entries = [_construct_summary(construct, p) for p in props]
            else:
                entries = [{"text": t["content"], "at": t["position"]}
                           for t in q.text_items(ir, pattern)]
            n = len(entries)
            file_total += n
            page_out = {"page": pg, "count": n}
            if (pattern is not None and construct is None and n == 0
                    and ir.entities
                    and not any(e.KIND == "text" for e in ir.entities)):
                page_out["no_text_layer"] = True
                page_out["note"] = (
                    "This page has NO extractable text at all (likely SHX/"
                    "stroked lettering) — a zero count here does not mean "
                    "the string is absent from the sheet. Re-run with "
                    "ocr_text=true to read the lettering optically "
                    "(planlens[ocr]), or inspect via snip_region + vision.")
            if n:
                page_out["matches"] = entries[:_SET_PAGE_RESULT_CAP]
                if n > _SET_PAGE_RESULT_CAP:
                    page_out["matches_truncated"] = True
            pages_out.append(page_out)
        total += file_total
        files_out.append({"file_path": fp, "n_pages": n_pages,
                          "count": file_total, "pages": pages_out})

    out = {
        "total_count": total,
        "files": files_out,
        "proposal_only": construct is not None,
    }
    if construct is not None:
        out["note"] = (f"'{construct}' matches are confidence-scored "
                       "PROPOSALS (min_confidence={:g}) composed from "
                       "geometry — confirm important ones visually "
                       "(snip_region / render_region on 'at')."
                       .format(min_conf))
    return clean_result(out)


METHOD_REGISTRY = {
    "digitize_drawing": _run_digitize_drawing,
    "query_drawing": _run_query_drawing,
    "get_entities": _run_get_entities,
    "snip_region": _run_snip_region,
    "search_drawing_set": _run_search_drawing_set,
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
            "ocr_text": {"type": "bool", "required": False, "default": False,
                         "description": (
                             "PDF only: additionally OCR the rendered page "
                             "and merge recognized text into the IR as "
                             "confidence<1 text entities (source='ocr'). "
                             "THE way to read SHX/stroked-lettering sheets "
                             "(has_text=false) — needs planlens[ocr] "
                             "(RapidOCR; auto-detects sideways plots). "
                             "Adds seconds to a-minute-ish per page.")},
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
            "has_text": ("Whether the sheet carries ANY extractable text. "
                         "False = likely SHX/stroked lettering (common on "
                         "agency PDFs): text queries return nothing there "
                         "unless digitized with ocr_text=true; geometry/"
                         "construct queries still work."),
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
                                       "entities_ending_near: {x,y,radius,entity_types?} "
                                       "(what TERMINATES at a point — arrow tips, callout tails); "
                                       "text_anchored_geometry: {pattern,radius?,entity_types?} "
                                       "(find text X -> geometry ending at it -> the far point it "
                                       "points at; pattern is any runtime string); "
                                       "find_leaders: {min_confidence?,exclude_dimensions?,...}; "
                                       "find_dimensions/find_title_block/find_bubble_callouts/"
                                       "find_revision_clouds: {min_confidence?,...} "
                                       "(annotation-construct PROPOSALS with confidence + "
                                       "evidence — never asserted facts; confirm visually). "
                                       "On no-text-layer (SHX-stroked) sheets find_dimensions "
                                       "detects real plotted dims (split-shaft + continuous "
                                       "legs, 13/16 native defpoints on validation sheets; "
                                       "ends = arrow apexes) but some high-confidence hits "
                                       "may be other line-plus-arrows constructs — verify "
                                       "via snip_region before reporting values; "
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
    "snip_region": {
        "category": "Drawing IR",
        "brief": ("Render a zoomed-in crop of a PDF drawing page to a PNG "
                  "file — the 'geometry says WHERE, vision says WHAT' "
                  "primitive. Feed exact coordinates from query_drawing "
                  "(e.g. a leader's tip_xy, a title block's region_bbox), "
                  "then analyze the saved PNG with analyze_image (or use "
                  "the one-step render_region tool)."),
        "parameters": {
            "file_path": {"type": "str", "required": True,
                          "description": "Path to the source PDF."},
            "output_path": {"type": "str", "required": True,
                            "description": "Where to save the PNG crop."},
            "page": {"type": "int", "required": False, "default": 0,
                     "description": "PDF page (0-indexed)."},
            "bbox": {"type": "array", "required": False,
                     "description": ("[x0,y0,x1,y1] region of interest. "
                                     "Omit for the full page.")},
            "frame": {"type": "str", "required": False, "default": "ir",
                      "allowed_values": ["ir", "pdf"],
                      "description": ("Coordinate frame of bbox/marks: 'ir' "
                                      "(bottom-left origin, y up — what "
                                      "digitize_drawing/query_drawing "
                                      "return, the default) or 'pdf' "
                                      "(top-left points, y down).")},
            "scale": {"type": "float", "required": False,
                      "description": ("Model units per PDF point, ONLY if "
                                      "the drawing was digitized with a "
                                      "scale (model-space IR coords are "
                                      "divided by this before the flip).")},
            "dpi": {"type": "int", "required": False, "default": 300,
                    "description": "Render resolution."},
            "pad_frac": {"type": "float", "required": False, "default": 0.15,
                         "description": "Fractional padding around bbox."},
            "marks": {"type": "array", "required": False,
                      "description": ("[[x,y,label], ...] numbered marker "
                                      "circles drawn at points of interest "
                                      "(set-of-marks: 'what is mark 1 "
                                      "pointing at?'). Same frame as bbox.")},
        },
        "returns": {"saved": "Absolute path of the written PNG."},
    },
    "search_drawing_set": {
        "category": "Drawing IR",
        "brief": ("Search EVERY page of one or more drawing files (PDF/DXF) "
                  "in one call: count/locate text matches ('how many times "
                  "does X occur in this set') or annotation constructs "
                  "(leaders, dimensions, title_block, bubble_callouts, "
                  "revision_clouds — confidence-scored proposals). Returns "
                  "per-file, per-page counts + compact match locations. "
                  "Digitized pages are cached across calls (first pass on a "
                  "dense sheet can take seconds)."),
        "parameters": {
            "file_paths": {"type": "array", "required": True,
                           "description": ("One path or a list of paths "
                                           "(.pdf or .dxf).")},
            "pattern": {"type": "str", "required": False,
                        "description": ("Text to find (regex, else literal "
                                        "substring; case-insensitive). Any "
                                        "runtime string. With 'construct', "
                                        "filters constructs by their text "
                                        "(title blocks: matched against all "
                                        "contained text; revision clouds "
                                        "carry no text and never match a "
                                        "pattern). "
                                        "NOTE: sheets plotted with SHX/"
                                        "stroked lettering have NO text "
                                        "layer — such pages are flagged "
                                        "no_text_layer and a zero count is "
                                        "inconclusive; pass ocr_text=true "
                                        "to read them optically.")},
            "ocr_text": {"type": "bool", "required": False, "default": False,
                         "description": (
                             "OCR pages that have no text layer before the "
                             "pattern search (planlens[ocr]; adds seconds "
                             "per no-text page, cached across calls).")},
            "construct": {"type": "str", "required": False,
                          "allowed_values": list(_CONSTRUCT_FINDERS),
                          "description": ("Annotation construct to search "
                                          "for instead of (or as well as) "
                                          "raw text.")},
            "pages": {"type": "array", "required": False,
                      "description": ("Page numbers (0-indexed) to search; "
                                      "omit for ALL pages of each file.")},
            "min_confidence": {"type": "float", "required": False,
                               "default": 0.5,
                               "description": ("Construct proposals below "
                                               "this are dropped.")},
        },
        "returns": {
            "total_count": "Matches across all files/pages.",
            "files": ("Per file: count + per-page {page, count, matches "
                      "[{text/kind, at, confidence}]} (capped per page)."),
            "proposal_only": "True when construct results are proposals.",
        },
    },
}

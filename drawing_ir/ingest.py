"""
Ingest adapters — build a :class:`drawing_ir.results.DrawingIR` from a source.

Three legs, each stamping provenance + confidence on every entity:

* :func:`from_dxf`        — exact CAD geometry via ezdxf. source='dxf',
  confidence 1.0. Native model-space coordinates converted to SI meters using
  the drawing's ``$INSUNITS`` (or a supplied ``units``). Carries layers/colors.
* :func:`from_pdf_vector` — exact PDF path coordinates via PyMuPDF, reusing the
  ``pdf_import`` extractor + scale module. source='pdf_vector', confidence 1.0.
  Page-point coordinates, promoted to model meters when a scale/calibration is
  given (else page space, units='pt') with scale candidates proposed as
  metadata.
* :func:`from_raster`     — best-effort OpenCV tracing (delegates to
  ``drawing_ir.raster``). source='raster_trace', confidence < 1.

Import convention: this module (the I/O ingest layer) legally imports the
``dxf_import`` / ``pdf_import`` I/O modules — the same pattern
``geo_project/ingest.py`` uses. The "no cross-module imports" house rule targets
the 30 computational *analysis* modules; the I/O modules
(dxf_import <- pdf_import <- geo_project) already form a dependency layer, which
``drawing_ir`` joins. ``results.py`` / ``queries.py`` stay pure-schema.

Geometry only: a drawing carries geometry, never soil properties — those always
come from the user/report downstream.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from drawing_ir.results import (
    Arc, Circle, DrawingIR, Line, Polyline, Region, TextItem,
)


# ---------------------------------------------------------------------------
# DXF
# ---------------------------------------------------------------------------

def _dxf_color(entity) -> Optional[str]:
    """Best-effort color: true-color hex, else an ACI token, else None."""
    try:
        rgb = entity.rgb
    except Exception:
        rgb = None
    if rgb:
        r, g, b = rgb
        return f"#{r:02x}{g:02x}{b:02x}"
    try:
        aci = entity.dxf.color
    except Exception:
        aci = 256
    if aci and 0 < aci < 256:
        return f"ACI{aci}"
    return None


def _dxf_style(entity) -> Optional[str]:
    try:
        lt = entity.dxf.linetype
    except Exception:
        return None
    if lt and lt.upper() != "BYLAYER":
        return lt
    return None


def from_dxf(filepath: str = None, content: bytes = None,
             units: Optional[str] = None, flip_y: bool = False,
             name: str = "DXF import") -> DrawingIR:
    """Build a DrawingIR from a DXF file's model space (exact coordinates).

    Extracts LINE, LWPOLYLINE/POLYLINE, ARC, CIRCLE, ELLIPSE/SPLINE (flattened),
    TEXT/MTEXT and (best-effort) HATCH, each with its layer, color and linetype.
    Coordinates are converted to SI meters using ``units`` (default: the DXF
    ``$INSUNITS`` header, else 'm').
    """
    import os
    import tempfile
    try:
        import ezdxf
    except ImportError as exc:
        raise ImportError("ezdxf is required for DXF ingest. Install with: "
                          "pip install ezdxf>=1.4") from exc
    from dxf_import.units import UNIT_FACTORS, detect_units_from_header

    if filepath is None and content is None:
        raise ValueError("Provide either filepath or content")

    tmp_path = None
    if content is not None:
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            doc = ezdxf.readfile(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        doc = ezdxf.readfile(filepath)

    resolved_units = units or detect_units_from_header(doc) or "m"
    factor = UNIT_FACTORS.get(resolved_units, 1.0)
    warnings: List[str] = []

    def conv(x, y):
        yy = -y if flip_y else y
        return (x * factor, yy * factor)

    msp = doc.modelspace()
    ir = DrawingIR(
        units="m", coordinate_space="model", origin="bottom_left",
        source="dxf", warnings=warnings,
        scale=(factor if factor != 1.0 else None),
        scale_provenance=(f"dxf_units:{resolved_units}->m"
                          if factor != 1.0 else "dxf_native_meters"),
    )

    layers = set()
    for ent in msp:
        etype = ent.dxftype()
        layer = getattr(ent.dxf, "layer", None)
        layers.add(layer)
        common = dict(layer=layer, color=_dxf_color(ent),
                      style=_dxf_style(ent), source="dxf", confidence=1.0)
        try:
            if etype == "LINE":
                s, e = ent.dxf.start, ent.dxf.end
                ir.add(Line(start=conv(s.x, s.y), end=conv(e.x, e.y), **common))
            elif etype == "LWPOLYLINE":
                verts = [conv(x, y) for x, y in ent.get_points(format="xy")]
                ir.add(Polyline(vertices=verts, closed=bool(ent.closed),
                                **common))
            elif etype == "POLYLINE":
                verts = [conv(v.dxf.location.x, v.dxf.location.y)
                         for v in ent.vertices]
                ir.add(Polyline(vertices=verts,
                                closed=bool(ent.is_closed), **common))
            elif etype == "ARC":
                c = ent.dxf.center
                sa = ent.dxf.start_angle
                ea = ent.dxf.end_angle
                if flip_y:
                    sa, ea = (-ea) % 360.0, (-sa) % 360.0
                ir.add(Arc(center=conv(c.x, c.y),
                           radius=ent.dxf.radius * factor,
                           start_angle=sa, end_angle=ea, **common))
            elif etype == "CIRCLE":
                c = ent.dxf.center
                ir.add(Circle(center=conv(c.x, c.y),
                              radius=ent.dxf.radius * factor, **common))
            elif etype in ("ELLIPSE", "SPLINE"):
                try:
                    verts = [conv(p.x, p.y) for p in ent.flattening(0.01)]
                except Exception:
                    verts = []
                if len(verts) >= 2:
                    st = dict(common)
                    st["style"] = f"approx_from_{etype.lower()}"
                    ir.add(Polyline(vertices=verts, closed=False, **st))
            elif etype == "TEXT":
                ins = ent.dxf.insert
                ir.add(TextItem(content=ent.dxf.text,
                                position=conv(ins.x, ins.y),
                                rotation=getattr(ent.dxf, "rotation", 0.0),
                                height=getattr(ent.dxf, "height", 0.0) * factor,
                                **common))
            elif etype == "MTEXT":
                ins = ent.dxf.insert
                ir.add(TextItem(content=ent.text,
                                position=conv(ins.x, ins.y),
                                rotation=getattr(ent.dxf, "rotation", 0.0),
                                height=getattr(ent.dxf, "char_height", 0.0)
                                * factor, **common))
            elif etype == "HATCH":
                region = _hatch_region(ent, conv, common)
                if region is not None:
                    ir.add(region)
            # other entity types are skipped (kept honest — not fabricated)
        except Exception as exc:  # pragma: no cover - malformed entity guard
            warnings.append(f"Skipped {etype} on layer '{layer}': {exc}")

    ir.metadata = {"dxf_units": resolved_units,
                   "n_layers": len([lyr for lyr in layers if lyr is not None])}
    if not ir.entities:
        warnings.append("No supported entities found in DXF model space.")
    return ir


def _hatch_region(ent, conv, common) -> Optional[Region]:
    """Best-effort HATCH → Region using the first boundary path's vertices."""
    try:
        pattern = getattr(ent.dxf, "pattern_name", None)
        for path in ent.paths:
            verts = []
            for v in getattr(path, "vertices", []) or []:
                verts.append(conv(v[0], v[1]))
            if len(verts) >= 3:
                st = dict(common)
                return Region(boundary=verts, pattern=pattern, **st)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# PDF (vector)
# ---------------------------------------------------------------------------

def _closed(pts, tol=1e-6) -> bool:
    return (len(pts) >= 3
            and abs(pts[0][0] - pts[-1][0]) <= tol
            and abs(pts[0][1] - pts[-1][1]) <= tol)


def from_pdf_vector(filepath: str = None, content: bytes = None,
                    page: int = 0, scale: Optional[float] = None,
                    calibration: Optional[Dict[str, Any]] = None,
                    origin: str = "bottom_left",
                    name: str = "PDF vector import") -> DrawingIR:
    """Build a DrawingIR from a PDF page's vector line-work + text.

    Reuses ``pdf_import.extract_colored_paths`` (per-path point lists + color)
    and ``pdf_import.discover_pdf_content`` (page size + text). When ``scale``
    (meters per PDF point) or a two-point ``calibration`` ({p1, p2, distance_m})
    is supplied, coordinates are promoted to model meters; otherwise the IR
    stays in page points and scale CANDIDATES are attached to metadata (via the
    ``pdf_import`` scale module) as proposals, never applied.
    """
    from pdf_import import (
        calibrate_scale, discover_pdf_content, extract_colored_paths,
        propose_scale,
    )

    info = discover_pdf_content(filepath=filepath, content=content, page=page)
    width_pt = info["page_size"]["width"]
    height_pt = info["page_size"]["height"]
    text_blocks = info.get("text_blocks", [])
    # Page-point path coordinates (flipped per origin, scale=1.0).
    regions = extract_colored_paths(filepath=filepath, content=content,
                                    page=page, scale=1.0, origin=origin)

    # Resolve scale factor (meters per point) if any.
    sf = None
    provenance = None
    metadata: Dict[str, Any] = {}
    if scale is not None:
        sf = float(scale)
        provenance = f"explicit_scale:{sf:g} m/pt"
    elif calibration:
        sf = calibrate_scale(tuple(calibration["p1"]), tuple(calibration["p2"]),
                             calibration["distance_m"])
        provenance = "two_point_calibration"
    else:
        proposals = propose_scale(text_blocks, calibration=None)
        if proposals.get("candidates"):
            metadata["scale_candidates"] = proposals["candidates"]
            metadata["scale_note"] = proposals["note"]

    is_model = sf is not None
    smul = sf if is_model else 1.0

    def apply_scale(x, y):
        return (x * smul, y * smul)

    def text_xy(x, y):
        yy = (height_pt - y) if origin == "bottom_left" else y
        return apply_scale(x, yy)

    ir = DrawingIR(
        width=(width_pt * smul), height=(height_pt * smul),
        units=("m" if is_model else "pt"),
        coordinate_space=("model" if is_model else "page"),
        scale=sf, scale_provenance=provenance, origin=origin,
        source="pdf_vector", metadata=metadata,
    )

    for reg in regions:
        pts = [apply_scale(x, y) for x, y in reg.get("points", [])]
        color = reg.get("color")
        if len(pts) < 2:
            continue
        if len(pts) == 2:
            ir.add(Line(start=pts[0], end=pts[1], color=color,
                        source="pdf_vector", confidence=1.0))
        else:
            closed = _closed(pts)
            verts = pts[:-1] if closed else pts
            ir.add(Polyline(vertices=verts, closed=closed, color=color,
                            source="pdf_vector", confidence=1.0))

    for tb in text_blocks:
        txt = (tb.get("text") or "").strip()
        if not txt:
            continue
        size = tb.get("size", 0.0)
        ir.add(TextItem(content=txt, position=text_xy(tb["x"], tb["y"]),
                        rotation=0.0, height=size * smul,
                        source="pdf_vector", confidence=1.0))

    ir.metadata.setdefault("page_number", page)
    if not ir.entities:
        ir.warnings.append(
            "No vector paths or text found on this page — the drawing may be a "
            "raster scan (try from_raster) or a different page.")
    return ir


# ---------------------------------------------------------------------------
# Raster (delegates to the OpenCV leg — keeps cv2 optional)
# ---------------------------------------------------------------------------

def from_raster(filepath: str = None, image: Any = None, **kwargs) -> DrawingIR:
    """Build a DrawingIR by tracing a raster image (source='raster_trace').

    Thin wrapper over :func:`drawing_ir.raster.trace_raster` so the OpenCV
    dependency stays isolated to the raster leg. See that function for the full
    parameter list (scale, detectors, Hough/contour tuning, OCR).
    """
    from drawing_ir.raster import trace_raster
    return trace_raster(filepath=filepath, image=image, **kwargs)


__all__ = ["from_dxf", "from_pdf_vector", "from_raster"]

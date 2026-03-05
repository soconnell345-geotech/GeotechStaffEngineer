"""
DXF export — write cross-section geometry to DXF files using ezdxf.

Layer conventions
-----------------
| Layer            | Color | Entity     | Source field          |
|------------------|-------|------------|-----------------------|
| SURFACE          | 3     | LWPOLYLINE | surface_points        |
| BOUNDARY_<name>  | 1     | LWPOLYLINE | boundary_profiles     |
| GWT              | 4     | LWPOLYLINE | gwt_points            |
| NAILS            | 6     | LINE       | nail_lines            |
| ANNOTATIONS      | 7     | TEXT       | text_annotations      |

Requires: ezdxf >= 1.4 (optional dependency: pip install geotech-staff-engineer[dxf])
"""

import io
from typing import Any, Dict, List, Optional, Tuple

from dxf_export.results import DxfExportResult

# $INSUNITS values
_INSUNITS = {
    "m": 6,
    "meters": 6,
    "mm": 4,
    "millimeters": 4,
    "cm": 5,
    "centimeters": 5,
    "ft": 2,
    "feet": 2,
    "in": 1,
    "inches": 1,
}

# Layer color assignments
_LAYER_COLORS = {
    "SURFACE": 3,       # green
    "GWT": 4,           # cyan
    "NAILS": 6,         # magenta
    "ANNOTATIONS": 7,   # white
}
_BOUNDARY_COLOR = 1     # red


def _build_dxf_document(
    surface_points: Optional[List[Tuple[float, float]]] = None,
    boundary_profiles: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    gwt_points: Optional[List[Tuple[float, float]]] = None,
    nail_lines: Optional[List[Dict[str, float]]] = None,
    text_annotations: Optional[List[Dict[str, Any]]] = None,
    units: str = "m",
    dxf_version: str = "R2010",
):
    """Build an ezdxf document from geometry inputs.

    Returns (doc, result_fields) where result_fields is a dict of counts.
    """
    import ezdxf

    surface_points = surface_points or []
    boundary_profiles = boundary_profiles or {}
    gwt_points = gwt_points or []
    nail_lines = nail_lines or []
    text_annotations = text_annotations or []

    doc = ezdxf.new(dxfversion=dxf_version)
    msp = doc.modelspace()

    # Set units header
    units_key = units.lower().strip()
    if units_key in _INSUNITS:
        doc.header["$INSUNITS"] = _INSUNITS[units_key]

    layers_created = []
    n_entities = 0
    warnings = []

    # --- Surface polyline ---
    surface_pts_written = 0
    if surface_points:
        doc.layers.add("SURFACE", color=_LAYER_COLORS["SURFACE"])
        layers_created.append("SURFACE")
        msp.add_lwpolyline(
            surface_points, dxfattribs={"layer": "SURFACE"}
        )
        n_entities += 1
        surface_pts_written = len(surface_points)

    # --- Boundary profiles ---
    boundary_count = 0
    for name, pts in boundary_profiles.items():
        if not pts:
            warnings.append(f"Boundary '{name}' has no points, skipped.")
            continue
        layer_name = f"BOUNDARY_{name}"
        doc.layers.add(layer_name, color=_BOUNDARY_COLOR)
        layers_created.append(layer_name)
        msp.add_lwpolyline(pts, dxfattribs={"layer": layer_name})
        n_entities += 1
        boundary_count += 1

    # --- GWT polyline ---
    gwt_pts_written = 0
    if gwt_points:
        doc.layers.add("GWT", color=_LAYER_COLORS["GWT"])
        layers_created.append("GWT")
        msp.add_lwpolyline(gwt_points, dxfattribs={"layer": "GWT"})
        n_entities += 1
        gwt_pts_written = len(gwt_points)

    # --- Nail lines ---
    nails_written = 0
    if nail_lines:
        doc.layers.add("NAILS", color=_LAYER_COLORS["NAILS"])
        layers_created.append("NAILS")
        for nl in nail_lines:
            start = (nl["x_head"], nl["z_head"])
            end = (nl["x_tip"], nl["z_tip"])
            msp.add_line(start=start, end=end, dxfattribs={"layer": "NAILS"})
            n_entities += 1
            nails_written += 1

    # --- Text annotations ---
    texts_written = 0
    if text_annotations:
        doc.layers.add("ANNOTATIONS", color=_LAYER_COLORS["ANNOTATIONS"])
        layers_created.append("ANNOTATIONS")
        for ann in text_annotations:
            text = ann.get("text", "")
            x = ann.get("x", 0.0)
            y = ann.get("y", 0.0)
            height = ann.get("height", 0.5)
            msp.add_text(
                text,
                dxfattribs={
                    "layer": "ANNOTATIONS",
                    "insert": (x, y),
                    "height": height,
                },
            )
            n_entities += 1
            texts_written += 1

    result_fields = {
        "n_layers": len(layers_created),
        "n_entities": n_entities,
        "layers_created": layers_created,
        "surface_points_written": surface_pts_written,
        "boundary_profiles_written": boundary_count,
        "gwt_points_written": gwt_pts_written,
        "nail_lines_written": nails_written,
        "text_annotations_written": texts_written,
        "warnings": warnings,
    }
    return doc, result_fields


def export_to_dxf(
    filepath: str,
    surface_points: Optional[List[Tuple[float, float]]] = None,
    boundary_profiles: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    gwt_points: Optional[List[Tuple[float, float]]] = None,
    nail_lines: Optional[List[Dict[str, float]]] = None,
    text_annotations: Optional[List[Dict[str, Any]]] = None,
    units: str = "m",
    dxf_version: str = "R2010",
) -> DxfExportResult:
    """Export cross-section geometry to a DXF file.

    Parameters
    ----------
    filepath : str
        Output file path for the DXF file.
    surface_points : list of (x, z), optional
        Ground surface profile coordinates.
    boundary_profiles : dict of {name: [(x, z), ...]}, optional
        Soil boundary profiles keyed by soil name.
    gwt_points : list of (x, z), optional
        Groundwater table profile coordinates.
    nail_lines : list of dict, optional
        Each dict: {x_head, z_head, x_tip, z_tip}.
    text_annotations : list of dict, optional
        Each dict: {text, x, y} and optional {height}.
    units : str
        Coordinate units ('m', 'ft', 'mm', etc.). Default 'm'.
    dxf_version : str
        DXF version string. Default 'R2010'.

    Returns
    -------
    DxfExportResult
        Export summary with entity counts and layer info.
    """
    doc, fields = _build_dxf_document(
        surface_points=surface_points,
        boundary_profiles=boundary_profiles,
        gwt_points=gwt_points,
        nail_lines=nail_lines,
        text_annotations=text_annotations,
        units=units,
        dxf_version=dxf_version,
    )
    doc.saveas(filepath)
    return DxfExportResult(filepath=filepath, **fields)


def to_dxf_bytes(
    surface_points: Optional[List[Tuple[float, float]]] = None,
    boundary_profiles: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    gwt_points: Optional[List[Tuple[float, float]]] = None,
    nail_lines: Optional[List[Dict[str, float]]] = None,
    text_annotations: Optional[List[Dict[str, Any]]] = None,
    units: str = "m",
    dxf_version: str = "R2010",
) -> bytes:
    """Export cross-section geometry to DXF content as bytes.

    Same as export_to_dxf but returns raw DXF bytes instead of saving to file.
    Useful for in-memory use and the funhouse agent adapter.

    Parameters
    ----------
    (same as export_to_dxf, minus filepath)

    Returns
    -------
    bytes
        DXF file content.
    """
    doc, _ = _build_dxf_document(
        surface_points=surface_points,
        boundary_profiles=boundary_profiles,
        gwt_points=gwt_points,
        nail_lines=nail_lines,
        text_annotations=text_annotations,
        units=units,
        dxf_version=dxf_version,
    )
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode("utf-8")


def export_parse_result(parse_result, filepath: str) -> DxfExportResult:
    """Export a DxfParseResult or PdfParseResult directly to a DXF file.

    If a PdfParseResult is given, it is automatically converted via
    ``pdf_import.to_dxf_parse_result()`` first.

    Parameters
    ----------
    parse_result : DxfParseResult or PdfParseResult
        Geometry result from dxf_import or pdf_import.
    filepath : str
        Output file path.

    Returns
    -------
    DxfExportResult
    """
    # Duck-type: PdfParseResult has 'extraction_method', DxfParseResult does not
    if hasattr(parse_result, "extraction_method"):
        from pdf_import import to_dxf_parse_result
        parse_result = to_dxf_parse_result(parse_result)

    return export_to_dxf(
        filepath=filepath,
        surface_points=parse_result.surface_points,
        boundary_profiles=parse_result.boundary_profiles,
        gwt_points=parse_result.gwt_points,
        nail_lines=parse_result.nail_lines,
        text_annotations=parse_result.text_annotations,
        units=parse_result.units_used,
    )

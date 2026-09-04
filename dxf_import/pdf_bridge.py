"""PDF -> DXF-parse bridge (geotech side).

``to_dxf_parse_result`` converts a ``planlens.pdf`` ``PdfParseResult`` into
the slope-profile ``DxfParseResult`` consumed by ``build_slope_geometry()`` /
``build_fem_inputs()``. It lives HERE (not in planlens) because the target
type and every consumer are geotech-side; planlens stays free of app coupling.
Moved from ``pdf_import/__init__.py`` in the 2026-09-04 planlens split.
"""

from planlens.pdf.results import PdfParseResult

from dxf_import.results import DxfParseResult


def to_dxf_parse_result(pdf_result: PdfParseResult) -> DxfParseResult:
    """Convert ``PdfParseResult`` to ``DxfParseResult`` for the converters."""
    return DxfParseResult(
        surface_points=list(pdf_result.surface_points),
        boundary_profiles=dict(pdf_result.boundary_profiles),
        gwt_points=list(pdf_result.gwt_points) if pdf_result.gwt_points else None,
        nail_lines=[],
        text_annotations=[
            {"text": a["text"], "x": a["x"], "y": a["y"], "layer": "PDF"}
            for a in pdf_result.text_annotations
        ],
        units_used="m",
        warnings=list(pdf_result.warnings),
    )

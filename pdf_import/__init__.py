"""
PDF Import Module for Geotechnical Cross-Section Extraction

Extracts geometry from PDF drawings using two methods:
    1. Vector extraction — PyMuPDF path analysis (exact, requires role_mapping)
    2. Vision extraction — LLM image analysis (approximate, any drawing)

Output is compatible with both slope_stability and fem2d via
to_dxf_parse_result() adapter for use with build_slope_geometry() and
build_fem_inputs().

Requires: PyMuPDF >= 1.23 (optional dependency)
"""

from pdf_import.results import PdfParseResult
from pdf_import.extractor import discover_pdf_content, extract_vector_geometry
from pdf_import.vision import extract_geometry_vision

__all__ = [
    'PdfParseResult',
    'discover_pdf_content',
    'extract_vector_geometry',
    'extract_geometry_vision',
]


def to_dxf_parse_result(pdf_result: PdfParseResult):
    """Convert PdfParseResult to DxfParseResult for use with build_slope_geometry() / build_fem_inputs().

    Parameters
    ----------
    pdf_result : PdfParseResult
        Output from extract_vector_geometry() or extract_geometry_vision().

    Returns
    -------
    DxfParseResult
        Compatible result for converter functions.
    """
    from dxf_import.results import DxfParseResult

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

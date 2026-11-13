"""
DXF Export Module for Geotechnical Cross-Section Geometry

Exports geometry (surface profiles, soil boundaries, groundwater table,
soil nails, text annotations) to DXF files using ezdxf.

Drawing sheets (BACKLOG GM-313): a typed description of a drawing
(``Drawing``: outlines, bars, dimensions, callouts, notes, a table, a
title block) is written by fixed code as a DXF with real DIMENSION, HATCH
and LEADER entities, and the same drawing as a one-page vector PDF and a
PNG (``write_sheet``). The first drawing is the reinforced cantilever wall
section (``wall_section_drawing``).

Compatible with DxfParseResult from dxf_import and PdfParseResult from
pdf_import for round-trip workflows.

Requires: ezdxf >= 1.4 (optional dependency: pip install geotech-staff-engineer[dxf])
"""

from dxf_export.results import DrawingSheetResult, DxfExportResult
from dxf_export.writer import export_to_dxf, to_dxf_bytes, export_parse_result
from dxf_export.drawing import (
    Bar, Callout, Dimension, Drawing, DrawingError, Line, Outline, Sheet,
    Table, TitleBlock,
)
from dxf_export.sheet import STAMP, build_dxf
from dxf_export.render import render_sheet, write_sheet
from dxf_export.wall_section import (
    validate_wall_description, wall_section_drawing,
)

__all__ = [
    "DrawingSheetResult",
    "Drawing", "DrawingError", "Outline", "Line", "Bar", "Dimension",
    "Callout", "Table", "TitleBlock", "Sheet", "STAMP",
    "build_dxf", "render_sheet", "write_sheet",
    "validate_wall_description", "wall_section_drawing",
    "DxfExportResult",
    "export_to_dxf",
    "to_dxf_bytes",
    "export_parse_result",
]

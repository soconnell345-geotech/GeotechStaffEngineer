"""
DXF Export Module for Geotechnical Cross-Section Geometry

Exports geometry (surface profiles, soil boundaries, groundwater table,
soil nails, text annotations) to DXF files using ezdxf.

Compatible with DxfParseResult from dxf_import and PdfParseResult from
pdf_import for round-trip workflows.

Requires: ezdxf >= 1.4 (optional dependency: pip install geotech-staff-engineer[dxf])
"""

from dxf_export.results import DxfExportResult
from dxf_export.writer import export_to_dxf, to_dxf_bytes, export_parse_result

__all__ = [
    "DxfExportResult",
    "export_to_dxf",
    "to_dxf_bytes",
    "export_parse_result",
]

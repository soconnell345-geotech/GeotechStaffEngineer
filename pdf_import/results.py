"""
Result containers for PDF import operations.

PdfParseResult mirrors the geometry fields of DxfParseResult to enable
duck-typing with build_slope_geometry() and build_fem_inputs() via
to_dxf_parse_result() adapter.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PdfParseResult:
    """Results from PDF geometry extraction.

    Attributes
    ----------
    surface_points : list of (float, float)
        Ground surface profile as (x, z) in meters.
    boundary_profiles : dict
        {soil_name: [(x, z), ...]} for each soil boundary.
    gwt_points : list of (float, float) or None
        Groundwater table profile, or None.
    text_annotations : list of dict
        Each dict: {text, x, y}.
    page_number : int
        PDF page from which geometry was extracted.
    extraction_method : str
        'vector' or 'vision'.
    scale_factor : float
        Scale applied to convert drawing coords to meters.
    confidence : float
        1.0 for vector, <1.0 for vision-based extraction.
    warnings : list of str
        Any warnings encountered during extraction.
    """
    surface_points: List[Tuple[float, float]] = field(default_factory=list)
    boundary_profiles: Dict[str, List[Tuple[float, float]]] = field(
        default_factory=dict
    )
    gwt_points: Optional[List[Tuple[float, float]]] = None
    text_annotations: List[Dict[str, Any]] = field(default_factory=list)
    page_number: int = 0
    extraction_method: str = "vector"
    scale_factor: float = 1.0
    confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  PDF PARSE RESULTS",
            "=" * 60,
            "",
            f"  Extraction method: {self.extraction_method}",
            f"  Page: {self.page_number}",
            f"  Scale factor: {self.scale_factor}",
            f"  Confidence: {self.confidence:.2f}",
            f"  Surface points: {len(self.surface_points)}",
            f"  Soil boundaries: {len(self.boundary_profiles)}",
        ]
        for name, pts in self.boundary_profiles.items():
            lines.append(f"    '{name}': {len(pts)} points")
        if self.gwt_points is not None:
            lines.append(f"  GWT points: {len(self.gwt_points)}")
        if self.text_annotations:
            lines.append(f"  Text annotations: {len(self.text_annotations)}")
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"  WARNING: {w}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "surface_points": [
                {"x": round(x, 4), "z": round(z, 4)}
                for x, z in self.surface_points
            ],
            "boundary_profiles": {
                name: [{"x": round(x, 4), "z": round(z, 4)} for x, z in pts]
                for name, pts in self.boundary_profiles.items()
            },
            "page_number": self.page_number,
            "extraction_method": self.extraction_method,
            "scale_factor": self.scale_factor,
            "confidence": round(self.confidence, 3),
            "warnings": self.warnings,
        }
        if self.gwt_points is not None:
            d["gwt_points"] = [
                {"x": round(x, 4), "z": round(z, 4)}
                for x, z in self.gwt_points
            ]
        if self.text_annotations:
            d["text_annotations"] = self.text_annotations
        return d

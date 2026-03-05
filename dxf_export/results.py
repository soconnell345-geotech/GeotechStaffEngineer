"""
Result container for DXF export operations.

DxfExportResult stores export outputs and provides:
- summary() -> formatted string for human reading
- to_dict() -> flat dict for LLM agent consumption
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DxfExportResult:
    """Results from DXF export.

    Attributes
    ----------
    filepath : str
        Path to the saved DXF file (or '<bytes>' if returned as bytes).
    n_layers : int
        Number of DXF layers created.
    n_entities : int
        Total entity count written.
    layers_created : list of str
        Layer names created in the DXF file.
    surface_points_written : int
        Number of surface profile points written.
    boundary_profiles_written : int
        Number of boundary profiles written.
    gwt_points_written : int
        Number of GWT points written.
    nail_lines_written : int
        Number of nail line entities written.
    text_annotations_written : int
        Number of text annotation entities written.
    warnings : list of str
        Any warnings encountered during export.
    """
    filepath: str = ""
    n_layers: int = 0
    n_entities: int = 0
    layers_created: List[str] = field(default_factory=list)
    surface_points_written: int = 0
    boundary_profiles_written: int = 0
    gwt_points_written: int = 0
    nail_lines_written: int = 0
    text_annotations_written: int = 0
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  DXF EXPORT RESULTS",
            "=" * 60,
            "",
            f"  File: {self.filepath}",
            f"  Total entities: {self.n_entities}",
            f"  Layers created: {self.n_layers}",
            "",
            f"  Surface points: {self.surface_points_written}",
            f"  Boundary profiles: {self.boundary_profiles_written}",
            f"  GWT points: {self.gwt_points_written}",
            f"  Nail lines: {self.nail_lines_written}",
            f"  Text annotations: {self.text_annotations_written}",
        ]
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"  WARNING: {w}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filepath": self.filepath,
            "n_layers": self.n_layers,
            "n_entities": self.n_entities,
            "layers_created": self.layers_created,
            "surface_points_written": self.surface_points_written,
            "boundary_profiles_written": self.boundary_profiles_written,
            "gwt_points_written": self.gwt_points_written,
            "nail_lines_written": self.nail_lines_written,
            "text_annotations_written": self.text_annotations_written,
            "warnings": self.warnings,
        }

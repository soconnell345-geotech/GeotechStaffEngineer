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


@dataclass
class DrawingSheetResult:
    """What ``write_sheet`` wrote: one drawing as a DXF, a PDF page and a
    PNG (BACKLOG GM-313).

    Attributes
    ----------
    filepath, pdf_path, png_path : str
        The three files (absolute paths). The PDF and the PNG are drawn
        from the DXF document.
    sheet : str
        Sheet size name ("letter", "tabloid", "a4", "a3"), landscape.
    sheet_width_mm, sheet_height_mm : float
        The sheet, mm.
    scale : int
        N of the sheet's scale 1:N.
    scale_chosen_by : str
        "code" (the largest standard scale that fits) or "given".
    counts : dict
        outlines, hatches, lines, bars (sets), bar_dots, dimensions,
        callouts, notes, table_rows.
    layers_created : list of str
        The DXF layers that hold something.
    n_entities : int
        Entities in model space.
    warnings : list of str
        What the reader should know (a bar drawn wider than to scale, a
        title-block field cut to fit).
    """
    filepath: str = ""
    pdf_path: str = ""
    png_path: str = ""
    sheet: str = "letter"
    sheet_width_mm: float = 0.0
    sheet_height_mm: float = 0.0
    scale: int = 1
    scale_chosen_by: str = "code"
    counts: Dict[str, int] = field(default_factory=dict)
    layers_created: List[str] = field(default_factory=list)
    n_entities: int = 0
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  DRAWING SHEET",
            "=" * 60,
            "",
            f"  DXF: {self.filepath}",
            f"  PDF: {self.pdf_path}",
            f"  PNG: {self.png_path}",
            f"  Sheet: {self.sheet} landscape "
            f"({self.sheet_width_mm:g} x {self.sheet_height_mm:g} mm), "
            f"scale 1:{self.scale} ({self.scale_chosen_by})",
            f"  Entities: {self.n_entities} on {len(self.layers_created)} "
            "layers",
        ]
        for key, value in self.counts.items():
            lines.append(f"  {key}: {value}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filepath": self.filepath,
            "pdf_path": self.pdf_path,
            "png_path": self.png_path,
            "sheet": self.sheet,
            "sheet_width_mm": self.sheet_width_mm,
            "sheet_height_mm": self.sheet_height_mm,
            "scale": f"1:{self.scale}",
            "scale_n": self.scale,
            "scale_chosen_by": self.scale_chosen_by,
            "counts": dict(self.counts),
            "layers_created": list(self.layers_created),
            "n_entities": self.n_entities,
            "warnings": list(self.warnings),
        }

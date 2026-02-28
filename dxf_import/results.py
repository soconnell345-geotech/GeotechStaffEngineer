"""
Result containers for DXF import operations.

Each dataclass stores import outputs and provides:
- summary() -> formatted string for human reading
- to_dict() -> flat dict for LLM agent consumption
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LayerInfo:
    """Information about a single DXF layer.

    Attributes
    ----------
    name : str
        Layer name as defined in the DXF file.
    n_entities : int
        Total entity count on this layer.
    entity_types : dict
        Entity type counts, e.g. {"LWPOLYLINE": 3, "LINE": 5}.
    sample_texts : list of str
        Sample TEXT/MTEXT content found on this layer (up to 5).
    bbox : tuple or None
        Bounding box (x_min, y_min, x_max, y_max) of entities, or None.
    """
    name: str = ""
    n_entities: int = 0
    entity_types: Dict[str, int] = field(default_factory=dict)
    sample_texts: List[str] = field(default_factory=list)
    bbox: Optional[Tuple[float, float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "n_entities": self.n_entities,
            "entity_types": self.entity_types,
            "sample_texts": self.sample_texts,
        }
        if self.bbox is not None:
            d["bbox"] = {
                "x_min": round(self.bbox[0], 3),
                "y_min": round(self.bbox[1], 3),
                "x_max": round(self.bbox[2], 3),
                "y_max": round(self.bbox[3], 3),
            }
        return d


@dataclass
class DxfDiscoveryResult:
    """Results from discover_layers().

    Attributes
    ----------
    filepath : str
        Path to the DXF file (or '<bytes>' if loaded from content).
    n_layers : int
        Number of layers with entities.
    layers : list of LayerInfo
        Per-layer information.
    units_hint : str or None
        Units detected from $INSUNITS header.
    n_total_entities : int
        Total entity count across all layers.
    warnings : list of str
        Any warnings encountered during discovery.
    """
    filepath: str = ""
    n_layers: int = 0
    layers: List[LayerInfo] = field(default_factory=list)
    units_hint: Optional[str] = None
    n_total_entities: int = 0
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  DXF LAYER DISCOVERY",
            "=" * 60,
            "",
            f"  File: {self.filepath}",
            f"  Total entities: {self.n_total_entities}",
            f"  Layers with entities: {self.n_layers}",
        ]
        if self.units_hint:
            lines.append(f"  Units hint ($INSUNITS): {self.units_hint}")
        lines.append("")
        for lyr in self.layers:
            types_str = ", ".join(
                f"{k}: {v}" for k, v in sorted(lyr.entity_types.items())
            )
            lines.append(f"  Layer '{lyr.name}': {lyr.n_entities} entities ({types_str})")
            if lyr.sample_texts:
                texts = "; ".join(lyr.sample_texts[:3])
                lines.append(f"    Texts: {texts}")
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
            "n_total_entities": self.n_total_entities,
            "units_hint": self.units_hint,
            "layers": [lyr.to_dict() for lyr in self.layers],
            "warnings": self.warnings,
        }


@dataclass
class DxfParseResult:
    """Results from parse_dxf_geometry().

    Attributes
    ----------
    surface_points : list of (float, float)
        Ground surface profile as (x, z) in target units.
    boundary_profiles : dict
        {soil_name: [(x, z), ...]} for each soil boundary.
    gwt_points : list of (float, float) or None
        Groundwater table profile, or None.
    nail_lines : list of dict
        Each dict: {x_head, z_head, x_tip, z_tip}.
    text_annotations : list of dict
        Each dict: {text, x, y, layer}.
    units_used : str
        Units that coordinates were converted from.
    warnings : list of str
        Any warnings encountered during parsing.
    """
    surface_points: List[Tuple[float, float]] = field(default_factory=list)
    boundary_profiles: Dict[str, List[Tuple[float, float]]] = field(
        default_factory=dict
    )
    gwt_points: Optional[List[Tuple[float, float]]] = None
    nail_lines: List[Dict[str, float]] = field(default_factory=list)
    text_annotations: List[Dict[str, Any]] = field(default_factory=list)
    units_used: str = "m"
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  DXF PARSE RESULTS",
            "=" * 60,
            "",
            f"  Surface points: {len(self.surface_points)}",
            f"  Soil boundaries: {len(self.boundary_profiles)}",
        ]
        for name, pts in self.boundary_profiles.items():
            lines.append(f"    '{name}': {len(pts)} points")
        if self.gwt_points is not None:
            lines.append(f"  GWT points: {len(self.gwt_points)}")
        if self.nail_lines:
            lines.append(f"  Nail lines: {len(self.nail_lines)}")
        if self.text_annotations:
            lines.append(f"  Text annotations: {len(self.text_annotations)}")
        lines.append(f"  Units converted from: {self.units_used}")
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
            "units_used": self.units_used,
            "warnings": self.warnings,
        }
        if self.gwt_points is not None:
            d["gwt_points"] = [
                {"x": round(x, 4), "z": round(z, 4)}
                for x, z in self.gwt_points
            ]
        if self.nail_lines:
            d["nail_lines"] = [
                {k: round(v, 4) for k, v in nl.items()}
                for nl in self.nail_lines
            ]
        if self.text_annotations:
            d["text_annotations"] = self.text_annotations
        return d

"""
Result container for subsurface profile schematics.

Carries the RESOLVED geometry (every layer's top/bottom elevation after
thickness stacking and validation) alongside the rendered PNG, so a caller can
check what was actually drawn without re-reading the image.

All units are SI: meters (m) for elevations/depths, kPa for surcharge.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProfileFigureResult:
    """Container for a rendered subsurface profile figure.

    Attributes
    ----------
    layers : list of dict
        Resolved layers, top-down. Each dict carries ``name``, ``top``,
        ``bottom``, ``thickness`` (m), ``settling``, ``description``.
    ground_elevation : float
        Elevation of the original ground surface (m). Top of the first layer.
    base_elevation : float
        Bottom elevation of the deepest layer (m).
    axis : str
        ``'elevation'`` or ``'depth'`` — how the vertical axis was labelled.
    image_base64 : str
        The rendered PNG, base64-encoded, WITHOUT a data-URI prefix.
        ``data_uri()`` adds the prefix for direct HTML embedding.
    png_bytes : bytes
        Raw PNG bytes (not included in ``to_dict()``).
    width_px, height_px : int
        Pixel dimensions of the rendered PNG.
    output_path : str or None
        Absolute path of the saved PNG, when ``output_path`` was given.
    water_elevation : float or None
        Water-table elevation (m), or None when no water table was drawn.
    fill : dict or None
        Resolved fill/embankment block ({name, top, bottom, thickness}).
    surcharge : dict or None
        Resolved surcharge ({pressure, label}), kPa.
    foundation : dict or None
        Resolved foundation overlay ({type, head, tip, size, length, label}).
    annotations : list of dict
        Resolved callouts ({elevation, text, side}).
    title : str
        Figure title.
    warnings : list of str
        Non-fatal QC notes (e.g. a pile tip below the deepest layer, a water
        table outside the section).
    """

    layers: List[dict]
    ground_elevation: float
    base_elevation: float
    axis: str
    image_base64: str
    png_bytes: bytes
    width_px: int
    height_px: int
    output_path: Optional[str] = None
    water_elevation: Optional[float] = None
    fill: Optional[dict] = None
    surcharge: Optional[dict] = None
    foundation: Optional[dict] = None
    annotations: List[dict] = field(default_factory=list)
    title: str = ""
    warnings: List[str] = field(default_factory=list)

    # -- convenience ------------------------------------------------------

    def data_uri(self) -> str:
        """The PNG as an ``<img src=...>`` data URI (base64 PNG)."""
        return "data:image/png;base64," + self.image_base64

    def img_tag(self, max_width_px: int = 640, alt: str = "") -> str:
        """A ready-to-paste ``<img>`` tag embedding the PNG as a data URI."""
        return (f'<img src="{self.data_uri()}" alt="{alt or self.title}" '
                f'style="width:100%;max-width:{max_width_px}px;">')

    def _depth(self, elevation: float) -> float:
        return self.ground_elevation - elevation

    def _elev_str(self, elevation: float) -> str:
        """Format an elevation the way the figure's axis labels it."""
        if self.axis == "depth":
            return f"{self._depth(elevation):.2f} m depth"
        return f"El. {elevation:.2f} m"

    def summary(self) -> str:
        """Human-readable summary of what the figure shows."""
        lines = [self.title or "Subsurface profile",
                 f"Ground surface: {self._elev_str(self.ground_elevation)}"]
        if self.fill:
            lines.append(f"Fill: {self.fill['name']}, "
                         f"{self.fill['thickness']:.2f} m thick "
                         f"(to {self._elev_str(self.fill['top'])})")
        if self.surcharge:
            lines.append(f"Surcharge: {self.surcharge['pressure']:.1f} kPa")
        for i, lay in enumerate(self.layers, 1):
            tag = " [settling]" if lay.get("settling") else ""
            desc = f" — {lay['description']}" if lay.get("description") else ""
            lines.append(
                f"  {i}. {lay['name']}: {self._elev_str(lay['top'])} to "
                f"{self._elev_str(lay['bottom'])} "
                f"({lay['thickness']:.2f} m){tag}{desc}")
        if self.water_elevation is not None:
            lines.append(f"Water table: {self._elev_str(self.water_elevation)}")
        else:
            lines.append("Water table: not shown")
        if self.foundation:
            f = self.foundation
            lines.append(
                f"{f['label']}: head {self._elev_str(f['head'])}, tip "
                f"{self._elev_str(f['tip'])}, length {f['length']:.2f} m")
        for note in self.annotations:
            lines.append(f"Callout at {self._elev_str(note['elevation'])}: "
                         f"{note['text']}")
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serializable dict. Includes ``image_base64``, not ``png_bytes``."""
        return {
            "title": self.title,
            "axis": self.axis,
            "ground_elevation": self.ground_elevation,
            "base_elevation": self.base_elevation,
            "layers": [dict(lay) for lay in self.layers],
            "water_elevation": self.water_elevation,
            "fill": dict(self.fill) if self.fill else None,
            "surcharge": dict(self.surcharge) if self.surcharge else None,
            "foundation": dict(self.foundation) if self.foundation else None,
            "annotations": [dict(a) for a in self.annotations],
            "output_path": self.output_path,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "image_base64": self.image_base64,
            "warnings": list(self.warnings),
        }

"""
Intermediate-representation (IR) schema for a digitized drawing page.

A :class:`DrawingIR` is a deterministic, JSON-round-trippable description of a
single drawing page: page metadata (size, units, scale + provenance) plus a flat
list of geometric/text entities. It is the hand-off surface between the
DETERMINISTIC extractor (which owns exact coordinates) and an LLM (which owns
semantics and requests *slices* of the drawing through ``drawing_ir.queries``).

Entity model
------------
All entities share a common envelope (:class:`Entity`):

- ``id``          — stable within a page ("e0", "e1", ...).
- ``layer``       — CAD layer / logical group (DXF), else None.
- ``color``       — hex "#rrggbb" or an "ACI<n>" token when only a DXF color
  index is known.
- ``style``       — linetype / line-weight / a note like "approx_from_spline".
- ``source``      — "dxf" | "pdf_vector" | "raster_trace" (provenance).
- ``confidence``  — 1.0 for deterministic sources (DXF, PDF vector); < 1.0 for
  raster tracing, tiered by how the entity was detected.
- ``bbox``        — (x_min, y_min, x_max, y_max); auto-computed if not supplied.

Concrete types: :class:`Line`, :class:`Polyline`, :class:`Arc`,
:class:`Circle`, :class:`TextItem`, :class:`Region` (hatch/filled area).

Coordinate space
----------------
Coordinates live in the IR-level space flagged by ``DrawingIR.coordinate_space``:

- ``"model"`` — calibrated engineering units (``units``, e.g. "m"); a scale has
  been applied (DXF native units, or a PDF/raster page-to-model scale).
- ``"page"``  — raw page/pixel units (``units`` = "pt" for PDF points, "px" for
  raster pixels, or the DXF drawing units); no calibrated scale is set.

``scale`` (model units per page unit that WAS applied) and ``scale_provenance``
record how a model-space IR was calibrated. All entities on a page share this
one space, so the flag lives on the page, not repeated per entity.

Units are SI when in model space (house convention: meters).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def _r(v: float, n: int = 4) -> float:
    """Round a float for compact JSON; pass through non-finite as-is."""
    if v is None:
        return v
    try:
        if math.isnan(v) or math.isinf(v):
            return v
    except TypeError:
        return v
    return round(float(v), n)


def _bbox_of_points(points: Sequence[Point]) -> Optional[BBox]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _arc_points(cx: float, cy: float, radius: float,
                start_deg: float, end_deg: float, n: int = 33) -> List[Point]:
    """Sample points along an arc (CCW from start to end, degrees)."""
    sweep = end_deg - start_deg
    # Normalize to a positive CCW sweep in (0, 360].
    while sweep <= 0:
        sweep += 360.0
    while sweep > 360.0:
        sweep -= 360.0
    pts = []
    for i in range(n + 1):
        a = math.radians(start_deg + sweep * i / n)
        pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return pts


# ---------------------------------------------------------------------------
# Entity base + concrete types
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """Common envelope for every drawing entity.

    Subclasses add geometry fields and set the ``KIND`` discriminator. ``bbox``
    is computed from the geometry in ``__post_init__`` when left as None.
    """
    KIND: ClassVar[str] = "entity"

    id: str = ""
    layer: Optional[str] = None
    color: Optional[str] = None
    style: Optional[str] = None
    source: str = "dxf"
    confidence: float = 1.0
    bbox: Optional[BBox] = None

    def __post_init__(self):
        if self.bbox is None:
            self.bbox = self.compute_bbox()

    # -- geometry hooks (overridden) --
    def compute_bbox(self) -> Optional[BBox]:  # pragma: no cover - overridden
        return None

    def points(self) -> List[Point]:
        """Representative vertices for spatial queries (overridden)."""
        return []

    def length(self) -> float:
        """Path length in coordinate units (0 for point-like entities)."""
        return 0.0

    # -- serialization --
    def _geom_dict(self) -> Dict[str, Any]:  # pragma: no cover - overridden
        return {}

    def _common_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "type": self.KIND,
            "source": self.source,
            "confidence": _r(self.confidence, 3),
        }
        if self.layer is not None:
            d["layer"] = self.layer
        if self.color is not None:
            d["color"] = self.color
        if self.style is not None:
            d["style"] = self.style
        if self.bbox is not None:
            d["bbox"] = [_r(v) for v in self.bbox]
        return d

    def to_dict(self) -> Dict[str, Any]:
        d = self._common_dict()
        d.update(self._geom_dict())
        return d


@dataclass
class Line(Entity):
    """A straight segment from ``start`` to ``end``."""
    KIND: ClassVar[str] = "line"
    start: Point = (0.0, 0.0)
    end: Point = (0.0, 0.0)

    def compute_bbox(self) -> Optional[BBox]:
        return _bbox_of_points([self.start, self.end])

    def points(self) -> List[Point]:
        return [tuple(self.start), tuple(self.end)]

    def length(self) -> float:
        return math.hypot(self.end[0] - self.start[0],
                          self.end[1] - self.start[1])

    def angle_deg(self) -> float:
        """Segment orientation in degrees, folded to [0, 180)."""
        a = math.degrees(math.atan2(self.end[1] - self.start[1],
                                    self.end[0] - self.start[0]))
        a %= 180.0
        return a

    def _geom_dict(self) -> Dict[str, Any]:
        return {
            "start": [_r(self.start[0]), _r(self.start[1])],
            "end": [_r(self.end[0]), _r(self.end[1])],
            "length": _r(self.length()),
            "angle_deg": _r(self.angle_deg(), 2),
        }


@dataclass
class Polyline(Entity):
    """An ordered vertex chain; ``closed`` marks a closed ring."""
    KIND: ClassVar[str] = "polyline"
    vertices: List[Point] = field(default_factory=list)
    closed: bool = False

    def compute_bbox(self) -> Optional[BBox]:
        return _bbox_of_points(self.vertices)

    def points(self) -> List[Point]:
        return [tuple(p) for p in self.vertices]

    def length(self) -> float:
        pts = self.vertices
        if len(pts) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(pts, pts[1:]):
            total += math.hypot(b[0] - a[0], b[1] - a[1])
        if self.closed and len(pts) > 2:
            total += math.hypot(pts[0][0] - pts[-1][0],
                                pts[0][1] - pts[-1][1])
        return total

    def _geom_dict(self) -> Dict[str, Any]:
        return {
            "vertices": [[_r(x), _r(y)] for x, y in self.vertices],
            "closed": bool(self.closed),
            "n_vertices": len(self.vertices),
            "length": _r(self.length()),
        }


@dataclass
class Arc(Entity):
    """A circular arc, CCW from ``start_angle`` to ``end_angle`` (degrees)."""
    KIND: ClassVar[str] = "arc"
    center: Point = (0.0, 0.0)
    radius: float = 0.0
    start_angle: float = 0.0
    end_angle: float = 0.0

    def _sample(self, n: int = 33) -> List[Point]:
        return _arc_points(self.center[0], self.center[1], self.radius,
                           self.start_angle, self.end_angle, n)

    def compute_bbox(self) -> Optional[BBox]:
        return _bbox_of_points(self._sample())

    def points(self) -> List[Point]:
        return self._sample(n=16)

    def length(self) -> float:
        sweep = self.end_angle - self.start_angle
        while sweep <= 0:
            sweep += 360.0
        while sweep > 360.0:
            sweep -= 360.0
        return self.radius * math.radians(sweep)

    def _geom_dict(self) -> Dict[str, Any]:
        return {
            "center": [_r(self.center[0]), _r(self.center[1])],
            "radius": _r(self.radius),
            "start_angle": _r(self.start_angle, 2),
            "end_angle": _r(self.end_angle, 2),
            "length": _r(self.length()),
        }


@dataclass
class Circle(Entity):
    """A full circle."""
    KIND: ClassVar[str] = "circle"
    center: Point = (0.0, 0.0)
    radius: float = 0.0

    def compute_bbox(self) -> Optional[BBox]:
        cx, cy, r = self.center[0], self.center[1], self.radius
        return (cx - r, cy - r, cx + r, cy + r)

    def points(self) -> List[Point]:
        return _arc_points(self.center[0], self.center[1], self.radius,
                           0.0, 360.0, 16)

    def length(self) -> float:
        return 2.0 * math.pi * self.radius

    def _geom_dict(self) -> Dict[str, Any]:
        return {
            "center": [_r(self.center[0]), _r(self.center[1])],
            "radius": _r(self.radius),
        }


@dataclass
class TextItem(Entity):
    """A text label placed at ``position`` (insertion point).

    ``rotation`` is in degrees CCW; ``height`` is the text cap/character height
    in coordinate units. The bbox is an APPROXIMATION (width estimated from the
    character count) — text metrics are font-dependent and not recovered here.
    """
    KIND: ClassVar[str] = "text"
    content: str = ""
    position: Point = (0.0, 0.0)
    rotation: float = 0.0
    height: float = 0.0

    def _corners(self) -> List[Point]:
        x0, y0 = self.position
        h = self.height or 0.0
        w = 0.6 * h * max(len(self.content), 1)
        a = math.radians(self.rotation)
        ca, sa = math.cos(a), math.sin(a)
        local = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        return [(x0 + lx * ca - ly * sa, y0 + lx * sa + ly * ca)
                for lx, ly in local]

    def compute_bbox(self) -> Optional[BBox]:
        return _bbox_of_points(self._corners())

    def points(self) -> List[Point]:
        return [tuple(self.position)]

    def _geom_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "position": [_r(self.position[0]), _r(self.position[1])],
            "rotation": _r(self.rotation, 2),
            "height": _r(self.height),
        }


@dataclass
class Region(Entity):
    """A filled area / hatch, described by its boundary ring."""
    KIND: ClassVar[str] = "region"
    boundary: List[Point] = field(default_factory=list)
    pattern: Optional[str] = None

    def compute_bbox(self) -> Optional[BBox]:
        return _bbox_of_points(self.boundary)

    def points(self) -> List[Point]:
        return [tuple(p) for p in self.boundary]

    def area(self) -> float:
        """Shoelace area of the boundary ring (absolute value)."""
        pts = self.boundary
        if len(pts) < 3:
            return 0.0
        s = 0.0
        for a, b in zip(pts, pts[1:] + pts[:1]):
            s += a[0] * b[1] - b[0] * a[1]
        return abs(s) * 0.5

    def _geom_dict(self) -> Dict[str, Any]:
        d = {
            "boundary": [[_r(x), _r(y)] for x, y in self.boundary],
            "n_vertices": len(self.boundary),
            "area": _r(self.area()),
        }
        if self.pattern is not None:
            d["pattern"] = self.pattern
        return d


_ENTITY_CLASSES: Dict[str, type] = {
    cls.KIND: cls
    for cls in (Line, Polyline, Arc, Circle, TextItem, Region)
}


def entity_from_dict(d: Dict[str, Any]) -> Entity:
    """Reconstruct the correct Entity subclass from its ``to_dict`` payload."""
    kind = d.get("type", "")
    cls = _ENTITY_CLASSES.get(kind)
    if cls is None:
        raise ValueError(f"Unknown entity type '{kind}'. "
                         f"Known: {sorted(_ENTITY_CLASSES)}")
    common = dict(
        id=d.get("id", ""),
        layer=d.get("layer"),
        color=d.get("color"),
        style=d.get("style"),
        source=d.get("source", "dxf"),
        confidence=d.get("confidence", 1.0),
        bbox=tuple(d["bbox"]) if d.get("bbox") is not None else None,
    )
    if kind == "line":
        return Line(start=tuple(d["start"]), end=tuple(d["end"]), **common)
    if kind == "polyline":
        return Polyline(vertices=[tuple(p) for p in d.get("vertices", [])],
                        closed=bool(d.get("closed", False)), **common)
    if kind == "arc":
        return Arc(center=tuple(d["center"]), radius=d["radius"],
                   start_angle=d["start_angle"], end_angle=d["end_angle"],
                   **common)
    if kind == "circle":
        return Circle(center=tuple(d["center"]), radius=d["radius"], **common)
    if kind == "text":
        return TextItem(content=d.get("content", ""),
                        position=tuple(d["position"]),
                        rotation=d.get("rotation", 0.0),
                        height=d.get("height", 0.0), **common)
    if kind == "region":
        return Region(boundary=[tuple(p) for p in d.get("boundary", [])],
                      pattern=d.get("pattern"), **common)
    raise ValueError(f"Unhandled entity type '{kind}'")  # pragma: no cover


# ---------------------------------------------------------------------------
# Page container
# ---------------------------------------------------------------------------

@dataclass
class DrawingIR:
    """A single digitized drawing page: metadata + a flat list of entities."""
    width: float = 0.0
    height: float = 0.0
    units: str = "px"
    coordinate_space: str = "page"      # "page" | "model"
    scale: Optional[float] = None       # model units per page unit (if applied)
    scale_provenance: Optional[str] = None
    origin: str = "bottom_left"         # y-orientation convention
    source: str = "dxf"                 # dominant provenance
    entities: List[Entity] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- assembly --
    def add(self, entity: Entity) -> Entity:
        """Append an entity, assigning an id if it lacks one."""
        if not entity.id:
            entity.id = f"e{len(self.entities)}"
        self.entities.append(entity)
        return entity

    def by_id(self, entity_id: str) -> Optional[Entity]:
        for e in self.entities:
            if e.id == entity_id:
                return e
        return None

    def bbox(self) -> Optional[BBox]:
        """Union bbox over all entities (in coordinate units)."""
        boxes = [e.bbox for e in self.entities if e.bbox is not None]
        if not boxes:
            return None
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    def counts_by_type(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.entities:
            out[e.KIND] = out.get(e.KIND, 0) + 1
        return out

    def counts_by_layer(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.entities:
            key = e.layer if e.layer is not None else "(none)"
            out[key] = out.get(key, 0) + 1
        return out

    # -- serialization --
    def to_dict(self, include_entities: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "page": {
                "width": _r(self.width),
                "height": _r(self.height),
                "units": self.units,
                "coordinate_space": self.coordinate_space,
                "origin": self.origin,
            },
            "scale": _r(self.scale, 9) if self.scale is not None else None,
            "scale_provenance": self.scale_provenance,
            "source": self.source,
            "n_entities": len(self.entities),
            "counts_by_type": self.counts_by_type(),
            "warnings": list(self.warnings),
        }
        bb = self.bbox()
        if bb is not None:
            d["bbox"] = [_r(v) for v in bb]
        if self.metadata:
            d["metadata"] = self.metadata
        if include_entities:
            d["entities"] = [e.to_dict() for e in self.entities]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DrawingIR":
        page = d.get("page", {})
        ir = cls(
            width=page.get("width", 0.0),
            height=page.get("height", 0.0),
            units=page.get("units", "px"),
            coordinate_space=page.get("coordinate_space", "page"),
            scale=d.get("scale"),
            scale_provenance=d.get("scale_provenance"),
            origin=page.get("origin", "bottom_left"),
            source=d.get("source", "dxf"),
            warnings=list(d.get("warnings", [])),
            metadata=dict(d.get("metadata", {})),
        )
        for ed in d.get("entities", []):
            ir.entities.append(entity_from_dict(ed))
        return ir

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  DRAWING IR",
            "=" * 60,
            "",
            f"  Source: {self.source}",
            f"  Page: {_r(self.width)} x {_r(self.height)} {self.units} "
            f"({self.coordinate_space} space, origin {self.origin})",
        ]
        if self.scale is not None:
            lines.append(f"  Scale: {self.scale:g} {self.units}/page-unit "
                         f"({self.scale_provenance or 'unspecified'})")
        lines.append(f"  Entities: {len(self.entities)}")
        for k, v in sorted(self.counts_by_type().items()):
            lines.append(f"    {k}: {v}")
        layers = self.counts_by_layer()
        if len(layers) > 1 or (layers and "(none)" not in layers):
            lines.append("  Layers/groups:")
            for k, v in sorted(layers.items()):
                lines.append(f"    '{k}': {v}")
        bb = self.bbox()
        if bb is not None:
            lines.append(f"  Extent: x[{_r(bb[0])}, {_r(bb[2])}]  "
                         f"y[{_r(bb[1])}, {_r(bb[3])}]")
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"  WARNING: {w}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

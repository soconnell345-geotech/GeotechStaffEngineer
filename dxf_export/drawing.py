"""
The typed description of a drawing sheet (BACKLOG GM-313).

A :class:`Drawing` is DATA: outlines, lines, bars, dimensions, callouts,
notes, one small table, the title block's fields and the sheet. Fixed code
(``dxf_export.sheet``) turns it into one DXF document, and
``dxf_export.render`` draws that same document as a PDF page and a PNG.
Nothing here runs code from a description, and a description cannot name a
layer, a colour, a line weight, a text height or a file path: those are the
code's (``dxf_export.sheet``), as are the border, the title block's layout,
the scale bar and the stamp "CALCULATION SKETCH - NOT FOR CONSTRUCTION".

Units: every coordinate and bar diameter is in MILLIMETRES at true size
(model space). Distances that are about the sheet and not the object (a
dimension line's offset, a callout's wrap width) are millimetres ON THE
SHEET, and say so.

Every type has ``from_dict`` (strict: an unknown key is refused by name)
and ``to_dict`` (JSON types only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: The JSON form's schema name.
SCHEMA = "drawing_sheet/1"

#: Sheet sizes, landscape, width x height in mm.
SHEETS: Dict[str, Tuple[float, float]] = {
    "letter": (279.4, 215.9),
    "tabloid": (431.8, 279.4),
    "a4": (297.0, 210.0),
    "a3": (420.0, 297.0),
}

#: The standard scales 1:N the code chooses from (the largest drawing that
#: fits), and the ones a description may ask for.
SCALES: Tuple[int, ...] = (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000)

#: Hatch patterns an outline may carry.
HATCHES: Tuple[str, ...] = ("none", "concrete", "earth", "gravel")

#: What an outline may be (the code maps each to its own layer).
OUTLINE_LAYERS: Tuple[str, ...] = ("concrete", "soil", "structure", "general")

#: What a line may be.
LINE_LAYERS: Tuple[str, ...] = ("general", "ground", "hidden", "centre")

BAR_KINDS: Tuple[str, ...] = ("path", "dots")
DIMENSION_DIRECTIONS: Tuple[str, ...] = ("horizontal", "vertical", "aligned")

#: A dimension's optional second unit, printed in brackets after the
#: measurement; the code works it out from the same measurement.
SECOND_UNITS: Tuple[str, ...] = ("in", "ft", "ft-in", "m")

#: Limits that keep a description a drawing and a DXF a file a CAD program
#: opens quickly.
MAX_ITEMS = 2000            # outlines + lines + bars + dimensions + callouts
MAX_POINTS = 5000           # points in one outline, line or bar
#: Points of every outline, line and bar together (each bar seen end-on
#: is one point and one filled circle): 2,000 items of 5,000 points each
#: would be ten million (GM-313 review B2).
MAX_TOTAL_POINTS = 20000
MAX_COORD = 1.0e7           # mm: 10 km either side of the origin
MAX_TEXT = 300              # characters in one callout or note
MAX_NOTES = 40
MAX_TABLE_ROWS = 40
MAX_TABLE_COLS = 8
MAX_CELL = 60
MAX_FIELD = 80              # a title-block field
MAX_SUFFIX = 8


class DrawingError(ValueError):
    """A description the layer refuses, with the reason in plain words."""


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------

def _keys(data: Any, what: str, allowed: Tuple[str, ...],
          required: Tuple[str, ...] = ()) -> dict:
    """``data`` as a dict holding only ``allowed`` keys and every
    ``required`` one; else a :class:`DrawingError` naming the key."""
    if not isinstance(data, dict):
        raise DrawingError(f"{what}: expected an object with the keys "
                           f"{list(allowed)}, got {type(data).__name__}.")
    unknown = [k for k in data if k not in allowed]
    if unknown:
        raise DrawingError(
            f"{what}: unknown key(s) {sorted(map(str, unknown))}. "
            f"The keys are {list(allowed)}.")
    missing = [k for k in required if k not in data or data[k] is None]
    if missing:
        raise DrawingError(f"{what}: missing required key(s) {missing}.")
    return data


def _number(value: Any, what: str, *, low: Optional[float] = None,
            high: Optional[float] = None, strict_low: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DrawingError(f"{what}: expected a number, got {value!r}.")
    v = float(value)
    if not math.isfinite(v):
        raise DrawingError(f"{what}: must be a finite number, got {value!r}.")
    if low is not None and (v <= low if strict_low else v < low):
        op = ">" if strict_low else ">="
        raise DrawingError(f"{what}: must be {op} {low:g}, got {v:g}.")
    if high is not None and v > high:
        raise DrawingError(f"{what}: must be <= {high:g}, got {v:g}.")
    return v


def _point(value: Any, what: str) -> Tuple[float, float]:
    if isinstance(value, dict):
        _keys(value, what, ("x", "y"), ("x", "y"))
        value = [value["x"], value["y"]]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise DrawingError(f"{what}: expected a point [x, y] in mm, "
                           f"got {value!r}.")
    return (_number(value[0], f"{what} x", low=-MAX_COORD, high=MAX_COORD),
            _number(value[1], f"{what} y", low=-MAX_COORD, high=MAX_COORD))


def _points(value: Any, what: str, least: int) -> List[Tuple[float, float]]:
    if not isinstance(value, (list, tuple)):
        raise DrawingError(f"{what}: expected a list of points "
                           f"[[x, y], ...] in mm, got {type(value).__name__}.")
    if len(value) < least:
        raise DrawingError(f"{what}: needs at least {least} points, "
                           f"got {len(value)}.")
    if len(value) > MAX_POINTS:
        raise DrawingError(f"{what}: at most {MAX_POINTS} points, "
                           f"got {len(value)}.")
    return [_point(p, f"{what}[{i}]") for i, p in enumerate(value)]


def _text(value: Any, what: str, limit: int, *, empty_ok: bool = True) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise DrawingError(f"{what}: expected text, got {value!r}.")
    # one line: a description lays out no text, the code does; control
    # characters are dropped (a CAD program may refuse them)
    value = " ".join("".join(c if c.isprintable() else " "
                              for c in value).split())
    if len(value) > limit:
        raise DrawingError(f"{what}: at most {limit} characters, "
                           f"got {len(value)}.")
    if not empty_ok and not value:
        raise DrawingError(f"{what}: must not be empty.")
    return value


def _choice(value: Any, what: str, allowed: Tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise DrawingError(f"{what}: must be one of {list(allowed)}, "
                           f"got {value!r}.")
    return value


def _list(value: Any, what: str, limit: int) -> list:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise DrawingError(f"{what}: expected a list, "
                           f"got {type(value).__name__}.")
    if len(value) > limit:
        raise DrawingError(f"{what}: at most {limit} entries, "
                           f"got {len(value)}.")
    return list(value)


def _pt(p: Tuple[float, float]) -> List[float]:
    return [float(p[0]), float(p[1])]


# ---------------------------------------------------------------------------
# the types
# ---------------------------------------------------------------------------

@dataclass
class Outline:
    """A closed polygon (mm), with an optional hatch.

    ``layer``: what it is (:data:`OUTLINE_LAYERS`); ``hatch``: one of
    :data:`HATCHES`."""
    points: List[Tuple[float, float]]
    layer: str = "general"
    hatch: str = "none"

    KEYS = ("points", "layer", "hatch")

    @classmethod
    def from_dict(cls, data: Any, what: str = "outline") -> "Outline":
        d = _keys(data, what, cls.KEYS, ("points",))
        return cls(points=_points(d["points"], f"{what}.points", 3),
                   layer=_choice(d.get("layer", "general"), f"{what}.layer",
                                 OUTLINE_LAYERS),
                   hatch=_choice(d.get("hatch", "none"), f"{what}.hatch",
                                 HATCHES))

    def to_dict(self) -> dict:
        return {"points": [_pt(p) for p in self.points], "layer": self.layer,
                "hatch": self.hatch}


@dataclass
class Line:
    """A line or an open polyline (mm). ``layer``: :data:`LINE_LAYERS`
    ("hidden" is dashed, "centre" is chain-dotted)."""
    points: List[Tuple[float, float]]
    layer: str = "general"

    KEYS = ("points", "layer")

    @classmethod
    def from_dict(cls, data: Any, what: str = "line") -> "Line":
        d = _keys(data, what, cls.KEYS, ("points",))
        return cls(points=_points(d["points"], f"{what}.points", 2),
                   layer=_choice(d.get("layer", "general"), f"{what}.layer",
                                 LINE_LAYERS))

    def to_dict(self) -> dict:
        return {"points": [_pt(p) for p in self.points], "layer": self.layer}


@dataclass
class Bar:
    """A reinforcing bar set as it is seen in the drawing.

    ``kind="path"``: the bar lies in the plane of the drawing; ``points``
    is its centreline, drawn as a heavy line ``diameter_mm`` wide with its
    corners rounded to ``bend_radius_mm`` (centreline radius; 0 = sharp).
    ``kind="dots"``: bars seen end-on; each point is one bar, drawn as a
    filled circle of ``diameter_mm``. A bar thinner than prints legibly at
    the sheet's scale is drawn at the least legible width; the result's
    warnings say so."""
    points: List[Tuple[float, float]]
    diameter_mm: float
    kind: str = "path"
    bend_radius_mm: float = 0.0

    KEYS = ("points", "diameter_mm", "kind", "bend_radius_mm")

    @classmethod
    def from_dict(cls, data: Any, what: str = "bar") -> "Bar":
        d = _keys(data, what, cls.KEYS, ("points", "diameter_mm"))
        kind = _choice(d.get("kind", "path"), f"{what}.kind", BAR_KINDS)
        return cls(
            points=_points(d["points"], f"{what}.points",
                           2 if kind == "path" else 1),
            diameter_mm=_number(d["diameter_mm"], f"{what}.diameter_mm",
                                low=0.0, high=500.0, strict_low=True),
            kind=kind,
            bend_radius_mm=_number(d.get("bend_radius_mm", 0.0),
                                   f"{what}.bend_radius_mm", low=0.0,
                                   high=MAX_COORD))

    def to_dict(self) -> dict:
        return {"points": [_pt(p) for p in self.points],
                "diameter_mm": self.diameter_mm, "kind": self.kind,
                "bend_radius_mm": self.bend_radius_mm}


@dataclass
class Dimension:
    """A linear dimension between two points (mm).

    The number printed is the DIMENSION entity's own measurement of the two
    points: a description cannot type a number over it. ``direction``:
    "horizontal" measures the x distance, "vertical" the y distance,
    "aligned" the distance between the points. ``offset_mm`` is where the
    dimension line sits, in mm ON THE SHEET: for "horizontal", above the
    higher point when positive and below the lower when negative; for
    "vertical", right of the right-hand point when positive and left of
    the left-hand one when negative; for "aligned", to the left of the
    direction p1 -> p2 when positive. ``suffix`` is printed after the
    number (for example " mm"); ``second_unit`` adds the same measurement
    in a second unit in brackets (:data:`SECOND_UNITS`). A number too wide
    for its dimension is printed beyond ``p2`` instead of between the
    extension lines."""
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    offset_mm: float = 8.0
    direction: str = "horizontal"
    suffix: str = ""
    second_unit: Optional[str] = None

    KEYS = ("p1", "p2", "offset_mm", "direction", "suffix", "second_unit")

    @classmethod
    def from_dict(cls, data: Any, what: str = "dimension") -> "Dimension":
        d = _keys(data, what, cls.KEYS, ("p1", "p2"))
        direction = _choice(d.get("direction", "horizontal"),
                            f"{what}.direction", DIMENSION_DIRECTIONS)
        p1 = _point(d["p1"], f"{what}.p1")
        p2 = _point(d["p2"], f"{what}.p2")
        measured = {"horizontal": abs(p2[0] - p1[0]),
                    "vertical": abs(p2[1] - p1[1]),
                    "aligned": math.hypot(p2[0] - p1[0], p2[1] - p1[1])}
        if measured[direction] < 1e-6:
            raise DrawingError(
                f"{what}: the two points are no {direction} distance apart "
                f"({p1} and {p2}); there is nothing to dimension.")
        offset = _number(d.get("offset_mm", 8.0), f"{what}.offset_mm",
                         low=-400.0, high=400.0)
        if abs(offset) < 1e-9:
            raise DrawingError(f"{what}.offset_mm: must not be 0 (the "
                               "dimension line would lie on the points).")
        second = d.get("second_unit")
        if second is not None:
            second = _choice(second, f"{what}.second_unit", SECOND_UNITS)
        suffix = d.get("suffix") or ""
        if not isinstance(suffix, str) or len(suffix) > MAX_SUFFIX:
            raise DrawingError(f"{what}.suffix: text of at most "
                               f"{MAX_SUFFIX} characters, got {suffix!r}.")
        if any(c in suffix for c in "<>\\{}\n\r"):
            raise DrawingError(f"{what}.suffix: must not hold any of "
                               "< > \\ { } or a line break.")
        return cls(p1=p1, p2=p2, offset_mm=offset, direction=direction,
                   suffix=suffix, second_unit=second)

    def to_dict(self) -> dict:
        return {"p1": _pt(self.p1), "p2": _pt(self.p2),
                "offset_mm": self.offset_mm, "direction": self.direction,
                "suffix": self.suffix, "second_unit": self.second_unit}


@dataclass
class Callout:
    """A leader from a point on the drawing to a text.

    ``at``: the point the arrow touches (mm). ``text_at``: where the leader
    lands (mm); the text is set to the right of it when it is right of
    ``at``, else to its left. ``mark``: an optional short mark (a bar
    mark) drawn in a circle before the text. ``width_mm``: the width the
    text is wrapped to, in mm ON THE SHEET."""
    at: Tuple[float, float]
    text_at: Tuple[float, float]
    text: str
    mark: str = ""
    width_mm: float = 45.0

    KEYS = ("at", "text_at", "text", "mark", "width_mm")

    @classmethod
    def from_dict(cls, data: Any, what: str = "callout") -> "Callout":
        d = _keys(data, what, cls.KEYS, ("at", "text_at", "text"))
        at = _point(d["at"], f"{what}.at")
        text_at = _point(d["text_at"], f"{what}.text_at")
        if math.hypot(at[0] - text_at[0], at[1] - text_at[1]) < 1e-6:
            raise DrawingError(f"{what}: 'at' and 'text_at' are the same "
                               "point; the leader would have no length.")
        return cls(at=at, text_at=text_at,
                   text=_text(d["text"], f"{what}.text", MAX_TEXT,
                              empty_ok=False),
                   mark=_text(d.get("mark", ""), f"{what}.mark", 3),
                   width_mm=_number(d.get("width_mm", 45.0),
                                    f"{what}.width_mm", low=15.0, high=200.0))

    def to_dict(self) -> dict:
        return {"at": _pt(self.at), "text_at": _pt(self.text_at),
                "text": self.text, "mark": self.mark,
                "width_mm": self.width_mm}


@dataclass
class Table:
    """One small table (the bar schedule): a title, headers and rows of
    text cells. It is set at the sheet's bottom left."""
    title: str = ""
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)

    KEYS = ("title", "headers", "rows")

    @classmethod
    def from_dict(cls, data: Any, what: str = "table") -> "Table":
        d = _keys(data, what, cls.KEYS, ("headers", "rows"))
        headers = [_text(h, f"{what}.headers[{i}]", MAX_CELL)
                   for i, h in enumerate(_list(d["headers"],
                                               f"{what}.headers",
                                               MAX_TABLE_COLS))]
        if not headers:
            raise DrawingError(f"{what}.headers: needs at least one column.")
        rows = []
        for i, row in enumerate(_list(d["rows"], f"{what}.rows",
                                      MAX_TABLE_ROWS)):
            if not isinstance(row, (list, tuple)) or len(row) != len(headers):
                raise DrawingError(
                    f"{what}.rows[{i}]: expected {len(headers)} cells (one "
                    f"per header), got {row!r}.")
            cells = []
            for j, cell in enumerate(row):
                if cell is None:
                    cell = "-"
                elif isinstance(cell, bool) or not isinstance(
                        cell, (str, int, float)):
                    raise DrawingError(f"{what}.rows[{i}][{j}]: expected "
                                       f"text or a number, got {cell!r}.")
                elif isinstance(cell, float):
                    cell = f"{cell:g}"
                cells.append(_text(str(cell), f"{what}.rows[{i}][{j}]",
                                   MAX_CELL))
            rows.append(cells)
        return cls(title=_text(d.get("title", ""), f"{what}.title", MAX_FIELD),
                   headers=headers, rows=rows)

    def to_dict(self) -> dict:
        return {"title": self.title, "headers": list(self.headers),
                "rows": [list(r) for r in self.rows]}


@dataclass
class TitleBlock:
    """The title block's fields. Its layout, and the scale and sheet size
    it prints, are the code's."""
    title: str = "Drawing"
    project: str = ""
    project_number: str = ""
    drawn_by: str = ""
    checked_by: str = ""
    date: str = ""
    drawing_number: str = ""

    KEYS = ("title", "project", "project_number", "drawn_by", "checked_by",
            "date", "drawing_number")

    @classmethod
    def from_dict(cls, data: Any, what: str = "title_block") -> "TitleBlock":
        d = _keys(data if data is not None else {}, what, cls.KEYS)
        values = {k: _text(d.get(k, ""), f"{what}.{k}",
                           160 if k == "title" else MAX_FIELD)
                  for k in cls.KEYS}
        if not values["title"]:
            values["title"] = "Drawing"
        return cls(**values)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.KEYS}


@dataclass
class Sheet:
    """The sheet: ``size`` (:data:`SHEETS`, always landscape) and
    ``scale`` (N of 1:N from :data:`SCALES`; ``None`` = the code chooses
    the largest standard scale at which the drawing fits)."""
    size: str = "letter"
    scale: Optional[int] = None

    KEYS = ("size", "scale")

    @classmethod
    def from_dict(cls, data: Any, what: str = "sheet") -> "Sheet":
        d = _keys(data if data is not None else {}, what, cls.KEYS)
        size = d.get("size", "letter")
        if isinstance(size, str):
            size = size.strip().lower()
        size = _choice(size, f"{what}.size", tuple(SHEETS))
        scale = d.get("scale")
        if scale is not None:
            if isinstance(scale, str):
                s = scale.replace(" ", "")
                if s.startswith("1:"):
                    s = s[2:]
                try:
                    scale = int(s)
                except ValueError:
                    raise DrawingError(
                        f"{what}.scale: give N of 1:N, one of "
                        f"{list(SCALES)}, got {scale!r}.") from None
            if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                    or int(scale) != scale or int(scale) not in SCALES):
                raise DrawingError(
                    f"{what}.scale: give N of 1:N, one of {list(SCALES)} "
                    f"(or leave it out and the code chooses), got {scale!r}.")
            scale = int(scale)
        return cls(size=size, scale=scale)

    def to_dict(self) -> dict:
        return {"size": self.size, "scale": self.scale}


@dataclass
class Drawing:
    """One drawing sheet as data. See the module's docstring."""
    outlines: List[Outline] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)
    bars: List[Bar] = field(default_factory=list)
    dimensions: List[Dimension] = field(default_factory=list)
    callouts: List[Callout] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    table: Optional[Table] = None
    title_block: TitleBlock = field(default_factory=TitleBlock)
    sheet: Sheet = field(default_factory=Sheet)

    KEYS = ("schema", "outlines", "lines", "bars", "dimensions", "callouts",
            "notes", "table", "title_block", "sheet")

    @classmethod
    def from_dict(cls, data: Any) -> "Drawing":
        d = _keys(data, "drawing", cls.KEYS)
        schema = d.get("schema", SCHEMA)
        if schema != SCHEMA:
            raise DrawingError(f"drawing.schema: this code reads "
                               f"{SCHEMA!r}, got {schema!r}.")

        def many(key, kind):
            return [kind.from_dict(item, f"{key}[{i}]") for i, item in
                    enumerate(_list(d.get(key), f"drawing.{key}", MAX_ITEMS))]

        drawing = cls(
            outlines=many("outlines", Outline),
            lines=many("lines", Line),
            bars=many("bars", Bar),
            dimensions=many("dimensions", Dimension),
            callouts=many("callouts", Callout),
            notes=[_text(n, f"notes[{i}]", MAX_TEXT, empty_ok=False)
                   for i, n in enumerate(_list(d.get("notes"),
                                               "drawing.notes", MAX_NOTES))],
            table=(Table.from_dict(d["table"])
                   if d.get("table") is not None else None),
            title_block=TitleBlock.from_dict(d.get("title_block")),
            sheet=Sheet.from_dict(d.get("sheet")),
        )
        drawing.validate()
        return drawing

    def validate(self) -> "Drawing":
        """Check the whole drawing (each part was checked as it was
        read); returns itself. Raises :class:`DrawingError`."""
        n = (len(self.outlines) + len(self.lines) + len(self.bars)
             + len(self.dimensions) + len(self.callouts))
        if n > MAX_ITEMS:
            raise DrawingError(f"drawing: at most {MAX_ITEMS} outlines, "
                               f"lines, bars, dimensions and callouts in "
                               f"all, got {n}.")
        points = sum(len(item.points) for group in
                     (self.outlines, self.lines, self.bars)
                     for item in group)
        if points > MAX_TOTAL_POINTS:
            raise DrawingError(
                f"drawing: at most {MAX_TOTAL_POINTS} points in all "
                f"(outlines, lines and bars, a bar seen end-on counting "
                f"one), got {points}.")
        if not (self.outlines or self.lines or self.bars):
            raise DrawingError("drawing: nothing to draw; give at least one "
                               "outline, line or bar.")
        return self

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "outlines": [o.to_dict() for o in self.outlines],
            "lines": [ln.to_dict() for ln in self.lines],
            "bars": [b.to_dict() for b in self.bars],
            "dimensions": [dm.to_dict() for dm in self.dimensions],
            "callouts": [c.to_dict() for c in self.callouts],
            "notes": list(self.notes),
            "table": self.table.to_dict() if self.table else None,
            "title_block": self.title_block.to_dict(),
            "sheet": self.sheet.to_dict(),
        }

    def geometry_points(self) -> List[Tuple[float, float]]:
        """Every point of the outlines, lines and bars (mm)."""
        pts: List[Tuple[float, float]] = []
        for group in (self.outlines, self.lines, self.bars):
            for item in group:
                pts.extend(item.points)
        return pts

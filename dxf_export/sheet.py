"""
A :class:`~dxf_export.drawing.Drawing` as ONE ezdxf document (GM-313).

The object is drawn in model space at true size in millimetres. The sheet
(border, title block, notes, table, scale bar, stamp) is drawn round it in
model space too, N times its printed size, so plotting the model at 1:N
gives the sheet; ``dxf_export.render`` does exactly that for the PDF page
and the PNG, from this same document.

Fixed here and closed to a description: the layers with their colours and
line weights (:data:`LAYERS`), the text heights (2.5 mm on the sheet, 3.5
for the title and the stamp), the border, the title block, the scale bar
and the stamp (:data:`STAMP`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dxf_export._fonts import (
    BOLD, REGULAR, WIDTH_FACTOR, ensure_fonts, fit, text_width, wrap,
)
from dxf_export.drawing import (
    SCALES, SHEETS, Bar, Callout, Dimension, Drawing, DrawingError,
)

#: Printed on every sheet; no input removes it.
STAMP = "CALCULATION SKETCH - NOT FOR CONSTRUCTION"

#: key -> (DXF layer name, AutoCAD colour index, line weight in 1/100 mm,
#: linetype).
LAYERS: Dict[str, Tuple[str, int, int, str]] = {
    "border": ("GSE-BORDER", 7, 50, "CONTINUOUS"),
    "title": ("GSE-TITLE", 7, 25, "CONTINUOUS"),
    "stamp": ("GSE-STAMP", 1, 35, "CONTINUOUS"),
    "concrete": ("GSE-CONCRETE", 7, 50, "CONTINUOUS"),
    "soil": ("GSE-SOIL", 30, 35, "CONTINUOUS"),
    "structure": ("GSE-STRUCTURE", 7, 50, "CONTINUOUS"),
    "general": ("GSE-GENERAL", 7, 25, "CONTINUOUS"),
    "ground": ("GSE-GROUND", 30, 35, "CONTINUOUS"),
    "hidden": ("GSE-HIDDEN", 8, 18, "DASHED"),
    "centre": ("GSE-CENTRE", 8, 18, "CENTER"),
    "hatch": ("GSE-HATCH", 8, 9, "CONTINUOUS"),
    "bars": ("GSE-BARS", 1, 0, "CONTINUOUS"),
    "dims": ("GSE-DIMS", 3, 18, "CONTINUOUS"),
    "callouts": ("GSE-CALLOUTS", 4, 18, "CONTINUOUS"),
    "notes": ("GSE-NOTES", 7, 18, "CONTINUOUS"),
    "table": ("GSE-TABLE", 7, 18, "CONTINUOUS"),
}

#: hatch name -> (DXF pattern, pattern scale per unit of N).
PATTERNS: Dict[str, Tuple[str, float]] = {
    "concrete": ("AR-CONC", 0.03),
    "earth": ("EARTH", 0.35),
    "gravel": ("GRAVEL", 0.35),
}

TEXT_STYLE = "GSE"
BOLD_STYLE = "GSE-BOLD"
DIM_STYLE = "GSE-MM"

# Sizes on the sheet, mm.
TEXT_H = 2.5                # lettering (cap height)
TITLE_H = 3.5               # the title and the stamp
PITCH = 4.2                 # line pitch of 2.5 mm lettering
MARGIN = 10.0               # sheet edge to border
PAD = 3.0
STRIP_W = 80.0              # the right-hand strip: notes over title block
ARROW = 2.0
DIM_GAP = 0.8               # dimension line to its number
MIN_BAR_W = 0.5             # least printed width of a bar drawn as a line
MIN_DOT_D = 0.9             # least printed diameter of a bar seen end-on
MARK_R = 2.9                # a callout mark's circle
LANDING = 2.0               # a leader's flat end
ROW_H = 5.0                 # table row
CELL_PAD = 1.5


@dataclass
class SheetInfo:
    """What :func:`build_dxf` drew."""
    sheet: str = "letter"
    sheet_width_mm: float = 0.0
    sheet_height_mm: float = 0.0
    scale: int = 1
    scale_chosen_by: str = "code"       # or "given"
    counts: Dict[str, int] = field(default_factory=dict)
    layers: List[str] = field(default_factory=list)
    n_entities: int = 0
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# geometry shared by the fit test and the drawing
# ---------------------------------------------------------------------------

def second_unit_text(mm: float, unit: str) -> str:
    """``mm`` in a dimension's second unit, as printed in its brackets."""
    if unit == "m":
        return f"{mm / 1000.0:.3f} m"
    inches = mm / 25.4
    if unit == "in":
        return f"{inches:.1f} in."
    if unit == "ft":
        return f"{inches / 12.0:.2f} ft"
    # feet and inches to the nearest quarter inch
    quarters = int(round(inches * 4))
    feet, rem = divmod(quarters, 48)
    whole, frac = divmod(rem, 4)
    part = {0: "", 1: " 1/4", 2: " 1/2", 3: " 3/4"}[frac]
    if feet:
        return f"{feet}'-{whole}{part}\""
    return f"{whole}{part}\"".strip()


@dataclass
class _DimGeom:
    measured: float
    angle: float                    # of the dimension line, degrees
    a: Tuple[float, float]          # the dimension line's ends
    b: Tuple[float, float]
    base: Tuple[float, float]
    p1: Tuple[float, float]         # the points as the entity takes them
    p2: Tuple[float, float]
    distance: float                 # an aligned dimension's offset, model
    text: str                       # the DIMENSION's text ("<>" = measured)
    printed: str                    # as it prints, for measuring
    fits: bool
    location: Optional[Tuple[float, float]]     # the number, if moved
    box: Tuple[float, float, float, float]      # x0, y0, x1, y1 envelope


def _dim_geometry(dim: Dimension, n: int) -> _DimGeom:
    (x1, y1), (x2, y2) = dim.p1, dim.p2
    off = dim.offset_mm * n
    if dim.direction == "horizontal":
        y = (max(y1, y2) if off > 0 else min(y1, y2)) + off
        a, b, angle = (x1, y), (x2, y), 0.0
        measured = abs(x2 - x1)
    elif dim.direction == "vertical":
        x = (max(x1, x2) if off > 0 else min(x1, x2)) + off
        a, b, angle = (x, y1), (x, y2), 90.0
        measured = abs(y2 - y1)
    else:
        if (x2, y2) < (x1, y1):
            # measured left to right, so the number reads upright; the
            # dimension line stays on the side the description asked for
            (x1, y1), (x2, y2), off = (x2, y2), (x1, y1), -off
        length = math.hypot(x2 - x1, y2 - y1)
        nx, ny = -(y2 - y1) / length, (x2 - x1) / length
        a = (x1 + nx * off, y1 + ny * off)
        b = (x2 + nx * off, y2 + ny * off)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        measured = length
    text = "<>" + dim.suffix
    printed = f"{measured:.0f}{dim.suffix}"
    if dim.second_unit:
        extra = f" ({second_unit_text(measured, dim.second_unit)})"
        text += extra
        printed += extra
    width = text_width(printed, TEXT_H) * n
    # room for the number and, under it, the two arrowheads
    fits = max(width + 2.0 * n, 3.5 * ARROW * n) <= measured
    # the side the number sits on: above a horizontal line, left of a
    # vertical one (read from the right), as ezdxf sets it
    ux, uy = (b[0] - a[0]), (b[1] - a[1])
    ul = math.hypot(ux, uy) or 1.0
    ux, uy = ux / ul, uy / ul
    px, py = -uy, ux
    if (dim.direction == "horizontal" and py < 0) or (
            dim.direction == "vertical" and px > 0) or (
            dim.direction == "aligned" and (py < 0 or (py == 0 and px > 0))):
        px, py = -px, -py
    lift = (DIM_GAP + TEXT_H / 2.0) * n
    location = None
    if fits:
        cx, cy = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
    else:
        reach = (ARROW + 1.5) * n + width / 2.0
        cx, cy = b[0] + ux * reach, b[1] + uy * reach
        location = (cx + px * lift, cy + py * lift)
    tx, ty = cx + px * lift, cy + py * lift
    half_w, half_h = width / 2.0, TEXT_H * n / 2.0
    ex = abs(ux) * half_w + abs(px) * half_h
    ey = abs(uy) * half_w + abs(py) * half_h
    xs = [x1, x2, a[0], b[0], tx - ex, tx + ex]
    ys = [y1, y2, a[1], b[1], ty - ey, ty + ey]
    return _DimGeom(measured=measured, angle=angle, a=a, b=b, base=a,
                    p1=(x1, y1), p2=(x2, y2), distance=off,
                    text=text, printed=printed, fits=fits, location=location,
                    box=(min(xs), min(ys), max(xs), max(ys)))


@dataclass
class _CalloutGeom:
    side: int                               # +1 text to the right, -1 left
    vertices: List[Tuple[float, float]]     # the leader
    mark_centre: Optional[Tuple[float, float]]
    text_x: float
    lines: List[str]
    line_ys: List[float]
    box: Tuple[float, float, float, float]


def callout_lines(callout: Callout) -> List[str]:
    """A callout's text as the sheet breaks it."""
    return wrap(callout.text, callout.width_mm, TEXT_H)


def callout_size(callout: Callout) -> Tuple[float, float]:
    """A callout's width and height on the sheet (mm), from the leader's
    landing to the far end of its text."""
    lines = callout_lines(callout)
    width = LANDING + max(text_width(s, TEXT_H) for s in lines) + 1.0
    if callout.mark:
        width += 2.0 * MARK_R + 0.6
    return width, max(len(lines) * PITCH, 2.0 * MARK_R)


def _callout_geometry(c: Callout, n: int) -> _CalloutGeom:
    side = 1 if c.text_at[0] >= c.at[0] else -1
    end = (c.text_at[0] + side * LANDING * n, c.text_at[1])
    x = end[0]
    mark_centre = None
    if c.mark:
        mark_centre = (x + side * MARK_R * n, end[1])
        x += side * (2.0 * MARK_R + 0.6) * n
    x += side * 1.0 * n
    lines = callout_lines(c)
    count = len(lines)
    ys = [end[1] + ((count - 1) / 2.0 - i) * PITCH * n for i in range(count)]
    width = max(text_width(s, TEXT_H) for s in lines) * n
    half = max(count * PITCH, 2.0 * MARK_R) * n / 2.0
    xs = [c.at[0], c.text_at[0], x, x + side * width]
    return _CalloutGeom(side=side, vertices=[c.at, c.text_at, end],
                        mark_centre=mark_centre, text_x=x, lines=lines,
                        line_ys=ys,
                        box=(min(xs), min(c.at[1], end[1] - half),
                             max(xs), max(c.at[1], end[1] + half)))


def bar_width(bar: Bar, n: int) -> float:
    """The width (or dot diameter) a bar is drawn at, model mm."""
    least = (MIN_BAR_W if bar.kind == "path" else MIN_DOT_D) * n
    return max(bar.diameter_mm, least)


def content_box(drawing: Drawing, n: int) -> Tuple[float, float, float, float]:
    """The box round everything of ``drawing`` at scale 1:``n`` (model
    mm): the object, its dimensions and its callouts."""
    xs: List[float] = []
    ys: List[float] = []
    for group in (drawing.outlines, drawing.lines):
        for item in group:
            xs.extend(p[0] for p in item.points)
            ys.extend(p[1] for p in item.points)
    for bar in drawing.bars:
        half = bar_width(bar, n) / 2.0
        xs.extend(v for p in bar.points for v in (p[0] - half, p[0] + half))
        ys.extend(v for p in bar.points for v in (p[1] - half, p[1] + half))
    boxes = [_dim_geometry(d, n).box for d in drawing.dimensions]
    boxes += [_callout_geometry(c, n).box for c in drawing.callouts]
    for x0, y0, x1, y1 in boxes:
        xs.extend((x0, x1))
        ys.extend((y0, y1))
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# the sheet's furniture, laid out on paper (mm, origin bottom left)
# ---------------------------------------------------------------------------

@dataclass
class _Plan:
    width: float
    height: float
    table_cols: List[float]             # column widths
    table_h: float
    note_lines: List[List[str]]         # per note, its wrapped lines
    notes_h: float
    title_lines: List[str]
    title_block_h: float
    area: Tuple[float, float, float, float]     # x0, y0, x1, y1


_TB_ROWS = 4                # the title block's rows under the title
_TB_ROW_H = 5.5
_NOTE_INDENT = 5.0


def _plan(drawing: Drawing) -> _Plan:
    width, height = SHEETS[drawing.sheet.size]
    inner_w = STRIP_W - 2.0 * PAD

    title_lines = wrap(drawing.title_block.title, inner_w, TITLE_H, True)[:3]
    title_block_h = (len(title_lines) * (TITLE_H * 1.6) + 3.0
                     + _TB_ROWS * _TB_ROW_H)

    note_lines = [wrap(note, inner_w - _NOTE_INDENT, TEXT_H)
                  for note in drawing.notes]
    notes_h = 0.0
    if note_lines:
        notes_h = (PITCH + 2.0 + sum(len(ls) * PITCH + 1.5
                                     for ls in note_lines))
    strip_room = height - 2.0 * MARGIN - title_block_h - 2.0 * PAD
    if notes_h > strip_room:
        raise DrawingError(
            f"The notes need {notes_h:.0f} mm of the sheet's notes column "
            f"and it has {strip_room:.0f} mm on a {drawing.sheet.size} "
            "sheet. Give fewer or shorter notes, or a larger sheet "
            "(tabloid, a3).")

    table_cols: List[float] = []
    table_h = 0.0
    table = drawing.table
    if table is not None:
        for j, header in enumerate(table.headers):
            cells = [header] + [row[j] for row in table.rows]
            widest = max(text_width(c, TEXT_H, bold=(i == 0))
                         for i, c in enumerate(cells))
            table_cols.append(widest + 2.0 * CELL_PAD)
        table_h = (len(table.rows) + 1) * ROW_H + (PITCH + 1.5 if table.title
                                                    else 0.0)
        room = width - 2.0 * MARGIN - STRIP_W - 2.0 * PAD - 55.0
        if sum(table_cols) > room:
            raise DrawingError(
                f"The table is {sum(table_cols):.0f} mm wide and a "
                f"{drawing.sheet.size} sheet has {room:.0f} mm for it. "
                "Shorten its cells or headers, give fewer columns, or a "
                "larger sheet (tabloid, a3).")

    band_h = max(table_h, 13.0)
    area = (MARGIN + PAD, MARGIN + PAD + band_h + PAD,
            width - MARGIN - STRIP_W - PAD,
            height - MARGIN - PAD - (TITLE_H + 3.0) - PAD)
    return _Plan(width=width, height=height, table_cols=table_cols,
                 table_h=table_h, note_lines=note_lines, notes_h=notes_h,
                 title_lines=title_lines, title_block_h=title_block_h,
                 area=area)


def choose_scale(drawing: Drawing) -> Tuple[int, str]:
    """N of 1:N for ``drawing``: the given one if it fits, else the
    smallest standard N at which the object with its dimensions and
    callouts fits the sheet's drawing area. Raises :class:`DrawingError`
    when the given scale, or none, fits."""
    plan = _plan(drawing)
    area_w = plan.area[2] - plan.area[0]
    area_h = plan.area[3] - plan.area[1]

    def fits(n: int) -> bool:
        x0, y0, x1, y1 = content_box(drawing, n)
        return (x1 - x0) <= area_w * n + 1e-6 and (y1 - y0) <= area_h * n + 1e-6

    first = next((n for n in SCALES if fits(n)), None)
    given = drawing.sheet.scale
    if given is not None:
        if fits(given):
            return given, "given"
        x0, y0, x1, y1 = content_box(drawing, given)
        hint = (f"The largest standard scale that fits is 1:{first}."
                if first else "No standard scale fits; use a larger sheet.")
        raise DrawingError(
            f"At 1:{given} the drawing with its dimensions and callouts is "
            f"{(x1 - x0) / given:.0f} x {(y1 - y0) / given:.0f} mm on the "
            f"sheet, and a {drawing.sheet.size} sheet's drawing area is "
            f"{area_w:.0f} x {area_h:.0f} mm. {hint}")
    if first is None:
        raise DrawingError(
            f"The drawing does not fit a {drawing.sheet.size} sheet at any "
            f"standard scale up to 1:{SCALES[-1]}. Use a larger sheet or "
            "check the coordinates (millimetres).")
    return first, "code"


# ---------------------------------------------------------------------------
# the DXF
# ---------------------------------------------------------------------------

def _filleted(points, radius):
    """``points`` as LWPOLYLINE (x, y, bulge) vertices, the corners
    rounded to ``radius``."""
    pts = [points[0]]
    for p in points[1:]:
        if math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 1e-9:
            pts.append(p)
    if len(pts) < 3 or radius <= 0:
        return [(p[0], p[1], 0.0) for p in pts]
    out = [(pts[0][0], pts[0][1], 0.0)]
    for i in range(1, len(pts) - 1):
        p0, p, p2 = pts[i - 1], pts[i], pts[i + 1]
        l1 = math.hypot(p[0] - p0[0], p[1] - p0[1])
        l2 = math.hypot(p2[0] - p[0], p2[1] - p[1])
        u1 = ((p[0] - p0[0]) / l1, (p[1] - p0[1]) / l1)
        u2 = ((p2[0] - p[0]) / l2, (p2[1] - p[1]) / l2)
        cross = u1[0] * u2[1] - u1[1] * u2[0]
        dot = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
        turn = math.acos(dot)
        if turn < 1e-6 or abs(math.pi - turn) < 1e-6:
            out.append((p[0], p[1], 0.0))
            continue
        t = min(radius * math.tan(turn / 2.0), 0.45 * l1, 0.45 * l2)
        bulge = math.tan(turn / 4.0) * (1.0 if cross > 0 else -1.0)
        out.append((p[0] - u1[0] * t, p[1] - u1[1] * t, bulge))
        out.append((p[0] + u2[0] * t, p[1] + u2[1] * t, 0.0))
    out.append((pts[-1][0], pts[-1][1], 0.0))
    return out


class _Builder:
    """Draws one :class:`Drawing` into a new ezdxf document."""

    def __init__(self, drawing: Drawing):
        import ezdxf

        ensure_fonts()
        self.drawing = drawing
        self.plan = _plan(drawing)
        self.n, chosen_by = choose_scale(drawing)
        self.info = SheetInfo(sheet=drawing.sheet.size,
                              sheet_width_mm=self.plan.width,
                              sheet_height_mm=self.plan.height,
                              scale=self.n, scale_chosen_by=chosen_by)
        n = self.n
        self.doc = doc = ezdxf.new("R2010", setup=["linetypes"])
        doc.units = 4                           # millimetres
        doc.header["$MEASUREMENT"] = 1          # metric
        doc.header["$LTSCALE"] = float(n)
        doc.header["$DIMSCALE"] = float(n)
        self.msp = doc.modelspace()

        doc.styles.add(TEXT_STYLE, font=REGULAR).dxf.width = WIDTH_FACTOR
        doc.styles.add(BOLD_STYLE, font=BOLD).dxf.width = WIDTH_FACTOR
        for name, colour, weight, linetype in LAYERS.values():
            doc.layers.add(name, color=colour, lineweight=weight,
                           linetype=linetype)
        # Our own dimension style: millimetres measured as drawn (the
        # stock EZDXF style multiplies the measurement by 100), whole
        # numbers, sized on the sheet and scaled by N.
        doc.dimstyles.new(DIM_STYLE, dxfattribs={
            "dimtxsty": TEXT_STYLE, "dimtxt": TEXT_H, "dimasz": ARROW,
            "dimexo": 1.0, "dimexe": 1.5, "dimgap": DIM_GAP,
            "dimscale": float(n), "dimlfac": 1.0, "dimdec": 0, "dimrnd": 0,
            "dimtad": 1, "dimzin": 8, "dimlunit": 2, "dimtih": 0,
            "dimtoh": 0, "dimclrd": 256, "dimclre": 256, "dimclrt": 256,
            "dimblk": "", "dimtsz": 0.0,
        })

        # where the sheet sits in model space: the content centred in the
        # drawing area
        x0, y0, x1, y1 = content_box(drawing, n)
        ax0, ay0, ax1, ay1 = self.plan.area
        self.ox = (x0 + x1) / 2.0 - n * (ax0 + ax1) / 2.0
        self.oy = (y0 + y1) / 2.0 - n * (ay0 + ay1) / 2.0
        self.counts = {"outlines": 0, "hatches": 0, "lines": 0, "bars": 0,
                       "bar_dots": 0, "dimensions": 0, "callouts": 0,
                       "notes": 0, "table_rows": 0}

    # -- paper helpers ----------------------------------------------------

    def P(self, x: float, y: float) -> Tuple[float, float]:
        """A point on the sheet (mm from its bottom-left corner) in model
        space."""
        return (self.ox + self.n * x, self.oy + self.n * y)

    def _layer(self, key: str) -> str:
        return LAYERS[key][0]

    def _text(self, text: str, at, height: float, layer: str, *,
              bold: bool = False, align: str = "LEFT"):
        """One line of text at model point ``at`` (``height`` on the
        sheet)."""
        from ezdxf.enums import TextEntityAlignment as A

        if not text:
            return None
        entity = self.msp.add_text(text, height=height * self.n, dxfattribs={
            "style": BOLD_STYLE if bold else TEXT_STYLE,
            "width": WIDTH_FACTOR, "layer": self._layer(layer)})
        entity.set_placement(at, align=getattr(A, align))
        return entity

    def _paper_text(self, text, x, y, height, layer, **kw):
        return self._text(text, self.P(x, y), height, layer, **kw)

    def _paper_line(self, x0, y0, x1, y1, layer):
        self.msp.add_line(self.P(x0, y0), self.P(x1, y1),
                          dxfattribs={"layer": self._layer(layer)})

    def _paper_box(self, x0, y0, x1, y1, layer):
        self.msp.add_lwpolyline(
            [self.P(x0, y0), self.P(x1, y0), self.P(x1, y1), self.P(x0, y1)],
            close=True, dxfattribs={"layer": self._layer(layer)})

    def _solid(self, path, layer: str):
        """A solid-filled closed path of (x, y[, bulge]) vertices."""
        hatch = self.msp.add_hatch(dxfattribs={"layer": self._layer(layer),
                                               "color": 256})
        hatch.paths.add_polyline_path(path, is_closed=True)
        return hatch

    # -- the object -------------------------------------------------------

    def _outlines(self):
        n = self.n
        for outline in self.drawing.outlines:
            if outline.hatch != "none":
                pattern, per_n = PATTERNS[outline.hatch]
                hatch = self.msp.add_hatch(dxfattribs={
                    "layer": self._layer("hatch"), "color": 256})
                hatch.set_pattern_fill(pattern, scale=per_n * n)
                hatch.paths.add_polyline_path(outline.points, is_closed=True)
                self.counts["hatches"] += 1
            self.msp.add_lwpolyline(outline.points, close=True, dxfattribs={
                "layer": self._layer(outline.layer)})
            self.counts["outlines"] += 1

    def _lines(self):
        for line in self.drawing.lines:
            attribs = {"layer": self._layer(line.layer)}
            if len(line.points) == 2:
                self.msp.add_line(line.points[0], line.points[1],
                                  dxfattribs=attribs)
            else:
                self.msp.add_lwpolyline(line.points, dxfattribs=attribs)
            self.counts["lines"] += 1

    def _bars(self):
        n = self.n
        thin_line = thin_dot = False
        for bar in self.drawing.bars:
            drawn = bar_width(bar, n)
            if bar.kind == "path":
                thin_line = thin_line or drawn > bar.diameter_mm
                self.msp.add_lwpolyline(
                    _filleted(bar.points, bar.bend_radius_mm), format="xyb",
                    dxfattribs={"layer": self._layer("bars"),
                                "const_width": drawn})
                self.counts["bars"] += 1
            else:
                thin_dot = thin_dot or drawn > bar.diameter_mm
                r = drawn / 2.0
                for x, y in bar.points:
                    self._solid([(x - r, y, 1.0), (x + r, y, 1.0)], "bars")
                    self.counts["bar_dots"] += 1
                self.counts["bars"] += 1
        if thin_line:
            self.info.warnings.append(
                f"At 1:{n} a bar thinner than {MIN_BAR_W * n:g} mm would "
                f"print under {MIN_BAR_W} mm wide; such bars are drawn "
                f"{MIN_BAR_W} mm wide on the sheet, wider than to scale.")
        if thin_dot:
            self.info.warnings.append(
                f"At 1:{n} a bar seen end-on thinner than {MIN_DOT_D * n:g} "
                f"mm would print under {MIN_DOT_D} mm across; such bars are "
                f"drawn {MIN_DOT_D} mm across on the sheet, larger than to "
                "scale.")

    def _dimensions(self):
        n = self.n
        attribs = {"layer": self._layer("dims")}
        for dim in self.drawing.dimensions:
            g = _dim_geometry(dim, n)
            if dim.direction == "aligned":
                style = self.msp.add_aligned_dim(
                    p1=g.p1, p2=g.p2, distance=g.distance,
                    text=g.text, dimstyle=DIM_STYLE, dxfattribs=attribs)
            else:
                style = self.msp.add_linear_dim(
                    base=g.base, p1=dim.p1, p2=dim.p2, angle=g.angle,
                    text=g.text, dimstyle=DIM_STYLE, dxfattribs=attribs)
            if g.location is not None:
                style.set_location(g.location, leader=False, relative=False)
            style.render()
            self.counts["dimensions"] += 1

    def _callouts(self):
        n = self.n
        layer = self._layer("callouts")
        for callout in self.drawing.callouts:
            g = _callout_geometry(callout, n)
            self.msp.add_leader(g.vertices, dimstyle=DIM_STYLE,
                                dxfattribs={"layer": layer})
            if g.mark_centre is not None:
                self.msp.add_circle(g.mark_centre, MARK_R * n,
                                    dxfattribs={"layer": layer})
                self._text(callout.mark, g.mark_centre, TEXT_H, "callouts",
                           bold=True, align="MIDDLE_CENTER")
            align = "MIDDLE_LEFT" if g.side > 0 else "MIDDLE_RIGHT"
            for line, y in zip(g.lines, g.line_ys):
                self._text(line, (g.text_x, y), TEXT_H, "callouts",
                           align=align)
            self.counts["callouts"] += 1

    # -- the sheet --------------------------------------------------------

    def _border_and_stamp(self):
        p = self.plan
        self._paper_box(MARGIN, MARGIN, p.width - MARGIN, p.height - MARGIN,
                        "border")
        x = p.width - MARGIN - STRIP_W
        self._paper_line(x, MARGIN, x, p.height - MARGIN, "title")
        # the stamp, top left, boxed
        w = text_width(STAMP, TITLE_H, bold=True)
        x0 = MARGIN + PAD
        y1 = p.height - MARGIN - PAD
        y0 = y1 - TITLE_H - 3.0
        self._paper_box(x0, y0, x0 + w + 4.0, y1, "stamp")
        self._paper_text(STAMP, x0 + 2.0, (y0 + y1) / 2.0, TITLE_H, "stamp",
                         bold=True, align="MIDDLE_LEFT")

    def _title_block(self):
        p = self.plan
        tb = self.drawing.title_block
        x0 = p.width - MARGIN - STRIP_W
        x1 = p.width - MARGIN
        y0 = MARGIN
        y1 = y0 + p.title_block_h
        self._paper_line(x0, y1, x1, y1, "title")
        y = y1 - 1.5
        for line in p.title_lines:
            y -= TITLE_H * 1.6
            self._paper_text(line, x0 + PAD, y + TITLE_H * 0.3, TITLE_H,
                             "title", bold=True)
        rows_top = y0 + _TB_ROWS * _TB_ROW_H
        self._paper_line(x0, rows_top, x1, rows_top, "title")
        mid = (x0 + x1) / 2.0
        half = STRIP_W / 2.0 - 2.0 * 1.5
        full = STRIP_W - 2.0 * 1.5
        rows = [
            [("PROJECT: " + tb.project, full)],
            [("NO.: " + tb.project_number, half),
             ("DWG: " + tb.drawing_number, half)],
            [("DRAWN: " + tb.drawn_by, half),
             ("CHECKED: " + tb.checked_by, half)],
            [("DATE: " + tb.date, half),
             (f"SCALE 1:{self.n}  ({self.drawing.sheet.size})", half)],
        ]
        for i, row in enumerate(rows):
            top = rows_top - i * _TB_ROW_H
            base = top - _TB_ROW_H
            if i:
                self._paper_line(x0, top, x1, top, "title")
            if len(row) == 2:
                self._paper_line(mid, base, mid, top, "title")
            for j, (text, room) in enumerate(row):
                shown = fit(text, room, TEXT_H)
                if shown != text:
                    self.info.warnings.append(
                        f"Title block: '{text}' is too long for its box "
                        f"and is printed as '{shown}'.")
                self._paper_text(shown, (x0 if j == 0 else mid) + 1.5,
                                 base + _TB_ROW_H / 2.0, TEXT_H, "title",
                                 align="MIDDLE_LEFT")

    def _notes(self):
        p = self.plan
        if not p.note_lines:
            return
        x = p.width - MARGIN - STRIP_W + PAD
        y = p.height - MARGIN - PAD - TEXT_H
        self._paper_text("NOTES", x, y, TEXT_H, "notes", bold=True)
        y -= 2.0
        for k, lines in enumerate(p.note_lines, start=1):
            for i, line in enumerate(lines):
                y -= PITCH
                if i == 0:
                    self._paper_text(f"{k}.", x, y, TEXT_H, "notes")
                self._paper_text(line, x + _NOTE_INDENT, y, TEXT_H, "notes")
            y -= 1.5
            self.counts["notes"] += 1

    def _table(self) -> float:
        """Draw the table; return its right edge on the sheet."""
        p = self.plan
        table = self.drawing.table
        x0 = MARGIN + PAD
        if table is None:
            return x0
        y0 = MARGIN + PAD
        rows = [table.headers] + table.rows
        top = y0 + len(rows) * ROW_H
        if table.title:
            self._paper_text(table.title, x0, top + 1.5, TEXT_H, "table",
                             bold=True)
        width = sum(p.table_cols)
        for i in range(len(rows) + 1):
            y = top - i * ROW_H
            self._paper_line(x0, y, x0 + width, y, "table")
        x = x0
        for w in p.table_cols + [0.0]:
            self._paper_line(x, y0, x, top, "table")
            x += w
        for i, row in enumerate(rows):
            x = x0
            y = top - (i + 0.5) * ROW_H
            for cell, w in zip(row, p.table_cols):
                self._paper_text(cell, x + CELL_PAD, y, TEXT_H, "table",
                                 bold=(i == 0), align="MIDDLE_LEFT")
                x += w
        self.counts["table_rows"] = len(table.rows)
        return x0 + width

    def _scale_bar(self, x: float):
        """The scale bar, its left end at ``x`` on the sheet."""
        n = self.n
        # the longest 1-2-5 length that prints 45 mm or less
        length = 1.0
        for exp in range(0, 8):
            for mant in (1.0, 2.0, 5.0):
                if mant * 10 ** exp / n <= 45.0:
                    length = mant * 10 ** exp
        mant = length / 10 ** math.floor(math.log10(length))
        parts = 5 if abs(mant - 5.0) < 1e-9 else 4
        y = MARGIN + PAD + 1.0
        h = 1.4
        lp = length / n
        step = lp / parts
        self._paper_box(x, y, x + lp, y + h, "title")
        for i in range(0, parts, 2):
            a, b = self.P(x + i * step, y), self.P(x + (i + 1) * step, y + h)
            self._solid([(a[0], a[1]), (b[0], a[1]), (b[0], b[1]),
                         (a[0], b[1])], "title")
        metres = length >= 1000.0
        unit = "m" if metres else "mm"

        def label(v):
            return f"{v / 1000.0:g}" if metres else f"{v:g}"

        ty = y + h + 1.2
        self._paper_text("0", x, ty, TEXT_H, "title", align="CENTER")
        if parts == 4:
            self._paper_text(label(length / 2.0), x + lp / 2.0, ty, TEXT_H,
                             "title", align="CENTER")
        self._paper_text(f"{label(length)} {unit}", x + lp, ty, TEXT_H,
                         "title", align="CENTER")
        self._paper_text(f"SCALE 1:{n}", x, ty + TEXT_H + 2.0, TEXT_H,
                         "title", bold=True)

    def build(self):
        self._outlines()
        self._lines()
        self._bars()
        self._dimensions()
        self._callouts()
        self._border_and_stamp()
        self._title_block()
        self._notes()
        right = self._table()
        self._scale_bar(right + (10.0 if self.drawing.table else 2.0))
        self.info.counts = dict(self.counts)
        self.info.layers = sorted(
            {e.dxf.layer for e in self.msp if e.dxf.hasattr("layer")})
        self.info.n_entities = len(self.msp)
        return self.doc, self.info


def build_dxf(drawing: Drawing):
    """``drawing`` as a new ezdxf document.

    Returns ``(doc, info)``: the document (model space, millimetres, the
    sheet drawn round the object at N times its printed size) and a
    :class:`SheetInfo`. Raises :class:`~dxf_export.drawing.DrawingError`
    when the drawing does not fit its sheet."""
    drawing.validate()
    return _Builder(drawing).build()

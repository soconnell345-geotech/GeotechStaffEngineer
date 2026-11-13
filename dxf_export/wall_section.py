"""
The first drawing: a reinforced cantilever retaining wall section
(BACKLOG GM-313).

:func:`wall_section_drawing` takes a ``reinforced_wall_section/1``
description (a plain dict: the wall's resolved geometry, its materials and
its bar sets, as the wall module's structural design returns it) and lays
out a :class:`~dxf_export.drawing.Drawing` by fixed rules. It designs
nothing and computes no steel: every size, spacing, cover and hook length
is the description's.

Frame: millimetres, x from the toe's outer face toward the heel (the
retained soil is on the right), y up from the underside of the base.

Drawing rules (version 1; the sheet's notes say so):

* a stem bar that lies in the section runs the full height of the stem
  and into the base, down to the base's bottom steel; a base bar that
  lies in the section runs the full width of the base; no bar is cut off
  or lapped;
* bars that run along the wall are drawn end-on, as dots at their
  spacing, inside the in-section bars of the same face;
* a stem bar with a standard hook is turned at the bottom of the base
  TOWARD THE OPPOSITE FACE of the stem, across the joint: a back-face
  (retained-side) bar turns toward the toe. That is the "Correct
  Reinforcement Orientation" of USACE EM 1110-2-2104 (1 November 2023)
  Figure 2-1, p. 9; turned toward the heel is the figure's "Incorrect".
  The hook's bend radius and the length of its tail are drawn
  schematically (the tail to the line of the stem's opposite face): no
  held reference prints a standard hook's bend diameter or extension, and
  the sheet says the hook is not detailed;
* the retained ground is drawn from the top of the stem's back face at
  the backfill slope; the ground in front at the embedment depth above
  the underside of the base.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from dxf_export.drawing import (
    SCALES, Bar, Callout, Dimension, Drawing, DrawingError, Line, Outline,
    Sheet, Table, TitleBlock, _choice, _keys, _list, _number, _text,
)
from dxf_export.sheet import _plan, callout_size, choose_scale

SCHEMA = "reinforced_wall_section/1"

DESCRIPTION_KEYS = ("schema", "title", "geometry", "materials", "bars",
                    "notes", "basis")
GEOMETRY_KEYS = ("wall_height", "base_width", "toe_length", "heel_length",
                 "stem_thickness_top", "stem_thickness_base",
                 "base_thickness", "stem_batter", "embedment_depth",
                 "backfill_slope_deg", "has_shear_key", "key_depth",
                 "key_width", "key_distance_from_toe")
MATERIAL_KEYS = ("fc_MPa", "fy_MPa")
BAR_KEYS = ("mark", "element", "face", "runs", "role", "size",
            "diameter_mm", "spacing_mm", "cover_mm", "end", "callout",
            "As_required_mm2_per_m", "As_provided_mm2_per_m")
END_KEYS = ("type", "into", "ldh_mm")
ELEMENTS = ("stem", "toe", "heel", "base")
STEM_FACES = ("back", "front", "both")
BASE_FACES = ("top", "bottom", "both")
RUNS = ("in_section", "along_wall")
ROLES = ("flexure", "temperature_shrinkage")

#: Callout text width on the sheet, mm.
CALLOUT_W = 38.0
#: The schematic hook's bend, as a radius on the sheet, mm.
HOOK_R = 2.0

#: Notes the drawing adds after the description's own.
DRAWING_NOTES = (
    "All dimensions are in millimetres.",
    "Bars are drawn full length: no bar is cut off or lapped in this "
    "sketch. Bars along the wall are drawn end-on.",
    "Hooked stem bars are turned toward the opposite face of the stem "
    "(EM 1110-2-2104 Fig. 2-1, correct orientation). The hook's bend and "
    "tail are schematic, not detailed.",
)


# ---------------------------------------------------------------------------
# the description, checked
# ---------------------------------------------------------------------------

def validate_wall_description(description: Any) -> dict:
    """``description`` checked against ``reinforced_wall_section/1`` and
    returned as a new dict (lengths still metres, as given). Unknown keys
    are refused by name. Raises :class:`DrawingError`."""
    d = _keys(description, "description", DESCRIPTION_KEYS,
              ("geometry", "bars"))
    schema = d.get("schema", SCHEMA)
    if schema != SCHEMA:
        raise DrawingError(f"description.schema: this code draws {SCHEMA!r}, "
                           f"got {schema!r}.")
    g = _keys(d["geometry"], "geometry", GEOMETRY_KEYS,
              ("wall_height", "base_width", "toe_length",
               "stem_thickness_top", "stem_thickness_base",
               "base_thickness"))
    geo: Dict[str, Any] = {}
    for key in ("wall_height", "base_width", "stem_thickness_top",
                "stem_thickness_base", "base_thickness"):
        geo[key] = _number(g[key], f"geometry.{key}", low=0.0, high=100.0,
                           strict_low=True)
    geo["toe_length"] = _number(g["toe_length"], "geometry.toe_length",
                                low=0.0, high=100.0)
    heel = (geo["base_width"] - geo["toe_length"]
            - geo["stem_thickness_base"])
    if g.get("heel_length") is not None:
        given = _number(g["heel_length"], "geometry.heel_length", low=0.0,
                        high=100.0)
        if abs(given - heel) > 0.002:
            raise DrawingError(
                f"geometry: heel_length {given:g} m does not equal "
                f"base_width - toe_length - stem_thickness_base = "
                f"{heel:.3f} m.")
    if heel < -1e-9:
        raise DrawingError("geometry: toe_length + stem_thickness_base "
                           "exceed base_width.")
    geo["heel_length"] = max(heel, 0.0)
    if geo["stem_thickness_top"] > geo["stem_thickness_base"] + 1e-9:
        raise DrawingError("geometry: stem_thickness_top exceeds "
                           "stem_thickness_base.")
    if geo["base_thickness"] >= geo["wall_height"]:
        raise DrawingError("geometry: base_thickness must be less than "
                           "wall_height (underside of base to top of stem).")
    geo["stem_batter"] = _choice(g.get("stem_batter") or "back",
                                 "geometry.stem_batter", ("back", "front"))
    geo["embedment_depth"] = _number(
        g.get("embedment_depth") if g.get("embedment_depth") is not None
        else geo["base_thickness"], "geometry.embedment_depth", low=0.0,
        high=geo["wall_height"])
    geo["backfill_slope_deg"] = _number(g.get("backfill_slope_deg") or 0.0,
                                        "geometry.backfill_slope_deg",
                                        low=0.0, high=45.0)
    has_key = g.get("has_shear_key") or False
    if not isinstance(has_key, bool):
        raise DrawingError("geometry.has_shear_key: true or false, "
                           f"got {has_key!r}.")
    geo["has_shear_key"] = has_key
    if has_key:
        for key in ("key_depth", "key_width", "key_distance_from_toe"):
            if g.get(key) is None:
                raise DrawingError(f"geometry.{key}: needed with a shear "
                                   "key.")
        geo["key_depth"] = _number(g["key_depth"], "geometry.key_depth",
                                   low=0.0, high=20.0, strict_low=True)
        geo["key_width"] = _number(g["key_width"], "geometry.key_width",
                                   low=0.0, high=100.0, strict_low=True)
        geo["key_distance_from_toe"] = _number(
            g["key_distance_from_toe"], "geometry.key_distance_from_toe",
            low=0.0, high=100.0)
        if (geo["key_distance_from_toe"] + geo["key_width"]
                > geo["base_width"] + 1e-9):
            raise DrawingError("geometry: the shear key runs past the heel "
                               "(key_distance_from_toe + key_width exceed "
                               "base_width).")
    else:
        geo["key_depth"] = 0.0
        geo["key_width"] = None
        geo["key_distance_from_toe"] = None

    materials = {}
    if d.get("materials") is not None:
        m = _keys(d["materials"], "materials", MATERIAL_KEYS)
        for key in MATERIAL_KEYS:
            if m.get(key) is not None:
                materials[key] = _number(m[key], f"materials.{key}", low=0.0,
                                         high=2000.0, strict_low=True)

    raw_bars = _list(d["bars"], "bars", 26)
    if not raw_bars:
        raise DrawingError("bars: the description holds no bar set; there "
                           "is no reinforcement to draw.")
    bars = []
    marks = set()
    for i, raw in enumerate(raw_bars):
        what = f"bars[{i}]"
        b = _keys(raw, what, BAR_KEYS,
                  ("mark", "element", "face", "runs", "size", "diameter_mm",
                   "spacing_mm", "cover_mm"))
        mark = _text(b["mark"], f"{what}.mark", 3, empty_ok=False)
        if mark in marks:
            raise DrawingError(f"{what}.mark: '{mark}' is used twice; each "
                               "bar set needs its own mark.")
        marks.add(mark)
        element = _choice(b["element"], f"{what}.element", ELEMENTS)
        face = _choice(b["face"], f"{what}.face",
                       STEM_FACES if element == "stem" else BASE_FACES)
        runs = _choice(b["runs"], f"{what}.runs", RUNS)
        if runs == "in_section" and face == "both":
            raise DrawingError(
                f"{what}: an in_section bar set with face 'both' is not "
                "drawn; give one set per face.")
        end = None
        if b.get("end") is not None:
            e = _keys(b["end"], f"{what}.end", END_KEYS, ("type",))
            end = {
                "type": _choice(e["type"], f"{what}.end.type",
                                ("standard_hook_90",)),
                "into": _choice(e.get("into") or "base", f"{what}.end.into",
                                ("base",)),
                "ldh_mm": (_number(e["ldh_mm"], f"{what}.end.ldh_mm",
                                   low=0.0, high=5000.0)
                           if e.get("ldh_mm") is not None else None),
            }
            if element != "stem" or runs != "in_section":
                raise DrawingError(
                    f"{what}.end: a hook is drawn only on a stem bar that "
                    "lies in the section (it turns into the base).")
        size = _text(b["size"], f"{what}.size", 12, empty_ok=False)
        area = {}
        for key in ("As_required_mm2_per_m", "As_provided_mm2_per_m"):
            area[key] = (_number(b[key], f"{what}.{key}", low=0.0, high=1e6)
                         if b.get(key) is not None else None)
        bars.append({
            "mark": mark, "element": element, "face": face, "runs": runs,
            "role": _choice(b.get("role") or "flexure", f"{what}.role",
                            ROLES),
            "size": size,
            "diameter_mm": _number(b["diameter_mm"], f"{what}.diameter_mm",
                                   low=0.0, high=100.0, strict_low=True),
            "spacing_mm": _number(b["spacing_mm"], f"{what}.spacing_mm",
                                  low=10.0, high=5000.0),
            "cover_mm": _number(b["cover_mm"], f"{what}.cover_mm", low=0.0,
                                high=1000.0),
            "end": end,
            "callout": _text(b.get("callout") or
                             f"{size} @ {b['spacing_mm']:g} mm",
                             f"{what}.callout", 200),
            **area,
        })
    return {
        "schema": SCHEMA,
        "title": _text(d.get("title") or
                       "Cantilever retaining wall: reinforcement",
                       "description.title", 160),
        "geometry": geo,
        "materials": materials,
        "bars": bars,
        "notes": [_text(n, f"notes[{i}]", 300, empty_ok=False)
                  for i, n in enumerate(_list(d.get("notes"), "notes", 30))],
        "basis": _text(d.get("basis") or "", "description.basis", 300),
    }


# ---------------------------------------------------------------------------
# the wall's geometry in mm
# ---------------------------------------------------------------------------

class _Wall:
    """The section in mm, with its faces as functions of height."""

    def __init__(self, geo: dict):
        mm = 1000.0
        self.H = geo["wall_height"] * mm
        self.B = geo["base_width"] * mm
        self.toe = geo["toe_length"] * mm
        self.ts_t = geo["stem_thickness_top"] * mm
        self.ts_b = geo["stem_thickness_base"] * mm
        self.tb = geo["base_thickness"] * mm
        self.emb = geo["embedment_depth"] * mm
        self.beta = math.radians(geo["backfill_slope_deg"])
        self.key = None
        if geo["has_shear_key"]:
            self.key = (geo["key_distance_from_toe"] * mm,
                        geo["key_width"] * mm, geo["key_depth"] * mm)
        # stem faces at the top of the base (b) and at the top (t)
        self.xf_b = self.toe
        self.xb_b = self.toe + self.ts_b
        if geo["stem_batter"] == "back":
            self.xf_t = self.toe
            self.xb_t = self.toe + self.ts_t
        else:
            self.xb_t = self.xb_b
            self.xf_t = self.xb_b - self.ts_t
        self.y_min = -self.key[2] if self.key else 0.0

    def face_x(self, face: str, y: float) -> float:
        """x of the stem's ``face`` line at height ``y`` (the line is
        carried on below the top of the base)."""
        b, t = ((self.xb_b, self.xb_t) if face == "back"
                else (self.xf_b, self.xf_t))
        return b + (t - b) * (y - self.tb) / (self.H - self.tb)

    def face_secant(self, face: str) -> float:
        """1 / cos of the face's batter: the horizontal distance per unit
        of distance square to the face."""
        b, t = ((self.xb_b, self.xb_t) if face == "back"
                else (self.xf_b, self.xf_t))
        return math.hypot(t - b, self.H - self.tb) / (self.H - self.tb)

    def inside(self, face: str) -> int:
        """The x direction into the concrete from a stem face."""
        return -1 if face == "back" else 1

    def outline(self) -> List[Tuple[float, float]]:
        pts = [(0.0, 0.0)]
        if self.key:
            x, w, depth = self.key
            pts += [(x, 0.0), (x, -depth), (x + w, -depth), (x + w, 0.0)]
        pts += [(self.B, 0.0), (self.B, self.tb), (self.xb_b, self.tb),
                (self.xb_t, self.H), (self.xf_t, self.H),
                (self.xf_b, self.tb), (0.0, self.tb)]
        out = [pts[0]]
        for p in pts[1:]:
            if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-6:
                out.append(p)
        if math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) < 1e-6:
            out.pop()
        return out


def _place(element: str, face: str, runs: str) -> str:
    where = {"stem": "Stem", "toe": "Toe", "heel": "Heel",
             "base": "Base"}[element]
    side = {"back": "back face", "front": "front face", "top": "top",
            "bottom": "bottom",
            "both": "each face" if element == "stem" else "top and bottom"}
    text = f"{where}, {side[face]}"
    return text + (", along" if runs == "along_wall" else "")


def _area(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.0f}"


# ---------------------------------------------------------------------------
# the layout
# ---------------------------------------------------------------------------

def _layout(desc: dict, sheet: Sheet, title_block: TitleBlock,
            n: int) -> Tuple[Drawing, List[str]]:
    """The drawing laid out for scale 1:``n``, and its warnings."""
    wall = _Wall(desc["geometry"])
    warnings: List[str] = []
    bars_out: List[Bar] = []
    H, B, tb = wall.H, wall.B, wall.tb

    # -- where each bar set sits ------------------------------------------
    # the in-section sets of each face, outermost first, in the order given
    sets = desc["bars"]
    stack: Dict[Tuple[str, str], float] = {}    # (part, face) -> depth used
    first_cover: Dict[Tuple[str, str], float] = {}
    outer_edge: Dict[Tuple[str, str], float] = {}
    centre: Dict[str, float] = {}               # mark -> offset to centre

    def part_of(element):
        return "stem" if element == "stem" else "base"

    for s in sets:
        if s["runs"] != "in_section":
            continue
        key = (part_of(s["element"]), s["face"])
        used = stack.get(key)
        if used is None:
            used = s["cover_mm"]
            first_cover[key] = s["cover_mm"]
            outer_edge[key] = s["cover_mm"]
        centre[s["mark"]] = used + s["diameter_mm"] / 2.0
        stack[key] = used + s["diameter_mm"]

    def along_offset(s, face):
        key = (part_of(s["element"]), face)
        inner = max(s["cover_mm"], stack.get(key, 0.0))
        if key not in first_cover:
            first_cover[key] = s["cover_mm"]
            outer_edge[key] = s["cover_mm"]
        return inner + s["diameter_mm"] / 2.0

    # the level of the base's bottom steel, for the stem bars to stand on
    bottom_top = stack.get(("base", "bottom"))

    # callout attach points: mark -> (x, y), and the side it leaves by
    attach: Dict[str, Tuple[float, float]] = {}
    side_of: Dict[str, int] = {}
    stem_h = H - tb

    stem_sets = [s for s in sets if s["element"] == "stem"]
    base_sets = [s for s in sets if s["element"] != "stem"]

    for s in stem_sets:
        d = s["diameter_mm"]
        faces = ["back", "front"] if s["face"] == "both" else [s["face"]]
        if s["runs"] == "in_section":
            face = faces[0]
            sec = wall.face_secant(face)
            off = centre[s["mark"]] * sec * wall.inside(face)
            y_top = H - s["cover_mm"]
            y_end = (bottom_top + d / 2.0 if bottom_top is not None
                     else s["cover_mm"] + d / 2.0)
            y_end = min(y_end, tb - d)
            pts = [(wall.face_x(face, y_top) + off, y_top),
                   (wall.face_x(face, y_end) + off, y_end)]
            radius = 0.0
            if s["end"]:
                other = "front" if face == "back" else "back"
                x_tail = wall.face_x(other, tb)
                x_tail = min(max(x_tail, s["cover_mm"]), B - s["cover_mm"])
                pts.append((x_tail, y_end))
                radius = HOOK_R * n
                ldh = s["end"]["ldh_mm"]
                room = tb - (y_end - d / 2.0)
                if ldh is not None and ldh > room + 0.5:
                    warnings.append(
                        f"Bar {s['mark']}: the hook development length "
                        f"given, {ldh:.0f} mm, is more than the {room:.0f} "
                        "mm from the top of the base to the outside of the "
                        "hook as drawn (the hook stands on the base's "
                        "bottom steel).")
            bars_out.append(Bar(points=pts, diameter_mm=d, kind="path",
                                bend_radius_mm=radius))
            y_at = tb + stem_h * 0.6
            attach[s["mark"]] = (wall.face_x(face, y_at) + off, y_at)
            side_of[s["mark"]] = 1 if face == "back" else -1
        else:
            dots: List[Tuple[float, float]] = []
            first_face_dots: List[Tuple[float, float]] = []
            for face in faces:
                sec = wall.face_secant(face)
                off = along_offset(s, face) * sec * wall.inside(face)
                y = tb + s["spacing_mm"] / 2.0
                mine = []
                while y <= H - s["cover_mm"] + 1e-6:
                    mine.append((wall.face_x(face, y) + off, y))
                    y += s["spacing_mm"]
                if not first_face_dots:
                    first_face_dots = mine
                dots.extend(mine)
            if dots:
                bars_out.append(Bar(points=dots, diameter_mm=d, kind="dots"))
                pick = first_face_dots[len(first_face_dots) * 2 // 5]
                attach[s["mark"]] = pick
                side_of[s["mark"]] = 1 if faces[0] == "back" else -1

    for s in base_sets:
        d = s["diameter_mm"]
        c = s["cover_mm"]
        faces = ["top", "bottom"] if s["face"] == "both" else [s["face"]]

        def level(face, off):
            return off if face == "bottom" else tb - off

        if s["runs"] == "in_section":
            y = level(faces[0], centre[s["mark"]])
            bars_out.append(Bar(points=[(c, y), (B - c, y)], diameter_mm=d,
                                kind="path"))
            # a bottom bar's leader starts further from the heel's end, so
            # it passes under the leaders of the sets above it
            back = 0.12 if faces[0] == "top" else 0.3
            attach[s["mark"]] = (B - c - back * max(B - wall.xb_b, 300.0), y)
        else:
            dots = []
            pick = None
            for face in faces:
                y = level(face, along_offset(s, face))
                room = B - 2.0 * c
                count = int(math.floor(room / s["spacing_mm"] + 1e-9)) + 1
                x = c + (room - (count - 1) * s["spacing_mm"]) / 2.0
                row = [(x + k * s["spacing_mm"], y) for k in range(count)]
                dots.extend(row)
                if row:         # the last row drawn: the lowest
                    pick = row[-1]
            if dots:
                bars_out.append(Bar(points=dots, diameter_mm=d, kind="dots"))
                attach[s["mark"]] = pick
        side_of[s["mark"]] = 1

    # -- callouts ---------------------------------------------------------
    by_mark = {s["mark"]: s for s in sets}

    def make(mark, text_at):
        s = by_mark[mark]
        return Callout(at=attach[mark], text_at=text_at, text=s["callout"],
                       mark=mark, width_mm=CALLOUT_W)

    callouts: List[Callout] = []
    left_reach = 0.0        # how far the left callouts reach, sheet mm
    for side in (1, -1):
        marks = [m for m in attach if side_of[m] == side]
        marks.sort(key=lambda m: -attach[m][1])
        # stem callouts spread over the stem so their leaders stay apart
        stems = [m for m in marks if by_mark[m]["element"] == "stem"]
        for k, m in enumerate(stems):
            if by_mark[m]["runs"] == "in_section":
                frac = 0.8 - 0.45 * (k / max(len(stems) - 1, 1))
                y = tb + stem_h * frac
                face = "back" if side == 1 else "front"
                x = attach[m][0] - wall.face_x(face, attach[m][1])
                attach[m] = (wall.face_x(face, y) + x, y)
        marks.sort(key=lambda m: (-attach[m][1], -attach[m][0]))
        if side == 1:
            x_text = B + 8.0 * n
        else:
            x_text = min(wall.xf_b, wall.xf_t) - 7.0 * n
        prev_low = None
        for m in marks:
            probe = make(m, (x_text, attach[m][1]))
            width, height = callout_size(probe)
            y = attach[m][1]
            if side == -1:
                # keep clear of the ground in front and of the base
                y = max(y, max(wall.emb, tb) + (height / 2.0 + 3.0) * n)
            if prev_low is not None:
                y = min(y, prev_low - (height / 2.0 + 1.5) * n)
            prev_low = y - height / 2.0 * n
            callouts.append(make(m, (x_text, y)))
            if side == -1:
                left_reach = max(left_reach, (0.0 - x_text) / n + width)

    # -- dimensions -------------------------------------------------------
    dims: List[Dimension] = []
    cover_zone = 10.0 if ("base", "bottom") in first_cover or (
        "base", "top") in first_cover else 2.0
    left1 = max(cover_zone, left_reach + 2.0) + 7.0
    # heights, left: base thickness and stem height, then the wall height
    dims.append(Dimension(p1=(0.0, 0.0), p2=(0.0, tb), offset_mm=-left1,
                          direction="vertical"))
    dims.append(Dimension(p1=(0.0, tb), p2=(wall.xf_t, H),
                          offset_mm=-left1, direction="vertical"))
    dims.append(Dimension(p1=(0.0, 0.0), p2=(wall.xf_t, H),
                          offset_mm=-(left1 + 8.0), direction="vertical"))
    # widths, below
    drop = -wall.y_min / n
    line = drop + 8.0
    if wall.key:
        kx, kw, kd = wall.key
        if kx > 1e-6:
            dims.append(Dimension(p1=(0.0, -kd), p2=(kx, -kd),
                                  offset_mm=-8.0))
        dims.append(Dimension(p1=(kx, -kd), p2=(kx + kw, -kd),
                              offset_mm=-8.0))
        dims.append(Dimension(p1=(kx + kw, 0.0), p2=(kx + kw, -kd),
                              offset_mm=6.0, direction="vertical"))
        line += 8.0
    chain = [0.0, wall.toe, wall.xb_b, B]
    for a, b in zip(chain, chain[1:]):
        if b - a > 1e-6:
            dims.append(Dimension(p1=(a, 0.0), p2=(b, 0.0), offset_mm=-line))
    dims.append(Dimension(p1=(0.0, 0.0), p2=(B, 0.0),
                          offset_mm=-(line + 8.0)))
    # the stem's top, above; its covers on the line below that
    top_line = 8.0
    stem_covers = []
    for face in ("front", "back"):
        key = ("stem", face)
        if key in outer_edge:
            sec = wall.face_secant(face)
            y = H - first_cover[key]
            x_face = wall.face_x(face, y)
            x_bar = x_face + outer_edge[key] * sec * wall.inside(face)
            stem_covers.append(((x_bar, y), (x_face, y), first_cover[key]))
    for p1, p2, cover in stem_covers:
        if cover > 0:
            dims.append(Dimension(p1=p1, p2=p2,
                                  offset_mm=top_line + cover / n))
    if stem_covers:
        top_line += 8.0
    dims.append(Dimension(p1=(wall.xf_t, H), p2=(wall.xb_t, H),
                          offset_mm=top_line))
    # the base's covers, at the toe's face
    if ("base", "bottom") in outer_edge and first_cover[("base", "bottom")]:
        dims.append(Dimension(p1=(0.0, outer_edge[("base", "bottom")]),
                              p2=(0.0, 0.0), offset_mm=-4.0,
                              direction="vertical"))
    if ("base", "top") in outer_edge and first_cover[("base", "top")]:
        dims.append(Dimension(p1=(0.0, tb - outer_edge[("base", "top")]),
                              p2=(0.0, tb), offset_mm=-4.0,
                              direction="vertical"))

    # -- the ground -------------------------------------------------------
    lines: List[Line] = []
    right_end = B + 30.0 * n
    lines.append(Line(points=[
        (wall.xb_t, H),
        (right_end, H + (right_end - wall.xb_t) * math.tan(wall.beta))],
        layer="ground"))
    if wall.emb > 0:
        x_face = 0.0 if wall.emb <= tb else wall.face_x("front", wall.emb)
        lines.append(Line(points=[(-(left1 - 2.0) * n, wall.emb),
                                  (x_face, wall.emb)], layer="ground"))

    # -- schedule, notes, title block --------------------------------------
    table = Table(
        title=("BAR SCHEDULE (steel areas in mm2 per metre of wall; "
               "along = bars along the wall)"),
        headers=["Mark", "Size", "Spacing", "Place", "As req.", "As prov."],
        rows=[[s["mark"], s["size"], f"{s['spacing_mm']:.0f} mm",
               _place(s["element"], s["face"], s["runs"]),
               _area(s["As_required_mm2_per_m"]),
               _area(s["As_provided_mm2_per_m"])] for s in sets])
    notes = list(desc["notes"])
    if desc["basis"]:
        notes.append("Design basis: " + desc["basis"])
    notes.extend(DRAWING_NOTES)

    block = TitleBlock(**title_block.to_dict())
    if block.title == "Drawing":
        block.title = desc["title"]
    drawing = Drawing(
        outlines=[Outline(points=wall.outline(), layer="concrete",
                          hatch="concrete")],
        lines=lines, bars=bars_out, dimensions=dims, callouts=callouts,
        notes=notes, table=table, title_block=block,
        sheet=Sheet(size=sheet.size, scale=n))
    return drawing, warnings


def wall_section_drawing(description: Any, *, sheet: Any = None,
                         title_block: Any = None) -> Drawing:
    """The reinforced wall section of ``description`` as a
    :class:`Drawing` with its scale set.

    ``description``: a ``reinforced_wall_section/1`` dict. ``sheet``:
    ``{"size": ..., "scale": ...}`` or a :class:`Sheet` (letter, scale
    chosen by the code, by default). ``title_block``: its fields (the
    title defaults to the description's). The layout rules are fixed
    (module docstring). Warnings about the description (a hook length
    that does not fit) are kept on the drawing as ``layout_warnings``."""
    desc = validate_wall_description(description)
    sheet = sheet if isinstance(sheet, Sheet) else Sheet.from_dict(sheet)
    block = (title_block if isinstance(title_block, TitleBlock)
             else TitleBlock.from_dict(title_block))
    # The callouts and dimension lines are placed in sheet millimetres, so
    # the layout depends on the scale: lay it out at each candidate.
    candidates = [sheet.scale] if sheet.scale else list(SCALES)
    last_error: Optional[DrawingError] = None
    for n in candidates:
        drawing, warnings = _layout(desc, sheet, block, n)
        _plan(drawing)      # notes or schedule too big for the sheet: say so
        try:
            choose_scale(drawing)
        except DrawingError as exc:
            last_error = exc
            continue
        drawing.sheet = Sheet(size=sheet.size, scale=n)
        drawing.scale_chosen_by = "given" if sheet.scale else "code"
        drawing.layout_warnings = warnings
        return drawing.validate()
    if sheet.scale:
        raise last_error
    raise DrawingError(
        f"The wall section does not fit a {sheet.size} sheet at any "
        f"standard scale up to 1:{SCALES[-1]}; use a larger sheet "
        "(tabloid, a3).")

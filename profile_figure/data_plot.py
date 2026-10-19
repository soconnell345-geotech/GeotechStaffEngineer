"""Generic data plot — one or more (x, y) series rendered to a PNG.

The "plots of data" half of the owner's 2026-09-11 feedback (calc packages
lacking figures). Every analysis module can produce numbers; until now only a
module with a canned calc package could turn them into a chart, and the agent
has no code-execution tool to improvise one. This is the small, general,
matplotlib-backed plot the agent CAN call: SPT/CPT vs depth, settlement vs
time, load vs displacement, a sensitivity sweep, a method comparison, and
(since 2026-10-15) a plan view drawn to scale: outlines with a fill, points
named beside them, hover text, separate cases on a case axis. Since BACKLOG
GM-109 it also places free text where the caller puts it (``annotations``:
a note, the name of a zone or a line, a reading at a point), with a leader
arrow to the point a note marks.

It draws exactly what it is handed and knows no analysis. Units are whatever
the caller labels the axes with — put them in ``xlabel`` / ``ylabel``.

Usage
-----
>>> from profile_figure import render_data_plot
>>> res = render_data_plot(
...     [{"x": [4, 8, 12, 9], "y": [1.5, 3.0, 4.5, 6.0], "label": "B-1 SPT N"}],
...     xlabel="SPT N (blows/0.3 m)", ylabel="Depth (m)", depth_axis=True,
...     title="Boring B-1 — SPT blow count vs depth")
>>> res.width_px > 0
True
"""

from __future__ import annotations

import base64
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

from profile_figure.plotting import figure_to_png

#: Series styles the caller may ask for. ``polygon`` draws a closed outline
#: (a footprint, a site boundary, a zone), optionally filled.
STYLES = ("line", "markers", "both", "step", "polygon")

#: Marker shapes, by the name the caller gives -> (matplotlib, Plotly).
MARKERS = {
    "circle": ("o", "circle"),
    "square": ("s", "square"),
    "triangle": ("^", "triangle-up"),
    "triangle-down": ("v", "triangle-down"),
    "diamond": ("D", "diamond"),
    "x": ("X", "x"),
    "cross": ("P", "cross"),
    "star": ("*", "star"),
    "hexagon": ("h", "hexagon"),
}

#: Line dashes, by name -> (matplotlib, Plotly).
DASHES = {"solid": ("-", "solid"), "dash": ("--", "dash"),
          "dot": (":", "dot"), "dashdot": ("-.", "dashdot")}

#: Hatch patterns for a polygon's fill (matplotlib and Plotly share them).
HATCHES = ("/", "\\", "x", "-", "|", "+", ".")

#: Where the legend goes. ``auto`` = outside the plot for a drawing to scale
#: (where the data fill the frame), inside otherwise.
LEGEND_POSITIONS = ("auto", "inside", "outside", "below")

#: The keys an annotation (a note of free text) may carry.
ANNOTATION_KEYS = ("text", "x", "y", "coords", "point", "align", "box",
                   "color", "font_size", "halo")

#: Where a note's x, y are measured: in the data's own units, or as
#: fractions of the plot (0, 0 at its bottom left, 1, 1 at its top right).
ANNOTATION_COORDS = ("data", "axes")

_PALETTE = ["#1f4e79", "#c0504d", "#4f8f3a", "#8064a2", "#e08a1e", "#3b8ea5",
            "#7f7f7f", "#a05d2c"]

# Shared drawing constants — the matplotlib (PNG) and Plotly (interactive)
# backends BOTH read these so the two renderings cannot drift apart.
_LINE_WIDTH = 1.6           #: series line width
_MARKER_SIZE = 4.5          #: marker size, style "markers" (no line)
_MARKER_SIZE_BOTH = 3.5     #: marker size, style "both" (line + markers)
_REF_COLOR = "#444444"      #: hline/vline reference-line color
_REF_WIDTH = 1.0            #: hline/vline reference-line width
_GRID_COLOR = "#dddddd"
_GRID_WIDTH = 0.7
_TEXT_COLOR = "#222222"     #: titles, labels, point names: dark on white
_LABEL_SIZE = 7.5           #: point-name font size (points)
_FILL_OPACITY = 0.2         #: default polygon fill opacity
_LEADER_WIDTH = 0.8         #: a note's leader arrow, line width (points)
#: A note's font size is held between these (points).
_NOTE_SIZE_RANGE = (6.0, 20.0)
#: plan_view: the margins round everything drawn, in points as printed:
#: the band below holds the scale bar with its end names, the coordinate
#: system under it and the north arrow beside it; above and at the sides a
#: margin keeps an outline off the edge of the view and leaves room for
#: the name of a point on it (BACKLOG GM-147: margins taken as fractions
#: of the site's height, and an equal aspect free to trim the view, cut
#: the top edge off a 40 x 25 m site and put a long flat site's scale bar
#: names on its property line)
_PLAN_PT = {"side": 14.0, "above": 14.0, "below": 58.0}
#: ... and where the furniture sits in that band, in points: the bar's
#: bottom below the lowest thing drawn, its height, the north arrow's
#: centre left of the right edge and below the lowest thing drawn, and the
#: gap from the bar's right end to the arrow's centre (the arrow and its N
#: reach about 24 pt from the centre at any angle, the bar's end name
#: about 14 pt past the bar).
_PLAN_BAR_DROP = 30.0
_PLAN_BAR_HEIGHT = 4.0
_PLAN_ARROW_IN = 12.0
_PLAN_ARROW_DROP = 32.0
_PLAN_BAR_TO_ARROW = 44.0
#: matplotlib sizes markers in POINTS, Plotly in PIXELS. At the PNG's default
#: 150 dpi one point is ~2 px, so the interactive chart reads like the image.
_PT_TO_PX = 2.0

#: Point-name positions tried in turn, as (dx, dy) in points from the point,
#: with matplotlib's alignment and Plotly's matching ``textposition``.
_LABEL_SPOTS = (
    (0, 5, "center", "bottom", "top center"),
    (5, 0, "left", "center", "middle right"),
    (0, -5, "center", "top", "bottom center"),
    (-5, 0, "right", "center", "middle left"),
    (5, 5, "left", "bottom", "top right"),
    (-5, 5, "right", "bottom", "top left"),
    (5, -5, "left", "top", "bottom right"),
    (-5, -5, "right", "top", "bottom left"),
)


@dataclass
class DataPlotResult:
    """A rendered data plot.

    Attributes
    ----------
    series : list of dict
        Resolved series as drawn: ``label``, ``n`` points, ``x_range``,
        ``y_range``, ``style`` (``x_categories`` instead of ``x_range`` when
        x named cases).
    image_base64 : str
        The PNG, base64-encoded (no data-URI prefix).
    png_bytes : bytes
        Raw PNG bytes (not in ``to_dict()``).
    width_px, height_px : int
        Pixel size of the PNG.
    title, xlabel, ylabel : str
    depth_axis : bool
        True when the y axis was inverted (depth increasing downward).
    equal_aspect : bool
        True when one unit across was drawn as long as one unit up.
    x_on_top : bool
        True when the x axis was drawn along the top (``depth_axis`` or
        ``x_on_top``).
    reversed_axes : list of str
        The axes drawn running from high to low because ``xlim`` / ``ylim``
        gave them that way (``["x"]``); empty otherwise.
    annotations : list of dict
        The notes as drawn: ``text``, ``at`` ([x, y] as given), ``coords``
        and, for a note with a leader, ``point``.
    warnings : list of str
        Non-fatal notes (a dropped non-finite point, an empty label, a note
        that overlaps other lettering or that a line runs through).
    figure : plotly Figure or None
        The interactive twin of the PNG, drawn from the same cleaned series
        (``None`` when ``interactive=False`` or plotly is unavailable). Named
        ``figure`` deliberately: the adapters' sidecar helper reads
        ``getattr(result, "figure", None)``, so this plugs into the existing
        ``.plotly.json`` plumbing unchanged. Not in ``to_dict()``.
    """

    series: List[dict]
    image_base64: str
    png_bytes: bytes
    width_px: int
    height_px: int
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    depth_axis: bool = False
    warnings: List[str] = field(default_factory=list)
    figure: object = None
    equal_aspect: bool = False
    plan_view: bool = False
    x_on_top: bool = False
    reversed_axes: List[str] = field(default_factory=list)
    annotations: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"'{self.title}'" if self.title else "data plot",
                 f"{len(self.series)} series"]
        for s in self.series:
            parts.append(f"{s['label']} ({s['n']} pts)")
        if self.annotations:
            n = len(self.annotations)
            parts.append(f"{n} note{'s' if n > 1 else ''}")
        if self.depth_axis:
            parts.append("depth axis (down)")
        elif self.x_on_top:
            parts.append("x axis along the top")
        for axis in self.reversed_axes:
            parts.append(f"{axis} axis runs high to low, as {axis}lim gives "
                         "it")
        if self.plan_view:
            parts.append("plan view (scale bar, no axes)")
        elif self.equal_aspect:
            parts.append("drawn to scale (1:1)")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "series": list(self.series),
            "title": self.title, "xlabel": self.xlabel, "ylabel": self.ylabel,
            "depth_axis": self.depth_axis,
            "equal_aspect": self.equal_aspect,
            "plan_view": self.plan_view,
            "x_on_top": self.x_on_top,
            "reversed_axes": list(self.reversed_axes),
            "annotations": list(self.annotations),
            "width_px": self.width_px, "height_px": self.height_px,
            "warnings": list(self.warnings),
        }


def _is_number(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    try:
        float(str(value).strip())
        return True
    except (TypeError, ValueError):
        return False


def _per_point(raw, key: str, n: int, idx: int) -> Optional[list]:
    """A per-point list (``labels``, ``hover``) as strings, or None."""
    values = raw.get(key)
    if values is None:
        return None
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise ValueError(f"series[{idx}] '{key}' must be a list with one "
                         f"entry per point")
    if len(values) != n:
        raise ValueError(f"series[{idx}] has {n} points but {len(values)} "
                         f"'{key}' -- give one per point (use \"\" to skip "
                         f"one)")
    return ["" if v is None else str(v) for v in values]


def _clean_series(raw, idx: int, warnings: list) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"series[{idx}] must be a dict with 'x' and 'y'")
    xs, ys = raw.get("x"), raw.get("y")
    if xs is None or ys is None:
        raise ValueError(f"series[{idx}] needs both 'x' and 'y' lists")
    if isinstance(xs, str) or isinstance(ys, str):
        raise ValueError(f"series[{idx}] 'x' and 'y' must be lists")
    xs, ys = list(xs), list(ys)
    if len(xs) != len(ys):
        raise ValueError(f"series[{idx}]: x has {len(xs)} values, y has "
                         f"{len(ys)} — they must match")
    if not xs:
        raise ValueError(f"series[{idx}] is empty")
    # x given as names (cases, scenarios, borings) -> a case axis.
    categorical = not all(_is_number(v) for v in xs)
    try:
        ys = [float(v) for v in ys]
        if not categorical:
            xs = [float(v) for v in xs]
        else:
            xs = [str(v) for v in xs]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"series[{idx}] has a non-numeric value: {exc}")
    labels = _per_point(raw, "labels", len(xs), idx)
    hover = _per_point(raw, "hover", len(xs), idx)
    keep = [i for i in range(len(xs))
            if math.isfinite(ys[i])
            and (categorical or math.isfinite(xs[i]))]
    if len(keep) < len(xs):
        warnings.append(f"series[{idx}]: dropped {len(xs) - len(keep)} "
                        "non-finite point(s)")
    if not keep:
        raise ValueError(f"series[{idx}] has no finite points")
    label = str(raw.get("label") or f"Series {idx + 1}")
    style = str(raw.get("style") or "both").lower()
    if style not in STYLES:
        raise ValueError(f"series[{idx}] style '{style}' not in {STYLES}")
    if style == "polygon" and categorical:
        raise ValueError(f"series[{idx}]: a polygon needs numeric x")
    marker = str(raw.get("marker") or "circle").lower()
    if marker not in MARKERS:
        raise ValueError(f"series[{idx}] marker '{marker}' not in "
                         f"{sorted(MARKERS)}")
    dash = str(raw.get("line_dash") or "solid").lower()
    if dash not in DASHES:
        raise ValueError(f"series[{idx}] line_dash '{dash}' not in "
                         f"{sorted(DASHES)}")
    hatch = raw.get("hatch")
    if hatch is not None:
        hatch = str(hatch)
        if hatch not in HATCHES:
            raise ValueError(f"series[{idx}] hatch '{hatch}' not in "
                             f"{list(HATCHES)}")
        if style != "polygon":
            warnings.append(f"series[{idx}]: 'hatch' applies to style "
                            "'polygon' only; ignored")
            hatch = None
    fill = raw.get("fill")
    if fill is not None and not isinstance(fill, (bool, str)):
        raise ValueError(f"series[{idx}] 'fill' is a colour or true/false")
    if fill and style != "polygon":
        warnings.append(f"series[{idx}]: 'fill' applies to style 'polygon' "
                        "only; ignored")
        fill = None

    def _num(key, default, lo, hi):
        value = raw.get(key)
        if value is None:
            return default
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"series[{idx}] '{key}' must be a number")
        return min(max(value, lo), hi)

    out_x = [xs[i] for i in keep]
    out_y = [ys[i] for i in keep]
    out_labels = [labels[i] for i in keep] if labels else None
    out_hover = [hover[i] for i in keep] if hover else None
    if style == "polygon" and (out_x[0], out_y[0]) != (out_x[-1], out_y[-1]):
        out_x.append(out_x[0])              # close the outline
        out_y.append(out_y[0])
        if out_labels:
            out_labels.append("")
        if out_hover:
            out_hover.append(out_hover[0])
    return {"label": label, "style": style,
            "color": raw.get("color"),
            "x": out_x, "y": out_y,
            "categorical": categorical,
            "marker": marker,
            "marker_size": _num("marker_size", None, 1.0, 30.0),
            "marker_edge": raw.get("marker_edge"),
            "marker_open": bool(raw.get("marker_open", False)),
            "line_dash": dash,
            "line_width": _num("line_width", _LINE_WIDTH, 0.2, 8.0),
            "fill": fill,
            "hatch": hatch,
            "fill_opacity": _num("fill_opacity", _FILL_OPACITY, 0.0, 1.0),
            "labels": out_labels,
            "hover": out_hover,
            "in_legend": raw.get("in_legend", True) is not False,
            "label_spots": None}


def _categories(cleaned) -> Optional[list]:
    """The case names along x, in first-seen order, or None for numeric x.
    Every series must agree: all cases or all numbers."""
    kinds = {s["categorical"] for s in cleaned}
    if len(kinds) > 1:
        raise ValueError("x is names (cases) in some series and numbers in "
                         "others -- use one kind for every series")
    if not kinds.pop():
        return None
    order: list = []
    for s in cleaned:
        for v in s["x"]:
            if v not in order:
                order.append(v)
    return order


def _note_lines(text, idx: int) -> List[str]:
    """A note's text as its lines: a line break may be written as a new
    line, as ``<br>`` (the hover text's way) or as the two characters
    ``\\n`` (a model escaping one twice)."""
    if text is None or isinstance(text, (dict, list, tuple, bool)):
        raise ValueError(f"annotations[{idx}] needs 'text': the words to "
                         "place")
    s = re.sub(r"<br\s*/?>", "\n", str(text), flags=re.IGNORECASE)
    lines = [ln.rstrip() for ln in s.replace("\\n", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError(f"annotations[{idx}] needs 'text': the words to "
                         "place")
    return lines


def _note_align(spec, idx: int) -> tuple:
    """``(ha, va)`` from ``align``: which side of the text stands at the
    note's x, y -- 'center' (the default), 'left', 'right', 'top',
    'bottom', or one of each ('top left')."""
    words = str(spec or "center").lower().replace("-", " ") \
        .replace("_", " ").split()
    ha = va = None
    for w in words:
        if w in ("left", "right"):
            if ha is not None:
                break
            ha = w
        elif w in ("top", "bottom"):
            if va is not None:
                break
            va = w
        elif w not in ("center", "centre", "middle"):
            break
    else:
        return ha or "center", va or "center"
    raise ValueError(f"annotations[{idx}] align '{spec}': give the side of "
                     "the text that stands at x, y -- center, left, right, "
                     "top, bottom, or one of each, e.g. 'top left'")


def _note_coord(value, idx: int, what: str, categories, log: bool,
                in_data: bool) -> float:
    """One coordinate of a note, as drawn: a number, or on a case axis a
    case's name (placed at that case)."""
    if (in_data and categories is not None and what.endswith("x")
            and isinstance(value, str) and not _is_number(value)):
        if value in categories:
            return float(categories.index(value))
        raise ValueError(f"annotations[{idx}] {what} '{value}' is not one "
                         f"of the cases {categories}")
    if not _is_number(value):
        cases = (" (or one of the cases along x)"
                 if in_data and categories is not None
                 and what.endswith("x") else "")
        raise ValueError(f"annotations[{idx}] {what} must be a number"
                         f"{cases}, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"annotations[{idx}] {what} must be a finite number")
    if in_data and log and v <= 0:
        raise ValueError(f"annotations[{idx}] {what} must be above zero on "
                         "a log axis")
    return v


def _note_colour(value, idx: int, what: str, default: str) -> str:
    from matplotlib.colors import is_color_like
    if value is None:
        return default
    if not isinstance(value, str) or not is_color_like(value):
        raise ValueError(f"annotations[{idx}] {what} {value!r} is not a "
                         "colour (a name such as 'gray', or '#1f4e79')")
    return value


def _clean_notes(raw, categories, logx: bool, logy: bool,
                 warnings: Optional[list] = None) -> List[dict]:
    """The notes (``annotations``) as both renderings draw them, checked:
    each needs its text and a number for x and y; ``coords`` is 'data' (the
    default) or 'axes' (fractions of the plot); ``point`` is the [x, y], in
    data units, that the note marks with a leader arrow. A ``font_size``
    outside 6-20 pt is drawn at the nearer end, with a warning."""
    import difflib
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("annotations must be a list of {text, x, y} dicts")
    notes = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"annotations[{idx}] must be a dict with "
                             "'text', 'x' and 'y'")
        unknown = sorted(str(k) for k in item if k not in ANNOTATION_KEYS)
        if unknown:
            near = [f"{k} -> {m[0]}" for k in unknown
                    for m in [difflib.get_close_matches(
                        k, ANNOTATION_KEYS, n=1, cutoff=0.6)] if m]
            hint = f" Did you mean: {'; '.join(near)}?" if near else ""
            raise ValueError(f"annotations[{idx}] has unknown key(s) "
                             f"{unknown}.{hint} Valid keys: "
                             f"{sorted(ANNOTATION_KEYS)}.")
        lines = _note_lines(item.get("text"), idx)
        coords = str(item.get("coords") or "data").lower().strip()
        if coords not in ANNOTATION_COORDS:
            raise ValueError(f"annotations[{idx}] coords '{item['coords']}' "
                             "not in ('data', 'axes'): 'data' for the data's "
                             "own units, 'axes' for fractions of the plot "
                             "(0, 0 bottom left; 1, 1 top right)")
        for key in ("x", "y"):
            if item.get(key) is None:
                raise ValueError(f"annotations[{idx}] needs '{key}' (where "
                                 "the text stands)")
        in_data = coords == "data"
        x = _note_coord(item["x"], idx, "x", categories, logx, in_data)
        y = _note_coord(item["y"], idx, "y", categories, logy, in_data)
        point = item.get("point")
        if point is not None:
            if (isinstance(point, (str, dict)) or not hasattr(point, "__len__")
                    or len(point) != 2):
                raise ValueError(f"annotations[{idx}] point must be [x, y] "
                                 "in data units: the point the note marks")
            point = (_note_coord(point[0], idx, "point x", categories, logx,
                                 True),
                     _note_coord(point[1], idx, "point y", categories, logy,
                                 True))
        ha, va = _note_align(item.get("align"), idx)
        box = item.get("box")
        if box is True:
            box = "white"
        elif box is False or box is None:
            box = None
        else:
            box = _note_colour(box, idx, "box", "white")
        halo = item.get("halo")
        if halo is True:
            halo = "background"         # the plot's own background colour
        elif halo is False or halo is None:
            halo = None
        else:
            halo = _note_colour(halo, idx, "halo", "white")
        size = item.get("font_size")
        if size is None:
            size = _LABEL_SIZE
        elif not _is_number(size) or not math.isfinite(float(size)):
            raise ValueError(f"annotations[{idx}] font_size must be a number "
                             "(points)")
        lo, hi = _NOTE_SIZE_RANGE
        size = float(size)
        if not lo <= size <= hi and warnings is not None:
            warnings.append(f"annotations[{idx}] font_size {size:g} pt drawn "
                            f"at {min(max(size, lo), hi):g} pt (a note is "
                            f"{lo:g} to {hi:g} pt)")
        echo = {"text": "\n".join(lines),
                "at": [item["x"], item["y"]], "coords": coords}
        if point is not None:
            echo["point"] = list(item["point"])
        notes.append({
            "index": idx, "lines": lines, "x": x, "y": y, "coords": coords,
            "point": point, "ha": ha, "va": va, "box": box, "halo": halo,
            "color": _note_colour(item.get("color"), idx, "color",
                                  _TEXT_COLOR),
            "font_size": min(max(size, lo), hi),
            "echo": echo})
    return notes


def _marker_size(s) -> float:
    if s["marker_size"] is not None:
        return s["marker_size"]
    return _MARKER_SIZE if s["style"] == "markers" else _MARKER_SIZE_BOTH


def _rgba(color: str, alpha: float) -> str:
    """``color`` (any matplotlib colour) as a Plotly rgba() string."""
    from matplotlib.colors import to_rgba
    r, g, b, _a = to_rgba(color)
    return (f"rgba({int(round(r * 255))},{int(round(g * 255))},"
            f"{int(round(b * 255))},{alpha:g})")


def _resolve_legend_position(position: Optional[str],
                             equal_aspect: bool) -> str:
    pos = str(position or "auto").lower()
    if pos not in LEGEND_POSITIONS:
        raise ValueError(f"legend_position '{position}' not in "
                         f"{LEGEND_POSITIONS}")
    if pos == "auto":
        return "outside" if equal_aspect else "inside"
    return pos


def _limits(spec, what: str) -> Optional[tuple]:
    """``[start, end]`` of an axis, in the order given: ``[100, 0.001]``
    draws the axis running high to low (live wave w9: a range given that
    way was sorted without a word and the axis came out the other way)."""
    if spec is None:
        return None
    try:
        lo, hi = (float(v) for v in spec)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be [start, end]: two numbers, either "
                         "way round (high to low runs the axis that way)")
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo == hi:
        raise ValueError(f"{what} must be two different finite numbers")
    return (lo, hi)


def _axis_limits(xlim, ylim, logx: bool, logy: bool, depth_axis: bool,
                 warnings: Optional[list] = None) -> tuple:
    """``(xl, yl, reversed_axes)`` as both renderings draw them: each range
    in its given order, except that ``depth_axis`` sets the direction of y
    itself; a range a log axis cannot take (zero or below) is left out with
    a warning."""
    xl, yl = _limits(xlim, "xlim"), _limits(ylim, "ylim")
    for name, lim, log in (("xlim", xl, logx), ("ylim", yl, logy)):
        if lim and log and min(lim) <= 0:
            if warnings is not None:
                warnings.append(f"{name} ignored: a log axis needs limits "
                                "above zero")
            if name == "xlim":
                xl = None
            else:
                yl = None
    if yl and depth_axis:
        yl = (min(yl), max(yl))
    reversed_axes = [axis for axis, lim in (("x", xl), ("y", yl))
                     if lim and lim[0] > lim[1]]
    return xl, yl, reversed_axes


def _plotly_range(lim, log: bool) -> list:
    """A Plotly axis range: on a log axis Plotly takes log10 of the values;
    a range high to low draws the axis that way."""
    return [math.log10(v) for v in lim] if log else list(lim)


def _axes_to_data(fx: float, fy: float, view, logx: bool,
                  logy: bool) -> tuple:
    """A point given in fractions of the plot, in data units, for the view
    ``((left, right), (bottom, top))`` (each in the order drawn)."""
    def along(f, ends, log):
        a, b = ends
        if log:
            return 10 ** (math.log10(a) + f * (math.log10(b) - math.log10(a)))
        return a + f * (b - a)
    return along(fx, view[0], logx), along(fy, view[1], logy)


def _fallback_view(cleaned, categories, xlim, ylim, logx, logy,
                   depth_axis) -> tuple:
    """The view a caller of the Plotly builder alone did not give: the
    axis limits, else the data's own extent."""
    xl, yl, _rev = _axis_limits(xlim, ylim, logx, logy, depth_axis)

    def extent(values):
        lo, hi = min(values), max(values)
        return (lo, hi) if hi > lo else (lo - 1.0, hi + 1.0)
    if categories is not None:
        xv = (-0.5, len(categories) - 0.5)
    else:
        xv = xl or extent([x for s in cleaned for x in s["x"]])
    yv = yl or extent([y for s in cleaned for y in s["y"]])
    if depth_axis:
        yv = (max(yv), min(yv))
    return xv, yv


def _plotly_notes(fig, notes, logx: bool, logy: bool, view) -> bool:
    """The notes on the Plotly twin, where the PNG has them: text at x, y in
    data units (log10 of them on a log axis, as Plotly takes positions
    there) or in fractions of the plot (the axis domain), anchored on the
    same side, on the same box, with the leader arrow from the text to the
    point. Plotly draws a leader's tail in the head's coordinates, so a note
    in fractions of the plot with a leader is placed in data units for the
    PNG's view, and the view is then held to the PNG's (returns True)."""
    def pos(v, log):
        return math.log10(v) if log else v

    hold_view = False
    for n in notes:
        kw = {"text": "<br>".join(n["lines"]), "showarrow": False,
              "xanchor": n["ha"],
              "yanchor": "middle" if n["va"] == "center" else n["va"],
              "align": n["ha"],
              "font": {"size": n["font_size"] * _PT_TO_PX * 0.75,
                       "color": n["color"]}}
        if n["box"]:
            kw.update(bgcolor=_rgba(n["box"], 1.0), borderpad=3,
                      borderwidth=0)
        x, y = n["x"], n["y"]
        if n["coords"] == "axes":
            if n["point"] is None:
                fig.add_annotation(xref="x domain", yref="y domain", x=x,
                                   y=y, **kw)
                continue
            x, y = _axes_to_data(x, y, view, logx, logy)
            hold_view = True
        if n["point"] is None:
            fig.add_annotation(xref="x", yref="y", x=pos(x, logx),
                               y=pos(y, logy), **kw)
            continue
        kw.update(showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
                  arrowcolor=n["color"], standoff=2, startstandoff=2)
        fig.add_annotation(xref="x", yref="y",
                           x=pos(n["point"][0], logx),
                           y=pos(n["point"][1], logy),
                           axref="x", ayref="y", ax=pos(x, logx),
                           ay=pos(y, logy), **kw)
    return hold_view


def build_data_plot_figure(cleaned_series, *, title: str = "",
                           xlabel: str = "", ylabel: str = "",
                           depth_axis: bool = False, logx: bool = False,
                           logy: bool = False, hlines=None, vlines=None,
                           grid: bool = True,
                           legend: Optional[bool] = None,
                           equal_aspect: bool = False, xlim=None, ylim=None,
                           legend_position: Optional[str] = None,
                           plan_view: bool = False,
                           north_deg: Optional[float] = None,
                           scale_units: str = "m", x_on_top: bool = False,
                           annotations=None, view=None, plan_layout=None,
                           ref_spots=None):
    """Build the interactive Plotly twin of :func:`render_data_plot`'s PNG.

    ``cleaned_series`` is what :func:`_clean_series` produced for the
    matplotlib pass — the SAME resolved points, labels, styles and colors, so
    the two backends draw identical data (including the dropped-non-finite
    points), and the point names stand where the PNG put them. Returns a
    ``plotly.graph_objects.Figure``, or ``None`` when plotly cannot be
    imported. ``plotly>=5`` is a core dependency, so that guard is belt and
    braces; the import stays inside the function per the
    ``docs/CALC_VIZ_PLAN.md`` ground rule (no plotting import at module load).

    Semantics mirror the matplotlib figure: ``depth_axis`` reverses the y axis
    and moves the x axis to the top (``x_on_top`` moves it alone), ``xlim`` /
    ``ylim`` keep the order given (high to low reverses the axis, on a log
    axis too), ``logx``/``logy`` set log axes, the
    styles map to lines / markers / lines+markers / step (``line_shape="hv"``,
    matplotlib's ``where="post"``) / a closed outline (``fill="toself"``),
    ``equal_aspect`` ties the y scale to the x scale, and reference lines are
    drawn in the same color, width and dash. One deliberate difference: a
    labelled reference line becomes a chart ANNOTATION here (matplotlib puts
    it in the legend) — Plotly shapes carry no legend entry.

    ``annotations`` are the notes as :func:`_clean_notes` resolved them for
    the PNG, drawn where the PNG has them. ``view`` is the PNG's own axis
    limits, ``((left, right), (bottom, top))``: a note placed in fractions
    of the plot with a leader to a data point is put in data units for that
    view, and the chart is drawn over the same view (without one, the axis
    limits or the data's extent stand in for it). ``plan_layout`` is the
    PNG's plan layout (:func:`_plan_layout`), so a plan's view and furniture
    stand where the PNG has them; ``ref_spots`` where the PNG put each
    reference line's name, ``{("h" | "v", k): "top right" | ...}`` (k the
    line's index in ``hlines`` / ``vlines``; top right when not given).
    """
    try:
        import plotly.graph_objects as go
    except Exception:                       # noqa: BLE001 — plotly absent
        return None

    categories = _categories(cleaned_series)
    fig = go.Figure()
    # outlines first, so points and lines stand on top of a filled zone
    order = sorted(range(len(cleaned_series)),
                   key=lambda i: cleaned_series[i]["style"] != "polygon")
    for i in order:
        s = cleaned_series[i]
        color = s["color"] or _PALETTE[i % len(_PALETTE)]
        style = s["style"]
        dash = DASHES[s["line_dash"]][1]
        shape, marker_pt = "linear", 0.0
        if style == "markers":
            mode, marker_pt = "markers", _marker_size(s)
        elif style in ("line", "polygon"):
            mode = "lines"
        elif style == "step":
            mode, shape = "lines", "hv"
        else:
            mode, marker_pt = "lines+markers", _marker_size(s)
        has_text = bool(s["labels"] and any(s["labels"]))
        if has_text:
            mode += "+text"
        trace = go.Scatter(
            x=list(s["x"]), y=list(s["y"]), name=s["label"], mode=mode,
            line={"color": color, "width": s["line_width"], "shape": shape,
                  "dash": dash},
            showlegend=s["in_legend"],
        )
        if marker_pt:
            symbol = MARKERS[s["marker"]][1]
            if s["marker_open"]:
                symbol += "-open"
            marker = {"color": color, "size": marker_pt * _PT_TO_PX,
                      "symbol": symbol}
            if s["marker_edge"]:
                marker["line"] = {"color": s["marker_edge"], "width": 1}
            trace.marker = marker
        if style == "polygon" and s["fill"]:
            fill_color = color if s["fill"] is True else s["fill"]
            trace.fill = "toself"
            trace.fillcolor = _rgba(fill_color, s["fill_opacity"])
        if style == "polygon" and s["hatch"]:
            trace.fill = "toself"
            if not s["fill"]:
                trace.fillcolor = "rgba(0,0,0,0)"
            trace.fillpattern = {"shape": s["hatch"], "fgcolor": color,
                                 "solidity": 0.15}
        if has_text:
            trace.text = list(s["labels"])
            trace.textposition = (list(s["label_spots"]) if s["label_spots"]
                                  else "top center")
            trace.textfont = {"size": _LABEL_SIZE * _PT_TO_PX * 0.75,
                              "color": _TEXT_COLOR}
        if s["hover"]:
            trace.hovertext = list(s["hover"])
            trace.hoverinfo = "text"
        fig.add_trace(trace)

    ref_spots = ref_spots or {}
    for k, spec in enumerate(hlines or []):
        v, lab = _ref(spec)
        pos = _axis_coord(v, logy)
        if pos is not None:
            fig.add_hline(y=pos, line_color=_REF_COLOR, line_width=_REF_WIDTH,
                          line_dash="dash",
                          **({"annotation_text": lab,
                              "annotation_position":
                                  ref_spots.get(("h", k), "top right"),
                              "annotation_font": {"size": 10,
                                                  "color": _REF_COLOR}}
                             if lab else {}))
    for k, spec in enumerate(vlines or []):
        v, lab = _ref(spec)
        pos = _axis_coord(v, logx)
        if pos is not None:
            fig.add_vline(x=pos, line_color=_REF_COLOR, line_width=_REF_WIDTH,
                          line_dash="dot",
                          **({"annotation_text": lab,
                              "annotation_position":
                                  ref_spots.get(("v", k), "top right"),
                              "annotation_font": {"size": 10,
                                                  "color": _REF_COLOR}}
                             if lab else {}))

    # A reference line is named on the line itself (both renderings), so
    # the legend is for the series only.
    shown = [s for s in cleaned_series if s["in_legend"]]
    show_legend = (legend if legend is not None else len(shown) > 1)
    position = _resolve_legend_position(legend_position, equal_aspect)
    legend_kw = {"font": {"size": 11, "color": _TEXT_COLOR}}
    if position == "below":
        legend_kw.update(orientation="h", x=0.0, y=-0.18, yanchor="top")
    elif position == "inside":
        legend_kw.update(x=0.01, y=0.99, xanchor="left", yanchor="top",
                         bgcolor="rgba(255,255,255,0.8)")
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white",
        font={"color": _TEXT_COLOR},
        showlegend=bool(show_legend),
        legend=legend_kw,
        margin={"l": 70, "r": 30, "t": 70 if title else 40, "b": 60},
    )
    if title:
        fig.update_layout(title={"text": title,
                                 "font": {"size": 15, "color": _TEXT_COLOR}})
    axis_kw = {"showgrid": bool(grid), "gridcolor": _GRID_COLOR,
               "gridwidth": _GRID_WIDTH, "zeroline": False,
               "title_font": {"size": 13, "color": _TEXT_COLOR},
               "tickfont": {"color": _TEXT_COLOR},
               "showline": True, "linecolor": "#888888"}
    fig.update_xaxes(title_text=xlabel or "", **axis_kw)
    fig.update_yaxes(title_text=ylabel or "", **axis_kw)
    if categories is not None:
        fig.update_xaxes(type="category", categoryorder="array",
                         categoryarray=list(categories))
    if logx:
        fig.update_xaxes(type="log")
    if logy:
        fig.update_yaxes(type="log")
    xl, yl, _rev = _axis_limits(xlim, ylim, logx, logy, depth_axis)
    if xl:
        fig.update_xaxes(range=_plotly_range(xl, logx))
    if depth_axis:                          # boring-log convention, as the PNG
        if yl:
            fig.update_yaxes(range=_plotly_range((yl[1], yl[0]), logy))
        else:
            fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(side="top")
    else:
        if yl:
            fig.update_yaxes(range=_plotly_range(yl, logy))
        if x_on_top:
            fig.update_xaxes(side="top")
    if equal_aspect or plan_view:
        # one unit up as long as one unit across: a plan or a section to scale
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    if plan_view:
        # a drawing, not a chart: no axes or grid; a scale bar and north
        # the view kept whole: the equal aspect fits the plot's box, never
        # trims the range (BACKLOG GM-147)
        fig.update_xaxes(visible=False, constrain="domain")
        fig.update_yaxes(visible=False, constrain="domain")
        if title:
            fig.update_layout(title={"x": 0.0, "xanchor": "left"})
        _plotly_plan_furniture(fig, cleaned_series, scale_units, north_deg,
                               xlabel, plan_layout)
    if annotations:
        if view is None:
            view = _fallback_view(cleaned_series, categories, xlim, ylim,
                                  logx, logy, depth_axis)
        if _plotly_notes(fig, annotations, logx, logy, view):
            fig.update_xaxes(range=_plotly_range(view[0], logx),
                             autorange=False)
            fig.update_yaxes(range=_plotly_range(view[1], logy),
                             autorange=False)
    return fig


from geotech_common.plan_furniture import nice_length as _nice_length


def _plan_layout(cleaned, axes_pt=None) -> dict:
    """Where plan_view's furniture goes, in data units, for both renderings:
    a band below the site holds the scale bar and, at its right, the north
    arrow (drafting practice keeps them together); the coordinate system is
    named under the bar. Nothing is drawn on the site itself.

    The view takes in every outline, line and point drawn with a margin
    all round (BACKLOG GM-147), and the band and margins are sized in
    points as printed, so the furniture reads the same on a long flat site
    as on a tall narrow one. ``axes_pt`` is the room the plot has, ``(width,
    height)`` in points (the matplotlib figure's); without it (the
    interactive chart drawn alone) a 6.5 in figure with the legend outside
    is assumed. ``scale`` is data units per point."""
    xs = [x for s in cleaned for x in s["x"]]
    ys = [y for s in cleaned for y in s["y"]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (x1 - x0) or 1.0, (y1 - y0) or 1.0
    if (x1 - x0) == 0 and (y1 - y0) > 0:
        dx = dy
    if (y1 - y0) == 0:
        dy = dx
    length = _nice_length(dx)
    side, above, below = (_PLAN_PT["side"], _PLAN_PT["above"],
                          _PLAN_PT["below"])
    if axes_pt is None:
        w = (6.5 - 1.3 - 1.9) * 72.0
        axes_pt = (w, max(w * (dy + 1e-9) / dx, 1.0) + above + below)
    w, h = (max(float(v), 1.0) for v in axes_pt)
    # the drawing's scale (data units per point): the view (everything
    # drawn, the bar and the margins) fills the room it has one way and
    # fits the other; the bar may reach left of a narrow site
    scale = max(dx / max(w - 2 * side, 0.2 * w),
                dy / max(h - above - below, 0.2 * h))
    for _ in range(20):
        bx = x1 - (_PLAN_ARROW_IN + _PLAN_BAR_TO_ARROW) * scale - length
        left = min(x0 - side * scale, bx - side * scale)
        right = x1 + side * scale
        bottom, top = y0 - below * scale, y1 + above * scale
        new = max((right - left) / w, (top - bottom) / h)
        if abs(new - scale) <= 1e-9 * scale:
            break
        scale = new
    bx = x1 - (_PLAN_ARROW_IN + _PLAN_BAR_TO_ARROW) * scale - length
    by = y0 - _PLAN_BAR_DROP * scale
    return {"xlim": (min(x0 - side * scale, bx - side * scale),
                     x1 + side * scale),
            "ylim": (y0 - below * scale, y1 + above * scale),
            "scale": scale, "length": length, "bx": bx, "by": by,
            "h": _PLAN_BAR_HEIGHT * scale,
            "arrow": (x1 - _PLAN_ARROW_IN * scale,
                      y0 - _PLAN_ARROW_DROP * scale),
            "caption_xy": (bx + length, by)}


def _draw_plan_furniture(ax, cleaned, units, north_deg, caption):
    """The scale bar, the north arrow beside it and the coordinate system
    under the bar, in a band below the site
    (:mod:`geotech_common.plan_furniture`), the view set to take in
    everything drawn with a margin and kept so (an equal aspect fitted by
    the axes' box, never by trimming the view). Returns the layout, the
    text artists and the other artists drawn (the bar, the arrow), so
    point names keep clear of them and the layout check names them."""
    from geotech_common import plan_furniture as pf

    fig = ax.figure
    pos = ax.get_position(original=True)
    fw, fh = fig.get_size_inches()
    lay = _plan_layout(cleaned, (pos.width * fw * 72.0,
                                 pos.height * fh * 72.0))
    ax.set_autoscale_on(False)
    ax.set_xlim(*lay["xlim"])
    ax.set_ylim(*lay["ylim"])
    ax.set_aspect("equal", adjustable="box")
    before = {id(a) for a in ax.get_children()}
    texts = pf.mpl_scale_bar(ax, lay["bx"], lay["by"], lay["length"],
                             lay["h"], units, fontsize=_LABEL_SIZE,
                             text_color=_TEXT_COLOR)
    if caption:
        texts.append(ax.annotate(
            caption, lay["caption_xy"], xytext=(0, -6),
            textcoords="offset points", ha="right", va="top",
            fontsize=_LABEL_SIZE - 0.5, color="#555555", zorder=7))
    angle, _note = pf.north_or_default(north_deg)
    texts += pf.mpl_north_arrow(ax, lay["arrow"], angle)
    named = {id(t) for t in texts}
    shapes = [a for a in ax.get_children()
              if id(a) not in before and id(a) not in named]
    return lay, texts, shapes


def _plotly_plan_furniture(fig, cleaned, units, north_deg, caption,
                           lay=None):
    """The Plotly twin's scale bar, north arrow and caption (the PNG's
    layout when given)."""
    from geotech_common import plan_furniture as pf

    lay = lay or _plan_layout(cleaned)
    fig.update_xaxes(range=list(lay["xlim"]))
    fig.update_yaxes(range=list(lay["ylim"]))
    pf.plotly_scale_bar(fig, lay["bx"], lay["by"], lay["length"], lay["h"],
                        units, text_color=_TEXT_COLOR)
    if caption:
        cx, cy = lay["caption_xy"]
        fig.add_annotation(x=cx, y=cy, xref="x", yref="y", text=caption,
                           showarrow=False, yanchor="top", yshift=-8,
                           xanchor="right",
                           font={"size": 9, "color": "#555555"})
    angle, _note = pf.north_or_default(north_deg)
    pf.plotly_north_arrow(fig, lay["arrow"], angle)


def _axis_coord(value: float, log_axis: bool):
    """Where Plotly wants a reference line drawn on a linear or log axis.

    Shape coordinates on a LOG axis are log10 of the data value (plotly.py
    does not convert them for ``add_hline``/``add_vline``), so a line at 100
    on a log axis is drawn at 2.0. A non-positive value has no place on a log
    axis — matplotlib would put it off the canvas; here it is skipped.
    """
    if not log_axis:
        return value
    if value <= 0:
        return None
    return math.log10(value)


def _scale_height(cleaned, width_in: float, legend_out: bool,
                  plan_view: bool = False) -> float:
    """A figure height for a drawing to scale: the data's own proportions in
    the space the axes get, so a long site is not drawn in a square box."""
    xs = [v for s in cleaned for v in s["x"]]
    ys = [v for s in cleaned for v in s["y"]]
    span_x = (max(xs) - min(xs)) or 1.0
    span_y = (max(ys) - min(ys)) or 1.0
    axes_w = max(width_in - 1.3 - (1.9 if legend_out else 0.0), 2.0)
    if plan_view:
        # the site to scale across the room left by the side margins, and
        # the bands above and below it as printed (_PLAN_PT)
        site_w = max(axes_w - 2 * _PLAN_PT["side"] / 72.0, 1.0)
        bands = (_PLAN_PT["above"] + _PLAN_PT["below"]) / 72.0
        return min(max(site_w * span_y / span_x + bands + 1.2, 3.0), 10.0)
    return min(max(axes_w * span_y / span_x + 1.2, 3.0), 10.0)


def _overlap(a, b) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if (w > 0 and h > 0) else 0.0


def _segment_boxes(p0, p1, half: float, skip=None) -> list:
    """Small boxes every few pixels along the segment ``p0``-``p1`` (pixel
    coordinates): a line as an obstacle; none inside ``skip`` (a box)."""
    from matplotlib.transforms import Bbox
    (x0, y0), (x1, y1) = p0, p1
    n = int(min(max(math.hypot(x1 - x0, y1 - y0) / 6.0, 1), 200))
    out = []
    for k in range(n + 1):
        cx = x0 + (x1 - x0) * k / n
        cy = y0 + (y1 - y0) * k / n
        if skip is not None and skip.contains(cx, cy):
            continue
        out.append(Bbox([[cx - half, cy - half], [cx + half, cy + half]]))
    return out


def _data_obstacles(fig, ax, drawn, lines=None) -> list:
    """Every point and line drawn, as pixel boxes (lines sampled every few
    pixels along them). ``drawn`` is ``[(series, xs_plotted)]``. With
    ``lines`` (a list) the lines' boxes go there instead."""
    from matplotlib.transforms import Bbox
    to_px = ax.transData.transform
    boxes = []
    line_boxes = boxes if lines is None else lines
    for s, xs in drawn:                     # every point is an obstacle
        size_px = _marker_size(s) * fig.dpi / 72.0
        for x, y in zip(xs, s["y"]):
            px, py = to_px((x, y))
            boxes.append(Bbox([[px - size_px, py - size_px],
                               [px + size_px, py + size_px]]))
        if s["style"] == "markers":
            continue
        # ... and so is every line
        pts = [to_px((x, y)) for x, y in zip(xs, s["y"])]
        half = max(s["line_width"] * fig.dpi / 72.0, 1.5)
        for p0, p1 in zip(pts, pts[1:]):
            line_boxes += _segment_boxes(p0, p1, half)
    return boxes


def _note_box(artist, renderer):
    """What a note covers: its text, or the box behind it when it has one
    (never its leader)."""
    from matplotlib.text import Text
    patch = artist.get_bbox_patch()
    if patch is not None:
        return patch.get_window_extent(renderer)
    return Text.get_window_extent(artist, renderer)


def _leader_boxes(fig, ax, artist, point, renderer) -> list:
    """A note's leader, from its text to the point it marks, as pixel
    boxes (the stretch under the text left out)."""
    text = _note_box(artist, renderer)
    start = ((text.x0 + text.x1) / 2.0, (text.y0 + text.y1) / 2.0)
    end = ax.transData.transform(point)
    half = max(_LEADER_WIDTH * fig.dpi / 72.0, 1.5)
    return _segment_boxes(start, end, half, skip=text)


def _draw_notes(ax, notes) -> list:
    """Draw the notes where they were put (never moved): text at x, y in
    data units or in fractions of the plot, aligned as asked, on a plain
    box when asked, with a leader arrow to the point a note marks. Returns
    the text artists, in order."""
    artists = []
    for n in notes:
        kw = {"fontsize": n["font_size"], "color": n["color"],
              "ha": n["ha"], "va": n["va"], "multialignment": n["ha"],
              "zorder": 6, "annotation_clip": False}
        if n["box"]:
            kw["bbox"] = {"boxstyle": "square,pad=0.25",
                          "facecolor": n["box"], "edgecolor": "none"}
        if n.get("halo"):
            # a thin ring of the background round each letter: hatching or
            # a grid line behind the text stops short of it (the PNG only;
            # the interactive chart's text has no such ring)
            from matplotlib import patheffects
            ring = (ax.get_facecolor() if n["halo"] == "background"
                    else n["halo"])
            kw["path_effects"] = [patheffects.withStroke(linewidth=2.5,
                                                         foreground=ring)]
        where = "data" if n["coords"] == "data" else "axes fraction"
        text = "\n".join(n["lines"])
        if n["point"] is None:
            artists.append(ax.annotate(text, (n["x"], n["y"]),
                                       xycoords=where, **kw))
            continue
        artists.append(ax.annotate(
            text, n["point"], xycoords="data", xytext=(n["x"], n["y"]),
            textcoords=where,
            arrowprops={"arrowstyle": "-|>", "color": n["color"],
                        "lw": _LEADER_WIDTH, "mutation_scale": 8,
                        "shrinkA": 2, "shrinkB": 2}, **kw))
    return artists


#: How much more a line through a point name's letters costs than its area
#: of overlap with the line's thin boxes (a marker's box is about 6 times
#: as wide as a line's); a line only in the margin round the name costs
#: its area, as anything else there does.
_LINE_COST = 6.0


def _ref_line_boxes(fig, ax, ref_lines) -> list:
    """Each reference line (``("h" | "v", value)``) across the plot as
    pixel boxes along it: an obstacle for point names, as a series' line is
    (BACKLOG GM-146: a vertical LL = 50 line ran through two names)."""
    half = max(_REF_WIDTH * fig.dpi / 72.0, 1.5)
    boxes = []
    for kind, v in ref_lines:
        if kind == "h":
            ends = ax.get_yaxis_transform().transform([(0.0, v), (1.0, v)])
        else:
            ends = ax.get_xaxis_transform().transform([(v, 0.0), (v, 1.0)])
        if not all(math.isfinite(c) for p in ends for c in p):
            continue
        boxes += _segment_boxes(tuple(ends[0]), tuple(ends[1]), half)
    return boxes


def _hatched_areas(ax, drawn) -> list:
    """The hatched outlines (style ``polygon`` with a hatch) as pixel paths:
    a name inside one has hatching through its letters."""
    from matplotlib.path import Path
    to_px = ax.transData.transform
    return [Path(to_px(list(zip(xs, s["y"]))))
            for s, xs in drawn
            if s["style"] == "polygon" and s.get("hatch") and len(xs) >= 3]


def _hatch_cost(bb, areas) -> float:
    """The share of a name's box inside hatched outlines, as an area: nine
    points of the box tested."""
    if not areas:
        return 0.0
    pts = [(bb.x0 + (bb.x1 - bb.x0) * i / 2.0,
            bb.y0 + (bb.y1 - bb.y0) * j / 2.0)
           for i in range(3) for j in range(3)]
    inside = 0
    for path in areas:
        inside = max(inside, int(path.contains_points(pts).sum()))
    return (bb.x1 - bb.x0) * (bb.y1 - bb.y0) * inside / 9.0


#: Where a reference line's name may stand, in turn: (axes fraction along
#: the line, offset in points, alignment, Plotly's annotation_position).
#: The first is where it always stood.
_REF_SPOTS = {
    "h": ((1.0, (-3, 2), "right", "bottom", "top right"),
          (1.0, (-3, -2), "right", "top", "bottom right"),
          (0.0, (3, 2), "left", "bottom", "top left"),
          (0.0, (3, -2), "left", "top", "bottom left")),
    "v": ((1.0, (3, -3), "left", "top", "top right"),
          (1.0, (-3, -3), "right", "top", "top left"),
          (0.0, (3, 3), "left", "bottom", "bottom right"),
          (0.0, (-3, 3), "right", "bottom", "bottom left")),
}


def _place_ref_names(fig, ax, ref_names, drawn, fixed_texts=(),
                     leaders=()) -> dict:
    """Put each reference line's name at whichever end of its line, above
    or below it (left or right of a vertical one), covers least of the
    points, the lines, the notes and the other names, inside the plot frame
    (the right end above the line, as before, wins a tie). ``ref_names`` is
    ``[(artist, "h" | "v", value, k)]``. Returns ``{(kind, k): Plotly's
    annotation_position}`` for the interactive twin (BACKLOG GM-152: a
    worked example's "Best estimate" stood on a data point)."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    pad = 2.0
    taken = [_note_box(t, renderer) for t in fixed_texts]
    for artist, point in leaders:
        taken += _leader_boxes(fig, ax, artist, point, renderer)
    taken += _data_obstacles(fig, ax, drawn)
    spots = {}
    for t, kind, v, k in ref_names:
        best = None
        for along, off, ha, va, plotly_pos in _REF_SPOTS[kind]:
            t.xy = (along, v) if kind == "h" else (v, along)
            t.xyann = off
            t.set_ha(ha)
            t.set_va(va)
            bb = t.get_window_extent(renderer)
            bb = Bbox([[bb.x0 - pad, bb.y0 - pad],
                       [bb.x1 + pad, bb.y1 + pad]])
            area = (bb.x1 - bb.x0) * (bb.y1 - bb.y0)
            cost = sum(_overlap(bb, o) for o in taken)
            cost += 3.0 * (area - _overlap(bb, frame))
            if best is None or cost < best[0]:
                best = (cost, along, off, ha, va, plotly_pos, bb)
            if cost == 0:
                break
        _c, along, off, ha, va, plotly_pos, bb = best
        t.xy = (along, v) if kind == "h" else (v, along)
        t.xyann = off
        t.set_ha(ha)
        t.set_va(va)
        taken.append(bb)
        spots[(kind, k)] = plotly_pos
    return spots


def _place_point_names(fig, ax, drawn, fixed_texts=(), leaders=(),
                       fixed_shapes=(), ref_lines=()) -> list:
    """Write each point's name beside it where it covers the fewest other
    names, points, lines, reference lines, hatched areas, notes and
    reference-line labels, and stays inside the plot frame (eight spots
    tried in turn), and record the spot so the interactive twin puts the
    name in the same place. ``drawn`` is ``[(series, xs_plotted)]``;
    ``fixed_texts`` are lettering that stays where it is (a note by its
    text or box), ``leaders`` ``[(note artist, point)]`` the notes' leader
    arrows, ``fixed_shapes`` other things drawn that stay (a plan's scale
    bar and north arrow) and ``ref_lines`` ``[("h" | "v", value)]`` the
    reference lines. Returns ``[(artist, series, name)]`` for the names
    written."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()                       # settle limits (equal aspect)
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    pad = 3.0
    taken = [_note_box(t, renderer) for t in fixed_texts]
    for shape in fixed_shapes:
        try:
            taken.append(shape.get_window_extent(renderer))
        except Exception:                               # noqa: BLE001
            pass
    # a line through a name (a series, a reference line, a leader) costs
    # more than its thin boxes' area: it cuts the letters, where a name
    # grazing a marker's box only sits close (_LINE_COST)
    strokes = []
    for artist, point in leaders:
        strokes += _leader_boxes(fig, ax, artist, point, renderer)
    strokes += _ref_line_boxes(fig, ax, ref_lines)
    taken += _data_obstacles(fig, ax, drawn, strokes)
    hatched = _hatched_areas(ax, drawn)
    written = []
    for s, xs in drawn:
        if not s["labels"]:
            continue
        spots = []
        for x, y, text in zip(xs, s["y"], s["labels"]):
            if not text:
                spots.append("top center")
                continue
            best = None
            for dx, dy, ha, va, plotly_pos in _LABEL_SPOTS:
                t = ax.annotate(text, (x, y), xytext=(dx, dy),
                                textcoords="offset points", ha=ha, va=va,
                                fontsize=_LABEL_SIZE, color=_TEXT_COLOR,
                                zorder=6)
                letters = t.get_window_extent(renderer)
                bb = Bbox([[letters.x0 - pad, letters.y0 - pad],
                           [letters.x1 + pad, letters.y1 + pad]])
                cost = sum(_overlap(bb, o) for o in taken + strokes)
                cost += _LINE_COST * sum(_overlap(letters, o)
                                         for o in strokes)
                on_hatch = _hatch_cost(bb, hatched)
                cost += on_hatch
                # a name outside the plot frame runs into the axis numbers
                # or off the figure: costlier than touching another name
                area = (bb.x1 - bb.x0) * (bb.y1 - bb.y0)
                cost += 3.0 * (area - _overlap(bb, frame))
                if best is None or cost < best[0]:
                    if best is not None:
                        best[1].remove()
                    best = (cost, t, bb, plotly_pos, on_hatch)
                else:
                    t.remove()
                if cost == 0:
                    break
            if best[4]:
                # a name on hatching (a point inside a hatched zone): a thin
                # ring of the background round its letters, so the hatch
                # stops short of them, as on a drawing (the PNG only, as a
                # note's halo)
                from matplotlib import patheffects
                best[1].set_path_effects([patheffects.withStroke(
                    linewidth=2.5, foreground=ax.get_facecolor())])
            taken.append(best[2])
            spots.append(best[3])
            written.append((best[1], s, text))
        s["label_spots"] = spots
    return written


def _short(text: str, n: int = 30) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[:n - 3] + "..."


def _lettering_names(ax, notes, artists, placed=()) -> dict:
    """What the warnings call the lettering a note can meet, by ``id``:
    another note, the title, an axis title, the axis numbers, the legend,
    and the lettering ``plot_data`` places itself (``placed``: ``[(artist,
    name)]``: reference-line names, the plan's furniture, point names)."""
    names = {id(a): f"annotations[{n['index']}]"
             for a, n in zip(artists, notes)}
    names[id(ax.title)] = "the title"
    names[id(ax.xaxis.label)] = "the x-axis title"
    names[id(ax.yaxis.label)] = "the y-axis title"
    for axis in (ax.xaxis, ax.yaxis):
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            names[id(tick.label1)] = names[id(tick.label2)] = \
                "the axis numbers"
    legend = ax.get_legend()
    if legend is not None:
        for a in legend.findobj():
            names[id(a)] = "the legend"
    for a, name in placed:
        names[id(a)] = name
    return names


#: What a warning on lettering plot_data places itself suggests.
_PLACED_HINTS = {
    "reference": "name the line with a note placed clear instead",
    "furniture": "move what runs into it off the band below the site",
    "point": "no spot beside the point was clear: name it with a note and "
             "a leader, or leave the name out",
}


def _check_lettering(fig, ax, notes, artists, dpi, lines_named=None,
                     placed=()) -> list:
    """The layout check over the lettering on the figure that is not the
    axes' own: the notes, which stay where they were put, and what
    plot_data places itself (``placed``: ``[(artist, name, kind)]``, kind
    ``reference`` for a reference line's name, ``furniture`` for a plan's
    scale bar, north arrow and caption, ``point`` for a point's name;
    BACKLOG GM-152). By the pixels drawn, with the plotting box's own check
    (``geotech_common.layout_ink``) at the PNG's dpi: lettering that
    overlaps or runs into other lettering, that a line, markers, an edge,
    hatching or an arrow shows through (a line under the lettering's own
    box does not show), or that the figure edge cuts. And by the numbers: a
    note, or the point it marks, outside the axis limits. ``lines_named``
    names the series and reference lines by artist ``id``. One warning per
    piece of lettering found, the notes first; two pieces that meet are
    reported once, on the note when one is a note."""
    from geotech_common.layout_ink import layout_check

    placed = list(placed)
    texts = list(artists) + [a for a, _n, _k in placed]
    res = layout_check(fig, "plot_data", dpi=int(dpi), texts=texts)
    names = _lettering_names(ax, notes, artists,
                             [(a, n) for a, n, _k in placed])
    n_notes = len(notes)
    parts = {k: [] for k in range(len(texts))}
    met = {k: [] for k in range(len(texts))}
    index = {id(a): k for k, a in enumerate(texts)}
    for k, words, other in res.get("text_overlaps", []):
        j = index.get(id(other))
        # a note and a piece plot_data placed: on the note; two pieces it
        # placed: on the first (two notes: on each, as before)
        if j is not None and (j < n_notes) != (k < n_notes):
            if k >= n_notes:
                continue
        elif j is not None and j < k and k >= n_notes:
            continue
        met[k].append(names.get(id(other), f"'{_short(words)}'"))
    for k, hits in met.items():
        if hits:
            parts[k].append("overlaps " + ", ".join(dict.fromkeys(hits)))
    for k, what, by in res.get("text_on_a_line", []):
        if id(by) in names and names[id(by)].startswith("annotations["):
            what = f"the leader of {names[id(by)]}"
        elif id(by) in (lines_named or {}):
            what = (f"the hatching of {lines_named[id(by)]}"
                    if what.startswith("the hatching") or what == "hatching"
                    else lines_named[id(by)])
        if not what.endswith("drawn over it"):
            what = f"crossed by {what}"
        if k < n_notes:
            hint = ("move it" if notes[k]["box"]
                    else "move it, or give it box: true")
            if "hatching" in what and not notes[k]["box"]:
                hint = "move it, or give it box: true or halo: true"
            parts[k].append(f"{what} ({hint})")
        else:
            parts[k].append(what)
    for k, where in res.get("text_cut_off", []):
        parts[k].append(f"cut off by {where}")

    xl, yl = sorted(ax.get_xlim()), sorted(ax.get_ylim())

    def inside(x, y):
        return xl[0] <= x <= xl[1] and yl[0] <= y <= yl[1]

    found = []
    if res.get("unchecked"):
        found.append(f"the lettering's overlaps were not checked "
                     f"({res['unchecked']})")
    for k, n in enumerate(notes):
        if n["coords"] == "data" and not inside(n["x"], n["y"]):
            parts[k].append("it stands outside the plot, beyond the axis "
                            "limits (set xlim/ylim to take it in, or give "
                            "it in axes fractions)")
        if n["point"] is not None and not inside(*n["point"]):
            parts[k].append("the point it marks is outside the plot")
        if parts[k]:
            found.append(f"annotations[{n['index']}] "
                         f"'{_short(n['lines'][0])}': " + "; ".join(parts[k]))
    for k, (_a, name, kind) in enumerate(placed, start=n_notes):
        if parts[k]:
            found.append(f"{name}: " + "; ".join(parts[k])
                         + f" ({_PLACED_HINTS[kind]})")
    return found


def render_data_plot(series, *, title: str = "", xlabel: str = "",
                     ylabel: str = "", depth_axis: bool = False,
                     logx: bool = False, logy: bool = False,
                     hlines=None, vlines=None, grid: bool = True,
                     legend: Optional[bool] = None, dpi: int = 150,
                     width_in: float = 6.5, height_in: Optional[float] = None,
                     interactive: bool = True, equal_aspect: bool = False,
                     xlim=None, ylim=None,
                     legend_position: Optional[str] = None,
                     plan_view: bool = False,
                     north_deg: Optional[float] = None,
                     scale_units: str = "m",
                     x_on_top: bool = False,
                     annotations=None) -> DataPlotResult:
    """Render ``series`` to a PNG (and its interactive twin).

    Each series is ``{x, y, label, style, color}`` plus, optionally,
    ``marker`` (a name in :data:`MARKERS`), ``marker_size`` (points),
    ``marker_edge`` (outline colour), ``marker_open``, ``line_dash``,
    ``line_width``, ``fill`` / ``fill_opacity`` (style ``polygon``),
    ``labels`` (a name beside each point), ``hover`` (text shown on hover in
    the interactive chart) and ``in_legend``. x given as names (cases,
    scenarios) puts the cases along the x axis in the order first seen.

    ``depth_axis=True`` inverts the y axis (depth increases downward) and puts
    the x axis at the top — the boring-log convention. ``x_on_top=True`` puts
    the x axis at the top alone: a profile against elevation read like a
    boring log (BACKLOG GM-106). ``xlim`` / ``ylim`` are ``[start, end]`` in
    the order given: ``[100, 0.001]`` runs the axis from high to low, on a
    log axis too (with ``depth_axis`` the direction of y is its own), and
    the result's ``reversed_axes`` and summary say so. ``equal_aspect=True``
    draws one unit across as long as one unit up (a plan view, a map, a
    section to scale) and, unless ``height_in`` is given, sizes the figure to
    the data's own proportions. ``hlines`` / ``vlines`` are
    ``[{"value": v, "label": "..."}]`` reference lines (a water table, an
    allowable value, a design load), each label written on its line.
    ``legend`` defaults to "when more than one series is in it";
    ``legend_position`` is
    ``auto`` (outside a drawing to scale, inside otherwise), ``inside``,
    ``outside`` or ``below``.

    ``plan_view=True`` draws a plan as a drawing, not a chart (owner,
    2026-10-15): drawn to scale, no axes, axis numbers or grid, a scale bar
    in ``scale_units`` with the x label (the coordinate system) under it, and
    a north arrow when ``north_deg`` gives the direction of north on the page
    (degrees clockwise from straight up, as the plan the data came from
    or the project's other plans show it; drawn straight up, with a note,
    when not given).

    ``annotations`` places free text where the caller puts it (BACKLOG
    GM-109): ``[{"text", "x", "y"}]`` in data units (a case's name for x on
    a case axis), or with ``coords="axes"`` in fractions of the plot (0, 0
    bottom left, 1, 1 top right); optional ``point`` ([x, y] in data units:
    the point the note marks, drawn as a leader arrow from the text),
    ``align`` (the side of the text at x, y: center, left, right, top,
    bottom, or one of each), ``box`` (true, or a colour: a plain box behind
    the text), ``halo`` (true, or a colour: a thin ring of the background
    round the letters, so hatching or a grid behind them stops short; the
    PNG only), ``color`` and ``font_size``. A line break is a new line or
    ``<br>``. Notes are never moved: point names keep clear of them, and
    the warnings say when a note overlaps other lettering, a line runs
    through it or it stands outside the plot. Both renderings draw them.

    ``interactive=True`` (the default) also builds the Plotly twin of the same
    figure on ``result.figure`` — what the chat renders as a real zoom/hover
    chart, while the PNG stays the thing a PDF report embeds. Building it can
    never cost the PNG: a failure is recorded as a warning and leaves
    ``figure`` as ``None``.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    if not isinstance(series, (list, tuple)) or not series:
        raise ValueError("series must be a non-empty list of {x, y, ...} dicts")
    warnings: List[str] = []
    cleaned = [_clean_series(s, i, warnings) for i, s in enumerate(series)]
    categories = _categories(cleaned)
    notes = _clean_notes(annotations, categories, bool(logx), bool(logy),
                         warnings)
    plan_view = bool(plan_view)
    if plan_view and (categories is not None or logx or logy
                      or depth_axis):
        warnings.append("plan_view ignored: a plan needs numeric x and y "
                        "on linear axes")
        plan_view = False
    if plan_view:
        equal_aspect, grid = True, False
        from geotech_common.plan_furniture import north_or_default
        _angle, north_note = north_or_default(north_deg)
        if north_note:
            warnings.append(north_note)
    equal_aspect = bool(equal_aspect)
    if equal_aspect and categories is not None:
        warnings.append("equal_aspect ignored: x names cases, not a length")
        equal_aspect = False
    if equal_aspect and (logx or logy):
        warnings.append("equal_aspect ignored on a log axis")
        equal_aspect = False
    xl, yl, reversed_axes = _axis_limits(xlim, ylim, logx, logy,
                                         bool(depth_axis), warnings)
    x_on_top = bool(x_on_top) and not plan_view
    position = _resolve_legend_position(legend_position, equal_aspect)
    shown = [s for s in cleaned if s["in_legend"]]
    show_legend = (legend if legend is not None else len(shown) > 1)
    if height_in is None:
        height_in = (_scale_height(cleaned, float(width_in),
                                   show_legend and position == "outside",
                                   plan_view)
                     if equal_aspect else 4.5)

    fig, ax = plt.subplots(figsize=(float(width_in), float(height_in)))
    drawn = []
    lines_named = {}            # the notes' check: artist id -> its line
    order = sorted(range(len(cleaned)),
                   key=lambda i: cleaned[i]["style"] != "polygon")
    for i in order:
        s = cleaned[i]
        before = {id(a) for a in ax.get_children()}
        xs = ([categories.index(v) for v in s["x"]] if categories is not None
              else s["x"])
        color = s["color"] or _PALETTE[i % len(_PALETTE)]
        label = s["label"] if s["in_legend"] else "_nolegend_"
        dash = DASHES[s["line_dash"]][0]
        kw = {"color": color, "label": label, "linewidth": s["line_width"],
              "zorder": 3}
        mk = {"marker": MARKERS[s["marker"]][0],
              "markersize": _marker_size(s)}
        if s["marker_edge"]:
            mk.update(markeredgecolor=s["marker_edge"], markeredgewidth=0.8)
        if s["marker_open"]:
            mk.update(markerfacecolor="none")
        if s["style"] == "markers":
            ax.plot(xs, s["y"], linestyle="none", **mk, **kw)
        elif s["style"] == "line":
            ax.plot(xs, s["y"], linestyle=dash, **kw)
        elif s["style"] == "step":
            ax.step(xs, s["y"], where="post", linestyle=dash, **kw)
        elif s["style"] == "polygon":
            if s["fill"]:
                fill_color = color if s["fill"] is True else s["fill"]
                ax.fill(xs, s["y"], facecolor=fill_color,
                        alpha=s["fill_opacity"], edgecolor="none", zorder=1)
            if s["hatch"]:
                # the hatch takes the outline's colour (a zone drawn over
                # what lies under it, as on a drawing); the legend shows a
                # hatched swatch, not a line
                ax.fill(xs, s["y"], facecolor="none", edgecolor=color,
                        hatch=s["hatch"] * 3, linewidth=0.8, zorder=1.5,
                        label=kw["label"])
                kw["label"] = "_nolegend_"
            kw["zorder"] = 2
            ax.plot(xs, s["y"], linestyle=dash, **kw)
        else:
            ax.plot(xs, s["y"], linestyle=dash, **mk, **kw)
        drawn.append((s, xs))
        if s["label"]:
            lines_named.update((id(a), f"'{_short(s['label'])}'")
                               for a in ax.get_children()
                               if id(a) not in before)

    # Reference lines are named on the line itself (labels on the data,
    # not in a legend the reader has to match).
    ref_texts, ref_lines, placed, ref_names = [], [], [], []
    for spec in (hlines or []):
        v, lab = _ref(spec)
        ref = ax.axhline(v, color=_REF_COLOR, linestyle="--",
                         linewidth=_REF_WIDTH, label="_nolegend_")
        lines_named[id(ref)] = f"the reference line at {v:g}"
        ref_lines.append(("h", v))
        if lab:
            ref_texts.append(ax.annotate(
                lab, xy=(1.0, v), xycoords=("axes fraction", "data"),
                xytext=(-3, 2), textcoords="offset points", ha="right",
                va="bottom", fontsize=_LABEL_SIZE, color=_REF_COLOR,
                zorder=5))
            placed.append((ref_texts[-1], f"the name of the reference "
                           f"line at {v:g} ('{_short(lab)}')", "reference"))
            ref_names.append((ref_texts[-1], "h", v, len(ref_lines) - 1))
    for spec in (vlines or []):
        v, lab = _ref(spec)
        ref = ax.axvline(v, color=_REF_COLOR, linestyle=":",
                         linewidth=_REF_WIDTH, label="_nolegend_")
        lines_named[id(ref)] = f"the reference line at {v:g}"
        ref_lines.append(("v", v))
        if lab:
            ref_texts.append(ax.annotate(
                lab, xy=(v, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -3), textcoords="offset points", ha="left",
                va="top", fontsize=_LABEL_SIZE, color=_REF_COLOR, zorder=5))
            placed.append((ref_texts[-1], f"the name of the reference "
                           f"line at {v:g} ('{_short(lab)}')", "reference"))
            ref_names.append((ref_texts[-1], "v", v,
                              sum(1 for k, _v in ref_lines if k == "v") - 1))
    # Notes go where they were put, before the legend, so a legend placed
    # "best" keeps clear of them too.
    note_artists = _draw_notes(ax, notes)

    if categories is not None:
        ax.set_xticks(range(len(categories)))
        long_names = max(len(c) for c in categories) > 6
        ax.set_xticklabels(categories, rotation=30 if long_names else 0,
                           ha="right" if long_names else "center",
                           fontsize=8.5)
        ax.set_xlim(-0.5, len(categories) - 0.5)
    if logx or logy:
        from matplotlib.ticker import FuncFormatter
        plain = FuncFormatter(lambda v, _pos: f"{v:g}")   # 1, 10, 100
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plain)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plain)
    if xl:
        ax.set_xlim(*xl)
    if yl:
        ax.set_ylim(*yl)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box" if (xl and yl)
                      else "datalim")
    if depth_axis:
        ax.invert_yaxis()
    if depth_axis or x_on_top:
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
    if grid:
        ax.grid(True, color=_GRID_COLOR, linewidth=_GRID_WIDTH)
        ax.set_axisbelow(True)
    for side in ("top", "right"):
        if not ((depth_axis or x_on_top) and side == "top"):
            ax.spines[side].set_visible(False)
    if plan_view:
        ax.set_axis_off()               # the scale bar carries the scale
    else:
        ax.set_xlabel(xlabel, fontsize=10, color=_TEXT_COLOR)
        ax.set_ylabel(ylabel, fontsize=10, color=_TEXT_COLOR)
    if title:
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10,
                     color=_TEXT_COLOR,
                     loc="left" if plan_view else "center")
    if show_legend:
        if position == "outside":
            ax.legend(fontsize=8.5, frameon=False, loc="upper left",
                      bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        elif position == "inside":
            ax.legend(fontsize=8.5, frameon=True, framealpha=0.85,
                      edgecolor="none")
    fig.tight_layout()
    if show_legend and position == "below":
        # Under everything the x axis takes up (slanted case names, the
        # axis title), not at a fixed offset that the labels can run into.
        fig.canvas.draw()
        bottom = ax.get_tightbbox(fig.canvas.get_renderer()).y0
        y_axes = ax.transAxes.inverted().transform((0.0, bottom))[1]
        ax.legend(fontsize=8.5, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, y_axes - 0.02),
                  ncol=min(3, max(1, len(shown))))
    ref_spots = {}
    if ref_names:
        ref_spots = _place_ref_names(fig, ax, ref_names, drawn, note_artists,
                                     [(a, n["point"]) for a, n in
                                      zip(note_artists, notes) if n["point"]])
    plan_lay, furniture = None, []
    if plan_view:
        fig.canvas.draw()               # settle the layout
        plan_lay, plan_texts, furniture = _draw_plan_furniture(
            ax, cleaned, scale_units, north_deg, xlabel)
        ref_texts += plan_texts
        for a in furniture:     # the bar's segments, the arrow
            lines_named[id(a)] = ("the north arrow"
                                  if a.__class__.__name__ == "Annotation"
                                  else "the scale bar")
        # the bar's two end names, the coordinate system when given, the N
        for n, t in enumerate(plan_texts):
            words = " ".join(t.get_text().split())
            what = (f"the scale bar's '{words}'" if n < 2
                    else "the north arrow's N" if n == len(plan_texts) - 1
                    else f"the coordinate system '{_short(words)}'")
            placed.append((t, what, "furniture"))
    if any(s["labels"] for s in cleaned):
        index_of = {id(s): i for i, s in enumerate(cleaned)}
        for t, s, text in _place_point_names(
                fig, ax, drawn, ref_texts + note_artists,
                [(a, n["point"]) for a, n in zip(note_artists, notes)
                 if n["point"]], furniture, ref_lines):
            placed.append((t, f"series[{index_of[id(s)]}] point name "
                              f"'{_short(text)}'", "point"))
    view = None
    if notes or placed:
        # the notes, the reference lines' names, the plan's furniture and
        # the point names (BACKLOG GM-152)
        try:
            warnings += _check_lettering(fig, ax, notes, note_artists, dpi,
                                         lines_named, placed)
        except Exception as exc:                        # noqa: BLE001
            warnings.append("the lettering's layout was not checked "
                            f"({type(exc).__name__}: {exc})")
    if notes:
        view = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))

    png = figure_to_png(fig, dpi=int(dpi))
    w_px, h_px = int(fig.get_size_inches()[0] * dpi), \
        int(fig.get_size_inches()[1] * dpi)
    plt.close(fig)
    try:                                    # exact pixel size if PIL is around
        from PIL import Image
        import io
        with Image.open(io.BytesIO(png)) as im:
            w_px, h_px = im.size
    except Exception:                       # noqa: BLE001
        pass

    figure = None
    if interactive:                 # the PNG is already safely in hand here
        try:
            figure = build_data_plot_figure(
                cleaned, title=title or "", xlabel=xlabel or "",
                ylabel=ylabel or "", depth_axis=bool(depth_axis),
                logx=bool(logx), logy=bool(logy), hlines=hlines,
                vlines=vlines, grid=bool(grid), legend=legend,
                equal_aspect=equal_aspect, xlim=xlim, ylim=ylim,
                legend_position=legend_position, plan_view=plan_view,
                north_deg=north_deg, scale_units=scale_units,
                x_on_top=x_on_top, annotations=notes, view=view,
                plan_layout=plan_lay, ref_spots=ref_spots)
        except Exception as exc:                        # noqa: BLE001
            warnings.append("interactive chart not built "
                            f"({type(exc).__name__}: {exc}); the PNG is fine")

    resolved = []
    for s in cleaned:
        row = {"label": s["label"], "n": len(s["x"]), "style": s["style"],
               "y_range": [min(s["y"]), max(s["y"])]}
        if s["categorical"]:
            row["x_categories"] = list(dict.fromkeys(s["x"]))
        else:
            row["x_range"] = [min(s["x"]), max(s["x"])]
        if s["labels"]:
            row["points_named"] = sum(1 for t in s["labels"] if t)
        resolved.append(row)
    return DataPlotResult(
        series=resolved,
        image_base64=base64.b64encode(png).decode("ascii"),
        png_bytes=png, width_px=w_px, height_px=h_px,
        title=title or "", xlabel=xlabel or "", ylabel=ylabel or "",
        depth_axis=bool(depth_axis), warnings=warnings, figure=figure,
        equal_aspect=equal_aspect, plan_view=plan_view,
        x_on_top=bool(depth_axis) or x_on_top,
        reversed_axes=reversed_axes,
        annotations=[n["echo"] for n in notes])


def _ref(spec) -> tuple:
    if isinstance(spec, dict):
        try:
            return float(spec.get("value")), str(spec.get("label") or "")
        except (TypeError, ValueError):
            raise ValueError(f"reference line needs a numeric 'value': {spec}")
    return float(spec), ""


__all__ = ["render_data_plot", "build_data_plot_figure", "DataPlotResult",
           "STYLES", "MARKERS", "DASHES", "HATCHES", "LEGEND_POSITIONS",
           "ANNOTATION_KEYS", "ANNOTATION_COORDS"]

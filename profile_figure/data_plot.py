"""Generic data plot — one or more (x, y) series rendered to a PNG.

The "plots of data" half of the owner's 2026-09-11 feedback (calc packages
lacking figures). Every analysis module can produce numbers; until now only a
module with a canned calc package could turn them into a chart, and the agent
has no code-execution tool to improvise one. This is the small, general,
matplotlib-backed plot the agent CAN call: SPT/CPT vs depth, settlement vs
time, load vs displacement, a sensitivity sweep, a method comparison, and
(since 2026-10-15) a plan view drawn to scale: outlines with a fill, points
named beside them, hover text, separate cases on a case axis.

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
#: plan_view: the margins, as fractions of the site's height, below it (the
#: scale bar and its caption) and above it (the north arrow)
_PLAN_BAND_BELOW = 0.26
_PLAN_BAND_ABOVE = 0.05
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
    warnings : list of str
        Non-fatal notes (a dropped non-finite point, an empty label).
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

    def summary(self) -> str:
        parts = [f"'{self.title}'" if self.title else "data plot",
                 f"{len(self.series)} series"]
        for s in self.series:
            parts.append(f"{s['label']} ({s['n']} pts)")
        if self.depth_axis:
            parts.append("depth axis (down)")
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
    if spec is None:
        return None
    try:
        lo, hi = (float(v) for v in spec)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be [min, max]")
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo == hi:
        raise ValueError(f"{what} must be two different finite numbers")
    return (min(lo, hi), max(lo, hi))


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
                           scale_units: str = "m"):
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
    and moves the x axis to the top, ``logx``/``logy`` set log axes, the
    styles map to lines / markers / lines+markers / step (``line_shape="hv"``,
    matplotlib's ``where="post"``) / a closed outline (``fill="toself"``),
    ``equal_aspect`` ties the y scale to the x scale, and reference lines are
    drawn in the same color, width and dash. One deliberate difference: a
    labelled reference line becomes a chart ANNOTATION here (matplotlib puts
    it in the legend) — Plotly shapes carry no legend entry.
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

    for spec in (hlines or []):
        v, lab = _ref(spec)
        pos = _axis_coord(v, logy)
        if pos is not None:
            fig.add_hline(y=pos, line_color=_REF_COLOR, line_width=_REF_WIDTH,
                          line_dash="dash",
                          **({"annotation_text": lab,
                              "annotation_font": {"size": 10,
                                                  "color": _REF_COLOR}}
                             if lab else {}))
    for spec in (vlines or []):
        v, lab = _ref(spec)
        pos = _axis_coord(v, logx)
        if pos is not None:
            fig.add_vline(x=pos, line_color=_REF_COLOR, line_width=_REF_WIDTH,
                          line_dash="dot",
                          **({"annotation_text": lab,
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
    xl, yl = _limits(xlim, "xlim"), _limits(ylim, "ylim")
    if xl and not logx:
        fig.update_xaxes(range=list(xl))
    if depth_axis:                          # boring-log convention, as the PNG
        if yl and not logy:
            fig.update_yaxes(range=[yl[1], yl[0]])
        else:
            fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(side="top")
    elif yl and not logy:
        fig.update_yaxes(range=list(yl))
    if equal_aspect or plan_view:
        # one unit up as long as one unit across: a plan or a section to scale
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    if plan_view:
        # a drawing, not a chart: no axes or grid; a scale bar and north
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        if title:
            fig.update_layout(title={"x": 0.0, "xanchor": "left"})
        _plotly_plan_furniture(fig, cleaned_series, scale_units, north_deg,
                               xlabel)
    return fig


from geotech_common.plan_furniture import nice_length as _nice_length


def _plan_layout(cleaned) -> dict:
    """Where plan_view's furniture goes, in data units, for both renderings:
    a band below the site holds the scale bar and, at its right, the north
    arrow (drafting practice keeps them together); the coordinate system is
    named under them. Nothing is drawn on the site itself."""
    xs = [x for s in cleaned for x in s["x"]]
    ys = [y for s in cleaned for y in s["y"]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (x1 - x0) or 1.0, (y1 - y0) or 1.0
    length = _nice_length(dx)
    bar_end = x1 - 0.09 * dx
    return {"xlim": (x0 - 0.04 * dx, x1 + 0.04 * dx),
            "ylim": (y0 - _PLAN_BAND_BELOW * dy, y1 + _PLAN_BAND_ABOVE * dy),
            "length": length, "bx": bar_end - length,
            "by": y0 - 0.12 * dy, "h": 0.014 * dy,
            "arrow": (x1 - 0.02 * dx, y0 - 0.15 * dy),
            "caption_xy": (x1, y0 - 0.12 * dy)}


def _draw_plan_furniture(ax, cleaned, units, north_deg, caption):
    """The scale bar, the north arrow beside it and the coordinate system
    under them, in a band below the site
    (:mod:`geotech_common.plan_furniture`). Returns the text artists, so
    point names keep clear of them."""
    from geotech_common import plan_furniture as pf

    lay = _plan_layout(cleaned)
    ax.set_autoscale_on(False)
    ax.set_xlim(*lay["xlim"])
    ax.set_ylim(*lay["ylim"])
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
    return texts


def _plotly_plan_furniture(fig, cleaned, units, north_deg, caption):
    """The Plotly twin's scale bar, north arrow and caption (same layout)."""
    from geotech_common import plan_furniture as pf

    lay = _plan_layout(cleaned)
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
    if plan_view:
        span_y *= 1.0 + _PLAN_BAND_BELOW + _PLAN_BAND_ABOVE
    axes_w = max(width_in - 1.3 - (1.9 if legend_out else 0.0), 2.0)
    return min(max(axes_w * span_y / span_x + 1.2, 3.0), 10.0)


def _overlap(a, b) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if (w > 0 and h > 0) else 0.0


def _place_point_names(fig, ax, drawn, fixed_texts=()) -> int:
    """Write each point's name beside it where it covers the fewest other
    names, points and reference-line labels, and stays inside the plot frame
    (eight spots tried in turn), and record the spot so the interactive twin
    puts the name in the same place. ``drawn`` is ``[(series,
    xs_plotted)]``. Returns how many names were written."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()                       # settle limits (equal aspect)
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    frame = ax.get_window_extent(renderer)
    pad = 3.0
    taken = [t.get_window_extent(renderer) for t in fixed_texts]
    for s, xs in drawn:                     # every point is an obstacle
        size_px = _marker_size(s) * fig.dpi / 72.0
        for x, y in zip(xs, s["y"]):
            px, py = to_px((x, y))
            taken.append(Bbox([[px - size_px, py - size_px],
                               [px + size_px, py + size_px]]))
        if s["style"] == "markers":
            continue
        # ... and so is every line, sampled every few pixels along it
        pts = [to_px((x, y)) for x, y in zip(xs, s["y"])]
        half = max(s["line_width"] * fig.dpi / 72.0, 1.5)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            n = int(min(max(math.hypot(x1 - x0, y1 - y0) / 6.0, 1), 200))
            for k in range(n + 1):
                cx = x0 + (x1 - x0) * k / n
                cy = y0 + (y1 - y0) * k / n
                taken.append(Bbox([[cx - half, cy - half],
                                   [cx + half, cy + half]]))
    written = 0
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
                bb = t.get_window_extent(renderer)
                bb = Bbox([[bb.x0 - pad, bb.y0 - pad],
                           [bb.x1 + pad, bb.y1 + pad]])
                cost = sum(_overlap(bb, o) for o in taken)
                # a name outside the plot frame runs into the axis numbers
                # or off the figure: costlier than touching another name
                area = (bb.x1 - bb.x0) * (bb.y1 - bb.y0)
                cost += 3.0 * (area - _overlap(bb, frame))
                if best is None or cost < best[0]:
                    if best is not None:
                        best[1].remove()
                    best = (cost, t, bb, plotly_pos)
                else:
                    t.remove()
                if cost == 0:
                    break
            taken.append(best[2])
            spots.append(best[3])
            written += 1
        s["label_spots"] = spots
    return written


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
                     scale_units: str = "m") -> DataPlotResult:
    """Render ``series`` to a PNG (and its interactive twin).

    Each series is ``{x, y, label, style, color}`` plus, optionally,
    ``marker`` (a name in :data:`MARKERS`), ``marker_size`` (points),
    ``marker_edge`` (outline colour), ``marker_open``, ``line_dash``,
    ``line_width``, ``fill`` / ``fill_opacity`` (style ``polygon``),
    ``labels`` (a name beside each point), ``hover`` (text shown on hover in
    the interactive chart) and ``in_legend``. x given as names (cases,
    scenarios) puts the cases along the x axis in the order first seen.

    ``depth_axis=True`` inverts the y axis (depth increases downward) and puts
    the x axis at the top — the boring-log convention. ``equal_aspect=True``
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
    xl, yl = _limits(xlim, "xlim"), _limits(ylim, "ylim")
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
    order = sorted(range(len(cleaned)),
                   key=lambda i: cleaned[i]["style"] != "polygon")
    for i in order:
        s = cleaned[i]
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

    # Reference lines are named on the line itself (labels on the data,
    # not in a legend the reader has to match).
    ref_texts = []
    for spec in (hlines or []):
        v, lab = _ref(spec)
        ax.axhline(v, color=_REF_COLOR, linestyle="--", linewidth=_REF_WIDTH,
                   label="_nolegend_")
        if lab:
            ref_texts.append(ax.annotate(
                lab, xy=(1.0, v), xycoords=("axes fraction", "data"),
                xytext=(-3, 2), textcoords="offset points", ha="right",
                va="bottom", fontsize=_LABEL_SIZE, color=_REF_COLOR,
                zorder=5))
    for spec in (vlines or []):
        v, lab = _ref(spec)
        ax.axvline(v, color=_REF_COLOR, linestyle=":", linewidth=_REF_WIDTH,
                   label="_nolegend_")
        if lab:
            ref_texts.append(ax.annotate(
                lab, xy=(v, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -3), textcoords="offset points", ha="left",
                va="top", fontsize=_LABEL_SIZE, color=_REF_COLOR, zorder=5))

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
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
    if grid:
        ax.grid(True, color=_GRID_COLOR, linewidth=_GRID_WIDTH)
        ax.set_axisbelow(True)
    for side in ("top", "right"):
        if not (depth_axis and side == "top"):
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
    if plan_view:
        fig.canvas.draw()               # settle the limits (equal aspect)
        ref_texts += _draw_plan_furniture(ax, cleaned, scale_units,
                                          north_deg, xlabel)
    if any(s["labels"] for s in cleaned):
        _place_point_names(fig, ax, drawn, ref_texts)

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
                north_deg=north_deg, scale_units=scale_units)
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
        equal_aspect=equal_aspect, plan_view=plan_view)


def _ref(spec) -> tuple:
    if isinstance(spec, dict):
        try:
            return float(spec.get("value")), str(spec.get("label") or "")
        except (TypeError, ValueError):
            raise ValueError(f"reference line needs a numeric 'value': {spec}")
    return float(spec), ""


__all__ = ["render_data_plot", "build_data_plot_figure", "DataPlotResult",
           "STYLES", "MARKERS", "DASHES", "HATCHES", "LEGEND_POSITIONS"]

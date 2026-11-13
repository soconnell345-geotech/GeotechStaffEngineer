"""Panels side by side on ONE vertical axis, drawn by ``plot_data`` with no
model-written code (BACKLOG GM-315).

What only plot code drew before: a wall or pile drawn to scale with its
pressure or load diagrams beside it, a soil column beside stress panels,
several profiles on one elevation (or depth) axis. :func:`render_panel_plot`
draws them from a description, ``panels``, one dict per panel, left to
right:

- ``kind``: ``chart`` (the default: an x axis of its own), ``section`` (a
  drawing to scale, one unit across as long as one unit up, no x axis: the
  shared vertical axis is its scale) or ``column`` (a soil column: no x
  axis, x runs 0 to 1 across it);
- ``series``: the panel's series, exactly as ``plot_data``'s own (a series
  may also be filled to a base line, ``fill_to_x``: a pressure diagram);
- ``title``, ``xlabel``, ``xlim``, ``vlines``, ``annotations``: as the
  figure's own, for this panel;
- ``width``: its width against the other chart panels (a section is as wide
  as its metres; a column is 0.95 in times ``width``);
- ``share_x``: a name; chart panels with the same name are drawn on one x
  scale (the same quantity: small multiples);
- ``bands``: layers across the panel, ``[{top, bottom, label, color, hatch,
  align, label_y}]``;
- ``dimensions``: dimension lines, ``[{from: [x, y], to: [x, y], text,
  offset: [dx, dy]}]``: two heads, the text beside the line, the line set
  off from the two points by ``offset`` (data units) with witness lines;
- ``arrows``: ``[{from: [x, y], to: [x, y], text, color}]``: an arrow from
  its tail to its head (a resultant, pointing the way it acts) with its
  text at the tail;
- ``grid``: true for a light grid in a chart panel where values are read
  off it (default none: the figure's own ``grid`` is not used with panels).

The figure's own ``title``, ``ylabel`` (the shared axis), ``ylim``,
``depth_axis``, ``hlines`` (drawn across every panel, named once) and
``footnote`` apply to the whole figure.

The layout is set in inches and then measured: the margins take what the
axis numbers, titles and footnote need, a section's width follows from the
axes' height, and the chart panels share the rest. The whole figure then
goes through the layout check (:func:`aec_agent_core.charts.layout_ink.
layout_check`) and what it finds is in the warnings.

The interactive twin is built where Plotly draws the same thing: chart
panels only, with no bands or dimension lines. Otherwise there is no twin,
the warnings say so, and the chat shows the PNG.
"""

from __future__ import annotations

import base64
import difflib
import math
import textwrap
from typing import List, Optional

from aec_agent_core.charts import data_plot as _dp
from aec_agent_core.charts.plotting import figure_to_png

PANEL_KINDS = ("chart", "section", "column")
PANEL_KEYS = ("kind", "title", "xlabel", "xlim", "width", "share_x",
              "series", "vlines", "annotations", "bands", "dimensions",
              "arrows", "grid")
BAND_KEYS = ("top", "bottom", "label", "color", "hatch", "align", "label_y")
DIMENSION_KEYS = ("from", "to", "text", "offset")
ARROW_KEYS = ("from", "to", "text", "color")
BAND_ALIGN = ("center", "left", "right")

#: Faint tints for layer bands given no colour, in turn.
_BAND_TINTS = ("#ece8df", "#e1e6da", "#f3ead0", "#eadfbf", "#e6e3ea")
_BAND_EDGE = "#737373"
_DIM_COLOR = "#262626"

_TITLE_PT = 9.5         #: the figure title
_PANEL_TITLE_PT = 8.0   #: a panel's title
_AXIS_PT = 8.0          #: axis titles
_TICK_PT = 7.5          #: axis numbers
_TEXT_PT = 7.5          #: band names, dimension and arrow text
_FOOT_PT = 7.5          #: the footnote

_AXES_H = 3.6           #: default height of the axes (inches)
_GAP = 0.16             #: between panels (inches)
_RIGHT = 0.08           #: right margin (inches)
_COLUMN_W = 0.95        #: a column panel's width at width = 1 (inches)
_MIN_CHART_W = 0.75     #: the narrowest chart panel drawn (inches)
_SECTION_SHARE = 0.62   #: of the room, the most sections take beside charts
_MAX_PANELS = 8

#: Top-level inputs that have no meaning with ``panels``, and where each
#: goes instead.
_NOT_WITH_PANELS = {
    "series": "give each panel its own 'series'",
    "xlabel": "give each panel its own 'xlabel'",
    "xlim": "give each panel its own 'xlim'",
    "vlines": "give each panel its own 'vlines'",
    "annotations": "give each panel its own 'annotations'",
    "logx": "panels are drawn on linear axes",
    "logy": "panels are drawn on linear axes",
    "equal_aspect": "a panel of kind 'section' is drawn to scale",
    "plan_view": "a plan is one drawing, not panels",
    "north_deg": "a plan is one drawing, not panels",
    "x_on_top": "the x axes of panels run along the bottom",
    "contour": "contours are drawn on one plot, not in panels",
}


def _unknown(item: dict, valid, where: str) -> None:
    unknown = sorted(str(k) for k in item if k not in valid)
    if unknown:
        near = [f"{k} -> {m[0]}" for k in unknown
                for m in [difflib.get_close_matches(k, valid, n=1,
                                                    cutoff=0.6)] if m]
        hint = f" Did you mean: {'; '.join(near)}?" if near else ""
        raise ValueError(f"{where} has unknown key(s) {unknown}.{hint} "
                         f"Valid keys: {sorted(valid)}.")


def _number(value, where: str) -> float:
    if not _dp._is_number(value) or not math.isfinite(float(value)):
        raise ValueError(f"{where} must be a number")
    return float(value)


def _point(value, where: str) -> tuple:
    if (isinstance(value, (str, dict)) or not hasattr(value, "__len__")
            or len(value) != 2):
        raise ValueError(f"{where} must be [x, y]")
    return (_number(value[0], f"{where} x"), _number(value[1], f"{where} y"))


def _list_of_dicts(raw, where: str, valid) -> list:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{where} must be a list of dicts")
    for k, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{where}[{k}] must be a dict")
        _unknown(item, valid, f"{where}[{k}]")
    return list(raw)


def _placed(exc: ValueError, where: str) -> ValueError:
    """A series' or note's own error, said with its panel."""
    return ValueError(f"{where}: {exc}")


def _clean_panel(raw, k: int, warnings: list) -> dict:
    where = f"panels[{k}]"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a dict (kind, title, series, ...)")
    _unknown(raw, PANEL_KEYS, where)
    kind = str(raw.get("kind") or "chart").lower().strip()
    if kind not in PANEL_KINDS:
        raise ValueError(f"{where} kind '{raw.get('kind')}' not in "
                         f"{PANEL_KINDS}")
    series_raw = raw.get("series")
    if series_raw is not None and not isinstance(series_raw, (list, tuple)):
        raise ValueError(f"{where} series must be a list of {{x, y, ...}} "
                         "dicts")
    own: List[str] = []
    try:
        series = [_dp._clean_series(s, i, own)
                  for i, s in enumerate(series_raw or [])]
        notes = _dp._clean_notes(raw.get("annotations"), None, False, False,
                                 own)
    except ValueError as exc:
        raise _placed(exc, where) from None
    warnings += [f"{where}: {w}" for w in own]
    if any(s["categorical"] for s in series):
        raise ValueError(f"{where}: a panel's x must be numbers (cases "
                         "along x are one plot, not panels)")
    bands = []
    for i, b in enumerate(_list_of_dicts(raw.get("bands"), f"{where} bands",
                                         BAND_KEYS)):
        at = f"{where} bands[{i}]"
        top, bottom = _number(b.get("top"), f"{at} top"), \
            _number(b.get("bottom"), f"{at} bottom")
        hatch = b.get("hatch")
        if hatch is not None and str(hatch) not in _dp.HATCHES:
            raise ValueError(f"{at} hatch '{hatch}' not in "
                             f"{list(_dp.HATCHES)}")
        align = str(b.get("align") or "center").lower().strip()
        if align not in BAND_ALIGN:
            raise ValueError(f"{at} align '{b.get('align')}' not in "
                             f"{BAND_ALIGN}")
        label_y = b.get("label_y")
        bands.append({
            "top": max(top, bottom), "bottom": min(top, bottom),
            "label": _dp._note_text(b.get("label"), f"{at} label"),
            "color": (_dp._note_colour(b.get("color"), i, "color",
                                       _BAND_TINTS[i % len(_BAND_TINTS)],
                                       where=at)),
            "hatch": None if hatch is None else str(hatch),
            "align": align,
            "label_y": (None if label_y is None
                        else _number(label_y, f"{at} label_y"))})
    dims = []
    for i, d in enumerate(_list_of_dicts(raw.get("dimensions"),
                                         f"{where} dimensions",
                                         DIMENSION_KEYS)):
        at = f"{where} dimensions[{i}]"
        p0, p1 = _point(d.get("from"), f"{at} from"), \
            _point(d.get("to"), f"{at} to")
        if p0 == p1:
            raise ValueError(f"{at}: 'from' and 'to' are one point")
        off = (0.0, 0.0) if d.get("offset") is None else \
            _point(d["offset"], f"{at} offset")
        dims.append({"from": p0, "to": p1, "offset": off,
                     "text": _dp._note_text(d.get("text"), f"{at} text")})
    arrows = []
    for i, a in enumerate(_list_of_dicts(raw.get("arrows"),
                                         f"{where} arrows", ARROW_KEYS)):
        at = f"{where} arrows[{i}]"
        p0, p1 = _point(a.get("from"), f"{at} from"), \
            _point(a.get("to"), f"{at} to")
        if p0 == p1:
            raise ValueError(f"{at}: 'from' and 'to' are one point")
        arrows.append({"from": p0, "to": p1,
                       "text": _dp._note_text(a.get("text"), f"{at} text"),
                       "color": _dp._note_colour(a.get("color"), i, "color",
                                                 "#1a1a1a", where=at)})
    vlines = [_dp._ref(v) for v in (raw.get("vlines") or [])]
    if kind != "chart" and vlines:
        warnings.append(f"{where}: 'vlines' applies to a chart panel; "
                        "ignored")
        vlines = []
    xl = _dp._limits(raw.get("xlim"), f"{where} xlim")
    if kind == "column":
        if xl:
            warnings.append(f"{where}: a column runs 0 to 1 across; 'xlim' "
                            "ignored")
        xl = (0.0, 1.0)
    width = raw.get("width")
    width = 1.0 if width is None else _number(width, f"{where} width")
    if not 0.2 <= width <= 5.0:
        raise ValueError(f"{where} width is a ratio between 0.2 and 5")
    share = raw.get("share_x")
    if share is not None and kind != "chart":
        warnings.append(f"{where}: 'share_x' applies to a chart panel; "
                        "ignored")
        share = None
    if not (series or bands or dims or arrows or notes):
        raise ValueError(f"{where} is empty: give it series, bands, "
                         "dimensions, arrows or annotations")
    return {"index": k, "kind": kind, "title": _dp._note_text(
                raw.get("title"), f"{where} title"),
            "xlabel": _dp._note_text(raw.get("xlabel"), f"{where} xlabel"),
            "xlim": xl,
            "width": width, "share_x": None if share is None else str(share),
            "series": series, "notes": notes, "bands": bands, "dims": dims,
            "arrows": arrows, "vlines": vlines,
            "grid": bool(raw.get("grid", False))}


def _dim_line(d) -> tuple:
    """A dimension line's two ends: the measured points moved by its
    offset."""
    (x0, y0), (x1, y1), (dx, dy) = d["from"], d["to"], d["offset"]
    return (x0 + dx, y0 + dy), (x1 + dx, y1 + dy)


def _extent(p, axis: int) -> list:
    """Every x (axis 0) or y (axis 1) value drawn in panel ``p``, notes
    left out (they are placed, never fitted)."""
    key = "x" if axis == 0 else "y"
    out = [v for s in p["series"] for v in s[key]]
    if axis == 0:
        out += [s["fill_to_x"] for s in p["series"]
                if s.get("fill_to_x") is not None]
        out += [v for v, _lab in p["vlines"]]
    else:
        out += [s["fill_to_y"] for s in p["series"]
                if s.get("fill_to_y") is not None]
        out += [v for b in p["bands"] for v in (b["top"], b["bottom"])]
    for d in p["dims"]:
        out += [pt[axis] for pt in (d["from"], d["to"]) + _dim_line(d)]
    for a in p["arrows"]:
        out += [a["from"][axis], a["to"][axis]]
    return out


def _x_view(p, values: list) -> tuple:
    """A panel's x limits when ``xlim`` does not give them: a section takes
    in what is drawn with a small margin; a chart starts at zero when every
    value is of one sign (magnitudes compare truly) with room past the
    largest."""
    if not values:
        return (0.0, 1.0)
    lo, hi = min(values), max(values)
    span = (hi - lo) or abs(hi) or 1.0
    if p["kind"] == "section":
        return (lo - 0.04 * span, hi + 0.04 * span)
    if lo >= 0:
        return (0.0, hi + 0.08 * (hi or 1.0))
    if hi <= 0:
        return (lo + 0.08 * lo, 0.0)
    return (lo - 0.06 * span, hi + 0.06 * span)


def _wrap(text: str, chars: int) -> str:
    return "\n".join(
        "\n".join(textwrap.wrap(line, max(chars, 20)) or [""])
        for line in text.split("\n"))


def _fit_text(artist, text: str, width_px: float, pt: float, dpi: float,
              renderer) -> None:
    """Break ``text`` so ``artist`` is no wider than ``width_px``."""
    chars = int(width_px / (0.5 * pt * dpi / 72.0))
    for _ in range(6):
        artist.set_text(_wrap(text, chars))
        wide = artist.get_window_extent(renderer).width
        if wide <= width_px or chars <= 20:
            return
        chars = max(20, int(chars * width_px / wide * 0.97))


def _thousands(v, _pos=None) -> str:
    return f"{v:,.10g}".replace("-", "−")


def _draw_bands(ax, p) -> list:
    """Layer bands across the panel, each named inside it."""
    from matplotlib import patheffects
    from matplotlib.transforms import blended_transform_factory
    texts = []
    across = blended_transform_factory(ax.transAxes, ax.transData)
    for b in p["bands"]:
        ax.axhspan(b["bottom"], b["top"], facecolor=b["color"],
                   edgecolor="none", zorder=0.2)
        if b["hatch"]:
            ax.axhspan(b["bottom"], b["top"], facecolor="none",
                       edgecolor=_BAND_EDGE, hatch=b["hatch"] * 2,
                       linewidth=0.0, zorder=0.3)
        for edge in (b["bottom"], b["top"]):
            ax.axhline(edge, color=_BAND_EDGE, linewidth=0.6, zorder=0.4)
        if not b["label"]:
            continue
        x = {"center": 0.5, "left": 0.05, "right": 0.95}[b["align"]]
        y = (b["label_y"] if b["label_y"] is not None
             else (b["top"] + b["bottom"]) / 2.0)
        t = ax.text(x, y, b["label"], transform=across, ha=b["align"],
                    va="center", multialignment=b["align"],
                    fontsize=_TEXT_PT, color=_dp._TEXT_COLOR, zorder=6)
        if b["hatch"]:              # the hatch stops short of the letters
            t.set_path_effects([patheffects.withStroke(
                linewidth=2.5, foreground=b["color"])])
        texts.append(t)
    return texts


def _draw_dimensions(ax, p) -> list:
    """Dimension lines: two heads, witness lines from the measured points,
    the text beside the line on the side away from what it measures."""
    texts = []
    for d in p["dims"]:
        a, b = _dim_line(d)
        dx, dy = d["offset"]
        if dx or dy:
            for (px, py), (qx, qy) in ((d["from"], a), (d["to"], b)):
                # from a little off the point to a little past the line
                ax.plot([px + 0.12 * dx, qx + 0.2 * (qx - px)],
                        [py + 0.12 * dy, qy + 0.2 * (qy - py)],
                        color=_DIM_COLOR, linewidth=0.5, zorder=4,
                        label="_nolegend_")
        ax.annotate("", xy=a, xytext=b, zorder=4,
                    arrowprops={"arrowstyle": "<|-|>", "linewidth": 0.6,
                                "color": _DIM_COLOR, "shrinkA": 0,
                                "shrinkB": 0, "mutation_scale": 7})
        if not d["text"]:
            continue
        # along the line, on the side the offset points to (below a
        # horizontal line and left of a vertical one when there is none)
        pa, pb = ax.transData.transform(a), ax.transData.transform(b)
        ang = math.degrees(math.atan2(pb[1] - pa[1], pb[0] - pa[0]))
        if ang > 90 or ang <= -90:
            ang += 180 if ang <= -90 else -180
        rad = math.radians(ang)
        normal = (-math.sin(rad), math.cos(rad))    # left of the line
        po = ax.transData.transform((a[0] + dx, a[1] + dy)) - pa
        side = po[0] * normal[0] + po[1] * normal[1]
        if not (dx or dy):
            side = -1.0 if abs(ang) < 45 else 1.0
        sign = 1.0 if side > 0 else -1.0
        t = ax.annotate(
            d["text"], xy=((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0),
            xytext=(sign * normal[0] * 3.0, sign * normal[1] * 3.0),
            textcoords="offset points", rotation=ang,
            rotation_mode="anchor", ha="center",
            va="bottom" if sign > 0 else "top", fontsize=_TEXT_PT,
            color=_DIM_COLOR, zorder=6, annotation_clip=False)
        texts.append(t)
    return texts


def _draw_arrows(ax, p) -> list:
    """Arrows from tail to head, each with its text at the tail."""
    texts = []
    for a in p["arrows"]:
        ax.annotate("", xy=a["to"], xytext=a["from"], zorder=5,
                    arrowprops={"arrowstyle": "-|>", "color": a["color"],
                                "linewidth": 1.0, "shrinkA": 0, "shrinkB": 0,
                                "mutation_scale": 9})
        if not a["text"]:
            continue
        pa, pb = ax.transData.transform(a["from"]), \
            ax.transData.transform(a["to"])
        if abs(pb[0] - pa[0]) >= abs(pb[1] - pa[1]):
            # across: above the tail, running back over the arrow
            kw = {"xytext": (0, 2.5), "va": "bottom",
                  "ha": "right" if pb[0] < pa[0] else "left"}
        else:
            kw = {"xytext": (4, 0), "ha": "left",
                  "va": "top" if pb[1] < pa[1] else "bottom"}
        texts.append(ax.annotate(
            a["text"], xy=a["from"], textcoords="offset points",
            fontsize=_TEXT_PT + 0.5, color=a["color"], zorder=6,
            annotation_clip=False, **kw))
    return texts


def _layout(fig_w, left, top, bottom, axes_h, cleaned, y_span) -> tuple:
    """Each panel's width in inches for axes ``axes_h`` high, and that
    height, lowered when the sections to scale would crowd out the charts:
    ``(widths, axes_h)``."""
    n = len(cleaned)
    room = fig_w - left - _RIGHT - _GAP * (n - 1)
    charts = [p for p in cleaned if p["kind"] == "chart"]
    fixed = sum(_COLUMN_W * p["width"] for p in cleaned
                if p["kind"] == "column")
    per_h = sum((p["view"][1] - p["view"][0]) / y_span for p in cleaned
                if p["kind"] == "section")      # section width per inch up
    if per_h:
        most = (room - fixed) * (_SECTION_SHARE if charts else 1.0)
        most -= 0 if charts else 0.0
        if axes_h * per_h > most:
            axes_h = most / per_h
    rest = room - fixed - axes_h * per_h
    total = sum(p["width"] for p in charts) or 1.0
    widths = []
    for p in cleaned:
        if p["kind"] == "section":
            widths.append(axes_h * (p["view"][1] - p["view"][0]) / y_span)
        elif p["kind"] == "column":
            widths.append(_COLUMN_W * p["width"])
        else:
            widths.append(rest * p["width"] / total)
    return widths, axes_h


def _numbers_touch(fig, ax) -> bool:
    """True when two of the x axis' numbers stand closer than a letter's
    width."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lo, hi = sorted(ax.get_xlim())
    boxes = sorted(
        (t.label1.get_window_extent(renderer)
         for t in ax.xaxis.get_major_ticks()
         if t.label1.get_visible() and t.label1.get_text()
         and lo <= t.get_loc() <= hi), key=lambda b: b.x0)
    gap = 0.06 * fig.dpi
    return any(b.x0 - a.x1 < gap for a, b in zip(boxes, boxes[1:]))


def _check(fig, dpi, names: dict, warnings: list) -> None:
    """The layout check over every piece of lettering on the figure, as
    the plotting box checks a figure drawn in code; what it finds goes in
    the warnings."""
    from aec_agent_core.charts.layout_ink import layout_check
    try:
        res = layout_check(fig, "plot_data", dpi=int(dpi))
    except Exception as exc:                            # noqa: BLE001
        warnings.append("the lettering's layout was not checked "
                        f"({type(exc).__name__}: {exc})")
        return
    if res.get("unchecked"):
        warnings.append("the lettering's overlaps were not checked "
                        f"({res['unchecked']})")
    for words, other in res.get("text_overlaps", []):
        warnings.append(f"lettering '{_dp._short(words)}' overlaps "
                        f"'{_dp._short(other)}': move one, or give its "
                        "panel more width or the axis more room (xlim, "
                        "ylim)")
    for words, what in res.get("text_on_a_line", []):
        if not str(what).endswith("drawn over it"):
            what = f"crossed by {what}"
        warnings.append(f"lettering '{_dp._short(words)}': {what} (move "
                        "it, or give a note box: true)")
    for words, where in res.get("text_cut_off", []):
        warnings.append(f"lettering '{_dp._short(words)}': cut off by "
                        f"{where}")


def render_panel_plot(panels, *, title: str = "", ylabel: str = "",
                      depth_axis: bool = False, ylim=None, hlines=None,
                      grid: bool = True, legend: Optional[bool] = None,
                      dpi: int = 150, width_in: float = 6.5,
                      height_in: Optional[float] = None,
                      interactive: bool = True, footnote: str = "",
                      not_with_panels: Optional[dict] = None):
    """Draw ``panels`` side by side on one vertical axis (the module's
    docstring describes a panel). ``not_with_panels`` is what the caller
    gave of ``plot_data``'s single-plot inputs: any of them set is refused
    by name, with where it goes instead. Returns a
    :class:`aec_agent_core.charts.data_plot.DataPlotResult`."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    for key, value in (not_with_panels or {}).items():
        if key == "legend_position":
            continue
        if value not in (None, False, "", [], ()):
            raise ValueError(f"'{key}' is not used with panels: "
                             f"{_NOT_WITH_PANELS.get(key, 'leave it out')}")
    if not isinstance(panels, (list, tuple)) or not panels:
        raise ValueError("panels must be a non-empty list of dicts, one "
                         "per panel, left to right")
    if len(panels) > _MAX_PANELS:
        raise ValueError(f"{len(panels)} panels: at most {_MAX_PANELS} fit "
                         "side by side; draw two figures")
    warnings: List[str] = []
    cleaned = [_clean_panel(p, k, warnings) for k, p in enumerate(panels)]
    title, footnote = _dp._note_text(title, "title"), \
        _dp._note_text(footnote)
    ylabel = _dp._note_text(ylabel, "ylabel")
    refs = [_dp._ref(spec) for spec in (hlines or [])]

    # the shared vertical axis
    _xl, yl, _rev = _dp._axis_limits(None, ylim, False, False,
                                     bool(depth_axis), warnings)
    if yl:
        y_view = (min(yl), max(yl))
    else:
        ys = [v for p in cleaned for v in _extent(p, 1)]
        ys += [v for v, _lab in refs]
        if not ys:
            raise ValueError("nothing gives the vertical axis its range: "
                             "give ylim")
        span = (max(ys) - min(ys)) or abs(max(ys)) or 1.0
        y_view = (min(ys) - 0.03 * span, max(ys) + 0.03 * span)
    y_span = y_view[1] - y_view[0]
    if y_span <= 0:
        raise ValueError("ylim needs two different values")

    # each panel's x view; chart panels of one share_x name on one scale
    for p in cleaned:
        p["view"] = p["xlim"] or _x_view(p, _extent(p, 0))
    for name in {p["share_x"] for p in cleaned if p["share_x"]}:
        group = [p for p in cleaned if p["share_x"] == name]
        given = [p["xlim"] for p in group if p["xlim"]]
        if len({tuple(g) for g in given}) > 1:
            warnings.append(f"panels sharing x scale '{name}' give "
                            "different xlim; the first is used")
        view = given[0] if given else (
            min(p["view"][0] for p in group), max(p["view"][1] for p in group))
        for p in group:
            p["view"] = view

    fig_w = float(width_in)
    axes_h = _AXES_H
    left, top, bottom = 0.62, 0.75, 0.9
    if height_in is not None:
        axes_h = max(float(height_in) - top - bottom, 1.0)
    widths, axes_h = _layout(fig_w, left, top, bottom, axes_h, cleaned,
                             y_span)
    fig = plt.figure(figsize=(fig_w, axes_h + top + bottom))

    axes, drawn_all, own_texts = [], [], []
    for k, p in enumerate(cleaned):
        ax = fig.add_axes([0.1 + 0.1 * k, 0.2, 0.08, 0.6],
                          sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.set_xlim(*p["view"])
        if k == 0:
            ax.set_ylim(*y_view)
            if depth_axis:
                ax.invert_yaxis()
    fig.canvas.draw()

    def place(left_in, top_in, bottom_in, heights):
        fig_h = heights + top_in + bottom_in
        fig.set_size_inches(fig_w, fig_h)
        x = left_in
        for ax, w in zip(axes, widths):
            ax.set_position([x / fig_w, bottom_in / fig_h, w / fig_w,
                             heights / fig_h])
            x += w + _GAP
        return fig_h

    fig_h = place(left, top, bottom, axes_h)
    plain = FuncFormatter(_thousands)
    names = {}
    for k, (p, ax) in enumerate(zip(cleaned, axes)):
        own_texts += _draw_bands(ax, p)
        drawn, lines_named = _dp._draw_series(ax, p["series"], None)
        ax.set_xlim(*p["view"])
        if k == 0:
            ax.set_ylim(*(y_view[::-1] if depth_axis else y_view))
        names.update(lines_named)
        for v, _lab in refs:
            ax.axhline(v, color=_dp._REF_COLOR, linestyle=(0, (5, 3)),
                       linewidth=0.8, label="_nolegend_", zorder=2.5)
        vnames = []
        for j, (v, lab) in enumerate(p["vlines"]):
            ax.axvline(v, color=_dp._REF_COLOR, linestyle=":",
                       linewidth=_dp._REF_WIDTH, label="_nolegend_")
            if lab:
                t = ax.annotate(
                    lab, xy=(v, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -3), textcoords="offset points", ha="left",
                    va="top", fontsize=_TEXT_PT, color=_dp._REF_COLOR,
                    zorder=5)
                vnames.append((t, "v", v, j))
        own_texts += _draw_dimensions(ax, p) + _draw_arrows(ax, p)
        note_artists = _dp._draw_notes(ax, p["notes"])
        p["drawn"], p["note_artists"], p["vnames"] = drawn, note_artists, \
            vnames
        drawn_all.append(drawn)

        ax.tick_params(axis="both", labelsize=_TICK_PT,
                       colors=_dp._TEXT_COLOR, length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if p["kind"] == "chart":
            ax.xaxis.set_major_formatter(plain)
            ax.xaxis.set_major_locator(MaxNLocator(
                nbins=max(3, min(7, int(widths[k] / 0.21))),
                steps=[1, 2, 2.5, 5, 10]))
            ax.set_xlabel(p["xlabel"], fontsize=_AXIS_PT,
                          color=_dp._TEXT_COLOR)
            if p["grid"]:
                ax.grid(True, color=_dp._GRID_COLOR,
                        linewidth=_dp._GRID_WIDTH)
                ax.set_axisbelow(True)
        else:
            ax.xaxis.set_visible(False)
            ax.spines["bottom"].set_visible(False)
            if p["xlabel"]:
                warnings.append(f"panels[{k}]: a {p['kind']} has no x axis; "
                                "'xlabel' ignored")
        if k == 0:
            ax.yaxis.set_major_formatter(plain)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=9,
                                                   steps=[1, 2, 5, 10]))
            ax.set_ylabel(ylabel, fontsize=_AXIS_PT + 0.5,
                          color=_dp._TEXT_COLOR)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        if p["title"]:
            ax.set_title(p["title"], fontsize=_PANEL_TITLE_PT,
                         color=_dp._TEXT_COLOR, pad=5,
                         loc="left" if p["kind"] == "section" else "center")
        shown = [s for s in p["series"] if s["in_legend"]]
        if legend and shown:
            ax.legend(fontsize=_TEXT_PT, frameon=True, framealpha=0.85,
                      edgecolor="none")

    head = fig.text(0.0, 1.0, title, fontsize=_TITLE_PT, fontweight="bold",
                    color=_dp._TEXT_COLOR, ha="left", va="top") \
        if title else None
    foot = fig.text(0.0, 0.0, footnote, fontsize=_FOOT_PT,
                    color=_dp._TEXT_COLOR, ha="left", va="bottom",
                    linespacing=1.25) if footnote else None

    # measure what the margins must hold, then set the layout for good
    given_h = None if height_in is None else float(height_in)
    for _round in range(2):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        px = fig.dpi
        above = below = 0.0
        for ax in axes:
            box = ax.get_window_extent(renderer)
            if ax.get_title() or ax.get_title(loc="left"):
                for t in (ax.title, ax._left_title):
                    if t.get_text():
                        above = max(above, t.get_window_extent(renderer).y1
                                    - box.y1)
            if ax.xaxis.get_visible():
                tb = ax.xaxis.get_tightbbox(renderer)
                if tb is not None:
                    below = max(below, box.y0 - tb.y0)
        yb = axes[0].yaxis.get_tightbbox(renderer)
        box0 = axes[0].get_window_extent(renderer)
        left = min(max((box0.x0 - yb.x0) / px + 0.05, 0.3), 1.5) \
            if yb is not None else 0.3
        text_w = (fig_w - left - _RIGHT) * px
        head_h = foot_h = 0.0
        if head is not None:
            _fit_text(head, title, text_w, _TITLE_PT, px, renderer)
            head_h = head.get_window_extent(renderer).height / px + 0.14
        if foot is not None:
            _fit_text(foot, footnote, text_w, _FOOT_PT, px, renderer)
            foot_h = foot.get_window_extent(renderer).height / px + 0.16
        top = min(above / px + head_h + 0.06, 2.5)
        bottom = min(max(below / px, 0.08) + foot_h + 0.06, 3.0)
        want_h = (_AXES_H if given_h is None
                  else max(given_h - top - bottom, 1.0))
        widths, axes_h = _layout(fig_w, left, top, bottom, want_h, cleaned,
                                 y_span)
        fig_h = place(left, top, bottom, axes_h)
        if head is not None:
            head.set_position((left / fig_w, 1.0 - 0.05 / fig_h))
        if foot is not None:
            foot.set_position((left / fig_w, 0.05 / fig_h))
    if given_h is not None and axes_h < want_h - 1e-6:
        warnings.append(
            f"height_in {given_h:g} drawn as {fig_h:.2f}: the section is "
            "to scale, and taller axes would leave its diagrams no room")
    chart_w = [w for p, w in zip(cleaned, widths) if p["kind"] == "chart"]
    if chart_w and min(chart_w) < _MIN_CHART_W:
        plt.close(fig)
        raise ValueError(
            f"the chart panels would be {min(chart_w):.2f} in wide: too "
            "narrow to read. Use fewer panels, a larger width_in, or a "
            "narrower section (its xlim)")

    # fewer axis numbers where they would run together in a narrow panel
    for k, (p, ax) in enumerate(zip(cleaned, axes)):
        if p["kind"] != "chart":
            continue
        nbins = max(3, min(7, int(widths[k] / 0.21)))
        while nbins > 2 and _numbers_touch(fig, ax):
            nbins -= 1
            ax.xaxis.set_major_locator(MaxNLocator(
                nbins=nbins, steps=[1, 2, 2.5, 5, 10]))
    # an end number that would run into the next panel's first is left out
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, nxt in zip(axes, axes[1:]):
        if not (ax.xaxis.get_visible() and nxt.xaxis.get_visible()):
            continue
        mine = [t.label1 for t in ax.xaxis.get_major_ticks()
                if t.label1.get_visible() and t.label1.get_text()
                and min(ax.get_xlim()) <= t.get_loc() <= max(ax.get_xlim())]
        theirs = [t.label1 for t in nxt.xaxis.get_major_ticks()
                  if t.label1.get_visible() and t.label1.get_text()
                  and min(nxt.get_xlim()) <= t.get_loc()
                  <= max(nxt.get_xlim())]
        if not (mine and theirs):
            continue
        last = max(mine, key=lambda t: t.get_window_extent(renderer).x1)
        first = min(theirs, key=lambda t: t.get_window_extent(renderer).x0)
        if (last.get_window_extent(renderer).x1 + 0.09 * fig.dpi
                > first.get_window_extent(renderer).x0):
            last.set_visible(False)

    # names placed last, clear of what is drawn: the reference lines'
    # (once, in the last panel), each panel's own, and the points'
    ref_lines = [("h", v) for v, _lab in refs]
    last_ax, last_p = axes[-1], cleaned[-1]
    ref_names = []
    for j, (v, lab) in enumerate(refs):
        if lab:
            t = last_ax.annotate(
                lab, xy=(1.0, v), xycoords=("axes fraction", "data"),
                xytext=(-3, 2), textcoords="offset points", ha="right",
                va="bottom", fontsize=_TEXT_PT, color=_dp._REF_COLOR,
                zorder=5)
            ref_names.append((t, "h", v, j))
    for p, ax in zip(cleaned, axes):
        leaders = [(a, n["point"]) for a, n in
                   zip(p["note_artists"], p["notes"]) if n["point"]]
        fixed = list(p["note_artists"])
        mine = p["vnames"] + (ref_names if ax is last_ax else [])
        lines_here = ref_lines + [("v", v) for v, _lab in p["vlines"]]
        if mine:
            _dp._place_ref_names(fig, ax, mine, p["drawn"], fixed, leaders,
                                 lines_here)
        if any(s["labels"] for s in p["series"]):
            _dp._place_point_names(
                fig, ax, p["drawn"], fixed + [m[0] for m in mine], leaders,
                (), lines_here)
    _check(fig, dpi, names, warnings)

    png = figure_to_png(fig, dpi=int(dpi))
    w_px, h_px = int(fig_w * dpi), int(fig_h * dpi)
    panel_rows = [{"kind": p["kind"], "title": p["title"],
                   "x_range": [float(v) for v in ax.get_xlim()],
                   "width_in": round(float(w), 2)}
                  for p, ax, w in zip(cleaned, axes, widths)]
    plt.close(fig)
    try:
        from PIL import Image
        import io
        with Image.open(io.BytesIO(png)) as im:
            w_px, h_px = im.size
    except Exception:                                   # noqa: BLE001
        pass

    figure = None
    if interactive:
        why = _no_twin(cleaned)
        if why:
            warnings.append(f"no interactive chart for this figure ({why}): "
                            "the chat shows the PNG; nothing to fix")
        else:
            try:
                figure = build_panel_figure(
                    cleaned, title=title, ylabel=ylabel,
                    depth_axis=bool(depth_axis), y_view=y_view, refs=refs,
                    grid=bool(grid), legend=legend, footnote=footnote,
                    widths=widths)
            except Exception as exc:                    # noqa: BLE001
                warnings.append("interactive chart not built "
                                f"({type(exc).__name__}: {exc}); the PNG is "
                                "fine")

    resolved, echo = [], []
    for p in cleaned:
        for s in p["series"]:
            row = {"panel": p["index"], "label": s["label"],
                   "n": len(s["x"]), "style": s["style"],
                   "x_range": [min(s["x"]), max(s["x"])],
                   "y_range": [min(s["y"]), max(s["y"])]}
            if s["labels"]:
                row["points_named"] = sum(1 for t in s["labels"] if t)
            resolved.append(row)
        echo += [dict(n["echo"], panel=p["index"]) for n in p["notes"]]
    return _dp.DataPlotResult(
        series=resolved, image_base64=base64.b64encode(png).decode("ascii"),
        png_bytes=png, width_px=w_px, height_px=h_px, title=title,
        xlabel="", ylabel=ylabel, depth_axis=bool(depth_axis),
        warnings=warnings, figure=figure,
        equal_aspect=any(p["kind"] == "section" for p in cleaned),
        annotations=echo, panels=panel_rows, footnote=footnote)


def _no_twin(cleaned) -> Optional[str]:
    """Why these panels have no interactive twin, or None when Plotly draws
    the same figure."""
    kinds = sorted({p["kind"] for p in cleaned if p["kind"] != "chart"})
    if kinds:
        return (f"a {' or '.join(kinds)} panel is drawn in the PNG "
                "only")
    if any(p["bands"] or p["dims"] for p in cleaned):
        return ("layer bands and dimension lines are drawn in the PNG "
                "only")
    return None


def build_panel_figure(cleaned, *, title, ylabel, depth_axis, y_view, refs,
                       grid, legend, footnote, widths):
    """The Plotly twin of chart panels sharing a vertical axis: subplots in
    one row sharing y, each with its series, filled diagrams, reference
    lines, notes and arrows where the PNG has them. ``None`` when plotly
    cannot be imported."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:                                   # noqa: BLE001
        return None
    n = len(cleaned)
    total = sum(widths)
    fig = make_subplots(
        rows=1, cols=n, shared_yaxes=True,
        column_widths=[w / total for w in widths],
        horizontal_spacing=min(0.035, 0.9 / max(n - 1, 1)),
        subplot_titles=[p["title"].replace("\n", "<br>") for p in cleaned])
    for ann in fig.layout.annotations:          # the panels' titles
        ann.font = {"size": 12, "color": _dp._TEXT_COLOR}
    px = _dp._PT_TO_PX * 0.75
    for k, p in enumerate(cleaned, start=1):
        ref = "" if k == 1 else str(k)
        order = sorted(range(len(p["series"])),
                       key=lambda i: p["series"][i]["style"] != "polygon")
        for i in order:
            s = p["series"][i]
            color = s["color"] or _dp._PALETTE[i % len(_dp._PALETTE)]
            for area in _dp._plotly_fill_to(go, s, color):
                fig.add_trace(area, row=1, col=k)
            style, shape, marker_pt = s["style"], "linear", 0.0
            if style == "markers":
                mode, marker_pt = "markers", _dp._marker_size(s)
            elif style in ("line", "polygon"):
                mode = "lines"
            elif style == "step":
                mode, shape = "lines", "hv"
            else:
                mode, marker_pt = "lines+markers", _dp._marker_size(s)
            has_text = bool(s["labels"] and any(s["labels"]))
            trace = go.Scatter(
                x=list(s["x"]), y=list(s["y"]), name=s["label"],
                mode=mode + ("+text" if has_text else ""),
                line={"color": color, "width": s["line_width"],
                      "shape": shape, "dash": _dp.DASHES[s["line_dash"]][1]},
                showlegend=bool(legend) and s["in_legend"])
            if marker_pt:
                symbol = _dp.MARKERS[s["marker"]][1]
                if s["marker_open"]:
                    symbol += "-open"
                trace.marker = {"color": color, "symbol": symbol,
                                "size": marker_pt * _dp._PT_TO_PX}
                if s["marker_edge"]:
                    trace.marker.line = {"color": s["marker_edge"],
                                         "width": 1}
            if style == "polygon" and s["fill"]:
                trace.fill = "toself"
                trace.fillcolor = _dp._rgba(
                    color if s["fill"] is True else s["fill"],
                    s["fill_opacity"])
            if has_text:
                trace.text = list(s["labels"])
                trace.textposition = (list(s["label_spots"])
                                      if s["label_spots"] else "top center")
                trace.textfont = {"size": _dp._LABEL_SIZE * px,
                                  "color": _dp._TEXT_COLOR}
            if s["hover"]:
                trace.hovertext = list(s["hover"])
                trace.hoverinfo = "text"
            fig.add_trace(trace, row=1, col=k)
        for v, lab in p["vlines"]:
            fig.add_vline(x=v, row=1, col=k, line_color=_dp._REF_COLOR,
                          line_width=_dp._REF_WIDTH, line_dash="dot",
                          **({"annotation_text": lab,
                              "annotation_font": {"size": 10,
                                                  "color": _dp._REF_COLOR}}
                             if lab else {}))
        for a in p["arrows"]:
            fig.add_annotation(
                xref=f"x{ref}", yref=f"y{ref}", x=a["to"][0], y=a["to"][1],
                axref=f"x{ref}", ayref=f"y{ref}", ax=a["from"][0],
                ay=a["from"][1], showarrow=True, arrowhead=2, arrowsize=1,
                arrowwidth=1.2, arrowcolor=a["color"], text="")
            if a["text"]:
                across = (abs(a["to"][0] - a["from"][0]) > 0)
                fig.add_annotation(
                    xref=f"x{ref}", yref=f"y{ref}", x=a["from"][0],
                    y=a["from"][1], showarrow=False,
                    text=a["text"].replace("\n", "<br>"),
                    xanchor=("right" if across and a["to"][0] < a["from"][0]
                             else "left"),
                    yanchor="bottom",
                    font={"size": (_TEXT_PT + 0.5) * px, "color": a["color"]})
        view = (p["view"], y_view)
        _dp._plotly_notes(fig, p["notes"], False, False, view,
                          xref=f"x{ref}", yref=f"y{ref}")
        fig.update_xaxes(range=list(p["view"]), autorange=False,
                         title_text=p["xlabel"].replace("\n", "<br>"),
                         row=1, col=k)
    for j, (v, lab) in enumerate(refs):
        fig.add_hline(y=v, line_color=_dp._REF_COLOR, line_width=0.9,
                      line_dash="dash")
        if lab:
            fig.add_annotation(
                xref=f"x{n if n > 1 else ''} domain",
                yref=f"y{n if n > 1 else ''}", x=1.0, y=v, text=lab,
                showarrow=False, xanchor="right", yanchor="bottom",
                font={"size": 10, "color": _dp._REF_COLOR})
    lines = footnote.split("\n") if footnote else []
    fig.update_layout(
        template="plotly_white", paper_bgcolor="white",
        plot_bgcolor="white", font={"color": _dp._TEXT_COLOR},
        showlegend=bool(legend),
        margin={"l": 70, "r": 30, "t": 100 if title else 60,
                "b": 60 + (22 + 14 * len(lines) if lines else 0)})
    if title:
        fig.update_layout(title={"text": title.replace("\n", "<br>"),
                                 "x": 0.0, "xanchor": "left",
                                 "font": {"size": 15,
                                          "color": _dp._TEXT_COLOR}})
    axis_kw = {"showgrid": False, "gridcolor": _dp._GRID_COLOR,
               "gridwidth": _dp._GRID_WIDTH, "zeroline": False,
               "title_font": {"size": 12, "color": _dp._TEXT_COLOR},
               "tickfont": {"color": _dp._TEXT_COLOR}, "showline": True,
               "linecolor": "#888888"}
    fig.update_xaxes(**axis_kw)
    fig.update_yaxes(**axis_kw)
    fig.update_yaxes(range=list(y_view[::-1] if depth_axis else y_view),
                     autorange=False)
    fig.update_yaxes(title_text=ylabel.replace("\n", "<br>"), row=1, col=1)
    for k, p in enumerate(cleaned, start=1):
        if p["grid"]:
            fig.update_xaxes(showgrid=True, row=1, col=k)
            fig.update_yaxes(showgrid=True, row=1, col=k)
    if lines:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.0, y=0.0, yshift=-58,
            xanchor="left", yanchor="top", align="left", showarrow=False,
            text="<br>".join(lines),
            font={"size": 10.5, "color": _dp._TEXT_COLOR})
    return fig


__all__ = ["render_panel_plot", "build_panel_figure", "PANEL_KEYS",
           "PANEL_KINDS", "BAND_KEYS", "DIMENSION_KEYS", "ARROW_KEYS"]

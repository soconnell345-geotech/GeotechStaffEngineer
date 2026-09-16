"""Generic data plot — one or more (x, y) series rendered to a PNG.

The "plots of data" half of the owner's 2026-09-11 feedback (calc packages
lacking figures). Every analysis module can produce numbers; until now only a
module with a canned calc package could turn them into a chart, and the agent
has no code-execution tool to improvise one. This is the small, general,
matplotlib-backed plot the agent CAN call: SPT/CPT vs depth, settlement vs
time, load vs displacement, a sensitivity sweep, a method comparison.

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

#: Series line styles the caller may ask for.
STYLES = ("line", "markers", "both", "step")

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
#: matplotlib sizes markers in POINTS, Plotly in PIXELS. At the PNG's default
#: 150 dpi one point is ~2 px, so the interactive chart reads like the image.
_PT_TO_PX = 2.0


@dataclass
class DataPlotResult:
    """A rendered data plot.

    Attributes
    ----------
    series : list of dict
        Resolved series as drawn: ``label``, ``n`` points, ``x_range``,
        ``y_range``, ``style``.
    image_base64 : str
        The PNG, base64-encoded (no data-URI prefix).
    png_bytes : bytes
        Raw PNG bytes (not in ``to_dict()``).
    width_px, height_px : int
        Pixel size of the PNG.
    title, xlabel, ylabel : str
    depth_axis : bool
        True when the y axis was inverted (depth increasing downward).
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

    def summary(self) -> str:
        parts = [f"'{self.title}'" if self.title else "data plot",
                 f"{len(self.series)} series"]
        for s in self.series:
            parts.append(f"{s['label']} ({s['n']} pts)")
        if self.depth_axis:
            parts.append("depth axis (down)")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "series": list(self.series),
            "title": self.title, "xlabel": self.xlabel, "ylabel": self.ylabel,
            "depth_axis": self.depth_axis,
            "width_px": self.width_px, "height_px": self.height_px,
            "warnings": list(self.warnings),
        }


def _clean_series(raw, idx: int, warnings: list) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"series[{idx}] must be a dict with 'x' and 'y'")
    xs, ys = raw.get("x"), raw.get("y")
    if xs is None or ys is None:
        raise ValueError(f"series[{idx}] needs both 'x' and 'y' lists")
    try:
        xs = [float(v) for v in xs]
        ys = [float(v) for v in ys]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"series[{idx}] has a non-numeric value: {exc}")
    if len(xs) != len(ys):
        raise ValueError(f"series[{idx}]: x has {len(xs)} values, y has "
                         f"{len(ys)} — they must match")
    if not xs:
        raise ValueError(f"series[{idx}] is empty")
    pairs = [(x, y) for x, y in zip(xs, ys)
             if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < len(xs):
        warnings.append(f"series[{idx}]: dropped {len(xs) - len(pairs)} "
                        "non-finite point(s)")
    if not pairs:
        raise ValueError(f"series[{idx}] has no finite points")
    label = str(raw.get("label") or f"Series {idx + 1}")
    style = str(raw.get("style") or "both").lower()
    if style not in STYLES:
        raise ValueError(f"series[{idx}] style '{style}' not in {STYLES}")
    return {"label": label, "style": style,
            "color": raw.get("color"),
            "x": [p[0] for p in pairs], "y": [p[1] for p in pairs]}


def build_data_plot_figure(cleaned_series, *, title: str = "",
                           xlabel: str = "", ylabel: str = "",
                           depth_axis: bool = False, logx: bool = False,
                           logy: bool = False, hlines=None, vlines=None,
                           grid: bool = True,
                           legend: Optional[bool] = None):
    """Build the interactive Plotly twin of :func:`render_data_plot`'s PNG.

    ``cleaned_series`` is what :func:`_clean_series` produced for the
    matplotlib pass — the SAME resolved points, labels, styles and colors, so
    the two backends draw identical data (including the dropped-non-finite
    points). Returns a ``plotly.graph_objects.Figure``, or ``None`` when
    plotly cannot be imported. ``plotly>=5`` is a core dependency, so that
    guard is belt and braces; the import stays inside the function per the
    ``docs/CALC_VIZ_PLAN.md`` ground rule (no plotting import at module load).

    Semantics mirror the matplotlib figure: ``depth_axis`` reverses the y axis
    and moves the x axis to the top, ``logx``/``logy`` set log axes, the four
    styles map to lines / markers / lines+markers / step (``line_shape="hv"``,
    matplotlib's ``where="post"``), and reference lines are drawn in the same
    color, width and dash. One deliberate difference: a labelled reference
    line becomes a chart ANNOTATION here (matplotlib puts it in the legend) —
    Plotly shapes carry no legend entry.
    """
    try:
        import plotly.graph_objects as go
    except Exception:                       # noqa: BLE001 — plotly absent
        return None

    fig = go.Figure()
    for i, s in enumerate(cleaned_series):
        color = s["color"] or _PALETTE[i % len(_PALETTE)]
        style = s["style"]
        shape, marker_pt = "linear", 0.0
        if style == "markers":
            mode, marker_pt = "markers", _MARKER_SIZE
        elif style == "line":
            mode = "lines"
        elif style == "step":
            mode, shape = "lines", "hv"
        else:
            mode, marker_pt = "lines+markers", _MARKER_SIZE_BOTH
        trace = go.Scatter(
            x=list(s["x"]), y=list(s["y"]), name=s["label"], mode=mode,
            line={"color": color, "width": _LINE_WIDTH, "shape": shape},
        )
        if marker_pt:
            trace.marker = {"color": color, "size": marker_pt * _PT_TO_PX}
        fig.add_trace(trace)

    any_ref_label = False
    for spec in (hlines or []):
        v, lab = _ref(spec)
        pos = _axis_coord(v, logy)
        if pos is not None:
            fig.add_hline(y=pos, line_color=_REF_COLOR, line_width=_REF_WIDTH,
                          line_dash="dash",
                          **({"annotation_text": lab} if lab else {}))
        any_ref_label = any_ref_label or bool(lab)
    for spec in (vlines or []):
        v, lab = _ref(spec)
        pos = _axis_coord(v, logx)
        if pos is not None:
            fig.add_vline(x=pos, line_color=_REF_COLOR, line_width=_REF_WIDTH,
                          line_dash="dot",
                          **({"annotation_text": lab} if lab else {}))
        any_ref_label = any_ref_label or bool(lab)

    show_legend = (legend if legend is not None
                   else (len(cleaned_series) > 1 or any_ref_label))
    fig.update_layout(
        template="plotly_white",
        showlegend=bool(show_legend),
        legend={"font": {"size": 11}},
        margin={"l": 70, "r": 30, "t": 70 if title else 40, "b": 60},
    )
    if title:
        fig.update_layout(title={"text": title,
                                 "font": {"size": 15, "color": "#222222"}})
    axis_kw = {"showgrid": bool(grid), "gridcolor": _GRID_COLOR,
               "gridwidth": _GRID_WIDTH, "zeroline": False,
               "title_font": {"size": 13}}
    fig.update_xaxes(title_text=xlabel or "", **axis_kw)
    fig.update_yaxes(title_text=ylabel or "", **axis_kw)
    if logx:
        fig.update_xaxes(type="log")
    if logy:
        fig.update_yaxes(type="log")
    if depth_axis:                          # boring-log convention, as the PNG
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(side="top")
    return fig


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


def render_data_plot(series, *, title: str = "", xlabel: str = "",
                     ylabel: str = "", depth_axis: bool = False,
                     logx: bool = False, logy: bool = False,
                     hlines=None, vlines=None, grid: bool = True,
                     legend: Optional[bool] = None, dpi: int = 150,
                     width_in: float = 6.5, height_in: float = 4.5,
                     interactive: bool = True) -> DataPlotResult:
    """Render ``series`` (list of ``{x, y, label, style, color}``) to a PNG.

    ``depth_axis=True`` inverts the y axis (depth increases downward) and puts
    the x axis at the top — the boring-log convention. ``hlines`` / ``vlines``
    are ``[{"value": v, "label": "..."}]`` reference lines (a water table, an
    allowable value, a design load). ``legend`` defaults to "when more than
    one series or any labelled reference line".

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

    fig, ax = plt.subplots(figsize=(float(width_in), float(height_in)))
    for i, s in enumerate(cleaned):
        color = s["color"] or _PALETTE[i % len(_PALETTE)]
        kw = {"color": color, "label": s["label"], "linewidth": _LINE_WIDTH}
        if s["style"] == "markers":
            ax.plot(s["x"], s["y"], linestyle="none", marker="o",
                    markersize=_MARKER_SIZE, **kw)
        elif s["style"] == "line":
            ax.plot(s["x"], s["y"], linestyle="-", **kw)
        elif s["style"] == "step":
            ax.step(s["x"], s["y"], where="post", **kw)
        else:
            ax.plot(s["x"], s["y"], linestyle="-", marker="o",
                    markersize=_MARKER_SIZE_BOTH, **kw)

    any_ref_label = False
    for spec in (hlines or []):
        v, lab = _ref(spec)
        ax.axhline(v, color=_REF_COLOR, linestyle="--", linewidth=_REF_WIDTH,
                   label=lab)
        any_ref_label = any_ref_label or bool(lab)
    for spec in (vlines or []):
        v, lab = _ref(spec)
        ax.axvline(v, color=_REF_COLOR, linestyle=":", linewidth=_REF_WIDTH,
                   label=lab)
        any_ref_label = any_ref_label or bool(lab)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
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
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
    show_legend = (legend if legend is not None
                   else (len(cleaned) > 1 or any_ref_label))
    if show_legend:
        ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()

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
                vlines=vlines, grid=bool(grid), legend=legend)
        except Exception as exc:                        # noqa: BLE001
            warnings.append("interactive chart not built "
                            f"({type(exc).__name__}: {exc}); the PNG is fine")

    resolved = [{"label": s["label"], "n": len(s["x"]), "style": s["style"],
                 "x_range": [min(s["x"]), max(s["x"])],
                 "y_range": [min(s["y"]), max(s["y"])]} for s in cleaned]
    return DataPlotResult(
        series=resolved,
        image_base64=base64.b64encode(png).decode("ascii"),
        png_bytes=png, width_px=w_px, height_px=h_px,
        title=title or "", xlabel=xlabel or "", ylabel=ylabel or "",
        depth_axis=bool(depth_axis), warnings=warnings, figure=figure)


def _ref(spec) -> tuple:
    if isinstance(spec, dict):
        try:
            return float(spec.get("value")), str(spec.get("label") or "")
        except (TypeError, ValueError):
            raise ValueError(f"reference line needs a numeric 'value': {spec}")
    return float(spec), ""


__all__ = ["render_data_plot", "build_data_plot_figure", "DataPlotResult",
           "STYLES"]

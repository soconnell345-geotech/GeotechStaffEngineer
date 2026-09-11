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


def render_data_plot(series, *, title: str = "", xlabel: str = "",
                     ylabel: str = "", depth_axis: bool = False,
                     logx: bool = False, logy: bool = False,
                     hlines=None, vlines=None, grid: bool = True,
                     legend: Optional[bool] = None, dpi: int = 150,
                     width_in: float = 6.5,
                     height_in: float = 4.5) -> DataPlotResult:
    """Render ``series`` (list of ``{x, y, label, style, color}``) to a PNG.

    ``depth_axis=True`` inverts the y axis (depth increases downward) and puts
    the x axis at the top — the boring-log convention. ``hlines`` / ``vlines``
    are ``[{"value": v, "label": "..."}]`` reference lines (a water table, an
    allowable value, a design load). ``legend`` defaults to "when more than
    one series or any labelled reference line".
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
        kw = {"color": color, "label": s["label"], "linewidth": 1.6}
        if s["style"] == "markers":
            ax.plot(s["x"], s["y"], linestyle="none", marker="o",
                    markersize=4.5, **kw)
        elif s["style"] == "line":
            ax.plot(s["x"], s["y"], linestyle="-", **kw)
        elif s["style"] == "step":
            ax.step(s["x"], s["y"], where="post", **kw)
        else:
            ax.plot(s["x"], s["y"], linestyle="-", marker="o",
                    markersize=3.5, **kw)

    any_ref_label = False
    for spec in (hlines or []):
        v, lab = _ref(spec)
        ax.axhline(v, color="#444444", linestyle="--", linewidth=1.0,
                   label=lab)
        any_ref_label = any_ref_label or bool(lab)
    for spec in (vlines or []):
        v, lab = _ref(spec)
        ax.axvline(v, color="#444444", linestyle=":", linewidth=1.0,
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
        ax.grid(True, color="#dddddd", linewidth=0.7)
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

    resolved = [{"label": s["label"], "n": len(s["x"]), "style": s["style"],
                 "x_range": [min(s["x"]), max(s["x"])],
                 "y_range": [min(s["y"]), max(s["y"])]} for s in cleaned]
    return DataPlotResult(
        series=resolved,
        image_base64=base64.b64encode(png).decode("ascii"),
        png_bytes=png, width_px=w_px, height_px=h_px,
        title=title or "", xlabel=xlabel or "", ylabel=ylabel or "",
        depth_axis=bool(depth_axis), warnings=warnings)


def _ref(spec) -> tuple:
    if isinstance(spec, dict):
        try:
            return float(spec.get("value")), str(spec.get("label") or "")
        except (TypeError, ValueError):
            raise ValueError(f"reference line needs a numeric 'value': {spec}")
    return float(spec), ""


__all__ = ["render_data_plot", "DataPlotResult", "STYLES"]

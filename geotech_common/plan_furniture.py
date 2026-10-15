"""Drawing furniture for a plan drawn as a drawing: scale bar, north arrow.

Owner, 2026-10-15, on the app's plan figures: "There isn't much use to the
grid lines or the multiple axis number... This could just look like a
typical plot with a scale bar ... a property line and north arrow any time
we do a plan view", and "what I'm talking about with the grid lines is
basically trying to follow Tufte principles: don't use unnecessary ink";
the north arrow "should follow what the plan you pulled data from uses or
other plans you reviewed on a project. If there is not, it can default to
north arrow up."

Shared by every module that draws a plan (``profile_figure.plot_data`` with
``plan_view``, the ``gstools`` probability map), so the drawings read alike.
Both renderings: matplotlib (the PNG for reports) and Plotly (the chat).
"""

from __future__ import annotations

import math
from typing import List, Optional

#: Alternating black and white segments in a scale bar.
BAR_SEGMENTS = 4
#: North arrow length, and how far past its head the "N" sits (points).
ARROW_PT = 22.0
N_PT = 8.0
INK = "#222222"
#: Plotly sizes in pixels, matplotlib in points: about this many px a point
#: at the app's 150 dpi PNGs, scaled to read alike.
PT_TO_PX = 1.5


def nice_length(span: float) -> float:
    """A scale-bar length: the largest 1, 2 or 5 x 10^k within a third of
    the width drawn."""
    target = span / 3.0
    best = None
    for k in range(-3, 9):
        for m in (1.0, 2.0, 5.0):
            c = m * 10.0 ** k
            if c <= target and (best is None or c > best):
                best = c
    return best or target


def north_or_default(north_deg: Optional[float]) -> tuple:
    """``(angle, note)``: the angle to draw (degrees clockwise from up) and a
    note when it was not given (drawn up, the owner's default)."""
    if north_deg is None:
        return 0.0, ("North drawn straight up: no north was given. If a plan "
                     "of the site shows north (the plan the data came from, "
                     "or the project's other plans), give that angle as "
                     "north_deg, degrees clockwise from up.")
    return float(north_deg), None


def mpl_scale_bar(ax, x0: float, y: float, length: float, height: float,
                  units: str = "m", fontsize: float = 7.5,
                  text_color: str = INK) -> List:
    """Scale bar from (x0, y) in data units: alternating segments, '0' and
    the length with its unit at the ends. Returns the text artists."""
    from matplotlib.patches import Rectangle
    seg = length / BAR_SEGMENTS
    for k in range(BAR_SEGMENTS):
        ax.add_patch(Rectangle((x0 + k * seg, y), seg, height,
                               facecolor=INK if k % 2 == 0 else "white",
                               edgecolor=INK, linewidth=0.8, zorder=7))
    return [ax.annotate(lab, (xv, y + height), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=fontsize, color=text_color, zorder=7)
            for xv, lab in ((x0, "0"),
                            (x0 + length, f"{length:g} {units}".strip()))]


def mpl_north_arrow(ax, xy, north_deg: float) -> List:
    """North arrow centred on ``xy`` (data units), pointing ``north_deg``
    clockwise from up. Returns the text artists."""
    from matplotlib.transforms import offset_copy
    th = math.radians(float(north_deg))
    ux, uy = math.sin(th), math.cos(th)
    head = offset_copy(ax.transData, fig=ax.figure, x=ux * ARROW_PT / 2,
                       y=uy * ARROW_PT / 2, units="points")
    ax.annotate("", xy=xy, xycoords=head,
                xytext=(-ux * ARROW_PT, -uy * ARROW_PT),
                textcoords="offset points", zorder=7,
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.4,
                            "mutation_scale": 12})
    return [ax.annotate("N", xy=xy, xycoords=head,
                        xytext=(ux * N_PT, uy * N_PT),
                        textcoords="offset points", ha="center",
                        va="center", fontsize=9, fontweight="bold",
                        color=INK, zorder=7)]


def plotly_scale_bar(fig, x0: float, y: float, length: float,
                     height: float, units: str = "m",
                     text_color: str = INK) -> None:
    """The Plotly twin of :func:`mpl_scale_bar` (data coordinates)."""
    seg = length / BAR_SEGMENTS
    for k in range(BAR_SEGMENTS):
        fig.add_shape(type="rect", x0=x0 + k * seg, x1=x0 + (k + 1) * seg,
                      y0=y, y1=y + height, xref="x", yref="y",
                      fillcolor=INK if k % 2 == 0 else "white",
                      line={"color": INK, "width": 1})
    for xv, lab in ((x0, "0"), (x0 + length, f"{length:g} {units}".strip())):
        fig.add_annotation(x=xv, y=y + height, xref="x", yref="y", text=lab,
                           showarrow=False, yanchor="bottom", yshift=2,
                           font={"size": 10, "color": text_color})


def plotly_north_arrow(fig, xy, north_deg: float) -> None:
    """The Plotly twin of :func:`mpl_north_arrow` (data coordinates)."""
    th = math.radians(float(north_deg))
    ux, uy = math.sin(th), math.cos(th)
    px = ARROW_PT * PT_TO_PX
    x, y = xy
    # the tail sits (ax, ay) px from the head, y downward; xshift / yshift
    # move the whole arrow so it is centred on its spot
    fig.add_annotation(x=x, y=y, xref="x", yref="y", text="",
                       showarrow=True, arrowhead=2, arrowsize=1.3,
                       arrowwidth=1.6, arrowcolor=INK,
                       ax=-ux * px, ay=uy * px,
                       xshift=ux * px / 2, yshift=uy * px / 2)
    fig.add_annotation(x=x, y=y, xref="x", yref="y", text="N",
                       showarrow=False, xshift=ux * (px / 2 + 11),
                       yshift=uy * (px / 2 + 11),
                       font={"size": 13, "color": INK})


__all__ = ["nice_length", "north_or_default", "mpl_scale_bar",
           "mpl_north_arrow", "plotly_scale_bar", "plotly_north_arrow",
           "BAR_SEGMENTS", "ARROW_PT"]

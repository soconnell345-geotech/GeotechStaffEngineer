"""Example figures that follow the figure checklist.

The owner's choice on 2026-10-15 (OWN-61, option B): a figure helper that
works from a checklist after Tufte and from "some actual good looking example
plots". These are those examples, two kinds:

- :data:`EXAMPLES`: one per kind of figure a geotechnical answer or report
  usually needs, each a complete ``plot_data`` parameter set;
- :data:`CODE_EXAMPLES`: figures ``plot_data`` cannot draw (a structure to
  scale with its diagrams beside it, panels sharing one elevation axis),
  each a complete matplotlib script for ``plot_code``'s locked box and the
  ``data`` it reads (BACKLOG GM-97: the owner's conventions for profiles and
  design figures: elevation on the vertical axis, kPa and kN, the wall
  drawn).

Each has what it shows and why it is drawn that way. The figure helper
(``funhouse_agent/deep/figure_helper.py``) reads them through its
``figure_examples`` tool, the code examples only when it has ``plot_code``,
and adapts the nearest one; ``render_examples`` draws the ``plot_data`` ones
(tests, the gallery); the code examples are drawn through the box by
``funhouse_agent/deep/tests/test_figure_helper.py``.

Every number here is invented. Nothing in this file describes a real site.
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional


def _boring(name: str, ground_el: float, depths: List[float],
            n60: List[int], marker: str, color: str) -> dict:
    """One boring's SPT series against elevation: named at its deepest test,
    each test's depth below ground kept in its hover text."""
    els = [round(ground_el - z, 2) for z in depths]
    label = f"{name} (ground El. {ground_el:g} m)"
    return {"x": list(n60), "y": els, "label": label, "style": "both",
            "marker": marker, "color": color,
            "labels": [""] * (len(els) - 1) + [label],
            "hover": [f"{name}, {z:g} m below ground<br>El. {e:.1f} m, "
                      f"N60 {n}" for z, e, n in zip(depths, els, n60)]}


_SPT_DEPTHS = [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0]

#: name -> {"shows", "why", "params"}; params is a plot_data call.
EXAMPLES: Dict[str, dict] = {
    "plan_borings": {
        "shows": "A site plan: the property line, a building footprint and "
                 "the borings, those that reached hard rock and those that "
                 "did not, in a local grid.",
        "why": [
            "plan_view: drawn to scale and read like a drawing sheet, not a "
            "chart: no axes, axis numbers or grid; a scale bar with the "
            "coordinate system under it, and a north arrow at the angle "
            "read off the source drawing (north_deg, here 20 degrees "
            "clockwise from up).",
            "The property line is a dash-dot outline (the drafting "
            "convention), the footprint a filled outline under the points, "
            "a zone of interest hatched over it.",
            "Borings are named beside each point (labels), so no legend "
            "entry per boring; two groups told apart by colour AND shape "
            "(reached: filled circle; not reached: grey x); the legend names "
            "the groups and sits outside the plan.",
            "Hover text carries what the plan cannot show (depth to rock).",
        ],
        "params": {
            "title": "Footprint and borings",
            "xlabel": "Local site grid, metres",
            "ylabel": "",
            "plan_view": True, "north_deg": 20, "scale_units": "m",
            "series": [
                {"x": [20, 255, 255, 20], "y": [25, 25, 175, 175],
                 "label": "Property line", "style": "polygon",
                 "line_dash": "dashdot", "color": "#444444",
                 "line_width": 1.2},
                {"x": [40, 80, 80, 70, 70, 80, 80, 40],
                 "y": [60, 60, 85, 85, 100, 100, 150, 150],
                 "label": "Building footprint", "style": "polygon",
                 "fill": True, "color": "#1f4e79", "line_width": 2.0,
                 "fill_opacity": 0.15},
                {"x": [40, 80, 80, 40], "y": [110, 110, 150, 150],
                 "label": "Rock likely above excavation level",
                 "style": "polygon", "hatch": "/", "color": "#c0504d",
                 "line_width": 1.0},
                {"x": [52, 150, 230], "y": [120, 40, 160],
                 "label": "Hard rock reached", "style": "markers",
                 "marker": "circle", "marker_size": 6.5, "color": "#c0504d",
                 "marker_edge": "white",
                 "labels": ["B-2", "B-6", "B-9"],
                 "hover": ["B-2<br>hard rock 14.2 m below ground",
                           "B-6<br>hard rock 11.8 m",
                           "B-9<br>hard rock 12.5 m"]},
                {"x": [45, 74, 95, 120, 160, 210],
                 "y": [70, 140, 90, 130, 95, 60],
                 "label": "Hard rock not reached", "style": "markers",
                 "marker": "x", "marker_size": 6.0, "color": "#7f7f7f",
                 "labels": ["B-1", "B-3", "B-4", "B-5", "B-7", "B-8"],
                 "hover": ["B-1<br>ended in soil at 12.0 m",
                           "B-3<br>ended in soil at 12.0 m",
                           "B-4", "B-5", "B-7", "B-8"]},
            ],
        },
    },
    "spt_profile": {
        "shows": "An SPT N60 profile of three borings whose ground surfaces "
                 "are at different elevations, drawn against elevation (not "
                 "depth below each boring's ground), with the groundwater "
                 "level.",
        "why": [
            "Elevation on the vertical axis, rising upward: the default for "
            "a profile. The borings start at different ground levels (24.6, "
            "23.1 and 25.4 m), so one depth below ground is a different "
            "elevation in each; on elevation the layers line up across the "
            "borings. Depth (depth_axis, ylabel 'Depth below ground') only "
            "when no elevations are given, and then the figure says so; "
            "never depth and elevation on two axes of one profile.",
            "Each boring's line is named at its deepest test with its "
            "ground-surface elevation (labels) instead of a legend; markers "
            "show where the tests are, the line joins tests in ONE boring "
            "(a continuous profile).",
            "Hover text keeps each test's depth below ground, which the "
            "elevation axis does not show.",
            "The groundwater level is a dashed reference line named with "
            "its elevation and where and when it was measured.",
            "The x axis starts at 0 and the units are on both axes.",
        ],
        "params": {
            "title": "SPT N60 against elevation",
            "xlabel": "SPT N60 (blows / 0.3 m)",
            "ylabel": "Elevation (m, site datum)",
            "xlim": [0, 50], "legend": False,
            "hlines": [{"value": 19.9,
                        "label": "Groundwater El. 19.9 m (B-2, 12 Sep)"}],
            "series": [
                _boring("B-1", 24.6, _SPT_DEPTHS[:7],
                        [5, 6, 8, 11, 16, 21, 31], "circle", "#1f4e79"),
                _boring("B-2", 23.1, _SPT_DEPTHS,
                        [5, 7, 10, 15, 19, 29, 34, 38], "square", "#c0504d"),
                _boring("B-3", 25.4, _SPT_DEPTHS[:6],
                        [4, 6, 7, 9, 14, 18], "triangle", "#4f8f3a"),
            ],
        },
    },
    "cases": {
        "shows": "One result across the cases of a sensitivity study, for "
                 "two interpretations.",
        "why": [
            "x names the cases, so the axis reads 'Exp 15 m', not '2.5'.",
            "Markers only: the cases are separate, a line between them "
            "would show a trend that is not there.",
            "Two interpretations told apart by shape and colour, each named "
            "as the logs name it (never a coined label); the best estimate "
            "is a labelled reference line.",
            "The y axis runs over the whole possible range (0 to 1) so "
            "differences are not exaggerated.",
        ],
        "params": {
            "title": "Fraction of the footprint above the excavation level, "
                     "by case",
            "xlabel": "Covariance model and length scale",
            "ylabel": "Fraction of footprint", "ylim": [0, 1],
            "legend_position": "below",
            "hlines": [{"value": 0.16, "label": "Best estimate"}],
            "series": [
                {"x": ["Exp 15 m", "Exp 30 m", "Exp 60 m", "Sph 30 m",
                       "Gau 30 m"],
                 "y": [0.42, 0.53, 0.42, 0.49, 0.44],
                 "label": "Top of weathered rock, first encountered",
                 "style": "markers",
                 "marker": "circle", "marker_size": 6.5, "color": "#c0504d"},
                {"x": ["Exp 15 m", "Exp 30 m", "Exp 60 m", "Sph 30 m",
                       "Gau 30 m"],
                 "y": [0.16, 0.14, 0.21, 0.21, 0.21],
                 "label": "Top of weathered rock, no soil below",
                 "style": "markers",
                 "marker": "diamond", "marker_size": 6.5,
                 "color": "#1f4e79"},
            ],
        },
    },
    "time_settlement": {
        "shows": "Consolidation settlement against time, with and without "
                 "wick drains.",
        "why": [
            "Time on a log axis because it spans decades (1 to 1,000 days); "
            "the label says so through the units.",
            "Settlement increases downward (depth_axis), as it happens.",
            "Each curve named where it stands apart from the other (not "
            "at the end, where the two meet), no legend.",
            "The end of filling is a labelled vertical reference line.",
        ],
        "params": {
            "title": "Settlement of the embankment centre",
            "xlabel": "Time since start of filling (days)",
            "ylabel": "Settlement (mm)", "logx": True, "depth_axis": True,
            "legend": False,
            "vlines": [{"value": 60, "label": "Filling complete"}],
            "series": [
                {"x": [1, 3, 10, 30, 60, 100, 200, 400, 1000],
                 "y": [5, 14, 40, 90, 150, 210, 300, 360, 395],
                 "label": "No drains", "style": "line", "color": "#c0504d",
                 "line_width": 2.0,
                 "labels": ["", "", "", "", "", "No drains", "", "", ""]},
                {"x": [1, 3, 10, 30, 60, 100, 200, 400, 1000],
                 "y": [12, 45, 130, 260, 340, 375, 392, 398, 400],
                 "label": "Wick drains at 1.5 m", "style": "line",
                 "color": "#1f4e79", "line_width": 2.0,
                 "labels": ["", "", "", "Wick drains at 1.5 m", "", "",
                            "", "", ""]},
            ],
        },
    },
    "load_settlement": {
        "shows": "A pile load test: load against head settlement, with the "
                 "Davisson offset line and the capacity it gives.",
        "why": [
            "Settlement down (depth_axis), load across: the usual load-test "
            "plot.",
            "Measured points are markers joined by a line (one test, "
            "continuous); the Davisson line is dashed and named.",
            "The capacity the text quotes is marked on the figure (a "
            "labelled point), not left for the reader to find.",
        ],
        "params": {
            "title": "Static load test, test pile TP-1",
            "xlabel": "Applied load (kN)", "ylabel": "Pile head settlement (mm)",
            "depth_axis": True, "xlim": [0, 2600], "ylim": [0, 40],
            "legend_position": "below",
            "series": [
                {"x": [0, 300, 600, 900, 1200, 1500, 1800, 2000, 2200, 2400],
                 "y": [0, 1.0, 2.2, 3.6, 5.3, 7.4, 10.4, 13.5, 18.5, 28.0],
                 "label": "Measured", "style": "both", "marker": "circle",
                 "color": "#1f4e79"},
                {"x": [0, 2600], "y": [5.6, 16.8],
                 "label": "Davisson offset line", "style": "line",
                 "line_dash": "dash", "color": "#7f7f7f"},
                {"x": [2050], "y": [14.6], "label": "Davisson capacity",
                 "style": "markers", "marker": "diamond", "marker_size": 8,
                 "color": "#c0504d", "in_legend": False,
                 "labels": ["Davisson capacity 2,050 kN"]},
            ],
        },
    },
    "exceedance": {
        "shows": "The chance that a total (here a rock volume) exceeds each "
                 "quantity, from a Monte Carlo run.",
        "why": [
            "Probability of exceedance on y from 0 to 1, the quantity on x "
            "with its units.",
            "The percentiles the text quotes (P50, P80, P90) are marked and "
            "named on the curve.",
            "The x range stops where the curve has reached about zero, so "
            "the useful part fills the figure.",
        ],
        "params": {
            "title": "Probability that the rock volume exceeds a quantity",
            "xlabel": "Rock volume above the excavation level (m³)",
            "ylabel": "Probability of exceedance", "ylim": [0, 1],
            "xlim": [0, 1200], "legend": False,
            "series": [
                {"x": [0, 50, 100, 150, 220, 300, 400, 500, 650, 800, 1000,
                       1200],
                 "y": [0.92, 0.80, 0.66, 0.55, 0.45, 0.34, 0.24, 0.17,
                       0.10, 0.06, 0.03, 0.015],
                 "label": "All realizations", "style": "line",
                 "color": "#1f4e79", "line_width": 2.0},
                {"x": [185, 457, 650], "y": [0.5, 0.2, 0.1],
                 "label": "Percentiles", "style": "markers",
                 "marker": "circle", "marker_size": 6, "color": "#c0504d",
                 "labels": ["P50 185 m³", "P80 457 m³", "P90 650 m³"]},
            ],
        },
    },
}


_WALL_CODE = r'''
# A cantilever wall drawn to scale, and beside it, on the same elevation
# axis, one diagram per lateral pressure with its resultant. Every number
# comes from `data`; only the layout is set here.
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

d = data
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
y0, y1 = d["elevation_range"]
x0, x1 = d["section_x_range"]
diagrams = d["diagrams"]

# Layout in inches: the section keeps one metre across = one metre up, so
# its width follows from the axes height; the diagrams share the rest.
fig_w, fig_h = 6.5, 4.9
left, bottom, height = 0.62, 0.95, 3.35
sec_w = height * (x1 - x0) / (y1 - y0)
gap, right = 0.16, 0.08
dia_w = (fig_w - left - sec_w - gap * (len(diagrams) + 1) - right) \
    / len(diagrams)
fig = plt.figure(figsize=(fig_w, fig_h))


def box(x, w):
    return [x / fig_w, bottom / fig_h, w / fig_w, height / fig_h]


ax = fig.add_axes(box(left, sec_w))
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect("equal", adjustable="datalim")

# The wall, the ground on both sides and the groundwater behind it.
ax.add_patch(Polygon(d["wall"]["outline"], closed=True, facecolor="0.85",
                     edgecolor="0.15", linewidth=1.0))
for g in (d["retained_ground"], d["front_ground"]):
    ax.plot(g["x"], g["y"], color="0.25", linewidth=1.0)
gw = d["groundwater"]
ax.plot(gw["x"], [gw["elevation"]] * 2, color="#1f4e79", linewidth=0.8)
ax.plot([gw["symbol_x"]], [gw["elevation"] + 0.12], marker="v",
        markersize=6, color="#1f4e79", linestyle="none")
ax.text(gw["symbol_x"] + 0.2, gw["elevation"] + 0.15, gw["label"],
        color="#1f4e79", fontsize=7.5, va="bottom")

# The plane the pressures act on, named along it (above the groundwater
# line, so no line runs through the name).
hp = d["heel_plane"]
ax.plot([hp["x"]] * 2, hp["elevation"], color="0.35", linewidth=0.8,
        linestyle=(0, (4, 3)))
ax.text(hp["x"] - 0.1, hp["label_elevation"], hp["label"], rotation=90,
        va="center", ha="right", fontsize=7, color="0.25")

# The base width, dimensioned with its unit.
dim = d["base_dimension"]
yd = dim["elevation"]
for xe in dim["x"]:
    ax.plot([xe, xe], [dim["from_elevation"], yd - 0.15], color="0.35",
            linewidth=0.5)
ax.annotate("", xy=(dim["x"][0], yd), xytext=(dim["x"][1], yd),
            arrowprops={"arrowstyle": "<|-|>", "linewidth": 0.6,
                        "color": "0.15", "shrinkA": 0, "shrinkB": 0,
                        "mutation_scale": 7})
ax.text(sum(dim["x"]) / 2, yd - 0.12, dim["label"], ha="center", va="top",
        fontsize=7.5)

ax.set_title(d["section_title"], loc="left")
ax.set_ylabel(d["elevation_label"])
ax.xaxis.set_visible(False)
for side in ("top", "right", "bottom"):
    ax.spines[side].set_visible(False)

# One diagram per pressure, all on ONE pressure scale (small multiples).
p_max = d["pressure_axis"]["max_kpa"]
x = left + sec_w + gap
for dia in diagrams:
    a = fig.add_axes(box(x, dia_w), sharey=ax)
    x += dia_w + gap
    c = dia["color"]
    a.fill_betweenx(dia["elevation"], 0, dia["kpa"], color=c, alpha=0.22,
                    linewidth=0)
    a.plot(dia["kpa"], dia["elevation"], color=c, linewidth=1.2)
    a.plot([dia["kpa"][0], 0], [dia["elevation"][0]] * 2, color=c,
           linewidth=1.2)
    a.plot([dia["kpa"][-1], 0], [dia["elevation"][-1]] * 2, color=c,
           linewidth=1.2)
    # the largest ordinate, written beside it
    i = max(range(len(dia["kpa"])), key=lambda j: dia["kpa"][j])
    a.text(dia["kpa"][i] + 1.0, dia["elevation"][i], f"{dia['kpa'][i]:.1f}",
           fontsize=7, va="center", ha="left", color="0.2")
    # the resultant: an arrow at its elevation pointing at the wall (the
    # way the pressure acts), its symbol at the tail, its value below
    zr = dia["resultant_elevation"]
    tail = 0.82 * p_max
    a.annotate("", xy=(0, zr), xytext=(tail, zr),
               arrowprops={"arrowstyle": "-|>", "color": "0.1",
                           "linewidth": 1.0, "shrinkA": 0, "shrinkB": 0,
                           "mutation_scale": 9})
    a.text(tail, zr + 0.1, dia["symbol"], ha="right", va="bottom")
    a.text(p_max / 2, d["resultant_label_elevation"],
           f"{dia['symbol']} = {dia['resultant_kn_per_m']:.1f} kN/m\n"
           f"at El. {zr:.2f} m", ha="center", va="top", fontsize=7.5)
    a.set_xlim(0, p_max)
    a.set_xticks(d["pressure_axis"]["ticks"])
    a.set_xlabel("Pressure (kPa)")
    a.set_title(dia["title"])
    a.tick_params(axis="y", left=False, labelleft=False)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)

fig.suptitle(d["title"], x=left / fig_w, ha="left", y=0.985,
             fontsize=9.5, fontweight="bold")
fig.text(left / fig_w, 0.02, d["notes"], fontsize=7, ha="left",
         va="bottom")
save_figure(fig, "wall_pressures")
'''

_SETTLEMENT_CODE = r'''
# The soil column (with the fill that loads it), then panels sharing ONE
# elevation axis and ONE stress scale: the initial vertical effective
# stress, the preconsolidation stress, the stress increase, and the final
# stress against the preconsolidation stress. Every number is from `data`.
import matplotlib.pyplot as plt
import numpy as np

d = data
plt.rcParams.update({"font.size": 8, "axes.titlesize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
y0, y1 = d["elevation_range"]
el = np.array(d["elevation"], dtype=float)
s_max = d["stress_axis"]["max_kpa"]

fig_w, fig_h = 6.5, 5.6
left, bottom, height = 0.62, 1.0, 3.95
col_w, gap, right = 0.95, 0.14, 0.06
n = 4
pan_w = (fig_w - left - col_w - gap * n - right) / n
fig = plt.figure(figsize=(fig_w, fig_h))


def box(x, w):
    return [x / fig_w, bottom / fig_h, w / fig_w, height / fig_h]


# --- the soil column: layers as filled bands named inside, the ground
# surface, the groundwater; no x axis (its width means nothing)
col = fig.add_axes(box(left, col_w))
col.set_xlim(0, 1)
col.set_ylim(y0, y1)
for lay in d["layers"]:
    col.axhspan(lay["bottom"], lay["top"], facecolor=lay["color"],
                edgecolor="none")
    col.axhline(lay["bottom"], color="0.45", linewidth=0.6)
    col.text(0.5, lay.get("label_elevation",
                          (lay["top"] + lay["bottom"]) / 2),
             lay["name"], ha="center", va="center", fontsize=7)
col.axhline(d["ground_elevation"], color="0.15", linewidth=1.0)
gw = d["groundwater"]
col.axhline(gw["elevation"], color="#1f4e79", linewidth=0.8)
col.plot([0.12], [gw["elevation"] + 0.25], marker="v", markersize=5.5,
         color="#1f4e79", linestyle="none")
col.text(0.22, gw["elevation"] + 0.15, gw["label"], fontsize=7,
         color="#1f4e79", va="bottom")
col.set_ylabel(d["elevation_label"])
col.set_title(d["column_title"])
col.xaxis.set_visible(False)
for side in ("top", "right", "bottom"):
    col.spines[side].set_visible(False)


def panel(k, title):
    a = fig.add_axes(box(left + col_w + gap + k * (pan_w + gap), pan_w),
                     sharey=col)
    a.set_xlim(0, s_max)
    a.set_xticks(d["stress_axis"]["ticks"])
    a.set_xlabel("Stress (kPa)")
    a.set_title(title)
    a.tick_params(axis="y", left=False, labelleft=False)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    return a


def name_line(a, xs, text, at, color, dx=6, ha="left"):
    """Name a line beside itself at elevation `at`."""
    x = np.interp(at, el[::-1], np.asarray(xs, dtype=float)[::-1])
    a.text(x + (dx if ha == "left" else -dx), at, text, color=color,
           fontsize=7, ha=ha, va="center")


def preconsolidation(a, pc, color, named):
    # Test results are POINTS (each a separate test, never joined). An
    # assumed design profile is a LINE, named as one; the figure says which
    # it shows.
    t, line = pc.get("tests"), pc.get("design")
    if t:
        a.plot(t["kpa"], t["elevation"], linestyle="none", marker="o",
               markersize=4.5, markerfacecolor="white",
               markeredgecolor=color, markeredgewidth=1.2)
        if named:
            # named under the lowest test, below any design line too
            low = min(range(len(t["elevation"])),
                      key=lambda j: t["elevation"][j])
            under = min(t["elevation"] + (line["elevation"] if line
                                          else []))
            a.text(t["kpa"][low], under - 0.5, t["label"], color=color,
                   fontsize=7, ha="center", va="top")
    if line:
        a.plot(line["kpa"], line["elevation"], color=color, linewidth=1.4)
        if named:
            a.text(line["kpa"][0], line["elevation"][0] + 0.4,
                   line["label"], color=color, fontsize=7, ha="center",
                   va="bottom")


s0, ds, sf = d["initial_kpa"], d["increase_kpa"], d["final_kpa"]
pc = d["preconsolidation"]
c0, cp, cd, cf = "0.35", "#c0504d", "#4f8f3a", "#1f4e79"

a = panel(0, d["titles"]["initial"])
a.plot(s0, el, color=c0, linewidth=1.4)

a = panel(1, d["titles"]["preconsolidation"])
preconsolidation(a, pc, cp, named=True)

a = panel(2, d["titles"]["increase"])
a.plot(ds, el, color=cd, linewidth=1.4)

a = panel(3, d["titles"]["final"])
a.plot(s0, el, color=c0, linewidth=0.8, linestyle=(0, (3, 2)))
a.plot(sf, el, color=cf, linewidth=1.4)
preconsolidation(a, pc, cp, named=False)
lab = d["final_labels"]
name_line(a, s0, lab["initial"], lab["initial_at"], c0, ha="right")
name_line(a, sf, lab["final"], lab["final_at"], cf)
a.text(lab["preconsolidation_x"], lab["preconsolidation_at"],
       lab["preconsolidation"], color=cp, fontsize=7, va="center")

fig.suptitle(d["title"], x=left / fig_w, ha="left", y=0.985,
             fontsize=9.5, fontweight="bold")
fig.text(left / fig_w, 0.015, d["notes"], fontsize=7, ha="left",
         va="bottom")
save_figure(fig, "settlement_stress_profile")
'''

_PILE_CODE = r'''
# The soil column with the pile drawn in it, then panels sharing ONE
# elevation axis: unit shaft resistance, the axial resistance accumulated
# down the pile, and the axial load in the pile with the neutral plane
# where the load curve meets the resistance curve. Forces in kN, stresses
# in kPa; every number is from `data`.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

d = data
plt.rcParams.update({"font.size": 8, "axes.titlesize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
y0, y1 = d["elevation_range"]
npl = d["neutral_plane"]
pile = d["pile"]

fig_w, fig_h = 6.5, 5.6
left, bottom, height = 0.62, 1.0, 3.95
col_w, gap, right = 1.15, 0.16, 0.06
ratios = d["panel_width_ratios"]
room = fig_w - left - col_w - gap * len(ratios) - right
widths = [room * w / sum(ratios) for w in ratios]
fig = plt.figure(figsize=(fig_w, fig_h))
thousands = FuncFormatter(lambda v, _pos: f"{v:,.0f}")


def box(x, w):
    return [x / fig_w, bottom / fig_h, w / fig_w, height / fig_h]


def np_line(a):
    """The neutral plane across a panel: one dashed reference line."""
    a.axhline(npl["elevation"], color="0.3", linewidth=0.8,
              linestyle=(0, (5, 3)))


# --- the soil column: layers named inside their bands, the groundwater,
# and the pile at its true head and toe elevations (its width is not to
# scale: the notes give its size)
col = fig.add_axes(box(left, col_w))
col.set_xlim(0, 1)
col.set_ylim(y0, y1)
for lay in d["layers"]:
    col.axhspan(lay["bottom"], lay["top"], facecolor=lay["color"],
                edgecolor="none")
    col.axhline(lay["bottom"], color="0.45", linewidth=0.6)
    col.text(0.05, lay.get("label_elevation",
                           (lay["top"] + lay["bottom"]) / 2),
             lay["name"], ha="left", va="center", fontsize=7)
col.axhline(d["ground_elevation"], color="0.15", linewidth=1.0)
gw = d["groundwater"]
col.axhline(gw["elevation"], color="#1f4e79", linewidth=0.8)
col.plot([0.1], [gw["elevation"] + 0.3], marker="v", markersize=5.5,
         color="#1f4e79", linestyle="none")
col.text(0.18, gw["elevation"] + 0.18, gw["label"], fontsize=7,
         color="#1f4e79", va="bottom")
col.add_patch(Rectangle((pile["x"][0], pile["toe_elevation"]),
                        pile["x"][1] - pile["x"][0],
                        pile["head_elevation"] - pile["toe_elevation"],
                        facecolor="0.55", edgecolor="0.15", linewidth=0.8))
np_line(col)
col.set_ylabel(d["elevation_label"])
col.set_title(d["titles"]["column"])
col.xaxis.set_visible(False)
for side in ("top", "right", "bottom"):
    col.spines[side].set_visible(False)


def panel(k, title, axis):
    x = left + col_w + gap + sum(widths[:k]) + k * gap
    a = fig.add_axes(box(x, widths[k]), sharey=col)
    a.set_xlim(0, axis["max"])
    a.set_xticks(axis["ticks"])
    a.xaxis.set_major_formatter(thousands)
    a.set_xlabel(axis["label"])
    a.set_title(title)
    a.tick_params(axis="y", left=False, labelleft=False)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    np_line(a)
    return a


def write(a, labels):
    for lab in labels:
        a.text(lab["x"], lab["elevation"], lab["text"], fontsize=7,
               color=lab.get("color", "0.15"), ha=lab.get("ha", "left"),
               va="center")


blue, red, grey = "#1f4e79", "#c0504d", "0.35"

# Unit shaft resistance: a line along the pile, stepping at layer
# boundaries (each elevation there is listed twice).
fs = d["unit_shaft"]
a = panel(0, d["titles"]["unit_shaft"], d["stress_axis"])
a.plot(fs["kpa"], fs["elevation"], color=grey, linewidth=1.4)

# The axial resistance accumulated from the head down, then the toe
# resistance added at the toe: the nominal resistance the text quotes.
r = d["resistance"]
a = panel(1, d["titles"]["resistance"], d["force_axis"])
a.plot(r["shaft_kn"], r["elevation"], color=grey, linewidth=1.4)
toe = pile["toe_elevation"]
a.plot([r["shaft_kn"][-1], r["nominal_kn"]], [toe, toe], color=grey,
       linewidth=1.4)
a.plot([r["nominal_kn"]], [toe], marker="o", markersize=5, color=red,
       linestyle="none")
write(a, r["labels"])

# The axial load in the pile: the load curve (permanent load plus drag
# from the head down) meets the resistance curve (toe resistance plus
# shaft resistance from the toe up) at the neutral plane. The load in the
# pile follows the lower of the two (heavy line); the rest of each curve
# is drawn thin, to show how the plane was found.
q = d["load"]
a = panel(2, d["titles"]["load"], d["force_axis"])
a.plot(q["load_curve_kn"], q["elevation"], color=blue, linewidth=0.7)
a.plot(q["resistance_curve_kn"], q["elevation"], color=red, linewidth=0.7)
in_pile = [min(p, s) for p, s in zip(q["load_curve_kn"],
                                     q["resistance_curve_kn"])]
a.plot(in_pile, q["elevation"], color="0.1", linewidth=2.0)
a.plot([npl["max_load_kn"]], [npl["elevation"]], marker="o", markersize=5,
       color="0.1", linestyle="none")
write(a, q["labels"])

fig.suptitle(d["title"], x=left / fig_w, ha="left", y=0.985,
             fontsize=9.5, fontweight="bold")
fig.text(left / fig_w, 0.015, d["notes"], fontsize=7, ha="left",
         va="bottom")
save_figure(fig, "pile_design_profile")
'''

#: Elevations at which the pile example's accumulated forces are given.
_PILE_EL = [12.0, 11.0, 10.8, 10.0, 9.5, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0,
            2.0, 1.5, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -6.5,
            -7.0, -7.5]

#: name -> {"shows", "why", "code", "data"}: figures plot_data cannot draw,
#: each a complete script for plot_code's locked box (``data`` is what the
#: script reads as ``data``). Drawn through the box by the tests; every
#: number made up but consistent (Rankine pressures, effective stresses,
#: the neutral plane where the two curves meet).
CODE_EXAMPLES: Dict[str, dict] = {
    "wall_pressures": {
        "shows": "A cantilever retaining wall drawn to scale with its "
                 "active earth pressure, surcharge pressure and water "
                 "pressure as separate diagrams beside it on one elevation "
                 "axis, each resultant and its elevation marked.",
        "why": [
            "The wall is drawn to scale (set_aspect('equal'), its panel as "
            "wide as its metres across): the elevation axis is its scale, "
            "its base width is dimensioned with its unit, and it has no x "
            "axis.",
            "Each pressure is its own diagram beside the wall, sharing the "
            "wall's elevation axis and ONE pressure scale in kPa, so their "
            "sizes compare truly (small multiples); each is named as the "
            "references name it (active earth pressure, surcharge "
            "pressure, water pressure).",
            "Each resultant is an arrow at its elevation pointing at the "
            "wall, the way the pressure acts, with its value in kN per "
            "metre run of wall and its elevation written under its "
            "diagram; the largest ordinate is written beside the diagram.",
            "The pressures act on the vertical plane at the back of the "
            "heel, drawn dashed and named along its length clear of the "
            "groundwater line; the soil properties, the surcharge and what "
            "was left out (passive resistance) are in the notes, so every "
            "quantity is defined.",
            "When the wall's configuration is not known (typically the "
            "structural engineer designs it), draw the pressure diagrams "
            "alone: leave out the section axes, put the elevation axis "
            "and its label on the first diagram, mark the ground surface "
            "and the groundwater as named reference lines across the "
            "diagrams, and say in the title that the wall is not drawn "
            "because it is not yet configured.",
        ],
        "code": _WALL_CODE,
        "data": {
            "title": "Cantilever retaining wall: lateral pressures for "
                     "design",
            "section_title": "Wall section, to scale",
            "elevation_label": "Elevation (m, site datum)",
            "elevation_range": [97.3, 105.9],
            "section_x_range": [-2.2, 3.3],
            "wall": {"outline": [[-1.35, 98.8], [2.45, 98.8], [2.45, 99.4],
                                 [0.0, 99.4], [0.0, 105.0], [-0.30, 105.0],
                                 [-0.45, 99.4], [-1.35, 99.4]]},
            "retained_ground": {"x": [0.0, 3.3], "y": [105.0, 105.0]},
            "front_ground": {"x": [-2.2, -0.43], "y": [100.0, 100.0]},
            "groundwater": {"elevation": 101.0, "x": [0.0, 3.3],
                            "symbol_x": 0.35,
                            "label": "Groundwater\nEl. 101.0 m"},
            "heel_plane": {"x": 2.45, "elevation": [98.8, 105.0],
                           "label": "Vertical plane at\nthe back of the "
                                    "heel",
                           "label_elevation": 103.1},
            "base_dimension": {"x": [-1.35, 2.45], "elevation": 98.35,
                               "from_elevation": 98.75,
                               "label": "B = 3.80 m"},
            "pressure_axis": {"max_kpa": 40, "ticks": [0, 10, 20, 30]},
            "resultant_label_elevation": 98.55,
            "diagrams": [
                {"title": "Active earth\npressure", "symbol": "$P_A$",
                 "elevation": [105.0, 101.0, 98.8],
                 "kpa": [0.0, 23.4, 30.2],
                 "resultant_kn_per_m": 105.7, "resultant_elevation": 100.95,
                 "color": "#8c6d46"},
                {"title": "Surcharge\npressure", "symbol": "$P_q$",
                 "elevation": [105.0, 98.8], "kpa": [3.7, 3.7],
                 "resultant_kn_per_m": 22.9, "resultant_elevation": 101.90,
                 "color": "#7f7f7f"},
                {"title": "Water\npressure", "symbol": "$P_W$",
                 "elevation": [101.0, 98.8], "kpa": [0.0, 21.6],
                 "resultant_kn_per_m": 23.7, "resultant_elevation": 99.53,
                 "color": "#1f4e79"},
            ],
            "notes": ("Backfill: unit weight 19 kN/m³ above the "
                      "groundwater, 20 kN/m³ below; φ′ = 32°; Rankine "
                      "Ka = 0.307.\n"
                      "Surcharge q = 12 kPa. Water pressure hydrostatic "
                      "below El. 101.0 m behind the wall, none in front.\n"
                      "Passive resistance in front of the toe not counted. "
                      "Pressures and resultants per metre run of wall."),
        },
    },
    "settlement_stress_profile": {
        "shows": "A settlement profile: the soil column with the fill that "
                 "loads it, then the initial vertical effective stress, the "
                 "preconsolidation stress (consolidation tests), the stress "
                 "increase, and the final stress against the "
                 "preconsolidation stress, side by side on one elevation "
                 "axis.",
        "why": [
            "The soil column, with the fill that loads it, stands beside "
            "the stress panels and every panel shares its elevation axis, "
            "so a layer boundary or the groundwater reads straight across.",
            "All four stress panels use ONE scale in kPa (small "
            "multiples): the stress increase is honestly small beside the "
            "effective stress.",
            "Preconsolidation stress from tests is POINTS, never joined "
            "(each is a separate test), named by the test that gave it. "
            "When the analysis uses an assumed design profile, give it as "
            "a line (data 'design': its elevations, kPa and a label naming "
            "it as assumed for design) and keep the test points it was "
            "drawn through; the figure says which it shows.",
            "The last panel sets the final stress (initial plus the "
            "increase) against the preconsolidation stress: where the line "
            "passes the points, the clay is loaded past its "
            "preconsolidation stress. Each line is named on the line, no "
            "legend.",
            "The terms are the references' (initial vertical effective "
            "stress, preconsolidation stress, stress increase) and the "
            "symbols are defined in the panel titles; the notes give the "
            "load, how the stress increase was found and the unit weights.",
        ],
        "code": _SETTLEMENT_CODE,
        "data": {
            "title": "Stresses in the clay under the embankment centreline",
            "column_title": "Soil column",
            "elevation_label": "Elevation (m, site datum)",
            "elevation_range": [2.5, 23.6],
            "ground_elevation": 20.0,
            "groundwater": {"elevation": 18.5, "label": "Groundwater"},
            "layers": [
                {"name": "Embankment fill\n3.0 m high", "top": 23.0,
                 "bottom": 20.0, "color": "#ece8df"},
                {"name": "Medium dense\nsand (SP)", "top": 20.0,
                 "bottom": 16.5, "color": "#f3ead0",
                 "label_elevation": 17.5},
                {"name": "Soft to firm\nclay (CL)", "top": 16.5,
                 "bottom": 6.5, "color": "#e1e6da"},
                {"name": "Dense sand\n(SP)", "top": 6.5, "bottom": 3.0,
                 "color": "#eadfbf"},
            ],
            "stress_axis": {"max_kpa": 205, "ticks": [0, 50, 100, 150]},
            "elevation": [20.0, 18.5, 16.5, 14.0, 11.5, 9.0, 6.5, 3.0],
            "initial_kpa": [0.0, 27.8, 47.1, 65.1, 83.1, 101.1, 119.0,
                            154.7],
            "increase_kpa": [60.0, 56.5, 52.4, 48.0, 44.3, 41.1, 38.4,
                             35.1],
            "final_kpa": [60.0, 84.3, 99.5, 113.1, 127.4, 142.2, 157.4,
                          189.8],
            "preconsolidation": {
                "tests": {"label": "Consolidation\ntests",
                          "elevation": [15.6, 14.0, 11.9, 9.8, 7.6],
                          "kpa": [98, 84, 92, 108, 126]},
            },
            "titles": {
                "initial": "Initial vertical\neffective stress "
                           "$\\sigma'_{v0}$",
                "preconsolidation": "Preconsolidation\nstress "
                                    "$\\sigma'_p$",
                "increase": "Stress increase\n$\\Delta\\sigma$ (2:1 method)",
                "final": "Final stress\nagainst $\\sigma'_p$",
            },
            "final_labels": {
                "initial": "$\\sigma'_{v0}$", "initial_at": 4.5,
                "final": "$\\sigma'_{v0} + \\Delta\\sigma$",
                "final_at": 18.0,
                "preconsolidation": "$\\sigma'_p$",
                "preconsolidation_x": 140, "preconsolidation_at": 9.8,
            },
            "notes": ("Embankment fill 3.0 m high, unit weight 20 kN/m³ "
                      "(q = 60 kPa), 24 m wide at its base;\n"
                      "stress increase by the 2:1 method at the "
                      "centreline.\n"
                      "Unit weights: sand 18.5 kN/m³ above the groundwater, "
                      "19.5 kN/m³ below; clay 17.0 kN/m³; dense sand "
                      "20.0 kN/m³."),
        },
    },
    "pile_design_profile": {
        "shows": "A pile design profile: the soil column with the pile "
                 "drawn, then unit shaft resistance, the axial resistance "
                 "accumulated down the pile, and the axial load in the pile "
                 "with the neutral plane, on one elevation axis.",
        "why": [
            "The soil column has the pile drawn at its true head and toe "
            "elevations (its width cannot be to scale at this size, and "
            "the notes say so); every panel shares the elevation axis.",
            "Unit shaft resistance is one line stepping at the layer "
            "boundaries (each boundary elevation is listed twice in the "
            "data).",
            "The axial resistance accumulates down the pile and steps out "
            "by the toe resistance at the toe; the nominal resistance the "
            "text quotes is a marked point with its value.",
            "The axial load in the pile is the heavy line: the load curve "
            "(permanent load plus drag from the head down) down to the "
            "neutral plane, the resistance curve (toe resistance plus "
            "shaft resistance from the toe up) below it, as UFC 3-220-20 "
            "and GEC-12 name them; the rest of each curve is thin, showing "
            "how the plane was found. The neutral plane is one dashed line "
            "across every panel, named once with its elevation, and the "
            "largest load in the pile is written at the point.",
            "Forces in kN and stresses in kPa, with thousands separators; "
            "the two force panels share one scale and get more width than "
            "the stress panel so their names stand clear of the curves.",
        ],
        "code": _PILE_CODE,
        "data": {
            "title": "Driven pile: shaft resistance, axial resistance and "
                     "the neutral plane",
            "elevation_label": "Elevation (m, site datum)",
            "elevation_range": [-9.6, 13.0],
            "ground_elevation": 12.0,
            "groundwater": {"elevation": 10.8, "label": "Groundwater"},
            "layers": [
                {"name": "Fill (SM)", "top": 12.0, "bottom": 9.5,
                 "color": "#ece8df", "label_elevation": 10.15},
                {"name": "Soft clay\n(CH)", "top": 9.5, "bottom": 1.5,
                 "color": "#e1e6da"},
                {"name": "Medium dense\nsand (SP)", "top": 1.5,
                 "bottom": -6.5, "color": "#f3ead0",
                 "label_elevation": -0.6},
                {"name": "Dense sand\n(SP)", "top": -6.5, "bottom": -9.6,
                 "color": "#eadfbf"},
            ],
            "pile": {"x": [0.83, 0.92], "head_elevation": 12.0,
                     "toe_elevation": -7.5},
            "neutral_plane": {"elevation": -2.85, "max_load_kn": 1638,
                              "drag_force_kn": 538},
            "titles": {"column": "Soil and pile",
                       "unit_shaft": "Unit shaft\nresistance $f_s$",
                       "resistance": "Axial resistance\n(nominal)",
                       "load": "Axial load\nin the pile"},
            "panel_width_ratios": [0.8, 1.0, 1.0],
            "stress_axis": {"max": 110, "ticks": [0, 25, 50, 75, 100],
                            "label": "Stress (kPa)"},
            "force_axis": {"max": 3000, "ticks": [0, 1000, 2000],
                           "label": "Force (kN)"},
            "unit_shaft": {
                "elevation": [12.0, 10.8, 9.5, 9.5, 1.5, 1.5, -6.5, -6.5,
                              -7.5],
                "kpa": [0.0, 8.2, 12.6, 9.0, 22.4, 35.8, 66.8, 91.9,
                        97.7]},
            "resistance": {
                "elevation": _PILE_EL,
                "shaft_kn": [0, 5, 8, 20, 29, 37, 54, 74, 96, 121, 149, 180,
                             213, 230, 260, 323, 393, 469, 551, 639, 734,
                             834, 887, 962, 1039],
                "toe_kn": 1137, "nominal_kn": 2176,
                "labels": [
                    {"x": 330, "elevation": 6.0,
                     "text": "Shaft resistance,\nfrom the head down"},
                    {"x": 2176, "elevation": -8.6, "ha": "right",
                     "color": "#c0504d",
                     "text": "$R_n$ = 2,176 kN\n(toe 1,137 kN)"},
                ]},
            "load": {
                "elevation": _PILE_EL,
                "load_curve_kn": [1100, 1105, 1108, 1120, 1129, 1137, 1154,
                                  1174, 1196, 1221, 1249, 1280, 1313, 1330,
                                  1360, 1423, 1493, 1569, 1651, 1739, 1834,
                                  1934, 1987, 2062, 2139],
                "resistance_curve_kn": [2176, 2171, 2168, 2156, 2147, 2139,
                                        2122, 2102, 2080, 2055, 2027, 1996,
                                        1963, 1946, 1916, 1853, 1783, 1707,
                                        1625, 1537, 1442, 1342, 1289, 1214,
                                        1137],
                "labels": [
                    {"x": 1050, "elevation": 7.0, "ha": "right",
                     "color": "#1f4e79", "text": "Load\ncurve"},
                    {"x": 2250, "elevation": 7.0, "color": "#c0504d",
                     "text": "Resistance\ncurve"},
                    {"x": 60, "elevation": -1.6,
                     "text": "Neutral plane\nEl. −2.85 m"},
                    {"x": 1870, "elevation": -1.9,
                     "text": "$Q_{np}$ =\n1,638 kN"},
                ]},
            "notes": ("400 mm square precast concrete pile, head El. "
                      "12.0 m, toe El. −7.5 m (drawn to its "
                      "elevations, not its width).\n"
                      "Permanent load 1,100 kN; drag force 538 kN. Neutral "
                      "plane where the load curve meets the resistance "
                      "curve\n"
                      "(UFC 3-220-20, GEC-12). Unit shaft resistance by the "
                      "effective stress (β) method: β = 0.35 "
                      "fill, 0.25 clay,\n"
                      "0.40 medium dense sand, 0.55 dense sand; toe "
                      "resistance with $N_t$ = 40."),
        },
    },
}


def example_names() -> List[str]:
    """The ``plot_data`` examples."""
    return list(EXAMPLES)


def code_example_names() -> List[str]:
    """The ``plot_code`` examples (scripts for the locked box)."""
    return list(CODE_EXAMPLES)


def example(name: str) -> dict:
    """A deep copy of one example with its name and kind: a ``plot_data``
    call ``{"name", "kind": "plot_data", "shows", "why", "params"}`` or a
    ``plot_code`` script ``{"name", "kind": "plot_code", "shows", "why",
    "code", "data"}``."""
    if name in EXAMPLES:
        out, kind = copy.deepcopy(EXAMPLES[name]), "plot_data"
    elif name in CODE_EXAMPLES:
        out, kind = copy.deepcopy(CODE_EXAMPLES[name]), "plot_code"
    else:
        raise KeyError(f"no example '{name}'; there are "
                       f"{example_names() + code_example_names()}")
    out["name"] = name
    out["kind"] = kind
    return out


def render_examples(out_dir: str, names: Optional[List[str]] = None) -> dict:
    """Draw the examples with plot_data: ``{name: png_path}``."""
    from profile_figure import render_data_plot
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name in names or example_names():
        res = render_data_plot(**example(name)["params"], interactive=False)
        path = os.path.join(out_dir, f"example_{name}.png")
        with open(path, "wb") as fh:
            fh.write(res.png_bytes)
        paths[name] = path
    return paths


__all__ = ["CODE_EXAMPLES", "EXAMPLES", "code_example_names", "example",
           "example_names", "render_examples"]

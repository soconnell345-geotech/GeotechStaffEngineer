"""Example figures: ``plot_data`` calls that follow the figure checklist.

The owner's choice on 2026-10-15 (OWN-61, option B): a figure helper that
works from a checklist after Tufte and from "some actual good looking example
plots". These are those examples: one per kind of figure a geotechnical
answer or report usually needs, each a complete ``plot_data`` parameter set
on made-up data, with what it shows and why it is drawn that way. The figure
helper (``funhouse_agent/deep/figure_helper.py``) reads them through its
``figure_examples`` tool and adapts the nearest one; ``render_examples``
draws them all (tests, the gallery).

Every number here is invented. Nothing in this file describes a real site.
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional

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
    "depth_profile": {
        "shows": "SPT blow counts against depth in three borings, with the "
                 "water table.",
        "why": [
            "depth_axis: depth increases downward and x sits on top, as on a "
            "boring log.",
            "Each boring's line is named at its deepest point (labels) "
            "instead of a legend; markers show where the tests are, the line "
            "joins tests in ONE boring (a continuous profile).",
            "The water table is a dashed reference line with its date.",
            "The x axis starts at 0 and the units are on both axes.",
        ],
        "params": {
            "title": "SPT N60 with depth",
            "xlabel": "SPT N60 (blows / 0.3 m)",
            "ylabel": "Depth below ground (m)",
            "depth_axis": True, "xlim": [0, 50], "legend": False,
            "hlines": [{"value": 3.2,
                        "label": "Water table (B-2, 12 Sep)"}],
            "series": [
                {"x": [4, 6, 9, 14, 18, 25, 31],
                 "y": [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5],
                 "label": "B-1", "style": "both", "marker": "circle",
                 "color": "#1f4e79",
                 "labels": ["", "", "", "", "", "", "B-1"]},
                {"x": [3, 5, 7, 8, 12, 19, 27, 36],
                 "y": [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0],
                 "label": "B-2", "style": "both", "marker": "square",
                 "color": "#c0504d",
                 "labels": ["", "", "", "", "", "", "", "B-2"]},
                {"x": [6, 10, 13, 17, 22, 30],
                 "y": [1.5, 3.0, 4.5, 6.0, 7.5, 9.0],
                 "label": "B-3", "style": "both", "marker": "triangle",
                 "color": "#4f8f3a",
                 "labels": ["", "", "", "", "", "B-3"]},
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


def example_names() -> List[str]:
    return list(EXAMPLES)


def example(name: str) -> dict:
    """A deep copy of one example: ``{"name", "shows", "why", "params"}``."""
    if name not in EXAMPLES:
        raise KeyError(f"no example '{name}'; there are {example_names()}")
    out = copy.deepcopy(EXAMPLES[name])
    out["name"] = name
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


__all__ = ["EXAMPLES", "example", "example_names", "render_examples"]

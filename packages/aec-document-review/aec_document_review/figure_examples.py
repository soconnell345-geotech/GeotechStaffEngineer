"""Document Review's own worked examples for the figure helper.

The owner's choice on 2026-10-21 (OWN-77, option (b); BACKLOG DR-35): after
the split, Document Review gets general examples (drawings, schedules,
quantities) in place of the geotechnical ones it carried along from the
core (:mod:`aec_agent_core.charts.figure_examples`). They are chosen by
what a document review produces: a plan redrawn to scale from the
dimensions on a sheet, a schedule against the marks counted on the sheets,
comments by sheet, values from a table against the limit a document states,
one value as each sheet gives it, and a submittal log as a timeline.

The set has the core set's shape (``figure_helper._example_set``):
:data:`EXAMPLES` are complete ``plot_data`` calls, :data:`CODE_EXAMPLES`
complete scripts for ``plot_code``'s locked box with the ``data`` each
reads, each with what it shows and why it is drawn that way, following the
figure checklist in ``aec_agent_core.deep.figure_helper``. The page hands
them to its figure helper only behind ``GEOTECH_REVIEW_FIGURE_EXAMPLES``
(``review_build.build_review_agent``); unset, the helper gets the core's.

Every number, sheet and name here is invented. Nothing in this file
describes a real project.
"""

from __future__ import annotations

import math
from typing import Dict

_BLUE, _RED, _GREEN, _GREY, _BROWN = ("#1f4e79", "#c0504d", "#4f8f3a",
                                      "#595959", "#8c6d46")

#: The plan example's grid (m, from grid A/1) and the two points measured.
_GRID_X = {"A": 0.0, "B": 7.5, "C": 15.0, "D": 22.5}
_GRID_Y = {"1": 0.0, "2": 6.0, "3": 12.0}
_STAIR_1, _STAIR_2 = (3.0, 9.5), (19.0, 3.5)
_MEASURED = math.hypot(_STAIR_2[0] - _STAIR_1[0], _STAIR_2[1] - _STAIR_1[1])


def _grid_lines() -> list:
    """The grid as thin chain lines past the outline, each named at its
    end as the sheet names it."""
    out = []
    for name, x in _GRID_X.items():
        out.append({"x": [x, x], "y": [-1.5, 13.5], "label": f"Grid {name}",
                    "style": "line", "line_dash": "dashdot",
                    "line_width": 0.6, "color": "#a6a6a6",
                    "labels": ["", name], "in_legend": False})
    for name, y in _GRID_Y.items():
        out.append({"x": [-1.5, 24.0], "y": [y, y], "label": f"Grid {name}",
                    "style": "line", "line_dash": "dashdot",
                    "line_width": 0.6, "color": "#a6a6a6",
                    "labels": [name, ""], "in_legend": False})
    return out


#: name -> {"shows", "why", "params"}; params is a plot_data call.
EXAMPLES: Dict[str, dict] = {
    "plan_dimensions": {
        "shows": "A floor plan's column grid redrawn to scale from the "
                 "dimensions written on the sheet, the exterior wall line, "
                 "and a distance measured between two points on it.",
        "why": [
            "plan_view: drawn from the sheet's own dimensions, to scale, "
            "read like a drawing: no axes, axis numbers or grid; a scale "
            "bar with the coordinate system under it, and north as the "
            "sheet's north arrow shows it (north_deg).",
            "The grid lines are thin grey chain lines named at their ends "
            "with the sheet's grid names (labels); the wall line is the "
            "one heavy line; no legend.",
            "Each bay dimension is written between its grid lines as the "
            "sheet gives it (annotations), so the drawing can be checked "
            "against the sheet.",
            "The distance is a dashed line between the two named points "
            "with its length written beside it; the footnote says where "
            "the dimensions came from and how the distance was found.",
        ],
        "params": {
            "title": "Level 1 grid redrawn from its dimensions, with the "
                     "distance between the stairs",
            "xlabel": "x and y in metres from grid A/1, the sheet's grid",
            "plan_view": True, "north_deg": 30, "scale_units": "m",
            "legend": False,
            "series": _grid_lines() + [
                {"x": [0, 22.5, 22.5, 0], "y": [0, 0, 12, 12],
                 "label": "Exterior wall line", "style": "polygon",
                 "color": "#262626", "line_width": 2.0},
                {"x": [_STAIR_1[0], _STAIR_2[0]],
                 "y": [_STAIR_1[1], _STAIR_2[1]],
                 "label": "Measured distance", "style": "line",
                 "line_dash": "dash", "color": _RED, "line_width": 1.2},
                {"x": [_STAIR_1[0], _STAIR_2[0]],
                 "y": [_STAIR_1[1], _STAIR_2[1]],
                 "label": "Stair centres", "style": "markers",
                 "marker": "circle", "marker_size": 6.5, "color": _RED,
                 "labels": ["Stair 1", "Stair 2"],
                 "hover": [f"Stair 1<br>x {_STAIR_1[0]:g} m, "
                           f"y {_STAIR_1[1]:g} m",
                           f"Stair 2<br>x {_STAIR_2[0]:g} m, "
                           f"y {_STAIR_2[1]:g} m"]},
            ],
            "annotations": [
                {"text": "7.50 m", "x": 3.75, "y": 12.9, "align": "bottom"},
                {"text": "7.50 m", "x": 11.25, "y": 12.9, "align": "bottom"},
                {"text": "7.50 m", "x": 18.75, "y": 12.9, "align": "bottom"},
                {"text": "6.00 m", "x": -0.4, "y": 3.0, "align": "right"},
                {"text": "6.00 m", "x": -0.4, "y": 9.0, "align": "right"},
                {"text": f"{_MEASURED:.2f} m", "x": 12.5, "y": 7.2,
                 "color": _RED, "halo": True},
            ],
            "footnote": "Grid dimensions as written on the Level 1 plan; "
                        "stair points at the centres of the stair "
                        "enclosures. Distance calculated from the "
                        "dimensions, not scaled off the print.",
        },
    },
    "schedule_count": {
        "shows": "A schedule checked against the sheets: for each door "
                 "type, the number the door schedule lists and the number "
                 "of tags counted on the floor plans, the types that "
                 "disagree labelled.",
        "why": [
            "x names the schedule's own marks in its order; markers only: "
            "the types are separate, so nothing joins them.",
            "Two groups, the schedule and the count off the plans, told "
            "apart by shape and colour (open circle, filled cross); the "
            "legend names the groups and sits below the axis.",
            "Only the types that disagree are labelled, with both numbers; "
            "the count on each sheet is in the hover text.",
            "The axis starts at zero because counts compare magnitudes; "
            "the footnote says what was counted on which sheets, so the "
            "count can be repeated.",
        ],
        "params": {
            "title": "Door schedule against the door tags on the floor "
                     "plans",
            "xlabel": "Door type, as the schedule marks it",
            "ylabel": "Number of doors", "ylim": [0, 16],
            "legend_position": "below",
            "series": [
                {"x": ["D1", "D2", "D3", "D4", "D5", "D6"],
                 "y": [14, 9, 6, 4, 2, 3],
                 "label": "Door schedule", "style": "markers",
                 "marker": "circle", "marker_open": True,
                 "marker_size": 9, "color": _GREY},
                {"x": ["D1", "D2", "D3", "D4", "D5", "D6"],
                 "y": [14, 8, 6, 5, 2, 3],
                 "label": "Tags counted on the floor plans",
                 "style": "markers", "marker": "cross", "marker_size": 6,
                 "color": _RED,
                 "labels": ["", "8 tagged, 9 scheduled", "",
                            "5 tagged, 4 scheduled", "", ""],
                 "hover": ["D1: first floor 8, second floor 6",
                           "D2: first floor 5, second floor 3",
                           "D3: first floor 3, second floor 3",
                           "D4: first floor 3, second floor 2",
                           "D5: first floor 2", "D6: second floor 3"]},
            ],
            "footnote": "Door tags counted on the first and second floor "
                        "plans; the schedule as it stands on the door "
                        "schedule sheet.",
        },
    },
    "comments_by_sheet": {
        "shows": "Quantities by sheet, grouped by discipline: the review "
                 "comments on each sheet of a set, in the set's order, one "
                 "colour and shape per discipline.",
        "why": [
            "x names the sheets in the order of the sheet index, so the "
            "figure reads like the set.",
            "Markers only: each sheet is a separate count; a line would "
            "draw a trend from sheet to sheet that is not there.",
            "The disciplines are groups: one colour and one shape each, "
            "named in a legend below the axis, the same in every figure "
            "of the review.",
            "The axis starts at zero (counts compare magnitudes); the "
            "count the text quotes is labelled on its point.",
        ],
        "params": {
            "title": "Review comments by sheet and discipline",
            "xlabel": "Sheet, in the order of the sheet index",
            "ylabel": "Number of review comments", "ylim": [0, 8],
            "legend_position": "below",
            "series": [
                {"x": ["A-001", "A-101", "A-102", "A-201", "A-301"],
                 "y": [2, 7, 5, 3, 4], "label": "Architectural",
                 "style": "markers", "marker": "circle", "color": _BLUE,
                 "labels": ["", "7 comments", "", "", ""]},
                {"x": ["S-101", "S-201"], "y": [6, 2],
                 "label": "Structural", "style": "markers",
                 "marker": "square", "color": _BROWN},
                {"x": ["M-101"], "y": [3], "label": "Mechanical",
                 "style": "markers", "marker": "triangle", "color": _GREEN},
                {"x": ["E-101"], "y": [1], "label": "Electrical",
                 "style": "markers", "marker": "diamond", "color": _GREY},
            ],
            "footnote": "From the comment log of one review: 33 comments "
                        "in all.",
        },
    },
    "values_against_limit": {
        "shows": "Values from a table against the limit a document "
                 "states: the 28-day strength of each mix in a concrete "
                 "submittal against the strength the specification "
                 "requires, the one below it labelled.",
        "why": [
            "The limit is a reference line (hlines) named on the line with "
            "its value and the document that states it, drawn from the "
            "value the brief gives, never from memory.",
            "x names each mix as the submittal names it, in its order; "
            "markers only, since separate mixes have no line between "
            "them; one group, so no legend.",
            "Only the value the text quotes, the one below the limit, is "
            "labelled; the others are in the hover text.",
            "The axis starts at zero, so the margin to the limit is not "
            "exaggerated; the quantity and unit are in the axis title.",
        ],
        "params": {
            "title": "Concrete mixes in the submittal against the "
                     "specified strength",
            "xlabel": "Mix, as the submittal names it",
            "ylabel": "28-day compressive strength (MPa)", "ylim": [0, 45],
            "hlines": [{"value": 30, "label": "Specified 28-day strength, "
                                              "30 MPa (specification)"}],
            "series": [
                {"x": ["Mix 1", "Mix 2", "Mix 3", "Mix 4", "Mix 5"],
                 "y": [36.5, 34.0, 28.5, 38.0, 31.5],
                 "label": "Mean of the trial batches", "style": "markers",
                 "marker": "circle", "marker_size": 7, "color": _BLUE,
                 "labels": ["", "", "28.5 MPa, below", "", ""],
                 "hover": ["Mix 1: 36.5 MPa", "Mix 2: 34.0 MPa",
                           "Mix 3: 28.5 MPa", "Mix 4: 38.0 MPa",
                           "Mix 5: 31.5 MPa"]},
            ],
            "footnote": "Each value the mean of the submittal's trial "
                        "batches at 28 days.",
        },
    },
    "level_by_sheet": {
        "shows": "One value as each sheet gives it: the Level 2 floor "
                 "level read off the elevations and sections of a set "
                 "against the level on the floor plans, the sheet that "
                 "gives it differently labelled.",
        "why": [
            "What the sheets disagree on is the point, so the vertical "
            "axis spans only 0.20 m around the value and the footnote "
            "says so; a comparison of magnitudes would start at zero.",
            "The floor plans' value is a reference line, named in a note "
            "just under it because the points sit on it; it is the "
            "reference, not a judgement of which sheet is right.",
            "x names the sheets in the set's order; markers only, since the "
            "sheets are separate readings; one group, so no legend.",
            "The sheet that disagrees is labelled with its value; what "
            "each sheet is (an elevation, a section) is in the hover text.",
        ],
        "params": {
            "title": "Level 2 finished floor as each sheet gives it",
            "xlabel": "Sheet",
            "ylabel": "Level 2 (m above the project datum)",
            "ylim": [4.10, 4.30],
            # the points sit on the line, so its name is a note beside it
            "hlines": [{"value": 4.2}],
            "annotations": [{"text": "On the floor plans, 4.200 m",
                             "x": 1.5, "y": 4.19, "align": "top",
                             "color": _GREY}],
            "series": [
                {"x": ["A-201", "A-202", "A-301", "A-302", "A-501"],
                 "y": [4.2, 4.2, 4.25, 4.2, 4.2],
                 "label": "Level 2 as the sheet gives it",
                 "style": "markers", "marker": "circle", "marker_size": 7,
                 "color": _BLUE,
                 "labels": ["", "", "4.250 m on A-301", "", ""],
                 "hover": ["A-201 north elevation: 4.200 m",
                           "A-202 south elevation: 4.200 m",
                           "A-301 building section: 4.250 m",
                           "A-302 wall section: 4.200 m",
                           "A-501 stair section: 4.200 m"]},
            ],
            "footnote": "The vertical axis spans 0.20 m only: the "
                        "difference is the point. Project datum: Level 1 "
                        "finished floor, 0.000 m.",
        },
    },
}


_TIMELINE_CODE = r'''
# A submittal log as a timeline: one row per submittal, each spell with the
# reviewer or with the contractor a bar on a date axis, and the part of a
# review past the period the specification allows drawn in its own colour
# with the days over written above it. Every number is from `data`. The
# axis counts days from the log's first date and is lettered with dates.
from datetime import date, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

d = data
plt.rcParams.update({"font.size": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5})
allowed = d["allowed_days"]
colours = d["colours"]
rows = d["submittals"]
day0 = date.fromisoformat(d["first_date"])
fig, ax = plt.subplots(figsize=(6.5, 3.4))
fig.subplots_adjust(left=0.34, right=0.97, top=0.88, bottom=0.27)


def day(text):
    return (date.fromisoformat(text) - day0).days


last = 0
for i, s in enumerate(rows):
    y = len(rows) - 1 - i                    # the first submittal on top
    for who, start, end in s["spells"]:
        x0, x1 = day(start), day(end)
        last = max(last, x1)
        if who == "reviewer" and x1 - x0 > allowed:
            ax.barh(y, allowed, left=x0, height=0.5,
                    color=colours["reviewer"])
            ax.barh(y, x1 - x0 - allowed, left=x0 + allowed, height=0.5,
                    color=colours["over"])
            # the days over, written just above their own part of the bar
            ax.text((x0 + allowed + x1) / 2, y + 0.28,
                    f"+{x1 - x0 - allowed} d", va="bottom", ha="center",
                    fontsize=7, color=colours["over"])
        else:
            ax.barh(y, x1 - x0, left=x0, height=0.5, color=colours[who])

ax.set_xlim(0, last + 2)
ticks = list(range(0, last + 1, d["tick_days"]))
ax.set_xticks(ticks)
ax.set_xticklabels([(day0 + timedelta(days=t)).strftime("%d %b")
                    for t in ticks])
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{s['id']}  {s['what']}" for s in reversed(rows)])
ax.tick_params(axis="y", length=0)
ax.set_xlabel(d["xlabel"])
ax.grid(axis="x", color="0.88", linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)

fig.legend(handles=[Patch(color=colours[k], label=d["groups"][k])
                    for k in ("reviewer", "over", "contractor")],
           loc="lower left", bbox_to_anchor=(0.02, 0.07), ncol=3,
           frameon=False, fontsize=7.5)
fig.suptitle(d["title"], x=0.02, ha="left", y=0.97, fontsize=9.5,
             fontweight="bold")
fig.text(0.02, 0.01, d["notes"], fontsize=7, ha="left", va="bottom")
save_figure(fig, "submittal_timeline")
'''

#: name -> {"shows", "why", "code", "data"}: figures plot_data cannot draw
#: (bars on a date axis), each a complete script for plot_code's locked box.
CODE_EXAMPLES: Dict[str, dict] = {
    "submittal_timeline": {
        "shows": "A submittal log as a timeline: each submittal a row, its "
                 "spells with the reviewer and with the contractor as bars "
                 "on a date axis, the part of a review past the period the "
                 "specification allows marked with the days over.",
        "why": [
            "Bars on a date axis (plot_data draws no bars): a spell is a "
            "length of time, so its length is the data; rows are named "
            "with the log's own number and title, the first submittal on "
            "top.",
            "Three groups told apart by colour and named once in a legend "
            "below the axis; the days past the allowed period are their "
            "own colour, with the number written just above them, clear "
            "of the bars.",
            "A light vertical grid only, for reading dates; no frame on "
            "three sides and no y axis line.",
            "The note says how days are counted and where the allowed "
            "period comes from, so the overruns can be checked.",
        ],
        "code": _TIMELINE_CODE,
        "data": {
            "title": "Submittal reviews: time with the reviewer and with "
                     "the contractor",
            "xlabel": "Date (2026)",
            "first_date": "2026-03-02", "tick_days": 14,
            "allowed_days": 14,
            "colours": {"reviewer": _BLUE, "over": _RED,
                        "contractor": "#bfbfbf"},
            "groups": {"reviewer": "With the reviewer, within 14 days",
                       "over": "With the reviewer, past 14 days",
                       "contractor": "With the contractor"},
            "submittals": [
                {"id": "03 30 00-01", "what": "Concrete mixes",
                 "spells": [["reviewer", "2026-03-02", "2026-03-13"],
                            ["contractor", "2026-03-13", "2026-03-20"],
                            ["reviewer", "2026-03-20", "2026-03-27"]]},
                {"id": "05 12 00-01", "what": "Steel shop drawings",
                 "spells": [["reviewer", "2026-03-09", "2026-03-30"],
                            ["contractor", "2026-03-30", "2026-04-10"],
                            ["reviewer", "2026-04-10", "2026-04-21"]]},
                {"id": "08 11 13-01", "what": "Doors and frames",
                 "spells": [["reviewer", "2026-03-16", "2026-03-27"],
                            ["contractor", "2026-03-27", "2026-04-17"],
                            ["reviewer", "2026-04-17", "2026-05-05"]]},
                {"id": "07 52 00-01", "what": "Roofing membrane",
                 "spells": [["reviewer", "2026-04-06", "2026-04-17"]]},
                {"id": "23 05 93-01", "what": "Testing and balancing",
                 "spells": [["reviewer", "2026-04-13", "2026-04-30"]]},
            ],
            "notes": "Calendar days from receipt to return. Allowed review "
                     "period: 14 days, as the specification states it.",
        },
    },
}

#: No plot_data redraws of the scripts (``figure_helper.nearest_example``).
REDRAWS: Dict[str, str] = {}

#: The kinds of figure the ``figure_examples`` tool's description names.
TOOL_KINDS = ("a plan redrawn to scale from a sheet's dimensions, a schedule "
              "against the count on the sheets, comments by sheet and "
              "discipline, values against a stated limit, one value as "
              "each sheet gives it")

__all__ = ["CODE_EXAMPLES", "EXAMPLES", "REDRAWS", "TOOL_KINDS"]

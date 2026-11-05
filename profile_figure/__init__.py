"""
profile_figure — generic subsurface profile schematics.

A parameter-driven matplotlib figure of a layered subsurface: strata bands with
names/hatches, the water table, an optional fill/surcharge block, an optional
foundation overlay (pile / micropile / drilled shaft / footing / wall), and
callouts at given elevations or depths.  Returns a PNG (file + base64) sized
for a calc-package page.

Every module family can use it — it renders whatever layers it is handed and
knows no analysis; the vertical scale is true, the horizontal is schematic.

Usage
-----
>>> from profile_figure import render_profile_figure
>>> result = render_profile_figure(
...     layers=[{"name": "Fill", "thickness": 3.0},
...             {"name": "Soft clay", "thickness": 9.0, "settling": True,
...              "description": "su = 25 kPa"},
...             {"name": "Dense sand", "thickness": 8.0}],
...     water_depth=2.0,
...     foundation={"type": "micropile", "diameter": 0.25, "tip_depth": 18.0},
...     annotations=[{"depth": 11.5, "text": "Neutral plane"}],
...     title="MCAC micropile — subsurface profile",
...     output_path="profile.png")
>>> result.output_path is not None
True
"""

from profile_figure.geometry import (AXIS_MODES, FOUNDATION_TYPES,
                                     resolve_profile)
from profile_figure.plotting import (LAYER_COLORS, LAYER_HATCHES,
                                     build_profile_figure)
from profile_figure.profile import render_profile_figure
from profile_figure.results import ProfileFigureResult


def __getattr__(name):
    """The names whose code is in funhouse_agent.charts (INF-31 step 6),
    imported on first use, so importing this package loads nothing of
    funhouse_agent (its first-import notice included). Each is the same
    object as before."""
    if name == "figure_to_png":
        from funhouse_agent.charts.plotting import figure_to_png
        return figure_to_png
    if name in ("render_data_plot", "DataPlotResult"):
        from funhouse_agent.charts import data_plot
        return getattr(data_plot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "render_profile_figure",
    "render_data_plot",
    "DataPlotResult",
    "ProfileFigureResult",
    "resolve_profile",
    "build_profile_figure",
    "figure_to_png",
    "FOUNDATION_TYPES",
    "AXIS_MODES",
    "LAYER_COLORS",
    "LAYER_HATCHES",
]

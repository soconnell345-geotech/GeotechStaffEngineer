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
                                     build_profile_figure, figure_to_png)
from profile_figure.profile import render_profile_figure
from profile_figure.results import ProfileFigureResult

__all__ = [
    "render_profile_figure",
    "ProfileFigureResult",
    "resolve_profile",
    "build_profile_figure",
    "figure_to_png",
    "FOUNDATION_TYPES",
    "AXIS_MODES",
    "LAYER_COLORS",
    "LAYER_HATCHES",
]

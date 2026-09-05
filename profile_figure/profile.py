"""
Subsurface profile schematic — the one public entry point.

``render_profile_figure`` resolves the layer stack (validating it), draws the
section, and returns a :class:`ProfileFigureResult` carrying both a saved PNG
and the base64 of the same image for direct HTML embedding.

Units are SI: meters (m) for every elevation/depth/dimension, kPa for
surcharge.
"""

import base64
import os
from pathlib import Path

from profile_figure.geometry import resolve_profile
from profile_figure.plotting import build_profile_figure, figure_to_png
from profile_figure.results import ProfileFigureResult


def render_profile_figure(layers, *, title="Subsurface Profile",
                          ground_elevation=None, water_elevation=None,
                          water_depth=None, fill=None, surcharge=None,
                          foundation=None, annotations=None, axis="auto",
                          output_path=None, dpi=150, width_in=6.5,
                          height_in=8.0) -> ProfileFigureResult:
    """Draw a layered subsurface profile schematic.

    Parameters
    ----------
    layers : list of dict
        Strata top-down. Each entry: ``name`` plus a ``thickness`` (m) — or a
        ``top_elevation``/``bottom_elevation`` (or ``top_depth``/
        ``bottom_depth``) pair. Optional ``description`` (e.g. "N=8, su=25
        kPa"), ``settling`` (marks a compressible stratum with down-arrows),
        ``color``, ``hatch``. Layers must stack continuously — a gap or
        overlap raises a ValueError naming the offending layer.
    title : str
        Figure title.
    ground_elevation : float, optional
        Original ground surface (m). Defaults to the first layer's
        ``top_elevation``, else 0.0 (so thickness-only input reads as depth
        below ground).
    water_elevation, water_depth : float, optional
        Water table, as an elevation or a depth below ground. Omit for none.
    fill : dict or float, optional
        Fill/embankment placed ABOVE the ground surface:
        ``{'name', 'thickness'}`` (or ``top_elevation``), plus the same
        optional keys as a layer. A bare number is read as a thickness.
    surcharge : dict or float, optional
        Surface surcharge drawn as load arrows: ``{'pressure': kPa, 'label'}``
        (a bare number is the pressure).
    foundation : dict, optional
        Foundation overlay: ``{'type': 'pile'|'micropile'|'drilled_shaft'|
        'footing'|'wall', 'head_elevation'|'head_depth', 'tip_elevation'|
        'tip_depth'|'length', 'diameter'|'width', 'label'}``. Head defaults to
        the top of the section.
    annotations : list of dict, optional
        Callouts: ``{'elevation'|'depth', 'text', 'side': 'left'|'right'}``
        — e.g. the neutral plane, a sample depth, a design assumption.
    axis : str
        ``'auto'`` (default; depth when the input is thickness/depth-based,
        elevation otherwise), ``'elevation'`` or ``'depth'``. Presentation
        only — the geometry is identical either way.
    output_path : str, optional
        Where to write the PNG. The parent directory is created. When omitted,
        only the in-memory PNG/base64 is returned.
    dpi : int
        Raster resolution. 150 (default) suits a calc-package page.
    width_in, height_in : float
        Figure size in inches.

    Returns
    -------
    ProfileFigureResult
        Resolved geometry + the PNG (``output_path``, ``png_bytes``,
        ``image_base64``, ``data_uri()``, ``img_tag()``).
    """
    resolved = resolve_profile(
        layers, ground_elevation=ground_elevation, fill=fill,
        water_elevation=water_elevation, water_depth=water_depth,
        surcharge=surcharge, foundation=foundation, annotations=annotations,
        axis=axis)

    fig, _ax = build_profile_figure(resolved, title=title, width_in=width_in,
                                    height_in=height_in)
    try:
        png = figure_to_png(fig, dpi=dpi)
        width_px, height_px = _png_size(png)
    finally:
        # Long-running hosts (the web app) must not accumulate open figures.
        from geotech_common.plotting import get_pyplot
        get_pyplot().close(fig)

    saved = None
    if output_path:
        saved = str(Path(os.path.abspath(output_path)))
        Path(saved).parent.mkdir(parents=True, exist_ok=True)
        Path(saved).write_bytes(png)

    return ProfileFigureResult(
        layers=resolved["layers"],
        ground_elevation=resolved["ground"],
        base_elevation=resolved["base"],
        axis=resolved["axis"],
        image_base64=base64.b64encode(png).decode("ascii"),
        png_bytes=png,
        width_px=width_px,
        height_px=height_px,
        output_path=saved,
        water_elevation=resolved["water"],
        fill=resolved["fill"],
        surcharge=resolved["surcharge"],
        foundation=resolved["foundation"],
        annotations=resolved["annotations"],
        title=title,
        warnings=resolved["warnings"],
    )


def _png_size(png: bytes):
    """Pixel width/height from the PNG IHDR chunk (no image library needed)."""
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
        return 0, 0
    width = int.from_bytes(png[16:20], "big")
    height = int.from_bytes(png[20:24], "big")
    return width, height

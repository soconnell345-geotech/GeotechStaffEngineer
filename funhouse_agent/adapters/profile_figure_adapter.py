"""Profile figure adapter — flat dict → profile_figure → saved PNG + echo.

Two methods. ``subsurface_profile`` draws a layered subsurface schematic
(strata, water table, fill/surcharge, an optional pile/shaft/footing/wall
overlay, callouts); ``plot_data`` draws a generic XY / depth data plot (SPT
vs depth, settlement vs time, a sweep). Both SAVE a PNG on the real
filesystem and return a ready-to-paste ``html_img_tag`` — the "figures for
any report" pair (owner feedback 2026-09-11: calc packages lacked figures,
and the agent had no general plotting tool to make one).

Context economy: the response carries the file path, the resolved geometry and
a ready-to-paste ``html_img_tag`` — NOT the base64 image, which would be tens
of thousands of characters of tool output. ``html_to_pdf`` inlines a real local
PNG path itself, so the agent never needs the base64; ``include_base64=true``
is there for the rare caller that does.
"""

import os
from datetime import datetime

from funhouse_agent.adapters import (apply_aliases, check_nested_keys,
                                     check_nested_list, reject_unknown_params,
                                     require_params, write_plotly_sidecar)
from funhouse_agent._fileio import (default_output_dir, resolve_output_path,
                                    save_verified)

_VALID = {
    "layers", "title", "ground_elevation", "water_elevation", "water_depth",
    "fill", "surcharge", "foundation", "annotations", "axis", "output_path",
    "dpi", "width_in", "height_in", "include_base64",
}


def _default_output_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(default_output_dir(), f"subsurface_profile_{ts}.png")


# Keys of the nested dicts profile_figure reads with .get. Anything else
# was silently left out (GM-58): a series' 'name' left the legend reading
# 'Series 1' and a reference line's 'text' drew it unnamed. The layers and
# fill are left open: an analysis' own layer dicts (unit_weight, phi, ...)
# draw as they are.
_SURCHARGE_KEYS = ("pressure", "label")
_FOUNDATION_KEYS = ("type", "head_elevation", "head_depth", "tip_elevation",
                    "tip_depth", "length", "embedded_length", "thickness",
                    "diameter", "width", "size", "label")
_ANNOTATION_KEYS = ("elevation", "depth", "text", "label", "side")
_SERIES_KEYS = ("x", "y", "label", "style", "color", "marker", "marker_size",
                "marker_edge", "marker_open", "line_dash", "line_width",
                "fill", "hatch", "fill_opacity", "labels", "hover",
                "in_legend")
_REF_LINE_KEYS = ("value", "label")
# Words the near-miss matcher cannot place. Suggestions only, never renamed.
_KEY_HINTS = {
    "name": ("label",), "legend": ("label", "in_legend"),
    "showlegend": ("in_legend",), "mode": ("style",), "kind": ("style",),
    "dash": ("line_dash",), "linestyle": ("line_dash",),
    "ls": ("line_dash",), "lw": ("line_width",), "width": ("line_width",),
    "size": ("marker_size",), "ms": ("marker_size",),
    "opacity": ("fill_opacity",), "alpha": ("fill_opacity",),
    "colour": ("color",), "text": ("labels", "label"),
    "pointlabels": ("labels",), "hovertext": ("hover",),
    "q": ("pressure",), "elev": ("elevation",), "z": ("elevation",),
    "note": ("text",), "position": ("side",), "y": ("value",),
    "x": ("value",), "diam": ("diameter",), "b": ("width",),
    "tip": ("tip_elevation", "tip_depth"),
    "head": ("head_elevation", "head_depth"),
    # a foundation's top is its head, never its tip (the spelling is close)
    "top": ("head_elevation", "head_depth"),
    "topelevation": ("head_elevation",), "topelevationm": ("head_elevation",),
    "topdepth": ("head_depth",), "topdepthm": ("head_depth",),
    "elevation": ("head_elevation", "tip_elevation"),
}


def _run_subsurface_profile(params: dict) -> dict:
    from profile_figure import render_profile_figure

    reject_unknown_params(params, _VALID, method="subsurface_profile")
    require_params(params, ["layers"], method="subsurface_profile",
                   valid=_VALID)
    for key, valid in (("surcharge", _SURCHARGE_KEYS),
                       ("foundation", _FOUNDATION_KEYS)):
        if isinstance(params.get(key), dict):
            check_nested_keys(params[key], valid,
                              method="subsurface_profile", label=key,
                              hints=_KEY_HINTS)
    check_nested_list(params.get("annotations"), _ANNOTATION_KEYS,
                      method="subsurface_profile", label="annotations",
                      hints=_KEY_HINTS)

    output_path = params.get("output_path") or _default_output_path()
    output_path = resolve_output_path(str(output_path))
    if not output_path.lower().endswith((".png", ".jpg", ".jpeg")):
        output_path += ".png"

    # Render in memory, then write through save_verified so a /Workspace
    # target goes via the durable Databricks workspace API and every write is
    # read-back-verified (same contract as save_file / the calc packages).
    result = render_profile_figure(
        params["layers"],
        title=params.get("title", "Subsurface Profile"),
        ground_elevation=params.get("ground_elevation"),
        water_elevation=params.get("water_elevation"),
        water_depth=params.get("water_depth"),
        fill=params.get("fill"),
        surcharge=params.get("surcharge"),
        foundation=params.get("foundation"),
        annotations=params.get("annotations"),
        axis=params.get("axis", "auto"),
        dpi=params.get("dpi", 150),
        width_in=params.get("width_in", 6.5),
        height_in=params.get("height_in", 8.0),
    )
    saved = save_verified(output_path, result.png_bytes)
    abs_path = saved.get("saved", os.path.abspath(output_path))

    profile = result.to_dict()
    for key in ("image_base64", "output_path", "warnings"):
        profile.pop(key, None)  # carried on the response, not duplicated here

    response = {
        "status": "success" if saved.get("file_exists") else "error",
        "analysis_type": "Subsurface profile figure",
        "output_path": abs_path,
        "file_exists": bool(saved.get("file_exists")),
        "file_size_bytes": saved.get("file_size_bytes", 0),
        "width_px": result.width_px,
        "height_px": result.height_px,
        "profile": profile,
        "summary": result.summary(),
        "html_img_tag": (
            f'<img src="{abs_path}" alt="{result.title}" '
            f'style="width:100%;max-width:640px;">'),
        "embed_note": (
            "The figure is saved as a PNG. The web app shows it under your "
            "reply (a file written outside the conversation folder is copied "
            "in at the end of the turn) -- do not put a markdown image link "
            "to a local path in the reply; it cannot display. To put it in an "
            "HTML report, paste html_img_tag as-is: html_to_pdf reads a real "
            "local PNG path and embeds it for you. Never write '[image]' or an "
            "inline <svg>."),
    }
    if result.warnings:
        response["warnings"] = list(result.warnings)
    for key in ("error", "rescue_path", "workspace_api_note"):
        if saved.get(key):
            response[key] = saved[key]
    if params.get("include_base64"):
        response["image_base64"] = result.image_base64
    return response


_VALID_PLOT = {
    "series", "title", "xlabel", "ylabel", "depth_axis", "logx", "logy",
    "hlines", "vlines", "grid", "legend", "output_path", "dpi", "width_in",
    "height_in", "include_base64", "interactive", "equal_aspect", "xlim",
    "ylim", "legend_position", "plan_view", "north_deg", "scale_units",
    "x_on_top",
}


def _default_plot_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(default_output_dir(), f"data_plot_{ts}.png")


#: The axis-label spellings with an underscore (matplotlib's own
#: set_xlabel/xlabel have none; agents often write one).
_PLOT_ALIASES = {"x_label": "xlabel", "y_label": "ylabel"}


def _run_plot_data(params: dict) -> dict:
    from profile_figure import render_data_plot

    params = apply_aliases(params, _PLOT_ALIASES)
    reject_unknown_params(params, _VALID_PLOT, method="plot_data")
    require_params(params, ["series"], method="plot_data", valid=_VALID_PLOT)
    check_nested_list(params["series"], _SERIES_KEYS, method="plot_data",
                      label="series", hints=_KEY_HINTS)
    for key in ("hlines", "vlines"):           # a bare number is a line too
        lines = params.get(key)
        for i, spec in enumerate(lines if isinstance(lines, list) else []):
            if isinstance(spec, dict):
                check_nested_keys(spec, _REF_LINE_KEYS, method="plot_data",
                                  label=f"{key}[{i}]", hints=_KEY_HINTS)

    output_path = params.get("output_path") or _default_plot_path()
    output_path = resolve_output_path(str(output_path))
    if not output_path.lower().endswith((".png", ".jpg", ".jpeg")):
        output_path += ".png"

    result = render_data_plot(
        params["series"],
        title=params.get("title", ""),
        xlabel=params.get("xlabel", ""),
        ylabel=params.get("ylabel", ""),
        depth_axis=bool(params.get("depth_axis", False)),
        logx=bool(params.get("logx", False)),
        logy=bool(params.get("logy", False)),
        hlines=params.get("hlines"),
        vlines=params.get("vlines"),
        grid=bool(params.get("grid", True)),
        legend=params.get("legend"),
        dpi=params.get("dpi", 150),
        width_in=params.get("width_in", 6.5),
        height_in=params.get("height_in"),
        interactive=bool(params.get("interactive", True)),
        equal_aspect=bool(params.get("equal_aspect", False)),
        xlim=params.get("xlim"),
        ylim=params.get("ylim"),
        legend_position=params.get("legend_position"),
        plan_view=bool(params.get("plan_view", False)),
        north_deg=params.get("north_deg"),
        scale_units=str(params.get("scale_units") or "m"),
        x_on_top=bool(params.get("x_on_top", False)),
    )
    saved = save_verified(output_path, result.png_bytes)
    abs_path = saved.get("saved", os.path.abspath(output_path))
    alt = result.title or "data plot"
    response = {
        "status": "success" if saved.get("file_exists") else "error",
        "analysis_type": "Data plot figure",
        "output_path": abs_path,
        "file_exists": bool(saved.get("file_exists")),
        "file_size_bytes": saved.get("file_size_bytes", 0),
        "width_px": result.width_px,
        "height_px": result.height_px,
        "series": result.series,
        "summary": result.summary(),
        "html_img_tag": (
            f'<img src="{abs_path}" alt="{alt}" '
            f'style="width:100%;max-width:640px;">'),
        "embed_note": (
            "Saved as a PNG plus an INTERACTIVE chart (unless "
            "interactive=false). The web app shows the interactive chart "
            "under your reply -- zoom and hover, one card; the PNG is the "
            "report copy (a file written outside the conversation folder is "
            "copied in at the end of the turn). Paste html_img_tag into "
            "report HTML and html_to_pdf embeds the real local PNG. Do not "
            "put a markdown image link to a local path in the reply; it "
            "cannot display. Never write '[image]' or an inline <svg>."),
    }
    # The interactive twin, written beside the PNG as <name>.plotly.json —
    # what the chat actually renders (webapp/core.classify_artifact -> the
    # st.plotly_chart card). Additive and best-effort: the PNG response above
    # stands whether or not this works.
    if saved.get("file_exists") and getattr(result, "figure", None) is not None:
        try:
            figure_json = result.figure.to_json()
        except Exception:                                  # noqa: BLE001
            figure_json = None
        side = write_plotly_sidecar(abs_path, figure_json)
        if side:
            response.update(side)
    if result.warnings:
        response["warnings"] = list(result.warnings)
    for key in ("error", "rescue_path", "workspace_api_note"):
        if saved.get(key):
            response[key] = saved[key]
    if params.get("include_base64"):
        response["image_base64"] = result.image_base64
    return response


METHOD_REGISTRY = {
    "subsurface_profile": _run_subsurface_profile,
    "plot_data": _run_plot_data,
}


METHOD_INFO = {
    "subsurface_profile": {
        "category": "Figure",
        "brief": ("Draw a subsurface profile schematic (PNG): soil layers with "
                  "names/descriptions, water table, optional fill + surcharge, "
                  "optional pile/micropile/drilled-shaft/footing/wall overlay, "
                  "and callouts (e.g. neutral plane) at any elevation or depth. "
                  "Saves the file and returns an <img> tag ready for a report."),
        "parameters": {
            "layers": {
                "type": "array", "required": True,
                "description": (
                    "Strata TOP-DOWN. Each dict: 'name' plus 'thickness' (m) — "
                    "or a 'top_elevation'/'bottom_elevation' (or 'top_depth'/"
                    "'bottom_depth') pair. Optional: 'description' (short "
                    "properties string shown under the name, e.g. 'su = 25 "
                    "kPa, N=8'), 'settling' (true marks a compressible layer "
                    "with down-arrows — use it for downdrag/consolidation "
                    "figures), 'color', 'hatch'. Layers must stack "
                    "continuously; a gap or overlap is an error."),
            },
            "title": {"type": "str", "required": False,
                      "default": "Subsurface Profile",
                      "description": "Figure title (name the project/boring)."},
            "ground_elevation": {
                "type": "float", "required": False,
                "description": ("Original ground surface (m). Defaults to the "
                                "first layer's top_elevation, else 0.0 — with "
                                "thickness-only layers the axis then reads as "
                                "depth below ground.")},
            "water_elevation": {"type": "float", "required": False,
                                "description": "Water-table elevation (m)."},
            "water_depth": {"type": "float", "required": False,
                            "description": ("Water-table depth below ground "
                                            "(m). Alternative to "
                                            "water_elevation.")},
            "fill": {"type": "object", "required": False,
                     "description": ("Fill/embankment ABOVE existing grade: "
                                     "{'name', 'thickness'} (or "
                                     "'top_elevation'), optional 'settling'. A "
                                     "bare number is read as a thickness.")},
            "surcharge": {"type": "object", "required": False,
                          "description": ("Surface surcharge drawn as load "
                                          "arrows: {'pressure': kPa, 'label'} "
                                          "— a bare number is the pressure.")},
            "foundation": {
                "type": "object", "required": False,
                "description": (
                    "Foundation overlay: {'type': 'pile'|'micropile'|"
                    "'drilled_shaft'|'footing'|'wall', 'diameter' (or 'width' "
                    "for a footing), 'tip_elevation' or 'tip_depth' or "
                    "'length', optional 'head_elevation'/'head_depth' "
                    "(defaults to the top of the section), optional 'label'}. "
                    "Drawn to true DEPTH; its width is exaggerated so it is "
                    "visible, with the real dimension in the label."),
            },
            "annotations": {
                "type": "array", "required": False,
                "description": ("Callouts: [{'elevation' or 'depth', 'text', "
                                "'side': 'left'|'right'}] — the neutral plane, "
                                "a sample depth, top of rock, a design "
                                "assumption. Drawn as a dashed line across the "
                                "section with a labelled leader."),
            },
            "axis": {"type": "str", "required": False, "default": "auto",
                     "allowed_values": ["auto", "elevation", "depth"],
                     "description": ("Vertical-axis labelling. 'auto' = depth "
                                     "for thickness-only input, elevation once "
                                     "any elevation is given. Presentation "
                                     "only — geometry is unchanged.")},
            "output_path": {"type": "str", "required": False,
                            "description": ("PNG path. A bare filename lands in "
                                            "the working folder. "
                                            "Auto-generated if omitted.")},
            "dpi": {"type": "int", "required": False, "default": 150,
                    "description": "Raster resolution; 150 suits a report page."},
            "width_in": {"type": "float", "required": False, "default": 6.5,
                         "description": "Figure width (inches)."},
            "height_in": {"type": "float", "required": False, "default": 8.0,
                          "description": "Figure height (inches)."},
            "include_base64": {
                "type": "bool", "required": False, "default": False,
                "description": ("Also return the PNG as base64. Rarely needed "
                                "and VERY large (tens of thousands of "
                                "characters) — html_to_pdf embeds a local PNG "
                                "path for you, so leave this off.")},
        },
        "returns": {
            "status": "success or error.",
            "output_path": ("The saved PNG: its name in the working folder (an "
                            "absolute path only outside one)."),
            "file_exists": ("True if the file was verified on disk. Trust this "
                            "field — do NOT try to verify with agent-side "
                            "filesystem tools (they may be sandboxed)."),
            "file_size_bytes": "Size of the saved PNG.",
            "width_px": "Rendered width in pixels.",
            "height_px": "Rendered height in pixels.",
            "profile": ("Resolved geometry actually drawn: every layer's top/"
                        "bottom elevation and thickness, ground/base elevation, "
                        "water table, fill, surcharge, foundation, callouts. "
                        "Check it against your inputs before citing the figure."),
            "summary": "Human-readable description of the figure.",
            "html_img_tag": ("Ready-to-paste <img> tag referencing the saved "
                             "PNG — drop it straight into report HTML for "
                             "html_to_pdf."),
            "warnings": ("Non-fatal QC notes, e.g. a pile tip below the "
                         "deepest layer or a water table off the section."),
            "image_base64": "Only when include_base64=true.",
        },
    },
    "plot_data": {
        "category": "Figure",
        "brief": ("Generic data plot: one or more x/y series — an SPT or "
                  "CPT profile against elevation, settlement vs time, load "
                  "vs displacement, a parameter sweep or method "
                  "comparison, the cases of a "
                  "sensitivity study, a plan view of borings and a "
                  "footprint drawn to scale. Shows in chat as an "
                  "INTERACTIVE chart (zoom/hover) and saves a PNG for "
                  "reports, with an <img> tag ready to paste."),
        "parameters": {
            "series": {
                "type": "array", "required": True,
                "description": (
                    "[{'x': [...], 'y': [...], 'label': str, optional "
                    "'style': 'both'|'line'|'markers'|'step'|'polygon', "
                    "'color'}]. x and y are equal-length lists. Optional per "
                    "series: 'marker' (circle, square, triangle, "
                    "triangle-down, diamond, x, cross, star, hexagon), "
                    "'marker_size' (points), 'marker_edge' (outline colour), "
                    "'marker_open' (hollow), 'line_dash' (solid, dash, dot, "
                    "dashdot), 'line_width', 'labels' (one name per point, "
                    "written beside it, '' to skip one), 'hover' (one text "
                    "per point, shown on hover in the interactive chart; "
                    "'<br>' breaks a line), 'in_legend' (false hides it). "
                    "Style 'polygon' closes the outline (a footprint, a site "
                    "boundary, a zone); 'fill': true or a colour, "
                    "'fill_opacity' (default 0.2), 'hatch' ('/', '\\', "
                    "'x', '-', '|', '+', '.') to mark a zone over what lies "
                    "under it. x given as names (cases, "
                    "scenarios, borings) puts them along the x axis in "
                    "order: draw separate cases with style 'markers', since "
                    "a line between unrelated cases shows a trend that is "
                    "not there. Name points on the figure with 'labels' "
                    "rather than a legend entry each."),
            },
            "equal_aspect": {
                "type": "bool", "required": False, "default": False,
                "description": (
                    "Draw one unit across as long as one unit up: set it "
                    "for every plan view, site map, footprint or section "
                    "drawn to scale (x and y both lengths in the same unit). "
                    "The figure takes the data's own proportions and the "
                    "legend moves outside the map.")},
            "plan_view": {
                "type": "bool", "required": False, "default": False,
                "description": (
                    "Draw a site plan as a drawing, not a chart: to scale, "
                    "no axes, axis numbers or grid, a scale bar with xlabel "
                    "(name the coordinate system there) under it, and a "
                    "north arrow from north_deg. Use it for a plan of a "
                    "site, compound or property; draw the property line as "
                    "a polygon with line_dash 'dashdot'.")},
            "north_deg": {
                "type": "float", "required": False,
                "description": (
                    "Direction of north on the page, degrees clockwise from "
                    "straight up, as the plan the data came from (or the "
                    "project's other plans) shows it. Not given: drawn "
                    "straight up, and the result says so.")},
            "scale_units": {"type": "str", "required": False, "default": "m",
                            "description": "Units written on the scale bar."},
            "xlim": {"type": "array", "required": False,
                     "description": ("[start, end] of the x axis, kept in the "
                                     "order given: [100, 0.001] runs it from "
                                     "high to low (on a log axis too). Else "
                                     "fitted.")},
            "ylim": {"type": "array", "required": False,
                     "description": ("[start, end] of the y axis, kept in the "
                                     "order given (with depth_axis, y runs "
                                     "downward whatever the order). Else "
                                     "fitted.")},
            "x_on_top": {"type": "bool", "required": False, "default": False,
                         "description": ("Put the x axis along the top, as a "
                                         "boring log reads, on a profile "
                                         "against elevation (depth_axis does "
                                         "this by itself).")},
            "legend_position": {
                "type": "str", "required": False, "default": "auto",
                "allowed_values": ["auto", "inside", "outside", "below"],
                "description": ("Where the legend goes; auto = outside a "
                                "drawing to scale, inside otherwise.")},
            "title": {"type": "str", "required": False, "default": "",
                      "description": "Figure title."},
            "xlabel": {"type": "str", "required": False, "default": "",
                       "description": "X-axis label WITH units."},
            "ylabel": {"type": "str", "required": False, "default": "",
                       "description": "Y-axis label WITH units."},
            "depth_axis": {"type": "bool", "required": False, "default": False,
                           "description": (
                               "A profile has elevation on y by default "
                               "(leave this false, y = elevations): borings "
                               "start at different ground levels. True when "
                               "y is depth (only when no elevations are "
                               "given; say so in ylabel) or settlement: y "
                               "increases DOWNWARD and the x axis sits at "
                               "the top (boring-log convention).")},
            "logx": {"type": "bool", "required": False, "default": False,
                     "description": "Log-scale x axis."},
            "logy": {"type": "bool", "required": False, "default": False,
                     "description": "Log-scale y axis."},
            "hlines": {"type": "array", "required": False,
                       "description": ("Horizontal reference lines: "
                                       "[{'value': y, 'label': str}] — e.g. a "
                                       "water table or an allowable value.")},
            "vlines": {"type": "array", "required": False,
                       "description": ("Vertical reference lines: "
                                       "[{'value': x, 'label': str}].")},
            "grid": {"type": "bool", "required": False, "default": True,
                     "description": "Draw a light grid."},
            "legend": {"type": "bool", "required": False,
                       "description": ("Show the legend. Default: when more "
                                       "than one series is in it (reference "
                                       "lines are named on the line).")},
            "output_path": {"type": "str", "required": False,
                            "description": ("PNG path. A bare filename lands in "
                                            "the working folder. "
                                            "Auto-generated if omitted.")},
            "interactive": {
                "type": "bool", "required": False, "default": True,
                "description": ("Also write the interactive chart beside the "
                                "PNG (<name>.plotly.json) — that is what the "
                                "chat displays. Set false only when a static "
                                "image is all you want.")},
            "dpi": {"type": "int", "required": False, "default": 150,
                    "description": "Raster resolution."},
            "width_in": {"type": "float", "required": False, "default": 6.5,
                         "description": "Figure width (inches)."},
            "height_in": {"type": "float", "required": False,
                          "description": ("Figure height (inches). Default "
                                          "4.5, or the data's proportions "
                                          "with equal_aspect.")},
            "include_base64": {
                "type": "bool", "required": False, "default": False,
                "description": ("Also return the PNG as base64 — VERY large; "
                                "html_to_pdf embeds the saved path for you, "
                                "so leave this off.")},
        },
        "returns": {
            "status": "success or error.",
            "output_path": ("The saved PNG: its name in the working folder (an "
                            "absolute path only outside one)."),
            "file_exists": "True if the file was verified on disk — trust it.",
            "file_size_bytes": "Size of the saved PNG.",
            "width_px": "Rendered width in pixels.",
            "height_px": "Rendered height in pixels.",
            "series": ("Each series as drawn: label, n points, x/y ranges "
                       "(x_categories when x named cases), points_named."),
            "summary": "Human-readable description of the figure.",
            "html_img_tag": ("Ready-to-paste <img> tag referencing the saved "
                             "PNG — drop it straight into report HTML for "
                             "html_to_pdf."),
            "plotly_json_path": ("Name of the interactive chart written beside "
                                 "the PNG; this is what the chat shows. Absent "
                                 "when interactive=false."),
            "warnings": "Non-fatal notes (e.g. non-finite points dropped).",
        },
    },
}

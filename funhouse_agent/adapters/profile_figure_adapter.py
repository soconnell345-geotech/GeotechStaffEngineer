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

from funhouse_agent.adapters import reject_unknown_params, require_params
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


def _run_subsurface_profile(params: dict) -> dict:
    from profile_figure import render_profile_figure

    reject_unknown_params(params, _VALID, method="subsurface_profile")
    require_params(params, ["layers"], method="subsurface_profile",
                   valid=_VALID)

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
            "The figure is saved as a PNG — the chat UI renders a saved PNG "
            "inline automatically, so point the user at output_path rather "
            "than describing the figure. To put it in an HTML report, paste "
            "html_img_tag as-is: html_to_pdf reads a real local PNG path and "
            "embeds it for you. Never write '[image]' or an inline <svg>."),
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
    "height_in", "include_base64",
}


def _default_plot_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(default_output_dir(), f"data_plot_{ts}.png")


def _run_plot_data(params: dict) -> dict:
    from profile_figure import render_data_plot

    reject_unknown_params(params, _VALID_PLOT, method="plot_data")
    require_params(params, ["series"], method="plot_data", valid=_VALID_PLOT)

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
        height_in=params.get("height_in", 4.5),
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
            "Saved as a PNG: the chat UI renders it inline, and html_to_pdf "
            "embeds the real local path when you paste html_img_tag into "
            "report HTML. Never write '[image]' or an inline <svg>."),
    }
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
            "output_path": "Absolute path of the saved PNG.",
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
        "brief": ("Generic data plot (PNG): one or more x/y series — SPT or "
                  "CPT vs depth, settlement vs time, load vs displacement, a "
                  "parameter sweep or method comparison. Saves the file and "
                  "returns an <img> tag ready for a report."),
        "parameters": {
            "series": {
                "type": "array", "required": True,
                "description": ("[{'x': [...], 'y': [...], 'label': str, "
                                "optional 'style': 'both'|'line'|'markers'|"
                                "'step', optional 'color'}]. x and y are equal-"
                                "length numeric lists."),
            },
            "title": {"type": "str", "required": False, "default": "",
                      "description": "Figure title."},
            "xlabel": {"type": "str", "required": False, "default": "",
                       "description": "X-axis label WITH units."},
            "ylabel": {"type": "str", "required": False, "default": "",
                       "description": "Y-axis label WITH units."},
            "depth_axis": {"type": "bool", "required": False, "default": False,
                           "description": ("True for a depth profile: y "
                                           "increases DOWNWARD and the x axis "
                                           "sits at the top (boring-log "
                                           "convention).")},
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
                                       "than one series or a labelled "
                                       "reference line.")},
            "output_path": {"type": "str", "required": False,
                            "description": ("PNG path. A bare filename lands in "
                                            "the working folder. "
                                            "Auto-generated if omitted.")},
            "dpi": {"type": "int", "required": False, "default": 150,
                    "description": "Raster resolution."},
            "width_in": {"type": "float", "required": False, "default": 6.5,
                         "description": "Figure width (inches)."},
            "height_in": {"type": "float", "required": False, "default": 4.5,
                          "description": "Figure height (inches)."},
            "include_base64": {
                "type": "bool", "required": False, "default": False,
                "description": ("Also return the PNG as base64 — VERY large; "
                                "html_to_pdf embeds the saved path for you, "
                                "so leave this off.")},
        },
        "returns": {
            "status": "success or error.",
            "output_path": "Absolute path of the saved PNG.",
            "file_exists": "True if the file was verified on disk — trust it.",
            "file_size_bytes": "Size of the saved PNG.",
            "width_px": "Rendered width in pixels.",
            "height_px": "Rendered height in pixels.",
            "series": "Each series as drawn: label, n points, x/y ranges.",
            "summary": "Human-readable description of the figure.",
            "html_img_tag": ("Ready-to-paste <img> tag referencing the saved "
                             "PNG — drop it straight into report HTML for "
                             "html_to_pdf."),
            "warnings": "Non-fatal notes (e.g. non-finite points dropped).",
        },
    },
}

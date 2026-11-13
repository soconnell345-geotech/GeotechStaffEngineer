"""``plot_data``, the generic x/y data plot both pages draw with.

The method, its parameters and its call, moved here from the geotechnical
``profile_figure`` adapter (INF-31, the app split's step 6) so a page
without the geotechnical dispatch draws through the same code:
:func:`run_plot_data` checks the arguments and draws (through
:mod:`funhouse_agent.charts.data_plot`), and :func:`call_plot_data` runs it
as ``call_agent('profile_figure', 'plot_data', ...)`` did, with the checks
and notes of :mod:`funhouse_agent.method_call` (the output path moved into
the working folder and said so, ``defaults_used``, paths relative to the
working folder).

``reference_chart`` (a standard chart's lines drawn under the data) is the
geotechnical app's: it is offered only where reference charts are
registered (:func:`funhouse_agent.charts.register_reference_charts`). The
adapter keeps ``plot_data`` under ``profile_figure`` for ``call_agent``,
drawing through :func:`run_plot_data`.
"""

import copy
import os
from datetime import datetime

from aec_agent_core._fileio import (default_output_dir, resolve_output_path,
                                    save_verified, write_plotly_sidecar)
from aec_agent_core.arg_checks import (apply_aliases, check_nested_keys,
                                       check_nested_list, declare_aliases,
                                       declare_when_absent,
                                       reject_unknown_params, require_params)
from aec_agent_core.charts import reference_charts

#: Use the registered reference charts' overlay (the default).
_REGISTERED = object()


def _image_name(path: str) -> str:
    """``path`` as an image file name: one already ending .png, .jpg or
    .jpeg is kept; a document extension (.html, .pdf, .svg) is replaced, not
    stacked (``plot.html`` made ``plot.html.png``; REF-02)."""
    if path.lower().endswith((".png", ".jpg", ".jpeg")):
        return path
    root, ext = os.path.splitext(path)
    if ext.lower() in (".html", ".htm", ".pdf", ".svg", ".json"):
        path = root
    return path + ".png"


_SERIES_KEYS = ("x", "y", "label", "style", "color", "marker", "marker_size",
                "marker_edge", "marker_open", "line_dash", "line_width",
                "fill", "hatch", "fill_opacity", "labels", "hover",
                "in_legend", "fill_to_x", "fill_to_y")
_REF_LINE_KEYS = ("value", "label")
#: A plot_data note's keys (profile_figure.data_plot.ANNOTATION_KEYS).
_NOTE_KEYS = ("text", "x", "y", "coords", "point", "align", "box", "color",
              "font_size", "halo", "rotation")
# Words the near-miss matcher cannot place. Suggestions only, never renamed.
_KEY_HINTS = {
    "name": ("label", "text"), "legend": ("label", "in_legend"),
    "showlegend": ("in_legend",), "mode": ("style",), "kind": ("style",),
    "dash": ("line_dash",), "linestyle": ("line_dash",),
    "ls": ("line_dash",), "lw": ("line_width",), "width": ("line_width",),
    "size": ("marker_size", "font_size"), "ms": ("marker_size",),
    "opacity": ("fill_opacity",), "alpha": ("fill_opacity",),
    "colour": ("color",), "text": ("labels", "label"),
    "pointlabels": ("labels",), "hovertext": ("hover",),
    "q": ("pressure",), "elev": ("elevation",), "z": ("elevation",),
    "note": ("text",), "position": ("side", "align"), "y": ("value",),
    # a plot_data note's own words, as matplotlib and Plotly spell them
    "label": ("text",), "txt": ("text",), "string": ("text",),
    "ha": ("align",), "va": ("align",), "anchor": ("align",),
    "horizontalalignment": ("align",), "verticalalignment": ("align",),
    "xanchor": ("align",), "yanchor": ("align",),
    "textposition": ("align",), "alignment": ("align",),
    "xy": ("point",), "arrow": ("point",), "arrowto": ("point",),
    "leader": ("point",), "target": ("point",), "marks": ("point",),
    "pointsto": ("point",), "xytext": ("x", "y"),
    "bbox": ("box",), "background": ("box",), "bgcolor": ("box",),
    "boxed": ("box",), "frame": ("box",),
    "fontsize": ("font_size",), "textsize": ("font_size",),
    "xycoords": ("coords",), "textcoords": ("coords",), "xref": ("coords",),
    "yref": ("coords",), "coordinates": ("coords",), "units": ("coords",),
    "x": ("value",), "diam": ("diameter",), "b": ("width",),
    "tip": ("tip_elevation", "tip_depth"),
    "head": ("head_elevation", "head_depth"),
    # a foundation's top is its head, never its tip (the spelling is close)
    "top": ("head_elevation", "head_depth"),
    "topelevation": ("head_elevation",), "topelevationm": ("head_elevation",),
    "topdepth": ("head_depth",), "topdepthm": ("head_depth",),
    "elevation": ("head_elevation", "tip_elevation"),
}


#: plot_data's parameters, without the geotechnical app's reference_chart.
VALID_PLOT = frozenset({
    "series", "title", "xlabel", "ylabel", "depth_axis", "logx", "logy",
    "hlines", "vlines", "grid", "legend", "output_path", "dpi", "width_in",
    "height_in", "include_base64", "interactive", "equal_aspect", "xlim",
    "ylim", "legend_position", "plan_view", "north_deg", "scale_units",
    "x_on_top", "annotations", "panels", "contour", "footnote",
})


def _default_plot_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(default_output_dir(), f"data_plot_{ts}.png")


#: The axis-label spellings with an underscore (matplotlib's own
#: set_xlabel/xlabel have none; agents often write one).
_PLOT_ALIASES = {"x_label": "xlabel", "y_label": "ylabel"}


#: The length units the scale bar is written in (GM-205); the bar's
#: length is in the data's own units, so the unit is only its label.
SCALE_UNITS = ("m", "km", "cm", "mm", "ft", "in", "yd", "mi")


def _scale_units(value) -> str:
    """``scale_units`` as one of ``SCALE_UNITS``, else refused by name."""
    if value is None or value == "":
        return "m"
    if isinstance(value, str) and value.strip().lower() in SCALE_UNITS:
        return value.strip().lower()
    raise ValueError(
        f"plot_data: scale_units {value!r} is not a length unit the scale "
        f"bar is written in; use one of {', '.join(SCALE_UNITS)}.")


def run_plot_data(params: dict, overlay=_REGISTERED) -> dict:
    """Draw ``plot_data`` (the method; :func:`call_plot_data` is the call
    with its checks and notes). ``overlay(params) -> (params, n_added,
    said)`` draws ``reference_chart`` under the data; by default the
    registered reference charts' (:func:`funhouse_agent.charts.
    register_reference_charts`), and without one the argument is refused
    by name."""
    from aec_agent_core.charts.data_plot import render_data_plot

    if overlay is _REGISTERED:
        provider = reference_charts()
        overlay = provider.overlay if provider is not None else None
    valid = VALID_PLOT | ({"reference_chart"} if overlay is not None
                          else set())
    params = apply_aliases(params, _PLOT_ALIASES)
    reject_unknown_params(params, valid, method="plot_data")
    if params.get("panels") is None and params.get("contour") is None:
        require_params(params, ["series"], method="plot_data", valid=valid)
    check_nested_list(params.get("series"), _SERIES_KEYS,
                      method="plot_data", label="series", hints=_KEY_HINTS)
    panels = params.get("panels")
    for k, panel in enumerate(panels if isinstance(panels, list) else []):
        if isinstance(panel, dict):     # a panel's series and notes too
            check_nested_list(panel.get("series"), _SERIES_KEYS,
                              method="plot_data",
                              label=f"panels[{k}].series", hints=_KEY_HINTS)
            check_nested_list(panel.get("annotations"), _NOTE_KEYS,
                              method="plot_data",
                              label=f"panels[{k}].annotations",
                              hints=_KEY_HINTS)
    check_nested_list(params.get("annotations"), _NOTE_KEYS,
                      method="plot_data", label="annotations",
                      hints=_KEY_HINTS)
    for key in ("hlines", "vlines"):           # a bare number is a line too
        lines = params.get(key)
        for i, spec in enumerate(lines if isinstance(lines, list) else []):
            if isinstance(spec, dict):
                check_nested_keys(spec, _REF_LINE_KEYS, method="plot_data",
                                  label=f"{key}[{i}]", hints=_KEY_HINTS)

    _scale_units(params.get("scale_units"))
    n_added, chart_said = 0, None
    if params.get("reference_chart"):
        params, n_added, chart_said = overlay(params)

    # Every figure saved reaches the user: one that only holds a place is
    # refused before anything is written (BACKLOG GM-190).
    from aec_agent_core.plot_box.figures import STAND_IN_NOTE, stand_in
    why = stand_in(**_plot_words(params))
    if why:
        raise ValueError(f"plot_data drew no figure: {why}. {STAND_IN_NOTE}")

    output_path = params.get("output_path") or _default_plot_path()
    output_path = resolve_output_path(str(output_path))
    output_path = _image_name(output_path)

    try:
        result = render_data_plot(
            params.get("series"),
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
            scale_units=_scale_units(params.get("scale_units")),
            x_on_top=bool(params.get("x_on_top", False)),
            annotations=params.get("annotations"),
            contour=params.get("contour"),
            panels=params.get("panels"),
            footnote=params.get("footnote") or "",
        )
    except ValueError as exc:
        if not n_added:
            raise
        raise ValueError(_own_places(str(exc), n_added)) from None
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
        "series": result.series[n_added:],
        "summary": result.summary(),
        "html_img_tag": (
            f'<img src="{abs_path}" alt="{alt}" '
            f'style="width:100%;max-width:640px;">'),
        "embed_note": (
            "Saved as a PNG plus an INTERACTIVE chart (unless "
            "interactive=false, or the warnings say this figure has none: "
            "the chat then shows the PNG). The web app shows the interactive chart "
            "under your reply -- zoom and hover, one card; the PNG is the "
            "report copy (a file written outside the conversation folder is "
            "copied in at the end of the turn). Paste html_img_tag into "
            "report HTML and html_to_pdf embeds the real local PNG. Do not "
            "put a markdown image link to a local path in the reply; it "
            "cannot display. Never write '[image]' or an inline <svg>."),
    }
    if response["status"] == "success":
        # a look at the figure, in a later step, before it is used: what
        # plot_code asks (BACKLOG GM-152, live wave w12)
        from aec_agent_core.plot_box import next_step
        response["next"] = next_step(None, renamed_drafts=False,
                                     checked_in="warnings")
    if result.annotations:
        response["annotations"] = list(result.annotations)
    if result.panels:
        response["panels"] = list(result.panels)
    if result.contour:
        response["contour"] = dict(result.contour)
    if chart_said:
        response["reference_chart"] = chart_said
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
        response["warnings"] = [_own_places(w, n_added)
                                for w in result.warnings]
    for key in ("error", "rescue_path", "workspace_api_note"):
        if saved.get(key):
            response[key] = saved[key]
    if params.get("include_base64"):
        response["image_base64"] = result.image_base64
    return response


def _plot_words(params: dict) -> dict:
    """The words a ``plot_data`` figure carries, for the stand-in check
    (:func:`funhouse_agent.plot_box.figures.stand_in`): its title, axis
    names, series names, and the point names, notes and reference-line
    names."""
    series = [s for s in params.get("series") or [] if isinstance(s, dict)]
    notes, titles, axes = [], [params.get("title")], \
        [params.get("xlabel"), params.get("ylabel")]
    raw_notes = list(params.get("annotations") or [])
    panels = params.get("panels")
    for panel in (panels if isinstance(panels, list) else []):
        if not isinstance(panel, dict):
            continue
        series += [s for s in panel.get("series") or []
                   if isinstance(s, dict)]
        titles.append(panel.get("title"))
        axes.append(panel.get("xlabel"))
        raw_notes += list(panel.get("annotations") or [])
        for key in ("bands", "dimensions", "arrows"):
            notes += [item.get("label") or item.get("text")
                      for item in panel.get(key) or []
                      if isinstance(item, dict)]
    if isinstance(params.get("contour"), dict):
        axes.append(params["contour"].get("label"))
    notes.append(params.get("footnote"))
    for s in series:
        if isinstance(s.get("labels"), (list, tuple)):
            notes += list(s["labels"])
    notes += [a.get("text") for a in raw_notes if isinstance(a, dict)]
    for key in ("hlines", "vlines"):
        lines = params.get(key)
        notes += [spec.get("label") for spec in
                  (lines if isinstance(lines, list) else [])
                  if isinstance(spec, dict)]
    return {"titles": titles, "axes": axes,
            "series": [s.get("label") for s in series], "notes": notes}


def _own_places(text: str, n_added: int) -> str:
    """``text`` with each ``series[i]`` numbered as the caller numbered it:
    a reference chart's series are drawn first, under the caller's."""
    import re

    def own(m):
        i = int(m.group(1))
        return (f"series[{i - n_added}]" if i >= n_added
                else "the reference chart")
    return re.sub(r"series\[(\d+)\]", own, text) if n_added else text


#: plot_data's entry in its method table: what describe_method shows and
#: what ``defaults_used`` reads.
PLOT_DATA_INFO = {
    "category": "Figure",
    "brief": ("Generic data plot: one or more x/y series — an SPT or "
              "CPT profile against elevation, settlement vs time, load "
              "vs displacement, a parameter sweep or method "
              "comparison, the cases of a "
              "sensitivity study, a plan view of borings and a "
              "footprint drawn to scale, panels on one elevation axis, "
              "contours. Shows in chat as an "
              "INTERACTIVE chart (zoom/hover) and saves a PNG for "
              "reports, with an <img> tag ready to paste."),
    "parameters": {
        "series": {
            "type": "array", "required": False,
            "description": (
                "Required unless panels or contour is given. "
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
                "under it. 'fill_to_x' / 'fill_to_y' (a number): filled "
                "to that base line. x given as names (cases, "
                "scenarios, borings) puts them along the x axis in "
                "order: draw separate cases with style 'markers', since "
                "a line between unrelated cases shows a trend that is "
                "not there. Name points on the figure with 'labels' "
                "rather than a legend entry each."),
        },
        "annotations": {
            "type": "array", "required": False,
            "description": (
                "Text placed where you put it: a note, the name of a "
                "zone or a line, a reading at a point. It is never "
                "moved (point names keep clear of it). [{'text', 'x', "
                "'y'}] in data units (a case's name for x on a case "
                "axis), or with 'coords': 'axes' in fractions of the "
                "plot (0, 0 bottom left; 1, 1 top right). Optional: "
                "'point': [x, y] in data units, the point the note "
                "marks, drawn as a leader arrow from the text; 'align': "
                "the side of the text that stands at x, y (center, "
                "left, right, top, bottom, or e.g. 'top left'); 'box': "
                "true for a plain white box behind the text (or a "
                "colour); 'halo': true for a thin ring of the "
                "background round the letters, so hatching or a grid "
                "behind them stops short (the PNG); 'color'; "
                "'font_size' (pt); 'rotation' (degrees). A new line or "
                "'<br>' breaks a line. The warnings say when a note "
                "overlaps other lettering, a line runs through it or it "
                "stands outside the plot."),
        },
        "panels": {
            "type": "array", "required": False,
            "description": (
                "In place of series: panels side by side on ONE vertical "
                "axis (ylabel, ylim; hlines cross them all). [{'kind': "
                "'chart', 'section' (to scale, no x axis) or 'column' "
                "(layer bands, x 0-1), 'title', 'series', 'xlabel', "
                "'xlim', 'annotations', 'vlines', 'width', 'share_x' (a "
                "name: one x scale), 'bands': [{top, bottom, label, color, "
                "hatch}], 'dimensions': [{from, to, offset, text}], "
                "'arrows': [{from, to, text}], 'grid'}]."),
        },
        "contour": {
            "type": "object", "required": False,
            "description": (
                "Contours under the series: {'x', 'y', 'z', 'label' "
                "(quantity, unit), 'levels', 'filled', 'line_labels'}; z "
                "as rows over an x-y grid, or one value per scattered "
                "point."),
        },
        "footnote": {
            "type": "str", "required": False, "default": "",
            "description": "Small text under the figure."},
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
                        "description": "Length unit written on the scale bar: "
                                       + ", ".join(SCALE_UNITS) + "."},
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
                 "description": ("Draw a light grid (with panels: each "
                                 "panel's own 'grid', default none).")},
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
                   "(x_categories when x named cases), points_named; its "
                   "panel with panels."),
        "panels": ("With panels, each as drawn: kind, title, x_range, "
                   "width_in."),
        "contour": ("With contour: label, from (grid or points), levels, "
                    "z_range, filled."),
        "annotations": ("Each note as drawn: text, at ([x, y] as "
                        "given), coords, and point for a note with a "
                        "leader. Absent when there are none."),
        "summary": "Human-readable description of the figure.",
        "html_img_tag": ("Ready-to-paste <img> tag referencing the saved "
                         "PNG — drop it straight into report HTML for "
                         "html_to_pdf."),
        "plotly_json_path": ("Name of the interactive chart written beside "
                             "the PNG; this is what the chat shows. Absent "
                             "when interactive=false."),
        "warnings": ("Non-fatal notes (e.g. non-finite points dropped, "
                     "a note that overlaps other lettering, a figure "
                     "with no interactive chart)."),
    },
}

# What a call with panels does not use (GM-315): these single-plot inputs'
# defaults are then left out of ``defaults_used``; a left-out ``series``
# is listed only where panels or contour stands in for it.
for _name in ("equal_aspect", "plan_view", "north_deg", "scale_units",
              "x_on_top", "legend_position", "logx", "logy", "grid"):
    PLOT_DATA_INFO["parameters"][_name]["applies_when"] = {"panels": "unset"}
PLOT_DATA_INFO["parameters"]["series"]["applies_when"] = [
    {"panels": "set"}, {"contour": "set"}]

declare_aliases({"plot_data": PLOT_DATA_INFO}, {"plot_data": _PLOT_ALIASES})
# What a left-out input with no default stands for (BACKLOG GM-165): north.
declare_when_absent({"plot_data": PLOT_DATA_INFO}, {
    "plot_data": {
        "north_deg": ("north straight up",
                      "the source plan's north was not given; the result "
                      "says so"),
        # left out only with panels or contour (GM-315); refused otherwise
        "series": ("no series of the figure's own",
                   "panels or contour gives what is drawn")},
})


def method_info() -> dict:
    """plot_data's entry, with ``reference_chart`` where reference charts
    are registered (a copy)."""
    info = copy.deepcopy(PLOT_DATA_INFO)
    provider = reference_charts()
    if provider is not None:
        info["parameters"]["reference_chart"] = copy.deepcopy(
            provider.parameter)
        info["returns"]["reference_chart"] = provider.returns
    return info


class _PlotMethod:
    """plot_data as a method table, the shape a method call reads."""

    __name__ = __name__

    @property
    def METHOD_INFO(self):                                  # noqa: N802
        return {"plot_data": method_info()}

    @property
    def METHOD_REGISTRY(self):                              # noqa: N802
        return {"plot_data": run_plot_data}


PLOT_METHOD = _PlotMethod()


def call_plot_data(parameters: dict, attachments: dict = None) -> dict:
    """``plot_data`` called as ``call_agent('profile_figure', 'plot_data',
    parameters)`` called it: its result, or ``{"error": ...}``."""
    from aec_agent_core.method_call import run_method
    return run_method(PLOT_METHOD, "profile_figure", "plot_data",
                      parameters, attachments)

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

import copy
import os
from datetime import datetime

from funhouse_agent.adapters import (apply_aliases, check_nested_keys,
                                     check_nested_list, declare_aliases,
                                     reject_unknown_params, require_params,
                                     write_plotly_sidecar)
from funhouse_agent._fileio import (default_output_dir, resolve_output_path,
                                    save_verified)
# plot_data itself is the chart stack's (INF-31 step 6): its method, its
# parameters and the nested-key lists it shares with subsurface_profile come
# from there; this adapter adds reference_chart (the reference library's
# chart lines) and keeps plot_data under profile_figure for call_agent.
from funhouse_agent import charts as _charts
from funhouse_agent.charts import plot_tool as _plot_tool
from funhouse_agent.charts.plot_tool import (  # noqa: F401
    _KEY_HINTS, _NOTE_KEYS, _PLOT_ALIASES, _REF_LINE_KEYS, _SERIES_KEYS,
    SCALE_UNITS, _default_plot_path, _image_name, _own_places, _plot_words,
    _scale_units,
)

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
    output_path = _image_name(output_path)

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


def _chart_lines():
    """``geotech_references.chart_lines``: standard chart lines as the
    references print them (BACKLOG GM-153), or None where the installed
    library has none (geotech-references 1.4.1 and before)."""
    import importlib
    try:
        return importlib.import_module("geotech_references.chart_lines")
    except Exception:                                      # noqa: BLE001
        return None


_VALID_PLOT = set(_plot_tool.VALID_PLOT) | (
    {"reference_chart"} if _chart_lines() else set())

#: How a chart's lines are drawn: in grey, thinner than the data, named on
#: the line (never in the legend), as a printed chart draws them.
_CHART_INK = "#555555"


def _data_range(series, axis, log):
    """(low, high) of the numbers on one axis of the data, or None."""
    vals = []
    for s in series if isinstance(series, list) else []:
        for v in (s.get(axis) or []) if isinstance(s, dict) else []:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f == f and f not in (float("inf"), float("-inf")) \
                    and (f > 0 or not log):
                vals.append(f)
    return (min(vals), max(vals)) if vals else None


def _chart_extent(frame, given, data, log):
    """What a chart's lines are drawn over: the axis limits given, else the
    printed chart's frame widened to take in the data."""
    if isinstance(given, (list, tuple)) and len(given) == 2:
        try:
            return sorted(float(v) for v in given)
        except (TypeError, ValueError):
            pass
    lo, hi = float(frame[0]), float(frame[1])
    if data:
        if log:
            if data[0] < lo:
                lo = data[0] / 1.5
            if data[1] > hi:
                hi = data[1] * 1.5
        else:
            pad = 0.05 * (max(hi, data[1]) - min(lo, data[0]))
            if data[0] < lo:
                lo = data[0] - pad
            if data[1] > hi:
                hi = data[1] + pad
    return [lo, hi]


def _data_points(series):
    """The caller's numeric (x, y) points."""
    pts = []
    for s in series if isinstance(series, list) else []:
        if not isinstance(s, dict):
            continue
        for x, y in zip(s.get("x") or [], s.get("y") or []):
            try:
                pts.append((float(x), float(y)))
            except (TypeError, ValueError):
                continue
    return pts


def _edge_gap(pt, outline, scale):
    """``(inside, distance)``: whether ``pt`` is inside ``outline``, and how
    far it is from the nearest edge, in fractions of the chart's width and
    height."""
    sx, sy = scale
    x, y = pt
    hit, gap = False, float("inf")
    for (x0, y0), (x1, y1) in zip(outline, outline[1:] + outline[:1]):
        if ((y0 > y) != (y1 > y)
                and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0)):
            hit = not hit
        dx, dy = (x1 - x0) / sx, (y1 - y0) / sy
        px, py = (x - x0) / sx, (y - y0) / sy
        den = dx * dx + dy * dy
        t = 0.0 if den == 0 else max(0.0, min(1.0, (px * dx + py * dy) / den))
        gap = min(gap, ((px - t * dx) ** 2 + (py - t * dy) ** 2) ** 0.5)
    return hit, gap


def _letter(params: dict):
    """An 8 pt letter's (width, height) as fractions of the plot's: about
    0.06 in by 0.14 in, on a plot the figure's size less about 0.9 in for
    the axes' numbers and names."""
    try:
        w_in = float(params.get("width_in") or 6.5)
        h_in = float(params.get("height_in") or 4.5)
    except (TypeError, ValueError):
        w_in, h_in = 6.5, 4.5
    return 0.06 / max(w_in - 0.9, 1.0), 0.14 / max(h_in - 0.9, 1.0)


def _footprint(text, at, scale, letter, pad=0.003):
    """The box a name written centred at ``at`` takes, ``pad`` (a fraction
    of the plot) round it: ``(x0, x1, y0, y1)`` in data units. A marker's
    room is a pad of 0.012."""
    half_w = (letter[0] * len(text) / 2.0 + pad) * scale[0]
    half_h = (letter[1] / 2.0 + pad) * scale[1]
    return (at[0] - half_w, at[0] + half_w, at[1] - half_h, at[1] + half_h)


def _covers(box, pt):
    return box[0] <= pt[0] <= box[1] and box[2] <= pt[1] <= box[3]


def _crosses(p, q, box):
    """Whether the segment p-q passes through ``box`` (Liang-Barsky)."""
    x0, x1, y0, y1 = box
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0),
                   (dy, y1 - p[1])):
        if pk == 0:
            if qk < 0:
                return False
            continue
        t = qk / pk
        if pk < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _spot_in_zone(text, outline, at, data, segments, scale, letter):
    """The spot nearest the printed one where the whole name fits inside its
    zone with no data point under it and no line through it, or None."""
    edges = list(zip(outline, outline[1:] + outline[:1]))
    xs = [p[0] for p in outline]
    ys = [p[1] for p in outline]
    best, spot = float("inf"), None
    n = 30
    for i in range(n + 1):
        for j in range(n + 1):
            c = (min(xs) + (max(xs) - min(xs)) * i / n,
                 min(ys) + (max(ys) - min(ys)) * j / n)
            if not _edge_gap(c, outline, scale)[0]:
                continue
            box = _footprint(text, c, scale, letter)
            room = _footprint(text, c, scale, letter, pad=0.012)
            if (any(_covers(room, p) for p in data)
                    or any(_crosses(p, q, box) for p, q in edges)
                    or any(_crosses(p, q, box) for p, q in segments)):
                continue
            d = ((c[0] - at[0]) / scale[0]) ** 2 \
                + ((c[1] - at[1]) / scale[1]) ** 2
            if d < best:
                best, spot = d, c
    return spot


def _data_segments(series):
    """The caller's drawn line segments (every style but markers)."""
    segs = []
    for s in series if isinstance(series, list) else []:
        if not isinstance(s, dict) or str(s.get("style") or "both") \
                .lower() == "markers":
            continue
        try:
            pts = [(float(x), float(y))
                   for x, y in zip(s.get("x") or [], s.get("y") or [])]
        except (TypeError, ValueError):
            continue
        segs += list(zip(pts, pts[1:]))
    return segs


def _clear_spot(outline, data, scale):
    """The point inside ``outline`` farthest from both its edges and the
    data: where a leader to the zone points without pointing at a point."""
    sx, sy = scale
    xs = [p[0] for p in outline]
    ys = [p[1] for p in outline]
    best, spot = -1.0, None
    n = 24
    for i in range(1, n):
        for j in range(1, n):
            pt = (min(xs) + (max(xs) - min(xs)) * i / n,
                  min(ys) + (max(ys) - min(ys)) * j / n)
            inside, gap = _edge_gap(pt, outline, scale)
            if not inside:
                continue
            for x, y in data:
                gap = min(gap, (((pt[0] - x) / sx) ** 2
                                + ((pt[1] - y) / sy) ** 2) ** 0.5)
            if gap > best:
                best, spot = gap, pt
    return spot


def _field_spots(extent, texts, width_in=6.5):
    """Where names moved off the data go: side by side along the top of the
    empty field above the PI = LL line, where no soil plots (PI = LL - PL),
    so each leader runs down clear of the other names; a full row goes on
    in a row below, set against the line, clear of the leaders from above.
    ``None`` for a name with no room before the line."""
    (x0, x1), (y0, y1) = extent["x"], extent["y"]
    w, h = x1 - x0, y1 - y0
    # a letter at 8 pt is about 0.06 in wide; the plot is the figure's
    # width less about 0.9 in for the y axis's numbers and name
    letter = 0.06 / max(float(width_in or 6.5) - 0.9, 1.0)
    left = x0 + 0.03 * w
    x, y = left, y1 - 0.07 * h
    rows, spots = [(y, [])], [None] * len(texts)
    for i, text in enumerate(texts):
        width = letter * w * len(text)
        if x > left and x + width > y - 0.03 * h:
            x, y = left, y - 0.08 * h           # the row is full
            rows.append((y, []))
        if x + width > y - 0.03 * h or y < y0:
            continue                            # it would reach the line
        rows[-1][1].append((i, x, width))
        x += width + 0.015 * w
    for k, (y, row) in enumerate(rows):
        shift = 0.0
        if k and row:
            shift = (y - 0.03 * h) - (row[-1][1] + row[-1][2])
        for i, x, _width in row:
            spots[i] = (x + shift, y)
    return spots


def _reference_overlay(params: dict):
    """``params`` with a reference chart's lines and zones put under the
    caller's series (drawn first) and its zone names after the caller's
    notes, the number of series added, and what the result says about the
    chart."""
    mod = _chart_lines()
    name = params.get("reference_chart")
    if mod is None:
        raise ValueError("plot_data: reference_chart needs a "
                         "geotech-references with chart_lines (released "
                         "after 1.4.1); the one installed has none.")
    try:
        probe = mod.chart_lines(name)
    except ValueError as exc:
        raise ValueError(f"plot_data: reference_chart: {exc}") from None
    log_x = bool(probe["x"]["log"])
    xr = _chart_extent(probe["frame"]["x"], params.get("xlim"),
                       _data_range(params["series"], "x", log_x), log_x)
    yr = _chart_extent(probe["frame"]["y"], params.get("ylim"),
                       _data_range(params["series"], "y", False), False)
    try:
        chart = mod.chart_lines(name, x_range=xr, y_range=yr)
    except ValueError as exc:
        raise ValueError(f"plot_data: reference_chart: {exc}") from None
    out = dict(params)
    added = []
    for line in chart["lines"]:
        pts = line["points"]
        if len(pts) < 2:
            continue
        tip = f"{line['name']}: {line['equation']}"
        added.append({
            "x": [p[0] for p in pts], "y": [p[1] for p in pts],
            "label": line["name"], "style": "line", "color": _CHART_INK,
            "line_dash": line["dash"], "line_width": 1.0,
            "in_legend": False, "hover": [tip] * len(pts),
            "labels": [line["name"] if p == line["label_at"] else ""
                       for p in pts]})
    notes, moved, left_out, in_zone = [], [], [], []
    data = _data_points(params["series"])
    scale = (chart["extent"]["x"][1] - chart["extent"]["x"][0],
             chart["extent"]["y"][1] - chart["extent"]["y"][0])
    letter = _letter(params)
    segments = _data_segments(params["series"]) + [
        (tuple(p), tuple(q)) for line in chart["lines"]
        for p, q in zip(line["points"], line["points"][1:])]
    for zone in chart["zones"]:
        outline = [tuple(p) for p in zone.get("outline") or []]
        if zone.get("hatched") and len(outline) >= 3:
            added.append({
                "x": [p[0] for p in outline], "y": [p[1] for p in outline],
                "label": zone["name"], "style": "polygon", "hatch": "/",
                "color": _CHART_INK, "line_width": 0.6, "in_legend": False,
                "hover": [zone["name"]] * len(outline)})
        if not zone.get("label_at"):
            continue
        at = tuple(zone["label_at"])
        note = {"text": zone["name"], "x": at[0], "y": at[1],
                "color": _CHART_INK, "font_size": 8, "halo": True}
        if len(outline) < 3 or not any(
                _covers(_footprint(zone["name"], at, scale, letter,
                                   pad=0.012), p)
                for p in data):
            notes.append(note)          # nothing under its printed place
            continue
        # A point lies under the name: written at the nearest spot in its
        # zone that is clear, else above the PI = LL line with a leader.
        spot = _spot_in_zone(zone["name"], outline, at, data, segments,
                             scale, letter)
        if spot:
            note["x"], note["y"] = round(spot[0], 3), round(spot[1], 3)
            notes.append(note)
            in_zone.append(zone["name"])
            continue
        spot = _clear_spot(outline, data, scale)
        if spot:
            note["x"], note["y"] = round(spot[0], 3), round(spot[1], 3)
        moved.append(note)
    # in the order of the points they mark, left to right (the lower
    # first), so the leaders fan out without crossing
    moved.sort(key=lambda n: (n["x"], n["y"]))
    spots = _field_spots(chart["extent"], [n["text"] for n in moved],
                         params.get("width_in") or 6.5)
    for note, spot in zip(moved, spots):
        if spot is None:
            left_out.append(note["text"])
            continue
        note["point"] = [note["x"], note["y"]]
        note["x"], note["y"] = spot
        note["align"] = "left"
        notes.append(note)
    # The chart goes under the data: its series first, and each of the
    # caller's keeps the colour and name its own place gives it.
    from funhouse_agent.charts.data_plot import _PALETTE
    own = []
    for i, s in enumerate(params["series"]):
        s = dict(s)
        s["color"] = s.get("color") or _PALETTE[i % len(_PALETTE)]
        s["label"] = s.get("label") or f"Series {i + 1}"
        own.append(s)
    out["series"] = added + own
    if notes:
        out["annotations"] = list(params.get("annotations") or []) + notes
    if not params.get("xlabel"):
        out["xlabel"] = chart["x"]["name"]
    if not params.get("ylabel"):
        out["ylabel"] = chart["y"]["name"]
    if params.get("xlim") is None:
        out["xlim"] = chart["extent"]["x"]
    if params.get("ylim") is None and not params.get("depth_axis"):
        out["ylim"] = chart["extent"]["y"]
    said = {
        "name": chart["name"], "title": chart["title"],
        "source": chart["source"],
        "also_printed_in": [f"{s['document']}, {s['where']}, p. "
                            f"{s['printed_page']}"
                            for s in chart["also_printed_in"]],
        "lines": [{"name": l["name"], "equation": l["equation"]}
                  for l in chart["lines"] if len(l["points"]) >= 2],
        "zones_named": [n["text"] for n in notes],
        "drawn_over": chart["extent"], "notes": chart["notes"]}
    shifted = [n["text"] for n in moved if n["text"] not in left_out]
    if in_zone or shifted:
        said["zone_names_moved"] = " ".join(filter(None, [
            (", ".join(in_zone) + ": a point lies where the name is printed; "
             "it is written at the nearest clear spot in its zone.")
            if in_zone else "",
            (", ".join(shifted) + ": no clear spot in the zone; the name is "
             "written above the PI = LL line, with a leader to the zone.")
            if shifted else ""]))
    if left_out:
        said["zone_names_left_out"] = (
            ", ".join(left_out) + ": data lie in the zone and the field "
            "above the PI = LL line is not on the plot; name the zone with "
            "an annotation where it is clear of the data.")
    if log_x and not params.get("logx"):
        out["logx"] = True
        said["axis_note"] = "logx set: the chart's x axis is logarithmic."
    return out, len(added), said










def _run_plot_data(params: dict) -> dict:
    """plot_data, drawn by the chart stack's method
    (:func:`funhouse_agent.charts.plot_tool.run_plot_data`), with this
    adapter's ``reference_chart`` where the reference library has chart
    lines."""
    overlay = _reference_overlay if "reference_chart" in _VALID_PLOT else None
    return _plot_tool.run_plot_data(params, overlay=overlay)






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
                     "allowed_values": ["auto", "elevation", "depth"], "any_case": "strip",
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
    "plot_data": copy.deepcopy(_plot_tool.PLOT_DATA_INFO),
}

#: reference_chart's entry in plot_data's parameters, and what the result
#: says of it: here and in the reference charts this adapter registers.
REFERENCE_CHART_PARAMETER = {
    "type": "str", "required": False,
    "description": (
        "A standard chart's lines drawn under the data as the reference "
        "prints them, not from memory: 'uscs_plasticity' (the plasticity "
        "chart, UFS 3-220-10 Figure 1-2 as in ASTM D2487: the A-line, "
        "level at PI = 4 up to LL = 25.5 and then PI = 0.73(LL - 20); "
        "the U-line; LL = 50; PI = LL; the CL-ML zone, hatched as GEC 5 "
        "Figure 4-11 prints it; the zone "
        "names), 'aashto_plasticity' (GEC 5 Figure 4-12: LL = 40, "
        "PI = 10, PI = LL - 30 and the group names) or "
        "'uscs_particle_sizes' (the sieve sizes that bound fines, sand, "
        "gravel, cobbles and boulders, on a log grain-size axis). The "
        "lines are grey and named on the line; the chart's own axis "
        "names and frame are used unless xlabel, ylabel, xlim or ylim "
        "are given. The result's reference_chart gives the source, page "
        "and each line's equation, for the caption.")}
REFERENCE_CHART_RETURNS = (
    "With reference_chart: the chart drawn, its source (document, "
    "figure, printed page), each line's equation, the zones named and "
    "the extent drawn over.")

if _chart_lines():
    METHOD_INFO["plot_data"]["parameters"]["reference_chart"] = dict(
        REFERENCE_CHART_PARAMETER)
    METHOD_INFO["plot_data"]["returns"]["reference_chart"] = (
        REFERENCE_CHART_RETURNS)



# What a left-out input with no default stands for (BACKLOG GM-165):
# no water table or surcharge drawn, the ground surface, and north.
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
_NO_WATER_DRAWN = ("no water table drawn", "none was given")
declare_when_absent(METHOD_INFO, {
    "subsurface_profile": {
        "water_elevation": _NO_WATER_DRAWN,
        "water_depth": _NO_WATER_DRAWN,
        "surcharge": ("no surcharge drawn", "none was given"),
        "ground_elevation": ("the first layer's top elevation, else 0",
                             "with thickness-only layers the axis reads as "
                             "depth below the ground")},
})
METHOD_INFO["subsurface_profile"]["parameters"]["water_elevation"][
    "applies_when"] = {"water_depth": "unset"}
METHOD_INFO["subsurface_profile"]["parameters"]["water_depth"][
    "applies_when"] = {"water_elevation": "unset"}


class _ReferenceCharts:
    """The reference library's standard charts, as the chart stack takes
    them (:func:`funhouse_agent.charts.register_reference_charts`):
    ``plot_data``'s ``reference_chart`` and the plotting box's
    ``chart_lines``."""

    parameter = REFERENCE_CHART_PARAMETER
    returns = REFERENCE_CHART_RETURNS

    def __init__(self, module):
        self.chart_lines = module

    @staticmethod
    def overlay(params: dict):
        return _reference_overlay(params)


def register_reference_charts() -> bool:
    """Hand the reference library's chart lines to the chart stack (INF-31
    step 6; ``None`` where the installed library has none). The geotech
    page's builds call it, and importing this adapter does; True when
    there were lines to hand in."""
    module = _chart_lines()
    _charts.register_reference_charts(
        _ReferenceCharts(module) if module is not None else None)
    return module is not None


register_reference_charts()

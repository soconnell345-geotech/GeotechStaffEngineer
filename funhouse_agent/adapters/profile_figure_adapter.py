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
                                     check_nested_list, declare_aliases,
                                     reject_unknown_params, require_params,
                                     write_plotly_sidecar)
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
#: A plot_data note's keys (profile_figure.data_plot.ANNOTATION_KEYS).
_NOTE_KEYS = ("text", "x", "y", "coords", "point", "align", "box", "color",
              "font_size", "halo")
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


def _chart_lines():
    """``geotech_references.chart_lines``: standard chart lines as the
    references print them (BACKLOG GM-153), or None where the installed
    library has none (geotech-references 1.4.1 and before)."""
    import importlib
    try:
        return importlib.import_module("geotech_references.chart_lines")
    except Exception:                                      # noqa: BLE001
        return None


_VALID_PLOT = {
    "series", "title", "xlabel", "ylabel", "depth_axis", "logx", "logy",
    "hlines", "vlines", "grid", "legend", "output_path", "dpi", "width_in",
    "height_in", "include_base64", "interactive", "equal_aspect", "xlim",
    "ylim", "legend_position", "plan_view", "north_deg", "scale_units",
    "x_on_top", "annotations",
} | ({"reference_chart"} if _chart_lines() else set())

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
    from profile_figure.data_plot import _PALETTE
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
    check_nested_list(params.get("annotations"), _NOTE_KEYS,
                      method="plot_data", label="annotations",
                      hints=_KEY_HINTS)
    for key in ("hlines", "vlines"):           # a bare number is a line too
        lines = params.get(key)
        for i, spec in enumerate(lines if isinstance(lines, list) else []):
            if isinstance(spec, dict):
                check_nested_keys(spec, _REF_LINE_KEYS, method="plot_data",
                                  label=f"{key}[{i}]", hints=_KEY_HINTS)

    n_added, chart_said = 0, None
    if params.get("reference_chart"):
        params, n_added, chart_said = _reference_overlay(params)

    output_path = params.get("output_path") or _default_plot_path()
    output_path = resolve_output_path(str(output_path))
    if not output_path.lower().endswith((".png", ".jpg", ".jpeg")):
        output_path += ".png"

    try:
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
            annotations=params.get("annotations"),
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
            "interactive=false). The web app shows the interactive chart "
            "under your reply -- zoom and hover, one card; the PNG is the "
            "report copy (a file written outside the conversation folder is "
            "copied in at the end of the turn). Paste html_img_tag into "
            "report HTML and html_to_pdf embeds the real local PNG. Do not "
            "put a markdown image link to a local path in the reply; it "
            "cannot display. Never write '[image]' or an inline <svg>."),
    }
    if result.annotations:
        response["annotations"] = list(result.annotations)
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


def _own_places(text: str, n_added: int) -> str:
    """``text`` with each ``series[i]`` numbered as the caller numbered it:
    a reference chart's series are drawn first, under the caller's."""
    import re

    def own(m):
        i = int(m.group(1))
        return (f"series[{i - n_added}]" if i >= n_added
                else "the reference chart")
    return re.sub(r"series\[(\d+)\]", own, text) if n_added else text


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
                    "'font_size' (pt). A new line or "
                    "'<br>' breaks a line. The warnings say when a note "
                    "overlaps other lettering, a line runs through it or it "
                    "stands outside the plot."),
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
                         "a note that overlaps other lettering)."),
        },
    },
}

if _chart_lines():
    METHOD_INFO["plot_data"]["parameters"]["reference_chart"] = {
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
    METHOD_INFO["plot_data"]["returns"]["reference_chart"] = (
        "With reference_chart: the chart drawn, its source (document, "
        "figure, printed page), each line's equation, the zones named and "
        "the extent drawn over.")

declare_aliases(METHOD_INFO, {"plot_data": _PLOT_ALIASES})

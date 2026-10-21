"""
Plan view map of investigation locations.

All functions return plotly.graph_objects.Figure.

Functions
---------
plot_plan_view : XY scatter of investigation locations with labels
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from subsurface_characterization.site_model import SiteModel
from subsurface_characterization.plot_utils import (
    get_plotly, apply_standard_layout,
    INVESTIGATION_MARKERS, INVESTIGATION_COLORS,
)
from subsurface_characterization.results import PlotResult


_GRID = re.compile(r"UTM|zone|State\s*Plane", re.IGNORECASE)
_GEOGRAPHIC = re.compile(
    r"EPSG\D{0,3}(?:4326|4269|4258)\b|CRS\s*84|WGS\s*-?\s*84|NAD\s*-?\s*83"
    r"|DIGGS:[\d.]+:4326_\d+|latitude|longitude|geographic", re.IGNORECASE)
_EPSG = re.compile(r"EPSG\D{0,3}\d{4,6}", re.IGNORECASE)


def _depth(value) -> str:
    """A total depth for a hover label; "not given" where the source gives none."""
    return "not given" if value is None else f"{value:.1f}m"


_NUM = r"(\d+(?:\.\d+)?)"
#: "+y is N 116 E", "y axis N 30 W": the bearing of +y, east or west of north.
_Y_BEARING = re.compile(
    r"\+?\s*y(?:\s*[- ]?axis)?\s*(?:is|=|:|points|at)?\s*(?:a\s+bearing\s+of\s+)?"
    r"N\s*" + _NUM + r"\s*(?:°|deg(?:rees?)?)?\s*([EW])\b", re.I)
#: "+y is 116 deg from north", "grid north is 116 deg east of true north":
#: how far +y (grid north) is turned from true north.
_Y_FROM_NORTH = re.compile(
    r"(?:\+?\s*y(?:\s*[- ]?axis)?|grid\s+north)\s*(?:is|=|:|points|at|rotated)?"
    r"[^.;,\d]{0,12}" + _NUM + r"\s*(?:°|deg(?:rees?)?)\s*"
    r"(?:(east|west|clockwise|counter-?clockwise|ccw|cw)\s+)?"
    r"(?:of|from)\s+(?:true\s+|geographic\s+)?north", re.I)
#: "rotation 116 deg", "rotated 116 deg": the same, with nothing said of
#: which way it runs.
_ROTATION = re.compile(r"\brotat(?:ion|ed)\b\D{0,15}" + _NUM +
                       r"\s*(?:°|deg(?:rees?)?)", re.I)
#: "north is 116 deg clockwise from +y": north's angle on the grid sheet.
_NORTH_FROM_Y = re.compile(
    r"\bnorth\b\s*(?:is|=|:|lies)?[^.;,\d]{0,12}" + _NUM +
    r"\s*(?:°|deg(?:rees?)?)\s*(clockwise|counter-?clockwise|ccw|cw)?"
    r"\s*(?:of|from)\s+\+?\s*y\b", re.I)


def north_from_coordinate_system(coordinate_system: str):
    """``(north_deg, how)`` from a coordinate system's own words, else None.

    ``north_deg`` is the direction of true north on the page, degrees
    clockwise from straight up, with +y drawn up. Only clear statements are
    read: the bearing of +y (``+y is N 116 E``), how far grid north is turned
    from true north (``grid north 116 deg from true north``), a rotation, or
    north's own angle from +y. An angle with no stated way round is taken
    clockwise from north, which ``how`` says. Anything else returns None.
    """
    text = coordinate_system or ""
    m = _Y_BEARING.search(text)
    if m:
        bearing = float(m.group(1)) * (1 if m.group(2).upper() == "E" else -1)
        return ((-bearing) % 360.0,
                f"stated: +y is N {m.group(1)} deg {m.group(2).upper()}")
    m = _NORTH_FROM_Y.search(text)
    if m:
        sign = -1 if (m.group(2) or "").lower().startswith(("counter", "ccw")) \
            else 1
        way = "counter-clockwise" if sign < 0 else "clockwise"
        return ((sign * float(m.group(1))) % 360.0,
                f"stated: north is {m.group(1)} deg {way} from +y")
    m = _Y_FROM_NORTH.search(text)
    if m:
        way = (m.group(2) or "").lower()
        sign = -1 if way.startswith(("west", "counter", "ccw")) else 1
        shown = {"": "clockwise (taken as clockwise: the sense is not stated)"
                 }.get(way, way)
        return ((-sign * float(m.group(1))) % 360.0,
                f"stated: +y is {m.group(1)} deg from north, {shown}")
    m = _ROTATION.search(text)
    if m:
        return ((-float(m.group(1))) % 360.0,
                f"stated: grid rotation {m.group(1)} deg, taken clockwise "
                f"from north (the sense is not stated)")
    return None


def north_for_plan(coordinate_system: str):
    """``(north_deg, note)`` for a plan: where north is drawn and why.

    A stated rotation is drawn at its angle. Geographic coordinates have
    north up. Otherwise north is drawn straight up and the note says it is
    assumed, since nothing on the record states where north is (the plan the
    data came from, or the project's other plans, may show it).
    """
    found = north_from_coordinate_system(coordinate_system)
    if found is not None:
        return found
    if _GEOGRAPHIC.search(coordinate_system or ""):
        return 0.0, "geographic coordinates: north is up"
    return 0.0, ("north assumed up: no rotation is stated for this grid "
                 "(give north_deg from a plan of the site if one shows it)")


def axis_titles(coordinate_system: str):
    """(x title, y title, note) for a plan in ``coordinate_system``.

    Geographic (EPSG:4326, WGS 84, ...): x is longitude, y latitude, as
    the app writes them (x first). A projected system (UTM, State Plane,
    any other EPSG code): easting and northing. Anything else stated (a local drawing grid) or nothing: plain
    x and y, and the note names the system, so a grid read off a plan sheet
    is never labelled as eastings and northings (GM-76).
    """
    cs = (coordinate_system or "").strip()
    if not cs:
        return "x", "y", "coordinate system not stated"
    if _GRID.search(cs):                  # a map grid, even "WGS 84 / UTM"
        return f"Easting ({cs})", f"Northing ({cs})", ""
    if _GEOGRAPHIC.search(cs):
        return "Longitude (deg)", "Latitude (deg)", cs
    if _EPSG.search(cs):                  # any other EPSG code: projected
        return f"Easting ({cs})", f"Northing ({cs})", ""
    return "x (local grid)", "y (local grid)", cs


def plot_plan_view(
    site: SiteModel,
    color_by: str = "type",
    label_field: str = "id",
    parameter_for_color: str = "",
    title: str = "",
) -> PlotResult:
    """Plan view map of investigation locations.

    Parameters
    ----------
    site : SiteModel
        Site model with investigations.
    color_by : str
        'type' (color by investigation type) or 'parameter' (color by avg value).
    label_field : str
        'id', 'depth_to_rock', 'gwl', 'fill_thickness', or a parameter name.
    parameter_for_color : str
        If color_by='parameter', the parameter to color by.
    title : str
        Custom title.

    Returns
    -------
    PlotResult
    """
    go = get_plotly()
    fig = go.Figure()

    if not title:
        title = "Site Plan View"

    if not site.investigations:
        apply_standard_layout(fig, title=title)
        return PlotResult(
            plot_type="plan_view", title=title, figure=fig,
            n_investigations=0, n_data_points=0, parameters=[],
        )

    # An investigation with no plan position is LEFT OFF the map and named in
    # ``skipped`` -- drawing it at 0, 0 would put it at the grid origin as
    # though that were where it is.
    placed = [inv for inv in site.investigations if inv.has_xy]
    skipped = [f"{inv.investigation_id}: no x, y given; not placed on the "
               f"plan" for inv in site.investigations if not inv.has_xy]

    if color_by == "parameter" and parameter_for_color:
        _plot_by_parameter(fig, placed, parameter_for_color, label_field, go)
    else:
        _plot_by_type(fig, placed, label_field, go)

    # Labels
    for inv in placed:
        label_text = _get_label(inv, label_field)
        if label_text:
            fig.add_annotation(
                x=inv.x, y=inv.y,
                text=label_text,
                showarrow=False,
                yshift=15,
                font=dict(size=10),
            )

    cs = site.coordinate_system or next(
        (inv.coordinate_system for inv in placed if inv.coordinate_system),
        "")
    x_title, y_title, note = axis_titles(cs)
    north_deg, north_note = north_for_plan(cs)
    if note:
        title = f"{title}<br><sup>{note}</sup>"
    title += f"<br><sup>N arrow: {north_note}</sup>"
    fig.update_layout(
        xaxis=dict(title=x_title, scaleanchor="y", scaleratio=1),
        yaxis=dict(title=y_title),
    )
    apply_standard_layout(fig, title=title)
    _draw_north(fig, placed, north_deg)

    return PlotResult(
        plot_type="plan_view",
        title=title,
        n_investigations=len(placed),
        n_data_points=len(placed),
        parameters=[parameter_for_color] if parameter_for_color else [],
        figure=fig,
        skipped=skipped,
        north_deg=north_deg,
        north_note=north_note,
    )


def _draw_north(fig, placed, north_deg: float) -> None:
    """One small north arrow with its N, in the upper right of the plan, in
    the margin the axes are padded by so it covers no exploration."""
    from geotech_common.plan_furniture import plotly_north_arrow

    xs = [inv.x for inv in placed]
    ys = [inv.y for inv in placed]
    if not xs:
        return
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    pad = 0.18 * span
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    fig.update_layout(xaxis=dict(range=[x0, x1]), yaxis=dict(range=[y0, y1]))
    plotly_north_arrow(fig, (x1 - 0.5 * pad, y1 - 0.5 * pad), north_deg)


def _metres(value) -> str:
    """A depth or an elevation for a hover label; 'unknown' when not given."""
    return "unknown" if value is None else f"{value:.1f}m"


def _plot_by_type(fig, invs, label_field, go):
    """Add traces colored by investigation type."""
    type_groups = {}
    for inv in invs:
        t = inv.investigation_type
        if t not in type_groups:
            type_groups[t] = []
        type_groups[t].append(inv)

    colors = {
        "boring": "#1f77b4",
        "cpt": "#ff7f0e",
        "test_pit": "#2ca02c",
        "monitoring_well": "#9467bd",
    }

    for inv_type, invs in type_groups.items():
        xs = [inv.x for inv in invs]
        ys = [inv.y for inv in invs]
        marker_symbol = INVESTIGATION_MARKERS.get(inv_type, "circle")
        color = colors.get(inv_type, "#7f7f7f")

        hover_texts = [
            f"ID: {inv.investigation_id}<br>"
            f"Type: {inv.investigation_type}<br>"
            f"Depth: {_depth(inv.total_depth_m)}<br>"
            f"GWL: {inv.gwl_depth_m:.1f}m" if inv.gwl_depth_m is not None else
            f"ID: {inv.investigation_id}<br>"
            f"Type: {inv.investigation_type}<br>"
            f"Depth: {_depth(inv.total_depth_m)}<br>"
            + ("GWL: not encountered<br>" if inv.gwl_not_encountered else "")
            + f"Elev: {_metres(inv.elevation_m)}<br>"
            f"({inv.x:.1f}, {inv.y:.1f})"
            for inv in invs
        ]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(size=12, color=color, symbol=marker_symbol,
                        line=dict(width=1, color="black")),
            name=inv_type,
            hovertext=hover_texts,
            hoverinfo="text",
        ))


def _plot_by_parameter(fig, invs, parameter, label_field, go):
    """Add traces colored by average parameter value."""
    xs, ys, avg_vals, hover_texts = [], [], [], []

    for inv in invs:
        meas = inv.get_measurements(parameter)
        avg = float(np.mean([m.value for m in meas])) if meas else 0.0
        xs.append(inv.x)
        ys.append(inv.y)
        avg_vals.append(avg)
        hover_texts.append(
            f"ID: {inv.investigation_id}<br>"
            f"Type: {inv.investigation_type}<br>"
            f"Depth: {_depth(inv.total_depth_m)}<br>"
            f"Avg {parameter}: {avg:.1f}<br>"
            f"({inv.x:.1f}, {inv.y:.1f})"
        )

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            size=14,
            color=avg_vals,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=parameter),
            line=dict(width=1, color="black"),
        ),
        hovertext=hover_texts,
        hoverinfo="text",
        name=parameter,
    ))


def _get_label(inv, label_field):
    """Get label text for an investigation."""
    if label_field == "id":
        return inv.investigation_id
    elif label_field == "depth_to_rock":
        dtr = inv.depth_to_rock_m()
        return f"DTR={dtr:.1f}m" if dtr is not None else ""
    elif label_field == "gwl":
        if inv.gwl_depth_m is not None:
            return f"GWL={inv.gwl_depth_m:.1f}m"
        return "GWL not encountered" if inv.gwl_not_encountered else ""
    elif label_field == "fill_thickness":
        ft = inv.fill_thickness_m()
        return f"Fill={ft:.1f}m" if ft is not None else ""
    else:
        # Try as parameter name
        meas = inv.get_measurements(label_field)
        if meas:
            avg = np.mean([m.value for m in meas])
            return f"{label_field}={avg:.1f}"
        return ""

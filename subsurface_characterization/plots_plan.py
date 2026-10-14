"""
Plan view map of investigation locations.

All functions return plotly.graph_objects.Figure.

Functions
---------
plot_plan_view : XY scatter of investigation locations with labels
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from subsurface_characterization.site_model import SiteModel
from subsurface_characterization.plot_utils import (
    get_plotly, apply_standard_layout,
    INVESTIGATION_MARKERS, INVESTIGATION_COLORS,
)
from subsurface_characterization.results import PlotResult


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

    fig.update_layout(
        xaxis=dict(title="X (Easting)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y (Northing)"),
    )
    apply_standard_layout(fig, title=title)

    return PlotResult(
        plot_type="plan_view",
        title=title,
        n_investigations=len(placed),
        n_data_points=len(placed),
        parameters=[parameter_for_color] if parameter_for_color else [],
        figure=fig,
        skipped=skipped,
    )


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
            f"Depth: {inv.total_depth_m:.1f}m<br>"
            f"GWL: {inv.gwl_depth_m:.1f}m" if inv.gwl_depth_m is not None else
            f"ID: {inv.investigation_id}<br>"
            f"Type: {inv.investigation_type}<br>"
            f"Depth: {inv.total_depth_m:.1f}m<br>"
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
            f"Depth: {inv.total_depth_m:.1f}m<br>"
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

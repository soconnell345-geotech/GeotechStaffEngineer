"""
Cross-section profile views for subsurface characterization.

All functions return plotly.graph_objects.Figure.

Functions
---------
plot_cross_section : Vertical columns showing lithology at selected locations
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from subsurface_characterization.site_model import SiteModel, Investigation
from subsurface_characterization.plot_utils import (
    get_plotly, apply_standard_layout, USCS_COLORS,
)
from subsurface_characterization.results import PlotResult


def plot_cross_section(
    site: SiteModel,
    investigation_ids: List[str],
    use_elevation: bool = True,
    annotate_parameter: str = "",
    column_width: float = 0.3,
    show_gwl: bool = True,
    title: str = "",
) -> PlotResult:
    """Cross-section profile view with lithology columns.

    X-axis: cumulative horizontal distance between selected investigations.
    Y-axis: elevation (or depth).
    Vertical columns at each location showing USCS-colored lithology.
    NO interpreted connections between borings — raw data only.

    Parameters
    ----------
    site : SiteModel
        Site model.
    investigation_ids : list of str
        Ordered list of investigation IDs defining the section line.
    use_elevation : bool
        If True, Y-axis is elevation; if False, depth (inverted).
    annotate_parameter : str
        If set, show parameter values next to columns.
    column_width : float
        Width of lithology columns in distance units (fraction of spacing).
    show_gwl : bool
        Show groundwater level dashed line.
    title : str
        Custom title.

    Returns
    -------
    PlotResult
    """
    go = get_plotly()
    fig = go.Figure()

    if not title:
        ids_str = " → ".join(investigation_ids[:4])
        if len(investigation_ids) > 4:
            ids_str += " → ..."
        title = f"Cross Section: {ids_str}"

    if len(investigation_ids) < 2:
        apply_standard_layout(fig, title=title)
        return PlotResult(
            plot_type="cross_section", title=title, figure=fig,
            n_investigations=len(investigation_ids), n_data_points=0, parameters=[],
        )

    # Get investigations in order
    invs = []
    for inv_id in investigation_ids:
        try:
            invs.append(site.get_investigation(inv_id))
        except KeyError:
            continue

    if len(invs) < 2:
        apply_standard_layout(fig, title=title)
        return PlotResult(
            plot_type="cross_section", title=title, figure=fig,
            n_investigations=len(invs), n_data_points=0, parameters=[],
        )

    # Compute cumulative distances
    distances = _compute_distances(invs)
    total_dist = distances[-1] if distances[-1] > 0 else 1.0
    col_half = column_width * total_dist / (2 * len(invs))

    n_lith = 0

    # Ground surface line
    if use_elevation:
        surface_y = [inv.elevation_m for inv in invs]
    else:
        surface_y = [0.0 for _ in invs]

    fig.add_trace(go.Scatter(
        x=distances, y=surface_y,
        mode="lines",
        line=dict(color="brown", width=2),
        name="Ground Surface",
        hoverinfo="skip",
    ))

    # Lithology columns
    for idx, (inv, dist) in enumerate(zip(invs, distances)):
        for lith in inv.lithology:
            if use_elevation:
                y_top = inv.elevation_m - lith.top_depth_m
                y_bot = inv.elevation_m - lith.bottom_depth_m
            else:
                y_top = lith.top_depth_m
                y_bot = lith.bottom_depth_m

            color = USCS_COLORS.get(lith.uscs.upper(), USCS_COLORS[""])

            hover = (
                f"{inv.investigation_id}<br>"
                f"{lith.description}<br>"
                f"USCS: {lith.uscs}<br>"
                f"Depth: {lith.top_depth_m:.1f}-{lith.bottom_depth_m:.1f}m"
            )

            # Draw filled rectangle for lithology
            x0, x1 = dist - col_half, dist + col_half
            fig.add_trace(go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y_top, y_top, y_bot, y_bot, y_top],
                fill="toself",
                fillcolor=color,
                line=dict(color="black", width=0.5),
                name=f"{lith.uscs}" if (idx == 0 and lith == inv.lithology[0]) else None,
                showlegend=False,
                hovertext=hover,
                hoverinfo="text",
            ))
            n_lith += 1

        # Investigation ID label at top
        label_y = surface_y[idx] + 0.5 if use_elevation else -0.5
        fig.add_annotation(
            x=dist, y=label_y,
            text=inv.investigation_id,
            showarrow=False,
            font=dict(size=10, color="black"),
        )

    # GWL line
    if show_gwl:
        gwl_x, gwl_y = [], []
        for inv, dist in zip(invs, distances):
            if inv.gwl_depth_m is not None:
                gwl_x.append(dist)
                if use_elevation:
                    gwl_y.append(inv.elevation_m - inv.gwl_depth_m)
                else:
                    gwl_y.append(inv.gwl_depth_m)

        if len(gwl_x) >= 2:
            fig.add_trace(go.Scatter(
                x=gwl_x, y=gwl_y,
                mode="lines",
                line=dict(color="blue", width=1.5, dash="dash"),
                name="GWL",
            ))

    # Parameter annotations
    parameters = []
    if annotate_parameter:
        parameters.append(annotate_parameter)
        for inv, dist in zip(invs, distances):
            meas = inv.get_measurements(annotate_parameter)
            for m in meas:
                if use_elevation:
                    y_pos = inv.elevation_m - m.depth_m
                else:
                    y_pos = m.depth_m
                fig.add_annotation(
                    x=dist + col_half * 1.5,
                    y=y_pos,
                    text=f"{m.value:.0f}",
                    showarrow=False,
                    font=dict(size=8, color="red"),
                )

    # Add USCS legend entries
    uscs_seen = set()
    for inv in invs:
        for lith in inv.lithology:
            uscs = lith.uscs.upper()
            if uscs and uscs not in uscs_seen:
                uscs_seen.add(uscs)
                color = USCS_COLORS.get(uscs, USCS_COLORS[""])
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=12, color=color, symbol="square"),
                    name=uscs,
                    showlegend=True,
                ))

    y_label = "Elevation (m)" if use_elevation else "Depth (m)"
    fig.update_layout(
        xaxis=dict(title="Distance Along Section (m)"),
        yaxis=dict(
            title=y_label,
            autorange="reversed" if not use_elevation else True,
        ),
    )
    apply_standard_layout(fig, title=title)

    return PlotResult(
        plot_type="cross_section",
        title=title,
        n_investigations=len(invs),
        n_data_points=n_lith,
        parameters=parameters,
        figure=fig,
    )


def _compute_distances(invs: List[Investigation]) -> List[float]:
    """Compute cumulative horizontal distances between investigations."""
    distances = [0.0]
    for i in range(1, len(invs)):
        dx = invs[i].x - invs[i - 1].x
        dy = invs[i].y - invs[i - 1].y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        distances.append(distances[-1] + dist)
    return distances

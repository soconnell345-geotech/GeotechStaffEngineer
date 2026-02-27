"""
XY depth plots for subsurface characterization.

All functions return plotly.graph_objects.Figure.

Functions
---------
plot_parameter_vs_depth : Scatter plot of a single parameter vs depth
plot_atterberg_limits : LL/PL bracket plot with wn overlay
plot_multi_parameter : Side-by-side subplots with shared Y-axis
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from subsurface_characterization.site_model import SiteModel
from subsurface_characterization.plot_utils import (
    get_plotly, get_subplots, apply_standard_layout,
    USCS_COLORS, INVESTIGATION_COLORS,
)
from subsurface_characterization.results import PlotResult


def plot_parameter_vs_depth(
    site: SiteModel,
    parameter: str,
    color_by: str = "investigation",
    use_elevation: bool = False,
    show_trend: bool = False,
    show_bands: bool = False,
    band_sigma: float = 1.0,
    group_trends_by: str = "",
    title: str = "",
) -> PlotResult:
    """Scatter plot of parameter value (X) vs depth/elevation (Y).

    Parameters
    ----------
    site : SiteModel
        Site model with investigations.
    parameter : str
        Parameter name to plot (e.g., 'N_spt', 'cu_kPa').
    color_by : str
        'investigation', 'uscs', or 'none'.
    use_elevation : bool
        If True, Y-axis is elevation; if False, depth (inverted).
    show_trend : bool
        Overlay trendline.
    show_bands : bool
        Overlay ±n_sigma prediction bands.
    band_sigma : float
        Number of sigma for bands (default 1.0).
    group_trends_by : str
        If 'uscs', compute separate trends per USCS class.
    title : str
        Custom title. If empty, auto-generated.

    Returns
    -------
    PlotResult
    """
    go = get_plotly()
    fig = go.Figure()

    if not title:
        title = f"{parameter} vs {'Elevation' if use_elevation else 'Depth'}"

    n_points = 0
    inv_ids = set()

    if color_by == "investigation":
        for i, inv in enumerate(site.investigations):
            meas = inv.get_measurements(parameter)
            if not meas:
                continue
            inv_ids.add(inv.investigation_id)
            depths = [m.depth_m for m in meas]
            values = [m.value for m in meas]
            y_vals = [inv.elevation_m - d for d in depths] if use_elevation else depths
            color = INVESTIGATION_COLORS[i % len(INVESTIGATION_COLORS)]

            hover_text = [
                f"{inv.investigation_id}<br>"
                f"Depth: {d:.1f}m<br>"
                f"{parameter}: {v:.2f}<br>"
                f"USCS: {inv.uscs_at_depth(d)}<br>"
                f"Sample: {m.sample_id}"
                for d, v, m in zip(depths, values, meas)
            ]

            fig.add_trace(go.Scatter(
                x=values, y=y_vals,
                mode="markers",
                marker=dict(size=8, color=color),
                name=inv.investigation_id,
                hovertext=hover_text,
                hoverinfo="text",
            ))
            n_points += len(meas)

    elif color_by == "uscs":
        uscs_groups = {}  # uscs -> (depths, values, hover_texts)
        for inv in site.investigations:
            inv_ids.add(inv.investigation_id)
            for m in inv.get_measurements(parameter):
                uscs = inv.uscs_at_depth(m.depth_m) or ""
                if uscs not in uscs_groups:
                    uscs_groups[uscs] = ([], [], [])
                d = m.depth_m
                y = inv.elevation_m - d if use_elevation else d
                uscs_groups[uscs][0].append(y)
                uscs_groups[uscs][1].append(m.value)
                uscs_groups[uscs][2].append(
                    f"{inv.investigation_id}<br>"
                    f"Depth: {d:.1f}m<br>"
                    f"{parameter}: {m.value:.2f}<br>"
                    f"USCS: {uscs}"
                )
                n_points += 1

        for uscs, (y_vals, values, hovers) in sorted(uscs_groups.items()):
            color = USCS_COLORS.get(uscs, USCS_COLORS[""])
            fig.add_trace(go.Scatter(
                x=values, y=y_vals,
                mode="markers",
                marker=dict(size=8, color=color),
                name=uscs or "Unknown",
                hovertext=hovers,
                hoverinfo="text",
            ))

    else:  # 'none'
        all_y = []
        all_x = []
        all_hover = []
        for inv in site.investigations:
            inv_ids.add(inv.investigation_id)
            for m in inv.get_measurements(parameter):
                d = m.depth_m
                y = inv.elevation_m - d if use_elevation else d
                all_y.append(y)
                all_x.append(m.value)
                all_hover.append(
                    f"{inv.investigation_id}<br>"
                    f"Depth: {d:.1f}m<br>"
                    f"{parameter}: {m.value:.2f}"
                )
                n_points += 1

        fig.add_trace(go.Scatter(
            x=all_x, y=all_y,
            mode="markers",
            marker=dict(size=8, color="#1f77b4"),
            name=parameter,
            hovertext=all_hover,
            hoverinfo="text",
        ))

    # Trend line and bands
    trend_results = []
    if show_trend and n_points >= 2:
        from subsurface_characterization.statistics import compute_trend, compute_grouped_trends

        if group_trends_by == "uscs":
            grouped = compute_grouped_trends(site, parameter, group_by="uscs")
            for label, trend in grouped.items():
                if trend.n_points < 2:
                    continue
                trend_results.append(trend)
                _add_trend_to_fig(fig, trend, site, parameter, use_elevation,
                                  show_bands, band_sigma, label)
        else:
            all_data = site.all_measurements(parameter)
            depths = np.array([m.depth_m for _, m in all_data])
            values = np.array([m.value for _, m in all_data])
            trend = compute_trend(depths, values, parameter=parameter)
            trend_results.append(trend)
            _add_trend_to_fig(fig, trend, site, parameter, use_elevation,
                              show_bands, band_sigma, "Trend")

    y_label = "Elevation (m)" if use_elevation else "Depth (m)"
    fig.update_layout(
        yaxis=dict(
            title=y_label,
            autorange="reversed" if not use_elevation else True,
        ),
        xaxis=dict(title=parameter),
    )
    apply_standard_layout(fig, title=title)

    return PlotResult(
        plot_type="parameter_vs_depth",
        title=title,
        n_investigations=len(inv_ids),
        n_data_points=n_points,
        parameters=[parameter],
        figure=fig,
        trend_results=trend_results,
    )


def _add_trend_to_fig(fig, trend, site, parameter, use_elevation,
                       show_bands, band_sigma, label):
    """Add trendline and optional bands to figure."""
    go = get_plotly()

    # Generate trend line points
    all_data = site.all_measurements(parameter)
    depths = np.array([m.depth_m for _, m in all_data])
    d_min, d_max = float(depths.min()), float(depths.max())
    d_range = np.linspace(d_min, d_max, 50)

    trend_vals = np.array([trend.predict(d) for d in d_range])

    if use_elevation:
        # Use average elevation for trend (approximate)
        avg_elev = np.mean([inv.elevation_m for inv in site.investigations])
        y_range = avg_elev - d_range
    else:
        y_range = d_range

    fig.add_trace(go.Scatter(
        x=trend_vals, y=y_range,
        mode="lines",
        line=dict(dash="dash", width=2),
        name=f"{label} (R²={trend.r_squared:.2f})",
        hoverinfo="skip",
    ))

    if show_bands and trend.std_residual > 0:
        lower = np.array([trend.band(d, band_sigma)[0] for d in d_range])
        upper = np.array([trend.band(d, band_sigma)[1] for d in d_range])

        fig.add_trace(go.Scatter(
            x=upper.tolist() + lower[::-1].tolist(),
            y=y_range.tolist() + y_range[::-1].tolist(),
            fill="toself",
            fillcolor="rgba(100,100,100,0.15)",
            line=dict(width=0),
            name=f"±{band_sigma}σ band",
            hoverinfo="skip",
        ))


def plot_atterberg_limits(
    site: SiteModel,
    use_elevation: bool = False,
    title: str = "",
) -> PlotResult:
    """Plot Atterberg limits (LL-PL bracket) with natural moisture content overlay.

    LL and PL shown as horizontal bar connecting them. wn shown as dot.

    Parameters
    ----------
    site : SiteModel
        Site model with LL_pct, PL_pct, and optionally wn_pct measurements.
    use_elevation : bool
        If True, Y-axis is elevation.
    title : str
        Custom title.

    Returns
    -------
    PlotResult
    """
    go = get_plotly()
    fig = go.Figure()

    if not title:
        title = "Atterberg Limits"

    n_points = 0
    inv_ids = set()

    for i, inv in enumerate(site.investigations):
        ll_meas = {m.depth_m: m.value for m in inv.get_measurements("LL_pct")}
        pl_meas = {m.depth_m: m.value for m in inv.get_measurements("PL_pct")}
        wn_meas = {m.depth_m: m.value for m in inv.get_measurements("wn_pct")}

        common_depths = sorted(set(ll_meas.keys()) & set(pl_meas.keys()))
        if not common_depths:
            continue

        inv_ids.add(inv.investigation_id)
        color = INVESTIGATION_COLORS[i % len(INVESTIGATION_COLORS)]

        for d in common_depths:
            ll = ll_meas[d]
            pl = pl_meas[d]
            pi = ll - pl
            wn = wn_meas.get(d)
            y = inv.elevation_m - d if use_elevation else d

            hover = (
                f"{inv.investigation_id}<br>"
                f"Depth: {d:.1f}m<br>"
                f"LL: {ll:.0f}%<br>"
                f"PL: {pl:.0f}%<br>"
                f"PI: {pi:.0f}%"
            )
            if wn is not None:
                hover += f"<br>wn: {wn:.0f}%"

            # Bracket line from PL to LL
            fig.add_trace(go.Scatter(
                x=[pl, ll], y=[y, y],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6, symbol="line-ns", color=color),
                name=inv.investigation_id if d == common_depths[0] else None,
                showlegend=(d == common_depths[0]),
                legendgroup=inv.investigation_id,
                hovertext=[hover, hover],
                hoverinfo="text",
            ))
            n_points += 1

        # Natural moisture content overlay
        for d, wn in wn_meas.items():
            y = inv.elevation_m - d if use_elevation else d
            fig.add_trace(go.Scatter(
                x=[wn], y=[y],
                mode="markers",
                marker=dict(size=10, color="blue", symbol="diamond"),
                name="wn (%)" if (i == 0 and d == list(wn_meas.keys())[0]) else None,
                showlegend=(i == 0 and d == list(wn_meas.keys())[0]),
                legendgroup="wn",
                hovertext=f"{inv.investigation_id}<br>Depth: {d:.1f}m<br>wn: {wn:.0f}%",
                hoverinfo="text",
            ))

    y_label = "Elevation (m)" if use_elevation else "Depth (m)"
    fig.update_layout(
        yaxis=dict(title=y_label, autorange="reversed" if not use_elevation else True),
        xaxis=dict(title="Moisture Content (%)"),
    )
    apply_standard_layout(fig, title=title)

    return PlotResult(
        plot_type="atterberg_limits",
        title=title,
        n_investigations=len(inv_ids),
        n_data_points=n_points,
        parameters=["LL_pct", "PL_pct", "wn_pct"],
        figure=fig,
    )


def plot_multi_parameter(
    site: SiteModel,
    parameters: List[str],
    use_elevation: bool = False,
    title: str = "",
) -> PlotResult:
    """Side-by-side subplots with shared Y-axis (depth aligned).

    Parameters
    ----------
    site : SiteModel
        Site model.
    parameters : list of str
        Parameters for each panel (e.g., ['N_spt', 'cu_kPa', 'wn_pct']).
    use_elevation : bool
        If True, Y-axis is elevation.
    title : str
        Custom title.

    Returns
    -------
    PlotResult
    """
    go = get_plotly()
    make_subplots = get_subplots()

    n_panels = len(parameters)
    if n_panels == 0:
        fig = go.Figure()
        return PlotResult(plot_type="multi_parameter", figure=fig)

    if not title:
        title = "Multi-Parameter Profile"

    fig = make_subplots(
        rows=1, cols=n_panels,
        shared_yaxes=True,
        subplot_titles=parameters,
        horizontal_spacing=0.05,
    )

    n_points = 0
    inv_ids = set()

    for col, param in enumerate(parameters, 1):
        for i, inv in enumerate(site.investigations):
            meas = inv.get_measurements(param)
            if not meas:
                continue
            inv_ids.add(inv.investigation_id)
            depths = [m.depth_m for m in meas]
            values = [m.value for m in meas]
            y_vals = [inv.elevation_m - d for d in depths] if use_elevation else depths
            color = INVESTIGATION_COLORS[i % len(INVESTIGATION_COLORS)]

            fig.add_trace(
                go.Scatter(
                    x=values, y=y_vals,
                    mode="markers",
                    marker=dict(size=6, color=color),
                    name=inv.investigation_id,
                    showlegend=(col == 1),
                    legendgroup=inv.investigation_id,
                    hovertext=[
                        f"{inv.investigation_id}<br>Depth: {d:.1f}m<br>{param}: {v:.2f}"
                        for d, v in zip(depths, values)
                    ],
                    hoverinfo="text",
                ),
                row=1, col=col,
            )
            n_points += len(meas)

        fig.update_xaxes(title_text=param, row=1, col=col)

    y_label = "Elevation (m)" if use_elevation else "Depth (m)"
    fig.update_yaxes(
        title_text=y_label,
        autorange="reversed" if not use_elevation else True,
        row=1, col=1,
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=600,
        width=300 * n_panels,
    )

    return PlotResult(
        plot_type="multi_parameter",
        title=title,
        n_investigations=len(inv_ids),
        n_data_points=n_points,
        parameters=parameters,
        figure=fig,
    )

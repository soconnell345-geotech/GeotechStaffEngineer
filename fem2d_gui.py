"""
FEM2D GUI — Interactive Plotly Dash Application

Browser-based 2D finite element analysis with contour fill visualization.
Supports: Gravity, Foundation, Slope SRM, Excavation, Seepage, Consolidation.

Run:
    python fem2d_gui.py          # http://127.0.0.1:8055
    # Jupyter:
    # from fem2d_gui import app
    # app.run(jupyter_mode="inline", port=8055)
"""

import json
import math
import traceback

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import Dash, html, dcc, dash_table, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from fem2d import (
    analyze_gravity, analyze_foundation, analyze_slope_srm,
    analyze_excavation, analyze_seepage, analyze_consolidation,
    generate_rect_mesh, generate_slope_mesh, detect_boundary_nodes,
    FEMResult, SeepageResult, ConsolidationResult,
)

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

ANALYSIS_TYPES = [
    "Gravity", "Foundation", "Slope SRM",
    "Excavation", "Seepage", "Consolidation",
]

MESH_PRESETS = {
    "Very Coarse": (5, 5),
    "Coarse": (8, 8),
    "Medium": (12, 12),
    "Fine": (18, 18),
    "Very Fine": (25, 25),
}

FIELD_OPTIONS_MECHANICAL = [
    {"label": "Displacement Magnitude", "value": "disp_mag"},
    {"label": "Displacement X", "value": "disp_x"},
    {"label": "Displacement Y", "value": "disp_y"},
    {"label": "Stress sigma_xx", "value": "sigma_xx"},
    {"label": "Stress sigma_yy", "value": "sigma_yy"},
    {"label": "Shear tau_xy", "value": "tau_xy"},
]

FIELD_OPTIONS_SEEPAGE = [
    {"label": "Total Head", "value": "head"},
    {"label": "Pore Pressure", "value": "pore_pressure"},
    {"label": "Velocity Magnitude", "value": "vel_mag"},
]

DEFAULT_LAYERS = [
    {
        "layer": "Layer 1", "thickness": 10, "E": 30000, "nu": 0.3,
        "gamma": 18, "c": 10, "phi": 25, "psi": 0, "model": "Elastic",
    },
]

DEFAULT_HS_PARAMS = [
    {"param": "E50_ref", "value": 25000},
    {"param": "Eur_ref", "value": 75000},
    {"param": "m", "value": 0.5},
    {"param": "p_ref", "value": 100},
    {"param": "R_f", "value": 0.9},
]

DEFAULT_STRUT_ROWS = [
    {"depth": 1.5, "stiffness": 50000},
]

DEFAULT_GWT_POLYLINE = [
    {"x": 0, "z_gwt": -2},
    {"x": 50, "z_gwt": -2},
]

# Colorscale helpers
COLORSCALES = {
    "disp_mag": "Viridis",
    "disp_x": "Viridis",
    "disp_y": "Viridis",
    "sigma_xx": "RdBu_r",
    "sigma_yy": "RdBu_r",
    "tau_xy": "RdBu_r",
    "head": "Blues",
    "pore_pressure": "Blues",
    "vel_mag": "YlOrRd",
}

# Sidebar width
SIDEBAR_W = "370px"

# Shared styles — match slope_stability_gui.py
SECTION_STYLE = {
    "borderBottom": "1px solid #e2e8f0",
    "paddingBottom": "12px",
    "marginBottom": "8px",
}
LABEL_STYLE = {
    "fontSize": "0.82rem",
    "fontWeight": "600",
    "color": "#475569",
    "marginBottom": "4px",
    "display": "block",
}
INPUT_STYLE = {
    "width": "100%",
    "padding": "4px 8px",
    "border": "1px solid #cbd5e1",
    "borderRadius": "4px",
    "fontSize": "0.85rem",
}
INPUT_NARROW = {
    **INPUT_STYLE,
    "width": "80px",
    "display": "inline-block",
}
BTN_STYLE = {
    "padding": "4px 12px",
    "border": "1px solid #cbd5e1",
    "borderRadius": "4px",
    "background": "#f8fafc",
    "cursor": "pointer",
    "fontSize": "0.8rem",
    "marginRight": "6px",
}
BTN_PRIMARY = {
    **BTN_STYLE,
    "background": "#2563eb",
    "color": "white",
    "border": "1px solid #2563eb",
    "fontWeight": "600",
    "padding": "8px 24px",
    "fontSize": "0.9rem",
}
TABLE_STYLE = {
    "overflowX": "auto",
    "fontSize": "0.8rem",
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe_float(val, default=0.0):
    """Convert to float, returning default on failure."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    """Convert to int, returning default on failure."""
    if val is None or val == "":
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _get_mesh_params(coarseness, analysis_type, width, depth,
                     slope_height=None, slope_angle=None):
    """Get nx, ny from coarseness preset with optional auto-refinement."""
    nx_base, ny_base = MESH_PRESETS.get(coarseness, (12, 12))

    # Scale to domain aspect ratio
    aspect = width / depth if depth > 0 else 1.0
    if aspect > 2:
        nx = int(nx_base * 1.5)
        ny = nx_base
    elif aspect < 0.5:
        nx = nx_base
        ny = int(nx_base * 1.5)
    else:
        nx = nx_base
        ny = nx_base

    # Auto-refine for slopes (more elements near slope face)
    if analysis_type == "Slope SRM" and coarseness != "Very Fine":
        nx = int(nx * 1.3)
        ny = int(ny * 1.2)

    # Auto-refine for excavation (need resolution near wall)
    if analysis_type == "Excavation" and coarseness != "Very Fine":
        nx = int(nx * 1.3)
        ny = int(ny * 1.3)

    return max(nx, 4), max(ny, 4)


def _value_to_color(val, vmin, vmax, colorscale_name="Viridis"):
    """Map a scalar value to an RGB color string using a Plotly colorscale."""
    import plotly.colors as pc

    if vmax == vmin:
        t = 0.5
    else:
        t = (val - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))

    # Get the colorscale as list of (fraction, color) tuples
    cs = pc.get_colorscale(colorscale_name)

    # Find the two bounding entries
    for i in range(len(cs) - 1):
        f0, c0 = cs[i]
        f1, c1 = cs[i + 1]
        if f0 <= t <= f1:
            if f1 == f0:
                frac = 0.0
            else:
                frac = (t - f0) / (f1 - f0)
            # Parse colors
            rgb0 = pc.unlabel_rgb(c0) if isinstance(c0, str) and c0.startswith("rgb") else pc.hex_to_rgb(c0) if isinstance(c0, str) and c0.startswith("#") else (0, 0, 0)
            rgb1 = pc.unlabel_rgb(c1) if isinstance(c1, str) and c1.startswith("rgb") else pc.hex_to_rgb(c1) if isinstance(c1, str) and c1.startswith("#") else (0, 0, 0)
            r = int(rgb0[0] + frac * (rgb1[0] - rgb0[0]))
            g = int(rgb0[1] + frac * (rgb1[1] - rgb0[1]))
            b = int(rgb0[2] + frac * (rgb1[2] - rgb0[2]))
            return f"rgb({r},{g},{b})"

    return "rgb(128,128,128)"


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _empty_figure(msg="Select analysis type and click Run Analysis"):
    """Return a placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=16, color="#94a3b8"))
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=520, margin=dict(l=40, r=20, t=30, b=30),
    )
    return fig


def _build_mesh_wireframe(nodes, elements, title="Mesh Preview",
                          bc_nodes=None, beam_nodes=None):
    """Build a wireframe figure of the triangular mesh."""
    fig = go.Figure()

    # Draw triangle edges
    edge_x, edge_y = [], []
    for elem in elements:
        for k in range(3):
            i0 = elem[k]
            i1 = elem[(k + 1) % 3]
            edge_x.extend([nodes[i0, 0], nodes[i1, 0], None])
            edge_y.extend([nodes[i0, 1], nodes[i1, 1], None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="#94a3b8", width=0.5),
        name="Elements", hoverinfo="skip",
    ))

    # Node markers
    fig.add_trace(go.Scatter(
        x=nodes[:, 0], y=nodes[:, 1], mode="markers",
        marker=dict(size=2, color="#475569"),
        name=f"Nodes ({len(nodes)})",
        hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>",
    ))

    # Boundary condition markers
    if bc_nodes is not None:
        # Fixed (bottom)
        fixed = bc_nodes.get("fixed", [])
        if len(fixed) > 0:
            fig.add_trace(go.Scatter(
                x=nodes[fixed, 0], y=nodes[fixed, 1], mode="markers",
                marker=dict(symbol="triangle-up", size=6, color="#dc2626"),
                name="Fixed", hoverinfo="name",
            ))
        # Roller (sides)
        roller_x = bc_nodes.get("roller_x", [])
        if len(roller_x) > 0:
            fig.add_trace(go.Scatter(
                x=nodes[roller_x, 0], y=nodes[roller_x, 1], mode="markers",
                marker=dict(symbol="circle-open", size=5, color="#2563eb"),
                name="Roller X", hoverinfo="name",
            ))

    # Beam nodes
    if beam_nodes is not None and len(beam_nodes) > 0:
        fig.add_trace(go.Scatter(
            x=nodes[beam_nodes, 0], y=nodes[beam_nodes, 1],
            mode="lines+markers",
            line=dict(color="#16a34a", width=3),
            marker=dict(size=4, color="#16a34a"),
            name="Wall",
        ))

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title="x (m)", scaleanchor="y", constrain="domain"),
        yaxis=dict(title="y (m)"),
        height=520,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(font=dict(size=10), x=1.0, y=1.0),
    )
    return fig


def _build_contour_figure(nodes, elements, field_values, field_name,
                          colorscale_name="Viridis", title=None,
                          deformed_nodes=None, beam_nodes=None,
                          beam_forces=None):
    """Build per-element contour fill figure on triangular mesh."""
    fig = go.Figure()

    display_nodes = deformed_nodes if deformed_nodes is not None else nodes
    vmin = float(np.min(field_values))
    vmax = float(np.max(field_values))

    # Draw filled triangles
    for i, elem in enumerate(elements):
        x0, y0 = display_nodes[elem[0]]
        x1, y1 = display_nodes[elem[1]]
        x2, y2 = display_nodes[elem[2]]
        color = _value_to_color(field_values[i], vmin, vmax, colorscale_name)

        fig.add_trace(go.Scatter(
            x=[x0, x1, x2, x0], y=[y0, y1, y2, y0],
            fill="toself", fillcolor=color,
            line=dict(width=0.3, color="rgba(100,100,100,0.3)"),
            showlegend=False,
            hovertemplate=(
                f"{field_name}: {field_values[i]:.4g}<br>"
                f"Element {i}<extra></extra>"
            ),
        ))

    # Colorbar via invisible scatter
    n_bar = 50
    bar_vals = np.linspace(vmin, vmax, n_bar)
    bar_colors = [
        _value_to_color(v, vmin, vmax, colorscale_name) for v in bar_vals
    ]
    fig.add_trace(go.Scatter(
        x=[None] * n_bar, y=[None] * n_bar,
        mode="markers",
        marker=dict(
            size=0,
            color=bar_vals,
            colorscale=colorscale_name,
            colorbar=dict(
                title=dict(text=field_name, font=dict(size=11)),
                thickness=15, len=0.7,
            ),
            showscale=True,
        ),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Beam overlay
    if beam_nodes is not None and len(beam_nodes) > 0:
        fig.add_trace(go.Scatter(
            x=display_nodes[beam_nodes, 0],
            y=display_nodes[beam_nodes, 1],
            mode="lines",
            line=dict(color="#16a34a", width=3),
            name="Wall",
        ))

    if title is None:
        title = field_name

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title="x (m)", scaleanchor="y", constrain="domain"),
        yaxis=dict(title="y (m)"),
        height=520,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(font=dict(size=10)),
    )
    return fig


def _build_seepage_figure(nodes, elements, result, field="head"):
    """Build seepage visualization with optional velocity arrows."""
    if field == "head":
        # Nodal field → average to elements for contour
        vals = np.array([
            np.mean(result.head[elem]) for elem in elements
        ])
        name = "Total Head (m)"
        cs = "Blues"
    elif field == "pore_pressure":
        vals = np.array([
            np.mean(result.pore_pressures[elem]) for elem in elements
        ])
        name = "Pore Pressure (kPa)"
        cs = "Blues"
    else:  # vel_mag
        vel = result.velocity
        vals = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
        name = "Velocity |v| (m/s)"
        cs = "YlOrRd"

    fig = _build_contour_figure(nodes, elements, vals, name, cs)

    # Add velocity arrows for head/pore_pressure views
    if field in ("head", "pore_pressure") and result.velocity is not None:
        vel = result.velocity
        v_mag = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
        v_max = v_mag.max()
        if v_max > 0:
            centroids = np.mean(nodes[elements], axis=1)
            # Subsample arrows for clarity (max ~100)
            step = max(1, len(elements) // 100)
            idx = np.arange(0, len(elements), step)
            arrow_scale = (nodes[:, 0].max() - nodes[:, 0].min()) * 0.03
            for i in idx:
                cx, cy = centroids[i]
                vx = vel[i, 0] / v_max * arrow_scale
                vy = vel[i, 1] / v_max * arrow_scale
                fig.add_annotation(
                    x=cx + vx, y=cy + vy, ax=cx, ay=cy,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                    arrowcolor="#d97706",
                )

    return fig


def _build_consolidation_time_figure(result):
    """Build settlement + pore pressure vs time subplot."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Surface Settlement", "Max Excess Pore Pressure"),
        vertical_spacing=0.12,
    )

    times = result.times
    if times is None:
        return _empty_figure("No consolidation time data")

    # Settlement vs time
    settlements = result.settlements
    if settlements is not None:
        fig.add_trace(go.Scatter(
            x=times, y=settlements * 1000,  # convert m to mm
            mode="lines+markers",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=4),
            name="Settlement",
        ), row=1, col=1)

    # Pore pressure vs time
    pp = result.pore_pressures
    if pp is not None:
        max_pp = np.max(pp, axis=1) if pp.ndim == 2 else pp
        fig.add_trace(go.Scatter(
            x=times, y=max_pp,
            mode="lines+markers",
            line=dict(color="#dc2626", width=2),
            marker=dict(size=4),
            name="Max Excess PP",
        ), row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Settlement (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Pore Pressure (kPa)", row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=60, r=20, t=40, b=40),
        showlegend=False,
    )
    return fig


def _build_beam_force_figure(nodes, beam_forces):
    """Build moment/shear diagram along beam elements."""
    if not beam_forces:
        return _empty_figure("No beam force data")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Bending Moment (kN*m/m)", "Shear Force (kN/m)"),
    )

    # Sort beam forces by elevation (top to bottom)
    sorted_bf = sorted(beam_forces, key=lambda bf: -nodes[bf.node_i, 1])

    elevations_i = [nodes[bf.node_i, 1] for bf in sorted_bf]
    elevations_j = [nodes[bf.node_j, 1] for bf in sorted_bf]
    moments_i = [bf.moment_i for bf in sorted_bf]
    moments_j = [bf.moment_j for bf in sorted_bf]
    shears_i = [bf.shear_i for bf in sorted_bf]
    shears_j = [bf.shear_j for bf in sorted_bf]

    # Interleave i and j for continuous diagram
    elev_all, mom_all, shear_all = [], [], []
    for k in range(len(sorted_bf)):
        elev_all.extend([elevations_i[k], elevations_j[k]])
        mom_all.extend([moments_i[k], moments_j[k]])
        shear_all.extend([shears_i[k], shears_j[k]])

    fig.add_trace(go.Scatter(
        x=mom_all, y=elev_all,
        mode="lines", line=dict(color="#2563eb", width=2),
        fill="tozerox", fillcolor="rgba(37,99,235,0.15)",
        name="Moment",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=shear_all, y=elev_all,
        mode="lines", line=dict(color="#dc2626", width=2),
        fill="tozerox", fillcolor="rgba(220,38,38,0.15)",
        name="Shear",
    ), row=1, col=2)

    fig.update_yaxes(title_text="Elevation (m)", row=1, col=1)
    fig.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=60, r=20, t=40, b=40),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Build summary text
# ---------------------------------------------------------------------------

def _format_result_summary(result, analysis_type):
    """Format analysis result as summary text."""
    if isinstance(result, FEMResult):
        lines = [
            f"Analysis: {result.analysis_type}",
            f"Mesh: {result.n_nodes} nodes, {result.n_elements} elements",
            f"Converged: {result.converged}",
            "",
            f"Max displacement: {result.max_displacement_m:.4f} m",
            f"  ux_max: {result.max_displacement_x_m:.4f} m",
            f"  uy_max: {result.max_displacement_y_m:.4f} m",
            "",
            f"Stress:",
            f"  sigma_xx max: {result.max_sigma_xx_kPa:.1f} kPa",
            f"  sigma_yy: {result.min_sigma_yy_kPa:.1f} to "
            f"{result.max_sigma_yy_kPa:.1f} kPa",
            f"  tau_xy max: {result.max_tau_xy_kPa:.1f} kPa",
        ]
        if result.FOS is not None:
            lines.extend([
                "",
                f"Factor of Safety (SRM): {result.FOS:.3f}",
                f"SRF trials: {result.n_srf_trials}",
            ])
        if result.n_beam_elements > 0:
            lines.extend([
                "",
                f"Beam elements: {result.n_beam_elements}",
                f"  Max moment: {result.max_beam_moment_kNm_per_m:.2f} kN*m/m",
                f"  Max shear: {result.max_beam_shear_kN_per_m:.2f} kN/m",
            ])
        return "\n".join(lines)

    elif isinstance(result, SeepageResult):
        return result.summary()

    elif isinstance(result, ConsolidationResult):
        return result.summary()

    return str(result)


# ---------------------------------------------------------------------------
# Extract field values for contour
# ---------------------------------------------------------------------------

def _extract_field(result, field_name, nodes, elements):
    """Extract per-element field values from a FEMResult."""
    u = result.displacements
    stresses = result.stresses
    n_nodes = len(nodes)

    if field_name == "disp_mag":
        ux = u[0::2][:n_nodes]
        uy = u[1::2][:n_nodes]
        nodal = np.sqrt(ux**2 + uy**2)
        return np.array([np.mean(nodal[elem]) for elem in elements])

    elif field_name == "disp_x":
        ux = u[0::2][:n_nodes]
        return np.array([np.mean(ux[elem]) for elem in elements])

    elif field_name == "disp_y":
        uy = u[1::2][:n_nodes]
        return np.array([np.mean(uy[elem]) for elem in elements])

    elif field_name == "sigma_xx" and stresses is not None:
        return stresses[:, 0]

    elif field_name == "sigma_yy" and stresses is not None:
        return stresses[:, 1]

    elif field_name == "tau_xy" and stresses is not None:
        return stresses[:, 2]

    # Fallback
    return np.zeros(len(elements))


# ---------------------------------------------------------------------------
# Layout builder functions
# ---------------------------------------------------------------------------

def _make_input_row(label, input_id, default, width="80px", **kwargs):
    """Create a label + number input row."""
    return html.Div([
        html.Label(label, style=LABEL_STYLE),
        dcc.Input(
            id=input_id, type="number", value=default,
            style={**INPUT_STYLE, "width": width},
            **kwargs,
        ),
    ], style={"marginBottom": "6px"})


def _make_section(title, children, section_id, open_default=True):
    """Create a collapsible <details> section."""
    return html.Details(
        [html.Summary(
            title,
            style={
                "fontWeight": "700", "fontSize": "0.92rem",
                "cursor": "pointer", "color": "#1e293b",
                "padding": "6px 0",
            },
        )] + children,
        open=open_default,
        id=section_id,
        style=SECTION_STYLE,
    )


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = Dash(
    __name__,
    title="FEM2D Analysis",
    suppress_callback_exceptions=True,
)

# ── Sidebar ─────────────────────────────────────────────────────────────

sidebar = html.Div([
    # Title
    html.H2("FEM2D Analysis", style={
        "margin": "0 0 4px 0", "fontSize": "1.1rem", "color": "#1e293b",
    }),
    html.Div(id="fos-badge", style={"marginBottom": "10px"}),

    # ── Analysis Type ──
    _make_section("Analysis Type", [
        dcc.Dropdown(
            id="analysis-type",
            options=[{"label": a, "value": a} for a in ANALYSIS_TYPES],
            value="Gravity",
            clearable=False,
            style={"fontSize": "0.85rem"},
        ),
    ], "section-analysis-type", open_default=True),

    # ── Domain Geometry ──
    _make_section("Domain Geometry", [
        _make_input_row("Width (m)", "domain-width", 20),
        _make_input_row("Depth (m)", "domain-depth", 10),
        html.Div(id="slope-geometry-container", children=[
            html.Hr(style={"margin": "8px 0"}),
            html.Label("Slope Parameters", style={
                **LABEL_STYLE, "color": "#1e293b"}),
            _make_input_row("Slope Height (m)", "slope-height", 5),
            _make_input_row("Slope Angle (deg)", "slope-angle", 30),
            _make_input_row("Crest Offset (m)", "crest-offset", 5),
        ], style={"display": "none"}),
        html.Div(id="excav-geometry-container", children=[
            html.Hr(style={"margin": "8px 0"}),
            html.Label("Excavation Parameters", style={
                **LABEL_STYLE, "color": "#1e293b"}),
            _make_input_row("Excavation Depth (m)", "excav-depth", 5),
            _make_input_row("Excavation Width (m)", "excav-width", 10),
        ], style={"display": "none"}),
    ], "section-domain", open_default=True),

    # ── Mesh Density ──
    _make_section("Mesh Density", [
        html.Label("Global Coarseness", style=LABEL_STYLE),
        dcc.Dropdown(
            id="mesh-coarseness",
            options=[{"label": k, "value": k} for k in MESH_PRESETS],
            value="Medium",
            clearable=False,
            style={"fontSize": "0.85rem", "marginBottom": "8px"},
        ),
        dcc.Checklist(
            id="auto-refine",
            options=[{"label": " Auto-refine near features", "value": "on"}],
            value=["on"],
            style={"fontSize": "0.82rem"},
        ),
        html.Button(
            "Generate Mesh", id="btn-gen-mesh",
            style={**BTN_STYLE, "marginTop": "8px"},
        ),
    ], "section-mesh", open_default=True),

    # ── Soil Layers ──
    html.Div(id="section-soil-container", children=[
        _make_section("Soil Layers", [
            dash_table.DataTable(
                id="soil-layers-table",
                columns=[
                    {"name": "Layer", "id": "layer", "editable": True},
                    {"name": "Thick (m)", "id": "thickness", "type": "numeric",
                     "editable": True},
                    {"name": "E (kPa)", "id": "E", "type": "numeric",
                     "editable": True},
                    {"name": "nu", "id": "nu", "type": "numeric",
                     "editable": True},
                    {"name": "gamma", "id": "gamma", "type": "numeric",
                     "editable": True},
                    {"name": "c (kPa)", "id": "c", "type": "numeric",
                     "editable": True},
                    {"name": "phi (deg)", "id": "phi", "type": "numeric",
                     "editable": True},
                    {"name": "psi (deg)", "id": "psi", "type": "numeric",
                     "editable": True},
                    {"name": "Model", "id": "model", "editable": True,
                     "presentation": "dropdown"},
                ],
                dropdown={
                    "model": {
                        "options": [
                            {"label": "Elastic", "value": "Elastic"},
                            {"label": "Mohr-Coulomb", "value": "MC"},
                            {"label": "Hardening Soil", "value": "HS"},
                        ],
                    },
                },
                data=DEFAULT_LAYERS.copy(),
                editable=True,
                row_deletable=True,
                style_table=TABLE_STYLE,
                style_cell={"padding": "4px 6px", "fontSize": "0.8rem",
                            "minWidth": "50px"},
                style_header={"fontWeight": "700", "backgroundColor": "#f1f5f9"},
            ),
            html.Div([
                html.Button("+ Add Layer", id="btn-add-layer", style=BTN_STYLE),
            ], style={"marginTop": "6px"}),
            # HS parameters (shown when any layer uses HS model)
            html.Div(id="hs-params-container", children=[
                html.Hr(style={"margin": "8px 0"}),
                html.Label("Hardening Soil Parameters (shared)", style={
                    **LABEL_STYLE, "color": "#1e293b"}),
                _make_input_row("E50_ref (kPa)", "hs-e50", 25000),
                _make_input_row("Eur_ref (kPa)", "hs-eur", 75000),
                _make_input_row("m", "hs-m", 0.5),
                _make_input_row("p_ref (kPa)", "hs-pref", 100),
                _make_input_row("R_f", "hs-rf", 0.9),
            ], style={"display": "none"}),
        ], "section-soil", open_default=True),
    ]),

    # ── Loading ──
    html.Div(id="section-loading-container", children=[
        _make_section("Loading", [
            _make_input_row("Surface Load q (kPa)", "load-q", 100),
            html.Div(id="load-width-container", children=[
                _make_input_row("Footing Width B (m)", "load-width", 2),
            ]),
        ], "section-loading", open_default=True),
    ], style={"display": "none"}),

    # ── Wall Properties (Excavation) ──
    html.Div(id="section-wall-container", children=[
        _make_section("Wall Properties", [
            _make_input_row("Wall Depth (m)", "wall-depth", 10),
            _make_input_row("Wall EI (kN*m2/m)", "wall-ei", 50000),
            _make_input_row("Wall EA (kN/m)", "wall-ea", 5000000),
            html.Hr(style={"margin": "8px 0"}),
            html.Label("Strut Levels", style=LABEL_STYLE),
            dash_table.DataTable(
                id="strut-table",
                columns=[
                    {"name": "Depth (m)", "id": "depth", "type": "numeric",
                     "editable": True},
                    {"name": "Stiffness (kN/m/m)", "id": "stiffness",
                     "type": "numeric", "editable": True},
                ],
                data=DEFAULT_STRUT_ROWS.copy(),
                editable=True,
                row_deletable=True,
                style_table=TABLE_STYLE,
                style_cell={"padding": "4px 6px", "fontSize": "0.8rem"},
                style_header={"fontWeight": "700", "backgroundColor": "#f1f5f9"},
            ),
            html.Button("+ Add Strut", id="btn-add-strut", style={
                **BTN_STYLE, "marginTop": "4px"}),
        ], "section-wall", open_default=True),
    ], style={"display": "none"}),

    # ── Groundwater ──
    html.Div(id="section-gwt-container", children=[
        _make_section("Groundwater", [
            html.Label("GWT Mode", style=LABEL_STYLE),
            dcc.Dropdown(
                id="gwt-mode",
                options=[
                    {"label": "None", "value": "none"},
                    {"label": "Constant Elevation", "value": "constant"},
                    {"label": "Polyline", "value": "polyline"},
                ],
                value="none",
                clearable=False,
                style={"fontSize": "0.85rem", "marginBottom": "8px"},
            ),
            html.Div(id="gwt-constant-container", children=[
                _make_input_row("GWT Elevation (m)", "gwt-elevation", -2),
            ], style={"display": "none"}),
            html.Div(id="gwt-polyline-container", children=[
                dash_table.DataTable(
                    id="gwt-polyline-table",
                    columns=[
                        {"name": "X (m)", "id": "x", "type": "numeric",
                         "editable": True},
                        {"name": "GWT Elev (m)", "id": "z_gwt",
                         "type": "numeric", "editable": True},
                    ],
                    data=DEFAULT_GWT_POLYLINE.copy(),
                    editable=True,
                    row_deletable=True,
                    style_table=TABLE_STYLE,
                    style_cell={"padding": "4px 6px", "fontSize": "0.8rem"},
                    style_header={"fontWeight": "700",
                                  "backgroundColor": "#f1f5f9"},
                ),
                html.Button("+ Add Point", id="btn-add-gwt-pt",
                            style={**BTN_STYLE, "marginTop": "4px"}),
            ], style={"display": "none"}),
            _make_input_row("gamma_w (kN/m3)", "gamma-w", 9.81),
        ], "section-gwt", open_default=True),
    ], style={"display": "none"}),

    # ── Seepage BCs ──
    html.Div(id="section-seepage-container", children=[
        _make_section("Seepage Boundary Conditions", [
            _make_input_row("Permeability k (m/s)", "seepage-k", 1e-5),
            html.Hr(style={"margin": "8px 0"}),
            html.Label("Prescribed Head Edges", style=LABEL_STYLE),
            dcc.Checklist(
                id="seepage-edges",
                options=[
                    {"label": " Left edge", "value": "left"},
                    {"label": " Right edge", "value": "right"},
                    {"label": " Top edge", "value": "top"},
                    {"label": " Bottom edge", "value": "bottom"},
                ],
                value=["left", "right"],
                style={"fontSize": "0.82rem", "marginBottom": "8px"},
            ),
            _make_input_row("Left Head (m)", "head-left", 10),
            _make_input_row("Right Head (m)", "head-right", 0),
            _make_input_row("Top Head (m)", "head-top", 5),
            _make_input_row("Bottom Head (m)", "head-bottom", 0),
        ], "section-seepage", open_default=True),
    ], style={"display": "none"}),

    # ── Consolidation ──
    html.Div(id="section-consol-container", children=[
        _make_section("Consolidation", [
            _make_input_row("Permeability k (m/s)", "consol-k", 1e-8),
            _make_input_row("Surface Load q (kPa)", "consol-q", 100),
            html.Hr(style={"margin": "8px 0"}),
            html.Label("Time Range", style=LABEL_STYLE),
            _make_input_row("Start Time (s)", "consol-t-start", 1),
            _make_input_row("End Time (s)", "consol-t-end", 1e7),
            _make_input_row("Number of Steps", "consol-n-steps", 15),
            dcc.Checklist(
                id="consol-log-space",
                options=[{"label": " Log-spaced", "value": "on"}],
                value=["on"],
                style={"fontSize": "0.82rem"},
            ),
        ], "section-consol", open_default=True),
    ], style={"display": "none"}),

    # ── Run button ──
    html.Div([
        html.Hr(style={"margin": "12px 0"}),
        html.Button(
            "Run Analysis", id="btn-run",
            style=BTN_PRIMARY,
        ),
    ]),

], style={
    "width": SIDEBAR_W, "minWidth": SIDEBAR_W, "maxWidth": SIDEBAR_W,
    "padding": "14px", "overflowY": "auto", "height": "100vh",
    "borderRight": "1px solid #e2e8f0", "background": "#f8fafc",
    "fontFamily": "'Segoe UI', system-ui, sans-serif",
})

# ── Main area ───────────────────────────────────────────────────────────

main_area = html.Div([
    # Field selector (for results)
    html.Div([
        html.Label("Display Field:", style={
            **LABEL_STYLE, "display": "inline-block", "marginRight": "8px"}),
        dcc.Dropdown(
            id="field-selector",
            options=FIELD_OPTIONS_MECHANICAL,
            value="disp_mag",
            clearable=False,
            style={"width": "250px", "fontSize": "0.85rem",
                   "display": "inline-block", "verticalAlign": "middle"},
        ),
        dcc.Checklist(
            id="show-deformed",
            options=[{"label": " Show deformed mesh", "value": "on"}],
            value=[],
            style={"fontSize": "0.82rem", "display": "inline-block",
                   "marginLeft": "16px", "verticalAlign": "middle"},
        ),
        html.Div([
            html.Label("Scale:", style={
                **LABEL_STYLE, "display": "inline-block", "marginRight": "4px"}),
            dcc.Input(
                id="deform-scale", type="number", value=10,
                style={**INPUT_STYLE, "width": "60px", "display": "inline-block"},
            ),
        ], id="deform-scale-container",
            style={"display": "inline-block", "marginLeft": "8px",
                   "verticalAlign": "middle"}),
    ], style={"marginBottom": "8px", "display": "flex",
              "alignItems": "center", "flexWrap": "wrap", "gap": "4px"}),

    # Main figure (mesh or contour)
    dcc.Loading(
        dcc.Graph(id="main-figure", figure=_empty_figure()),
        type="circle",
    ),

    # Secondary figures (beam forces, consolidation time history)
    html.Div(id="secondary-figure-container", children=[
        dcc.Graph(id="secondary-figure", figure=_empty_figure(""),
                  style={"display": "none"}),
    ]),

    # Results summary
    html.Div(id="results-summary-container", children=[
        html.Pre(
            id="results-summary",
            style={
                "background": "#f1f5f9",
                "padding": "12px",
                "borderRadius": "6px",
                "fontSize": "0.82rem",
                "fontFamily": "'Consolas', 'Courier New', monospace",
                "whiteSpace": "pre-wrap",
                "maxHeight": "300px",
                "overflowY": "auto",
                "marginTop": "8px",
            },
            children="",
        ),
    ]),

], style={
    "flex": "1",
    "padding": "14px",
    "overflowY": "auto",
    "height": "100vh",
    "fontFamily": "'Segoe UI', system-ui, sans-serif",
})

# ── Stores ──────────────────────────────────────────────────────────────

stores = html.Div([
    dcc.Store(id="store-mesh", data=None),
    dcc.Store(id="store-result", data=None),
    dcc.Store(id="store-analysis-type-done", data=None),
])

# ── Layout ──────────────────────────────────────────────────────────────

app.layout = html.Div([
    sidebar,
    main_area,
    stores,
], style={
    "display": "flex",
    "flexDirection": "row",
    "height": "100vh",
    "margin": "0",
    "padding": "0",
    "overflow": "hidden",
})

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# ── 1. Input visibility based on analysis type ──

@app.callback(
    [
        Output("slope-geometry-container", "style"),
        Output("excav-geometry-container", "style"),
        Output("section-soil-container", "style"),
        Output("section-loading-container", "style"),
        Output("section-wall-container", "style"),
        Output("section-gwt-container", "style"),
        Output("section-seepage-container", "style"),
        Output("section-consol-container", "style"),
        Output("field-selector", "options"),
        Output("field-selector", "value"),
        Output("load-width-container", "style"),
    ],
    Input("analysis-type", "value"),
)
def update_visibility(atype):
    show = {"display": "block"}
    hide = {"display": "none"}

    # Slope geometry: only for SRM
    slope_vis = show if atype == "Slope SRM" else hide
    # Excavation geometry: only for Excavation
    excav_vis = show if atype == "Excavation" else hide
    # Soil layers: all except Seepage
    soil_vis = show if atype != "Seepage" else hide
    # Loading: Foundation and Consolidation
    loading_vis = show if atype in ("Foundation", "Consolidation") else hide
    # Wall: Excavation only
    wall_vis = show if atype == "Excavation" else hide
    # GWT: SRM, Excavation, Seepage, Consolidation
    gwt_vis = show if atype in ("Slope SRM", "Excavation", "Consolidation") else hide
    # Seepage BCs: Seepage only
    seepage_vis = show if atype == "Seepage" else hide
    # Consolidation: Consolidation only
    consol_vis = show if atype == "Consolidation" else hide

    # Field options
    if atype == "Seepage":
        field_opts = FIELD_OPTIONS_SEEPAGE
        field_val = "head"
    else:
        field_opts = FIELD_OPTIONS_MECHANICAL
        field_val = "disp_mag"

    # Load width: only for Foundation
    load_w_vis = show if atype == "Foundation" else hide

    return (slope_vis, excav_vis, soil_vis, loading_vis, wall_vis,
            gwt_vis, seepage_vis, consol_vis,
            field_opts, field_val, load_w_vis)


# ── 2. GWT mode visibility ──

@app.callback(
    [
        Output("gwt-constant-container", "style"),
        Output("gwt-polyline-container", "style"),
    ],
    Input("gwt-mode", "value"),
)
def update_gwt_mode(mode):
    show = {"display": "block"}
    hide = {"display": "none"}
    if mode == "constant":
        return show, hide
    elif mode == "polyline":
        return hide, show
    return hide, hide


# ── 3. HS params visibility ──

@app.callback(
    Output("hs-params-container", "style"),
    Input("soil-layers-table", "data"),
)
def update_hs_visibility(rows):
    if rows and any(r.get("model") == "HS" for r in rows):
        return {"display": "block"}
    return {"display": "none"}


# ── 4. Add layer button ──

@app.callback(
    Output("soil-layers-table", "data"),
    Input("btn-add-layer", "n_clicks"),
    State("soil-layers-table", "data"),
    prevent_initial_call=True,
)
def add_layer(n_clicks, rows):
    if rows is None:
        rows = []
    n = len(rows) + 1
    rows.append({
        "layer": f"Layer {n}", "thickness": 5, "E": 30000, "nu": 0.3,
        "gamma": 18, "c": 10, "phi": 25, "psi": 0, "model": "Elastic",
    })
    return rows


# ── 5. Add strut button ──

@app.callback(
    Output("strut-table", "data"),
    Input("btn-add-strut", "n_clicks"),
    State("strut-table", "data"),
    prevent_initial_call=True,
)
def add_strut(n_clicks, rows):
    if rows is None:
        rows = []
    rows.append({"depth": 3, "stiffness": 50000})
    return rows


# ── 6. Add GWT polyline point ──

@app.callback(
    Output("gwt-polyline-table", "data"),
    Input("btn-add-gwt-pt", "n_clicks"),
    State("gwt-polyline-table", "data"),
    prevent_initial_call=True,
)
def add_gwt_point(n_clicks, rows):
    if rows is None:
        rows = []
    last_x = _safe_float(rows[-1]["x"], 0) + 10 if rows else 0
    rows.append({"x": last_x, "z_gwt": -2})
    return rows


# ── 7. Generate Mesh Preview ──

@app.callback(
    [
        Output("main-figure", "figure", allow_duplicate=True),
        Output("store-mesh", "data"),
    ],
    Input("btn-gen-mesh", "n_clicks"),
    [
        State("analysis-type", "value"),
        State("domain-width", "value"),
        State("domain-depth", "value"),
        State("mesh-coarseness", "value"),
        State("slope-height", "value"),
        State("slope-angle", "value"),
        State("crest-offset", "value"),
        State("excav-depth", "value"),
        State("excav-width", "value"),
    ],
    prevent_initial_call=True,
)
def generate_mesh(n_clicks, atype, width, depth, coarseness,
                  slope_h, slope_ang, crest_off, excav_d, excav_w):
    width = _safe_float(width, 20)
    depth = _safe_float(depth, 10)

    nx, ny = _get_mesh_params(coarseness, atype, width, depth)

    try:
        if atype == "Slope SRM":
            # Build surface profile from slope params
            sh = _safe_float(slope_h, 5)
            sa = _safe_float(slope_ang, 30)
            co = _safe_float(crest_off, 5)
            run = sh / max(math.tan(math.radians(sa)), 0.01)
            surface_pts = [
                (0, 0),
                (co, 0),
                (co + run, sh),
                (co + run + width, sh),
            ]
            nodes, elements = generate_slope_mesh(
                surface_pts, depth, nx, ny,
                x_extend_left=width * 0.3,
                x_extend_right=width * 0.3)
        elif atype == "Excavation":
            ed = _safe_float(excav_d, 5)
            ew = _safe_float(excav_w, 10)
            wall_d = ed * 2
            x_left = -2.0 * wall_d
            x_right = ew + 2.0 * wall_d
            y_bottom = -max(wall_d + ed, 2.0 * wall_d)
            nodes, elements = generate_rect_mesh(
                x_left, x_right, y_bottom, 0, nx, ny)
        else:
            nodes, elements = generate_rect_mesh(
                0, width, -depth, 0, nx, ny)

        bc_nodes = detect_boundary_nodes(nodes)

        # Store mesh data as JSON-serializable
        mesh_data = {
            "nodes": nodes.tolist(),
            "elements": elements.tolist(),
            "bc_fixed": bc_nodes["fixed"].tolist(),
            "bc_roller_x": bc_nodes["roller_x"].tolist(),
        }

        fig = _build_mesh_wireframe(
            nodes, elements,
            title=f"Mesh: {len(nodes)} nodes, {len(elements)} elements",
            bc_nodes=bc_nodes,
        )
        return fig, mesh_data

    except Exception as e:
        return _empty_figure(f"Mesh error: {e}"), None


# ── 8. Run Analysis ──

@app.callback(
    [
        Output("main-figure", "figure", allow_duplicate=True),
        Output("secondary-figure", "figure"),
        Output("secondary-figure", "style"),
        Output("results-summary", "children"),
        Output("store-result", "data"),
        Output("store-analysis-type-done", "data"),
        Output("fos-badge", "children"),
    ],
    Input("btn-run", "n_clicks"),
    [
        State("analysis-type", "value"),
        State("domain-width", "value"),
        State("domain-depth", "value"),
        State("mesh-coarseness", "value"),
        State("soil-layers-table", "data"),
        # Loading
        State("load-q", "value"),
        State("load-width", "value"),
        # Slope
        State("slope-height", "value"),
        State("slope-angle", "value"),
        State("crest-offset", "value"),
        # Excavation
        State("excav-depth", "value"),
        State("excav-width", "value"),
        State("wall-depth", "value"),
        State("wall-ei", "value"),
        State("wall-ea", "value"),
        State("strut-table", "data"),
        # Groundwater
        State("gwt-mode", "value"),
        State("gwt-elevation", "value"),
        State("gwt-polyline-table", "data"),
        State("gamma-w", "value"),
        # Seepage
        State("seepage-k", "value"),
        State("seepage-edges", "value"),
        State("head-left", "value"),
        State("head-right", "value"),
        State("head-top", "value"),
        State("head-bottom", "value"),
        # Consolidation
        State("consol-k", "value"),
        State("consol-q", "value"),
        State("consol-t-start", "value"),
        State("consol-t-end", "value"),
        State("consol-n-steps", "value"),
        State("consol-log-space", "value"),
        # HS params
        State("hs-e50", "value"),
        State("hs-eur", "value"),
        State("hs-m", "value"),
        State("hs-pref", "value"),
        State("hs-rf", "value"),
        # Display
        State("field-selector", "value"),
        State("show-deformed", "value"),
        State("deform-scale", "value"),
    ],
    prevent_initial_call=True,
)
def run_analysis(n_clicks, atype, width, depth, coarseness,
                 soil_rows,
                 load_q, load_width,
                 slope_h, slope_ang, crest_off,
                 excav_d, excav_w, wall_d, wall_ei, wall_ea, strut_rows,
                 gwt_mode, gwt_elev, gwt_poly_data, gamma_w,
                 seepage_k, seepage_edges, head_left, head_right,
                 head_top, head_bottom,
                 consol_k, consol_q, consol_t_start, consol_t_end,
                 consol_n_steps, consol_log_space,
                 hs_e50, hs_eur, hs_m, hs_pref, hs_rf,
                 field_sel, show_deformed, deform_scale):

    empty_fig = _empty_figure("")
    sec_style_hide = {"display": "none"}
    sec_style_show = {"display": "block"}
    fos_badge = ""

    width = _safe_float(width, 20)
    depth = _safe_float(depth, 10)
    gamma_w = _safe_float(gamma_w, 9.81)

    nx, ny = _get_mesh_params(coarseness, atype, width, depth)

    try:
        # ── Parse soil layers ──
        def _parse_soil_layers(rows, for_mc=False):
            """Parse soil table rows into layer dicts."""
            layers = []
            elev = 0.0
            for row in (rows or DEFAULT_LAYERS):
                thick = _safe_float(row.get("thickness"), 5)
                bot = elev - thick
                layer = {
                    "name": row.get("layer", "Layer"),
                    "bottom_elevation": bot,
                    "E": _safe_float(row.get("E"), 30000),
                    "nu": _safe_float(row.get("nu"), 0.3),
                    "gamma": _safe_float(row.get("gamma"), 18),
                    "c": _safe_float(row.get("c"), 10),
                    "phi": _safe_float(row.get("phi"), 25),
                    "psi": _safe_float(row.get("psi"), 0),
                }
                model = row.get("model", "Elastic")
                if model == "HS":
                    layer["model"] = "hs"
                    layer["E50_ref"] = _safe_float(hs_e50, 25000)
                    layer["Eur_ref"] = _safe_float(hs_eur, 75000)
                    layer["m"] = _safe_float(hs_m, 0.5)
                    layer["p_ref"] = _safe_float(hs_pref, 100)
                    layer["R_f"] = _safe_float(hs_rf, 0.9)
                layers.append(layer)
                elev = bot
            return layers

        # ── Parse GWT ──
        def _parse_gwt():
            if gwt_mode == "constant":
                return _safe_float(gwt_elev, -2)
            elif gwt_mode == "polyline":
                pts = []
                for row in (gwt_poly_data or []):
                    pts.append([_safe_float(row.get("x"), 0),
                                _safe_float(row.get("z_gwt"), -2)])
                if len(pts) >= 2:
                    return np.array(sorted(pts, key=lambda p: p[0]))
            return None

        # ── Dispatch analysis ──

        if atype == "Gravity":
            sl = _parse_soil_layers(soil_rows)
            result = analyze_gravity(
                width=width, depth=depth,
                gamma=sl[0]["gamma"],
                E=sl[0]["E"], nu=sl[0]["nu"],
                nx=nx, ny=ny,
            )

        elif atype == "Foundation":
            sl = _parse_soil_layers(soil_rows)
            q = _safe_float(load_q, 100)
            B = _safe_float(load_width, 2)
            result = analyze_foundation(
                B=B, q=q, depth=depth,
                E=sl[0]["E"], nu=sl[0]["nu"],
                gamma=sl[0]["gamma"],
                nx=nx, ny=ny,
            )

        elif atype == "Slope SRM":
            sl = _parse_soil_layers(soil_rows, for_mc=True)
            sh = _safe_float(slope_h, 5)
            sa = _safe_float(slope_ang, 30)
            co = _safe_float(crest_off, 5)
            run = sh / max(math.tan(math.radians(sa)), 0.01)
            surface_pts = [
                (0, 0),
                (co, 0),
                (co + run, sh),
                (co + run + width, sh),
            ]
            gwt = _parse_gwt()
            result = analyze_slope_srm(
                surface_points=surface_pts,
                soil_layers=sl,
                depth=depth,
                nx=nx, ny=ny,
                gwt=gwt, gamma_w=gamma_w,
            )
            if result.FOS is not None:
                fos_badge = html.Span(
                    f"FOS = {result.FOS:.3f}",
                    style={
                        "background": "#16a34a" if result.FOS >= 1.5
                                  else "#f59e0b" if result.FOS >= 1.0
                                  else "#dc2626",
                        "color": "white",
                        "padding": "4px 12px",
                        "borderRadius": "12px",
                        "fontSize": "0.9rem",
                        "fontWeight": "700",
                    },
                )

        elif atype == "Excavation":
            sl = _parse_soil_layers(soil_rows, for_mc=True)
            ed = _safe_float(excav_d, 5)
            ew = _safe_float(excav_w, 10)
            wd = _safe_float(wall_d, 10)
            wei = _safe_float(wall_ei, 50000)
            wea = _safe_float(wall_ea, 5000000)
            gwt = _parse_gwt()
            result = analyze_excavation(
                width=ew, depth=ed, wall_depth=wd,
                soil_layers=sl,
                wall_EI=wei, wall_EA=wea,
                nx=nx, ny=ny,
                gwt=gwt, gamma_w=gamma_w,
            )

        elif atype == "Seepage":
            k = _safe_float(seepage_k, 1e-5)
            nodes, elements = generate_rect_mesh(
                0, width, -depth, 0, nx, ny)

            # Build head BCs from edge selections
            head_bcs = []
            tol = 0.01
            edges = seepage_edges or []
            if "left" in edges:
                h = _safe_float(head_left, 10)
                left_nodes = np.where(
                    np.abs(nodes[:, 0] - nodes[:, 0].min()) < tol)[0]
                head_bcs.extend([(int(n), h) for n in left_nodes])
            if "right" in edges:
                h = _safe_float(head_right, 0)
                right_nodes = np.where(
                    np.abs(nodes[:, 0] - nodes[:, 0].max()) < tol)[0]
                head_bcs.extend([(int(n), h) for n in right_nodes])
            if "top" in edges:
                h = _safe_float(head_top, 5)
                top_nodes = np.where(
                    np.abs(nodes[:, 1] - nodes[:, 1].max()) < tol)[0]
                head_bcs.extend([(int(n), h) for n in top_nodes])
            if "bottom" in edges:
                h = _safe_float(head_bottom, 0)
                bot_nodes = np.where(
                    np.abs(nodes[:, 1] - nodes[:, 1].min()) < tol)[0]
                head_bcs.extend([(int(n), h) for n in bot_nodes])

            result = analyze_seepage(
                nodes=nodes, elements=elements,
                k=k, head_bcs=head_bcs, gamma_w=gamma_w,
            )

        elif atype == "Consolidation":
            sl = _parse_soil_layers(soil_rows)
            k = _safe_float(consol_k, 1e-8)
            q = _safe_float(consol_q, 100)
            t_start = _safe_float(consol_t_start, 1)
            t_end = _safe_float(consol_t_end, 1e7)
            n_steps = _safe_int(consol_n_steps, 15)

            if "on" in (consol_log_space or []):
                time_pts = np.logspace(
                    np.log10(max(t_start, 0.1)),
                    np.log10(t_end),
                    n_steps,
                )
            else:
                time_pts = np.linspace(t_start, t_end, n_steps)

            gwt_val = 0.0
            gwt = _parse_gwt()
            if isinstance(gwt, (int, float)):
                gwt_val = gwt
            elif gwt is not None:
                gwt_val = float(np.mean(gwt[:, 1]))

            result = analyze_consolidation(
                width=width, depth=depth,
                soil_layers=sl, k=k, load_q=q,
                time_points=time_pts,
                gwt=gwt_val, gamma_w=gamma_w,
                nx=nx, ny=ny,
            )

        else:
            return (_empty_figure("Unknown analysis type"),
                    empty_fig, sec_style_hide, "Unknown type", None, None, "")

        # ── Build output figures ──

        if atype == "Seepage":
            nodes_arr = np.array(result.nodes)
            elem_arr = np.array(result.elements)
            main_fig = _build_seepage_figure(
                nodes_arr, elem_arr, result, field=field_sel)
            summary = _format_result_summary(result, atype)

            # Store result data
            result_data = {
                "type": "seepage",
                "nodes": nodes_arr.tolist(),
                "elements": elem_arr.tolist(),
                "head": result.head.tolist(),
                "pore_pressures": result.pore_pressures.tolist(),
                "velocity": result.velocity.tolist(),
            }
            return (main_fig, empty_fig, sec_style_hide,
                    summary, result_data, atype, fos_badge)

        elif atype == "Consolidation":
            # Time history figure
            sec_fig = _build_consolidation_time_figure(result)
            summary = _format_result_summary(result, atype)

            result_data = {
                "type": "consolidation",
                "summary": summary,
            }
            return (_empty_figure("See time history below"),
                    sec_fig, sec_style_show,
                    summary, result_data, atype, fos_badge)

        else:
            # Mechanical result (Gravity, Foundation, SRM, Excavation)
            nodes_arr = result.nodes
            elem_arr = result.elements
            field_values = _extract_field(
                result, field_sel, nodes_arr, elem_arr)
            cs_name = COLORSCALES.get(field_sel, "Viridis")

            # Deformed mesh
            deformed = None
            if "on" in (show_deformed or []) and result.displacements is not None:
                scale = _safe_float(deform_scale, 10)
                u = result.displacements
                n_nodes = len(nodes_arr)
                ux = u[0::2][:n_nodes]
                uy = u[1::2][:n_nodes]
                deformed = nodes_arr.copy()
                deformed[:, 0] += ux * scale
                deformed[:, 1] += uy * scale

            # Beam nodes for overlay
            beam_nodes = None
            if result.beam_forces:
                bn_set = set()
                for bf in result.beam_forces:
                    bn_set.add(bf.node_i)
                    bn_set.add(bf.node_j)
                beam_nodes = sorted(bn_set, key=lambda n: -nodes_arr[n, 1])

            main_fig = _build_contour_figure(
                nodes_arr, elem_arr, field_values, field_sel,
                cs_name, deformed_nodes=deformed,
                beam_nodes=beam_nodes,
            )

            # Secondary figure: beam forces for excavation
            sec_fig = empty_fig
            sec_style = sec_style_hide
            if atype == "Excavation" and result.beam_forces:
                sec_fig = _build_beam_force_figure(nodes_arr, result.beam_forces)
                sec_style = sec_style_show

            summary = _format_result_summary(result, atype)

            # Store result (serialize only what we need for field switching)
            result_data = {
                "type": "mechanical",
                "nodes": nodes_arr.tolist(),
                "elements": elem_arr.tolist(),
                "displacements": result.displacements.tolist()
                    if result.displacements is not None else None,
                "stresses": result.stresses.tolist()
                    if result.stresses is not None else None,
                "beam_nodes": beam_nodes,
            }
            if result.beam_forces:
                result_data["beam_forces"] = [
                    {"node_i": bf.node_i, "node_j": bf.node_j,
                     "moment_i": bf.moment_i, "moment_j": bf.moment_j,
                     "shear_i": bf.shear_i, "shear_j": bf.shear_j}
                    for bf in result.beam_forces
                ]

            return (main_fig, sec_fig, sec_style,
                    summary, result_data, atype, fos_badge)

    except Exception as e:
        tb = traceback.format_exc()
        err_msg = f"Analysis error:\n{e}\n\n{tb}"
        return (_empty_figure(f"Error: {e}"),
                empty_fig, sec_style_hide,
                err_msg, None, None, "")


# ── 9. Update field display (re-render contour without re-running analysis) ──

@app.callback(
    Output("main-figure", "figure", allow_duplicate=True),
    [
        Input("field-selector", "value"),
        Input("show-deformed", "value"),
        Input("deform-scale", "value"),
    ],
    [
        State("store-result", "data"),
        State("store-analysis-type-done", "data"),
    ],
    prevent_initial_call=True,
)
def update_field_display(field_sel, show_deformed, deform_scale,
                         result_data, atype_done):
    if result_data is None:
        raise PreventUpdate

    rtype = result_data.get("type")

    if rtype == "seepage":
        nodes = np.array(result_data["nodes"])
        elements = np.array(result_data["elements"])
        # Reconstruct a minimal SeepageResult for visualization
        sr = SeepageResult()
        sr.head = np.array(result_data["head"])
        sr.pore_pressures = np.array(result_data["pore_pressures"])
        sr.velocity = np.array(result_data["velocity"])
        return _build_seepage_figure(nodes, elements, sr, field=field_sel)

    elif rtype == "mechanical":
        nodes = np.array(result_data["nodes"])
        elements = np.array(result_data["elements"])
        displacements = (np.array(result_data["displacements"])
                         if result_data.get("displacements") else None)
        stresses = (np.array(result_data["stresses"])
                    if result_data.get("stresses") else None)
        beam_nodes = result_data.get("beam_nodes")

        # Build a minimal FEMResult for field extraction
        fr = FEMResult()
        fr.displacements = displacements
        fr.stresses = stresses

        field_values = _extract_field(fr, field_sel, nodes, elements)
        cs_name = COLORSCALES.get(field_sel, "Viridis")

        # Deformed mesh
        deformed = None
        if "on" in (show_deformed or []) and displacements is not None:
            scale = _safe_float(deform_scale, 10)
            n_nodes = len(nodes)
            ux = displacements[0::2][:n_nodes]
            uy = displacements[1::2][:n_nodes]
            deformed = nodes.copy()
            deformed[:, 0] += ux * scale
            deformed[:, 1] += uy * scale

        bn = None
        if beam_nodes:
            bn = sorted(beam_nodes, key=lambda n: -nodes[n, 1])

        return _build_contour_figure(
            nodes, elements, field_values, field_sel, cs_name,
            deformed_nodes=deformed, beam_nodes=bn,
        )

    raise PreventUpdate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("FEM2D GUI starting at http://127.0.0.1:8055")
    app.run(debug=True, port=8055)

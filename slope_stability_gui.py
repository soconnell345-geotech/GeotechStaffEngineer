"""
Slope Stability GUI — Interactive Plotly Dash Application

Browser-based slope stability analysis with live cross-section preview,
Bishop/Fellenius/Spencer methods, and grid search for critical surfaces.

Run:
    python slope_stability_gui.py          # http://127.0.0.1:8051
    # Jupyter:
    # from slope_stability_gui import app
    # app.run(jupyter_mode="inline", port=8051)
"""

import math
import traceback

import numpy as np
import plotly.graph_objects as go
import dash
from dash import Dash, html, dcc, dash_table, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from slope_stability import (
    SlopeGeometry, SlopeSoilLayer, SoilNail, analyze_slope,
    search_critical_surface,
)

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

LAYER_COLORS = [
    "#f5e6c8", "#d4a574", "#a0c4a0", "#c8b4a0", "#e6d4a0",
]

DEFAULT_SURFACE = [
    {"x": 0, "z": 10},
    {"x": 10, "z": 10},
    {"x": 30, "z": 0},
    {"x": 50, "z": 0},
]

DEFAULT_LAYERS = [
    {
        "name": "Fill",
        "top_elev": 10,
        "bot_elev": -5,
        "gamma": 18,
        "gamma_sat": 20,
        "phi": 25,
        "c_prime": 10,
        "cu": 0,
        "mode": "drained",
    },
]

DEFAULT_GWT = [
    {"x": 0, "z": 5},
    {"x": 50, "z": -1},
]

DEFAULT_NAILS = [
    {"x_head": 30, "z_head": 0, "length": 8, "incl": 15},
    {"x_head": 25, "z_head": 2.5, "length": 10, "incl": 15},
    {"x_head": 20, "z_head": 5, "length": 12, "incl": 15},
]

# Sidebar width
SIDEBAR_W = "370px"

# Shared styles
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
# Helper: build SlopeGeometry from DataTable inputs
# ---------------------------------------------------------------------------

def _safe_float(val, default=0.0):
    """Convert to float, returning default on failure."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _build_geometry(surface_data, layers_data, gwt_enabled, gwt_data,
                    surcharge, surcharge_start, surcharge_end, kh,
                    nails_enabled=False, nails_data=None,
                    nail_bar_dia=25, nail_ddh=150, nail_fy=420,
                    nail_bond=100, nail_spacing=1.5):
    """Parse DataTable rows into a SlopeGeometry.

    Returns (SlopeGeometry, None) on success or (None, error_string) on failure.
    """
    # Surface points
    try:
        pts = []
        for row in (surface_data or []):
            x = _safe_float(row.get("x"))
            z = _safe_float(row.get("z"))
            pts.append((x, z))
        pts.sort(key=lambda p: p[0])
        if len(pts) < 2:
            return None, "Need at least 2 surface points."
    except Exception as e:
        return None, f"Surface error: {e}"

    # Soil layers
    try:
        layers = []
        for row in (layers_data or []):
            layers.append(SlopeSoilLayer(
                name=str(row.get("name", "Layer")),
                top_elevation=_safe_float(row.get("top_elev"), 10),
                bottom_elevation=_safe_float(row.get("bot_elev"), -5),
                gamma=_safe_float(row.get("gamma"), 18),
                gamma_sat=_safe_float(row.get("gamma_sat"), 20) or None,
                phi=_safe_float(row.get("phi"), 0),
                c_prime=_safe_float(row.get("c_prime"), 0),
                cu=_safe_float(row.get("cu"), 0),
                analysis_mode=str(row.get("mode", "drained")),
            ))
        if not layers:
            return None, "Need at least 1 soil layer."
    except ValueError as e:
        return None, f"Layer error: {e}"

    # GWT
    gwt_pts = None
    if gwt_enabled:
        try:
            gwt_pts = []
            for row in (gwt_data or []):
                gx = _safe_float(row.get("x"))
                gz = _safe_float(row.get("z"))
                gwt_pts.append((gx, gz))
            gwt_pts.sort(key=lambda p: p[0])
            if len(gwt_pts) < 2:
                gwt_pts = None
        except Exception:
            gwt_pts = None

    # Surcharge
    q = _safe_float(surcharge, 0)
    q_range = None
    s_start = _safe_float(surcharge_start)
    s_end = _safe_float(surcharge_end)
    if s_start and s_end and s_end > s_start:
        q_range = (s_start, s_end)

    kh_val = max(0.0, _safe_float(kh, 0))

    # Soil nails
    nails = None
    if nails_enabled and nails_data:
        try:
            nails = []
            for row in nails_data:
                nails.append(SoilNail(
                    x_head=_safe_float(row.get("x_head"), 30),
                    z_head=_safe_float(row.get("z_head"), 0),
                    length=_safe_float(row.get("length"), 10),
                    inclination=_safe_float(row.get("incl"), 15),
                    bar_diameter=_safe_float(nail_bar_dia, 25),
                    drill_hole_diameter=_safe_float(nail_ddh, 150),
                    fy=_safe_float(nail_fy, 420),
                    bond_stress=_safe_float(nail_bond, 100),
                    spacing_h=_safe_float(nail_spacing, 1.5),
                ))
        except ValueError as e:
            return None, f"Nail error: {e}"

    try:
        geom = SlopeGeometry(
            surface_points=pts,
            soil_layers=layers,
            gwt_points=gwt_pts,
            surcharge=q,
            surcharge_x_range=q_range,
            kh=kh_val,
            nails=nails,
        )
        return geom, None
    except ValueError as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------------

def _empty_figure(msg="Define geometry and click Run Analysis"):
    """Return a placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=16, color="#94a3b8"))
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=480, margin=dict(l=40, r=20, t=30, b=30),
    )
    return fig


def build_cross_section(geom, result=None):
    """Build a Plotly cross-section figure from geometry and optional result.

    Translates the matplotlib logic from slope_stability/calc_steps.py:340-440.
    """
    fig = go.Figure()

    xs_surface = [p[0] for p in geom.surface_points]
    zs_surface = [p[1] for p in geom.surface_points]
    x_min, x_max = min(xs_surface), max(xs_surface)

    # ── Soil layer fills ─────────────────────────────────────────
    x_fill = np.linspace(x_min, x_max, 300)
    for i, layer in enumerate(geom.soil_layers):
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        z_top = np.array([
            min(geom.ground_elevation_at(x), layer.top_elevation)
            for x in x_fill
        ])
        z_bot = np.full_like(x_fill, layer.bottom_elevation)
        mask = z_top > z_bot
        if not np.any(mask):
            continue

        # Build closed polygon for fill='toself'
        valid_x = x_fill[mask]
        valid_top = z_top[mask]
        valid_bot = z_bot[mask]
        poly_x = list(valid_x) + list(valid_x[::-1])
        poly_z = list(valid_top) + list(valid_bot[::-1])

        fig.add_trace(go.Scatter(
            x=poly_x, y=poly_z,
            fill="toself", fillcolor=color,
            line=dict(color="#888", width=0.5),
            opacity=0.65,
            name=layer.name,
            hoverinfo="name",
        ))

    # ── Ground surface ───────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=xs_surface, y=zs_surface,
        mode="lines",
        line=dict(color="black", width=2.5),
        name="Ground Surface",
    ))

    # ── GWT line ─────────────────────────────────────────────────
    if geom.gwt_points:
        gwt_x = [p[0] for p in geom.gwt_points]
        gwt_z = [p[1] for p in geom.gwt_points]
        fig.add_trace(go.Scatter(
            x=gwt_x, y=gwt_z,
            mode="lines",
            line=dict(color="#2563eb", width=1.5, dash="dash"),
            name="Water Table",
            opacity=0.7,
        ))

    # ── Soil nails ─────────────────────────────────────────────
    if geom.nails:
        nail_x, nail_z = [], []
        for nail in geom.nails:
            beta = math.radians(nail.inclination)
            x0, z0 = nail.x_head, nail.z_head
            x1 = x0 + nail.length * math.cos(beta)
            z1 = z0 - nail.length * math.sin(beta)
            nail_x.extend([x0, x1, None])
            nail_z.extend([z0, z1, None])
        fig.add_trace(go.Scatter(
            x=nail_x, y=nail_z,
            mode="lines",
            line=dict(color="#16a34a", width=2.5),
            name="Soil Nails",
            connectgaps=False,
        ))
        # Nail head markers
        hx = [n.x_head for n in geom.nails]
        hz = [n.z_head for n in geom.nails]
        fig.add_trace(go.Scatter(
            x=hx, y=hz,
            mode="markers",
            marker=dict(symbol="square", size=6, color="#16a34a",
                        line=dict(width=1, color="black")),
            name="Nail Heads",
            showlegend=False,
        ))

    # ── Nail-circle intersection markers (if result + nails) ──
    if result is not None and geom.nails:
        from slope_stability.nails import compute_all_nail_contributions
        contribs = compute_all_nail_contributions(
            geom.nails, result.xc, result.yc, result.radius
        )
        if contribs:
            ix = [c.x_intersect for c in contribs]
            iz = [c.z_intersect for c in contribs]
            labels = [f"T={c.T_design:.0f} kN/m" for c in contribs]
            fig.add_trace(go.Scatter(
                x=ix, y=iz,
                mode="markers+text",
                marker=dict(symbol="diamond", size=10, color="#16a34a",
                            line=dict(width=1.5, color="black")),
                text=labels,
                textposition="top right",
                textfont=dict(size=9, color="#16a34a"),
                name="Nail Intersections",
                showlegend=False,
            ))

    # ── Slip circle + result overlays ────────────────────────────
    if result is not None:
        # Slip circle arc (lower arc between entry/exit)
        theta = np.linspace(0, 2 * np.pi, 720)
        cx = result.xc + result.radius * np.cos(theta)
        cz = result.yc + result.radius * np.sin(theta)

        # Mask: within entry-exit x range and below center
        mask = (
            (cx >= result.x_entry - 0.3) &
            (cx <= result.x_exit + 0.3) &
            (cz <= result.yc)
        )
        arc_x = np.where(mask, cx, np.nan)
        arc_z = np.where(mask, cz, np.nan)

        fig.add_trace(go.Scatter(
            x=arc_x.tolist(), y=arc_z.tolist(),
            mode="lines",
            line=dict(color="red", width=3),
            name=f"Slip Circle (FOS={result.FOS:.3f})",
            connectgaps=False,
        ))

        # Circle center marker
        fig.add_trace(go.Scatter(
            x=[result.xc], y=[result.yc],
            mode="markers+text",
            marker=dict(symbol="cross-thin", size=14, color="red",
                        line=dict(width=2, color="red")),
            text=[f"({result.xc:.1f}, {result.yc:.1f})"],
            textposition="top right",
            textfont=dict(size=10, color="red"),
            name="Circle Center",
            showlegend=False,
        ))

        # Entry / exit markers
        z_entry = geom.ground_elevation_at(result.x_entry)
        z_exit = geom.ground_elevation_at(result.x_exit)
        fig.add_trace(go.Scatter(
            x=[result.x_entry, result.x_exit],
            y=[z_entry, z_exit],
            mode="markers+text",
            marker=dict(
                symbol=["triangle-down", "triangle-up"],
                size=10, color="red",
            ),
            text=["Entry", "Exit"],
            textposition=["bottom center", "bottom center"],
            textfont=dict(size=9),
            name="Entry/Exit",
            showlegend=False,
        ))

        # Slice boundary lines (single trace with None separators)
        if result.slice_data:
            sl_x, sl_z = [], []
            for s in result.slice_data:
                sl_x.extend([s.x_mid, s.x_mid, None])
                sl_z.extend([s.z_base, s.z_top, None])
            fig.add_trace(go.Scatter(
                x=sl_x, y=sl_z,
                mode="lines",
                line=dict(color="black", width=0.4),
                opacity=0.35,
                name="Slices",
                showlegend=False,
                connectgaps=False,
            ))

    # ── Layout ───────────────────────────────────────────────────
    # Axis padding
    z_min_surf = min(zs_surface)
    z_max_surf = max(zs_surface)
    # Include layer bottoms in vertical range
    z_bot_all = min(l.bottom_elevation for l in geom.soil_layers)
    z_plot_min = min(z_min_surf, z_bot_all) - 2
    z_plot_max = z_max_surf + 5
    if result is not None:
        z_plot_max = max(z_plot_max, result.yc + 2)

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            title="Distance (m)",
            scaleanchor="y", scaleratio=1,
            range=[x_min - 3, x_max + 3],
        ),
        yaxis=dict(
            title="Elevation (m)",
            range=[z_plot_min, z_plot_max],
        ),
        height=480,
        margin=dict(l=50, r=20, t=35, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=10),
        ),
        title=dict(text="Slope Cross-Section", font=dict(size=14)),
    )
    return fig


def build_search_heatmap(search_result):
    """Build a scatter heatmap of FOS at each trial circle center."""
    if not search_result.grid_fos:
        return _empty_figure("No search data")

    xc_vals = [g["xc"] for g in search_result.grid_fos]
    yc_vals = [g["yc"] for g in search_result.grid_fos]
    fos_vals = [g["FOS"] for g in search_result.grid_fos]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xc_vals, y=yc_vals,
        mode="markers",
        marker=dict(
            size=14,
            color=fos_vals,
            colorscale="RdYlGn",
            cmin=min(fos_vals),
            cmax=max(fos_vals),
            colorbar=dict(title="FOS"),
            line=dict(width=0.5, color="black"),
        ),
        text=[f"FOS={f:.3f}" for f in fos_vals],
        hoverinfo="text+x+y",
        name="Trial Centers",
    ))

    if search_result.critical:
        fig.add_trace(go.Scatter(
            x=[search_result.critical.xc],
            y=[search_result.critical.yc],
            mode="markers+text",
            marker=dict(symbol="star", size=18, color="red",
                        line=dict(width=1.5, color="black")),
            text=[f"Critical\nFOS={search_result.critical.FOS:.3f}"],
            textposition="top right",
            textfont=dict(size=10, color="red"),
            name="Critical Center",
        ))

    fig.update_layout(
        template="plotly_white",
        title="Critical Surface Search — FOS at Each Trial Center",
        xaxis_title="Circle Center X (m)",
        yaxis_title="Circle Center Y (m)",
        height=380,
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(scaleanchor="y", scaleratio=1),
    )
    return fig


# ---------------------------------------------------------------------------
# HTML builders for results
# ---------------------------------------------------------------------------

def build_fos_badge(result):
    """Return a styled FOS badge div."""
    if result is None:
        return html.Div("--", style={
            "fontSize": "1.5rem", "color": "#94a3b8", "textAlign": "center",
        })
    color = "#16a34a" if result.is_stable else "#dc2626"
    label = "STABLE" if result.is_stable else "UNSTABLE"
    bg = "rgba(22,163,74,0.1)" if result.is_stable else "rgba(220,38,38,0.1)"
    return html.Div([
        html.Div(f"FOS = {result.FOS:.3f}", style={
            "fontSize": "2rem", "fontWeight": "700", "color": color,
        }),
        html.Div(label, style={
            "fontSize": "0.85rem", "fontWeight": "600", "color": color,
            "letterSpacing": "0.1em",
        }),
    ], style={
        "textAlign": "center", "padding": "10px",
        "background": bg, "borderRadius": "8px",
        "border": f"2px solid {color}",
    })


def build_results_summary(result):
    """Return an HTML div with results summary."""
    if result is None:
        return html.Div()
    rows = [
        ("Method", result.method),
        ("Circle Center", f"({result.xc:.2f}, {result.yc:.2f}) m"),
        ("Radius", f"{result.radius:.2f} m"),
        ("Entry x", f"{result.x_entry:.2f} m"),
        ("Exit x", f"{result.x_exit:.2f} m"),
        ("Slices", str(result.n_slices)),
    ]
    if result.has_seismic:
        rows.append(("Seismic kh", f"{result.kh:.3f}"))
    if result.n_nails_active > 0:
        rows.append(("Nails Active", str(result.n_nails_active)))
        rows.append(("Nail Resisting", f"{result.nail_resisting_kN_per_m:.1f} kN/m"))

    return html.Table([
        html.Tbody([
            html.Tr([
                html.Td(label, style={
                    "fontWeight": "600", "padding": "3px 12px 3px 0",
                    "color": "#475569", "fontSize": "0.85rem",
                }),
                html.Td(val, style={
                    "padding": "3px 0", "fontSize": "0.85rem",
                }),
            ]) for label, val in rows
        ]),
    ], style={"borderCollapse": "collapse"})


def build_comparison_table(result):
    """Return an HTML table comparing Fellenius/Bishop/Spencer FOS."""
    if result is None:
        return html.Div()
    rows = []
    if result.FOS_fellenius is not None:
        rows.append(("Fellenius", f"{result.FOS_fellenius:.3f}"))
    if result.FOS_bishop is not None:
        rows.append(("Bishop", f"{result.FOS_bishop:.3f}"))
    if result.method == "Spencer" or result.theta_spencer is not None:
        rows.append(("Spencer", f"{result.FOS:.3f}"))
    elif result.method == "Bishop" and result.FOS_bishop is None:
        rows.append(("Bishop", f"{result.FOS:.3f}"))
    elif result.method == "Fellenius" and result.FOS_fellenius is None:
        rows.append(("Fellenius", f"{result.FOS:.3f}"))

    if not rows:
        return html.Div()

    th_style = {
        "padding": "6px 14px", "textAlign": "left", "fontWeight": "600",
        "borderBottom": "2px solid #e2e8f0", "fontSize": "0.82rem",
        "color": "#334155",
    }
    td_style = {
        "padding": "5px 14px", "fontSize": "0.85rem",
        "borderBottom": "1px solid #f1f5f9",
    }

    return html.Div([
        html.Div("Method Comparison", style={
            "fontWeight": "600", "fontSize": "0.9rem", "marginBottom": "6px",
            "color": "#1e293b",
        }),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Method", style=th_style),
                html.Th("FOS", style=th_style),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(m, style=td_style),
                    html.Td(f, style=td_style),
                ]) for m, f in rows
            ]),
        ], style={"borderCollapse": "collapse", "width": "100%"}),
    ], style={"marginTop": "12px"})


def build_slice_table(result):
    """Return an expandable details element with per-slice data."""
    if result is None or not result.slice_data:
        return html.Div()

    th_style = {
        "padding": "4px 8px", "textAlign": "right", "fontWeight": "600",
        "borderBottom": "2px solid #e2e8f0", "fontSize": "0.75rem",
        "color": "#475569",
    }
    td_style = {
        "padding": "3px 8px", "textAlign": "right", "fontSize": "0.78rem",
        "borderBottom": "1px solid #f1f5f9",
    }
    headers = ["#", "x_mid", "z_top", "z_base", "Weight", "alpha", "u (kPa)"]
    head_row = html.Tr([
        html.Th(h, style={**th_style, "textAlign": "left" if i == 0 else "right"})
        for i, h in enumerate(headers)
    ])
    body_rows = []
    for j, s in enumerate(result.slice_data):
        body_rows.append(html.Tr([
            html.Td(str(j + 1), style={**td_style, "textAlign": "left"}),
            html.Td(f"{s.x_mid:.2f}", style=td_style),
            html.Td(f"{s.z_top:.2f}", style=td_style),
            html.Td(f"{s.z_base:.2f}", style=td_style),
            html.Td(f"{s.weight:.1f}", style=td_style),
            html.Td(f"{s.alpha_deg:.1f}", style=td_style),
            html.Td(f"{s.pore_pressure:.1f}", style=td_style),
        ]))

    return html.Details([
        html.Summary("Slice Data", style={
            "cursor": "pointer", "fontWeight": "600", "fontSize": "0.9rem",
            "color": "#1e293b", "marginBottom": "6px",
        }),
        html.Div([
            html.Table([
                html.Thead(head_row),
                html.Tbody(body_rows),
            ], style={"borderCollapse": "collapse", "width": "100%"}),
        ], style={"maxHeight": "300px", "overflowY": "auto", "marginTop": "6px"}),
    ], style={"marginTop": "12px"})


# ---------------------------------------------------------------------------
# Dash App
# ---------------------------------------------------------------------------

app = Dash(__name__)
app.title = "Slope Stability Analysis"

# -- DataTable column definitions --

surface_cols = [
    {"name": "x (m)", "id": "x", "type": "numeric", "editable": True},
    {"name": "z (m)", "id": "z", "type": "numeric", "editable": True},
]

layer_cols = [
    {"name": "Name", "id": "name", "type": "text", "editable": True},
    {"name": "Top Elev", "id": "top_elev", "type": "numeric", "editable": True},
    {"name": "Bot Elev", "id": "bot_elev", "type": "numeric", "editable": True},
    {"name": "gamma", "id": "gamma", "type": "numeric", "editable": True},
    {"name": "gamma_sat", "id": "gamma_sat", "type": "numeric", "editable": True},
    {"name": "phi (deg)", "id": "phi", "type": "numeric", "editable": True},
    {"name": "c' (kPa)", "id": "c_prime", "type": "numeric", "editable": True},
    {"name": "cu (kPa)", "id": "cu", "type": "numeric", "editable": True},
    {"name": "Mode", "id": "mode", "type": "text", "editable": True,
     "presentation": "dropdown"},
]

gwt_cols = [
    {"name": "x (m)", "id": "x", "type": "numeric", "editable": True},
    {"name": "z (m)", "id": "z", "type": "numeric", "editable": True},
]

nail_cols = [
    {"name": "x_head (m)", "id": "x_head", "type": "numeric", "editable": True},
    {"name": "z_head (m)", "id": "z_head", "type": "numeric", "editable": True},
    {"name": "Length (m)", "id": "length", "type": "numeric", "editable": True},
    {"name": "Incl (deg)", "id": "incl", "type": "numeric", "editable": True},
]

# Common DataTable styling
DT_STYLE_CELL = {
    "textAlign": "right", "padding": "4px 8px", "fontSize": "0.8rem",
    "fontFamily": "Segoe UI, system-ui, sans-serif",
    "minWidth": "55px",
}
DT_STYLE_HEADER = {
    "fontWeight": "600", "textAlign": "center", "fontSize": "0.75rem",
    "backgroundColor": "#f1f5f9", "color": "#334155",
}
DT_STYLE_TABLE = {"overflowX": "auto"}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

# Pre-compute initial cross-section from default geometry
_init_geom, _ = _build_geometry(DEFAULT_SURFACE, DEFAULT_LAYERS, False, None,
                                0, None, None, 0)
_init_fig = build_cross_section(_init_geom) if _init_geom else _empty_figure()

app.layout = html.Div([

    # ═══════════════════════ HEADER ═══════════════════════════════
    html.Div([
        html.Div([
            html.Span("SLOPE STABILITY ANALYSIS", style={
                "fontSize": "1.1rem", "fontWeight": "700", "color": "#1e293b",
                "letterSpacing": "0.05em",
            }),
        ], style={"flex": "1"}),
        html.Div(id="fos-badge", style={"flex": "0 0 auto"}),
    ], style={
        "display": "flex", "alignItems": "center", "justifyContent": "space-between",
        "padding": "10px 20px",
        "borderBottom": "2px solid #e2e8f0",
        "background": "#f8fafc",
    }),

    # ═══════════════════════ BODY ═════════════════════════════════
    html.Div([

        # ────────── SIDEBAR ──────────────────────────────────────
        html.Div([

            # ── DXF Import ────────────────────────────────────────
            html.Details([
                html.Summary("DXF Import", style={
                    "fontWeight": "700", "fontSize": "0.95rem", "cursor": "pointer",
                    "color": "#1e293b", "padding": "6px 0",
                }),
                html.Div([
                    # Hidden stores for DXF data
                    dcc.Store(id="dxf-store"),
                    dcc.Store(id="dxf-discovery-store"),

                    # Upload zone
                    dcc.Upload(
                        id="dxf-upload",
                        children=html.Div([
                            html.Span("Drag & drop .dxf or "),
                            html.A("browse", style={"color": "#2563eb",
                                                     "textDecoration": "underline"}),
                        ], style={"textAlign": "center", "fontSize": "0.82rem",
                                  "color": "#64748b"}),
                        style={
                            "border": "2px dashed #cbd5e1",
                            "borderRadius": "6px",
                            "padding": "16px 8px",
                            "cursor": "pointer",
                            "background": "#f8fafc",
                        },
                        accept=".dxf",
                    ),

                    # Status / error message
                    html.Div(id="dxf-status", style={
                        "marginTop": "6px", "fontSize": "0.82rem",
                        "color": "#16a34a", "minHeight": "18px",
                    }),

                    # Mapping section (hidden until upload)
                    html.Div(id="dxf-mapping-section", children=[
                        # Units + Flip Y row
                        html.Div([
                            html.Div([
                                html.Label("Units", style=LABEL_STYLE),
                                dcc.Dropdown(
                                    id="dxf-units-dropdown",
                                    options=[
                                        {"label": "meters (m)", "value": "m"},
                                        {"label": "feet (ft)", "value": "ft"},
                                        {"label": "millimeters (mm)", "value": "mm"},
                                        {"label": "centimeters (cm)", "value": "cm"},
                                        {"label": "inches (in)", "value": "in"},
                                    ],
                                    value="m",
                                    clearable=False,
                                    style={"fontSize": "0.82rem"},
                                ),
                            ], style={"flex": "1", "marginRight": "6px"}),
                            html.Div([
                                html.Label("\u00a0", style=LABEL_STYLE),
                                dcc.Checklist(
                                    id="dxf-flip-y",
                                    options=[{"label": " Flip Y", "value": "on"}],
                                    value=[],
                                    style={"fontSize": "0.82rem", "paddingTop": "4px"},
                                ),
                            ], style={"flex": "0 0 auto"}),
                        ], style={"display": "flex", "marginBottom": "8px"}),

                        # Surface layer (required)
                        html.Label("Surface Layer (required)", style=LABEL_STYLE),
                        dcc.Dropdown(
                            id="dxf-surface-dropdown",
                            placeholder="Select surface layer...",
                            style={"fontSize": "0.82rem", "marginBottom": "8px"},
                        ),

                        # Soil boundaries
                        html.Label("Soil Boundaries", style=LABEL_STYLE),
                        *[html.Div(
                            id=f"dxf-boundary-row-{i}",
                            children=[
                                html.Div([
                                    dcc.Dropdown(
                                        id=f"dxf-boundary-layer-{i}",
                                        placeholder="DXF layer...",
                                        style={"fontSize": "0.82rem"},
                                    ),
                                ], style={"flex": "1", "marginRight": "4px"}),
                                html.Div([
                                    dcc.Input(
                                        id=f"dxf-boundary-name-{i}",
                                        placeholder="Soil name",
                                        style={**INPUT_STYLE, "fontSize": "0.82rem"},
                                    ),
                                ], style={"flex": "1"}),
                            ],
                            style={"display": "flex" if i == 0 else "none",
                                   "marginBottom": "4px"},
                        ) for i in range(5)],
                        html.Button("+ Add Boundary", id="btn-add-boundary",
                                    style={**BTN_STYLE, "marginTop": "2px",
                                           "marginBottom": "8px"}),

                        # Water table layer (optional)
                        html.Label("Water Table Layer", style=LABEL_STYLE),
                        dcc.Dropdown(
                            id="dxf-gwt-dropdown",
                            placeholder="(none)",
                            style={"fontSize": "0.82rem", "marginBottom": "8px"},
                        ),

                        # Nail layer (optional)
                        html.Label("Nail Layer", style=LABEL_STYLE),
                        dcc.Dropdown(
                            id="dxf-nail-dropdown",
                            placeholder="(none)",
                            style={"fontSize": "0.82rem", "marginBottom": "10px"},
                        ),

                        # Import button
                        html.Button("Import DXF", id="btn-import-dxf",
                                    style=BTN_PRIMARY),
                    ], style={"display": "none", "marginTop": "8px"}),
                ], style={"marginTop": "6px"}),
            ], open=False, style=SECTION_STYLE),

            # ── Geometry ─────────────────────────────────────────
            html.Details([
                html.Summary("Geometry", style={
                    "fontWeight": "700", "fontSize": "0.95rem", "cursor": "pointer",
                    "color": "#1e293b", "padding": "6px 0",
                }),
                html.Div([
                    html.Label("Ground Surface Points", style=LABEL_STYLE),
                    dash_table.DataTable(
                        id="surface-table",
                        columns=surface_cols,
                        data=DEFAULT_SURFACE.copy(),
                        editable=True,
                        row_deletable=True,
                        style_cell=DT_STYLE_CELL,
                        style_header=DT_STYLE_HEADER,
                        style_table=DT_STYLE_TABLE,
                    ),
                    html.Div([
                        html.Button("Add Point", id="btn-add-point", style=BTN_STYLE),
                        html.Button("Reset", id="btn-reset-surface", style=BTN_STYLE),
                    ], style={"marginTop": "6px"}),
                ], style={"marginTop": "6px"}),
            ], open=True, style=SECTION_STYLE),

            # ── Soil Layers ──────────────────────────────────────
            html.Details([
                html.Summary("Soil Layers", style={
                    "fontWeight": "700", "fontSize": "0.95rem", "cursor": "pointer",
                    "color": "#1e293b", "padding": "6px 0",
                }),
                html.Div([
                    dash_table.DataTable(
                        id="layers-table",
                        columns=layer_cols,
                        data=DEFAULT_LAYERS.copy(),
                        editable=True,
                        row_deletable=True,
                        dropdown={
                            "mode": {
                                "options": [
                                    {"label": "drained", "value": "drained"},
                                    {"label": "undrained", "value": "undrained"},
                                ],
                            },
                        },
                        style_cell={**DT_STYLE_CELL, "minWidth": "50px"},
                        style_header=DT_STYLE_HEADER,
                        style_table=DT_STYLE_TABLE,
                    ),
                    html.Div([
                        html.Button("Add Layer", id="btn-add-layer", style=BTN_STYLE),
                    ], style={"marginTop": "6px"}),
                ], style={"marginTop": "6px"}),
            ], open=True, style=SECTION_STYLE),

            # ── Water & Loading ──────────────────────────────────
            html.Details([
                html.Summary("Water & Loading", style={
                    "fontWeight": "700", "fontSize": "0.95rem", "cursor": "pointer",
                    "color": "#1e293b", "padding": "6px 0",
                }),
                html.Div([
                    dcc.Checklist(
                        id="gwt-toggle",
                        options=[{"label": " Enable Water Table", "value": "on"}],
                        value=[],
                        style={"marginBottom": "6px", "fontSize": "0.85rem"},
                    ),
                    html.Div(id="gwt-section", children=[
                        html.Label("GWT Points", style=LABEL_STYLE),
                        dash_table.DataTable(
                            id="gwt-table",
                            columns=gwt_cols,
                            data=DEFAULT_GWT.copy(),
                            editable=True,
                            row_deletable=True,
                            style_cell=DT_STYLE_CELL,
                            style_header=DT_STYLE_HEADER,
                            style_table=DT_STYLE_TABLE,
                        ),
                        html.Button("Add GWT Point", id="btn-add-gwt",
                                    style={**BTN_STYLE, "marginTop": "4px"}),
                    ], style={"display": "none"}),

                    html.Hr(style={"margin": "10px 0", "borderColor": "#e2e8f0"}),

                    html.Label("Surcharge (kPa)", style=LABEL_STYLE),
                    dcc.Input(id="input-surcharge", type="number", value=0,
                              style=INPUT_STYLE),

                    html.Div([
                        html.Div([
                            html.Label("x start", style=LABEL_STYLE),
                            dcc.Input(id="input-surcharge-start", type="number",
                                      placeholder="start", style=INPUT_STYLE),
                        ], style={"flex": "1", "marginRight": "6px"}),
                        html.Div([
                            html.Label("x end", style=LABEL_STYLE),
                            dcc.Input(id="input-surcharge-end", type="number",
                                      placeholder="end", style=INPUT_STYLE),
                        ], style={"flex": "1"}),
                    ], style={"display": "flex", "marginTop": "6px"}),

                    html.Hr(style={"margin": "10px 0", "borderColor": "#e2e8f0"}),

                    html.Label("Seismic kh", style=LABEL_STYLE),
                    dcc.Input(id="input-kh", type="number", value=0,
                              min=0, max=1, step=0.01, style=INPUT_STYLE),
                ], style={"marginTop": "6px"}),
            ], open=False, style=SECTION_STYLE),

            # ── Soil Nails ───────────────────────────────────────
            html.Details([
                html.Summary("Soil Nails", style={
                    "fontWeight": "700", "fontSize": "0.95rem", "cursor": "pointer",
                    "color": "#1e293b", "padding": "6px 0",
                }),
                html.Div([
                    dcc.Checklist(
                        id="nail-toggle",
                        options=[{"label": " Enable Soil Nails", "value": "on"}],
                        value=[],
                        style={"marginBottom": "6px", "fontSize": "0.85rem"},
                    ),
                    html.Div(id="nail-section", children=[
                        html.Div("Nail Properties (shared)", style={
                            "fontWeight": "600", "fontSize": "0.82rem",
                            "color": "#475569", "marginBottom": "6px",
                        }),
                        html.Div([
                            html.Div([
                                html.Label("Bar dia (mm)", style=LABEL_STYLE),
                                dcc.Input(id="input-nail-bar-dia", type="number",
                                          value=25, style=INPUT_STYLE),
                            ], style={"flex": "1", "marginRight": "6px"}),
                            html.Div([
                                html.Label("DDH (mm)", style=LABEL_STYLE),
                                dcc.Input(id="input-nail-ddh", type="number",
                                          value=150, style=INPUT_STYLE),
                            ], style={"flex": "1"}),
                        ], style={"display": "flex", "marginBottom": "6px"}),
                        html.Div([
                            html.Div([
                                html.Label("fy (MPa)", style=LABEL_STYLE),
                                dcc.Input(id="input-nail-fy", type="number",
                                          value=420, style=INPUT_STYLE),
                            ], style={"flex": "1", "marginRight": "6px"}),
                            html.Div([
                                html.Label("Bond (kPa)", style=LABEL_STYLE),
                                dcc.Input(id="input-nail-bond", type="number",
                                          value=100, style=INPUT_STYLE),
                            ], style={"flex": "1"}),
                        ], style={"display": "flex", "marginBottom": "6px"}),
                        html.Div([
                            html.Label("H spacing (m)", style=LABEL_STYLE),
                            dcc.Input(id="input-nail-spacing", type="number",
                                      value=1.5, step=0.1, style=INPUT_STYLE),
                        ], style={"marginBottom": "8px"}),

                        html.Div("Nail Locations", style={
                            "fontWeight": "600", "fontSize": "0.82rem",
                            "color": "#475569", "marginBottom": "4px",
                        }),
                        dash_table.DataTable(
                            id="nail-table",
                            columns=nail_cols,
                            data=DEFAULT_NAILS.copy(),
                            editable=True,
                            row_deletable=True,
                            style_cell={**DT_STYLE_CELL, "minWidth": "60px"},
                            style_header=DT_STYLE_HEADER,
                            style_table=DT_STYLE_TABLE,
                        ),
                        html.Div([
                            html.Button("Add Nail", id="btn-add-nail", style=BTN_STYLE),
                            html.Button("Reset", id="btn-reset-nails", style=BTN_STYLE),
                        ], style={"marginTop": "6px"}),
                    ], style={"display": "none"}),
                ], style={"marginTop": "6px"}),
            ], open=False, style=SECTION_STYLE),

            # ── Analysis Controls ────────────────────────────────
            html.Details([
                html.Summary("Analysis", style={
                    "fontWeight": "700", "fontSize": "0.95rem", "cursor": "pointer",
                    "color": "#1e293b", "padding": "6px 0",
                }),
                html.Div([
                    html.Label("Method", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="dd-method",
                        options=[
                            {"label": "Bishop (simplified)", "value": "bishop"},
                            {"label": "Fellenius (OMS)", "value": "fellenius"},
                            {"label": "Spencer", "value": "spencer"},
                        ],
                        value="bishop",
                        clearable=False,
                        style={"fontSize": "0.85rem", "marginBottom": "8px"},
                    ),

                    html.Label("Search Mode", style=LABEL_STYLE),
                    dcc.RadioItems(
                        id="radio-mode",
                        options=[
                            {"label": "Single Circle", "value": "single"},
                            {"label": "Grid Search", "value": "search"},
                        ],
                        value="single",
                        inline=True,
                        style={"fontSize": "0.85rem", "marginBottom": "10px"},
                        labelStyle={"marginRight": "14px"},
                    ),

                    # -- Single circle inputs --
                    html.Div(id="single-inputs", children=[
                        html.Div([
                            html.Div([
                                html.Label("xc (m)", style=LABEL_STYLE),
                                dcc.Input(id="input-xc", type="number",
                                          value=20, style=INPUT_STYLE),
                            ], style={"flex": "1", "marginRight": "6px"}),
                            html.Div([
                                html.Label("yc (m)", style=LABEL_STYLE),
                                dcc.Input(id="input-yc", type="number",
                                          value=15, style=INPUT_STYLE),
                            ], style={"flex": "1", "marginRight": "6px"}),
                            html.Div([
                                html.Label("R (m)", style=LABEL_STYLE),
                                dcc.Input(id="input-radius", type="number",
                                          value=13, style=INPUT_STYLE),
                            ], style={"flex": "1"}),
                        ], style={"display": "flex"}),
                    ]),

                    # -- Grid search inputs --
                    html.Div(id="search-inputs", children=[
                        html.Div([
                            html.Div([
                                html.Label("x_min", style=LABEL_STYLE),
                                dcc.Input(id="input-x-min", type="number",
                                          value=5, style=INPUT_STYLE),
                            ], style={"flex": "1", "marginRight": "6px"}),
                            html.Div([
                                html.Label("x_max", style=LABEL_STYLE),
                                dcc.Input(id="input-x-max", type="number",
                                          value=35, style=INPUT_STYLE),
                            ], style={"flex": "1"}),
                        ], style={"display": "flex", "marginBottom": "6px"}),
                        html.Div([
                            html.Div([
                                html.Label("y_min", style=LABEL_STYLE),
                                dcc.Input(id="input-y-min", type="number",
                                          value=11, style=INPUT_STYLE),
                            ], style={"flex": "1", "marginRight": "6px"}),
                            html.Div([
                                html.Label("y_max", style=LABEL_STYLE),
                                dcc.Input(id="input-y-max", type="number",
                                          value=25, style=INPUT_STYLE),
                            ], style={"flex": "1"}),
                        ], style={"display": "flex", "marginBottom": "6px"}),
                        html.Div([
                            html.Div([
                                html.Label("nx", style=LABEL_STYLE),
                                dcc.Input(id="input-nx", type="number",
                                          value=10, style=INPUT_STYLE),
                            ], style={"flex": "1", "marginRight": "6px"}),
                            html.Div([
                                html.Label("ny", style=LABEL_STYLE),
                                dcc.Input(id="input-ny", type="number",
                                          value=10, style=INPUT_STYLE),
                            ], style={"flex": "1"}),
                        ], style={"display": "flex"}),
                    ], style={"display": "none"}),

                    html.Hr(style={"margin": "10px 0", "borderColor": "#e2e8f0"}),

                    html.Div([
                        html.Div([
                            html.Label("N slices", style=LABEL_STYLE),
                            dcc.Input(id="input-nslices", type="number",
                                      value=30, min=5, max=100, style=INPUT_STYLE),
                        ], style={"flex": "1", "marginRight": "6px"}),
                        html.Div([
                            html.Label("FOS req", style=LABEL_STYLE),
                            dcc.Input(id="input-fos-req", type="number",
                                      value=1.5, step=0.1, style=INPUT_STYLE),
                        ], style={"flex": "1"}),
                    ], style={"display": "flex", "marginBottom": "8px"}),

                    dcc.Checklist(
                        id="chk-compare",
                        options=[{"label": " Compare All Methods", "value": "on"}],
                        value=[],
                        style={"fontSize": "0.85rem", "marginBottom": "12px"},
                    ),

                    html.Button("Run Analysis", id="btn-run",
                                style=BTN_PRIMARY),

                ], style={"marginTop": "6px"}),
            ], open=True, style=SECTION_STYLE),

            # Error message area
            html.Div(id="error-msg", style={
                "marginTop": "10px", "color": "#dc2626", "fontSize": "0.85rem",
            }),

        ], style={
            "width": SIDEBAR_W, "minWidth": SIDEBAR_W,
            "padding": "12px 14px",
            "borderRight": "1px solid #e2e8f0",
            "overflowY": "auto",
            "height": "calc(100vh - 55px)",
            "background": "#ffffff",
        }),

        # ────────── MAIN AREA ────────────────────────────────────
        html.Div([
            # Cross-section plot
            dcc.Graph(id="cross-section", figure=_init_fig,
                      config={"displayModeBar": True, "scrollZoom": True}),

            # Results row
            html.Div([
                html.Div(id="results-summary", style={"flex": "1"}),
                html.Div(id="comparison-table", style={"flex": "1"}),
            ], style={"display": "flex", "gap": "20px", "padding": "0 16px",
                       "flexWrap": "wrap"}),

            # Search heatmap (hidden until grid search)
            html.Div(id="heatmap-container", children=[
                dcc.Graph(id="search-heatmap", figure=_empty_figure(""),
                          config={"displayModeBar": True}),
            ], style={"display": "none"}),

            # Slice data table
            html.Div(id="slice-data-container", style={"padding": "0 16px"}),

        ], style={
            "flex": "1",
            "overflowY": "auto",
            "height": "calc(100vh - 55px)",
            "padding": "8px 0",
        }),

    ], style={"display": "flex"}),

], style={
    "fontFamily": "Segoe UI, system-ui, -apple-system, sans-serif",
    "margin": "0", "padding": "0",
    "height": "100vh", "overflow": "hidden",
})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# Callback 1: Live cross-section preview on any geometry edit
@app.callback(
    Output("cross-section", "figure", allow_duplicate=True),
    [Input("surface-table", "data"),
     Input("layers-table", "data"),
     Input("gwt-toggle", "value"),
     Input("gwt-table", "data"),
     Input("nail-toggle", "value"),
     Input("nail-table", "data")],
    prevent_initial_call=True,
)
def update_preview(surface_data, layers_data, gwt_toggle, gwt_data,
                   nail_toggle, nail_data):
    gwt_on = "on" in (gwt_toggle or [])
    nail_on = "on" in (nail_toggle or [])
    geom, err = _build_geometry(
        surface_data, layers_data, gwt_on, gwt_data,
        surcharge=0, surcharge_start=None, surcharge_end=None, kh=0,
        nails_enabled=nail_on, nails_data=nail_data,
    )
    if geom is None:
        return _empty_figure(err or "Invalid geometry")
    return build_cross_section(geom)


# Callback 2: Add point / reset surface table
@app.callback(
    Output("surface-table", "data"),
    [Input("btn-add-point", "n_clicks"),
     Input("btn-reset-surface", "n_clicks")],
    State("surface-table", "data"),
    prevent_initial_call=True,
)
def modify_surface(add_clicks, reset_clicks, current_data):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "btn-reset-surface":
        return DEFAULT_SURFACE.copy()
    # Add point — append a row with x slightly past the last
    data = list(current_data or [])
    if data:
        last_x = _safe_float(data[-1].get("x"), 50)
        last_z = _safe_float(data[-1].get("z"), 0)
        data.append({"x": last_x + 10, "z": last_z})
    else:
        data.append({"x": 0, "z": 0})
    return data


# Callback 3: Add layer
@app.callback(
    Output("layers-table", "data"),
    Input("btn-add-layer", "n_clicks"),
    State("layers-table", "data"),
    prevent_initial_call=True,
)
def add_layer(n_clicks, current_data):
    data = list(current_data or [])
    if len(data) >= 5:
        raise PreventUpdate
    # Default new layer below the last one
    if data:
        prev_bot = _safe_float(data[-1].get("bot_elev"), -5)
        new_top = prev_bot
        new_bot = prev_bot - 5
    else:
        new_top = 10
        new_bot = -5
    data.append({
        "name": f"Layer {len(data) + 1}",
        "top_elev": new_top,
        "bot_elev": new_bot,
        "gamma": 18,
        "gamma_sat": 20,
        "phi": 30,
        "c_prime": 0,
        "cu": 0,
        "mode": "drained",
    })
    return data


# Callback 4: Add GWT point
@app.callback(
    Output("gwt-table", "data"),
    Input("btn-add-gwt", "n_clicks"),
    State("gwt-table", "data"),
    prevent_initial_call=True,
)
def add_gwt_point(n_clicks, current_data):
    data = list(current_data or [])
    if data:
        last_x = _safe_float(data[-1].get("x"), 50)
        last_z = _safe_float(data[-1].get("z"), 0)
        data.append({"x": last_x + 10, "z": last_z})
    else:
        data.append({"x": 0, "z": 0})
    return data


# Callback 4b: Add nail / reset nails
@app.callback(
    Output("nail-table", "data"),
    [Input("btn-add-nail", "n_clicks"),
     Input("btn-reset-nails", "n_clicks")],
    State("nail-table", "data"),
    prevent_initial_call=True,
)
def modify_nails(add_clicks, reset_clicks, current_data):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "btn-reset-nails":
        return DEFAULT_NAILS.copy()
    data = list(current_data or [])
    if data:
        last_x = _safe_float(data[-1].get("x_head"), 30)
        last_z = _safe_float(data[-1].get("z_head"), 0)
        last_l = _safe_float(data[-1].get("length"), 10)
        data.append({"x_head": last_x, "z_head": last_z + 2.5,
                     "length": last_l, "incl": 15})
    else:
        data.append({"x_head": 30, "z_head": 0, "length": 10, "incl": 15})
    return data


# Callback 5: Toggle single/search input visibility + GWT + nail sections
@app.callback(
    [Output("single-inputs", "style"),
     Output("search-inputs", "style"),
     Output("gwt-section", "style"),
     Output("nail-section", "style")],
    [Input("radio-mode", "value"),
     Input("gwt-toggle", "value"),
     Input("nail-toggle", "value")],
)
def toggle_mode(mode, gwt_toggle, nail_toggle):
    single_vis = {"display": "block"} if mode == "single" else {"display": "none"}
    search_vis = {"display": "block"} if mode == "search" else {"display": "none"}
    gwt_vis = {"display": "block"} if "on" in (gwt_toggle or []) else {"display": "none"}
    nail_vis = {"display": "block"} if "on" in (nail_toggle or []) else {"display": "none"}
    return single_vis, search_vis, gwt_vis, nail_vis


# Callback 7: Auto-populate search bounds from geometry
@app.callback(
    [Output("input-x-min", "value"),
     Output("input-x-max", "value"),
     Output("input-y-min", "value"),
     Output("input-y-max", "value")],
    Input("radio-mode", "value"),
    State("surface-table", "data"),
    prevent_initial_call=True,
)
def auto_populate_bounds(mode, surface_data):
    if mode != "search":
        raise PreventUpdate
    if not surface_data:
        raise PreventUpdate
    xs = [_safe_float(r.get("x")) for r in surface_data]
    zs = [_safe_float(r.get("z")) for r in surface_data]
    x_min, x_max = min(xs), max(xs)
    z_min, z_max = min(zs), max(zs)
    slope_height = z_max - z_min
    return (
        round(x_min, 1),
        round(x_max, 1),
        round(z_max + 1, 1),
        round(z_max + 2 * slope_height, 1),
    )


# Callback 6: MAIN ANALYSIS — the core callback
@app.callback(
    [Output("cross-section", "figure"),
     Output("fos-badge", "children"),
     Output("results-summary", "children"),
     Output("comparison-table", "children"),
     Output("heatmap-container", "style"),
     Output("search-heatmap", "figure"),
     Output("slice-data-container", "children"),
     Output("error-msg", "children")],
    Input("btn-run", "n_clicks"),
    [State("surface-table", "data"),
     State("layers-table", "data"),
     State("gwt-toggle", "value"),
     State("gwt-table", "data"),
     State("input-surcharge", "value"),
     State("input-surcharge-start", "value"),
     State("input-surcharge-end", "value"),
     State("input-kh", "value"),
     State("nail-toggle", "value"),
     State("nail-table", "data"),
     State("input-nail-bar-dia", "value"),
     State("input-nail-ddh", "value"),
     State("input-nail-fy", "value"),
     State("input-nail-bond", "value"),
     State("input-nail-spacing", "value"),
     State("dd-method", "value"),
     State("radio-mode", "value"),
     State("input-xc", "value"),
     State("input-yc", "value"),
     State("input-radius", "value"),
     State("input-x-min", "value"),
     State("input-x-max", "value"),
     State("input-y-min", "value"),
     State("input-y-max", "value"),
     State("input-nx", "value"),
     State("input-ny", "value"),
     State("input-nslices", "value"),
     State("input-fos-req", "value"),
     State("chk-compare", "value")],
    prevent_initial_call=True,
)
def run_analysis(n_clicks,
                 surface_data, layers_data, gwt_toggle, gwt_data,
                 surcharge, surcharge_start, surcharge_end, kh,
                 nail_toggle, nail_data,
                 nail_bar_dia, nail_ddh, nail_fy, nail_bond, nail_spacing,
                 method, mode,
                 xc, yc, radius,
                 x_min, x_max, y_min, y_max, nx, ny,
                 n_slices, fos_req, compare):
    if not n_clicks:
        raise PreventUpdate

    # Defaults for empty outputs
    empty_outputs = (
        _empty_figure(), build_fos_badge(None), html.Div(), html.Div(),
        {"display": "none"}, _empty_figure(""), html.Div(), "",
    )

    gwt_on = "on" in (gwt_toggle or [])
    nail_on = "on" in (nail_toggle or [])
    geom, err = _build_geometry(
        surface_data, layers_data, gwt_on, gwt_data,
        surcharge, surcharge_start, surcharge_end, kh,
        nails_enabled=nail_on, nails_data=nail_data,
        nail_bar_dia=nail_bar_dia, nail_ddh=nail_ddh,
        nail_fy=nail_fy, nail_bond=nail_bond, nail_spacing=nail_spacing,
    )
    if geom is None:
        return (*empty_outputs[:7], err)

    n_sl = int(_safe_float(n_slices, 30))
    fos_r = _safe_float(fos_req, 1.5)
    do_compare = "on" in (compare or [])

    try:
        if mode == "single":
            xc_v = _safe_float(xc, 20)
            yc_v = _safe_float(yc, 15)
            r_v = _safe_float(radius, 13)
            if r_v <= 0:
                return (*empty_outputs[:7], "Radius must be positive.")

            result = analyze_slope(
                geom, xc_v, yc_v, r_v,
                method=method,
                n_slices=n_sl,
                FOS_required=fos_r,
                include_slice_data=True,
                compare_methods=do_compare,
            )

            fig = build_cross_section(geom, result)
            heatmap_style = {"display": "none"}
            heatmap_fig = _empty_figure("")

        else:
            # Grid search
            xr = (_safe_float(x_min, 5), _safe_float(x_max, 35))
            yr = (_safe_float(y_min, 11), _safe_float(y_max, 25))
            nx_v = max(3, int(_safe_float(nx, 10)))
            ny_v = max(3, int(_safe_float(ny, 10)))

            search_res = search_critical_surface(
                geom,
                x_range=xr,
                y_range=yr,
                nx=nx_v,
                ny=ny_v,
                method=method,
                n_slices=n_sl,
                FOS_required=fos_r,
            )

            if search_res.critical is None:
                return (*empty_outputs[:7],
                        "No valid slip surfaces found. Try adjusting search bounds.")

            # Re-run on critical surface with slice data + comparison
            result = analyze_slope(
                geom,
                search_res.critical.xc,
                search_res.critical.yc,
                search_res.critical.radius,
                method=method,
                n_slices=n_sl,
                FOS_required=fos_r,
                include_slice_data=True,
                compare_methods=do_compare,
            )

            fig = build_cross_section(geom, result)
            heatmap_fig = build_search_heatmap(search_res)
            heatmap_style = {"display": "block"}

        # Build output components
        badge = build_fos_badge(result)
        summary = build_results_summary(result)
        comp_table = build_comparison_table(result) if do_compare else html.Div()
        slice_table = build_slice_table(result)

        return fig, badge, summary, comp_table, heatmap_style, heatmap_fig, slice_table, ""

    except ValueError as e:
        return (*empty_outputs[:7], f"Analysis error: {e}")
    except Exception as e:
        tb = traceback.format_exc()
        return (*empty_outputs[:7], f"Unexpected error: {e}\n{tb}")


# ---------------------------------------------------------------------------
# DXF Import Callbacks
# ---------------------------------------------------------------------------

# Callback A: Upload → Discovery
# Decodes base64, runs discover_layers, populates all dropdown options
@app.callback(
    [Output("dxf-store", "data"),
     Output("dxf-discovery-store", "data"),
     Output("dxf-mapping-section", "style"),
     Output("dxf-status", "children"),
     Output("dxf-status", "style"),
     Output("dxf-units-dropdown", "value"),
     Output("dxf-surface-dropdown", "options"),
     Output("dxf-gwt-dropdown", "options"),
     Output("dxf-nail-dropdown", "options"),
     *[Output(f"dxf-boundary-layer-{i}", "options") for i in range(5)]],
    Input("dxf-upload", "contents"),
    State("dxf-upload", "filename"),
    prevent_initial_call=True,
)
def dxf_upload_discovery(contents, filename):
    import base64

    # Number of boundary dropdown outputs
    n_boundary = 5
    error_style = {"marginTop": "6px", "fontSize": "0.82rem",
                   "color": "#dc2626", "minHeight": "18px"}
    ok_style = {"marginTop": "6px", "fontSize": "0.82rem",
                "color": "#16a34a", "minHeight": "18px"}
    hidden = {"display": "none", "marginTop": "8px"}
    empty_opts = []

    if contents is None:
        raise PreventUpdate

    # Check for DWG by filename
    if filename and filename.lower().endswith(".dwg"):
        msg = ("DWG files are not supported. Convert to DXF using "
               "ODA File Converter or save as DXF from AutoCAD.")
        return (None, None, hidden, msg, error_style, "m",
                empty_opts, empty_opts, empty_opts,
                *[empty_opts for _ in range(n_boundary)])

    # Decode base64 content
    try:
        content_type, content_string = contents.split(",", 1)
        raw_bytes = base64.b64decode(content_string)
    except Exception as e:
        msg = f"Failed to decode uploaded file: {e}"
        return (None, None, hidden, msg, error_style, "m",
                empty_opts, empty_opts, empty_opts,
                *[empty_opts for _ in range(n_boundary)])

    # Lazy import — ezdxf stays optional
    try:
        from dxf_import import discover_layers
    except ImportError:
        msg = ("ezdxf is required for DXF import. "
               "Install with: pip install ezdxf>=1.4")
        return (None, None, hidden, msg, error_style, "m",
                empty_opts, empty_opts, empty_opts,
                *[empty_opts for _ in range(n_boundary)])

    # Run discovery
    try:
        result = discover_layers(content=raw_bytes)
    except Exception as e:
        msg = f"DXF read error: {e}"
        return (None, None, hidden, msg, error_style, "m",
                empty_opts, empty_opts, empty_opts,
                *[empty_opts for _ in range(n_boundary)])

    # Build dropdown options from layer names
    layer_names = [lyr.name for lyr in result.layers]
    layer_options = [{"label": name, "value": name} for name in layer_names]
    optional_options = [{"label": "(none)", "value": ""}] + layer_options

    # Auto-detect units
    units_map = {"m": "m", "ft": "ft", "mm": "mm", "cm": "cm", "in": "in",
                 "meters": "m", "feet": "ft", "millimeters": "mm",
                 "centimeters": "cm", "inches": "in"}
    detected_units = "m"
    if result.units_hint:
        detected_units = units_map.get(result.units_hint.lower(), "m")

    fname = filename or "DXF file"
    msg = f"Loaded {fname}: {result.n_layers} layers, {result.n_total_entities} entities"
    show = {"display": "block", "marginTop": "8px"}

    # Store raw bytes as base64 string for reuse in import callback
    return (contents, result.to_dict(), show, msg, ok_style, detected_units,
            layer_options, optional_options, optional_options,
            *[layer_options for _ in range(n_boundary)])


# Callback B: Add Boundary Row
@app.callback(
    [Output(f"dxf-boundary-row-{i}", "style") for i in range(5)],
    Input("btn-add-boundary", "n_clicks"),
    [State(f"dxf-boundary-row-{i}", "style") for i in range(5)],
    prevent_initial_call=True,
)
def add_boundary_row(n_clicks, *row_styles):
    if not n_clicks:
        raise PreventUpdate
    styles = list(row_styles)
    # Find first hidden row and show it
    for i in range(5):
        if styles[i] and styles[i].get("display") == "none":
            styles[i] = {"display": "flex", "marginBottom": "4px"}
            break
    return styles


# Callback C: Import DXF → Populate Tables
@app.callback(
    [Output("surface-table", "data", allow_duplicate=True),
     Output("layers-table", "data", allow_duplicate=True),
     Output("gwt-toggle", "value", allow_duplicate=True),
     Output("gwt-table", "data", allow_duplicate=True),
     Output("nail-toggle", "value", allow_duplicate=True),
     Output("nail-table", "data", allow_duplicate=True),
     Output("dxf-status", "children", allow_duplicate=True),
     Output("dxf-status", "style", allow_duplicate=True)],
    Input("btn-import-dxf", "n_clicks"),
    [State("dxf-store", "data"),
     State("dxf-units-dropdown", "value"),
     State("dxf-flip-y", "value"),
     State("dxf-surface-dropdown", "value"),
     State("dxf-gwt-dropdown", "value"),
     State("dxf-nail-dropdown", "value"),
     *[State(f"dxf-boundary-layer-{i}", "value") for i in range(5)],
     *[State(f"dxf-boundary-name-{i}", "value") for i in range(5)]],
    prevent_initial_call=True,
)
def import_dxf(n_clicks, dxf_contents, units, flip_y_val,
               surface_layer, gwt_layer, nail_layer,
               *boundary_args):
    import base64

    error_style = {"marginTop": "6px", "fontSize": "0.82rem",
                   "color": "#dc2626", "minHeight": "18px"}
    ok_style = {"marginTop": "6px", "fontSize": "0.82rem",
                "color": "#16a34a", "minHeight": "18px"}
    no_update = [dash.no_update] * 6

    if not n_clicks or not dxf_contents:
        raise PreventUpdate

    if not surface_layer:
        return (*no_update, "Select a surface layer before importing.", error_style)

    # Split boundary_args into layers (first 5) and names (last 5)
    boundary_layers = list(boundary_args[:5])
    boundary_names = list(boundary_args[5:])

    # Build LayerMapping
    try:
        from dxf_import import LayerMapping, parse_dxf_geometry
    except ImportError:
        return (*no_update,
                "ezdxf is required. Install with: pip install ezdxf>=1.4",
                error_style)

    # Build soil_boundaries dict from paired dropdowns/inputs
    soil_boundaries = {}
    for i in range(5):
        lyr = boundary_layers[i]
        name = boundary_names[i]
        if lyr and name:
            soil_boundaries[lyr] = name.strip()

    mapping = LayerMapping(
        surface=surface_layer,
        soil_boundaries=soil_boundaries,
        water_table=gwt_layer if gwt_layer else None,
        nails=nail_layer if nail_layer else None,
    )

    flip_y = "on" in (flip_y_val or [])
    units_val = units or "m"

    # Decode base64 content
    try:
        _ct, content_string = dxf_contents.split(",", 1)
        raw_bytes = base64.b64decode(content_string)
    except Exception as e:
        return (*no_update, f"Failed to decode DXF data: {e}", error_style)

    # Parse geometry
    try:
        result = parse_dxf_geometry(
            content=raw_bytes,
            layer_mapping=mapping,
            units=units_val,
            flip_y=flip_y,
        )
    except Exception as e:
        return (*no_update, f"Parse error: {e}", error_style)

    # --- Populate surface table ---
    surface_data = [{"x": round(x, 3), "z": round(z, 3)}
                    for x, z in result.surface_points]

    # --- Populate layers table ---
    # Build layers from boundary profiles + surface (top layer goes from
    # surface down to first boundary, subsequent boundaries stack below)
    layers_data = []
    if result.boundary_profiles:
        # Surface z-range for the topmost layer
        z_vals = [z for _, z in result.surface_points]
        surface_max_z = max(z_vals) if z_vals else 10

        boundary_names_ordered = list(result.boundary_profiles.keys())
        for idx, soil_name in enumerate(boundary_names_ordered):
            bnd_pts = result.boundary_profiles[soil_name]
            bnd_zs = [z for _, z in bnd_pts]
            avg_z = sum(bnd_zs) / len(bnd_zs) if bnd_zs else 0

            if idx == 0:
                top_e = round(surface_max_z, 2)
            else:
                # Top is previous boundary's average
                prev_pts = result.boundary_profiles[
                    boundary_names_ordered[idx - 1]]
                prev_zs = [z for _, z in prev_pts]
                top_e = round(sum(prev_zs) / len(prev_zs), 2) if prev_zs else 0

            layers_data.append({
                "name": soil_name,
                "top_elev": top_e,
                "bot_elev": round(avg_z, 2),
                "gamma": 18, "gamma_sat": 20,
                "phi": 30, "c_prime": 0, "cu": 0,
                "mode": "drained",
            })
    else:
        # No boundaries — single layer from surface max to surface min - 5
        z_vals = [z for _, z in result.surface_points]
        z_max = max(z_vals) if z_vals else 10
        z_min = min(z_vals) if z_vals else 0
        layers_data.append({
            "name": "Layer 1",
            "top_elev": round(z_max, 2),
            "bot_elev": round(z_min - 5, 2),
            "gamma": 18, "gamma_sat": 20,
            "phi": 30, "c_prime": 0, "cu": 0,
            "mode": "drained",
        })

    # --- GWT ---
    gwt_toggle_val = []
    gwt_data = DEFAULT_GWT.copy()
    if result.gwt_points:
        gwt_toggle_val = ["on"]
        gwt_data = [{"x": round(x, 3), "z": round(z, 3)}
                    for x, z in result.gwt_points]

    # --- Nails ---
    nail_toggle_val = []
    nail_data = DEFAULT_NAILS.copy()
    if result.nail_lines:
        nail_toggle_val = ["on"]
        nail_data = []
        for nl in result.nail_lines:
            dx = nl["x_tip"] - nl["x_head"]
            dz = nl["z_tip"] - nl["z_head"]
            length = (dx**2 + dz**2) ** 0.5
            incl = math.degrees(math.atan2(abs(dz), abs(dx))) if length > 0 else 15
            nail_data.append({
                "x_head": round(nl["x_head"], 3),
                "z_head": round(nl["z_head"], 3),
                "length": round(length, 2),
                "incl": round(incl, 1),
            })

    # Build status message
    parts = [f"{len(surface_data)} surface pts"]
    if layers_data:
        parts.append(f"{len(layers_data)} layers")
    if result.gwt_points:
        parts.append("GWT")
    if result.nail_lines:
        parts.append(f"{len(result.nail_lines)} nails")
    warnings_text = ""
    if result.warnings:
        warnings_text = " | Warnings: " + "; ".join(result.warnings)
    msg = f"Imported: {', '.join(parts)}{warnings_text}"

    return (surface_data, layers_data, gwt_toggle_val, gwt_data,
            nail_toggle_val, nail_data, msg, ok_style)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  Slope Stability GUI running at: http://127.0.0.1:8051\n")
    app.run(debug=True, host="127.0.0.1", port=8051)

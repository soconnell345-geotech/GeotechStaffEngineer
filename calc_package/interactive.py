"""
Interactive single-file HTML viewers (plotly) for analysis results.

STRETCH feature of the calc-viz build: `save_interactive_report`
writes a self-contained HTML file (plotly.js embedded INLINE — no
network needed to view) from a result object:

- slope_stability.SlopeStabilityResult (+ SlopeGeometry, optional
  SearchResult): section with toggleable layers/GWT/reinforcement,
  trial surfaces colored by FOS (legend toggle), slice hover tooltips
  (W, alpha, N', S_mob, E/X), line-of-thrust toggle.
- fem2d.FEMResult: mesh/deformed toggle, contour dropdown
  (|u| / ux / uy / sigma_yy / tau_max), plastic-point toggle and the
  SRF curve in a side subplot for SRM results.

plotly is an OPTIONAL dependency (`pip install
geotech-staff-engineer[interactive]`); everything degrades to a clear
ImportError. matplotlib is NOT required here.
"""

import math
from pathlib import Path

import numpy as np

_LAYER_COLORS = ['#e8d9b0', '#cfa97c', '#a9c3a4', '#c9b8a6',
                 '#dfd0a2', '#b3c4cf', '#d2c3ab', '#bfd3bf']


def _get_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        raise ImportError(
            "plotly is required for interactive reports. "
            "Install with: pip install plotly "
            "(or geotech-staff-engineer[interactive])")


# ---------------------------------------------------------------------------
# Slope (LE) viewer
# ---------------------------------------------------------------------------

def _fos_color(fos, vmin, vmax):
    """Red (low FOS) -> yellow -> green (high FOS) hex color."""
    if vmax - vmin < 1e-9:
        t = 1.0
    else:
        t = min(max((fos - vmin) / (vmax - vmin), 0.0), 1.0)
    if t < 0.5:
        r, g = 204, int(40 + (204 - 40) * (t * 2))
    else:
        r, g = int(204 - (204 - 30) * ((t - 0.5) * 2)), 204
    return f'rgb({r},{g},40)'


def slope_interactive_figure(result, geom, search=None):
    """Build the interactive LE figure (plotly Figure).

    Parameters
    ----------
    result : SlopeStabilityResult
    geom : SlopeGeometry
    search : SearchResult, optional — adds toggleable trial surfaces
        colored by FOS.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _get_plotly()
    fig = go.Figure()

    x_min, x_max = geom.x_range
    xs = np.linspace(x_min, x_max, 300)
    zg = np.array([geom.ground_elevation_at(x) for x in xs])

    # Soil layers as filled polygons
    for i, layer in enumerate(geom.soil_layers):
        top = np.minimum(zg, np.array([layer.top_at(x) for x in xs]))
        bot = np.array([layer.bottom_at(x) for x in xs])
        mask = top > bot
        if not np.any(mask):
            continue
        xv = xs[mask]
        poly_x = np.concatenate([xv, xv[::-1]])
        poly_z = np.concatenate([top[mask], bot[mask][::-1]])
        fig.add_trace(go.Scatter(
            x=poly_x, y=poly_z, fill='toself', mode='none',
            fillcolor=_LAYER_COLORS[i % len(_LAYER_COLORS)],
            opacity=0.65, name=layer.name, legendgroup='layers',
            hoverinfo='name',
        ))

    # Ground surface
    sx = [p[0] for p in geom.surface_points]
    sz = [p[1] for p in geom.surface_points]
    fig.add_trace(go.Scatter(
        x=sx, y=sz, mode='lines', name='Ground surface',
        line=dict(color='black', width=2.5), hoverinfo='x+y'))

    # GWT
    if geom.gwt_points:
        gx = [p[0] for p in geom.gwt_points]
        gz = [p[1] for p in geom.gwt_points]
        fig.add_trace(go.Scatter(
            x=gx, y=gz, mode='lines', name='GWT',
            line=dict(color='#1f6fbf', width=1.8, dash='dash')))

    # Trial surfaces (toggleable as a group)
    if search is not None:
        trials = []
        from slope_stability.slip_surface import CircularSlipSurface
        for g in search.grid_fos:
            fos = g.get('FOS', 999.9)
            R = g.get('R', 0.0)
            if fos >= 900 or not R or R <= 0:
                continue
            try:
                slip = CircularSlipSurface(g['xc'], g['yc'], R)
                x_en, x_ex = slip.find_entry_exit(geom)
            except ValueError:
                continue
            txs = np.linspace(x_en, x_ex, 60)
            dx = np.clip(txs - g['xc'], -R, R)
            tzs = g['yc'] - np.sqrt(np.maximum(R**2 - dx**2, 0.0))
            trials.append((fos, txs, tzs))
        for t in getattr(search, 'trial_surfaces', None) or []:
            fos = t.get('FOS', 999.9)
            if fos >= 900 or not t.get('points'):
                continue
            pts = np.asarray(t['points'], dtype=float)
            trials.append((fos, pts[:, 0], pts[:, 1]))

        if trials:
            fvals = [t[0] for t in trials]
            vmin = min(fvals)
            vmax = float(np.percentile(fvals, 90))
            if len(trials) > 150:
                trials = sorted(trials, key=lambda t: t[0])
                idx = np.linspace(0, len(trials) - 1, 150).astype(int)
                trials = [trials[i] for i in idx]
            for k, (fos, txs, tzs) in enumerate(
                    sorted(trials, key=lambda t: -t[0])):
                fig.add_trace(go.Scatter(
                    x=txs, y=tzs, mode='lines',
                    line=dict(color=_fos_color(fos, vmin, vmax),
                              width=1),
                    opacity=0.6,
                    name=f'Trial surfaces ({len(trials)})',
                    legendgroup='trials',
                    showlegend=(k == 0),
                    visible='legendonly',
                    hovertemplate=f'FOS = {fos:.3f}<extra></extra>',
                ))

    # Analyzed / critical slip surface
    if result.slip_points:
        pts = np.asarray(result.slip_points, dtype=float)
        cx, cz = pts[:, 0], pts[:, 1]
    else:
        cx = np.linspace(result.x_entry, result.x_exit, 150)
        dx = np.clip(cx - result.xc, -result.radius, result.radius)
        cz = result.yc - np.sqrt(
            np.maximum(result.radius**2 - dx**2, 0.0))
    fig.add_trace(go.Scatter(
        x=cx, y=cz, mode='lines',
        name=f'Slip surface (FOS = {result.FOS:.3f})',
        line=dict(color='#cc0000', width=3.5),
        hovertemplate='x = %{x:.2f} m<br>z = %{y:.2f} m'
                      '<extra>Slip surface</extra>'))

    # Slices with rich hover
    if result.slice_data:
        sd = result.slice_data
        hover = []
        for i, s in enumerate(sd):
            txt = (f"Slice {i + 1}<br>W = {s.weight:.1f} kN/m"
                   f"<br>α = {s.alpha_deg:.1f}°"
                   f"<br>N′ = {s.N_eff_kN:.1f} kN/m"
                   f"<br>S_mob = {s.S_mob_kN:.1f} kN/m"
                   f"<br>u·l = {s.U_base_kN:.1f} kN/m")
            if s.E_left_kN is not None:
                txt += (f"<br>E = {s.E_left_kN:.1f} / "
                        f"{s.E_right_kN:.1f} kN/m"
                        f"<br>X = {s.X_left_kN:.1f} / "
                        f"{s.X_right_kN:.1f} kN/m")
            hover.append(txt)
        fig.add_trace(go.Scatter(
            x=[s.x_mid for s in sd], y=[s.z_base for s in sd],
            mode='markers', name=f'Slices ({len(sd)})',
            marker=dict(size=6, color='#7c3aed', symbol='circle',
                        line=dict(color='white', width=0.5)),
            text=hover, hoverinfo='text'))

    # Line of thrust
    if result.thrust_line:
        tx = [p[0] for p in result.thrust_line]
        tz = [p[1] for p in result.thrust_line]
        fig.add_trace(go.Scatter(
            x=tx, y=tz, mode='lines', name='Line of thrust',
            line=dict(color='#9333ea', width=2, dash='dashdot')))

    # Reinforcement
    shown = set()
    for kind, items, color in (
            ('Soil nails', geom.nails or [], '#444444'),
            ('Anchors', geom.anchors or [], '#0e7490')):
        for el in items:
            dxn = math.cos(math.radians(el.inclination))
            dzn = -math.sin(math.radians(el.inclination))
            sgn = 1.0 if geom.ground_elevation_at(el.x_head + 1.0) >= \
                geom.ground_elevation_at(el.x_head - 1.0) else -1.0
            fig.add_trace(go.Scatter(
                x=[el.x_head, el.x_head + sgn * dxn * el.length],
                y=[el.z_head, el.z_head + dzn * el.length],
                mode='lines', name=kind, legendgroup=kind,
                showlegend=kind not in shown,
                line=dict(color=color, width=2.5)))
            shown.add(kind)
    for g in (geom.geosynthetics or []):
        x0 = g.x_start if g.x_start is not None else x_min
        x1 = g.x_end if g.x_end is not None else x_max
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[g.elevation, g.elevation], mode='lines',
            name='Geosynthetics', legendgroup='Geosynthetics',
            showlegend='Geosynthetics' not in shown,
            line=dict(color='#15803d', width=2.5)))
        shown.add('Geosynthetics')

    fig.update_layout(
        title=dict(
            text=f"Slope Stability — {result.method}, "
                 f"FOS = {result.FOS:.3f}",
            font=dict(size=16)),
        xaxis=dict(title='Distance (m)'),
        yaxis=dict(title='Elevation (m)', scaleanchor='x',
                   scaleratio=1),
        legend=dict(font=dict(size=10), groupclick='togglegroup'),
        template='plotly_white',
        hovermode='closest',
    )
    return fig


# ---------------------------------------------------------------------------
# FEM viewer
# ---------------------------------------------------------------------------

def _grid_field(nodes, corner, values, nx=160, ny=110):
    """Interpolate nodal values onto a regular grid, NaN outside the
    mesh (vectorized barycentric point-in-triangle, no matplotlib)."""
    nodes = np.asarray(nodes, dtype=float)
    gx = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), nx)
    gy = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), ny)
    GX, GY = np.meshgrid(gx, gy)
    P = np.column_stack([GX.ravel(), GY.ravel()])
    Z = np.full(len(P), np.nan)

    tri_xy = nodes[corner]                       # (n_e, 3, 2)
    vals = np.asarray(values, dtype=float)[corner]  # (n_e, 3)

    for e in range(len(corner)):
        (x1, y1), (x2, y2), (x3, y3) = tri_xy[e]
        xmin, xmax = min(x1, x2, x3), max(x1, x2, x3)
        ymin, ymax = min(y1, y2, y3), max(y1, y2, y3)
        cand = np.where(
            (P[:, 0] >= xmin) & (P[:, 0] <= xmax) &
            (P[:, 1] >= ymin) & (P[:, 1] <= ymax) &
            np.isnan(Z))[0]
        if len(cand) == 0:
            continue
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(det) < 1e-14:
            continue
        px, py = P[cand, 0], P[cand, 1]
        l1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / det
        l2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / det
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        idx = cand[inside]
        Z[idx] = (l1[inside] * vals[e, 0] + l2[inside] * vals[e, 1]
                  + l3[inside] * vals[e, 2])
    return gx, gy, Z.reshape(ny, nx)


def _mesh_edge_trace(go, nodes, corner, name, color, width=0.7,
                     visible=True):
    """All element edges as one Scatter trace (None-separated)."""
    edges = set()
    for tri in corner:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]),
                     (tri[2], tri[0])):
            edges.add((min(a, b), max(a, b)))
    xs, ys = [], []
    for a, b in edges:
        xs += [nodes[a, 0], nodes[b, 0], None]
        ys += [nodes[a, 1], nodes[b, 1], None]
    return go.Scatter(x=xs, y=ys, mode='lines', name=name,
                      line=dict(color=color, width=width),
                      visible=visible, hoverinfo='skip')


def fem_interactive_figure(result):
    """Build the interactive FEM figure (plotly Figure).

    Mesh/deformed toggle (legend), contour dropdown (|u|, ux, uy,
    sigma_yy, tau_max), plastic-point toggle, and — for SRM results —
    the SRF curve in a side subplot.

    Parameters
    ----------
    result : FEMResult (needs nodes, elements, displacements)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _get_plotly()
    from plotly.subplots import make_subplots

    nodes = np.asarray(result.nodes, dtype=float)
    corner = np.asarray(result.elements, dtype=int)[:, :3]
    n_nodes = len(nodes)
    u = np.asarray(result.displacements, dtype=float)[:2 * n_nodes]
    ux, uy = u[0::2], u[1::2]
    umag = np.sqrt(ux**2 + uy**2)

    # nodal stress fields
    fields = {'|u| (m)': umag, 'u_x (m)': ux, 'u_y (m)': uy}
    if result.stresses is not None:
        sig = np.asarray(result.stresses, dtype=float)
        sig_sum = np.zeros((n_nodes, 3))
        count = np.zeros(n_nodes)
        for e in range(len(corner)):
            for nd in corner[e]:
                sig_sum[nd] += sig[e]
                count[nd] += 1
        count[count == 0] = 1
        sig_nodal = sig_sum / count[:, None]
        fields['sigma_yy (kPa)'] = sig_nodal[:, 1]
        fields['tau_max (kPa)'] = np.sqrt(
            (0.5 * (sig_nodal[:, 0] - sig_nodal[:, 1]))**2
            + sig_nodal[:, 2]**2)

    curve = getattr(result, 'srf_curve', None)
    has_srf = curve is not None and len(curve[0]) > 0

    if has_srf:
        fig = make_subplots(
            rows=1, cols=2, column_widths=[0.72, 0.28],
            horizontal_spacing=0.08,
            subplot_titles=("Model", "SRF vs displacement"))
    else:
        fig = make_subplots(rows=1, cols=1,
                            subplot_titles=("Model",))

    # Contour traces (one per field, first visible)
    field_names = list(fields)
    n_contours = len(field_names)
    for i, fname in enumerate(field_names):
        gx, gy, Z = _grid_field(nodes, corner, fields[fname])
        fig.add_trace(go.Heatmap(
            x=gx, y=gy, z=Z, colorscale='Viridis',
            colorbar=dict(title=dict(text=fname, font=dict(size=10)),
                          x=0.66 if has_srf else 1.0, len=0.8),
            visible=(i == 0), name=fname,
            hovertemplate='x=%{x:.1f}, y=%{y:.1f}<br>'
                          + fname + ' = %{z:.4g}<extra></extra>',
        ), row=1, col=1)

    # Mesh / deformed mesh toggles
    fig.add_trace(_mesh_edge_trace(
        go, nodes, corner, 'Mesh', 'rgba(60,60,60,0.45)',
        visible='legendonly'), row=1, col=1)

    diag = math.hypot(float(np.ptp(nodes[:, 0])),
                      float(np.ptp(nodes[:, 1])))
    scale = float(f"{0.05 * diag / max(float(umag.max()), 1e-12):.1g}")
    deformed = nodes + scale * np.column_stack([ux, uy])
    fig.add_trace(_mesh_edge_trace(
        go, deformed, corner, f'Deformed mesh (x{scale:g})',
        'rgba(29,78,216,0.6)', visible='legendonly'), row=1, col=1)

    # Plastic points toggle
    pp = getattr(result, 'plastic_points', None)
    if pp and pp.get('points'):
        pts = np.asarray(pp['points'], dtype=float)
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode='markers',
            name=f"Plastic points ({pp['n_plastic']})",
            marker=dict(size=3, color='#dc2626', symbol='square'),
            visible='legendonly',
            hovertemplate='plastic GP<extra></extra>'), row=1, col=1)

    # SRF curve subplot
    if has_srf:
        srf, dd = np.asarray(curve[0]), np.asarray(curve[1])
        order = np.argsort(srf)
        fig.add_trace(go.Scatter(
            x=srf[order], y=dd[order], mode='lines+markers',
            name='SRF trials (converged)',
            line=dict(color='#2563eb', width=2),
            hovertemplate='SRF = %{x:.3f}<br>'
                          'E·δ/(γH²) = %{y:.3g}<extra></extra>'),
            row=1, col=2)
        history = getattr(result, 'srf_history', None) or []
        failed = [(h['srf'], h['dimensionless_disp'])
                  for h in history if h.get('failed')]
        if failed:
            fx, fy = zip(*failed)
            fig.add_trace(go.Scatter(
                x=fx, y=fy, mode='markers', name='Failed trials',
                marker=dict(symbol='x', size=9, color='#dc2626')),
                row=1, col=2)
        if result.FOS is not None:
            fig.add_vline(x=result.FOS, line_dash='dash',
                          line_color='#16a34a', row=1, col=2,
                          annotation_text=f'FOS = {result.FOS:.3f}',
                          annotation_font_size=10)
        fig.update_xaxes(title_text='SRF', row=1, col=2)
        fig.update_yaxes(title_text='E·δ/(γH²)', row=1, col=2)

    # Contour dropdown
    n_extra = len(fig.data) - n_contours
    buttons = []
    for i, fname in enumerate(field_names):
        vis = [j == i for j in range(n_contours)] + [None] * n_extra
        buttons.append(dict(
            label=fname, method='restyle',
            args=[{'visible': vis},
                  list(range(n_contours))]))
    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons, direction='down', x=0.0, y=1.15,
            xanchor='left', yanchor='top', font=dict(size=11))],
        title=dict(
            text=("2D FEM Results — "
                  + (f"SRM, FOS = {result.FOS:.3f}"
                     if result.FOS is not None
                     else result.analysis_type)),
            x=0.5, font=dict(size=16)),
        legend=dict(font=dict(size=10), orientation='h',
                    yanchor='bottom', y=-0.18),
        template='plotly_white',
    )
    fig.update_xaxes(title_text='x (m)', row=1, col=1)
    fig.update_yaxes(title_text='y (m)', scaleanchor='x',
                     scaleratio=1, row=1, col=1)
    return fig


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_interactive_report(result, path, geom=None, search=None,
                            title=None, include_plotlyjs=True):
    """Write a self-contained interactive HTML viewer for a result.

    Parameters
    ----------
    result : SlopeStabilityResult or FEMResult
        Dispatch is by result type. Slope results REQUIRE ``geom``.
    path : str — output .html path (parents created).
    geom : SlopeGeometry — required for slope results.
    search : SearchResult, optional — slope trial surfaces.
    title : str, optional — HTML page title.
    include_plotlyjs : bool or str — True (default) embeds plotly.js
        INLINE (~3 MB, viewable offline); pass 'cdn' for a small file
        that needs network access.

    Returns
    -------
    str — absolute path of the written file.
    """
    type_name = type(result).__name__
    if type_name == "SlopeStabilityResult":
        if geom is None:
            raise ValueError("geom (SlopeGeometry) is required for "
                             "slope results")
        fig = slope_interactive_figure(result, geom, search=search)
        default_title = "Slope Stability — Interactive Viewer"
    elif type_name == "FEMResult":
        fig = fem_interactive_figure(result)
        default_title = "2D FEM — Interactive Viewer"
    else:
        raise TypeError(
            f"Unsupported result type '{type_name}' — expected "
            f"SlopeStabilityResult or FEMResult")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = fig.to_html(full_html=True,
                       include_plotlyjs=include_plotlyjs,
                       config={'displaylogo': False})
    if title or default_title:
        html = html.replace(
            "<head>",
            f"<head><title>{title or default_title}</title>", 1)
    out.write_text(html, encoding='utf-8')
    return str(out.resolve())

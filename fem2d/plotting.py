"""
PLAXIS-style report figures for fem2d results.

All functions return a matplotlib Figure and import matplotlib lazily
(matplotlib is an optional dependency — project pattern:
geotech_common.plotting.get_pyplot).

Functions
---------
plot_mesh               — mesh w/ material coloring + BC symbols
plot_deformed_mesh      — scaled deformed mesh over undeformed outline
plot_contour            — nodal contour of |u| / ux / uy / sigma_xx /
                          sigma_yy / tau_xy / tau_max
plot_plastic_points     — plastic Gauss-point map (SRM failure state)
plot_srf_curve          — SRF vs dimensionless displacement, FOS mark
plot_failure_mechanism  — |u| contour at SRM failure state
plot_seepage            — head contours + Darcy flow vectors

T6 meshes are contoured on their 3-node corner skeleton (quadratic
midside values are not needed for report-quality fills). Everything
RENDERS stored result arrays; derived display fields are limited to
standard visual transforms (|u|, tau_max Mohr radius).
"""

import math

import numpy as np

_MAT_COLORS = ['#e8d9b0', '#cfa97c', '#a9c3a4', '#c9b8a6',
               '#dfd0a2', '#b3c4cf', '#d2c3ab', '#bfd3bf']


def _get_plt():
    from geotech_common.plotting import get_pyplot
    return get_pyplot()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _corner_elements(elements):
    """3-node corner connectivity for CST or T6 meshes."""
    elements = np.asarray(elements, dtype=int)
    return elements[:, :3]


def _triangulation(nodes, elements):
    import matplotlib.tri as mtri
    nodes = np.asarray(nodes, dtype=float)
    return mtri.Triangulation(nodes[:, 0], nodes[:, 1],
                              _corner_elements(elements))


def _boundary_edges(elements):
    """Edges used by exactly one corner triangle (the mesh outline)."""
    corner = _corner_elements(elements)
    count = {}
    for tri in corner:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]),
                     (tri[2], tri[0])):
            key = (min(a, b), max(a, b))
            count[key] = count.get(key, 0) + 1
    return [e for e, c in count.items() if c == 1]


def _draw_outline(ax, nodes, elements, color='#888888', lw=1.0,
                  label=None):
    nodes = np.asarray(nodes, dtype=float)
    first = True
    for a, b in _boundary_edges(elements):
        ax.plot([nodes[a, 0], nodes[b, 0]], [nodes[a, 1], nodes[b, 1]],
                '-', color=color, linewidth=lw,
                label=label if first else None, zorder=3)
        first = False


def _style_axes(ax, title, xlabel='x (m)', ylabel='y (m)'):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=9)


def _nodal_field(result, field):
    """Nodal scalar field + label/unit for contouring.

    Displacement fields come straight from result.displacements;
    stress fields are element values averaged to nodes
    (fem2d.assembly.nodal_stresses).
    """
    nodes = np.asarray(result.nodes, dtype=float)
    n_nodes = len(nodes)
    u = np.asarray(result.displacements, dtype=float)[:2 * n_nodes]
    ux = u[0::2]
    uy = u[1::2]

    if field == 'u_mag':
        return np.sqrt(ux**2 + uy**2), '|u| (m)'
    if field == 'ux':
        return ux, 'u_x (m)'
    if field == 'uy':
        return uy, 'u_y (m)'

    if result.stresses is None:
        raise ValueError(f"Result carries no stresses for field "
                         f"'{field}'")
    from fem2d.assembly import nodal_stresses
    sig_nodal = nodal_stresses(nodes, _corner_elements(result.elements),
                               np.asarray(result.stresses, dtype=float))
    if field == 'sigma_xx':
        return sig_nodal[:, 0], 'sigma_xx (kPa)'
    if field == 'sigma_yy':
        return sig_nodal[:, 1], 'sigma_yy (kPa)'
    if field == 'tau_xy':
        return sig_nodal[:, 2], 'tau_xy (kPa)'
    if field == 'tau_max':
        tmax = np.sqrt((0.5 * (sig_nodal[:, 0] - sig_nodal[:, 1]))**2
                       + sig_nodal[:, 2]**2)
        return tmax, 'tau_max (kPa)'
    raise ValueError(
        f"Unknown field '{field}'. Choose from: u_mag, ux, uy, "
        f"sigma_xx, sigma_yy, tau_xy, tau_max")


CONTOUR_FIELDS = ('u_mag', 'ux', 'uy', 'sigma_xx', 'sigma_yy',
                  'tau_xy', 'tau_max')


# ---------------------------------------------------------------------------
# 1. Mesh plot
# ---------------------------------------------------------------------------

def plot_mesh(nodes, elements, material_ids=None, material_names=None,
              bc_nodes=None, title='Finite Element Mesh',
              figsize=(10, 6.5)):
    """Mesh plot with material coloring, BC symbols and mesh stats.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elem, 3 or 6) array — CST or T6 connectivity.
    material_ids : (n_elem,) int array, optional — material/layer id
        per element (colors the mesh).
    material_names : list of str, optional — legend names per id.
    bc_nodes : dict, optional — from fem2d.detect_boundary_nodes():
        fixed_base drawn as filled triangles, rollers as open circles.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    corner = _corner_elements(elements)

    fig, ax = plt.subplots(figsize=figsize)

    if material_ids is not None:
        material_ids = np.asarray(material_ids, dtype=int)
        verts = nodes[corner]
        colors = [_MAT_COLORS[m % len(_MAT_COLORS)]
                  for m in material_ids]
        pc = PolyCollection(verts, facecolors=colors,
                            edgecolors='#666666', linewidths=0.3,
                            alpha=0.85)
        ax.add_collection(pc)
        handles = []
        for m in sorted(set(material_ids.tolist())):
            name = (material_names[m] if material_names is not None
                    and m < len(material_names) else f'Material {m + 1}')
            handles.append(Patch(
                facecolor=_MAT_COLORS[m % len(_MAT_COLORS)],
                edgecolor='#666666', label=name))
        ax.legend(handles=handles, loc='lower right', fontsize=8,
                  framealpha=0.9)
        ax.autoscale_view()
    else:
        tri = _triangulation(nodes, elements)
        ax.triplot(tri, color='#666666', linewidth=0.35)

    # BC symbols
    if bc_nodes is not None:
        base = bc_nodes.get('fixed_base', [])
        if len(base):
            ax.plot(nodes[base, 0], nodes[base, 1], marker='^',
                    linestyle='none', markersize=5, color='#1d4ed8',
                    markerfacecolor='#1d4ed8', zorder=6,
                    label='Fixed (u = v = 0)')
        rollers = list(bc_nodes.get('roller_left', [])) + \
            list(bc_nodes.get('roller_right', []))
        if rollers:
            ax.plot(nodes[rollers, 0], nodes[rollers, 1], marker='o',
                    linestyle='none', markersize=4, color='#1d4ed8',
                    markerfacecolor='white', zorder=6,
                    label='Roller (u = 0)')
        leg1 = ax.get_legend()
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        if leg1 is not None and material_ids is not None:
            ax.add_artist(leg1)

    elem_type = 'T6' if elements.shape[1] == 6 else (
        'Q4' if elements.shape[1] == 4 else 'CST')
    ax.text(0.02, 0.98,
            f"{len(nodes)} nodes, {len(elements)} {elem_type} elements",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.9, edgecolor='#999999'), zorder=10)

    _style_axes(ax, title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Deformed mesh
# ---------------------------------------------------------------------------

def plot_deformed_mesh(result, scale=None, title='Deformed Mesh',
                       figsize=(10, 6.5)):
    """Scaled deformed mesh over the undeformed outline.

    Parameters
    ----------
    result : FEMResult (needs nodes, elements, displacements)
    scale : float, optional — displacement magnification. Default:
        5% of the model diagonal over the max displacement, rounded
        to one significant figure (annotated on the figure).

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    nodes = np.asarray(result.nodes, dtype=float)
    n_nodes = len(nodes)
    u = np.asarray(result.displacements, dtype=float)[:2 * n_nodes]
    ux = u[0::2]
    uy = u[1::2]
    umax = float(np.sqrt(ux**2 + uy**2).max())

    if scale is None:
        diag = math.hypot(float(np.ptp(nodes[:, 0])), float(np.ptp(nodes[:, 1])))
        raw = 0.05 * diag / max(umax, 1e-12)
        scale = float(f"{raw:.1g}") if raw > 0 else 1.0

    deformed = nodes + scale * np.column_stack([ux, uy])

    fig, ax = plt.subplots(figsize=figsize)
    _draw_outline(ax, nodes, result.elements, color='#999999', lw=1.0,
                  label='Undeformed outline')
    tri = _triangulation(deformed, result.elements)
    ax.triplot(tri, color='#1d4ed8', linewidth=0.35)
    ax.plot([], [], '-', color='#1d4ed8', linewidth=1.0,
            label=f'Deformed mesh (x{scale:g})')

    ax.text(0.02, 0.98,
            f"u_max = {umax:.4g} m\ndisplacement scale x{scale:g}",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.9, edgecolor='#999999'), zorder=10)

    _style_axes(ax, title)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Contours
# ---------------------------------------------------------------------------

def plot_contour(result, field='u_mag', n_levels=12, cmap=None,
                 show_mesh=False, title=None, figsize=(10, 6.5)):
    """Filled nodal contour of a displacement or stress field.

    Parameters
    ----------
    result : FEMResult
    field : str — one of u_mag, ux, uy, sigma_xx, sigma_yy, tau_xy,
        tau_max.
    n_levels : int — contour levels.
    cmap : str, optional — default 'viridis' for displacements,
        'RdYlBu' for stresses.
    show_mesh : bool — overlay element edges.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    values, label = _nodal_field(result, field)
    if cmap is None:
        cmap = 'viridis' if field in ('u_mag', 'ux', 'uy') else 'RdYlBu'

    tri = _triangulation(result.nodes, result.elements)

    fig, ax = plt.subplots(figsize=figsize)
    if float(np.ptp(values)) < 1e-15:
        values = values + np.linspace(0, 1e-12, len(values))
    cf = ax.tricontourf(tri, values, levels=n_levels, cmap=cmap)
    if show_mesh:
        ax.triplot(tri, color='#444444', linewidth=0.2, alpha=0.5)
    _draw_outline(ax, result.nodes, result.elements, color='#555555',
                  lw=0.8)

    cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if title is None:
        title = f'Contour: {label}'
    _style_axes(ax, title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Plastic Gauss-point map
# ---------------------------------------------------------------------------

def plot_plastic_points(result, title='Plastic Points at Failure State',
                        figsize=(10, 6.5)):
    """PLAXIS-style plastic-point map (SRM results).

    Requires result.plastic_points (stored by analyze_slope_srm —
    Gauss points on the reduced-strength MC yield surface at the last
    stable SRF).

    Returns
    -------
    matplotlib.figure.Figure
    """
    pp = getattr(result, 'plastic_points', None)
    if not pp or not pp.get('points'):
        raise ValueError("Result carries no plastic_points "
                         "(SRM analysis with the modern engine "
                         "required)")
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=figsize)
    tri = _triangulation(result.nodes, result.elements)
    ax.triplot(tri, color='#bbbbbb', linewidth=0.25, zorder=2)
    _draw_outline(ax, result.nodes, result.elements, color='#555555',
                  lw=0.9)

    pts = np.asarray(pp['points'], dtype=float)
    ax.plot(pts[:, 0], pts[:, 1], 's', color='#dc2626', markersize=2.2,
            linestyle='none', zorder=5,
            label='Mohr-Coulomb plastic point')

    frac = pp['n_plastic'] / max(pp['n_gp_total'], 1)
    ax.text(0.02, 0.98,
            f"{pp['n_plastic']} of {pp['n_gp_total']} Gauss points "
            f"plastic ({frac:.0%})\nat SRF = {pp['srf']:.3f} "
            f"(last stable state)",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.9, edgecolor='#999999'), zorder=10)

    _style_axes(ax, title)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. SRF curve
# ---------------------------------------------------------------------------

def plot_srf_curve(result, title='Strength Reduction — SRF vs '
                                 'Displacement', figsize=(8.5, 5.5)):
    """SRF vs dimensionless displacement with the FOS annotated.

    Requires result.srf_curve / srf_history (analyze_slope_srm).

    Returns
    -------
    matplotlib.figure.Figure
    """
    curve = getattr(result, 'srf_curve', None)
    history = getattr(result, 'srf_history', None)
    if curve is None or len(curve[0]) == 0:
        raise ValueError("Result carries no srf_curve (SRM analysis "
                         "required)")
    plt = _get_plt()

    srf, dim_disp = np.asarray(curve[0]), np.asarray(curve[1])
    order = np.argsort(srf)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(srf[order], dim_disp[order], 'o-', color='#2563eb',
            markersize=5, linewidth=1.6,
            label='Converged trials')

    if history:
        failed = [(h['srf'], h['dimensionless_disp']) for h in history
                  if h.get('failed')]
        if failed:
            fx, fy = zip(*failed)
            ax.plot(fx, fy, 'x', color='#dc2626', markersize=8,
                    markeredgewidth=2, linestyle='none',
                    label='Failed trials (no equilibrium)')

    if result.FOS is not None:
        ax.axvline(result.FOS, color='#16a34a', linestyle='--',
                   linewidth=1.6, label=f'FOS = {result.FOS:.3f}')
        basis = getattr(result, 'fos_basis', None)
        if basis:
            ax.text(result.FOS, ax.get_ylim()[1] * 0.55,
                    f'  failure basis: {basis}', rotation=90,
                    fontsize=8, va='center', color='#16a34a')

    ax.set_xlabel('Strength Reduction Factor (SRF)', fontsize=10)
    ax.set_ylabel('Dimensionless displacement  '
                  'E·δ_max / (γ·H²)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Failure mechanism
# ---------------------------------------------------------------------------

def plot_failure_mechanism(result, show_plastic_points=True,
                           figsize=(10, 6.5)):
    """|u| contour at the SRM failure (last stable) state.

    The displacement pattern at the bracketed SRF localizes along the
    critical mechanism — the FEM analogue of the LE critical surface.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fos_txt = (f" (FOS = {result.FOS:.3f})"
               if result.FOS is not None else "")
    fig = plot_contour(
        result, field='u_mag',
        title=f'Failure Mechanism — |u| at last stable SRF{fos_txt}',
        cmap='inferno', figsize=figsize)
    ax = fig.axes[0]
    pp = getattr(result, 'plastic_points', None)
    if show_plastic_points and pp and pp.get('points'):
        pts = np.asarray(pp['points'], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], 's', color='#ffffff',
                markersize=1.6, alpha=0.6, linestyle='none',
                label='Plastic points', zorder=6)
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    return fig


# ---------------------------------------------------------------------------
# 6b. Local factor-of-safety heatmap
# ---------------------------------------------------------------------------

def plot_local_fos(field_or_result, c=None, phi=None, *, cap=None,
                   vmax=None, n_levels=14,
                   title='Local Factor of Safety (mobilized strength)',
                   figsize=(10, 6.5)):
    """Filled contour of the pointwise local FOS over the mesh.

    Same look as ``plot_contour`` — low FOS is red (critical), high is green
    (safe), on the RdYlGn scale, clipped for display at ``vmax``.

    Parameters
    ----------
    field_or_result : LocalFOSField or FEMResult
        Either a precomputed ``LocalFOSField`` (from ``local_fos_field`` or a
        result's ``.local_fos``), or a FEMResult plus ``c`` and ``phi`` to
        compute one on the fly.
    c, phi : float or array, optional
        Original c' (kPa) / phi' (deg), required only when passing a raw result.
    cap : float, optional
        Local-FOS cap when computing from a result (default 10).
    vmax : float, optional
        Upper limit of the color scale (display only). Default: 3x the global
        FOS if known, else ~2x the minimum (keeps the critical band legible).
    n_levels : int — contour levels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from fem2d.local_fos import LocalFOSField, local_fos_field
    if isinstance(field_or_result, LocalFOSField):
        lf = field_or_result
    else:
        lf = getattr(field_or_result, 'local_fos', None)
        if lf is None:
            if c is None or phi is None:
                raise ValueError(
                    "Pass a LocalFOSField, a result with .local_fos, or a "
                    "result together with c and phi.")
            lf = local_fos_field(field_or_result, c, phi,
                                 cap=cap if cap is not None else 10.0)

    plt = _get_plt()
    tri = _triangulation(lf.nodes, lf.elements)
    vals = np.asarray(lf.nodal_values, dtype=float).copy()

    if vmax is None:
        # keep the near-critical (mobilized) band legible: scale a little above
        # the global FOS, or ~2x the minimum, whichever gives useful contrast.
        if lf.global_fos is not None:
            vmax = 2.2 * lf.global_fos
        else:
            vmax = max(2.0 * lf.min_fos, lf.min_fos + 1.0)
    vmax = float(min(vmax, lf.cap))
    vmin = float(max(0.0, min(1.0, lf.min_fos)))
    vals = np.clip(vals, vmin, vmax)

    levels = np.linspace(vmin, vmax, n_levels + 1)
    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.tricontourf(tri, vals, levels=levels, cmap='RdYlGn', extend='both')
    # highlight the local_FOS = 1 (incipient yield) contour if in range
    if vmin < 1.0 < vmax:
        ax.tricontour(tri, vals, levels=[1.0], colors='#111111',
                      linewidths=1.4, linestyles='--')
    _draw_outline(ax, lf.nodes, lf.elements, color='#555555', lw=0.8)
    ax.plot(lf.min_location[0], lf.min_location[1], marker='v',
            markersize=9, color='#111111', markerfacecolor='#dc2626',
            zorder=6, label=f'min local FOS = {lf.min_fos:.2f}')

    cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('local FOS = tau_available / tau_mobilized', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    txt = (f"min local FOS = {lf.min_fos:.3f}\n"
           f"median = {lf.median_fos:.2f}")
    if lf.global_fos is not None:
        txt += (f"\nglobal SRM FOS = {lf.global_fos:.3f}"
                f"\nmin/global = {lf.min_fos / lf.global_fos:.2f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.9, edgecolor='#999999'), zorder=10)

    _style_axes(ax, title)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Seepage
# ---------------------------------------------------------------------------

def plot_seepage(seepage_result, n_levels=14, show_vectors=True,
                 title='Steady-State Seepage — Total Head',
                 figsize=(10, 6.5)):
    """Head contours with Darcy flow vectors.

    Parameters
    ----------
    seepage_result : SeepageResult (carries nodes, elements, head,
        velocity arrays)
    show_vectors : bool — overlay element-centroid Darcy velocity
        vectors.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if seepage_result.head is None or seepage_result.nodes is None:
        raise ValueError("SeepageResult carries no head/node arrays")
    plt = _get_plt()

    nodes = np.asarray(seepage_result.nodes, dtype=float)
    elements = np.asarray(seepage_result.elements, dtype=int)
    head = np.asarray(seepage_result.head, dtype=float)
    tri = _triangulation(nodes, elements)

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.tricontourf(tri, head, levels=n_levels, cmap='Blues')
    lines = ax.tricontour(tri, head, levels=n_levels,
                          colors='#1e3a8a', linewidths=0.5)
    ax.clabel(lines, fontsize=6.5, fmt='%.1f')
    _draw_outline(ax, nodes, elements, color='#555555', lw=0.9)

    if show_vectors and seepage_result.velocity is not None:
        vel = np.asarray(seepage_result.velocity, dtype=float)
        cent = nodes[_corner_elements(elements)].mean(axis=1)
        vmag = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
        keep = vmag > 1e-15
        if np.any(keep):
            ax.quiver(cent[keep, 0], cent[keep, 1],
                      vel[keep, 0], vel[keep, 1],
                      color='#b91c1c', width=0.0025, alpha=0.8,
                      zorder=6)

    cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Total head (m)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.text(0.02, 0.98,
            f"head: {seepage_result.min_head_m:.2f} to "
            f"{seepage_result.max_head_m:.2f} m\n"
            f"Q = {seepage_result.total_flow_m3_per_s_per_m:.3e} "
            f"m³/s per m\nv_max = "
            f"{seepage_result.max_velocity_m_per_s:.3e} m/s",
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.9, edgecolor='#999999'), zorder=10)

    _style_axes(ax, title)
    fig.tight_layout()
    return fig



# ---------------------------------------------------------------------------
# 8. Consolidation time history
# ---------------------------------------------------------------------------

def plot_consolidation_history(consolidation_result,
                               title='Consolidation Time History',
                               figsize=(8.5, 5)):
    """Settlement and excess pore pressure dissipation vs time.

    Parameters
    ----------
    consolidation_result : ConsolidationResult (carries times,
        settlements, pore_pressures arrays)

    Returns
    -------
    matplotlib.figure.Figure
    """
    r = consolidation_result
    if r.times is None or r.settlements is None:
        raise ValueError("ConsolidationResult carries no time histories")
    plt = _get_plt()

    times = np.asarray(r.times, dtype=float)
    # settlements: (n_steps,) max settlement per step OR (n_steps, n)
    sett = np.asarray(r.settlements, dtype=float)
    if sett.ndim > 1:
        sett = sett.min(axis=1)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(times, sett * 1000.0, 'o-', color='#2563eb',
             markersize=3.5, linewidth=1.5, label='Settlement')
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Settlement (mm)', fontsize=10, color='#2563eb')
    ax1.tick_params(axis='y', labelcolor='#2563eb', labelsize=9)
    ax1.tick_params(axis='x', labelsize=9)
    if times.max() / max(times[times > 0].min(), 1e-12) > 100:
        ax1.set_xscale('log')

    if r.pore_pressures is not None:
        pp = np.asarray(r.pore_pressures, dtype=float)
        if pp.ndim > 1:
            pp = pp.max(axis=1)
        ax2 = ax1.twinx()
        ax2.plot(times, pp, 's--', color='#dc2626', markersize=3.5,
                 linewidth=1.4, label='Max excess pore pressure')
        ax2.set_ylabel('Excess pore pressure (kPa)', fontsize=10,
                       color='#dc2626')
        ax2.tick_params(axis='y', labelcolor='#dc2626', labelsize=9)

    ax1.set_title(
        f'{title}  (U_final = {r.degree_of_consolidation:.2f})',
        fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig

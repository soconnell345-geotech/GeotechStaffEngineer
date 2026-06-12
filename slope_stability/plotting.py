"""
SLOPE/W-style report figures for slope stability results.

All functions return a matplotlib Figure and import matplotlib lazily,
so the module imports cleanly when matplotlib is absent (optional
dependency, project pattern: geotech_common.plotting.get_pyplot).

Functions
---------
plot_cross_section        - annotated section: layers, GWT, slip
                            surface, slices, reinforcement, thrust line
plot_trial_surface_map    - all trial surfaces colored by FOS,
                            critical surface highlighted
plot_slice_forces         - per-slice W / N' / S_mob distribution
plot_interslice_forces    - interslice E/X distribution + line of
                            thrust (rigorous methods)
plot_mc_histogram         - Monte Carlo FOS histogram + lognormal fit
plot_fosm_tornado         - FOSM variance-contribution chart

Everything here RENDERS result fields (CS-3) - no engineering values
are re-derived. The only geometry computed locally is the drawing of
circular arcs from stored (xc, yc, R) triples.
"""

import math
from typing import Optional

import numpy as np

# Engineering-report palette (muted earth tones, SLOPE/W-like)
LAYER_COLORS = ['#e8d9b0', '#cfa97c', '#a9c3a4', '#c9b8a6',
                '#dfd0a2', '#b3c4cf', '#d2c3ab', '#bfd3bf']
LAYER_HATCHES = ['', '..', '//', '\\\\', 'xx', '--', '++', 'oo']

_CRIT_COLOR = '#cc0000'
_GWT_COLOR = '#1f6fbf'
_THRUST_COLOR = '#9333ea'


def _get_plt():
    from geotech_common.plotting import get_pyplot
    return get_pyplot()


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def _section_limits(geom):
    x_min, x_max = geom.x_range
    z_bot = min(L.bottom_elevation for L in geom.soil_layers)
    z_top = max(z for _, z in geom.surface_points)
    return x_min, x_max, z_bot, z_top


def _draw_section_base(ax, geom, layer_alpha=0.75, show_legend_layers=True,
                       lw_surface=1.8):
    """Draw soil layers, ground surface, GWT and ponded water."""
    x_min, x_max, z_bot, z_top = _section_limits(geom)
    xs = np.linspace(x_min, x_max, 400)
    zg = np.array([geom.ground_elevation_at(x) for x in xs])

    for i, layer in enumerate(geom.soil_layers):
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        hatch = LAYER_HATCHES[i % len(LAYER_HATCHES)]
        top = np.array([min(geom.ground_elevation_at(x), layer.top_at(x))
                        for x in xs])
        bot = np.array([layer.bottom_at(x) for x in xs])
        mask = top > bot
        if np.any(mask):
            ax.fill_between(
                xs, bot, top, where=mask, facecolor=color,
                alpha=layer_alpha, hatch=hatch, edgecolor='#777777',
                linewidth=0.4,
                label=layer.name if show_legend_layers else None)

    # Ground surface
    sx = [p[0] for p in geom.surface_points]
    sz = [p[1] for p in geom.surface_points]
    ax.plot(sx, sz, 'k-', linewidth=lw_surface, label='Ground surface',
            zorder=5)

    # GWT + ponded water
    if geom.gwt_points:
        gz = np.array([geom.gwt_elevation_at(x) for x in xs])
        ax.plot(xs, gz, '--', color=_GWT_COLOR, linewidth=1.4,
                label='GWT', zorder=6)
        # ponded water: GWT above ground surface
        pond = gz > zg + 1e-9
        if np.any(pond):
            ax.fill_between(xs, zg, gz, where=pond, facecolor='#9ecbff',
                            alpha=0.5, edgecolor='none', zorder=2,
                            label='Ponded water')
        # standard water-table marker (inverted triangle)
        xm = xs[len(xs) // 3]
        ax.plot([xm], [geom.gwt_elevation_at(xm)], marker='v',
                color=_GWT_COLOR, markersize=7, zorder=7)

    return xs, zg


def _slip_polyline(result, geom, n=200):
    """Sample the slip surface stored on a result as (x, z) arrays."""
    if result.slip_points:
        pts = np.asarray(result.slip_points, dtype=float)
        return pts[:, 0], pts[:, 1]
    if result.radius > 0:
        xs = np.linspace(result.x_entry, result.x_exit, n)
        dx = xs - result.xc
        dx = np.clip(dx, -result.radius, result.radius)
        zs = result.yc - np.sqrt(np.maximum(result.radius**2 - dx**2, 0.0))
        return xs, zs
    raise ValueError("Result carries neither slip_points nor a circle")


def _circle_arc(geom, xc, yc, R, n=80):
    """Ground-clipped lower arc for a stored trial circle, or None."""
    from slope_stability.slip_surface import CircularSlipSurface
    try:
        slip = CircularSlipSurface(xc, yc, R)
        x_en, x_ex = slip.find_entry_exit(geom)
    except ValueError:
        return None
    xs = np.linspace(x_en, x_ex, n)
    dx = np.clip(xs - xc, -R, R)
    zs = yc - np.sqrt(np.maximum(R**2 - dx**2, 0.0))
    return xs, zs


def _draw_reinforcement(ax, geom, result=None):
    """Draw nails / anchors / geosynthetics; mark mobilized crossings."""
    seen = set()

    def _label(kind):
        if kind in seen:
            return None
        seen.add(kind)
        return kind

    for nail in (geom.nails or []):
        dx = math.cos(math.radians(nail.inclination))
        dz = -math.sin(math.radians(nail.inclination))
        # nails point INTO the slope (toward rising ground)
        sgn = 1.0 if geom.ground_elevation_at(nail.x_head + 1.0) >= \
            geom.ground_elevation_at(nail.x_head - 1.0) else -1.0
        ax.plot([nail.x_head, nail.x_head + sgn * dx * nail.length],
                [nail.z_head, nail.z_head + dz * nail.length],
                '-', color='#444444', linewidth=1.6, zorder=8,
                label=_label('Soil nails'))
        ax.plot([nail.x_head], [nail.z_head], 's', color='#444444',
                markersize=3.5, zorder=8)

    for a in (geom.anchors or []):
        dx = math.cos(math.radians(a.inclination))
        dz = -math.sin(math.radians(a.inclination))
        sgn = 1.0 if geom.ground_elevation_at(a.x_head + 1.0) >= \
            geom.ground_elevation_at(a.x_head - 1.0) else -1.0
        x_tip = a.x_head + sgn * dx * a.length
        z_tip = a.z_head + dz * a.length
        ax.plot([a.x_head, x_tip], [a.z_head, z_tip], '-',
                color='#0e7490', linewidth=1.6, zorder=8,
                label=_label('Anchors'))
        # bond-zone tick at the tip
        ax.plot([x_tip], [z_tip], 'D', color='#0e7490', markersize=4,
                zorder=8)

    x_min, x_max, _, _ = _section_limits(geom)
    for g in (geom.geosynthetics or []):
        x0 = g.x_start if g.x_start is not None else x_min
        x1 = g.x_end if g.x_end is not None else x_max
        ax.plot([x0, x1], [g.elevation, g.elevation], '-',
                color='#15803d', linewidth=1.8, zorder=8,
                label=_label('Geosynthetics'))

    # Mobilized crossings stored on the result (engine-computed)
    if result is not None and result.reinforcements:
        for r in result.reinforcements:
            ax.plot([r['x_m']], [r['z_m']], 'o', color='#b91c1c',
                    markersize=5, zorder=9,
                    label=_label('Slip crossing (T mobilized)'))


def _style_section_axes(ax, title):
    ax.set_xlabel('Distance (m)', fontsize=10)
    ax.set_ylabel('Elevation (m)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=9)


# ---------------------------------------------------------------------------
# 1. Cross-section
# ---------------------------------------------------------------------------

def plot_cross_section(result, geom, show_slices=True,
                       show_thrust_line=True, show_reinforcement=True,
                       fos_required: Optional[float] = None,
                       title: str = "Slope Stability Analysis - Cross-Section",
                       figsize=(10, 6)):
    """SLOPE/W-style annotated cross-section for one analyzed surface.

    Parameters
    ----------
    result : SlopeStabilityResult
    geom : SlopeGeometry
    show_slices : bool - draw the slice discretization (needs
        result.slice_data).
    show_thrust_line : bool - draw the line of thrust (rigorous
        methods only; needs result.thrust_line).
    show_reinforcement : bool - draw nails/anchors/geosynthetics and
        mark mobilized slip-surface crossings.
    fos_required : float, optional - annotate STABLE/UNSTABLE against
        this threshold instead of the default 1.5.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=figsize)

    _draw_section_base(ax, geom)

    # Slices (under the slip surface line)
    if show_slices and result.slice_data:
        for s in result.slice_data:
            x_l = s.x_mid - s.width / 2.0
            x_r = s.x_mid + s.width / 2.0
            for xb in (x_l, x_r):
                # vertical slice boundary, clipped between base and top
                if result.radius > 0:
                    zb = result.yc - math.sqrt(max(
                        result.radius**2 - (xb - result.xc)**2, 0.0))
                else:
                    zb = s.z_base
                zt = geom.ground_elevation_at(xb)
                if zt > zb:
                    ax.plot([xb, xb], [zb, zt], '-', color='#555555',
                            linewidth=0.35, alpha=0.55, zorder=4)

    # Slip surface
    sxs, szs = _slip_polyline(result, geom)
    ax.plot(sxs, szs, '-', color=_CRIT_COLOR, linewidth=2.4, zorder=7,
            label=f'Slip surface (FOS = {result.FOS:.3f})')

    # entry/exit markers
    for x_pt, mk, lbl in ((result.x_entry, 'v', 'Entry'),
                          (result.x_exit, '^', 'Exit')):
        z_pt = geom.ground_elevation_at(x_pt)
        ax.plot([x_pt], [z_pt], mk, color=_CRIT_COLOR, markersize=7,
                zorder=8)
        ax.annotate(lbl, xy=(x_pt, z_pt), fontsize=7,
                    xytext=(4, -12), textcoords='offset points')

    # circle center + radius callout
    if result.radius > 0:
        ax.plot([result.xc], [result.yc], '+', color=_CRIT_COLOR,
                markersize=12, markeredgewidth=1.8, zorder=8)
        x_mid_arc = 0.5 * (result.x_entry + result.x_exit)
        z_mid_arc = result.yc - math.sqrt(max(
            result.radius**2 - (x_mid_arc - result.xc)**2, 0.0))
        ax.plot([result.xc, x_mid_arc], [result.yc, z_mid_arc], ':',
                color=_CRIT_COLOR, linewidth=0.9, zorder=6)
        ax.annotate(
            f'({result.xc:.1f}, {result.yc:.1f})\nR = {result.radius:.1f} m',
            xy=(result.xc, result.yc), fontsize=7,
            xytext=(6, 4), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.85, edgecolor='#999999'))

    # Tension crack at the upslope end
    if result.tension_crack_depth > 0:
        z_en = geom.ground_elevation_at(result.x_entry)
        z_ex = geom.ground_elevation_at(result.x_exit)
        x_crk = result.x_entry if z_en >= z_ex else result.x_exit
        z_top_crk = geom.ground_elevation_at(x_crk)
        z_bot_crk = z_top_crk - result.tension_crack_depth
        ax.plot([x_crk, x_crk], [z_bot_crk, z_top_crk], '-',
                color='#7c2d12', linewidth=2.2, zorder=8,
                label=f'Tension crack ({result.tension_crack_depth:.1f} m)')
        if result.tension_crack_water_depth > 0:
            zw = z_bot_crk + result.tension_crack_water_depth
            ax.plot([x_crk, x_crk], [z_bot_crk, zw], '-',
                    color=_GWT_COLOR, linewidth=4.0, alpha=0.6, zorder=8)

    # Thrust line
    if show_thrust_line and result.thrust_line:
        tx = [p[0] for p in result.thrust_line]
        tz = [p[1] for p in result.thrust_line]
        ax.plot(tx, tz, '-.', color=_THRUST_COLOR, linewidth=1.6,
                zorder=8, label='Line of thrust')

    if show_reinforcement:
        _draw_reinforcement(ax, geom, result)

    # FOS annotation box
    req = fos_required if fos_required is not None else 1.5
    ok = result.FOS >= req
    color = '#16a34a' if ok else '#dc2626'
    lines = [f"FOS = {result.FOS:.3f}  ({result.method})"]
    if result.theta_spencer is not None:
        lines.append(f"theta = {result.theta_spencer:.1f} deg")
    if result.lambda_mp is not None:
        lines.append(f"lambda = {result.lambda_mp:.3f}")
    lines.append(f"[{'STABLE' if ok else 'UNSTABLE'} vs FOS_req = {req:g}]")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=9.5, fontweight='bold', va='top', color=color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=color, alpha=0.92), zorder=10)

    _style_section_axes(ax, title)
    ax.legend(loc='lower right', fontsize=7, ncol=2, framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Trial-surface map
# ---------------------------------------------------------------------------

def plot_trial_surface_map(search, geom, max_surfaces: int = 200,
                           title: str = "Critical Surface Search - Trial Surfaces",
                           figsize=(10, 6.5)):
    """SLOPE/W-style map of all trial surfaces colored by FOS.

    Parameters
    ----------
    search : SearchResult
        From search_critical_surface() / grid_search() / the
        noncircular searches. Circular trials are re-drawn from the
        stored (xc, yc, R) triples in ``grid_fos``; noncircular trials
        come from ``trial_surfaces`` (stored by the search).
    geom : SlopeGeometry
    max_surfaces : int - subsample ceiling so dense searches stay
        readable.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    from matplotlib import cm, colors as mcolors

    # Collect drawable trials: list of (fos, xs, zs)
    trials = []
    for g in search.grid_fos:
        fos = g.get('FOS', 999.9)
        if fos >= 900:
            continue
        R = g.get('R', 0.0)
        if R and R > 0:
            arc = _circle_arc(geom, g['xc'], g['yc'], R)
            if arc is not None:
                trials.append((fos, arc[0], arc[1]))
    for t in getattr(search, 'trial_surfaces', None) or []:
        fos = t.get('FOS', 999.9)
        if fos >= 900 or not t.get('points'):
            continue
        pts = np.asarray(t['points'], dtype=float)
        trials.append((fos, pts[:, 0], pts[:, 1]))

    if not trials:
        raise ValueError(
            "SearchResult contains no drawable trial surfaces "
            "(grid_fos empty and no trial_surfaces stored)")

    # Subsample evenly across the FOS ordering, keeping the extremes
    if len(trials) > max_surfaces:
        trials.sort(key=lambda t: t[0])
        idx = np.linspace(0, len(trials) - 1, max_surfaces).astype(int)
        trials = [trials[i] for i in idx]

    fos_vals = np.array([t[0] for t in trials])
    vmin = float(fos_vals.min())
    vmax = float(np.percentile(fos_vals, 90))
    if vmax - vmin < 1e-6:
        vmax = vmin + 0.1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap('RdYlGn')

    fig, ax = plt.subplots(figsize=figsize)
    _draw_section_base(ax, geom, layer_alpha=0.45)

    # Draw worst (lowest FOS) last so they stay visible
    for fos, xs, zs in sorted(trials, key=lambda t: -t[0]):
        ax.plot(xs, zs, '-', color=cmap(norm(fos)), linewidth=0.8,
                alpha=0.75, zorder=4)

    # Critical surface
    crit = search.critical
    if crit is not None:
        cxs, czs = _slip_polyline(crit, geom)
        ax.plot(cxs, czs, '-', color='black', linewidth=2.8, zorder=7)
        ax.plot(cxs, czs, '-', color=_CRIT_COLOR, linewidth=1.8, zorder=8,
                label=f'Critical surface (FOS = {crit.FOS:.3f})')
        if crit.radius > 0:
            ax.plot([crit.xc], [crit.yc], '*', color=_CRIT_COLOR,
                    markersize=14, markeredgecolor='black',
                    markeredgewidth=0.6, zorder=9, label='Critical center')

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Factor of Safety', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.text(0.02, 0.98,
            f"{search.n_surfaces_evaluated} surfaces evaluated\n"
            f"{len(trials)} shown",
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.9, edgecolor='#999999'), zorder=10)

    _style_section_axes(ax, title)
    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Slice force diagram
# ---------------------------------------------------------------------------

def plot_slice_forces(result, figsize=(9.5, 5)):
    """Per-slice force distribution: W bars + N', S_mob, u*l lines.

    Requires ``result.slice_data`` (analyze_slope(...,
    include_slice_data=True)).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not result.slice_data:
        raise ValueError("No slice data on result - run analyze_slope "
                         "with include_slice_data=True")
    plt = _get_plt()
    sd = result.slice_data
    x = [s.x_mid for s in sd]
    widths = [s.width * 0.85 for s in sd]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, [s.weight for s in sd], width=widths, color='#cbd5e1',
           edgecolor='#64748b', linewidth=0.5, label='W (slice weight)')
    ax.plot(x, [s.N_eff_kN for s in sd], 'o-', color='#2563eb',
            markersize=3.5, linewidth=1.4, label="N' (eff. base normal)")
    ax.plot(x, [s.S_mob_kN for s in sd], 's--', color='#dc2626',
            markersize=3.5, linewidth=1.4, label='S_mob (mobilized shear)')
    ax.plot(x, [s.U_base_kN for s in sd], '^:', color='#0891b2',
            markersize=3.5, linewidth=1.2, label='u*l (base water force)')
    ax.axhline(0, color='black', linewidth=0.6)

    ax.set_xlabel('Slice midpoint x (m)', fontsize=10)
    ax.set_ylabel('Force (kN/m)', fontsize=10)
    ax.set_title(
        f'Slice Forces - {result.method}, FOS = {result.FOS:.3f} '
        f'({result.n_slices} slices)',
        fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Interslice forces + thrust line (rigorous methods)
# ---------------------------------------------------------------------------

def plot_interslice_forces(result, geom=None, figsize=(9.5, 7)):
    """Interslice E/X distribution and line of thrust.

    Top panel: interslice normal E and shear X at every slice
    boundary. Bottom panel: line of thrust between the slip surface
    and the ground line.

    Requires the rigorous GLE engine output (Spencer / M-P / GLE):
    slice_data with E/X and result.thrust_line.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sd = result.slice_data
    if not sd or sd[0].E_left_kN is None:
        raise ValueError(
            "Interslice forces unavailable - rigorous method "
            "(Spencer / Morgenstern-Price / GLE) with "
            "include_slice_data=True required")
    plt = _get_plt()

    # boundary x and forces: left edge of each slice + right edge of last
    bx = [s.x_mid - s.width / 2.0 for s in sd] + \
         [sd[-1].x_mid + sd[-1].width / 2.0]
    E = [s.E_left_kN for s in sd] + [sd[-1].E_right_kN]
    X = [s.X_left_kN for s in sd] + [sd[-1].X_right_kN]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.2]})

    ax1.plot(bx, E, 'o-', color='#2563eb', markersize=3.5,
             linewidth=1.5, label='E (interslice normal)')
    ax1.plot(bx, X, 's--', color='#dc2626', markersize=3.5,
             linewidth=1.5, label='X (interslice shear)')
    ax1.axhline(0, color='black', linewidth=0.6)
    note = []
    if result.theta_spencer is not None:
        note.append(f'theta = {result.theta_spencer:.1f} deg')
    if result.lambda_mp is not None:
        note.append(f'lambda = {result.lambda_mp:.3f}')
    ax1.set_title(
        f'Interslice Forces - {result.method}, FOS = {result.FOS:.3f}'
        + (f'  ({", ".join(note)})' if note else ''),
        fontsize=12, fontweight='bold')
    ax1.set_ylabel('Force (kN/m)', fontsize=10)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)
    ax1.tick_params(labelsize=9)

    # Bottom: thrust line in elevation view
    if geom is not None:
        _draw_section_base(ax2, geom, layer_alpha=0.35,
                           show_legend_layers=False, lw_surface=1.2)
    else:
        ax2.plot([s.x_mid for s in sd], [s.z_top for s in sd], 'k-',
                 linewidth=1.2, label='Ground surface')
    ax2.plot([s.x_mid for s in sd], [s.z_base for s in sd], '-',
             color=_CRIT_COLOR, linewidth=2.0, label='Slip surface')
    if result.thrust_line:
        tx = [p[0] for p in result.thrust_line]
        tz = [p[1] for p in result.thrust_line]
        ax2.plot(tx, tz, '-.', color=_THRUST_COLOR, linewidth=1.8,
                 label='Line of thrust', zorder=8)
    ax2.set_xlabel('x (m)', fontsize=10)
    ax2.set_ylabel('Elevation (m)', fontsize=10)
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=7.5, loc='lower right')
    ax2.tick_params(labelsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Monte Carlo histogram
# ---------------------------------------------------------------------------

def plot_mc_histogram(mc, figsize=(8.5, 5)):
    """Monte Carlo FOS histogram with moment-matched lognormal overlay.

    Parameters
    ----------
    mc : MonteCarloResult

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not mc.histogram_bins or not mc.histogram_counts:
        raise ValueError("MonteCarloResult carries no histogram data")
    plt = _get_plt()

    edges = np.asarray(mc.histogram_bins, dtype=float)
    counts = np.asarray(mc.histogram_counts, dtype=float)
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2.0
    density = counts / max(counts.sum(), 1.0) / np.maximum(widths, 1e-12)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(centers, density, width=widths * 0.97, color='#93c5fd',
           edgecolor='#1d4ed8', linewidth=0.5,
           label=f'MC samples (n = {mc.n})')

    # Moment-matched lognormal from stored mean / COV (result fields)
    if mc.fos_mean > 0 and mc.fos_cov > 0:
        s_ln = math.sqrt(math.log(1.0 + mc.fos_cov ** 2))
        mu_ln = math.log(mc.fos_mean) - 0.5 * s_ln ** 2
        xs = np.linspace(max(edges[0] * 0.9, 1e-3), edges[-1] * 1.05, 300)
        pdf = (1.0 / (xs * s_ln * math.sqrt(2 * math.pi))
               * np.exp(-(np.log(xs) - mu_ln) ** 2 / (2 * s_ln ** 2)))
        ax.plot(xs, pdf, '-', color='#b91c1c', linewidth=1.8,
                label='Lognormal fit')

    ax.axvline(1.0, color='#dc2626', linestyle='--', linewidth=1.4,
               label='FOS = 1.0')
    ax.axvline(mc.fos_mean, color='#16a34a', linestyle=':',
               linewidth=1.4, label=f'Mean = {mc.fos_mean:.3f}')

    box = (f"FOS mean = {mc.fos_mean:.3f}  (COV = {mc.fos_cov:.3f})\n"
           f"beta_LN = {mc.beta_lognormal:.2f}\n"
           f"pf (count)  = {mc.pf:.2%}\n"
           f"pf (logn.)  = {mc.pf_lognormal:.2%}")
    ax.text(0.98, 0.97, box, transform=ax.transAxes, fontsize=8.5,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#666666', alpha=0.92))

    ax.set_xlabel('Factor of Safety', fontsize=10)
    ax.set_ylabel('Probability density', fontsize=10)
    ax.set_title(
        f'Monte Carlo FOS Distribution - {mc.method.capitalize()} method',
        fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. FOSM tornado / variance-contribution chart
# ---------------------------------------------------------------------------

def plot_fosm_tornado(fosm, figsize=(8.5, 4.8)):
    """FOSM variance-contribution chart (Duncan 2000 tornado style).

    Parameters
    ----------
    fosm : FOSMResult

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not fosm.variable_variance_pct:
        raise ValueError("FOSMResult carries no variance contributions")
    plt = _get_plt()

    items = sorted(fosm.variable_variance_pct.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    pcts = [v for _, v in items]
    deltas = [fosm.variable_deltas.get(k, 0.0) for k in names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(names, pcts, color='#60a5fa', edgecolor='#1e40af',
                   linewidth=0.6, height=0.6)
    for bar, dF in zip(bars, deltas):
        ax.text(bar.get_width() + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f'dF = {dF:+.3f}', va='center', fontsize=8)

    box = (f"F_MLV = {fosm.fos_mlv:.3f}\n"
           f"sigma_F = {fosm.sigma_f:.3f}  (COV = {fosm.cov_f:.3f})\n"
           f"beta (normal) = {fosm.beta_normal:.2f}  "
           f"(pf = {fosm.pf_normal:.2%})\n"
           f"beta (lognormal) = {fosm.beta_lognormal:.2f}  "
           f"(pf = {fosm.pf_lognormal:.2%})")
    ax.text(0.98, 0.04, box, transform=ax.transAxes, fontsize=8.5,
            va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#666666', alpha=0.92))

    ax.set_xlabel('Contribution to Var(FOS) (%)', fontsize=10)
    ax.set_xlim(0, max(100.0, max(pcts) * 1.15))
    ax.set_title('FOSM Reliability - Variance Contributions',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.25)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig

"""Echo-back rendering — the human-confirmation artifact.

:func:`echo_back` draws the Project AS THE MACHINE UNDERSTANDS IT — layers,
GWT, reinforcement, surcharge — to a PNG, plus a numbered vertex table
(plain text) keyed to markers on the figure. The human verifies the picture
against reality/the original drawing and confirms (or corrects vertex by
number). This direction — numbers → image — is the one a human can check at
a glance; the reverse (image → numbers, i.e. trusting an LLM's read of a
drawing) is NOT trusted anywhere in this system.

Works PRE-ANALYSIS and with incomplete materials (renders straight off the
Project; no SlopeGeometry needed). Matplotlib is an optional dependency,
imported lazily via geotech_common.plotting.get_pyplot (project pattern).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from geo_project.schema import Project

# Same engineering palette as slope_stability.plotting.
LAYER_COLORS = ['#e8d9b0', '#cfa97c', '#a9c3a4', '#c9b8a6',
                '#dfd0a2', '#b3c4cf', '#d2c3ab', '#bfd3bf']
LAYER_HATCHES = ['', '..', '//', '\\\\', 'xx', '--', '++', 'oo']
_GWT_COLOR = '#1f6fbf'
_SURCHARGE_COLOR = '#b45309'


def _get_plt():
    from geotech_common.plotting import get_pyplot
    return get_pyplot()


def _interp(points: List[Tuple[float, float]], x: float) -> float:
    """Linear interpolation with constant extrapolation."""
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for i in range(len(points) - 1):
        x0, z0 = points[i]
        x1, z1 = points[i + 1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return z0 + t * (z1 - z0)
    return points[-1][1]


@dataclass
class EchoBack:
    """The confirmation artifact: image path + the numbered vertex table."""
    image_path: Optional[str]
    vertex_table: str

    def to_dict(self) -> dict:
        return {"image_path": self.image_path,
                "vertex_table": self.vertex_table}


# ---------------------------------------------------------------------------
# Vertex table (text — returned alongside the PNG)
# ---------------------------------------------------------------------------

def vertex_table(project: Project) -> str:
    """Numbered vertex listing for every polyline the figure shows.

    Vertex ids (S1.., B<k>.1.., W1..) match the markers on the echo-back
    figure so the human can call out a correction by id ("S3 should be z=2,
    not 0").
    """
    lines: List[str] = []
    g = project.geometry
    lines.append(f"PROVENANCE: {g.provenance}"
                 + ("  ** UNCONFIRMED VISION DRAFT — verify every number **"
                    if g.provenance == "vision_draft" else ""))
    lines.append("")
    lines.append("Ground surface (S#: x, z [m]):")
    for i, (x, z) in enumerate(g.surface_points, start=1):
        lines.append(f"  S{i}: ({x:g}, {z:g})")

    for k, (bname, pts) in enumerate(g.layer_boundaries.items(), start=1):
        lines.append(f"Layer boundary '{bname}' (B{k}.#: x, z [m]):")
        for i, (x, z) in enumerate(sorted(pts, key=lambda p: p[0]), start=1):
            lines.append(f"  B{k}.{i}: ({x:g}, {z:g})")

    lines.append("Layers (top -> bottom):")
    for i, layer in enumerate(project.stratigraphy):
        top = project.layer_top(i)
        bot = project.layer_bottom(i)
        bb = (f", bottom = boundary '{layer.bottom_boundary}'"
              if layer.bottom_boundary else "")
        lines.append(
            f"  {i + 1}. {layer.name or f'layer_{i}'}: top "
            f"{'?' if top is None else f'{top:g}'} m, bottom "
            f"{'?' if bot is None else f'{bot:g}'} m{bb}")

    if project.water.gwt_points:
        lines.append("GWT (W#: x, z [m]):")
        for i, (x, z) in enumerate(project.water.gwt_points, start=1):
            lines.append(f"  W{i}: ({x:g}, {z:g})")
    elif project.water.ru:
        lines.append(f"Water: ru = {project.water.ru:g} (no GWT line)")
    else:
        lines.append("Water: none defined")

    if project.loads.surcharges:
        lines.append("Surcharges:")
        for k, s in enumerate(project.loads.surcharges, start=1):
            band = ("full surface" if s.x_start is None or s.x_end is None
                    else f"x = {s.x_start:g} to {s.x_end:g} m")
            lines.append(f"  Q{k}: {s.q:g} kPa over {band}"
                         + (f" ({s.label})" if s.label else ""))
    if project.loads.kh:
        lines.append(f"Seismic: kh = {project.loads.kh:g}")

    r = project.reinforcement
    if r.nails:
        lines.append("Nails (N#: head x, z / length / incl):")
        for i, n in enumerate(r.nails, start=1):
            lines.append(f"  N{i}: ({n.x_head:g}, {n.z_head:g}) "
                         f"L={n.length:g} m @ {n.inclination:g} deg")
    if r.anchors:
        lines.append("Anchors (A#: head x, z / length / incl / T_allow):")
        for i, a in enumerate(r.anchors, start=1):
            lines.append(f"  A{i}: ({a.x_head:g}, {a.z_head:g}) "
                         f"L={a.length:g} m @ {a.inclination:g} deg, "
                         f"T={a.T_allow:g} kN/m")
    if r.geosynthetics:
        lines.append("Geosynthetics (G#: elevation / extent / T_allow):")
        for i, gl in enumerate(r.geosynthetics, start=1):
            ext = ("full width" if gl.x_start is None or gl.x_end is None
                   else f"x = {gl.x_start:g} to {gl.x_end:g} m")
            lines.append(f"  G{i}: z={gl.elevation:g} m, {ext}, "
                         f"T={gl.T_allow:g} kN/m")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def echo_back(project: Project, path: Optional[str] = None,
              figsize=(11, 6.5), dpi: int = 130) -> EchoBack:
    """Render the echo-back cross-section PNG + numbered vertex table.

    Parameters
    ----------
    project : Project
        Any state — works pre-analysis and with incomplete materials.
    path : str, optional
        PNG output path. When None, only the vertex table is produced
        (image_path=None) — still useful where matplotlib is absent.
    figsize, dpi
        Figure size/resolution.

    Returns
    -------
    EchoBack
        ``image_path`` (None when no path given) + ``vertex_table`` text.
        Vertex markers on the figure carry the same S#/B#.#/W# ids as the
        table.
    """
    table = vertex_table(project)
    if path is None:
        return EchoBack(image_path=None, vertex_table=table)

    plt = _get_plt()
    g = project.geometry
    if len(g.surface_points) < 2:
        raise ValueError("Cannot render: fewer than 2 surface points")

    fig, ax = plt.subplots(figsize=figsize)
    x_min, x_max = g.x_range
    z_min_surf, z_max_surf = g.z_surface_range
    bot = project.section_bottom()
    z_floor = bot if bot is not None else z_min_surf - 5.0

    xs = np.linspace(x_min, x_max, 400)
    surf = [(float(x), float(z)) for x, z in g.surface_points]
    zg = np.array([_interp(surf, x) for x in xs])

    # Layer fills (clipped to the ground surface).
    n_layers = len(project.stratigraphy)
    for i in range(n_layers):
        top_e = project.layer_top(i)
        bpts = project.boundary_points(i)
        bot_e = project.layer_bottom(i)
        if top_e is None or (bot_e is None and not bpts):
            continue
        top_prev = project.boundary_points(i - 1) if i > 0 else None
        top_arr = np.array([
            min(_interp(surf, x),
                _interp(top_prev, x) if top_prev else top_e)
            for x in xs])
        bot_arr = np.array([
            _interp(bpts, x) if bpts else bot_e for x in xs])
        mask = top_arr > bot_arr
        if np.any(mask):
            ax.fill_between(
                xs, bot_arr, top_arr, where=mask,
                facecolor=LAYER_COLORS[i % len(LAYER_COLORS)],
                hatch=LAYER_HATCHES[i % len(LAYER_HATCHES)],
                alpha=0.7, edgecolor='#777777', linewidth=0.4,
                label=project.stratigraphy[i].name or f"layer_{i}")

    # Ground surface + numbered vertices.
    sx = [p[0] for p in surf]
    sz = [p[1] for p in surf]
    ax.plot(sx, sz, 'k-', linewidth=2.0, label='Ground surface', zorder=6)
    for i, (x, z) in enumerate(surf, start=1):
        ax.plot([x], [z], 'o', color='black', markersize=4, zorder=7)
        ax.annotate(f"S{i}", xy=(x, z), xytext=(3, 5),
                    textcoords='offset points', fontsize=7,
                    fontweight='bold', zorder=8)

    # Boundary polylines + numbered vertices.
    for k, (bname, pts) in enumerate(g.layer_boundaries.items(), start=1):
        spts = sorted(((float(x), float(z)) for x, z in pts),
                      key=lambda p: p[0])
        bx = [p[0] for p in spts]
        bz = [p[1] for p in spts]
        ax.plot(bx, bz, '-', color='#555555', linewidth=1.2, zorder=5)
        for i, (x, z) in enumerate(spts, start=1):
            ax.plot([x], [z], 's', color='#555555', markersize=3, zorder=7)
            ax.annotate(f"B{k}.{i}", xy=(x, z), xytext=(3, -9),
                        textcoords='offset points', fontsize=6.5, zorder=8)

    # GWT.
    if project.water.gwt_points:
        wpts = [(float(x), float(z)) for x, z in project.water.gwt_points]
        gz = np.array([_interp(wpts, x) for x in xs])
        ax.plot(xs, gz, '--', color=_GWT_COLOR, linewidth=1.5, label='GWT',
                zorder=6)
        pond = gz > zg + 1e-9
        if np.any(pond):
            ax.fill_between(xs, zg, gz, where=pond, facecolor='#9ecbff',
                            alpha=0.5, edgecolor='none', zorder=2,
                            label='Ponded water')
        for i, (x, z) in enumerate(wpts, start=1):
            ax.plot([x], [z], 'v', color=_GWT_COLOR, markersize=5, zorder=7)
            ax.annotate(f"W{i}", xy=(x, z), xytext=(3, 4),
                        textcoords='offset points', fontsize=6.5,
                        color=_GWT_COLOR, zorder=8)

    # Surcharge arrows (first band shown solid; extras dashed).
    for k, s in enumerate(project.loads.surcharges):
        if s.q <= 0:
            continue
        x0 = s.x_start if s.x_start is not None else x_min
        x1 = s.x_end if s.x_end is not None else x_max
        x0, x1 = max(x0, x_min), min(x1, x_max)
        if x1 <= x0:
            continue
        arrows = np.linspace(x0, x1, max(int((x1 - x0) / 2.0), 3))
        h = 0.06 * max(z_max_surf - z_floor, 1.0)
        for xa in arrows:
            za = _interp(surf, xa)
            ax.annotate(
                "", xy=(xa, za), xytext=(xa, za + h),
                arrowprops=dict(arrowstyle='-|>', color=_SURCHARGE_COLOR,
                                lw=1.2 if k == 0 else 0.8,
                                linestyle='-' if k == 0 else '--'),
                zorder=7)
        za0 = _interp(surf, 0.5 * (x0 + x1))
        ax.annotate(f"q={s.q:g} kPa", xy=(0.5 * (x0 + x1), za0 + h),
                    xytext=(0, 3), textcoords='offset points', fontsize=7.5,
                    color=_SURCHARGE_COLOR, ha='center', zorder=8)

    # Reinforcement.
    import math
    for i, n in enumerate(project.reinforcement.nails, start=1):
        dx = math.cos(math.radians(n.inclination))
        dz = -math.sin(math.radians(n.inclination))
        sgn = (1.0 if _interp(surf, n.x_head + 1.0)
               >= _interp(surf, n.x_head - 1.0) else -1.0)
        ax.plot([n.x_head, n.x_head + sgn * dx * n.length],
                [n.z_head, n.z_head + dz * n.length], '-', color='#444444',
                linewidth=1.6, zorder=7,
                label='Soil nails' if i == 1 else None)
        ax.annotate(f"N{i}", xy=(n.x_head, n.z_head), xytext=(-10, 0),
                    textcoords='offset points', fontsize=6.5, zorder=8)
    for i, a in enumerate(project.reinforcement.anchors, start=1):
        dx = math.cos(math.radians(a.inclination))
        dz = -math.sin(math.radians(a.inclination))
        sgn = (1.0 if _interp(surf, a.x_head + 1.0)
               >= _interp(surf, a.x_head - 1.0) else -1.0)
        x_tip = a.x_head + sgn * dx * a.length
        z_tip = a.z_head + dz * a.length
        ax.plot([a.x_head, x_tip], [a.z_head, z_tip], '-', color='#0e7490',
                linewidth=1.6, zorder=7,
                label='Anchors' if i == 1 else None)
        ax.plot([x_tip], [z_tip], 'D', color='#0e7490', markersize=4,
                zorder=7)
        ax.annotate(f"A{i}", xy=(a.x_head, a.z_head), xytext=(-10, 0),
                    textcoords='offset points', fontsize=6.5,
                    color='#0e7490', zorder=8)
    for i, gl in enumerate(project.reinforcement.geosynthetics, start=1):
        x0 = gl.x_start if gl.x_start is not None else x_min
        x1 = gl.x_end if gl.x_end is not None else x_max
        ax.plot([x0, x1], [gl.elevation, gl.elevation], '-',
                color='#15803d', linewidth=1.8, zorder=7,
                label='Geosynthetics' if i == 1 else None)
        ax.annotate(f"G{i}", xy=(x0, gl.elevation), xytext=(-12, 0),
                    textcoords='offset points', fontsize=6.5,
                    color='#15803d', zorder=8)

    # Status box: provenance + confirmation gates.
    c = project.confirmations
    status = [f"provenance: {g.provenance}"]
    for stage in ("geometry", "materials", "water_loads"):
        status.append(
            f"{'[x]' if getattr(c, stage) else '[ ]'} {stage} confirmed")
    box_color = ('#dc2626' if g.provenance == 'vision_draft'
                 and not c.geometry else '#2563eb')
    if g.provenance == "vision_draft" and not c.geometry:
        status.append("VISION DRAFT — DO NOT TRUST UNTIL CONFIRMED")
    ax.text(0.02, 0.98, "\n".join(status), transform=ax.transAxes,
            fontsize=8.5, va='top', color=box_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=box_color, alpha=0.92), zorder=10)

    title = (f"ECHO-BACK — confirm this is your section: "
             f"{project.meta.name}")
    ax.set_xlabel('Distance (m)', fontsize=10)
    ax.set_ylabel('Elevation (m)', fontsize=10)
    ax.set_title(title, fontsize=11.5, fontweight='bold')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=9)
    ax.legend(loc='lower right', fontsize=7, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return EchoBack(image_path=str(path), vertex_table=table)


__all__ = ["echo_back", "vertex_table", "EchoBack"]

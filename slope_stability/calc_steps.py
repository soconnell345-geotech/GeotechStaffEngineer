"""
Calculation package steps for slope stability analysis.

Provides input summaries, step-by-step equation output, and SLIDE-style
figures for Mathcad-style calc package generation.

The `analysis` parameter for this module should be a dict with:
    - "geom": SlopeGeometry
    - "xc", "yc", "radius": circle parameters
    - "method": analysis method name
    - "n_slices": number of slices

References:
    Duncan, Wright & Brandon (2014) — Soil Strength and Slope Stability
    Fellenius, W. (1927) — Method of Slices
    Bishop, A.W. (1955) — Simplified method
    Spencer, E. (1967) — Complete equilibrium method
"""

import math
from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "Slope Stability Analysis"

REFERENCES = [
    'Duncan, J.M., Wright, S.G. & Brandon, T.L. (2014). Soil Strength '
    'and Slope Stability, 2nd Ed. Wiley.',
    'Fellenius, W. (1927). "Erdstatische Berechnungen." W. Ernst & Sohn.',
    'Bishop, A.W. (1955). "The Use of the Slip Circle in the Stability '
    'Analysis of Slopes." Geotechnique, 5(1), 7-17.',
    'Spencer, E. (1967). "A Method of Analysis of the Stability of '
    'Embankments Assuming Parallel Interslice Forces." Geotechnique, 17(1), 11-26.',
    'Abramson, L.W. et al. (2002). Slope Stability and Stabilization Methods, '
    '2nd Ed. Wiley.',
]


def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for slope stability calc package.

    Parameters
    ----------
    result : SlopeStabilityResult
        Computed results.
    analysis : dict
        Must contain "geom" (SlopeGeometry) plus analysis parameters.

    Returns
    -------
    list of InputItem
    """
    geom = analysis["geom"]

    items = [
        InputItem("Method", "Analysis method", result.method, ""),
        InputItem("x_c", "Circle center x", f"{result.xc:.2f}", "m"),
        InputItem("y_c", "Circle center y (elev.)", f"{result.yc:.2f}", "m"),
        InputItem("R", "Circle radius", f"{result.radius:.2f}", "m"),
        InputItem("N", "Number of slices", result.n_slices, ""),
        InputItem("FOS_req", "Required FOS", result.FOS_required, ""),
    ]

    if result.has_seismic:
        items.append(InputItem("k_h", "Seismic coefficient", result.kh, ""))

    if geom.surcharge > 0:
        items.append(InputItem("q_s", "Surcharge", geom.surcharge, "kPa"))

    return items


def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections.

    Parameters
    ----------
    result : SlopeStabilityResult
    analysis : dict

    Returns
    -------
    list of CalcSection
    """
    geom = analysis["geom"]
    sections = []

    # ── Slope Geometry Summary ──────────────────────────────────
    geom_items = []

    # Surface profile table
    surface_rows = [[f"{x:.1f}", f"{z:.1f}"] for x, z in geom.surface_points]
    geom_items.append(TableData(
        title="Ground Surface Profile",
        headers=["x (m)", "Elevation (m)"],
        rows=surface_rows,
    ))

    # Soil layers table
    layer_rows = []
    for layer in geom.soil_layers:
        c, phi = layer.shear_strength_params
        layer_rows.append([
            layer.name,
            f"{layer.top_elevation:.1f}",
            f"{layer.bottom_elevation:.1f}",
            f"{layer.gamma:.1f}",
            f"{c:.1f}",
            f"{phi:.1f}",
            layer.analysis_mode,
        ])
    geom_items.append(TableData(
        title="Soil Layer Properties",
        headers=["Layer", "Top (m)", "Bottom (m)", "\u03b3 (kN/m\u00b3)",
                 "c/c_u (kPa)", "\u03c6 (deg)", "Mode"],
        rows=layer_rows,
    ))

    if geom.gwt_points is not None:
        gwt_rows = [[f"{x:.1f}", f"{z:.1f}"] for x, z in geom.gwt_points]
        geom_items.append(TableData(
            title="Groundwater Table",
            headers=["x (m)", "GWT Elev. (m)"],
            rows=gwt_rows,
        ))

    sections.append(CalcSection(title="Slope Geometry & Soil Properties", items=geom_items))

    # ── Slip Circle ─────────────────────────────────────────────
    circle_items = []

    circle_items.append(CalcStep(
        title="Slip Circle Definition",
        equation="Circle: (x - x_c)\u00b2 + (z - y_c)\u00b2 = R\u00b2",
        substitution=(
            f"(x - {result.xc:.2f})\u00b2 + (z - {result.yc:.2f})\u00b2 "
            f"= {result.radius:.2f}\u00b2"
        ),
        result_name="R",
        result_value=f"{result.radius:.2f}",
        result_unit="m",
    ))

    circle_items.append(CalcStep(
        title="Slip Surface Intersection",
        equation="Entry and exit points where circle intersects ground surface",
        substitution="",
        result_name="x_entry, x_exit",
        result_value=f"{result.x_entry:.2f} m, {result.x_exit:.2f} m",
    ))

    sections.append(CalcSection(title="Slip Circle", items=circle_items))

    # ── Analysis Method ─────────────────────────────────────────
    method_items = []

    if result.method == "Fellenius":
        method_items.append(CalcStep(
            title="Ordinary Method of Slices (Fellenius)",
            equation=(
                "FOS = \u03a3[c_i \u00d7 b_i / cos(\u03b1_i) + "
                "(W_i \u00d7 cos(\u03b1_i) - u_i \u00d7 b_i / cos(\u03b1_i)) "
                "\u00d7 tan(\u03c6_i)] / \u03a3[W_i \u00d7 sin(\u03b1_i)]"
            ),
            substitution="",
            result_name="FOS",
            result_value=f"{result.FOS:.3f}",
            reference="Fellenius (1927)",
            notes="Satisfies moment equilibrium only. Does not satisfy force equilibrium.",
        ))
    elif result.method == "Bishop":
        method_items.append(CalcStep(
            title="Bishop's Simplified Method",
            equation=(
                "FOS = \u03a3[(c_i\u00b7b_i + (W_i - u_i\u00b7b_i)\u00b7tan(\u03c6_i)) "
                "/ m_\u03b1_i] / \u03a3[W_i \u00d7 sin(\u03b1_i)]"
            ),
            substitution="m_\u03b1 = cos(\u03b1) + sin(\u03b1)\u00b7tan(\u03c6)/FOS",
            result_name="FOS",
            result_value=f"{result.FOS:.3f}",
            reference="Bishop (1955)",
            notes="Iterative solution — satisfies moment equilibrium. "
                  "Assumes zero interslice shear forces.",
        ))
    elif result.method == "Spencer":
        method_items.append(CalcStep(
            title="Spencer's Method",
            equation=(
                "FOS satisfies both force and moment equilibrium.\n"
                "Interslice forces inclined at angle \u03b8 from horizontal."
            ),
            substitution="",
            result_name="FOS",
            result_value=f"{result.FOS:.3f}",
            reference="Spencer (1967)",
        ))
        if result.theta_spencer is not None:
            method_items.append(CalcStep(
                title="Interslice Force Angle",
                equation="\u03b8 = angle of resultant interslice force from horizontal",
                substitution="",
                result_name="\u03b8",
                result_value=f"{result.theta_spencer:.2f}",
                result_unit="deg",
            ))

    sections.append(CalcSection(title="Analysis Method & Factor of Safety", items=method_items))

    # ── Method Comparison ──────────────────────────────────────
    if result.FOS_fellenius is not None or result.FOS_bishop is not None:
        compare_rows = []
        if result.FOS_fellenius is not None:
            compare_rows.append(["Fellenius", f"{result.FOS_fellenius:.3f}"])
        if result.FOS_bishop is not None:
            compare_rows.append(["Bishop", f"{result.FOS_bishop:.3f}"])
        compare_rows.append([result.method, f"{result.FOS:.3f}"])

        sections.append(CalcSection(
            title="Method Comparison",
            items=[TableData(
                title="FOS by Method",
                headers=["Method", "FOS"],
                rows=compare_rows,
                notes="All methods use the same slip circle and slice geometry.",
            )]
        ))

    # ── Slice Data Table ────────────────────────────────────────
    if result.slice_data:
        # Show a representative subset (every Nth slice)
        n = len(result.slice_data)
        if n > 12:
            step = max(1, n // 10)
            indices = list(range(0, n, step))
            if (n - 1) not in indices:
                indices.append(n - 1)
        else:
            indices = list(range(n))

        slice_rows = []
        for i in indices:
            s = result.slice_data[i]
            slice_rows.append([
                str(i + 1),
                f"{s.x_mid:.2f}",
                f"{s.width:.3f}",
                f"{s.weight:.1f}",
                f"{s.alpha_deg:.1f}",
                f"{s.c:.1f}",
                f"{s.phi:.1f}",
                f"{s.pore_pressure:.1f}",
            ])

        slice_table = TableData(
            title="Representative Slice Data",
            headers=["#", "x_mid (m)", "Width (m)", "Weight (kN/m)",
                     "\u03b1 (deg)", "c (kPa)", "\u03c6 (deg)", "u (kPa)"],
            rows=slice_rows,
            notes=f"Showing {len(indices)} of {n} slices.",
        )
        sections.append(CalcSection(title="Slice Data", items=[slice_table]))

    # ── Stability Check ─────────────────────────────────────────
    check_items = [
        CheckItem(
            description="Slope stability adequacy",
            demand=result.FOS_required,
            demand_label="FOS_required",
            capacity=result.FOS,
            capacity_label="FOS_computed",
            unit="",
            passes=result.is_stable,
        ),
    ]
    sections.append(CalcSection(title="Stability Check", items=check_items))

    return sections


def get_figures(result, analysis) -> List[FigureData]:
    """Generate SLIDE-style figures for slope stability calc package.

    Parameters
    ----------
    result : SlopeStabilityResult
    analysis : dict
        Must contain "geom" (SlopeGeometry).

    Returns
    -------
    list of FigureData
    """
    geom = analysis["geom"]
    figures = []

    try:
        fig = _plot_slip_circle(result, geom)
        b64 = figure_to_base64(fig, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        figures.append(FigureData(
            title="Slope Cross-Section with Slip Circle",
            image_base64=b64,
            caption=(
                f"Figure 1: Slope cross-section with critical slip circle. "
                f"FOS = {result.FOS:.3f} ({result.method} method). "
                f"Entry x = {result.x_entry:.1f} m, Exit x = {result.x_exit:.1f} m."
            ),
            width_percent=90,
        ))
    except ImportError:
        pass

    # Slice force diagram if data available
    if result.slice_data:
        try:
            fig2 = _plot_slice_forces(result)
            b64_2 = figure_to_base64(fig2, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig2)
            figures.append(FigureData(
                title="Slice Weight and Base Angle Distribution",
                image_base64=b64_2,
                caption=(
                    f"Figure 2: Slice weight and base inclination angle "
                    f"along the slip surface ({result.n_slices} slices)."
                ),
                width_percent=80,
            ))
        except ImportError:
            pass

    return figures


def _plot_slip_circle(result, geom):
    """Create a SLIDE-style cross-section plot with slip circle."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette for soil layers
    layer_colors = ['#f5e6c8', '#d4a574', '#a0c4a0', '#c8b4a0',
                    '#e6d4a0', '#b8c8d0', '#d4c4b0', '#c0d4c0']

    # Draw soil layers as filled regions
    xs_surface = [p[0] for p in geom.surface_points]
    zs_surface = [p[1] for p in geom.surface_points]
    x_min = min(xs_surface)
    x_max = max(xs_surface)

    for i, layer in enumerate(geom.soil_layers):
        color = layer_colors[i % len(layer_colors)]
        # Fill between layer top/bottom, clipped to surface
        x_fill = np.linspace(x_min, x_max, 200)
        z_top_fill = np.array([
            min(geom.ground_elevation_at(x), layer.top_elevation) for x in x_fill
        ])
        z_bot_fill = np.full_like(x_fill, layer.bottom_elevation)
        # Only show where top > bottom
        mask = z_top_fill > z_bot_fill
        if np.any(mask):
            ax.fill_between(x_fill, z_bot_fill, z_top_fill, where=mask,
                           color=color, alpha=0.6, label=layer.name,
                           edgecolor='#666', linewidth=0.5)

    # Ground surface
    ax.plot(xs_surface, zs_surface, 'k-', linewidth=2, label='Ground surface')

    # GWT line
    if geom.gwt_points is not None:
        gwt_x = [p[0] for p in geom.gwt_points]
        gwt_z = [p[1] for p in geom.gwt_points]
        ax.plot(gwt_x, gwt_z, 'b--', linewidth=1.5, label='GWT', alpha=0.7)

    # Slip circle arc
    theta_range = np.linspace(0, 2 * np.pi, 360)
    circle_x = result.xc + result.radius * np.cos(theta_range)
    circle_z = result.yc + result.radius * np.sin(theta_range)

    # Only show the part below the ground surface (the actual slip surface)
    # Clip to the entry-exit x range with a margin
    x_entry = result.x_entry
    x_exit = result.x_exit
    mask = (circle_x >= x_entry - 0.5) & (circle_x <= x_exit + 0.5)
    # Also only show the lower arc (below center)
    mask &= (circle_z <= result.yc)

    ax.plot(circle_x[mask], circle_z[mask], 'r-', linewidth=2.5,
            label=f'Slip circle (FOS={result.FOS:.3f})')

    # Circle center
    ax.plot(result.xc, result.yc, 'r+', markersize=12, markeredgewidth=2)
    ax.annotate(f'Center\n({result.xc:.1f}, {result.yc:.1f})',
                xy=(result.xc, result.yc), fontsize=7,
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Entry and exit points
    z_entry = geom.ground_elevation_at(x_entry)
    z_exit = geom.ground_elevation_at(x_exit)
    ax.plot(x_entry, z_entry, 'rv', markersize=8)
    ax.plot(x_exit, z_exit, 'r^', markersize=8)
    ax.annotate(f'Entry', xy=(x_entry, z_entry), fontsize=7,
                xytext=(-15, -15), textcoords='offset points')
    ax.annotate(f'Exit', xy=(x_exit, z_exit), fontsize=7,
                xytext=(5, -15), textcoords='offset points')

    # Slice lines (if slice data available)
    if result.slice_data:
        for s in result.slice_data:
            ax.plot([s.x_mid, s.x_mid], [s.z_base, s.z_top],
                    'k-', linewidth=0.3, alpha=0.4)

    # FOS annotation box
    status = "STABLE" if result.is_stable else "UNSTABLE"
    status_color = '#16a34a' if result.is_stable else '#dc2626'
    ax.text(0.02, 0.98,
            f"FOS = {result.FOS:.3f} ({result.method})\n[{status}]",
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', color=status_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor=status_color, alpha=0.9))

    ax.set_xlabel('Distance (m)', fontsize=10)
    ax.set_ylabel('Elevation (m)', fontsize=10)
    ax.set_title('Slope Stability Analysis — Cross-Section', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def _plot_slice_forces(result):
    """Create a dual-axis plot of slice weights and base angles."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    slices = result.slice_data
    x_mids = [s.x_mid for s in slices]
    weights = [s.weight for s in slices]
    alphas = [s.alpha_deg for s in slices]

    fig, ax1 = plt.subplots(figsize=(8, 4))

    color1 = '#2563eb'
    ax1.bar(x_mids, weights, width=[s.width * 0.8 for s in slices],
            color=color1, alpha=0.6, label='Slice weight')
    ax1.set_xlabel('x position (m)', fontsize=10)
    ax1.set_ylabel('Slice Weight (kN/m)', fontsize=10, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = '#dc2626'
    ax2.plot(x_mids, alphas, 'o-', color=color2, markersize=3,
             linewidth=1.5, label='Base angle \u03b1')
    ax2.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax2.set_ylabel('Base Angle \u03b1 (deg)', fontsize=10, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('Slice Weight and Base Angle Distribution', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    plt.tight_layout()
    return fig

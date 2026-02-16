"""
Calculation package steps for pile group analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation.

References:
    CPGA User's Guide (USACE ITL-89-4, Hartman et al., 1989)
    USACE EM 1110-2-2906: Design of Pile Foundations
    FHWA GEC-12 (FHWA-NHI-16-009), Chapter 9 — Group Effects
    Converse, F.J. (1962). "Foundations Subjected to Dynamic Forces."
"""

import math
from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "Pile Group Analysis"

REFERENCES = [
    'FHWA GEC-12 (FHWA-NHI-16-009): Design and Construction of Driven '
    'Pile Foundations, Volumes I & II, Chapter 9 — Group Effects.',
    'USACE EM 1110-2-2906: Design of Pile Foundations.',
    'CPGA User\'s Guide (USACE ITL-89-4, Hartman et al., 1989). '
    'Computer Program for the Analysis of Pile Groups with Rigid Caps.',
    'Converse, F.J. (1962). "Foundations Subjected to Dynamic Forces." '
    'In Foundation Engineering, G.A. Leonards (ed.), McGraw-Hill.',
    'Reese, L.C. & Van Impe, W.F. (2001). Single Piles and Pile Groups '
    'Under Lateral Loading. Balkema.',
    'Brown, D.A., Morrison, C. & Reese, L.C. (1988). "Lateral Load '
    'Behavior of Pile Group in Sand." JGED, ASCE, Vol. 114, No. 11.',
]


# ---------------------------------------------------------------------------
# Attribute access helpers
# ---------------------------------------------------------------------------

def _get(obj, key, default=None):
    """Get attribute or dict key from analysis/result object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_piles(analysis):
    """Extract pile list from analysis object."""
    return _get(analysis, 'piles', [])


def _get_load(analysis):
    """Extract GroupLoad from analysis object."""
    return _get(analysis, 'load', None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for pile group calc package.

    Parameters
    ----------
    result : PileGroupResult
        Computed results.
    analysis : object
        Analysis object or dict holding piles, load, and optional metadata
        (method, pile_diameter, spacing, n_rows, n_cols, efficiency, etc.).

    Returns
    -------
    list of InputItem
    """
    items = []

    # -- Group layout --
    n_piles = _get(result, 'n_piles', 0)
    n_rows = _get(analysis, 'n_rows', None)
    n_cols = _get(analysis, 'n_cols', None)

    if n_rows and n_cols:
        items.append(InputItem(
            "Layout", "Group configuration",
            f"{n_rows} rows x {n_cols} columns ({n_rows * n_cols} piles)", "",
        ))
    else:
        items.append(InputItem("n", "Number of piles", n_piles, ""))

    # Spacing
    spacing_x = _get(analysis, 'spacing_x', None)
    spacing_y = _get(analysis, 'spacing_y', None)
    spacing = _get(analysis, 'spacing', None)

    if spacing_x is not None and spacing_y is not None:
        if abs(spacing_x - spacing_y) < 0.001:
            items.append(InputItem("s", "Pile spacing (c/c)", f"{spacing_x:.2f}", "m"))
        else:
            items.append(InputItem("s_x", "Spacing in X-direction", f"{spacing_x:.2f}", "m"))
            items.append(InputItem("s_y", "Spacing in Y-direction", f"{spacing_y:.2f}", "m"))
    elif spacing is not None:
        items.append(InputItem("s", "Pile spacing (c/c)", f"{spacing:.2f}", "m"))

    # Pile properties
    pile_diameter = _get(analysis, 'pile_diameter', None)
    pile_length = _get(analysis, 'pile_length', None)

    if pile_diameter is not None:
        items.append(InputItem("D", "Pile diameter/width", f"{pile_diameter:.3f}", "m"))
    if pile_length is not None:
        items.append(InputItem("L", "Pile embedment length", f"{pile_length:.2f}", "m"))

    # Pile stiffness (from first pile or analysis-level)
    piles = _get_piles(analysis)
    if piles:
        p0 = piles[0]
        ka = _get(p0, 'axial_stiffness', None)
        kl = _get(p0, 'lateral_stiffness', None)
        if ka is not None:
            items.append(InputItem(
                "k_a", "Axial stiffness (per pile)", f"{ka:,.0f}", "kN/m",
            ))
        if kl is not None:
            items.append(InputItem(
                "k_l", "Lateral stiffness (per pile)", f"{kl:,.0f}", "kN/m",
            ))

        # Capacity
        cap_comp = _get(p0, 'axial_capacity_compression', None)
        cap_tens = _get(p0, 'axial_capacity_tension', None)
        if cap_comp is not None:
            items.append(InputItem(
                "Q_comp", "Compression capacity (per pile)",
                f"{cap_comp:,.1f}", "kN",
            ))
        if cap_tens is not None:
            items.append(InputItem(
                "Q_tens", "Tension capacity (per pile)",
                f"{cap_tens:,.1f}", "kN",
            ))

        # Batter
        battered = [p for p in piles if not _get(p, 'is_vertical', True)]
        if battered:
            items.append(InputItem(
                "Batter", "Battered piles in group",
                f"{len(battered)} of {len(piles)} piles battered", "",
            ))

    # -- Applied loading --
    load = _get_load(analysis)
    if load is not None:
        Vx = _get(load, 'Vx', 0.0)
        Vy = _get(load, 'Vy', 0.0)
        Vz = _get(load, 'Vz', 0.0)
        Mx = _get(load, 'Mx', 0.0)
        My = _get(load, 'My', 0.0)
        Mz = _get(load, 'Mz', 0.0)

        items.append(InputItem("V_z", "Vertical load (compression +)", f"{Vz:,.1f}", "kN"))
        if abs(Vx) > 0.01:
            items.append(InputItem("V_x", "Lateral load in X", f"{Vx:,.1f}", "kN"))
        if abs(Vy) > 0.01:
            items.append(InputItem("V_y", "Lateral load in Y", f"{Vy:,.1f}", "kN"))
        if abs(Mx) > 0.01:
            items.append(InputItem("M_x", "Moment about X-axis", f"{Mx:,.1f}", "kN-m"))
        if abs(My) > 0.01:
            items.append(InputItem("M_y", "Moment about Y-axis", f"{My:,.1f}", "kN-m"))
        if abs(Mz) > 0.01:
            items.append(InputItem("M_z", "Torsion about Z-axis", f"{Mz:,.1f}", "kN-m"))

    # Analysis method
    method = _get(analysis, 'method', None)
    if method:
        method_display = {
            'simple': 'Simplified Elastic (P = V/n \u00b1 M\u00b7x/\u03a3x\u00b2)',
            '6dof': 'General 6-DOF Rigid Cap (CPGA)',
        }.get(method, method)
        items.append(InputItem("Method", "Analysis method", method_display, ""))

    return items


def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections.

    Parameters
    ----------
    result : PileGroupResult
        Computed results.
    analysis : object
        Analysis object or dict with piles, load, and optional metadata.

    Returns
    -------
    list of CalcSection
    """
    sections = []
    piles = _get_piles(analysis)
    load = _get_load(analysis)
    pile_forces = _get(result, 'pile_forces', [])
    n_piles = _get(result, 'n_piles', len(piles))

    # =================================================================
    # Section 1: Pile Layout
    # =================================================================
    layout_items = []

    if piles:
        layout_rows = []
        for i, p in enumerate(piles):
            label = _get(p, 'label', '') or f"P{i+1}"
            x = _get(p, 'x', 0.0)
            y = _get(p, 'y', 0.0)
            bx = _get(p, 'batter_x', 0.0)
            by = _get(p, 'batter_y', 0.0)
            vert = "Yes" if _get(p, 'is_vertical', True) else f"No ({bx:.1f}\u00b0, {by:.1f}\u00b0)"
            layout_rows.append([
                label,
                f"{x:+.3f}",
                f"{y:+.3f}",
                vert,
            ])

        layout_items.append(TableData(
            title="Pile Coordinates (relative to cap centroid)",
            headers=["Pile", "X (m)", "Y (m)", "Vertical"],
            rows=layout_rows,
        ))

    # Geometric sums for simplified method
    if piles:
        xs = [_get(p, 'x', 0.0) for p in piles]
        ys = [_get(p, 'y', 0.0) for p in piles]
        sum_x2 = sum(x**2 for x in xs)
        sum_y2 = sum(y**2 for y in ys)

        layout_items.append(CalcStep(
            title="Sum of Squared Distances",
            equation="\u03a3x\u00b2 = sum of x_i\u00b2 for all piles\n"
                     "\u03a3y\u00b2 = sum of y_i\u00b2 for all piles",
            substitution=_sum_squares_substitution(xs, ys),
            result_name="\u03a3x\u00b2, \u03a3y\u00b2",
            result_value=f"\u03a3x\u00b2 = {sum_x2:.3f} m\u00b2, "
                         f"\u03a3y\u00b2 = {sum_y2:.3f} m\u00b2",
            reference="USACE EM 1110-2-2906, Ch. 4",
            notes="Used in simplified elastic load distribution.",
        ))

    sections.append(CalcSection(title="Pile Layout", items=layout_items))

    # =================================================================
    # Section 2: Group Efficiency (Converse-Labarre)
    # =================================================================
    efficiency_items = []

    n_rows = _get(analysis, 'n_rows', None)
    n_cols = _get(analysis, 'n_cols', None)
    pile_diameter = _get(analysis, 'pile_diameter', None)
    spacing = _get(analysis, 'spacing', None) or _get(analysis, 'spacing_x', None)
    efficiency = _get(analysis, 'efficiency', None)

    if n_rows and n_cols and pile_diameter and spacing:
        theta = math.degrees(math.atan(pile_diameter / spacing))
        m, n = n_rows, n_cols
        Eg = 1.0 - theta / (90.0 * m * n) * (n * (m - 1) + m * (n - 1))
        Eg = max(Eg, 0.0)

        efficiency_items.append(CalcStep(
            title="Converse-Labarre Group Efficiency",
            equation="E_g = 1 - \u03b8 / (90\u00b7m\u00b7n) \u00d7 "
                     "[n\u00b7(m-1) + m\u00b7(n-1)]",
            substitution=(
                f"\u03b8 = arctan(D/s) = arctan({pile_diameter:.3f}/{spacing:.3f}) "
                f"= {theta:.2f}\u00b0\n"
                f"E_g = 1 - {theta:.2f}/(90\u00d7{m}\u00d7{n}) \u00d7 "
                f"[{n}\u00d7({m}-1) + {m}\u00d7({n}-1)]"
            ),
            result_name="E_g",
            result_value=f"{Eg:.3f}",
            result_unit="",
            reference="FHWA GEC-12, Eq. 9-1",
            notes=(
                f"D = {pile_diameter:.3f} m, s = {spacing:.3f} m, "
                f"s/D = {spacing/pile_diameter:.1f}. "
                "Group efficiency should typically be > 0.7 for driven piles."
            ),
        ))

        # Reduced group capacity
        Vz = _get(load, 'Vz', 0.0) if load else 0.0
        if Vz > 0:
            piles_list = _get_piles(analysis)
            cap_comp = _get(piles_list[0], 'axial_capacity_compression', None) if piles_list else None
            if cap_comp:
                Q_group_ind = n_piles * cap_comp
                Q_group_eff = Eg * Q_group_ind
                efficiency_items.append(CalcStep(
                    title="Reduced Group Capacity",
                    equation="Q_group = E_g \u00d7 n \u00d7 Q_individual",
                    substitution=(
                        f"Q_group = {Eg:.3f} \u00d7 {n_piles} \u00d7 "
                        f"{cap_comp:,.1f}"
                    ),
                    result_name="Q_group",
                    result_value=f"{Q_group_eff:,.1f}",
                    result_unit="kN",
                    reference="FHWA GEC-12, Section 9.4",
                ))

    elif efficiency is not None:
        # Efficiency provided directly
        efficiency_items.append(CalcStep(
            title="Group Efficiency (Provided)",
            equation="E_g (user-specified)",
            substitution="",
            result_name="E_g",
            result_value=f"{efficiency:.3f}",
            notes="Group efficiency factor applied to individual pile capacities.",
        ))
    else:
        efficiency_items.append(
            "Group efficiency not computed (Converse-Labarre parameters "
            "not provided: n_rows, n_cols, pile_diameter, spacing)."
        )

    sections.append(CalcSection(
        title="Group Efficiency", items=efficiency_items,
    ))

    # =================================================================
    # Section 3: Load Distribution
    # =================================================================
    load_items = []

    method = _get(analysis, 'method', None)
    if method == '6dof' or method == 'general':
        load_items.append(CalcStep(
            title="6-DOF Rigid Cap Stiffness Matrix",
            equation="[K_group]{U} = {F}",
            substitution=(
                "K_group assembled from per-pile axial and lateral stiffnesses "
                "transformed to global coordinates via direction cosines."
            ),
            result_name="Method",
            result_value="General 6-DOF (CPGA)",
            reference="CPGA User's Guide (ITL-89-4); EM 1110-2-2906, Ch. 4",
            notes=(
                "Solves for 6 cap DOFs (dx, dy, dz, rx, ry, rz), then "
                "back-calculates individual pile forces."
            ),
        ))
    else:
        # Simplified elastic
        Vz = _get(load, 'Vz', 0.0) if load else 0.0
        My = _get(load, 'My', 0.0) if load else 0.0
        Mx = _get(load, 'Mx', 0.0) if load else 0.0

        load_items.append(CalcStep(
            title="Simplified Elastic Load Distribution",
            equation="P_i = V_z/n \u00b1 M_y\u00b7x_i/\u03a3x\u00b2 "
                     "\u00b1 M_x\u00b7y_i/\u03a3y\u00b2",
            substitution=_simple_load_substitution(n_piles, Vz, My, Mx, piles),
            result_name="Method",
            result_value="Simplified Elastic",
            reference="USACE EM 1110-2-2906, Eq. 4-1",
            notes="Assumes vertical piles, rigid cap, and elastic behavior.",
        ))

    # Cap displacements
    cap_d = _get(result, 'cap_displacements', {})
    if cap_d:
        dx_mm = cap_d.get('dx', 0) * 1000
        dy_mm = cap_d.get('dy', 0) * 1000
        dz_mm = cap_d.get('dz', 0) * 1000
        rx_mrad = cap_d.get('rx', 0) * 1000
        ry_mrad = cap_d.get('ry', 0) * 1000
        rz_mrad = cap_d.get('rz', 0) * 1000

        disp_rows = [
            ["dx (lateral X)", f"{dx_mm:.3f}", "mm"],
            ["dy (lateral Y)", f"{dy_mm:.3f}", "mm"],
            ["dz (vertical)", f"{dz_mm:.3f}", "mm"],
            ["rx (rotation about X)", f"{rx_mrad:.5f}", "mrad"],
            ["ry (rotation about Y)", f"{ry_mrad:.5f}", "mrad"],
            ["rz (torsion about Z)", f"{rz_mrad:.5f}", "mrad"],
        ]
        load_items.append(TableData(
            title="Cap Displacements",
            headers=["DOF", "Value", "Unit"],
            rows=disp_rows,
        ))

    sections.append(CalcSection(
        title="Load Distribution (Rigid Cap)", items=load_items,
    ))

    # =================================================================
    # Section 4: Individual Pile Forces
    # =================================================================
    force_items = []

    if pile_forces:
        force_rows = []
        for pf in pile_forces:
            label = pf.get('label', '?')
            x_m = pf.get('x_m', 0.0)
            y_m = pf.get('y_m', 0.0)
            axial = pf.get('axial_kN', 0.0)
            util = pf.get('utilization', 0.0)
            status = "Comp" if axial >= 0 else "Tension"
            force_rows.append([
                label,
                f"{x_m:+.3f}",
                f"{y_m:+.3f}",
                f"{axial:+,.1f}",
                status,
                f"{util:.3f}" if util else "N/A",
            ])

        force_items.append(TableData(
            title="Per-Pile Axial Forces",
            headers=["Pile", "X (m)", "Y (m)", "Axial (kN)",
                     "Type", "Utilization"],
            rows=force_rows,
        ))

    # Summary results
    max_comp = _get(result, 'max_compression', 0.0)
    max_tens = _get(result, 'max_tension', 0.0)
    max_util = _get(result, 'max_utilization', 0.0)

    force_items.append(CalcStep(
        title="Maximum Pile Forces",
        equation="Max compression and tension from load distribution",
        substitution="",
        result_name="P_max",
        result_value=(
            f"Compression = {max_comp:,.1f} kN, "
            f"Tension = {max_tens:,.1f} kN"
        ),
        result_unit="",
    ))

    force_items.append(CalcStep(
        title="Maximum Utilization Ratio",
        equation="Utilization = P_demand / P_capacity",
        substitution="",
        result_name="UR_max",
        result_value=f"{max_util:.3f}",
        reference="FHWA GEC-12, Section 9.5",
        notes="Utilization < 1.0 required for adequacy." if max_util > 0 else
              "Capacity not specified; utilization ratio not computed.",
    ))

    # Capacity check
    if max_util > 0:
        force_items.append(CheckItem(
            description="Pile group adequacy (max utilization \u2264 1.0)",
            demand=max_util,
            demand_label="UR_max",
            capacity=1.0,
            capacity_label="UR_limit",
            unit="",
            passes=max_util <= 1.0,
        ))

    sections.append(CalcSection(
        title="Individual Pile Forces & Utilization", items=force_items,
    ))

    return sections


def get_figures(result, analysis) -> List[FigureData]:
    """Generate figures for the pile group calc package.

    Parameters
    ----------
    result : PileGroupResult
        Computed results.
    analysis : object
        Analysis object or dict with piles, load, and optional metadata.

    Returns
    -------
    list of FigureData
    """
    figures = []
    pile_forces = _get(result, 'pile_forces', [])

    if not pile_forces:
        return figures

    # Figure 1: Plan view of pile layout with load distribution
    try:
        fig1 = _plot_plan_view(pile_forces, result, analysis)
        b64 = figure_to_base64(fig1, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig1)
        figures.append(FigureData(
            title="Pile Group Plan View",
            image_base64=b64,
            caption=(
                f"Figure 1: Plan view of pile group layout showing axial "
                f"force distribution ({_get(result, 'n_piles', len(pile_forces))} piles). "
                f"Circle size and color indicate load magnitude. "
                f"Positive = compression, negative = tension."
            ),
            width_percent=80,
        ))
    except (ImportError, Exception):
        pass

    # Figure 2: Bar chart of individual pile loads
    try:
        fig2 = _plot_pile_load_bar(pile_forces, result)
        b64_2 = figure_to_base64(fig2, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig2)
        figures.append(FigureData(
            title="Individual Pile Axial Forces",
            image_base64=b64_2,
            caption=(
                f"Figure 2: Axial force per pile. "
                f"Max compression = {_get(result, 'max_compression', 0):,.1f} kN, "
                f"max tension = {_get(result, 'max_tension', 0):,.1f} kN."
            ),
            width_percent=80,
        ))
    except (ImportError, Exception):
        pass

    return figures


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sum_squares_substitution(xs, ys):
    """Build substitution string for sum-of-squares calculation."""
    if len(xs) <= 8:
        x_terms = " + ".join(f"({x:.3f})\u00b2" for x in xs)
        y_terms = " + ".join(f"({y:.3f})\u00b2" for y in ys)
    else:
        x_terms = " + ".join(f"({x:.3f})\u00b2" for x in xs[:4]) + " + ..."
        y_terms = " + ".join(f"({y:.3f})\u00b2" for y in ys[:4]) + " + ..."

    return f"\u03a3x\u00b2 = {x_terms}\n\u03a3y\u00b2 = {y_terms}"


def _simple_load_substitution(n_piles, Vz, My, Mx, piles):
    """Build substitution string for simplified elastic formula."""
    parts = [f"P_i = {Vz:,.1f}/{n_piles}"]

    if piles:
        xs = [_get(p, 'x', 0.0) for p in piles]
        ys = [_get(p, 'y', 0.0) for p in piles]
        sum_x2 = sum(x**2 for x in xs)
        sum_y2 = sum(y**2 for y in ys)

        if abs(My) > 0.01 and sum_x2 > 0:
            parts.append(f"\u00b1 {My:,.1f}\u00b7x_i/{sum_x2:.3f}")
        if abs(Mx) > 0.01 and sum_y2 > 0:
            parts.append(f"\u00b1 {Mx:,.1f}\u00b7y_i/{sum_y2:.3f}")

    return " ".join(parts)


def _plot_plan_view(pile_forces, result, analysis):
    """Create plan view of pile layout with colored/sized markers
    showing axial force distribution."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 7))

    xs = [pf['x_m'] for pf in pile_forces]
    ys = [pf['y_m'] for pf in pile_forces]
    forces = [pf['axial_kN'] for pf in pile_forces]
    labels = [pf['label'] for pf in pile_forces]

    forces_arr = np.array(forces)
    abs_max = max(abs(forces_arr.max()), abs(forces_arr.min()), 1.0)

    # Color mapping: blue = compression, red = tension
    colors = []
    for f in forces:
        if f >= 0:
            # Compression: shades of blue
            intensity = min(abs(f) / abs_max, 1.0)
            colors.append((0.15, 0.39, 0.92, 0.3 + 0.7 * intensity))
        else:
            # Tension: shades of red
            intensity = min(abs(f) / abs_max, 1.0)
            colors.append((0.86, 0.20, 0.16, 0.3 + 0.7 * intensity))

    # Marker sizes proportional to load magnitude
    min_size = 200
    max_size = 1200
    abs_forces = np.abs(forces_arr)
    if abs_max > 0:
        sizes = min_size + (max_size - min_size) * abs_forces / abs_max
    else:
        sizes = np.full(len(forces), min_size + (max_size - min_size) / 2)

    # Plot piles
    scatter = ax.scatter(xs, ys, s=sizes, c=colors, edgecolors='#333333',
                         linewidths=1.5, zorder=5)

    # Labels with force values
    x_range = max(xs) - min(xs) if len(xs) > 1 else 1.0
    y_range = max(ys) - min(ys) if len(ys) > 1 else 1.0
    offset_y = max(y_range, x_range) * 0.06

    for x, y, label, force in zip(xs, ys, labels, forces):
        ax.annotate(
            f"{label}\n{force:+,.1f} kN",
            (x, y), textcoords="offset points", xytext=(0, -offset_y * 30),
            ha='center', va='top', fontsize=7.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#cccccc', alpha=0.85),
        )

    # Add crosshairs at centroid
    ax.axhline(y=0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)

    # Legend
    import matplotlib.patches as mpatches
    comp_patch = mpatches.Patch(color=(0.15, 0.39, 0.92, 0.8), label='Compression (+)')
    tens_patch = mpatches.Patch(color=(0.86, 0.20, 0.16, 0.8), label='Tension (\u2212)')
    ax.legend(handles=[comp_patch, tens_patch], loc='upper right', fontsize=9)

    # Axis setup
    margin = max(x_range, y_range, 1.0) * 0.4
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_title(
        f'Pile Group Plan View  ({_get(result, "n_piles", len(pile_forces))} piles)',
        fontsize=11, fontweight='bold',
    )
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig


def _plot_pile_load_bar(pile_forces, result):
    """Create a bar chart of individual pile axial forces."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [pf['label'] for pf in pile_forces]
    forces = [pf['axial_kN'] for pf in pile_forces]
    forces_arr = np.array(forces)

    # Color by compression (blue) vs tension (red)
    colors = ['#2563eb' if f >= 0 else '#dc2626' for f in forces]

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.6), 5))

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, forces, color=colors, edgecolor='#333333',
                  linewidth=0.8, width=0.65)

    # Value labels on bars
    for bar, val in zip(bars, forces):
        ypos = bar.get_height()
        va = 'bottom' if val >= 0 else 'top'
        offset = 2 if val >= 0 else -2
        ax.text(bar.get_x() + bar.get_width() / 2, ypos + offset,
                f'{val:+,.1f}', ha='center', va=va, fontsize=7.5,
                fontweight='bold')

    # Zero line
    ax.axhline(y=0, color='#333333', linewidth=0.8)

    # Utilization markers on secondary axis if available
    utils = [pf.get('utilization', 0) for pf in pile_forces]
    if any(u > 0 for u in utils):
        ax2 = ax.twinx()
        ax2.plot(x_pos, utils, 'ko--', markersize=5, linewidth=1.2,
                 alpha=0.7, label='Utilization')
        ax2.axhline(y=1.0, color='#dc2626', linestyle=':', linewidth=1.0,
                     alpha=0.5, label='UR = 1.0')
        ax2.set_ylabel('Utilization Ratio', fontsize=10, color='#555555')
        ax2.set_ylim(0, max(max(utils) * 1.3, 1.2))
        ax2.tick_params(labelsize=8, colors='#555555')
        ax2.legend(loc='upper right', fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45 if len(labels) > 6 else 0,
                       ha='right' if len(labels) > 6 else 'center',
                       fontsize=8)
    ax.set_xlabel('Pile', fontsize=10)
    ax.set_ylabel('Axial Force (kN)', fontsize=10)

    max_comp = _get(result, 'max_compression', 0)
    max_tens = _get(result, 'max_tension', 0)
    ax.set_title(
        f'Individual Pile Forces  '
        f'(Max comp = {max_comp:,.1f} kN, Max tens = {max_tens:,.1f} kN)',
        fontsize=10, fontweight='bold',
    )
    ax.grid(True, axis='y', alpha=0.2)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig

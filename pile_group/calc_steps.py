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
    # One spacing for Converse-Labarre: the rows' and columns' spacings as
    # the layout took them (a single 'spacing' for a direction not given),
    # and where they differ the smaller, where the piles are closer and
    # E_g is lower
    s_one = _get(analysis, 'spacing', None)
    s_x = _get(analysis, 'spacing_x', None)
    s_y = _get(analysis, 'spacing_y', None)
    s_x = s_one if s_x is None else s_x
    s_y = s_one if s_y is None else s_y
    s_xy = [s for s in (s_x, s_y) if s is not None]
    spacing = _get(analysis, 'efficiency_spacing', None)
    if spacing is None and s_xy:
        spacing = min(s_xy)
    spacing_note = ""
    if s_x is not None and s_y is not None and s_x != s_y:
        spacing_note = (f"Converse-Labarre takes one spacing; of s_x = "
                        f"{s_x:.3f} m and s_y = {s_y:.3f} m the smaller is "
                        "used (the closer piles, the lower E_g). ")
    # E_g as the run worked it out (the group_efficiency tool), else by
    # the same function, never by a formula of the steps' own (GM-141);
    # theta below only shows the substitution
    efficiency = _get(analysis, 'efficiency', None)

    if n_rows and n_cols and pile_diameter and spacing:
        if efficiency is None:
            from pile_group.group_efficiency import converse_labarre
            efficiency = converse_labarre(n_rows, n_cols, pile_diameter,
                                          spacing)
        theta = math.degrees(math.atan(pile_diameter / spacing))
        m, n = n_rows, n_cols
        Eg = efficiency

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
                spacing_note +
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

    else:
        missing = [k for k, v in (("n_rows", n_rows), ("n_cols", n_cols),
                                  ("pile_diameter", pile_diameter),
                                  ("spacing", spacing)) if not v]
        efficiency_items.append(
            "Group efficiency not computed (Converse-Labarre needs "
            + ", ".join(missing) + ")."
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
            equation="P_i = V_z/n + M_y\u00b7x_i/\u03a3x\u00b2 "
                     "- M_x\u00b7y_i/\u03a3y\u00b2",
            substitution=_simple_load_substitution(n_piles, Vz, My, Mx, piles),
            result_name="Method",
            result_value="Simplified Elastic",
            reference=("classical rigid-cap combined-stress formula (the EM "
                       "1110-2-2906 Ch. 4 equations are unnumbered and this "
                       "approximate method is described, not printed, there — "
                       "citation corrected 2026-07-19, wiki-verification)"),
            notes=(
                "Assumes vertical piles, rigid cap, and elastic behavior. "
                "Right-hand-rule convention (z up, V_z positive down): "
                "+M_y compresses the +x side; +M_x uplifts the +y side."
            ),
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
        drawn = ("drawn to scale" if _get(analysis, 'pile_diameter')
                 else "drawn to scale, the piles as symbols (no diameter "
                      "given)")
        given = [name for name, key in (("the cap outline", 'cap_outline'),
                                        ("the property line",
                                         'property_line'))
                 if _get(analysis, key)]
        if given:
            drawn += ", with " + " and ".join(given) + " as given"
        figures.append(FigureData(
            title="Pile Group Plan View",
            image_base64=b64,
            caption=(
                f"Figure 1: Plan of the pile group "
                f"({_get(result, 'n_piles', len(pile_forces))} piles), "
                f"{drawn}, with the axial force at each pile head "
                f"(positive = compression, negative = tension and "
                f"hatched); the darker the fill, the larger the force."
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
            parts.append(f"+ {My:,.1f}\u00b7x_i/{sum_x2:.3f}")
        if abs(Mx) > 0.01 and sum_y2 > 0:
            parts.append(f"- {Mx:,.1f}\u00b7y_i/{sum_y2:.3f}")

    return " ".join(parts)


#: The plan view, drawn at the width it prints at so that a size in
#: points here is the size printed (GM-186, GM-206): the margins round
#: the piles, in points. At the sides and above, room for the cap axes'
#: names; below, a pile's name and then the band holding the scale bar
#: and the north arrow.
_PLAN_WIDTH_IN = 5.0
_PLAN_MAX_HEIGHT_IN = 6.0
_PLAN_SIDE_PT = 30.0
_PLAN_ABOVE_PT = 18.0
_PLAN_NAME_PT = 26.0
_PLAN_BAND_PT = 56.0
#: the least width of the view, in points: the scale bar, its end names
#: and the north arrow side by side
_PLAN_MIN_VIEW_PT = 170.0
#: a pile drawn as a symbol, where no diameter is given: its radius
_PLAN_SYMBOL_PT = 5.0
#: the names of the piles (pile and force), in points
_PLAN_NAME_SIZE = 7.0
#: a pile in tension (and its key): hatched as well as red
_TENSION_HATCH = '////'


def _plot_plan_view(pile_forces, result, analysis):
    """The pile group in plan, drawn as a drawing (GM-206; the owner's
    rule for plans, 2026-10-15): no axes, grid or frame; a scale bar; a
    north arrow at ``north_deg`` (clockwise from up, as the plan the
    layout came from shows it), straight up when none is given, and the
    figure says so; the cap's x and y axes through its origin. A pile is
    a circle at its diameter when one is given, else a symbol not to
    scale; its fill is blue in compression and red and hatched in
    tension, darker the larger the axial force, and its name and force
    are written under it.
    Circles sized by the force read as the piles' size against a scale
    bar and covered the names (GM-206). Where the names cannot all be
    written clear of each other and of the piles, the forces are left to the
    force table and the bar chart, then the names too, and the key says
    so."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from geotech_common import plan_furniture as pf
    from geotech_common.plotting import halo, key_under_axes

    xs = [float(p['x_m']) for p in pile_forces]
    ys = [float(p['y_m']) for p in pile_forces]
    forces = [float(p['axial_kN']) for p in pile_forces]
    labels = [str(p['label']) for p in pile_forces]
    D = _get(analysis, 'pile_diameter')
    r = 0.5 * float(D) if D else 0.0
    north = _get(analysis, 'north_deg')
    # The cap outline and the property line, drawn only where given
    # (GM-216): never made up
    cap = _get(analysis, 'cap_outline') or []
    prop = _get(analysis, 'property_line') or []
    lines_x = [float(p[0]) for p in cap + prop]
    lines_y = [float(p[1]) for p in cap + prop]

    # The view: the piles, the cap's origin and the lines given, with the
    # margins in points, at the scale (m per point) that fits it to the
    # width it prints at, or to the height allowed
    x0 = min(min(xs + [0.0]) - r, min(lines_x, default=0.0))
    x1 = max(max(xs + [0.0]) + r, max(lines_x, default=0.0))
    y0 = min(min(ys + [0.0]) - r, min(lines_y, default=0.0))
    y1 = max(max(ys + [0.0]) + r, max(lines_y, default=0.0))
    below = _PLAN_NAME_PT + _PLAN_BAND_PT
    room_w = _PLAN_WIDTH_IN * 72.0 - 20.0 - 2 * _PLAN_SIDE_PT
    room_h = (_PLAN_MAX_HEIGHT_IN * 72.0 - 40.0 - _PLAN_ABOVE_PT
              - below)
    span = max(x1 - x0, y1 - y0, 1e-3)
    scale = max((x1 - x0) / room_w, (y1 - y0) / room_h,
                0.2 * span / room_w)
    left, right = x0 - _PLAN_SIDE_PT * scale, x1 + _PLAN_SIDE_PT * scale
    widen = _PLAN_MIN_VIEW_PT * scale - (right - left)
    if widen > 0:
        left, right = left - widen / 2, right + widen / 2
    bottom, top = y0 - below * scale, y1 + _PLAN_ABOVE_PT * scale
    height_in = min(max((top - bottom) / scale / 72.0 + 0.4, 1.5),
                    _PLAN_MAX_HEIGHT_IN)
    fig, ax = plt.subplots(figsize=(_PLAN_WIDTH_IN, height_in))
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()               # the scale bar carries the scale

    # The cap's x and y axes through its origin, named at their ends
    cap_axis = dict(color='#888888', lw=0.7, linestyle=(0, (6, 3)),
                    zorder=2)
    x_end = x1 + 0.45 * _PLAN_SIDE_PT * scale
    y_end = y1 + 0.35 * _PLAN_ABOVE_PT * scale
    ax.plot([x0 - 0.45 * _PLAN_SIDE_PT * scale, x_end], [0, 0],
            **cap_axis)
    ax.plot([0, 0], [y0, y_end], **cap_axis)
    if cap:
        ax.add_patch(mpatches.Polygon(
            cap, closed=True, facecolor='#e5e7eb', edgecolor='#444444',
            linewidth=1.0, zorder=1))
    if prop:
        ax.plot([p[0] for p in prop], [p[1] for p in prop],
                color=pf.INK, lw=1.2, linestyle='-.', zorder=3)
    axis_names = [
        ax.annotate('x', (x_end, 0), xytext=(2, 0),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=8, style='italic', color='#555555'),
        ax.annotate('y', (0, y_end), xytext=(0, 1),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=8, style='italic', color='#555555')]

    # The piles: blue in compression, red and hatched in tension, darker
    # the larger the force. The hatch tells tension from compression in
    # grey print, where the two colours print alike and the forces may be
    # left to the table (GM-200 review)
    f_max = max(max(abs(f) for f in forces), 1e-9)
    fills = [(0.15, 0.39, 0.92, 0.25 + 0.65 * abs(f) / f_max) if f >= 0
             else (0.86, 0.20, 0.16, 0.25 + 0.65 * abs(f) / f_max)
             for f in forces]
    piles = []
    if r > 0:
        for x, y, c, f in zip(xs, ys, fills, forces):
            piles.append(ax.add_patch(mpatches.Circle(
                (x, y), r, facecolor=c, edgecolor='#333333', linewidth=0.9,
                hatch=_TENSION_HATCH if f < 0 else None, zorder=4)))
    else:
        for tension in (False, True):
            at = [i for i, f in enumerate(forces) if (f < 0) == tension]
            if at:
                ax.scatter([xs[i] for i in at], [ys[i] for i in at],
                           s=(2 * _PLAN_SYMBOL_PT) ** 2,
                           c=[fills[i] for i in at], edgecolors='#333333',
                           linewidths=0.9, zorder=4,
                           hatch=_TENSION_HATCH if tension else None)

    # The names under the piles, set off from each pile's edge in points
    names = []
    for x, y, lab, f in zip(xs, ys, labels, forces):
        if r > 0:
            at, off = (x, y - r), -2
        else:
            at, off = (x, y), -(_PLAN_SYMBOL_PT + 2)
        names.append(ax.annotate(
            f"{lab}\n{f:+,.1f} kN", at, xytext=(0, off),
            textcoords='offset points', ha='center', va='top',
            fontsize=_PLAN_NAME_SIZE, color='#222222', zorder=6))
    halo(names + axis_names)

    # The scale bar and, right of it, the north arrow, in the band under
    # the names
    length = pf.nice_length(right - left)
    arrow_xy = (right - 18.0 * scale, y0 - (_PLAN_NAME_PT + 28.0) * scale)
    bar_x = arrow_xy[0] - 36.0 * scale - length
    bar_y = y0 - (_PLAN_NAME_PT + 18.0) * scale
    pf.mpl_scale_bar(ax, bar_x, bar_y, length, 4.0 * scale, "m",
                     fontsize=7.5)
    pf.mpl_north_arrow(ax, arrow_xy, 0.0 if north is None else float(north))

    ax.set_title(f'Pile Group Plan  ({len(pile_forces)} piles)',
                 fontsize=10, fontweight='bold', loc='left')

    # Names that cannot be written clear of each other and of the piles at
    # this scale: the forces are left to the force table, then the names too
    fig.tight_layout()
    written = _clear_names(fig, ax, names, piles, xs, ys, labels, r)

    # the key's lines no wider than the figure prints (a wider key keeps
    # the figure from being drawn at its printed size)
    notes = ["Axial force at each pile head (kN): + compression, "
             "− tension (hatched);",
             "the darker the fill, the larger the force."]
    if written != "all":
        notes += [("Too many piles to write each force at this scale:"
                   if written == "names" else
                   "Too many piles to name at this scale:"),
                  "see the table Per-Pile Axial Forces."]
    notes.append(f"Piles to scale, D = {float(D):.2f} m." if r > 0 else
                 "Piles drawn as symbols, not to scale: no diameter given.")
    notes.append("x, y: the cap's axes, through its origin."
                 + (" North drawn up: none was given." if north is None
                    else ""))
    missing = [name for name, given in (("cap outline", cap),
                                        ("property line", prop))
               if not given]
    if missing:
        notes.append(f"No {' or '.join(missing)} given: not drawn.")
    handles = []
    if cap:
        handles.append(mpatches.Patch(facecolor='#e5e7eb',
                                      edgecolor='#444444', linewidth=1.0,
                                      label='Pile cap'))
    if prop:
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], color=pf.INK, lw=1.2,
                              linestyle='-.', label='Property line'))
    if any(f >= 0 for f in forces):
        handles.append(mpatches.Patch(color=(0.15, 0.39, 0.92, 0.8),
                                      label='Compression (+)'))
    if any(f < 0 for f in forces):
        handles.append(mpatches.Patch(facecolor=(0.86, 0.20, 0.16, 0.8),
                                      edgecolor='#333333', linewidth=0.9,
                                      hatch=_TENSION_HATCH,
                                      label='Tension (−)'))
    key_under_axes(fig, ax, handles=handles, note="\n".join(notes),
                   fontsize=8)
    return fig


def _clear_names(fig, ax, names, piles, xs, ys, labels, r) -> str:
    """Write the piles' names as they fit at the printed size: each pile's
    name and force ("all"), else the name only ("names"), else none
    ("none"). A name fits when it is clear of every other name and of
    every pile but its own."""
    from matplotlib.transforms import Bbox

    pt = fig.dpi / 72.0
    gap = 2.0 * pt
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    if r > 0:
        circles = [p.get_window_extent(rend) for p in piles]
    else:
        rad = _PLAN_SYMBOL_PT * pt
        circles = [Bbox.from_extents(cx - rad, cy - rad, cx + rad, cy + rad)
                   for cx, cy in ax.transData.transform(list(zip(xs, ys)))]

    def apart(a, b):
        return (a.x1 + gap < b.x0 or b.x1 + gap < a.x0
                or a.y1 + gap < b.y0 or b.y1 + gap < a.y0)

    def clear():
        fig.canvas.draw()
        boxes = [t.get_window_extent(fig.canvas.get_renderer())
                 for t in names]
        for i, a in enumerate(boxes):
            if not all(apart(a, b) for b in boxes[i + 1:]):
                return False
            if not all(apart(a, c) for j, c in enumerate(circles)
                       if j != i):
                return False
        return True

    if clear():
        return "all"
    for t, lab in zip(names, labels):
        t.set_text(lab)
    if clear():
        return "names"
    for t in names:
        t.remove()
    return "none"


#: The bar chart's room for its bars at the width it prints at (points),
#: and the most pile names its axis carries there.
_BAR_ROOM_PT = 300.0
_BAR_NAMES = 30


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

    # 7 in wide, printed at 5 in: a figure 0.6 in a pile wide was shrunk
    # to the print width and its lettering ran together (GM-206, 64 piles)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(7, 5))

    x_pos = np.arange(n)
    bars = ax.bar(x_pos, forces, color=colors, edgecolor='#333333',
                  linewidth=0.8, width=0.65)

    # Value labels on bars, 2 pt off the bar end (an offset of 2 kN put
    # them 68 in off the figure when the forces were near zero; GM-186):
    # across the bar while they fit side by side at the printed width,
    # along it while a bar is wider than a line of lettering, else on the
    # largest compression and tension only
    per_bar_pt = _BAR_ROOM_PT / max(n, 1)
    turned = per_bar_pt < 36.0
    shown = range(n)
    if per_bar_pt < 10.0:
        shown = {int(np.argmax(forces_arr))}
        if forces_arr.min() < 0:
            shown.add(int(np.argmin(forces_arr)))
    for i, (bar, val) in enumerate(zip(bars, forces)):
        if i not in shown:
            continue
        ypos = bar.get_height()
        va = 'bottom' if val >= 0 else 'top'
        offset = 2 if val >= 0 else -2
        ax.annotate(f'{val:+,.1f}', (bar.get_x() + bar.get_width() / 2, ypos),
                    xytext=(0, offset), textcoords='offset points',
                    ha='center', va=va, fontsize=7.5, fontweight='bold',
                    rotation=90 if turned else 0)

    # room past the longest bar for its value (GM-186)
    ax.margins(y=0.25 if turned else 0.12)

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

    # every pile named while the names fit at the printed width, else
    # every k-th
    step = max(1, math.ceil(n / _BAR_NAMES))
    ax.set_xticks(x_pos)
    ax.set_xticklabels([lab if i % step == 0 else ''
                        for i, lab in enumerate(labels)],
                       rotation=45 if 6 < n <= 12 else 90 if n > 12 else 0,
                       ha='right' if 6 < n <= 12 else 'center',
                       fontsize=8)
    ax.set_xlabel('Pile', fontsize=10)
    ax.set_ylabel('Axial Force (kN)', fontsize=10)

    max_comp = _get(result, 'max_compression', 0)
    max_tens = _get(result, 'max_tension', 0)
    ax.set_title(
        f'Individual Pile Forces\n'
        f'max compression = {max_comp:,.1f} kN, '
        f'max tension = {max_tens:,.1f} kN',
        fontsize=10, fontweight='bold',
    )
    ax.grid(True, axis='y', alpha=0.2)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig

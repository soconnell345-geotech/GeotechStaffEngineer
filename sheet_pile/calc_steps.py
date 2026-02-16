"""
Calculation package steps for sheet pile wall analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation.

Handles both cantilever and anchored walls with Rankine/Coulomb
earth pressure theory.

References:
    USACE EM 1110-2-2504 (Design of Sheet Pile Walls)
    USS Steel Sheet Piling Design Manual
    Rankine (1857) — Active/passive pressure coefficients
    Coulomb (1776) — With wall friction
    Das, B.M. — Principles of Foundation Engineering, Ch 9
"""

import math
from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "Sheet Pile Wall Analysis"

REFERENCES = [
    'USACE EM 1110-2-2504: Design of Sheet Pile Walls.',
    'USS Steel Sheet Piling Design Manual.',
    'Dawkins, W.P. (1991). CWALSHT — Computerized Analysis of Sheet Pile '
    'Walls by Classical Methods. USACE ITL.',
    'Rankine, W.J.M. (1857). "On the Stability of Loose Earth." '
    'Philosophical Transactions of the Royal Society of London, Vol. 147.',
    'Coulomb, C.A. (1776). "Essai sur une Application des Regles de Maximis '
    'et Minimis." Memoires de Mathematique et de Physique, Vol. 7.',
    'Das, B.M. "Principles of Foundation Engineering", Chapter 9.',
]


def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for sheet pile wall calc package.

    Parameters
    ----------
    result : CantileverWallResult or AnchoredWallResult
        Computed results.
    analysis : dict
        Analysis parameters dict with keys such as 'wall_type',
        'excavation_depth', 'soil_layers', 'gwt_depth_active',
        'gwt_depth_passive', 'surcharge', 'FOS_passive',
        'pressure_method', and optionally 'anchor_depth'.

    Returns
    -------
    list of InputItem
    """
    items = []

    # Wall type
    wall_type = analysis.get("wall_type", "cantilever")
    items.append(InputItem(
        "Type", "Wall type",
        "Anchored" if wall_type == "anchored" else "Cantilever", "",
    ))

    # Excavation depth
    H = analysis.get("excavation_depth", result.excavation_depth)
    items.append(InputItem("H", "Excavation depth", f"{H:.2f}", "m"))

    # Anchor depth (anchored walls)
    if wall_type == "anchored" and hasattr(result, "anchor_depth"):
        items.append(InputItem(
            "h_a", "Anchor depth from top", f"{result.anchor_depth:.2f}", "m",
        ))

    # Surcharge
    q = analysis.get("surcharge", 0.0)
    items.append(InputItem("q", "Surface surcharge", f"{q:.1f}", "kPa"))

    # Pressure method
    method = analysis.get("pressure_method", "rankine").capitalize()
    items.append(InputItem("Method", "Earth pressure theory", method, ""))

    # Factor of safety on passive
    FOS = analysis.get("FOS_passive", result.FOS_passive)
    items.append(InputItem("FS_p", "FOS on passive resistance", f"{FOS:.2f}", ""))

    # Water levels
    gwt_a = analysis.get("gwt_depth_active", None)
    gwt_p = analysis.get("gwt_depth_passive", None)
    if gwt_a is not None:
        items.append(InputItem(
            "GWT_a", "GWT depth (active side)", f"{gwt_a:.2f}", "m",
        ))
    if gwt_p is not None:
        items.append(InputItem(
            "GWT_p", "GWT depth (passive side)", f"{gwt_p:.2f}", "m",
        ))

    # Soil layers
    layers = analysis.get("soil_layers", [])
    for i, layer in enumerate(layers, 1):
        desc = layer.description if layer.description else f"Layer {i}"
        items.append(InputItem(
            f"\u03c6_{i}", f"Friction angle ({desc})",
            f"{layer.friction_angle:.1f}", "deg",
        ))
        items.append(InputItem(
            f"c_{i}", f"Cohesion ({desc})",
            f"{layer.cohesion:.1f}", "kPa",
        ))
        items.append(InputItem(
            f"\u03b3_{i}", f"Unit weight ({desc})",
            f"{layer.unit_weight:.1f}", "kN/m\u00b3",
        ))
        items.append(InputItem(
            f"t_{i}", f"Thickness ({desc})",
            f"{layer.thickness:.2f}", "m",
        ))

    return items


def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections for sheet pile wall.

    Parameters
    ----------
    result : CantileverWallResult or AnchoredWallResult
        Computed results.
    analysis : dict
        Analysis parameters dict.

    Returns
    -------
    list of CalcSection
    """
    sections = []
    wall_type = analysis.get("wall_type", "cantilever")
    layers = analysis.get("soil_layers", [])
    H = analysis.get("excavation_depth", result.excavation_depth)
    q = analysis.get("surcharge", 0.0)
    FOS = analysis.get("FOS_passive", result.FOS_passive)
    method = analysis.get("pressure_method", "rankine").lower()

    # ── Section 1: Earth Pressure Coefficients ──────────────────
    ep_items = []

    # Description of method
    if method == "coulomb":
        ep_items.append(
            "Earth pressures computed using Coulomb (1776) theory, which "
            "accounts for wall-soil interface friction."
        )
    else:
        ep_items.append(
            "Earth pressures computed using Rankine (1857) theory, which "
            "assumes a smooth vertical wall and horizontal backfill."
        )

    # Ka/Kp for each unique friction angle
    seen_phis = set()
    layer_rows = []
    for i, layer in enumerate(layers, 1):
        phi = layer.friction_angle
        phi_rad = math.radians(phi)

        if method == "coulomb":
            from sheet_pile.earth_pressure import coulomb_Ka, coulomb_Kp
            Ka = coulomb_Ka(phi)
            Kp = coulomb_Kp(phi)
            ka_eq = "Coulomb Ka"
            kp_eq = "Coulomb Kp"
        else:
            Ka = math.tan(math.pi / 4 - phi_rad / 2) ** 2
            Kp = math.tan(math.pi / 4 + phi_rad / 2) ** 2
            ka_eq = "tan\u00b2(45\u00b0 - \u03c6/2)"
            kp_eq = "tan\u00b2(45\u00b0 + \u03c6/2)"

        desc = layer.description if layer.description else f"Layer {i}"
        layer_rows.append([
            desc,
            f"{phi:.1f}\u00b0",
            f"{layer.cohesion:.1f}",
            f"{layer.unit_weight:.1f}",
            f"{Ka:.4f}",
            f"{Kp:.4f}",
        ])

        if phi not in seen_phis:
            seen_phis.add(phi)
            ep_items.append(CalcStep(
                title=f"Active Coefficient Ka (\u03c6 = {phi:.1f}\u00b0)",
                equation=f"Ka = {ka_eq}",
                substitution=f"Ka = tan\u00b2(45\u00b0 - {phi/2:.1f}\u00b0)"
                if method == "rankine" else f"Ka (Coulomb, \u03c6={phi:.1f}\u00b0)",
                result_name="Ka",
                result_value=f"{Ka:.4f}",
                reference=f"{'Rankine (1857)' if method == 'rankine' else 'Coulomb (1776)'}; "
                          f"USACE EM 1110-2-2504",
            ))
            ep_items.append(CalcStep(
                title=f"Passive Coefficient Kp (\u03c6 = {phi:.1f}\u00b0)",
                equation=f"Kp = {kp_eq}",
                substitution=f"Kp = tan\u00b2(45\u00b0 + {phi/2:.1f}\u00b0)"
                if method == "rankine" else f"Kp (Coulomb, \u03c6={phi:.1f}\u00b0)",
                result_name="Kp",
                result_value=f"{Kp:.4f}",
                reference=f"{'Rankine (1857)' if method == 'rankine' else 'Coulomb (1776)'}; "
                          f"USACE EM 1110-2-2504",
            ))

    # Summary table of layer properties and coefficients
    ep_items.append(TableData(
        title="Soil Layer Summary",
        headers=["Layer", "\u03c6 (deg)", "c (kPa)", "\u03b3 (kN/m\u00b3)", "Ka", "Kp"],
        rows=layer_rows,
    ))

    sections.append(CalcSection(
        title="Earth Pressure Coefficients", items=ep_items,
    ))

    # ── Section 2: Active Pressure Distribution ─────────────────
    active_items = []

    active_items.append(CalcStep(
        title="Active Earth Pressure",
        equation="\u03c3_a = Ka \u00d7 \u03c3'_v - 2c\u221aKa + Ka \u00d7 q",
        substitution="Applied at each depth increment along the retained side",
        result_name="\u03c3_a(z)",
        result_value="(varies with depth)",
        reference="USACE EM 1110-2-2504, Eq 3-1",
        notes="Negative active pressures (tension zone) set to zero",
    ))

    # Show tension crack depth for cohesive layers
    for i, layer in enumerate(layers, 1):
        if layer.cohesion > 0:
            phi_rad = math.radians(layer.friction_angle)
            Ka = math.tan(math.pi / 4 - phi_rad / 2) ** 2
            if Ka > 0 and layer.unit_weight > 0:
                z_crack = (2.0 * layer.cohesion / math.sqrt(Ka) - q) / layer.unit_weight
                z_crack = max(z_crack, 0.0)
                desc = layer.description if layer.description else f"Layer {i}"
                active_items.append(CalcStep(
                    title=f"Tension Crack Depth ({desc})",
                    equation="z_c = (2c/\u221aKa - q) / \u03b3",
                    substitution=(
                        f"z_c = (2\u00d7{layer.cohesion:.1f}/\u221a{Ka:.4f} "
                        f"- {q:.1f}) / {layer.unit_weight:.1f}"
                    ),
                    result_name="z_c",
                    result_value=f"{z_crack:.2f}",
                    result_unit="m",
                    reference="USACE EM 1110-2-2504",
                    notes="Active pressure is zero from surface to z_c",
                ))

    sections.append(CalcSection(
        title="Active Pressure Distribution", items=active_items,
    ))

    # ── Section 3: Passive Pressure Distribution ────────────────
    passive_items = []

    passive_items.append(CalcStep(
        title="Passive Earth Pressure (below excavation)",
        equation="\u03c3_p = Kp \u00d7 \u03c3'_v + 2c\u221aKp",
        substitution="Applied on the excavation side below dredge line",
        result_name="\u03c3_p(z)",
        result_value="(varies with depth)",
        reference="USACE EM 1110-2-2504, Eq 3-2",
    ))

    passive_items.append(CalcStep(
        title="Factored Passive Resistance",
        equation="\u03c3_p,d = \u03c3_p / FS_p",
        substitution=f"\u03c3_p,d = \u03c3_p / {FOS:.2f}",
        result_name="FS_p",
        result_value=f"{FOS:.2f}",
        reference="USACE EM 1110-2-2504, Chapter 4",
        notes="Factor of safety applied to reduce passive resistance",
    ))

    sections.append(CalcSection(
        title="Passive Pressure Distribution", items=passive_items,
    ))

    # ── Section 4: Embedment Depth ──────────────────────────────
    embed_items = []

    if wall_type == "anchored":
        embed_items.append(
            "Free Earth Support Method: Embedment depth is found by summing "
            "moments about the anchor location until passive resistance moment "
            "equals or exceeds active driving moment."
        )
        embed_items.append(CalcStep(
            title="Moment Equilibrium about Anchor",
            equation="\u03a3M_passive \u2265 \u03a3M_active (about anchor at h_a)",
            substitution=(
                f"Moments summed about anchor at depth "
                f"{result.anchor_depth:.2f} m from top"
            ),
            result_name="D_calc",
            result_value=f"{result.embedment_depth / 1.2:.2f}",
            result_unit="m",
            reference="USACE EM 1110-2-2504, Chapter 5",
            notes="Embedment at moment balance before safety factor increase",
        ))
    else:
        embed_items.append(
            "Simplified Method: Embedment depth is found by summing moments "
            "about the wall base until factored passive moment equals or "
            "exceeds active driving moment."
        )
        embed_items.append(CalcStep(
            title="Moment Equilibrium about Base",
            equation="\u03a3M_passive \u2265 \u03a3M_active (about wall base)",
            substitution="Iterative solution with numerical integration",
            result_name="D_calc",
            result_value=f"{result.embedment_depth / 1.2:.2f}",
            result_unit="m",
            reference="USACE EM 1110-2-2504, Chapter 4",
            notes="Embedment at moment balance before safety factor increase",
        ))

    # Design embedment (with 20% increase)
    embed_items.append(CalcStep(
        title="Design Embedment Depth",
        equation="D_design = 1.20 \u00d7 D_calc (USACE 20% increase)",
        substitution=f"D_design = 1.20 \u00d7 {result.embedment_depth / 1.2:.2f}",
        result_name="D_design",
        result_value=f"{result.embedment_depth:.2f}",
        result_unit="m",
        reference="USACE EM 1110-2-2504",
        notes="20% increase per USACE guidance for simplified method",
    ))

    # Total wall length
    embed_items.append(CalcStep(
        title="Total Wall Length",
        equation="L = H + D_design",
        substitution=f"L = {H:.2f} + {result.embedment_depth:.2f}",
        result_name="L",
        result_value=f"{result.total_wall_length:.2f}",
        result_unit="m",
    ))

    sections.append(CalcSection(
        title="Required Embedment Depth", items=embed_items,
    ))

    # ── Section 5: Maximum Bending Moment ───────────────────────
    moment_items = []

    moment_items.append(CalcStep(
        title="Maximum Bending Moment",
        equation="M_max occurs where shear V = 0 (net pressure sign change)",
        substitution="Computed by numerical integration of net pressure diagram",
        result_name="M_max",
        result_value=f"{result.max_moment:.1f}",
        result_unit="kN\u00b7m/m",
        reference="USACE EM 1110-2-2504",
    ))

    moment_items.append(CalcStep(
        title="Depth of Maximum Moment",
        equation="z at V = 0",
        substitution="",
        result_name="z_Mmax",
        result_value=f"{result.max_moment_depth:.2f}",
        result_unit="m",
        notes="Depth from top of wall",
    ))

    sections.append(CalcSection(
        title="Maximum Bending Moment", items=moment_items,
    ))

    # ── Section 6: Anchor Force (anchored walls only) ───────────
    if wall_type == "anchored" and hasattr(result, "anchor_force"):
        anchor_items = []

        anchor_items.append(CalcStep(
            title="Anchor Force from Horizontal Equilibrium",
            equation="T = \u03a3F_active - \u03a3F_passive",
            substitution="Sum of all horizontal forces along the wall",
            result_name="T",
            result_value=f"{result.anchor_force:.1f}",
            result_unit="kN/m",
            reference="USACE EM 1110-2-2504, Chapter 5",
            notes="Force per unit length of wall (per meter run)",
        ))

        sections.append(CalcSection(
            title="Anchor Force", items=anchor_items,
        ))

    # ── Section 7: Results Summary ──────────────────────────────
    summary_items = []

    if wall_type == "anchored" and hasattr(result, "anchor_force"):
        summary_items.append(TableData(
            title="Design Summary",
            headers=["Quantity", "Value", "Unit"],
            rows=[
                ["Excavation depth (H)", f"{H:.2f}", "m"],
                ["Anchor depth", f"{result.anchor_depth:.2f}", "m"],
                ["Required embedment", f"{result.embedment_depth:.2f}", "m"],
                ["Total wall length", f"{result.total_wall_length:.2f}", "m"],
                ["FOS on passive", f"{FOS:.2f}", ""],
                ["Anchor force", f"{result.anchor_force:.1f}", "kN/m"],
                ["Maximum moment", f"{result.max_moment:.1f}", "kN\u00b7m/m"],
                ["Depth of max moment", f"{result.max_moment_depth:.2f}", "m"],
            ],
        ))
    else:
        summary_items.append(TableData(
            title="Design Summary",
            headers=["Quantity", "Value", "Unit"],
            rows=[
                ["Excavation depth (H)", f"{H:.2f}", "m"],
                ["Required embedment", f"{result.embedment_depth:.2f}", "m"],
                ["Total wall length", f"{result.total_wall_length:.2f}", "m"],
                ["FOS on passive", f"{FOS:.2f}", ""],
                ["Maximum moment", f"{result.max_moment:.1f}", "kN\u00b7m/m"],
                ["Depth of max moment", f"{result.max_moment_depth:.2f}", "m"],
            ],
        ))

    sections.append(CalcSection(
        title="Results Summary", items=summary_items,
    ))

    return sections


def get_figures(result, analysis) -> List[FigureData]:
    """Generate figures for the sheet pile wall calc package.

    Parameters
    ----------
    result : CantileverWallResult or AnchoredWallResult
        Computed results.
    analysis : dict
        Analysis parameters dict.

    Returns
    -------
    list of FigureData
    """
    figures = []

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Figure 1: Net pressure diagram
        fig1 = _plot_pressure_diagram(result, analysis)
        b64 = figure_to_base64(fig1, dpi=150)
        plt.close(fig1)
        figures.append(FigureData(
            title="Earth Pressure Diagram",
            image_base64=b64,
            caption=(
                f"Figure 1: Active and passive earth pressure distributions "
                f"along the wall (H = {result.excavation_depth:.2f} m, "
                f"D = {result.embedment_depth:.2f} m)."
            ),
            width_percent=70,
        ))

        # Figure 2: Shear and moment diagrams
        fig2 = _plot_shear_moment(result, analysis)
        b64_2 = figure_to_base64(fig2, dpi=150)
        plt.close(fig2)
        figures.append(FigureData(
            title="Shear Force and Bending Moment Diagrams",
            image_base64=b64_2,
            caption=(
                f"Figure 2: Shear force and bending moment along the wall. "
                f"M_max = {result.max_moment:.1f} kN\u00b7m/m at depth "
                f"{result.max_moment_depth:.2f} m."
            ),
            width_percent=85,
        ))

        # Figure 3: Wall cross-section schematic
        fig3 = _plot_wall_section(result, analysis)
        b64_3 = figure_to_base64(fig3, dpi=150)
        plt.close(fig3)
        figures.append(FigureData(
            title="Wall Cross-Section",
            image_base64=b64_3,
            caption=(
                f"Figure 3: Sheet pile wall geometry showing excavation depth, "
                f"embedment, and key dimensions."
            ),
            width_percent=65,
        ))

    except ImportError:
        pass

    return figures


# ── Private helper functions ─────────────────────────────────────


def _compute_pressures(result, analysis, n_points=200):
    """Compute active and passive pressure arrays along the wall.

    Returns (depths, p_active, p_passive, p_net) arrays.
    """
    import numpy as np
    from sheet_pile.earth_pressure import rankine_Ka, rankine_Kp
    from sheet_pile.cantilever import (
        _get_soil_at_depth, _cumulative_stress, _effective_gamma,
    )
    from geotech_common.water import GAMMA_W

    H = analysis.get("excavation_depth", result.excavation_depth)
    layers = analysis.get("soil_layers", [])
    q = analysis.get("surcharge", 0.0)
    FOS = analysis.get("FOS_passive", result.FOS_passive)
    gwt_a = analysis.get("gwt_depth_active", None)
    gwt_p = analysis.get("gwt_depth_passive", None)
    gamma_w = analysis.get("gamma_w", GAMMA_W)
    total_length = result.total_wall_length

    depths = np.linspace(0, total_length, n_points)
    p_active = np.zeros(n_points)
    p_passive = np.zeros(n_points)

    for idx, z in enumerate(depths):
        layer = _get_soil_at_depth(z, layers)
        Ka = rankine_Ka(layer.friction_angle)
        Kp = rankine_Kp(layer.friction_angle)

        if z <= H:
            # Above excavation: active pressure only
            sigma_v = q + _cumulative_stress(z, layers, gwt_a, gamma_w)
            pa = Ka * sigma_v - 2 * layer.cohesion * math.sqrt(Ka)
            pa = max(pa, 0)
            # Add water pressure on active side
            if gwt_a is not None and z > gwt_a:
                pa += gamma_w * (z - gwt_a)
            p_active[idx] = pa
        else:
            z_below = z - H
            # Active side
            sigma_v_a = q + _cumulative_stress(z, layers, gwt_a, gamma_w)
            pa = Ka * sigma_v_a - 2 * layer.cohesion * math.sqrt(Ka)
            pa = max(pa, 0)
            if gwt_a is not None and z > gwt_a:
                pa += gamma_w * (z - gwt_a)

            # Passive side (factored)
            sigma_v_p = _cumulative_stress(z_below, layers, gwt_p, gamma_w)
            pp = Kp * sigma_v_p + 2 * layer.cohesion * math.sqrt(Kp)
            pp_reduced = pp / FOS

            # Differential water pressure
            if gwt_p is not None and z > gwt_p:
                u_p = gamma_w * max(0, z_below - max(0, (gwt_p or 1e10) - H))
                pp_reduced += u_p

            p_active[idx] = pa
            p_passive[idx] = pp_reduced

    p_net = p_active - p_passive
    return depths, p_active, p_passive, p_net


def _plot_pressure_diagram(result, analysis):
    """Create earth pressure distribution diagram."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    depths, p_active, p_passive, p_net = _compute_pressures(result, analysis)
    H = analysis.get("excavation_depth", result.excavation_depth)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), sharey=True)

    # Left panel: active and passive pressures
    ax1.plot(p_active, depths, color='#dc2626', linewidth=1.8, label='Active')
    ax1.plot(-p_passive, depths, color='#2563eb', linewidth=1.8, label='Passive')
    ax1.fill_betweenx(depths, 0, p_active, alpha=0.15, color='#dc2626')
    ax1.fill_betweenx(depths, 0, -p_passive, alpha=0.15, color='#2563eb')
    ax1.axhline(y=H, color='#16a34a', linestyle='--', linewidth=1.2,
                label=f'Excavation line (H={H:.1f} m)')
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Lateral Pressure (kPa)', fontsize=10)
    ax1.set_ylabel('Depth from Top of Wall (m)', fontsize=10)
    ax1.set_title('Active & Passive Pressures', fontsize=11, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend(fontsize=8, loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=9)

    # Right panel: net pressure
    pos_net = np.maximum(p_net, 0)
    neg_net = np.minimum(p_net, 0)
    ax2.fill_betweenx(depths, 0, pos_net, alpha=0.3, color='#dc2626',
                       label='Net active (driving)')
    ax2.fill_betweenx(depths, 0, neg_net, alpha=0.3, color='#2563eb',
                       label='Net passive (resisting)')
    ax2.plot(p_net, depths, color='#1a1a1a', linewidth=1.5)
    ax2.axhline(y=H, color='#16a34a', linestyle='--', linewidth=1.2)
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Net Pressure (kPa)', fontsize=10)
    ax2.set_title('Net Pressure Diagram', fontsize=11, fontweight='bold')
    ax2.invert_yaxis()
    ax2.legend(fontsize=8, loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=9)

    plt.tight_layout()
    return fig


def _plot_shear_moment(result, analysis):
    """Create shear force and bending moment diagrams."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    depths, p_active, p_passive, p_net = _compute_pressures(result, analysis)
    H = analysis.get("excavation_depth", result.excavation_depth)
    wall_type = analysis.get("wall_type", "cantilever")

    n = len(depths)
    dz = depths[1] - depths[0] if n > 1 else 0.01

    # Compute shear and moment by integration
    shear = np.zeros(n)
    moment = np.zeros(n)

    if wall_type == "anchored" and hasattr(result, "anchor_force"):
        # Anchor reaction at the anchor depth
        anchor_depth = result.anchor_depth
        anchor_idx = np.searchsorted(depths, anchor_depth)
        cumulative_shear = 0.0
        cumulative_moment = 0.0
        for i in range(n):
            if i == anchor_idx:
                cumulative_shear -= result.anchor_force
            cumulative_shear += p_net[i] * dz
            shear[i] = cumulative_shear
            cumulative_moment += cumulative_shear * dz
            moment[i] = cumulative_moment
    else:
        cumulative_shear = 0.0
        cumulative_moment = 0.0
        for i in range(n):
            cumulative_shear += p_net[i] * dz
            shear[i] = cumulative_shear
            cumulative_moment += cumulative_shear * dz
            moment[i] = cumulative_moment

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), sharey=True)

    # Shear force diagram
    ax1.plot(shear, depths, color='#d97706', linewidth=1.8)
    ax1.fill_betweenx(depths, 0, shear, alpha=0.15, color='#d97706')
    ax1.axhline(y=H, color='#16a34a', linestyle='--', linewidth=1.2,
                label=f'Excavation line')
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Shear Force (kN/m)', fontsize=10)
    ax1.set_ylabel('Depth from Top of Wall (m)', fontsize=10)
    ax1.set_title('Shear Force Diagram', fontsize=11, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=9)

    # Bending moment diagram
    ax2.plot(moment, depths, color='#7c3aed', linewidth=1.8)
    ax2.fill_betweenx(depths, 0, moment, alpha=0.15, color='#7c3aed')
    ax2.axhline(y=H, color='#16a34a', linestyle='--', linewidth=1.2)
    ax2.axvline(x=0, color='black', linewidth=0.5)

    # Mark maximum moment
    max_idx = np.argmax(np.abs(moment))
    ax2.plot(moment[max_idx], depths[max_idx], 'ro', markersize=8, zorder=5)
    ax2.annotate(
        f'M_max = {abs(moment[max_idx]):.0f} kN\u00b7m/m\n'
        f'at z = {depths[max_idx]:.1f} m',
        xy=(moment[max_idx], depths[max_idx]),
        xytext=(moment[max_idx] + abs(moment).max() * 0.15,
                depths[max_idx] - 0.5),
        fontsize=8,
        arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
    )

    ax2.set_xlabel('Bending Moment (kN\u00b7m/m)', fontsize=10)
    ax2.set_title('Bending Moment Diagram', fontsize=11, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=9)

    plt.tight_layout()
    return fig


def _plot_wall_section(result, analysis):
    """Create a schematic cross-section of the sheet pile wall."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    H = analysis.get("excavation_depth", result.excavation_depth)
    D = result.embedment_depth
    L = result.total_wall_length
    wall_type = analysis.get("wall_type", "cantilever")
    gwt_a = analysis.get("gwt_depth_active", None)
    gwt_p = analysis.get("gwt_depth_passive", None)
    layers = analysis.get("soil_layers", [])

    fig, ax = plt.subplots(figsize=(8, 7))

    wall_x = 0.0
    left_extent = -4.0
    right_extent = 4.0

    # Retained soil (left side, full depth)
    ax.add_patch(patches.Rectangle(
        (left_extent, 0), abs(left_extent), L,
        facecolor='#f5e6c8', edgecolor='none',
    ))

    # Excavated side soil (right side, below excavation)
    ax.add_patch(patches.Rectangle(
        (wall_x, H), right_extent, D,
        facecolor='#e8d5a8', edgecolor='none',
    ))

    # Ground surface lines
    ax.plot([left_extent, wall_x], [0, 0], color='#16a34a', linewidth=2.5)
    ax.plot([wall_x, right_extent], [H, H], color='#16a34a', linewidth=2.5)

    # Excavation cut face
    ax.plot([wall_x, right_extent * 0.15], [0, 0], color='#16a34a', linewidth=2.5)
    ax.plot([right_extent * 0.15, right_extent * 0.15], [0, H],
            color='#8b6914', linewidth=1.5, linestyle=':')
    ax.plot([right_extent * 0.15, wall_x + 0.05], [H, H],
            color='#16a34a', linewidth=2.5)

    # Sheet pile wall (thick vertical line)
    ax.plot([wall_x, wall_x], [0, L], color='#374151', linewidth=4,
            solid_capstyle='butt')
    ax.plot([wall_x, wall_x], [0, L], color='#6b7280', linewidth=2,
            solid_capstyle='butt')

    # GWT on active side
    if gwt_a is not None and gwt_a < L:
        ax.plot([left_extent, wall_x - 0.1], [gwt_a, gwt_a],
                color='#2563eb', linestyle='--', linewidth=1.5)
        ax.text(left_extent + 0.2, gwt_a - 0.15,
                f'GWT = {gwt_a:.1f} m', fontsize=8, color='#2563eb')
        # Water shading below GWT on active side
        gwt_bottom = L
        ax.add_patch(patches.Rectangle(
            (left_extent, gwt_a), abs(left_extent), gwt_bottom - gwt_a,
            facecolor='#dbeafe', edgecolor='none', alpha=0.4,
        ))

    # GWT on passive side
    if gwt_p is not None and gwt_p < L:
        gwt_p_draw = max(gwt_p, H)
        if gwt_p_draw < L:
            ax.plot([wall_x + 0.1, right_extent], [gwt_p_draw, gwt_p_draw],
                    color='#2563eb', linestyle='--', linewidth=1.5)

    # Anchor (if anchored)
    if wall_type == "anchored" and hasattr(result, "anchor_depth"):
        a_depth = result.anchor_depth
        ax.plot([-0.3, -2.5], [a_depth, a_depth], color='#1a1a1a',
                linewidth=2.5, solid_capstyle='butt')
        ax.plot(-2.5, a_depth, 'v', color='#1a1a1a', markersize=10)
        ax.annotate(
            f'Anchor\nT = {result.anchor_force:.0f} kN/m',
            xy=(-1.3, a_depth),
            xytext=(-3.0, a_depth - 0.8),
            fontsize=8, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#1a1a1a', lw=1),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef3c7',
                      edgecolor='#1a1a1a', alpha=0.9),
        )

    # Dimension annotations: H
    ax.annotate('', xy=(right_extent - 0.5, 0),
                xytext=(right_extent - 0.5, H),
                arrowprops=dict(arrowstyle='<->', color='#1a1a1a', lw=1.2))
    ax.text(right_extent - 0.3, H / 2, f'H = {H:.1f} m',
            ha='left', va='center', fontsize=9, fontweight='bold')

    # Dimension annotations: D
    ax.annotate('', xy=(right_extent - 0.5, H),
                xytext=(right_extent - 0.5, L),
                arrowprops=dict(arrowstyle='<->', color='#1a1a1a', lw=1.2))
    ax.text(right_extent - 0.3, H + D / 2, f'D = {D:.1f} m',
            ha='left', va='center', fontsize=9, fontweight='bold')

    # Soil layer labels on retained side
    z_cum = 0.0
    for i, layer in enumerate(layers, 1):
        z_mid = z_cum + layer.thickness / 2
        if z_mid < L:
            desc_parts = []
            if layer.description:
                desc_parts.append(layer.description)
            desc_parts.append(
                f"\u03c6={layer.friction_angle:.0f}\u00b0, "
                f"c={layer.cohesion:.0f} kPa, "
                f"\u03b3={layer.unit_weight:.0f}"
            )
            ax.text(left_extent + 0.3, z_mid, '\n'.join(desc_parts),
                    fontsize=7.5, va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.85, edgecolor='#ccc'))
        z_cum += layer.thickness

    # Labels
    ax.text(-2.0, -0.4, 'Retained Side', ha='center', fontsize=10,
            fontweight='bold', color='#374151')
    ax.text(2.0, H - 0.4, 'Excavation Side', ha='center', fontsize=10,
            fontweight='bold', color='#374151')

    # Surcharge arrow
    q = analysis.get("surcharge", 0.0)
    if q > 0:
        for x_arr in [-3.0, -2.0, -1.0]:
            ax.annotate('', xy=(x_arr, 0),
                        xytext=(x_arr, -0.6),
                        arrowprops=dict(arrowstyle='->', color='#dc2626',
                                        lw=1.5))
        ax.text(-2.0, -0.8, f'q = {q:.0f} kPa', ha='center', fontsize=9,
                color='#dc2626', fontweight='bold')

    ax.set_xlim(left_extent - 0.5, right_extent + 1.0)
    ax.set_ylim(L + 0.5, min(-1.5, -1.0))
    ax.set_xlabel('Distance (m)', fontsize=10)
    ax.set_ylabel('Depth (m)', fontsize=10)
    title = ("Anchored" if wall_type == "anchored" else "Cantilever")
    ax.set_title(f'{title} Sheet Pile Wall Cross-Section',
                 fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig

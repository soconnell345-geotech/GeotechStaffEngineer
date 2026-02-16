"""
Calculation package steps for lateral pile analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation (LPILE-style output).

References:
    COM624P Manual: FHWA-SA-91-048 (Wang & Reese, 1993)
    FHWA GEC-13: FHWA-HIF-18-031
"""

from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "Lateral Pile Analysis"

REFERENCES = [
    'Wang, S.T. & Reese, L.C. (1993). COM624P — Laterally Loaded Pile '
    'Analysis Program for the Microcomputer. FHWA-SA-91-048.',
    'Reese, L.C. & Matlock, H. (1956). Non-dimensional Solutions for '
    'Laterally Loaded Piles with Soil Modulus Assumed Proportional to Depth.',
    'Matlock, H. (1970). Correlations for Design of Laterally Loaded Piles '
    'in Soft Clay. OTC-1204.',
    'Reese, L.C. et al. (1974). Analysis of Laterally Loaded Piles in Sand. '
    'OTC-2080.',
    'API RP 2GEO (2014). Geotechnical and Foundation Design Considerations.',
    'Jeanjean, P. (2009). Re-Assessment of p-y Curves for Soft Clays from '
    'Centrifuge Testing and Finite Element Modeling. OTC-20158.',
    'FHWA GEC-13 (FHWA-HIF-18-031): Ground Modification Methods.',
]


def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for lateral pile calc package.

    Parameters
    ----------
    result : lateral_pile.results.Results
        Computed results.
    analysis : LateralPileAnalysis
        Analysis object holding pile and soil layer inputs.

    Returns
    -------
    list of InputItem
    """
    pile = analysis.pile
    items = [
        InputItem("L", "Pile length", f"{pile.length:.2f}", "m"),
        InputItem("D", "Pile diameter", f"{pile.diameter:.3f}", "m"),
        InputItem("E", "Young's modulus", f"{pile.E:.0f}", "kPa"),
    ]

    if pile.thickness is not None:
        items.append(InputItem("t", "Wall thickness", f"{pile.thickness:.4f}", "m"))

    items.append(InputItem("EI", "Flexural rigidity", f"{pile.EI:.0f}", "kN-m\u00b2"))

    items.extend([
        InputItem("V_t", "Lateral load at head", f"{result.Vt:.1f}", "kN"),
        InputItem("M_t", "Moment at head", f"{result.Mt:.1f}", "kN-m"),
        InputItem("Q", "Axial load", f"{result.Q:.1f}", "kN"),
    ])

    return items


def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections.

    Parameters
    ----------
    result : lateral_pile.results.Results
        Computed results.
    analysis : LateralPileAnalysis
        Analysis object.

    Returns
    -------
    list of CalcSection
    """
    sections = []

    # ── Soil Layer Summary ─────────────────────────────────────
    layer_rows = []
    for i, layer in enumerate(analysis.layers, 1):
        model = layer.py_model
        model_name = type(model).__name__
        desc = layer.description or model_name
        # Extract key parameters from the model
        params = _extract_model_params(model)
        layer_rows.append([
            str(i),
            f"{layer.top:.1f}",
            f"{layer.bottom:.1f}",
            desc,
            model_name,
            params,
        ])

    layer_table = TableData(
        title="Soil Layer Definition",
        headers=["#", "Top (m)", "Bottom (m)", "Description", "p-y Model", "Key Parameters"],
        rows=layer_rows,
    )
    sections.append(CalcSection(title="Soil Profile", items=[layer_table]))

    # ── Pile Section Properties ─────────────────────────────────
    pile = analysis.pile
    pile_steps = []

    if pile.thickness is not None:
        # Pipe pile
        pile_steps.append(CalcStep(
            title="Moment of Inertia (hollow pipe)",
            equation="I = \u03c0/64 \u00d7 (D\u2074 - d\u2074)",
            substitution=(
                f"I = \u03c0/64 \u00d7 ({pile.diameter:.4f}\u2074 - "
                f"{pile.diameter - 2*pile.thickness:.4f}\u2074)"
            ),
            result_name="I",
            result_value=f"{pile.moment_of_inertia:.6e}",
            result_unit="m\u2074",
        ))
    else:
        # Solid pile
        pile_steps.append(CalcStep(
            title="Moment of Inertia (solid circular)",
            equation="I = \u03c0/64 \u00d7 D\u2074",
            substitution=f"I = \u03c0/64 \u00d7 {pile.diameter:.4f}\u2074",
            result_name="I",
            result_value=f"{pile.moment_of_inertia:.6e}",
            result_unit="m\u2074",
        ))

    pile_steps.append(CalcStep(
        title="Flexural Rigidity",
        equation="EI = E \u00d7 I",
        substitution=f"EI = {pile.E:.0f} \u00d7 {pile.moment_of_inertia:.6e}",
        result_name="EI",
        result_value=f"{pile.EI:,.0f}",
        result_unit="kN-m\u00b2",
    ))

    sections.append(CalcSection(title="Pile Section Properties", items=pile_steps))

    # ── Solver Summary ──────────────────────────────────────────
    solver_items = []
    solver_items.append(CalcStep(
        title="Finite Difference Solution",
        equation="Iterative p-y method: EI \u00d7 d\u2074y/dz\u2074 + Q \u00d7 d\u00b2y/dz\u00b2 - p(y,z) = 0",
        substitution="",
        result_name="Converged",
        result_value="Yes" if result.converged else "No",
        notes=f"Iterations: {result.iterations}",
        reference="COM624P finite difference formulation",
    ))

    if result.ei_iterations > 0:
        solver_items.append(CalcStep(
            title="Cracked-Section EI Iteration",
            equation="Branson's equation: EI_eff = EI_cr + (EI_g - EI_cr)(M_cr/M_a)\u00b3",
            substitution="",
            result_name="EI iterations",
            result_value=str(result.ei_iterations),
        ))

    sections.append(CalcSection(title="Solver Performance", items=solver_items))

    # ── Key Results ─────────────────────────────────────────────
    results_items = []

    results_items.append(TableData(
        title="Pile Head Response",
        headers=["Quantity", "Value", "Unit"],
        rows=[
            ["Head deflection", f"{result.y_top * 1000:.2f}", "mm"],
            ["Head rotation", f"{result.rotation_top * 1000:.4f}", "mrad"],
            ["Maximum moment", f"{result.max_moment:.1f}", "kN-m"],
            ["Depth of max moment", f"{result.max_moment_depth:.2f}", "m"],
            ["Maximum shear", f"{result.max_shear:.1f}", "kN"],
            ["Maximum deflection", f"{result.max_deflection * 1000:.2f}", "mm"],
            ["Depth of zero deflection", f"{result.depth_of_zero_deflection():.2f}", "m"],
        ],
    ))

    sections.append(CalcSection(title="Key Results", items=results_items))

    return sections


def get_figures(result, analysis) -> List[FigureData]:
    """Generate LPILE-style figures for the lateral pile calc package.

    Captures the existing plot_all() from the Results class.

    Parameters
    ----------
    result : lateral_pile.results.Results
    analysis : LateralPileAnalysis

    Returns
    -------
    list of FigureData
    """
    figures = []

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Use existing plot_all() for the 4-panel figure
        fig, axes = result.plot_all(show=False)
        b64 = figure_to_base64(fig, dpi=150)
        plt.close(fig)
        figures.append(FigureData(
            title="Pile Response Profiles",
            image_base64=b64,
            caption=(
                f"Figure 1: Deflection, bending moment, shear force, and soil reaction "
                f"profiles along the pile (V_t = {result.Vt:.0f} kN, "
                f"M_t = {result.Mt:.0f} kN-m, Q = {result.Q:.0f} kN)."
            ),
            width_percent=95,
        ))

        # Individual deflection plot (larger, for detail)
        fig2, ax2 = plt.subplots(figsize=(6, 8))
        result.plot_deflection(ax=ax2, show=False, color='#2563eb', linewidth=2)
        ax2.axhline(y=result.max_moment_depth, color='#dc2626', linestyle='--',
                    alpha=0.6, label=f'Max moment depth = {result.max_moment_depth:.2f} m')
        zz = result.depth_of_zero_deflection()
        if zz < result.pile_length:
            ax2.axhline(y=zz, color='#16a34a', linestyle='--',
                        alpha=0.6, label=f'Zero deflection = {zz:.2f} m')
        ax2.legend(fontsize=8)
        b64_2 = figure_to_base64(fig2, dpi=150)
        plt.close(fig2)
        figures.append(FigureData(
            title="Deflection Profile (Detail)",
            image_base64=b64_2,
            caption=(
                f"Figure 2: Detailed deflection profile. "
                f"Head deflection = {result.y_top * 1000:.2f} mm."
            ),
            width_percent=60,
        ))

    except ImportError:
        pass

    return figures


def _extract_model_params(model) -> str:
    """Extract key parameters from a p-y model as a summary string."""
    parts = []
    if hasattr(model, 'c'):
        parts.append(f"c={model.c:.0f} kPa")
    if hasattr(model, 'phi'):
        parts.append(f"\u03c6={model.phi:.0f}\u00b0")
    if hasattr(model, 'gamma'):
        parts.append(f"\u03b3={model.gamma:.1f} kN/m\u00b3")
    if hasattr(model, 'eps50'):
        parts.append(f"\u03b550={model.eps50}")
    if hasattr(model, 'k'):
        parts.append(f"k={model.k:.0f} kN/m\u00b3")
    if hasattr(model, 'J'):
        parts.append(f"J={model.J}")
    return ", ".join(parts) if parts else "—"

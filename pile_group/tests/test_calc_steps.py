"""
Tests for pile_group/calc_steps.py calculation package generator.
"""

import pytest

from pile_group.calc_steps import (
    DISPLAY_NAME, REFERENCES,
    get_input_summary, get_calc_steps, get_figures,
)
from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from pile_group import (
    create_rectangular_layout, GroupLoad,
    analyze_vertical_group_simple, analyze_group_6dof,
)


class _Analysis:
    """Simple namespace for analysis parameters."""
    pass


def _make_analysis_and_result(method='simple', n_rows=3, n_cols=3,
                               spacing=2.0, with_capacity=True):
    """Helper to create a standard analysis + result pair."""
    piles = create_rectangular_layout(
        n_rows, n_cols, spacing, spacing,
        axial_stiffness=50000, lateral_stiffness=5000,
    )
    if with_capacity:
        for p in piles:
            p.axial_capacity_compression = 800.0
            p.axial_capacity_tension = 400.0

    load = GroupLoad(Vz=2700, My=500, Mx=200)

    if method == '6dof':
        result = analyze_group_6dof(piles, load)
    else:
        result = analyze_vertical_group_simple(piles, load)

    analysis = _Analysis()
    analysis.piles = piles
    analysis.load = load
    analysis.method = method
    analysis.n_rows = n_rows
    analysis.n_cols = n_cols
    analysis.spacing_x = spacing
    analysis.spacing_y = spacing
    analysis.spacing = spacing
    analysis.pile_diameter = 0.3
    analysis.pile_length = 15.0

    return result, analysis


# ===================================================================
# Module-level constants
# ===================================================================

class TestConstants:

    def test_display_name(self):
        assert DISPLAY_NAME == "Pile Group Analysis"

    def test_references_not_empty(self):
        assert len(REFERENCES) >= 4

    def test_references_are_strings(self):
        assert all(isinstance(r, str) for r in REFERENCES)


# ===================================================================
# get_input_summary
# ===================================================================

class TestGetInputSummary:

    def test_returns_input_items(self):
        result, analysis = _make_analysis_and_result()
        items = get_input_summary(result, analysis)
        assert all(isinstance(i, InputItem) for i in items)

    def test_has_layout_info(self):
        result, analysis = _make_analysis_and_result()
        items = get_input_summary(result, analysis)
        names = [i.name for i in items]
        assert "Layout" in names

    def test_has_loading(self):
        result, analysis = _make_analysis_and_result()
        items = get_input_summary(result, analysis)
        names = [i.name for i in items]
        assert "V_z" in names

    def test_has_pile_properties(self):
        result, analysis = _make_analysis_and_result()
        items = get_input_summary(result, analysis)
        names = [i.name for i in items]
        assert "k_a" in names
        assert "Q_comp" in names

    def test_shows_moment_loads(self):
        result, analysis = _make_analysis_and_result()
        items = get_input_summary(result, analysis)
        names = [i.name for i in items]
        assert "M_y" in names
        assert "M_x" in names

    def test_dict_based_analysis(self):
        result, analysis = _make_analysis_and_result()
        analysis_dict = {
            'piles': analysis.piles,
            'load': analysis.load,
            'method': 'simple',
            'n_rows': 3, 'n_cols': 3,
            'spacing': 2.0, 'pile_diameter': 0.3,
        }
        items = get_input_summary(result, analysis_dict)
        assert len(items) > 5

    def test_method_displayed(self):
        result, analysis = _make_analysis_and_result()
        items = get_input_summary(result, analysis)
        names = [i.name for i in items]
        assert "Method" in names


# ===================================================================
# get_calc_steps
# ===================================================================

class TestGetCalcSteps:

    def test_returns_calc_sections(self):
        result, analysis = _make_analysis_and_result()
        sections = get_calc_steps(result, analysis)
        assert all(isinstance(s, CalcSection) for s in sections)

    def test_four_sections(self):
        result, analysis = _make_analysis_and_result()
        sections = get_calc_steps(result, analysis)
        assert len(sections) == 4

    def test_section_names(self):
        result, analysis = _make_analysis_and_result()
        sections = get_calc_steps(result, analysis)
        titles = [s.title for s in sections]
        assert "Pile Layout" in titles
        assert "Group Efficiency" in titles
        assert "Load Distribution (Rigid Cap)" in titles
        assert "Individual Pile Forces & Utilization" in titles

    def test_pile_layout_has_table(self):
        result, analysis = _make_analysis_and_result()
        sections = get_calc_steps(result, analysis)
        layout_section = sections[0]
        tables = [i for i in layout_section.items if isinstance(i, TableData)]
        assert len(tables) >= 1
        assert tables[0].title == "Pile Coordinates (relative to cap centroid)"
        assert len(tables[0].rows) == 9  # 3x3

    def test_efficiency_has_converse_labarre(self):
        result, analysis = _make_analysis_and_result()
        sections = get_calc_steps(result, analysis)
        eff_section = sections[1]
        eff_steps = [i for i in eff_section.items if isinstance(i, CalcStep)]
        assert any("Converse-Labarre" in s.title for s in eff_steps)

    def test_efficiency_missing_params_gives_text(self):
        """When efficiency parameters not provided, show info note."""
        result, analysis = _make_analysis_and_result()
        # Remove efficiency params
        del analysis.n_rows
        del analysis.n_cols
        del analysis.pile_diameter
        del analysis.spacing
        sections = get_calc_steps(result, analysis)
        eff_section = sections[1]
        text_items = [i for i in eff_section.items if isinstance(i, str)]
        assert len(text_items) >= 1

    def test_load_distribution_simplified(self):
        result, analysis = _make_analysis_and_result(method='simple')
        sections = get_calc_steps(result, analysis)
        load_section = sections[2]
        steps = [i for i in load_section.items if isinstance(i, CalcStep)]
        assert any("Simplified" in s.title for s in steps)

    def test_load_distribution_6dof(self):
        result, analysis = _make_analysis_and_result(method='6dof')
        sections = get_calc_steps(result, analysis)
        load_section = sections[2]
        steps = [i for i in load_section.items if isinstance(i, CalcStep)]
        assert any("6-DOF" in s.title for s in steps)

    def test_cap_displacements_table(self):
        result, analysis = _make_analysis_and_result()
        sections = get_calc_steps(result, analysis)
        load_section = sections[2]
        tables = [i for i in load_section.items if isinstance(i, TableData)]
        assert any("Cap Displacements" in t.title for t in tables)

    def test_pile_forces_table(self):
        result, analysis = _make_analysis_and_result()
        sections = get_calc_steps(result, analysis)
        forces_section = sections[3]
        tables = [i for i in forces_section.items if isinstance(i, TableData)]
        assert len(tables) >= 1
        assert len(tables[0].rows) == 9  # 3x3 group

    def test_utilization_check(self):
        result, analysis = _make_analysis_and_result(with_capacity=True)
        sections = get_calc_steps(result, analysis)
        forces_section = sections[3]
        checks = [i for i in forces_section.items if isinstance(i, CheckItem)]
        assert len(checks) == 1

    def test_max_forces_step(self):
        result, analysis = _make_analysis_and_result()
        sections = get_calc_steps(result, analysis)
        forces_section = sections[3]
        force_steps = [i for i in forces_section.items
                       if isinstance(i, CalcStep) and "Maximum Pile Forces" in i.title]
        assert len(force_steps) == 1


# ===================================================================
# get_figures
# ===================================================================

class TestGetFigures:

    def test_returns_figure_data(self):
        result, analysis = _make_analysis_and_result()
        figs = get_figures(result, analysis)
        assert all(isinstance(f, FigureData) for f in figs)

    def test_two_figures(self):
        result, analysis = _make_analysis_and_result()
        figs = get_figures(result, analysis)
        assert len(figs) == 2

    def test_plan_view_figure(self):
        result, analysis = _make_analysis_and_result()
        figs = get_figures(result, analysis)
        assert figs[0].title == "Pile Group Plan View"
        assert len(figs[0].image_base64) > 100

    def test_bar_chart_figure(self):
        result, analysis = _make_analysis_and_result()
        figs = get_figures(result, analysis)
        assert figs[1].title == "Individual Pile Axial Forces"
        assert len(figs[1].image_base64) > 100

    def test_empty_result_no_figures(self):
        """No pile forces => no figures."""
        from pile_group.rigid_cap import PileGroupResult
        empty_result = PileGroupResult()
        figs = get_figures(empty_result, None)
        assert len(figs) == 0


# ===================================================================
# Full integration with calc_package generator
# ===================================================================

class TestCalcPackageIntegration:

    def test_generate_html(self):
        from calc_package import generate_calc_package
        result, analysis = _make_analysis_and_result()
        html = generate_calc_package(
            module='pile_group',
            result=result,
            analysis=analysis,
            project_name='Test Bridge',
            engineer='Test Engineer',
        )
        assert '<html' in html.lower()
        assert 'Pile Group Analysis' in html
        assert 'Converse-Labarre' in html

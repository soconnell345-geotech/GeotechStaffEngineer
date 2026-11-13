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
    analysis.soil_type = 'cohesive'
    analysis.cu = 50.0
    analysis.cap_in_contact = False

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
        # the coordinates as given (GM-319 review: the simplified method
        # measures from the piles' centroid, its own step)
        assert tables[0].title == "Pile Coordinates"
        assert len(tables[0].rows) == 9  # 3x3

    def test_efficiency_read_from_the_printed_row(self):
        """GEC-12 Sec. 7.2.2.2: soft clay, cap not in contact, s/D = 3:
        0.65 + 0.35 x 0.5/3.5 = 0.70."""
        result, analysis = _make_analysis_and_result(spacing=0.9)
        sections = get_calc_steps(result, analysis)
        eff_steps = [i for i in sections[1].items if isinstance(i, CalcStep)]
        step = next(s for s in eff_steps
                    if s.title == "Group Efficiency Factor")
        assert step.result_value == "0.700"
        assert "GEC-12" in step.reference and "7.2.2" in step.reference
        assert "2.5" in step.substitution and "6" in step.substitution
        # the substitution is raw markup: the row's "<" as an entity
        assert "su &lt; 2 ksf" in step.substitution
        assert not any("Converse" in s.title for s in eff_steps)

    def test_efficiency_without_soil_type_gives_text(self):
        result, analysis = _make_analysis_and_result()
        del analysis.soil_type
        sections = get_calc_steps(result, analysis)
        text = [i for i in sections[1].items if isinstance(i, str)]
        assert text and "soil_type" in text[0]

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


class TestCentroidSteps:
    """GM-319 review: the simplified method takes the loads at the piles'
    centroid and measures x and y from it; the package says so. Made-up
    values."""

    @staticmethod
    def _steps(xs, method='simple'):
        from pile_group import GroupPile
        piles = [GroupPile(x=x, y=0.0, axial_stiffness=5e4) for x in xs]
        load = GroupLoad(Vz=900, My=200)
        analyze = (analyze_group_6dof if method == '6dof'
                   else analyze_vertical_group_simple)
        analysis = _Analysis()
        analysis.piles, analysis.load, analysis.method = piles, load, method
        sections = get_calc_steps(analyze(piles, load), analysis)
        return {i.title: i for s in sections for i in s.items
                if isinstance(i, CalcStep)}

    def test_off_centre_layout(self):
        steps = self._steps((-1.0, 1.0, 3.0))
        c = steps["Centroid of the Piles"]
        assert c.result_value == "x_c = 1.000 m, y_c = 0.000 m"
        assert "WARNING: The loads are taken at the centroid" in c.notes
        assert "-900.0 kN-m to My" in c.notes
        s = steps["Sum of Squared Distances"]
        assert "(x_i − x_c)" in s.equation
        assert s.result_value.startswith("Σx² = 8.000 m²")
        d = steps["Simplified Elastic Load Distribution"]
        assert d.substitution == ("P_i = 900.0/3 + 200.0·(x_i − "
                                  "1.000)/8.000")
        assert "Eq. 8-79 (p. 468)" in d.reference

    def test_centred_layout_unchanged_but_for_the_centroid(self):
        steps = self._steps((-2.0, 0.0, 2.0))
        c = steps["Centroid of the Piles"]
        assert c.result_value == "x_c = 0.000 m, y_c = 0.000 m"
        assert "WARNING" not in c.notes
        assert steps["Sum of Squared Distances"].equation.startswith(
            "Σx² = sum of x_i²")
        assert steps["Simplified Elastic Load Distribution"].substitution \
            == "P_i = 900.0/3 + 200.0·x_i/8.000"

    def test_6dof_has_no_centroid_step(self):
        steps = self._steps((-1.0, 1.0, 3.0), method='6dof')
        assert "Centroid of the Piles" not in steps
        # its sums step as before, from the coordinates as given
        assert steps["Sum of Squared Distances"].result_value.startswith(
            "Σx² = 11.000 m²")


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
        assert 'Group Efficiency Factor' in html
        assert 'Converse' not in html

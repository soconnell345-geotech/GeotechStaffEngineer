"""Tests for the upgraded slope_stability calc_steps (Phase 2, calc-viz).

Verifies the new sections render the modern engine output: interslice
statement, per-slice force table, reinforcement, method comparison,
search summary, probabilistic section, and the embedded figures.
"""

import pytest

from slope_stability import (
    SlopeGeometry, SlopeSoilLayer, Anchor, Geosynthetic, SoilNail,
    analyze_slope, search_critical_surface,
)
from slope_stability import calc_steps as cs
from calc_package.data_model import (
    CalcSection, CalcStep, TableData, CheckItem,
)


@pytest.fixture(scope="module")
def geom():
    return SlopeGeometry(
        surface_points=[(0, 20), (10, 20), (30, 10), (50, 10)],
        soil_layers=[
            SlopeSoilLayer("Fill", top_elevation=20, bottom_elevation=12,
                           gamma=19, phi=28, c_prime=5),
            SlopeSoilLayer("Clay", top_elevation=12, bottom_elevation=0,
                           gamma=18, phi=22, c_prime=12),
        ],
        gwt_points=[(0, 16), (30, 9), (50, 9)],
    )


@pytest.fixture(scope="module")
def geom_reinforced():
    return SlopeGeometry(
        surface_points=[(0, 20), (10, 20), (30, 10), (50, 10)],
        soil_layers=[
            SlopeSoilLayer("Clay", top_elevation=20, bottom_elevation=0,
                           gamma=18, phi=22, c_prime=12),
        ],
        nails=[SoilNail(x_head=18, z_head=16, length=8)],
        anchors=[Anchor(x_head=24, z_head=13, length=12, T_allow=80)],
        geosynthetics=[Geosynthetic(elevation=11, T_allow=40)],
    )


@pytest.fixture(scope="module")
def spencer_result(geom):
    return analyze_slope(geom, xc=22, yc=28, radius=20, method="spencer",
                         include_slice_data=True, compare_methods=True)


def _titles(sections):
    return [s.title for s in sections]


def _section(sections, title):
    for s in sections:
        if s.title == title:
            return s
    raise AssertionError(f"section '{title}' missing: {_titles(sections)}")


def _flat_text(section):
    out = []
    for item in section.items:
        if isinstance(item, CalcStep):
            out += [item.title, item.equation, item.notes,
                    str(item.result_value), item.result_name]
        elif isinstance(item, TableData):
            out += [item.title, item.notes] + item.headers
            for row in item.rows:
                out += [str(c) for c in row]
        elif isinstance(item, str):
            out.append(item)
    return " | ".join(out)


class TestMethodSections:
    def test_spencer_interslice_statement(self, spencer_result, geom):
        sections = cs.get_calc_steps(spencer_result, {"geom": geom})
        sec = _section(sections, "Analysis Method & Factor of Safety")
        text = _flat_text(sec)
        assert "Spencer" in text
        assert "Interslice Force Angle" in text
        assert "Line of Thrust" in text
        assert "Interslice Force Extremes" in text

    def test_morgenstern_price_lambda(self, geom):
        res = analyze_slope(geom, xc=22, yc=28, radius=20,
                            method="morgenstern_price",
                            include_slice_data=True)
        sections = cs.get_calc_steps(
            res, {"geom": geom, "f_interslice": "half_sine"})
        text = _flat_text(
            _section(sections, "Analysis Method & Factor of Safety"))
        assert "Morgenstern-Price" in text
        assert "half_sine" in text
        assert "λ" in text

    def test_janbu_f0(self, geom):
        res = analyze_slope(geom, xc=22, yc=28, radius=20, method="janbu")
        sections = cs.get_calc_steps(res, {"geom": geom})
        text = _flat_text(
            _section(sections, "Analysis Method & Factor of Safety"))
        assert "Janbu" in text
        assert "Correction" in text

    def test_bishop_unchanged(self, geom):
        res = analyze_slope(geom, xc=22, yc=28, radius=20, method="bishop")
        sections = cs.get_calc_steps(res, {"geom": geom})
        text = _flat_text(
            _section(sections, "Analysis Method & Factor of Safety"))
        assert "Bishop" in text


class TestSliceForceTable:
    def test_interslice_columns_for_rigorous(self, spencer_result, geom):
        sections = cs.get_calc_steps(spencer_result, {"geom": geom})
        sec = _section(sections, "Slice Force Table")
        table = sec.items[0]
        assert "E_L (kN/m)" in table.headers
        assert "X_R (kN/m)" in table.headers
        assert "N' (kN/m)" in table.headers
        assert len(table.rows) == spencer_result.n_slices

    def test_no_interslice_columns_for_fellenius(self, geom):
        res = analyze_slope(geom, xc=22, yc=28, radius=20,
                            method="fellenius", include_slice_data=True)
        sections = cs.get_calc_steps(res, {"geom": geom})
        table = _section(sections, "Slice Force Table").items[0]
        assert "E_L (kN/m)" not in table.headers
        assert "N' (kN/m)" in table.headers

    def test_subset_for_many_slices(self, geom):
        res = analyze_slope(geom, xc=22, yc=28, radius=20,
                            method="bishop", n_slices=60,
                            include_slice_data=True)
        table = _section(cs.get_calc_steps(res, {"geom": geom}),
                         "Slice Force Table").items[0]
        assert len(table.rows) < 60
        assert "of 60 slices" in table.notes


class TestReinforcementSection:
    def test_layout_and_mobilized_tables(self, geom_reinforced):
        res = analyze_slope(geom_reinforced, xc=22, yc=28, radius=20,
                            method="bishop", include_slice_data=True)
        sections = cs.get_calc_steps(res, {"geom": geom_reinforced})
        sec = _section(sections, "Reinforcement")
        text = _flat_text(sec)
        assert "Nail" in text
        assert "Anchor" in text
        assert "Geosynthetic" in text
        assert "T_allow" in text
        # mobilized table present (anchor + geosynthetic cross)
        assert res.reinforcements
        assert "Mobilized Reinforcement Forces" in text
        assert "Total stabilizing reinforcement force" in text

    def test_absent_for_unreinforced(self, spencer_result, geom):
        sections = cs.get_calc_steps(spencer_result, {"geom": geom})
        assert "Reinforcement" not in _titles(sections)


class TestComparisonSection:
    def test_all_methods_listed(self, spencer_result, geom):
        sections = cs.get_calc_steps(spencer_result, {"geom": geom})
        text = _flat_text(_section(sections, "Method Comparison"))
        assert "Fellenius" in text
        assert "Bishop" in text
        assert "Janbu" in text
        assert "Spencer" in text
        assert "Morgenstern-Price" in text

    def test_absent_without_compare(self, geom):
        res = analyze_slope(geom, xc=22, yc=28, radius=20, method="bishop")
        sections = cs.get_calc_steps(res, {"geom": geom})
        assert "Method Comparison" not in _titles(sections)


class TestSearchSection:
    def test_circular_search_summary(self, geom):
        search = search_critical_surface(geom, nx=3, ny=3,
                                         method="bishop", n_slices=20)
        res = analyze_slope(geom, xc=search.critical.xc,
                            yc=search.critical.yc,
                            radius=search.critical.radius)
        sections = cs.get_calc_steps(res, {"geom": geom, "search": search})
        text = _flat_text(_section(sections, "Critical Surface Search"))
        assert "Surfaces evaluated" in text
        assert "Critical circle center" in text

    def test_noncircular_vertices_table(self, geom):
        search = search_critical_surface(
            geom, surface_type="noncircular", n_trials=25, seed=3)
        crit = search.critical
        res = analyze_slope(geom, slip_surface=None, xc=22, yc=28,
                            radius=20)
        sections = cs.get_calc_steps(res, {"geom": geom, "search": search})
        text = _flat_text(_section(sections, "Critical Surface Search"))
        assert "Noncircular" in text
        assert "Critical Surface Vertices" in text


class TestProbabilisticSection:
    @pytest.fixture(scope="class")
    def prob(self, geom):
        from slope_stability import monte_carlo_fos, fosm_fos
        variables = {
            "phi:Clay": {"mean": 22, "cov": 0.10, "source": "CPT corr."},
            "c_prime:Clay": {"mean": 12, "cov": 0.30,
                             "dist": "lognormal"},
        }
        mc = monte_carlo_fos(geom, variables, xc=22, yc=28, radius=20,
                             n=80, seed=42)
        fosm = fosm_fos(geom, variables, xc=22, yc=28, radius=20)
        return variables, mc, fosm

    def test_full_section(self, geom, spencer_result, prob):
        variables, mc, fosm = prob
        sections = cs.get_calc_steps(spencer_result, {
            "geom": geom, "mc": mc, "fosm": fosm,
            "variables": variables})
        sec = _section(sections, "Probabilistic Analysis (Reliability)")
        text = _flat_text(sec)
        assert "Random Variables" in text
        assert "lognormal" in text
        assert "CPT corr." in text          # source column
        assert "Reliability Index (normal)" in text
        assert "Reliability Index (lognormal)" in text
        assert "Variance Contributions" in text
        assert "Monte Carlo FOS Distribution" in text
        assert "P(FOS < 1)" in text

    def test_absent_without_prob_inputs(self, spencer_result, geom):
        sections = cs.get_calc_steps(spencer_result, {"geom": geom})
        assert "Probabilistic Analysis (Reliability)" \
            not in _titles(sections)


class TestInputSummary:
    def test_interslice_function_echoed(self, geom):
        res = analyze_slope(geom, xc=22, yc=28, radius=20,
                            method="morgenstern_price")
        items = cs.get_input_summary(
            res, {"geom": geom, "f_interslice": "clipped_sine"})
        text = " ".join(f"{i.name} {i.value}" for i in items)
        assert "clipped_sine" in text

    def test_reinforcement_counts(self, geom_reinforced):
        res = analyze_slope(geom_reinforced, xc=22, yc=28, radius=20)
        items = cs.get_input_summary(res, {"geom": geom_reinforced})
        names = [i.name for i in items]
        assert "n_nail" in names
        assert "n_anch" in names
        assert "n_geo" in names


class TestFigures:
    def test_figure_set(self, geom, spencer_result):
        pytest.importorskip("matplotlib")
        from slope_stability import monte_carlo_fos, fosm_fos
        variables = {"phi:Clay": {"mean": 22, "cov": 0.10}}
        mc = monte_carlo_fos(geom, variables, xc=22, yc=28, radius=20,
                             n=50, seed=1)
        fosm = fosm_fos(geom, variables, xc=22, yc=28, radius=20)
        search = search_critical_surface(geom, nx=3, ny=3,
                                         method="bishop", n_slices=20)
        figs = cs.get_figures(spencer_result, {
            "geom": geom, "search": search, "mc": mc, "fosm": fosm})
        titles = [f.title for f in figs]
        assert "Slope Cross-Section" in titles
        assert "Trial Surface Map" in titles
        assert "Slice Force Distribution" in titles
        assert "Interslice Forces & Line of Thrust" in titles
        assert "Monte Carlo FOS Distribution" in titles
        assert "FOSM Variance Contributions" in titles
        for f in figs:
            assert len(f.image_base64) > 1000

    def test_minimal_figure_set(self, geom):
        pytest.importorskip("matplotlib")
        res = analyze_slope(geom, xc=22, yc=28, radius=20)
        figs = cs.get_figures(res, {"geom": geom})
        assert [f.title for f in figs] == ["Slope Cross-Section"]


class TestEndToEnd:
    def test_html_package(self, geom, spencer_result):
        pytest.importorskip("matplotlib")
        pytest.importorskip("jinja2")
        from calc_package import generate_calc_package
        html = generate_calc_package(
            "slope_stability", spencer_result, {"geom": geom},
            project_name="Test", engineer="Test")
        assert "Slice Force Table" in html
        assert "Method Comparison" in html
        assert "data:image/png;base64" in html

"""
Tests for calc_package module.

Tests the data model, renderer, and per-module calc_steps.
"""

import pytest
from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, TableData,
    CalcSection, CalcPackageData,
)
from calc_package.renderer import render_html, _preprocess_sections, _item_type
from calc_package import generate_calc_package, list_supported_modules


# ── Data model tests ──────────────────────────────────────────────

class TestInputItem:
    def test_basic(self):
        item = InputItem(name="B", description="Footing width", value=2.0, unit="m")
        assert item.name == "B"
        assert item.value == 2.0

    def test_no_unit(self):
        item = InputItem(name="FS", description="Factor of safety", value=3.0)
        assert item.unit == ""


class TestCalcStep:
    def test_basic(self):
        step = CalcStep(
            title="Nq Factor",
            equation="Nq = exp(pi*tan(phi)) * tan^2(45 + phi/2)",
            substitution="Nq = exp(pi*tan(30)) * tan^2(60)",
            result_name="Nq",
            result_value=18.40,
            reference="Vesic (1973)",
        )
        assert step.result_value == 18.40
        assert step.result_unit == ""

    def test_with_unit(self):
        step = CalcStep(
            title="Ultimate capacity",
            equation="qult = term_c + term_q + term_g",
            substitution="qult = 0 + 706.8 + 257.4",
            result_name="qult",
            result_value=964.2,
            result_unit="kPa",
        )
        assert step.result_unit == "kPa"


class TestCheckItem:
    def test_passing(self):
        check = CheckItem(
            description="Bearing capacity adequacy",
            demand=150.0,
            demand_label="q_applied",
            capacity=321.4,
            capacity_label="q_allowable",
            unit="kPa",
            passes=True,
        )
        assert check.passes is True

    def test_failing(self):
        check = CheckItem(
            description="Settlement limit",
            demand=50.0,
            demand_label="s_total",
            capacity=25.0,
            capacity_label="s_allowable",
            unit="mm",
            passes=False,
        )
        assert check.passes is False


class TestTableData:
    def test_basic(self):
        table = TableData(
            title="Layer Breakdown",
            headers=["Depth (m)", "phi (deg)", "c (kPa)"],
            rows=[[0.0, 30, 0], [3.0, 25, 10]],
        )
        assert len(table.rows) == 2
        assert table.headers[0] == "Depth (m)"


class TestCalcSection:
    def test_mixed_items(self):
        section = CalcSection(title="Test Section", items=[
            InputItem("B", "Width", 2.0, "m"),
            CalcStep("Step", "eq", "sub", "x", 1.0),
            "Some explanatory text.",
        ])
        assert len(section.items) == 3


class TestCalcPackageData:
    def test_defaults(self):
        data = CalcPackageData()
        assert data.project_name == "Project"
        assert data.sections == []
        assert data.references == []

    def test_full(self):
        data = CalcPackageData(
            project_name="Test Bridge",
            project_number="2024-001",
            analysis_type="Bearing Capacity",
            engineer="J. Doe",
            sections=[CalcSection(title="Inputs", items=[])],
            references=["Vesic (1973)"],
        )
        assert data.engineer == "J. Doe"
        assert len(data.references) == 1


# ── Renderer tests ────────────────────────────────────────────────

class TestItemType:
    def test_calc_step(self):
        assert _item_type(CalcStep("t", "e", "s", "r", 1.0)) == "calc_step"

    def test_check(self):
        assert _item_type(CheckItem("d", 1, "d", 2, "c")) == "check"

    def test_figure(self):
        assert _item_type(FigureData("t", "abc123")) == "figure"

    def test_table(self):
        assert _item_type(TableData("t")) == "table"

    def test_text(self):
        assert _item_type("hello") == "text"


class TestPreprocessSections:
    def test_input_items_become_table(self):
        section = CalcSection(title="Inputs", items=[
            InputItem("B", "Width", 2.0, "m"),
            InputItem("L", "Length", 3.0, "m"),
        ])
        processed = _preprocess_sections([section])
        assert len(processed) == 1
        assert len(processed[0].items) == 1  # collapsed into one table
        assert _item_type(processed[0].items[0]) == "input_table"

    def test_mixed_items_preserved(self):
        section = CalcSection(title="Mixed", items=[
            InputItem("B", "Width", 2.0, "m"),
            CalcStep("Step", "eq", "sub", "x", 1.0),
            InputItem("L", "Length", 3.0, "m"),
        ])
        processed = _preprocess_sections([section])
        items = processed[0].items
        # First InputItem → table, then CalcStep, then second InputItem → table
        assert len(items) == 3
        assert _item_type(items[0]) == "input_table"
        assert _item_type(items[1]) == "calc_step"
        assert _item_type(items[2]) == "input_table"


class TestRenderHtml:
    def test_basic_render(self):
        data = CalcPackageData(
            project_name="Test Project",
            analysis_type="Test Analysis",
            engineer="Tester",
            date="2026-01-01",
            sections=[
                CalcSection(title="Inputs", items=[
                    InputItem("B", "Width", 2.0, "m"),
                ]),
                CalcSection(title="Calc", items=[
                    CalcStep("Step 1", "x = a + b", "x = 1 + 2",
                             "x", 3.0, "", "Ref 1"),
                ]),
            ],
            references=["Reference 1"],
        )
        html = render_html(data)
        assert "<!DOCTYPE html>" in html
        assert "Test Project" in html
        assert "Test Analysis" in html
        assert "Tester" in html
        assert "Width" in html
        assert "2.0" in html
        assert "Step 1" in html
        assert "x = a + b" in html
        assert "x = 1 + 2" in html
        assert "3.0" in html
        assert "Reference 1" in html

    def test_check_item_render(self):
        data = CalcPackageData(
            project_name="P",
            analysis_type="A",
            date="2026-01-01",
            sections=[CalcSection(title="Checks", items=[
                CheckItem("BC check", 100, "q_app", 300, "q_all", "kPa", True),
            ])],
        )
        html = render_html(data)
        assert "PASS" in html
        assert "q_app" in html
        assert "q_all" in html

    def test_figure_render(self):
        data = CalcPackageData(
            project_name="P",
            analysis_type="A",
            date="2026-01-01",
            sections=[CalcSection(title="Figs", items=[
                FigureData("Test Fig", "AAAA", "Figure 1: Test"),
            ])],
        )
        html = render_html(data)
        assert "data:image/png;base64,AAAA" in html
        assert "Figure 1: Test" in html

    def test_table_render(self):
        data = CalcPackageData(
            project_name="P",
            analysis_type="A",
            date="2026-01-01",
            sections=[CalcSection(title="Tables", items=[
                TableData("Layer Data", ["Depth", "phi"], [[0, 30], [3, 25]]),
            ])],
        )
        html = render_html(data)
        assert "Layer Data" in html
        assert "Depth" in html

    def test_auto_date(self):
        data = CalcPackageData(project_name="P", analysis_type="A")
        html = render_html(data)
        assert "2026" in html  # auto-filled today's date


class TestFigureToBase64:
    def test_converts_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from calc_package.renderer import figure_to_base64

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        b64 = figure_to_base64(fig)
        plt.close(fig)

        assert isinstance(b64, str)
        assert len(b64) > 100  # should be a valid base64 string
        # Verify it's valid base64
        import base64
        decoded = base64.b64decode(b64)
        assert decoded[:4] == b'\x89PNG'  # PNG header


# ── Module registry tests ─────────────────────────────────────────

class TestSupportedModules:
    def test_list(self):
        modules = list_supported_modules()
        assert "bearing_capacity" in modules
        assert "lateral_pile" in modules
        assert "slope_stability" in modules

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="does not have calc package support"):
            generate_calc_package("nonexistent_module", None)


# ── Integration tests (end-to-end with real modules) ──────────────

class TestBearingCapacityIntegration:
    def test_sand_footing(self):
        from bearing_capacity.footing import Footing
        from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer
        from bearing_capacity.capacity import BearingCapacityAnalysis

        footing = Footing(width=2.0, depth=1.5, shape="square")
        soil = BearingSoilProfile(layer1=SoilLayer(friction_angle=30, unit_weight=18))
        analysis = BearingCapacityAnalysis(footing=footing, soil=soil)
        result = analysis.compute()

        html = generate_calc_package(
            module="bearing_capacity",
            result=result,
            analysis=analysis,
            project_name="Test",
            engineer="Test",
        )
        assert "<!DOCTYPE html>" in html
        assert "Bearing Capacity" in html
        assert "N_q" in html
        assert "data:image/png;base64" in html
        assert len(html) > 10000

    def test_clay_footing(self):
        from bearing_capacity.footing import Footing
        from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer
        from bearing_capacity.capacity import BearingCapacityAnalysis

        footing = Footing(width=1.5, length=3.0, depth=1.0, shape="rectangular")
        soil = BearingSoilProfile(
            layer1=SoilLayer(cohesion=50, friction_angle=0, unit_weight=17),
        )
        analysis = BearingCapacityAnalysis(footing=footing, soil=soil, factor_of_safety=2.5)
        result = analysis.compute()

        html = generate_calc_package("bearing_capacity", result, analysis)
        assert "Prandtl" in html  # phi=0 uses Prandtl Nc=5.14
        assert "5.14" in html


class TestLateralPileIntegration:
    def test_pipe_pile(self):
        from lateral_pile import Pile, SoilLayer, LateralPileAnalysis
        from lateral_pile.py_curves import SoftClayMatlock, SandAPI

        pile = Pile(length=20.0, diameter=0.6, thickness=0.012, E=200e6)
        layers = [
            SoilLayer(top=0.0, bottom=5.0,
                      py_model=SoftClayMatlock(c=25, gamma=8, eps50=0.02, J=0.5)),
            SoilLayer(top=5.0, bottom=20.0,
                      py_model=SandAPI(phi=35, gamma=10, k=16000)),
        ]
        analysis = LateralPileAnalysis(pile, layers)
        result = analysis.solve(Vt=100, Mt=0, Q=500, head_condition='free')

        html = generate_calc_package("lateral_pile", result, analysis,
                                     project_name="Test", engineer="Test")
        assert "Lateral Pile" in html
        assert "Deflection" in html or "deflection" in html
        assert "data:image/png;base64" in html
        assert len(html) > 50000  # should have substantial content + figures


class TestSlopeStabilityIntegration:
    def test_simple_slope(self):
        from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
        from slope_stability.analysis import analyze_slope

        surface = [(0, 20), (10, 20), (30, 10), (50, 10)]
        layers = [
            SlopeSoilLayer('Clay', top_elevation=20, bottom_elevation=0,
                           gamma=18, phi=25, c_prime=10),
        ]
        geom = SlopeGeometry(surface_points=surface, soil_layers=layers)
        result = analyze_slope(geom, xc=20, yc=25, radius=18,
                               include_slice_data=True)

        analysis_dict = {"geom": geom}
        html = generate_calc_package("slope_stability", result, analysis_dict,
                                     project_name="Test", engineer="Test")
        assert "Slope Stability" in html
        assert "FOS" in html
        assert "data:image/png;base64" in html
        assert "Slip" in html or "slip" in html

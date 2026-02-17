"""
Tests for the LaTeX calc package renderer.

Tests equation converter, LaTeX renderer, and compiler wrapper.
PDF compilation tests are conditional on pdflatex being available.
"""

import shutil
import pytest

from calc_package.equation_converter import unicode_to_latex, escape_latex
from calc_package.latex_compiler import find_latex_compiler
from calc_package.latex_renderer import render_latex, save_latex, _prepare_item
from calc_package.data_model import (
    CalcPackageData, CalcSection, CalcStep, InputItem,
    CheckItem, FigureData, TableData,
)
from calc_package import generate_calc_package


# ── Equation converter: Greek letters ────────────────────────────────

class TestUnicodeGreek:
    def test_phi(self):
        assert r"\varphi" in unicode_to_latex("\u03c6")

    def test_gamma(self):
        assert r"\gamma" in unicode_to_latex("\u03b3")

    def test_alpha(self):
        assert r"\alpha" in unicode_to_latex("\u03b1")

    def test_beta(self):
        assert r"\beta" in unicode_to_latex("\u03b2")

    def test_sigma(self):
        assert r"\sigma" in unicode_to_latex("\u03c3")

    def test_pi(self):
        assert r"\pi" in unicode_to_latex("\u03c0")

    def test_delta(self):
        assert r"\delta" in unicode_to_latex("\u03b4")

    def test_theta(self):
        assert r"\theta" in unicode_to_latex("\u03b8")

    def test_uppercase_delta(self):
        assert r"\Delta" in unicode_to_latex("\u0394")

    def test_uppercase_sigma(self):
        assert r"\Sigma" in unicode_to_latex("\u03a3")


# ── Equation converter: operators and symbols ────────────────────────

class TestUnicodeOperators:
    def test_multiplication(self):
        assert r"\times" in unicode_to_latex("a \u00d7 b")

    def test_center_dot(self):
        assert r"\cdot" in unicode_to_latex("a \u00b7 b")

    def test_plus_minus(self):
        assert r"\pm" in unicode_to_latex("\u00b1")

    def test_leq(self):
        assert r"\leq" in unicode_to_latex("x \u2264 5")

    def test_geq(self):
        assert r"\geq" in unicode_to_latex("x \u2265 5")

    def test_approx(self):
        assert r"\approx" in unicode_to_latex("\u2248")

    def test_degree(self):
        result = unicode_to_latex("45\u00b0")
        assert r"^{\circ}" in result


# ── Equation converter: superscripts ─────────────────────────────────

class TestUnicodeSuperscripts:
    def test_squared(self):
        result = unicode_to_latex("x\u00b2")
        assert "^{2}" in result

    def test_cubed(self):
        result = unicode_to_latex("D\u00b3")
        assert "^{3}" in result

    def test_fourth(self):
        result = unicode_to_latex("z\u2074")
        assert "^{4}" in result


# ── Equation converter: subscripts ───────────────────────────────────

class TestSubscripts:
    def test_single_char(self):
        result = unicode_to_latex("N_q")
        assert "N_{q}" in result

    def test_multi_char(self):
        result = unicode_to_latex("q_ult")
        assert "q_{ult}" in result

    def test_sigma_v(self):
        # σ'_v should become \sigma '_{v}
        result = unicode_to_latex("\u03c3'_v")
        assert r"\sigma" in result


# ── Equation converter: square root ──────────────────────────────────

class TestSqrt:
    def test_sqrt_parens(self):
        result = unicode_to_latex("\u221a(Ka)")
        assert r"\sqrt{Ka}" in result

    def test_sqrt_complex(self):
        result = unicode_to_latex("\u221a(q_u \u00d7 p_a)")
        assert r"\sqrt{" in result
        assert r"\times" in result

    def test_sqrt_single_token(self):
        result = unicode_to_latex("\u221aKa")
        assert r"\sqrt{Ka}" in result


# ── Equation converter: trig functions ───────────────────────────────

class TestTrigFunctions:
    def test_tan(self):
        result = unicode_to_latex("tan(x)")
        assert r"\tan" in result

    def test_sin(self):
        result = unicode_to_latex("sin(x)")
        assert r"\sin" in result

    def test_cos(self):
        result = unicode_to_latex("cos(x)")
        assert r"\cos" in result

    def test_exp(self):
        result = unicode_to_latex("exp(x)")
        assert r"\exp" in result

    def test_arctan(self):
        result = unicode_to_latex("arctan(x)")
        assert r"\arctan" in result

    def test_min(self):
        result = unicode_to_latex("min(a, b)")
        assert r"\min" in result

    def test_max(self):
        result = unicode_to_latex("max(a, b)")
        assert r"\max" in result


# ── Equation converter: full equations ───────────────────────────────

class TestFullEquations:
    def test_bearing_capacity_nq(self):
        eq = "N_q = exp(\u03c0 \u00d7 tan(\u03c6)) \u00d7 tan\u00b2(45 + \u03c6/2)"
        result = unicode_to_latex(eq)
        assert r"\exp" in result
        assert r"\pi" in result
        assert r"\tan" in result
        assert "^{2}" in result
        assert r"\varphi" in result

    def test_drilled_shaft_rock_socket(self):
        eq = "f_s = C \u00d7 \u03b1_E \u00d7 \u221a(q_u \u00d7 p_a)"
        result = unicode_to_latex(eq)
        assert r"\times" in result
        assert r"\alpha" in result
        assert r"\sqrt{" in result

    def test_drilled_shaft_beta(self):
        eq = "\u03b2 = 1.5 - 0.245\u00d7\u221a(z\u00d73.281)"
        result = unicode_to_latex(eq)
        assert r"\beta" in result
        assert r"\sqrt{" in result

    def test_multiline_equation(self):
        eq = "line1\nline2\nline3"
        result = unicode_to_latex(eq)
        assert r"\\" in result

    def test_empty_string(self):
        assert unicode_to_latex("") == ""

    def test_plain_ascii(self):
        result = unicode_to_latex("x = 2 + 3")
        assert "x = 2 + 3" == result


# ── Escape LaTeX ─────────────────────────────────────────────────────

class TestEscapeLatex:
    def test_percent(self):
        assert r"\%" in escape_latex("50%")

    def test_ampersand(self):
        assert r"\&" in escape_latex("A & B")

    def test_hash(self):
        assert r"\#" in escape_latex("#1")

    def test_underscore(self):
        assert r"\_" in escape_latex("q_ult")

    def test_dollar(self):
        assert r"\$" in escape_latex("$100")

    def test_braces(self):
        assert r"\{" in escape_latex("{x}")
        assert r"\}" in escape_latex("{x}")

    def test_empty(self):
        assert escape_latex("") == ""

    def test_greek_in_text(self):
        result = escape_latex("Friction angle \u03c6")
        assert r"$\varphi$" in result


# ── Compiler tests ───────────────────────────────────────────────────

class TestFindCompiler:
    def test_missing_compiler_raises(self):
        with pytest.raises(FileNotFoundError, match="not found on PATH"):
            find_latex_compiler("nonexistent_latex_compiler_xyz")

    def test_helpful_error_message(self):
        with pytest.raises(FileNotFoundError, match="MiKTeX"):
            find_latex_compiler("nonexistent_compiler")


# ── Renderer unit tests ──────────────────────────────────────────────

class TestRenderLatex:
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
            ],
        )
        tex = render_latex(data)
        assert r"\documentclass" in tex
        assert "Test Project" in tex
        assert "Test Analysis" in tex
        assert "Tester" in tex
        assert "Width" in tex

    def test_calc_step_render(self):
        data = CalcPackageData(
            project_name="P",
            analysis_type="A",
            date="2026-01-01",
            sections=[CalcSection(title="Calc", items=[
                CalcStep("Step 1", "x = a + b", "x = 1 + 2",
                         "x", 3.0, "m", "Reference"),
            ])],
        )
        tex = render_latex(data)
        assert "Step 1" in tex
        assert "x = a + b" in tex
        assert "3.0" in tex
        assert "Reference" in tex

    def test_check_item_pass(self):
        data = CalcPackageData(
            project_name="P",
            analysis_type="A",
            date="2026-01-01",
            sections=[CalcSection(title="Checks", items=[
                CheckItem("BC check", 100, "q_app", 300, "q_all", "kPa", True),
            ])],
        )
        tex = render_latex(data)
        assert "PASS" in tex
        assert "PassGreen" in tex

    def test_check_item_fail(self):
        data = CalcPackageData(
            project_name="P",
            analysis_type="A",
            date="2026-01-01",
            sections=[CalcSection(title="Checks", items=[
                CheckItem("BC check", 400, "q_app", 300, "q_all", "kPa", False),
            ])],
        )
        tex = render_latex(data)
        assert "FAIL" in tex
        assert "FailRed" in tex

    def test_table_render(self):
        data = CalcPackageData(
            project_name="P",
            analysis_type="A",
            date="2026-01-01",
            sections=[CalcSection(title="T", items=[
                TableData("Data", ["Col1", "Col2"], [[1, 2], [3, 4]]),
            ])],
        )
        tex = render_latex(data)
        assert r"\begin{longtable}" in tex
        assert r"\toprule" in tex
        assert "Col1" in tex
        assert "Col2" in tex

    def test_text_render(self):
        data = CalcPackageData(
            project_name="P",
            analysis_type="A",
            date="2026-01-01",
            sections=[CalcSection(title="Notes", items=[
                "Some explanatory text.",
            ])],
        )
        tex = render_latex(data)
        assert "Some explanatory text." in tex

    def test_references_render(self):
        data = CalcPackageData(
            project_name="P",
            analysis_type="A",
            date="2026-01-01",
            sections=[],
            references=["Vesic (1973)", "Meyerhof (1963)"],
        )
        tex = render_latex(data)
        assert "Vesic (1973)" in tex
        assert "Meyerhof (1963)" in tex

    def test_auto_date(self):
        data = CalcPackageData(project_name="P", analysis_type="A")
        tex = render_latex(data)
        assert "2026" in tex

    def test_unicode_equation_in_step(self):
        """Verify Unicode equations get converted to LaTeX math."""
        data = CalcPackageData(
            project_name="P",
            analysis_type="A",
            date="2026-01-01",
            sections=[CalcSection(title="Calc", items=[
                CalcStep(
                    "Nq Factor",
                    "N_q = exp(\u03c0 \u00d7 tan(\u03c6)) \u00d7 tan\u00b2(45 + \u03c6/2)",
                    "",
                    "N_q",
                    18.40,
                ),
            ])],
        )
        tex = render_latex(data)
        assert r"\exp" in tex
        assert r"\pi" in tex
        assert r"\varphi" in tex

    def test_special_chars_escaped_in_title(self):
        data = CalcPackageData(
            project_name="Test & Demo #1",
            analysis_type="50% Capacity Check",
            date="2026-01-01",
            sections=[],
        )
        tex = render_latex(data)
        assert r"\&" in tex
        assert r"\%" in tex
        assert r"\#" in tex


# ── Prepare item tests ───────────────────────────────────────────────

class TestPrepareItem:
    def test_calc_step_type(self):
        step = CalcStep("Title", "eq", "sub", "x", 1.0)
        ns = _prepare_item(step)
        assert ns._type == "calc_step"
        assert ns.title == "Title"

    def test_check_type(self):
        check = CheckItem("desc", 100, "d", 200, "c", "kPa", True)
        ns = _prepare_item(check)
        assert ns._type == "check"
        assert ns.passes is True
        assert ns.dc_ratio == "0.50"

    def test_text_type(self):
        ns = _prepare_item("hello")
        assert ns._type == "text"
        assert ns.text == "hello"


# ── Integration: generate_calc_package with format="latex" ───────────

class TestGenerateLatex:
    def test_bearing_capacity_latex(self):
        from bearing_capacity.footing import Footing
        from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer
        from bearing_capacity.capacity import BearingCapacityAnalysis

        footing = Footing(width=2.0, depth=1.5, shape="square")
        soil = BearingSoilProfile(layer1=SoilLayer(friction_angle=30, unit_weight=18))
        analysis = BearingCapacityAnalysis(footing=footing, soil=soil)
        result = analysis.compute()

        tex = generate_calc_package(
            module="bearing_capacity",
            result=result,
            analysis=analysis,
            project_name="Test",
            engineer="Test",
            format="latex",
        )
        assert r"\documentclass" in tex
        assert "Bearing Capacity" in tex
        assert r"\begin{equation*}" in tex
        assert len(tex) > 2000

    def test_format_default_is_html(self):
        """Existing default produces HTML."""
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

    def test_pdf_requires_output_path(self):
        with pytest.raises(ValueError, match="output_path is required"):
            generate_calc_package(
                module="bearing_capacity",
                result=None,
                format="pdf",
            )


# ── Save latex to file ───────────────────────────────────────────────

class TestSaveLatex:
    def test_save_to_file(self, tmp_path):
        data = CalcPackageData(
            project_name="Save Test",
            analysis_type="Test",
            date="2026-01-01",
            sections=[CalcSection(title="Inputs", items=[
                InputItem("B", "Width", 2.0, "m"),
            ])],
        )
        tex_path = str(tmp_path / "test.tex")
        result = save_latex(data, tex_path)
        assert result.endswith("test.tex")
        content = (tmp_path / "test.tex").read_text()
        assert r"\documentclass" in content


# ── PDF compilation (conditional) ────────────────────────────────────

_HAS_PDFLATEX = shutil.which("pdflatex") is not None


@pytest.mark.skipif(not _HAS_PDFLATEX, reason="pdflatex not installed")
class TestPdfCompilation:
    def test_basic_pdf(self, tmp_path):
        from calc_package.latex_renderer import save_pdf

        data = CalcPackageData(
            project_name="PDF Test",
            analysis_type="Test Analysis",
            engineer="Tester",
            date="2026-01-01",
            sections=[CalcSection(title="Inputs", items=[
                InputItem("B", "Width", 2.0, "m"),
            ])],
        )
        pdf_path = str(tmp_path / "test.pdf")
        result = save_pdf(data, pdf_path)
        from pathlib import Path
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_keep_tex(self, tmp_path):
        from calc_package.latex_renderer import save_pdf

        data = CalcPackageData(
            project_name="Keep Test",
            analysis_type="Test",
            date="2026-01-01",
            sections=[],
        )
        pdf_path = str(tmp_path / "keep.pdf")
        save_pdf(data, pdf_path, keep_tex=True)
        from pathlib import Path
        assert Path(tmp_path / "keep.tex").exists()

    def test_bearing_capacity_pdf(self, tmp_path):
        from bearing_capacity.footing import Footing
        from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer
        from bearing_capacity.capacity import BearingCapacityAnalysis

        footing = Footing(width=2.0, depth=1.5, shape="square")
        soil = BearingSoilProfile(layer1=SoilLayer(friction_angle=30, unit_weight=18))
        analysis = BearingCapacityAnalysis(footing=footing, soil=soil)
        result = analysis.compute()

        pdf_path = str(tmp_path / "bearing.pdf")
        output = generate_calc_package(
            module="bearing_capacity",
            result=result,
            analysis=analysis,
            project_name="Test",
            engineer="Test",
            output_path=pdf_path,
            format="pdf",
        )
        from pathlib import Path
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

"""A canned calc package's figures print with lettering at 7 pt or more
(BACKLOG GM-186), and a PDF package says when one does not.

Every package figure is drawn for the width the package prints it at
(``calc_package.renderer.figure_to_base64``); a figure that still prints
under the plotting box's 7 pt is warned on in the tool's result, so a
regression shows.
"""

import pytest

pytest.importorskip("pymupdf")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

BEARING = {"width": 2.0, "unit_weight": 18.0, "friction_angle": 30.0,
           "depth": 1.5, "shape": "square", "format": "pdf"}

LATERAL = {"pile_length": 15.0, "pile_diameter": 0.5, "pile_E": 200e6,
           "Vt": 100.0, "format": "pdf",
           "soil_layers": [{"top": 0, "bottom": 15, "py_model": "SandReese",
                            "phi": 35, "gamma": 18.0, "k": 30000}]}


@pytest.mark.parametrize("package, params", [
    ("_generate_bearing_capacity_package", BEARING),
    ("_generate_lateral_pile_package", LATERAL),
])
def test_a_pdf_package_prints_its_figures_at_7_pt(tmp_path, monkeypatch,
                                                  package, params):
    from funhouse_agent.adapters import calc_package as adapter
    from calc_package import latex_renderer
    seen = {}
    real = latex_renderer.render_html_pdf_story

    def spy(html, filepath):
        out = real(html, filepath)
        seen["figures"] = out["figures"]
        return out

    monkeypatch.setattr(latex_renderer, "render_html_pdf_story", spy)
    result = getattr(adapter, package)(
        {**params, "output_path": str(tmp_path / "pkg.pdf")})
    assert result["status"] == "success", result.get("error")
    if result.get("renderer") != "pymupdf_story":
        pytest.skip("pdflatex prints the figures; the Story engine is tested")
    letterings = [f["lettering_pt"] for f in seen["figures"]
                  if "lettering_pt" in f]
    assert letterings and min(letterings) >= 7.0, seen["figures"]
    assert "figure_warnings" not in result


def test_a_package_figure_printed_too_small_is_warned(tmp_path, monkeypatch):
    """The regression the warning is for: package figures saved as drawn
    (not fitted to the print width) print under 7 pt, and the result
    says which."""
    from funhouse_agent.adapters import calc_package as adapter
    from geotech_common import layout_ink
    monkeypatch.setattr(layout_ink, "draw_for_print",
                        lambda fig, width_in, **kw: {})
    result = adapter._generate_lateral_pile_package(
        {**LATERAL, "output_path": str(tmp_path / "lp.pdf")})
    assert result["status"] == "success", result.get("error")
    if result.get("renderer") != "pymupdf_story":
        pytest.skip("pdflatex prints the figures; the Story engine is tested")
    warnings = result["figure_warnings"]
    assert warnings and all("under 7 pt" in w for w in warnings)
    # the module's figure: reported, not the caller's to redraw
    assert all("package's own figure" in w and "not yours to redraw" in w
               for w in warnings)

"""html_to_pdf figure-embedding contract.

Covers the 2026-09-04 field feedback (Praia downdrag): the renderer
silently dropped figures it could not embed, so a report PDF shipped with a
literal "[image]" placeholder where the profile figure belonged. Now a real
local PNG path is embedded for the caller, and anything still unembeddable
is a loud, itemised error.
"""

import base64

import pytest

from funhouse_agent.adapters.calc_package import _generate_html_to_pdf
from funhouse_agent.adapters.profile_figure_adapter import (
    _run_subsurface_profile)

LAYERS = [
    {"name": "Fill", "thickness": 3.0, "description": "N=5"},
    {"name": "Soft clay", "thickness": 9.0, "settling": True},
    {"name": "Dense sand", "thickness": 8.0},
]


def _doc(body: str) -> str:
    return f"<html><body><h1>Report</h1>{body}</body></html>"


# ---------------------------------------------------------------------------
# html_to_pdf — loud failure on figures it cannot embed
# ---------------------------------------------------------------------------

class TestHtmlToPdfLoudFailure:
    def test_inline_svg_is_refused_by_name(self):
        res = _generate_html_to_pdf({
            "html": _doc('<svg width="200" height="100"><rect/></svg>')})
        assert res["status"] == "error"
        assert "<svg" in res["error"]
        assert "not rendered" in res["error"].lower()
        assert "base64" in res["error"]
        assert len(res["unembeddable"]) == 1

    def test_image_placeholder_text_is_refused(self):
        res = _generate_html_to_pdf({
            "html": _doc("<p>[image]</p><p>Figure 1 — profile</p>")})
        assert res["status"] == "error"
        assert "[image]" in res["error"]
        assert "placeholder is not a figure" in res["error"]

    def test_remote_image_src_is_refused(self):
        res = _generate_html_to_pdf({
            "html": _doc('<img src="https://example.com/fig.png">')})
        assert res["status"] == "error"
        assert "example.com" in res["error"]
        assert "network" in res["error"]

    def test_missing_local_image_names_the_path(self, tmp_path):
        missing = tmp_path / "nope.png"
        res = _generate_html_to_pdf({
            "html": _doc(f'<img src="{missing}">')})
        assert res["status"] == "error"
        assert "no such file" in res["error"]
        assert "nope.png" in res["error"]

    def test_svg_file_src_is_refused(self, tmp_path):
        svg = tmp_path / "fig.svg"
        svg.write_text("<svg/>", encoding="utf-8")
        res = _generate_html_to_pdf({"html": _doc(f'<img src="{svg}">')})
        assert res["status"] == "error"
        assert "not an embeddable image format" in res["error"]

    def test_every_offender_is_listed(self, tmp_path):
        res = _generate_html_to_pdf({"html": _doc(
            '<svg></svg><img src="https://x.com/a.png"><p>[figure]</p>')})
        assert res["status"] == "error"
        assert len(res["unembeddable"]) == 3
        assert res["error"].startswith("html_to_pdf: 3 figure reference(s)")

    def test_error_points_at_the_profile_figure_tool(self):
        res = _generate_html_to_pdf({"html": _doc("<p>[image]</p>")})
        assert "profile_figure" in res["error"]
        assert "allow_unembeddable" in res["error"]

    def test_figure_cross_reference_is_not_a_placeholder(self, tmp_path):
        """'[Figure 3]' is a citation, not a missing figure — no false alarm."""
        pytest.importorskip("fitz")
        res = _generate_html_to_pdf({
            "html": _doc("<p>See [Figure 3] and [Table 2].</p>"),
            "output_path": str(tmp_path / "r.pdf")})
        assert res["status"] == "success"

    def test_allow_unembeddable_renders_and_warns(self, tmp_path):
        pytest.importorskip("fitz")
        res = _generate_html_to_pdf({
            "html": _doc("<p>[image]</p>"), "allow_unembeddable": True,
            "output_path": str(tmp_path / "r.pdf")})
        assert res["status"] == "success"
        assert res["file_exists"] is True
        assert res["unembeddable"]
        assert "missing from the PDF" in res["renderer_warnings"][0]

    def test_base64_data_uri_still_passes(self, tmp_path):
        pytest.importorskip("fitz")
        png = base64.b64encode(_tiny_png()).decode("ascii")
        res = _generate_html_to_pdf({
            "html": _doc(f'<img src="data:image/png;base64,{png}">'),
            "output_path": str(tmp_path / "r.pdf")})
        assert res["status"] == "success"
        assert res.get("images_embedded") is None  # nothing to inline

    def test_unknown_param_still_rejected(self):
        with pytest.raises(ValueError, match="html_to_pdf"):
            _generate_html_to_pdf({"html": "<p>x</p>", "renderer": "latex"})


class TestHtmlToPdfEmbedsLocalImages:
    def test_local_png_path_is_inlined(self, tmp_path):
        pytest.importorskip("fitz")
        png = tmp_path / "fig.png"
        png.write_bytes(_tiny_png())
        out = tmp_path / "r.pdf"
        res = _generate_html_to_pdf({"html": _doc(f'<img src="{png}">'),
                                     "output_path": str(out)})
        assert res["status"] == "success"
        assert res["images_embedded"] == 1
        assert res["embedded_images"] == [str(png)]
        assert out.read_bytes()[:5] == b"%PDF-"

    def test_relative_src_resolves_in_the_working_folder(self, tmp_path,
                                                         monkeypatch):
        pytest.importorskip("fitz")
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        (tmp_path / "fig.png").write_bytes(_tiny_png())
        res = _generate_html_to_pdf({"html": _doc('<img src="fig.png">'),
                                     "output_path": str(tmp_path / "r.pdf")})
        assert res["status"] == "success"
        assert res["images_embedded"] == 1

    def test_relative_src_resolves_beside_the_html_file(self, tmp_path):
        pytest.importorskip("fitz")
        (tmp_path / "fig.png").write_bytes(_tiny_png())
        page = tmp_path / "report.html"
        page.write_text(_doc('<img src="fig.png">'), encoding="utf-8")
        res = _generate_html_to_pdf({"html_path": str(page),
                                     "output_path": str(tmp_path / "r.pdf")})
        assert res["status"] == "success"
        assert res["embedded_images"] == [str(tmp_path / "fig.png")]

    def test_profile_figure_output_flows_into_a_pdf_report(self, tmp_path):
        """The workflow the field session could not do: draw it, then embed it."""
        pytest.importorskip("fitz")
        fig = _run_subsurface_profile({
            "layers": LAYERS, "water_depth": 2.0,
            "foundation": {"type": "micropile", "diameter": 0.25,
                           "tip_depth": 18.0, "label": "MCAC micropile"},
            "annotations": [{"depth": 11.0, "text": "Neutral plane"}],
            "title": "Micropile — subsurface profile",
            "output_path": str(tmp_path / "profile.png")})
        assert fig["status"] == "success"

        out = tmp_path / "report.pdf"
        res = _generate_html_to_pdf({
            "html": _doc("<h2>Subsurface profile</h2>" + fig["html_img_tag"]),
            "output_path": str(out)})
        assert res["status"] == "success"
        assert res["images_embedded"] == 1
        # The figure really rides inside the PDF, not as a placeholder.
        assert out.stat().st_size > fig["file_size_bytes"] * 0.5


def _tiny_png() -> bytes:
    """A 1x1 PNG — enough for the embed path without a plotting round-trip."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
        "hQGAhKmMIQAAAABJRU5ErkJggg==")

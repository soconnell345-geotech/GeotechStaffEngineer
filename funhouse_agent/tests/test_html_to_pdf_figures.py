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


# ---------------------------------------------------------------------------
# html_to_pdf — what each figure prints at (BACKLOG GM-168)
# ---------------------------------------------------------------------------

TEXT = "<p>" + ("The bearing resistance of the footing is checked against "
                "the factored load for each load case. ") * 6 + "</p>"


def _plot(tmp_path, name, width_in, height_in, fontsize=9.0):
    """A PNG saved the way the calc packages save theirs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from calc_package.renderer import figure_to_base64
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("Distance (m)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    try:
        png = base64.b64decode(figure_to_base64(fig, dpi=100))
    finally:
        plt.close(fig)
    path = tmp_path / name
    path.write_bytes(png)
    return path


class TestHtmlToPdfFigureSizes:
    def test_page_contents_count_from_one_as_pdf_page(self, tmp_path):
        """GM-154's convention: ``pdf_page`` counts from 1, ``page`` from
        0; a 1-based number is never called ``page``."""
        pytest.importorskip("fitz")
        _plot(tmp_path, "a.png", 6.5, 4)
        res = _generate_html_to_pdf({
            "html": _doc(f'{TEXT}<img src="{tmp_path / "a.png"}" '
                         'style="width:100%">'),
            "output_path": str(tmp_path / "r.pdf")})
        assert res["status"] == "success", res.get("error")
        rows = res["page_contents"]
        assert [r["pdf_page"] for r in rows] == list(range(1, len(rows) + 1))
        assert all("page" not in r for r in rows)

    def test_each_figure_reports_its_printed_size_and_lettering(
            self, tmp_path):
        pytest.importorskip("fitz")
        _plot(tmp_path, "a.png", 6.5, 4, fontsize=10)
        res = _generate_html_to_pdf({
            "html": _doc(f'<img src="{tmp_path / "a.png"}" '
                         'style="width:100%">'),
            "output_path": str(tmp_path / "r.pdf")})
        (fig,) = res["page_contents"][0]["figures"]
        assert fig["figure"] == 1 and fig["width_in"] > 6.0
        assert fig["height_in"] > 3.0
        # 10 pt on a figure saved about 5.5 in wide, printed wider
        assert 10.0 < fig["lettering_pt"] < 14.0
        assert "figure_warnings" not in res

    def test_a_figure_at_a_page_foot_moves_to_the_next_page(self, tmp_path):
        """The live case: a figure after most of a page of text printed
        0.3 in wide; it now starts the next page at its width."""
        pytest.importorskip("fitz")
        _plot(tmp_path, "wall.png", 6.5, 5)
        res = _generate_html_to_pdf({
            "html": _doc(TEXT * 4 + f'<img src="{tmp_path / "wall.png"}" '
                         'style="width:100%"><p>Figure 1: Wall</p>'),
            "output_path": str(tmp_path / "r.pdf")})
        on_page = {r["pdf_page"]: r.get("figures") for r in
                   res["page_contents"]}
        assert not on_page[1]
        (fig,) = on_page[2]
        assert fig["width_in"] > 6.0 and fig.get("moved")

    def test_a_narrow_figure_with_small_lettering_is_warned(self, tmp_path):
        pytest.importorskip("fitz")
        _plot(tmp_path, "small.png", 10, 4, fontsize=8)
        res = _generate_html_to_pdf({
            "html": _doc(f'<img src="{tmp_path / "small.png"}" '
                         'style="width:35%">'),
            "output_path": str(tmp_path / "r.pdf")})
        (fig,) = res["page_contents"][0]["figures"]
        assert fig["width_in"] < 3.0 and fig["lettering_pt"] < 7.0
        (warning,) = res["figure_warnings"]
        assert warning.startswith("Figure 1 (pdf_page 1)")
        assert "under 7 pt" in warning

    def test_a_figure_with_no_lettering_record_is_judged_by_width(
            self, tmp_path):
        """Without a record of its lettering, a figure printed narrower
        than the 6.5 in the lettering check measures at is warned."""
        pytest.importorskip("fitz")
        (tmp_path / "dot.png").write_bytes(_tiny_png())
        res = _generate_html_to_pdf({
            "html": _doc(f'<img src="{tmp_path / "dot.png"}" '
                         'style="width:3in">'),
            "output_path": str(tmp_path / "r.pdf")})
        (fig,) = res["page_contents"][0]["figures"]
        assert "lettering_pt" not in fig
        (warning,) = res["figure_warnings"]
        assert "narrower than the 6.5 in" in warning

    def test_the_numbers_are_the_lettering_checks(self):
        from funhouse_agent.adapters.calc_package import _figure_warnings
        from funhouse_agent.plot_box.child import (DEFAULT_PRINT_WIDTH_IN,
                                                   MIN_PRINT_PT)
        base = {"pdf_page": 1, "height_in": 3.0}
        assert _figure_warnings([{**base, "width_in": 6.5}]) == []
        assert _figure_warnings([{**base, "width_in":
                                  DEFAULT_PRINT_WIDTH_IN - 0.5}])
        ok = {**base, "width_in": 5.0, "lettering_pt": MIN_PRINT_PT}
        assert _figure_warnings([ok]) == []
        checked = {**ok, "checked_width_in": 6.5}
        assert "checked at" in _figure_warnings([checked])[0]
        tall = {**base, "width_in": 6.5, "scaled": 0.4}
        assert "taller than a page" in _figure_warnings([tall])[0]

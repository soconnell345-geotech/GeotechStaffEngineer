"""Drawing sheets at the end of a calc package (BACKLOG GM-313).

A sheet is appended to the PDF as its own vector page, at its own size, on
either PDF leg; HTML and LaTeX source show its PNG under "Drawings"; and a
package with no sheet is exactly what it was. Every value is made up.
"""

import base64
import shutil

import pytest

pymupdf = pytest.importorskip("pymupdf")

from calc_package import (                                  # noqa: E402
    CalcPackageData, CalcSection, CalcStep, InputItem, SheetData,
    generate_calc_package, render_html, render_latex,
)
from calc_package.latex_renderer import append_sheets, render_pdf  # noqa: E402

TABLOID = (1224.0, 792.0)       # points, landscape


def _sheet_pdf(size=TABLOID) -> bytes:
    """A one-page vector PDF standing in for a drawing sheet."""
    doc = pymupdf.open()
    page = doc.new_page(width=size[0], height=size[1])
    page.draw_rect(pymupdf.Rect(30, 30, size[0] - 30, size[1] - 30))
    page.draw_line((100, 100), (600, 500))
    data = doc.tobytes()
    doc.close()
    return data


def _png() -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
    pix.clear_with(200)
    return pix.tobytes("png")


def _sheet(title="Wall section") -> SheetData:
    return SheetData(title=title, caption="Drawing 1: the wall section",
                     pdf_bytes=_sheet_pdf(),
                     image_base64=base64.b64encode(_png()).decode("ascii"))


def _data(sheets=()) -> CalcPackageData:
    return CalcPackageData(
        project_name="Made-up project", analysis_type="Example",
        date="2026-01-01",
        sections=[CalcSection(title="Inputs", items=[
            InputItem("B", "Width", 2.0, "m")]),
            CalcSection(title="Steps", items=[
                CalcStep("Area", "A = B L", "A = 2.0 x 3.0", "A", 6.0,
                         "m2")])],
        references=["A made-up reference."], sheets=list(sheets))


def _pages(path):
    with pymupdf.open(path) as doc:
        return [(round(p.rect.width), round(p.rect.height),
                 len(p.get_drawings()), len(p.get_images())) for p in doc]


def test_a_sheet_is_one_more_vector_page_of_its_own_size(tmp_path):
    plain = render_pdf(_data(), str(tmp_path / "plain.pdf"),
                       renderer="pymupdf_story")
    with_sheet = render_pdf(_data([_sheet()]), str(tmp_path / "sheet.pdf"),
                            renderer="pymupdf_story")
    before, after = _pages(plain["path"]), _pages(with_sheet["path"])
    assert len(after) == len(before) + 1
    # the package's own pages are as they were (letter, portrait)
    assert [p[:2] for p in after[:-1]] == [p[:2] for p in before]
    width, height, paths, images = after[-1]
    assert (width, height) == (1224, 792)       # the sheet's size, landscape
    assert paths >= 2 and images == 0           # vector, not a picture
    assert with_sheet["sheets"] == [{
        "title": "Wall section", "pdf_page": len(after),
        "width_in": 17.0, "height_in": 11.0}]
    assert "sheets" not in plain


def test_the_story_pdf_holds_no_picture_of_the_sheet(tmp_path):
    """The Story leg would re-encode a PNG as a JPEG: the sheet's PNG is
    left out of the HTML that leg prints."""
    report = render_pdf(_data([_sheet()]), str(tmp_path / "p.pdf"),
                        renderer="pymupdf_story")
    assert all(images == 0 for *_, images in _pages(report["path"]))
    with pymupdf.open(report["path"]) as doc:
        assert "Drawings" not in "".join(p.get_text() for p in doc)


def test_two_sheets_keep_their_order_and_sizes(tmp_path):
    a = _sheet("First")
    b = SheetData(title="Second", pdf_bytes=_sheet_pdf((792.0, 612.0)))
    report = render_pdf(_data([a, b]), str(tmp_path / "p.pdf"),
                        renderer="pymupdf_story")
    pages = _pages(report["path"])
    assert [p[:2] for p in pages[-2:]] == [(1224, 792), (792, 612)]
    assert [s["title"] for s in report["sheets"]] == ["First", "Second"]


def test_a_package_with_no_sheet_is_unchanged():
    """HTML and LaTeX source of a package without sheets hold nothing of
    this feature (the PDF legs render from these)."""
    html = render_html(_data())
    tex = render_latex(_data())
    assert "Drawings" not in html and "drawings" not in html
    assert "Drawings" not in tex and "sheet_" not in tex
    assert render_html(_data(), with_sheets=False) == html
    assert render_latex(_data(), with_sheets=False) == tex
    # and a sheet adds only its own section, after the references
    with_sheet = render_html(_data([_sheet()]))
    cut = html.index("<!-- Footer -->")
    assert with_sheet.startswith(html[:cut])
    assert with_sheet.endswith(html[cut:])
    assert '<div class="section drawings">' in with_sheet[cut:]


def test_html_ends_with_a_drawings_section_showing_the_png():
    html = render_html(_data([_sheet()]))
    assert html.index("References") < html.index("Drawings")
    assert base64.b64encode(_png()).decode("ascii") in html
    assert "Drawing 1: the wall section" in html
    assert html.index("Drawings") < html.index("Generated by")


def test_latex_source_shows_the_png(tmp_path):
    tex = render_latex(_data([_sheet()]), figure_dir=str(tmp_path))
    assert r"\section*{Drawings}" in tex
    assert (tmp_path / "sheet_1.png").read_bytes() == _png()
    assert "sheet_1.png" in tex
    assert tex.index("References") < tex.index("Drawings")


def test_append_sheets_leaves_a_pdf_alone_when_there_is_none(tmp_path):
    path = tmp_path / "p.pdf"
    render_pdf(_data(), str(path), renderer="pymupdf_story")
    before = path.read_bytes()
    assert append_sheets(str(path), []) == []
    assert append_sheets(str(path), [SheetData(title="no page")]) == []
    assert path.read_bytes() == before


def test_a_sheet_that_cannot_be_read_is_warned_of_not_fatal(tmp_path):
    bad = SheetData(title="Bad", pdf_bytes=b"not a pdf")
    report = render_pdf(_data([bad]), str(tmp_path / "p.pdf"),
                        renderer="pymupdf_story")
    assert any("could not be appended" in w for w in report["warnings"])
    assert len(_pages(report["path"])) >= 1


def test_generate_calc_package_takes_sheets(tmp_path):
    from bearing_capacity import BearingCapacityAnalysis, Footing
    from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer
    analysis = BearingCapacityAnalysis(
        footing=Footing(width=2.0, depth=1.5, shape="square"),
        soil=BearingSoilProfile(layer1=SoilLayer(friction_angle=30,
                                                 unit_weight=18)))
    result = analysis.compute()
    common = dict(module="bearing_capacity", result=result,
                  analysis=analysis, date="2026-01-01", tables=False)

    html = generate_calc_package(**common)
    assert generate_calc_package(**common, sheets=[]) == html
    assert generate_calc_package(**common, sheets=None) == html
    assert "Drawings" in generate_calc_package(**common, sheets=[_sheet()])

    report = {}
    generate_calc_package(**common, format="pdf", sheets=[_sheet()],
                          output_path=str(tmp_path / "bc.pdf"),
                          render_report=report)
    assert report["sheets"][0]["pdf_page"] == len(_pages(report["path"]))
    assert _pages(report["path"])[-1][:2] == (1224, 792)

    with pytest.raises(TypeError, match="SheetData"):
        generate_calc_package(**common, sheets=[{"title": "a dict"}])


@pytest.mark.skipif(shutil.which("pdflatex") is None,
                    reason="pdflatex is not installed")
def test_the_latex_pdf_leg_appends_the_sheet_too(tmp_path):
    report = render_pdf(_data([_sheet()]), str(tmp_path / "p.pdf"),
                        renderer="pdflatex")
    assert _pages(report["path"])[-1][:2] == (1224, 792)
    assert report["sheets"][0]["title"] == "Wall section"


def test_a_real_sheet_from_the_drawing_layer_goes_in(tmp_path):
    """The two halves together; calc_package itself imports nothing of
    the drawing layer."""
    pytest.importorskip("ezdxf")
    import calc_package
    import sys
    from dxf_export import Drawing, render_sheet
    files = render_sheet(Drawing.from_dict({
        "outlines": [{"points": [[0, 0], [900, 0], [900, 300], [0, 300]],
                      "layer": "concrete", "hatch": "concrete"}],
        "dimensions": [{"p1": [0, 0], "p2": [900, 0], "offset_mm": -8}]}))
    sheet = SheetData(title="Detail", pdf_bytes=files["pdf"],
                      image_base64=base64.b64encode(
                          files["png"]).decode("ascii"))
    report = render_pdf(_data([sheet]), str(tmp_path / "p.pdf"),
                        renderer="pymupdf_story")
    width, height, paths, images = _pages(report["path"])[-1]
    assert (width, height) == (792, 612) and paths > 20 and images == 0
    source = "".join(
        open(getattr(sys.modules[name], "__file__")).read()
        for name in ("calc_package", "calc_package.data_model",
                     "calc_package.renderer", "calc_package.latex_renderer"))
    assert "dxf_export" not in source.replace("``dxf_export", "")


def test_the_latex_leg_appends_the_sheet_with_the_compile_stood_in_for(
        tmp_path, monkeypatch):
    """The pdflatex leg's own path through render_pdf (GM-313 review B3):
    pdflatex is on neither this machine nor CI, so the compile is stood in
    for by the Story renderer; the append after it is the real one."""
    from calc_package import latex_renderer

    def compile_stand_in(data, filepath, keep_tex=False, compiler="pdflatex"):
        html = latex_renderer.render_html(data, with_sheets=False)
        return latex_renderer.render_html_pdf_story(html, filepath)["path"]

    monkeypatch.setattr(latex_renderer, "_save_pdf_latex", compile_stand_in)
    monkeypatch.setattr(latex_renderer.shutil, "which",
                        lambda name, *a, **k: "/usr/bin/" + name)
    plain = render_pdf(_data(), str(tmp_path / "plain.pdf"),
                       renderer="pdflatex")
    report = render_pdf(_data([_sheet()]), str(tmp_path / "p.pdf"),
                        renderer="pdflatex")
    assert report["renderer"] == "pdflatex"
    pages = _pages(report["path"])
    assert len(pages) == len(_pages(plain["path"])) + 1
    assert pages[-1][:2] == (1224, 792) and pages[-1][3] == 0
    assert report["sheets"][0]["title"] == "Wall section"
    assert "sheets" not in plain

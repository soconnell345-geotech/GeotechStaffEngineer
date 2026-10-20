"""The Story PDF never shrinks a figure to fit what is left of a page
(BACKLOG GM-168; live wave w13: figures printed 0.11, 0.3, 1.6 and 2.1 in
wide at page feet). A figure that does not fit starts the next page at its
width; one taller than a page starts a page and is scaled to fit it.
"""

import base64
import io
import re

import pytest

fitz = pytest.importorskip("fitz")

from calc_package import latex_renderer                     # noqa: E402
from calc_package.latex_renderer import render_html_pdf_story  # noqa: E402

TEXT = "<p>" + ("The bearing resistance of the footing is checked against "
                "the factored load for each load case. ") * 6 + "</p>"


def _png(width_in: float, height_in: float, dpi: int = 100,
         fontsize: float = 10.0) -> str:
    """A base64 PNG plot ``width_in`` x ``height_in`` at ``dpi``, saved as
    drawn (not fitted to a package's print width: these tests are about
    the page layout)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from calc_package.renderer import figure_to_base64
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("Distance (m)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    try:
        return figure_to_base64(fig, dpi=dpi, print_width_in=None)
    finally:
        plt.close(fig)


def _images(path) -> list:
    """``[(pdf_page, width_in, height_in, y0_in)]`` of every picture."""
    out = []
    with fitz.open(path) as doc:
        for page in doc:
            for info in page.get_image_info():
                r = fitz.Rect(info["bbox"])
                out.append((page.number + 1, r.width / 72, r.height / 72,
                            r.y0 / 72))
    return out


def _figure(b64: str, caption: str = "Figure 1: Section") -> str:
    return (f'<div class="figure-block"><img src="data:image/png;base64,'
            f'{b64}" style="width:100%"><div class="caption">{caption}'
            f"</div></div>")


def test_a_figure_after_most_of_a_page_moves_to_the_next_page_whole(
        tmp_path):
    """The case the backlog item names: a large figure placed after most
    of a page of text (it ends 7.7 in down an 11 in page) lands on the
    next page at its full width, and the text after it follows it
    there."""
    html = (f"<html><body><h1>Footing</h1>{TEXT * 4}{_figure(_png(6.5, 5))}"
            "<p>After the drawing.</p></body></html>")
    out = render_html_pdf_story(html, str(tmp_path / "moved.pdf"))
    (page, width, height, top), = _images(out["path"])
    assert page == 2
    assert width == pytest.approx(out["figures"][0]["laid_out_width_in"],
                                  abs=0.02)
    assert width > 6.0 and top < 1.0                 # full width, page top
    fig = out["figures"][0]
    assert fig["pdf_page"] == 2 and fig["moved"] and fig["scaled"] is None
    with fitz.open(out["path"]) as doc:
        assert "After the drawing." in doc[1].get_text()
        assert "After the drawing." not in doc[0].get_text()


def test_master_squeezed_that_figure(tmp_path):
    """The engine on its own shrinks it into what is left of page 1: the
    case above would fail without the layout rule."""
    html = (f"<html><body><h1>Footing</h1>{TEXT * 4}{_figure(_png(6.5, 5))}"
            "</body></html>")
    pdf = latex_renderer._story_bytes(
        html, fitz.paper_rect("letter"),
        fitz.paper_rect("letter") + (54, 54, -54, -54))
    (page, rect, _top), = latex_renderer._story_images(pdf)
    assert page == 0 and rect.width / 72 < 4.0


def test_a_figure_that_fits_stays_where_it_is(tmp_path):
    html = (f"<html><body>{TEXT}{_figure(_png(6.5, 4))}{TEXT}"
            "</body></html>")
    out = render_html_pdf_story(html, str(tmp_path / "stays.pdf"))
    fig, = out["figures"]
    assert fig["pdf_page"] == 1 and not fig["moved"]
    assert fig["width_in"] == pytest.approx(fig["laid_out_width_in"],
                                            abs=0.02)


def test_the_heading_goes_with_its_figure(tmp_path):
    html = (f"<html><body>{TEXT * 4}<h2>8. Figures</h2>"
            f"{_figure(_png(6.5, 5))}</body></html>")
    out = render_html_pdf_story(html, str(tmp_path / "heading.pdf"))
    assert out["figures"][0]["pdf_page"] == 2
    with fitz.open(out["path"]) as doc:
        assert "8. Figures" in doc[1].get_text()
        assert "8. Figures" not in doc[0].get_text()


def test_a_figure_taller_than_a_page_is_scaled_to_one_page(tmp_path):
    """It starts a page, fits it with its caption, and says it was
    scaled; it is never shrunk into the foot of the page before."""
    html = (f"<html><body>{TEXT * 2}{_figure(_png(4, 14), 'Figure 2: Log')}"
            "<p>After.</p></body></html>")
    out = render_html_pdf_story(html, str(tmp_path / "tall.pdf"))
    fig, = out["figures"]
    assert fig["pdf_page"] == 2 and fig["moved"]
    assert fig["scaled"] is not None and fig["scaled"] < 1
    assert fig["height_in"] <= 9.5 - 0.4               # room for a caption
    with fitz.open(out["path"]) as doc:
        assert "Figure 2: Log" in doc[1].get_text()


def test_a_tall_first_figure_is_not_moved(tmp_path):
    """A page break before the document's first element makes the engine
    repeat an empty page for ever, so a figure that opens the document is
    scaled where it is."""
    html = f"<html><body>{_figure(_png(4, 14))}<p>After.</p></body></html>"
    out = render_html_pdf_story(html, str(tmp_path / "first.pdf"))
    fig, = out["figures"]
    assert fig["pdf_page"] == 1 and not fig["moved"]
    with fitz.open(out["path"]) as doc:
        assert doc.page_count <= 2


def test_a_page_break_on_the_first_element_fails_instead_of_hanging(
        tmp_path, monkeypatch):
    monkeypatch.setattr(latex_renderer, "_MAX_STORY_PAGES", 10)
    html = ('<html><body><div style="page-break-before:always">'
            "<p>A</p></div></body></html>")
    with pytest.raises(RuntimeError, match="page break"):
        render_html_pdf_story(html, str(tmp_path / "loop.pdf"))


def test_the_printed_lettering_comes_from_the_png(tmp_path):
    """figure_to_base64 writes the smallest lettering into the PNG; the
    renderer scales it to the printed width."""
    b64 = _png(10, 5, fontsize=8)
    html = f"<html><body>{_figure(b64)}</body></html>"
    out = render_html_pdf_story(html, str(tmp_path / "letters.pdf"))
    fig, = out["figures"]
    from geotech_common.layout_ink import (LETTERING_KEY, parse_lettering,
                                           png_texts)
    saved = parse_lettering(png_texts(base64.b64decode(b64))[LETTERING_KEY])
    assert saved["smallest_pt"] == pytest.approx(8.0, abs=0.01)
    assert fig["lettering_pt"] == pytest.approx(
        8.0 * fig["width_in"] / saved["saved_width_in"], abs=0.1)
    assert fig["lettering_pt"] < 8.0                   # printed smaller


def test_figures_are_listed_in_document_order(tmp_path):
    small, wide = _png(3, 2), _png(6.5, 3)
    html = (f'<html><body><p>Start.</p><img src="data:image/png;base64,'
            f'{small}"><p>t</p><img src="data:image/png;base64,{wide}" '
            'style="width:100%"></body></html>')
    out = render_html_pdf_story(html, str(tmp_path / "order.pdf"))
    first, second = out["figures"]
    assert first["width_in"] < second["width_in"]


A_PNG = 'src="data:image/png;base64,QQ=="'
B_PNG = 'src="data:image/png;base64,Qg=="'


def test_figure_marks_find_the_block_heading_and_title():
    html = ('<html><head><title>T</title><style>p{}</style></head><body>'
            '<h1>Report</h1><p>Text.</p>'
            '<h2>Profile</h2><p><b>Figure 1.</b> Soil profile</p>'
            f'<div class="figure-block"><img {A_PNG}><div>cap</div></div>'
            f'<p>More text.</p><table><tr><td><img {B_PNG}></td></tr>'
            '</table></body></html>')
    marks = latex_renderer._figure_marks(html)
    assert len(marks["spans"]) == 2
    assert html[marks["breaks"][0]:].startswith("<h2>Profile")
    assert html[marks["breaks"][1]:].startswith("<table>")
    assert html[slice(*marks["spans"][0])] == f"<img {A_PNG}>"


def test_an_image_the_engine_does_not_draw_is_not_a_figure():
    """Only an image carried inline is drawn, so only it is matched to the
    pictures on the pages."""
    html = (f'<body><p>x</p><img src="missing.png"><p>y</p><img {A_PNG}>'
            "</body>")
    marks = latex_renderer._figure_marks(html)
    assert len(marks["spans"]) == 1
    assert html[slice(*marks["spans"][0])] == f"<img {A_PNG}>"
    assert latex_renderer._png_letterings(html) == [{}]


def test_no_break_before_the_first_thing_in_the_document():
    html = f"<html><body><div><img {A_PNG}></div><p>x</p></body></html>"
    assert latex_renderer._figure_marks(html)["breaks"] == [None]


def test_edits_write_the_break_and_the_width():
    html = (f'<body><p>x</p><div><img {A_PNG} style="max-width:80%;">'
            "</div></body>")
    marks = latex_renderer._figure_marks(html)
    out = latex_renderer._with_figure_edits(
        html, marks, {0: {"break": True, "width_pt": 300.0}})
    assert latex_renderer._PAGE_BREAK + "<div><img" in out
    assert 'style="max-width:80%;width:300.0pt"' in out


# ── review fixes (GM-168 review, 2026-10-19) ──────────────────────────

TABLE = ("<table border='1'><tr><th>Case</th><th>FS</th></tr>"
         + "".join(f"<tr><td>LC{i}</td><td>3.0</td></tr>" for i in range(6))
         + "</table>")


def _words(path) -> str:
    with fitz.open(path) as doc:
        return " ".join(page.get_text() for page in doc)


def test_a_page_break_after_a_table_does_not_make_the_engine_loop(
        tmp_path, monkeypatch):
    """PyMuPDF's Story starts the document again for ever at a page break
    anywhere after a table whose borders are not collapsed: moving a
    figure after such a table made html_to_pdf fail on a normal report.
    Collapsed borders end the loop; the figure still moves."""
    monkeypatch.setattr(latex_renderer, "_MAX_STORY_PAGES", 60)
    html = (f"<html><body><h1>Report</h1>{TABLE}{TEXT * 3}"
            f"<img src=\"data:image/png;base64,{_png(6.5, 5)}\" "
            'style="width:100%"><p>After the drawing.</p></body></html>')
    with pytest.raises(RuntimeError):                  # the engine's trap
        latex_renderer._story_bytes(
            latex_renderer._PAGE_BREAK.join(html.split("<img", 1)),
            fitz.paper_rect("letter"),
            fitz.paper_rect("letter") + (54, 54, -54, -54))
    out = render_html_pdf_story(html, str(tmp_path / "table.pdf"))
    fig, = out["figures"]
    assert fig["moved"] and fig["width_in"] > 6.0
    assert "After the drawing." in _words(out["path"])


def test_a_page_break_the_html_carries_after_a_table_renders(tmp_path,
                                                             monkeypatch):
    """The caller's own page break after a table hung master for ever."""
    monkeypatch.setattr(latex_renderer, "_MAX_STORY_PAGES", 60)
    html = (f"<html><body><h1>Report</h1>{TABLE}"
            f"{latex_renderer._PAGE_BREAK}<p>Appendix.</p></body></html>")
    out = render_html_pdf_story(html, str(tmp_path / "own.pdf"))
    with fitz.open(out["path"]) as doc:
        assert doc.page_count == 2
        assert "Appendix." in doc[1].get_text()


def test_no_text_is_lost_after_figures_at_a_page_foot(tmp_path):
    """The engine shrinks a figure into a page foot, draws the next one
    past the page's edge and drops the text after them: master lost the
    captions and the section after. Every word now prints, and no picture
    runs off a page."""
    html = (f"<html><body><h1>Report</h1>{TEXT * 4}"
            f"<img src=\"data:image/png;base64,{_png(6.5, 8)}\" "
            'style="width:100%"><p>Figure 1: Wall section</p>'
            f"<img src=\"data:image/png;base64,{_png(6.5, 3.2)}\" "
            'style="width:100%"><p>Figure 2: Base pressure</p>'
            "<h2>Conclusions</h2><p>The wall is adequate.</p>"
            "</body></html>")
    master, _at = latex_renderer._story_pass(
        html, fitz.paper_rect("letter"),
        fitz.paper_rect("letter") + (54, 54, -54, -54))
    with fitz.open("pdf", master) as doc:
        assert "Conclusions" not in " ".join(p.get_text() for p in doc)
    out = render_html_pdf_story(html, str(tmp_path / "lost.pdf"))
    text = _words(out["path"])
    for words in ("Figure 1: Wall section", "Figure 2: Base pressure",
                  "Conclusions", "The wall is adequate."):
        assert words in text
    for _page, _w, height, top in _images(out["path"]):
        assert top + height <= 11 - 0.75 + 0.02
    assert all(f["width_in"] > 6.0 for f in out["figures"])


def test_a_caption_under_a_figure_stays_with_it():
    """Captions under their figures (render_figures says "use the caption
    text under it"): the next figure's break goes before that figure, not
    before the caption of the one above it."""
    html = ("<html><body><h1>R</h1><p>Text.</p>"
            f"<img {A_PNG}><p>Figure 1: Wall</p>"
            f"<img {B_PNG}><p>Figure 2: Pressure</p></body></html>")
    marks = latex_renderer._figure_marks(html)
    assert html[marks["breaks"][1]:].startswith(f"<img {B_PNG}")
    assert marks["titled"] == [False, False]


def test_a_title_above_its_figure_still_goes_with_it():
    html = ("<html><body><h1>R</h1><p>Text.</p>"
            f"<p>Figure 1: Wall</p><img {A_PNG}>"
            f"<p>Figure 2: Pressure</p><img {B_PNG}></body></html>")
    marks = latex_renderer._figure_marks(html)
    assert html[marks["breaks"][1]:].startswith("<p>Figure 2")
    assert marks["titled"] == [True, True]


def test_a_heading_left_at_the_foot_of_the_page_before_follows(tmp_path):
    """A figure the engine starts on a new page left its section heading
    alone at the foot of the page before (the bearing package's
    "8. Figures"): the heading now starts the page with it."""
    filler = "<p>Line of text in the report body.</p>" * 22
    block = (f'<div style="page-break-inside:avoid"><img src="data:image/'
             f'png;base64,{_png(6.5, 4.5)}" style="width:100%">'
             "<div>Figure 1: Section</div></div>")
    html = (f"<html><body><h1>Report</h1>{filler}<h2>8. Figures</h2>"
            f"{block}</body></html>")
    master = latex_renderer._story_bytes(
        html, fitz.paper_rect("letter"),
        fitz.paper_rect("letter") + (54, 54, -54, -54))
    with fitz.open("pdf", master) as doc:
        assert doc.page_count == 2
        assert "8. Figures" in doc[0].get_text()       # the orphan
    out = render_html_pdf_story(html, str(tmp_path / "orphan.pdf"))
    with fitz.open(out["path"]) as doc:
        assert "8. Figures" not in doc[0].get_text()
        assert "8. Figures" in doc[1].get_text()
    assert out["figures"][0]["pdf_page"] == 2


def test_a_commented_out_picture_does_not_shift_the_lettering(tmp_path):
    """The lettering is read off the same tags the pictures drawn are
    matched to: an image in an HTML comment is neither."""
    big, small = _png(6.5, 3, fontsize=12), _png(10, 4, fontsize=7)
    html = (f'<html><body><p>Start.</p><!-- <img src="data:image/png;'
            f'base64,{small}"> --><img src="data:image/png;base64,{big}" '
            'style="width:100%"></body></html>')
    out = render_html_pdf_story(html, str(tmp_path / "comment.pdf"))
    fig, = out["figures"]
    assert fig["lettering_pt"] > 10


# ---------------------------------------------------------------------------
# package figures drawn for the width they print at (BACKLOG GM-186)
# ---------------------------------------------------------------------------

def _package_pdf(tmp_path, fig, name="pkg.pdf"):
    """``fig`` as a calc package carries it: saved by figure_to_base64,
    laid out by the package template, printed by the Story engine."""
    from calc_package.data_model import CalcPackageData, CalcSection, FigureData
    from calc_package.renderer import figure_to_base64, render_html
    data = CalcPackageData(project_name="T", analysis_type="Test", sections=[
        CalcSection(title="Figures", items=[FigureData(
            title="F", image_base64=figure_to_base64(fig), caption="c")])])
    return render_html_pdf_story(render_html(data), str(tmp_path / name))


def test_a_package_figure_prints_its_lettering_at_7_pt(tmp_path):
    """Drawn 10 in wide with 8 pt lettering, a package figure printed at
    the template's 5 in with 4 pt lettering; it is now drawn for 5 in."""
    import matplotlib.pyplot as plt
    from calc_package.renderer import FIGURE_PRINT_WIDTH_IN
    from geotech_common.layout_ink import MIN_PRINT_PT
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("Distance (m)", fontsize=8)
    ax.tick_params(labelsize=8)
    try:
        out = _package_pdf(tmp_path, fig)
    finally:
        plt.close(fig)
    printed, = out["figures"]
    assert printed["width_in"] == pytest.approx(FIGURE_PRINT_WIDTH_IN,
                                                abs=0.05)
    assert printed["lettering_pt"] >= MIN_PRINT_PT


def test_a_tall_package_figure_keeps_the_text_width(tmp_path):
    """A tall figure's pixels are scaled to the PDF's cap, not halved: a
    half left it under the text width at the engine's 96 px an inch, and
    its lettering printed under 7 pt."""
    import matplotlib.pyplot as plt
    from calc_package.renderer import FIGURE_PRINT_WIDTH_IN
    from geotech_common.layout_ink import MIN_PRINT_PT
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot([0, 1], [0, 1])
    ax.set_ylabel("Depth (m)", fontsize=9)
    ax.tick_params(labelsize=9)
    try:
        out = _package_pdf(tmp_path, fig, "tall.pdf")
    finally:
        plt.close(fig)
    printed, = out["figures"]
    assert printed["width_in"] >= FIGURE_PRINT_WIDTH_IN - 0.05
    assert printed["lettering_pt"] >= MIN_PRINT_PT



def _foot_html(margin_pt: float) -> str:
    """A package's figures section after text that ends ``margin_pt`` down
    the first page, in the package template's own CSS: at 583 pt the
    engine leaves out the first figure, with its heading (GM-198)."""
    from calc_package.template import CALC_PACKAGE_CSS
    figures = (f'<div class="section"><h2>8. Figures</h2>'
               f'{_figure(_png(5, 5), "Figure 1: Plan view")}'
               f'{_figure(_png(5, 3), "Figure 2: Pile forces")}</div>')
    return (f"<html><head><style>{CALC_PACKAGE_CSS}</style></head><body>"
            f'<p style="margin-top:{margin_pt}pt">The last step.</p>'
            f"{figures}</body></html>")


def _steps_html(margin_pt: float) -> str:
    """Fourteen lines, a paragraph ``margin_pt`` below them, then a section
    of two calculation steps and a closing paragraph: at 40 pt the engine
    leaves out the closing paragraph, at 198 pt the steps under the
    section heading (GM-198)."""
    from calc_package.template import CALC_PACKAGE_CSS
    step = ('<div class="calc-step"><div class="step-title">Maximum shear'
            '</div><div class="result">V = 136.2 kN/m</div></div>')
    return (f"<html><head><style>{CALC_PACKAGE_CSS}</style></head><body>"
            + "".join(f"<p>Line {i}.</p>" for i in range(14))
            + f'<p style="margin-top:{margin_pt}pt">The last step.</p>'
            f'<div class="section"><h2>6. Wall Demands</h2>{step}{step}'
            "</div><p>References follow.</p></body></html>")


_LETTER = fitz.paper_rect("letter")
_WHERE = _LETTER + (54, 54, -54, -54)


def _texts(pdf: bytes) -> list:
    with fitz.open("pdf", pdf) as doc:
        return [page.get_text() for page in doc]


@pytest.mark.parametrize("case, margin_pt", [
    ("figures", 583), ("steps", 40), ("steps", 198)])
def test_the_engine_drops_what_it_opens_past_a_page_foot(case, margin_pt):
    """The engine on its own: each case loses something, and the layout
    finds where."""
    html = (_foot_html if case == "figures" else _steps_html)(margin_pt)
    pdf, at = latex_renderer._story_pass(html, _LETTER, _WHERE)
    text = " ".join(_texts(pdf))
    lost = (len(latex_renderer._story_images(pdf)) < html.count("<img")
            or text.count("Maximum shear") < 2
            and "Wall Demands" in html
            or "References follow." not in text and "References" in html)
    if not lost:
        pytest.skip("this PyMuPDF prints it all on its own")
    assert at is not None and html[at] == "<"


def test_a_dropped_figure_prints_on_the_next_page(tmp_path):
    """GM-198: every figure prints, the first under its heading and over
    its caption, and the figures are reported."""
    out = render_html_pdf_story(_foot_html(583), str(tmp_path / "foot.pdf"))
    assert out["figures"] is not None and len(out["figures"]) == 2
    assert "figures_not_printed" not in out
    first, second = out["figures"]
    assert first["scaled"] is None
    with fitz.open(out["path"]) as doc:
        assert sum(len(page.get_image_info()) for page in doc) == 2
        text = doc[first["pdf_page"] - 1].get_text()
    assert "8. FIGURES" in text.upper() and "Figure 1: Plan view" in text


@pytest.mark.parametrize("margin_pt", [40, 198])
def test_dropped_steps_print_on_the_next_page_under_their_heading(
        margin_pt):
    """GM-198, the same for text: every step and the closing paragraph
    print, and a heading starts the page its steps do."""
    pages = _texts(latex_renderer._story_bytes(
        _steps_html(margin_pt), _LETTER, _WHERE))
    text = " ".join(pages)
    assert text.count("Maximum shear") == 2 and "References follow." in text
    heading = next(p for p in pages if "Wall Demands" in p)
    assert "Maximum shear" in heading


def test_block_elements_get_an_id_to_find_them_by():
    html = ('<div id="a"><p>x</p><div id="a"></div>'
            '<h2 class="z">h</h2><span>s</span></div>')
    tagged, starts = latex_renderer._tag_blocks(html)
    assert tagged.count(' id="gse-block-') == 2           # p and h2
    assert "a" not in starts                               # used twice
    assert sorted(html[at:at + 3] for at in starts.values()) == ["<h2", "<p>"]


def _cells_html(margin_pt: float) -> str:
    """Thirty table rows with a paragraph in each first cell, after a
    paragraph ``margin_pt`` down the page, then a closing paragraph: the
    engine leaves out the rows it opens past a page foot (at 12 pt, rows
    10 to 20), and at 188 pt the closing paragraph after the table's last
    row too, once the rows are moved."""
    from calc_package.template import CALC_PACKAGE_CSS
    rows = "".join(f"<tr><td><p>Row {i} " + "word " * 12 + "</p></td>"
                   f"<td><div>{i}.5 kPa</div></td></tr>" for i in range(30))
    return (f"<html><head><style>{CALC_PACKAGE_CSS}</style></head><body>"
            f'<p style="margin-top:{margin_pt}pt">Before the table.</p>'
            f"<table>{rows}</table><p>References follow.</p></body></html>")


def _items_html(margin_pt: float) -> str:
    """Twelve list items of 26 words after a paragraph ``margin_pt`` down
    the page: the engine leaves out the lines of an item it runs past a
    page foot, and the items after it."""
    from calc_package.template import CALC_PACKAGE_CSS
    items = "".join(f"<li>Item {i} " + "word " * 25 + "</li>"
                    for i in range(12))
    return (f"<html><head><style>{CALC_PACKAGE_CSS}</style></head><body>"
            f'<p style="margin-top:{margin_pt}pt">Before the list.</p>'
            f"<ul>{items}</ul><p>References follow.</p></body></html>")


def _numbered(text: str, word: str) -> set:
    return {int(n) for n in re.findall(rf"\b{word} (\d+)\b", text)}


@pytest.mark.parametrize("margin_pt", [12, 188, 424])
def test_table_rows_opened_past_a_page_foot_print(margin_pt):
    """GM-198 for table rows: the table ends before the row and starts
    again on the next page (or before the row above it, or its last row
    goes with the paragraph after it), and every row prints."""
    html = _cells_html(margin_pt)
    raw = " ".join(_texts(latex_renderer._story_pass(html, _LETTER,
                                                     _WHERE)[0]))
    if len(_numbered(raw, "Row")) == 30:
        pytest.skip("this PyMuPDF prints every row on its own")
    text = " ".join(_texts(latex_renderer._story_bytes(html, _LETTER,
                                                       _WHERE)))
    assert _numbered(text, "Row") == set(range(30))
    assert text.count("word") == 360 and "References follow." in text


@pytest.mark.parametrize("margin_pt", [124, 388])
def test_list_items_run_past_a_page_foot_print(margin_pt):
    """GM-198 for list items: one opened or run past the foot goes to the
    next page, with every line of it."""
    html = _items_html(margin_pt)
    raw = " ".join(_texts(latex_renderer._story_pass(html, _LETTER,
                                                     _WHERE)[0]))
    if raw.count("word") == 300:
        pytest.skip("this PyMuPDF prints every item on its own")
    text = " ".join(_texts(latex_renderer._story_bytes(html, _LETTER,
                                                       _WHERE)))
    assert _numbered(text, "Item") == set(range(12))
    assert text.count("word") == 300 and "References follow." in text


def test_one_that_no_break_moves_is_left_and_the_next_is_moved(monkeypatch):
    """An element no break moves (a row taller than a page) is left as it
    is, with everything in it, and the ones after it are still moved."""
    html = _cells_html(12)
    raw = " ".join(_texts(latex_renderer._story_pass(html, _LETTER,
                                                     _WHERE)[0]))
    if 10 in _numbered(raw, "Row"):
        pytest.skip("this PyMuPDF prints row 10 on its own")
    real, seen = latex_renderer._foot_breaks, []

    def first_unmovable(text, at):
        seen.append(text[at:at + 18])
        return [] if len(seen) == 1 else real(text, at)
    monkeypatch.setattr(latex_renderer, "_foot_breaks", first_unmovable)
    text = " ".join(_texts(latex_renderer._story_bytes(html, _LETTER,
                                                       _WHERE)))
    assert seen[0].startswith("<tr><td><p>Row 10")
    assert not any("Row 10 " in s for s in seen[1:])   # nor what is in it
    # (moving the next one can bring it back too)
    assert _numbered(text, "Row") >= set(range(30)) - {10}
    assert "References follow." in text


def test_a_table_row_is_found_with_the_body_rows_before_it():
    html = ("<p>x</p><table><thead><tr><th>A</th></tr></thead>"
            "<tr><td>1</td></tr><tr><td><table><tr><td>in</td></tr></table>"
            "</td></tr><tr><td>3</td></tr></table>")
    rows = [m.start() for m in re.finditer("<tr><td>", html)]
    table = html.find("<table>")
    assert latex_renderer._table_row(html, 0) is None
    assert latex_renderer._table_row(html, rows[0]) == (table, rows[0], [])
    assert latex_renderer._table_row(html, rows[3]) == (
        table, rows[3], [rows[0], rows[1]])               # not the inner row
    inner = html.find("<table>", table + 1)
    assert latex_renderer._table_row(html, rows[2]) == (inner, rows[2], [])
    end = latex_renderer._element_end(html, table)
    assert html[:end].endswith("</table>") and end == len(html)


def test_the_breaks_tried_for_a_row_and_for_a_step_result():
    br = latex_renderer._PAGE_BREAK
    rows = "".join(f"<tr><td>{i}</td></tr>" for i in range(4))
    html = f"<h2>T</h2><table>{rows}</table><p>After.</p>"
    starts = [m.start() for m in re.finditer("<tr>", html)]
    table, again = html.find("<table>"), "</table>" + br + "<table>"
    # the third row: before it, before the row above it, then the table
    # under its heading
    assert latex_renderer._foot_breaks(html, starts[2])[:4] == [
        (starts[2], again), (starts[1], again), (0, br), (table, br)]
    # the first body row goes with the table
    assert latex_renderer._foot_breaks(html, starts[0])[:2] == [
        (0, br), (table, br)]
    # after a table: the paragraph, then with the table's last row
    after = html.find("<p>After")
    assert latex_renderer._foot_breaks(html, after)[:2] == [
        (after, br), (starts[3], again)]
    step = ('<div class="calc-step"><div class="step-title">T</div>'
            '<div class="result">R</div></div>')
    result = step.find('<div class="result">')
    previous, holding = latex_renderer._blocks_before(step, result)
    assert step[previous:].startswith('<div class="step-title">')
    assert holding == [0]
    assert latex_renderer._foot_breaks(step, result) == [
        (result, br), (previous, br), (0, br)]


def test_a_figure_that_still_does_not_print_is_said(tmp_path, monkeypatch):
    """Should the restore fail, the result says how many figures are
    missing, and the warning says so plainly."""
    html = _foot_html(583)
    raw, _at = latex_renderer._story_pass(html, _LETTER, _WHERE)
    if len(latex_renderer._story_images(raw)) == 2:
        pytest.skip("this PyMuPDF prints both figures on its own")
    monkeypatch.setattr(latex_renderer, "_lay_out_figures",
                        lambda *a: (raw, None))
    out = render_html_pdf_story(html, str(tmp_path / "missing.pdf"))
    assert out["figures"] is None and out["figures_not_printed"] >= 1
    warning = latex_renderer.figures_not_printed_warning(out)
    assert "did not print" in warning and "Tell the user" in warning
    assert latex_renderer.figures_not_printed_warning(
        {"figures": []}) is None

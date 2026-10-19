"""The Story PDF never shrinks a figure to fit what is left of a page
(BACKLOG GM-168; live wave w13: figures printed 0.11, 0.3, 1.6 and 2.1 in
wide at page feet). A figure that does not fit starts the next page at its
width; one taller than a page starts a page and is scaled to fit it.
"""

import base64
import io

import pytest

fitz = pytest.importorskip("fitz")

from calc_package import latex_renderer                     # noqa: E402
from calc_package.latex_renderer import render_html_pdf_story  # noqa: E402

TEXT = "<p>" + ("The bearing resistance of the footing is checked against "
                "the factored load for each load case. ") * 6 + "</p>"


def _png(width_in: float, height_in: float, dpi: int = 100,
         fontsize: float = 10.0) -> str:
    """A base64 PNG plot ``width_in`` x ``height_in`` at ``dpi``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from calc_package.renderer import figure_to_base64
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("Distance (m)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    try:
        return figure_to_base64(fig, dpi=dpi)
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

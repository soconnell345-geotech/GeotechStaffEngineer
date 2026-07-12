"""Tests for the calc_package PDF fallback chain (5.4.1).

pdflatex (best, unchanged) -> PyMuPDF Story (pure-Python HTML->PDF) -> clear
error. All branches are forced offline WITHOUT a LaTeX install by mocking
``shutil.which``; the Story branch is exercised for real (PyMuPDF is a [pdf]
dependency) and the produced PDF is opened + read back with fitz.
"""

import os
from unittest import mock

import pytest

from calc_package.data_model import (
    CalcPackageData, CalcSection, CalcStep, InputItem, CheckItem,
    FigureData, TableData,
)
from calc_package import latex_renderer
from calc_package.latex_renderer import (
    render_pdf, save_pdf, _compress_pdf_images, _fit_wide_tables_for_pdf,
)


def _sample() -> CalcPackageData:
    return CalcPackageData(
        project_name="Test Project",
        analysis_type="Bearing Capacity",
        engineer="Eng",
        date="2026-01-01",
        sections=[
            CalcSection(title="Inputs", items=[
                InputItem("B", "Width", 2.0, "m"),
                InputItem("phi", "Friction angle", 30, "deg"),
            ]),
            CalcSection(title="Calc", items=[
                CalcStep("Nq", "Nq = e^(pi tan phi)", "Nq = 18.4",
                         "Nq", 18.4, "", "Vesic"),
            ]),
            CalcSection(title="Checks", items=[
                CheckItem("BC", 300.0, "q_app", 450.0, "q_all", "kPa", True),
            ]),
        ],
    )


def test_story_branch_produces_valid_readable_pdf(tmp_path):
    """The pure-Python Story renderer produces a valid, non-empty PDF whose
    text carries the package content."""
    import fitz
    out = str(tmp_path / "calc.pdf")
    rep = render_pdf(_sample(), out, renderer="pymupdf_story")
    assert rep["renderer"] == "pymupdf_story"
    assert rep["path"] == out and os.path.getsize(out) > 0
    doc = fitz.open(out)
    assert doc.page_count >= 1
    text = "".join(doc[i].get_text() for i in range(doc.page_count))
    assert "Bearing Capacity" in text and "Width" in text and "Nq" in text


def test_auto_falls_back_to_story_when_no_pdflatex(tmp_path):
    """With pdflatex absent, renderer='auto' uses the Story fallback and warns."""
    out = str(tmp_path / "calc.pdf")
    with mock.patch("shutil.which", return_value=None):
        rep = render_pdf(_sample(), out, renderer="auto")
    assert rep["renderer"] == "pymupdf_story"
    assert rep["warnings"] and "pdflatex" in rep["warnings"][0].lower() \
        or "not found" in rep["warnings"][0].lower()


def test_auto_prefers_pdflatex_when_present(tmp_path):
    """Default-preserving: with pdflatex on PATH, renderer='auto' takes the
    LaTeX leg (mocked here so the test needs no LaTeX install)."""
    out = str(tmp_path / "calc.pdf")
    with mock.patch("shutil.which", return_value="/usr/bin/pdflatex"), \
         mock.patch.object(latex_renderer, "_save_pdf_latex",
                           return_value=out) as m_latex:
        rep = render_pdf(_sample(), out, renderer="auto")
    assert rep["renderer"] == "pdflatex"
    assert rep["path"] == out
    m_latex.assert_called_once()


def test_force_pdflatex_missing_raises_informative(tmp_path):
    """renderer='pdflatex' with the compiler absent raises the install-guidance
    FileNotFoundError (unchanged behaviour)."""
    with mock.patch("shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError, match="MiKTeX"):
            render_pdf(_sample(), str(tmp_path / "x.pdf"), renderer="pdflatex")


def test_error_branch_when_neither_available(tmp_path):
    """When pdflatex is absent AND the Story renderer fails (e.g. PyMuPDF
    missing), render_pdf raises a clear error with the print-to-PDF advice."""
    with mock.patch("shutil.which", return_value=None), \
         mock.patch.object(latex_renderer, "_save_pdf_story",
                           side_effect=RuntimeError("fitz missing")):
        with pytest.raises(RuntimeError, match="Print -> Save as PDF"):
            render_pdf(_sample(), str(tmp_path / "x.pdf"), renderer="auto")


def test_invalid_renderer_rejected(tmp_path):
    with pytest.raises(ValueError, match="renderer must be"):
        render_pdf(_sample(), str(tmp_path / "x.pdf"), renderer="bogus")


def test_save_pdf_returns_path_string(tmp_path):
    """save_pdf keeps its str return; render_pdf carries the renderer report."""
    out = str(tmp_path / "calc.pdf")
    path = save_pdf(_sample(), out, renderer="pymupdf_story")
    assert isinstance(path, str) and path == out and os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# Story-path size + wide-table fixes (F7 QC)
# ---------------------------------------------------------------------------

def _big_png_b64() -> str:
    """A 1400x900 PNG that would embed as a ~3.8 MB RAW image in a Story PDF."""
    import base64
    import fitz
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 1400, 900), False)
    pix.clear_with(210)
    return base64.b64encode(pix.tobytes("png")).decode("ascii")


def _sample_with_figure_and_wide_table() -> CalcPackageData:
    headers = [f"C{i}" for i in range(15)]           # 15-column (wide) table
    rows = [[str(r)] + ["18.84"] * 14 for r in range(1, 8)]
    return CalcPackageData(
        project_name="T", analysis_type="Slope",
        sections=[
            CalcSection(title="Slice", items=[
                TableData(title="Per-Slice", headers=headers, rows=rows)]),
            CalcSection(title="Figures", items=[
                FigureData(title="F", image_base64=_big_png_b64(),
                           caption="c", width_percent=90)]),
        ],
    )


def test_compress_pdf_images_reencodes_png_to_jpeg():
    html = f'<img src="data:image/png;base64,{_big_png_b64()}">'
    out = _compress_pdf_images(html)
    assert 'data:image/jpeg;base64,' in out
    assert 'data:image/png;base64,' not in out


def test_fit_wide_tables_only_tags_wide_tables():
    def _tbl(n):
        ths = "".join(f"<th>H{i}</th>" for i in range(n))
        return (f'<table class="data-table"><thead><tr>{ths}</tr></thead>'
                "<tbody></tbody></table>")
    out = _fit_wide_tables_for_pdf(_tbl(3) + _tbl(15))
    assert out.count("wide-pdf-table") == 1              # only the 15-col table
    assert '<table class="data-table"><thead><tr><th>H0</th>' in out  # 3-col untouched


def test_story_pdf_is_compact_and_wide_table_not_clipped(tmp_path):
    """The Story PDF re-encodes figures to JPEG (so it stays small) and compacts
    a wide table so every column survives (no right-edge clip)."""
    import fitz
    out = str(tmp_path / "r.pdf")
    rep = render_pdf(_sample_with_figure_and_wide_table(), out,
                     renderer="pymupdf_story")
    assert rep["renderer"] == "pymupdf_story"
    # raw-embedding one 1400x900 figure alone would be ~3.8 MB; JPEG keeps it small
    assert os.path.getsize(out) < 2_000_000, os.path.getsize(out)
    doc = fitz.open(out)
    text = "".join(doc[i].get_text() for i in range(doc.page_count))
    doc.close()
    assert all(f"C{i}" in text for i in range(15)), "wide table columns clipped"

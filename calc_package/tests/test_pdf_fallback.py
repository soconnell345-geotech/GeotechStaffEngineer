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
)
from calc_package import latex_renderer
from calc_package.latex_renderer import render_pdf, save_pdf


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

"""Tests for pdf_import.extractor — PyMuPDF vector extraction."""

import pytest

fitz = pytest.importorskip("fitz")

from pdf_import.extractor import (
    discover_pdf_content,
    extract_vector_geometry,
    _color_to_hex,
    _extract_path_points,
)
from pdf_import.results import PdfParseResult


# ---------------------------------------------------------------------------
# Helpers to create test PDFs programmatically
# ---------------------------------------------------------------------------

def _make_simple_pdf(tmp_path, lines=None, texts=None):
    """Create a minimal PDF with colored lines and text."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)

    if lines:
        for (x0, y0, x1, y1), color in lines:
            shape = page.new_shape()
            shape.draw_line(fitz.Point(x0, y0), fitz.Point(x1, y1))
            shape.finish(color=color, width=1)
            shape.commit()

    if texts:
        for text, (x, y), size in texts:
            page.insert_text(fitz.Point(x, y), text, fontsize=size)

    path = tmp_path / "test.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


def _make_cross_section_pdf(tmp_path):
    """Create PDF with a cross-section: surface (black), boundary (gray), gwt (blue)."""
    doc = fitz.open()
    page = doc.new_page(width=500, height=400)
    shape = page.new_shape()

    # Surface (black) — Y inverted in PDF: top is y=0
    # Engineering: surface at z=10, which in PDF-space is y=100 from top
    surface_pts = [(50, 100), (150, 100), (300, 200), (450, 200)]
    for i in range(len(surface_pts) - 1):
        shape.draw_line(fitz.Point(*surface_pts[i]), fitz.Point(*surface_pts[i + 1]))
    shape.finish(color=(0, 0, 0), width=2)
    shape.commit()

    # Boundary (gray)
    shape = page.new_shape()
    boundary_pts = [(50, 200), (150, 200), (300, 250), (450, 250)]
    for i in range(len(boundary_pts) - 1):
        shape.draw_line(fitz.Point(*boundary_pts[i]), fitz.Point(*boundary_pts[i + 1]))
    shape.finish(color=(0.5, 0.5, 0.5), width=1)
    shape.commit()

    # GWT (blue)
    shape = page.new_shape()
    gwt_pts = [(50, 150), (150, 150), (300, 220), (450, 220)]
    for i in range(len(gwt_pts) - 1):
        shape.draw_line(fitz.Point(*gwt_pts[i]), fitz.Point(*gwt_pts[i + 1]))
    shape.finish(color=(0, 0, 1), width=1)
    shape.commit()

    # Text annotation
    page.insert_text(fitz.Point(200, 170), "Stiff Clay", fontsize=10)

    path = tmp_path / "cross_section.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


# ---------------------------------------------------------------------------
# Tests: _color_to_hex
# ---------------------------------------------------------------------------

class TestColorToHex:
    def test_rgb(self):
        assert _color_to_hex((1.0, 0, 0)) == "#ff0000"

    def test_grayscale(self):
        assert _color_to_hex((0.5,)) == "#7f7f7f"

    def test_none(self):
        assert _color_to_hex(None) == "#000000"

    def test_empty(self):
        assert _color_to_hex(()) == "#000000"

    def test_black(self):
        assert _color_to_hex((0, 0, 0)) == "#000000"

    def test_white(self):
        assert _color_to_hex((1, 1, 1)) == "#ffffff"


# ---------------------------------------------------------------------------
# Tests: discover_pdf_content
# ---------------------------------------------------------------------------

class TestDiscoverPdfContent:
    def test_returns_dict(self, tmp_path):
        path = _make_simple_pdf(tmp_path, lines=[
            ((10, 10, 100, 10), (0, 0, 0)),
        ])
        result = discover_pdf_content(filepath=path)
        assert isinstance(result, dict)
        assert "page_size" in result
        assert "n_drawings" in result
        assert "colors" in result

    def test_counts_drawings(self, tmp_path):
        path = _make_simple_pdf(tmp_path, lines=[
            ((10, 10, 100, 10), (0, 0, 0)),
            ((10, 50, 100, 50), (1, 0, 0)),
        ])
        result = discover_pdf_content(filepath=path)
        assert result["n_drawings"] >= 2

    def test_colors_detected(self, tmp_path):
        path = _make_simple_pdf(tmp_path, lines=[
            ((10, 10, 100, 10), (0, 0, 0)),
            ((10, 50, 100, 50), (1, 0, 0)),
        ])
        result = discover_pdf_content(filepath=path)
        assert "#000000" in result["colors"] or "#ff0000" in result["colors"]

    def test_text_blocks(self, tmp_path):
        path = _make_simple_pdf(tmp_path, texts=[
            ("Sand Layer", (50, 50), 12),
        ])
        result = discover_pdf_content(filepath=path)
        assert any("Sand" in tb["text"] for tb in result["text_blocks"])

    def test_page_size(self, tmp_path):
        path = _make_simple_pdf(tmp_path)
        result = discover_pdf_content(filepath=path)
        assert result["page_size"]["width"] == 400
        assert result["page_size"]["height"] == 300

    def test_bytes_input(self, tmp_path):
        path = _make_simple_pdf(tmp_path, lines=[
            ((10, 10, 100, 10), (0, 0, 0)),
        ])
        with open(path, "rb") as f:
            data = f.read()
        result = discover_pdf_content(content=data)
        assert isinstance(result, dict)
        assert result["n_drawings"] >= 1

    def test_page_out_of_range(self, tmp_path):
        path = _make_simple_pdf(tmp_path)
        with pytest.raises(ValueError, match="out of range"):
            discover_pdf_content(filepath=path, page=5)

    def test_empty_page(self, tmp_path):
        doc = fitz.open()
        doc.new_page()
        path = tmp_path / "empty.pdf"
        doc.save(str(path))
        doc.close()
        result = discover_pdf_content(filepath=str(path))
        assert result["n_drawings"] == 0


# ---------------------------------------------------------------------------
# Tests: extract_vector_geometry
# ---------------------------------------------------------------------------

class TestExtractVectorGeometry:
    def test_returns_pdf_parse_result(self, tmp_path):
        path = _make_cross_section_pdf(tmp_path)
        result = extract_vector_geometry(
            filepath=path,
            role_mapping={"#000000": "surface"},
        )
        assert isinstance(result, PdfParseResult)
        assert result.extraction_method == "vector"
        assert result.confidence == 1.0

    def test_surface_extraction(self, tmp_path):
        path = _make_cross_section_pdf(tmp_path)
        result = extract_vector_geometry(
            filepath=path,
            role_mapping={"#000000": "surface"},
        )
        assert len(result.surface_points) >= 2

    def test_boundary_extraction(self, tmp_path):
        path = _make_cross_section_pdf(tmp_path)
        result = extract_vector_geometry(
            filepath=path,
            role_mapping={
                "#000000": "surface",
                "#7f7f7f": "boundary_Clay",
            },
        )
        assert "Clay" in result.boundary_profiles

    def test_gwt_extraction(self, tmp_path):
        path = _make_cross_section_pdf(tmp_path)
        result = extract_vector_geometry(
            filepath=path,
            role_mapping={
                "#000000": "surface",
                "#0000ff": "gwt",
            },
        )
        assert result.gwt_points is not None

    def test_scale_factor(self, tmp_path):
        path = _make_cross_section_pdf(tmp_path)
        result_unscaled = extract_vector_geometry(
            filepath=path, scale=1.0,
            role_mapping={"#000000": "surface"},
        )
        result_scaled = extract_vector_geometry(
            filepath=path, scale=0.01,
            role_mapping={"#000000": "surface"},
        )
        if result_unscaled.surface_points and result_scaled.surface_points:
            x_unscaled = result_unscaled.surface_points[0][0]
            x_scaled = result_scaled.surface_points[0][0]
            assert abs(x_scaled - x_unscaled * 0.01) < 0.1

    def test_missing_color_warning(self, tmp_path):
        path = _make_cross_section_pdf(tmp_path)
        result = extract_vector_geometry(
            filepath=path,
            role_mapping={"#ff00ff": "surface"},
        )
        assert any("not found" in w for w in result.warnings)

    def test_no_role_mapping(self, tmp_path):
        path = _make_cross_section_pdf(tmp_path)
        result = extract_vector_geometry(filepath=path)
        # Should still extract something from first color
        assert result.extraction_method == "vector"

    def test_bytes_input(self, tmp_path):
        path = _make_cross_section_pdf(tmp_path)
        with open(path, "rb") as f:
            data = f.read()
        result = extract_vector_geometry(
            content=data,
            role_mapping={"#000000": "surface"},
        )
        assert isinstance(result, PdfParseResult)

    def test_page_out_of_range(self, tmp_path):
        path = _make_cross_section_pdf(tmp_path)
        with pytest.raises(ValueError, match="out of range"):
            extract_vector_geometry(filepath=path, page=10)

    def test_text_annotations_extracted(self, tmp_path):
        path = _make_cross_section_pdf(tmp_path)
        result = extract_vector_geometry(filepath=path)
        assert any("Clay" in a["text"] for a in result.text_annotations)

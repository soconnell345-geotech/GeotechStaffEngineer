"""PDF-vector ingest tests (PyMuPDF)."""

import pytest

fitz = pytest.importorskip("fitz")

from drawing_ir import DrawingIR, from_pdf_vector
from drawing_ir.results import Line, Polyline, TextItem


def _single_line_pdf(tmp_path):
    """A 400x300 page with one horizontal line y=100 (top-left) from x=50..150."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    sh = page.new_shape()
    sh.draw_line(fitz.Point(50, 100), fitz.Point(150, 100))
    sh.finish(color=(0, 0, 0), width=1)
    sh.commit()
    path = tmp_path / "line.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


class TestPdfPageSpace:
    def test_default_is_page_space(self, cross_section_pdf):
        ir = from_pdf_vector(cross_section_pdf)
        assert ir.source == "pdf_vector"
        assert ir.coordinate_space == "page"
        assert ir.units == "pt"
        assert ir.scale is None
        assert all(e.confidence == 1.0 for e in ir.entities)

    def test_text_and_scale_candidates(self, cross_section_pdf):
        ir = from_pdf_vector(cross_section_pdf)
        texts = [e for e in ir.entities if isinstance(e, TextItem)]
        assert any("SCALE 1:100" in t.content for t in texts)
        # The scale module proposed a candidate from the "SCALE 1:100" note.
        assert ir.metadata.get("scale_candidates")

    def test_y_flip_bottom_left(self, tmp_path):
        ir = from_pdf_vector(_single_line_pdf(tmp_path), origin="bottom_left")
        seg = [e for e in ir.entities if isinstance(e, (Line, Polyline))][0]
        pts = seg.points()
        # PDF y=100 (from top) on a 300-tall page -> y=200 bottom-left.
        assert all(p[1] == pytest.approx(200.0, abs=0.5) for p in pts)

    def test_top_left_origin_keeps_raw_y(self, tmp_path):
        ir = from_pdf_vector(_single_line_pdf(tmp_path), origin="top_left")
        seg = [e for e in ir.entities if isinstance(e, (Line, Polyline))][0]
        assert all(p[1] == pytest.approx(100.0, abs=0.5) for p in seg.points())


class TestPdfModelSpace:
    def test_explicit_scale(self, tmp_path):
        ir = from_pdf_vector(_single_line_pdf(tmp_path), scale=0.05)
        assert ir.coordinate_space == "model"
        assert ir.units == "m"
        assert ir.scale == pytest.approx(0.05)
        seg = [e for e in ir.entities if isinstance(e, (Line, Polyline))][0]
        # x=50 pt -> 2.5 m ; x=150 -> 7.5 m
        xs = sorted(p[0] for p in seg.points())
        assert xs[0] == pytest.approx(2.5, abs=1e-3)
        assert xs[-1] == pytest.approx(7.5, abs=1e-3)

    def test_two_point_calibration(self, tmp_path):
        # 100 pt of drawing == 10 m  ->  0.1 m/pt.
        ir = from_pdf_vector(
            _single_line_pdf(tmp_path),
            calibration={"p1": [50, 100], "p2": [150, 100], "distance_m": 10.0})
        assert ir.coordinate_space == "model"
        assert ir.scale == pytest.approx(0.1)
        assert ir.scale_provenance == "two_point_calibration"

    def test_round_trip(self, cross_section_pdf):
        ir = from_pdf_vector(cross_section_pdf)
        d = ir.to_dict()
        assert DrawingIR.from_dict(d).to_dict() == d

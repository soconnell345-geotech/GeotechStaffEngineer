"""Raster ingest tests (OpenCV) — tolerance-based, since tracing is inexact."""

import pytest

pytest.importorskip("cv2")

from drawing_ir import DrawingIR, from_raster
from drawing_ir.results import Circle, Line, Polyline


class TestRasterTrace:
    def test_detects_lines_and_circle(self, synthetic_raster_array):
        ir = from_raster(image=synthetic_raster_array, detect_contours=False)
        assert ir.source == "raster_trace"
        assert ir.coordinate_space == "page"
        assert ir.units == "px"
        n_lines = len([e for e in ir.entities if isinstance(e, Line)])
        n_circ = len([e for e in ir.entities if isinstance(e, Circle)])
        assert n_lines >= 2          # two strokes (each edge may trace twice)
        assert n_circ >= 1

    def test_confidence_below_one(self, synthetic_raster_array):
        ir = from_raster(image=synthetic_raster_array, detect_contours=False)
        assert ir.entities
        assert all(e.confidence < 1.0 for e in ir.entities)
        assert all(e.source == "raster_trace" for e in ir.entities)

    def test_ocr_degrades_gracefully(self, synthetic_raster_array):
        ir = from_raster(image=synthetic_raster_array, detect_contours=False)
        # Either OCR produced text, or it warned that it was skipped — never
        # invents text silently.
        from drawing_ir.results import TextItem
        has_text = any(isinstance(e, TextItem) for e in ir.entities)
        warned = any("OCR" in w for w in ir.warnings)
        assert has_text or warned

    def test_scale_promotes_to_model(self, synthetic_raster_array):
        ir = from_raster(image=synthetic_raster_array, scale=0.01,
                         detect_contours=False)
        assert ir.coordinate_space == "model"
        assert ir.units == "m"
        circ = [e for e in ir.entities if isinstance(e, Circle)]
        assert circ  # 40 px radius * 0.01 -> ~0.4 m
        assert 0.2 < circ[0].radius < 0.6

    def test_detector_toggles(self, synthetic_raster_array):
        ir = from_raster(image=synthetic_raster_array, detect_lines=False,
                         detect_circles=True, detect_contours=False, ocr=False)
        assert all(not isinstance(e, Line) for e in ir.entities)
        assert any(isinstance(e, Circle) for e in ir.entities)

    def test_contours_produce_closed_polylines(self, synthetic_raster_array):
        ir = from_raster(image=synthetic_raster_array, detect_lines=False,
                         detect_circles=False, detect_contours=True, ocr=False)
        polys = [e for e in ir.entities if isinstance(e, Polyline)]
        assert polys
        assert all(p.closed for p in polys)

    def test_png_path(self, synthetic_raster_png):
        ir = from_raster(filepath=synthetic_raster_png, detect_contours=False)
        assert ir.entities

    def test_round_trip(self, synthetic_raster_array):
        ir = from_raster(image=synthetic_raster_array, detect_contours=False)
        d = ir.to_dict()
        assert DrawingIR.from_dict(d).to_dict() == d

    def test_requires_source(self):
        with pytest.raises(ValueError):
            from_raster()

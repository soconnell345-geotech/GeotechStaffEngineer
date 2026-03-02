"""Integration tests for pdf_import module.

Tests the conversion bridge between PDF results and slope_stability/fem2d inputs.
"""

import json
import pytest

from pdf_import import to_dxf_parse_result, PdfParseResult
from pdf_import.results import PdfParseResult as PdfResult
from dxf_import.results import DxfParseResult
from dxf_import.converter import (
    SoilPropertyAssignment,
    build_slope_geometry,
    FEMSoilPropertyAssignment,
    build_fem_inputs,
)


def _sample_pdf_result():
    """Create a sample PdfParseResult for testing."""
    return PdfParseResult(
        surface_points=[(0, 10), (10, 10), (20, 5), (30, 5)],
        boundary_profiles={
            "Clay": [(0, 5), (10, 5), (20, 3), (30, 3)],
        },
        gwt_points=[(0, 8), (10, 8), (20, 4), (30, 4)],
        text_annotations=[
            {"text": "Stiff Clay", "x": 15, "y": 4},
        ],
        page_number=0,
        extraction_method="vision",
        scale_factor=1.0,
        confidence=0.8,
    )


class TestToDxfParseResult:
    """Tests for the PdfParseResult → DxfParseResult adapter."""

    def test_returns_dxf_parse_result(self):
        pdf = _sample_pdf_result()
        dxf = to_dxf_parse_result(pdf)
        assert isinstance(dxf, DxfParseResult)

    def test_surface_points_preserved(self):
        pdf = _sample_pdf_result()
        dxf = to_dxf_parse_result(pdf)
        assert len(dxf.surface_points) == 4
        assert dxf.surface_points[0] == (0, 10)

    def test_boundary_profiles_preserved(self):
        pdf = _sample_pdf_result()
        dxf = to_dxf_parse_result(pdf)
        assert "Clay" in dxf.boundary_profiles
        assert len(dxf.boundary_profiles["Clay"]) == 4

    def test_gwt_preserved(self):
        pdf = _sample_pdf_result()
        dxf = to_dxf_parse_result(pdf)
        assert dxf.gwt_points is not None
        assert len(dxf.gwt_points) == 4

    def test_no_gwt(self):
        pdf = PdfParseResult(surface_points=[(0, 10), (20, 10)])
        dxf = to_dxf_parse_result(pdf)
        assert dxf.gwt_points is None

    def test_text_annotations_converted(self):
        pdf = _sample_pdf_result()
        dxf = to_dxf_parse_result(pdf)
        assert len(dxf.text_annotations) == 1
        assert dxf.text_annotations[0]["text"] == "Stiff Clay"
        assert dxf.text_annotations[0]["layer"] == "PDF"

    def test_nail_lines_empty(self):
        pdf = _sample_pdf_result()
        dxf = to_dxf_parse_result(pdf)
        assert dxf.nail_lines == []


class TestPdfToSlopeGeometry:
    """Tests for PDF → DxfParseResult → build_slope_geometry() roundtrip."""

    def test_roundtrip(self):
        pdf = _sample_pdf_result()
        dxf = to_dxf_parse_result(pdf)
        props = [
            SoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0, c_prime=5.0),
            SoilPropertyAssignment(name="Clay", gamma=19.0, phi=25.0, c_prime=10.0),
        ]
        geom = build_slope_geometry(dxf, props)
        assert len(geom.soil_layers) == 2
        assert geom.gwt_points is not None
        assert len(geom.surface_points) == 4


class TestPdfToFEMInputs:
    """Tests for PDF → DxfParseResult → build_fem_inputs() roundtrip."""

    def test_roundtrip(self):
        pdf = _sample_pdf_result()
        dxf = to_dxf_parse_result(pdf)
        props = [
            FEMSoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0, E=40000),
            FEMSoilPropertyAssignment(name="Clay", gamma=19.0, phi=25.0, E=20000),
        ]
        result = build_fem_inputs(dxf, props)
        assert len(result["soil_layers"]) == 2
        assert result["gwt"] is not None
        assert result["soil_layers"][0]["E"] == 40000


class TestPdfParseResultToDict:
    """Tests for PdfParseResult.to_dict() and summary()."""

    def test_to_dict_keys(self):
        pdf = _sample_pdf_result()
        d = pdf.to_dict()
        assert "surface_points" in d
        assert "boundary_profiles" in d
        assert "extraction_method" in d
        assert "confidence" in d
        assert d["extraction_method"] == "vision"

    def test_to_dict_roundtrip(self):
        pdf = _sample_pdf_result()
        d = pdf.to_dict()
        # Verify points are serialized as dicts with x/z keys
        assert d["surface_points"][0] == {"x": 0.0, "z": 10.0}
        assert d["gwt_points"][0] == {"x": 0.0, "z": 8.0}

    def test_summary_contains_method(self):
        pdf = _sample_pdf_result()
        s = pdf.summary()
        assert "vision" in s
        assert "Surface points: 4" in s

    def test_to_dict_json_serializable(self):
        pdf = _sample_pdf_result()
        d = pdf.to_dict()
        # Should be JSON-serializable
        text = json.dumps(d)
        assert isinstance(text, str)

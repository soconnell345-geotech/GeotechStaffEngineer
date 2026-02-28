"""Tests for dxf_import.units — unit conversion and header detection."""

import pytest

ezdxf = pytest.importorskip("ezdxf")

from dxf_import.units import (
    UNIT_FACTORS,
    convert_coords,
    detect_units_from_header,
)


class TestConvertCoords:
    """Tests for convert_coords()."""

    def test_ft_to_m(self):
        pts = [(10.0, 20.0), (30.0, 40.0)]
        result = convert_coords(pts, "ft", "m")
        assert len(result) == 2
        assert abs(result[0][0] - 10.0 * 0.3048) < 1e-6
        assert abs(result[0][1] - 20.0 * 0.3048) < 1e-6
        assert abs(result[1][0] - 30.0 * 0.3048) < 1e-6

    def test_mm_to_m(self):
        pts = [(1000.0, 2000.0)]
        result = convert_coords(pts, "mm", "m")
        assert abs(result[0][0] - 1.0) < 1e-6
        assert abs(result[0][1] - 2.0) < 1e-6

    def test_identity_m_to_m(self):
        pts = [(5.0, 10.0)]
        result = convert_coords(pts, "m", "m")
        assert result[0] == (5.0, 10.0)

    def test_in_to_m(self):
        pts = [(12.0, 24.0)]  # 1 foot = 12 inches
        result = convert_coords(pts, "in", "m")
        assert abs(result[0][0] - 12.0 * 0.0254) < 1e-6

    def test_cm_to_m(self):
        pts = [(100.0, 200.0)]
        result = convert_coords(pts, "cm", "m")
        assert abs(result[0][0] - 1.0) < 1e-6

    def test_unknown_source_unit(self):
        with pytest.raises(ValueError, match="Unknown source unit"):
            convert_coords([(0, 0)], "furlongs", "m")

    def test_unknown_target_unit(self):
        with pytest.raises(ValueError, match="Unknown target unit"):
            convert_coords([(0, 0)], "m", "cubits")

    def test_empty_list(self):
        result = convert_coords([], "ft", "m")
        assert result == []


class TestDetectUnitsFromHeader:
    """Tests for detect_units_from_header()."""

    def test_detect_feet(self):
        doc = ezdxf.new()
        doc.header["$INSUNITS"] = 2
        assert detect_units_from_header(doc) == "ft"

    def test_detect_meters(self):
        doc = ezdxf.new()
        doc.header["$INSUNITS"] = 6
        assert detect_units_from_header(doc) == "m"

    def test_detect_mm(self):
        doc = ezdxf.new()
        doc.header["$INSUNITS"] = 4
        assert detect_units_from_header(doc) == "mm"

    def test_detect_inches(self):
        doc = ezdxf.new()
        doc.header["$INSUNITS"] = 1
        assert detect_units_from_header(doc) == "in"

    def test_no_insunits_returns_none(self):
        doc = ezdxf.new()
        doc.header["$INSUNITS"] = 0  # Unitless
        result = detect_units_from_header(doc)
        assert result is None

    def test_unknown_insunits_returns_none(self):
        doc = ezdxf.new()
        doc.header["$INSUNITS"] = 99  # not a recognized code
        assert detect_units_from_header(doc) is None

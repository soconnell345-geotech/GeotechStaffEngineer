"""DXF ingest tests (ezdxf)."""

import pytest

pytest.importorskip("ezdxf")

from drawing_ir import DrawingIR, from_dxf
from drawing_ir.results import Arc, Circle, Line, Polyline, TextItem


def _by_type(ir, cls):
    return [e for e in ir.entities if isinstance(e, cls)]


class TestDxfIngest:
    def test_entity_types_and_space(self, simple_dxf):
        ir = from_dxf(simple_dxf)
        assert ir.source == "dxf"
        assert ir.coordinate_space == "model"
        assert ir.units == "m"
        assert ir.counts_by_type() == {
            "polyline": 1, "line": 1, "circle": 1, "arc": 1, "text": 1}
        assert all(e.confidence == 1.0 for e in ir.entities)
        assert all(e.source == "dxf" for e in ir.entities)

    def test_exact_coordinates_preserved(self, simple_dxf):
        ir = from_dxf(simple_dxf)
        poly = _by_type(ir, Polyline)[0]
        assert poly.vertices == [(0, 10), (10, 10), (20, 5), (30, 5)]
        ln = _by_type(ir, Line)[0]
        assert ln.start == (5, 8) and ln.end == (25, 3)
        circ = _by_type(ir, Circle)[0]
        assert circ.center == (15, 7) and circ.radius == 2.0
        arc = _by_type(ir, Arc)[0]
        assert arc.center == (10, 4) and arc.radius == 3.0
        assert arc.start_angle == pytest.approx(0)
        assert arc.end_angle == pytest.approx(90)

    def test_layers_and_text(self, simple_dxf):
        ir = from_dxf(simple_dxf)
        layers = set(ir.counts_by_layer())
        assert {"SURFACE", "GWT", "DETAIL", "NOTES"} <= layers
        txt = _by_type(ir, TextItem)[0]
        assert txt.content == "Clay"
        assert txt.position == (12, 6)
        assert txt.layer == "NOTES"

    def test_native_meters_no_scale(self, simple_dxf):
        ir = from_dxf(simple_dxf)
        assert ir.scale is None
        assert ir.scale_provenance == "dxf_native_meters"
        assert ir.metadata["dxf_units"] == "m"

    def test_imperial_units_converted_to_meters(self, imperial_dxf):
        ir = from_dxf(imperial_dxf)
        assert ir.units == "m"
        assert ir.scale == pytest.approx(0.3048)
        assert ir.scale_provenance == "dxf_units:ft->m"
        poly = _by_type(ir, Polyline)[0]
        # (0,30) ft -> (0, 9.144) m ; (100,15) ft -> (30.48, 4.572) m
        assert poly.vertices[0] == pytest.approx((0.0, 9.144))
        assert poly.vertices[-1] == pytest.approx((30.48, 4.572))

    def test_units_override(self, simple_dxf):
        # Force mm interpretation: every coord scales by 0.001.
        ir = from_dxf(simple_dxf, units="mm")
        poly = _by_type(ir, Polyline)[0]
        assert poly.vertices[1] == pytest.approx((0.01, 0.01))

    def test_content_bytes_path(self, simple_dxf):
        with open(simple_dxf, "rb") as f:
            data = f.read()
        ir = from_dxf(content=data)
        assert len(ir.entities) == 5

    def test_round_trip_after_ingest(self, simple_dxf):
        ir = from_dxf(simple_dxf)
        d = ir.to_dict()
        assert DrawingIR.from_dict(d).to_dict() == d

    def test_requires_a_source(self):
        with pytest.raises(ValueError):
            from_dxf()

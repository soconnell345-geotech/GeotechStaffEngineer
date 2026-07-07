"""Schema tests — entities, bbox computation, JSON round-trip."""

import math

import pytest

from drawing_ir.results import (
    Arc, Circle, DrawingIR, Line, Polyline, Region, TextItem,
    entity_from_dict,
)


class TestEntityGeometry:
    def test_line_bbox_length_angle(self):
        ln = Line(start=(0, 0), end=(3, 4))
        assert ln.bbox == (0, 0, 3, 4)
        assert ln.length() == pytest.approx(5.0)
        assert ln.angle_deg() == pytest.approx(53.130102, abs=1e-3)

    def test_line_angle_folds_to_0_180(self):
        # A segment pointing down-left has the same folded angle as up-right.
        assert Line(start=(0, 0), end=(-1, -1)).angle_deg() == pytest.approx(45.0)

    def test_polyline_bbox_length_closed(self):
        pl = Polyline(vertices=[(0, 0), (10, 0), (10, 10)], closed=True)
        assert pl.bbox == (0, 0, 10, 10)
        # open would be 20; closed adds the closing edge back to start (~14.14)
        assert pl.length() == pytest.approx(20 + math.hypot(10, 10))

    def test_circle_bbox(self):
        c = Circle(center=(5, 5), radius=2)
        assert c.bbox == (3, 3, 7, 7)
        assert c.length() == pytest.approx(2 * math.pi * 2)

    def test_arc_bbox_within_circle_bbox(self):
        arc = Arc(center=(0, 0), radius=5, start_angle=0, end_angle=90)
        x0, y0, x1, y1 = arc.bbox
        # Quarter arc in the first quadrant: spans roughly [0,5]x[0,5].
        assert x0 == pytest.approx(0, abs=0.2)
        assert y0 == pytest.approx(0, abs=0.2)
        assert x1 == pytest.approx(5, abs=1e-6)
        assert y1 == pytest.approx(5, abs=1e-6)
        assert arc.length() == pytest.approx(5 * math.pi / 2)

    def test_text_bbox_positive_extent(self):
        t = TextItem(content="Clay", position=(2, 3), rotation=0, height=1.0)
        x0, y0, x1, y1 = t.bbox
        assert (x0, y0) == (2, 3)
        assert x1 > x0 and y1 > y0

    def test_region_area(self):
        r = Region(boundary=[(0, 0), (4, 0), (4, 3), (0, 3)])
        assert r.area() == pytest.approx(12.0)


class TestRoundTrip:
    def _entities(self):
        return [
            Line(id="e0", start=(0, 0), end=(3, 4), layer="L", color="#ff0000"),
            Polyline(id="e1", vertices=[(0, 0), (1, 1), (2, 0)], closed=False,
                     source="pdf_vector"),
            Arc(id="e2", center=(1, 1), radius=2, start_angle=10, end_angle=80),
            Circle(id="e3", center=(5, 5), radius=1.5, confidence=0.5,
                   source="raster_trace"),
            TextItem(id="e4", content="N-1", position=(2, 2), rotation=30,
                     height=0.5),
            Region(id="e5", boundary=[(0, 0), (2, 0), (2, 2)], pattern="EARTH"),
        ]

    def test_entity_round_trip_each_type(self):
        for e in self._entities():
            d = e.to_dict()
            assert d["type"] == e.KIND
            e2 = entity_from_dict(d)
            assert type(e2) is type(e)
            assert e2.to_dict() == d  # idempotent

    def test_entity_from_dict_unknown_type(self):
        with pytest.raises(ValueError):
            entity_from_dict({"type": "nope"})

    def test_drawing_ir_round_trip(self):
        ir = DrawingIR(width=30, height=10, units="m",
                       coordinate_space="model", scale=0.01,
                       scale_provenance="test", source="dxf")
        for e in self._entities():
            ir.entities.append(e)
        d = ir.to_dict()
        ir2 = DrawingIR.from_dict(d)
        assert ir2.to_dict() == d
        assert len(ir2.entities) == 6
        assert ir2.coordinate_space == "model"
        assert ir2.scale == 0.01

    def test_to_dict_can_omit_entities(self):
        ir = DrawingIR()
        ir.add(Line(start=(0, 0), end=(1, 0)))
        d = ir.to_dict(include_entities=False)
        assert "entities" not in d
        assert d["n_entities"] == 1


class TestDrawingIRHelpers:
    def test_add_assigns_ids(self):
        ir = DrawingIR()
        a = ir.add(Line(start=(0, 0), end=(1, 1)))
        b = ir.add(Circle(center=(0, 0), radius=1))
        assert a.id == "e0" and b.id == "e1"
        assert ir.by_id("e1") is b
        assert ir.by_id("zzz") is None

    def test_counts_and_bbox(self):
        ir = DrawingIR()
        ir.add(Line(start=(0, 0), end=(10, 0)))
        ir.add(Line(start=(0, 5), end=(10, 5)))
        ir.add(Circle(center=(20, 0), radius=2))
        assert ir.counts_by_type() == {"line": 2, "circle": 1}
        assert ir.bbox() == (0, -2, 22, 5)

    def test_counts_by_layer(self):
        ir = DrawingIR()
        ir.add(Line(start=(0, 0), end=(1, 0), layer="A"))
        ir.add(Line(start=(0, 0), end=(1, 0)))  # no layer
        cbl = ir.counts_by_layer()
        assert cbl["A"] == 1 and cbl["(none)"] == 1

    def test_summary_is_string(self):
        ir = DrawingIR(source="dxf")
        ir.add(Line(start=(0, 0), end=(1, 1), layer="X"))
        s = ir.summary()
        assert "DRAWING IR" in s and "line" in s

"""Query correctness — hand-built IR with known coordinates in/out."""

import pytest

from drawing_ir import queries
from drawing_ir.results import (
    Circle, DrawingIR, Line, Polyline, TextItem,
)


@pytest.fixture
def ir():
    """A small, fully known drawing.

    e0  surface polyline (0,10)-(10,10)-(20,5)-(30,5)  layer SURFACE
    e1  horizontal line   (0,0)-(30,0)                  layer BASE
    e2  vertical line     (0,0)-(0,10)                   layer LEFT
    e3  diagonal line     (0,0)-(10,10)  (45 deg)        layer CUT
    e4  circle center (15,7) r 2                          layer DETAIL, blue
    e5  text "Clay" at (16,7)                             layer NOTES
    e6  text "Sand" at (2,1)                              layer NOTES
    """
    d = DrawingIR(width=30, height=10, units="m", coordinate_space="model",
                  source="dxf")
    d.add(Polyline(vertices=[(0, 10), (10, 10), (20, 5), (30, 5)],
                   layer="SURFACE"))
    d.add(Line(start=(0, 0), end=(30, 0), layer="BASE"))
    d.add(Line(start=(0, 0), end=(0, 10), layer="LEFT"))
    d.add(Line(start=(0, 0), end=(10, 10), layer="CUT"))
    d.add(Circle(center=(15, 7), radius=2, layer="DETAIL", color="#0000ff"))
    d.add(TextItem(content="Clay", position=(16, 7), height=1, layer="NOTES"))
    d.add(TextItem(content="Sand", position=(2, 1), height=1, layer="NOTES"))
    return d


class TestSpatial:
    def test_entities_in_bbox_intersect(self, ir):
        hits = queries.entities_in_bbox(ir, 13, 5, 17, 9)
        ids = {h["id"] for h in hits}
        assert "e4" in ids  # circle
        assert "e5" in ids  # "Clay" text at (16,7)

    def test_entities_in_bbox_contain_mode(self, ir):
        # Only fully-contained entities. The circle bbox is (13,5,17,9).
        hits = queries.entities_in_bbox(ir, 12, 4, 18, 10, mode="contain")
        ids = {h["id"] for h in hits}
        assert "e4" in ids
        assert "e1" not in ids  # base line spans x 0..30, not contained

    def test_entities_in_bbox_type_filter(self, ir):
        hits = queries.entities_in_bbox(ir, 0, 0, 30, 10, entity_type="text")
        assert {h["id"] for h in hits} == {"e5", "e6"}

    def test_nearest_entity(self, ir):
        # (2,1) coincides with the "Sand" text insertion; everything else is
        # farther (base line 1.0 below, diagonal ~0.71, vertical 2.0 away).
        near = queries.nearest_entity(ir, 2.0, 1.0, k=1)
        assert near[0]["id"] == "e6"
        assert near[0]["distance"] == pytest.approx(0.0, abs=1e-6)

    def test_nearest_entity_ring_distance(self, ir):
        # Straight up from the circle center, clear of the weaving surface.
        near = queries.nearest_entity(ir, 15.0, 2.0, entity_type="circle")
        assert near[0]["id"] == "e4"
        # center (15,7) r2 -> point (15,2) is |5 - 2| = 3.0 from the ring.
        assert near[0]["distance"] == pytest.approx(3.0, abs=1e-6)

    def test_nearest_entity_type_filter(self, ir):
        near = queries.nearest_entity(ir, 2.0, 1.0, entity_type="text", k=1)
        assert near[0]["id"] == "e6"


class TestAngles:
    def test_horizontal_lines(self, ir):
        hits = queries.horizontal_lines(ir)
        ids = {h["entity_id"] for h in hits}
        assert "e1" in ids           # base horizontal line
        assert "e0" in ids           # polyline has flat top segments
        assert "e2" not in ids       # vertical

    def test_vertical_lines(self, ir):
        hits = queries.vertical_lines(ir)
        assert {h["entity_id"] for h in hits} == {"e2"}

    def test_lines_by_angle_band(self, ir):
        hits = queries.lines_by_angle(ir, 40, 50)
        # The 45-degree diagonal e3, plus the descending polyline segments.
        ids = {h["entity_id"] for h in hits}
        assert "e3" in ids

    def test_lines_by_angle_reports_geometry(self, ir):
        hits = queries.lines_by_angle(ir, 44, 46)
        e3 = [h for h in hits if h["entity_id"] == "e3"][0]
        assert e3["angle_deg"] == pytest.approx(45.0)
        assert e3["length"] == pytest.approx(14.1421, abs=1e-3)


class TestLengthLayerColor:
    def test_polylines_longer_than(self, ir):
        hits = queries.polylines_longer_than(ir, 10)
        assert len(hits) == 1 and hits[0]["id"] == "e0"
        assert queries.polylines_longer_than(ir, 1e6) == []

    def test_entities_on_layer(self, ir):
        assert {h["id"] for h in queries.entities_on_layer(ir, "NOTES")} == {
            "e5", "e6"}

    def test_entities_by_color(self, ir):
        hits = queries.entities_by_color(ir, "#0000FF")  # case-insensitive
        assert {h["id"] for h in hits} == {"e4"}


class TestText:
    def test_text_items_all(self, ir):
        assert {h["content"] for h in queries.text_items(ir)} == {"Clay", "Sand"}

    def test_text_items_pattern(self, ir):
        hits = queries.text_items(ir, pattern="cla")
        assert {h["content"] for h in hits} == {"Clay"}

    def test_text_near_entity(self, ir):
        # "Clay" (16,7) sits on the circle e4; "Sand" (2,1) is far.
        hits = queries.text_near(ir, "e4", radius=2.0)
        assert {h["content"] for h in hits} == {"Clay"}

    def test_text_near_unknown_id(self, ir):
        out = queries.text_near(ir, "zzz", radius=1)
        assert "error" in out[0]


class TestHeuristicsAndStats:
    def test_candidate_ground_surface(self, ir):
        cand = queries.candidate_ground_surface(ir)
        # e0 (width 30) and e1 (width 30) tie; tie broken toward higher-y (e0).
        assert cand["candidate"]["id"] == "e0"
        assert cand["proposal_only"] is True
        assert cand["width"] == pytest.approx(30.0)

    def test_candidate_ground_surface_empty(self):
        empty = DrawingIR()
        empty.add(TextItem(content="x", position=(0, 0), height=1))
        assert queries.candidate_ground_surface(empty)["candidate"] is None

    def test_summary_stats(self, ir):
        s = queries.summary_stats(ir)
        assert s["n_entities"] == 7
        assert s["counts_by_type"]["line"] == 3
        assert s["counts_by_type"]["text"] == 2
        assert s["page"]["coordinate_space"] == "model"

    def test_get_entities_selective(self, ir):
        got = queries.get_entities(ir, ["e0", "missing"])
        assert got[0]["type"] == "polyline"
        assert "vertices" in got[0]
        assert got[1]["error"] == "not found"

"""Geometric correctness tests for B4: entities_ending_near + text_anchored_geometry.

A small hand-built IR with known coordinates (same style as test_queries.py's
``ir`` fixture) so every expected match/miss is exact, not fixture-derived.
"""

import pytest

from drawing_ir import queries
from drawing_ir.results import (
    Arc, Circle, DrawingIR, Line, Polyline, TextItem,
)


@pytest.fixture
def ir():
    """
    e0  Line (0,0)-(10,0)                       -- endpoints (0,0),(10,0)
    e1  Line (10,0)-(10,10)                      -- endpoints (10,0),(10,10)
        (e0.end and e1.start coincide at (10,0) -- a shared vertex)
    e2  open Polyline (50,50)-(60,60)-(70,50)    -- endpoints (50,50),(70,50)
    e3  closed Polyline (200,200)-(210,200)-(205,210) -- a small triangle
    e4  Circle center (0,0) r 5                   -- NO endpoints
    e5  Arc center (100,100) r 10, 0->90 deg      -- endpoints (110,100),(100,110)
    e6  TextItem "SAE" at (70,50.4)               -- near e2's "end" endpoint
    e7  TextItem "OTHER" at (500,500)              -- isolated, nothing nearby
    """
    d = DrawingIR(width=600, height=600, units="m", coordinate_space="model")
    d.add(Line(start=(0, 0), end=(10, 0)))
    d.add(Line(start=(10, 0), end=(10, 10)))
    d.add(Polyline(vertices=[(50, 50), (60, 60), (70, 50)], closed=False))
    d.add(Polyline(vertices=[(200, 200), (210, 200), (205, 210)], closed=True))
    d.add(Circle(center=(0, 0), radius=5))
    d.add(Arc(center=(100, 100), radius=10, start_angle=0, end_angle=90))
    d.add(TextItem(content="SAE", position=(70, 50.4), height=2))
    d.add(TextItem(content="OTHER", position=(500, 500), height=2))
    return d


class TestEntitiesEndingNear:
    def test_shared_vertex_matches_both_lines(self, ir):
        hits = queries.entities_ending_near(ir, (10, 0), 0.5)
        ids = {h["id"] for h in hits}
        assert ids == {"e0", "e1"}
        by_id = {h["id"]: h for h in hits}
        assert by_id["e0"]["end"] == "end"
        assert by_id["e0"]["end_point"] == [10.0, 0.0]
        assert by_id["e0"]["other_end"] == [0.0, 0.0]
        assert by_id["e1"]["end"] == "start"
        assert by_id["e1"]["other_end"] == [10.0, 10.0]

    def test_all_results_within_distance_zero(self, ir):
        hits = queries.entities_ending_near(ir, (10, 0), 0.5)
        assert all(h["distance"] == pytest.approx(0.0, abs=1e-9) for h in hits)

    def test_polyline_endpoint(self, ir):
        hits = queries.entities_ending_near(ir, (70, 50), 1.0)
        assert len(hits) == 1
        assert hits[0]["id"] == "e2"
        assert hits[0]["end"] == "end"
        assert hits[0]["other_end"] == [50.0, 50.0]

    def test_polyline_start_endpoint(self, ir):
        hits = queries.entities_ending_near(ir, (50, 50), 1.0)
        assert len(hits) == 1
        assert hits[0]["id"] == "e2"
        assert hits[0]["end"] == "start"
        assert hits[0]["other_end"] == [70.0, 50.0]

    def test_polyline_bend_is_not_an_endpoint(self, ir):
        # (60,60) is a mid-vertex bend, not a terminal point.
        hits = queries.entities_ending_near(ir, (60, 60), 1.0)
        assert hits == []

    def test_arc_endpoints_computed_correctly(self, ir):
        hits = queries.entities_ending_near(ir, (110, 100), 0.5)
        assert len(hits) == 1
        assert hits[0]["id"] == "e5"
        assert hits[0]["end"] == "start"
        assert hits[0]["end_point"] == pytest.approx([110.0, 100.0])
        assert hits[0]["other_end"] == pytest.approx([100.0, 110.0])

        hits2 = queries.entities_ending_near(ir, (100, 110), 0.5)
        assert hits2[0]["id"] == "e5"
        assert hits2[0]["end"] == "end"

    def test_circle_never_matches(self, ir):
        # Even a point ON the circle boundary (5,0) never matches -- a circle
        # has no endpoints.
        hits = queries.entities_ending_near(ir, (5, 0), 2.0)
        assert all(h["id"] != "e4" for h in hits)

    def test_text_never_matches(self, ir):
        hits = queries.entities_ending_near(ir, (70, 50.4), 2.0)
        assert all(h["type"] != "text" for h in hits)

    def test_radius_boundary(self, ir):
        # (70,50) to (70+0.99,50) is just inside radius=1.0.
        assert queries.entities_ending_near(ir, (70.99, 50), 1.0)
        # radius=0.5 excludes it.
        assert not queries.entities_ending_near(ir, (70.99, 50), 0.5)

    def test_entity_types_filter(self, ir):
        hits = queries.entities_ending_near(ir, (10, 0), 0.5,
                                            entity_types=["line"])
        assert {h["id"] for h in hits} == {"e0", "e1"}
        hits_none = queries.entities_ending_near(ir, (10, 0), 0.5,
                                                 entity_types=["polyline"])
        assert hits_none == []

    def test_closed_polyline_endpoints_first_last_vertex(self, ir):
        hits = queries.entities_ending_near(ir, (200, 200), 0.5)
        assert len(hits) == 1 and hits[0]["id"] == "e3"
        assert hits[0]["end"] == "start"
        assert hits[0]["other_end"] == [205.0, 210.0]

    def test_sorted_nearest_first(self, ir):
        # Two lines end at (10,0) exactly (distance 0); adding a point offset
        # slightly toward e1's endpoint should not break sort stability for
        # tied distances, and a real distance gradient should still sort.
        hits = queries.entities_ending_near(ir, (10, 0.3), 1.0)
        dists = [h["distance"] for h in hits]
        assert dists == sorted(dists)

    def test_empty_when_nothing_in_radius(self, ir):
        assert queries.entities_ending_near(ir, (300, 300), 5.0) == []


class TestTextAnchoredGeometry:
    def test_finds_connected_geometry(self, ir):
        out = queries.text_anchored_geometry(ir, "SAE")
        assert len(out) == 1
        entry = out[0]
        assert entry["text"]["content"] == "SAE"
        assert entry["anchor"] == [70.0, 50.4]
        assert entry["connected"]
        assert entry["connected"][0]["id"] == "e2"
        assert entry["points_at"] == [50.0, 50.0]
        assert entry["proposal_only"] is True

    def test_isolated_text_has_no_connection(self, ir):
        out = queries.text_anchored_geometry(ir, "OTHER")
        assert len(out) == 1
        assert out[0]["connected"] == []
        assert out[0]["points_at"] is None

    def test_pattern_no_match_returns_empty(self, ir):
        assert queries.text_anchored_geometry(ir, "NOPE_NOT_THERE") == []

    def test_explicit_radius_overrides_default(self, ir):
        # e6 text at (70,50.4) is 0.4 above e2's endpoint (70,50); with a
        # radius smaller than 0.4 nothing should connect.
        out = queries.text_anchored_geometry(ir, "SAE", radius=0.1)
        assert out[0]["connected"] == []
        assert out[0]["search_radius"] == pytest.approx(0.1)

    def test_default_radius_derived_from_text_height(self, ir):
        # height=2 -> default radius = 3*2 = 6.0.
        out = queries.text_anchored_geometry(ir, "SAE")
        assert out[0]["search_radius"] == pytest.approx(6.0)

    def test_default_radius_small_text_not_floored(self):
        # Model-space regression: small (but known) text height must give
        # 3*height, NOT a fixed 5.0-unit floor — in meters a 5 m reach would
        # connect geometry across half a cross-section. Here 3*0.2 = 0.6, so
        # a line ending 1.0 away must NOT connect.
        d = DrawingIR(width=100, height=100, units="m",
                      coordinate_space="model")
        d.add(Line(start=(10, 11), end=(30, 25)))       # nearest end 1.0 away
        d.add(TextItem(content="N1", position=(10, 10), height=0.2))
        out = queries.text_anchored_geometry(d, "N1")
        assert out[0]["search_radius"] == pytest.approx(0.6)
        assert out[0]["connected"] == []

    def test_default_radius_unknown_height_falls_back(self):
        # height=0 (unknown) -> the 5.0-unit fallback applies.
        d = DrawingIR(width=100, height=100, units="pt")
        d.add(TextItem(content="N2", position=(50, 50), height=0.0))
        out = queries.text_anchored_geometry(d, "N2")
        assert out[0]["search_radius"] == pytest.approx(5.0)

    def test_entity_types_filter_passthrough(self, ir):
        out = queries.text_anchored_geometry(ir, "SAE", entity_types=["arc"])
        assert out[0]["connected"] == []   # e2 is a polyline, filtered out

    def test_regex_pattern(self, ir):
        out = queries.text_anchored_geometry(ir, "^SA")
        assert len(out) == 1 and out[0]["text"]["content"] == "SAE"

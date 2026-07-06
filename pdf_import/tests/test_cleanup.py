"""Tests for pdf_import.cleanup — C3 geometry cleanup (v5.3)."""

import math

import pytest

from pdf_import.cleanup import (
    dedupe_consecutive, merge_collinear, cleanup_polyline,
    snap_endpoints, join_polylines, cleanup_geometry,
)


class TestDedupe:
    def test_removes_consecutive_duplicates(self):
        assert dedupe_consecutive([(0, 0), (0, 0), (1, 1), (1, 1), (2, 2)]) == \
            [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]

    def test_keeps_distinct(self):
        pts = [(0, 0), (1, 0), (2, 0)]
        assert dedupe_consecutive(pts) == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]

    def test_tolerance(self):
        assert dedupe_consecutive([(0, 0), (0.0005, 0), (1, 0)], tol=1e-3) == \
            [(0.0, 0.0), (1.0, 0.0)]


class TestMergeCollinear:
    def test_thins_straight_run(self):
        assert merge_collinear([(0, 0), (1, 0), (2, 0), (3, 0)]) == \
            [(0.0, 0.0), (3.0, 0.0)]

    def test_keeps_corner(self):
        # an L-shape: the 90-degree corner at (3,0) must survive
        assert merge_collinear([(0, 0), (1, 0), (2, 0), (3, 0), (3, 2), (3, 4)]) == \
            [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]

    def test_keeps_real_bend(self):
        out = merge_collinear([(0, 0), (5, 0), (10, 1)], angle_tol_deg=1.0)
        assert len(out) == 3            # ~11 deg bend -> kept

    def test_short_polyline_unchanged(self):
        assert merge_collinear([(0, 0), (1, 1)]) == [(0.0, 0.0), (1.0, 1.0)]


class TestSnapEndpoints:
    def test_snaps_near_endpoints(self):
        out = snap_endpoints([[(0, 0), (10, 0)], [(10.0005, 0), (20, 0)]], tol=1e-2)
        # the two touching endpoints now share the exact averaged vertex
        assert out[0][-1] == out[1][0]

    def test_leaves_interior_untouched(self):
        out = snap_endpoints([[(0, 0), (5, 0.4), (10, 0)]], tol=1e-2)
        assert out[0][1] == (5.0, 0.4)

    def test_far_endpoints_not_snapped(self):
        out = snap_endpoints([[(0, 0), (10, 0)], [(50, 0), (60, 0)]], tol=1e-2)
        assert out[0][-1] == (10.0, 0.0)
        assert out[1][0] == (50.0, 0.0)


class TestJoinPolylines:
    def test_joins_chain(self):
        out = join_polylines([[(0, 0), (10, 0)], [(10, 0), (20, 0)],
                              [(20, 0), (30, 0)]], tol=1e-6)
        assert len(out) == 1
        assert out[0] == [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]

    def test_joins_with_reversal(self):
        # second segment is reversed but shares the endpoint
        out = join_polylines([[(0, 0), (10, 0)], [(20, 0), (10, 0)]], tol=1e-6)
        assert len(out) == 1
        assert out[0][0] == (0.0, 0.0) and out[0][-1] == (20.0, 0.0)

    def test_disjoint_stay_separate(self):
        out = join_polylines([[(0, 0), (10, 0)], [(50, 0), (60, 0)]], tol=1e-6)
        assert len(out) == 2


class TestCleanupGeometry:
    def test_report_and_thinning(self):
        r = cleanup_geometry(
            surface_points=[(0, 0), (1, 0), (2, 0), (3, 0)],
            boundary_profiles={"Clay": [(0, -5), (1, -5), (2, -5)]},
            gwt_points=[(0, -2), (1, -2), (2, -2)])
        assert r["surface_points"] == [(0.0, 0.0), (3.0, 0.0)]
        assert r["boundary_profiles"]["Clay"] == [(0.0, -5.0), (2.0, -5.0)]
        assert r["report"]["before"]["surface"] == 4
        assert r["report"]["after"]["surface"] == 2
        assert r["report"]["after"]["gwt"] == 2

    def test_non_destructive(self):
        surf = [(0, 0), (1, 0), (2, 0)]
        cleanup_geometry(surface_points=surf)
        assert surf == [(0, 0), (1, 0), (2, 0)]     # input untouched

    def test_snaps_surface_to_boundary_endpoint(self):
        r = cleanup_geometry(
            surface_points=[(0, 0), (10.0005, 0)],
            boundary_profiles={"Clay": [(10.0, 0), (20, -5)]},
            snap_tol=1e-2)
        assert r["surface_points"][-1] == r["boundary_profiles"]["Clay"][0]

    def test_empty_inputs(self):
        r = cleanup_geometry()
        assert r["surface_points"] == []
        assert r["boundary_profiles"] == {}
        assert r["gwt_points"] is None

"""Tests for pdf_import.crosscheck — C5 vision<->vector cross-check (v5.3)."""

import pytest

from pdf_import.crosscheck import cross_check, polyline_deviation
from pdf_import.results import PdfParseResult


class TestPolylineDeviation:
    def test_identical(self):
        d = polyline_deviation([(0, 10), (20, 10)], [(0, 10), (20, 10)])
        assert d["max"] == 0.0 and d["rms"] == 0.0

    def test_constant_offset(self):
        d = polyline_deviation([(0, 10), (20, 10)], [(0, 12), (20, 12)])
        assert d["max"] == pytest.approx(2.0)
        assert d["mean"] == pytest.approx(2.0)

    def test_no_x_overlap(self):
        assert polyline_deviation([(0, 0), (5, 0)], [(10, 0), (20, 0)]) is None

    def test_empty(self):
        assert polyline_deviation([], [(0, 0)]) is None

    def test_accepts_dict_points(self):
        d = polyline_deviation([{"x": 0, "z": 10}, {"x": 20, "z": 10}],
                               [{"x": 0, "z": 11}, {"x": 20, "z": 11}])
        assert d["max"] == pytest.approx(1.0)


class TestCrossCheck:
    def _vec(self):
        return {"surface_points": [(0, 10), (10, 10), (20, 5)],
                "boundary_profiles": {"Clay": [(0, 0), (20, 0)]},
                "gwt_points": [(0, 3), (20, 3)]}

    def test_agreement_within_tol(self):
        vec = self._vec()
        vis = {"surface_points": [(0, 10.2), (10, 10.1), (20, 5.2)],
               "boundary_profiles": {"Clay": [(0, 0.1), (20, -0.1)]},
               "gwt_points": [(0, 3.1), (20, 2.9)]}
        r = cross_check(vec, vis, tol=0.5)
        assert r["agree"] is True
        assert r["surface"]["agrees"]
        assert "nothing merged" in r["note"]

    def test_feature_only_in_one(self):
        vec = self._vec()
        vis = {"surface_points": [(0, 10), (20, 5)],
               "boundary_profiles": {"Clay": [(0, 0), (20, 0)]},
               "gwt_points": None}     # vision missed GWT
        r = cross_check(vec, vis, tol=0.5)
        assert r["agree"] is False
        assert r["gwt"]["in_vector"] and not r["gwt"]["in_vision"]
        assert "only in vector" in r["gwt"]["note"]

    def test_large_deviation_flagged(self):
        vec = self._vec()
        vis = {"surface_points": [(0, 30), (20, 25)],   # way off
               "boundary_profiles": {"Clay": [(0, 0), (20, 0)]},
               "gwt_points": [(0, 3), (20, 3)]}
        r = cross_check(vec, vis, tol=0.5)
        assert r["surface"]["agrees"] is False
        assert r["surface"]["deviation"]["max"] > 0.5
        assert r["agree"] is False

    def test_extra_boundary_in_vision(self):
        vec = self._vec()
        vis = dict(self._vec())
        vis["boundary_profiles"] = {"Clay": [(0, 0), (20, 0)],
                                    "Sand": [(0, -5), (20, -5)]}
        r = cross_check(vec, vis, tol=0.5)
        assert "Sand" in r["boundaries"]
        assert r["boundaries"]["Sand"]["note"] == "only in vision"
        assert r["agree"] is False

    def test_accepts_pdfparseresult(self):
        vec = PdfParseResult(surface_points=[(0, 10), (20, 10)])
        vis = PdfParseResult(surface_points=[(0, 10.1), (20, 10.1)])
        r = cross_check(vec, vis, tol=0.5)
        assert r["surface"]["agrees"]

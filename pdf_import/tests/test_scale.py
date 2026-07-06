"""Tests for pdf_import.scale — C1 scale calibration (v5.3)."""

import math

import pytest

from pdf_import.scale import (
    calibrate_scale, parse_scale_annotations, propose_scale,
)

_M_PER_PT = 0.0254 / 72.0


class TestCalibrateScale:
    def test_two_point_exact(self):
        # 100 drawing units span 10 m -> 0.1 m per unit
        assert calibrate_scale((0, 0), (100, 0), 10.0) == pytest.approx(0.1)

    def test_diagonal_distance(self):
        s = calibrate_scale((0, 0), (30, 40), 5.0)   # hypot=50
        assert s == pytest.approx(0.1)

    def test_bad_distance(self):
        with pytest.raises(ValueError):
            calibrate_scale((0, 0), (100, 0), 0.0)
        with pytest.raises(ValueError):
            calibrate_scale((0, 0), (100, 0), -1.0)

    def test_coincident_points(self):
        with pytest.raises(ValueError):
            calibrate_scale((5, 5), (5, 5), 10.0)


class TestParseScaleAnnotations:
    def test_ratio_scale(self):
        c = parse_scale_annotations([{"text": "SCALE 1:100"}])
        assert len(c) == 1
        assert c[0]["basis"] == "ratio_1_to_N"
        assert c[0]["scale_factor"] == pytest.approx(100 * _M_PER_PT)
        assert c[0]["applied"] is False
        assert "1:100" in c[0]["provenance"] or "SCALE 1:100" in c[0]["provenance"]

    def test_ratio_confidence_higher_with_keyword(self):
        with_kw = parse_scale_annotations([{"text": "SCALE 1:200"}])[0]
        without = parse_scale_annotations([{"text": "detail 1:200"}])[0]
        assert with_kw["confidence"] > without["confidence"]

    def test_engineering_imperial(self):
        for txt in ('1" = 20 ft', "1 in = 20 ft", '1"=20\''):
            c = parse_scale_annotations([{"text": txt}])
            assert c, txt
            assert c[0]["basis"] == "engineering_imperial"
            assert c[0]["scale_factor"] == pytest.approx(20 * 0.3048 / 72.0)

    def test_engineering_in_to_m(self):
        c = parse_scale_annotations([{"text": '1" = 5 m'}])
        assert c[0]["basis"] == "engineering_in_to_m"
        assert c[0]["scale_factor"] == pytest.approx(5.0 / 72.0)

    def test_metric_cm(self):
        c = parse_scale_annotations([{"text": "1 cm = 2 m"}])
        assert c[0]["basis"] == "metric_cm"
        # 1 pt = 2.54/72 cm of paper; 1 cm -> 2 m
        assert c[0]["scale_factor"] == pytest.approx((2.54 / 72.0) * 2.0)

    def test_no_scale_text(self):
        assert parse_scale_annotations([{"text": "Borehole BH-1"},
                                        {"text": ""}]) == []

    def test_dedup_and_sorted_by_confidence(self):
        c = parse_scale_annotations([
            {"text": "SCALE 1:100"}, {"text": "SCALE 1:100"},
            {"text": '1" = 20 ft'}])
        bases = [x["basis"] for x in c]
        assert bases.count("ratio_1_to_N") == 1            # deduped
        confs = [x["confidence"] for x in c]
        assert confs == sorted(confs, reverse=True)         # sorted desc

    def test_ratio_ignores_one_to_one(self):
        assert parse_scale_annotations([{"text": "1:1 detail"}]) == []


class TestProposeScale:
    def test_calibration_is_recommended_and_deterministic(self):
        out = propose_scale(
            [{"text": "SCALE 1:100"}],
            calibration={"p1": [0, 0], "p2": [100, 0], "distance_m": 10.0})
        rec = out["recommended"]
        assert rec["basis"] == "two_point_calibration"
        assert rec["scale_factor"] == pytest.approx(0.1)
        assert rec["confidence"] == 1.0
        assert rec["applied"] is False
        # both the calibration and the annotation candidate are present
        assert len(out["candidates"]) == 2
        assert "NOT applied" in out["note"]

    def test_no_candidates(self):
        out = propose_scale([{"text": "no scale here"}])
        assert out["candidates"] == []
        assert out["recommended"] is None

    def test_text_only(self):
        out = propose_scale([{"text": '1" = 40 ft'}])
        assert out["recommended"]["basis"] == "engineering_imperial"

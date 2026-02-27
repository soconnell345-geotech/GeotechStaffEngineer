"""Tests for statistics.py — trend regression and variance bands."""

import pytest
import numpy as np

from subsurface_characterization.statistics import (
    compute_trend,
    compute_grouped_trends,
    TrendAnalysisResult,
)
from subsurface_characterization.site_model import SiteModel


class TestComputeTrend:
    def test_linear_perfect(self):
        """Perfect linear data: value = 2*depth + 5."""
        depths = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        values = 2 * depths + 5
        result = compute_trend(depths, values, parameter="N_spt")
        assert result.slope == pytest.approx(2.0, abs=0.01)
        assert result.intercept == pytest.approx(5.0, abs=0.01)
        assert result.r_squared == pytest.approx(1.0, abs=0.001)
        assert result.n_points == 10

    def test_linear_with_noise(self):
        """Linear trend with noise — R² < 1 but > 0."""
        np.random.seed(42)
        depths = np.arange(1, 21, dtype=float)
        values = 3 * depths + 10 + np.random.normal(0, 5, 20)
        result = compute_trend(depths, values, parameter="cu_kPa")
        assert result.trend_type == "linear"
        assert 0.5 < result.r_squared < 1.0
        assert result.std_residual > 0

    def test_log_linear(self):
        """Log-linear trend: value = exp(0.5 * ln(depth) + 2)."""
        depths = np.array([1, 2, 4, 8, 16], dtype=float)
        values = np.exp(0.5 * np.log(depths) + 2)
        result = compute_trend(depths, values, trend_type="log_linear", parameter="cu_kPa")
        assert result.trend_type == "log_linear"
        assert result.slope == pytest.approx(0.5, abs=0.01)
        assert result.intercept == pytest.approx(2.0, abs=0.01)
        assert result.r_squared == pytest.approx(1.0, abs=0.001)

    def test_edge_case_single_point(self):
        """With n=1, should return zeroed result."""
        result = compute_trend(np.array([5.0]), np.array([10.0]))
        assert result.n_points == 1
        assert result.r_squared == 0.0
        assert result.slope == 0.0

    def test_edge_case_two_points(self):
        """With n=2, should fit exactly."""
        result = compute_trend(np.array([1.0, 5.0]), np.array([10.0, 30.0]))
        assert result.n_points == 2
        assert result.slope == pytest.approx(5.0, abs=0.01)
        assert result.r_squared == pytest.approx(1.0, abs=0.01)

    def test_predict_linear(self):
        depths = np.array([1, 2, 3, 4, 5], dtype=float)
        values = 2 * depths + 10
        result = compute_trend(depths, values)
        assert result.predict(3.0) == pytest.approx(16.0, abs=0.1)
        assert result.predict(0.0) == pytest.approx(10.0, abs=0.1)

    def test_band(self):
        depths = np.arange(1, 11, dtype=float)
        values = 2 * depths + 10 + np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        result = compute_trend(depths, values)
        lower, upper = result.band(5.0, n_sigma=1.0)
        assert lower < result.predict(5.0) < upper

    def test_cov(self):
        """COV = std_residual / mean_value."""
        depths = np.arange(1, 11, dtype=float)
        values = 2 * depths + 10 + np.array([2, -2, 2, -2, 2, -2, 2, -2, 2, -2], dtype=float)
        result = compute_trend(depths, values)
        assert result.cov > 0
        assert result.cov < 1.0  # should be small for nearly linear data

    def test_to_dict(self):
        result = TrendAnalysisResult(parameter="N_spt", n_points=10, slope=2.0, intercept=5.0)
        d = result.to_dict()
        assert d["parameter"] == "N_spt"
        assert d["slope"] == 2.0

    def test_summary(self):
        result = TrendAnalysisResult(parameter="N_spt", n_points=10, r_squared=0.85, cov=0.15)
        s = result.summary()
        assert "N_spt" in s
        assert "0.85" in s


class TestComputeGroupedTrends:
    def test_group_by_uscs(self, rich_site):
        results = compute_grouped_trends(rich_site, "N_spt", group_by="uscs")
        assert isinstance(results, dict)
        # Should have groups for USCS classes present in the data
        assert len(results) > 0
        for label, trend in results.items():
            assert trend.group_label == label
            assert trend.n_points > 0

    def test_group_by_investigation(self, simple_site):
        results = compute_grouped_trends(simple_site, "N_spt", group_by="investigation")
        assert "B-1" in results
        assert "B-2" in results
        assert results["B-1"].n_points == 8
        assert results["B-2"].n_points == 6

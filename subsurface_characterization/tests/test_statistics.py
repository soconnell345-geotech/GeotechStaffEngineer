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
        """With fewer than 2 points there is no line: refused, never a
        result with slope 0 and R-squared 0 (GM-326)."""
        with pytest.raises(ValueError, match="at least 2 points; 1 given"):
            compute_trend(np.array([5.0]), np.array([10.0]))
        with pytest.raises(ValueError, match="at least 2 points; 0 given"):
            compute_trend(np.array([]), np.array([]))

    def test_unequal_lengths_refused(self):
        with pytest.raises(ValueError, match="one value per depth"):
            compute_trend([1.0, 2.0, 3.0], [10.0, 12.0])

    def test_two_points_say_scatter_is_not_defined(self):
        """A line through 2 points fits both: R-squared is 1 by
        construction, and the scatter and COV are null with a note."""
        d = compute_trend([1.0, 3.0], [10.0, 20.0]).to_dict()
        assert d["n_points"] == 2 and d["fitted"] is True
        assert d["r_squared"] == 1.0
        assert d["std_residual"] is None and d["cov"] is None
        assert "exactly 2 points" in d["notes"][0]

    def test_log_fit_counts_what_it_dropped(self):
        """ln(depth) and ln(value) need positive numbers: the points that
        cannot enter the fit are counted, and n_points is what was used."""
        depths = [0.0, 1.0, 2.0, 4.0, 8.0, 3.0]
        values = [5.0, 10.0, 20.0, 40.0, 80.0, -1.0]
        r = compute_trend(depths, values, trend_type="log_linear")
        assert r.n_points == 4 and r.n_dropped == 2
        assert r.slope == pytest.approx(1.0) and r.r_squared == pytest.approx(1.0)
        d = r.to_dict()
        assert d["n_points"] == 4 and d["n_dropped"] == 2
        assert "2 of 6 points left out" in d["notes"][0]
        # the same data with nothing to drop reports none
        clean = compute_trend(depths[1:5], values[1:5],
                              trend_type="log_linear").to_dict()
        assert clean["n_dropped"] == 0 and "notes" not in clean
        # a linear fit drops nothing
        assert compute_trend(depths, values).n_dropped == 0

    def test_log_fit_with_too_few_positive_points_refused(self):
        with pytest.raises(ValueError, match="1 of the 3 given qualify"):
            compute_trend([0.0, 1.0, 2.0], [5.0, 10.0, -3.0],
                          trend_type="log_linear")

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


class TestGroupsThatCannotBeFitted:
    """A group with too few points is kept and said, never a zero fit."""

    def _site(self):
        from subsurface_characterization.site_model import (
            Investigation, PointMeasurement,
        )
        return SiteModel(investigations=[
            Investigation(investigation_id="B-1", measurements=[
                PointMeasurement(1.0, "N_spt", 5, "field", "SPT"),
                PointMeasurement(2.0, "N_spt", 9, "field", "SPT"),
                PointMeasurement(3.0, "N_spt", 13, "field", "SPT")]),
            Investigation(investigation_id="B-2", measurements=[
                PointMeasurement(1.5, "N_spt", 7, "field", "SPT")]),
        ])

    def test_grouped_trends_flag_the_group(self):
        results = compute_grouped_trends(self._site(), "N_spt",
                                         group_by="investigation")
        assert results["B-1"].fitted and results["B-1"].slope == pytest.approx(4.0)
        lone = results["B-2"]
        assert lone.fitted is False and lone.n_points == 1
        d = lone.to_dict()
        assert d["slope"] is None and d["r_squared"] is None
        assert d["cov"] is None and "at least 2 points" in d["notes"][0]
        assert "not fitted" in lone.summary()
        with pytest.raises(ValueError, match="No trend was fitted"):
            lone.predict(2.0)

    def test_the_plot_names_what_it_did_not_fit(self):
        pytest.importorskip("plotly")
        from subsurface_characterization.plots_xy import (
            plot_parameter_vs_depth,
        )
        from subsurface_characterization.site_model import (
            Investigation, PointMeasurement,
        )
        one = SiteModel(investigations=[Investigation(
            investigation_id="B-3", measurements=[
                PointMeasurement(1.5, "N_spt", 7, "field", "SPT")])])
        result = plot_parameter_vs_depth(one, "N_spt", show_trend=True)
        assert not result.trend_results
        assert "no trend fitted" in result.skipped[-1]

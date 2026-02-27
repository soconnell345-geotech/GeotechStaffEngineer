"""Tests for plots_xy.py — XY depth plots, Atterberg, multi-parameter."""

import pytest

from subsurface_characterization.plots_xy import (
    plot_parameter_vs_depth,
    plot_atterberg_limits,
    plot_multi_parameter,
)
from subsurface_characterization.results import PlotResult


class TestPlotParameterVsDepth:
    def test_basic_scatter(self, simple_site):
        result = plot_parameter_vs_depth(simple_site, "N_spt")
        assert isinstance(result, PlotResult)
        assert result.plot_type == "parameter_vs_depth"
        assert result.n_investigations == 2
        assert result.n_data_points == 14
        assert result.figure is not None

    def test_color_by_investigation(self, simple_site):
        result = plot_parameter_vs_depth(simple_site, "N_spt", color_by="investigation")
        fig = result.figure
        # Should have traces for each investigation
        assert len(fig.data) >= 2

    def test_color_by_uscs(self, simple_site):
        result = plot_parameter_vs_depth(simple_site, "N_spt", color_by="uscs")
        assert result.n_data_points == 14

    def test_color_by_none(self, simple_site):
        result = plot_parameter_vs_depth(simple_site, "N_spt", color_by="none")
        assert result.n_data_points == 14
        # Single trace
        assert len(result.figure.data) == 1

    def test_with_trend(self, simple_site):
        result = plot_parameter_vs_depth(simple_site, "N_spt", show_trend=True)
        assert len(result.trend_results) == 1
        # Figure should have trend line trace
        assert len(result.figure.data) > 2

    def test_with_bands(self, simple_site):
        result = plot_parameter_vs_depth(
            simple_site, "N_spt", show_trend=True, show_bands=True
        )
        assert len(result.trend_results) == 1
        # Extra traces for band fill
        assert len(result.figure.data) > 3

    def test_grouped_trends(self, simple_site):
        result = plot_parameter_vs_depth(
            simple_site, "N_spt",
            show_trend=True, group_trends_by="uscs",
        )
        assert len(result.trend_results) > 0

    def test_elevation_mode(self, simple_site):
        result = plot_parameter_vs_depth(
            simple_site, "N_spt", use_elevation=True
        )
        assert result.figure is not None
        # Y-axis should not be reversed in elevation mode
        y_autorange = result.figure.layout.yaxis.autorange
        assert y_autorange is True or y_autorange is None

    def test_custom_title(self, simple_site):
        result = plot_parameter_vs_depth(
            simple_site, "N_spt", title="Custom SPT Plot"
        )
        assert result.title == "Custom SPT Plot"

    def test_html_output(self, simple_site):
        result = plot_parameter_vs_depth(simple_site, "N_spt")
        html = result.to_html()
        assert "<html>" in html.lower() or "plotly" in html.lower()
        assert len(html) > 100

    def test_empty_parameter(self, simple_site):
        """Parameter with no data should still return a PlotResult."""
        result = plot_parameter_vs_depth(simple_site, "qc_kPa")
        assert result.n_data_points == 0
        assert result.figure is not None

    def test_to_dict(self, simple_site):
        result = plot_parameter_vs_depth(simple_site, "N_spt")
        d = result.to_dict()
        assert "html" in d
        assert d["plot_type"] == "parameter_vs_depth"
        assert d["n_data_points"] == 14


class TestPlotAtterbergLimits:
    def test_basic_atterberg(self, rich_site):
        result = plot_atterberg_limits(rich_site)
        assert isinstance(result, PlotResult)
        assert result.plot_type == "atterberg_limits"
        assert result.n_investigations >= 1
        assert result.n_data_points >= 1

    def test_atterberg_with_wn(self, rich_site):
        result = plot_atterberg_limits(rich_site)
        # Should have bracket traces + wn overlay
        assert len(result.figure.data) > 2


class TestPlotMultiParameter:
    def test_basic_multi(self, rich_site):
        result = plot_multi_parameter(rich_site, ["N_spt", "cu_kPa", "wn_pct"])
        assert isinstance(result, PlotResult)
        assert result.plot_type == "multi_parameter"
        assert result.n_data_points > 0
        assert result.parameters == ["N_spt", "cu_kPa", "wn_pct"]

    def test_empty_parameters(self, simple_site):
        result = plot_multi_parameter(simple_site, [])
        assert result.figure is not None

    def test_single_panel(self, simple_site):
        result = plot_multi_parameter(simple_site, ["N_spt"])
        assert result.n_data_points == 14

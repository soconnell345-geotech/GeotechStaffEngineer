"""Tests for plots_profile.py — cross-section profile views."""

import pytest

from subsurface_characterization.plots_profile import plot_cross_section
from subsurface_characterization.results import PlotResult
from subsurface_characterization.site_model import SiteModel


class TestPlotCrossSection:
    def test_two_boring_section(self, simple_site):
        result = plot_cross_section(simple_site, ["B-1", "B-2"])
        assert isinstance(result, PlotResult)
        assert result.plot_type == "cross_section"
        assert result.n_investigations == 2
        assert result.n_data_points > 0  # lithology intervals

    def test_four_investigation_section(self, rich_site):
        result = plot_cross_section(rich_site, ["B-1", "CPT-1", "B-2", "TP-1"])
        assert result.n_investigations == 4

    def test_lithology_columns(self, simple_site):
        result = plot_cross_section(simple_site, ["B-1", "B-2"])
        # Should have filled rectangle traces for lithology
        assert len(result.figure.data) > 2  # surface + lithology + legend

    def test_gwl_line(self, simple_site):
        result = plot_cross_section(simple_site, ["B-1", "B-2"], show_gwl=True)
        # Should have GWL trace
        trace_names = [t.name for t in result.figure.data if t.name]
        assert "GWL" in trace_names

    def test_no_gwl(self, simple_site):
        result = plot_cross_section(simple_site, ["B-1", "B-2"], show_gwl=False)
        trace_names = [t.name for t in result.figure.data if t.name]
        assert "GWL" not in trace_names

    def test_parameter_annotation(self, simple_site):
        result = plot_cross_section(
            simple_site, ["B-1", "B-2"], annotate_parameter="N_spt"
        )
        assert "N_spt" in result.parameters
        # Should have annotations for SPT values
        assert len(result.figure.layout.annotations) > 2

    def test_depth_mode(self, simple_site):
        result = plot_cross_section(
            simple_site, ["B-1", "B-2"], use_elevation=False
        )
        # Y-axis should be reversed for depth mode
        assert result.figure.layout.yaxis.autorange == "reversed"

    def test_single_investigation(self, simple_site):
        """Less than 2 investigations should return empty plot."""
        result = plot_cross_section(simple_site, ["B-1"])
        assert result.n_investigations == 1
        assert result.figure is not None

    def test_distance_calculation(self, simple_site):
        """Verify cumulative distance is computed correctly."""
        result = plot_cross_section(simple_site, ["B-1", "B-2"])
        # B-1 at (100,200), B-2 at (150,200) → distance = 50m
        surface_trace = result.figure.data[0]  # Ground Surface
        assert surface_trace.x[0] == 0.0
        assert surface_trace.x[1] == pytest.approx(50.0, abs=0.1)

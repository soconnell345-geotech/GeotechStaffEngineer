"""Tests for plots_plan.py — plan view map."""

import pytest

from subsurface_characterization.plots_plan import plot_plan_view
from subsurface_characterization.results import PlotResult
from subsurface_characterization.site_model import SiteModel


class TestPlotPlanView:
    def test_basic_plan(self, rich_site):
        result = plot_plan_view(rich_site)
        assert isinstance(result, PlotResult)
        assert result.plot_type == "plan_view"
        assert result.n_investigations == 4

    def test_color_by_type(self, rich_site):
        result = plot_plan_view(rich_site, color_by="type")
        # Should have traces grouped by type
        assert len(result.figure.data) >= 2  # boring + cpt + test_pit

    def test_color_by_parameter(self, rich_site):
        result = plot_plan_view(
            rich_site, color_by="parameter", parameter_for_color="N_spt"
        )
        assert result.n_investigations == 4
        # Should use colorscale
        assert len(result.figure.data) >= 1

    def test_label_id(self, rich_site):
        result = plot_plan_view(rich_site, label_field="id")
        # Annotations for labels
        assert len(result.figure.layout.annotations) >= 4

    def test_label_gwl(self, rich_site):
        result = plot_plan_view(rich_site, label_field="gwl")
        # Only borings with GWL get labels
        assert result.figure is not None

    def test_empty_site(self):
        site = SiteModel()
        result = plot_plan_view(site)
        assert result.n_investigations == 0
        assert result.figure is not None

    def test_hover_data(self, rich_site):
        result = plot_plan_view(rich_site)
        # All traces should have hover text
        for trace in result.figure.data:
            assert trace.hovertext is not None or trace.hoverinfo == "text"

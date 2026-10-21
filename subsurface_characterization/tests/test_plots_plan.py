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


# GM-76 (field session 2026-10-14): a local grid read off a plan sheet was
# drawn as "X (Easting)" / "Y (Northing)". The axes now follow the
# coordinate system the DIGGS file states.

from subsurface_characterization.plots_plan import axis_titles  # noqa: E402


@pytest.mark.parametrize("cs, want", [
    ("", ("x", "y")),
    ("EPSG:4326", ("Longitude (deg)", "Latitude (deg)")),
    ("urn:ogc:def:crs:EPSG::4326", ("Longitude (deg)", "Latitude (deg)")),
    ("urn:diggs:def:crs:DIGGS:0.1:4326_5702",
     ("Longitude (deg)", "Latitude (deg)")),
    ("EPSG:2263", ("Easting (EPSG:2263)", "Northing (EPSG:2263)")),
    ("WGS 84 / UTM zone 33N", ("Easting (WGS 84 / UTM zone 33N)",
                               "Northing (WGS 84 / UTM zone 33N)")),
    ("Figure 2 local grid, approximate", ("x (local grid)", "y (local grid)")),
])
def test_axis_titles_follow_the_coordinate_system(cs, want):
    assert axis_titles(cs)[:2] == want


def _diggs_with_srs(srs):
    pos = (f'<gml:pos srsDimension="3" srsName="{srs}">10 20 NaN</gml:pos>'
           if srs else '<gml:pos srsDimension="3">10 20 NaN</gml:pos>')
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Diggs xmlns="http://diggsml.org/schemas/2.6"
       xmlns:gml="http://www.opengis.net/gml/3.2">
  <Borehole gml:id="h1">
    <gml:name>H-1</gml:name>
    <referencePoint><PointLocation gml:id="p1">{pos}</PointLocation>
    </referencePoint>
  </Borehole>
</Diggs>"""


def test_a_local_grid_in_a_diggs_file_is_not_called_easting():
    from subsurface_characterization.diggs_parser import parse_diggs
    site = parse_diggs(
        content=_diggs_with_srs("Figure 2 local grid, approximate")).site
    assert site.coordinate_system == "Figure 2 local grid, approximate"
    fig = plot_plan_view(site).figure
    assert fig.layout.xaxis.title.text == "x (local grid)"
    assert "Easting" not in str(fig.layout.yaxis.title.text)
    assert "Figure 2 local grid" in fig.layout.title.text


def test_no_stated_system_is_plain_x_and_y():
    from subsurface_characterization.diggs_parser import parse_diggs
    site = parse_diggs(content=_diggs_with_srs("")).site
    assert site.coordinate_system == ""
    fig = plot_plan_view(site).figure
    assert fig.layout.xaxis.title.text == "x"


class TestNorthArrow:
    """GM-76: one north arrow, from the grid's own words, else up and said."""

    def test_stated_rotations(self):
        from subsurface_characterization.plots_plan import (
            north_from_coordinate_system as f)
        assert f("local grid, +y is N 116 E")[0] == pytest.approx(244.0)
        assert f("+y is N 30 W")[0] == pytest.approx(30.0)
        assert f("grid north 116 deg from true north")[0] == pytest.approx(244.0)
        assert f("grid north 20 deg west of true north")[0] == pytest.approx(20.0)
        assert f("rotation 116 deg")[0] == pytest.approx(244.0)
        assert f("north is 116 deg clockwise from +y")[0] == pytest.approx(116.0)
        assert f("north is 40 deg counterclockwise from +y")[0] == pytest.approx(320.0)
        assert "not stated" in f("grid north 116 deg from true north")[1]
        assert f("local grid") is None and f("") is None
        assert f("State Plane zone 6") is None

    def _site(self, cs):
        from subsurface_characterization import (
            SiteModel, Investigation)
        return SiteModel(project_name="t", coordinate_system=cs, investigations=[
            Investigation(investigation_id="B-1", x=0.0, y=0.0),
            Investigation(investigation_id="B-2", x=50.0, y=30.0)])

    def test_the_arrow_is_drawn_once_with_its_source(self):
        from subsurface_characterization.plots_plan import plot_plan_view
        res = plot_plan_view(self._site("local grid, +y is N 116 E"))
        assert res.north_deg == pytest.approx(244.0)
        assert "stated" in res.north_note and "N arrow" in res.title
        notes = [a.text for a in res.figure.layout.annotations]
        assert notes.count("N") == 1

    def test_nothing_stated_is_up_and_says_assumed(self):
        from subsurface_characterization.plots_plan import plot_plan_view
        res = plot_plan_view(self._site("local grid"))
        assert res.north_deg == 0.0 and "assumed up" in res.north_note

    def test_geographic_is_plainly_up(self):
        from subsurface_characterization.plots_plan import plot_plan_view
        res = plot_plan_view(self._site("EPSG:4326"))
        assert res.north_deg == 0.0 and "geographic" in res.north_note

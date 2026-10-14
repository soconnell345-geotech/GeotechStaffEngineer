"""An unknown position stays unknown; "not encountered" is not a level.

RI-27 (field session 2026-10-14): a DIGGS file whose writer filled 0 for
every position nobody printed read back with 33 of 40 explorations at
elevation 0 and one at the grid's origin -- the reader defaulted to 0 as
well, and the site model's x, y and elevation were plain floats. Now each
is None when not given, every plot leaves such an investigation out and
says so, and a borehole or pit that found no groundwater carries
``gwl_not_encountered`` instead of a level.
"""

import math

import pytest

from subsurface_characterization.csv_loader import load_site_from_csv
from subsurface_characterization.diggs_parser import parse_diggs
from subsurface_characterization.plots_plan import plot_plan_view
from subsurface_characterization.plots_profile import plot_cross_section
from subsurface_characterization.plots_xy import (
    plot_atterberg_limits, plot_multi_parameter, plot_parameter_vs_depth,
)
from subsurface_characterization.site_model import (
    Investigation, LithologyInterval, PointMeasurement, SiteModel,
    optional_float,
)


def _inv(inv_id, x=None, y=None, elevation_m=None, **over):
    data = dict(
        investigation_id=inv_id, x=x, y=y, elevation_m=elevation_m,
        total_depth_m=10.0,
        lithology=[LithologyInterval(0, 10, "Sand", uscs="SP")],
        measurements=[PointMeasurement(1.5, "N_spt", 10, "field", "SPT"),
                      PointMeasurement(3.0, "N_spt", 14, "field", "SPT"),
                      PointMeasurement(1.5, "LL_pct", 40, "lab"),
                      PointMeasurement(1.5, "PL_pct", 20, "lab")])
    data.update(over)
    return Investigation(**data)


@pytest.fixture
def site():
    """Two placed borings and one nobody gave a position for."""
    return SiteModel(project_name="S", investigations=[
        _inv("B-1", 0.0, 0.0, 100.0),
        _inv("B-2", 30.0, 40.0, 101.0),
        _inv("B-9"),
    ])


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class TestTheModel:
    def test_a_position_not_given_is_none(self):
        inv = Investigation(investigation_id="B-1")
        assert (inv.x, inv.y, inv.elevation_m) == (None, None, None)
        assert not inv.has_xy and not inv.has_elevation
        assert inv.gwl_not_encountered is False

    def test_a_zero_given_is_a_zero(self):
        inv = Investigation(investigation_id="B-1", x=0.0, y=0.0,
                            elevation_m=0.0)
        assert inv.has_xy and inv.has_elevation

    @pytest.mark.parametrize("raw, want", [
        (None, None), ("", None), ("  ", None), (float("nan"), None),
        ("NaN", None), ("12.5", 12.5), (0, 0.0), ("-3", -3.0)])
    def test_optional_float(self, raw, want):
        assert optional_float(raw) == want

    def test_to_dict_and_back_keep_unknown_unknown(self):
        inv = Investigation(investigation_id="B-1", gwl_not_encountered=True)
        d = inv.to_dict()
        assert (d["x"], d["y"], d["elevation_m"]) == (None, None, None)
        assert d["gwl_not_encountered"] is True
        back = Investigation.from_dict(d)
        assert (back.x, back.y, back.elevation_m) == (None, None, None)
        assert back.gwl_not_encountered is True
        # a dict with no position keys at all is unknown too, not 0
        bare = Investigation.from_dict({"investigation_id": "B-2"})
        assert (bare.x, bare.y, bare.elevation_m) == (None, None, None)

    def test_the_bounding_box_leaves_out_what_has_no_place(self, site):
        assert site.bounding_box() == (0.0, 0.0, 30.0, 40.0)
        lone = SiteModel(investigations=[_inv("B-9")])
        assert lone.bounding_box() == (0.0, 0.0, 0.0, 0.0)

    def test_summary_says_not_encountered(self):
        inv = Investigation(investigation_id="B-1", gwl_not_encountered=True)
        assert "GWL not encountered" in inv.summary()


class TestTheCsvLoader:
    def test_a_blank_or_missing_cell_is_unknown(self):
        borings = ("investigation_id,investigation_type,x,y,elevation_m,"
                   "total_depth_m\n"
                   "B-1,boring,100.5,200.0,,10\n"
                   "B-2,boring,,,12.0,8\n")
        meas = "investigation_id,depth_m,parameter,value\nB-1,1.5,N_spt,10\n"
        site = load_site_from_csv(borings, meas)
        b1, b2 = site.investigations
        assert (b1.x, b1.y, b1.elevation_m) == (100.5, 200.0, None)
        assert (b2.x, b2.y, b2.elevation_m) == (None, None, 12.0)

    def test_a_hole_only_the_measurements_name_has_no_position(self):
        borings = ("investigation_id,investigation_type,x,y,elevation_m,"
                   "total_depth_m\nB-1,boring,1,2,3,10\n")
        meas = "investigation_id,depth_m,parameter,value\nB-7,1.5,N_spt,10\n"
        site = load_site_from_csv(borings, meas)
        b7 = site.get_investigation("B-7")
        assert (b7.x, b7.y, b7.elevation_m) == (None, None, None)


# ---------------------------------------------------------------------------
# the reader
# ---------------------------------------------------------------------------

def _diggs(borehole_body: str, feature: str = "Borehole") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Diggs xmlns="http://diggsml.org/schemas/2.6"
       xmlns:gml="http://www.opengis.net/gml/3.2"
       xmlns:xlink="http://www.w3.org/1999/xlink">
  <{feature} gml:id="h1">
    <gml:name>H-1</gml:name>
    {borehole_body}
  </{feature}>
</Diggs>"""


class TestTheReader:
    def test_nan_is_unknown(self):
        xml = _diggs("<referencePoint><PointLocation gml:id='p'>"
                     "<gml:pos srsDimension='3'>12.5 NaN NaN</gml:pos>"
                     "</PointLocation></referencePoint>")
        (inv,) = parse_diggs(content=xml).site.investigations
        assert (inv.x, inv.y, inv.elevation_m) == (12.5, None, None)

    def test_no_reference_point_is_unknown_not_zero(self):
        (inv,) = parse_diggs(content=_diggs("")).site.investigations
        assert (inv.x, inv.y, inv.elevation_m) == (None, None, None)

    def test_a_two_number_position_has_no_elevation(self):
        xml = _diggs("<referencePoint><gml:Point>"
                     "<gml:pos>50.0 60.0</gml:pos>"
                     "</gml:Point></referencePoint>")
        (inv,) = parse_diggs(content=xml).site.investigations
        assert (inv.x, inv.y, inv.elevation_m) == (50.0, 60.0, None)

    def test_a_borehole_that_found_no_water_says_so(self):
        xml = _diggs("<waterStrike><WaterStrike gml:id='w'>"
                     "<notEncountered>true</notEncountered>"
                     "</WaterStrike></waterStrike>")
        (inv,) = parse_diggs(content=xml).site.investigations
        assert inv.gwl_depth_m is None and inv.gwl_not_encountered is True

    def test_a_pit_remark_says_so_and_only_that_remark(self):
        said = _diggs("<remark><Remark><content>Groundwater not encountered."
                      "</content></Remark></remark>", feature="TrialPit")
        (inv,) = parse_diggs(content=said).site.investigations
        assert inv.gwl_not_encountered is True
        other = _diggs("<remark><Remark><content>Seepage at 2 m."
                       "</content></Remark></remark>", feature="TrialPit")
        (inv,) = parse_diggs(content=other).site.investigations
        assert inv.gwl_not_encountered is False

    def test_a_reading_is_a_level_not_a_flag(self):
        xml = _diggs(
            "<waterStrike><WaterStrike gml:id='w'><initialWaterStrikeReading>"
            "<WaterStrikeReading gml:id='r'><waterLocation>"
            "<PointLocation gml:id='l' srsDimension='1'><gml:pos>3.5</gml:pos>"
            "</PointLocation></waterLocation></WaterStrikeReading>"
            "</initialWaterStrikeReading></WaterStrike></waterStrike>")
        (inv,) = parse_diggs(content=xml).site.investigations
        assert inv.gwl_depth_m == pytest.approx(3.5)
        assert inv.gwl_not_encountered is False


# ---------------------------------------------------------------------------
# the plots: what cannot be placed is left out, and said
# ---------------------------------------------------------------------------

class TestThePlots:
    def test_plan_view_leaves_out_an_unplaced_hole(self, site):
        result = plot_plan_view(site)
        assert result.n_investigations == 2
        assert result.skipped == ["B-9: no x, y given; not placed on the plan"]
        xs = [x for trace in result.figure.data for x in (trace.x or ())]
        assert len(xs) == 2 and not any(
            x is None or (isinstance(x, float) and math.isnan(x)) for x in xs)
        assert result.to_dict(output_format="metadata")["skipped"]
        by_parameter = plot_plan_view(site, color_by="parameter",
                                      parameter_for_color="N_spt")
        assert by_parameter.n_investigations == 2

    def test_plan_view_hover_says_unknown_elevation(self):
        site = SiteModel(investigations=[_inv("B-1", 1.0, 2.0)])
        hover = plot_plan_view(site).figure.data[0].hovertext[0]
        assert "Elev: unknown" in hover

    def test_cross_section_leaves_out_an_unplaced_hole(self, site):
        result = plot_cross_section(site, ["B-1", "B-9", "B-2"])
        assert result.n_investigations == 2
        assert result.skipped[0].startswith("B-9: no x, y given")
        surface = result.figure.data[0]
        assert list(surface.x) == [0.0, pytest.approx(50.0)]

    def test_cross_section_by_elevation_needs_an_elevation(self):
        site = SiteModel(investigations=[
            _inv("B-1", 0.0, 0.0, 100.0), _inv("B-2", 30.0, 40.0),
            _inv("B-3", 60.0, 80.0, 99.0)])
        by_elevation = plot_cross_section(site, ["B-1", "B-2", "B-3"])
        assert by_elevation.n_investigations == 2
        assert "no ground elevation" in by_elevation.skipped[0]
        by_depth = plot_cross_section(site, ["B-1", "B-2", "B-3"],
                                      use_elevation=False)
        assert by_depth.n_investigations == 3 and not by_depth.skipped

    @pytest.mark.parametrize("color_by", ["investigation", "uscs", "none"])
    def test_values_by_elevation_need_an_elevation(self, site, color_by):
        result = plot_parameter_vs_depth(site, "N_spt", color_by=color_by,
                                         use_elevation=True)
        assert result.n_investigations == 2
        assert result.n_data_points == 4
        assert result.skipped[0].startswith("B-9: no ground elevation")
        by_depth = plot_parameter_vs_depth(site, "N_spt", color_by=color_by)
        assert by_depth.n_data_points == 6 and not by_depth.skipped

    def test_a_trend_by_elevation_uses_known_elevations_only(self, site):
        result = plot_parameter_vs_depth(site, "N_spt", use_elevation=True,
                                         show_trend=True)
        line = [t for t in result.figure.data if t.mode == "lines"][0]
        # the mean of the KNOWN elevations (100.5) minus the shallowest depth
        assert line.y[0] == pytest.approx(100.5 - 1.5)
        # with no elevation known anywhere, nothing is drawn -- no points
        # and no line at an invented elevation -- and each hole is named
        nowhere = SiteModel(investigations=[_inv("B-8"), _inv("B-9")])
        alone = plot_parameter_vs_depth(nowhere, "N_spt",
                                        use_elevation=True, show_trend=True)
        assert alone.n_data_points == 0 and len(alone.skipped) == 2
        assert not [t for t in alone.figure.data if t.mode == "lines"]

    def test_atterberg_and_multi_parameter_by_elevation(self, site):
        atterberg = plot_atterberg_limits(site, use_elevation=True)
        assert atterberg.n_investigations == 2 and atterberg.skipped
        multi = plot_multi_parameter(site, ["N_spt", "LL_pct"],
                                     use_elevation=True)
        assert multi.n_investigations == 2
        assert len(multi.skipped) == 1          # said once, not per panel

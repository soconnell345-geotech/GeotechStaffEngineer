"""Tests for site_model.py — data model classes."""

import pytest
import numpy as np

from subsurface_characterization.site_model import (
    SiteModel, Investigation, PointMeasurement, LithologyInterval,
    STANDARD_PARAMETERS,
)


class TestPointMeasurement:
    def test_create_basic(self):
        m = PointMeasurement(depth_m=3.0, parameter="N_spt", value=15)
        assert m.depth_m == 3.0
        assert m.parameter == "N_spt"
        assert m.value == 15

    def test_to_dict(self):
        m = PointMeasurement(depth_m=3.0, parameter="N_spt", value=15, source="field")
        d = m.to_dict()
        assert d["depth_m"] == 3.0
        assert d["parameter"] == "N_spt"
        assert d["value"] == 15
        assert d["source"] == "field"

    def test_from_dict(self):
        d = {"depth_m": 5.5, "parameter": "cu_kPa", "value": 50, "source": "lab"}
        m = PointMeasurement.from_dict(d)
        assert m.depth_m == 5.5
        assert m.parameter == "cu_kPa"
        assert m.value == 50.0
        assert m.source == "lab"


class TestLithologyInterval:
    def test_create_basic(self):
        l = LithologyInterval(top_depth_m=0, bottom_depth_m=3, description="Fill", uscs="SM")
        assert l.top_depth_m == 0
        assert l.bottom_depth_m == 3
        assert l.uscs == "SM"

    def test_thickness_and_mid_depth(self):
        l = LithologyInterval(top_depth_m=2.0, bottom_depth_m=8.0)
        assert l.thickness_m == 6.0
        assert l.mid_depth_m == 5.0

    def test_to_dict_roundtrip(self):
        l = LithologyInterval(top_depth_m=0, bottom_depth_m=3, description="Clay", uscs="CL")
        d = l.to_dict()
        l2 = LithologyInterval.from_dict(d)
        assert l2.uscs == "CL"
        assert l2.thickness_m == pytest.approx(3.0)


class TestInvestigation:
    def test_get_measurements(self, simple_site):
        inv = simple_site.get_investigation("B-1")
        spt = inv.get_measurements("N_spt")
        assert len(spt) == 8
        # Sorted by depth
        assert spt[0].depth_m <= spt[-1].depth_m

    def test_get_parameter_arrays(self, simple_site):
        inv = simple_site.get_investigation("B-1")
        depths, values = inv.get_parameter_arrays("N_spt")
        assert len(depths) == 8
        assert isinstance(depths, np.ndarray)
        assert values[0] == 8  # N=8 at 1.5m

    def test_available_parameters(self, rich_site):
        inv = rich_site.get_investigation("B-1")
        params = inv.available_parameters()
        assert "N_spt" in params
        assert "LL_pct" in params
        assert "cu_kPa" in params

    def test_depth_to_rock(self, rich_site):
        inv = rich_site.get_investigation("B-1")
        dtr = inv.depth_to_rock_m()
        assert dtr == pytest.approx(18.0)

    def test_depth_to_rock_none(self, simple_site):
        inv = simple_site.get_investigation("B-1")
        assert inv.depth_to_rock_m() is None

    def test_fill_thickness(self, simple_site):
        inv = simple_site.get_investigation("B-1")
        ft = inv.fill_thickness_m()
        assert ft == pytest.approx(2.0)

    def test_uscs_at_depth(self, simple_site):
        inv = simple_site.get_investigation("B-1")
        assert inv.uscs_at_depth(1.0) == "SM"  # fill
        assert inv.uscs_at_depth(5.0) == "CL"  # clay
        assert inv.uscs_at_depth(10.0) == "SP"  # sand
        assert inv.uscs_at_depth(20.0) == ""    # below total depth

    def test_to_dict(self, simple_site):
        inv = simple_site.get_investigation("B-1")
        d = inv.to_dict()
        assert d["investigation_id"] == "B-1"
        assert d["n_measurements"] == 8
        assert d["n_lithology"] == 3
        assert "N_spt" in d["available_parameters"]

    def test_from_dict_roundtrip(self, simple_site):
        inv = simple_site.get_investigation("B-1")
        d = inv.to_dict()
        inv2 = Investigation.from_dict(d)
        assert inv2.investigation_id == "B-1"
        assert len(inv2.measurements) == 8
        assert len(inv2.lithology) == 3

    def test_to_soil_profile(self, simple_site):
        inv = simple_site.get_investigation("B-1")
        profile = inv.to_soil_profile()
        assert len(profile.layers) == 3
        assert profile.layers[0].uscs == "SM"
        assert profile.groundwater.depth == 3.0

    def test_summary(self, simple_site):
        inv = simple_site.get_investigation("B-1")
        s = inv.summary()
        assert "B-1" in s
        assert "boring" in s
        assert "8 measurements" in s


class TestSiteModel:
    def test_investigation_ids(self, simple_site):
        ids = simple_site.investigation_ids()
        assert ids == ["B-1", "B-2"]

    def test_get_investigation(self, simple_site):
        inv = simple_site.get_investigation("B-1")
        assert inv.investigation_id == "B-1"

    def test_get_investigation_not_found(self, simple_site):
        with pytest.raises(KeyError):
            simple_site.get_investigation("B-99")

    def test_borings_filter(self, rich_site):
        borings = rich_site.borings()
        assert len(borings) == 2
        assert all(b.investigation_type == "boring" for b in borings)

    def test_cpts_filter(self, rich_site):
        cpts = rich_site.cpts()
        assert len(cpts) == 1
        assert cpts[0].investigation_id == "CPT-1"

    def test_all_measurements(self, simple_site):
        all_spt = simple_site.all_measurements("N_spt")
        assert len(all_spt) == 14  # 8 + 6
        # Each is (inv_id, PointMeasurement) tuple
        assert all_spt[0][0] in ("B-1", "B-2")

    def test_available_parameters(self, rich_site):
        params = rich_site.available_parameters()
        assert "N_spt" in params
        assert "qc_kPa" in params
        assert "LL_pct" in params

    def test_bounding_box(self, simple_site):
        bb = simple_site.bounding_box()
        assert bb == (100.0, 200.0, 150.0, 200.0)

    def test_bounding_box_empty(self):
        site = SiteModel()
        assert site.bounding_box() == (0.0, 0.0, 0.0, 0.0)

    def test_to_dict_from_dict_roundtrip(self, simple_site):
        d = simple_site.to_dict()
        site2 = SiteModel.from_dict(d)
        assert site2.project_name == "Test Site"
        assert len(site2.investigations) == 2

    def test_summary(self, simple_site):
        s = simple_site.summary()
        assert "Test Site" in s
        assert "Investigations: 2" in s
        assert "Borings: 2" in s

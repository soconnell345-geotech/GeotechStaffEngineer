"""Tests for csv_loader.py — CSV/dict loading and CPT bridge."""

import pytest
import numpy as np

from subsurface_characterization.csv_loader import (
    load_site_from_dict,
    load_site_from_csv,
    load_cpt_to_investigation,
)
from subsurface_characterization.site_model import SiteModel, Investigation


class TestLoadSiteFromDict:
    def test_basic_dict(self):
        data = {
            "project_name": "Dict Project",
            "investigations": [
                {
                    "investigation_id": "B-1",
                    "investigation_type": "boring",
                    "x": 100, "y": 200,
                    "elevation_m": 10,
                    "total_depth_m": 15,
                    "gwl_depth_m": 3,
                    "measurements": [
                        {"depth_m": 1.5, "parameter": "N_spt", "value": 10},
                        {"depth_m": 3.0, "parameter": "N_spt", "value": 15},
                    ],
                    "lithology": [
                        {"top_depth_m": 0, "bottom_depth_m": 5, "description": "Fill", "uscs": "SM"},
                    ],
                }
            ],
        }
        site = load_site_from_dict(data)
        assert isinstance(site, SiteModel)
        assert site.project_name == "Dict Project"
        assert len(site.investigations) == 1
        assert len(site.investigations[0].measurements) == 2
        assert len(site.investigations[0].lithology) == 1

    def test_empty_dict(self):
        site = load_site_from_dict({})
        assert len(site.investigations) == 0

    def test_missing_optional_fields(self):
        data = {
            "investigations": [
                {
                    "investigation_id": "B-1",
                    "measurements": [
                        {"depth_m": 1.5, "parameter": "N_spt", "value": 8},
                    ],
                }
            ],
        }
        site = load_site_from_dict(data)
        assert site.investigations[0].investigation_id == "B-1"
        assert site.investigations[0].gwl_depth_m is None


class TestLoadSiteFromCsv:
    def test_csv_strings(self):
        borings_csv = (
            "investigation_id,investigation_type,x,y,elevation_m,total_depth_m,gwl_depth_m\n"
            "B-1,boring,100,200,10,15,3\n"
            "B-2,boring,150,200,11,12,4\n"
        )
        measurements_csv = (
            "investigation_id,depth_m,parameter,value\n"
            "B-1,1.5,N_spt,8\n"
            "B-1,3.0,N_spt,12\n"
            "B-2,1.5,N_spt,6\n"
        )
        site = load_site_from_csv(borings_csv, measurements_csv)
        assert len(site.investigations) == 2
        inv1 = [i for i in site.investigations if i.investigation_id == "B-1"][0]
        assert len(inv1.measurements) == 2
        assert inv1.x == 100.0

    def test_csv_with_lithology(self):
        borings_csv = (
            "investigation_id,investigation_type,x,y,elevation_m,total_depth_m\n"
            "B-1,boring,100,200,10,15\n"
        )
        measurements_csv = (
            "investigation_id,depth_m,parameter,value\n"
            "B-1,1.5,N_spt,8\n"
        )
        lithology_csv = (
            "investigation_id,top_depth_m,bottom_depth_m,description,uscs\n"
            "B-1,0,3,Fill,SM\n"
            "B-1,3,10,Clay,CL\n"
        )
        site = load_site_from_csv(borings_csv, measurements_csv, lithology_csv)
        inv = site.investigations[0]
        assert len(inv.lithology) == 2
        assert inv.lithology[0].uscs == "SM"

    def test_type_coercion(self):
        """Numeric strings are properly converted to floats."""
        borings_csv = (
            "investigation_id,investigation_type,x,y,elevation_m,total_depth_m\n"
            "B-1,boring,100.5,200.3,10.2,15.0\n"
        )
        measurements_csv = (
            "investigation_id,depth_m,parameter,value\n"
            "B-1,1.5,N_spt,8\n"
        )
        site = load_site_from_csv(borings_csv, measurements_csv)
        inv = site.investigations[0]
        assert isinstance(inv.x, float)
        assert inv.x == pytest.approx(100.5)

    def test_measurement_creates_investigation(self):
        """If a measurement references unknown ID, it auto-creates the investigation."""
        borings_csv = (
            "investigation_id,investigation_type,x,y,elevation_m,total_depth_m\n"
            "B-1,boring,100,200,10,15\n"
        )
        measurements_csv = (
            "investigation_id,depth_m,parameter,value\n"
            "B-1,1.5,N_spt,8\n"
            "B-99,3.0,N_spt,12\n"
        )
        site = load_site_from_csv(borings_csv, measurements_csv)
        ids = [i.investigation_id for i in site.investigations]
        assert "B-99" in ids


class TestLoadCptToInvestigation:
    def test_basic_cpt_bridge(self):
        """Mock CPTParseResult and convert to Investigation."""
        class MockCPT:
            alias = "CPT-1"
            x = 100.0
            y = 200.0
            final_depth_m = 15.0
            gwl_m = 3.0
            srs_name = "EPSG:2263"
            depth_m = np.array([1.0, 2.0, 3.0])
            q_c_kPa = np.array([500, 1000, 1500])
            f_s_kPa = np.array([5, 10, 15])
            u_2_kPa = np.array([0, 50, 100])
            Rf_pct = np.array([1.0, 1.0, 1.0])

        cpt = MockCPT()
        inv = load_cpt_to_investigation(cpt)
        assert inv.investigation_id == "CPT-1"
        assert inv.investigation_type == "cpt"
        assert inv.x == 100.0
        assert inv.gwl_depth_m == 3.0
        # 3 depths * 4 parameters = 12 measurements
        assert len(inv.measurements) == 12
        qc_meas = inv.get_measurements("qc_kPa")
        assert len(qc_meas) == 3
        assert qc_meas[0].value == 500

    def test_empty_cpt(self):
        class MockCPT:
            alias = "CPT-empty"
            x = None
            y = None
            final_depth_m = 0.0
            gwl_m = None
            srs_name = ""
            depth_m = np.array([])
            q_c_kPa = np.array([])
            f_s_kPa = np.array([])
            u_2_kPa = np.array([])
            Rf_pct = np.array([])

        inv = load_cpt_to_investigation(MockCPT())
        assert inv.investigation_id == "CPT-empty"
        assert len(inv.measurements) == 0

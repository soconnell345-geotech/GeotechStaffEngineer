"""Tests for diggs_parser.py — DIGGS 2.6 XML parser."""

import pytest

from subsurface_characterization.diggs_parser import parse_diggs
from subsurface_characterization.results import DiggsParseResult


class TestParseSingleBoring:
    def test_parse_basic(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        assert isinstance(result, DiggsParseResult)
        assert result.n_investigations == 1
        assert result.site.project_name == "Test Project"

    def test_coordinates(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        inv = result.site.investigations[0]
        assert inv.x == pytest.approx(100.0)
        assert inv.y == pytest.approx(200.0)
        assert inv.elevation_m == pytest.approx(10.0)

    def test_total_depth(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        inv = result.site.investigations[0]
        assert inv.total_depth_m == pytest.approx(15.0)

    def test_lithology(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        inv = result.site.investigations[0]
        assert len(inv.lithology) == 3
        assert inv.lithology[0].uscs == "SM"
        assert inv.lithology[1].description == "Soft gray clay"
        assert inv.lithology[2].top_depth_m == pytest.approx(10.0)

    def test_spt_data(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        inv = result.site.investigations[0]
        spt = inv.get_measurements("N_spt")
        assert len(spt) == 4
        assert spt[0].depth_m == pytest.approx(1.5)
        assert spt[0].value == pytest.approx(8)
        assert spt[-1].value == pytest.approx(30)

    def test_gwl(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        inv = result.site.investigations[0]
        assert inv.gwl_depth_m == pytest.approx(3.5)


class TestParseMultiBoring:
    def test_two_borings(self, multi_boring_diggs_xml):
        result = parse_diggs(content=multi_boring_diggs_xml)
        assert result.n_investigations == 2
        ids = result.site.investigation_ids()
        assert "B-1" in ids
        assert "B-2" in ids

    def test_spt_association(self, multi_boring_diggs_xml):
        """SPT tests are associated with correct borings via xlink:href."""
        result = parse_diggs(content=multi_boring_diggs_xml)
        b1 = result.site.get_investigation("B-1")
        b2 = result.site.get_investigation("B-2")
        assert len(b1.get_measurements("N_spt")) == 2
        assert len(b2.get_measurements("N_spt")) == 2

    def test_atterberg(self, multi_boring_diggs_xml):
        result = parse_diggs(content=multi_boring_diggs_xml)
        b1 = result.site.get_investigation("B-1")
        ll = b1.get_measurements("LL_pct")
        pl = b1.get_measurements("PL_pct")
        assert len(ll) == 1
        assert ll[0].value == pytest.approx(55)
        assert len(pl) == 1
        assert pl[0].value == pytest.approx(22)

    def test_moisture(self, multi_boring_diggs_xml):
        result = parse_diggs(content=multi_boring_diggs_xml)
        b1 = result.site.get_investigation("B-1")
        wn = b1.get_measurements("wn_pct")
        assert len(wn) == 1
        assert wn[0].value == pytest.approx(42)

    def test_gwl_per_boring(self, multi_boring_diggs_xml):
        result = parse_diggs(content=multi_boring_diggs_xml)
        b1 = result.site.get_investigation("B-1")
        b2 = result.site.get_investigation("B-2")
        assert b1.gwl_depth_m == pytest.approx(3.5)
        assert b2.gwl_depth_m == pytest.approx(4.0)

    def test_lithology_counts(self, multi_boring_diggs_xml):
        result = parse_diggs(content=multi_boring_diggs_xml)
        assert result.n_lithology_intervals == 6  # 3 + 3


class TestNamespaceDetection:
    def test_25a_namespace(self, diggs_25a_xml):
        result = parse_diggs(content=diggs_25a_xml)
        assert result.n_investigations == 1
        inv = result.site.investigations[0]
        assert inv.investigation_id == "B-1"
        assert inv.x == pytest.approx(50.0)

    def test_25a_spt(self, diggs_25a_xml):
        result = parse_diggs(content=diggs_25a_xml)
        inv = result.site.investigations[0]
        spt = inv.get_measurements("N_spt")
        assert len(spt) == 1
        assert spt[0].value == pytest.approx(12)


class TestErrorHandling:
    def test_no_input(self):
        with pytest.raises(ValueError, match="Must provide"):
            parse_diggs()

    def test_invalid_xml(self):
        with pytest.raises(ValueError, match="Invalid XML"):
            parse_diggs(content="<not valid xml")

    def test_empty_xml(self):
        """XML with no borings should return empty SiteModel."""
        xml = """<?xml version="1.0"?>
        <Diggs xmlns="http://diggsml.org/schemas/2.6"
               xmlns:gml="http://www.opengis.net/gml/3.2">
          <Project><gml:name>Empty</gml:name></Project>
        </Diggs>"""
        result = parse_diggs(content=xml)
        assert result.n_investigations == 0

    def test_summary(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        s = result.summary()
        assert "Test Project" in s
        assert "Investigations: 1" in s

    def test_to_dict(self, single_boring_diggs_xml):
        result = parse_diggs(content=single_boring_diggs_xml)
        d = result.to_dict()
        assert d["n_investigations"] == 1
        assert d["project_name"] == "Test Project"
        assert d["site"] is not None

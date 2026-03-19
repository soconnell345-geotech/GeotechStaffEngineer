"""Tests for subsurface_adapter — DIGGS ingestion, site cache, and all plot methods."""

import pytest

from funhouse_agent.adapters import subsurface_adapter
from funhouse_agent.adapters.subsurface_adapter import (
    METHOD_INFO, METHOD_REGISTRY, _site_cache,
)
from funhouse_agent.dispatch import list_methods, describe_method, call_agent


REQUIRED_INFO_FIELDS = {"category", "brief", "parameters", "returns"}

# ---------------------------------------------------------------------------
# Test XML fixtures (inline — no conftest dependency)
# ---------------------------------------------------------------------------

SINGLE_BORING_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Diggs xmlns="http://diggsml.org/schemas/2.6"
       xmlns:gml="http://www.opengis.net/gml/3.2"
       xmlns:xlink="http://www.w3.org/1999/xlink">
  <Project>
    <gml:name>Test Project</gml:name>
  </Project>
  <Borehole gml:id="BH1">
    <gml:name>B-1</gml:name>
    <referencePoint>
      <gml:Point>
        <gml:pos>100.0 200.0 10.0</gml:pos>
      </gml:Point>
    </referencePoint>
    <totalMeasuredDepth>15.0</totalMeasuredDepth>
    <LithologyObservation>
      <depthTop>0.0</depthTop>
      <depthBase>3.0</depthBase>
      <description>Fill - sand and gravel</description>
      <uscs>SM</uscs>
    </LithologyObservation>
    <LithologyObservation>
      <depthTop>3.0</depthTop>
      <depthBase>10.0</depthBase>
      <description>Soft gray clay</description>
      <uscs>CL</uscs>
    </LithologyObservation>
    <LithologyObservation>
      <depthTop>10.0</depthTop>
      <depthBase>15.0</depthBase>
      <description>Dense sand</description>
      <uscs>SP</uscs>
    </LithologyObservation>
  </Borehole>
  <DrivenPenetrationTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>1.5</depth>
    <blowCount>8</blowCount>
  </DrivenPenetrationTest>
  <DrivenPenetrationTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>4.5</depth>
    <blowCount>5</blowCount>
  </DrivenPenetrationTest>
  <DrivenPenetrationTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>7.5</depth>
    <blowCount>7</blowCount>
  </DrivenPenetrationTest>
  <DrivenPenetrationTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>10.5</depth>
    <blowCount>30</blowCount>
  </DrivenPenetrationTest>
  <WaterLevelObservation>
    <investigationRef xlink:href="#BH1"/>
    <waterDepth>3.5</waterDepth>
  </WaterLevelObservation>
</Diggs>"""

MULTI_BORING_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Diggs xmlns="http://diggsml.org/schemas/2.6"
       xmlns:gml="http://www.opengis.net/gml/3.2"
       xmlns:xlink="http://www.w3.org/1999/xlink">
  <Project>
    <gml:name>Multi-Boring Project</gml:name>
  </Project>
  <Borehole gml:id="BH1">
    <gml:name>B-1</gml:name>
    <referencePoint>
      <gml:Point>
        <gml:pos>100.0 200.0 10.0</gml:pos>
      </gml:Point>
    </referencePoint>
    <totalMeasuredDepth>20.0</totalMeasuredDepth>
    <LithologyObservation>
      <depthTop>0.0</depthTop>
      <depthBase>5.0</depthBase>
      <description>Fill</description>
      <uscs>SM</uscs>
    </LithologyObservation>
    <LithologyObservation>
      <depthTop>5.0</depthTop>
      <depthBase>15.0</depthBase>
      <description>Clay</description>
      <uscs>CH</uscs>
    </LithologyObservation>
    <LithologyObservation>
      <depthTop>15.0</depthTop>
      <depthBase>20.0</depthBase>
      <description>Rock</description>
      <uscs>R</uscs>
    </LithologyObservation>
  </Borehole>
  <Borehole gml:id="BH2">
    <gml:name>B-2</gml:name>
    <referencePoint>
      <gml:Point>
        <gml:pos>150.0 200.0 11.0</gml:pos>
      </gml:Point>
    </referencePoint>
    <totalMeasuredDepth>15.0</totalMeasuredDepth>
    <LithologyObservation>
      <depthTop>0.0</depthTop>
      <depthBase>4.0</depthBase>
      <description>Fill</description>
      <uscs>SM</uscs>
    </LithologyObservation>
    <LithologyObservation>
      <depthTop>4.0</depthTop>
      <depthBase>12.0</depthBase>
      <description>Stiff clay</description>
      <uscs>CL</uscs>
    </LithologyObservation>
    <LithologyObservation>
      <depthTop>12.0</depthTop>
      <depthBase>15.0</depthBase>
      <description>Bedrock</description>
      <uscs>R</uscs>
    </LithologyObservation>
  </Borehole>
  <DrivenPenetrationTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>2.0</depth>
    <blowCount>8</blowCount>
  </DrivenPenetrationTest>
  <DrivenPenetrationTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>7.0</depth>
    <blowCount>5</blowCount>
  </DrivenPenetrationTest>
  <DrivenPenetrationTest>
    <investigationRef xlink:href="#BH2"/>
    <depth>2.0</depth>
    <blowCount>10</blowCount>
  </DrivenPenetrationTest>
  <DrivenPenetrationTest>
    <investigationRef xlink:href="#BH2"/>
    <depth>6.0</depth>
    <blowCount>12</blowCount>
  </DrivenPenetrationTest>
  <AtterbergLimitsTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>7.0</depth>
    <liquidLimit>55</liquidLimit>
    <plasticLimit>22</plasticLimit>
  </AtterbergLimitsTest>
  <AtterbergLimitsTest>
    <investigationRef xlink:href="#BH2"/>
    <depth>6.0</depth>
    <liquidLimit>38</liquidLimit>
    <plasticLimit>18</plasticLimit>
  </AtterbergLimitsTest>
  <MoistureContent>
    <investigationRef xlink:href="#BH1"/>
    <depth>7.0</depth>
    <moistureContent>42</moistureContent>
  </MoistureContent>
  <WaterLevelObservation>
    <investigationRef xlink:href="#BH1"/>
    <waterDepth>3.5</waterDepth>
  </WaterLevelObservation>
  <WaterLevelObservation>
    <investigationRef xlink:href="#BH2"/>
    <waterDepth>4.0</waterDepth>
  </WaterLevelObservation>
</Diggs>"""


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the site cache before each test."""
    _site_cache.clear()
    yield
    _site_cache.clear()


# ============================================================================
# METHOD_INFO / REGISTRY consistency
# ============================================================================

class TestMethodInfo:
    def test_keys_match(self):
        assert set(METHOD_INFO.keys()) == set(METHOD_REGISTRY.keys())

    def test_required_fields(self):
        for name, info in METHOD_INFO.items():
            for field in REQUIRED_INFO_FIELDS:
                assert field in info, f"{name} missing {field}"

    def test_expected_methods(self):
        expected = {
            "parse_diggs", "load_site",
            "plot_parameter_vs_depth", "plot_atterberg_limits",
            "plot_multi_parameter", "plot_plan_view", "plot_cross_section",
            "compute_trend",
        }
        assert set(METHOD_INFO.keys()) == expected


# ============================================================================
# Dispatch layer
# ============================================================================

class TestDispatch:
    def test_list_methods(self):
        result = list_methods("subsurface")
        total = sum(len(v) for v in result.values())
        assert total == 8

    def test_describe_parse_diggs(self):
        info = describe_method("subsurface", "parse_diggs")
        assert "parameters" in info
        assert "file_path" in info["parameters"]
        assert "content" in info["parameters"]

    def test_describe_plot_plan_view(self):
        info = describe_method("subsurface", "plot_plan_view")
        assert "parameters" in info
        assert "color_by" in info["parameters"]

    def test_describe_plot_cross_section(self):
        info = describe_method("subsurface", "plot_cross_section")
        assert "parameters" in info
        assert "investigation_ids" in info["parameters"]


# ============================================================================
# parse_diggs
# ============================================================================

class TestParseDiggs:
    def test_single_boring(self):
        result = call_agent("subsurface", "parse_diggs", {"content": SINGLE_BORING_XML})
        assert "error" not in result
        assert result["project_name"] == "Test Project"
        assert result["n_investigations"] == 1
        assert result["n_measurements"] == 4  # 4 SPT
        assert result["n_lithology_intervals"] == 3
        assert "site_key" in result

    def test_multi_boring(self):
        result = call_agent("subsurface", "parse_diggs", {"content": MULTI_BORING_XML})
        assert result["n_investigations"] == 2
        assert len(result["investigations"]) == 2
        # B-1 has SPT(2) + Atterberg(LL,PL) + moisture = 5
        # B-2 has SPT(2) + Atterberg(LL,PL) = 4
        assert result["n_measurements"] >= 4

    def test_site_key_cached(self):
        result = call_agent("subsurface", "parse_diggs", {"content": SINGLE_BORING_XML})
        key = result["site_key"]
        assert key in _site_cache
        assert _site_cache[key].project_name == "Test Project"

    def test_investigation_summary_includes_parameters(self):
        result = call_agent("subsurface", "parse_diggs", {"content": SINGLE_BORING_XML})
        inv = result["investigations"][0]
        assert inv["investigation_id"] == "B-1"
        assert "N_spt" in inv["parameters"]
        assert inv["total_depth_m"] == 15.0

    def test_duplicate_key_incremented(self):
        """Parsing the same project twice produces unique keys."""
        r1 = call_agent("subsurface", "parse_diggs", {"content": SINGLE_BORING_XML})
        r2 = call_agent("subsurface", "parse_diggs", {"content": SINGLE_BORING_XML})
        assert r1["site_key"] != r2["site_key"]
        assert len(_site_cache) == 2


# ============================================================================
# site_key resolution in plot methods
# ============================================================================

class TestSiteKeyResolution:
    def test_plot_with_site_key(self):
        """Parse DIGGS, then plot using the returned site_key."""
        parse_result = call_agent("subsurface", "parse_diggs", {"content": SINGLE_BORING_XML})
        key = parse_result["site_key"]
        plot_result = call_agent("subsurface", "plot_parameter_vs_depth", {
            "site_key": key,
            "parameter": "N_spt",
            "output_format": "metadata",
        })
        assert "error" not in plot_result
        assert plot_result["n_data_points"] == 4

    def test_invalid_site_key_error(self):
        result = call_agent("subsurface", "plot_parameter_vs_depth", {
            "site_key": "nonexistent",
            "parameter": "N_spt",
        })
        assert "error" in result

    def test_no_site_key_or_data_error(self):
        result = call_agent("subsurface", "plot_parameter_vs_depth", {
            "parameter": "N_spt",
        })
        assert "error" in result


# ============================================================================
# load_site (with caching)
# ============================================================================

class TestLoadSite:
    def test_load_site_returns_key(self):
        site_data = {
            "project_name": "Dict Project",
            "investigations": [{
                "investigation_id": "B-1",
                "investigation_type": "boring",
                "x": 0, "y": 0, "elevation_m": 10, "total_depth_m": 15,
                "measurements": [
                    {"depth_m": 1.5, "parameter": "N_spt", "value": 8},
                ],
                "lithology": [],
            }],
        }
        result = call_agent("subsurface", "load_site", {"site_data": site_data})
        assert "error" not in result
        assert "site_key" in result
        assert result["site_key"] in _site_cache

    def test_load_site_then_plot(self):
        site_data = {
            "project_name": "Quick Test",
            "investigations": [{
                "investigation_id": "B-1",
                "investigation_type": "boring",
                "x": 0, "y": 0, "elevation_m": 10, "total_depth_m": 15,
                "measurements": [
                    {"depth_m": 1.5, "parameter": "N_spt", "value": 8},
                    {"depth_m": 4.5, "parameter": "N_spt", "value": 12},
                ],
                "lithology": [],
            }],
        }
        load_result = call_agent("subsurface", "load_site", {"site_data": site_data})
        plot_result = call_agent("subsurface", "plot_parameter_vs_depth", {
            "site_key": load_result["site_key"],
            "parameter": "N_spt",
            "output_format": "metadata",
        })
        assert plot_result["n_data_points"] == 2


# ============================================================================
# plot_plan_view
# ============================================================================

class TestPlotPlanView:
    def test_plan_view_metadata(self):
        parse_result = call_agent("subsurface", "parse_diggs", {"content": MULTI_BORING_XML})
        result = call_agent("subsurface", "plot_plan_view", {
            "site_key": parse_result["site_key"],
            "output_format": "metadata",
        })
        assert "error" not in result
        assert result["plot_type"] == "plan_view"
        assert result["n_investigations"] == 2

    def test_plan_view_html(self):
        parse_result = call_agent("subsurface", "parse_diggs", {"content": MULTI_BORING_XML})
        result = call_agent("subsurface", "plot_plan_view", {
            "site_key": parse_result["site_key"],
            "output_format": "html",
        })
        assert "html" in result
        assert "<html>" in result["html"].lower() or "plotly" in result["html"].lower()


# ============================================================================
# plot_cross_section
# ============================================================================

class TestPlotCrossSection:
    def test_cross_section_metadata(self):
        parse_result = call_agent("subsurface", "parse_diggs", {"content": MULTI_BORING_XML})
        result = call_agent("subsurface", "plot_cross_section", {
            "site_key": parse_result["site_key"],
            "investigation_ids": ["B-1", "B-2"],
            "output_format": "metadata",
        })
        assert "error" not in result
        assert result["plot_type"] == "cross_section"
        assert result["n_investigations"] == 2

    def test_cross_section_html(self):
        parse_result = call_agent("subsurface", "parse_diggs", {"content": MULTI_BORING_XML})
        result = call_agent("subsurface", "plot_cross_section", {
            "site_key": parse_result["site_key"],
            "investigation_ids": ["B-1", "B-2"],
            "use_elevation": True,
            "show_gwl": True,
            "output_format": "html",
        })
        assert "html" in result

    def test_cross_section_with_annotation(self):
        parse_result = call_agent("subsurface", "parse_diggs", {"content": MULTI_BORING_XML})
        result = call_agent("subsurface", "plot_cross_section", {
            "site_key": parse_result["site_key"],
            "investigation_ids": ["B-1", "B-2"],
            "annotate_parameter": "N_spt",
            "output_format": "metadata",
        })
        assert "error" not in result


# ============================================================================
# plot_atterberg_limits (with site_key)
# ============================================================================

class TestPlotAtterbergWithKey:
    def test_atterberg_via_site_key(self):
        parse_result = call_agent("subsurface", "parse_diggs", {"content": MULTI_BORING_XML})
        result = call_agent("subsurface", "plot_atterberg_limits", {
            "site_key": parse_result["site_key"],
            "output_format": "metadata",
        })
        assert "error" not in result
        assert result["n_data_points"] >= 2


# ============================================================================
# plot_multi_parameter (with site_key)
# ============================================================================

class TestPlotMultiParameterWithKey:
    def test_multi_param_via_site_key(self):
        parse_result = call_agent("subsurface", "parse_diggs", {"content": MULTI_BORING_XML})
        result = call_agent("subsurface", "plot_multi_parameter", {
            "site_key": parse_result["site_key"],
            "parameters": ["N_spt"],
            "output_format": "metadata",
        })
        assert "error" not in result
        assert result["n_data_points"] >= 2


# ============================================================================
# End-to-end: DIGGS → parse → plot workflow
# ============================================================================

class TestEndToEnd:
    def test_diggs_to_multiple_plots(self):
        """Full workflow: parse DIGGS, then generate multiple plot types."""
        parse_result = call_agent("subsurface", "parse_diggs", {"content": MULTI_BORING_XML})
        key = parse_result["site_key"]

        # SPT vs depth
        r1 = call_agent("subsurface", "plot_parameter_vs_depth", {
            "site_key": key, "parameter": "N_spt", "output_format": "metadata",
        })
        assert "error" not in r1

        # Plan view
        r2 = call_agent("subsurface", "plot_plan_view", {
            "site_key": key, "output_format": "metadata",
        })
        assert "error" not in r2

        # Cross-section
        r3 = call_agent("subsurface", "plot_cross_section", {
            "site_key": key,
            "investigation_ids": ["B-1", "B-2"],
            "output_format": "metadata",
        })
        assert "error" not in r3

        # Atterberg
        r4 = call_agent("subsurface", "plot_atterberg_limits", {
            "site_key": key, "output_format": "metadata",
        })
        assert "error" not in r4

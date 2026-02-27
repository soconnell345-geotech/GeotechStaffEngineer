"""
Shared fixtures for subsurface_characterization tests.

Provides synthetic DIGGS XML strings and SiteModel builders.
"""

import pytest
import numpy as np

from subsurface_characterization.site_model import (
    SiteModel, Investigation, PointMeasurement, LithologyInterval,
)


# ---------------------------------------------------------------------------
# Site model builders
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_site():
    """Minimal site with 2 borings, SPT data, and lithology."""
    inv1 = Investigation(
        investigation_id="B-1",
        investigation_type="boring",
        x=100.0, y=200.0,
        elevation_m=10.0,
        total_depth_m=15.0,
        gwl_depth_m=3.0,
        lithology=[
            LithologyInterval(0, 2, "Fill - sand and gravel", uscs="SM"),
            LithologyInterval(2, 8, "Soft gray clay", uscs="CL"),
            LithologyInterval(8, 15, "Dense sand", uscs="SP"),
        ],
        measurements=[
            PointMeasurement(1.5, "N_spt", 8, "field", "SPT"),
            PointMeasurement(3.0, "N_spt", 4, "field", "SPT"),
            PointMeasurement(4.5, "N_spt", 5, "field", "SPT"),
            PointMeasurement(6.0, "N_spt", 6, "field", "SPT"),
            PointMeasurement(7.5, "N_spt", 7, "field", "SPT"),
            PointMeasurement(9.0, "N_spt", 25, "field", "SPT"),
            PointMeasurement(10.5, "N_spt", 30, "field", "SPT"),
            PointMeasurement(12.0, "N_spt", 35, "field", "SPT"),
        ],
    )
    inv2 = Investigation(
        investigation_id="B-2",
        investigation_type="boring",
        x=150.0, y=200.0,
        elevation_m=11.0,
        total_depth_m=12.0,
        gwl_depth_m=4.0,
        lithology=[
            LithologyInterval(0, 3, "Fill - misc debris", uscs="SM"),
            LithologyInterval(3, 7, "Stiff brown clay", uscs="CL"),
            LithologyInterval(7, 12, "Medium dense sand", uscs="SP"),
        ],
        measurements=[
            PointMeasurement(1.5, "N_spt", 6, "field", "SPT"),
            PointMeasurement(3.0, "N_spt", 10, "field", "SPT"),
            PointMeasurement(4.5, "N_spt", 12, "field", "SPT"),
            PointMeasurement(6.0, "N_spt", 14, "field", "SPT"),
            PointMeasurement(7.5, "N_spt", 20, "field", "SPT"),
            PointMeasurement(9.0, "N_spt", 22, "field", "SPT"),
        ],
    )
    return SiteModel(
        project_name="Test Site",
        investigations=[inv1, inv2],
    )


@pytest.fixture
def rich_site():
    """Site with 4 investigations, mixed types, Atterberg/moisture data."""
    inv1 = Investigation(
        investigation_id="B-1",
        investigation_type="boring",
        x=100.0, y=200.0,
        elevation_m=10.0,
        total_depth_m=20.0,
        gwl_depth_m=3.0,
        lithology=[
            LithologyInterval(0, 2, "Fill", uscs="SM"),
            LithologyInterval(2, 10, "Soft gray clay", uscs="CH"),
            LithologyInterval(10, 18, "Dense sand", uscs="SP"),
            LithologyInterval(18, 20, "Bedrock - sandstone", uscs="R"),
        ],
        measurements=[
            PointMeasurement(1.5, "N_spt", 8, "field", "SPT"),
            PointMeasurement(3.0, "N_spt", 3, "field", "SPT"),
            PointMeasurement(4.5, "N_spt", 4, "field", "SPT"),
            PointMeasurement(6.0, "N_spt", 5, "field", "SPT"),
            PointMeasurement(7.5, "N_spt", 6, "field", "SPT"),
            PointMeasurement(9.0, "N_spt", 7, "field", "SPT"),
            PointMeasurement(10.5, "N_spt", 28, "field", "SPT"),
            PointMeasurement(12.0, "N_spt", 35, "field", "SPT"),
            PointMeasurement(15.0, "N_spt", 45, "field", "SPT"),
            # Atterberg
            PointMeasurement(3.0, "LL_pct", 52, "lab", "Atterberg"),
            PointMeasurement(3.0, "PL_pct", 22, "lab", "Atterberg"),
            PointMeasurement(6.0, "LL_pct", 48, "lab", "Atterberg"),
            PointMeasurement(6.0, "PL_pct", 20, "lab", "Atterberg"),
            PointMeasurement(9.0, "LL_pct", 55, "lab", "Atterberg"),
            PointMeasurement(9.0, "PL_pct", 24, "lab", "Atterberg"),
            # Moisture
            PointMeasurement(3.0, "wn_pct", 45, "lab", "moisture_content"),
            PointMeasurement(6.0, "wn_pct", 42, "lab", "moisture_content"),
            PointMeasurement(9.0, "wn_pct", 38, "lab", "moisture_content"),
            # cu
            PointMeasurement(3.0, "cu_kPa", 25, "lab", "UU_triaxial"),
            PointMeasurement(6.0, "cu_kPa", 35, "lab", "UU_triaxial"),
            PointMeasurement(9.0, "cu_kPa", 40, "lab", "UU_triaxial"),
        ],
    )
    inv2 = Investigation(
        investigation_id="B-2",
        investigation_type="boring",
        x=160.0, y=210.0,
        elevation_m=11.0,
        total_depth_m=15.0,
        gwl_depth_m=4.0,
        lithology=[
            LithologyInterval(0, 3, "Fill", uscs="SM"),
            LithologyInterval(3, 10, "Stiff clay", uscs="CL"),
            LithologyInterval(10, 15, "Dense sand", uscs="SP"),
        ],
        measurements=[
            PointMeasurement(1.5, "N_spt", 6, "field", "SPT"),
            PointMeasurement(3.0, "N_spt", 10, "field", "SPT"),
            PointMeasurement(6.0, "N_spt", 12, "field", "SPT"),
            PointMeasurement(9.0, "N_spt", 15, "field", "SPT"),
            PointMeasurement(12.0, "N_spt", 30, "field", "SPT"),
            PointMeasurement(3.0, "LL_pct", 38, "lab", "Atterberg"),
            PointMeasurement(3.0, "PL_pct", 18, "lab", "Atterberg"),
            PointMeasurement(6.0, "LL_pct", 35, "lab", "Atterberg"),
            PointMeasurement(6.0, "PL_pct", 16, "lab", "Atterberg"),
            PointMeasurement(3.0, "wn_pct", 30, "lab", "moisture_content"),
            PointMeasurement(6.0, "wn_pct", 28, "lab", "moisture_content"),
        ],
    )
    inv3 = Investigation(
        investigation_id="CPT-1",
        investigation_type="cpt",
        x=130.0, y=205.0,
        elevation_m=10.5,
        total_depth_m=15.0,
        measurements=[
            PointMeasurement(d, "qc_kPa", 500 + d * 200, "field", "CPTu")
            for d in np.arange(0.5, 15.0, 0.5)
        ] + [
            PointMeasurement(d, "fs_kPa", 5 + d * 2, "field", "CPTu")
            for d in np.arange(0.5, 15.0, 0.5)
        ],
    )
    inv4 = Investigation(
        investigation_id="TP-1",
        investigation_type="test_pit",
        x=120.0, y=195.0,
        elevation_m=9.5,
        total_depth_m=3.0,
        lithology=[
            LithologyInterval(0, 1.5, "Topsoil", uscs="OL"),
            LithologyInterval(1.5, 3.0, "Brown clay", uscs="CL"),
        ],
        measurements=[
            PointMeasurement(0.5, "N_spt", 5, "field", "SPT"),
            PointMeasurement(2.0, "N_spt", 8, "field", "SPT"),
        ],
    )
    return SiteModel(
        project_name="Rich Test Site",
        investigations=[inv1, inv2, inv3, inv4],
        coordinate_system="EPSG:2263",
    )


# ---------------------------------------------------------------------------
# Synthetic DIGGS XML fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_boring_diggs_xml():
    """Minimal DIGGS 2.6 XML with one boring, lithology, and SPT."""
    return """<?xml version="1.0" encoding="UTF-8"?>
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


@pytest.fixture
def multi_boring_diggs_xml():
    """DIGGS 2.6 XML with two borings, lithology, SPT, Atterberg, moisture."""
    return """<?xml version="1.0" encoding="UTF-8"?>
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


@pytest.fixture
def diggs_25a_xml():
    """DIGGS 2.5.a namespace XML (minimal)."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<Diggs xmlns="http://diggsml.org/schemas/2.5.a"
       xmlns:gml="http://www.opengis.net/gml/3.2"
       xmlns:xlink="http://www.w3.org/1999/xlink">
  <Project>
    <gml:name>Legacy Project</gml:name>
  </Project>
  <Borehole gml:id="BH1">
    <gml:name>B-1</gml:name>
    <referencePoint>
      <gml:Point>
        <gml:pos>50.0 60.0</gml:pos>
      </gml:Point>
    </referencePoint>
    <totalMeasuredDepth>10.0</totalMeasuredDepth>
  </Borehole>
  <DrivenPenetrationTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>2.0</depth>
    <blowCount>12</blowCount>
  </DrivenPenetrationTest>
</Diggs>"""

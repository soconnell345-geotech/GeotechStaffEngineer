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


# ---------------------------------------------------------------------------
# DIGGS XML fixtures for new test types
# ---------------------------------------------------------------------------

_DIGGS_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<Diggs xmlns="http://diggsml.org/schemas/2.6"
       xmlns:gml="http://www.opengis.net/gml/3.2"
       xmlns:xlink="http://www.w3.org/1999/xlink">
  <Project><gml:name>Test Project</gml:name></Project>
  <Borehole gml:id="BH1">
    <gml:name>B-1</gml:name>
    <totalMeasuredDepth>20.0</totalMeasuredDepth>
  </Borehole>"""

_DIGGS_HEADER_TWO_BORINGS = """<?xml version="1.0" encoding="UTF-8"?>
<Diggs xmlns="http://diggsml.org/schemas/2.6"
       xmlns:gml="http://www.opengis.net/gml/3.2"
       xmlns:xlink="http://www.w3.org/1999/xlink">
  <Project><gml:name>Test Project</gml:name></Project>
  <Borehole gml:id="BH1">
    <gml:name>B-1</gml:name>
    <totalMeasuredDepth>20.0</totalMeasuredDepth>
  </Borehole>
  <Borehole gml:id="BH2">
    <gml:name>B-2</gml:name>
    <totalMeasuredDepth>15.0</totalMeasuredDepth>
  </Borehole>"""

_DIGGS_FOOTER = "\n</Diggs>"


@pytest.fixture
def cpt_diggs_xml():
    """DIGGS XML with StaticConePenetrationTest (single-depth elements)."""
    return _DIGGS_HEADER + """
  <StaticConePenetrationTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>1.0</depth>
    <coneResistance>2500</coneResistance>
    <sleeveFriction>35</sleeveFriction>
    <porePressure>50</porePressure>
    <frictionRatio>1.4</frictionRatio>
  </StaticConePenetrationTest>
  <StaticConePenetrationTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>2.0</depth>
    <coneResistance>4800</coneResistance>
    <sleeveFriction>60</sleeveFriction>
    <porePressure>110</porePressure>
    <frictionRatio>1.25</frictionRatio>
  </StaticConePenetrationTest>""" + _DIGGS_FOOTER


@pytest.fixture
def cpt_nested_diggs_xml():
    """DIGGS XML with nested ConePenetrationReading elements."""
    return _DIGGS_HEADER + """
  <StaticConePenetrationTest>
    <investigationRef xlink:href="#BH1"/>
    <ConePenetrationReading>
      <depth>0.5</depth>
      <coneResistance>1200</coneResistance>
      <sleeveFriction>18</sleeveFriction>
    </ConePenetrationReading>
    <ConePenetrationReading>
      <depth>1.0</depth>
      <coneResistance>2400</coneResistance>
      <sleeveFriction>30</sleeveFriction>
    </ConePenetrationReading>
    <ConePenetrationReading>
      <depth>1.5</depth>
      <coneResistance>3600</coneResistance>
      <sleeveFriction>42</sleeveFriction>
    </ConePenetrationReading>
  </StaticConePenetrationTest>""" + _DIGGS_FOOTER


@pytest.fixture
def vane_diggs_xml():
    """DIGGS XML with InsituVaneTest."""
    return _DIGGS_HEADER + """
  <InsituVaneTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>5.0</depth>
    <peakShearStrength>45</peakShearStrength>
    <remoldedShearStrength>9</remoldedShearStrength>
    <sensitivity>5</sensitivity>
  </InsituVaneTest>""" + _DIGGS_FOOTER


@pytest.fixture
def vane_computed_st_xml():
    """Vane test without explicit sensitivity — should be computed."""
    return _DIGGS_HEADER + """
  <InsituVaneTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>4.0</depth>
    <peakShearStrength>60</peakShearStrength>
    <remoldedShearStrength>10</remoldedShearStrength>
  </InsituVaneTest>""" + _DIGGS_FOOTER


@pytest.fixture
def consolidation_diggs_xml():
    """DIGGS XML with ConsolidationTest."""
    return _DIGGS_HEADER + """
  <ConsolidationTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>6.0</depth>
    <initialVoidRatio>0.95</initialVoidRatio>
    <compressionIndex>0.35</compressionIndex>
    <recompressionIndex>0.06</recompressionIndex>
    <preconsolidationPressure>120</preconsolidationPressure>
  </ConsolidationTest>""" + _DIGGS_FOOTER


@pytest.fixture
def triaxial_uu_diggs_xml():
    """DIGGS XML with TriaxialTest (UU — cu only)."""
    return _DIGGS_HEADER + """
  <TriaxialTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>5.0</depth>
    <undrainedShearStrength>75</undrainedShearStrength>
  </TriaxialTest>""" + _DIGGS_FOOTER


@pytest.fixture
def triaxial_cd_diggs_xml():
    """DIGGS XML with TriaxialTest (CD — phi and c)."""
    return _DIGGS_HEADER + """
  <TriaxialTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>8.0</depth>
    <frictionAngle>32</frictionAngle>
    <cohesion>5</cohesion>
  </TriaxialTest>""" + _DIGGS_FOOTER


@pytest.fixture
def direct_shear_diggs_xml():
    """DIGGS XML with DirectShearTest."""
    return _DIGGS_HEADER + """
  <DirectShearTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>3.0</depth>
    <frictionAngle>28</frictionAngle>
    <cohesion>10</cohesion>
  </DirectShearTest>""" + _DIGGS_FOOTER


@pytest.fixture
def ucs_diggs_xml():
    """DIGGS XML with UnconfinedCompressiveStrengthTest."""
    return _DIGGS_HEADER + """
  <UnconfinedCompressiveStrengthTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>4.0</depth>
    <compressiveStrength>150</compressiveStrength>
  </UnconfinedCompressiveStrengthTest>""" + _DIGGS_FOOTER


@pytest.fixture
def point_load_diggs_xml():
    """DIGGS XML with PointLoadTest."""
    return _DIGGS_HEADER + """
  <PointLoadTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>12.0</depth>
    <pointLoadIndex>3.5</pointLoadIndex>
  </PointLoadTest>""" + _DIGGS_FOOTER


@pytest.fixture
def specific_gravity_diggs_xml():
    """DIGGS XML with SpecificGravityTest."""
    return _DIGGS_HEADER + """
  <SpecificGravityTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>5.0</depth>
    <specificGravity>2.68</specificGravity>
  </SpecificGravityTest>""" + _DIGGS_FOOTER


@pytest.fixture
def gradation_diggs_xml():
    """DIGGS XML with ParticleSizeAnalysis."""
    return _DIGGS_HEADER + """
  <ParticleSizeAnalysis>
    <investigationRef xlink:href="#BH1"/>
    <depth>3.0</depth>
    <percentGravel>15</percentGravel>
    <percentSand>55</percentSand>
    <percentFines>30</percentFines>
    <D10>0.002</D10>
    <D30>0.05</D30>
    <D60>0.5</D60>
  </ParticleSizeAnalysis>""" + _DIGGS_FOOTER


@pytest.fixture
def gradation_alt_tag_xml():
    """DIGGS XML with MaterialGradationTest (alternate tag)."""
    return _DIGGS_HEADER + """
  <MaterialGradationTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>6.0</depth>
    <percentGravel>40</percentGravel>
    <percentSand>45</percentSand>
    <percentFines>15</percentFines>
  </MaterialGradationTest>""" + _DIGGS_FOOTER


@pytest.fixture
def permeability_diggs_xml():
    """DIGGS XML with LabPermeabilityTest."""
    return _DIGGS_HEADER + """
  <LabPermeabilityTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>7.0</depth>
    <permeability>1.5e-8</permeability>
  </LabPermeabilityTest>""" + _DIGGS_FOOTER


@pytest.fixture
def compaction_diggs_xml():
    """DIGGS XML with LabCompactionTest."""
    return _DIGGS_HEADER + """
  <LabCompactionTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>2.0</depth>
    <maxDryDensity>19.5</maxDryDensity>
    <optimumMoistureContent>12.5</optimumMoistureContent>
  </LabCompactionTest>""" + _DIGGS_FOOTER


@pytest.fixture
def dynamic_probe_diggs_xml():
    """DIGGS XML with DynamicProbeTest."""
    return _DIGGS_HEADER + """
  <DynamicProbeTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>3.0</depth>
    <blowCount>18</blowCount>
  </DynamicProbeTest>
  <DynamicProbeTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>6.0</depth>
    <blowCount>25</blowCount>
  </DynamicProbeTest>""" + _DIGGS_FOOTER


@pytest.fixture
def pressuremeter_diggs_xml():
    """DIGGS XML with PressuremeterTest."""
    return _DIGGS_HEADER + """
  <PressuremeterTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>8.0</depth>
    <pressureMeterModulus>15000</pressureMeterModulus>
    <limitPressure>1200</limitPressure>
  </PressuremeterTest>""" + _DIGGS_FOOTER


@pytest.fixture
def dmt_diggs_xml():
    """DIGGS XML with FlatPlateDilatometerTest."""
    return _DIGGS_HEADER + """
  <FlatPlateDilatometerTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>5.0</depth>
    <dilatometerModulus>22000</dilatometerModulus>
    <materialIndex>1.8</materialIndex>
    <horizontalStressIndex>4.5</horizontalStressIndex>
  </FlatPlateDilatometerTest>""" + _DIGGS_FOOTER


@pytest.fixture
def multi_boring_xlink_xml():
    """DIGGS XML with two borings and tests associated via xlink."""
    return _DIGGS_HEADER_TWO_BORINGS + """
  <InsituVaneTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>5.0</depth>
    <peakShearStrength>40</peakShearStrength>
  </InsituVaneTest>
  <InsituVaneTest>
    <investigationRef xlink:href="#BH2"/>
    <depth>6.0</depth>
    <peakShearStrength>55</peakShearStrength>
  </InsituVaneTest>
  <ConsolidationTest>
    <investigationRef xlink:href="#BH2"/>
    <depth>7.0</depth>
    <compressionIndex>0.40</compressionIndex>
    <preconsolidationPressure>150</preconsolidationPressure>
  </ConsolidationTest>
  <StaticConePenetrationTest>
    <investigationRef xlink:href="#BH1"/>
    <depth>3.0</depth>
    <coneResistance>5000</coneResistance>
    <sleeveFriction>70</sleeveFriction>
  </StaticConePenetrationTest>""" + _DIGGS_FOOTER

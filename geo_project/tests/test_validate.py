"""Validator tests — every rule exercised BOTH ways (fires / stays quiet)."""

import pytest

from geo_project.schema import (
    Anchor,
    FEMAnalysis,
    GeosyntheticLayer,
    Geometry,
    Layer,
    LEAnalysis,
    LEProbabilistic,
    Loads,
    Material,
    Nail,
    Project,
    Reinforcement,
    Surcharge,
    Water,
)
from geo_project.validate import has_errors, summarize, validate


def codes(issues, level=None):
    return [i.code for i in issues
            if level is None or i.level == level]


def make_clean_le_project() -> Project:
    """A complete, analysis-ready single-bench slope (LE requested)."""
    return Project(
        geometry=Geometry(
            surface_points=[(0, 10), (10, 10), (30, 0), (45, 0)],
        ),
        stratigraphy=[
            Layer(name="Sand",
                  material=Material(gamma=19.0, phi=33.0, c_prime=0.0),
                  bottom_elevation=2.0),
            Layer(name="Clay",
                  material=Material(strength_model="undrained",
                                    gamma=17.0, cu=50.0),
                  bottom_elevation=-8.0),
        ],
        water=Water(gwt_points=[(0, 5.0), (45, -2.0)]),
        analyses=[LEAnalysis(method="bishop")],
    )


def test_clean_project_validates_clean():
    issues = validate(make_clean_le_project())
    assert not has_errors(issues), summarize(issues)
    assert issues == []  # not even warnings


# --- geometry --------------------------------------------------------------

def test_geom001_too_few_surface_points():
    p = make_clean_le_project()
    p.geometry.surface_points = [(0, 10)]
    assert "GEOM001" in codes(validate(p), "error")


def test_geom002_non_monotonic_surface():
    p = make_clean_le_project()
    p.geometry.surface_points = [(0, 10), (30, 0), (10, 10), (45, 0)]
    assert "GEOM002" in codes(validate(p), "error")


def test_geom003_no_layers():
    p = make_clean_le_project()
    p.stratigraphy = []
    assert "GEOM003" in codes(validate(p), "error")


def test_geom004_gap_and_overlap():
    p = make_clean_le_project()
    p.stratigraphy[1].top_elevation = 1.0  # gap below Sand bottom (2.0)
    assert "GEOM004" in codes(validate(p), "error")
    p2 = make_clean_le_project()
    p2.stratigraphy[1].top_elevation = 3.0  # overlap
    assert "GEOM004" in codes(validate(p2), "error")


def test_geom004_zero_thickness():
    p = make_clean_le_project()
    p.stratigraphy[0].top_elevation = 2.0  # == its bottom
    assert "GEOM004" in codes(validate(p), "error")


def test_geom005_top_layer_below_crest():
    p = make_clean_le_project()
    p.stratigraphy[0].top_elevation = 8.0  # crest at 10
    assert "GEOM005" in codes(validate(p), "error")


def test_geom006_shallow_section_warns():
    p = make_clean_le_project()
    p.stratigraphy[1].bottom_elevation = 1.0
    p.stratigraphy[1].top_elevation = None
    # bottom (1.0) above surface low point (0.0) → warning
    assert "GEOM006" in codes(validate(p), "warning")


def test_geom007_vision_draft_blocks_until_confirmed():
    p = make_clean_le_project()
    p.geometry.provenance = "vision_draft"
    issues = validate(p)
    assert "GEOM007" in codes(issues, "error")
    # Confirming the geometry stage clears the block.
    p.confirmations.geometry = True
    assert "GEOM007" not in codes(validate(p))


def test_geom008_undetermined_bottom():
    p = make_clean_le_project()
    p.stratigraphy[1].bottom_elevation = None
    assert "GEOM008" in codes(validate(p), "error")


def test_geom009_short_boundary_warns():
    p = make_clean_le_project()
    p.geometry.layer_boundaries["b1"] = [(5, 2.0), (20, 2.0)]  # short span
    p.stratigraphy[0].bottom_boundary = "b1"
    p.stratigraphy[0].bottom_elevation = None
    assert "GEOM009" in codes(validate(p), "warning")
    # Full-span boundary stays quiet.
    p.geometry.layer_boundaries["b1"] = [(0, 2.0), (45, 2.0)]
    assert "GEOM009" not in codes(validate(p))


# --- water -------------------------------------------------------------------

def test_water001_ponded_warning_vs_declared_info():
    p = make_clean_le_project()
    p.water.gwt_points = [(0, 12.0), (45, -2.0)]  # above crest at x=0
    issues = validate(p)
    ponded = [i for i in issues if i.code == "WATER001"]
    assert ponded and ponded[0].level == "warning"
    p.water.ponded = True
    issues2 = validate(p)
    ponded2 = [i for i in issues2 if i.code == "WATER001"]
    assert ponded2 and ponded2[0].level == "info"


def test_water002_gwt_below_section():
    p = make_clean_le_project()
    p.water.gwt_points = [(0, -20.0), (45, -20.0)]
    assert "WATER002" in codes(validate(p), "warning")


def test_water003_ru_out_of_range():
    p = make_clean_le_project()
    p.water.ru = 1.2
    assert "WATER003" in codes(validate(p), "error")
    p.water.ru = 0.3
    assert "WATER003" not in codes(validate(p))


# --- loads -------------------------------------------------------------------

def test_load001_negative_surcharge():
    p = make_clean_le_project()
    p.loads.surcharges = [Surcharge(q=-5.0)]
    assert "LOAD001" in codes(validate(p), "error")


def test_load002_multiple_surcharges_warn():
    p = make_clean_le_project()
    p.loads.surcharges = [Surcharge(q=10.0), Surcharge(q=5.0)]
    assert "LOAD002" in codes(validate(p), "warning")


def test_load003_band_outside_section():
    p = make_clean_le_project()
    p.loads.surcharges = [Surcharge(q=10.0, x_start=100.0, x_end=120.0)]
    assert "LOAD003" in codes(validate(p), "error")
    p.loads.surcharges = [Surcharge(q=10.0, x_start=8.0, x_end=2.0)]
    assert "LOAD003" in codes(validate(p), "error")
    p.loads.surcharges = [Surcharge(q=10.0, x_start=0.0, x_end=8.0)]
    assert "LOAD003" not in codes(validate(p))


def test_load004_kh_range():
    p = make_clean_le_project()
    p.loads.kh = -0.1
    assert "LOAD004" in codes(validate(p), "error")
    p.loads.kh = 0.6
    assert "LOAD004" in codes(validate(p), "warning")
    p.loads.kh = 0.15
    assert "LOAD004" not in codes(validate(p))


# --- reinforcement ------------------------------------------------------------

def test_reinf001_outside_section():
    p = make_clean_le_project()
    p.reinforcement = Reinforcement(nails=[Nail(x_head=99.0, z_head=5.0,
                                                length=8.0)])
    assert "REINF001" in codes(validate(p), "error")
    p.reinforcement = Reinforcement(anchors=[Anchor(x_head=20.0, z_head=99.0,
                                                    length=8.0,
                                                    T_allow=100.0)])
    assert "REINF001" in codes(validate(p), "error")
    p.reinforcement = Reinforcement(
        geosynthetics=[GeosyntheticLayer(elevation=-50.0, T_allow=20.0)])
    assert "REINF001" in codes(validate(p), "error")
    # Inside: quiet.
    p.reinforcement = Reinforcement(nails=[Nail(x_head=22.0, z_head=4.0,
                                                length=8.0)])
    assert "REINF001" not in codes(validate(p))


# --- materials (completeness FOR REQUESTED ANALYSES) --------------------------

def test_materials_not_required_when_no_analyses():
    p = make_clean_le_project()
    p.analyses = []
    p.stratigraphy[0].material = Material()  # everything unset
    issues = validate(p)
    assert not [c for c in codes(issues) if c.startswith("MAT")]


def test_mat001_gamma_missing():
    p = make_clean_le_project()
    p.stratigraphy[0].material.gamma = None
    assert "MAT001" in codes(validate(p), "error")


def test_mat002_mohr_coulomb_incomplete():
    p = make_clean_le_project()
    p.stratigraphy[0].material.phi = None
    assert "MAT002" in codes(validate(p), "error")


def test_mat003_undrained_needs_cu():
    p = make_clean_le_project()
    p.stratigraphy[1].material.cu = None
    assert "MAT003" in codes(validate(p), "error")
    p.stratigraphy[1].material.cu = 0.0
    assert "MAT003" in codes(validate(p), "error")


def test_mat004_shansep_incomplete_and_ranges():
    p = make_clean_le_project()
    p.stratigraphy[1].material = Material(strength_model="shansep",
                                          gamma=17.0)
    assert "MAT004" in codes(validate(p), "error")
    p.stratigraphy[1].material = Material(
        strength_model="shansep", gamma=17.0,
        shansep_S=0.22, shansep_m=0.8, ocr=0.5)  # OCR < 1
    assert "MAT004" in codes(validate(p), "error")
    p.stratigraphy[1].material.ocr = 2.0
    assert "MAT004" not in codes(validate(p))


def test_mat005_hoek_brown_incomplete_and_ranges():
    p = make_clean_le_project()
    p.stratigraphy[1].material = Material(strength_model="hoek_brown",
                                          gamma=24.0)
    assert "MAT005" in codes(validate(p), "error")
    p.stratigraphy[1].material = Material(
        strength_model="hoek_brown", gamma=24.0,
        hb_sigci=-5.0, hb_gsi=45.0, hb_mi=10.0)
    assert "MAT005" in codes(validate(p), "error")
    p.stratigraphy[1].material.hb_sigci = 25000.0
    assert "MAT005" not in codes(validate(p))


def test_mat006_fem_needs_stiffness():
    p = make_clean_le_project()
    p.analyses.append(FEMAnalysis())
    issues = validate(p)
    assert "MAT006" in codes(issues, "error")
    for layer in p.stratigraphy:
        layer.material.E = 30000.0
        layer.material.nu = 0.3
    assert "MAT006" not in codes(validate(p))


def test_mat007_fem_with_shansep_blocked():
    p = make_clean_le_project()
    p.stratigraphy[1].material = Material(
        strength_model="shansep", gamma=17.0,
        shansep_S=0.22, shansep_m=0.8, ocr=2.0, E=20000.0, nu=0.35)
    p.stratigraphy[0].material.E = 30000.0
    p.stratigraphy[0].material.nu = 0.3
    p.analyses.append(FEMAnalysis())
    assert "MAT007" in codes(validate(p), "error")


def test_mat008_unknown_probabilistic_key():
    p = make_clean_le_project()
    p.stratigraphy[0].material.probabilistic = {"phee": {"cov": 0.1}}
    assert "MAT008" in codes(validate(p), "error")
    p.stratigraphy[0].material.probabilistic = {"phi": {"cov": 0.1}}
    assert "MAT008" not in codes(validate(p))


# --- unit sanity ---------------------------------------------------------------

def test_unit001_gamma_range():
    p = make_clean_le_project()
    p.stratigraphy[0].material.gamma = 120.0  # pcf entered as kN/m3
    assert "UNIT001" in codes(validate(p), "warning")


def test_unit002_phi_range():
    p = make_clean_le_project()
    p.stratigraphy[0].material.phi = 55.0
    assert "UNIT002" in codes(validate(p), "warning")


def test_unit003_nu_range():
    p = make_clean_le_project()
    p.stratigraphy[0].material.nu = 0.55
    assert "UNIT003" in codes(validate(p), "warning")


def test_unit004_strength_magnitude():
    p = make_clean_le_project()
    p.stratigraphy[0].material.c_prime = 2000.0
    assert "UNIT004" in codes(validate(p), "warning")
    p2 = make_clean_le_project()
    p2.stratigraphy[1].material.cu = 5000.0
    assert "UNIT004" in codes(validate(p2), "warning")


def test_unit005_E_range():
    p = make_clean_le_project()
    p.stratigraphy[0].material.E = 30.0  # MPa entered as kPa
    assert "UNIT005" in codes(validate(p), "warning")


# --- analyses -------------------------------------------------------------------

def test_anal001_unknown_method_and_kind():
    p = make_clean_le_project()
    p.analyses = [LEAnalysis(method="slide2")]
    assert "ANAL001" in codes(validate(p), "error")
    p.analyses = [LEAnalysis(probabilistic=LEProbabilistic(kind="bayes"))]
    assert "ANAL001" in codes(validate(p), "error")
    p.analyses = [LEAnalysis(probabilistic=LEProbabilistic(
        kind="fosm", variables={"banana": {"cov": 0.1}}))]
    assert "ANAL001" in codes(validate(p), "error")
    p.analyses = [FEMAnalysis(element_type="q8")]
    assert "ANAL001" in codes(validate(p), "error")


def test_summarize_formats():
    p = make_clean_le_project()
    p.stratigraphy[0].material.gamma = None
    text = summarize(validate(p))
    assert "MAT001" in text and "[ERROR]" in text
    assert summarize([]) == "Validation: clean (no issues)."

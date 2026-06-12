"""Builder tests against a hand-built slope_stability reference."""

import numpy as np
import pytest

from geo_project.builders import (
    ProjectValidationError,
    run_analyses,
    to_fem_kwargs,
    to_slope_geometry,
)
from geo_project.schema import (
    Anchor,
    FEMAnalysis,
    GeosyntheticLayer,
    Geometry,
    Layer,
    LEAnalysis,
    LEProbabilistic,
    LESearch,
    Loads,
    Material,
    Nail,
    Project,
    Reinforcement,
    Surcharge,
    Water,
)


def make_project() -> Project:
    return Project(
        geometry=Geometry(
            surface_points=[(0, 10), (10, 10), (30, 0), (45, 0)],
            layer_boundaries={"clay_top": [(0, 2.0), (45, 1.0)]},
        ),
        stratigraphy=[
            Layer(name="Sand",
                  material=Material(gamma=19.0, gamma_sat=20.0, phi=33.0,
                                    c_prime=0.0, E=30000.0, nu=0.3),
                  bottom_boundary="clay_top"),
            Layer(name="Clay",
                  material=Material(strength_model="undrained", gamma=17.0,
                                    cu=50.0, E=15000.0, nu=0.4),
                  bottom_elevation=-8.0),
        ],
        water=Water(gwt_points=[(0, 5.0), (45, -2.0)]),
        loads=Loads(surcharges=[Surcharge(q=12.0, x_start=0.0, x_end=8.0)],
                    kh=0.1),
        reinforcement=Reinforcement(
            nails=[Nail(x_head=25.0, z_head=2.5, length=9.0,
                        inclination=20.0, spacing_h=2.0)],
            anchors=[Anchor(x_head=20.0, z_head=5.0, length=12.0,
                            T_allow=150.0, inclination=10.0)],
            geosynthetics=[GeosyntheticLayer(elevation=1.0, T_allow=30.0,
                                             x_start=10.0, x_end=40.0)],
        ),
        analyses=[LEAnalysis(method="bishop")],
    )


# --- to_slope_geometry -------------------------------------------------------

def test_to_slope_geometry_matches_hand_built():
    from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
    p = make_project()
    geom = to_slope_geometry(p)

    # Hand-built reference.
    ref = SlopeGeometry(
        surface_points=[(0, 10), (10, 10), (30, 0), (45, 0)],
        soil_layers=[
            SlopeSoilLayer(name="Sand", top_elevation=10.0,
                           bottom_elevation=2.0, gamma=19.0, gamma_sat=20.0,
                           phi=33.0, c_prime=0.0,
                           bottom_boundary_points=[(0, 2.0), (45, 1.0)]),
            SlopeSoilLayer(name="Clay", top_elevation=2.0,
                           bottom_elevation=-8.0, gamma=17.0, cu=50.0,
                           analysis_mode="undrained"),
        ],
        gwt_points=[(0, 5.0), (45, -2.0)],
        surcharge=12.0, surcharge_x_range=(0.0, 8.0), kh=0.1,
    )
    assert geom.surface_points == ref.surface_points
    assert geom.gwt_points == ref.gwt_points
    assert geom.surcharge == ref.surcharge
    assert geom.surcharge_x_range == ref.surcharge_x_range
    assert geom.kh == ref.kh
    for g, r in zip(geom.soil_layers, ref.soil_layers):
        assert g.name == r.name
        assert g.top_elevation == r.top_elevation
        assert g.bottom_elevation == r.bottom_elevation
        assert g.gamma == r.gamma
        assert g.gamma_sat == r.gamma_sat
        assert g.analysis_mode == r.analysis_mode
        assert g.shear_strength_params == r.shear_strength_params
    assert geom.soil_layers[0].bottom_boundary_points == [(0, 2.0), (45, 1.0)]
    # Position-dependent bottom is live.
    assert geom.soil_layers[0].bottom_at(45.0) == 1.0

    # Reinforcement objects.
    assert len(geom.nails) == 1
    assert geom.nails[0].x_head == 25.0
    assert geom.nails[0].spacing_h == 2.0
    assert len(geom.anchors) == 1
    assert geom.anchors[0].T_allow == 150.0
    assert len(geom.geosynthetics) == 1
    assert geom.geosynthetics[0].x_start == 10.0


def test_to_slope_geometry_shansep_and_hoek_brown():
    p = make_project()
    p.reinforcement = Reinforcement()
    p.stratigraphy[0].material = Material(
        strength_model="shansep", gamma=17.0, shansep_S=0.25,
        shansep_m=0.85, ocr=2.0, su_min=5.0)
    p.stratigraphy[1].material = Material(
        strength_model="hoek_brown", gamma=24.0, hb_sigci=25000.0,
        hb_gsi=45.0, hb_mi=10.0, hb_D=0.7)
    geom = to_slope_geometry(p)
    l0, l1 = geom.soil_layers
    assert l0.strength_model == "shansep"
    assert (l0.shansep_S, l0.shansep_m, l0.ocr, l0.su_min) == \
        (0.25, 0.85, 2.0, 5.0)
    assert l1.strength_model == "hoek_brown"
    assert (l1.hb_sigci, l1.hb_gsi, l1.hb_mi, l1.hb_D) == \
        (25000.0, 45.0, 10.0, 0.7)


def test_to_slope_geometry_ru_applied():
    p = make_project()
    p.water = Water(ru=0.25)
    geom = to_slope_geometry(p)
    assert all(L.ru == 0.25 for L in geom.soil_layers)


def test_builder_raises_on_invalid_project():
    p = make_project()
    p.stratigraphy[0].material.gamma = None
    with pytest.raises(ProjectValidationError) as ei:
        to_slope_geometry(p)
    assert "MAT001" in str(ei.value)


def test_vision_draft_blocks_builders_until_confirmed():
    p = make_project()
    p.geometry.provenance = "vision_draft"
    with pytest.raises(ProjectValidationError) as ei:
        to_slope_geometry(p)
    assert "GEOM007" in str(ei.value)
    p.confirmations.geometry = True
    geom = to_slope_geometry(p)  # confirmed → builds
    assert len(geom.soil_layers) == 2


# --- to_fem_kwargs -------------------------------------------------------------

def test_to_fem_kwargs_shape():
    p = make_project()
    p.analyses.append(FEMAnalysis(nx=20, ny=10, depth=12.0, x_extend=0.0))
    kw = to_fem_kwargs(p)
    assert kw["surface_points"] == [(0, 10), (10, 10), (30, 0), (45, 0)]
    assert kw["nx"] == 20 and kw["ny"] == 10
    assert kw["depth"] == 12.0 and kw["x_extend"] == 0.0
    assert isinstance(kw["gwt"], np.ndarray) and kw["gwt"].shape == (2, 2)
    layers = kw["soil_layers"]
    assert [L["name"] for L in layers] == ["Sand", "Clay"]
    sand, clay = layers
    assert sand["E"] == 30000.0 and sand["nu"] == 0.3
    assert sand["phi"] == 33.0 and sand["c"] == 0.0
    # Undrained → c = cu, phi = 0 (total stress).
    assert clay["c"] == 50.0 and clay["phi"] == 0.0
    assert clay["bottom_elevation"] == -8.0
    # Named boundary → layer_polylines for assign_layers_by_polylines.
    assert "layer_polylines" in kw
    np.testing.assert_allclose(kw["layer_polylines"][0],
                               [[0, 2.0], [45, 1.0]])


def test_to_fem_kwargs_requires_stiffness():
    p = make_project()
    p.analyses.append(FEMAnalysis())
    p.stratigraphy[0].material.E = None
    with pytest.raises(ProjectValidationError) as ei:
        to_fem_kwargs(p)
    assert "MAT006" in str(ei.value)


# --- run_analyses ----------------------------------------------------------------

def _small_le_project() -> Project:
    """Tiny homogeneous slope for a fast LE search."""
    return Project(
        geometry=Geometry(
            surface_points=[(0.0, 8.0), (8.0, 8.0), (24.0, 0.0), (36.0, 0.0)],
        ),
        stratigraphy=[Layer(
            name="Soil",
            material=Material(gamma=18.0, phi=25.0, c_prime=8.0,
                              probabilistic={
                                  "phi": {"cov": 0.10, "dist": "lognormal",
                                          "source": "Duncan (2000) Table 1"},
                              }),
            bottom_elevation=-6.0)],
        analyses=[LEAnalysis(
            name="LE-bishop", method="bishop", n_slices=20,
            search=LESearch(nx=4, ny=4),
            probabilistic=LEProbabilistic(kind="fosm"))],
    )


def test_run_analyses_le_matches_direct_search():
    from slope_stability.analysis import search_critical_surface
    p = _small_le_project()
    results = run_analyses(p)
    assert "LE-bishop" in results
    r = results["LE-bishop"]
    assert r["type"] == "le"
    assert r["FOS"] is not None and 0.5 < r["FOS"] < 4.0

    # Direct reference run on the equivalent hand mapping.
    geom = to_slope_geometry(p, check=False)
    ref = search_critical_surface(geom, method="bishop", n_slices=20,
                                  nx=4, ny=4)
    assert r["FOS"] == pytest.approx(ref.critical.FOS, rel=1e-9)

    # Probabilistic block assembled from the material's cited COV.
    prob = r["probabilistic"]
    assert prob["kind"] == "fosm"
    assert "phi:Soil" in prob["variables"]
    assert prob["fos_mlv"] == pytest.approx(r["FOS"], rel=1e-6)
    assert prob["beta_lognormal"] > 0


def test_run_analyses_refuses_invalid():
    p = _small_le_project()
    p.stratigraphy[0].material.phi = None
    with pytest.raises(ProjectValidationError):
        run_analyses(p)

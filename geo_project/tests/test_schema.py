"""Schema round-trip + unknown-key tolerance tests."""

import json

import pytest

from geo_project.schema import (
    SCHEMA_VERSION,
    Anchor,
    Confirmations,
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
    ProjectMeta,
    Reinforcement,
    Surcharge,
    Water,
)


def make_full_project() -> Project:
    """A project exercising every schema branch."""
    return Project(
        meta=ProjectMeta(name="Test slope", description="2H:1V cut"),
        geometry=Geometry(
            surface_points=[(0, 10), (10, 10), (30, 0), (45, 0)],
            layer_boundaries={"Clay": [(0, 4.0), (45, 3.0)]},
            provenance="user",
        ),
        stratigraphy=[
            Layer(name="Fill",
                  material=Material(strength_model="mohr_coulomb",
                                    gamma=19.0, phi=32.0, c_prime=2.0,
                                    E=30000.0, nu=0.3,
                                    probabilistic={"phi": {
                                        "cov": 0.08, "dist": "lognormal",
                                        "source": "Duncan (2000) Table 1"}}),
                  bottom_boundary="Clay"),
            Layer(name="Clay",
                  material=Material(strength_model="undrained",
                                    gamma=17.5, cu=45.0),
                  bottom_elevation=-8.0),
        ],
        water=Water(gwt_points=[(0, 6.0), (45, -1.0)], ru=0.0, ponded=False),
        loads=Loads(surcharges=[Surcharge(q=12.0, x_start=0.0, x_end=8.0,
                                          label="traffic")],
                    kh=0.0),
        reinforcement=Reinforcement(
            nails=[Nail(x_head=22.0, z_head=4.0, length=8.0)],
            anchors=[Anchor(x_head=20.0, z_head=6.0, length=10.0,
                            T_allow=150.0)],
            geosynthetics=[GeosyntheticLayer(elevation=2.0, T_allow=30.0)],
        ),
        analyses=[
            LEAnalysis(method="spencer", n_slices=40,
                       search=LESearch(surface_type="circular", nx=8, ny=8),
                       probabilistic=LEProbabilistic(kind="fosm")),
            FEMAnalysis(nx=24, ny=12, srf_tol=0.02),
        ],
        confirmations=Confirmations(geometry=True),
    )


def test_round_trip_json():
    p = make_full_project()
    p.add_assumption("stratigraphy[0].material.phi", 32.0,
                     source="assumed compacted fill")
    text = p.to_json()
    p2 = Project.from_json(text)
    # Tuples become lists in JSON; compare canonical dicts.
    assert p2.to_dict() == p.to_dict()
    # Types survive.
    assert isinstance(p2.analyses[0], LEAnalysis)
    assert isinstance(p2.analyses[1], FEMAnalysis)
    assert p2.geometry.surface_points[2] == (30.0, 0.0)
    assert p2.stratigraphy[0].material.probabilistic["phi"]["cov"] == 0.08
    assert p2.confirmations.geometry is True
    assert p2.confirmations.materials is False
    assert p2.assumptions[0].field == "stratigraphy[0].material.phi"


def test_schema_version_present():
    p = Project()
    d = p.to_dict()
    assert d["meta"]["schema_version"] == SCHEMA_VERSION


def test_unknown_keys_tolerated_everywhere():
    d = make_full_project().to_dict()
    # Sprinkle unknown keys at several depths (forward compatibility).
    d["totally_new_top_level"] = {"x": 1}
    d["meta"]["new_meta_field"] = "hello"
    d["geometry"]["crs"] = "EPSG:9999"
    d["stratigraphy"][0]["color"] = "#ff0000"
    d["stratigraphy"][0]["material"]["new_strength_param"] = 42
    d["water"]["aquifer_name"] = "A1"
    d["loads"]["surcharges"][0]["applied_by"] = "truck"
    d["reinforcement"]["nails"][0]["coating"] = "epoxy"
    d["analyses"][0]["solver_hint"] = "fast"
    d["confirmations"]["extra_stage"] = True
    p = Project.from_dict(d)
    assert p.meta.name == "Test slope"
    assert p.stratigraphy[0].material.gamma == 19.0
    assert p.analyses[0].method == "spencer"


def test_layer_top_bottom_resolution():
    p = make_full_project()
    # Layer 0: top from surface crest, bottom from the 'Clay' boundary max z.
    assert p.layer_top(0) == 10.0
    assert p.layer_bottom(0) == 4.0
    # Layer 1: top chains from layer 0 bottom; bottom explicit.
    assert p.layer_top(1) == 4.0
    assert p.layer_bottom(1) == -8.0
    assert p.section_bottom() == -8.0
    # Boundary points come back sorted by x.
    bp = p.boundary_points(0)
    assert bp == [(0.0, 4.0), (45.0, 3.0)]


def test_bad_provenance_rejected():
    with pytest.raises(ValueError):
        Geometry(surface_points=[(0, 0), (1, 0)], provenance="hallucinated")


def test_bad_strength_model_rejected():
    with pytest.raises(ValueError):
        Material(strength_model="tresca")


def test_confirmations_missing_and_all():
    c = Confirmations()
    assert c.all_confirmed() is False
    assert c.missing() == ["geometry", "materials", "water_loads"]
    c.geometry = c.materials = c.water_loads = True
    assert c.all_confirmed() is True
    assert c.missing() == []


def test_add_assumption_dedupes():
    p = Project()
    p.add_assumption("water.ru", 0.0, source="default")
    p.add_assumption("water.ru", 0.0, source="default")
    assert len(p.assumptions) == 1


def test_from_json_points_are_tuples():
    p = Project.from_json(json.dumps({
        "geometry": {"surface_points": [[0, 1], [2, 3]],
                     "layer_boundaries": {"b": [[0, 0], [2, 0]]}},
        "water": {"gwt_points": [[0, 0.5], [2, 0.5]]},
    }))
    assert p.geometry.surface_points == [(0.0, 1.0), (2.0, 3.0)]
    assert p.geometry.layer_boundaries["b"][1] == (2.0, 0.0)
    assert p.water.gwt_points == [(0.0, 0.5), (2.0, 0.5)]

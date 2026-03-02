"""Tests for dxf_import.converter — build_fem_inputs() FEM conversion."""

import numpy as np
import pytest

ezdxf = pytest.importorskip("ezdxf")

from dxf_import.parser import LayerMapping, parse_dxf_geometry
from dxf_import.converter import (
    FEMSoilPropertyAssignment, build_fem_inputs, build_slope_geometry,
    SoilPropertyAssignment,
)
from dxf_import.results import DxfParseResult


class TestFEMSoilPropertyAssignment:
    """Tests for the FEMSoilPropertyAssignment dataclass."""

    def test_defaults(self):
        sp = FEMSoilPropertyAssignment()
        assert sp.E == 30000.0
        assert sp.nu == 0.3
        assert sp.model == "mc"
        assert sp.hs_params is None

    def test_custom_values(self):
        sp = FEMSoilPropertyAssignment(
            name="Sand", gamma=19.0, phi=35.0, c=0.0,
            E=50000.0, nu=0.25, psi=5.0, model="mc",
        )
        assert sp.name == "Sand"
        assert sp.E == 50000.0
        assert sp.psi == 5.0

    def test_hs_params(self):
        hs = {"E50_ref": 25000, "Eur_ref": 75000, "m": 0.5,
               "p_ref": 100, "R_f": 0.9}
        sp = FEMSoilPropertyAssignment(name="Clay", model="hs", hs_params=hs)
        assert sp.model == "hs"
        assert sp.hs_params["E50_ref"] == 25000


class TestBuildFEMInputsTwoLayer:
    """Tests for build_fem_inputs with a two-layer profile."""

    def test_returns_dict(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [
            FEMSoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0, E=40000),
            FEMSoilPropertyAssignment(name="Clay", gamma=19.0, phi=25.0, c=10.0, E=20000),
        ]
        result = build_fem_inputs(parse, props)
        assert isinstance(result, dict)
        assert "surface_points" in result
        assert "soil_layers" in result
        assert "gwt" in result
        assert "boundary_polylines" in result

    def test_surface_passthrough(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [
            FEMSoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0),
            FEMSoilPropertyAssignment(name="Clay", gamma=19.0, phi=25.0),
        ]
        result = build_fem_inputs(parse, props)
        assert len(result["surface_points"]) == 4

    def test_layer_count(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [
            FEMSoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0),
            FEMSoilPropertyAssignment(name="Clay", gamma=19.0, phi=25.0),
        ]
        result = build_fem_inputs(parse, props)
        assert len(result["soil_layers"]) == 2

    def test_stiffness_params_in_layers(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [
            FEMSoilPropertyAssignment(name="Surface", E=40000, nu=0.25),
            FEMSoilPropertyAssignment(name="Clay", E=15000, nu=0.35),
        ]
        result = build_fem_inputs(parse, props)
        assert result["soil_layers"][0]["E"] == 40000
        assert result["soil_layers"][0]["nu"] == 0.25
        assert result["soil_layers"][1]["E"] == 15000
        assert result["soil_layers"][1]["nu"] == 0.35

    def test_layer_elevations(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [
            FEMSoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0),
            FEMSoilPropertyAssignment(name="Clay", gamma=19.0, phi=25.0),
        ]
        result = build_fem_inputs(parse, props)
        # Top layer: surface_max=10 → clay_max=3
        assert result["soil_layers"][0]["top_elevation"] == 10.0
        assert result["soil_layers"][0]["bottom_elevation"] == 3.0
        # Clay: top=3, bottom extended below
        assert result["soil_layers"][1]["top_elevation"] == 3.0

    def test_boundary_polylines(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [
            FEMSoilPropertyAssignment(name="Surface"),
            FEMSoilPropertyAssignment(name="Clay"),
        ]
        result = build_fem_inputs(parse, props)
        assert len(result["boundary_polylines"]) == 1
        poly = result["boundary_polylines"][0]
        assert isinstance(poly, np.ndarray)
        assert poly.shape[1] == 2

    def test_gwt_passthrough(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
            water_table="WATER_TABLE",
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [
            FEMSoilPropertyAssignment(name="Surface"),
            FEMSoilPropertyAssignment(name="Clay"),
        ]
        result = build_fem_inputs(parse, props)
        assert result["gwt"] is not None
        assert isinstance(result["gwt"], np.ndarray)
        assert result["gwt"].shape == (4, 2)

    def test_no_gwt(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [
            FEMSoilPropertyAssignment(name="Surface"),
            FEMSoilPropertyAssignment(name="Clay"),
        ]
        result = build_fem_inputs(parse, props)
        assert result["gwt"] is None


class TestBuildFEMInputsMultiLayer:
    """Tests for multi-layer FEM inputs."""

    def test_four_layers(self, multi_layer_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={
                "FILL_SAND_BOUNDARY": "Sand",
                "SAND_CLAY_BOUNDARY": "Clay",
                "CLAY_ROCK_BOUNDARY": "Rock",
            },
        )
        parse = parse_dxf_geometry(
            filepath=multi_layer_dxf, layer_mapping=mapping
        )
        props = [
            FEMSoilPropertyAssignment(name="Fill", gamma=17.0, phi=28.0, E=25000),
            FEMSoilPropertyAssignment(name="Sand", gamma=19.0, phi=35.0, E=50000),
            FEMSoilPropertyAssignment(name="Clay", gamma=18.0, phi=20.0, c=15.0, E=12000),
            FEMSoilPropertyAssignment(name="Rock", gamma=22.0, phi=45.0, c=50.0, E=200000),
        ]
        result = build_fem_inputs(parse, props)
        assert len(result["soil_layers"]) == 4
        assert len(result["boundary_polylines"]) == 3

    def test_layer_ordering(self, multi_layer_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={
                "FILL_SAND_BOUNDARY": "Sand",
                "SAND_CLAY_BOUNDARY": "Clay",
                "CLAY_ROCK_BOUNDARY": "Rock",
            },
        )
        parse = parse_dxf_geometry(
            filepath=multi_layer_dxf, layer_mapping=mapping
        )
        props = [
            FEMSoilPropertyAssignment(name="Fill", gamma=17.0, phi=28.0),
            FEMSoilPropertyAssignment(name="Sand", gamma=19.0, phi=35.0),
            FEMSoilPropertyAssignment(name="Clay", gamma=18.0, phi=20.0),
            FEMSoilPropertyAssignment(name="Rock", gamma=22.0, phi=45.0),
        ]
        result = build_fem_inputs(parse, props)
        for i in range(len(result["soil_layers"]) - 1):
            assert (
                result["soil_layers"][i]["top_elevation"]
                >= result["soil_layers"][i + 1]["top_elevation"]
            )


class TestBuildFEMInputsHSModel:
    """Tests for Hardening Soil model propagation."""

    def test_hs_params_in_output(self):
        parse = DxfParseResult(
            surface_points=[(0, 10), (20, 10), (40, 5)],
            boundary_profiles={"Clay": [(0, 5), (20, 5), (40, 3)]},
        )
        hs = {"E50_ref": 25000, "Eur_ref": 75000, "m": 0.5,
              "p_ref": 100, "R_f": 0.9}
        props = [
            FEMSoilPropertyAssignment(name="Surface", model="elastic", E=50000),
            FEMSoilPropertyAssignment(name="Clay", model="hs", hs_params=hs),
        ]
        result = build_fem_inputs(parse, props)
        clay_layer = result["soil_layers"][1]
        assert clay_layer["model"] == "hs"
        assert clay_layer["hs_params"]["E50_ref"] == 25000
        assert clay_layer["hs_params"]["m"] == 0.5

    def test_mc_no_hs_params(self):
        parse = DxfParseResult(
            surface_points=[(0, 10), (20, 10)],
        )
        props = [
            FEMSoilPropertyAssignment(name="Soil", model="mc", phi=30, c=5),
        ]
        result = build_fem_inputs(parse, props)
        assert "hs_params" not in result["soil_layers"][0]


class TestBuildFEMInputsEdgeCases:
    """Tests for error handling in build_fem_inputs()."""

    def test_no_surface_points_raises(self):
        parse = DxfParseResult(surface_points=[])
        props = [FEMSoilPropertyAssignment(name="Soil")]
        with pytest.raises(ValueError, match="No surface points"):
            build_fem_inputs(parse, props)

    def test_no_properties_raises(self):
        parse = DxfParseResult(surface_points=[(0, 10), (20, 10)])
        with pytest.raises(ValueError, match="At least one"):
            build_fem_inputs(parse, [])

    def test_missing_boundary_property_raises(self):
        parse = DxfParseResult(
            surface_points=[(0, 10), (20, 10)],
            boundary_profiles={"Clay": [(0, 5), (20, 5)]},
        )
        props = [FEMSoilPropertyAssignment(name="Surface")]
        with pytest.raises(ValueError, match="No FEMSoilPropertyAssignment"):
            build_fem_inputs(parse, props)

    def test_single_layer_no_boundary(self):
        parse = DxfParseResult(
            surface_points=[(0, 10), (20, 10), (40, 5)],
        )
        props = [FEMSoilPropertyAssignment(name="Soil", E=30000)]
        result = build_fem_inputs(parse, props)
        assert len(result["soil_layers"]) == 1
        assert result["soil_layers"][0]["name"] == "Soil"
        assert result["boundary_polylines"] == []

    def test_model_field_preserved(self):
        parse = DxfParseResult(
            surface_points=[(0, 10), (20, 10)],
        )
        props = [FEMSoilPropertyAssignment(name="Soil", model="elastic")]
        result = build_fem_inputs(parse, props)
        assert result["soil_layers"][0]["model"] == "elastic"

    def test_psi_field_preserved(self):
        parse = DxfParseResult(
            surface_points=[(0, 10), (20, 10)],
        )
        props = [FEMSoilPropertyAssignment(name="Soil", psi=8.0)]
        result = build_fem_inputs(parse, props)
        assert result["soil_layers"][0]["psi"] == 8.0

"""Tests for dxf_import.converter — build_slope_geometry()."""

import math
import pytest

ezdxf = pytest.importorskip("ezdxf")

from dxf_import.parser import LayerMapping, parse_dxf_geometry
from dxf_import.converter import SoilPropertyAssignment, build_slope_geometry
from dxf_import.results import DxfParseResult
from slope_stability.geometry import SlopeGeometry


class TestBuildTwoLayerSlope:
    """Tests for a simple 2-layer slope build."""

    def test_returns_slope_geometry(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [
            SoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0, c_prime=5.0),
            SoilPropertyAssignment(name="Clay", gamma=19.0, cu=50.0, analysis_mode="undrained"),
        ]
        geom = build_slope_geometry(parse, props)
        assert isinstance(geom, SlopeGeometry)

    def test_layer_count(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [
            SoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0),
            SoilPropertyAssignment(name="Clay", gamma=19.0, cu=50.0, analysis_mode="undrained"),
        ]
        geom = build_slope_geometry(parse, props)
        assert len(geom.soil_layers) == 2

    def test_layer_elevations(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [
            SoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0),
            SoilPropertyAssignment(name="Clay", gamma=19.0, phi=25.0),
        ]
        geom = build_slope_geometry(parse, props)
        # Top layer: surface_max=10 → clay_max=3
        assert geom.soil_layers[0].top_elevation == 10.0
        assert geom.soil_layers[0].bottom_elevation == 3.0
        # Clay layer: top=3, bottom extended below
        assert geom.soil_layers[1].top_elevation == 3.0

    def test_surface_points_preserved(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [
            SoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0),
            SoilPropertyAssignment(name="Clay", gamma=19.0, phi=25.0),
        ]
        geom = build_slope_geometry(parse, props)
        assert len(geom.surface_points) == 4

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
            SoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0),
            SoilPropertyAssignment(name="Clay", gamma=19.0, phi=25.0),
        ]
        geom = build_slope_geometry(parse, props)
        assert geom.gwt_points is not None
        assert len(geom.gwt_points) == 4


class TestBuildMultiLayer:
    """Tests for multi-layer slope build."""

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
            SoilPropertyAssignment(name="Fill", gamma=17.0, phi=28.0),
            SoilPropertyAssignment(name="Sand", gamma=19.0, phi=35.0),
            SoilPropertyAssignment(name="Clay", gamma=18.0, cu=40.0, analysis_mode="undrained"),
            SoilPropertyAssignment(name="Rock", gamma=22.0, phi=45.0, c_prime=50.0),
        ]
        geom = build_slope_geometry(parse, props)
        # Fill (topmost, not a boundary name) + Sand + Clay + Rock = 4
        assert len(geom.soil_layers) == 4

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
            SoilPropertyAssignment(name="Fill", gamma=17.0, phi=28.0),
            SoilPropertyAssignment(name="Sand", gamma=19.0, phi=35.0),
            SoilPropertyAssignment(name="Clay", gamma=18.0, cu=40.0, analysis_mode="undrained"),
            SoilPropertyAssignment(name="Rock", gamma=22.0, phi=45.0, c_prime=50.0),
        ]
        geom = build_slope_geometry(parse, props)
        # Layers descend by top elevation
        for i in range(len(geom.soil_layers) - 1):
            assert (
                geom.soil_layers[i].top_elevation
                >= geom.soil_layers[i + 1].top_elevation
            )


class TestBuildNails:
    """Tests for nail conversion."""

    def test_nails_converted(self, nailed_slope_dxf):
        mapping = LayerMapping(
            surface="SURFACE",
            soil_boundaries={"SOIL_BOUNDARY": "Clay"},
            nails="NAILS",
        )
        parse = parse_dxf_geometry(
            filepath=nailed_slope_dxf, layer_mapping=mapping
        )
        props = [
            SoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0),
            SoilPropertyAssignment(name="Clay", gamma=19.0, phi=25.0),
        ]
        geom = build_slope_geometry(parse, props)
        assert geom.nails is not None
        assert len(geom.nails) == 3

    def test_nail_length(self, nailed_slope_dxf):
        mapping = LayerMapping(
            surface="SURFACE",
            nails="NAILS",
        )
        parse = parse_dxf_geometry(
            filepath=nailed_slope_dxf, layer_mapping=mapping
        )
        props = [SoilPropertyAssignment(name="Soil", gamma=18.0, phi=30.0)]
        geom = build_slope_geometry(parse, props)
        for nail in geom.nails:
            assert abs(nail.length - 6.0) < 0.1

    def test_nail_inclination(self, nailed_slope_dxf):
        mapping = LayerMapping(
            surface="SURFACE",
            nails="NAILS",
        )
        parse = parse_dxf_geometry(
            filepath=nailed_slope_dxf, layer_mapping=mapping
        )
        props = [SoilPropertyAssignment(name="Soil", gamma=18.0, phi=30.0)]
        geom = build_slope_geometry(parse, props)
        for nail in geom.nails:
            # Should be ~15° (created at 15° in fixture)
            assert abs(nail.inclination - 15.0) < 1.0

    def test_nail_defaults_applied(self, nailed_slope_dxf):
        mapping = LayerMapping(
            surface="SURFACE",
            nails="NAILS",
        )
        parse = parse_dxf_geometry(
            filepath=nailed_slope_dxf, layer_mapping=mapping
        )
        props = [SoilPropertyAssignment(name="Soil", gamma=18.0, phi=30.0)]
        defaults = {"bar_diameter": 32.0, "bond_stress": 150.0, "spacing_h": 2.0}
        geom = build_slope_geometry(parse, props, nail_defaults=defaults)
        for nail in geom.nails:
            assert nail.bar_diameter == 32.0
            assert nail.bond_stress == 150.0
            assert nail.spacing_h == 2.0


class TestBuildEdgeCases:
    """Tests for error handling in build_slope_geometry()."""

    def test_no_surface_points_raises(self):
        parse = DxfParseResult(surface_points=[])
        props = [SoilPropertyAssignment(name="Soil", gamma=18.0)]
        with pytest.raises(ValueError, match="No surface points"):
            build_slope_geometry(parse, props)

    def test_no_properties_raises(self, simple_slope_dxf):
        mapping = LayerMapping(surface="GROUND_SURFACE")
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        with pytest.raises(ValueError, match="At least one"):
            build_slope_geometry(parse, [])

    def test_missing_boundary_property_raises(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        # Only provide Surface, missing Clay
        props = [SoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0)]
        with pytest.raises(ValueError, match="No SoilPropertyAssignment"):
            build_slope_geometry(parse, props)

    def test_single_layer_no_boundary(self, simple_slope_dxf):
        mapping = LayerMapping(surface="GROUND_SURFACE")
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [SoilPropertyAssignment(name="Soil", gamma=18.0, phi=30.0)]
        geom = build_slope_geometry(parse, props)
        assert len(geom.soil_layers) == 1
        assert geom.soil_layers[0].name == "Soil"

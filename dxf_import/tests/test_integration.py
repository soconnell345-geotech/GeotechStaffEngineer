"""Integration tests for the full DXF import → slope stability pipeline.

Tests the complete workflow: discover → parse → build → analyze_slope.
"""

import math
import pytest

ezdxf = pytest.importorskip("ezdxf")

from dxf_import import (
    discover_layers,
    parse_dxf_geometry,
    build_slope_geometry,
    LayerMapping,
    SoilPropertyAssignment,
)
from slope_stability import analyze_slope, search_critical_surface


class TestFullPipeline:
    """End-to-end: DXF → SlopeGeometry → analyze_slope."""

    def test_discover_parse_build_analyze(self, simple_slope_dxf):
        # Step 1: Discover
        disc = discover_layers(filepath=simple_slope_dxf)
        assert disc.n_layers > 0

        # Step 2: Parse
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
            water_table="WATER_TABLE",
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        assert len(parse.surface_points) == 4

        # Step 3: Build
        props = [
            SoilPropertyAssignment(
                name="Surface", gamma=18.0, phi=30.0, c_prime=5.0
            ),
            SoilPropertyAssignment(
                name="Clay", gamma=19.0, phi=25.0, c_prime=10.0
            ),
        ]
        geom = build_slope_geometry(parse, props)

        # Step 4: Analyze
        result = analyze_slope(
            geom, xc=15.0, yc=18.0, radius=12.0, method="bishop"
        )
        assert result.FOS > 0
        assert result.method == "Bishop"

    def test_pipeline_produces_valid_fos(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [
            SoilPropertyAssignment(name="Surface", gamma=18.0, phi=32.0, c_prime=5.0),
            SoilPropertyAssignment(name="Clay", gamma=19.0, phi=28.0, c_prime=15.0),
        ]
        geom = build_slope_geometry(parse, props)
        result = analyze_slope(
            geom, xc=15.0, yc=15.0, radius=10.0, method="fellenius"
        )
        # FOS should be reasonable (0.5 < FOS < 10)
        assert 0.5 < result.FOS < 10.0

    def test_pipeline_with_search(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        props = [
            SoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0, c_prime=5.0),
            SoilPropertyAssignment(name="Clay", gamma=19.0, phi=25.0, c_prime=10.0),
        ]
        geom = build_slope_geometry(parse, props)
        search = search_critical_surface(
            geom, method="bishop", nx=3, ny=3
        )
        assert search.critical is not None
        assert search.critical.FOS > 0
        assert search.n_surfaces_evaluated > 0


class TestImperialPipeline:
    """End-to-end with imperial units."""

    def test_imperial_conversion_and_analyze(self, imperial_dxf):
        # Discover — should hint feet
        disc = discover_layers(filepath=imperial_dxf)
        assert disc.units_hint == "ft"

        # Parse in feet → converted to meters
        mapping = LayerMapping(
            surface="GROUND",
            soil_boundaries={"CLAY": "Clay"},
        )
        parse = parse_dxf_geometry(
            filepath=imperial_dxf, layer_mapping=mapping, units="ft"
        )

        # Surface first point was (0, 30ft) → ~(0, 9.144m)
        assert abs(parse.surface_points[0][1] - 30.0 * 0.3048) < 0.01

        # Build and analyze
        props = [
            SoilPropertyAssignment(name="Surface", gamma=18.0, phi=30.0, c_prime=5.0),
            SoilPropertyAssignment(name="Clay", gamma=19.0, phi=25.0, c_prime=10.0),
        ]
        geom = build_slope_geometry(parse, props)
        result = analyze_slope(
            geom, xc=10.0, yc=12.0, radius=8.0, method="bishop"
        )
        assert result.FOS > 0


class TestNailedPipeline:
    """End-to-end with nails."""

    def test_nailed_slope_analysis(self, nailed_slope_dxf):
        mapping = LayerMapping(
            surface="SURFACE",
            soil_boundaries={"SOIL_BOUNDARY": "Clay"},
            nails="NAILS",
        )
        parse = parse_dxf_geometry(
            filepath=nailed_slope_dxf, layer_mapping=mapping
        )
        props = [
            SoilPropertyAssignment(
                name="Surface", gamma=18.0, phi=28.0, c_prime=5.0
            ),
            SoilPropertyAssignment(
                name="Clay", gamma=19.0, phi=24.0, c_prime=8.0
            ),
        ]
        geom = build_slope_geometry(parse, props)

        # Verify nails exist
        assert geom.nails is not None
        assert len(geom.nails) == 3

        # Analyze — nails should be considered
        result = analyze_slope(
            geom, xc=12.0, yc=14.0, radius=10.0, method="bishop"
        )
        assert result.FOS > 0

    def test_nails_improve_fos(self, nailed_slope_dxf):
        """Nails should increase FOS compared to unreinforced slope."""
        # Parse with nails
        mapping_nails = LayerMapping(
            surface="SURFACE",
            soil_boundaries={"SOIL_BOUNDARY": "Clay"},
            nails="NAILS",
        )
        parse_nails = parse_dxf_geometry(
            filepath=nailed_slope_dxf, layer_mapping=mapping_nails
        )

        # Parse without nails
        mapping_plain = LayerMapping(
            surface="SURFACE",
            soil_boundaries={"SOIL_BOUNDARY": "Clay"},
        )
        parse_plain = parse_dxf_geometry(
            filepath=nailed_slope_dxf, layer_mapping=mapping_plain
        )

        props = [
            SoilPropertyAssignment(
                name="Surface", gamma=18.0, phi=28.0, c_prime=5.0
            ),
            SoilPropertyAssignment(
                name="Clay", gamma=19.0, phi=24.0, c_prime=8.0
            ),
        ]
        geom_nails = build_slope_geometry(parse_nails, props)
        geom_plain = build_slope_geometry(parse_plain, props)

        xc, yc, R = 12.0, 14.0, 10.0
        fos_nails = analyze_slope(
            geom_nails, xc=xc, yc=yc, radius=R, method="bishop"
        ).FOS
        fos_plain = analyze_slope(
            geom_plain, xc=xc, yc=yc, radius=R, method="bishop"
        ).FOS

        # Nails should improve stability (or at least not reduce it)
        assert fos_nails >= fos_plain


class TestDwgError:
    """Test DWG file rejection."""

    def test_dwg_file_gives_helpful_error(self, tmp_path):
        dwg_path = tmp_path / "model.dwg"
        dwg_path.write_bytes(b"fake")
        with pytest.raises(ValueError, match="DWG files are not supported"):
            discover_layers(filepath=str(dwg_path))

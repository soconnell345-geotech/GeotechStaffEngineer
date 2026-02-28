"""Tests for dxf_import.parser — LayerMapping and parse_dxf_geometry()."""

import math
import pytest

ezdxf = pytest.importorskip("ezdxf")

from dxf_import.parser import LayerMapping, parse_dxf_geometry


class TestParseBasicSurface:
    """Tests for surface extraction."""

    def test_surface_point_count(self, simple_slope_dxf):
        mapping = LayerMapping(surface="GROUND_SURFACE")
        result = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        assert len(result.surface_points) == 4

    def test_surface_sorted_by_x(self, simple_slope_dxf):
        mapping = LayerMapping(surface="GROUND_SURFACE")
        result = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        xs = [p[0] for p in result.surface_points]
        assert xs == sorted(xs)

    def test_surface_coordinates(self, simple_slope_dxf):
        mapping = LayerMapping(surface="GROUND_SURFACE")
        result = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        # First point: (0, 10)
        assert abs(result.surface_points[0][0] - 0.0) < 1e-4
        assert abs(result.surface_points[0][1] - 10.0) < 1e-4
        # Last point: (30, 5)
        assert abs(result.surface_points[-1][0] - 30.0) < 1e-4
        assert abs(result.surface_points[-1][1] - 5.0) < 1e-4


class TestParseBoundaries:
    """Tests for soil boundary extraction."""

    def test_single_boundary(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        result = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        assert "Clay" in result.boundary_profiles
        assert len(result.boundary_profiles["Clay"]) == 4

    def test_boundary_coordinates(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        result = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        pts = result.boundary_profiles["Clay"]
        # First point: (0, 3)
        assert abs(pts[0][0] - 0.0) < 1e-4
        assert abs(pts[0][1] - 3.0) < 1e-4

    def test_missing_boundary_layer_warns(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"NONEXISTENT": "MissingSoil"},
        )
        result = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        assert len(result.warnings) > 0
        assert "NONEXISTENT" in result.warnings[0]

    def test_multi_layer_boundaries(self, multi_layer_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={
                "FILL_SAND_BOUNDARY": "Sand",
                "SAND_CLAY_BOUNDARY": "Clay",
                "CLAY_ROCK_BOUNDARY": "Rock",
            },
        )
        result = parse_dxf_geometry(
            filepath=multi_layer_dxf, layer_mapping=mapping
        )
        assert len(result.boundary_profiles) == 3
        assert "Sand" in result.boundary_profiles
        assert "Clay" in result.boundary_profiles
        assert "Rock" in result.boundary_profiles


class TestParseGWT:
    """Tests for groundwater table extraction."""

    def test_gwt_extracted(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            water_table="WATER_TABLE",
        )
        result = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        assert result.gwt_points is not None
        assert len(result.gwt_points) == 4

    def test_gwt_none_when_not_mapped(self, simple_slope_dxf):
        mapping = LayerMapping(surface="GROUND_SURFACE")
        result = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        assert result.gwt_points is None


class TestParseNails:
    """Tests for nail LINE extraction."""

    def test_nail_count(self, nailed_slope_dxf):
        mapping = LayerMapping(
            surface="SURFACE",
            soil_boundaries={"SOIL_BOUNDARY": "Clay"},
            nails="NAILS",
        )
        result = parse_dxf_geometry(
            filepath=nailed_slope_dxf, layer_mapping=mapping
        )
        assert len(result.nail_lines) == 3

    def test_nail_head_is_leftmost(self, nailed_slope_dxf):
        mapping = LayerMapping(
            surface="SURFACE",
            nails="NAILS",
        )
        result = parse_dxf_geometry(
            filepath=nailed_slope_dxf, layer_mapping=mapping
        )
        for nl in result.nail_lines:
            assert nl["x_head"] < nl["x_tip"]

    def test_nail_geometry(self, nailed_slope_dxf):
        mapping = LayerMapping(
            surface="SURFACE",
            nails="NAILS",
        )
        result = parse_dxf_geometry(
            filepath=nailed_slope_dxf, layer_mapping=mapping
        )
        nl = result.nail_lines[0]
        # Head at x=8
        assert abs(nl["x_head"] - 8.0) < 0.1
        # Tip below head
        assert nl["z_tip"] < nl["z_head"]


class TestParseAnnotations:
    """Tests for TEXT/MTEXT extraction."""

    def test_text_annotations(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            annotations=["ANNOTATIONS"],
        )
        result = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        assert len(result.text_annotations) == 1
        assert result.text_annotations[0]["text"] == "Stiff Clay"


class TestParseUnits:
    """Tests for unit conversion during parsing."""

    def test_imperial_to_meters(self, imperial_dxf):
        mapping = LayerMapping(surface="GROUND")
        result = parse_dxf_geometry(
            filepath=imperial_dxf, layer_mapping=mapping, units="ft"
        )
        # First point: (0, 30ft) → (0, 9.144m)
        assert abs(result.surface_points[0][1] - 30.0 * 0.3048) < 1e-4
        assert result.units_used == "ft"

    def test_invalid_units_raises(self, simple_slope_dxf):
        mapping = LayerMapping(surface="GROUND_SURFACE")
        with pytest.raises(ValueError, match="Unknown units"):
            parse_dxf_geometry(
                filepath=simple_slope_dxf,
                layer_mapping=mapping,
                units="furlongs",
            )


class TestParseFlipY:
    """Tests for flip_y option."""

    def test_flip_y_negates(self, tmp_path):
        """Create a DXF with downward-positive Y and flip it."""
        doc = ezdxf.new()
        msp = doc.modelspace()
        # Y is negative (downward positive convention)
        msp.add_lwpolyline(
            [(0, -10), (10, -10), (20, -5)],
            dxfattribs={"layer": "SURFACE"},
        )
        path = tmp_path / "flip_test.dxf"
        doc.saveas(str(path))

        mapping = LayerMapping(surface="SURFACE")
        result = parse_dxf_geometry(
            filepath=str(path), layer_mapping=mapping, flip_y=True
        )
        # After flip: Y becomes positive
        assert result.surface_points[0][1] > 0


class TestParseLinesOnly:
    """Tests for surface from LINE entities (not polylines)."""

    def test_lines_merged_to_surface(self, lines_only_dxf):
        mapping = LayerMapping(surface="SURFACE")
        result = parse_dxf_geometry(
            filepath=lines_only_dxf, layer_mapping=mapping
        )
        # 3 LINE segments → should merge to 4 unique points
        assert len(result.surface_points) == 4
        # Sorted by x
        xs = [p[0] for p in result.surface_points]
        assert xs == sorted(xs)


class TestParseEdgeCases:
    """Tests for error handling and edge cases."""

    def test_missing_surface_layer_raises(self, simple_slope_dxf):
        mapping = LayerMapping(surface="DOES_NOT_EXIST")
        with pytest.raises(ValueError, match="Surface layer"):
            parse_dxf_geometry(
                filepath=simple_slope_dxf, layer_mapping=mapping
            )

    def test_no_layer_mapping_raises(self, simple_slope_dxf):
        with pytest.raises(ValueError, match="layer_mapping is required"):
            parse_dxf_geometry(filepath=simple_slope_dxf, layer_mapping=None)

    def test_empty_surface_name_raises(self, simple_slope_dxf):
        mapping = LayerMapping(surface="")
        with pytest.raises(ValueError, match="surface must be specified"):
            parse_dxf_geometry(
                filepath=simple_slope_dxf, layer_mapping=mapping
            )

    def test_no_input_raises(self):
        mapping = LayerMapping(surface="X")
        with pytest.raises(ValueError, match="Provide either"):
            parse_dxf_geometry(layer_mapping=mapping)

    def test_content_bytes_input(self, simple_slope_dxf_bytes):
        mapping = LayerMapping(surface="GROUND_SURFACE")
        result = parse_dxf_geometry(
            content=simple_slope_dxf_bytes, layer_mapping=mapping
        )
        assert len(result.surface_points) == 4

    def test_summary_string(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
        )
        result = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        s = result.summary()
        assert "DXF PARSE RESULTS" in s
        assert "Surface points: 4" in s

    def test_to_dict(self, simple_slope_dxf):
        mapping = LayerMapping(
            surface="GROUND_SURFACE",
            soil_boundaries={"CLAY_BOTTOM": "Clay"},
            water_table="WATER_TABLE",
        )
        result = parse_dxf_geometry(
            filepath=simple_slope_dxf, layer_mapping=mapping
        )
        d = result.to_dict()
        assert len(d["surface_points"]) == 4
        assert "Clay" in d["boundary_profiles"]
        assert "gwt_points" in d

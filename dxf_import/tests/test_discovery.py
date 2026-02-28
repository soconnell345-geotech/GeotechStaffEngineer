"""Tests for dxf_import.discovery — discover_layers()."""

import pytest

ezdxf = pytest.importorskip("ezdxf")

from dxf_import.discovery import discover_layers


class TestDiscoverLayers:
    """Tests for discover_layers()."""

    def test_simple_slope_layer_count(self, simple_slope_dxf):
        result = discover_layers(filepath=simple_slope_dxf)
        # GROUND_SURFACE, CLAY_BOTTOM, WATER_TABLE, ANNOTATIONS
        assert result.n_layers == 4

    def test_layer_names(self, simple_slope_dxf):
        result = discover_layers(filepath=simple_slope_dxf)
        names = {lyr.name for lyr in result.layers}
        assert "GROUND_SURFACE" in names
        assert "CLAY_BOTTOM" in names
        assert "WATER_TABLE" in names
        assert "ANNOTATIONS" in names

    def test_entity_counts(self, simple_slope_dxf):
        result = discover_layers(filepath=simple_slope_dxf)
        layer_map = {lyr.name: lyr for lyr in result.layers}
        # Each polyline layer has 1 entity
        assert layer_map["GROUND_SURFACE"].n_entities == 1
        assert layer_map["CLAY_BOTTOM"].n_entities == 1
        assert layer_map["WATER_TABLE"].n_entities == 1
        # Annotation has 1 text
        assert layer_map["ANNOTATIONS"].n_entities == 1

    def test_entity_types(self, simple_slope_dxf):
        result = discover_layers(filepath=simple_slope_dxf)
        layer_map = {lyr.name: lyr for lyr in result.layers}
        assert "LWPOLYLINE" in layer_map["GROUND_SURFACE"].entity_types
        assert "TEXT" in layer_map["ANNOTATIONS"].entity_types

    def test_sample_texts(self, simple_slope_dxf):
        result = discover_layers(filepath=simple_slope_dxf)
        layer_map = {lyr.name: lyr for lyr in result.layers}
        assert "Stiff Clay" in layer_map["ANNOTATIONS"].sample_texts

    def test_bbox(self, simple_slope_dxf):
        result = discover_layers(filepath=simple_slope_dxf)
        layer_map = {lyr.name: lyr for lyr in result.layers}
        bbox = layer_map["GROUND_SURFACE"].bbox
        assert bbox is not None
        x_min, y_min, x_max, y_max = bbox
        assert abs(x_min - 0.0) < 0.1
        assert abs(x_max - 30.0) < 0.1
        assert abs(y_min - 5.0) < 0.1
        assert abs(y_max - 10.0) < 0.1

    def test_total_entities(self, simple_slope_dxf):
        result = discover_layers(filepath=simple_slope_dxf)
        assert result.n_total_entities == 4

    def test_units_hint_meters_by_default(self, simple_slope_dxf):
        result = discover_layers(filepath=simple_slope_dxf)
        # ezdxf.new("R2010") defaults to $INSUNITS=6 (meters)
        assert result.units_hint == "m"

    def test_units_hint_feet(self, imperial_dxf):
        result = discover_layers(filepath=imperial_dxf)
        assert result.units_hint == "ft"

    def test_empty_dxf(self, empty_dxf):
        result = discover_layers(filepath=empty_dxf)
        assert result.n_layers == 0
        assert result.n_total_entities == 0
        assert result.layers == []

    def test_dwg_raises_error(self, tmp_path):
        dwg_path = tmp_path / "test.dwg"
        dwg_path.write_bytes(b"fake dwg content")
        with pytest.raises(ValueError, match="DWG files are not supported"):
            discover_layers(filepath=str(dwg_path))

    def test_summary_string(self, simple_slope_dxf):
        result = discover_layers(filepath=simple_slope_dxf)
        s = result.summary()
        assert "DXF LAYER DISCOVERY" in s
        assert "GROUND_SURFACE" in s
        assert "Total entities" in s

    def test_to_dict(self, simple_slope_dxf):
        result = discover_layers(filepath=simple_slope_dxf)
        d = result.to_dict()
        assert d["n_layers"] == 4
        assert len(d["layers"]) == 4
        assert all("name" in lyr for lyr in d["layers"])

    def test_content_bytes(self, simple_slope_dxf_bytes):
        result = discover_layers(content=simple_slope_dxf_bytes)
        assert result.filepath == "<bytes>"
        assert result.n_layers == 4

    def test_no_input_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            discover_layers()

    def test_multi_layer_discovery(self, multi_layer_dxf):
        result = discover_layers(filepath=multi_layer_dxf)
        assert result.n_layers == 4
        names = {lyr.name for lyr in result.layers}
        assert "GROUND_SURFACE" in names
        assert "FILL_SAND_BOUNDARY" in names
        assert "SAND_CLAY_BOUNDARY" in names
        assert "CLAY_ROCK_BOUNDARY" in names

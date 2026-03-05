"""
Tests for dxf_export module.

Covers:
- DxfExportResult (summary, to_dict)
- export_to_dxf (file creation, layers, entities)
- to_dxf_bytes (in-memory export)
- export_parse_result (DxfParseResult and PdfParseResult)
- Round-trip: export → re-import via dxf_import
- Edge cases (empty inputs, missing optional fields)
"""

import math
import os

import pytest

ezdxf = pytest.importorskip("ezdxf")

from dxf_export import DxfExportResult, export_to_dxf, to_dxf_bytes, export_parse_result
from dxf_export.writer import _INSUNITS


# ---------------------------------------------------------------------------
# Sample geometry fixtures
# ---------------------------------------------------------------------------

SURFACE_PTS = [(0, 10), (10, 10), (20, 5), (30, 5)]
BOUNDARY_PROFILES = {
    "Clay": [(0, 5), (10, 5), (20, 2), (30, 2)],
    "Sand": [(0, 8), (10, 8), (20, 4), (30, 4)],
}
GWT_PTS = [(0, 8), (30, 7)]
NAIL_LINES = [
    {"x_head": 8, "z_head": 8.5, "x_tip": 13.8, "z_tip": 6.95},
    {"x_head": 10, "z_head": 7.5, "x_tip": 15.8, "z_tip": 5.95},
]
TEXT_ANNOTATIONS = [
    {"text": "Sand", "x": 5, "y": 8},
    {"text": "Stiff Clay", "x": 5, "y": 3},
]


# ===================================================================
# DxfExportResult tests
# ===================================================================

class TestDxfExportResult:

    def test_summary_format(self):
        r = DxfExportResult(
            filepath="test.dxf",
            n_layers=3,
            n_entities=5,
            layers_created=["SURFACE", "BOUNDARY_Clay", "GWT"],
            surface_points_written=4,
            boundary_profiles_written=1,
            gwt_points_written=2,
        )
        s = r.summary()
        assert "DXF EXPORT RESULTS" in s
        assert "test.dxf" in s
        assert "Total entities: 5" in s
        assert "Surface points: 4" in s

    def test_summary_with_warnings(self):
        r = DxfExportResult(warnings=["Empty boundary skipped."])
        s = r.summary()
        assert "WARNING: Empty boundary skipped." in s

    def test_to_dict_keys(self):
        r = DxfExportResult(
            filepath="out.dxf",
            n_layers=2,
            n_entities=3,
            layers_created=["SURFACE", "GWT"],
            surface_points_written=4,
            boundary_profiles_written=0,
            gwt_points_written=2,
            nail_lines_written=0,
            text_annotations_written=0,
            warnings=[],
        )
        d = r.to_dict()
        expected_keys = {
            "filepath", "n_layers", "n_entities", "layers_created",
            "surface_points_written", "boundary_profiles_written",
            "gwt_points_written", "nail_lines_written",
            "text_annotations_written", "warnings",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_no_numpy(self):
        """Ensure to_dict returns plain Python types."""
        r = DxfExportResult(n_layers=3, n_entities=5)
        d = r.to_dict()
        assert isinstance(d["n_layers"], int)
        assert isinstance(d["n_entities"], int)
        assert isinstance(d["layers_created"], list)
        assert isinstance(d["warnings"], list)


# ===================================================================
# export_to_dxf tests
# ===================================================================

class TestExportToDxf:

    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "out.dxf")
        result = export_to_dxf(path, surface_points=SURFACE_PTS)
        assert os.path.isfile(path)
        assert result.filepath == path

    def test_ezdxf_can_read_output(self, tmp_path):
        path = str(tmp_path / "readable.dxf")
        export_to_dxf(path, surface_points=SURFACE_PTS)
        doc = ezdxf.readfile(path)
        assert doc is not None

    def test_surface_layer(self, tmp_path):
        path = str(tmp_path / "surface.dxf")
        result = export_to_dxf(path, surface_points=SURFACE_PTS)
        assert "SURFACE" in result.layers_created
        assert result.surface_points_written == 4
        assert result.n_entities >= 1

    def test_surface_entity_is_lwpolyline(self, tmp_path):
        path = str(tmp_path / "surface_entity.dxf")
        export_to_dxf(path, surface_points=SURFACE_PTS)
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        entities = list(msp.query("LWPOLYLINE[layer=='SURFACE']"))
        assert len(entities) == 1
        pts = list(entities[0].get_points(format="xy"))
        assert len(pts) == 4
        assert pts[0] == pytest.approx((0, 10), abs=0.01)

    def test_boundary_profiles(self, tmp_path):
        path = str(tmp_path / "boundaries.dxf")
        result = export_to_dxf(path, boundary_profiles=BOUNDARY_PROFILES)
        assert "BOUNDARY_Clay" in result.layers_created
        assert "BOUNDARY_Sand" in result.layers_created
        assert result.boundary_profiles_written == 2

    def test_boundary_entities(self, tmp_path):
        path = str(tmp_path / "boundary_ents.dxf")
        export_to_dxf(path, boundary_profiles=BOUNDARY_PROFILES)
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        clay_entities = list(msp.query("LWPOLYLINE[layer=='BOUNDARY_Clay']"))
        assert len(clay_entities) == 1
        pts = list(clay_entities[0].get_points(format="xy"))
        assert len(pts) == 4

    def test_gwt_polyline(self, tmp_path):
        path = str(tmp_path / "gwt.dxf")
        result = export_to_dxf(path, gwt_points=GWT_PTS)
        assert "GWT" in result.layers_created
        assert result.gwt_points_written == 2

    def test_gwt_entity(self, tmp_path):
        path = str(tmp_path / "gwt_entity.dxf")
        export_to_dxf(path, gwt_points=GWT_PTS)
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        gwt = list(msp.query("LWPOLYLINE[layer=='GWT']"))
        assert len(gwt) == 1

    def test_no_gwt_when_none(self, tmp_path):
        path = str(tmp_path / "no_gwt.dxf")
        result = export_to_dxf(path, surface_points=SURFACE_PTS, gwt_points=None)
        assert "GWT" not in result.layers_created
        assert result.gwt_points_written == 0

    def test_nail_lines(self, tmp_path):
        path = str(tmp_path / "nails.dxf")
        result = export_to_dxf(path, nail_lines=NAIL_LINES)
        assert "NAILS" in result.layers_created
        assert result.nail_lines_written == 2

    def test_nail_line_coordinates(self, tmp_path):
        path = str(tmp_path / "nail_coords.dxf")
        export_to_dxf(path, nail_lines=NAIL_LINES)
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        lines = list(msp.query("LINE[layer=='NAILS']"))
        assert len(lines) == 2
        # Check first nail start point
        start = lines[0].dxf.start
        assert start.x == pytest.approx(8, abs=0.01)
        assert start.y == pytest.approx(8.5, abs=0.01)

    def test_text_annotations(self, tmp_path):
        path = str(tmp_path / "text.dxf")
        result = export_to_dxf(path, text_annotations=TEXT_ANNOTATIONS)
        assert "ANNOTATIONS" in result.layers_created
        assert result.text_annotations_written == 2

    def test_text_insert_position(self, tmp_path):
        path = str(tmp_path / "text_pos.dxf")
        export_to_dxf(path, text_annotations=TEXT_ANNOTATIONS)
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        texts = list(msp.query("TEXT[layer=='ANNOTATIONS']"))
        assert len(texts) == 2

    def test_all_geometry(self, tmp_path):
        """Export with all geometry types at once."""
        path = str(tmp_path / "all.dxf")
        result = export_to_dxf(
            path,
            surface_points=SURFACE_PTS,
            boundary_profiles=BOUNDARY_PROFILES,
            gwt_points=GWT_PTS,
            nail_lines=NAIL_LINES,
            text_annotations=TEXT_ANNOTATIONS,
        )
        # 1 surface + 2 boundaries + 1 gwt + 2 nails + 2 texts = 8
        assert result.n_entities == 8
        # SURFACE + BOUNDARY_Clay + BOUNDARY_Sand + GWT + NAILS + ANNOTATIONS = 6
        assert result.n_layers == 6

    def test_empty_inputs(self, tmp_path):
        """Export with no geometry produces valid but empty DXF."""
        path = str(tmp_path / "empty.dxf")
        result = export_to_dxf(path)
        assert result.n_entities == 0
        assert result.n_layers == 0
        assert os.path.isfile(path)
        # Should be readable
        doc = ezdxf.readfile(path)
        assert doc is not None

    def test_empty_lists(self, tmp_path):
        """Explicit empty lists produce no entities."""
        path = str(tmp_path / "empty_lists.dxf")
        result = export_to_dxf(
            path,
            surface_points=[],
            boundary_profiles={},
            gwt_points=[],
            nail_lines=[],
            text_annotations=[],
        )
        assert result.n_entities == 0

    def test_units_meters(self, tmp_path):
        path = str(tmp_path / "meters.dxf")
        export_to_dxf(path, surface_points=SURFACE_PTS, units="m")
        doc = ezdxf.readfile(path)
        assert doc.header.get("$INSUNITS") == 6

    def test_units_feet(self, tmp_path):
        path = str(tmp_path / "feet.dxf")
        export_to_dxf(path, surface_points=SURFACE_PTS, units="ft")
        doc = ezdxf.readfile(path)
        assert doc.header.get("$INSUNITS") == 2

    def test_units_mm(self, tmp_path):
        path = str(tmp_path / "mm.dxf")
        export_to_dxf(path, surface_points=SURFACE_PTS, units="mm")
        doc = ezdxf.readfile(path)
        assert doc.header.get("$INSUNITS") == 4

    def test_empty_boundary_warning(self, tmp_path):
        """Boundary with empty points list produces a warning."""
        path = str(tmp_path / "warn.dxf")
        result = export_to_dxf(
            path,
            boundary_profiles={"Empty": []},
        )
        assert result.boundary_profiles_written == 0
        assert len(result.warnings) == 1
        assert "Empty" in result.warnings[0]

    def test_layer_colors(self, tmp_path):
        path = str(tmp_path / "colors.dxf")
        export_to_dxf(
            path,
            surface_points=SURFACE_PTS,
            gwt_points=GWT_PTS,
            boundary_profiles={"Clay": [(0, 5), (10, 5)]},
        )
        doc = ezdxf.readfile(path)
        assert doc.layers.get("SURFACE").color == 3
        assert doc.layers.get("GWT").color == 4
        assert doc.layers.get("BOUNDARY_Clay").color == 1

    def test_text_annotation_height(self, tmp_path):
        """Custom text height is written."""
        path = str(tmp_path / "height.dxf")
        export_to_dxf(
            path,
            text_annotations=[{"text": "Label", "x": 0, "y": 0, "height": 1.5}],
        )
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        texts = list(msp.query("TEXT[layer=='ANNOTATIONS']"))
        assert texts[0].dxf.height == pytest.approx(1.5)


# ===================================================================
# to_dxf_bytes tests
# ===================================================================

class TestToDxfBytes:

    def test_returns_bytes(self):
        data = to_dxf_bytes(surface_points=SURFACE_PTS)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_bytes_is_valid_dxf(self, tmp_path):
        """Write bytes to file and verify ezdxf can read it."""
        data = to_dxf_bytes(
            surface_points=SURFACE_PTS,
            boundary_profiles=BOUNDARY_PROFILES,
        )
        path = tmp_path / "from_bytes.dxf"
        path.write_bytes(data)
        doc = ezdxf.readfile(str(path))
        msp = doc.modelspace()
        surfaces = list(msp.query("LWPOLYLINE[layer=='SURFACE']"))
        assert len(surfaces) == 1

    def test_empty_bytes(self):
        """Empty geometry still produces valid DXF bytes."""
        data = to_dxf_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0


# ===================================================================
# export_parse_result tests
# ===================================================================

class TestExportParseResult:

    def test_from_dxf_parse_result(self, tmp_path):
        from dxf_import.results import DxfParseResult

        pr = DxfParseResult(
            surface_points=SURFACE_PTS,
            boundary_profiles={"Clay": [(0, 5), (10, 5), (20, 2), (30, 2)]},
            gwt_points=GWT_PTS,
            nail_lines=[
                {"x_head": 8, "z_head": 8.5, "x_tip": 14, "z_tip": 7},
            ],
            text_annotations=[{"text": "Sand", "x": 5, "y": 8, "layer": "ANN"}],
            units_used="m",
        )
        path = str(tmp_path / "from_parse.dxf")
        result = export_parse_result(pr, path)
        assert os.path.isfile(path)
        assert result.surface_points_written == 4
        assert result.boundary_profiles_written == 1
        assert result.gwt_points_written == 2
        assert result.nail_lines_written == 1
        assert result.text_annotations_written == 1

    def test_from_pdf_parse_result(self, tmp_path):
        from pdf_import.results import PdfParseResult

        pr = PdfParseResult(
            surface_points=SURFACE_PTS,
            boundary_profiles={"Clay": [(0, 5), (10, 5), (20, 2), (30, 2)]},
            gwt_points=GWT_PTS,
            text_annotations=[{"text": "Sand", "x": 5, "y": 8}],
            extraction_method="vector",
        )
        path = str(tmp_path / "from_pdf.dxf")
        result = export_parse_result(pr, path)
        assert os.path.isfile(path)
        assert result.surface_points_written == 4
        assert result.boundary_profiles_written == 1


# ===================================================================
# Round-trip tests: export → re-import via dxf_import
# ===================================================================

class TestRoundTrip:

    def _make_mapping(self, surface="SURFACE", boundaries=None,
                      water_table=None, nails=None):
        from dxf_import.parser import LayerMapping
        return LayerMapping(
            surface=surface,
            soil_boundaries=boundaries or {},
            water_table=water_table,
            nails=nails,
        )

    def test_surface_round_trip(self, tmp_path):
        """Export surface -> import -> compare points."""
        from dxf_import import parse_dxf_geometry

        path = str(tmp_path / "rt_surface.dxf")
        export_to_dxf(path, surface_points=SURFACE_PTS)

        mapping = self._make_mapping()
        imported = parse_dxf_geometry(path, layer_mapping=mapping)
        assert len(imported.surface_points) == len(SURFACE_PTS)
        for (ex, ez), (ix, iz) in zip(SURFACE_PTS, imported.surface_points):
            assert ex == pytest.approx(ix, abs=0.01)
            assert ez == pytest.approx(iz, abs=0.01)

    def test_boundary_round_trip(self, tmp_path):
        """Export boundary -> import -> compare."""
        from dxf_import import parse_dxf_geometry

        path = str(tmp_path / "rt_boundary.dxf")
        boundary = {"Clay": [(0, 5), (10, 5), (20, 2), (30, 2)]}
        export_to_dxf(
            path,
            surface_points=SURFACE_PTS,
            boundary_profiles=boundary,
        )
        mapping = self._make_mapping(
            boundaries={"BOUNDARY_Clay": "Clay"},
        )
        imported = parse_dxf_geometry(path, layer_mapping=mapping)
        assert "Clay" in imported.boundary_profiles
        pts = imported.boundary_profiles["Clay"]
        assert len(pts) == 4

    def test_gwt_round_trip(self, tmp_path):
        """Export GWT -> import -> compare."""
        from dxf_import import parse_dxf_geometry

        path = str(tmp_path / "rt_gwt.dxf")
        export_to_dxf(
            path,
            surface_points=SURFACE_PTS,
            gwt_points=GWT_PTS,
        )
        mapping = self._make_mapping(water_table="GWT")
        imported = parse_dxf_geometry(path, layer_mapping=mapping)
        assert imported.gwt_points is not None
        assert len(imported.gwt_points) == len(GWT_PTS)

    def test_full_round_trip(self, tmp_path):
        """Export all geometry -> import -> verify counts."""
        from dxf_import import parse_dxf_geometry

        path = str(tmp_path / "rt_full.dxf")
        export_to_dxf(
            path,
            surface_points=SURFACE_PTS,
            boundary_profiles=BOUNDARY_PROFILES,
            gwt_points=GWT_PTS,
            nail_lines=NAIL_LINES,
            text_annotations=TEXT_ANNOTATIONS,
        )
        mapping = self._make_mapping(
            boundaries={
                "BOUNDARY_Clay": "Clay",
                "BOUNDARY_Sand": "Sand",
            },
            water_table="GWT",
            nails="NAILS",
        )
        imported = parse_dxf_geometry(path, layer_mapping=mapping)
        assert len(imported.surface_points) == 4
        assert len(imported.boundary_profiles) == 2
        assert imported.gwt_points is not None
        assert len(imported.nail_lines) == 2

    def test_dxf_parse_result_round_trip(self, tmp_path):
        """DxfParseResult -> export -> re-import -> compare."""
        from dxf_import.results import DxfParseResult
        from dxf_import import parse_dxf_geometry

        original = DxfParseResult(
            surface_points=SURFACE_PTS,
            boundary_profiles={"Clay": [(0, 5), (10, 5), (20, 2), (30, 2)]},
            gwt_points=GWT_PTS,
            nail_lines=[],
            text_annotations=[],
            units_used="m",
        )
        path = str(tmp_path / "rt_parse.dxf")
        export_parse_result(original, path)

        mapping = self._make_mapping(
            boundaries={"BOUNDARY_Clay": "Clay"},
            water_table="GWT",
        )
        imported = parse_dxf_geometry(path, layer_mapping=mapping)
        assert len(imported.surface_points) == len(original.surface_points)
        assert "Clay" in imported.boundary_profiles

    def test_pdf_parse_result_round_trip(self, tmp_path):
        """PdfParseResult -> export -> re-import -> verify readable."""
        from pdf_import.results import PdfParseResult

        pdf_result = PdfParseResult(
            surface_points=[(0, 10), (20, 10), (40, 5)],
            boundary_profiles={"Silt": [(0, 6), (20, 6), (40, 3)]},
            gwt_points=[(0, 9), (40, 8)],
            extraction_method="vision",
            confidence=0.85,
        )
        path = str(tmp_path / "rt_pdf.dxf")
        export_parse_result(pdf_result, path)

        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        surfaces = list(msp.query("LWPOLYLINE[layer=='SURFACE']"))
        assert len(surfaces) == 1
        gwts = list(msp.query("LWPOLYLINE[layer=='GWT']"))
        assert len(gwts) == 1

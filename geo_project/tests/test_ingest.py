"""Ingest tests: synthetic DXF (built with ezdxf), points, vision-draft."""

import math

import pytest

from geo_project.ingest import (
    discover_dxf,
    from_dxf,
    from_points,
    from_vision_draft,
)
from geo_project.validate import has_errors, validate

ezdxf = pytest.importorskip("ezdxf")


# ---------------------------------------------------------------------------
# Synthetic DXF fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def dxf_path(tmp_path):
    """A synthetic 2-boundary slope drawing with GWT and one nail line."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for lname in ("TOPO", "STRATUM_CLAY", "STRATUM_TILL", "GWT", "NAILS"):
        doc.layers.add(lname)
    msp.add_lwpolyline([(0, 10), (10, 10), (30, 0), (45, 0)],
                       dxfattribs={"layer": "TOPO"})
    msp.add_lwpolyline([(0, 4), (45, 3)],
                       dxfattribs={"layer": "STRATUM_CLAY"})
    msp.add_lwpolyline([(0, -4), (45, -4)],
                       dxfattribs={"layer": "STRATUM_TILL"})
    msp.add_lwpolyline([(0, 6), (45, -1)], dxfattribs={"layer": "GWT"})
    msp.add_line((25, 2.5), (25 + 8 * math.cos(math.radians(15)),
                             2.5 - 8 * math.sin(math.radians(15))),
                 dxfattribs={"layer": "NAILS"})
    path = tmp_path / "slope.dxf"
    doc.saveas(path)
    return str(path)


def test_discover_dxf_inventory(dxf_path):
    inv = discover_dxf(dxf_path)
    names = {ly["name"] for ly in inv["layers"]}
    assert {"TOPO", "STRATUM_CLAY", "STRATUM_TILL", "GWT", "NAILS"} <= names
    topo = next(ly for ly in inv["layers"] if ly["name"] == "TOPO")
    assert topo["entity_types"].get("LWPOLYLINE") == 1
    nails = next(ly for ly in inv["layers"] if ly["name"] == "NAILS")
    assert nails["entity_types"].get("LINE") == 1


def test_from_dxf_builds_project(dxf_path):
    p = from_dxf(dxf_path, layer_mapping={
        "surface": "TOPO",
        "soil_boundaries": {"STRATUM_CLAY": "Stiff clay",
                            "STRATUM_TILL": "Till"},
        "water_table": "GWT",
        "nails": "NAILS",
    })
    assert p.geometry.provenance == "dxf"
    assert p.geometry.surface_points == [(0, 10), (10, 10), (30, 0), (45, 0)]
    # Boundaries named by SOIL name, ordered top-down.
    assert set(p.geometry.layer_boundaries) == {"Stiff clay", "Till"}
    # Layers: Surface (above clay boundary), Stiff clay, Till.
    assert [L.name for L in p.stratigraphy] == ["Surface", "Stiff clay",
                                                "Till"]
    assert p.stratigraphy[0].bottom_boundary == "Stiff clay"
    assert p.stratigraphy[1].bottom_boundary == "Till"
    # Bottom layer flat bottom: lowest point (-4) - 5.
    assert p.stratigraphy[2].bottom_elevation == pytest.approx(-9.0)
    # Elevation chain resolves.
    assert p.layer_top(0) == 10.0
    assert p.layer_bottom(0) == 4.0   # max z of clay boundary
    assert p.layer_bottom(1) == -4.0
    # GWT + nail carried over.
    assert p.water.gwt_points == [(0, 6), (45, -1)]
    assert len(p.reinforcement.nails) == 1
    n = p.reinforcement.nails[0]
    assert n.x_head == pytest.approx(25.0)
    assert n.length == pytest.approx(8.0, rel=1e-6)
    assert n.inclination == pytest.approx(15.0, rel=1e-6)
    # Geometry-only ingest validates clean (no analyses yet → no MAT errors).
    assert not has_errors(validate(p))
    # The materials-are-empty fact is on the ledger.
    assert any("material" in a.field for a in p.assumptions)


def test_from_dxf_requires_mapping(dxf_path):
    with pytest.raises(ValueError, match="discover_dxf"):
        from_dxf(dxf_path)


def test_from_dxf_units_conversion(tmp_path):
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.add("TOPO")
    # Same shape in millimeters.
    msp.add_lwpolyline([(0, 10000), (45000, 0)], dxfattribs={"layer": "TOPO"})
    path = tmp_path / "mm.dxf"
    doc.saveas(path)
    p = from_dxf(str(path), layer_mapping={"surface": "TOPO"}, units="mm")
    assert p.geometry.surface_points == [(0.0, 10.0), (45.0, 0.0)]


# ---------------------------------------------------------------------------
# from_points
# ---------------------------------------------------------------------------

def test_from_points_single_layer():
    p = from_points([(0, 5), (20, 0)], section_bottom=-6.0)
    assert p.geometry.provenance == "user"
    assert [L.name for L in p.stratigraphy] == ["Soil"]
    assert p.stratigraphy[0].bottom_elevation == -6.0
    assert not has_errors(validate(p))


def test_from_points_with_boundaries_and_names():
    p = from_points(
        [(0, 5), (20, 0), (30, 0)],
        layer_boundaries={"clay_top": [(0, 1.0), (30, 0.5)]},
        layer_names=["Fill", "Clay"],
        gwt_points=[(0, 3.0), (30, -1.0)],
    )
    assert [L.name for L in p.stratigraphy] == ["Fill", "Clay"]
    assert p.stratigraphy[0].bottom_boundary == "clay_top"
    assert p.water.gwt_points == [(0.0, 3.0), (30.0, -1.0)]
    assert not has_errors(validate(p))


def test_from_points_wrong_name_count():
    with pytest.raises(ValueError, match="layer_names"):
        from_points([(0, 5), (20, 0)], layer_names=["a", "b", "c"])


# ---------------------------------------------------------------------------
# Vision draft: quarantined until confirmed
# ---------------------------------------------------------------------------

def test_vision_draft_blocks_then_unblocks():
    draft = {
        "surface_points": [(0, 8), (10, 8), (26, 0), (36, 0)],
        "boundary_profiles": {"Clay": [(0, 2.0), (36, 1.5)]},
        "gwt_points": [(0, 4.0), (36, -1.0)],
    }
    p = from_vision_draft(draft)
    assert p.geometry.provenance == "vision_draft"
    issues = validate(p)
    assert has_errors(issues)
    assert any(i.code == "GEOM007" for i in issues)
    # The ledger carries the vision warning.
    assert any("vision" in (a.source + a.value if isinstance(a.value, str)
                            else a.source).lower()
               for a in p.assumptions)
    # Human confirms the geometry stage → block lifts.
    p.confirmations.geometry = True
    assert not has_errors(validate(p))


def test_vision_draft_from_pdf_parse_result():
    from pdf_import.results import PdfParseResult
    draft = PdfParseResult(
        surface_points=[(0, 8), (26, 0), (36, 0)],
        boundary_profiles={"Clay": [(0, 2.0), (36, 1.5)]},
        extraction_method="vision", confidence=0.5,
    )
    p = from_vision_draft(draft)
    assert p.geometry.provenance == "vision_draft"
    assert any(i.code == "GEOM007" for i in validate(p))


def test_vision_draft_rejects_garbage():
    with pytest.raises(TypeError):
        from_vision_draft(42)

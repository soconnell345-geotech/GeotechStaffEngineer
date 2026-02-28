"""
Programmatic DXF fixtures for dxf_import tests.

All DXF test files are created in-memory using ezdxf — no external files needed.
Each fixture writes a DXF file to tmp_path and returns its path.
"""

import math
import pytest

ezdxf = pytest.importorskip("ezdxf")


@pytest.fixture
def simple_slope_dxf(tmp_path):
    """DXF with GROUND_SURFACE polyline + CLAY_BOTTOM boundary + WATER_TABLE.

    Surface: (0,10) → (10,10) → (20,5) → (30,5)
    Clay bottom: (0,3) → (10,3) → (20,2) → (30,2)
    GWT: (0,8) → (10,8) → (20,4) → (30,4)
    """
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    # Surface polyline
    msp.add_lwpolyline(
        [(0, 10), (10, 10), (20, 5), (30, 5)],
        dxfattribs={"layer": "GROUND_SURFACE"},
    )

    # Clay bottom boundary
    msp.add_lwpolyline(
        [(0, 3), (10, 3), (20, 2), (30, 2)],
        dxfattribs={"layer": "CLAY_BOTTOM"},
    )

    # Water table
    msp.add_lwpolyline(
        [(0, 8), (10, 8), (20, 4), (30, 4)],
        dxfattribs={"layer": "WATER_TABLE"},
    )

    # A text annotation
    msp.add_text(
        "Stiff Clay",
        dxfattribs={"layer": "ANNOTATIONS", "insert": (5, 6)},
    )

    path = tmp_path / "simple_slope.dxf"
    doc.saveas(str(path))
    return str(path)


@pytest.fixture
def nailed_slope_dxf(tmp_path):
    """DXF with surface + boundary + nail LINE entities.

    Surface: (0,10) → (5,10) → (15,5) → (25,5)
    Boundary: (0,4) → (5,4) → (15,2) → (25,2)
    Nails: 3 lines from face into slope at ~15° below horizontal
    """
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    msp.add_lwpolyline(
        [(0, 10), (5, 10), (15, 5), (25, 5)],
        dxfattribs={"layer": "SURFACE"},
    )

    msp.add_lwpolyline(
        [(0, 4), (5, 4), (15, 2), (25, 2)],
        dxfattribs={"layer": "SOIL_BOUNDARY"},
    )

    # Nails: LINE from head (face) to tip (into slope)
    # ~15° below horizontal, 6m long
    nail_length = 6.0
    angle_deg = 15.0
    angle_rad = math.radians(angle_deg)
    dx = nail_length * math.cos(angle_rad)
    dz = nail_length * math.sin(angle_rad)

    nail_heads = [(8, 8.5), (10, 7.5), (12, 6.5)]
    for xh, zh in nail_heads:
        msp.add_line(
            start=(xh, zh),
            end=(xh + dx, zh - dz),
            dxfattribs={"layer": "NAILS"},
        )

    path = tmp_path / "nailed_slope.dxf"
    doc.saveas(str(path))
    return str(path)


@pytest.fixture
def imperial_dxf(tmp_path):
    """DXF in feet with $INSUNITS=2 (feet).

    Surface: (0,30) → (30,30) → (60,15) → (100,15) [feet]
    Boundary: (0,10) → (30,10) → (60,5) → (100,5) [feet]
    """
    doc = ezdxf.new(dxfversion="R2010")
    doc.header["$INSUNITS"] = 2  # feet

    msp = doc.modelspace()

    msp.add_lwpolyline(
        [(0, 30), (30, 30), (60, 15), (100, 15)],
        dxfattribs={"layer": "GROUND"},
    )

    msp.add_lwpolyline(
        [(0, 10), (30, 10), (60, 5), (100, 5)],
        dxfattribs={"layer": "CLAY"},
    )

    path = tmp_path / "imperial_slope.dxf"
    doc.saveas(str(path))
    return str(path)


@pytest.fixture
def multi_layer_dxf(tmp_path):
    """DXF with 3 soil boundaries (4 layers total).

    Surface:    (0,20) → (20,20) → (40,10) → (60,10)
    Boundary 1: (0,16) → (20,16) → (40, 8) → (60, 8)  (Fill/Sand interface)
    Boundary 2: (0,12) → (20,12) → (40, 6) → (60, 6)  (Sand/Clay interface)
    Boundary 3: (0, 5) → (20, 5) → (40, 3) → (60, 3)  (Clay/Rock interface)
    """
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    msp.add_lwpolyline(
        [(0, 20), (20, 20), (40, 10), (60, 10)],
        dxfattribs={"layer": "GROUND_SURFACE"},
    )
    msp.add_lwpolyline(
        [(0, 16), (20, 16), (40, 8), (60, 8)],
        dxfattribs={"layer": "FILL_SAND_BOUNDARY"},
    )
    msp.add_lwpolyline(
        [(0, 12), (20, 12), (40, 6), (60, 6)],
        dxfattribs={"layer": "SAND_CLAY_BOUNDARY"},
    )
    msp.add_lwpolyline(
        [(0, 5), (20, 5), (40, 3), (60, 3)],
        dxfattribs={"layer": "CLAY_ROCK_BOUNDARY"},
    )

    path = tmp_path / "multi_layer.dxf"
    doc.saveas(str(path))
    return str(path)


@pytest.fixture
def empty_dxf(tmp_path):
    """DXF file with no entities."""
    doc = ezdxf.new(dxfversion="R2010")
    path = tmp_path / "empty.dxf"
    doc.saveas(str(path))
    return str(path)


@pytest.fixture
def lines_only_dxf(tmp_path):
    """DXF with surface defined by individual LINE segments (not polyline).

    Surface: (0,10) → (10,10) → (20,5) → (30,5) as 3 LINE entities.
    """
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    segments = [
        ((0, 10), (10, 10)),
        ((10, 10), (20, 5)),
        ((20, 5), (30, 5)),
    ]
    for start, end in segments:
        msp.add_line(start=start, end=end, dxfattribs={"layer": "SURFACE"})

    path = tmp_path / "lines_surface.dxf"
    doc.saveas(str(path))
    return str(path)


@pytest.fixture
def simple_slope_dxf_bytes(simple_slope_dxf):
    """Return bytes content of the simple_slope DXF file."""
    with open(simple_slope_dxf, "rb") as f:
        return f.read()

"""
DXF Import Module for Slope Stability Analysis

Imports CAD cross-sections (DXF format) and translates them into
SlopeGeometry and soil stratigraphy for slope stability analysis.

Three-step workflow:
    1. discover_layers() — Inventory DXF layers for user mapping
    2. parse_dxf_geometry() — Extract coordinates from mapped layers
    3. build_slope_geometry() — Assemble SlopeGeometry with soil properties

DXF provides geometry only; soil strength parameters (gamma, phi, c')
must always come from the user.

Requires: ezdxf >= 1.4 (optional dependency)
"""

from dxf_import.results import (
    LayerInfo,
    DxfDiscoveryResult,
    DxfParseResult,
)
from dxf_import.discovery import discover_layers
from dxf_import.parser import LayerMapping, parse_dxf_geometry
from dxf_import.converter import SoilPropertyAssignment, build_slope_geometry

__all__ = [
    'discover_layers',
    'parse_dxf_geometry',
    'build_slope_geometry',
    'LayerMapping',
    'SoilPropertyAssignment',
    'LayerInfo',
    'DxfDiscoveryResult',
    'DxfParseResult',
]

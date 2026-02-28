"""
DXF layer discovery — inventory all layers with entity counts and types.

Reads a DXF file and produces a DxfDiscoveryResult listing every layer
that contains entities, along with entity type counts, sample text content,
and bounding box estimates. This is step 1 of the discover-then-parse
workflow: the user inspects layer names to create a LayerMapping.
"""

import os
import tempfile
from typing import Optional

from dxf_import.results import DxfDiscoveryResult, LayerInfo
from dxf_import.units import detect_units_from_header

try:
    import ezdxf
except ImportError:
    ezdxf = None

_EZDXF_INSTALL_MSG = (
    "ezdxf is required for DXF import. Install it with: "
    "pip install ezdxf>=1.4"
)

# Entity types that carry text content
_TEXT_TYPES = {"TEXT", "MTEXT"}

# Entity types we extract coordinates from
_GEOMETRY_TYPES = {
    "LINE", "LWPOLYLINE", "POLYLINE", "SPLINE", "ARC", "CIRCLE", "POINT",
}


def discover_layers(
    filepath: str = None,
    content: bytes = None,
) -> DxfDiscoveryResult:
    """Read a DXF file and inventory all layers.

    Parameters
    ----------
    filepath : str, optional
        Path to a DXF file on disk.
    content : bytes, optional
        Raw DXF file content (for GUI upload or agent base64 decode).
        Exactly one of filepath or content must be provided.

    Returns
    -------
    DxfDiscoveryResult
        Layer inventory with entity counts, types, sample texts, and bbox.

    Raises
    ------
    ImportError
        If ezdxf is not installed.
    ValueError
        If the file is a DWG or neither filepath nor content is provided.
    """
    if ezdxf is None:
        raise ImportError(_EZDXF_INSTALL_MSG)

    if filepath is None and content is None:
        raise ValueError("Provide either filepath or content")

    # Check for DWG files
    if filepath is not None and filepath.lower().endswith(".dwg"):
        raise ValueError(
            "DWG files are not supported. Convert to DXF using the free "
            "ODA File Converter (https://www.opendesign.com/guestfiles/oda_file_converter) "
            "or save as DXF from AutoCAD."
        )

    warnings = []

    # Read the DXF document
    if content is not None:
        # Write to temp file — ezdxf.readfile handles encoding detection
        with tempfile.NamedTemporaryFile(
            suffix=".dxf", delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            doc = ezdxf.readfile(tmp_path)
        finally:
            os.unlink(tmp_path)
        source_name = "<bytes>"
    else:
        doc = ezdxf.readfile(filepath)
        source_name = filepath

    # Detect units from header
    units_hint = detect_units_from_header(doc)

    # Iterate all entities in modelspace
    msp = doc.modelspace()
    layer_data = {}  # layer_name -> {types: Counter, texts: [], coords: []}

    for entity in msp:
        layer_name = entity.dxf.layer
        if layer_name not in layer_data:
            layer_data[layer_name] = {
                "types": {},
                "texts": [],
                "coords": [],
            }
        ld = layer_data[layer_name]
        etype = entity.dxftype()
        ld["types"][etype] = ld["types"].get(etype, 0) + 1

        # Collect text content
        if etype in _TEXT_TYPES:
            text = ""
            if etype == "TEXT":
                text = entity.dxf.text
            elif etype == "MTEXT":
                text = entity.text  # ezdxf resolves MTEXT content
            if text and len(ld["texts"]) < 5:
                ld["texts"].append(text.strip())

        # Collect coordinates for bbox
        _collect_coords(entity, etype, ld["coords"])

    # Build LayerInfo list
    layers = []
    total_entities = 0
    for name, ld in sorted(layer_data.items()):
        n_ents = sum(ld["types"].values())
        total_entities += n_ents
        bbox = _compute_bbox(ld["coords"]) if ld["coords"] else None
        layers.append(LayerInfo(
            name=name,
            n_entities=n_ents,
            entity_types=dict(ld["types"]),
            sample_texts=ld["texts"],
            bbox=bbox,
        ))

    return DxfDiscoveryResult(
        filepath=source_name,
        n_layers=len(layers),
        layers=layers,
        units_hint=units_hint,
        n_total_entities=total_entities,
        warnings=warnings,
    )


def _collect_coords(entity, etype, coords_list):
    """Extract representative coordinates from an entity for bbox calculation."""
    try:
        if etype == "LINE":
            s = entity.dxf.start
            e = entity.dxf.end
            coords_list.append((s.x, s.y))
            coords_list.append((e.x, e.y))
        elif etype == "LWPOLYLINE":
            for x, y, *_ in entity.get_points(format="xy"):
                coords_list.append((x, y))
        elif etype == "POLYLINE":
            for v in entity.vertices:
                coords_list.append((v.dxf.location.x, v.dxf.location.y))
        elif etype == "SPLINE":
            for pt in entity.control_points:
                coords_list.append((pt[0], pt[1]))
        elif etype == "CIRCLE" or etype == "ARC":
            c = entity.dxf.center
            r = entity.dxf.radius
            coords_list.append((c.x - r, c.y - r))
            coords_list.append((c.x + r, c.y + r))
        elif etype == "POINT":
            loc = entity.dxf.location
            coords_list.append((loc.x, loc.y))
        elif etype in _TEXT_TYPES:
            if hasattr(entity.dxf, "insert"):
                ins = entity.dxf.insert
                coords_list.append((ins.x, ins.y))
    except Exception:
        pass  # Skip entities with missing geometry data


def _compute_bbox(coords):
    """Compute (x_min, y_min, x_max, y_max) from a list of (x, y) tuples."""
    if not coords:
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (min(xs), min(ys), max(xs), max(ys))

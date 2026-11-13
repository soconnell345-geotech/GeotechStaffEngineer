"""
DXF geometry extraction — parse mapped layers into slope geometry coordinates.

Step 2 of the discover-then-parse workflow. The user provides a LayerMapping
that assigns DXF layer names to geometric roles (surface, boundaries, GWT, nails).
This module extracts polyline coordinates from those layers and converts units.

Supported entity types:
- LWPOLYLINE, POLYLINE → sorted (x, z) vertex lists
- LINE → individual segments (merged and sorted for surface/boundaries)
- SPLINE → flattened to polyline via ezdxf flattening()
- TEXT/MTEXT → annotation text with insertion point
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dxf_import.results import DxfParseResult
from dxf_import.units import (
    UNIT_FACTORS, convert_coords, detect_units_from_header,
)

try:
    import ezdxf
except ImportError:
    ezdxf = None

_EZDXF_INSTALL_MSG = (
    "ezdxf is required for DXF import. Install it with: "
    "pip install ezdxf>=1.4"
)


@dataclass
class LayerMapping:
    """User-provided mapping from DXF layers to slope geometry roles.

    Parameters
    ----------
    surface : str
        DXF layer name for the ground surface polyline.
    soil_boundaries : dict
        {dxf_layer_name: soil_name} for each soil boundary.
        Each boundary is the TOP of the soil it names (that soil lies
        below it, down to the next boundary), and the soil between the
        ground surface and the highest boundary takes the soil property
        named "Surface", or else the first property that no boundary
        names (``converter.build_slope_geometry``).
    water_table : str, optional
        DXF layer name for the groundwater table profile.
    nails : str, optional
        DXF layer name for soil nail LINE entities.
    annotations : list of str, optional
        DXF layer names for TEXT/MTEXT labels.
    """
    surface: str = ""
    soil_boundaries: Dict[str, str] = field(default_factory=dict)
    water_table: Optional[str] = None
    nails: Optional[str] = None
    annotations: Optional[List[str]] = None


def parse_dxf_geometry(
    filepath: str = None,
    content: bytes = None,
    layer_mapping: LayerMapping = None,
    units: Optional[str] = None,
    flip_y: bool = False,
    ignore_header_units: bool = False,
) -> DxfParseResult:
    """Extract slope geometry from mapped DXF layers.

    Parameters
    ----------
    filepath : str, optional
        Path to DXF file on disk.
    content : bytes, optional
        Raw DXF file content.
    layer_mapping : LayerMapping
        Mapping from DXF layers to geometric roles.
    units : str, optional
        Drawing units ('m', 'mm', 'cm', 'ft', 'in'). Left out, the units
        the drawing's own header states ($INSUNITS) are used; a drawing
        whose header states none is refused until ``units`` is given.
        Given, they are compared with the header: a difference is refused
        unless ``ignore_header_units`` is True. There is no silent default.
    flip_y : bool
        If True, negate Y values (for drawings with downward-positive Y).
    ignore_header_units : bool
        If True, ``units`` is used even where the drawing's header states
        other units (a header left at a template's value is common); the
        result's warnings then name both.

    Returns
    -------
    DxfParseResult
        Extracted coordinates in meters: surface, boundaries, water table,
        nails and text annotation positions alike. ``units_used`` is what
        the coordinates were converted from, ``units_source`` where that
        came from and ``header_units`` what the drawing's header states.

    Raises
    ------
    ImportError
        If ezdxf is not installed.
    ValueError
        If layer_mapping is None, units unknown, units not stated by the
        caller or the drawing, units differing from the drawing's header
        (without ``ignore_header_units``), or required layers missing.
    """
    if ezdxf is None:
        raise ImportError(_EZDXF_INSTALL_MSG)

    if layer_mapping is None:
        raise ValueError("layer_mapping is required")

    if not layer_mapping.surface:
        raise ValueError("layer_mapping.surface must be specified")

    if units is not None and units not in UNIT_FACTORS:
        raise ValueError(
            f"Unknown units '{units}'. Supported: {sorted(UNIT_FACTORS.keys())}"
        )

    if filepath is None and content is None:
        raise ValueError("Provide either filepath or content")

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
    else:
        doc = ezdxf.readfile(filepath)

    msp = doc.modelspace()
    warnings = []

    # --- Units: the caller's against the drawing's own header ---
    units, units_source, header_units = _resolve_units(
        doc, units, ignore_header_units, warnings)

    # Index entities by layer
    layer_entities = {}
    for entity in msp:
        lname = entity.dxf.layer
        if lname not in layer_entities:
            layer_entities[lname] = []
        layer_entities[lname].append(entity)

    # --- Surface ---
    if layer_mapping.surface not in layer_entities:
        raise ValueError(
            f"Surface layer '{layer_mapping.surface}' not found in DXF. "
            f"Available layers: {sorted(layer_entities.keys())}"
        )
    surface_raw = _extract_polyline_coords(
        layer_entities[layer_mapping.surface], flip_y
    )
    if not surface_raw:
        raise ValueError(
            f"No polyline/line geometry found on surface layer "
            f"'{layer_mapping.surface}'"
        )
    surface_points = convert_coords(surface_raw, units, "m")

    # --- Soil boundaries ---
    boundary_profiles = {}
    for dxf_layer, soil_name in layer_mapping.soil_boundaries.items():
        if dxf_layer not in layer_entities:
            warnings.append(
                f"Boundary layer '{dxf_layer}' (soil '{soil_name}') "
                f"not found in DXF — skipped"
            )
            continue
        raw = _extract_polyline_coords(layer_entities[dxf_layer], flip_y)
        if not raw:
            warnings.append(
                f"No geometry on boundary layer '{dxf_layer}' — skipped"
            )
            continue
        boundary_profiles[soil_name] = convert_coords(raw, units, "m")

    # --- GWT ---
    gwt_points = None
    if layer_mapping.water_table is not None:
        if layer_mapping.water_table not in layer_entities:
            warnings.append(
                f"Water table layer '{layer_mapping.water_table}' "
                f"not found in DXF — skipped"
            )
        else:
            raw = _extract_polyline_coords(
                layer_entities[layer_mapping.water_table], flip_y
            )
            if raw:
                gwt_points = convert_coords(raw, units, "m")
            else:
                warnings.append("No geometry on water table layer — skipped")

    # --- Nails ---
    nail_lines = []
    if layer_mapping.nails is not None:
        if layer_mapping.nails not in layer_entities:
            warnings.append(
                f"Nail layer '{layer_mapping.nails}' not found in DXF — skipped"
            )
        else:
            nail_entities = layer_entities[layer_mapping.nails]
            nail_lines = _extract_nail_lines(nail_entities, flip_y, units)
            not_lines = {}
            for ent in nail_entities:
                if ent.dxftype() != "LINE":
                    not_lines[ent.dxftype()] = (
                        not_lines.get(ent.dxftype(), 0) + 1)
            if not_lines:
                warnings.append(
                    f"Nail layer '{layer_mapping.nails}': only LINE "
                    f"entities are read as nails; left out: "
                    + ", ".join(f"{n} {t}" for t, n in sorted(
                        not_lines.items())))

    # --- Text annotations ---
    # Positions are converted to metres like every other coordinate
    # (GM-327: they were copied in drawing units beside geometry in m).
    factor = UNIT_FACTORS[units] / UNIT_FACTORS["m"]
    text_annotations = []
    if layer_mapping.annotations:
        for ann_layer in layer_mapping.annotations:
            if ann_layer not in layer_entities:
                warnings.append(
                    f"Annotation layer '{ann_layer}' not found in DXF "
                    f"— skipped"
                )
                continue
            for ent in layer_entities[ann_layer]:
                etype = ent.dxftype()
                if etype == "TEXT":
                    text = ent.dxf.text
                elif etype == "MTEXT":
                    text = ent.text
                else:
                    continue
                ins = ent.dxf.insert
                y = -ins.y if flip_y else ins.y
                text_annotations.append({
                    "text": text,
                    "x": ins.x * factor,
                    "y": y * factor,
                    "layer": ann_layer,
                })

    return DxfParseResult(
        surface_points=surface_points,
        boundary_profiles=boundary_profiles,
        gwt_points=gwt_points,
        nail_lines=nail_lines,
        text_annotations=text_annotations,
        units_used=units,
        units_source=units_source,
        header_units=header_units,
        warnings=warnings,
    )


#: $INSUNITS codes the DXF reference gives that this importer does not
#: convert (the supported ones are in dxf_import.units).
_INSUNITS_UNSUPPORTED = {
    3: "miles", 7: "kilometres", 8: "microinches", 9: "mils", 10: "yards",
    11: "angstroms", 12: "nanometres", 13: "microns", 14: "decimetres",
    15: "decametres", 16: "hectometres", 17: "gigametres",
    18: "astronomical units", 19: "light years", 20: "parsecs",
    21: "US survey feet", 22: "US survey inches", 23: "US survey yards",
    24: "US survey miles",
}


def _resolve_units(doc, units, ignore_header_units, warnings):
    """``(units, units_source, header_units)``: the units the coordinates
    are converted from, where they came from, and what the drawing's own
    $INSUNITS header states (None when unitless).

    The caller's ``units`` and the header are compared; nothing is assumed:

    * neither states units -> refused;
    * only the header -> the header's, said in ``units_source``;
    * only the caller (header unitless) -> the caller's;
    * both and equal -> those;
    * both and different -> refused, or the caller's with a warning that
      names both when ``ignore_header_units`` is True.
    """
    supported = sorted(UNIT_FACTORS.keys())
    try:
        code = doc.header.get("$INSUNITS", 0)
    except Exception:
        code = 0
    header_units = detect_units_from_header(doc)
    header_label = header_units
    if header_units is None and code in _INSUNITS_UNSUPPORTED:
        header_label = _INSUNITS_UNSUPPORTED[code]

    if units is None:
        if header_units is not None:
            return (header_units, "the drawing's header ($INSUNITS)",
                    header_units)
        if header_label is not None:
            raise ValueError(
                f"The drawing's header ($INSUNITS = {code}) states "
                f"{header_label}, which this importer does not convert. "
                f"Give units (one of {supported}) for the units the "
                f"coordinates are really drawn in."
            )
        raise ValueError(
            f"The drawing's header states no units ($INSUNITS is unitless) "
            f"and no units were given, so the coordinates cannot be turned "
            f"into metres. Give units (one of {supported}); a drawing in mm "
            f"read as m is 1000 times too large."
        )

    if header_label is None:
        return (units, "given (the drawing's header states no units)", None)
    if header_units == units:
        return (units, "given, and the drawing's header ($INSUNITS) agrees",
                header_units)
    if not ignore_header_units:
        raise ValueError(
            f"units='{units}' was given, and the drawing's own header "
            f"($INSUNITS = {code}) states {header_label}. Leave units out "
            f"to use the header's, or pass ignore_header_units=True if the "
            f"header is wrong (one left at a template's value is common) "
            f"and the coordinates really are in {units}."
        )
    warnings.append(
        f"The coordinates were converted from {units} as given; the "
        f"drawing's header ($INSUNITS = {code}) states {header_label} and "
        f"was ignored (ignore_header_units)."
    )
    return (units, f"given; the drawing's header states {header_label} "
                   f"and was ignored", header_units)


def _extract_polyline_coords(
    entities: list, flip_y: bool
) -> List[Tuple[float, float]]:
    """Extract and merge (x, y) coordinates from polyline/line entities.

    Collects vertices from LWPOLYLINE, POLYLINE, LINE, and SPLINE entities,
    merges all points, removes duplicates, and sorts by x.

    Parameters
    ----------
    entities : list
        List of ezdxf entities from a single layer.
    flip_y : bool
        If True, negate Y values.

    Returns
    -------
    list of (float, float)
        Sorted, deduplicated coordinates.
    """
    all_points = []
    for ent in entities:
        etype = ent.dxftype()
        if etype == "LWPOLYLINE":
            for x, y in ent.get_points(format="xy"):
                y_val = -y if flip_y else y
                all_points.append((x, y_val))
        elif etype == "POLYLINE":
            for v in ent.vertices:
                loc = v.dxf.location
                y_val = -loc.y if flip_y else loc.y
                all_points.append((loc.x, y_val))
        elif etype == "LINE":
            s = ent.dxf.start
            e = ent.dxf.end
            s_y = -s.y if flip_y else s.y
            e_y = -e.y if flip_y else e.y
            all_points.append((s.x, s_y))
            all_points.append((e.x, e_y))
        elif etype == "SPLINE":
            try:
                # Flatten spline to polyline approximation
                for pt in ent.flattening(0.01):
                    y_val = -pt.y if flip_y else pt.y
                    all_points.append((pt.x, y_val))
            except Exception:
                # Fallback: use control points
                for pt in ent.control_points:
                    y_val = -pt[1] if flip_y else pt[1]
                    all_points.append((pt[0], y_val))

    if not all_points:
        return []

    # Deduplicate (within tolerance) and sort by x
    all_points.sort(key=lambda p: (p[0], p[1]))
    deduped = [all_points[0]]
    for pt in all_points[1:]:
        if abs(pt[0] - deduped[-1][0]) > 1e-6 or abs(pt[1] - deduped[-1][1]) > 1e-6:
            deduped.append(pt)
    return deduped


def _extract_nail_lines(
    entities: list, flip_y: bool, units: str
) -> List[Dict[str, float]]:
    """Extract nail lines from LINE entities.

    Each LINE entity represents one nail from head to tip.

    Parameters
    ----------
    entities : list
        ezdxf entities on the nail layer.
    flip_y : bool
        If True, negate Y values.
    units : str
        Source units for conversion.

    Returns
    -------
    list of dict
        Each dict has {x_head, z_head, x_tip, z_tip} in meters.
    """
    factor = UNIT_FACTORS[units] / UNIT_FACTORS["m"]
    nails = []
    for ent in entities:
        if ent.dxftype() != "LINE":
            continue
        s = ent.dxf.start
        e = ent.dxf.end
        s_y = -s.y if flip_y else s.y
        e_y = -e.y if flip_y else e.y
        # Convention: head is the leftmost point (closer to slope face)
        if s.x <= e.x:
            nails.append({
                "x_head": s.x * factor,
                "z_head": s_y * factor,
                "x_tip": e.x * factor,
                "z_tip": e_y * factor,
            })
        else:
            nails.append({
                "x_head": e.x * factor,
                "z_head": e_y * factor,
                "x_tip": s.x * factor,
                "z_tip": s_y * factor,
            })
    return nails

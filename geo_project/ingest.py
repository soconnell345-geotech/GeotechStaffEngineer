"""Ingest geometry into a Project from DXF / PDF / raw points / vision drafts.

Every entry point stamps the geometry's PROVENANCE:

* :func:`from_points`       — provenance='user' (typed-in coordinates).
* :func:`from_dxf`          — provenance='dxf' (exact CAD coordinates; the
  agent must run :func:`discover_dxf` first and ASK the user which CAD layer
  is which — never guess a layer mapping).
* :func:`from_pdf_vector`   — provenance='pdf_vector' (exact PDF path
  coordinates via PyMuPDF).
* :func:`from_vision_draft` — provenance='vision_draft'. The QUARANTINED
  path: validate() emits a blocking error (GEOM007) until the human confirms
  the rendered echo-back. The agent never claims to have read geometry
  correctly — numbers→image is easy to verify, image→numbers is not to be
  trusted.

Geometry only: soil properties NEVER come from a drawing. Layers are created
named (from boundary names) with empty materials for the materials stage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from geo_project.schema import (
    Geometry,
    Layer,
    Material,
    Nail,
    Project,
    ProjectMeta,
    Reinforcement,
    Water,
)

#: Extension below the deepest boundary when no explicit section bottom is
#: known (mirrors dxf_import.converter's -2.0/-5.0 defaults, made uniform).
_DEFAULT_BOTTOM_EXTENSION = 5.0


# ---------------------------------------------------------------------------
# Shared assembly from a DxfParseResult-shaped object
# ---------------------------------------------------------------------------

def _project_from_parse_result(parse_result, provenance: str,
                               name: str,
                               section_bottom: Optional[float] = None,
                               ) -> Project:
    """Assemble a Project from a :class:`dxf_import.results.DxfParseResult`.

    Boundaries are ordered top-down by their max elevation (the
    dxf_import.converter convention); each becomes a named polyline in
    ``geometry.layer_boundaries`` and the bottom of one named layer. The
    bottom-most layer gets a flat bottom ``section_bottom`` (default: lowest
    boundary/surface point minus 5 m).
    """
    surface = [(float(x), float(z)) for x, z in parse_result.surface_points]
    if len(surface) < 2:
        raise ValueError("Parse result carries fewer than 2 surface points")

    boundaries = {
        str(bname): sorted(((float(x), float(z)) for x, z in pts),
                           key=lambda p: p[0])
        for bname, pts in (parse_result.boundary_profiles or {}).items()
        if pts
    }
    ordered = sorted(boundaries,
                     key=lambda n: -max(z for _, z in boundaries[n]))

    z_floor_candidates = [min(z for _, z in surface)]
    for pts in boundaries.values():
        z_floor_candidates.append(min(z for _, z in pts))
    bottom = (float(section_bottom) if section_bottom is not None
              else min(z_floor_candidates) - _DEFAULT_BOTTOM_EXTENSION)

    project = Project(
        meta=ProjectMeta(name=name),
        geometry=Geometry(surface_points=surface,
                          layer_boundaries=boundaries,
                          provenance=provenance),
    )

    if not ordered:
        project.stratigraphy.append(Layer(
            name="Soil", material=Material(), bottom_elevation=bottom))
    else:
        # Top layer: surface down to the first boundary.
        project.stratigraphy.append(Layer(
            name="Surface", material=Material(),
            bottom_boundary=ordered[0]))
        # Each boundary bounds the BOTTOM of the layer named after it...
        # wait, converter convention: boundary k is the bottom of the layer
        # ABOVE it and the layer NAMED after it sits BELOW the boundary.
        # Layer named b_k spans boundary k (top) to boundary k+1 (bottom).
        for i, bname in enumerate(ordered):
            if i + 1 < len(ordered):
                project.stratigraphy.append(Layer(
                    name=bname, material=Material(),
                    bottom_boundary=ordered[i + 1]))
            else:
                project.stratigraphy.append(Layer(
                    name=bname, material=Material(),
                    bottom_elevation=bottom))

    if getattr(parse_result, "gwt_points", None):
        project.water = Water(gwt_points=[
            (float(x), float(z)) for x, z in parse_result.gwt_points])

    nail_lines = getattr(parse_result, "nail_lines", None) or []
    if nail_lines:
        import math
        nails = []
        for nl in nail_lines:
            dx = nl["x_tip"] - nl["x_head"]
            dz = nl["z_tip"] - nl["z_head"]
            length = math.hypot(dx, dz)
            incl = math.degrees(math.atan2(-dz, dx)) if length > 1e-6 else 15.0
            nails.append(Nail(x_head=float(nl["x_head"]),
                              z_head=float(nl["z_head"]),
                              length=float(length), inclination=float(incl)))
            project.add_assumption(
                "reinforcement.nails[*].bar/bond/spacing defaults",
                "bar 25 mm, hole 150 mm, fy 420 MPa, bond 100 kPa, s_h 1.5 m",
                source="DXF gives nail lines only; structural defaults per "
                       "SoilNail (FHWA GEC-7 typical) — confirm at the "
                       "materials stage")
        project.reinforcement = Reinforcement(nails=nails)

    for w in (getattr(parse_result, "warnings", None) or []):
        project.add_assumption("ingest.warning", w, source=provenance)

    project.add_assumption(
        "stratigraphy[*].material", "EMPTY — drawings carry geometry only",
        source="Soil properties must come from the user/report, never from "
               "the drawing.")
    if section_bottom is None:
        project.add_assumption(
            "stratigraphy[-1].bottom_elevation", bottom,
            source=f"default: lowest imported point - "
                   f"{_DEFAULT_BOTTOM_EXTENSION:g} m")
    return project


# ---------------------------------------------------------------------------
# DXF
# ---------------------------------------------------------------------------

def discover_dxf(filepath: str = None, content: bytes = None) -> Dict[str, Any]:
    """Inventory a DXF's layers so the USER can map them to roles.

    Step 1 of the DXF flow. Returns the
    :meth:`dxf_import.results.DxfDiscoveryResult.to_dict` payload: per-layer
    entity counts/types, sample texts, bounding boxes, and a units hint.
    The agent presents this to the user and ASKS which layer is the surface,
    which are soil boundaries (and the soil names), water table, nails —
    it never guesses.
    """
    from dxf_import.discovery import discover_layers
    return discover_layers(filepath=filepath, content=content).to_dict()


def from_dxf(filepath: str = None, content: bytes = None,
             layer_mapping=None, units: str = "m", flip_y: bool = False,
             section_bottom: Optional[float] = None,
             name: str = "DXF import") -> Project:
    """Build a Project from a DXF using a USER-CONFIRMED layer mapping.

    Parameters
    ----------
    filepath, content
        DXF source (path or raw bytes).
    layer_mapping : dxf_import.parser.LayerMapping or dict
        The user-confirmed mapping. As a dict::

            {"surface": "TOPO",
             "soil_boundaries": {"STRATUM_1": "Stiff clay"},
             "water_table": "GWT",          # optional
             "nails": "NAILS"}              # optional

    units : str
        Drawing units ('m', 'mm', 'cm', 'ft', 'in').
    flip_y : bool
        Negate Y for downward-positive drawings.
    section_bottom : float, optional
        Flat bottom elevation of the modeled section (m). Default: lowest
        imported point minus 5 m (logged as an assumption).
    """
    from dxf_import.parser import LayerMapping, parse_dxf_geometry

    if layer_mapping is None:
        raise ValueError(
            "layer_mapping is required — run discover_dxf() first and ask "
            "the user to map CAD layers to roles (never guess).")
    if isinstance(layer_mapping, dict):
        layer_mapping = LayerMapping(
            surface=layer_mapping.get("surface", ""),
            soil_boundaries=dict(layer_mapping.get("soil_boundaries") or {}),
            water_table=layer_mapping.get("water_table"),
            nails=layer_mapping.get("nails"),
            annotations=layer_mapping.get("annotations"),
        )

    parsed = parse_dxf_geometry(filepath=filepath, content=content,
                                layer_mapping=layer_mapping, units=units,
                                flip_y=flip_y)
    return _project_from_parse_result(parsed, "dxf", name,
                                      section_bottom=section_bottom)


# ---------------------------------------------------------------------------
# PDF (vector)
# ---------------------------------------------------------------------------

def from_pdf_vector(filepath: str = None, content: bytes = None,
                    role_mapping: Optional[Dict[str, str]] = None,
                    page: int = 0, scale: float = 1.0,
                    origin: str = "bottom_left",
                    section_bottom: Optional[float] = None,
                    name: str = "PDF vector import") -> Project:
    """Build a Project from a PDF's VECTOR line work (exact coordinates).

    Uses :func:`pdf_import.extractor.extract_vector_geometry` with a
    color→role mapping the user confirms (e.g. ``{"#000000": "surface",
    "#0000ff": "gwt", "#808080": "boundary_Clay"}``), then adapts via
    ``pdf_import.to_dxf_parse_result``. NOT the vision path — vector
    extraction reads the drawing's actual path coordinates.
    """
    from pdf_import import to_dxf_parse_result
    from pdf_import.extractor import extract_vector_geometry

    pdf_result = extract_vector_geometry(
        filepath=filepath, content=content, page=page, scale=scale,
        origin=origin, role_mapping=role_mapping)
    parsed = to_dxf_parse_result(pdf_result)
    return _project_from_parse_result(parsed, "pdf_vector", name,
                                      section_bottom=section_bottom)


# ---------------------------------------------------------------------------
# Raw points
# ---------------------------------------------------------------------------

def from_points(surface_points: Sequence[Tuple[float, float]],
                layer_boundaries: Optional[Dict[str, Sequence]] = None,
                layer_names: Optional[List[str]] = None,
                gwt_points: Optional[Sequence] = None,
                section_bottom: Optional[float] = None,
                name: str = "User-defined section") -> Project:
    """Build a Project from user-supplied coordinates (provenance='user').

    Parameters
    ----------
    surface_points : list of (x, z)
        Ground surface, left to right.
    layer_boundaries : dict name -> list of (x, z), optional
        Bottom polylines, top-down. Layer names default to the boundary
        names (the layer ABOVE the boundary takes the boundary's name when
        ``layer_names`` is not given the converter way; here the FIRST layer
        is named 'Surface' and each boundary names the layer BELOW it,
        matching from_dxf).
    layer_names : list of str, optional
        Explicit names for the layers, top to bottom (must be
        len(boundaries) + 1 when boundaries are given, else 1).
    gwt_points : list of (x, z), optional
    section_bottom : float, optional
        Flat bottom of the section (default lowest point - 5 m).
    """
    from dxf_import.results import DxfParseResult

    parsed = DxfParseResult(
        surface_points=[(float(x), float(z)) for x, z in surface_points],
        boundary_profiles={k: [(float(x), float(z)) for x, z in v]
                           for k, v in (layer_boundaries or {}).items()},
        gwt_points=([(float(x), float(z)) for x, z in gwt_points]
                    if gwt_points else None),
    )
    project = _project_from_parse_result(parsed, "user", name,
                                         section_bottom=section_bottom)
    if layer_names:
        if len(layer_names) != len(project.stratigraphy):
            raise ValueError(
                f"layer_names has {len(layer_names)} entries but the section "
                f"resolves to {len(project.stratigraphy)} layer(s)")
        for layer, lname in zip(project.stratigraphy, layer_names):
            layer.name = lname
    return project


# ---------------------------------------------------------------------------
# Vision draft (QUARANTINED)
# ---------------------------------------------------------------------------

def from_vision_draft(draft: Any, name: str = "Vision draft (UNCONFIRMED)",
                      section_bottom: Optional[float] = None) -> Project:
    """Build a QUARANTINED Project from a vision-extracted draft.

    ``draft`` may be a :class:`pdf_import.results.PdfParseResult` (from
    ``pdf_import.vision.extract_geometry_vision``), a DxfParseResult, or a
    plain dict with ``surface_points`` / ``boundary_profiles`` /
    ``gwt_points``.

    EVERY geometry element gets provenance='vision_draft' and
    :func:`geo_project.validate.validate` emits the blocking GEOM007 error
    until the user has seen the rendered echo-back and confirmed the
    geometry stage. The numbers below are a STARTING GUESS, not data.
    """
    if isinstance(draft, dict):
        from dxf_import.results import DxfParseResult
        parsed = DxfParseResult(
            surface_points=[(float(x), float(z))
                            for x, z in (draft.get("surface_points") or [])],
            boundary_profiles={
                k: [(float(x), float(z)) for x, z in v]
                for k, v in (draft.get("boundary_profiles") or {}).items()},
            gwt_points=([(float(x), float(z))
                         for x, z in draft["gwt_points"]]
                        if draft.get("gwt_points") else None),
            warnings=list(draft.get("warnings") or []),
        )
    elif hasattr(draft, "boundary_profiles"):
        if hasattr(draft, "nail_lines"):
            parsed = draft  # already DxfParseResult-shaped
        else:
            # PdfParseResult → adapt
            from pdf_import import to_dxf_parse_result
            parsed = to_dxf_parse_result(draft)
    else:
        raise TypeError(
            "draft must be a dict, PdfParseResult or DxfParseResult, got "
            f"{type(draft).__name__}")

    project = _project_from_parse_result(parsed, "vision_draft", name,
                                         section_bottom=section_bottom)
    project.add_assumption(
        "geometry.*", "ALL geometry is a vision DRAFT",
        source="LLM vision read — must be visually confirmed against the "
               "original drawing via the echo-back render before any "
               "analysis (validate() blocks until then).")
    return project


__all__ = ["discover_dxf", "from_dxf", "from_pdf_vector", "from_points",
           "from_vision_draft"]

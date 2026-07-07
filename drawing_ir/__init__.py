"""
Drawing IR — an LLM-ready intermediate representation for a drawing page.

Design north star: LLM vision is unreliable on precise geometry, so the
DETERMINISTIC extractor owns coordinates and the LLM owns semantics. A drawing
(DXF, PDF vector, or raster scan) is digitized once into a unified
:class:`DrawingIR` — every line/polyline/arc/circle/text with coordinates,
provenance and per-entity confidence — and the LLM then requests *slices* of it
through :mod:`drawing_ir.queries` (spatial windows, angle bands, text near an
entity, layer/color groups) instead of interpreting pixels.

Quick start::

    from drawing_ir import from_dxf, queries

    ir = from_dxf("section.dxf")           # exact CAD geometry -> DrawingIR
    print(ir.summary())
    surf = queries.candidate_ground_surface(ir)   # a PROPOSAL to confirm
    labels = queries.text_items(ir, pattern="clay")

Ingest legs: :func:`from_dxf` (ezdxf, confidence 1.0), :func:`from_pdf_vector`
(PyMuPDF + the pdf_import scale module, confidence 1.0), :func:`from_raster`
(OpenCV tracing, confidence < 1.0). See ``DESIGN.md`` for schema, confidence /
provenance semantics, and the raster leg's honest limits.
"""

from drawing_ir.results import (
    Arc,
    Circle,
    DrawingIR,
    Entity,
    Line,
    Polyline,
    Region,
    TextItem,
    entity_from_dict,
)
from drawing_ir.ingest import from_dxf, from_pdf_vector, from_raster
from drawing_ir import queries

__all__ = [
    # schema
    "DrawingIR",
    "Entity",
    "Line",
    "Polyline",
    "Arc",
    "Circle",
    "TextItem",
    "Region",
    "entity_from_dict",
    # ingest
    "from_dxf",
    "from_pdf_vector",
    "from_raster",
    # queries (module)
    "queries",
]

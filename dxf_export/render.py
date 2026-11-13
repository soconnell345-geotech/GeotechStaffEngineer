"""
A sheet's DXF document drawn as a PDF page and a PNG (GM-313).

The page is drawn from the ezdxf document itself, by ezdxf's drawing
add-on with its PyMuPDF backend, at the sheet's size and at exactly 1:N;
the PNG is that PDF page rasterised. So the sheet in a calc package IS the
DXF the engineer opens, not a second drawing that could drift from it.
The PDF is vector; its lettering is drawn as outlines, so it is not
selectable text.
"""

from __future__ import annotations

import io
import os
from typing import Optional

from dxf_export._fonts import ensure_fonts
from dxf_export.drawing import Drawing, DrawingError
from dxf_export.results import DrawingSheetResult
from dxf_export.sheet import MARGIN, SheetInfo, build_dxf

#: The PNG's resolution: a letter sheet is 2,200 x 1,700 px.
PNG_DPI = 200


def dxf_bytes(doc) -> bytes:
    """The document as DXF text, in the DXF's own encoding."""
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode(doc.output_encoding, errors="replace")


def pdf_page_bytes(doc, info: SheetInfo) -> bytes:
    """``doc``'s model space as a one-page vector PDF of the sheet's size,
    black on white, at 1:``info.scale``."""
    from ezdxf.addons.drawing import Frontend, RenderContext, config, layout
    from ezdxf.addons.drawing.pymupdf import PyMuPdfBackend

    ensure_fonts()
    backend = PyMuPdfBackend()
    cfg = config.Configuration(
        background_policy=config.BackgroundPolicy.WHITE,
        color_policy=config.ColorPolicy.BLACK,
        lineweight_policy=config.LineweightPolicy.ABSOLUTE,
    )
    Frontend(RenderContext(doc), backend, config=cfg).draw_layout(
        doc.modelspace())
    page = layout.Page(info.sheet_width_mm, info.sheet_height_mm,
                       layout.Units.mm, margins=layout.Margins.all(MARGIN))
    # The border is the content's outermost box and sits MARGIN inside the
    # sheet's edge, so centring the content at 1:N puts the sheet on the
    # page exactly.
    settings = layout.Settings(fit_page=False, scale=1.0 / info.scale)
    return backend.get_pdf_bytes(page, settings=settings)


def png_bytes(pdf: bytes, dpi: int = PNG_DPI) -> bytes:
    """The PDF page as a PNG."""
    import pymupdf

    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        return document[0].get_pixmap(dpi=dpi, alpha=False).tobytes("png")


def render_sheet(drawing: Drawing, *, png_dpi: int = PNG_DPI) -> dict:
    """``drawing`` as its three files, in memory.

    Returns ``{"dxf": bytes, "pdf": bytes, "png": bytes, "info":
    SheetInfo}``; the PDF and the PNG are drawn from the DXF document."""
    doc, info = build_dxf(drawing)
    pdf = pdf_page_bytes(doc, info)
    return {"dxf": dxf_bytes(doc), "pdf": pdf,
            "png": png_bytes(pdf, png_dpi), "info": info}


def _stem(path: str) -> str:
    root, ext = os.path.splitext(path)
    return root if ext.lower() in (".dxf", ".pdf", ".png") else path


def write_sheet(drawing: Drawing, filepath: str, *,
                png_dpi: int = PNG_DPI) -> DrawingSheetResult:
    """Write ``drawing`` as ``<name>.dxf``, ``<name>.pdf`` and
    ``<name>.png`` (``filepath`` with or without one of those endings).

    Raises :class:`~dxf_export.drawing.DrawingError` for a drawing the
    layer refuses; nothing is written then."""
    if not isinstance(drawing, Drawing):
        raise DrawingError("write_sheet takes a Drawing "
                           "(Drawing.from_dict(description)).")
    files = render_sheet(drawing, png_dpi=png_dpi)
    info: SheetInfo = files["info"]
    stem = os.path.abspath(_stem(filepath))
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    paths = {}
    for ext in ("dxf", "pdf", "png"):
        paths[ext] = f"{stem}.{ext}"
        with open(paths[ext], "wb") as handle:
            handle.write(files[ext])
    return DrawingSheetResult(
        filepath=paths["dxf"], pdf_path=paths["pdf"], png_path=paths["png"],
        sheet=info.sheet, sheet_width_mm=info.sheet_width_mm,
        sheet_height_mm=info.sheet_height_mm, scale=info.scale,
        scale_chosen_by=getattr(drawing, "scale_chosen_by",
                                info.scale_chosen_by),
        counts=dict(info.counts), layers_created=list(info.layers),
        n_entities=info.n_entities,
        # a builder's own warnings (wall_section_drawing) come first
        warnings=(list(getattr(drawing, "layout_warnings", []))
                  + list(info.warnings)))

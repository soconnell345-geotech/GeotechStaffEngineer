"""
The region-snip vision primitive: render a zoomed-in crop of a PDF page.

Geometry says WHERE (a bbox from ``drawing_ir.queries`` or any other exact
source); this module turns that WHERE into a high-DPI PNG so a vision model can
answer WHAT. This is the "agentic zoom" pattern: instead of feeding a whole page
(where small annotations are illegible) or asking the model to guess pixel
coordinates, the deterministic layer picks the crop and the model only has to
look at it.

Coordinate contract
--------------------
``bbox`` and ``marks`` coordinates are **PDF points in PyMuPDF's native page
space**: origin at the page's TOP-LEFT corner, y increasing DOWNWARD — the same
frame as ``page.rect`` / ``fitz.Rect`` / ``page.get_pixmap(clip=...)``. This is
the natural frame for "render this part of the source PDF" and requires no page
lookup to interpret.

It is **not** the same as a :class:`drawing_ir.results.DrawingIR` built with
the (default) ``origin="bottom_left"`` convention, where y is flipped to
increase upward. Converting an IR point to this module's frame:

    x_pdf = x_ir
    y_pdf = page_height_pt - y_ir      # only when the IR used bottom_left

A ``coordinate_space="page", origin="top_left"`` IR needs no conversion.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Mark = Tuple[float, float, str]

#: Minimum crop size (PDF points) before padding, so a degenerate/point-like
#: bbox still yields a legible (not razor-thin) render.
MIN_CROP_SIZE = 20.0

#: Default mark circle radius (PDF points) — scales with dpi at render time
#: (more pixels per point), so it stays legible in a tight crop.
DEFAULT_MARK_RADIUS = 5.0


def _open_document(filepath=None, content=None):
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF (fitz) is required for render_region. "
            "Install with: pip install PyMuPDF>=1.23"
        ) from exc
    if content is not None:
        return fitz.open(stream=content, filetype="pdf")
    if filepath is not None:
        return fitz.open(filepath)
    raise ValueError("Provide either filepath or content")


def clip_rect_for_bbox(
    bbox: Optional[BBox], page_rect: Tuple[float, float, float, float],
    pad_frac: float = 0.15,
) -> BBox:
    """The padded, page-clamped crop rect for ``bbox`` (pure — no PDF I/O).

    ``page_rect`` is ``(x0, y0, x1, y1)`` (typically ``(0, 0, width, height)``).
    Padding is ``pad_frac`` of ``max(bbox_width, bbox_height, MIN_CROP_SIZE)``
    on every side, then clamped to the page. ``bbox=None`` returns the full
    page. Exposed so a caller can predict the exact render size before calling
    :func:`render_region` (``(x1-x0)*dpi/72`` x ``(y1-y0)*dpi/72`` pixels,
    rounded).
    """
    px0, py0, px1, py1 = page_rect
    if bbox is None:
        return (px0, py0, px1, py1)
    x0, y0, x1, y1 = bbox
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    span = max(x1 - x0, y1 - y0, MIN_CROP_SIZE)
    pad = pad_frac * span
    x0, x1 = x0 - pad, x1 + pad
    y0, y1 = y0 - pad, y1 + pad
    # Clamp to the page (never request pixels outside the rendered surface).
    x0, x1 = max(px0, x0), min(px1, x1)
    y0, y1 = max(py0, y0), min(py1, y1)
    if x1 <= x0:
        x0, x1 = px0, px1
    if y1 <= y0:
        y0, y1 = py0, py1
    return (x0, y0, x1, y1)


def _draw_marks(page, marks: Sequence[Mark], radius: float,
                mark_color, fill_color, text_color) -> None:
    fs = max(4.0, min(10.0, radius * 1.6))
    for i, m in enumerate(marks):
        x, y, label = (m[0], m[1], m[2]) if len(m) >= 3 else (m[0], m[1], str(i + 1))
        center = (float(x), float(y))
        page.draw_circle(center, radius, color=mark_color, fill=fill_color,
                         width=1.2)
        text = str(label)
        # Offset the label so it doesn't sit on top of the circle.
        page.insert_text((center[0] + radius + 1.5, center[1] - radius),
                         text, fontsize=fs, color=text_color)


def render_region(
    filepath: Optional[str] = None,
    content: Optional[bytes] = None,
    page: int = 0,
    bbox: Optional[BBox] = None,
    dpi: int = 300,
    pad_frac: float = 0.15,
    marks: Optional[Sequence[Mark]] = None,
    mark_radius: float = DEFAULT_MARK_RADIUS,
    mark_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> bytes:
    """Render a zoomed-in crop of a PDF page to PNG bytes.

    Parameters
    ----------
    filepath, content : source PDF (one required) — same convention as
        ``pdf_import.vision._render_pdf_page``.
    page : int
        0-indexed page number.
    bbox : (x0, y0, x1, y1), optional
        Region of interest in PDF points, PyMuPDF page space (top-left origin,
        y down) — see the module docstring's coordinate contract. ``None``
        renders the full page.
    dpi : int
        Render resolution (default 300 — higher than the whole-page vision
        tools, since this is the "zoom in and look closely" primitive).
    pad_frac : float
        Fractional padding added around ``bbox`` (of ``max(width, height,
        MIN_CROP_SIZE)``) before clamping to the page, so the crop isn't
        razor-tight around the geometry of interest.
    marks : list of (x, y, label), optional
        Points (same coordinate frame as ``bbox``) to annotate with a small
        numbered/labelled circle — set-of-marks prompting: "what is mark 2
        pointing at?" turns free-form pixel description into a multiple-choice
        question. Drawn directly on the page before rendering (PyMuPDF), so
        marks outside the final crop are simply not visible.
    mark_radius, mark_color : mark circle styling (PDF points / RGB 0-1).

    Returns
    -------
    bytes
        PNG image bytes.
    """
    doc = _open_document(filepath, content)
    try:
        if page >= len(doc):
            raise ValueError(
                f"Page {page} out of range (document has {len(doc)} pages)")
        pg = doc[page]
        page_rect = (0.0, 0.0, pg.rect.width, pg.rect.height)
        clip = clip_rect_for_bbox(bbox, page_rect, pad_frac=pad_frac)

        if marks:
            _draw_marks(pg, marks, mark_radius, mark_color,
                       fill_color=None, text_color=mark_color)

        import fitz
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = pg.get_pixmap(matrix=matrix, clip=fitz.Rect(*clip))
        return pix.tobytes("png")
    finally:
        doc.close()  # any mark annotations are discarded, never saved


__all__ = ["render_region", "clip_rect_for_bbox", "MIN_CROP_SIZE",
          "DEFAULT_MARK_RADIUS"]

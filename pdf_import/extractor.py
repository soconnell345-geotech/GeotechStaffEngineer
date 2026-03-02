"""
Programmatic vector extraction from PDF files via PyMuPDF.

Uses page.get_drawings() to extract vector paths, groups them by stroke color,
and assigns geometric roles via user-supplied role_mapping.

Requires: PyMuPDF >= 1.23 (optional dependency)
"""

import tempfile
from typing import Any, Dict, List, Optional, Tuple

from pdf_import.results import PdfParseResult


def _import_fitz():
    """Import PyMuPDF with helpful error message."""
    try:
        import fitz
        return fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF vector extraction. "
            "Install with: pip install PyMuPDF>=1.23"
        )


def _open_document(filepath=None, content=None):
    """Open PDF document from filepath or bytes content."""
    fitz = _import_fitz()
    if content is not None:
        if isinstance(content, (bytes, bytearray)):
            return fitz.open(stream=content, filetype="pdf")
        raise TypeError("content must be bytes or bytearray")
    if filepath is not None:
        return fitz.open(filepath)
    raise ValueError("Provide either filepath or content")


def _color_to_hex(color) -> str:
    """Convert PyMuPDF color tuple to hex string."""
    if color is None or len(color) == 0:
        return "#000000"
    if len(color) == 1:
        # Grayscale
        v = int(color[0] * 255)
        return f"#{v:02x}{v:02x}{v:02x}"
    if len(color) == 3:
        r, g, b = (int(c * 255) for c in color)
        return f"#{r:02x}{g:02x}{b:02x}"
    return "#000000"


def discover_pdf_content(
    filepath=None, content=None, page: int = 0,
) -> Dict[str, Any]:
    """Inventory a PDF page: vector paths by color, text blocks, dimensions.

    Parameters
    ----------
    filepath : str, optional
        Path to PDF file.
    content : bytes, optional
        PDF file content as bytes.
    page : int
        Page number (0-indexed).

    Returns
    -------
    dict with keys:
        'page_size' : dict with 'width' and 'height' in points
        'n_drawings' : int — total vector path count
        'colors' : dict — {hex_color: count}
        'text_blocks' : list of dict — {text, x, y, size}
        'has_images' : bool — whether page contains raster images
    """
    doc = _open_document(filepath, content)
    n_pages = len(doc)
    if page >= n_pages:
        doc.close()
        raise ValueError(f"Page {page} out of range (document has {n_pages} pages)")

    pg = doc[page]
    rect = pg.rect

    # Count drawings by color
    drawings = pg.get_drawings()
    colors: Dict[str, int] = {}
    for d in drawings:
        c = _color_to_hex(d.get("color"))
        colors[c] = colors.get(c, 0) + 1

    # Extract text blocks
    text_blocks = []
    text_dict = pg.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type") == 0:  # text block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_blocks.append({
                        "text": span.get("text", "").strip(),
                        "x": round(span.get("origin", (0, 0))[0], 2),
                        "y": round(span.get("origin", (0, 0))[1], 2),
                        "size": round(span.get("size", 0), 1),
                    })

    # Check for images
    has_images = len(pg.get_images()) > 0

    doc.close()
    return {
        "page_size": {
            "width": round(rect.width, 2),
            "height": round(rect.height, 2),
        },
        "n_drawings": len(drawings),
        "colors": colors,
        "text_blocks": [tb for tb in text_blocks if tb["text"]],
        "has_images": has_images,
    }


def extract_vector_geometry(
    filepath=None, content=None, page: int = 0,
    scale: float = 1.0, origin: str = "bottom_left",
    role_mapping: Optional[Dict[str, str]] = None,
) -> PdfParseResult:
    """Extract geometry from PDF vector drawings via PyMuPDF.

    Parameters
    ----------
    filepath : str, optional
        Path to PDF file.
    content : bytes, optional
        PDF file content as bytes.
    page : int
        Page number (0-indexed).
    scale : float
        Scale factor: drawing_units * scale = meters.
    origin : str
        Coordinate origin: 'bottom_left' (default, flips Y from PDF top-left)
        or 'top_left' (raw PDF coordinates).
    role_mapping : dict, optional
        Maps hex color strings to roles:
            {"#000000": "surface", "#0000ff": "gwt", "#808080": "boundary_Clay"}
        Boundary roles must start with "boundary_" prefix.

    Returns
    -------
    PdfParseResult
        Extracted geometry with coordinates in meters.
    """
    doc = _open_document(filepath, content)
    n_pages = len(doc)
    if page >= n_pages:
        doc.close()
        raise ValueError(f"Page {page} out of range (document has {n_pages} pages)")

    pg = doc[page]
    page_height = pg.rect.height
    drawings = pg.get_drawings()

    # Group paths by color
    paths_by_color: Dict[str, List[List[Tuple[float, float]]]] = {}
    for d in drawings:
        color_hex = _color_to_hex(d.get("color"))
        items = d.get("items", [])
        path_points = _extract_path_points(items)
        if path_points:
            if color_hex not in paths_by_color:
                paths_by_color[color_hex] = []
            paths_by_color[color_hex].append(path_points)

    # Extract text
    text_annotations = []
    text_dict = pg.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type") == 0:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        ox, oy = span.get("origin", (0, 0))
                        if origin == "bottom_left":
                            oy = page_height - oy
                        text_annotations.append({
                            "text": text,
                            "x": round(ox * scale, 4),
                            "y": round(oy * scale, 4),
                        })

    doc.close()

    # Apply role mapping
    surface_points = []
    boundary_profiles = {}
    gwt_points = None
    warnings = []

    if role_mapping:
        for color_hex, role in role_mapping.items():
            color_lower = color_hex.lower()
            if color_lower not in paths_by_color:
                warnings.append(f"Color {color_hex} (role '{role}') not found in drawings")
                continue

            all_points = []
            for path in paths_by_color[color_lower]:
                all_points.extend(path)

            # Transform coordinates
            transformed = []
            for x, y in all_points:
                if origin == "bottom_left":
                    y = page_height - y
                transformed.append((round(x * scale, 4), round(y * scale, 4)))

            # Sort by x
            transformed.sort(key=lambda p: p[0])

            if role == "surface":
                surface_points = transformed
            elif role == "gwt":
                gwt_points = transformed
            elif role.startswith("boundary_"):
                name = role[len("boundary_"):]
                boundary_profiles[name] = transformed
            else:
                warnings.append(f"Unknown role '{role}' for color {color_hex}")
    else:
        # No mapping — collect all paths as surface (first color found)
        if paths_by_color:
            first_color = next(iter(paths_by_color))
            all_points = []
            for path in paths_by_color[first_color]:
                all_points.extend(path)
            transformed = []
            for x, y in all_points:
                if origin == "bottom_left":
                    y = page_height - y
                transformed.append((round(x * scale, 4), round(y * scale, 4)))
            transformed.sort(key=lambda p: p[0])
            surface_points = transformed
            if len(paths_by_color) > 1:
                warnings.append(
                    f"No role_mapping provided — only extracted {len(surface_points)} "
                    f"points from color {first_color}. Provide role_mapping for "
                    f"multi-color drawings."
                )

    return PdfParseResult(
        surface_points=surface_points,
        boundary_profiles=boundary_profiles,
        gwt_points=gwt_points,
        text_annotations=text_annotations,
        page_number=page,
        extraction_method="vector",
        scale_factor=scale,
        confidence=1.0,
        warnings=warnings,
    )


def _extract_path_points(items) -> List[Tuple[float, float]]:
    """Extract (x, y) points from PyMuPDF drawing items."""
    points = []
    for item in items:
        kind = item[0]  # "l" for line, "c" for curve, "re" for rect
        if kind == "l":
            # Line: ("l", Point(x1,y1), Point(x2,y2))
            p1, p2 = item[1], item[2]
            points.append((p1.x, p1.y))
            points.append((p2.x, p2.y))
        elif kind == "c":
            # Bezier curve: ("c", p1, p2, p3, p4)
            # Use start and end points
            points.append((item[1].x, item[1].y))
            points.append((item[4].x, item[4].y))
        elif kind == "re":
            # Rectangle: ("re", Rect)
            r = item[1]
            points.append((r.x0, r.y0))
            points.append((r.x1, r.y0))
            points.append((r.x1, r.y1))
            points.append((r.x0, r.y1))
    # Deduplicate consecutive identical points
    if len(points) > 1:
        deduped = [points[0]]
        for p in points[1:]:
            if abs(p[0] - deduped[-1][0]) > 1e-6 or abs(p[1] - deduped[-1][1]) > 1e-6:
                deduped.append(p)
        return deduped
    return points

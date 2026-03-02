"""
LLM vision-based geometry extraction from PDF/image files.

Uses a pluggable image_fn callable for vision analysis — compatible with
PrompterAPI.analyze_image(), Claude vision, or any similar service.

No dependency on any specific LLM provider.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from pdf_import.results import PdfParseResult


VISION_PROMPT = """\
You are analyzing a geotechnical cross-section drawing. Extract the geometry as JSON.

Return ONLY a JSON object with these fields:
{
    "surface_points": [[x1, z1], [x2, z2], ...],
    "boundary_profiles": {
        "layer_name": [[x1, z1], [x2, z2], ...],
        ...
    },
    "gwt_points": [[x1, z1], [x2, z2], ...] or null,
    "scale_info": "description of any scale/dimensions visible"
}

Rules:
- Coordinates should be in the drawing's native units
- x increases left to right, z increases upward (elevation)
- The surface is the topmost ground profile line
- Boundaries separate different soil layers
- GWT (groundwater table) is typically shown as a dashed/blue line
- Include at least the endpoints and any slope breaks
- If you cannot identify a feature, omit it rather than guess
"""


def extract_geometry_vision(
    image_fn: Callable,
    filepath: Optional[str] = None,
    content: Optional[bytes] = None,
    page: int = 0,
    dpi: int = 200,
    custom_prompt: Optional[str] = None,
    scale: float = 1.0,
) -> PdfParseResult:
    """Extract geometry from PDF/image using LLM vision.

    Parameters
    ----------
    image_fn : callable
        Vision function: image_fn(image_bytes, prompt) -> str.
        Compatible with PrompterAPI.analyze_image().
    filepath : str, optional
        Path to PDF or image file.
    content : bytes, optional
        File content as bytes.
    page : int
        PDF page number (0-indexed). Ignored for image files.
    dpi : int
        Resolution for PDF→image rendering (default 200).
    custom_prompt : str, optional
        Custom prompt to use instead of the default.
    scale : float
        Scale factor: drawing_units * scale = meters.

    Returns
    -------
    PdfParseResult
        Extracted geometry with confidence < 1.0.
    """
    image_bytes = _get_image_bytes(filepath, content, page, dpi)
    prompt = custom_prompt or VISION_PROMPT

    # Call the vision function
    response_text = image_fn(image_bytes, prompt)

    # Parse the JSON response
    result = _parse_vision_response(response_text, scale, page)
    return result


def _get_image_bytes(
    filepath: Optional[str] = None,
    content: Optional[bytes] = None,
    page: int = 0,
    dpi: int = 200,
) -> bytes:
    """Get image bytes from filepath or content.

    If PDF, renders the specified page to PNG.
    If image, returns bytes directly.
    """
    if content is not None:
        # Check if it's a PDF by magic bytes
        if content[:5] == b"%PDF-":
            return _render_pdf_page(content=content, page=page, dpi=dpi)
        # Assume it's already an image
        return content

    if filepath is not None:
        # Check file extension
        lower = filepath.lower()
        if lower.endswith(".pdf"):
            return _render_pdf_page(filepath=filepath, page=page, dpi=dpi)
        # Image file — read bytes
        with open(filepath, "rb") as f:
            return f.read()

    raise ValueError("Provide either filepath or content")


def _render_pdf_page(
    filepath: Optional[str] = None,
    content: Optional[bytes] = None,
    page: int = 0,
    dpi: int = 200,
) -> bytes:
    """Render a PDF page to PNG bytes using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF rendering. "
            "Install with: pip install PyMuPDF>=1.23"
        )

    if content is not None:
        doc = fitz.open(stream=content, filetype="pdf")
    else:
        doc = fitz.open(filepath)

    if page >= len(doc):
        doc.close()
        raise ValueError(f"Page {page} out of range (document has {len(doc)} pages)")

    pg = doc[page]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = pg.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


def _parse_vision_response(
    response_text: str,
    scale: float = 1.0,
    page: int = 0,
) -> PdfParseResult:
    """Parse LLM vision response into PdfParseResult.

    Attempts to extract JSON from the response text, even if wrapped in
    markdown code blocks or surrounded by explanatory text.
    """
    warnings = []
    confidence = 0.7  # default for vision-based extraction

    # Try to extract JSON from the response
    json_data = _extract_json(response_text)
    if json_data is None:
        return PdfParseResult(
            page_number=page,
            extraction_method="vision",
            scale_factor=scale,
            confidence=0.0,
            warnings=["Could not parse JSON from vision response"],
        )

    # Extract surface points
    surface_points = []
    raw_surface = json_data.get("surface_points", [])
    if isinstance(raw_surface, list):
        for pt in raw_surface:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                try:
                    x, z = float(pt[0]) * scale, float(pt[1]) * scale
                    surface_points.append((round(x, 4), round(z, 4)))
                except (ValueError, TypeError):
                    pass
    if surface_points:
        confidence = max(confidence, 0.7)
    else:
        confidence = min(confidence, 0.3)
        warnings.append("No valid surface points extracted")

    # Extract boundary profiles
    boundary_profiles = {}
    raw_boundaries = json_data.get("boundary_profiles", {})
    if isinstance(raw_boundaries, dict):
        for name, pts in raw_boundaries.items():
            if isinstance(pts, list):
                parsed = []
                for pt in pts:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        try:
                            x, z = float(pt[0]) * scale, float(pt[1]) * scale
                            parsed.append((round(x, 4), round(z, 4)))
                        except (ValueError, TypeError):
                            pass
                if parsed:
                    boundary_profiles[str(name)] = parsed

    # Extract GWT
    gwt_points = None
    raw_gwt = json_data.get("gwt_points")
    if raw_gwt is not None and isinstance(raw_gwt, list):
        parsed_gwt = []
        for pt in raw_gwt:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                try:
                    x, z = float(pt[0]) * scale, float(pt[1]) * scale
                    parsed_gwt.append((round(x, 4), round(z, 4)))
                except (ValueError, TypeError):
                    pass
        if parsed_gwt:
            gwt_points = parsed_gwt

    # Check scale info
    scale_info = json_data.get("scale_info")
    if scale_info and isinstance(scale_info, str) and scale_info.strip():
        warnings.append(f"Scale info from vision: {scale_info}")

    return PdfParseResult(
        surface_points=surface_points,
        boundary_profiles=boundary_profiles,
        gwt_points=gwt_points,
        page_number=page,
        extraction_method="vision",
        scale_factor=scale,
        confidence=round(confidence, 2),
        warnings=warnings,
    )


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON object from text, handling markdown code blocks."""
    # Try direct JSON parse first
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting from markdown code block
    pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in text
    brace_start = text.find("{")
    if brace_start >= 0:
        # Find matching closing brace
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break

    return None

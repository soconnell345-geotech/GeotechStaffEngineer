"""
Raster tracing leg — turn a scanned/exported drawing image into a DrawingIR.

This is the LOW-CONFIDENCE, best-effort source. Where DXF and PDF-vector give
exact path coordinates (confidence 1.0), a raster image gives only pixels, so
every entity here is a DETECTION with a sub-1.0 confidence and pixel-quantized
coordinates. Use it to bootstrap an IR from a scan, then have a human confirm.

What it does (OpenCV):
- Straight segments via probabilistic Hough transform on Canny edges → Line.
- Circles via the Hough gradient transform → Circle.
- Closed shapes via external-contour tracing + polygon approximation → Polyline
  (closed). Best for filled/outlined regions; on pure line-work it may also
  trace the *outline* of a thick stroke, so it overlaps the Hough line result —
  enable/disable per drawing.
- Text via OCR **only if** an OCR backend (pytesseract + a Tesseract binary) is
  importable and working; otherwise text is skipped with a warning (positions
  are not invented).

Honest limits: no layers (raster has none), colors are sampled per detection,
arcs are not recovered (partial circles are missed or seen as contours), curved
paths become polygon approximations, and coordinates carry pixel-rounding error.
Confidence tiers are fixed per detector (see the CONF_* constants).

Requires: opencv-python-headless (``cv2``). Install with
``pip install opencv-python-headless``.
"""

from __future__ import annotations

from typing import Any, Optional

from drawing_ir.results import Circle, DrawingIR, Line, Polyline, TextItem

# Fixed confidence tiers by detector (deterministic sources use 1.0).
CONF_HOUGH_LINE = 0.6
CONF_CIRCLE = 0.5
CONF_CONTOUR = 0.5
CONF_OCR_TEXT = 0.4


def _require_cv2():
    try:
        import cv2  # noqa: F401
        return cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "opencv is required for the raster tracing leg. Install with: "
            "pip install opencv-python-headless"
        ) from exc


def _load_gray(cv2, filepath, image):
    import numpy as np
    if image is not None:
        arr = np.asarray(image)
        if arr.ndim == 3:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr
    if filepath is None:
        raise ValueError("Provide either filepath or image")
    gray = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {filepath}")
    return gray


def _make_converter(height, origin, scale):
    """Return f(px_x, px_y) -> output coords with y-flip and scale applied."""
    def conv(x, y):
        yy = (height - y) if origin == "bottom_left" else y
        if scale:
            return (x * scale, yy * scale)
        return (float(x), float(yy))
    return conv


def _sample_color(cv2, img_bgr, x, y):
    """Sample a hex color near a pixel (returns None if unavailable)."""
    if img_bgr is None:
        return None
    import numpy as np
    h, w = img_bgr.shape[:2]
    xi = int(min(max(x, 0), w - 1))
    yi = int(min(max(y, 0), h - 1))
    b, g, r = (int(v) for v in img_bgr[yi, xi][:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def trace_raster(
    filepath: str = None,
    image: Any = None,
    scale: Optional[float] = None,
    scale_provenance: Optional[str] = None,
    origin: str = "bottom_left",
    detect_lines: bool = True,
    detect_circles: bool = True,
    detect_contours: bool = True,
    ocr: bool = True,
    canny_lo: int = 50,
    canny_hi: int = 150,
    hough_threshold: int = 50,
    min_line_length: float = 20.0,
    max_line_gap: float = 5.0,
    min_contour_area: float = 100.0,
    approx_eps_frac: float = 0.01,
    circle_param2: int = 30,
    min_radius: int = 0,
    max_radius: int = 0,
    name: str = "raster trace",
) -> DrawingIR:
    """Trace an image into a DrawingIR (source='raster_trace', confidence < 1).

    Parameters mirror the OpenCV knobs so the caller can tune per drawing.
    ``scale`` (meters per pixel, e.g. from a two-point calibration) promotes the
    IR to model space; without it the IR stays in pixel space (units='px').
    """
    cv2 = _require_cv2()
    import numpy as np

    gray = _load_gray(cv2, filepath, image)
    img_bgr = None
    if image is not None:
        arr = np.asarray(image)
        if arr.ndim == 3:
            img_bgr = arr
    elif filepath is not None:
        img_bgr = cv2.imread(str(filepath), cv2.IMREAD_COLOR)

    h, w = gray.shape[:2]
    conv = _make_converter(h, origin, scale)
    warnings = []

    coordinate_space = "model" if scale else "page"
    units = "m" if scale else "px"
    ir = DrawingIR(
        width=(w * scale) if scale else float(w),
        height=(h * scale) if scale else float(h),
        units=units,
        coordinate_space=coordinate_space,
        scale=scale,
        scale_provenance=(scale_provenance or "raster_calibration") if scale else None,
        origin=origin,
        source="raster_trace",
        warnings=warnings,
        metadata={"detectors": {
            "lines": detect_lines, "circles": detect_circles,
            "contours": detect_contours, "ocr": ocr}},
    )

    # --- Lines (Hough on Canny edges) ---
    if detect_lines:
        edges = cv2.Canny(gray, canny_lo, canny_hi)
        segments = cv2.HoughLinesP(
            edges, 1, np.pi / 180.0, hough_threshold,
            minLineLength=min_line_length, maxLineGap=max_line_gap)
        if segments is not None:
            for seg in segments:
                x1, y1, x2, y2 = (float(v) for v in np.asarray(seg).reshape(-1)[:4])
                ir.add(Line(
                    start=conv(x1, y1), end=conv(x2, y2),
                    source="raster_trace", confidence=CONF_HOUGH_LINE,
                    style="hough",
                    color=_sample_color(cv2, img_bgr, (x1 + x2) / 2,
                                        (y1 + y2) / 2)))

    # --- Circles (Hough gradient) ---
    if detect_circles:
        blurred = cv2.medianBlur(gray, 3)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=max(10, h // 8),
            param1=canny_hi, param2=circle_param2,
            minRadius=min_radius, maxRadius=max_radius)
        if circles is not None:
            for c in circles[0]:
                cx, cy, r = (float(v) for v in np.asarray(c).reshape(-1)[:3])
                radius = r * scale if scale else r
                ir.add(Circle(
                    center=conv(cx, cy), radius=radius,
                    source="raster_trace", confidence=CONF_CIRCLE,
                    style="hough_circle",
                    color=_sample_color(cv2, img_bgr, cx, cy)))

    # --- Closed shapes (external contours) ---
    if detect_contours:
        _, bw = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        found = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area:
                continue
            eps = approx_eps_frac * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            verts = [conv(float(p[0][0]), float(p[0][1])) for p in approx]
            if len(verts) < 2:
                continue
            ir.add(Polyline(
                vertices=verts, closed=True,
                source="raster_trace", confidence=CONF_CONTOUR,
                style="contour"))

    # --- Text (OCR, best-effort) ---
    if ocr:
        texts, ocr_warn = _ocr_words(gray, conv, scale)
        for t in texts:
            ir.add(t)
        if ocr_warn:
            warnings.append(ocr_warn)

    if not ir.entities:
        warnings.append("No entities detected — try lowering hough_threshold / "
                        "min_line_length or check the image contrast.")
    return ir


def _ocr_words(gray, conv, scale):
    """Best-effort OCR via pytesseract; returns (TextItems, warning-or-None)."""
    try:
        import pytesseract
    except ImportError:
        return [], ("OCR skipped: pytesseract not installed — raster text was "
                    "not extracted (positions-only leg). Install pytesseract + "
                    "a Tesseract binary to enable.")
    try:
        data = pytesseract.image_to_data(
            gray, output_type=pytesseract.Output.DICT)
    except Exception as exc:  # pragma: no cover - needs tesseract binary
        return [], f"OCR skipped: Tesseract not available ({exc})."
    items = []
    n = len(data.get("text", []))
    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue
        x, y = data["left"][i], data["top"][i]
        th = data["height"][i]
        # OCR box is top-left; place the insertion at the text baseline.
        px, py = conv(x, y + th)
        items.append(TextItem(
            content=word, position=(px, py), rotation=0.0,
            height=(th * scale) if scale else float(th),
            source="raster_trace", confidence=CONF_OCR_TEXT, style="ocr"))
    return items, None

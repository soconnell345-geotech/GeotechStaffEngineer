"""
Scale calibration for PDF cross-section import (C1).

Two ways to establish the drawing-units -> meters scale factor used by
``extract_vector_geometry(scale=...)``:

1. ``calibrate_scale(p1, p2, distance_m)`` — DETERMINISTIC two-point calibration:
   the user identifies two points on the drawing whose real-world separation is
   known (a dimension line, a grid interval, a scale bar). Exact; no assumption
   about the PDF's physical size.

2. ``propose_scale_candidates(text_blocks, ...)`` — parse scale ANNOTATIONS from
   the page text ("SCALE 1:100", '1" = 20 ft', "1 cm = 2 m", ...) and return a
   list of scale CANDIDATES, each tagged with its provenance and confidence.
   These are PROPOSALS ONLY — never silently applied; the caller (or user)
   confirms one and passes it as ``scale`` to the extractor. The ratio/engineering
   candidates assume the PDF coordinates are at true plot size (1 pt = 1/72 in),
   which holds for most CAD-exported PDFs but should be spot-checked against a
   known dimension (``calibrate_scale``).

All returned scale factors are "meters per drawing unit (PDF point)".

References
----------
Companion to pdf_import.extractor (vector) / pdf_import.vision.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple


# 1 PDF point = 1/72 inch (at true plot size). Unit conversions:
_M_PER_INCH = 0.0254
_M_PER_FT = 0.3048
_M_PER_CM = 0.01
_M_PER_MM = 0.001
_M_PER_PT = _M_PER_INCH / 72.0        # meters of paper per PDF point


def calibrate_scale(p1: Tuple[float, float], p2: Tuple[float, float],
                    distance_m: float) -> float:
    """Two-point scale calibration: meters per drawing unit.

    Given two points ``p1``, ``p2`` in the drawing's coordinate system (the same
    units as ``discover_pdf_content`` / the raw vector geometry — PDF points) and
    the known real-world ``distance_m`` between them, returns the scale factor to
    pass to ``extract_vector_geometry(scale=...)`` so that drawing_units * scale =
    meters.

    Parameters
    ----------
    p1, p2 : tuple of (float, float)
        Two points in drawing units (PDF points).
    distance_m : float
        Known real-world distance between p1 and p2 (meters). Must be > 0.

    Returns
    -------
    float
        Scale factor (meters per drawing unit).

    Raises
    ------
    ValueError
        If distance_m <= 0 or the two points coincide.
    """
    if distance_m <= 0:
        raise ValueError(f"distance_m must be positive, got {distance_m}")
    d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if d <= 0:
        raise ValueError("p1 and p2 must be distinct points")
    return distance_m / d


def _candidate(scale_factor: float, basis: str, provenance: str,
               confidence: float, note: str = "") -> Dict[str, Any]:
    return {
        "scale_factor": scale_factor,
        "basis": basis,
        "provenance": provenance,
        "confidence": round(confidence, 3),
        "note": note,
        "applied": False,   # proposals are NEVER silently applied
    }


# Regexes for scale annotations (case-insensitive).
_RE_RATIO = re.compile(r"\b1\s*[:/]\s*(\d{1,6}(?:\.\d+)?)\b")
# 1" = 20', 1 in = 20 ft, 1"=20 ft, 1 inch = 20 feet
_RE_ENG_IMP = re.compile(
    r'1\s*(?:"|in\b|inch\b)\s*=\s*(\d+(?:\.\d+)?)\s*(?:\'|ft\b|feet\b|foot\b)',
    re.I)
# 1" = 20 m, 1 in = 5 m
_RE_ENG_IN_M = re.compile(r'1\s*(?:"|in\b|inch\b)\s*=\s*(\d+(?:\.\d+)?)\s*m\b', re.I)
# 1 cm = 2 m, 1 cm = 200 cm
_RE_METRIC_CM = re.compile(
    r'1\s*cm\s*=\s*(\d+(?:\.\d+)?)\s*(m|cm|mm)\b', re.I)
_RE_METRIC_MM = re.compile(
    r'1\s*mm\s*=\s*(\d+(?:\.\d+)?)\s*(m|cm|mm)\b', re.I)


def _has_scale_keyword(text: str) -> bool:
    return "scale" in text.lower()


def parse_scale_annotations(text_blocks: List[Dict[str, Any]]
                            ) -> List[Dict[str, Any]]:
    """Scan page text blocks for scale annotations -> scale CANDIDATES.

    Recognizes bare/keyworded ratio scales ("1:100", "SCALE 1:200"), engineering
    scales ('1" = 20 ft', "1 in = 5 m") and metric scales ("1 cm = 2 m"). Each
    candidate is returned with its raw provenance text and a confidence; ratio and
    engineering scales assume the PDF is at true plot size (1 pt = 1/72 in).

    Parameters
    ----------
    text_blocks : list of dict
        Text blocks as produced by ``discover_pdf_content`` (each has a "text").

    Returns
    -------
    list of dict
        Scale candidates, highest-confidence first. NEVER applied automatically.
    """
    candidates: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(sf, basis, prov, conf, note=""):
        key = (round(sf, 9), basis)
        if key in seen:
            return
        seen.add(key)
        candidates.append(_candidate(sf, basis, prov, conf, note))

    for tb in text_blocks:
        text = (tb.get("text") or "").strip()
        if not text:
            continue
        kw = _has_scale_keyword(text)

        m = _RE_ENG_IMP.search(text)
        if m:
            ft = float(m.group(1))
            sf = ft * _M_PER_FT / 72.0     # 1 pt = 1/72 in -> ft/72 -> m
            _add(sf, "engineering_imperial", text, 0.85 if kw else 0.7,
                 "assumes PDF at true plot size (1 pt = 1/72 in)")
            continue
        m = _RE_ENG_IN_M.search(text)
        if m:
            mm = float(m.group(1))
            sf = mm / 72.0
            _add(sf, "engineering_in_to_m", text, 0.85 if kw else 0.7,
                 "assumes PDF at true plot size (1 pt = 1/72 in)")
            continue
        m = _RE_METRIC_CM.search(text)
        if m:
            val, unit = float(m.group(1)), m.group(2).lower()
            real_m = val * {"m": 1.0, "cm": _M_PER_CM, "mm": _M_PER_MM}[unit]
            # 1 pt = 2.54/72 cm of paper; "1 cm of paper = real_m meters".
            sf = (2.54 / 72.0) * real_m
            _add(sf, "metric_cm", text, 0.85 if kw else 0.7,
                 "assumes PDF at true plot size (1 pt = 1/72 in)")
            continue
        m = _RE_METRIC_MM.search(text)
        if m:
            val, unit = float(m.group(1)), m.group(2).lower()
            real_m = val * {"m": 1.0, "cm": _M_PER_CM, "mm": _M_PER_MM}[unit]
            sf = (25.4 / 72.0) * real_m      # 1 pt = 25.4/72 mm of paper
            _add(sf, "metric_mm", text, 0.85 if kw else 0.7,
                 "assumes PDF at true plot size (1 pt = 1/72 in)")
            continue
        m = _RE_RATIO.search(text)
        if m:
            N = float(m.group(1))
            if N <= 1:
                continue
            sf = N * _M_PER_PT               # 1 pt of paper = N pt of reality
            _add(sf, "ratio_1_to_N", text, 0.8 if kw else 0.45,
                 "1:N assumes PDF at true plot size (1 pt = 1/72 in); "
                 "verify against a known dimension")

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates


def propose_scale(text_blocks: List[Dict[str, Any]],
                  calibration: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
    """Assemble scale proposals from text annotations (+ an optional two-point
    calibration), returned as candidates for the user to confirm.

    Parameters
    ----------
    text_blocks : list of dict
        Text blocks from ``discover_pdf_content``.
    calibration : dict, optional
        {"p1": [x, y], "p2": [x, y], "distance_m": float} to add a deterministic
        two-point candidate (confidence 1.0).

    Returns
    -------
    dict
        {"candidates": [...], "recommended": <best candidate or None>,
         "note": "proposals only — confirm one and pass it as scale=..."}
    """
    candidates = parse_scale_annotations(text_blocks)
    if calibration:
        sf = calibrate_scale(tuple(calibration["p1"]), tuple(calibration["p2"]),
                             calibration["distance_m"])
        candidates.insert(0, _candidate(
            sf, "two_point_calibration",
            f"p1={calibration['p1']}, p2={calibration['p2']}, "
            f"distance={calibration['distance_m']} m", 1.0,
            "deterministic; no plot-size assumption"))
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return {
        "candidates": candidates,
        "recommended": candidates[0] if candidates else None,
        "note": ("Scale proposals only — NOT applied. Confirm one and pass its "
                 "scale_factor as extract_vector_geometry(scale=...)."),
    }

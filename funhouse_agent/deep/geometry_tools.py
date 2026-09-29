"""Drawing-geometry tools on the lean review agent (``GEOTECH_REVIEW_GEOMETRY=1``).

WHY. The Document Review page's agent reads text and looks at pages, but it
has never had the drawing geometry planlens computes from a sheet's line-work
(``planlens.ir``: the leader, dimension, title-block and revision-cloud
finders). The geotechnical page reaches that geometry through its
``call_agent`` dispatch; the review page has no such tool. On a sheet whose
lettering is drawn as lines there is no text to search, and a whole-sheet
image is too small to read small callouts, so the reviewer needs to know
WHERE to zoom. "Geometry says where, vision says what": these tools find the
leader, the dimension line, the title block and the cloud; ``render_region``
then reads the lettering there (plan S1.1,
``module_work/REVIEW_ARCHITECTURE.md``).

WHAT. Four read-only tools, each taking ``source`` (an attachment key or a
path, resolved like the document tools), a 0-based ``page`` and an optional
``bbox`` that limits the answer to one region:

* ``drawing_callouts`` - leaders and callout arrows: the tip (what it points
  at), the tail (where its label sits), the label when the PDF's text layer
  has one there, and a ``label_zoom`` box to read it by looking;
* ``drawing_dimensions`` - dimension lines: both ends, the drawn length in
  PDF points, the value text when the text layer has it (else a
  ``value_zoom`` box), and the real-world length ONLY when the PDF itself
  stores a calibrated scale (``planlens.document.scale``) - never a guess;
* ``title_block`` - the title block's box and the fields its text gives;
* ``revision_clouds`` - clouded regions and the revision triangles near them.

THE FRAME. Every coordinate a result carries is in the DISPLAYED page frame:
PDF points on the page as a viewer shows it, top-left origin, y down - the
frame ``render_region``, ``read_document(with_locations)`` and
``annotate_document`` use. The planlens IR is not in that frame: it holds the
UNROTATED page geometry y-flipped with the ROTATED page height, so on a
``/Rotate 90`` or ``270`` sheet (two of the Mecklenburg sheets) a bare y-flip
points at unrelated content. :class:`_Sheet` converts with the page's own
rotation matrix, the mapping ``planlens.document.frame.from_ir_point``
documents.

LABELS come from the document layer's located lines (the same text
``read_document`` gives, AutoCAD's hidden SHX text included), not from the
IR's text spans, so the finders run exactly as planlens validated them.

Results are PROPOSALS with a confidence, strongest first, each kept under
:data:`RESULT_CHARS` (the rest are counted and left out). The ingested IR and
each finder's output are cached per (document hash, page) in this process.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from funhouse_agent import document_tools

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]

#: The tools this module adds, in the order they are offered.
GEOMETRY_TOOLS = ("drawing_callouts", "drawing_dimensions", "title_block",
                  "revision_clouds")

#: Every result stays under this many characters (the review tools' general
#: cap is 8,000-12,000); the weakest proposals are left out first.
RESULT_CHARS = 9000
#: Never list more than this many proposals, whatever the budget allows.
MAX_ITEMS = 40
#: The call threshold planlens' finders are documented against.
DEFAULT_MIN_CONFIDENCE = 0.5
#: Revision clouds are a best-effort tier (<= ~0.65 by design).
CLOUD_MIN_CONFIDENCE = 0.3
#: A callout whose tip and tail are further apart than this fraction of the
#: sheet's short side is held below the call threshold (see callouts()).
LONG_LEADER = 0.35
#: Sheets kept in memory (a dense sheet's IR is a few MB).
MAX_SHEETS = 16

PROPOSAL_NOTE = (
    "Geometric PROPOSALS found from the drawn lines, each with a confidence: "
    "confirm one by zooming on its box with render_region before you quote "
    "it.")
FRAME_NOTE = (
    "Coordinates are PDF points on the page as displayed (top-left origin, y "
    "down): pass a box straight to render_region as bbox, with the same "
    "document as attachment_key and the same page.")
NO_TEXT_NOTE = (
    "No text layer here: the lettering is drawn as lines, so read each "
    "label or value by zooming on its box.")

CALLOUTS_DESCRIPTION = (
    "Find the leaders and callout arrows drawn on a drawing page, from its "
    "line-work: for each, the tip (the spot it points at), the tail (where "
    "its label sits), the label when the PDF's text layer has it, and a "
    "label_zoom box for render_region to read a label drawn as lines. Use it "
    "to see what a note points at, or to list a sheet's callouts before "
    "zooming on each. source = the document as uploaded (or a path); page "
    "0-based; bbox [x0, y0, x1, y1] in PDF points limits it to one region. "
    "Proposals with a confidence: zoom to confirm before you quote one.")
DIMENSIONS_DESCRIPTION = (
    "Find the dimension lines drawn on a drawing page: each one's two ends, "
    "its drawn length in PDF points, its value text when the text layer has "
    "it (else a value_zoom box for render_region to read it), and the "
    "real-world length only when the PDF itself stores a scale. Use it to "
    "find the dimensions to check - against the notes, another sheet, or "
    "the drawn length at the sheet's scale. source, page (0-based), "
    "optional bbox in PDF points. Proposals with a confidence: read the "
    "value by zooming before you quote it.")
TITLE_BLOCK_DESCRIPTION = (
    "Find the title block of a drawing page: its box and the fields its text "
    "layer gives (sheet number, revision, date, scale, and every line in "
    "it); on a sheet whose lettering is drawn as lines, the box to read with "
    "render_region. source, page (0-based), optional bbox to search in. A "
    "proposal with a confidence.")
CLOUDS_DESCRIPTION = (
    "Find revision clouds drawn on a drawing page (the scalloped outlines "
    "that mark changed work) and revision triangles with their numbers, "
    "each with a box to look at with render_region. Clouds a reviewer added "
    "as PDF comments are listed by document_markups, not here. source, page "
    "(0-based), optional bbox. Low-confidence proposals by design: look "
    "before you report a change.")


# ---------------------------------------------------------------------------
# Small geometry helpers (displayed frame)
# ---------------------------------------------------------------------------

def _r(v: float) -> float:
    return round(float(v), 1)


def _rp(p: Sequence[float]) -> List[float]:
    return [_r(p[0]), _r(p[1])]


def _rb(b: Sequence[float]) -> List[float]:
    return [_r(v) for v in b]


def _union(*boxes: Optional[Sequence[float]]) -> Optional[Box]:
    bs = [b for b in boxes if b is not None]
    if not bs:
        return None
    return (min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs))


def _pt_box(p: Point) -> Box:
    return (p[0], p[1], p[0], p[1])


def _pad(b: Sequence[float], d: float) -> Box:
    return (b[0] - d, b[1] - d, b[2] + d, b[3] + d)


def _inside(p: Point, b: Sequence[float]) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def _intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _box_dist(p: Point, b: Sequence[float]) -> float:
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return math.hypot(dx, dy)


def _centre(b: Sequence[float]) -> Point:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


# ---------------------------------------------------------------------------
# One ingested page
# ---------------------------------------------------------------------------

class _Sheet:
    """One page's IR, its frame, its located text lines and its scale.

    ``point``/``box`` map IR coordinates (unrotated geometry, y-flipped with
    the rotated page height) onto the displayed frame with the page's own
    rotation matrix - planlens' ``from_ir_point``.
    """

    def __init__(self, ir, width: float, height: float,
                 matrix: Tuple[float, ...], rotation: int,
                 lines: List[Tuple[str, Box]], viewports: list):
        self.ir = ir
        self.width = float(width)
        self.height = float(height)
        self.rotation = int(rotation)
        self._m = tuple(float(v) for v in matrix)
        self.lines = lines
        self.viewports = viewports
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # -- frame --------------------------------------------------------------
    def point(self, x_ir: float, y_ir: float) -> Point:
        a, b, c, d, e, f = self._m
        x, y = float(x_ir), self.height - float(y_ir)
        return (a * x + c * y + e, b * x + d * y + f)

    def box(self, bbox_ir: Sequence[float]) -> Box:
        x0, y0, x1, y1 = bbox_ir
        pts = [self.point(x0, y0), self.point(x1, y1),
               self.point(x0, y1), self.point(x1, y0)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def clamp(self, b: Sequence[float]) -> Box:
        return (max(0.0, b[0]), max(0.0, b[1]),
                min(self.width, b[2]), min(self.height, b[3]))

    # -- scale of the lettering ---------------------------------------------
    @property
    def unit(self) -> float:
        """A lettering-sized length: the median letter height of short text
        lines (the smaller side of a line's box, so text that reads up the
        page counts the same), or (no text layer) a small fraction of the
        page diagonal."""
        if "unit" not in self._cache:
            hs = sorted(min(b[2] - b[0], b[3] - b[1]) for t, b in self.lines
                        if len(t) <= 60 and b[3] > b[1] and b[2] > b[0])
            if hs:
                u = hs[len(hs) // 2]
            else:
                u = 0.012 * math.hypot(self.width, self.height)
            self._cache["unit"] = max(4.0, min(float(u), 60.0))
        return self._cache["unit"]

    @property
    def has_text(self) -> bool:
        return bool(self.lines)

    def has_linework(self) -> bool:
        return any(getattr(e, "KIND", "") in ("line", "polyline")
                   for e in self.ir.entities)

    def cached(self, name: str, compute: Callable[[], Any]) -> Any:
        with self._lock:
            if name not in self._cache:
                self._cache[name] = compute()
            return self._cache[name]

    # -- text near a spot ----------------------------------------------------
    def nearest_line(self, p: Point, within: float,
                     region: Optional[Box] = None,
                     max_chars: Optional[int] = None
                     ) -> Optional[Tuple[str, Box]]:
        best, best_d = None, None
        for text, b in self.lines:
            if max_chars is not None and len(text) > max_chars:
                continue
            if region is not None and not _inside(_centre(b), region):
                continue
            d = _box_dist(p, b)
            if d <= within and (best_d is None or d < best_d):
                best, best_d = (text, b), d
        return best

    def lines_in(self, region: Sequence[float]) -> List[Tuple[str, Box]]:
        return [(t, b) for t, b in self.lines if _inside(_centre(b), region)]


class _PageRange(ValueError):
    def __init__(self, n: int):
        super().__init__(n)
        self.n = n


_SHEETS: "OrderedDict[Tuple[str, int], _Sheet]" = OrderedDict()
_SHEETS_GUARD = threading.Lock()
_KEY_LOCKS: Dict[Tuple[str, int], threading.Lock] = {}
_PATH_DIGESTS: Dict[Tuple[str, int, int], str] = {}


def _digest(src: Any) -> str:
    """The content hash a sheet is cached under (a path's is remembered by
    its size and modification time, so a big file is read once)."""
    if isinstance(src, (bytes, bytearray)):
        return hashlib.sha256(bytes(src)).hexdigest()
    path = os.path.abspath(str(src))
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    got = _PATH_DIGESTS.get(key)
    if got is None:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        got = _PATH_DIGESTS[key] = h.hexdigest()
    return got


def _is_pdf(src: Any) -> bool:
    if isinstance(src, (bytes, bytearray)):
        head = bytes(src[:1024])
    else:
        with open(str(src), "rb") as fh:
            head = fh.read(1024)
    return b"%PDF" in head


def _viewports(doc, page, index: int) -> list:
    """The page's stored, calibrated scales (planlens >= 0.4), or none."""
    try:
        from planlens.document.scale import page_viewports
        vps, _warnings = page_viewports(doc, page, index)
        return list(vps)
    except Exception:  # noqa: BLE001 - an unreadable scale is no scale
        return []


def _ingest(src: Any, page: int) -> _Sheet:
    import fitz
    from planlens.ir import from_pdf_vector

    from funhouse_agent import vision_view

    data = bytes(src) if isinstance(src, (bytes, bytearray)) else None
    doc = (fitz.open(stream=data, filetype="pdf") if data is not None
           else fitz.open(str(src)))
    try:
        n = len(doc)
        if not 0 <= page < n:
            raise _PageRange(n)
        pg = doc[page]
        m = pg.rotation_matrix
        matrix = (m.a, m.b, m.c, m.d, m.e, m.f)
        width, height = pg.rect.width, pg.rect.height
        rotation = pg.rotation
        viewports = _viewports(doc, pg, page)
    finally:
        doc.close()
    ir = (from_pdf_vector(content=data, page=page) if data is not None
          else from_pdf_vector(filepath=str(src), page=page))
    got = vision_view.page_lines(data if data is not None else str(src), page)
    lines = list(got[0]) if got else []
    return _Sheet(ir, width, height, matrix, rotation, lines, viewports)


def _sheet(src: Any, page: int) -> _Sheet:
    key = (_digest(src), int(page))
    with _SHEETS_GUARD:
        sheet = _SHEETS.get(key)
        if sheet is not None:
            _SHEETS.move_to_end(key)
            return sheet
        lock = _KEY_LOCKS.setdefault(key, threading.Lock())
    with lock:
        with _SHEETS_GUARD:
            sheet = _SHEETS.get(key)
        if sheet is None:
            sheet = _ingest(src, int(page))
            with _SHEETS_GUARD:
                _SHEETS[key] = sheet
                while len(_SHEETS) > MAX_SHEETS:
                    old, _ = _SHEETS.popitem(last=False)
                    _KEY_LOCKS.pop(old, None)
    return sheet


def clear_cache() -> None:
    """Forget every ingested sheet (tests; a host that wants the memory)."""
    with _SHEETS_GUARD:
        _SHEETS.clear()
        _KEY_LOCKS.clear()
        _PATH_DIGESTS.clear()


def _load(source: str, page: Any, attachments: Dict[str, bytes]
          ) -> Tuple[Optional[_Sheet], Optional[Dict[str, Any]]]:
    """``(sheet, None)`` or ``(None, error result)``."""
    try:
        page = int(page)
    except (TypeError, ValueError):
        return None, {"error": f"page must be a 0-based page number, not "
                               f"{page!r}"}
    try:
        src = document_tools.resolve_document_source(source, attachments)
    except Exception as exc:  # noqa: BLE001 - planlens' ToolError and kin
        out = {"error": str(exc) or type(exc).__name__}
        hint = getattr(exc, "hint", None)
        if hint:
            out["hint"] = hint
        return None, out
    try:
        if not _is_pdf(src):
            return None, {"error": f"'{source}' is not a PDF: these tools "
                                   f"read drawing pages of a PDF"}
        return _sheet(src, page), None
    except _PageRange as exc:
        return None, {"error": f"page {page} is out of range: the document "
                               f"has {exc.n} pages (0 to {exc.n - 1})"}
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return None, {"error": f"could not read page {page} of '{source}': "
                               f"{type(exc).__name__}: {exc}"}


def _region(bbox: Any) -> Tuple[Optional[Box], Optional[Dict[str, Any]]]:
    if bbox is None or bbox == []:
        return None, None
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None, {"error": "bbox must be [x0, y0, x1, y1] in PDF points "
                               "(top-left origin, y down)"}
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), None


# ---------------------------------------------------------------------------
# The result envelope and its budget
# ---------------------------------------------------------------------------

def _envelope(page: int, sheet: _Sheet) -> Dict[str, Any]:
    return {"page": page, "pdf_page": page + 1,
            "page_size": [_r(sheet.width), _r(sheet.height)],
            "text_layer": sheet.has_text}


def _fit(out: Dict[str, Any], key: str, limit: Optional[int] = None) -> str:
    """``out`` as JSON with its ``key`` list cut (weakest last) to fit
    ``limit`` (default :data:`RESULT_CHARS`) and :data:`MAX_ITEMS`;
    ``left_out`` says how many were cut."""
    limit = RESULT_CHARS if limit is None else int(limit)
    items = list(out.get(key) or [])
    total = len(items)
    kept = items[:MAX_ITEMS]

    def dump(n: int) -> str:
        o = dict(out)
        o[key] = kept[:n]
        o["shown"] = n
        if total - n:
            o["left_out"] = total - n
            o["left_out_note"] = ("weaker proposals were left out to fit: "
                                  "narrow with bbox, or raise min_confidence")
        return json.dumps(o, ensure_ascii=False)

    n = len(kept)
    text = dump(n)
    while n > 0 and len(text) > limit:
        n = max(0, n - max(1, n // 8))
        text = dump(n)
    return text


def _no_linework(out: Dict[str, Any], key: str) -> str:
    out[key] = []
    out["count"] = 0
    out["note"] = ("No drawn line-work on this page (a scan, a photo or a "
                   "text page): there is no geometry to find. Look at it "
                   "with analyze_pdf_page.")
    return json.dumps(out, ensure_ascii=False)


# ---------------------------------------------------------------------------
# drawing_callouts
# ---------------------------------------------------------------------------

def _label_box(tail: Point, prev: Point, u: float) -> Box:
    """Where a leader's label probably sits: past the tail, along the
    shaft's last segment (a landing runs into its text), sized to the
    lettering."""
    dx, dy = tail[0] - prev[0], tail[1] - prev[1]
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    w, h = 16.0 * u, 6.0 * u
    if abs(uy) > 2.0 * abs(ux):          # steep: text above or below the end
        x0, x1 = tail[0] - w / 2.0, tail[0] + w / 2.0
        if uy > 0:                        # y down: the leader runs downward
            y0, y1 = tail[1] - 0.25 * h, tail[1] + 0.75 * h
        else:
            y0, y1 = tail[1] - 0.75 * h, tail[1] + 0.25 * h
    else:
        if ux >= 0:
            x0, x1 = tail[0] - 0.15 * w, tail[0] + 0.85 * w
        else:
            x0, x1 = tail[0] - 0.85 * w, tail[0] + 0.15 * w
        y0, y1 = tail[1] - h / 2.0, tail[1] + h / 2.0
    return (x0, y0, x1, y1)


def _before_tail(verts: List[Point], tail: Point, tip: Point) -> Point:
    if len(verts) < 2:
        return tip
    if math.dist(verts[0], tail) <= math.dist(verts[-1], tail):
        return verts[1]
    return verts[-2]


def _short(text: str, n: int = 120) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def callouts(sheet: _Sheet, page: int, region: Optional[Box],
             min_confidence: float) -> str:
    from planlens.ir import queries as q

    out = _envelope(page, sheet)
    if not sheet.has_linework():
        return _no_linework(out, "callouts")
    raw = sheet.cached("leaders", lambda: q.find_leaders(
        sheet.ir, exclude_dimensions=True))
    u = sheet.unit
    items, weaker, seen = [], 0, set()
    for p in raw:
        conf = float(p.get("confidence") or 0.0)
        tip = sheet.point(*p["tip_xy"])
        tail = sheet.point(*p["tail_xy"])
        if region is not None and not (_inside(tip, region)
                                       or _inside(tail, region)):
            continue
        verts = [sheet.point(*v) for v in (p.get("vertices") or [])]
        # A tip within a letter's height of its own tail is a stroke of a
        # drawn letter, not a callout (measured: 4-9 pt "leaders", often a
        # glyph outline that loops back, inside the lettering of a
        # stroke-plotted sheet). Held below the call threshold, not deleted
        # - the finders' own cap-not-delete rule.
        if math.dist(tip, tail) < u:
            conf = min(conf, 0.45)
        # One spanning more than a third of the sheet is a border or a long
        # outline paired with a corner shape, not an annotation leader
        # (measured: 280-720 pt "leaders" along the border of a letter-size
        # sheet whose real leaders are 25-50 pt). Same cap.
        if math.dist(tip, tail) > LONG_LEADER * min(sheet.width, sheet.height):
            conf = min(conf, 0.45)
        # The same leader proposed twice (one arrowhead read two ways).
        key = (round(tip[0]), round(tip[1]), round(tail[0]), round(tail[1]))
        if key in seen or (key[2], key[3], key[0], key[1]) in seen:
            continue
        if conf < min_confidence:
            weaker += 1
            continue
        seen.add(key)
        guess = _label_box(tail, _before_tail(verts, tail, tip), u)
        found = sheet.nearest_line(tail, within=3.0 * u, max_chars=400)
        if found is None:
            found = sheet.nearest_line(tail, within=6.0 * u, region=guess,
                                       max_chars=400)
        item: Dict[str, Any] = {"confidence": round(conf, 2),
                                "tip": _rp(tip), "tail": _rp(tail)}
        if found is not None:
            item["label"] = _short(found[0])
            label_zoom = _pad(_union(found[1], _pt_box(tail)), u)
        else:
            item["label"] = None
            label_zoom = guess
        item["label_zoom"] = _rb(sheet.clamp(label_zoom))
        item["zoom"] = _rb(sheet.clamp(_pad(
            _union(_pt_box(tip), _pt_box(tail), label_zoom), u)))
        items.append(item)
    # The caps above can move a proposal down: strongest first again.
    items.sort(key=lambda i: i["confidence"], reverse=True)
    out["count"] = len(items)
    out["callouts"] = items
    if weaker:
        out["below_min_confidence"] = weaker
    if items and any(i["label"] is None for i in items):
        out["label_note"] = (NO_TEXT_NOTE if not sheet.has_text else
                             "label null = no text-layer line at that tail: "
                             "read it by zooming on label_zoom.")
    if not items:
        out["note"] = ("No leader or callout arrow found here at this "
                       "confidence" + (f" ({weaker} weaker candidates: lower "
                                       f"min_confidence to see them)"
                                       if weaker else "")
                       + ". Look at the page to be sure: an arrow drawn "
                         "unusually is missed.")
    else:
        out["note"] = PROPOSAL_NOTE
    out["frame"] = FRAME_NOTE
    return _fit(out, "callouts")


# ---------------------------------------------------------------------------
# drawing_dimensions
# ---------------------------------------------------------------------------

def _orientation(a: Point, b: Point) -> str:
    deg = math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0])) % 180.0
    if deg <= 2.0 or deg >= 178.0:
        return "horizontal"
    if abs(deg - 90.0) <= 2.0:
        return "vertical"
    return f"{deg:.0f} deg"


def _band(a: Point, b: Point, half: float) -> Box:
    """The axis-aligned box around the strip ``half`` either side of a-b."""
    L = math.dist(a, b) or 1.0
    px, py = -(b[1] - a[1]) / L * half, (b[0] - a[0]) / L * half
    pts = [(a[0] + px, a[1] + py), (a[0] - px, a[1] - py),
           (b[0] + px, b[1] + py), (b[0] - px, b[1] - py)]
    return (min(p[0] for p in pts), min(p[1] for p in pts),
            max(p[0] for p in pts), max(p[1] for p in pts))


def _real_length(sheet: _Sheet, mid: Point, length_pt: float
                 ) -> Optional[Dict[str, str]]:
    """The length through a scale the PDF itself stores, or ``None``."""
    if not sheet.viewports:
        return None
    try:
        from planlens.document.scale import to_quantity, viewport_at
        vp = viewport_at(sheet.viewports, mid, (sheet.width, sheet.height))
        if vp is None or not vp.is_calibrated:
            return None
        if vp.y_per_point and vp.x_per_point and \
                abs(vp.y_per_point - vp.x_per_point) > 1e-9 * vp.x_per_point:
            return None                 # a different scale per axis: not ours
        qty = to_quantity(vp, length_pt)
        if not getattr(qty, "scale_known", False):
            return None
        return {"real_length": f"{qty.value:.2f} {qty.units}",
                "scale": vp.label}
    except Exception:  # noqa: BLE001 - no scale is better than a wrong one
        return None


def dimensions(sheet: _Sheet, page: int, region: Optional[Box],
               min_confidence: float) -> str:
    from planlens.ir import queries as q

    out = _envelope(page, sheet)
    if not sheet.has_linework():
        return _no_linework(out, "dimensions")
    raw = sheet.cached("dimensions", lambda: q.find_dimensions(sheet.ir))
    u = sheet.unit
    items, weaker, scaled = [], 0, 0
    for p in raw:
        conf = float(p.get("confidence") or 0.0)
        a = sheet.point(*p["end_a_xy"])
        b = sheet.point(*p["end_b_xy"])
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        if region is not None and not (_inside(mid, region)
                                       or _inside(a, region)
                                       or _inside(b, region)):
            continue
        if conf < min_confidence:
            weaker += 1
            continue
        length = math.dist(a, b)
        zoom = _band(a, b, max(3.0 * u, 0.15 * length))
        found = sheet.nearest_line(mid, within=max(3.0 * u, 0.6 * length),
                                   region=zoom, max_chars=40)
        item: Dict[str, Any] = {"confidence": round(conf, 2),
                                "end_a": _rp(a), "end_b": _rp(b),
                                "length_pt": _r(length),
                                "orientation": _orientation(a, b)}
        if found is not None:
            item["text"] = _short(found[0], 60)
            zoom = _union(zoom, found[1])
        elif p.get("text") and sheet.has_text:
            item["text"] = _short(p["text"], 60)
        else:
            item["text"] = None
        item["value_zoom"] = _rb(sheet.clamp(_pad(zoom, u)))
        real = _real_length(sheet, mid, length)
        if real:
            item.update(real)
            scaled += 1
        items.append(item)
    out["count"] = len(items)
    out["dimensions"] = items
    if weaker:
        out["below_min_confidence"] = weaker
    if items and any(i["text"] is None for i in items):
        out["text_note"] = (NO_TEXT_NOTE if not sheet.has_text else
                            "text null = no text-layer value near that "
                            "line: read it by zooming on value_zoom.")
    out["scale"] = (
        "a scale stored in the PDF gives real_length" if scaled else
        "no calibrated scale is stored in this PDF for these lines: "
        "length_pt is the drawn length in points, not a real length - read "
        "the dimension's own value")
    if not items:
        out["note"] = ("No dimension line found here at this confidence"
                       + (f" ({weaker} weaker candidates: lower "
                          f"min_confidence to see them)" if weaker else "")
                       + ". Look at the page to be sure.")
    else:
        out["note"] = PROPOSAL_NOTE
    out["frame"] = FRAME_NOTE
    return _fit(out, "dimensions")


# ---------------------------------------------------------------------------
# title_block
# ---------------------------------------------------------------------------

def _rect_like(e) -> Optional[Box]:
    """The IR bbox of a rectangle-like ring (closed 4-5 vertices, or the open
    4-corner polyline a PDF ``re`` ingests as), else ``None``."""
    from planlens.ir.results import Polyline
    if not isinstance(e, Polyline) or e.bbox is None:
        return None
    n = len(e.vertices)
    if not ((e.closed and 4 <= n <= 5) or (not e.closed and n == 4)):
        return None
    x0, y0, x1, y1 = e.bbox
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return None
    verts = [tuple(v) for v in e.vertices]
    s = sum(a[0] * b[1] - b[0] * a[1]
            for a, b in zip(verts, verts[1:] + verts[:1]))
    if abs(s) * 0.5 / (bw * bh) < 0.8:
        return None
    return tuple(e.bbox)


def _segments(sheet: _Sheet) -> List[Tuple[Point, Point]]:
    """Every straight piece of line-work, in the displayed frame."""
    from planlens.ir.results import Line, Polyline
    segs: List[Tuple[Point, Point]] = []
    for e in sheet.ir.entities:
        if isinstance(e, Line):
            segs.append((sheet.point(*e.start), sheet.point(*e.end)))
        elif isinstance(e, Polyline):
            pts = [sheet.point(*v) for v in e.vertices]
            if e.closed and len(pts) > 2:
                pts.append(pts[0])
            segs.extend(zip(pts, pts[1:]))
    return segs


#: Title blocks sit along the right or bottom edge; a sheet drawn sideways
#: on an upright page puts its own bottom on the displayed left.
_EDGE_PRIOR = {"bottom": 1.0, "right": 1.0, "left": 0.7}


def _strips(sheet: _Sheet, segs) -> List[Dict[str, Any]]:
    """Edge strips: the band between the sheet border (or the page edge)
    and the next long line parallel to it - how a title block is drawn when
    it is not one closed rectangle."""
    W, H = sheet.width, sheet.height
    rows: Dict[str, List[Tuple[float, float, float]]] = {
        "bottom": [], "right": [], "left": []}
    for a, b in segs:
        dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
        if dy <= 0.01 * dx and dx >= 0.4 * W:
            y = (a[1] + b[1]) / 2.0
            if y >= 0.65 * H:
                rows["bottom"].append((H - y, min(a[0], b[0]),
                                       max(a[0], b[0])))
        elif dx <= 0.01 * dy and dy >= 0.4 * H:
            x = (a[0] + b[0]) / 2.0
            lo, hi = min(a[1], b[1]), max(a[1], b[1])
            if x >= 0.65 * W:
                rows["right"].append((W - x, lo, hi))
            elif x <= 0.35 * W:
                rows["left"].append((x, lo, hi))
    out = []
    for edge, found in rows.items():
        if not found:
            continue
        dim = H if edge == "bottom" else W
        found.sort()
        clusters: List[List[float]] = []          # [pos, lo, hi]
        for pos, lo, hi in found:
            if clusters and pos - clusters[-1][0] <= 0.01 * dim:
                c = clusters[-1]
                c[1], c[2] = min(c[1], lo), max(c[2], hi)
            else:
                clusters.append([pos, lo, hi])
        # The sheet border: the outermost long line within a plot margin
        # (measured 6-9 % of a letter page's short side on the public
        # sheets); else the page edge itself. A band narrower than 5 % is a
        # table's header row, not a title block.
        border = clusters[0][0] if clusters[0][0] <= 0.10 * dim else 0.0
        inner = [c for c in clusters
                 if 0.05 * dim <= c[0] - border and c[0] <= 0.35 * dim]
        if not inner:
            continue
        pos, lo, hi = inner[0]
        if edge == "bottom":
            box = (lo, H - pos, hi, H - border)
        elif edge == "right":
            box = (W - pos, lo, W - border, hi)
        else:
            box = (border, lo, pos, hi)
        out.append({"box": box, "edge": edge, "path": "edge_strip"})
    return out


def _cells(box: Box, segs, edge: str) -> int:
    """Dividers inside a strip: lines across it spanning most of its width."""
    x0, y0, x1, y1 = box
    across_h = edge in ("right", "left")        # dividers run horizontally
    span = (x1 - x0) if across_h else (y1 - y0)
    n = 0
    for a, b in segs:
        mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
        if not (x0 < mx < x1 and y0 < my < y1):
            continue
        dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
        if across_h and dy <= 0.05 * dx and dx >= 0.5 * span:
            n += 1
        elif not across_h and dx <= 0.05 * dy and dy >= 0.5 * span:
            n += 1
    return n


def _title_candidates(sheet: _Sheet) -> List[Dict[str, Any]]:
    W, H = sheet.width, sheet.height
    area = W * H
    segs = _segments(sheet)
    rects = [sheet.box(bb) for bb in
             (_rect_like(e) for e in sheet.ir.entities) if bb is not None]
    cands: List[Dict[str, Any]] = []
    for bb in rects:
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if not 0.005 * area <= w * h <= 0.40 * area:
            continue
        d = {"right": (W - bb[2]) / W, "bottom": (H - bb[3]) / H,
             "left": bb[0] / W}
        in_band = {"right": bb[0] >= 0.6 * W, "bottom": bb[1] >= 0.6 * H,
                   "left": bb[2] <= 0.4 * W}
        edges = [e for e in d if in_band[e] and d[e] <= 0.05]
        if not edges:
            continue
        edge = min(edges, key=lambda e: (d[e], -_EDGE_PRIOR[e]))
        nested = sum(1 for r in rects if r is not bb
                     and bb[0] <= r[0] and bb[1] <= r[1]
                     and r[2] <= bb[2] and r[3] <= bb[3]
                     and (r[2] - r[0]) * (r[3] - r[1]) < 0.95 * w * h)
        # 1.0 on the edge, 0.5 at the 5 % limit of the band.
        cands.append({"box": bb, "edge": edge, "path": "rectangle",
                      "edge_score": _EDGE_PRIOR[edge]
                      * max(0.0, 1.0 - 10.0 * d[edge]),
                      "cells": nested})
    for s in _strips(sheet, segs):
        bb = s["box"]
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if not 0.005 * area <= w * h <= 0.40 * area:
            continue
        s["edge_score"] = _EDGE_PRIOR[s["edge"]]
        s["cells"] = _cells(bb, segs, s["edge"])
        cands.append(s)
    for c in cands:
        n_text = len(sheet.lines_in(c["box"]))
        text_score = min(1.0, n_text / 6.0)
        cell_score = min(1.0, c["cells"] / (4.0 if c["path"] == "rectangle"
                                            else 3.0))
        edge = max(0.0, min(1.0, c["edge_score"]))
        if sheet.has_text:
            raw = 0.35 * edge + 0.35 * text_score + 0.30 * cell_score
            conf = raw if n_text else min(raw, 0.3)
        else:
            # No text to corroborate it: placed from the line-work alone.
            raw = (0.35 * edge + 0.30 * cell_score) / 0.65
            conf = min(0.45, raw)
        if c["path"] == "edge_strip":
            conf = min(conf, 0.6)
        c["confidence"] = round(conf, 2)
        c["raw"] = raw
        c["n_text"] = n_text
    cands.sort(key=lambda c: (c["confidence"], c["raw"]), reverse=True)
    return cands


_VALUE = r"([A-Z]{0,4}[-.]?\d[\w.\-/]*)"
_FIELD_RES = {
    "sheet": re.compile(r"\b(?:sheet|dwg|drawing|std\.?)\s*(?:no\.?|number|#)?"
                        r"\s*[:#]?\s*" + _VALUE, re.IGNORECASE),
    "revision": re.compile(r"\brev(?:ision)?\b\.?\s*(?:no\.?)?\s*[:#]?\s*"
                           r"([A-Z0-9][\w.\-/]*)", re.IGNORECASE),
    "scale": re.compile(r"\bscale\b\s*[:=]?\s*(\S.*)", re.IGNORECASE),
}
_DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+"
    r"(?:\d{1,2},?\s+)?\d{4})\b", re.IGNORECASE)
_LABEL_ONLY = {
    "sheet": re.compile(r"^\s*(?:sheet|dwg|drawing|std\.?)\s*(?:no\.?|number|"
                        r"#)?\s*[:#]?\s*$", re.IGNORECASE),
    "revision": re.compile(r"^\s*rev(?:ision)?\.?\s*(?:no\.?)?\s*[:#]?\s*$",
                           re.IGNORECASE),
    "date": re.compile(r"^\s*date\s*[:#]?\s*$", re.IGNORECASE),
    "scale": re.compile(r"^\s*scale\s*[:#]?\s*$", re.IGNORECASE),
}


def _below(label: Box, lines: List[Tuple[str, Box]]) -> Optional[str]:
    """The line just under (or right of) a label with no value of its own."""
    h = max(label[3] - label[1], 1.0)
    best, best_d = None, None
    for text, b in lines:
        if b is label:
            continue
        under = (b[1] >= label[1] + 0.5 * h and b[1] - label[3] <= 2.0 * h
                 and not (b[2] < label[0] or b[0] > label[2]))
        right = (b[0] >= label[2] - 0.5 * h and b[0] - label[2] <= 3.0 * h
                 and abs(_centre(b)[1] - _centre(label)[1]) <= 0.6 * h)
        if under or right:
            d = math.dist(_centre(b), _centre(label))
            if best_d is None or d < best_d:
                best, best_d = text, d
    return best


def _fields(lines: List[Tuple[str, Box]]) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for text, b in lines:
        for name, rx in _LABEL_ONLY.items():
            if name not in fields and rx.match(text):
                v = _below(b, lines)
                if v and not any(r.match(v) for r in _LABEL_ONLY.values()):
                    fields[name] = _short(v, 60)
        for name, rx in _FIELD_RES.items():
            if name in fields:
                continue
            m = rx.search(text)
            if m:
                fields[name] = _short(m.group(1), 60)
        if "date" not in fields:
            m = _DATE_RE.search(text)
            if m:
                fields["date"] = m.group(1)
        if "scale" not in fields and re.search(r"\bnot\s+to\s+scale\b|\bNTS\b",
                                               text, re.IGNORECASE):
            fields["scale"] = "not to scale"
    return fields


def title_block(sheet: _Sheet, page: int, region: Optional[Box]) -> str:
    out = _envelope(page, sheet)
    if not sheet.has_linework():
        out["title_block"] = None
        out["note"] = ("No drawn line-work on this page (a scan, a photo or "
                       "a text page): look at it with analyze_pdf_page.")
        return json.dumps(out, ensure_ascii=False)
    cands = sheet.cached("title_block", lambda: _title_candidates(sheet))
    if region is not None:
        cands = [c for c in cands if _intersects(c["box"], region)]
    u = sheet.unit
    if not cands:
        out["title_block"] = None
        out["note"] = ("No title block found from the line-work (it is "
                       "usually along the right or bottom edge): look at "
                       "the page with analyze_pdf_page.")
        out["frame"] = FRAME_NOTE
        return json.dumps(out, ensure_ascii=False)
    best = cands[0]
    lines = sorted(sheet.lines_in(best["box"]),
                   key=lambda tb: (round(tb[1][1] / max(u, 1.0)), tb[1][0]))
    block: Dict[str, Any] = {"confidence": best["confidence"],
                             "box": _rb(best["box"]),
                             "zoom": _rb(sheet.clamp(_pad(best["box"], u))),
                             "found_as": ("a rectangle" if best["path"] ==
                                          "rectangle" else
                                          f"a strip along the {best['edge']} "
                                          f"edge")}
    if lines:
        block["fields"] = _fields(lines)
        shown, used = [], 0
        for text, _b in lines:
            t = _short(text, 160)
            if used + len(t) > 3000:
                break
            shown.append(t)
            used += len(t)
        block["lines"] = shown
        if len(shown) < len(lines):
            block["lines_left_out"] = len(lines) - len(shown)
    else:
        block["fields"] = {}
        block["fields_note"] = (
            NO_TEXT_NOTE if not sheet.has_text else
            "no text-layer lines inside this box: read the sheet number, "
            "title, date and revision by zooming on zoom.")
    out["title_block"] = block
    others = [{"confidence": c["confidence"], "box": _rb(c["box"])}
              for c in cands[1:3]]
    if others:
        out["other_candidates"] = others
    out["count"] = len(cands)
    out["note"] = PROPOSAL_NOTE
    if not sheet.has_text:
        out["note"] += (" With no text layer the title block is placed from "
                        "the line-work alone: check the box by zooming; "
                        "other_candidates are the next guesses.")
    out["frame"] = FRAME_NOTE
    return json.dumps(out, ensure_ascii=False)


# ---------------------------------------------------------------------------
# revision_clouds
# ---------------------------------------------------------------------------

def clouds(sheet: _Sheet, page: int, region: Optional[Box],
           min_confidence: float) -> str:
    from planlens.ir import queries as q

    out = _envelope(page, sheet)
    if not sheet.has_linework():
        return _no_linework(out, "clouds")
    raw = sheet.cached("clouds", lambda: q.find_revision_clouds(sheet.ir))
    u = sheet.unit
    cloud_items, tags, weaker = [], [], 0
    for p in raw:
        conf = float(p.get("confidence") or 0.0)
        if p.get("kind") == "cloud" and p.get("bbox"):
            box = sheet.box(p["bbox"])
            if region is not None and not _intersects(box, region):
                continue
            if conf < min_confidence:
                weaker += 1
                continue
            cloud_items.append({"confidence": round(conf, 2), "box": box})
        elif p.get("kind") == "revision_delta" and p.get("center_xy"):
            at = sheet.point(*p["center_xy"])
            if region is not None and not _inside(at, region):
                continue
            tags.append({"confidence": round(conf, 2), "text": p.get("text"),
                         "at": at})
    free_tags = []
    for t in tags:
        near = None
        for c in cloud_items:
            b = c["box"]
            reach = 0.25 * math.hypot(b[2] - b[0], b[3] - b[1]) + 4.0 * u
            if _box_dist(t["at"], b) <= reach and (
                    near is None or _box_dist(t["at"], b)
                    < _box_dist(t["at"], near["box"])):
                near = c
        if near is not None and "tag" not in near:
            near["tag"] = {"text": t["text"], "at": _rp(t["at"])}
        else:
            free_tags.append({"confidence": t["confidence"],
                              "text": t["text"], "at": _rp(t["at"]),
                              "zoom": _rb(sheet.clamp(_pad(
                                  _pt_box(t["at"]), 4.0 * u)))})
    items = []
    for c in cloud_items:
        item = {"confidence": c["confidence"], "box": _rb(c["box"]),
                "zoom": _rb(sheet.clamp(_pad(c["box"], 2.0 * u)))}
        if "tag" in c:
            item["tag"] = c["tag"]
        items.append(item)
    out["count"] = len(items)
    out["clouds"] = items
    if free_tags:
        out["revision_tags"] = free_tags[:10]
    if weaker:
        out["below_min_confidence"] = weaker
    out["note"] = (PROPOSAL_NOTE if items or free_tags else
                   "No revision cloud found in the line-work"
                   + (f" ({weaker} weaker candidates: lower min_confidence "
                      f"to see them)" if weaker else "")
                   + ". Clouds added as PDF comments are listed by "
                     "document_markups; look at the page to be sure.")
    out["frame"] = FRAME_NOTE
    return _fit(out, "clouds")


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------

def _conf(v: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def make_geometry_tools(attachments: Optional[Dict[str, bytes]] = None
                        ) -> list:
    """``drawing_callouts``, ``drawing_dimensions``, ``title_block`` and
    ``revision_clouds`` bound to this agent's uploads (the live dict the
    host mutates). All four only read."""
    from langchain_core.tools import StructuredTool

    attachments = {} if attachments is None else attachments

    def run(source: str, page: Any, bbox: Any, body) -> str:
        region, err = _region(bbox)
        if err:
            return json.dumps(err)
        sheet, err = _load(source, page, attachments)
        if err:
            return json.dumps(err)
        try:
            return body(sheet, int(page), region)
        except Exception as exc:  # noqa: BLE001 - a finder failure is a result
            return json.dumps({"error": f"{type(exc).__name__}: {exc}",
                               "hint": "look at the page with "
                                       "analyze_pdf_page instead"})

    def drawing_callouts(source: str, page: int = 0,
                         bbox: Optional[List[float]] = None,
                         min_confidence: float = DEFAULT_MIN_CONFIDENCE
                         ) -> str:
        mc = _conf(min_confidence, DEFAULT_MIN_CONFIDENCE)
        return run(source, page, bbox,
                   lambda s, p, r: callouts(s, p, r, mc))

    def drawing_dimensions(source: str, page: int = 0,
                           bbox: Optional[List[float]] = None,
                           min_confidence: float = DEFAULT_MIN_CONFIDENCE
                           ) -> str:
        mc = _conf(min_confidence, DEFAULT_MIN_CONFIDENCE)
        return run(source, page, bbox,
                   lambda s, p, r: dimensions(s, p, r, mc))

    def title_block_tool(source: str, page: int = 0,
                         bbox: Optional[List[float]] = None) -> str:
        return run(source, page, bbox, title_block)

    def revision_clouds(source: str, page: int = 0,
                        bbox: Optional[List[float]] = None,
                        min_confidence: float = CLOUD_MIN_CONFIDENCE) -> str:
        mc = _conf(min_confidence, CLOUD_MIN_CONFIDENCE)
        return run(source, page, bbox,
                   lambda s, p, r: clouds(s, p, r, mc))

    return [
        StructuredTool.from_function(drawing_callouts,
                                     name="drawing_callouts",
                                     description=CALLOUTS_DESCRIPTION),
        StructuredTool.from_function(drawing_dimensions,
                                     name="drawing_dimensions",
                                     description=DIMENSIONS_DESCRIPTION),
        StructuredTool.from_function(title_block_tool, name="title_block",
                                     description=TITLE_BLOCK_DESCRIPTION),
        StructuredTool.from_function(revision_clouds,
                                     name="revision_clouds",
                                     description=CLOUDS_DESCRIPTION),
    ]


__all__ = ["make_geometry_tools", "GEOMETRY_TOOLS", "RESULT_CHARS",
           "CALLOUTS_DESCRIPTION", "DIMENSIONS_DESCRIPTION",
           "TITLE_BLOCK_DESCRIPTION", "CLOUDS_DESCRIPTION", "clear_cache"]

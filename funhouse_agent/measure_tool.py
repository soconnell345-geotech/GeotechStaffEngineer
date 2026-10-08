"""The ``measure`` and ``log_grid`` agent tools: planlens' scales, for the app.

WHY. A position read off an image by eye is approximate. On the 2026-10-06
field session the agent read every printed number on ten scanned log sheets
right and placed 31 layer boundaries a median 0.20 m off, by eye against the
depth scale, where code measures the same lines to about +/-0.02 m (design
``module_work/VISUAL_SCALES_DESIGN.md``, E4). Geometry says where, the model
says what.

WHAT. planlens' ``measure`` and ``log_grid`` (``planlens.tools``) do the
geometry and call no model. This module connects them to the app the way
:mod:`funhouse_agent.document_tools` connects the other document tools, and
adds the two things only the app can do:

* **A look's location.** ``measure`` takes the ``view`` + ``image_box`` pair
  every vision result carries (the same pair ``render_region`` and
  ``annotate_document`` take), converted in code
  (:func:`funhouse_agent.vision_view.image_box_to_page`), with the search
  window padded by that view's location error
  (:func:`funhouse_agent.vision_view.location_error`) — so a box off a whole
  sheet lists its candidates instead of choosing, and a box off a zoom snaps.
* **The labels' values on a scan with no text.** When a scale waits for
  them, ONE vision side call reads numbered crops of its printed labels
  (:mod:`funhouse_agent.scale_labels`) and the call is made again with them
  — the agent never sees the round trip. The values are the labels AS
  PRINTED, so a reading is never written finer than the print.

The tools are optional (``document_tools.OPTIONAL_DOCUMENT_TOOL_NAMES``) and
feature-detected on the installed planlens; they describe themselves and no
prompt mentions them (owner's rule).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: What the model is told ``measure`` does (design §6.2, the app's text).
MEASURE_DESCRIPTION = (
    "Measure a position on a page through the page's own scale: a log's "
    "depth or elevation ruler, a plot's axes (linear or log), a profile's "
    "stations and elevations, a plan's scale bar, stated scale or the scale "
    "stored in the PDF. A position read off an image by eye is approximate; "
    "this finds the drawn thing itself (a line, a symbol, a curve) and "
    "converts its position, returning the value +/- its uncertainty, the "
    "scale it used and what it snapped to. Call it with only source and "
    "page to list the scales on the page. To measure, give where the thing "
    "is - a bbox in PDF points, or a look's view + the thing's image_box "
    "(from a zoom, so the box is close) - and kind: line (a drawn "
    "boundary), lines (every line in the box, e.g. all layer boundaries in "
    "a log's description column), point (a symbol or marker), edge (side: "
    "top, bottom, left or right of the ink), curve with at (a curve's value "
    "at an axis value, e.g. {\"x\": 0.3}), or distance with to (a second "
    "box, on the same view when view is given). When two things fit the "
    "box it lists both instead of choosing; with no scale on the page it "
    "returns points and says so. source is the file (an attachment name or "
    "path) or an open_document handle; page is 0-based.")

#: Said in a result when the box came from a look.
LOCATED_NOTE = ("the box was read off a {w:.0f} x {h:.0f} pt view and "
                "searched within {pad:.0f} pt of it (that view's location "
                "error){more}")


def _json_error(msg: str, hint: str = "") -> str:
    out = {"error": msg}
    if hint:
        out["hint"] = hint
    return json.dumps(out)


def _entry_for(source: str, attachments: Optional[Dict[str, bytes]]):
    """The toolkit's open entry for a handle, an attachment key or a path."""
    from funhouse_agent import document_tools as dt
    entry = dt.document_entry(source) if source else None
    if entry is not None:
        return entry
    return dt.open_document_entry(source, attachments)


def _box(v: Any, name: str) -> List[float]:
    try:
        b = [float(x) for x in v]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be four numbers [x0, y0, x1, y1]")
    if len(b) != 4:
        raise ValueError(f"{name} must be four numbers [x0, y0, x1, y1]")
    return b


def _window(box: Sequence[float], pad: Optional[float]
            ) -> Tuple[float, float, float, float]:
    """The search window planlens' ``measure`` builds (its own rule), so the
    scales looked up here are the ones it measured with."""
    from planlens.document.measuring import default_pad
    x0, x1 = sorted((float(box[0]), float(box[2])))
    y0, y1 = sorted((float(box[1]), float(box[3])))
    nb = (x0, y0, x1, y1)
    p = float(pad) if pad is not None else default_pad(nb)
    return (x0 - p, y0 - p, x1 + p, y1 + p)


def _pending(entry, page: int, near=None):
    """The scales on ``page`` waiting for their label values (the analysis
    ``measure`` used: same page, same window — so the same ids)."""
    from planlens.document.scalefinder import find_scales
    with entry.lock:
        ps = find_scales(entry.doc, int(page), near=near)
    return ps.needing_values()


def _read_values(entry, page: int, scales, engine
                 ) -> Tuple[Dict[str, List[Optional[str]]], Dict[str, Any]]:
    """``{scale_id: [label as printed or None, ...]}`` for ``scales``, and
    what reading them cost."""
    from funhouse_agent import scale_labels
    items: List[Tuple[int, Sequence[float]]] = []
    spans: List[Tuple[str, int, int]] = []
    for sc in scales:
        boxes = list(getattr(sc, "label_boxes", None) or [])
        spans.append((sc.id, len(items), len(items) + len(boxes)))
        items.extend((int(page), b) for b in boxes)
    with entry.lock:
        texts, info = scale_labels.read_labels(entry.doc, items, engine,
                                               doc_key=entry.handle)
    values = {sid: texts[a:b] for sid, a, b in spans}
    return values, info


def _labels_block(values: Dict[Any, List[Optional[str]]],
                  info: Dict[str, Any]) -> Dict[str, Any]:
    from funhouse_agent import scale_labels
    return {
        "read_by": scale_labels.describe(info),
        "values": {str(k): ", ".join("?" if v is None else v for v in vals)
                   for k, vals in values.items()},
        "note": ("label values read by vision off the scan, as printed; the "
                 "positions are the drawing's. A misread label breaks the "
                 "even run and is left out (one) or the scale refused "
                 "(two)"),
    }


def run_measure(arguments: Dict[str, Any],
                attachments: Optional[Dict[str, bytes]] = None,
                engine: Any = None, *, max_chars: Optional[int] = None,
                cap: Optional[int] = None) -> str:
    """One ``measure`` call: a JSON string within the document tools' budget.

    ``arguments``: ``source`` (or ``handle``), ``page``, ``kind``, and where
    — ``bbox`` (PDF points) or ``view`` + ``image_box`` — plus ``at``,
    ``to``, ``scale``, ``side`` as planlens takes them.
    """
    from funhouse_agent import document_tools as dt
    from funhouse_agent import vision_view
    from planlens.tools import ToolError

    source = str(arguments.get("source") or arguments.get("handle")
                 or arguments.get("attachment_key") or "")
    try:
        page = int(arguments.get("page", 0) or 0)
    except (TypeError, ValueError):
        return _json_error("page must be a whole number (0-based)")
    kind = str(arguments.get("kind") or "line")
    bbox, view, image_box = (arguments.get("bbox"), arguments.get("view"),
                             arguments.get("image_box"))
    to = arguments.get("to")
    pad: Optional[float] = None
    located = None
    try:
        if view is not None or image_box is not None:
            if bbox is not None:
                return _json_error("pass bbox OR view + image_box, not both")
            if view is None or image_box is None:
                return _json_error(
                    "view and image_box go together",
                    "view = the 'view' of an earlier vision result; "
                    "image_box = the thing's 0-999 box on that image")
            v = _box(view, "view")
            where = list(vision_view.image_box_to_page(v, _box(image_box,
                                                               "image_box")))
            ex, ey = vision_view.location_error(v)
            pad = round(max(ex, ey), 2)
            if to is not None:
                to = list(vision_view.image_box_to_page(v, _box(to, "to")))
            located = (v, pad)
        elif bbox is not None:
            where = _box(bbox, "bbox")
            if to is not None:
                to = _box(to, "to")
        else:
            where = None
    except ValueError as exc:
        return _json_error(f"{exc}",
                           "bbox is [x0, y0, x1, y1] in PDF points; or pass "
                           "a vision result's view + the thing's image_box")
    try:
        entry = _entry_for(source, attachments)
    except ToolError as exc:
        return _json_error(str(exc), getattr(exc, "hint", "") or "")

    args: Dict[str, Any] = {"handle": entry.handle, "page": page,
                            "kind": kind}
    if where is not None:
        args["bbox"] = [round(float(x), 2) for x in where]
    if pad is not None:
        args["pad"] = pad
    for k in ("at", "scale", "side"):
        if arguments.get(k) not in (None, ""):
            args[k] = arguments[k]
    if to is not None:
        args["to"] = [round(float(x), 2) for x in to]

    def call(extra: Optional[Dict[str, Any]] = None) -> str:
        return dt.dispatch_document_tool("measure", {**args, **(extra or {})},
                                         attachments=attachments,
                                         max_chars=max_chars, cap=cap)

    raw = call()
    labels = None
    if "needs_values" in raw and not raw.lstrip().startswith('{"error"'):
        near = _window(args["bbox"], pad) if where is not None else None
        try:
            pend = _pending(entry, page, near)
        except Exception:            # noqa: BLE001 - the first answer stands
            pend = []
        if pend and engine is not None:
            values, info = _read_values(entry, page, pend, engine)
            if any(t is not None for vals in values.values() for t in vals):
                raw = call({"values": values})
            labels = _labels_block(values, info)
        elif pend:
            labels = {"note": "the labels of this scale have no text and no "
                              "vision engine is available here to read them"}
    try:
        out = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(out, dict) or "error" in out:
        return raw
    if located is not None:
        v, p = located
        more = (" - a box off a view this wide lists candidates rather than "
                "choosing one; zoom on the thing and measure from the zoom's "
                "view for one answer" if p >= 25.0 else "")
        out["located_from"] = {
            "view": [round(x, 1) for x in v],
            "note": LOCATED_NOTE.format(w=v[2] - v[0], h=v[3] - v[1], pad=p,
                                        more=more)}
    if labels is not None:
        out["labels_read"] = labels
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def log_grid_description() -> str:
    """planlens' own words for ``log_grid``, fitted to this app: no pointer
    at a tool the agent does not have, and the labels read for it."""
    from funhouse_agent import document_tools as dt
    text = dt.tool_description("log_grid")
    text = text.replace(
        " (its continuation sheets included, as document_roles groups "
        "them)", " (its continuation sheets included)")
    cut = text.find(" A scanned log is read from its pixels")
    if cut > 0:
        text = text[:cut]
    return text + (
        " A scanned log is read from its pixels: its ruled columns and every "
        "stratum line, each layer top with its +/-; where the scan has no "
        "text, its depth labels are read for you (one small vision call "
        "over numbered crops of them). pages are 0-based.")


def run_log_grid(arguments: Dict[str, Any],
                 attachments: Optional[Dict[str, bytes]] = None,
                 engine: Any = None, *, max_chars: Optional[int] = None,
                 cap: Optional[int] = None) -> str:
    """One ``log_grid`` call; a scan's depth labels read when they must be."""
    from funhouse_agent import document_tools as dt
    from funhouse_agent import scale_labels

    args = {k: arguments[k] for k in ("handle", "pages", "rows", "offset")
            if arguments.get(k) not in (None, "")}

    def call(extra: Optional[Dict[str, Any]] = None) -> str:
        return dt.dispatch_document_tool("log_grid", {**args, **(extra or {})},
                                         attachments=attachments,
                                         max_chars=max_chars, cap=cap)

    raw = call()
    try:
        out = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(out, dict) or not out.get("needs_values"):
        return raw
    entry = dt.document_entry(str(args.get("handle") or ""))
    if entry is None or engine is None:
        out["labels_read"] = {"note": "the depth labels of these pages have "
                                      "no text and no vision engine is "
                                      "available here to read them"}
        return json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    items: List[Tuple[int, Sequence[float]]] = []
    spans: List[Tuple[int, int, int]] = []
    for pg, nv in sorted(out["needs_values"].items(), key=lambda kv: int(kv[0])):
        boxes = list(nv.get("labels") or [])
        spans.append((int(pg), len(items), len(items) + len(boxes)))
        items.extend((int(pg), b) for b in boxes)
    with entry.lock:
        texts, info = scale_labels.read_labels(entry.doc, items, engine,
                                               doc_key=entry.handle)
    values = {str(pg): texts[a:b] for pg, a, b in spans}
    if any(t is not None for t in texts):
        raw = call({"values": values})
        try:
            out = json.loads(raw)
        except ValueError:
            return raw
        if not isinstance(out, dict):
            return raw
    out["labels_read"] = _labels_block(
        {f"page {pg}": vals for pg, vals in values.items()}, info)
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


__all__ = ["run_measure", "run_log_grid", "log_grid_description",
           "MEASURE_DESCRIPTION"]

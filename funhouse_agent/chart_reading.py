"""Code's reading of a reference chart, beside the vision read-off.

``read_reference_figure`` has always answered by eye: one vision side call
reads a value off a catalogued design chart, "accurate to a few percent on
linear axes, looser on log axes". The W3 design (``module_work/
VISUAL_SCALES_DESIGN.md`` §5.4, §6.4, E8) adds a SECOND VOTER from the
drawing itself, with planlens' scales (numpy and PyMuPDF; no OpenCV):

1. the side call, besides its answer, writes one ``READ`` line per curve it
   read — the axis value it read at, the curve's label, the value it read,
   and a pixel box where that curve crosses there (:data:`READ_INSTRUCTION`);
2. code fits the chart's axes from its gridlines and printed labels
   (``planlens.document.scalefinder.find_scales``; on an image with no text
   the labels are read by one more small vision call over numbered crops,
   :mod:`funhouse_agent.scale_labels`), finds the curve itself in the drawing
   where it crosses that axis value near the box, and converts its position
   (``planlens.document.measuring.measure(kind="curve")``);
3. both are returned: code's value with its +/-, the vision value beside it,
   and a flag where they differ by more than about 3x code's +/- (owner
   decision 4, 2026-10-08). Where the vision interpolated between two
   labelled curves (an ``ANSWER`` line), code interpolates between its own
   two readings, linearly in the curve parameter, and reports both.

Code never chooses between curves: where several cross near the box it lists
their values and gives none (design E5). Where it finds no curve, no scale,
or cannot read the labels, it says so and the vision value stands alone.

``GEOTECH_CHART_CODE_VOTER=0`` turns the second voter off (an arm for the
suite; on by default).
"""

from __future__ import annotations

import math
import os
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

VOTER_ENV = "GEOTECH_CHART_CODE_VOTER"

#: Appended to the chart read-off prompt: what code needs to measure the same
#: thing. Boxes in pixels of the image, as every vision prompt asks.
READ_INSTRUCTION = (
    "\n\nAFTER your answer, for EACH value you read off a curve, add one line "
    "at the very end, exactly in this form:\n"
    "READ | <x or y: the axis your input value is on> = <the input value> | "
    "curve = <the label of the curve you read, or -> | value = <the value "
    "you read> | px=[x0, y0, x1, y1]\n"
    "where px= is a small box (about 20 pixels across) in PIXELS of this "
    "image, centred on the point where that curve crosses the input value. "
    "If you interpolated between two labelled curves, give a READ line for "
    "each of the two and then one line:\n"
    "ANSWER | curve = <the curve value you interpolated for> | value = <the "
    "interpolated value>")

#: A box read off a whole page in PIXELS was measured 1-6 pt from the thing
#: on an 11 x 17 sheet (2026-10-07, 2,048 px): about 0.8 % of the view's
#: longer side. The window code searches in is that, at least 3 pt — the
#: same rule planlens' toolkit uses for a box read off one of its renders.
PX_BOX_ERROR_FRAC = 0.008

#: Code's value and vision's disagree when they differ by more than this
#: many times code's +/- ("about 3x", owner decision 4).
DISAGREE_FACTOR = 3.0

_NUM = r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"
_PX = re.compile(rf"px\s*[=:]?\s*[\[(]\s*({_NUM})\s*,\s*({_NUM})\s*,\s*"
                 rf"({_NUM})\s*,\s*({_NUM})\s*[\])]", re.I)

_DOCS: "OrderedDict[str, Any]" = OrderedDict()
_DOCS_LOCK = threading.Lock()
_DOCS_MAX = 4
#: One code reading at a time: the cached documents are not thread-safe, and
#: the references helper can run read-offs in parallel.
_READ_LOCK = threading.Lock()


def voter_on() -> bool:
    """Whether code reads the chart too (``GEOTECH_CHART_CODE_VOTER``)."""
    return (os.environ.get(VOTER_ENV) or "1").strip().lower() not in (
        "0", "off", "false", "no", "none")


def with_read_lines(prompt: str) -> str:
    """The chart prompt with :data:`READ_INSTRUCTION` appended."""
    return prompt.rstrip() + READ_INSTRUCTION


def _num(s: str) -> Optional[float]:
    try:
        return float(str(s).strip().replace(",", ""))
    except (TypeError, ValueError):
        m = re.search(_NUM, str(s or ""))
        if not m:
            return None
        try:
            return float(m.group(0).replace(",", ""))
        except ValueError:
            return None


def parse_reads(text: str) -> Tuple[List[Dict[str, Any]],
                                    Optional[Dict[str, Any]]]:
    """The ``READ`` lines and the ``ANSWER`` line of a raw answer (its pixel
    boxes not yet rewritten): ``([{axis, at, curve, value, px}], answer)``."""
    reads: List[Dict[str, Any]] = []
    answer = None
    for line in (text or "").splitlines():
        s = line.strip().strip("`*- ")
        head = s.split("|", 1)[0].strip().upper()
        if head not in ("READ", "ANSWER"):
            continue
        fields: Dict[str, str] = {}
        axis = at = None
        for part in s.split("|")[1:]:
            if "=" not in part and ":" not in part:
                continue
            k, _, v = part.partition("=") if "=" in part else \
                part.partition(":")
            k = k.strip().lower()
            if k in ("x", "y") and axis is None:
                axis, at = k, _num(v)
            else:
                fields[k] = v.strip()
        if head == "ANSWER":
            val = _num(fields.get("value", ""))
            cur = _num(fields.get("curve", ""))
            if val is not None:
                answer = {"curve": cur, "value": val}
            continue
        m = _PX.search(s)
        val = _num(fields.get("value", ""))
        if axis is None or at is None or val is None or not m:
            continue
        curve = fields.get("curve", "").strip()
        reads.append({"axis": axis, "at": at,
                      "curve": None if curve in ("", "-") else curve,
                      "value": val,
                      "px": [float(m.group(i).replace(",", "."))
                             for i in range(1, 5)]})
    return reads, answer


def _document(path: str):
    """An open planlens document for a reference PDF, kept open (a few), so
    the page's scales are found once."""
    from planlens.document import Document
    key = os.path.abspath(str(path))
    with _DOCS_LOCK:
        doc = _DOCS.get(key)
        if doc is not None:
            _DOCS.move_to_end(key)
            return doc
        doc = Document(filepath=key)
        _DOCS[key] = doc
        while len(_DOCS) > _DOCS_MAX:
            _old, gone = _DOCS.popitem(last=False)
            try:
                gone.close()
            except Exception:           # noqa: BLE001 - closing a cache entry
                pass
        return doc


def close_documents() -> None:
    """Close the cached reference documents (tests)."""
    with _DOCS_LOCK:
        while _DOCS:
            _k, doc = _DOCS.popitem()
            try:
                doc.close()
            except Exception:           # noqa: BLE001
                pass


def _number(v: Any) -> Optional[float]:
    if not isinstance(v, dict):
        return None
    for k, x in v.items():
        if k in ("plus_minus", "unit", "confidence", "display",
                 "plus_minus_pt", "warnings"):
            continue
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return float(x)
    return None


def _agreement(code: float, pm: float, vision: float) -> Dict[str, Any]:
    gap = abs(vision - code)
    pm = abs(pm) if pm else 0.0
    ratio = gap / pm if pm > 0 else math.inf
    out = {"gap": round(gap, 6)}
    if pm > 0:
        out["gap_in_code_plus_minus"] = round(ratio, 1)
    if ratio > DISAGREE_FACTOR:
        out["agreement"] = "disagree"
        out["flag"] = (f"the vision value differs from code's by "
                       f"{ratio:.1f}x code's +/- (over "
                       f"{DISAGREE_FACTOR:g}x): check before relying on "
                       f"either")
    else:
        out["agreement"] = "agree"
    return out


def code_reading(pdf_path: str, page: int, raw_answer: str,
                 view: Sequence[float], size: Optional[Sequence[int]],
                 engine: Any = None) -> Dict[str, Any]:
    """Code's reading of each ``READ`` line of a chart read-off.

    ``view`` is the page rect the image showed and ``size`` the image's
    (w, h) as sent; ``engine`` reads label crops when the chart's labels
    have no text. Returns the ``code_reading`` block for the result.
    """
    with _READ_LOCK:
        return _code_reading(pdf_path, page, raw_answer, view, size, engine)


def _code_reading(pdf_path: str, page: int, raw_answer: str,
                  view: Sequence[float], size: Optional[Sequence[int]],
                  engine: Any = None) -> Dict[str, Any]:
    from planlens.document.measuring import measure
    from planlens.document.scalefinder import find_scales
    from funhouse_agent import scale_labels, vision_view

    reads, answer = parse_reads(raw_answer)
    if not reads:
        return {"status": "no_read_lines",
                "note": "the vision answer gave no READ line with a pixel "
                        "box, so code had nothing to measure; the value is "
                        "the vision read-off alone"}
    if not size:
        return {"status": "unavailable",
                "note": "the image size was not known, so a pixel box "
                        "could not be placed on the page"}
    doc = _document(pdf_path)
    vw = max(float(view[2]) - float(view[0]), float(view[3]) - float(view[1]))
    pad = max(3.0, PX_BOX_ERROR_FRAC * vw)
    labels_info: Optional[Dict[str, Any]] = None
    values: Dict[str, List[Optional[str]]] = {}
    out_reads: List[Dict[str, Any]] = []
    for rd in reads:
        box = vision_view.px_box_to_page(view, rd["px"], size)
        # a box drawn round the crossing point; a degenerate one is a point
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        half = max(1.5, min(6.0, 0.5 * max(box[2] - box[0],
                                            box[3] - box[1])))
        box = (cx - half, cy - half, cx + half, cy + half)
        window = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
        try:
            pend = find_scales(doc, int(page), values=values or None,
                               near=window).needing_values()
            if pend and engine is not None:
                items = []
                spans = []
                for sc in pend:
                    spans.append((sc.id, len(items),
                                  len(items) + len(sc.label_boxes)))
                    items.extend((int(page), b) for b in sc.label_boxes)
                texts, info = scale_labels.read_labels(
                    doc, items, engine, doc_key=("reference", pdf_path))
                for sid, a, b in spans:
                    values[sid] = texts[a:b]
                labels_info = info if labels_info is None else {
                    k: (labels_info.get(k, 0) + info.get(k, 0)
                        if isinstance(info.get(k), int) else info.get(k))
                    for k in info}
            res = measure(doc, int(page), where=list(box), kind="curve",
                          at={rd["axis"]: rd["at"]}, pad=pad,
                          values=values or None)
        except Exception as exc:              # noqa: BLE001 - a voter only
            out_reads.append({"input": {rd["axis"]: rd["at"]},
                              "curve": rd["curve"], "vision": rd["value"],
                              "code": None,
                              "status": f"code could not read it: "
                                        f"{type(exc).__name__}: {exc}"})
            continue
        row: Dict[str, Any] = {"input": {rd["axis"]: rd["at"]},
                               "curve": rd["curve"], "vision": rd["value"]}
        val = res.get("value")
        num = _number(val)
        if num is not None and not res.get("unsnapped"):
            pm = float(val.get("plus_minus") or 0.0)
            row["code"] = {"value": num, "plus_minus": val.get("plus_minus"),
                           "display": val.get("display"),
                           "unit": val.get("unit"),
                           "confidence": val.get("confidence")}
            row.update(_agreement(num, pm, rd["value"]))
            row["status"] = "measured"
            sc = res.get("scale") or {}
            if sc.get("id"):
                row["scale"] = sc.get("id")
        elif res.get("ambiguous"):
            alts = []
            for a in res.get("alternatives") or []:
                n = _number(a.get("value"))
                if n is not None:
                    alts.append(n)
            row["code"] = None
            row["status"] = "several_curves"
            row["candidates"] = alts[:6]
            row["note"] = ("several curves cross that value near the box: "
                           "code lists them and chooses none")
        elif res.get("unsnapped"):
            row["code"] = None
            row["status"] = "no_curve"
            row["note"] = ("no drawn curve was found crossing that value "
                           "near the box: no code reading")
        else:
            row["code"] = None
            if res.get("needs_values"):
                row["status"] = "labels_unread"
                row["note"] = ("the chart's axis labels have no text and "
                               "could not be read, so its scale is not "
                               "known")
            else:
                row["status"] = "no_scale"
                row["note"] = (res.get("note") or res.get("refused")
                               or "no fitted scale answers here")
        out_reads.append(row)
    block: Dict[str, Any] = {
        "method": ("the chart's axes fitted from its gridlines and printed "
                   "labels, and the curve found in the drawing where it "
                   "crosses the input value near the vision call's box; "
                   "code chooses no curve the box does not single out"),
        "readings": out_reads,
    }
    measured = [r for r in out_reads if r.get("status") == "measured"]
    if answer is not None:
        block["answer"] = _interpolate(answer, measured)
    if labels_info is not None:
        block["labels_read"] = scale_labels.describe(labels_info)
    block["status"] = ("measured" if measured else
                       (out_reads[0].get("status") if out_reads
                        else "no_read_lines"))
    return block


def _interpolate(answer: Dict[str, Any], measured: List[Dict[str, Any]]
                 ) -> Dict[str, Any]:
    """Code's value for an interpolated answer: linear in the curve parameter
    between the two code readings that bracket it (at one input value)."""
    out = {"curve": answer.get("curve"), "vision": answer.get("value")}
    p = answer.get("curve")
    pairs = []
    for r in measured:
        c = _num(r.get("curve")) if r.get("curve") is not None else None
        if c is not None:
            pairs.append((c, r))
    if p is None or len(pairs) < 2:
        out["code"] = None
        out["note"] = ("code reads an interpolation only between two "
                       "labelled curves it measured")
        return out
    pairs.sort(key=lambda t: t[0])
    lo = hi = None
    for (c1, r1), (c2, r2) in zip(pairs, pairs[1:]):
        if c1 <= p <= c2 and r1["input"] == r2["input"]:
            lo, hi = (c1, r1), (c2, r2)
            break
    if lo is None:
        out["code"] = None
        out["note"] = "the curve value asked for is not between two measured curves"
        return out
    (c1, r1), (c2, r2) = lo, hi
    t = 0.0 if c2 == c1 else (p - c1) / (c2 - c1)
    v1, v2 = r1["code"]["value"], r2["code"]["value"]
    pm1 = float(r1["code"].get("plus_minus") or 0.0)
    pm2 = float(r2["code"].get("plus_minus") or 0.0)
    v = v1 + t * (v2 - v1)
    pm = (1 - t) * pm1 + t * pm2
    out["code"] = {"value": round(v, 6), "plus_minus": round(pm, 6),
                   "between": [r1.get("curve"), r2.get("curve")],
                   "how": "linear in the curve parameter between code's two "
                          "readings"}
    if out["vision"] is not None:
        out.update(_agreement(v, pm, float(out["vision"])))
    return out


__all__ = ["code_reading", "parse_reads", "with_read_lines", "voter_on",
           "READ_INSTRUCTION", "VOTER_ENV", "DISAGREE_FACTOR",
           "close_documents"]

"""Reading a scale's printed labels off a scan: numbered crops, one side call.

planlens finds a page's scales from its drawing and never calls a model
(``planlens.document.scalefinder``). On a scan with no text it finds WHERE
the labels of a depth ruler, a plot axis or a scale bar are, and not what
they say: the scale comes back with ``needs_values`` and the label boxes in
run order. This module is the app's answer (design
``module_work/VISUAL_SCALES_DESIGN.md`` §4.5, owner decision 3):

1. every label box is cut out, enlarged and numbered on ONE contact sheet
   (:func:`label_sheets`; PyMuPDF only — no OpenCV, no PIL);
2. ONE vision call reads each number as printed (:data:`LABEL_PROMPT`);
3. the reply is parsed into the labels' printed text in box order, ``None``
   where a cell was not read (:func:`parse_label_reply`) — the text, not a
   float, so planlens keeps the print's resolution and a reading is never
   written finer than the print (owner decision 5).

The POSITIONS stay code's: a misread value breaks the run's even steps and
planlens drops it (one) or refuses the scale (two). What was read is
remembered per document, page and box (:func:`read_labels`), so a second
measurement on the same page spends no call.

Used by the ``measure`` and ``log_grid`` agent tools
(:mod:`funhouse_agent.measure_tool`) and by ``read_reference_figure``'s code
reading. Report ingest reads labels through its own engine
(:mod:`report_ingest.visual_scales`) with the same sheets and parser.
"""

from __future__ import annotations

import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Labels on one contact sheet. A depth ruler has about ten, a plot axis
#: ten to twenty; two pending axes fit one sheet.
PER_SHEET = 40
#: Cells across a sheet.
COLS = 4
#: One cell, in pixels: the crop is fitted inside, the number band above.
CELL_W, CELL_H, BAND = 240, 84, 26
#: How tall a label's lettering is drawn in its cell, in pixels.
LABEL_PX = 40.0
#: Parallel side calls when there is more than one sheet.
WORKERS = 4

LABEL_PROMPT = (
    "Each numbered cell (#N) shows ONE number printed on a scale - a depth "
    "or elevation ruler, a plot axis, a scale bar - cut from a page and "
    "enlarged. For EVERY cell reply with exactly one line:\n"
    "#N | <the number exactly as printed>\n"
    "Copy the digits, any decimal point, minus sign or plus sign (a station "
    "such as 12+50) as printed, with every trailing zero (write 4.0, not "
    "4); leave out units. Write a power of ten as printed, e.g. 10^-3. If "
    "a cell shows no number, or you cannot read it with confidence, write "
    "#N | - . Output only those lines.")

_LINE = re.compile(r"^\s*#\s*(\d+)\s*[|:)\]]?\s*(.*?)\s*$|"
                   r"^\s*(\d+)\s*[|:)\]]\s*(.*?)\s*$")
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
#: A run of superscript characters: an exponent written as printed (10⁻³).
_SUPER_RUN = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+")
#: A power of ten, with its exponent marked (``^``, ``**``, ``e``): a bare
#: "100" is one hundred, never ten to the zero.
_POWER = re.compile(
    r"^(?:(\d+(?:\.\d+)?)\s*[x×*]\s*)?10\s*(?:\^|\*\*)\s*\(?\s*([-−–]?\d+)"
    r"\s*\)?$")
_E_NOTATION = re.compile(r"^[-+]?\d+(?:\.\d+)?[eE][-+]?\d+$")
_UNIT_TAIL = re.compile(
    r"\s*(?:m|mm|cm|km|ft|feet|in|inch|inches|%|'|\"|kpa|mpa|psf|tsf|ksf|"
    r"bpf|blows)\.?$", re.I)

_CACHE: Dict[Tuple[Any, int, Tuple[float, ...]], Optional[str]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 20000


def clear_cache() -> None:
    """Forget every label read (tests; a long-lived process never needs to)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _box_key(box: Sequence[float]) -> Tuple[float, ...]:
    return tuple(round(float(v), 1) for v in box)


# ---------------------------------------------------------------------------
# The sheet
# ---------------------------------------------------------------------------

def label_sheets(doc: Any, items: Sequence[Tuple[int, Sequence[float]]],
                 start: int = 1, per_sheet: int = PER_SHEET
                 ) -> List[Tuple[bytes, List[int], Tuple[int, int]]]:
    """Numbered contact sheets of label crops, for a vision model to read.

    ``items`` are ``(page, box)`` with the box in PDF points, displayed
    frame (the frame planlens reports label boxes in, and the frame PyMuPDF
    clips a render in). Each crop is the box with a small margin, drawn so
    its lettering is about :data:`LABEL_PX` tall, fitted into its cell under
    a band carrying ``#n`` (``n`` counting from ``start``). Returns
    ``[(png, [item indices], (width, height))]``.
    """
    import fitz
    fz = getattr(doc, "_doc", doc)
    out: List[Tuple[bytes, List[int], Tuple[int, int]]] = []
    for s0 in range(0, len(items), max(1, int(per_sheet))):
        chunk = list(range(s0, min(len(items), s0 + per_sheet)))
        rows = math.ceil(len(chunk) / COLS)
        cols = min(COLS, len(chunk))
        width, height = cols * CELL_W, rows * (CELL_H + BAND)
        sheet = fitz.open()
        spg = sheet.new_page(width=width, height=height)
        for n, i in enumerate(chunk):
            page, box = items[i]
            x0, y0, x1, y1 = (float(v) for v in box)
            h = max(1.0, y1 - y0)
            w = max(1.0, x1 - x0)
            # wide enough each side that a digit the box cut off still shows
            mx, my = max(3.0, 1.2 * h), max(1.5, 0.3 * h)
            src = fz[int(page)]
            r = src.rect
            clip = fitz.Rect(max(r.x0, x0 - mx), max(r.y0, y0 - my),
                             min(r.x1, x1 + mx), min(r.y1, y1 + my))
            c0 = (n % COLS) * CELL_W
            r0 = (n // COLS) * (CELL_H + BAND)
            spg.draw_rect(fitz.Rect(c0 + 1, r0 + 1, c0 + CELL_W - 1,
                                    r0 + BAND + CELL_H - 1),
                          color=(0.55, 0.55, 0.55), width=1)
            spg.insert_text((c0 + 8, r0 + BAND - 7), f"#{start + i}",
                            fontsize=17, color=(0, 0, 0))
            if clip.is_empty or clip.width < 0.5 or clip.height < 0.5:
                continue
            zoom = min(LABEL_PX / max(h, 2.0), 1200.0 / 72.0)
            # a long label (a station, a coordinate) must still fit its cell
            zoom = min(zoom, (CELL_W - 12) / max(clip.width, 1.0),
                       (CELL_H - 8) / max(clip.height, 1.0) * 1.0)
            zoom = max(zoom, 0.5)
            pix = src.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip,
                                 colorspace=fitz.csGRAY, alpha=False)
            pw, ph = pix.width, pix.height
            ox = c0 + (CELL_W - pw) / 2.0
            oy = r0 + BAND + (CELL_H - ph) / 2.0
            spg.insert_image(fitz.Rect(ox, oy, ox + pw, oy + ph), pixmap=pix)
        png = spg.get_pixmap(dpi=72, alpha=False).tobytes("png")
        sheet.close()
        out.append((png, chunk, (int(width), int(height))))
    return out


# ---------------------------------------------------------------------------
# The reply
# ---------------------------------------------------------------------------

def normalise_label(text: Any) -> Optional[str]:
    """One cell's reading as a label planlens can parse, or ``None``.

    Keeps what was printed (``"4.0"`` stays ``"4.0"``, so its resolution is
    kept); drops a trailing unit; writes a power of ten (``10^-3``,
    ``10⁻³``, ``2x10^2``) as its plain value.
    """
    t = str(text or "").strip().strip("`").strip()
    if not t or t in ("-", "?", "—", "–", "none", "None", "null"):
        return None
    t = _SUPER_RUN.sub(lambda m: "^" + m.group(0).translate(_SUPERSCRIPT), t)
    t = t.replace("−", "-").replace("–", "-")
    t = _UNIT_TAIL.sub("", t).strip()
    m = _POWER.match(t.replace(" ", ""))
    if m:
        mant = float(m.group(1)) if m.group(1) else 1.0
        v = mant * 10.0 ** int(m.group(2).replace("−", "-").replace("–", "-"))
        t = f"{v:.12g}"
    elif _E_NOTATION.match(t):
        t = f"{float(t):.12g}"
    try:
        from planlens.document.scalefinder import parse_label_value
    except ImportError:                       # pragma: no cover - old planlens
        try:
            float(t.replace(",", ""))
            return t
        except ValueError:
            return None
    return t if parse_label_value(t) is not None else None


def parse_label_reply(text: str, ids: Sequence[int]) -> Dict[int, Optional[str]]:
    """``{number: label text or None}`` for every number in ``ids`` from the
    model's ``#N | 4.0`` lines; a number it did not answer is ``None``."""
    got: Dict[int, Optional[str]] = {}
    for line in (text or "").splitlines():
        m = _LINE.match(line.strip().strip("`"))
        if m:
            num = m.group(1) or m.group(3)
            val = m.group(2) if m.group(1) else m.group(4)
            got[int(num)] = normalise_label(val)
    return {i: got.get(i) for i in ids}


# ---------------------------------------------------------------------------
# One side call (or one per sheet), remembered
# ---------------------------------------------------------------------------

def read_labels(doc: Any, items: Sequence[Tuple[int, Sequence[float]]],
                engine: Any, *, doc_key: Any = None,
                prompt: str = LABEL_PROMPT) -> Tuple[List[Optional[str]],
                                                     Dict[str, Any]]:
    """Read every label box in ``items`` with the vision ``engine``.

    Returns the labels' printed text in ``items`` order (``None`` where a
    cell was not read) and what was spent: ``{"calls", "labels", "read",
    "unreadable", "from_memory", "errors"}``. With a ``doc_key`` (a document
    handle) a label already read on that page is not read again.
    """
    texts: List[Optional[str]] = [None] * len(items)
    info: Dict[str, Any] = {"calls": 0, "labels": len(items), "read": 0,
                            "unreadable": 0, "from_memory": 0, "errors": []}
    todo: List[int] = []
    for i, (page, box) in enumerate(items):
        key = (doc_key, int(page), _box_key(box))
        if doc_key is not None:
            with _CACHE_LOCK:
                if key in _CACHE:
                    texts[i] = _CACHE[key]
                    info["from_memory"] += 1
                    continue
        todo.append(i)
    if todo and engine is not None:
        sub = [items[i] for i in todo]
        sheets = label_sheets(doc, sub)

        def read(sheet):
            png, idx, _size = sheet
            try:
                return idx, engine.analyze_image(png, prompt), None
            except Exception as exc:          # a sheet, not the measurement
                return idx, "", f"{type(exc).__name__}: {exc}"

        import contextvars
        if len(sheets) == 1:
            results = [read(sheets[0])]
        else:
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = [ex.submit(contextvars.copy_context().run, read, s)
                        for s in sheets]
                results = [f.result() for f in futs]
        for idx, reply, err in results:
            info["calls"] += 1
            if err:
                info["errors"].append(err)
                continue
            got = parse_label_reply(reply, [k + 1 for k in idx])
            for k in idx:
                i = todo[k]
                texts[i] = got.get(k + 1)
                if doc_key is not None:
                    page, box = items[i]
                    with _CACHE_LOCK:
                        if len(_CACHE) >= _CACHE_MAX:
                            _CACHE.clear()
                        _CACHE[(doc_key, int(page), _box_key(box))] = texts[i]
    info["read"] = sum(1 for t in texts if t is not None)
    info["unreadable"] = len(items) - info["read"]
    return texts, info


def describe(info: Dict[str, Any]) -> str:
    """One line on how the label values were come by, for a result."""
    calls = info.get("calls", 0)
    mem = info.get("from_memory", 0)
    how = []
    if calls:
        how.append(f"{calls} vision call{'s' if calls != 1 else ''} over "
                   f"numbered crops of the printed labels")
    if mem:
        how.append(f"{mem} remembered from an earlier read of this page")
    txt = " and ".join(how) or "no read"
    if info.get("unreadable"):
        txt += (f"; {info['unreadable']} of {info.get('labels', 0)} not "
                f"read (left out of the fit)")
    return txt


__all__ = ["label_sheets", "parse_label_reply", "normalise_label",
           "read_labels", "describe", "clear_cache", "LABEL_PROMPT",
           "PER_SHEET"]

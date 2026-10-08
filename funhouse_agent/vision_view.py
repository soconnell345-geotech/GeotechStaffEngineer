"""How a page reaches the vision model: its size, its detail level, and the
grid the model answers locations on.

Every vision tool in the app (``analyze_pdf_page``, ``render_region``,
``read_reference_figure``, ``view_worked_example_source``) renders a PDF page or
a region of it and hands the image to a one-shot vision call. This module is
the one place that decides how:

* **Size — the image budget.** A vision model shrinks every image to its own
  limits before it looks (GPT-5.4 at ``detail="high"``: 2048 px and 2,500
  patches of 32 px, about 1.6 MP; at ``"original"``: 6000 px and 10,000
  patches). planlens renders the page or region to EXACTLY the largest image
  that budget holds, re-drawing a zoomed region from the PDF so it fills it.
  ``GEOTECH_VISION_BUDGET`` names the budget (default
  :data:`DEFAULT_BUDGET`; ``none`` restores the old fixed sizes). Chart
  read-offs run inside :func:`chart_reading` at a larger budget
  (``GEOTECH_CHART_BUDGET``, default :data:`DEFAULT_CHART_BUDGET`).
* **Detail.** OpenAI's ``detail`` field decides which of those limits
  applies, so it must match the budget. :func:`detail` gives the budget's own
  value; ``GEOTECH_VISION_DETAIL`` overrides it (``none`` omits the field).
* **Where things are — pixels in, a 0-999 grid out.** The main agent never
  sees an image; it reads what the vision call wrote. Each vision call is
  told the image's size and asked for locations as boxes in PIXELS of that
  image (``px=[x0, y0, x1, y1]``); the tool converts them in code, with the
  size it SENT, into boxes on a 0-999 grid over the image
  (:func:`boxes_to_grid`) before the agent reads the answer, and returns the
  ``view`` — the page rect the image showed. ``render_region(view=...,
  image_box=...)`` turns the two back into a box on the page
  (:func:`image_box_to_page`). Measured on Funhouse GPT-5.4 (2026-10-07,
  module_work/harness_theory/locating_things_on_a_page.md §5.1): on one
  whole-sheet image its own 0-999 boxes were 57-91 pt off, with a scale that
  changed between identical calls, while its PIXEL boxes converted with the
  image's true size were within 1-6 pt of every tag — and its own statement
  of the image's size was wrong, so the size used is the one sent. A 0-999
  answer (an older prompt, a model that ignores the instruction) still
  parses.
* **Never more than the host delivers.** Funhouse shrinks any image over
  2,048 px on its long side before the model sees it (measured by token
  counts, 2026-10-07), so an image rendered larger is described by a size
  the model never saw: lettering the app thought was 14 px arrived at 7 px,
  and the tiling rule never fired. Every render is capped at
  :data:`DEFAULT_MAX_PX` (``GEOTECH_VISION_MAX_PX``; ``none`` lifts the cap)
  and at whatever smaller edge the probe measured for the host, so the size
  a box is converted with, the legibility estimate and auto-tiling all use
  the image the model actually got.
* **How far to trust a location.** A box is only as good as the view it was
  read off: off by up to about a tenth of a whole sheet in the measurements,
  a few points from a zoom. Every result says so (:func:`precision_note`),
  ``render_region(view=, image_box=)`` pads its window by that error
  (:func:`zoom_pad`), and ``annotate_document`` refuses a small mark read off
  a wide view (planlens ``markup_writer.VIEW_ANCHOR_MAX_PT``).

* **Which budget — measured, not assumed.** A deployment name is an alias
  that can be re-pointed at another model. The first time a vision engine is
  used, :mod:`funhouse_agent.vision_probe` asks the model what it is and
  measures what size of image it really looks at; that profile picks the
  budget for ordinary views and for chart read-offs. The env settings, when
  present, still win (an owner's override); without a profile the defaults
  below stand.
* **Legibility.** planlens reports how tall the page's small lettering is in
  each image (``text_px``). Below :data:`LEGIBLE_TEXT_PX` the result says so
  and points at the text layer and at a zoom window small enough to read in.

planlens gained budgets after the version this app pins; on an older planlens
the renders fall back to the fixed sizes and everything else still works.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

BUDGET_ENV = "GEOTECH_VISION_BUDGET"
DETAIL_ENV = "GEOTECH_VISION_DETAIL"
CHART_BUDGET_ENV = "GEOTECH_CHART_BUDGET"
POLICY_ENV = "GEOTECH_VISION_POLICY"
MAX_PX_ENV = "GEOTECH_VISION_MAX_PX"

#: The longest side any image is rendered at. Funhouse delivers at most
#: 2,048 px (a 3957 x 2560 image cost exactly the tokens of 2048 x 1326,
#: 2026-10-07); GPT-5.6 on Foundry, sent the full 10,000-patch image, drew
#: its boxes shrunk toward the top-left, and was accurate at about 2,000 px.
#: ``GEOTECH_VISION_MAX_PX`` overrides it (``none`` lifts it); a smaller edge
#: the probe measured for the host always wins.
DEFAULT_MAX_PX = 2048

#: An image edge below this is not a sensible cap (a typo, not a choice).
_MIN_MAX_PX = 256

#: ``1`` renders every image with both sides a whole number of
#: :data:`PATCH_PX` patches (OFF by default — a hypothesis to measure, not a
#: fix). GPT-5.4's pixel boxes came back stretched in y by 1.4-1.7 % and not
#: at all in x, on four images whose height was not a multiple of 32 px and
#: whose width (2,048) was; each stretch matched the height rounded up to
#: whole 32 px patches (1344 / 1325 = 1.014; Foundry brief 4, part A). If the
#: model works in a frame of whole patches, an image that already is one
#: removes the stretch. Sol showed none either way. See
#: :func:`align_to_patches` and module_work/LIVE_TEST_QUEUE.md.
PATCH_ALIGN_ENV = "GEOTECH_VISION_PATCH_ALIGN"

#: The patch a vision model of the GPT-5 family cuts an image into.
PATCH_PX = 32

#: ``robust`` (default, owner 2026-09-25: "robust first, then maybe we can
#: dial back for efficiency later"): every image goes at the largest detail
#: the model was measured to honour, and small lettering is tiled.
#: ``efficient``: ordinary views at the model's ``high`` budget, only chart
#: read-offs at the larger one, no automatic tiling — the dial-back, noted in
#: module_work/FUTURE_IDEAS.md "VISION EFFICIENCY".
POLICIES = ("robust", "efficient")

#: GPT-5.4's ``high`` detail (its ``auto`` too): what the side call already
#: got, now rendered to the size it actually looks at. ``openai-original`` is
#: four times the pixels and four times the image tokens — the owner's call.
DEFAULT_BUDGET = "openai-high"

#: Chart read-offs (``read_reference_figure``, ``view_worked_example_source``)
#: are detail-bound and few — a handful a conversation — so they get GPT-5.4's
#: ``original`` level: up to 10,000 patches instead of 2,500 (owner,
#: 2026-09-23; ``funhouse-gpt-high`` is GPT-5.4, which supports it).
DEFAULT_CHART_BUDGET = "openai-original"

#: True while a chart read-off is rendering and sending (see
#: :func:`chart_reading`): the detail-bound budget applies.
_CHART: ContextVar[bool] = ContextVar("gse_chart_reading", default=False)

#: Lettering shorter than this in an image is not read reliably (5 pt CAD
#: lettering at ~5 px smeared bar sizes and spacings for GPT-5.1, 2026-09-24).
LEGIBLE_TEXT_PX = 12.0

#: The lettering height a suggested zoom window aims for.
TARGET_TEXT_PX = 16.0

#: How every vision prompt asks for codes to be read: a best reading, with
#: alternatives only where a character really cannot be told in THIS image.
#: It used to ask for brackets wherever a character "could be another", and
#: GPT-5.4 bracketed tags it could read — "[G/C]CE", "G[C/O]E" on zooms at
#: 300-380 px, even "[G/C/O][C/E]E" at 70 px lettering — and an agent dropped
#: two real tags on those hedges (Foundry brief 4, 2026-10-07). A bracket
#: that remains is settled by a closer zoom (:data:`BRACKETED_NOTE`).
READING_INSTRUCTION = (
    "Read codes, tags and numbers character by character and give your best "
    "reading of each. Only where a character truly cannot be told apart in "
    "this image (the usual confusions are G/Q/O/C/D, E/F, B/8, S/5, I/1/L, "
    "Z/2) write the alternatives in brackets, e.g. A[B/8]C; lettering you "
    "can read, write plainly. If the lettering is too small to read "
    "reliably, say so rather than guessing.")

#: A bracketed reading, as :data:`READING_INSTRUCTION` asks for one.
_BRACKETED = re.compile(r"\[[A-Za-z0-9](?:\s*/\s*[A-Za-z0-9])+\]")

#: Said beside an answer that still holds a bracketed reading.
BRACKETED_NOTE = (
    "this answer has a bracketed reading (a character it could not settle). "
    "Settle it with a closer zoom — render_region on a smaller box round "
    "that thing — rather than dropping the thing or counting it on the "
    "hedge.")


def bracketed_note(text: Any) -> Optional[str]:
    """:data:`BRACKETED_NOTE` when ``text`` holds a bracketed reading."""
    return BRACKETED_NOTE if _BRACKETED.search(str(text or "")) else None

#: The old location sentence: boxes on a 0-999 grid. Used only where the
#: image's size is not known; a 0-999 answer is still understood everywhere.
GRID_INSTRUCTION = (
    "If you give the location of anything in this image, give it as a box "
    "[x0, y0, x1, y1] on a 0-999 grid over the whole image (origin at the "
    "top-left corner, x to the right, y down). " + READING_INSTRUCTION)

#: The location sentence every vision prompt ends with when the image's size
#: is known: boxes in PIXELS of the image, tagged ``px=`` so they cannot be
#: mistaken for anything else (:func:`boxes_to_grid` converts them).
PIXEL_INSTRUCTION = (
    "This image is {w} x {h} pixels. If you give the location of anything "
    "in it, give it as a box in PIXELS of this image, written px=[x0, y0, "
    "x1, y1]: origin at the top-left corner, x to the right (0 to {w}), y "
    "down (0 to {h}). Box the thing itself, tightly, and write px= before "
    "every box. " + READING_INSTRUCTION)

#: What the agent is told about the ``view`` a vision result carries.
ZOOM_HINT = ("to zoom on something the analysis located, call "
             "render_region(attachment_key=<same source>, view=<this view>, "
             "image_box=<its 0-999 box>, prompt=...)")

#: A box read off a view can be off by up to about this fraction of the view
#: (per axis). Measured 2026-10-07 on an 11 x 17 sheet: 0-999 boxes off
#: whole-sheet images were 14-90 pt off (to 0.11 of the sheet's height), zooms
#: of 80-350 pt 0.2-5 pt. Pixel boxes did far better on the whole sheet (1-6
#: pt, one call), but a model can still answer on the grid, so locations are
#: trusted as if it had.
LOCATION_ERROR_FRAC = 0.10

#: ... and never less than this, in points (a zoom's own boxes were up to
#: 12 pt off in y twice in seven, GPT-5.4 on 40-pt crops).
MIN_LOCATION_ERROR_PT = 12.0

#: A view no wider than this (its longer side, pt) is one a mark may be
#: placed from — the same limit planlens' writer enforces
#: (``markup_writer.VIEW_ANCHOR_MAX_PT``).
MARK_VIEW_PT = 300.0

_OFF = ("", "none", "off", "0", "false")

JPEG_MAGIC = bytes([0xFF, 0xD8, 0xFF])


@contextmanager
def chart_reading() -> Iterator[None]:
    """Render and send images at the chart read-off budget for this block.

    Both the render (:func:`render_view`) and the engine's ``detail`` field
    (:func:`detail`) read the budget at call time, so one ``with`` covers
    both. ``GEOTECH_CHART_BUDGET`` overrides :data:`DEFAULT_CHART_BUDGET`.
    """
    token = _CHART.set(True)
    try:
        yield
    finally:
        _CHART.reset(token)


def policy() -> str:
    """``robust`` (default) or ``efficient`` — see :data:`POLICIES`."""
    p = os.environ.get(POLICY_ENV, "robust").strip().lower()
    return p if p in POLICIES else "robust"


def engine_profile(engine):
    """The vision profile an engine measured (``vision_probe``), or ``None``."""
    fn = getattr(engine, "vision_profile", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:             # a probe must never break a vision call
        return None


def budget_name(engine=None) -> str:
    """Which budget applies now, by name (``none`` = the old fixed sizes).

    An env setting wins; then what the engine's model was measured to take;
    then the defaults.
    """
    chart = _CHART.get()
    chart_env = os.environ.get(CHART_BUDGET_ENV)
    general_env = os.environ.get(BUDGET_ENV)
    if chart and chart_env:
        return chart_env.strip()
    if general_env is not None and (
            not chart or general_env.strip().lower() in _OFF):
        return general_env.strip()
    prof = engine_profile(engine)
    if prof is not None and prof.general:
        if chart or policy() == "robust":
            return prof.detailed
        return prof.general
    return DEFAULT_CHART_BUDGET if chart else DEFAULT_BUDGET


def max_px(engine=None) -> Optional[int]:
    """The longest side, in pixels, any image may be rendered at — or
    ``None`` for no cap.

    ``GEOTECH_VISION_MAX_PX`` (a number, or ``none``) or else
    :data:`DEFAULT_MAX_PX`; then never more than the edge the engine's probe
    measured the host to deliver (``VisionProfile.max_edge``), because an
    image larger than that is shrunk before the model sees it and every size
    the app reasons with would be wrong."""
    raw = os.environ.get(MAX_PX_ENV)
    cap: Optional[int] = DEFAULT_MAX_PX
    if raw is not None:
        text = raw.strip().lower()
        if text in _OFF:
            cap = None
        else:
            try:
                cap = int(float(text))
                if cap < _MIN_MAX_PX:
                    raise ValueError(f"under {_MIN_MAX_PX} px")
            except ValueError as exc:
                log.warning("%s=%r ignored (%s); using %d", MAX_PX_ENV, raw,
                            exc, DEFAULT_MAX_PX)
                cap = DEFAULT_MAX_PX
    prof = engine_profile(engine)
    host = getattr(prof, "max_edge", None) if prof is not None else None
    if host:
        cap = int(host) if cap is None else min(cap, int(host))
    return cap


def _capped(bud, cap: Optional[int]):
    """``bud`` with its longest side held to ``cap`` (name and detail kept:
    on Funhouse the 2,048 px image still needs ``detail="original"``, since
    its ``high`` is cut down like a tile model's)."""
    if bud is None or not cap or bud.max_edge <= cap:
        return bud
    return dataclasses.replace(bud, max_edge=int(cap))


def budget(engine=None):
    """The planlens ``ImageBudget`` renders are sized to, or ``None`` —
    held to :func:`max_px`."""
    name = budget_name(engine)
    if name.lower() in _OFF:
        return None
    try:
        from planlens.document.budget import resolve_budget
    except ImportError:           # planlens older than image budgets
        return None
    try:
        bud = resolve_budget(name)
    except ValueError as exc:
        log.warning("%s=%r ignored: %s", BUDGET_ENV, name, exc)
        return None
    return _capped(bud, max_px(engine))


def detail(engine=None) -> Optional[str]:
    """The ``detail`` value image blocks carry, or ``None`` to omit it."""
    raw = os.environ.get(DETAIL_ENV)
    if raw is not None:
        return None if raw.strip().lower() in _OFF else raw.strip()
    bud = budget(engine)
    return bud.detail if bud is not None else None


def image_media_type(data: bytes) -> str:
    """``image/jpeg`` or ``image/png`` from the bytes themselves."""
    if data[:3] == JPEG_MAGIC:
        return "image/jpeg"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _render_accepts_budget() -> bool:
    from planlens.document import Document
    return "budget" in inspect.signature(Document.render).parameters


def render_view(source, page: int = 0, bbox: Optional[Sequence[float]] = None,
                marks: Optional[Sequence[Sequence[Any]]] = None,
                dpi: Optional[float] = None, pad_frac: float = 0.1,
                allow_jpeg: bool = False,
                engine=None) -> Tuple[bytes, Dict[str, Any]]:
    """Render ``page`` (or ``bbox`` on it) of a PDF for the vision model.

    ``source`` is PDF bytes or a file path. Returns ``(image_bytes, info)``
    where ``info`` is planlens' render info (``page``, ``clip``, ``dpi``,
    ``width_px``, ``height_px`` and, with a budget, ``format``/``budget``).
    ``allow_jpeg`` lets a scan go as JPEG when that is smaller (the engine
    must label the bytes by their real type); line art stays PNG either way.
    ``engine`` is the vision engine the image goes to: its measured model
    profile picks the budget (see :func:`budget_name`). The image is never
    larger than :func:`max_px` on its long side, so ``info``'s size is the
    size the model sees.
    """
    from planlens.document import Document
    doc = (Document(content=source) if isinstance(source, (bytes, bytearray))
           else Document(filepath=str(source)))
    try:
        kwargs: Dict[str, Any] = {"bbox": tuple(bbox) if bbox is not None else None,
                                  "pad_frac": pad_frac,
                                  "marks": [tuple(m) for m in marks] if marks else None}
        bud = budget(engine)
        if bud is not None and _render_accepts_budget():
            kwargs.update(budget=bud, fmt="auto" if allow_jpeg else "png",
                          dpi=dpi)
        else:
            # The fixed sizes this app always used: 300 dpi for a zoom.
            kwargs["dpi"] = dpi if dpi is not None else (300 if bbox is not None else None)
        data, info = doc.render(int(page), **kwargs)
        cap = max_px(engine)
        # A budget already held the size; the fixed sizes (or a dpi asked
        # for by hand) may not have: render again just under the cap.
        for _ in range(3):
            side = max(info["width_px"], info["height_px"])
            if not cap or side <= cap:
                break
            kwargs["dpi"] = float(info["dpi"]) * (cap - 1.0) / side
            data, info = doc.render(int(page), **kwargs)
        if patch_align():
            data, info = align_to_patches(data, info, cap)
        return data, info
    finally:
        doc.close()


def patch_align() -> bool:
    """Whether images are padded to whole :data:`PATCH_PX` patches
    (``GEOTECH_VISION_PATCH_ALIGN``, OFF by default)."""
    return str(os.environ.get(PATCH_ALIGN_ENV, "")).strip().lower() in (
        "1", "true", "yes", "on")


def align_to_patches(data: bytes, info: Dict[str, Any],
                     cap: Optional[int] = None
                     ) -> Tuple[bytes, Dict[str, Any]]:
    """``(image, info)`` with both sides of the image rounded UP to whole
    :data:`PATCH_PX` patches — at most 31 px of white paper added on the
    right and at the bottom — and the ``clip`` (the view) widened by exactly
    the points those pixels show, so a box converted with the view and the
    size sent lands where it did before. ``info['patch_aligned']`` is the
    ``[x, y]`` pixels added. Left as it was when it already fits, when the
    padded side would pass ``cap``, or when the image cannot be decoded."""
    import math
    w, h = int(info["width_px"]), int(info["height_px"])
    tw = int(math.ceil(w / PATCH_PX)) * PATCH_PX
    th = int(math.ceil(h / PATCH_PX)) * PATCH_PX
    if (tw, th) == (w, h) or (cap and max(tw, th) > cap):
        return data, info
    try:
        import fitz
        src = fitz.Pixmap(data)
        if src.alpha:
            src = fitz.Pixmap(src, 0)
        if (src.width, src.height) != (w, h):
            return data, info
        canvas = fitz.Pixmap(src.colorspace, fitz.IRect(0, 0, tw, th), False)
        canvas.clear_with(255)
        canvas.copy(src, fitz.IRect(0, 0, w, h))
        fmt = str(info.get("format") or "png").lower()
        out = (canvas.tobytes("jpeg", jpg_quality=85) if fmt == "jpeg"
               else canvas.tobytes("png"))
    except Exception:                     # a padding that fails changes nothing
        return data, info
    x0, y0, x1, y1 = (float(v) for v in info["clip"])
    aligned = dict(info)
    aligned["clip"] = [x0, y0, x0 + (x1 - x0) * tw / w,
                       y0 + (y1 - y0) * th / h]
    aligned["width_px"], aligned["height_px"] = tw, th
    aligned["patch_aligned"] = [tw - w, th - h]
    return out, aligned


def view_payload(info: Dict[str, Any], engine=None) -> Dict[str, Any]:
    """The fields a vision result carries so the agent can zoom on it — and,
    when the page's lettering is too small in this image, what to do instead
    of calling it unreadable."""
    out: Dict[str, Any] = {
        "view": [round(float(v), 1) for v in info["clip"]],
        "view_px": [info["width_px"], info["height_px"]],
        "zoom_hint": ZOOM_HINT,
        "precision": precision_note(info["clip"])}
    prof = engine_profile(engine)
    if prof is not None and prof.answered_by:
        out["vision_model"] = prof.answered_by
    if info.get("budget"):
        out["budget"] = info["budget"]
    d = detail(engine)
    if d:
        out["detail"] = d
    text_px = info.get("text_px")
    if text_px is None and info.get("text_chars", 1 << 30) < 40:
        out["legibility"] = (
            "! this sheet's lettering is not in its text layer (drawn as "
            "lines by CAD, or scanned): text search cannot see it. To read "
            "small lettering, zoom with render_region. Never conclude "
            "something is absent from a whole-sheet view.")
    if text_px:
        out["text_px"] = text_px
        if text_px < LEGIBLE_TEXT_PX:
            x0, y0, x1, y1 = info["clip"]
            window = max(20, round(min(x1 - x0, y1 - y0) * float(text_px)
                                   / TARGET_TEXT_PX))
            out["legibility"] = (
                f"! the page's small lettering is only ~{text_px:g} px tall in "
                f"this image, too small to read reliably. Read the words from "
                f"the text layer (read_document / search_document on this "
                f"file: exact, no guessing), and to SEE a detail zoom with "
                f"render_region on a box about {window} pt across (a bbox, or "
                f"this view + an image_box). Do not report the page as "
                f"unreadable or ask for a better file before doing both.")
    return out


#: Lettering assumed on a CAD sheet whose lettering cannot be measured (drawn
#: as lines): 0.06 in, the small end of plotted half-size sheets — the IZD set
#: of 2026-09-25.
ASSUMED_CAD_LETTERING_PT = 4.3

#: Most tiles one sheet is split into for a read (4 x 4).
MAX_TILES_PER_SIDE = 4


def tile_grid(info: Dict[str, Any]) -> int:
    """Tiles per side needed for the page's small lettering to arrive at
    :data:`TARGET_TEXT_PX` — 1 when the whole page already reads.

    Uses the measured ``text_px`` where the text layer gives one; on a sheet
    whose lettering is drawn as lines (no measurable text), assumes
    :data:`ASSUMED_CAD_LETTERING_PT`."""
    import math
    x0, y0, x1, y1 = info["clip"]
    px_per_pt = info["width_px"] / max(x1 - x0, 1e-6)
    text_px = info.get("text_px")
    if text_px is None:
        if info.get("text_chars", 1 << 30) >= 40:
            return 1                         # prose page with no size: fine
        text_px = ASSUMED_CAD_LETTERING_PT * px_per_pt
    if text_px >= LEGIBLE_TEXT_PX:
        return 1
    n = math.ceil(TARGET_TEXT_PX / max(text_px, 0.1))
    return max(1, min(MAX_TILES_PER_SIDE, n))


def tile_boxes(clip: Sequence[float], n: int, overlap: float = 0.08):
    """``n`` x ``n`` boxes over ``clip`` (points), each widened by
    ``overlap`` of its size so nothing on a seam is cut in two. Row-major,
    top-left first."""
    x0, y0, x1, y1 = (float(v) for v in clip)
    w, h = (x1 - x0) / n, (y1 - y0) / n
    ox, oy = overlap * w, overlap * h
    out = []
    for r in range(n):
        for c in range(n):
            out.append((max(x0, x0 + c * w - ox), max(y0, y0 + r * h - oy),
                        min(x1, x0 + (c + 1) * w + ox),
                        min(y1, y0 + (r + 1) * h + oy)))
    return out


def with_grid(prompt: str) -> str:
    """``prompt`` with the old 0-999 location instruction appended (for an
    image whose size is not known; see :func:`with_location`)."""
    return f"{prompt.rstrip()}\n\n{GRID_INSTRUCTION}"


def pixel_instruction(size: Sequence[int]) -> str:
    """:data:`PIXEL_INSTRUCTION` for an image of ``size`` = (w, h) px."""
    w, h = (int(v) for v in size)
    return PIXEL_INSTRUCTION.format(w=w, h=h)


def with_location(prompt: str, size: Optional[Sequence[int]] = None) -> str:
    """``prompt`` with the location instruction appended: boxes in pixels of
    an image of ``size`` (w, h) — or, with no size, the old 0-999 grid."""
    if not size:
        return with_grid(prompt)
    return f"{prompt.rstrip()}\n\n{pixel_instruction(size)}"


# ---------------------------------------------------------------------------
# Locations: pixel boxes from the vision call, 0-999 boxes for the agent
# ---------------------------------------------------------------------------

_NUM = r"-?\d+(?:\.\d+)?"
#: A box as a vision answer writes one: ``px=[x0, y0, x1, y1]`` (also ``px:``,
#: ``px [..]``, ``px=(..)``), ``[..] px`` / ``[..] pixels``, or a bare
#: ``[a, b, c, d]``.
_BOX = re.compile(
    rf"(?P<pre>\bpx\s*[=:]?\s*)?(?P<open>[\[(])\s*(?P<a>{_NUM})\s*,\s*"
    rf"(?P<b>{_NUM})\s*,\s*(?P<c>{_NUM})\s*,\s*(?P<d>{_NUM})\s*[\])]"
    rf"(?P<post>\s*(?:px|pixels?)\b)?", re.IGNORECASE)

#: A box value over this cannot be on the 0-999 grid.
_GRID_MAX = 999.5


def _box_values(m) -> List[float]:
    return [float(m.group(k)) for k in "abcd"]


def _is_box(m) -> bool:
    """A tagged box, or a bare one in square brackets — ``(1, 2, 3, 4)``
    with no tag is prose, not a box."""
    return bool(m.group("pre") or m.group("post") or m.group("open") == "[")


def px_to_grid(px_box: Sequence[float], size: Sequence[int]
               ) -> List[int]:
    """A pixel box on an image of ``size`` (w, h) as a whole-number box on
    the 0-999 grid over the same image (corners ordered, clamped)."""
    w, h = (max(float(v), 1.0) for v in size)
    x0, y0, x1, y1 = (float(v) for v in px_box)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    def g(v, s):
        return int(max(0, min(999, round(v / s * 999.0))))
    return [g(x0, w), g(y0, h), g(x1, w), g(y1, h)]


def px_box_to_page(view: Sequence[float], px_box: Sequence[float],
                   size: Sequence[int]) -> Tuple[float, float, float, float]:
    """A pixel box on the image of ``view`` (``size`` = the image's w, h as
    SENT) as PDF points on the page — the exact conversion, with no grid in
    between."""
    if len(view) != 4 or len(px_box) != 4:
        raise ValueError("view and the box must each be [x0, y0, x1, y1]")
    vx0, vy0, vx1, vy1 = (float(v) for v in view)
    if vx1 <= vx0 or vy1 <= vy0:
        raise ValueError("view must be a non-empty [x0, y0, x1, y1] rect")
    w, h = (max(float(v), 1.0) for v in size)
    x0, y0, x1, y1 = (float(v) for v in px_box)
    fx0, fx1 = sorted((min(1.0, max(0.0, x0 / w)), min(1.0, max(0.0, x1 / w))))
    fy0, fy1 = sorted((min(1.0, max(0.0, y0 / h)), min(1.0, max(0.0, y1 / h))))
    vw, vh = vx1 - vx0, vy1 - vy0
    return (vx0 + fx0 * vw, vy0 + fy0 * vh, vx0 + fx1 * vw, vy0 + fy1 * vh)


def boxes_to_grid(text: str, size: Optional[Sequence[int]],
                  tagged_only: bool = False) -> Tuple[str, Dict[str, int]]:
    """A vision answer with every pixel box rewritten as a 0-999 box on the
    same image, so the agent reads ONE convention — the one
    ``render_region(view=, image_box=)`` and ``annotate_document`` take.

    A box tagged ``px=`` (or followed by ``px``) is in pixels. An untagged
    ``[a, b, c, d]`` is in pixels too when the answer shows it is answering
    in pixels — a tagged box anywhere in it, or a value past 999, which no
    grid box has (a model that forgot the tag on some boxes) — provided it
    is shaped like a pixel box on this image (whole numbers, or one past
    999; corners in order; inside the image), so a list of four readings off
    a chart is never rewritten. Without that evidence every box in the
    answer is taken as an old-style 0-999 box and left as written.
    ``tagged_only`` converts tagged boxes alone — for a chart read-off, whose
    answer is about values and may hold a bracketed list of four readings.
    Returns the text and ``{"converted": n, "grid": m}`` — the pixel boxes
    rewritten and the untagged boxes left as grid boxes."""
    counts = {"converted": 0, "grid": 0}
    if not text or not size:
        return text, counts
    w, h = (int(v) for v in size)
    found = [m for m in _BOX.finditer(text) if _is_box(m)]
    in_pixels = any(m.group("pre") or m.group("post") for m in found) or any(
        v > _GRID_MAX for m in found for v in _box_values(m))

    def pixel_shaped(vals: List[float]) -> bool:
        x0, y0, x1, y1 = vals
        if not (x1 > x0 and y1 > y0):
            return False
        if not (-0.02 * w <= x0 and x1 <= 1.02 * w
                and -0.02 * h <= y0 and y1 <= 1.02 * h):
            return False
        return (any(v > _GRID_MAX for v in vals)
                or all(float(v).is_integer() for v in vals))

    def rewrite(m) -> str:
        if not _is_box(m):
            return m.group(0)
        vals = _box_values(m)
        tagged = bool(m.group("pre") or m.group("post"))
        if tagged or (not tagged_only and in_pixels and pixel_shaped(vals)):
            counts["converted"] += 1
            return "[{}, {}, {}, {}]".format(*px_to_grid(vals, (w, h)))
        counts["grid"] += 1
        return m.group(0)

    return _BOX.sub(rewrite, text), counts


def boxes_note(counts: Dict[str, int], size: Sequence[int]) -> Optional[str]:
    """What the agent is told about the boxes in an answer, or ``None`` when
    it gave none."""
    w, h = (int(v) for v in size)
    if counts.get("converted"):
        left = counts.get("grid") or 0
        return (f"the boxes in this analysis are on the 0-999 grid over this "
                f"view's image, converted in code from the vision call's "
                f"pixel boxes on the {w} x {h} px image it was sent: pass one "
                f"as image_box with this view"
                + (f" ({left} bracketed list(s) that were not pixel boxes "
                   f"were left as written)" if left else ""))
    if counts.get("grid"):
        return ("the vision call gave its boxes without units; they are "
                "taken as 0-999 grid boxes over this view's image — zoom "
                "before relying on one")
    return None


def location_error(view: Sequence[float]) -> Tuple[float, float]:
    """How far (x, y, in points) a box read off ``view`` may be from the
    thing: :data:`LOCATION_ERROR_FRAC` of the view on each axis, at least
    :data:`MIN_LOCATION_ERROR_PT`."""
    vx0, vy0, vx1, vy1 = (float(v) for v in view)
    return (max(MIN_LOCATION_ERROR_PT, LOCATION_ERROR_FRAC * (vx1 - vx0)),
            max(MIN_LOCATION_ERROR_PT, LOCATION_ERROR_FRAC * (vy1 - vy0)))


#: Said on every look's precision line, in general words that name no tool
#: (owner decision 7, 2026-10-08, module_work/VISUAL_SCALES_DESIGN.md §10):
#: a box says where a thing is to look at, not where it is to the point.
NOT_A_MEASUREMENT = ("A box read off an image locates a thing; it is not a "
                     "measurement.")


def precision_note(view: Sequence[float]) -> str:
    """The line every vision result carries on how far its boxes can be
    trusted, and what that means for zooming and for placing a mark — and
    that a box is not a measurement (:data:`NOT_A_MEASUREMENT`)."""
    vx0, vy0, vx1, vy1 = (float(v) for v in view)
    vw, vh = vx1 - vx0, vy1 - vy0
    ex, ey = location_error(view)
    if max(vw, vh) <= MARK_VIEW_PT:
        return (f"a box read off this {vw:.0f} x {vh:.0f} pt view is good to "
                f"a few points: a mark may be anchored on this view + the "
                f"thing's image_box. {NOT_A_MEASUREMENT}")
    return (f"a box read off this {vw:.0f} x {vh:.0f} pt view can be up to "
            f"~{ex:.0f} x {ey:.0f} pt from the thing: good for finding where "
            f"to zoom (render_region with this view + image_box pads its "
            f"window by that much), NOT for placing a mark — zoom until the "
            f"thing is legible in a view of {MARK_VIEW_PT:.0f} pt or less and "
            f"anchor the mark on that zoom's view + image_box. "
            f"{NOT_A_MEASUREMENT}")


def zoom_window(view: Sequence[float], box: Sequence[float]
                ) -> Optional[List[float]]:
    """``box`` (PDF points, read off ``view``) padded by the view's location
    error — a ``render_region(bbox=...)`` that will hold the thing — or
    ``None`` when the view is narrow enough that the box itself can be used
    (:data:`MARK_VIEW_PT`)."""
    vx0, vy0, vx1, vy1 = (float(v) for v in view)
    if max(vx1 - vx0, vy1 - vy0) <= MARK_VIEW_PT:
        return None
    px, py = zoom_pad(view, box)
    bx0, by0, bx1, by1 = (float(v) for v in box)
    return [round(bx0 - px, 1), round(by0 - py, 1),
            round(bx1 + px, 1), round(by1 + py, 1)]


def zoom_pad(view: Sequence[float], box: Sequence[float],
             pad_frac: float = 0.15) -> Tuple[float, float]:
    """How much to pad, each side (x, y, points), a zoom on ``box`` read off
    ``view``: the location error of the SOURCE view (:func:`location_error`),
    or ``pad_frac`` of the box (at least 20 pt across, planlens' own rule)
    when that is more. A 15 % pad of a tag-sized box is a 30 x 14 pt window
    — far smaller than the error of a whole-sheet box, and the first zooms
    of 2026-10-07 came back as blank paper 11 times in 11."""
    ex, ey = location_error(view)
    bx0, by0, bx1, by1 = (float(v) for v in box)
    own = float(pad_frac) * max(bx1 - bx0, by1 - by0, 20.0)
    return max(ex, own), max(ey, own)


#: How far, as a fraction of the SOURCE view's longer side, a zoom's answer
#: may sit from the box it was aimed at before the result says the answer
#: may be about another thing. Pixel boxes off a whole 11 x 17 sheet landed
#: 1-6 pt from their tags (15-20 pt where a model boxed tag and leader
#: together; Foundry brief 4, part A), so 2.5 % — 31 pt on that sheet — is
#: past any of them. The zoom window itself is padded far wider
#: (:data:`LOCATION_ERROR_FRAC`), and in brief 4 a 259 pt window aimed at a
#: look-alike also held a tag 98 pt away, and the look answered about the
#: tag (Sol r3, the tag ringed twice).
AIM_TOLERANCE_FRAC = 0.025


def aim_tolerance(view: Sequence[float]) -> float:
    """Points a zoom's answer may sit from its aim, for a zoom on a box read
    off ``view`` (:data:`AIM_TOLERANCE_FRAC`, at least
    :data:`MIN_LOCATION_ERROR_PT`)."""
    vx0, vy0, vx1, vy1 = (float(v) for v in view)
    return max(MIN_LOCATION_ERROR_PT,
               AIM_TOLERANCE_FRAC * max(vx1 - vx0, vy1 - vy0))


#: A box as an answer reads AFTER :func:`boxes_to_grid`: four numbers in
#: square brackets, each on the 0-999 grid.
_GRID_BOX_RE = re.compile(
    rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]")

#: A box this big on the 0-999 grid (either side) is a region, not a thing.
REGION_GRID = 300


def _label_text(text: str) -> str:
    """The words before a box, cleaned for a label: markdown, list numbers,
    table bars and a trailing "at"/"px=" dropped."""
    t = re.sub(r"[*`|#>]", " ", text)
    t = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", t)
    t = re.sub(r"\b(?:px|box|at|in|bbox)\s*[=:]?\s*$", "", t.strip(),
               flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip(" ,;:-–—()")
    return t[:60]


def answer_boxes(text: str, view: Sequence[float]
                 ) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """``(label, page box)`` for every thing-sized 0-999 box in an answer
    on ``view`` — the label is the words just before the box on its line
    (or the line's words, when none) — in PDF points. Region-sized boxes
    (:data:`REGION_GRID`) and anything that is not a box are left out."""
    out = []
    for line in str(text or "").splitlines():
        prev = 0
        for m in _GRID_BOX_RE.finditer(line):
            vals = [float(v) for v in m.groups()]
            before, prev = line[prev:m.start()], m.end()
            x0, y0, x1, y1 = vals
            if not (all(0 <= v <= 999 for v in vals) and x1 > x0 and y1 > y0):
                continue
            if x1 - x0 > REGION_GRID or y1 - y0 > REGION_GRID:
                continue
            label = _label_text(before) or _label_text(
                _GRID_BOX_RE.sub(" ", line))
            try:
                out.append((label, image_box_to_page(view, vals)))
            except ValueError:
                continue
    return out


# ---------------------------------------------------------------------------
# Grounding a vision call (GEOTECH_VISION_TEXT_CONTEXT / _STRUCTURED)
# ---------------------------------------------------------------------------
#
# A vision call is a fresh, one-shot call: it sees the image and the agent's
# prompt and nothing else. On most CAD sheets the exact words are in the PDF's
# text layer, and a vision model reading 5-point lettering off pixels misreads
# characters the text layer already has right (the review of 2026-09-26). So,
# behind a switch, the call is told which strings the text layer holds inside
# its view and where — or that there are none — and, behind another, it is
# asked for what it located in a form the tools can turn into page boxes.

#: Most characters of text-layer lines put in front of one vision call.
TEXT_CONTEXT_CHARS = 3500

#: A line more than this fraction U+FFFD is not text worth quoting.
_UNMAPPED_LINE = 0.3


def page_lines(source, page: int
               ) -> Optional[Tuple[List[Tuple[str, Tuple[float, ...]]], bool]]:
    """The page's text-layer lines as ``(text, bbox)`` in displayed points,
    and whether the page's text layer is reliable. ``([], True)`` when the
    page has no text layer; ``None`` when the page could not be read at all
    (so nobody tells a vision call "no text layer" on a guess). Never
    raises."""
    try:
        from planlens.document import Document
        doc = (Document(content=source) if isinstance(source, (bytes, bytearray))
               else Document(filepath=str(source)))
    except Exception:
        return None
    try:
        try:
            reliable = bool(getattr(doc.summary(int(page)), "text_reliable", True))
        except Exception:
            reliable = True
        pc = doc.page(int(page), tables=False)
        out = []
        for ln in pc.lines:
            text = " ".join(str(ln.text or "").split())
            if not text:
                continue
            if text.count("�") > _UNMAPPED_LINE * len(text):
                continue
            out.append((text, tuple(float(v) for v in ln.bbox)))
        return out, reliable
    except Exception:
        return None
    finally:
        try:
            doc.close()
        except Exception:
            pass


def text_context(lines, clip, reliable: bool = True,
                 limit: int = TEXT_CONTEXT_CHARS,
                 size: Optional[Sequence[int]] = None) -> str:
    """The block put in front of a vision prompt: the text-layer lines whose
    centre falls inside ``clip`` (points), each with its box on the image of
    that view — in pixels (``px=[..]``) when the image's ``size`` (w, h) is
    given, the same convention the answer is asked for, otherwise on the
    0-999 grid — or a plain statement that the view has no text layer, so
    every word must be read off the image."""
    x0, y0, x1, y1 = (float(v) for v in clip)
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    rows = []
    for text, (bx0, by0, bx1, by1) in lines:
        cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        if size:
            sw, sh = (float(v) for v in size)
            box = [max(0, min(round(sw), round((bx0 - x0) / w * sw))),
                   max(0, min(round(sh), round((by0 - y0) / h * sh))),
                   max(0, min(round(sw), round((bx1 - x0) / w * sw))),
                   max(0, min(round(sh), round((by1 - y0) / h * sh)))]
            rows.append(f"px={box} {text}")
            continue
        box = [max(0, min(999, round((bx0 - x0) / w * 999))),
               max(0, min(999, round((by0 - y0) / h * 999))),
               max(0, min(999, round((bx1 - x0) / w * 999))),
               max(0, min(999, round((by1 - y0) / h * 999)))]
        rows.append(f"{box} {text}")
    if not rows:
        return ("This view has NO text layer: every word in it has to be read "
                "from the image itself.")
    grid = ("in pixels of the image (px=[x0, y0, x1, y1])" if size
            else "on the same 0-999 grid as the image")
    head = (f"The PDF's own text layer inside this view (exact strings, each "
            f"with its box {grid}). Where a string "
            "below covers what you are reading, use it exactly; read from the "
            "image only what it does not cover (lettering drawn as lines, "
            "symbols, how things connect), and say which is which.")
    if not reliable:
        head += (" WARNING: this page's text layer is partly undecodable, so "
                 "check each string against the image.")
    body, used, dropped = [], 0, 0
    for r in rows:
        if used + len(r) + 1 > limit:
            dropped += 1
            continue
        body.append(r)
        used += len(r) + 1
    tail = f"\n... and {dropped} more line(s) not listed." if dropped else ""
    return head + "\n" + "\n".join(body) + tail


#: Appended to a vision prompt when structured locations are switched on.
LOCATED_INSTRUCTION = (
    "After your answer, end with ONE line that starts with LOCATED: followed by "
    "a JSON list of the things you located, each "
    '{"what": short label, "text": the exact characters you read or null, '
    '"box": [x0, y0, x1, y1] on the 0-999 grid, "sure": true or false}. '
    "Write LOCATED: [] if you located nothing.")


#: The same, asking for each box in pixels of the image (``{w}`` x ``{h}``).
LOCATED_PX_INSTRUCTION = (
    "After your answer, end with ONE line that starts with LOCATED: followed by "
    "a JSON list of the things you located, each "
    '{{"what": short label, "text": the exact characters you read or null, '
    '"px": [x0, y0, x1, y1] in pixels of this {w} x {h} image, "sure": true '
    "or false}}. Write LOCATED: [] if you located nothing.")


def with_locations(prompt: str, size: Optional[Sequence[int]] = None) -> str:
    """``prompt`` with the structured LOCATED instruction appended — boxes in
    pixels of an image of ``size`` (w, h), or with no size on the 0-999
    grid."""
    if size:
        w, h = (int(v) for v in size)
        return (f"{prompt.rstrip()}\n\n"
                f"{LOCATED_PX_INSTRUCTION.format(w=w, h=h)}")
    return f"{prompt.rstrip()}\n\n{LOCATED_INSTRUCTION}"


def split_located(text: str, view, size: Optional[Sequence[int]] = None
                  ) -> Tuple[str, List[Dict[str, Any]]]:
    """``(answer without the LOCATED line, located items)`` — each item's
    box given as ``image_box`` (0-999) and as ``page_bbox`` in PDF points on
    the page, which ``render_region(bbox=...)`` takes directly. An item's
    ``px`` box (pixels of the image, ``size`` = its w, h as sent) is
    converted exactly; an old-style ``box`` is read on the 0-999 grid (in
    pixels if a value is past 999 and the size is known). Never raises: an
    answer with no parseable LOCATED line comes back whole with no items."""
    import json as _json
    if not text or "LOCATED:" not in text:
        return text, []
    at = text.rfind("LOCATED:")
    head, tail = text[:at].rstrip(), text[at + len("LOCATED:"):]
    start = tail.find("[")
    if start < 0:
        return text, []
    depth, end = 0, None
    for i, ch in enumerate(tail[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return text, []
    try:
        items = _json.loads(tail[start:end])
    except ValueError:
        return text, []
    after = tail[end:].strip()
    if after:                       # anything written after the list stays
        head = f"{head}\n{after}" if head else after
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        row = {k: it.get(k) for k in ("what", "text", "sure") if k in it}
        px, box = it.get("px"), it.get("box")
        try:
            if size and isinstance(px, (list, tuple)) and len(px) == 4:
                pxb = [float(v) for v in px]
            elif (size and isinstance(box, (list, tuple)) and len(box) == 4
                  and any(float(v) > _GRID_MAX for v in box)):
                pxb = [float(v) for v in box]        # a pixel box, untagged
            else:
                pxb = None
            if pxb is not None:
                row["image_box"] = px_to_grid(pxb, size)
                row["page_bbox"] = [round(v, 1) for v in
                                    px_box_to_page(view, pxb, size)]
            elif isinstance(box, (list, tuple)) and len(box) == 4:
                row["image_box"] = [float(v) for v in box]
                row["page_bbox"] = [round(v, 1) for v in
                                    image_box_to_page(view, box)]
            if "page_bbox" in row:
                window = zoom_window(view, row["page_bbox"])
                if window is not None:
                    row["zoom_bbox"] = window
        except (TypeError, ValueError):
            for k in ("image_box", "page_bbox", "zoom_bbox"):
                row.pop(k, None)
        out.append(row)
    return head, out


def image_box_to_page(view: Sequence[float], image_box: Sequence[float]
                      ) -> Tuple[float, float, float, float]:
    """A 0-999 box on the image of ``view`` as PDF points on the page."""
    if len(view) != 4 or len(image_box) != 4:
        raise ValueError("view and image_box must each be [x0, y0, x1, y1]")
    vx0, vy0, vx1, vy1 = (float(v) for v in view)
    if vx1 <= vx0 or vy1 <= vy0:
        raise ValueError("view must be a non-empty [x0, y0, x1, y1] rect")
    x0, y0, x1, y1 = (min(999.0, max(0.0, float(v))) for v in image_box)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    w, h = vx1 - vx0, vy1 - vy0
    return (vx0 + x0 / 999.0 * w, vy0 + y0 / 999.0 * h,
            vx0 + x1 / 999.0 * w, vy0 + y1 / 999.0 * h)


__all__ = ["BUDGET_ENV", "DETAIL_ENV", "CHART_BUDGET_ENV", "POLICY_ENV",
           "MAX_PX_ENV", "DEFAULT_MAX_PX", "max_px",
           "POLICIES", "policy", "ASSUMED_CAD_LETTERING_PT", "tile_grid",
           "tile_boxes", "DEFAULT_BUDGET",
           "DEFAULT_CHART_BUDGET", "LEGIBLE_TEXT_PX", "TARGET_TEXT_PX",
           "chart_reading", "READING_INSTRUCTION", "GRID_INSTRUCTION",
           "PIXEL_INSTRUCTION", "pixel_instruction", "with_location",
           "ZOOM_HINT", "engine_profile",
           "budget_name", "budget", "detail", "image_media_type",
           "render_view", "view_payload", "with_grid", "image_box_to_page",
           "px_to_grid", "px_box_to_page", "boxes_to_grid", "boxes_note",
           "LOCATION_ERROR_FRAC", "MIN_LOCATION_ERROR_PT", "MARK_VIEW_PT",
           "location_error", "precision_note", "NOT_A_MEASUREMENT",
           "zoom_pad", "zoom_window",
           "TEXT_CONTEXT_CHARS", "page_lines", "text_context",
           "LOCATED_INSTRUCTION", "LOCATED_PX_INSTRUCTION", "with_locations",
           "split_located", "BRACKETED_NOTE", "bracketed_note",
           "PATCH_ALIGN_ENV", "PATCH_PX", "patch_align", "align_to_patches",
           "AIM_TOLERANCE_FRAC", "aim_tolerance", "answer_boxes",
           "REGION_GRID"]

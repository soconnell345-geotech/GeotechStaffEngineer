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
* **Turned lettering is sent upright.** Where the page's text layer shows
  the lettering in a view running up, down or upside down, the image is sent
  turned so it reads upright (:func:`reading_turn`, :func:`turn_image`) —
  Sol misread turned lettering far more often than upright (Foundry brief 5;
  those sheets had no text layer, so this does not reach them). The view
  stays the page rect as displayed, and every pixel box is turned back in
  code before it becomes a 0-999 box, so ``view`` + ``image_box`` mean the
  same thing whether or not the image was turned. A sheet with no text
  layer (lettering drawn as strokes, a scan) is sent as it is.
* **Lettering cut by the window's edge.** A zoom whose edge cuts text-layer
  lines is widened, within a limit, to take them in whole
  (:func:`widen_to_lines`), and the vision call is told which lines its
  edges still cut (:func:`cut_note`).

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
from typing import (
    Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple,
)

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
#: Its last sentence (VIS-04): a zoom window that cut a turned note's lines
#: came back with the cut words completed as plausible values -- a 4 in pipe
#: and 6 in spacing where 6 in and 3 in are printed (Foundry brief 5,
#: 2026-10-08) -- while the zooms either side said "cut off".
READING_INSTRUCTION = (
    "Read codes, tags and numbers character by character and give your best "
    "reading of each. Only where a character truly cannot be told apart in "
    "this image (the usual confusions are G/Q/O/C/D, E/F, B/8, S/5, I/1/L, "
    "Z/2) write the alternatives in brackets, e.g. A[B/8]C; lettering you "
    "can read, write plainly. If the lettering is too small to read "
    "reliably, say so rather than guessing. Where lettering runs off the "
    "edge of the image, quote only the characters you can see and write "
    "(cut) where they stop; never complete a cut word or line.")

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

#: Said with every location request (VIS-05): one whole-page answer on a
#: sheet drawn sideways gave every box in the frame of the sheet turned
#: upright to read -- the same box turned 90 degrees, about 300 pt from the
#: thing (Foundry brief 5, 2026-10-08, 1 look of 28).
FRAME_INSTRUCTION = (
    "Give every box in this image as it is, even where the drawing or its "
    "lettering is turned sideways: never in a frame turned so that the "
    "lettering reads upright.")

#: The old location sentence: boxes on a 0-999 grid. Used only where the
#: image's size is not known; a 0-999 answer is still understood everywhere.
GRID_INSTRUCTION = (
    "If you give the location of anything in this image, give it as a box "
    "[x0, y0, x1, y1] on a 0-999 grid over the whole image (origin at the "
    "top-left corner, x to the right, y down). " + FRAME_INSTRUCTION + " "
    + READING_INSTRUCTION)

#: The location sentence every vision prompt ends with when the image's size
#: is known: boxes in PIXELS of the image, tagged ``px=`` so they cannot be
#: mistaken for anything else (:func:`boxes_to_grid` converts them).
PIXEL_INSTRUCTION = (
    "This image is {w} x {h} pixels. If you give the location of anything "
    "in it, give it as a box in PIXELS of this image, written px=[x0, y0, "
    "x1, y1]: origin at the top-left corner, x to the right (0 to {w}), y "
    "down (0 to {h}). Box the thing itself, tightly, and write px= before "
    "every box. " + FRAME_INSTRUCTION + " " + READING_INSTRUCTION)

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
                engine=None, upright: bool = False,
                turn: Optional[int] = None,
                text_layer: Optional["TextLayer"] = None,
                default_turn: int = 0,
                ) -> Tuple[bytes, Dict[str, Any]]:
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

    ``upright`` sends the view turned so that its lettering reads upright,
    when the page's text layer says which way it runs (:func:`reading_turn`;
    ``text_layer`` is that layer when the caller already has it), and turned
    ``default_turn`` when the text layer cannot say (a turn earlier looks
    found for the page). ``turn`` (0/90/180/270, clockwise) turns it by hand
    instead. A turned image carries ``info['turn']`` and where the turn came
    from (``info['turn_from']``: ``text_layer``, ``remembered`` or
    ``asked``), and ``width_px``/``height_px`` are its size AS SENT; ``clip``
    is still the page rect it shows, as the page displays it — pixel boxes
    on it go through :func:`unturn_px_box` (which :func:`px_to_grid`,
    :func:`px_box_to_page`, :func:`boxes_to_grid` and :func:`split_located`
    do when given the turn). ``info['direction_known']`` says the text layer
    decided the view's direction (upright included).
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
        quarter, source, known = 0, None, None
        if turn is not None:
            quarter = int(turn) % 360 if int(turn) % 90 == 0 else 0
            source = "asked"
        elif upright:
            layer = (text_layer if text_layer is not None
                     else _text_layer_of(doc, int(page)))
            known = reading_turn(layer, info["clip"], default=None)
            if known is None:
                quarter = int(default_turn or 0) % 360
                quarter = quarter if quarter in TURNS else 0
                source = "remembered"
            else:
                quarter, source = known, "text_layer"
        if quarter:
            data, info = turn_image(data, info, quarter)
            if info.get("turn"):
                info["turn_from"] = source
        if known is not None:
            info = dict(info)
            info["direction_known"] = True
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
        import pymupdf
        src = pymupdf.Pixmap(data)
        if src.alpha:
            src = pymupdf.Pixmap(src, 0)
        if (src.width, src.height) != (w, h):
            return data, info
        canvas = pymupdf.Pixmap(src.colorspace, pymupdf.IRect(0, 0, tw, th), False)
        canvas.clear_with(255)
        canvas.copy(src, pymupdf.IRect(0, 0, w, h))
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
    quarter = int(info.get("turn") or 0)
    if quarter:
        out["turned"] = turned_note(quarter, info.get("turn_from"))
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
#: as lines): 0.06 in, the small end of plotted half-size sheets — the
#: drawing set of the 2026-09-25 test run.
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
    # The image's width ALONG the view's x: its height when sent turned.
    across, _down = unturned_size((info["width_px"], info["height_px"]),
                                  info.get("turn") or 0)
    px_per_pt = across / max(x1 - x0, 1e-6)
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


def px_to_grid(px_box: Sequence[float], size: Sequence[int],
               turn: int = 0) -> List[int]:
    """A pixel box on an image of ``size`` (w, h) as a whole-number box on
    the 0-999 grid over the same image (corners ordered, clamped). For an
    image sent turned ``turn`` degrees clockwise (``size`` as sent), the
    grid is over the view as the page shows it: the box is turned back
    first (:func:`unturn_px_box`)."""
    if int(turn or 0) % 360:
        px_box = unturn_px_box(px_box, size, turn)
        size = unturned_size(size, turn)
    w, h = (max(float(v), 1.0) for v in size)
    x0, y0, x1, y1 = (float(v) for v in px_box)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    def g(v, s):
        return int(max(0, min(999, round(v / s * 999.0))))
    return [g(x0, w), g(y0, h), g(x1, w), g(y1, h)]


def px_box_to_page(view: Sequence[float], px_box: Sequence[float],
                   size: Sequence[int], turn: int = 0
                   ) -> Tuple[float, float, float, float]:
    """A pixel box on the image of ``view`` (``size`` = the image's w, h as
    SENT) as PDF points on the page — the exact conversion, with no grid in
    between. ``turn`` is how far (degrees clockwise) the image was turned
    before it was sent; the box is turned back first."""
    if len(view) != 4 or len(px_box) != 4:
        raise ValueError("view and the box must each be [x0, y0, x1, y1]")
    vx0, vy0, vx1, vy1 = (float(v) for v in view)
    if vx1 <= vx0 or vy1 <= vy0:
        raise ValueError("view must be a non-empty [x0, y0, x1, y1] rect")
    if int(turn or 0) % 360:
        px_box = unturn_px_box(px_box, size, turn)
        size = unturned_size(size, turn)
    w, h = (max(float(v), 1.0) for v in size)
    x0, y0, x1, y1 = (float(v) for v in px_box)
    fx0, fx1 = sorted((min(1.0, max(0.0, x0 / w)), min(1.0, max(0.0, x1 / w))))
    fy0, fy1 = sorted((min(1.0, max(0.0, y0 / h)), min(1.0, max(0.0, y1 / h))))
    vw, vh = vx1 - vx0, vy1 - vy0
    return (vx0 + fx0 * vw, vy0 + fy0 * vh, vx0 + fx1 * vw, vy0 + fy1 * vh)


def boxes_to_grid(text: str, size: Optional[Sequence[int]],
                  tagged_only: bool = False,
                  turn: int = 0) -> Tuple[str, Dict[str, int]]:
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
    ``turn``: the image was sent turned that many degrees clockwise
    (``size`` is its size as sent); each pixel box is turned back, so the
    0-999 box is on the view as the page shows it.
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
            return "[{}, {}, {}, {}]".format(*px_to_grid(vals, (w, h), turn))
        counts["grid"] += 1
        return m.group(0)

    return _BOX.sub(rewrite, text), counts


def boxes_note(counts: Dict[str, int], size: Sequence[int],
               turn: int = 0, what: str = "page") -> Optional[str]:
    """What the agent is told about the boxes in an answer, or ``None`` when
    it gave none. ``what`` names the thing looked at (``page``, or
    ``image`` for an image file)."""
    w, h = (int(v) for v in size)
    quarter = int(turn or 0) % 360
    if counts.get("converted"):
        left = counts.get("grid") or 0
        turned = (f" (sent turned {quarter} degrees clockwise; each box was "
                  f"turned back, so it is on this view as the {what} shows "
                  f"it)" if quarter else "")
        return (f"the boxes in this analysis are on the 0-999 grid over this "
                f"view's image, converted in code from the vision call's "
                f"pixel boxes on the {w} x {h} px image it was sent{turned}: "
                f"pass one as image_box with this view"
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
                 size: Optional[Sequence[int]] = None,
                 turn: int = 0) -> str:
    """The block put in front of a vision prompt: the text-layer lines whose
    centre falls inside ``clip`` (points), each with its box on the image of
    that view — in pixels (``px=[..]``) when the image's ``size`` (w, h) is
    given, the same convention the answer is asked for, otherwise on the
    0-999 grid — or a plain statement that the view has no text layer, so
    every word must be read off the image. ``turn``: the image is sent
    turned that many degrees clockwise (``size`` as sent), and the pixel
    boxes are given on it as sent."""
    x0, y0, x1, y1 = (float(v) for v in clip)
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    rows = []
    for text, (bx0, by0, bx1, by1) in lines:
        cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        if size:
            sw, sh = (float(v) for v in size)
            uw, uh = unturned_size(size, turn)
            box = turn_px_box([(bx0 - x0) / w * uw, (by0 - y0) / h * uh,
                               (bx1 - x0) / w * uw, (by1 - y0) / h * uh],
                              (uw, uh), turn)
            box = [max(0, min(round(sw), round(box[0]))),
                   max(0, min(round(sh), round(box[1]))),
                   max(0, min(round(sw), round(box[2]))),
                   max(0, min(round(sh), round(box[3])))]
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


def split_located(text: str, view, size: Optional[Sequence[int]] = None,
                  turn: int = 0) -> Tuple[str, List[Dict[str, Any]]]:
    """``(answer without the LOCATED line, located items)`` — each item's
    box given as ``image_box`` (0-999) and as ``page_bbox`` in PDF points on
    the page, which ``render_region(bbox=...)`` takes directly. An item's
    ``px`` box (pixels of the image, ``size`` = its w, h as sent, ``turn``
    = degrees clockwise it was turned before sending) is converted exactly;
    an old-style ``box`` is read on the 0-999 grid (in pixels if a value is
    past 999 and the size is known). Never raises: an answer with no
    parseable LOCATED line comes back whole with no items."""
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
                row["image_box"] = px_to_grid(pxb, size, turn)
                row["page_bbox"] = [round(v, 1) for v in
                                    px_box_to_page(view, pxb, size, turn)]
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


# ---------------------------------------------------------------------------
# Turned lettering (VIS-01, VIS-05) and lettering cut by a window (VIS-04)
# ---------------------------------------------------------------------------
#
# Foundry brief 5 (2026-10-08, module_work/review_eval_results/
# 2026-10-08_foundry_5.33.0rc1/): Sol read turned lettering markedly worse
# than upright (6 of 15 turned look-alike tags misread, 0 of 11 upright;
# sheet numbers and titles misread on sideways sheets); one whole-page
# answer gave its boxes in the frame of the sheet turned to read; and a zoom
# whose edge cut a note's lines came back with the cut words made up. Where
# the page's text layer says which way its lines run, the view is sent so
# they read upright and every box is turned back in code; a zoom is widened
# to take in whole the lines its edge cuts, and the lines it still cuts are
# named to the vision call. Where there is no text layer the image goes as
# before, and the general sentences in :data:`FRAME_INSTRUCTION` and
# :data:`READING_INSTRUCTION` are all the help it gets. Every sheet of the
# brief-5 cases above is such a sheet: the Mecklenburg set and the synthetic
# tag sets carry their lettering as strokes, with no text layer.

#: The turns a view's image can be sent at: degrees CLOCKWISE from the page
#: as displayed (0 = as the page shows it).
TURNS = (0, 90, 180, 270)

#: A view is turned when at least this share of its lettering (characters,
#: each line weighted by how much of it lies in the view) runs one turned
#: way ...
TURN_SHARE = 0.6
#: ... and that is at least this many characters ...
TURN_MIN_CHARS = 3
#: ... and at least this many per square inch of the view: a whole sheet is
#: not turned for a vertical stamp when the rest of its lettering is drawn
#: as strokes the text layer cannot see.
TURN_MIN_CHARS_PER_SQ_IN = 0.5
#: A line within this many degrees of a quarter turn runs that way; lettering
#: at other angles (a name along a curved road) counts against every turn.
TURN_SNAP_DEG = 10.0

#: The quality a turned JPEG is written at (planlens' own render quality).
TURN_JPEG_QUALITY = 92

#: Which way lettering runs on the page, by the turn that sets it upright.
_RUNS = {90: "from bottom to top", 180: "upside down",
         270: "from top to bottom"}

#: The image's edges in clockwise order: a quarter turn clockwise moves each
#: edge one place on.
EDGES = ("top", "right", "bottom", "left")

#: A text-layer line is cut by a view's edge when it runs at least this far
#: past it, and lies at least this far inside it (glyph boxes carry slack).
CUT_TOL_PT = 1.0
#: A zoom window is widened to take in whole the lines its edges cut, by at
#: most this fraction of its own size on each side ...
WIDEN_MAX_FRAC = 0.25
#: ... or this many points when that is more (a small window).
WIDEN_MIN_PT = 12.0
#: Paper left round a line taken in.
WIDEN_MARGIN_PT = 2.0
#: Most cut lines named to one vision call.
CUT_LINES_LISTED = 8

#: planlens' ``model.SOURCE_CAD_HIDDEN``: AutoCAD's hidden "SHX Text" — one
#: box for a whole block of rows and no direction, so it neither turns a view
#: nor counts as a cut line.
_CAD_HIDDEN = "cad_hidden_text"


class LayerLine(NamedTuple):
    """One line of the page's text layer: its text, its box (displayed
    points, top-left origin) and its reading direction (degrees
    counter-clockwise as displayed; ``None`` when not known)."""
    text: str
    bbox: Tuple[float, float, float, float]
    rotation: Optional[float]


class TextLayer(NamedTuple):
    """A page's text-layer lines and the page's displayed size in points."""
    lines: List[LayerLine]
    width: float
    height: float


def _text_layer_of(doc, page: int) -> Optional[TextLayer]:
    """:func:`page_text_layer` on an open planlens ``Document``."""
    try:
        pc = doc.page(int(page), tables=False)
        lines = []
        for ln in pc.lines:
            if getattr(ln, "source", None) == _CAD_HIDDEN:
                continue
            text = " ".join(str(ln.text or "").split())
            if not text:
                continue
            rot = getattr(ln, "rotation", None)
            lines.append(LayerLine(text, tuple(float(v) for v in ln.bbox),
                                   None if rot is None else float(rot)))
        return TextLayer(lines, float(pc.width), float(pc.height))
    except Exception:  # noqa: BLE001 - no layer is never a failed render
        return None


def page_text_layer(source, page: int) -> Optional[TextLayer]:
    """The page's text-layer lines with their boxes and reading directions,
    and its size — ``None`` when the page cannot be read. Never raises."""
    try:
        from planlens.document import Document
        doc = (Document(content=source) if isinstance(source, (bytes, bytearray))
               else Document(filepath=str(source)))
    except Exception:  # noqa: BLE001
        return None
    try:
        return _text_layer_of(doc, page)
    finally:
        try:
            doc.close()
        except Exception:  # noqa: BLE001
            pass


def _inside(bbox, clip) -> Tuple[float, float]:
    """How far (x, y, points) a box and a clip overlap; <= 0 when apart."""
    return (min(bbox[2], clip[2]) - max(bbox[0], clip[0]),
            min(bbox[3], clip[3]) - max(bbox[1], clip[1]))


def reading_turn(layer: Optional[TextLayer], clip: Sequence[float],
                 default: Optional[int] = 0) -> Optional[int]:
    """The turn (degrees clockwise, :data:`TURNS`) that sets the lettering
    in ``clip`` upright, from the text layer's own reading directions: 90
    for lines that run from bottom to top, 270 from top to bottom, 180
    upside down. 0 when the lettering already reads upright, or when no one
    way holds :data:`TURN_SHARE` of it. ``default`` when the view has too
    little lettering with a known direction to say
    (:data:`TURN_MIN_CHARS`, :data:`TURN_MIN_CHARS_PER_SQ_IN`) — ``None``
    asks to be told so."""
    if not layer or not layer.lines:
        return default
    x0, y0, x1, y1 = (float(v) for v in clip)
    weight = {t: 0.0 for t in TURNS}
    total = 0.0
    for ln in layer.lines:
        if ln.rotation is None:
            continue
        ix, iy = _inside(ln.bbox, (x0, y0, x1, y1))
        if ix <= 0 or iy <= 0:
            continue
        area = (max(ln.bbox[2] - ln.bbox[0], 1e-6)
                * max(ln.bbox[3] - ln.bbox[1], 1e-6))
        n = len(ln.text.replace(" ", "")) * min(1.0, ix * iy / area)
        total += n
        r = float(ln.rotation) % 360.0
        quarter = (int(round(r / 90.0)) % 4) * 90
        if abs((r - quarter + 180.0) % 360.0 - 180.0) <= TURN_SNAP_DEG:
            weight[quarter] += n
    best = max((90, 180, 270), key=lambda t: weight[t])
    sq_in = (x1 - x0) * (y1 - y0) / (72.0 * 72.0)
    floor = max(float(TURN_MIN_CHARS), TURN_MIN_CHARS_PER_SQ_IN * sq_in)
    if total < floor:
        return default
    if weight[best] >= floor and weight[best] >= TURN_SHARE * total:
        return best
    return 0


def turn_image(data: bytes, info: Dict[str, Any], turn: int
               ) -> Tuple[bytes, Dict[str, Any]]:
    """The image turned ``turn`` degrees clockwise (a quarter turn moves
    whole pixels: nothing is resampled), in its own format, with ``info``
    giving its size AS SENT and ``turn``. ``clip`` — the page rect the image
    shows — is unchanged. Left as it was for no turn, or an image that
    cannot be decoded."""
    quarter = int(turn or 0) % 360
    if quarter not in TURNS or not quarter:
        return data, info
    try:
        import pymupdf
        import numpy as np
        pix = pymupdf.Pixmap(data)
        a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        r = np.ascontiguousarray(np.rot90(a, k=-(quarter // 90)))
        out = pymupdf.Pixmap(pix.colorspace, r.shape[1], r.shape[0], r.tobytes(),
                          pix.alpha)
        fmt = str(info.get("format") or "png").lower()
        body = (out.tobytes("jpeg", jpg_quality=TURN_JPEG_QUALITY)
                if fmt == "jpeg" else out.tobytes("png"))
    except Exception:  # noqa: BLE001 - an image that will not turn goes as is
        return data, info
    turned = dict(info)
    turned["width_px"], turned["height_px"] = out.width, out.height
    turned["turn"] = quarter
    return body, turned


def turned_note(turn: int, source: Optional[str] = "text_layer",
                what: str = "page") -> str:
    """What a result says about a view sent turned ``turn`` degrees, by
    where the turn came from (``render_view``'s ``info['turn_from']``, or
    ``retry`` for a view read again after a first look found it turned).
    ``what`` names the thing looked at (``page``, or ``image`` for an image
    file)."""
    quarter = int(turn or 0) % 360
    runs = _RUNS.get(quarter, "turned")
    why = {
        "asked": "as asked",
        "remembered": (f"because earlier looks at this {what} found its "
                       f"lettering running {runs}: the vision call read it "
                       f"upright"),
        "retry": (f"because a first look, as the {what} shows it, found its "
                  f"lettering running {runs}: this is a second reading, "
                  f"turned so the lettering read upright"),
    }.get(str(source), (f"because the page's text layer shows its lettering "
                        f"running {runs}: the vision call read it upright"))
    return (f"this view's image was sent turned {quarter} degrees clockwise, "
            f"{why}. Its boxes were turned back in code, so every image_box "
            f"is on this view as the {what} shows it; words such as top or "
            f"left in the analysis describe the turned image.")


#: Asked of a look whose image goes as the page shows it, with
#: ``GEOTECH_VISION_TURN_RETRY`` on (:func:`read_turn_tag` reads the line).
TURN_TAG_INSTRUCTION = (
    "If most of the lettering in this image does not read from left to "
    "right, end your answer with one line saying which way it runs: "
    "TURNED: bottom-to-top, TURNED: top-to-bottom or TURNED: upside-down.")

#: The tagged line, however a model dresses it (bold, a list mark, a full
#: stop).
_TURN_TAG = re.compile(r"^[ \t*_>`\-]*TURNED[ \t*]*:[ \t]*(.*?)[ \t*_`.]*$",
                       re.IGNORECASE | re.MULTILINE)


def _turn_for(phrase: str) -> Optional[int]:
    """The turn (degrees clockwise) that sets lettering running ``phrase``
    upright, or ``None`` for a phrase that names no turned direction."""
    p = " " + re.sub(r"[-_]", " ", str(phrase or "").lower()) + " "
    if "upside" in p:
        return 180
    b, t = p.find("bottom"), p.find(" top")
    if b >= 0 and t >= 0:
        return 90 if b < t else 270
    if " up " in p or "upward" in p:
        return 90
    if " down " in p or "downward" in p:
        return 270
    return None


def read_turn_tag(text: Any) -> Tuple[Any, Optional[int]]:
    """``(answer without its TURNED line, the turn that sets its lettering
    upright)`` for an answer to :data:`TURN_TAG_INSTRUCTION`; the turn is
    ``None`` when there is no such line or it names no turned direction (the
    line is dropped either way). The last such line counts."""
    if not isinstance(text, str) or "turned" not in text.lower():
        return text, None
    found = list(_TURN_TAG.finditer(text))
    if not found:
        return text, None
    m = found[-1]
    rest = (text[:m.start()].rstrip() + "\n" + text[m.end():].lstrip()).strip()
    return rest, _turn_for(m.group(1))


def unturned_size(size: Sequence[int], turn: int = 0) -> Tuple[int, int]:
    """(w, h) of the image as the page shows it, for one of ``size`` (w, h)
    sent turned ``turn`` degrees — and the other way round: a quarter turn
    swaps the sides."""
    w, h = (int(v) for v in size)
    return (h, w) if int(turn or 0) % 180 == 90 else (w, h)


def _ordered(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def unturn_px_box(px_box: Sequence[float], size: Sequence[int],
                  turn: int = 0) -> List[float]:
    """A pixel box on an image sent turned ``turn`` degrees clockwise
    (``size`` = its w, h AS SENT) as the same box on the image before the
    turn — the view as the page shows it. Corners ordered."""
    x0, y0, x1, y1 = _ordered(px_box)
    w, h = (float(v) for v in size)
    quarter = int(turn or 0) % 360
    if quarter == 90:
        return [y0, w - x1, y1, w - x0]
    if quarter == 180:
        return [w - x1, h - y1, w - x0, h - y0]
    if quarter == 270:
        return [h - y1, x0, h - y0, x1]
    return [x0, y0, x1, y1]


def turn_px_box(px_box: Sequence[float], size: Sequence[int],
                turn: int = 0) -> List[float]:
    """The reverse of :func:`unturn_px_box`: a pixel box on the image as the
    page shows it (``size`` = ITS w, h) as the box on that image turned
    ``turn`` degrees clockwise."""
    u0, v0, u1, v1 = _ordered(px_box)
    w, h = (float(v) for v in size)
    quarter = int(turn or 0) % 360
    if quarter == 90:
        return [h - v1, u0, h - v0, u1]
    if quarter == 180:
        return [w - u1, h - v1, w - u0, h - v0]
    if quarter == 270:
        return [v0, w - u1, v1, w - u0]
    return [u0, v0, u1, v1]


def turn_edge(edge: str, turn: int = 0) -> str:
    """Which edge of an image turned ``turn`` degrees clockwise the given
    edge of the unturned image becomes (the top goes to the right)."""
    return EDGES[(EDGES.index(edge) + int(turn or 0) // 90) % 4]


def cut_lines(clip: Sequence[float], layer: Optional[TextLayer]
              ) -> List[Tuple[LayerLine, List[str]]]:
    """Every text-layer line that ``clip``'s edges cut, with the edges
    (``top``/``right``/``bottom``/``left`` of the view as the page shows it)
    it runs past — a line partly in the view, not one wholly inside or
    wholly outside it."""
    if not layer or not layer.lines:
        return []
    x0, y0, x1, y1 = (float(v) for v in clip)
    out = []
    for ln in layer.lines:
        ix, iy = _inside(ln.bbox, (x0, y0, x1, y1))
        if ix <= CUT_TOL_PT or iy <= CUT_TOL_PT:
            continue
        bx0, by0, bx1, by1 = ln.bbox
        edges = [e for e, past in (("top", y0 - by0), ("right", bx1 - x1),
                                   ("bottom", by1 - y1), ("left", x0 - bx0))
                 if past > CUT_TOL_PT]
        if edges:
            out.append((ln, edges))
    return out


def widen_to_lines(clip: Sequence[float], layer: Optional[TextLayer]
                   ) -> Tuple[Tuple[float, float, float, float], int]:
    """``(clip, n)``: ``clip`` widened to take in whole the ``n`` text-layer
    lines its edges cut — each only when all of it lies within
    :data:`WIDEN_MAX_FRAC` of the clip's size (at least
    :data:`WIDEN_MIN_PT`) of the edge, so a zoom is never spread so wide its
    lettering shrinks away; held to the page. A window narrow enough to
    place a mark from (:data:`MARK_VIEW_PT`) stays that narrow: a line that
    would take it past is left out. One pass, nearest lines first: a line
    the wider window then cuts is named to the vision call
    (:func:`cut_note`), not chased."""
    x0, y0, x1, y1 = (float(v) for v in clip)
    lim_x = max(WIDEN_MIN_PT, WIDEN_MAX_FRAC * (x1 - x0))
    lim_y = max(WIDEN_MIN_PT, WIDEN_MAX_FRAC * (y1 - y0))
    cap = MARK_VIEW_PT if max(x1 - x0, y1 - y0) <= MARK_VIEW_PT else None
    page_w = layer.width if layer is not None and layer.width > 0 else None
    page_h = layer.height if layer is not None and layer.height > 0 else None

    def grown(box, bx0, by0, bx1, by1):
        g = (min(box[0], bx0 - WIDEN_MARGIN_PT),
             min(box[1], by0 - WIDEN_MARGIN_PT),
             max(box[2], bx1 + WIDEN_MARGIN_PT),
             max(box[3], by1 + WIDEN_MARGIN_PT))
        if page_w is not None and page_h is not None:
            g = (max(0.0, g[0]), max(0.0, g[1]), min(page_w, g[2]),
                 min(page_h, g[3]))
        return g

    def need(ln):
        bx0, by0, bx1, by1 = ln.bbox
        return max(0.0, x0 - bx0, bx1 - x1, y0 - by0, by1 - y1)

    box = (x0, y0, x1, y1)
    taken = 0
    for ln, _edges in sorted(cut_lines(box, layer), key=lambda c: need(c[0])):
        bx0, by0, bx1, by1 = ln.bbox
        if (x0 - bx0 > lim_x or bx1 - x1 > lim_x
                or y0 - by0 > lim_y or by1 - y1 > lim_y):
            continue
        wider = grown(box, bx0, by0, bx1, by1)
        if cap is not None and max(wider[2] - wider[0],
                                   wider[3] - wider[1]) > cap:
            continue
        box = wider
        taken += 1
    return box, taken


def cut_note(cuts: Sequence[Tuple[LayerLine, List[str]]],
             clip: Sequence[float], size: Optional[Sequence[int]],
             turn: int = 0) -> Optional[str]:
    """What a vision call is told about the lines its image's edges cut
    (:func:`cut_lines`): where each one's visible part is — in pixels of the
    image as sent (``size``, turned ``turn`` degrees) — and which edge cuts
    it; ``None`` when none is cut. The text itself is not given: the call
    reads what it can see and says where it stops."""
    if not cuts or not size:
        return None
    x0, y0, x1, y1 = (float(v) for v in clip)
    vw, vh = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    uw, uh = unturned_size(size, turn)
    sw, sh = (int(v) for v in size)
    rows = []
    for ln, edges in list(cuts)[:CUT_LINES_LISTED]:
        bx0, by0 = max(ln.bbox[0], x0), max(ln.bbox[1], y0)
        bx1, by1 = min(ln.bbox[2], x1), min(ln.bbox[3], y1)
        box = turn_px_box([(bx0 - x0) / vw * uw, (by0 - y0) / vh * uh,
                           (bx1 - x0) / vw * uw, (by1 - y0) / vh * uh],
                          (uw, uh), turn)
        box = [max(0, min(sw, round(box[0]))), max(0, min(sh, round(box[1]))),
               max(0, min(sw, round(box[2]))), max(0, min(sh, round(box[3])))]
        sides = " and ".join(turn_edge(e, turn) for e in edges)
        rows.append(f"px={box} (cut by the {sides} edge)")
    more = len(cuts) - len(rows)
    return (f"The page's text layer shows {len(cuts)} line(s) of lettering "
            f"running past the edge of this image: " + "; ".join(rows)
            + (f"; and {more} more" if more > 0 else "")
            + ". Quote only what you can see of each and write (cut) where it "
              "stops.")


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
           "REGION_GRID", "FRAME_INSTRUCTION",
           "TURNS", "TURN_SHARE", "TURN_MIN_CHARS", "TURN_MIN_CHARS_PER_SQ_IN",
           "TURN_SNAP_DEG", "EDGES", "CUT_TOL_PT", "WIDEN_MAX_FRAC",
           "WIDEN_MIN_PT", "WIDEN_MARGIN_PT", "CUT_LINES_LISTED",
           "LayerLine", "TextLayer", "page_text_layer", "reading_turn",
           "turn_image", "turned_note", "unturned_size", "unturn_px_box",
           "turn_px_box", "turn_edge", "cut_lines", "widen_to_lines",
           "cut_note", "TURN_TAG_INSTRUCTION", "read_turn_tag"]

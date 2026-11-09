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
    Any, Dict, Iterator, Optional, Sequence, Tuple,
)

# The page and image frames, tiles, zoom windows and turned lettering moved
# to planlens.review.frames (BACKLOG PL-29, the app split's step 9); the
# names stay here for the app's callers until step 16.
from planlens.review.frames import (  # noqa: F401 - moved, aliased
    aim_tolerance, AIM_TOLERANCE_FRAC, answer_boxes, ASSUMED_CAD_LETTERING_PT,
    _BOX, _box_values, boxes_note, boxes_to_grid, _CAD_HIDDEN, cut_lines,
    CUT_LINES_LISTED, cut_note, CUT_TOL_PT, EDGES, _GRID_BOX_RE, _GRID_MAX,
    image_box_to_page, _inside, _is_box, _label_text, LayerLine,
    LEGIBLE_TEXT_PX, location_error, LOCATION_ERROR_FRAC, MARK_VIEW_PT,
    MAX_TILES_PER_SIDE, MIN_LOCATION_ERROR_PT, NOT_A_MEASUREMENT, _NUM,
    _ordered, page_text_layer, precision_note, px_box_to_page, px_to_grid,
    read_turn_tag, reading_turn, REGION_GRID, _RUNS, TARGET_TEXT_PX,
    _text_layer_of, TextLayer, tile_boxes, tile_grid, turn_edge, _turn_for,
    turn_image, TURN_JPEG_QUALITY, TURN_MIN_CHARS, TURN_MIN_CHARS_PER_SQ_IN,
    turn_px_box, TURN_SHARE, TURN_SNAP_DEG, _TURN_TAG, TURN_TAG_INSTRUCTION,
    turned_note, TURNS, unturn_px_box, unturned_size, WIDEN_MARGIN_PT,
    WIDEN_MAX_FRAC, WIDEN_MIN_PT, widen_to_lines, zoom_pad, zoom_window,
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


def probe_fallback_note(prof) -> Optional[str]:
    """The line a vision result carries when its model's probe FAILED and a
    fallback budget was used (``VisionProfile.fallback_note``; DR-36), else
    ``None``."""
    fn = getattr(prof, "fallback_note", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:             # a record, never the call
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
    ``text_layer`` is that layer when the caller already has it); when the
    text layer cannot say it goes as the page shows it. ``turn``
    (0/90/180/270, clockwise) turns it by hand instead. A turned image
    carries ``info['turn']`` and where the turn came from
    (``info['turn_from']``: ``text_layer`` or ``asked``), and ``width_px``/``height_px`` are its size AS SENT; ``clip``
    is still the page rect it shows, as the page displays it — pixel boxes
    on it go through :func:`unturn_px_box` (which :func:`px_to_grid`,
    :func:`px_box_to_page` and :func:`boxes_to_grid` do when given the
    turn). ``info['direction_known']`` says the text layer
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
        quarter, source, known = 0, None, None
        if turn is not None:
            quarter = int(turn) % 360 if int(turn) % 90 == 0 else 0
            source = "asked"
        elif upright:
            layer = (text_layer if text_layer is not None
                     else _text_layer_of(doc, int(page)))
            known = reading_turn(layer, info["clip"], default=None)
            if known is not None:
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
    note = probe_fallback_note(prof)
    if note:
        # A failed probe's fallback budget, said on the result (DR-36).
        out["vision_probe"] = note
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
           "BRACKETED_NOTE", "bracketed_note",
           "AIM_TOLERANCE_FRAC", "aim_tolerance", "answer_boxes",
           "REGION_GRID", "FRAME_INSTRUCTION",
           "TURNS", "TURN_SHARE", "TURN_MIN_CHARS", "TURN_MIN_CHARS_PER_SQ_IN",
           "TURN_SNAP_DEG", "EDGES", "CUT_TOL_PT", "WIDEN_MAX_FRAC",
           "WIDEN_MIN_PT", "WIDEN_MARGIN_PT", "CUT_LINES_LISTED",
           "LayerLine", "TextLayer", "page_text_layer", "reading_turn",
           "turn_image", "turned_note", "unturned_size", "unturn_px_box",
           "turn_px_box", "turn_edge", "cut_lines", "widen_to_lines",
           "cut_note", "TURN_TAG_INSTRUCTION", "read_turn_tag"]

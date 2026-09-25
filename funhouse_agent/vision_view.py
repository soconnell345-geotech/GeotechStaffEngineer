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
* **Where things are — a 0-999 grid.** The main agent never sees an image; it
  reads what the vision call wrote. So each vision call is asked to give
  locations as a box on a 0-999 grid over the image (OpenAI's recommended
  convention, and immune to any resize), and each result returns the
  ``view`` — the page rect the image showed. ``render_region(view=...,
  image_box=...)`` turns the two back into a box on the page
  (:func:`image_box_to_page`), so the agent can zoom on what the vision call
  found.

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

import inspect
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

BUDGET_ENV = "GEOTECH_VISION_BUDGET"
DETAIL_ENV = "GEOTECH_VISION_DETAIL"
CHART_BUDGET_ENV = "GEOTECH_CHART_BUDGET"
POLICY_ENV = "GEOTECH_VISION_POLICY"

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

#: The sentence every vision prompt ends with, so a location comes back in a
#: form the agent can zoom on.
GRID_INSTRUCTION = (
    "If you give the location of anything in this image, give it as a box "
    "[x0, y0, x1, y1] on a 0-999 grid over the whole image (origin at the "
    "top-left corner, x to the right, y down). Read codes, tags and numbers "
    "character by character: where a character could be another (G/Q/O/C/D, "
    "E/F, B/8, S/5, I/1/L, Z/2), write the alternatives in brackets, e.g. "
    "[G/Q]CE, and say the lettering is too small to be sure, rather than "
    "picking one.")

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


def budget(engine=None):
    """The planlens ``ImageBudget`` renders are sized to, or ``None``."""
    name = budget_name(engine)
    if name.lower() in _OFF:
        return None
    try:
        from planlens.document.budget import resolve_budget
    except ImportError:           # planlens older than image budgets
        return None
    try:
        return resolve_budget(name)
    except ValueError as exc:
        log.warning("%s=%r ignored: %s", BUDGET_ENV, name, exc)
        return None


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
    profile picks the budget (see :func:`budget_name`).
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
        return doc.render(int(page), **kwargs)
    finally:
        doc.close()


def view_payload(info: Dict[str, Any], engine=None) -> Dict[str, Any]:
    """The fields a vision result carries so the agent can zoom on it — and,
    when the page's lettering is too small in this image, what to do instead
    of calling it unreadable."""
    out: Dict[str, Any] = {
        "view": [round(float(v), 1) for v in info["clip"]],
        "view_px": [info["width_px"], info["height_px"]],
        "zoom_hint": ZOOM_HINT}
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
    """``prompt`` with the 0-999 location instruction appended."""
    return f"{prompt.rstrip()}\n\n{GRID_INSTRUCTION}"


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
           "POLICIES", "policy", "ASSUMED_CAD_LETTERING_PT", "tile_grid",
           "tile_boxes", "DEFAULT_BUDGET",
           "DEFAULT_CHART_BUDGET", "LEGIBLE_TEXT_PX", "TARGET_TEXT_PX",
           "chart_reading", "GRID_INSTRUCTION", "ZOOM_HINT", "engine_profile",
           "budget_name", "budget", "detail", "image_media_type",
           "render_view", "view_payload", "with_grid", "image_box_to_page"]

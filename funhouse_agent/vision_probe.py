"""Which model really answers the vision calls, and what it really looks at.

A deployment name is an alias. ``tinyapp-gpt-medium`` turned out to be GPT-5.1
(a TILE model: every image cut down to 768 px on its short side) and
``funhouse-gpt-high`` is GPT-5.4 (2,500 patches at ``high``, 10,000 at
``original``) — and either can be re-pointed at another model tomorrow without
the key or the name changing. So the app does not trust a name. The first time
a vision engine is used, :func:`probe` asks the model itself, four small calls
(a fifth where the model takes images past 2,048 px, below):

1. a text-only call — the reply's ``model`` field names the model that
   answered, and its input tokens are the baseline;
2. a blank 1024 px square at ``detail="high"``;
3. a blank 2048 px square at ``detail="high"``;
4. the 2048 square again at ``detail="original"``.

The image tokens of 2-4 (input tokens less the baseline) are read by
``planlens.document.budget.budget_from_probe``: the RATIOS cancel whatever
per-token multiplier a model applies, so a tile model, a 2,500-patch model, a
6,144-patch model and whether ``original`` is really honoured (GPT-5.1 accepts
it and ignores it) all come out of the numbers, for a model no table has heard
of. Where the numbers are missing the model's name is looked up instead
(``budget_for_model``); where that fails too, the app's defaults stand.

5. When ``original`` is honoured (a budget past 2,048 px), one more image:
   a blank :data:`WIDE_PX` (3072 x 1024) at ``original``. A HOST can shrink
   an image before the model sees it — Funhouse delivers at most 2,048 px on
   the long side (2026-10-07: a 3957 x 2560 page cost exactly the tokens of
   2048 x 1326) — and the four calls above, none over 2,048 px, cannot see
   that. Taken whole the wide image costs 3,072 patches, 0.75 of the 2048
   square; shrunk to 2,048 px it costs 1,408, 0.34 of it; the ratio in
   between gives the edge (:func:`host_edge_from_probe`). The profile's
   ``max_edge`` then caps every render (``vision_view.max_px``). An edge
   BELOW 2,048 px is not seen (the square itself would be shrunk too, and
   both ratios read the same); the app's 2,048 px default covers the hosts
   measured so far.

One probe per model per process (a few thousand input tokens, about a cent on
GPT-5.4; with call 5, about 8,000 more where ``original`` is honoured, so
about three cents), cached by the model object's identity and configured
name. Turn it
off with ``GEOTECH_VISION_PROBE=0``; ``GEOTECH_VISION_BUDGET`` /
``GEOTECH_CHART_BUDGET`` still override whatever it finds.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

PROBE_ENV = "GEOTECH_VISION_PROBE"

_OFF = ("0", "false", "no", "off", "none")

#: The wide test image that shows whether the HOST shrinks large images.
WIDE_PX: Tuple[int, int] = (3072, 1024)

#: The 2048 square of the probe (planlens ``PROBE_LARGE_PX``).
_SQUARE_PX = 2048

#: Patch size of the patch models this test is for.
_PATCH = 32


def _patches(w: float, h: float) -> int:
    return math.ceil(w / _PATCH) * math.ceil(h / _PATCH)


def _ratio_at_edge(edge: int) -> float:
    """The wide image's patches over the square's, on a host that shrinks
    anything longer than ``edge`` px to ``edge``."""
    ww, wh = WIDE_PX
    s = min(1.0, edge / max(ww, wh))
    wide = _patches(round(ww * s), round(wh * s))
    q = min(_SQUARE_PX, edge)
    return wide / _patches(q, q)


def host_edge_from_probe(square_tokens: Optional[float],
                         wide_tokens: Optional[float]) -> Optional[int]:
    """The longest side, in px, the host delivers — from what the 2048 square
    and the :data:`WIDE_PX` image cost at ``original``. ``None`` when the
    wide image went through whole (no cap up to its 3,072 px) or the numbers
    are missing; 2,048 when it was shrunk to 2,048 px or less (an edge under
    2,048 is not told apart: the square is shrunk with it)."""
    if not square_tokens or not wide_tokens or square_tokens <= 0:
        return None
    r = float(wide_tokens) / float(square_tokens)
    whole = _ratio_at_edge(max(WIDE_PX))
    if r >= 0.9 * whole:
        return None
    if r <= 1.1 * _ratio_at_edge(_SQUARE_PX):
        return _SQUARE_PX
    best = min(range(_SQUARE_PX, max(WIDE_PX) + 1, 16),
               key=lambda e: abs(_ratio_at_edge(e) - r))
    return int(best)


@dataclass
class VisionProfile:
    """What was learned about one vision model."""

    #: The model the responses say answered (``gpt-5.1-2025-11-13``).
    answered_by: Optional[str] = None
    #: planlens budget names: for any image, and for a detail-bound one.
    general: Optional[str] = None
    detailed: Optional[str] = None
    #: ``probe`` (measured), ``model name`` (looked up) or ``none``.
    source: str = "none"
    ratios: Dict[str, float] = field(default_factory=dict)
    image_tokens: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    #: The longest side the HOST delivers (px), when the probe saw it shrink
    #: a larger image; ``None`` when none was seen (or not measured).
    max_edge: Optional[int] = None

    def summary(self) -> str:
        who = self.answered_by or "an unknown model"
        if not self.general:
            return f"{who}: image budget not determined ({self.error or self.source})"
        charts = ("" if self.detailed == self.general
                  else f", charts at {self.detailed}")
        host = (f"; the host delivers at most {self.max_edge} px"
                if self.max_edge else "")
        return (f"{who}: images at {self.general}{charts}{host} "
                f"(from {self.source})")

    def to_dict(self) -> Dict[str, Any]:
        return {"answered_by": self.answered_by, "general": self.general,
                "detailed": self.detailed, "source": self.source,
                "ratios": dict(self.ratios),
                "image_tokens": dict(self.image_tokens), "error": self.error,
                "max_edge": self.max_edge}


_CACHE: Dict[Any, VisionProfile] = {}
_LOCK = threading.Lock()


def enabled() -> bool:
    return os.environ.get(PROBE_ENV, "1").strip().lower() not in _OFF


def _cache_key(model) -> Any:
    name = (getattr(model, "model_name", None) or getattr(model, "model", None)
            or getattr(model, "deployment_name", None))
    base = (getattr(model, "openai_api_base", None)
            or getattr(model, "base_url", None))
    return (type(model).__name__, str(name), str(base), id(model)
            if name is None else None)


def _blank_png(side: int, height: Optional[int] = None) -> str:
    import fitz  # PyMuPDF, a planlens dependency
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, side, height or side), 0)
    pix.clear_with(255)
    return base64.b64encode(pix.tobytes("png")).decode()


def _call(model, content):
    from langchain_core.messages import HumanMessage
    msg = model.invoke([HumanMessage(content=content)])
    usage = getattr(msg, "usage_metadata", None) or {}
    meta = getattr(msg, "response_metadata", None) or {}
    answered = meta.get("model_name") or meta.get("model")
    tokens = usage.get("input_tokens")
    return (int(tokens) if tokens is not None else None), answered


def _image_content(b64: str, detail: str):
    return [{"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}",
                           "detail": detail}},
            {"type": "text", "text": "Reply with OK."}]


def probe(model) -> VisionProfile:
    """Measure ``model`` (a LangChain chat model). Never raises."""
    prof = VisionProfile()
    try:
        from planlens.document.budget import (
            PROBE_LARGE_PX, PROBE_SMALL_PX, budget_for_model, budget_from_probe)
    except ImportError:          # planlens older than model-aware budgets
        prof.error = "planlens has no budget_from_probe (needs 0.9)"
        return prof
    try:
        base, answered = _call(model, "Reply with OK.")
        prof.answered_by = answered
        small_b64 = _blank_png(PROBE_SMALL_PX)
        large_b64 = _blank_png(PROBE_LARGE_PX)
        small, a1 = _call(model, _image_content(small_b64, "high"))
        large, a2 = _call(model, _image_content(large_b64, "high"))
        prof.answered_by = prof.answered_by or a1 or a2
        try:
            orig, _ = _call(model, _image_content(large_b64, "original"))
        except Exception as exc:          # refused: original unsupported
            orig = None
            prof.ratios["original_refused"] = 1.0
            log.info("vision probe: detail=original refused: %s", exc)
        if None not in (base, small, large):
            toks = {"small_high": small - base, "large_high": large - base}
            if orig is not None:
                toks["large_original"] = orig - base
            prof.image_tokens = toks
            general, detailed, ratios = budget_from_probe(
                toks["small_high"], toks["large_high"],
                toks.get("large_original"))
            prof.general, prof.detailed = general.name, detailed.name
            prof.ratios.update(ratios)
            prof.source = "probe"
            if detailed.max_edge > PROBE_LARGE_PX and "large_original" in toks:
                _probe_host_edge(model, prof, base, toks["large_original"])
            return prof
        prof.error = "the responses carried no token counts"
    except Exception as exc:
        prof.error = f"{type(exc).__name__}: {exc}"[:300]
    found = budget_for_model(prof.answered_by)
    if found:
        prof.general, prof.detailed = found[0].name, found[1].name
        prof.source = "model name"
    return prof


def _probe_host_edge(model, prof: VisionProfile, base: int,
                     square_original: int) -> None:
    """Call 5: does the host shrink an image past 2,048 px? Fills
    ``prof.max_edge``; a failure leaves it unset and never fails the probe."""
    try:
        w, h = WIDE_PX
        wide, _ = _call(model, _image_content(_blank_png(w, h), "original"))
        if wide is None:
            return
        wide_tokens = wide - base
        prof.image_tokens["wide_original"] = wide_tokens
        if square_original and square_original > 0:
            prof.ratios["wide"] = round(wide_tokens / square_original, 3)
        prof.max_edge = host_edge_from_probe(square_original, wide_tokens)
    except Exception as exc:     # the budget stands without it
        log.info("vision probe: host edge not measured: %s", exc)


def profile_for(model) -> Optional[VisionProfile]:
    """The cached profile for ``model``, probing it the first time.

    ``None`` when probing is switched off. One probe per model per process,
    even when several conversations ask at once.
    """
    if model is None or not enabled():
        return None
    key = _cache_key(model)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        prof = probe(model)
        _CACHE[key] = prof
    log.info("vision model: %s", prof.summary())
    return prof


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def cached_profile(model) -> Optional[VisionProfile]:
    """The profile already measured for ``model`` in this process, WITHOUT
    probing — for a run record (``None`` when it was never probed)."""
    if model is None:
        return None
    with _LOCK:
        return _CACHE.get(_cache_key(model))


def cached_profiles() -> List[VisionProfile]:
    """Every profile measured in this process, without probing."""
    with _LOCK:
        return list(_CACHE.values())


__all__ = ["PROBE_ENV", "WIDE_PX", "VisionProfile", "enabled", "probe",
           "profile_for", "clear_cache", "host_edge_from_probe",
           "cached_profile", "cached_profiles"]

"""Which model really answers the vision calls, and what it really looks at.

A deployment name is an alias. ``tinyapp-gpt-medium`` turned out to be GPT-5.1
(a TILE model: every image cut down to 768 px on its short side) and
``funhouse-gpt-high`` is GPT-5.4 (2,500 patches at ``high``, 10,000 at
``original``) — and either can be re-pointed at another model tomorrow without
the key or the name changing. So the app does not trust a name. The first time
a vision engine is used, :func:`probe` asks the model itself, four small calls:

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

One probe per model per process (a few thousand input tokens, about a cent on
GPT-5.4), cached by the model object's identity and configured name. Turn it
off with ``GEOTECH_VISION_PROBE=0``; ``GEOTECH_VISION_BUDGET`` /
``GEOTECH_CHART_BUDGET`` still override whatever it finds.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

PROBE_ENV = "GEOTECH_VISION_PROBE"

_OFF = ("0", "false", "no", "off", "none")


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

    def summary(self) -> str:
        who = self.answered_by or "an unknown model"
        if not self.general:
            return f"{who}: image budget not determined ({self.error or self.source})"
        charts = ("" if self.detailed == self.general
                  else f", charts at {self.detailed}")
        return f"{who}: images at {self.general}{charts} (from {self.source})"

    def to_dict(self) -> Dict[str, Any]:
        return {"answered_by": self.answered_by, "general": self.general,
                "detailed": self.detailed, "source": self.source,
                "ratios": dict(self.ratios),
                "image_tokens": dict(self.image_tokens), "error": self.error}


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


def _blank_png(side: int) -> str:
    import fitz  # PyMuPDF, a planlens dependency
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, side, side), 0)
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
            return prof
        prof.error = "the responses carried no token counts"
    except Exception as exc:
        prof.error = f"{type(exc).__name__}: {exc}"[:300]
    found = budget_for_model(prof.answered_by)
    if found:
        prof.general, prof.detailed = found[0].name, found[1].name
        prof.source = "model name"
    return prof


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


__all__ = ["PROBE_ENV", "VisionProfile", "enabled", "probe", "profile_for",
           "clear_cache"]

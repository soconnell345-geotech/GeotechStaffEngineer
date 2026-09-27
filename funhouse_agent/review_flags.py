"""Switches for the Document Review harness changes of 2026-09-26.

The review of the Document Review page (plan: "Document Review harness —
big-picture review", 2026-09-26) found that every recent change had been judged
on the one example that prompted it. So each behaviour change from that review
ships behind a switch here, OFF by default, and the review suite
(:mod:`funhouse_agent.review_eval`) runs the same tasks with the switches off
and on from ONE install. A switch is turned on by default only once the suite
says it helps.

Every switch is read from the environment AT CALL TIME, so a notebook (or the
suite) can flip one between two runs without rebuilding anything but the agent.

``GEOTECH_REVIEW_AGENT``
    ``legacy`` (default) builds the review page's agent the way it has been
    built since 5.26: the geotechnical builder with the modules subtracted.
    ``lean`` builds it on its own (:mod:`funhouse_agent.deep.review_agent`):
    only the reading, looking and writing tools, no coding-agent prompt or
    scratch-filesystem tools, a reading sub-agent that carries the review
    rules, and a model-call budget that ends a long turn with an answer.
``GEOTECH_VISION_TEXT_CONTEXT``
    ``1`` gives every page/region vision call the PDF's own text-layer lines
    inside the view it is looking at (exact strings, with positions), or says
    there are none.
``GEOTECH_VISION_STRUCTURED``
    ``1`` asks every page/region vision call to end with a machine-readable
    list of what it located, and returns those locations as boxes in PDF
    points the agent can pass straight back to ``render_region``.
``GEOTECH_VISION_INLINE``
    ``1`` lets the reasoning model look: the page/region tools stop making the
    separate one-shot vision call and the newest images are shown to the main
    model itself at its next call (lean agent only).
``GEOTECH_REVIEW_SWEEP``
    ``1`` offers the ``sweep_pages`` tool (lean agent only): one question asked
    of every page in a range, in parallel, answered per page with citations.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, Iterator, Mapping, Optional

AGENT_ENV = "GEOTECH_REVIEW_AGENT"
VISION_TEXT_ENV = "GEOTECH_VISION_TEXT_CONTEXT"
VISION_STRUCTURED_ENV = "GEOTECH_VISION_STRUCTURED"
VISION_INLINE_ENV = "GEOTECH_VISION_INLINE"
SWEEP_ENV = "GEOTECH_REVIEW_SWEEP"

ALL_ENVS = (AGENT_ENV, VISION_TEXT_ENV, VISION_STRUCTURED_ENV,
            VISION_INLINE_ENV, SWEEP_ENV)

_ON = ("1", "true", "yes", "on")


def _on(env: str) -> bool:
    return str(os.environ.get(env, "")).strip().lower() in _ON


def lean_agent() -> bool:
    """Whether the review page builds its own lean agent."""
    return str(os.environ.get(AGENT_ENV, "")).strip().lower() == "lean"


def vision_text_context() -> bool:
    """Whether vision calls are given the text layer inside their view."""
    return _on(VISION_TEXT_ENV)


def vision_structured() -> bool:
    """Whether vision calls return located findings as page-point boxes."""
    return _on(VISION_STRUCTURED_ENV)


def vision_inline() -> bool:
    """Whether the main model is shown the images itself (lean agent only)."""
    return _on(VISION_INLINE_ENV)


def sweep() -> bool:
    """Whether the ``sweep_pages`` tool is offered (lean agent only)."""
    return _on(SWEEP_ENV)


#: Named configurations the review suite compares. ``baseline`` is exactly
#: what the page does with no switch set. Each later arm adds one change, so
#: a gain or a loss is attributable to the change that was added.
ARMS: Dict[str, Dict[str, str]] = {
    "baseline": {},
    "lean": {AGENT_ENV: "lean"},
    "grounded": {AGENT_ENV: "lean", VISION_TEXT_ENV: "1",
                 VISION_STRUCTURED_ENV: "1"},
    "inline": {AGENT_ENV: "lean", VISION_TEXT_ENV: "1",
               VISION_STRUCTURED_ENV: "1", VISION_INLINE_ENV: "1"},
    "sweep": {AGENT_ENV: "lean", VISION_TEXT_ENV: "1",
              VISION_STRUCTURED_ENV: "1", SWEEP_ENV: "1"},
}


@contextmanager
def switches(values: Optional[Mapping[str, str]]) -> Iterator[None]:
    """Set exactly ``values`` for the block: every switch NOT named is unset,
    so an arm is what it says and nothing leaks in from the notebook's own
    environment. Any other variable the arm names (a budget, a vision
    policy) is set too. The previous environment is restored afterwards."""
    values = dict(values or {})
    keys = set(ALL_ENVS) | set(values)
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            if k in values and values[k] not in (None, ""):
                os.environ[k] = str(values[k])
            else:
                os.environ.pop(k, None)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def describe() -> str:
    """One line naming the switches that are on (for diagnostics and logs)."""
    on = [f"{k}={os.environ[k]}" for k in ALL_ENVS
          if str(os.environ.get(k, "")).strip()]
    return ", ".join(on) if on else "none (legacy review page)"


__all__ = ["AGENT_ENV", "VISION_TEXT_ENV", "VISION_STRUCTURED_ENV",
           "VISION_INLINE_ENV", "SWEEP_ENV", "ALL_ENVS", "ARMS",
           "lean_agent", "vision_text_context", "vision_structured",
           "vision_inline", "sweep", "switches", "describe"]

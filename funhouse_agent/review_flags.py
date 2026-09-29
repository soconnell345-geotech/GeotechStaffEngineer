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
    model itself at its next call (lean agent only). ``analyze_image`` keeps
    its one-shot vision call, exactly as released in 5.30, unless
    ``GEOTECH_REVIEW_OVERVIEW`` is on too (see there).
``GEOTECH_REVIEW_SWEEP``
    ``1`` offers the ``sweep_pages`` tool (lean agent only): one question asked
    of every page in a range, in parallel, answered per page with citations.
``GEOTECH_REVIEW_FINDINGS``
    ``1`` offers the findings tools (lean agent only): ``record_finding``,
    ``list_findings`` and ``findings_report``. Each finding is kept in
    ``findings.json`` in the conversation's working folder in the shared
    format of :mod:`funhouse_agent.review_findings`, its quotes checked
    against the pages it cites, and the Word comment log and marked-up PDFs
    are rendered from it (plan S1.5, ``module_work/REVIEW_ARCHITECTURE.md``).
``GEOTECH_REVIEW_OVERVIEW``
    ``1`` makes the orientation turn after an upload ask for the contact
    sheets explicitly when the new uploads run past
    ``GEOTECH_REVIEW_OVERVIEW_PAGES`` pages in all (default 20), and to skip
    them at or below it (plan S1.2). Only with BOTH this and
    ``GEOTECH_VISION_INLINE`` on is the lean agent's ``analyze_image`` of an
    image FILE (a contact sheet) shown to the model itself, like the inline
    page tools; either one alone leaves ``analyze_image`` as released.
``GEOTECH_REVIEW_GEOMETRY``
    ``1`` offers the drawing-geometry tools (lean agent only):
    ``drawing_callouts``, ``drawing_dimensions``, ``title_block`` and
    ``revision_clouds`` (:mod:`funhouse_agent.deep.geometry_tools`). They
    find leaders, dimension lines, the title block and revision clouds from
    a sheet's line-work and return boxes to zoom on - "geometry says where,
    vision says what" (plan S1.1).
``GEOTECH_REVIEW_DIGEST``
    ``1`` offers the digest tools (lean agent only; its reading helper gets
    them too): ``document_inventory``, ``digest_search``, ``digest_pages``
    and ``digest_references`` (:mod:`funhouse_agent.deep.digest_tools`) over
    the FREE layer of each upload's digest (:mod:`funhouse_agent.
    review_digest`: page map, text index, references, built by code in
    seconds and kept in the working folder) - plan shape 2, Stage A. Its
    small/large hint uses ``GEOTECH_REVIEW_SHAPE1_PAGES`` (default 20).
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
FINDINGS_ENV = "GEOTECH_REVIEW_FINDINGS"
OVERVIEW_ENV = "GEOTECH_REVIEW_OVERVIEW"
GEOMETRY_ENV = "GEOTECH_REVIEW_GEOMETRY"
DIGEST_ENV = "GEOTECH_REVIEW_DIGEST"

ALL_ENVS = (AGENT_ENV, VISION_TEXT_ENV, VISION_STRUCTURED_ENV,
            VISION_INLINE_ENV, SWEEP_ENV, FINDINGS_ENV, OVERVIEW_ENV,
            GEOMETRY_ENV, DIGEST_ENV)

#: Not a switch but the overview's setting: above this many pages in the
#: new uploads (all of them together) the orientation asks for contact sheets.
OVERVIEW_PAGES_ENV = "GEOTECH_REVIEW_OVERVIEW_PAGES"
DEFAULT_OVERVIEW_PAGES = 20

#: The switches' SETTINGS (numbers the switched behaviour reads, not
#: switches): the overview's page threshold, the digest's small/large hint,
#: how many inline images each call carries and the lean agent's model-call
#: budget. :func:`switches` clears them too unless the arm names them, so an
#: arm is exactly what it says. Named here as strings so this module imports
#: nothing.
SETTINGS_ENVS = (OVERVIEW_PAGES_ENV, "GEOTECH_REVIEW_SHAPE1_PAGES",
                 "GEOTECH_VISION_INLINE_KEEP",
                 "GEOTECH_REVIEW_MAX_MODEL_CALLS")

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


def findings() -> bool:
    """Whether the findings tools are offered (lean agent only)."""
    return _on(FINDINGS_ENV)


def overview() -> bool:
    """Whether the orientation turn decides on contact sheets by page count."""
    return _on(OVERVIEW_ENV)


def geometry() -> bool:
    """Whether the drawing-geometry tools are offered (lean agent only)."""
    return _on(GEOMETRY_ENV)


def digest() -> bool:
    """Whether the digest tools are offered (lean agent only)."""
    return _on(DIGEST_ENV)


def overview_pages() -> int:
    """The page count above which the orientation asks for contact sheets
    (``GEOTECH_REVIEW_OVERVIEW_PAGES``, default 20)."""
    try:
        return max(1, int(os.environ.get(OVERVIEW_PAGES_ENV,
                                         DEFAULT_OVERVIEW_PAGES)))
    except (TypeError, ValueError):
        return DEFAULT_OVERVIEW_PAGES


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
    # Only the orientation turn differs: run it with orientation=True.
    "overview": {AGENT_ENV: "lean", VISION_TEXT_ENV: "1",
                 VISION_STRUCTURED_ENV: "1", OVERVIEW_ENV: "1"},
    "geometry": {AGENT_ENV: "lean", VISION_TEXT_ENV: "1",
                 VISION_STRUCTURED_ENV: "1", GEOMETRY_ENV: "1"},
    "digest": {AGENT_ENV: "lean", VISION_TEXT_ENV: "1",
               VISION_STRUCTURED_ENV: "1", DIGEST_ENV: "1"},
}


@contextmanager
def switches(values: Optional[Mapping[str, str]]) -> Iterator[None]:
    """Set exactly ``values`` for the block: every switch (:data:`ALL_ENVS`)
    and every switch setting (:data:`SETTINGS_ENVS`) NOT named is unset, so
    an arm is what it says and nothing leaks in from the notebook's own
    environment. Any other variable the arm names (a budget, a vision
    policy) is set too. The previous environment is restored afterwards."""
    values = dict(values or {})
    keys = set(ALL_ENVS) | set(SETTINGS_ENVS) | set(values)
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
           "VISION_INLINE_ENV", "SWEEP_ENV", "FINDINGS_ENV", "OVERVIEW_ENV",
           "GEOMETRY_ENV", "DIGEST_ENV", "OVERVIEW_PAGES_ENV",
           "DEFAULT_OVERVIEW_PAGES", "ALL_ENVS", "SETTINGS_ENVS", "ARMS",
           "lean_agent",
           "vision_text_context", "vision_structured", "vision_inline",
           "sweep", "findings", "overview", "geometry", "digest",
           "overview_pages", "switches", "describe"]

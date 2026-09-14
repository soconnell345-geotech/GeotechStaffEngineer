"""Whole-document review tools for the agent, served by ``planlens.tools``.

The deep agent's primary tool surface gets five tools — ``open_document``,
``document_page_map``, ``read_document``, ``search_document`` and
``document_markups`` — that read a whole PDF as located, attributed data: a
page map, text with boxes, tables, the review markups and the hidden text CAD
programs leave behind. planlens owns the behaviour and the size discipline
(every result valid JSON inside the budget, longer results paged through
cursors). This module only connects it to the app:

- a ``source`` resolves against the conversation's attachments first, then
  real file paths — the same order as ``read_pdf_text``;
- one process-wide toolkit, so a document opened in one turn is still open,
  by handle, in the next;
- attachments reach the resolver through a context variable set for the
  duration of each call, so concurrent conversations resolve only their own
  uploads;
- the "how to look" instruction planlens appends to every ``! look:`` line
  names THIS app's vision tools (``analyze_pdf_page``, ``render_region``),
  which take the same ``source`` and the same displayed-frame boxes. The
  toolkit's own ``render_page`` / ``render_region`` are not exposed here —
  the app already routes images to the model through its vision engine.
"""

from __future__ import annotations

import json
import os
import threading
from contextvars import ContextVar
from typing import Any, Dict, Optional

DOCUMENT_TOOL_NAMES = (
    "open_document",
    "document_page_map",
    "read_document",
    "search_document",
    "document_markups",
)

#: Appended by planlens to every "! look:" line. It must name the app's own
#: vision tools and their argument conventions, because that line is the
#: model's cue to switch from reading to looking.
VISION_HINT = (
    "to look: analyze_pdf_page(attachment_key=<the source you opened>, "
    "page=N, prompt=<what to find>) views the whole page; "
    "render_region(attachment_key=<source>, page=N, bbox=[x0,y0,x1,y1], "
    "prompt=...) zooms on a box from read_document or a markup (same frame, "
    "no conversion); marks=[[x,y,label],...] numbers spots to ask about"
)

#: Budget used when the host has disabled truncation: generous, still a bound.
UNCAPPED_BUDGET = 60000

#: Room left under the host's cap, so planlens' own limit is what binds and the
#: host's string truncation (which breaks JSON) never fires.
_CAP_MARGIN = 200

_ATTACHMENTS: ContextVar[Optional[Dict[str, bytes]]] = ContextVar(
    "gse_document_attachments", default=None)
_KIT = None
_KIT_LOCK = threading.Lock()


def available() -> bool:
    """Whether the installed planlens has the tool layer."""
    try:
        import planlens.tools  # noqa: F401
    except ImportError:
        return False
    return True


def budget_for_cap(cap: int) -> int:
    """planlens' size limit for a host result cap (``<= 0`` = uncapped)."""
    if cap <= 0:
        return UNCAPPED_BUDGET
    return max(1000, int(cap) - _CAP_MARGIN)


def tool_description(name: str) -> str:
    """The model-facing description planlens publishes for ``name``."""
    from planlens.tools.specs import TOOL_SPECS
    for spec in TOOL_SPECS:
        if spec["name"] == name:
            return spec["description"]
    raise KeyError(name)


def _resolve(source: str):
    from planlens.tools import ToolError
    attachments = _ATTACHMENTS.get() or {}
    if source in attachments:
        return attachments[source]
    if source and os.path.isfile(source):
        return source
    raise ToolError(
        f"'{source}' is not an attachment key or a readable file path",
        hint=(f"attachment keys: {sorted(attachments) or 'none'}; real paths "
              f"(/tmp/..., /Volumes/...) also work"))


def _toolkit():
    global _KIT
    with _KIT_LOCK:
        if _KIT is None:
            from planlens.tools import ReviewToolkit
            _KIT = ReviewToolkit(resolve_source=_resolve,
                                 vision_hint=VISION_HINT)
        return _KIT


def dispatch_document_tool(name: str, arguments: Dict[str, Any],
                           attachments: Optional[Dict[str, bytes]] = None,
                           max_chars: Optional[int] = None) -> str:
    """Run one document tool; returns a JSON string within ``max_chars``."""
    if not available():
        return json.dumps({
            "error": "document tools need planlens with planlens.tools "
                     "(0.3 or later)",
            "hint": "pip install -U planlens"})
    # A shallow snapshot: the host mutates its attachments dict between turns.
    token = _ATTACHMENTS.set(dict(attachments or {}))
    try:
        return _toolkit().call_json(name, arguments, max_chars=max_chars)
    finally:
        _ATTACHMENTS.reset(token)

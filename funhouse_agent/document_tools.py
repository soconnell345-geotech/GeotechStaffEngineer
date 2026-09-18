"""Whole-document review tools for the agent, served by ``planlens.tools``.

The deep agent's primary tool surface gets ``open_document``,
``document_structure``, ``document_page_map``, ``read_document``,
``search_document``, ``document_markups`` and ``render_page_thumbnails`` —
tools that read a whole PDF as located, attributed data: a page map, text with
boxes, tables, the review markups and the hidden text CAD programs leave
behind. planlens owns the behaviour and the size discipline (every result valid
JSON inside the budget, longer results paged through cursors). This module only
connects it to the app:

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

Tools and parameters planlens gained after the version this app pins are
FEATURE-DETECTED from the installed package's own specs
(:func:`has_tool`, :func:`search_supports_fuzzy`) and stay off the surface
otherwise — the cluster installs planlens from PyPI, which can be older than
the development checkout.
"""

from __future__ import annotations

import json
import os
import threading
from contextvars import ContextVar
from typing import Any, Dict, Optional

#: The tools every planlens with the tool layer serves (0.3 and later). Kept as
#: a module constant because callers import it; ``document_tool_names()`` is
#: what the INSTALLED planlens actually offers.
DOCUMENT_TOOL_NAMES = (
    "open_document",
    "document_structure",
    "document_page_map",
    "read_document",
    "search_document",
    "document_markups",
    "render_page_thumbnails",
)

#: Served only by a newer planlens. Each one is advertised to the model only
#: when the installed package publishes a spec for it.
OPTIONAL_DOCUMENT_TOOL_NAMES = (
    "find_quantities",
)

#: How the model views a PNG planlens wrote (the thumbnail contact sheets):
#: the app's analyze_image accepts a real path as its attachment_key.
IMAGE_VIEW_HINT = ("view the image with analyze_image(attachment_key="
                   "<image_path>, prompt=<what to look for>)")

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
    spec = _spec(name)
    if spec is None:
        raise KeyError(name)
    return spec["description"]


def _spec(name: str) -> Optional[Dict[str, Any]]:
    """The installed planlens' spec for ``name``, or ``None``."""
    try:
        from planlens.tools.specs import TOOL_SPECS
    except ImportError:
        return None
    for spec in TOOL_SPECS:
        if spec.get("name") == name:
            return spec
    return None


def has_tool(name: str) -> bool:
    """Whether the installed planlens publishes a tool called ``name``."""
    return _spec(name) is not None


def document_tool_names() -> tuple:
    """The document tools to put on the agent's surface.

    The fixed seven, plus every optional tool the installed planlens actually
    publishes. Computed per call, not at import, so the set follows the package
    that is installed rather than the one this app was written against.
    """
    return DOCUMENT_TOOL_NAMES + tuple(
        name for name in OPTIONAL_DOCUMENT_TOOL_NAMES if has_tool(name))


def search_supports_fuzzy() -> bool:
    """Whether the installed ``search_document`` takes ``fuzzy``."""
    spec = _spec("search_document")
    if spec is None:
        return False
    schema = spec.get("parameters") or spec.get("input_schema") or {}
    return "fuzzy" in (schema.get("properties") or {})


def _resolve(source: str):
    from planlens.tools import ToolError
    attachments = _ATTACHMENTS.get() or {}
    if source in attachments:
        return attachments[source]
    if source and os.path.isfile(source):
        return source
    from funhouse_agent._fileio import find_in_working_folder
    found = find_in_working_folder(source)
    if found:
        return found
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
                                 vision_hint=VISION_HINT,
                                 image_view_hint=IMAGE_VIEW_HINT)
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


#: The planlens tools the report ingest is built on: the page roles and work
#: items it walks, and the log grid its log reader reads rows off. Both
#: arrived in planlens 0.5; on anything older the ingest cannot run at all,
#: so the app's tool hides itself rather than failing in front of the user.
REPORT_INGEST_TOOLS = ("document_roles", "log_grid")


def report_ingest_supported() -> bool:
    """Whether the installed planlens can serve the whole-report ingest."""
    return available() and all(has_tool(name) for name in REPORT_INGEST_TOOLS)


def resolve_document_source(source: str,
                            attachments: Optional[Dict[str, bytes]] = None):
    """An attachment key or a path, resolved the way the document tools do.

    Returns the attachment's BYTES or a real path, which is what planlens'
    ``open_document`` takes either of. Raises planlens' ``ToolError`` when it
    is neither, with the same wording the document tools give.
    """
    token = _ATTACHMENTS.set(dict(attachments or {}))
    try:
        return _resolve(source)
    finally:
        _ATTACHMENTS.reset(token)

"""The document and drawing tools kept off the geotech page's tool list
until the conversation holds a document (BACKLOG INF-67).

The build i trace review (§2.2) found the document and drawing tools are
about a third of every model call's prefix on the geotech page, on turns
with no document at all. Measured offline (``module_work/inf67/CHECK.md``,
tiktoken ``o200k_base``): 18 tools, 7,078 of the 22,035-token prefix.

:class:`DocumentToolsGate` hides :data:`DOCUMENT_TOOLS` from the model's tool
list on every model call where the conversation holds no document, and shows
them from the first model call after one arrives -- in the same turn when it
arrives mid-turn, because the check runs again on each call. The tools stay
REGISTERED (as :class:`~funhouse_agent.deep.limits.HideTools` does for
``delete`` and ``execute``): a call to one by name still runs.

``read_pdf_text`` and ``analyze_pdf_page`` are NOT hidden: the prompt tells
the agent to read back and look at a PDF it built itself before handing it
over, which happens on turns with no document of the user's.

A document is held when any of these is true (checked on each model call):

* the conversation's attachments are not empty (an upload, a pasted
  screenshot, a restored conversation's uploads);
* a fetch tool (:data:`FETCH_TOOLS`) has returned in the conversation's
  messages (a SharePoint download, this turn or an earlier one);
* the host's ``signal`` says so (the web app: the conversation's downloads
  ledger is not empty, or the working folder is one the user chose);
* no working folder is bound (library or notebook use, where a document may
  be any path): the gate hides nothing there.

The Document Review page is built elsewhere and never carries this gate.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional

from funhouse_agent.deep.limits import HideTools

#: The tools kept off the list until a document is held: the planlens
#: document tools, the app's look tools that need a document or image, and
#: the DXF markup tool.
DOCUMENT_TOOLS = frozenset({
    "open_document", "read_document", "search_document",
    "document_structure", "document_page_map", "document_markups",
    "render_page_thumbnails", "render_region", "analyze_image",
    "find_like", "find_quantities", "log_grid", "measure",
    "kept_readings", "annotate_document", "annotate_dxf",
})

#: Tools whose result puts a document into the conversation.
FETCH_TOOLS = frozenset({"sharepoint_download_file"})


def _fetched(messages) -> bool:
    for m in messages or ():
        if getattr(m, "type", None) == "tool" and \
                getattr(m, "name", None) in FETCH_TOOLS and \
                getattr(m, "status", "success") != "error":
            return True
    return False


class DocumentToolsGate(HideTools):
    """Hide :data:`DOCUMENT_TOOLS` while the conversation holds no document;
    show them from the next model call once it does."""

    def __init__(self, attachments: Optional[Mapping] = None,
                 signal: Optional[Callable[[], bool]] = None,
                 names=DOCUMENT_TOOLS):
        super().__init__(names)
        self._attachments = attachments
        self._signal = signal

    def holds_document(self, messages=()) -> bool:
        from funhouse_agent.vision_tools import _host_folder
        if not _host_folder():
            return True
        if self._attachments:
            return True
        if _fetched(messages):
            return True
        if self._signal is not None:
            try:
                return bool(self._signal())
            except Exception:                            # noqa: BLE001
                return True       # unsure: offer the tools, as before
        return False

    def _gated(self, request):
        if self.holds_document(getattr(request, "messages", None) or ()):
            return request
        return self._filtered(request)

    def wrap_model_call(self, request, handler):
        return handler(self._gated(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._gated(request))

    # A tool's result and an unknown-tool error are left as they are: the
    # tools are only hidden for now, and a call to one still runs.
    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)


__all__ = ["DOCUMENT_TOOLS", "FETCH_TOOLS", "DocumentToolsGate"]

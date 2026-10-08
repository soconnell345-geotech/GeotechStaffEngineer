"""The Document Review page's own agent (``GEOTECH_REVIEW_AGENT=lean``).

WHY. The review of 2026-09-26 measured what the page's agent carried when it
was built by SUBTRACTING from the geotechnical builder: 29 tools, of whose
~38K characters of schema ~22K were deepagents' generic coding tools (``task``,
``write_todos``, a shell ``execute`` that cannot work here, a scratch
filesystem the prompt then had to warn against); two geotech chart tools that
cannot work on this page; tool descriptions pointing at geometry tools the page
does not have; ~7.7K characters of generic coding-agent prompt ("read relevant
files, check existing patterns", "mimic existing style"); a general-purpose
helper with a 286-character prompt and none of the review rules; and a turn
that ended in a GraphRecursionError after about 17 model calls.

WHAT. This builds the page's agent by CHOOSING instead:

* the reading, looking and writing tools only (:data:`REVIEW_TOOLS`), with
  descriptions written for this page (:data:`REVIEW_DESCRIPTIONS`);
* the review prompt without the scratch filesystem, and nothing else in front
  of it but the host's own extras (SharePoint, feedback);
* ``write_todos`` with a short description, a summarizer for long
  conversations, and dangling-tool-call repair for replayed histories;
* ``task`` → a ``page_reader`` helper that carries the SAME reading rules and
  only read-only tools;
* a model-call budget whose last call answers from what was gathered, so a
  long turn ends with findings and a list of what was not checked — never an
  error. The step cap the host passes is raised to fit it.

With ``GEOTECH_VISION_INLINE`` the main model is shown page images itself
(:mod:`funhouse_agent.deep.inline_images`), and - only with
``GEOTECH_REVIEW_OVERVIEW`` on too - image files such as the contact sheets
(so the released ``inline`` arm is exactly what it was); with
``GEOTECH_REVIEW_SWEEP`` it gets ``sweep_pages``
(:mod:`funhouse_agent.deep.sweep`); with ``GEOTECH_REVIEW_FINDINGS`` it gets
``record_finding`` / ``list_findings`` / ``findings_report``
(:mod:`funhouse_agent.deep.findings_tools`); with ``GEOTECH_REVIEW_GEOMETRY``
it and its reading helper get ``drawing_callouts`` / ``drawing_dimensions`` /
``title_block`` / ``revision_clouds``
(:mod:`funhouse_agent.deep.geometry_tools`); with ``GEOTECH_REVIEW_DIGEST``
it and its reading helper get ``document_inventory`` / ``digest_search`` /
``digest_pages`` / ``digest_references``
(:mod:`funhouse_agent.deep.digest_tools`).
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, List, Optional

from funhouse_agent import review_flags
from funhouse_agent.deep.limits import ModelCallBudgetMiddleware
from funhouse_agent.deep.prompt import (
    DOCUMENT_REVIEW_READER_PROMPT,
    build_document_review_prompt,
)
from funhouse_agent.deep.tools import (
    DEFAULT_MAX_RESULT_CHARS,
    DEFAULT_VISION_RESULT_CHARS,
    _find_like_available,
    make_vision_tools,
)

#: The page agent's tools, all built by ``make_vision_tools``.
REVIEW_TOOLS = (
    "open_document", "document_structure", "document_page_map",
    "read_document", "search_document", "document_markups",
    "render_page_thumbnails", "find_quantities", "annotate_document",
    "analyze_pdf_page", "render_region", "analyze_image", "find_like",
    "write_docx", "save_file", "list_files", "read_text_file",
    # Visual scales: read-only, so the reading helper gets them too. Owner,
    # 2026-10-08: on the legacy and lean builds, NOT the minimal one.
    "measure", "log_grid",
)

#: The reading helper's tools: nothing that writes a file or a markup.
READER_TOOLS = tuple(t for t in REVIEW_TOOLS
                     if t not in ("annotate_document", "write_docx",
                                  "save_file"))

#: Descriptions written for this page (the geotech page's versions point at
#: drawing_ir tools this page does not have, and the core "look" tool had one
#: line).
REVIEW_DESCRIPTIONS: Dict[str, str] = {
    "analyze_pdf_page": (
        "Look at one whole page. It is rendered at the largest size the "
        "vision model reads, and a vision call answers YOUR prompt about it "
        "- so say exactly what to read, find or check ('read the general "
        "notes', 'list every callout with what it points at', 'what does the "
        "legend say this symbol means'); a vague prompt gets a vague "
        "description. Use it on EVERY page a question about what pages show "
        "covers - drawing sheets, scans, figures and forms - not only when a "
        "text result looks wrong; a text search never decides which pages "
        "are worth a look. Locations come "
        "back as 0-999 boxes on the image with the page's view (and, when "
        "given, 'located' items with a page_bbox in PDF points); zoom on one "
        "with render_region - a whole-page box says where to zoom, not where "
        "to put a mark. When the page's small lettering is too small "
        "for one image, the page is also read in tiles (tiles='off' skips "
        "that; N or 'NxN' with N from 2 to 4, e.g. '3x3', forces it). page "
        "is 0-based."),
    "render_region": (
        "Zoom on part of a page and look at it. The region is re-drawn from "
        "the PDF to fill the image, so a smaller box shows finer lettering - "
        "zoom before you call anything unreadable. Say where with bbox = "
        "[x0, y0, x1, y1] in PDF points (top-left origin, y down: the boxes "
        "read_document(with_locations=true), search_document, "
        "document_markups and a vision result's located items give), or with "
        "an earlier vision result's view + a 0-999 image_box from its "
        "answer (the window is padded by that view's location error, so the "
        "thing is in it). marks = [[x, y, label], ...] numbers spots so you can ask "
        "what is at each. Use it to read small lettering, confirm a "
        "character, see what a note, markup or leader points at, or check a "
        "dimension before you quote it. page is 0-based."),
    "analyze_image": (
        "Look at an image file - an uploaded picture, or a contact sheet "
        "that render_page_thumbnails wrote (pass its path) - and answer your "
        "prompt about it."),
    "save_file": (
        "Save text or data (CSV, JSON, Markdown, HTML) to a file. A bare "
        "name lands in the working folder and appears as a download card; "
        "the result is the proof the file exists. content is text, or base64 "
        "with encoding='base64'."),
    "write_docx": (
        "Write a Word (.docx) document from Markdown - a review memo, a "
        "comment log, a compliance matrix, a summary to pass on. Headings, "
        "bold/italic/code, bullet and numbered lists, pipe tables (an italic "
        "line after a table is its caption), block quotes, and "
        "![alt](figure.png) for a figure already in the working folder; --- "
        "is a page break. A figure that is not found is reported in warnings "
        "and the document is still written."),
}

#: With GEOTECH_VISION_INLINE the page and region tools show YOU the image
#: (no separate vision call answers a prompt), so they are described that way.
#: ``analyze_image``'s entry applies only when image FILES are shown too
#: (GEOTECH_VISION_INLINE and GEOTECH_REVIEW_OVERVIEW both on).
INLINE_DESCRIPTIONS: Dict[str, str] = {
    "analyze_pdf_page": (
        "Look at one whole page yourself: it is rendered at the largest size "
        "the model reads and shown to you with your next step. Use it on "
        "EVERY page a question about what pages show covers - drawing "
        "sheets, scans, figures and forms - not only when a text result "
        "looks wrong; where the lettering is too small, zoom "
        "with render_region. prompt is only a note of what you are after. "
        "page is 0-based."),
    "render_region": (
        "Zoom on part of a page and look at it yourself: the region is "
        "re-drawn from the PDF to fill the image and shown to you with your "
        "next step, so a smaller box shows finer lettering - zoom before you "
        "call anything unreadable. Say where with bbox = [x0, y0, x1, y1] in "
        "PDF points (top-left origin, y down: the boxes "
        "read_document(with_locations=true), search_document and "
        "document_markups give), or with an earlier view + a 0-999 image_box "
        "on its image. marks = [[x, y, label], ...] numbers spots on the "
        "image. page is 0-based."),
    "analyze_image": (
        "Look at an image. An image FILE - a contact sheet that "
        "render_page_thumbnails wrote (pass its path) - is shown to you "
        "yourself with your next step, and prompt is only a note of what "
        "you are after; an uploaded picture (pass its name) is described by "
        "a vision call that answers your prompt."),
}

#: Default model calls in one request before the last one must answer.
DEFAULT_MAX_MODEL_CALLS = 40
MAX_MODEL_CALLS_ENV = "GEOTECH_REVIEW_MAX_MODEL_CALLS"
#: The reading helper's budget per job.
READER_MAX_MODEL_CALLS = 14

FINAL_NUDGE = (
    "[Step budget reached] This is your last step for this request and no "
    "tools are available. Answer now from what you have already gathered: "
    "give the findings you have, each with its citation; say plainly which "
    "parts you did not get to check; and offer to continue in the next "
    "message.")
EXHAUSTED = (
    "I reached this request's step budget before I could write up the "
    "findings. Ask me to continue and I will pick up from the pages already "
    "read.")
READER_NUDGE = (
    "[Step budget reached] Last step, no tools: return the findings you have, "
    "each with its citation, and one line on what you did not get to check.")
READER_EXHAUSTED = (
    "The reading helper ran out of steps before it could report. Hand it a "
    "narrower job (fewer pages, one question).")

TODO_PROMPT = (
    "## `write_todos`\n\n"
    "For a job with several parts - a full-set review, a specification-versus-"
    "submittal check, a comment-response round - list the parts with "
    "`write_todos` first and mark each done as you finish it; skip it for a "
    "single question. Give your final answer in a message after your last "
    "`write_todos` call.")
TODO_TOOL_DESCRIPTION = (
    "Write or update your to-do list for a multi-part job: send the whole "
    "list each time, each item with its content and a status (pending, "
    "in_progress or completed).")

TASK_DESCRIPTION = (
    "Hand a self-contained reading job to a helper that has your reading and "
    "looking tools and your reading rules, and get back a compact list of "
    "findings with citations - e.g. 'open <source>, read sheets 12-20 and "
    "list every note that sets a concrete strength, with the sheet', or one "
    "section of a long specification. The helper sees NOTHING of this "
    "conversation: put everything it needs in description (the source name "
    "or handle, the pages, what to find, what to return). It cannot write "
    "files or markups. Several can run at once. subagent_type: "
    "'page_reader'.")


def max_model_calls() -> int:
    """The request budget (``GEOTECH_REVIEW_MAX_MODEL_CALLS``, default 40)."""
    try:
        return max(4, int(os.environ.get(MAX_MODEL_CALLS_ENV,
                                         DEFAULT_MAX_MODEL_CALLS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_MODEL_CALLS


def _patch_tool_calls():
    try:
        from deepagents.middleware.patch_tool_calls import (
            PatchToolCallsMiddleware)
        return PatchToolCallsMiddleware()
    except Exception:  # noqa: BLE001 - a repair, not the agent
        return None


def _summarizer(model):
    """A summarizer for long conversations (the one deepagents added for the
    legacy agent), at the same trigger the rest of the app uses."""
    try:
        from langchain.agents.middleware import SummarizationMiddleware
        from funhouse_agent.deep.agent import (
            _AUTO_SUMMARIZATION_TRIGGER, _resolve_summarization_trigger)
        return SummarizationMiddleware(
            model, trigger=_resolve_summarization_trigger(
                _AUTO_SUMMARIZATION_TRIGGER, model),
            keep=("messages", 12))
    except Exception:  # noqa: BLE001 - optional
        return None


def _final_text(result) -> str:
    from funhouse_agent.deep.vision_engine import _content_to_text
    for m in reversed((result or {}).get("messages", [])):
        if getattr(m, "type", "") == "ai":
            text = _content_to_text(getattr(m, "content", ""))
            if text.strip():
                return text
    return "(the reading helper returned no text)"


def make_reader_tool(model, reader_tools: list,
                     budget: int = READER_MAX_MODEL_CALLS,
                     extra_middleware: Optional[list] = None):
    """The ``task`` tool: one ``page_reader`` helper, built on first use.

    Named ``task`` with a ``subagent_type`` like deepagents' own delegation,
    so the app's activity log files the helper's calls under it.
    ``extra_middleware`` is added to the helper's (the coverage recorder,
    so the pages a helper reads count).
    """
    from langchain.agents import create_agent
    from langchain_core.tools import StructuredTool

    state: Dict[str, Any] = {"agent": None}
    lock = threading.Lock()

    def reader():
        with lock:
            if state["agent"] is None:
                mw = [m for m in (
                    _patch_tool_calls(),
                    ModelCallBudgetMiddleware(budget,
                                              final_turn_nudge=READER_NUDGE,
                                              exhausted_message=READER_EXHAUSTED),
                ) if m is not None] + [m for m in (extra_middleware or [])
                                       if m is not None]
                state["agent"] = create_agent(
                    model, tools=reader_tools,
                    system_prompt=DOCUMENT_REVIEW_READER_PROMPT,
                    middleware=mw)
            return state["agent"]

    def task(description: str, subagent_type: str = "page_reader") -> str:
        # A helper's failure (a rate limit, a gateway error) is the helper's
        # answer, not the end of the reviewer's turn.
        try:
            result = reader().invoke(
                {"messages": [{"role": "user", "content": description}]},
                config={"recursion_limit": 10 * budget + 30})
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            return (f"The reading helper failed: {type(exc).__name__}: "
                    f"{str(exc)[:300]}. Do this reading yourself, or hand it "
                    f"over again as a smaller job.")
        return _final_text(result)

    return StructuredTool.from_function(task, name="task",
                                        description=TASK_DESCRIPTION)


def build_review_agent(model, *, engine=None,
                       attachments: Optional[Dict[str, bytes]] = None,
                       save_fn: Optional[Callable] = None,
                       extra_tools=None,
                       extra_system_prompt: Optional[str] = None,
                       markup_author: Optional[str] = None,
                       max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
                       reference_result_chars: Optional[int] = None,
                       checkpointer=None, store=None,
                       model_calls: Optional[int] = None,
                       working_dir: Optional[str] = None,
                       coverage_dir: Optional[str] = None,
                       **_ignored):
    """Build the lean Document Review agent (see the module docstring).

    Takes the keyword arguments :func:`build_deep_agent` does, so a host can
    call either; the geotechnical ones (module scope, reference and calc
    sub-agents, analysis depth) have nothing to act on here and are ignored.

    ``working_dir`` is the conversation's working folder, bound NOW: the
    findings ledger and the digests are kept there even if another
    conversation in the same process re-points the process-wide working
    folder meanwhile. ``None`` looks the working folder up at each call.
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import TodoListMiddleware

    if engine is None and not isinstance(model, str):
        from funhouse_agent.deep.vision_engine import LangChainVisionEngine
        engine = LangChainVisionEngine(model)
    attachments = {} if attachments is None else attachments
    inline = review_flags.vision_inline()
    # Image FILES (contact sheets) are shown to the model only when the
    # overview switch is on as well: the released ``inline`` arm keeps its
    # one-shot analyze_image.
    inline_files = inline and review_flags.overview()
    budget = int(model_calls or max_model_calls())

    def tools_for(names, inline_images: bool, inline_image_files: bool):
        wanted = set(names)
        if not _find_like_available():
            wanted.discard("find_like")
        descriptions = dict(REVIEW_DESCRIPTIONS)
        if inline_images:
            descriptions.update(INLINE_DESCRIPTIONS)
            if not inline_image_files:
                descriptions["analyze_image"] = \
                    REVIEW_DESCRIPTIONS["analyze_image"]
        return make_vision_tools(
            engine=engine, attachments=attachments, save_fn=save_fn,
            include=wanted, max_result_chars=max_result_chars,
            reference_result_chars=reference_result_chars,
            markup_author=markup_author,
            description_overrides=descriptions,
            inline_images=inline_images,
            inline_image_files=inline_image_files)

    tools = tools_for(REVIEW_TOOLS, inline, inline_files)
    reader_tools = tools_for(READER_TOOLS, False, False)
    if review_flags.geometry():
        # Read-only, so the reading helper gets them too.
        from funhouse_agent.deep.geometry_tools import make_geometry_tools
        tools += make_geometry_tools(attachments)
        reader_tools += make_geometry_tools(attachments)
    if review_flags.digest():
        # They only read (the digest cache is all they write, never a
        # deliverable or a markup), so the reading helper gets them too.
        from funhouse_agent.deep.digest_tools import make_digest_tools
        tools += make_digest_tools(attachments, working_dir=working_dir)
        reader_tools += make_digest_tools(attachments,
                                          working_dir=working_dir)
    # Coverage (GEOTECH_COVERAGE / GEOTECH_REVIEW_CHECKLIST, OFF by default):
    # the gate on this agent, a recorder on its reading helper.
    from funhouse_agent.deep.coverage_tools import coverage_kit
    coverage = coverage_kit(attachments=attachments, folder=coverage_dir,
                            max_result_chars=max_result_chars)
    tools += list(coverage.tools)
    tools.append(make_reader_tool(
        model, reader_tools,
        extra_middleware=[coverage.recorder("page_reader")]))
    if review_flags.sweep():
        from funhouse_agent.deep.sweep import make_sweep_tool
        tools.append(make_sweep_tool(
            engine, attachments, model=model,
            max_result_chars=DEFAULT_VISION_RESULT_CHARS))
    if review_flags.findings():
        from funhouse_agent.deep.findings_tools import make_findings_tools
        tools += make_findings_tools(attachments, save_fn=save_fn,
                                     markup_author=markup_author,
                                     working_dir=working_dir)
    if extra_tools:
        seen = {t.name for t in tools}
        tools += [t for t in extra_tools if getattr(t, "name", None) not in seen]

    system_prompt = build_document_review_prompt(lean=True)
    if extra_system_prompt:
        system_prompt = system_prompt + "\n\n" + extra_system_prompt

    images = None
    if inline:
        from funhouse_agent.deep.inline_images import InlineImageMiddleware
        images = InlineImageMiddleware(engine=engine)
    # The image middleware wraps OUTSIDE the budget: it adds the newest
    # images first and the budget's last-call instruction follows them, so
    # the final answer is written with the pages the model was just shown.
    middleware: List[Any] = [m for m in (
        _patch_tool_calls(),
        TodoListMiddleware(system_prompt=TODO_PROMPT,
                           tool_description=TODO_TOOL_DESCRIPTION),
        _summarizer(model),
        images,
        ModelCallBudgetMiddleware(budget, final_turn_nudge=FINAL_NUDGE,
                                  exhausted_message=EXHAUSTED),
        coverage.primary(budget=budget),
    ) if m is not None]

    kwargs: Dict[str, Any] = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    if store is not None:
        kwargs["store"] = store
    agent = create_agent(model, tools=tools, system_prompt=system_prompt,
                         middleware=middleware, **kwargs)
    try:
        agent.geotech_attachments = attachments
        # Each model call is several graph steps (measured 6 on deepagents
        # 0.6.8 / langchain 1.3: two before-model nodes, the model, two
        # after-model nodes, the tools); the budget, not the host's step cap,
        # must be what ends a long request.
        agent.geotech_min_recursion_limit = 8 * budget + 30
        agent.geotech_review_agent = "lean"
        agent.geotech_working_dir = working_dir
        if coverage.on:
            agent.geotech_coverage_ledger = coverage.ledger
    except Exception:  # noqa: BLE001 - attributes are conveniences
        pass
    return agent


__all__ = ["build_review_agent", "make_reader_tool", "REVIEW_TOOLS",
           "READER_TOOLS", "REVIEW_DESCRIPTIONS", "max_model_calls",
           "DEFAULT_MAX_MODEL_CALLS", "MAX_MODEL_CALLS_ENV"]

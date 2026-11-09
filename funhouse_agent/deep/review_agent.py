"""The Document Review page's shared parts: the tool lists, descriptions,
phrase swaps, budgets and helper pieces that the page's default build
(:mod:`funhouse_agent.review_build`) and the looking-only ``minimal`` build
(:mod:`funhouse_agent.deep.minimal_agent`) both read.

This file once also built the page's lean agent (``GEOTECH_REVIEW_AGENT=lean``)
and its experiments behind switches - the main model shown images itself
(``GEOTECH_VISION_INLINE``), ``sweep_pages``, the findings tools, the
drawing-geometry tools, the digest tools and the overview orientation. None
showed a gain on the review suite (``module_work/REVIEW_HARNESS.md`` runs
3-4), and BACKLOG DR-39 removed them (``module_work/dr39/CHECK.md``). The
default build took over the lean build's fixes long ago (DR-01 to DR-03).
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

from funhouse_agent.vision_tools import (
    IMAGE_LOOK_NOTE, IMAGE_ZOOM_NOTE, TURN_ARG_NOTE,
)

#: The page agent's tools, all built by ``make_vision_tools``.
REVIEW_TOOLS = (
    "open_document", "document_structure", "document_page_map",
    "read_document", "search_document", "document_markups",
    "render_page_thumbnails", "find_quantities", "annotate_document",
    "analyze_pdf_page", "render_region", "analyze_image", "find_like",
    "write_docx", "save_file", "list_files", "read_text_file",
    # Visual scales: read-only, so the reading helper gets it too. Owner,
    # 2026-10-08: on the legacy and lean builds, NOT the minimal one.
    # log_grid (a boring or test-pit log read as its grid) is the geotech
    # page's alone since BACKLOG RI-34.
    "measure",
    # A chart and a spreadsheet, as on the legacy build (live smoke wave 1,
    # A12); each is built only where its library imports.
    "plot_data", "write_xlsx",
    # Notes on a DXF copy, written by a CAD library (live smoke wave 2c, E4).
    "annotate_dxf",
)

#: The reading helper's tools: nothing that writes a file or a markup.
READER_TOOLS = tuple(t for t in REVIEW_TOOLS
                     if t not in ("annotate_document", "write_docx",
                                  "save_file", "plot_data", "write_xlsx",
                                  "annotate_dxf"))

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
        "that; N or 'NxN' with N from 2 to 4, e.g. '3x3', forces it). "
        "Every reading is kept for the conversation: reuse=true returns the "
        "latest reading of the page - saying what it was asked - instead of "
        "a new look, for a follow-up that reading already answers; look "
        "again when the question needs more. " + TURN_ARG_NOTE + " page "
        "is 0-based; or give pdf_page, 1-based, as a PDF viewer shows."),
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
        "dimension before you quote it. Every zoom is kept for the "
        "conversation: reuse=true returns the kept zoom of the same region - "
        "saying what it was asked - instead of a new look. " + IMAGE_ZOOM_NOTE
        + " " + TURN_ARG_NOTE
        + " page is 0-based; or give pdf_page, 1-based, as a PDF viewer "
        "shows."),
    "analyze_image": (
        "Look at an image file - an uploaded picture or pasted screenshot, "
        "or a contact sheet that render_page_thumbnails wrote (pass its "
        "path) - and answer your prompt about it. " + IMAGE_LOOK_NOTE),
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

#: Phrases swapped inside descriptions built elsewhere (planlens' own, the
#: measure tool's, plot_data's from profile_figure), so this page's examples
#: name no kind of geotechnical report (BACKLOG RI-34: geotech reports are
#: not this page's documents; the geotech page keeps the original wording).
#: A phrase that no longer matches leaves the original in place, and
#: ``test_review_page_names_no_geotech_report_things`` then fails.
REVIEW_REWRITES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "search_document": (
        ("a value, a boring or sheet id", "a value, an item or sheet id"),),
    "render_page_thumbnails": (
        ("spot the plan, the logs, the tables",
         "spot the plans, the drawings, the tables"),),
    "measure": (
        ("a log's depth or elevation ruler, a plot's axes",
         "a section's depth or elevation ruler, a plot's axes"),
        ("e.g. all layer boundaries in a log's description column",
         "e.g. all the boundaries drawn across a column"),),
    "plot_data": (
        ("an SPT or CPT profile against elevation, settlement vs time, load "
         "vs displacement, a parameter sweep or method comparison, the cases "
         "of a sensitivity study, a plan view of borings and a footprint "
         "drawn to scale",
         "quantities read from a document, values against date or "
         "elevation, a comparison of options or cases, a site plan or a "
         "footprint drawn to scale"),
        ("(cases, scenarios, borings)", "(cases, scenarios, items)"),
        ("borings start at different ground levels",
         "points along a section start at different levels"),
        ("(boring-log convention)", "(the depth convention)"),
        ("as a boring log reads", "as a depth profile reads"),),
}

#: The same for a helper's description in the ``task`` tool: the figure
#: helper (on by default since GM-227) is described by its examples. Its own
#: prompt and worked examples stay as they are (INF-26 plan, D8).
REVIEW_HELPER_REWRITES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "figure": (
        ("a plan of borings and a footprint, a profile (SPT, CPT, su, water "
         "content against elevation)",
         "a site plan and a footprint, a profile of values against "
         "elevation"),),
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


def _busy_retry():
    """A busy model is waited for on a ladder, not the end of the agent
    (BACKLOG TA-20; :mod:`funhouse_agent.deep.busy_retry`)."""
    from funhouse_agent.deep.busy_retry import AgentBusyRetry
    return AgentBusyRetry()


def _summarizer(model):
    """A summarizer for long conversations (the one deepagents added for the
    legacy agent), at the same trigger the rest of the app uses."""
    try:
        from langchain.agents.middleware import SummarizationMiddleware
        from funhouse_agent.deep.assembly import (
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


__all__ = ["REVIEW_TOOLS",
           "READER_TOOLS", "REVIEW_DESCRIPTIONS", "REVIEW_REWRITES",
           "REVIEW_HELPER_REWRITES",
           "max_model_calls",
           "DEFAULT_MAX_MODEL_CALLS", "MAX_MODEL_CALLS_ENV"]

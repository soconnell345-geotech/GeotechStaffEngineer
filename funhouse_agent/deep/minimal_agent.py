"""A deliberately minimal Document Review agent, for measurement only
(``GEOTECH_REVIEW_AGENT=minimal``; suite arm ``minimal``).

WHY. The owner, 2026-10-02: "if we built a very stripped down harness that
just has the zoom tool and ability to make subagents (who also have the zoom
tool), all on GPT 5.6 sol, how do you think it would perform? ... Would it be
worth building as a base-line comparison?" The Foundry run of 5.31 showed the
page as released passing all 35 tasks on GPT-5.6 Sol and no switch adding a
measurable gain, which is exactly when the question is worth asking: which of
the harness's tools carry weight on a strong model, and which are habit?

WHAT. The model LOOKS and nothing else:

* ``analyze_pdf_page`` — a whole page rendered and shown to the model itself;
* ``render_region`` (zoom) — a region re-drawn from the PDF to fill the image, shown too
  (the move the Opus 5.5 system card's Chartography result came from);
* ``analyze_image`` — an uploaded picture, shown;
* ``page_count`` — how many pages a PDF has and their sizes, nothing more;
* ``task`` — a helper with the same looking tools and the same short rules,
  for splitting a long set;
* ``write_docx`` and ``mark_up`` — the page's purpose includes handing back
  files, so the two output tools stay (``mark_up`` takes a location as the
  zoom it came from: view + image_box).

No text layer, no search, no page map or document structure, no markup
reading, no sweep, no geometry, no digest — so the suite says what those were
worth. The prompt is short and generic. It needs an endpoint that takes an
image in a user message after tool results (``inline_images.probe``).
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Dict, List, Optional

from funhouse_agent.deep.limits import ModelCallBudgetMiddleware
from funhouse_agent.deep.review_agent import (
    EXHAUSTED, FINAL_NUDGE, READER_EXHAUSTED, READER_MAX_MODEL_CALLS,
    READER_NUDGE, _final_text, _patch_tool_calls, _summarizer,
    max_model_calls,
)
from funhouse_agent.deep.tools import DEFAULT_MAX_RESULT_CHARS, make_vision_tools

MINIMAL_PROMPT = """\
You review documents for people who design and build things: drawing sets,
specifications, submittals, reports, calculation packages. A document is
attached to the conversation; its file name is the `source` your tools take.

You read by LOOKING. `page_count` tells you how many pages a PDF has.
`analyze_pdf_page` shows you a whole page. A page is shrunk to fit the image
you see, so small lettering on a drawing or a dense table comes out too small
to read: `render_region` zooms - it re-draws a part of the page from the PDF
at full size and shows it to you. To zoom on something you saw, pass the
`view` of the image it was in and its `image_box` on a 0-999 grid over that
image (0,0 top-left, 999,999 bottom-right); zoom again on a smaller box when
it is still small. Never call anything unreadable before you have zoomed on
it. An uploaded picture is shown with `analyze_image`.

For a long document or a question about every page, split the pages among
helpers with `task`: each has the same looking tools, sees nothing of this
conversation, and reports back what it found with page numbers. Give each a
self-contained job (the source, the pages, what to find, what to return).

Pages in your tools count from 0; cite them as a reader counts them, from 1
(the tool's page + 1), or by a sheet number printed on the page. Say which
pages you looked at, and that "not found" means not found on those. When you
are asked for a memo or a summary to pass on, write it with `write_docx`;
for comments on the pages, `mark_up` a copy, each mark located by the view
and image_box of the zoom where you saw the thing - never a guessed spot.
"""

READER_PROMPT = """\
You are looking at part of a document for a reviewer who handed this job to
you. You read by LOOKING: `analyze_pdf_page` shows a whole page, shrunk to
fit the image; `render_region` re-draws a part of a page at full size (pass
the `view` of an image and an `image_box` on a 0-999 grid over it). Zoom
before calling anything unreadable. Do the job you were given and return
ONLY a compact list of what you found, each with its page (counted from 1 =
the tool's page + 1), and one line on anything you could not check.
"""

DESCRIPTIONS = {
    "analyze_pdf_page": (
        "Look at one whole page of a PDF: it is rendered and shown to you "
        "with your next step. source = the attachment's file name; page is "
        "0-based. The result gives the page's `view` (the rect the image "
        "shows) for zooming."),
    "render_region": (
        "Zoom: re-draw part of a page from the PDF at full size and look at "
        "it - it is shown to you with your next step. Say where with view + "
        "image_box (the view of an image you saw and a box on a 0-999 grid "
        "over that image), or bbox = [x0, y0, x1, y1] in PDF points from the "
        "top-left. A smaller box shows finer lettering. page is 0-based."),
    "analyze_image": (
        "Look at an uploaded picture (a screenshot, a photo): it is shown to "
        "you with your next step."),
    "write_docx": (
        "Write a Word (.docx) document from Markdown - a memo, a comment "
        "log, a summary. A bare file name lands in the working folder."),
}

TASK_DESCRIPTION = (
    "Hand a self-contained looking job to a helper with your looking tools - "
    "e.g. 'in <source>, look at pages 12-20 and list every callout reading "
    "X, with its page'. It sees nothing of this conversation: put the "
    "source, the pages, what to find and what to return in description. "
    "Several can run at once. subagent_type: 'page_looker'.")

MARK_UP_DESCRIPTION = (
    "Write comments onto a COPY of a PDF. markups = a list of {kind: 'box' | "
    "'circle' | 'note' | 'callout', page (0-based), comment, label (short "
    "text drawn beside a box or circle), and WHERE: view + image_box from "
    "the render_region where you saw the thing}. output_path = a bare file name. Each "
    "box or circle is checked on the marked copy and the result says which "
    "are misplaced; redo those (append=false) before handing the file over.")


def _looking_tools(engine, attachments, save_fn, markup_author,
                   max_result_chars, *, with_output: bool) -> list:
    names = {"analyze_pdf_page", "render_region", "analyze_image"}
    if with_output:
        names |= {"write_docx", "open_document", "annotate_document"}
    built = make_vision_tools(
        engine=engine, attachments=attachments, save_fn=save_fn,
        include=names, max_result_chars=max_result_chars,
        markup_author=markup_author, description_overrides=DESCRIPTIONS,
        inline_images=True, inline_image_files=True)
    # The standard names stay: the notes a look returns name its tools
    # ("zoom with render_region"), so renaming them would point the model at
    # tools that are not there.
    by_name = {t.name: t for t in built}
    out = [by_name[n] for n in ("analyze_pdf_page", "render_region",
                                "analyze_image") if n in by_name]
    out.append(_page_count_tool(attachments))
    if with_output:
        if "write_docx" in by_name:
            out.append(by_name["write_docx"])
        if {"open_document", "annotate_document"} <= set(by_name):
            out.append(_mark_up_tool(by_name["open_document"],
                                     by_name["annotate_document"]))
    return out


def _page_count_tool(attachments):
    from langchain_core.tools import StructuredTool

    def page_count(source: str) -> str:
        """How many pages a PDF has, and each page's size in points."""
        from funhouse_agent.vision_tools import _resolve_attachment_or_path
        try:
            data, _src = _resolve_attachment_or_path(source, attachments)
            import fitz
            with fitz.open(stream=data, filetype="pdf") as doc:
                sizes = [[round(p.rect.width), round(p.rect.height)]
                         for p in doc]
        except Exception as exc:  # noqa: BLE001 - reported to the model
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        out: Dict[str, Any] = {"source": source, "pages": len(sizes)}
        if len({tuple(s) for s in sizes}) == 1:
            out["page_size_pt"] = sizes[0] if sizes else None
        else:
            out["page_sizes_pt"] = sizes
        return json.dumps(out)

    return StructuredTool.from_function(page_count, name="page_count")


def _mark_up_tool(open_tool, annotate_tool):
    from langchain_core.tools import StructuredTool

    def mark_up(source: str, markups: list, output_path: str = "",
                append: bool = True) -> str:
        """Comments on a copy of the PDF (see the tool description)."""
        try:
            opened = json.loads(open_tool.invoke({"source": source}))
            handle = opened.get("handle")
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"could not open {source}: {exc}"})
        if not handle:
            return json.dumps({"error": f"could not open {source}",
                               "detail": str(opened)[:300]})
        return annotate_tool.invoke({"handle": handle, "markups": markups,
                                     "output_path": output_path,
                                     "append": append})

    return StructuredTool.from_function(mark_up, name="mark_up",
                                        description=MARK_UP_DESCRIPTION)


def _helper_tool(model, engine, helper_tools: list,
                 budget: int = READER_MAX_MODEL_CALLS):
    """``task`` -> a helper that looks with the same tools (and is shown its
    own images by its own image middleware)."""
    from langchain.agents import create_agent
    from langchain_core.tools import StructuredTool
    from funhouse_agent.deep.inline_images import InlineImageMiddleware

    state: Dict[str, Any] = {"agent": None}
    lock = threading.Lock()

    def helper():
        with lock:
            if state["agent"] is None:
                mw = [m for m in (
                    _patch_tool_calls(),
                    InlineImageMiddleware(engine=engine),
                    ModelCallBudgetMiddleware(
                        budget, final_turn_nudge=READER_NUDGE,
                        exhausted_message=READER_EXHAUSTED),
                ) if m is not None]
                state["agent"] = create_agent(model, tools=helper_tools,
                                              system_prompt=READER_PROMPT,
                                              middleware=mw)
            return state["agent"]

    def task(description: str, subagent_type: str = "page_looker") -> str:
        try:
            result = helper().invoke(
                {"messages": [{"role": "user", "content": description}]},
                config={"recursion_limit": 10 * budget + 30})
        except Exception as exc:  # noqa: BLE001 - the helper's answer
            return (f"The helper failed: {type(exc).__name__}: "
                    f"{str(exc)[:300]}. Look yourself, or hand it a smaller "
                    f"job.")
        return _final_text(result)

    return StructuredTool.from_function(task, name="task",
                                        description=TASK_DESCRIPTION)


def build_minimal_agent(model, *, engine=None,
                        attachments: Optional[Dict[str, bytes]] = None,
                        save_fn: Optional[Callable] = None,
                        extra_tools=None,
                        extra_system_prompt: Optional[str] = None,
                        markup_author: Optional[str] = None,
                        max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
                        checkpointer=None, store=None,
                        model_calls: Optional[int] = None,
                        working_dir: Optional[str] = None,
                        **_ignored):
    """Build the minimal looking-only agent (see the module docstring).
    Takes the keyword arguments :func:`build_deep_agent` does; the host's
    extra tools (SharePoint, feedback) are left out on purpose."""
    from langchain.agents import create_agent
    from funhouse_agent.deep.inline_images import InlineImageMiddleware

    if engine is None and not isinstance(model, str):
        from funhouse_agent.deep.vision_engine import LangChainVisionEngine
        engine = LangChainVisionEngine(model)
    attachments = {} if attachments is None else attachments
    budget = int(model_calls or max_model_calls())

    tools = _looking_tools(engine, attachments, save_fn, markup_author,
                           max_result_chars, with_output=True)
    helper_tools = _looking_tools(engine, attachments, save_fn, markup_author,
                                  max_result_chars, with_output=False)
    tools.append(_helper_tool(model, engine, helper_tools))

    system_prompt = MINIMAL_PROMPT
    if extra_system_prompt:
        system_prompt = system_prompt + "\n\n" + extra_system_prompt
    middleware: List[Any] = [m for m in (
        _patch_tool_calls(),
        _summarizer(model),
        InlineImageMiddleware(engine=engine),
        ModelCallBudgetMiddleware(budget, final_turn_nudge=FINAL_NUDGE,
                                  exhausted_message=EXHAUSTED),
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
        agent.geotech_min_recursion_limit = 8 * budget + 30
        agent.geotech_review_agent = "minimal"
        agent.geotech_working_dir = working_dir
    except Exception:  # noqa: BLE001 - conveniences
        pass
    return agent


__all__ = ["build_minimal_agent", "MINIMAL_PROMPT", "READER_PROMPT"]

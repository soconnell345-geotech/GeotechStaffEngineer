"""``sweep_pages``: one question asked of every page in a range, in parallel.

The review of 2026-09-26 found that questions that need COVERAGE — "every",
"all", "count", "check each sheet", "where do the specs and the drawings
disagree" — had no general mechanism. The agent either paged through whole
sheets one call at a time until its step budget ran out, or a special-purpose
tool was built for one kind of mark. This is the general mechanism: the same
question goes to every page, each page is answered on its own (from its text
where the text is the page, by looking where it is a picture), and the answers
come back per page with citations for the agent to aggregate, check and zoom
into.

It is offered only with ``GEOTECH_REVIEW_SWEEP`` on, to the lean review agent,
and is known to the agent only through its own tool description.
"""

from __future__ import annotations

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

#: Pages answered at once.
SWEEP_WORKERS = 6
#: Most pages one call covers; the rest come back as ``next_pages``.
SWEEP_MAX_PAGES = 60
#: Characters of page text given to a text-page answer.
PAGE_TEXT_CHARS = 12000
#: Page kinds read by looking rather than from their text.
LOOK_KINDS = ("drawing_sheet", "figure", "scanned", "form")

DESCRIPTION = (
    "Ask ONE question of every page in a range and get a per-page answer — "
    "for questions that need coverage: every/all/each, counts, 'which sheets "
    "show X', 'check every page for Y'. Each page is answered on its own, "
    "from its text where the text is the page and by looking at it where it "
    "is a drawing, scan or figure (look='always' looks at every page, "
    "look='never' reads text only). Returns the pages where the answer is "
    "yes/relevant with what was found there (and page boxes to zoom on), the "
    "pages checked with nothing found, and the pages whose answer was "
    "unsure — zoom with render_region before relying on those. It is a "
    "first pass: verify what you will report. source = the attachment key or "
    "path; pages like '0-40' (0-based; default all, up to 60 a call, the "
    "result's next_pages continues).")

_PAGE_PROMPT = (
    "You are checking ONE page of a document for a reviewer. Answer ONLY "
    "from this page; say nothing about other pages.\n\nQuestion: {question}\n\n"
    "Reply with ONLY a JSON object: {{\"relevant\": true or false (does this "
    "page contain anything that answers the question), \"answer\": what this "
    "page says or shows about it, in one or two sentences, \"items\": [each "
    "thing found: {{\"text\": the exact characters as printed, \"box\": [x0, "
    "y0, x1, y1] on the 0-999 grid over the image, or null}}], \"sure\": "
    "true or false}}. If a character could be another, bracket the "
    "alternatives, e.g. A[B/8]C, and set sure to false.")


def _parse(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _text_call(model, prompt: str) -> str:
    from langchain_core.messages import HumanMessage
    from funhouse_agent.deep.vision_engine import _content_to_text, call_slot
    with call_slot():
        resp = model.invoke([HumanMessage(content=prompt)])
    return _content_to_text(getattr(resp, "content", resp))


def sweep(pdf, question: str, engine, *, pages=None, look: str = "auto",
          model=None) -> Dict[str, Any]:
    """Answer ``question`` for each page of ``pdf`` (bytes or path)."""
    from planlens.document import Document
    from planlens.document.document import parse_pages
    from funhouse_agent import vision_view

    doc = (Document(content=pdf) if isinstance(pdf, (bytes, bytearray))
           else Document(filepath=str(pdf)))
    try:
        wanted = parse_pages(pages, doc.n_pages) if pages not in (
            None, "") else list(range(doc.n_pages))
        todo, rest = wanted[:SWEEP_MAX_PAGES], wanted[SWEEP_MAX_PAGES:]
        kinds = {s.page: s.kind for s in doc.page_map(todo)}
        texts: Dict[int, str] = {}
        lines: Dict[int, Any] = {}
        for p in todo:
            pc = doc.page(p, tables=False)
            texts[p] = pc.text()[:PAGE_TEXT_CHARS]
            lines[p] = [(" ".join(ln.text.split()), tuple(ln.bbox))
                        for ln in pc.lines if ln.text and ln.text.strip()]
    finally:
        doc.close()
    source = pdf

    text_model = model if model is not None else getattr(engine, "model", None)

    def one(p: int) -> Dict[str, Any]:
        by_eye = (look == "always" or (look == "auto" and (
            kinds.get(p) in LOOK_KINDS or len(texts.get(p, "").strip()) < 40)))
        if look == "never" or (not by_eye and text_model is not None):
            prompt = (_PAGE_PROMPT.format(question=question)
                      + "\n\nThe page's text (drafting order):\n"
                      + (texts.get(p) or "[no text]"))
            if text_model is None:
                return {"page": p, "error": "no text model"}
            raw, how, view = _text_call(text_model, prompt), "text", None
        else:
            img, info = vision_view.render_view(
                source, page=p, allow_jpeg=getattr(engine, "accepts_jpeg",
                                                   False), engine=engine)
            view = info["clip"]
            prompt = (vision_view.text_context(lines.get(p, []), view) + "\n\n"
                      + _PAGE_PROMPT.format(question=question))
            raw, how = engine.analyze_image(img, prompt), "looked"
        data = _parse(raw) or {"relevant": None, "answer": raw[:400],
                               "sure": False}
        row = {"page": p, "pdf_page": p + 1, "how": how,
               "relevant": data.get("relevant"),
               "answer": str(data.get("answer") or "")[:600],
               "sure": bool(data.get("sure", False))}
        items = []
        for it in data.get("items") or []:
            if not isinstance(it, dict):
                continue
            item = {"text": it.get("text")}
            box = it.get("box")
            if view is not None and isinstance(box, (list, tuple)) and len(box) == 4:
                try:
                    item["page_bbox"] = [round(v, 1) for v in
                                         vision_view.image_box_to_page(view, box)]
                except (TypeError, ValueError):
                    pass
            items.append(item)
        if items:
            row["items"] = items[:25]
        return row

    def safe(p: int) -> Dict[str, Any]:
        try:
            return one(p)
        except Exception as exc:  # noqa: BLE001 - one page, not the sweep
            return {"page": p, "pdf_page": p + 1,
                    "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=SWEEP_WORKERS) as ex:
        # Each page's calls run in a copy of this context, so the run's
        # callbacks (token counts, the activity log) see them.
        futs = [ex.submit(contextvars.copy_context().run, safe, p) for p in todo]
        rows = [f.result() for f in futs]

    from planlens.tools.formatting import compact_ranges
    relevant = [r for r in rows if r.get("relevant") is True]
    unsure = [r["page"] for r in rows if r.get("relevant") is True
              and not r.get("sure")]
    out: Dict[str, Any] = {
        "question": question,
        "pages_checked": compact_ranges(todo),
        "relevant": relevant,
        "nothing_found": compact_ranges(
            r["page"] for r in rows if r.get("relevant") is False),
        "unsure": compact_ranges(unsure),
        "unanswered": [{"page": r["page"], "pdf_page": r["page"] + 1,
                        "why": r.get("error") or "no JSON answer"}
                       for r in rows if r.get("relevant") is None],
        "note": ("pages are 0-based (pdf_page = viewer page); a first pass — "
                 "zoom on anything you will report, especially unsure pages"),
    }
    if rest:
        out["next_pages"] = compact_ranges(rest)
    return out


def fit(result: Dict[str, Any], limit: int) -> str:
    """``result`` as JSON within ``limit`` characters, shortening per-page
    answers evenly (never cutting the JSON)."""
    text = json.dumps(result, ensure_ascii=False)
    while len(text) > limit and result.get("relevant"):
        longest = max(result["relevant"], key=lambda r: len(r.get("answer", "")))
        if len(longest.get("answer", "")) > 80:
            longest["answer"] = longest["answer"][: int(len(longest["answer"]) * 0.7)] + "…"
        elif any(r.get("items") for r in result["relevant"]):
            for r in result["relevant"]:
                if r.get("items"):
                    r["items"] = r["items"][: max(0, len(r["items"]) // 2)]
        else:
            result["relevant"] = result["relevant"][:-1]
            result["truncated"] = "some relevant pages dropped; sweep a narrower range"
        text = json.dumps(result, ensure_ascii=False)
    return text


def make_sweep_tool(engine, attachments: Dict[str, bytes], model=None,
                    max_result_chars: int = 32000):
    """The ``sweep_pages`` LangChain tool bound to this agent's uploads."""
    from langchain_core.tools import StructuredTool
    from funhouse_agent import document_tools

    def sweep_pages(source: str, question: str, pages: str = "",
                    look: str = "auto") -> str:
        try:
            resolved = document_tools.resolve_document_source(source,
                                                              attachments)
            if not isinstance(resolved, (bytes, bytearray)):
                with open(resolved, "rb") as fh:
                    resolved = fh.read()
        except Exception as exc:  # noqa: BLE001 - reported to the model
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        if look not in ("auto", "always", "never"):
            look = "auto"
        try:
            result = sweep(resolved, question, engine, pages=pages or None,
                           look=look, model=model)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"sweep failed: {type(exc).__name__}: "
                                        f"{exc}"})
        return fit(result, max_result_chars)

    return StructuredTool.from_function(sweep_pages, name="sweep_pages",
                                        description=DESCRIPTION)


__all__ = ["sweep", "make_sweep_tool", "fit", "DESCRIPTION", "SWEEP_MAX_PAGES"]

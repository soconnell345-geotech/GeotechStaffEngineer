"""Let the reasoning model look: show it the newest rendered page images.

The review of 2026-09-26 found that the Document Review agent never sees a
page. Every look was a separate, one-shot vision call that got only the image
and the agent's question, and the agent reasoned from whatever that call chose
to write. With ``GEOTECH_VISION_INLINE`` on (lean review agent only), the page
and region tools store the image (:mod:`funhouse_agent.inline_store`) and this
middleware puts the newest ones in front of the main model at its next call,
as a user message after the tool results — the shape every OpenAI-compatible
chat endpoint accepts (a tool message may carry text only).

The images are added to the REQUEST, never to the conversation state: they
cost tokens only on the calls that follow while they are among the newest
``keep``, and the saved history holds the text results alone.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any, List, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage

from funhouse_agent import inline_store
# Newest images shown at each call, and its setting: defined beside the
# store so the tools' notes can say how long an image stays in view.
from funhouse_agent.inline_store import (  # noqa: F401 - re-exported
    DEFAULT_KEEP, KEEP_ENV, keep_from_env)

_ID = re.compile(r'"image_id"\s*:\s*"(img_[0-9a-f]+)"')


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                       for b in content)
    return str(content or "")


def newest_image_ids(messages, keep: int = DEFAULT_KEEP) -> List[str]:
    """The ids of the newest ``keep`` stored images named by tool results
    since the last user message (oldest first)."""
    ids: List[str] = []
    for m in reversed(list(messages or [])):
        # The image message itself is added to the request only, never to
        # the state, so the first user message met is the user's own.
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage):
            for found in reversed(_ID.findall(_text(m.content))):
                if found not in ids and inline_store.get(found) is not None:
                    ids.append(found)
        if len(ids) >= keep:
            break
    return list(reversed(ids[:keep]))


def image_message(ids: List[str], detail: Optional[str] = None) -> HumanMessage:
    """One user message carrying the images ``ids``, each labelled with where
    it came from so the model can refer to it."""
    from funhouse_agent import vision_view
    blocks: List[dict] = [{
        "type": "text",
        "text": ("[Images returned by your last page/region views, newest "
                 "last. Look at them yourself to answer. A location you give "
                 "on one can be a 0-999 box on that image; render_region "
                 "zooms with its view + image_box.]")}]
    for image_id in ids:
        got = inline_store.get(image_id)
        if got is None:
            continue
        data, meta = got
        if meta.get("label"):
            # an image file (a contact sheet), not a view of a page
            text = f"Image {image_id}: {meta['label']}"
        else:
            text = (f"Image {image_id}: PDF page {meta.get('pdf_page')} (tool "
                    f"page {meta.get('page')}), view "
                    f"{json.dumps(meta.get('view'))}")
        blocks.append({"type": "text", "text": text})
        url = (f"data:{vision_view.image_media_type(data)};base64,"
               f"{base64.b64encode(data).decode()}")
        image_url = {"url": url}
        if detail:
            image_url["detail"] = detail
        blocks.append({"type": "image_url", "image_url": image_url})
    return HumanMessage(content=blocks)


class InlineImageMiddleware(AgentMiddleware):
    """Append the newest stored images to each model request (never to the
    state)."""

    def __init__(self, keep: Optional[int] = None, engine: Any = None) -> None:
        super().__init__()
        self.keep = max(1, int(keep)) if keep else keep_from_env()
        self.engine = engine

    def _with_images(self, request):
        ids = newest_image_ids(request.messages, self.keep)
        if not ids:
            return request
        from funhouse_agent import vision_view
        msg = image_message(ids, detail=vision_view.detail(self.engine))
        return request.override(messages=[*request.messages, msg])

    def wrap_model_call(self, request, handler):
        return handler(self._with_images(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._with_images(request))


def _probe_png() -> bytes:
    """A 96 px red square, drawn with PyMuPDF (no other image library)."""
    import fitz
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 96, 96), False)
    pix.set_rect(pix.irect, (210, 20, 20))
    return pix.tobytes("png")


def probe(model) -> dict:
    """ONE live call: does this endpoint answer from an image sent the way
    :class:`InlineImageMiddleware` sends it — a user message carrying the
    image AFTER an assistant tool call and its tool result?

    Run it once per deployment before trusting the ``inline`` arm::

        from funhouse_agent.deep.inline_images import probe
        probe(PrompterChatModel(prompter=fh_prompter, model="funhouse-gpt-high"))

    Returns ``{"ok", "answer", "error"}``; ``ok`` means the reply names the
    square's colour. Never raises.
    """
    from langchain_core.messages import AIMessage, SystemMessage
    from langchain_core.tools import StructuredTool

    def render_region(page: int = 0) -> str:
        """Render a region of a page (probe stub)."""
        return "{}"

    image_id = inline_store.put(_probe_png(), {"page": 0, "pdf_page": 1,
                                               "view": [0, 0, 96, 96]})
    msgs = [
        SystemMessage(content="Answer in one short sentence. Do not call "
                              "any tool."),
        HumanMessage(content="What colour is the square in the image the "
                             "tool returned?"),
        AIMessage(content="", tool_calls=[{"name": "render_region",
                                           "args": {"page": 0},
                                           "id": "call_probe_1"}]),
        ToolMessage(content=json.dumps({"image_id": image_id,
                                        "note": "shown next"}),
                    tool_call_id="call_probe_1"),
    ]
    msgs.append(image_message(newest_image_ids(msgs)))
    try:
        bound = model.bind_tools([StructuredTool.from_function(render_region)])
    except Exception:  # noqa: BLE001 - a model without tool binding
        bound = model
    try:
        resp = bound.invoke(msgs)
    except Exception as exc:  # noqa: BLE001 - the answer IS the error
        return {"ok": False, "answer": "", "error": f"{type(exc).__name__}: "
                                                   f"{str(exc)[:400]}"}
    text = _text(getattr(resp, "content", resp))
    return {"ok": "red" in text.lower(), "answer": text[:300], "error": None}


__all__ = ["InlineImageMiddleware", "newest_image_ids", "image_message",
           "probe", "DEFAULT_KEEP"]

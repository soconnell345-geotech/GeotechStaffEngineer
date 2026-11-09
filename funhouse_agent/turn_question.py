"""The turn's own user question, handed to the vision side calls (DR-37).

A vision side call (``analyze_image``, ``analyze_pdf_page``,
``render_region`` and the chart read-offs) answers the main model's
``prompt``: its rewording of what it wants from the picture. On the chart
suite (EV-35, 2026-11-08, TRACE_REVIEW section 4.2) that rewording folded the
answer-format line into the question on task 2089, the side call answered
the narrower question, and both Sol page arms copied it; the three arms that
saw the question itself were right.

So the host binds the user's latest message to the turn, the way it binds
the working folder (:func:`funhouse_agent._fileio.bind_working_dir`):
LangGraph copies the context into the threads it runs tools in, so the tool
needs no argument the model must fill. :func:`with_question` then adds it,
capped at :data:`MAX_CHARS`, under the main model's prompt, saying that the
prompt is what to answer and the question is context.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Optional

#: The user's latest message for the turn running in this context.
_QUESTION: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "geotech_turn_question", default=None)

#: The longest part of the user's message a side call is handed.
MAX_CHARS = 600

#: The vision calls whose reading goes back to the main model as an answer
#: to its request, by ``ask_vision`` kind. Left out: ``check`` (the markup
#: check's look at one placed mark) and ``find`` (find_like's read of a
#: sheet of candidates), which read one thing for code, not for the user.
QUESTION_KINDS = frozenset({"image", "page", "overview", "region", "tile",
                            "reference_figure", "worked_example_page"})

_HEAD = (
    "\n\nFor context only -- the user's own latest message in this "
    "conversation. The request above, from the agent working on it, is what "
    "to answer; use this message only to understand what that request is "
    "for. Do not answer it instead of the request, and do not add what it "
    "asks for that the request leaves out.\n"
    "User's message: <<<")
_TAIL = ">>>"


def bind(text) -> "contextvars.Token":
    """Bind ``text`` (the user's own message) to this context's turn;
    returns the token for :func:`unbind`. Blank binds nothing."""
    text = " ".join(str(text or "").split())
    return _QUESTION.set(text or None)


def unbind(token) -> None:
    try:
        _QUESTION.reset(token)
    except (ValueError, RuntimeError):     # set in another context
        pass


@contextlib.contextmanager
def bound(text):
    """``with bound(question):`` -- the turn's question for this context."""
    token = bind(text)
    try:
        yield
    finally:
        unbind(token)


def current() -> Optional[str]:
    """The bound question, capped at :data:`MAX_CHARS`, or ``None``."""
    text = _QUESTION.get()
    if not text:
        return None
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS - 1].rstrip() + "…"
    return text


def with_question(prompt: str, kind: str) -> str:
    """``prompt`` with the turn's question under it as context, for a side
    call of ``kind`` in :data:`QUESTION_KINDS`; ``prompt`` unchanged when no
    question is bound, for any other kind, or when the prompt already
    carries the question word for word."""
    if kind not in QUESTION_KINDS or not isinstance(prompt, str):
        return prompt
    question = current()
    if not question or question in " ".join(prompt.split()):
        return prompt
    return f"{prompt}{_HEAD}{question}{_TAIL}"


__all__ = ["MAX_CHARS", "QUESTION_KINDS", "bind", "unbind", "bound",
           "current", "with_question"]

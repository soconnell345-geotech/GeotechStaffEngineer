"""Two guards on every agent's loop: a tool that raises, and a reply cut off.

**A tool that raises** (live smoke wave 2b, C3). LangGraph's tool node turns
only argument-validation errors into a message for the model; any other
exception ends the run. In F31 ``annotate_document`` raised MuPDF's
``FzErrorSystem`` and the tester's whole turn ended on "(no answer text)",
with the server path in the error. :class:`ToolErrorGuard` catches what a
tool raises and hands the model a clear JSON error instead
(:func:`funhouse_agent.error_text.tool_error`, no server path), so the turn
goes on. LangGraph's own control flow (an ``interrupt()``, a sub-graph's
command) passes through untouched.

**A reply cut off at the output-token limit** (C5). F40 sent the same
``save_file`` five times: each call was cut at exactly 8,192 output tokens,
arrived as ``{"path": ...}`` without its content, and drew "content: Field
required" -- an answer to the wrong question. F42's whole-page transcription
stopped mid-sentence with no flag. :class:`OutputLimitGuard` reads the
provider's own stop reason (:func:`output_was_cut`):

* a cut TOOL CALL is not run: its arguments are incomplete, so the model is
  told so, with the limit, and to make the call smaller -- the earlier calls
  in the same reply were complete and run as usual;
* a cut TEXT reply is continued: the model is asked to carry on exactly
  where it stopped, and the two parts are delivered as one reply
  (at most :data:`MAX_TEXT_CONTINUES` times).
"""

from __future__ import annotations

import json
import threading
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:  # pragma: no cover - older layout
    from langchain.agents.middleware.types import AgentMiddleware

try:
    from langgraph.errors import GraphBubbleUp
except Exception:  # pragma: no cover - control flow then has no base class
    class GraphBubbleUp(Exception):  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# A tool that raises
# ---------------------------------------------------------------------------

def _error_message(request, exc: BaseException) -> ToolMessage:
    from funhouse_agent.error_text import tool_error
    call = getattr(request, "tool_call", None) or {}
    name = call.get("name") or "tool"
    return ToolMessage(
        content=json.dumps(tool_error(name, exc), ensure_ascii=False),
        name=name, tool_call_id=call.get("id") or "", status="error")


class ToolErrorGuard(AgentMiddleware):
    """Turn an exception out of any tool into a tool error the model reads
    (see the module docstring). Put it just inside
    :class:`~funhouse_agent.deep.delivered_log.DeliveredToolResults`, so the
    log records what the model was told, and outside the other tool guards,
    so their own failures are caught too."""

    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the model
            return _error_message(request, exc)

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the model
            return _error_message(request, exc)


# ---------------------------------------------------------------------------
# A reply cut off at the output-token limit
# ---------------------------------------------------------------------------

#: Stop reasons that mean "the output limit was reached" (OpenAI chat
#: ``length``; Anthropic ``max_tokens``; the Responses API's
#: ``max_output_tokens``).
CUT_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})

#: Times a cut text reply is continued before it is delivered as it is.
MAX_TEXT_CONTINUES = 2

CONTINUE_CUT_TEXT = (
    "[Your last reply was cut off at the model's output limit, mid-text. "
    "Continue it exactly where it stopped -- do not repeat anything already "
    "written, and do not start over.]")


def output_was_cut(message: Any) -> bool:
    """Whether a model reply stopped at the output-token limit, read off
    the provider's own stop reason in ``response_metadata``."""
    meta = getattr(message, "response_metadata", None) or {}
    if not isinstance(meta, dict):
        return False
    for key in ("finish_reason", "stop_reason"):
        v = meta.get(key)
        if isinstance(v, str) and v.strip().lower() in CUT_REASONS:
            return True
    if str(meta.get("status", "")).strip().lower() == "incomplete":
        details = meta.get("incomplete_details") or {}
        reason = (details.get("reason") if isinstance(details, dict)
                  else getattr(details, "reason", None))
        return reason in (None, "max_output_tokens")
    return False


def _output_tokens(message: Any) -> Optional[int]:
    usage = getattr(message, "usage_metadata", None) or {}
    n = usage.get("output_tokens") if isinstance(usage, dict) else None
    return int(n) if isinstance(n, (int, float)) and n > 0 else None


def cut_call_error(name: str, args: Any, tokens: Optional[int]) -> str:
    """What the model is told about a tool call its reply was cut off in."""
    got = sorted(args) if isinstance(args, dict) else []
    limit = f"about {tokens:,} tokens" if tokens else "its limit"
    return json.dumps({
        "error": (f"Not run: this call to {name} was CUT OFF at the model's "
                  f"output limit ({limit} for one reply) before its "
                  f"arguments were complete"
                  + (f" -- only {', '.join(got)} arrived" if got else "")
                  + ". Sending the same call again will be cut off again."),
        "hint": ("Make the call smaller: send less in one call (a shorter "
                 "text, fewer rows or markups per call -- annotate_document "
                 "with append=true adds to the same marked copy), or split "
                 "the work into several calls."),
    }, ensure_ascii=False)


def _ai_of(response) -> Optional[AIMessage]:
    """The AI message of a model response (``ModelResponse``, an
    ``ExtendedModelResponse`` or a bare ``AIMessage``)."""
    if isinstance(response, AIMessage):
        return response
    inner = getattr(response, "model_response", None)
    holder = inner if inner is not None else response
    result = getattr(holder, "result", None)
    if isinstance(result, list):
        for msg in reversed(result):
            if isinstance(msg, AIMessage):
                return msg
    return None


def _with_ai(response, old: AIMessage, new: AIMessage):
    """``response`` with ``old`` replaced by ``new``."""
    if isinstance(response, AIMessage):
        return new
    inner = getattr(response, "model_response", None)
    holder = inner if inner is not None else response
    result = getattr(holder, "result", None)
    if isinstance(result, list):
        holder.result = [new if m is old else m for m in result]
    return response


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                       for b in content
                       if not isinstance(b, dict) or b.get("type") in
                       (None, "text"))
    return "" if content is None else str(content)


class OutputLimitGuard(AgentMiddleware):
    """Catch a reply cut off at the output-token limit (see the module
    docstring). Put it LAST among the app's middleware: innermost, it sees
    the model's reply before any other middleware and answers a cut tool
    call before the tool would run."""

    def __init__(self, max_text_continues: int = MAX_TEXT_CONTINUES):
        super().__init__()
        self.max_text_continues = max(0, int(max_text_continues))
        self._cut: dict = {}
        self._lock = threading.Lock()

    # -- model side ---------------------------------------------------------
    def _mark(self, msg: AIMessage) -> AIMessage:
        """A cut reply with tool calls: the LAST call (the one the limit
        fell in) and any whose arguments did not parse are answered with
        :func:`cut_call_error` instead of being run. A call the provider
        could not parse at all is put back as a call, so the model hears
        about it rather than the turn ending on an empty reply."""
        calls = list(getattr(msg, "tool_calls", None) or [])
        bad = list(getattr(msg, "invalid_tool_calls", None) or [])
        revived = [{"name": b.get("name"), "args": {}, "id": b.get("id"),
                    "type": "tool_call"}
                   for b in bad if b.get("name") and b.get("id")]
        cut_ids = [c["id"] for c in revived]
        if calls and calls[-1].get("id") and not revived:
            cut_ids.append(calls[-1]["id"])
        tokens = _output_tokens(msg)
        with self._lock:
            for c in calls + revived:
                if c.get("id") in cut_ids:
                    self._cut[c["id"]] = (c.get("name") or "tool",
                                          c.get("args"), tokens)
            while len(self._cut) > 256:
                self._cut.pop(next(iter(self._cut)))
        if not revived:
            return msg
        return msg.model_copy(update={"tool_calls": calls + revived,
                                      "invalid_tool_calls": []})

    def _continue_text(self, request, response, msg: AIMessage, handler):
        """A cut text reply, carried on where it stopped and joined."""
        text = _text(msg.content)
        last = msg
        for _ in range(self.max_text_continues):
            more = handler(request.override(messages=list(request.messages) + [
                AIMessage(content=text), HumanMessage(content=CONTINUE_CUT_TEXT)]))
            nxt = _ai_of(more)
            if nxt is None:
                break
            text = text + _text(nxt.content)
            last = nxt
            if not output_was_cut(nxt) or nxt.tool_calls:
                break
        merged = last.model_copy(update={"content": text})
        return _with_ai(response, msg, merged)

    async def _acontinue_text(self, request, response, msg, handler):
        text = _text(msg.content)
        last = msg
        for _ in range(self.max_text_continues):
            more = await handler(request.override(
                messages=list(request.messages) + [
                    AIMessage(content=text),
                    HumanMessage(content=CONTINUE_CUT_TEXT)]))
            nxt = _ai_of(more)
            if nxt is None:
                break
            text = text + _text(nxt.content)
            last = nxt
            if not output_was_cut(nxt) or nxt.tool_calls:
                break
        merged = last.model_copy(update={"content": text})
        return _with_ai(response, msg, merged)

    def _after(self, response):
        msg = _ai_of(response)
        if msg is None or not output_was_cut(msg):
            return response, None
        if msg.tool_calls or getattr(msg, "invalid_tool_calls", None):
            marked = self._mark(msg)
            return (_with_ai(response, msg, marked) if marked is not msg
                    else response), None
        if not _text(msg.content).strip():
            return response, None
        return response, msg

    def wrap_model_call(self, request, handler):
        response = handler(request)
        response, cut_text = self._after(response)
        if cut_text is not None and self.max_text_continues:
            response = self._continue_text(request, response, cut_text,
                                           handler)
            final = _ai_of(response)
            if final is not None and output_was_cut(final) and (
                    final.tool_calls
                    or getattr(final, "invalid_tool_calls", None)):
                marked = self._mark(final)
                if marked is not final:
                    response = _with_ai(response, final, marked)
        return response

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        response, cut_text = self._after(response)
        if cut_text is not None and self.max_text_continues:
            response = await self._acontinue_text(request, response,
                                                  cut_text, handler)
            final = _ai_of(response)
            if final is not None and output_was_cut(final) and (
                    final.tool_calls
                    or getattr(final, "invalid_tool_calls", None)):
                marked = self._mark(final)
                if marked is not final:
                    response = _with_ai(response, final, marked)
        return response

    # -- tool side ----------------------------------------------------------
    def _answer_cut(self, request) -> Optional[ToolMessage]:
        call = getattr(request, "tool_call", None) or {}
        with self._lock:
            hit = self._cut.pop(call.get("id"), None)
        if hit is None:
            return None
        name, args, tokens = hit
        return ToolMessage(content=cut_call_error(name, args, tokens),
                           name=call.get("name") or name,
                           tool_call_id=call.get("id") or "", status="error")

    def wrap_tool_call(self, request, handler):
        cut = self._answer_cut(request)
        return cut if cut is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        cut = self._answer_cut(request)
        return cut if cut is not None else await handler(request)


# ---------------------------------------------------------------------------
# The scratch filesystem, off the Document Review page
# ---------------------------------------------------------------------------

#: deepagents' scratch-filesystem tools (and its shell), which see an empty
#: in-memory space and answer "not found" about the user's real files.
SCRATCH_TOOLS = frozenset({"ls", "read_file", "write_file", "edit_file",
                           "glob", "grep", "execute"})

#: The sections deepagents' filesystem middleware adds to the system prompt,
#: about exactly those tools ("## Following Conventions" is its
#: coding-agent advice).
_SCRATCH_SECTIONS = ("Following Conventions", "Filesystem Tools",
                     "Large Tool Results", "Execute Tool")


def strip_scratch_prompt(text: str) -> str:
    """``text`` without the scratch-filesystem sections deepagents appends
    (each from its ``## <title>`` heading to the next heading or the end)."""
    import re
    out = str(text or "")
    for title in _SCRATCH_SECTIONS:
        out = re.sub(r"(?ms)^## " + re.escape(title) + r"\b.*?(?=^## |\Z)",
                     "", out)
    return re.sub(r"\n{3,}", "\n\n", out).rstrip() + (
        "\n" if out.endswith("\n") else "")


def _hide_tools_base():
    from funhouse_agent.deep.limits import HideTools
    return HideTools


class HideScratchFilesystem(_hide_tools_base()):
    """The Document Review page without deepagents' scratch filesystem:
    ``ls``, ``read_file``, ``write_file``, ``edit_file``, ``glob``,
    ``grep`` and ``execute`` are kept off the model's tool list, and their
    section of the system prompt with them (live smoke wave 2b: F41's
    ``read_file('/Submittal Log.xlsx')`` answered "not found" for a file the
    conversation held, and an unknown-tool error listed them as if they
    were the page's tools). They stay registered, so a result deepagents
    saved to scratch (a very large tool result) can still be read back."""

    def __init__(self, names=SCRATCH_TOOLS):
        super().__init__(names)

    def _unlisted(self, request, result):
        """An unknown-tool error lists every REGISTERED tool; the hidden ones
        are taken out of that list too (F31's listed them as the page's)."""
        import re
        if getattr(request, "tool", None) is not None or \
                not isinstance(result, ToolMessage) or \
                not isinstance(result.content, str):
            return result
        m = re.search(r"try one of \[([^\]]*)\]", result.content)
        if not m:
            return result
        kept = [n.strip() for n in m.group(1).split(",")
                if n.strip() and n.strip().strip("'\"") not in self.hidden]
        return result.model_copy(update={"content": (
            result.content[:m.start(1)] + ", ".join(kept)
            + result.content[m.end(1):])})

    def wrap_tool_call(self, request, handler):
        return self._unlisted(request, handler(request))

    async def awrap_tool_call(self, request, handler):
        return self._unlisted(request, await handler(request))

    def _filtered(self, request):
        request = super()._filtered(request)
        sm = getattr(request, "system_message", None)
        if sm is None:
            return request
        content = sm.content
        if isinstance(content, str):
            new = strip_scratch_prompt(content)
            if new == content:
                return request
            return request.override(
                system_message=sm.model_copy(update={"content": new}))
        if isinstance(content, list):
            blocks, changed = [], False
            for b in content:
                if isinstance(b, dict) and isinstance(b.get("text"), str):
                    t = strip_scratch_prompt(b["text"])
                    if t != b["text"]:
                        changed = True
                        if not t.strip():
                            continue
                        b = {**b, "text": t}
                blocks.append(b)
            if changed:
                return request.override(
                    system_message=sm.model_copy(update={"content": blocks}))
        return request


def loop_guards() -> list:
    """The two guards in the order an agent's middleware list wants them:
    ``ToolErrorGuard`` near the outside of the tool middleware, and
    ``OutputLimitGuard`` (to go LAST)."""
    return [ToolErrorGuard(), OutputLimitGuard()]


__all__ = ["ToolErrorGuard", "OutputLimitGuard", "output_was_cut",
           "cut_call_error", "CUT_REASONS", "CONTINUE_CUT_TEXT",
           "MAX_TEXT_CONTINUES", "loop_guards", "HideScratchFilesystem",
           "SCRATCH_TOOLS", "strip_scratch_prompt"]

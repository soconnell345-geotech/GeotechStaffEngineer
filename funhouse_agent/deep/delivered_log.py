"""Record each tool result AS THE MODEL RECEIVED IT.

Live smoke wave 1 (geotech questions, G11): ``activity.jsonl`` logs a tool's
own output -- the ``on_tool_end`` callback fires inside ``tool.run``, BEFORE
the agent's middleware has touched the result. So every empty ``grep`` was
logged as "No matches found" although the model was also told what that
does NOT mean (``ScratchFilesystemGuard``), a tool call the graph refused
(an unknown tool name, DD-2) was missing from the log altogether, and the
guards' interceptions were invisible. The owner's rule: when the record
cannot answer "why", fix the logging.

:class:`DeliveredToolResults` wraps every tool call, outside the app's own
tool middleware, and reports the message the model will see as a
``tool_delivered`` custom event (``webapp.activity_log`` writes it when it
differs from the tool's own output, and writes a refusal -- no tool ran --
as ``tool_refused``). It never changes the result, and never fails a call.
"""

from __future__ import annotations

from langchain_core.messages import ToolMessage

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:  # pragma: no cover - older layout
    from langchain.agents.middleware.types import AgentMiddleware

#: The custom-event name the activity log listens for.
EVENT = "tool_delivered"


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if isinstance(b.get("text"), str):
                    parts.append(b["text"])
                elif b.get("type") in ("image", "image_url"):
                    parts.append("[image]")
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _messages_of(result) -> list:
    """The ToolMessages a tool call's result puts in the conversation."""
    if isinstance(result, ToolMessage):
        return [result]
    update = getattr(result, "update", None)            # a Command
    if isinstance(update, dict):
        msgs = update.get("messages")
        if isinstance(msgs, (list, tuple)):
            return [m for m in msgs if isinstance(m, ToolMessage)]
    if isinstance(result, (list, tuple)):
        out = []
        for r in result:
            out.extend(_messages_of(r))
        return out
    return []


def _report(request, result) -> None:
    try:
        from langchain_core.callbacks.manager import dispatch_custom_event
        call = getattr(request, "tool_call", None) or {}
        call_id = call.get("id")
        refused = getattr(request, "tool", None) is None
        for msg in _messages_of(result):
            if call_id and getattr(msg, "tool_call_id", None) not in (
                    None, "", call_id):
                continue
            dispatch_custom_event(EVENT, {
                "tool_call_id": call_id,
                "name": call.get("name"),
                "content": _text(msg.content),
                "status": getattr(msg, "status", None),
                "refused": refused,
            })
    except Exception:  # noqa: BLE001 - logging must never cost the call
        pass


class DeliveredToolResults(AgentMiddleware):
    """Report each tool result as delivered to the model (see module doc).
    Put it FIRST among the app's tool middleware, so it sees what the others
    made of the result."""

    def wrap_tool_call(self, request, handler):
        result = handler(request)
        _report(request, result)
        return result

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        _report(request, result)
        return result


__all__ = ["DeliveredToolResults", "EVENT"]

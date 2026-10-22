"""A LangChain chat model that replays canned replies, for
:class:`~report_ingest.engine.ChatModelEngine` (BACKLOG RI-03).

It stands where the app's own chat model stands -- on Foundry,
``webapp.palantir_sdk_engine``'s classes -- and is as strict as the
providers behind those are about the shape of a conversation: a request in
which a tool call has no result, or a result answers no call, is refused
with an AssertionError, because that is a 400 on every real route.

Every request is recorded in :attr:`ScriptedChatModel.requests` -- the
LangChain messages, the tools bound and the output cap the model carried --
on a list every copy of the model shares, so a copy made by ``bind_tools``
or by the engine's per-call cap still records into the same list.

:func:`from_turns` turns a :class:`~report_ingest.tests.fake_engine.FakeEngine`
script into replies, a structured answer becoming a call to the answer
tool, so a whole ingest can run through the engine on the scripts the
pipeline's own tests use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from report_ingest.engine import ANSWER_TOOL


class ScriptExhausted(AssertionError):
    """The engine asked for more replies than the script has."""


def check_conversation(messages: Sequence[BaseMessage]) -> None:
    """Refuse a conversation a provider would refuse: every tool call
    answered straight after its turn, every result answering a call."""
    i = 0
    while i < len(messages):
        message = messages[i]
        i += 1
        if isinstance(message, ToolMessage):
            raise AssertionError(
                f"a tool result answers no call: {message.tool_call_id!r}")
        if not isinstance(message, AIMessage):
            continue
        # ChatOpenAI sends a call whose arguments did not parse back as a
        # call, so it needs a result as much as any other.
        sent = {c["id"] for c in list(message.tool_calls or [])
                + list(getattr(message, "invalid_tool_calls", None) or [])}
        while i < len(messages) and isinstance(messages[i], ToolMessage):
            if messages[i].tool_call_id not in sent:
                raise AssertionError(
                    f"a tool result answers no call: "
                    f"{messages[i].tool_call_id!r}")
            sent.discard(messages[i].tool_call_id)
            i += 1
        if sent:
            raise AssertionError(f"tool calls with no result: {sorted(sent)}")


class ScriptedChatModel(BaseChatModel):
    """Replays :attr:`replies` in order and records every request."""

    replies: List[Any] = Field(default_factory=list)
    requests: List[Dict[str, Any]] = Field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None
    max_tokens: Optional[int] = 32000
    model_name: str = "scripted-model"

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        return self.model_copy(update={
            "tools": [convert_to_openai_tool(t) for t in tools]})

    def _generate(self, messages: List[BaseMessage], stop: Any = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        check_conversation(messages)
        self.requests.append({
            "messages": list(messages),
            "tools": [t["function"]["name"] for t in (self.tools or [])],
            "specs": list(self.tools or []),
            "max_tokens": self.max_tokens,
        })
        if not self.replies:
            raise ScriptExhausted(
                f"request {len(self.requests)} but the script is used up")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])


def answer_call(args: Dict[str, Any], call_id: str = "call_answer",
                **meta: Any) -> AIMessage:
    """A reply that answers through the answer tool."""
    return AIMessage(
        content="", tool_calls=[{"name": ANSWER_TOOL, "args": dict(args),
                                 "id": call_id, "type": "tool_call"}],
        response_metadata={"finish_reason": "tool_calls",
                           "model_name": "scripted-model", **meta},
        usage_metadata={"input_tokens": 1000, "output_tokens": 100,
                        "total_tokens": 1100})


def from_turns(turns: Sequence[Dict[str, Any]]) -> List[AIMessage]:
    """A FakeEngine script as chat-model replies."""
    out: List[AIMessage] = []
    for n, turn in enumerate(turns):
        usage = {"input_tokens": 1000, "output_tokens": 100,
                 "total_tokens": 1100}
        meta = {"model_name": "scripted-model"}
        if "tools" in turn:
            calls = [{"name": name, "args": dict(args),
                      "id": f"call_{n}_{i}", "type": "tool_call"}
                     for i, (name, args) in enumerate(turn["tools"])]
            out.append(AIMessage(content=turn.get("text", ""),
                                 tool_calls=calls,
                                 response_metadata={
                                     **meta, "finish_reason": "tool_calls"},
                                 usage_metadata=usage))
        elif turn.get("final") is not None:
            out.append(answer_call(turn["final"].model_dump(mode="json"),
                                   call_id=f"call_{n}_answer"))
        else:
            out.append(AIMessage(
                content=turn.get("text", ""),
                response_metadata={
                    **meta, "finish_reason": turn.get("stop_reason", "stop")},
                usage_metadata=usage))
    return out

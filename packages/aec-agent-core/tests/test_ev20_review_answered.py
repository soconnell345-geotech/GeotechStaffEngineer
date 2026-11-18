"""Review of BACKLOG EV-20 (``calls_not_started`` on ``turn_end``): a call
that was ANSWERED without its tool running (a guard's refusal, an unknown
tool name) is told apart from one that was cut off, and a turn's refusals
never push the cut-off call past the cap. Made-up tools and replies only;
no model is called.
"""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from aec_agent_core import activity_log
from aec_agent_core.activity_log import MAX_CALLS_NOT_STARTED, ActivityLogger
from aec_agent_core.deep.delivered_log import DeliveredToolResults

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:  # pragma: no cover - older layout
    from langchain.agents.middleware.types import AgentMiddleware


@tool
def made_up_lookup(term: str) -> str:
    """Look up a made-up term."""
    return "found " + term


class _Refuses(AgentMiddleware):
    """A made-up guard: a lookup of a term starting with ``no`` is answered
    without the tool running."""

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        if str(call["args"].get("term", "")).startswith("no"):
            return ToolMessage(content="Refused by a made-up guard.",
                               tool_call_id=call["id"], name=call["name"])
        return handler(request)


class _Script(FakeMessagesListChatModel):
    """Plays ``script``: one entry per model step, a list of
    ``(id, name, args)`` calls or the answer's text; the last repeats."""

    script: list = []

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        n = sum(1 for m in messages if getattr(m, "type", "") == "ai")
        step = self.script[min(n, len(self.script) - 1)]
        if isinstance(step, str):
            msg = AIMessage(content=step)
        else:
            msg = AIMessage(content="", tool_calls=[
                {"name": name, "args": args, "id": cid}
                for cid, name, args in step])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _turn(tmp_path, script, limit=25):
    from langchain.agents import create_agent
    agent = create_agent(_Script(responses=[], script=script),
                         tools=[made_up_lookup],
                         middleware=[DeliveredToolResults(), _Refuses()])
    log = ActivityLogger(str(tmp_path), turn=1)
    error = None
    try:
        agent.invoke({"messages": [HumanMessage(content="made-up")]},
                     config={"callbacks": [log], "recursion_limit": limit})
    except Exception as exc:                           # noqa: BLE001
        error = type(exc).__name__
    log.turn_end(turn_tokens=0, error=error)
    (end,) = [r for r in activity_log.load(str(tmp_path))
              if r.get("event") == "turn_end"]
    return end, error


def test_a_refused_call_on_a_finished_turn_is_marked_answered(tmp_path):
    end, error = _turn(tmp_path, [
        [("a", "made_up_lookup", {"term": "x"}),
         ("b", "made_up_lookup", {"term": "no_entry"}),
         ("c", "no_such_tool", {"q": 1})],
        "A made-up answer."])
    assert error is None
    assert end["calls_not_started"] == [
        {"agent": "primary", "name": "made_up_lookup",
         "args": {"term": "no_entry"}, "answered": True},
        {"agent": "primary", "name": "no_such_tool", "args": {"q": 1},
         "answered": True}]
    assert end["n_calls_not_started"] == 2
    # nothing was cut off on this turn, and the record says so
    assert end["n_calls_not_answered"] == 0


def test_the_cut_off_call_is_listed_ahead_of_the_refusals(tmp_path):
    n = MAX_CALLS_NOT_STARTED + 5
    script = [[(f"r{i}", "made_up_lookup", {"term": f"no{i}"})]
              for i in range(n)]
    script.append([("cut", "made_up_lookup", {"term": "the_cut_off_one"})])
    end, error = _turn(tmp_path, script, limit=2 * n + 1)
    assert error == "GraphRecursionError"
    listed = end["calls_not_started"]
    assert len(listed) == MAX_CALLS_NOT_STARTED
    # the call nothing answered comes first and carries no mark
    assert listed[0] == {"agent": "primary", "name": "made_up_lookup",
                         "args": {"term": "the_cut_off_one"}}
    assert all(c.get("answered") is True for c in listed[1:])
    assert end["n_calls_not_started"] == n + 1
    assert end["n_calls_not_answered"] == 1


def test_an_id_asked_again_after_a_refusal_is_not_marked_from_before(
        tmp_path):
    # a client that numbers its calls per step: the same id, first refused,
    # then asked again and cut off
    end, error = _turn(tmp_path, [
        [("call_0", "made_up_lookup", {"term": "no_entry"})],
        [("call_0", "made_up_lookup", {"term": "second"})]], limit=3)
    assert error == "GraphRecursionError"
    assert end["calls_not_started"] == [
        {"agent": "primary", "name": "made_up_lookup",
         "args": {"term": "second"}}]
    assert end["n_calls_not_answered"] == 1

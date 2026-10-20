"""A helper on a model of its own (BACKLOG INF-41): ``build_deep_agent(
subagent_models={name: model})``. Offline, with scripted chat models."""

import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel)
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

import funhouse_agent.deep.agent as agent_mod  # noqa: E402
from funhouse_agent.deep.agent import build_deep_agent  # noqa: E402


class _Scripted(FakeMessagesListChatModel):
    """Answers with ``script`` in turn, then ``reply``; keeps what it was
    asked."""

    script: list = []
    reply: str = "done"
    asked: list = []

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.asked.append([getattr(m, "content", "") for m in messages])
        msg = self.script.pop(0) if self.script else AIMessage(self.reply)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _model(script=(), reply="done"):
    return _Scripted(responses=[AIMessage("x")], script=list(script),
                     reply=reply, asked=[])


def _ask_references():
    return AIMessage(content="", tool_calls=[{
        "name": "task", "id": "call_1",
        "args": {"description": "Find Nq at phi' 30 in DM7.",
                 "subagent_type": "references"}}])


def test_a_named_helper_runs_on_its_own_model():
    main = _model([_ask_references()])
    sonnet = _model(reply="Nq is 18.4 (DM7.2 Table 4-3).")
    agent = build_deep_agent(main, subagent_models={"references": sonnet})
    out = agent.invoke(
        {"messages": [{"role": "user", "content": "Ask the librarian."}]},
        config={"recursion_limit": 40})
    assert len(sonnet.asked) == 1
    assert any("Find Nq" in str(c) for c in sonnet.asked[0])
    assert len(main.asked) == 2              # its own two turns, not the helper's
    assert "Nq is 18.4" in str(out["messages"][-2].content)
    assert set(agent.geotech_build["helper_models"]) == {"references"}


def test_only_the_named_helper_gets_a_model(monkeypatch):
    seen = {}
    real = agent_mod.create_deep_agent

    def spy(**kw):
        seen.update(kw)
        return real(**kw)

    monkeypatch.setattr(agent_mod, "create_deep_agent", spy)
    main, sonnet = _model(), _model()
    agent = build_deep_agent(main, enable_calc_subagent=True,
                             subagent_models={"references": sonnet,
                                              "no-such-helper": _model()})
    specs = {s["name"]: s for s in seen["subagents"]}
    assert specs["references"]["model"] is sonnet
    assert {"calc", "reviewer", "general-purpose"} <= set(specs)
    assert [n for n, s in specs.items() if "model" in s] == ["references"]
    assert set(agent.geotech_build["helper_models"]) == {"references"}


def test_without_a_map_every_helper_runs_on_the_main_model(monkeypatch):
    seen = {}
    real = agent_mod.create_deep_agent

    def spy(**kw):
        seen.update(kw)
        return real(**kw)

    monkeypatch.setattr(agent_mod, "create_deep_agent", spy)
    agent = build_deep_agent(_model(), enable_calc_subagent=True)
    assert not any("model" in s for s in seen["subagents"])
    assert "helper_models" not in agent.geotech_build

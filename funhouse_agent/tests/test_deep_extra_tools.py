"""build_deep_agent(extra_tools=...) — host-injected tools reach the agent."""

import pytest

pytest.importorskip("deepagents")

import funhouse_agent.deep.agent as da


def test_extra_tools_appended_to_primary_tools(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)

        class _A:                     # attribute-assignable stand-in
            pass
        return _A()

    monkeypatch.setattr(da, "create_deep_agent", fake_create_deep_agent)
    sentinel = object()
    da.build_deep_agent("provider:model", extra_tools=[sentinel],
                        reference_mode="off")
    assert sentinel in captured["tools"]
    assert captured["tools"].index(sentinel) == len(captured["tools"]) - 1


def test_no_extra_tools_changes_nothing(monkeypatch):
    counts = []

    def fake_create_deep_agent(**kwargs):
        counts.append(len(kwargs["tools"]))

        class _A:
            pass
        return _A()

    monkeypatch.setattr(da, "create_deep_agent", fake_create_deep_agent)
    da.build_deep_agent("provider:model", reference_mode="off")
    da.build_deep_agent("provider:model", extra_tools=[], reference_mode="off")
    assert counts[0] == counts[1]

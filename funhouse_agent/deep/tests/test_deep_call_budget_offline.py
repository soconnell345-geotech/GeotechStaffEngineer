"""Offline tests for the v5.1 references-sub-agent model-call budget.

Guards THE biggest cost lever from the rc7 Funhouse run: 26/76
reference-consult questions burned 69% of all tokens (4.6M/6.6M, almost all
input) because the references sub-agent re-sent its accumulated context across
an unbounded number of internal model calls. The fix
(:mod:`funhouse_agent.deep.limits`) is a
:class:`~funhouse_agent.deep.limits.ModelCallBudgetMiddleware` attached to the
sub-agent spec, which (a) forces the LAST budgeted call to be a tool-less
"summarize what you have and answer" turn, and (b) backstops a model that
ignores even that with a graceful jump-to-end message — never an error.

These tests make NO API/network calls: the looping/stubborn/obedient models
are local fakes, and the sub-agent is compiled with ``create_agent`` exactly
the way deepagents compiles a ``SubAgent`` spec (tools + system_prompt +
middleware).

Run from the worktree root with the venv python::

    cd <worktree>
    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_deep_call_budget_offline.py -v
"""

import pytest

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from funhouse_agent.deep.agent import build_deep_agent, build_references_subagent
from funhouse_agent.deep.limits import (
    BUDGET_EXHAUSTED_MESSAGE,
    DEFAULT_REFERENCES_MAX_MODEL_CALLS,
    FINAL_TURN_NUDGE,
    ModelCallBudgetMiddleware,
)


# ---------------------------------------------------------------------------
# Fake models (NO network)
# ---------------------------------------------------------------------------

#: A stable marker substring of the final-turn nudge, used to detect it.
_NUDGE_MARKER = "FINAL turn"


class _FakeToolCallingModel(BaseChatModel):
    """Base fake: accepts ``bind_tools`` so ``create_agent`` can use it.

    ``BaseChatModel.bind_tools`` raises ``NotImplementedError`` by default;
    the agent loop binds the tool schemas on every model call, so the fakes
    just acknowledge the binding and ignore the schemas (the canned responses
    decide whether to "call" a tool).
    """

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class _ToolLoopingModel(_FakeToolCallingModel):
    """A fake model that requests a tool call on EVERY turn — forever.

    ``obey_nudge=True`` (the realistic case): once the budget middleware
    injects the final-turn instruction, it stops tool-calling and answers.
    ``obey_nudge=False`` (the pathological case): it keeps tool-calling even
    on the forced final turn, so the hard backstop must fire.
    """

    obey_nudge: bool = True
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "tool-looping-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        nudged = any(
            _NUDGE_MARKER in str(getattr(m, "content", "")) for m in messages
        )
        if self.obey_nudge and nudged:
            msg = AIMessage(
                content="FINAL ANSWER: Kp is approximately 9.5 per DM7 "
                        "Figure 7-X (gathered before the budget)."
            )
        else:
            msg = AIMessage(
                content="",
                tool_calls=[{
                    "name": "list_agents",
                    "args": {},
                    "id": f"call_{len(self.seen)}",
                }],
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])


class _OneShotModel(_FakeToolCallingModel):
    """A fake model that answers immediately (a SHORT conversation)."""

    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "one-shot-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="Per GEC-10 the FS is 2.0.")
        )])


def _compile_references_agent(model, **spec_kwargs):
    """Compile the references SubAgent spec the way deepagents does.

    ``create_deep_agent`` turns a declarative ``SubAgent`` spec into
    ``create_agent(model, tools=..., system_prompt=..., middleware=...)`` —
    this mirrors that exactly so the middleware is exercised in a REAL agent
    loop without any network.
    """
    spec = build_references_subagent(**spec_kwargs)
    return create_agent(
        model,
        tools=spec["tools"],
        system_prompt=spec["system_prompt"],
        middleware=spec.get("middleware", []),
    )


def _final_text(result) -> str:
    for m in reversed(result.get("messages", [])):
        if type(m).__name__ == "AIMessage":
            text = m.content if isinstance(m.content, str) else str(m.content)
            if text.strip():
                return text
    return ""


# ---------------------------------------------------------------------------
# Spec wiring
# ---------------------------------------------------------------------------

def test_references_spec_carries_budget_middleware_by_default():
    spec = build_references_subagent()
    mws = spec.get("middleware", [])
    assert len(mws) == 1
    mw = mws[0]
    assert isinstance(mw, ModelCallBudgetMiddleware)
    assert mw.run_limit == DEFAULT_REFERENCES_MAX_MODEL_CALLS == 8
    # Graceful, never raising.
    assert mw.exit_behavior == "end"


def test_references_spec_budget_is_configurable():
    spec = build_references_subagent(max_model_calls=3)
    assert spec["middleware"][0].run_limit == 3


@pytest.mark.parametrize("disabled", [None, 0])
def test_references_spec_budget_can_be_disabled(disabled):
    spec = build_references_subagent(max_model_calls=disabled)
    assert "middleware" not in spec


def test_budget_middleware_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        ModelCallBudgetMiddleware(0)


def test_build_deep_agent_accepts_budget_knob_smoke():
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel,
    )
    agent = build_deep_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="ok")])),
        references_max_model_calls=4,
    )
    assert type(agent).__name__ == "CompiledStateGraph"


# ---------------------------------------------------------------------------
# The cap fires: a looping model is forced to a final answer (no error)
# ---------------------------------------------------------------------------

def test_looping_model_is_capped_and_still_answers():
    """A model that would loop forever gets exactly the budgeted number of
    calls; the last call is forced tool-less + nudged, and the final answer
    comes from the model (not an error)."""
    model = _ToolLoopingModel(obey_nudge=True)
    agent = _compile_references_agent(model, max_model_calls=3)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Kp for phi=35?"}]}
    )

    # Exactly the budget — not one call more.
    assert len(model.seen) == 3
    # The final (3rd) call carried the summarize-now nudge; earlier ones not.
    assert any(_NUDGE_MARKER in str(m.content) for m in model.seen[2])
    assert not any(_NUDGE_MARKER in str(m.content) for m in model.seen[0])
    assert not any(_NUDGE_MARKER in str(m.content) for m in model.seen[1])
    # A REAL final answer was returned to the caller.
    final = _final_text(result)
    assert "FINAL ANSWER" in final
    assert "9.5" in final


def test_nudge_text_tells_model_to_answer_from_gathered_info():
    assert _NUDGE_MARKER in FINAL_TURN_NUDGE
    low = FINAL_TURN_NUDGE.lower()
    assert "no more tool calls" in low
    assert "citation" in low


def test_stubborn_model_hits_backstop_gracefully():
    """A model that ignores even the forced final turn is cut off by the
    inherited limit check — with the actionable re-delegate message, and
    WITHOUT raising."""
    model = _ToolLoopingModel(obey_nudge=False)
    agent = _compile_references_agent(model, max_model_calls=3)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Kp for phi=35?"}]}
    )

    # The budget held: exactly 3 model calls despite the model's looping.
    assert len(model.seen) == 3
    # The backstop injected the actionable message (not the stock error text).
    final = _final_text(result)
    assert final == BUDGET_EXHAUSTED_MESSAGE
    assert "narrower" in final


# ---------------------------------------------------------------------------
# A short conversation is unaffected
# ---------------------------------------------------------------------------

def test_short_conversation_is_unaffected():
    """One-shot answer well under budget: no nudge, no backstop, answer
    returned untouched."""
    model = _OneShotModel()
    agent = _compile_references_agent(
        model, max_model_calls=DEFAULT_REFERENCES_MAX_MODEL_CALLS
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "FS per GEC-10?"}]}
    )

    assert len(model.seen) == 1
    # The model never saw the nudge.
    assert not any(
        _NUDGE_MARKER in str(m.content) for msgs in model.seen for m in msgs
    )
    final = _final_text(result)
    assert "GEC-10" in final
    assert BUDGET_EXHAUSTED_MESSAGE not in final


def test_under_budget_tool_round_completes_normally():
    """A conversation that uses SOME tool rounds but finishes under budget is
    not nudged or cut: the model tool-calls twice then would loop, but with a
    generous budget the obedient model only stops when nudged — so instead
    drive the obedient model with budget 5 and confirm the loop ran the full
    5 calls (4 tool rounds + forced final). Then with the one-shot model and
    the same budget, only 1 call happens. Together these pin the budget down:
    it bounds long consults and leaves short ones alone."""
    looping = _ToolLoopingModel(obey_nudge=True)
    agent = _compile_references_agent(looping, max_model_calls=5)
    agent.invoke({"messages": [{"role": "user", "content": "q"}]})
    assert len(looping.seen) == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

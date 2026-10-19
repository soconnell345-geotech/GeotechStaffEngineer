"""The calc helper knows its call budget and is warned near its end
(BACKLOG GM-101).

Live wave w13: both calc helpers ran out of tool rounds -- one before
``html_to_pdf``, one before its PDF -- and the main agent built the PDF; in
F55j it went out unchecked. The helper's prompt never stated its budget (the
references helper's does), and nothing warned it as the end came near.

No network: the looping model is a local fake, compiled with
``create_agent`` the way deepagents compiles a sub-agent spec.
"""

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from funhouse_agent.deep import agent as agent_mod
from funhouse_agent.deep.limits import (LOW_BUDGET_NOTE,
                                        ModelCallBudgetMiddleware)
from funhouse_agent.deep.prompt import build_domain_prompt

_WARN = "[Model-call budget]"
_FINAL = "FINAL turn"


class _Looping(BaseChatModel):
    """Calls a tool on every turn until told it is the final one."""

    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "looping-fake"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        if any(_FINAL in str(m.content) for m in messages):
            msg = AIMessage(content="done")
        else:
            msg = AIMessage(content="", tool_calls=[{
                "name": "list_agents", "args": {},
                "id": f"c{len(self.seen)}"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _notes(messages):
    return [str(m.content) for m in messages
            if isinstance(m, HumanMessage) and _WARN in str(m.content)]


def _run(budget, warn):
    from funhouse_agent.deep.tools import make_core_tools
    model = _Looping()
    agent = create_agent(model, tools=make_core_tools(),
                         system_prompt="calc",
                         middleware=[ModelCallBudgetMiddleware(
                             budget, warn_remaining=warn)])
    agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    return model.seen


def test_from_four_calls_left_each_call_says_how_many_are_left():
    seen = _run(budget=7, warn=4)
    assert len(seen) == 7
    # calls 1-3: no note; calls 4-6: 4, 3 and 2 left; call 7 is the final
    # turn, which says so itself.
    assert [len(_notes(m)) for m in seen] == [0, 0, 0, 1, 1, 1, 0]
    assert [_notes(m)[0].split()[2] for m in seen[3:6]] == ["4", "3", "2"]
    assert any(_FINAL in str(m.content) for m in seen[6])
    # The note is shown on its call only, never kept in the conversation.
    assert len(_notes(seen[5])) == 1


def test_no_warning_unless_asked():
    seen = _run(budget=5, warn=None)
    assert not any(_notes(m) for m in seen)


def test_the_boundaries():
    # 1 would warn only on the final turn, which says so itself: no note.
    assert [len(_notes(m)) for m in _run(budget=5, warn=1)] == [0] * 5
    assert [len(_notes(m)) for m in _run(budget=5, warn=2)] ==         [0, 0, 0, 1, 0]
    assert [len(_notes(m)) for m in _run(budget=1, warn=4)] == [0]


def test_a_small_budget_warns_from_the_start():
    seen = _run(budget=3, warn=4)
    assert [len(_notes(m)) for m in seen] == [1, 1, 0]


def test_the_note_counts_the_call_it_is_on():
    note = LOW_BUDGET_NOTE.format(left=3)
    assert "3 model calls left, this one included" in note
    assert "last is your reply, with no tools" in note


def test_the_calc_helper_is_told_its_budget_and_warned():
    spec = agent_mod.build_calc_subagent()
    n = agent_mod.DEFAULT_CALC_MAX_MODEL_CALLS
    assert f"you have {n} model calls for this task" in spec["system_prompt"]
    assert "with calls to spare" in spec["system_prompt"]
    (mw,) = spec["middleware"]
    assert mw.run_limit == n
    assert mw.warn_remaining == agent_mod.CALC_WARN_REMAINING == 4
    spec = agent_mod.build_calc_subagent(max_model_calls=9)
    assert "you have 9 model calls" in spec["system_prompt"]


def test_no_budget_no_budget_line():
    spec = agent_mod.build_calc_subagent(max_model_calls=0)
    assert "CALL BUDGET" not in spec["system_prompt"]
    assert "middleware" not in spec


def test_the_other_budgets_are_unchanged():
    refs = agent_mod.build_references_subagent()
    assert refs["middleware"][0].warn_remaining == 0
    rev = agent_mod.build_reviewer_subagent()
    assert rev["middleware"][0].warn_remaining == 0


def test_a_pdf_the_main_agent_builds_is_read_back_and_looked_at():
    prompt = build_domain_prompt()
    assert ("A PDF you build\n  yourself you read back (`read_pdf_text`) "
            "and look at (`analyze_pdf_page` on\n  each page with a figure) "
            "before you hand it over.") in prompt

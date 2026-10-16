"""The agent states its judgment calls (BACKLOG GM-116, behind
``GEOTECH_JUDGMENT_CALLS``).

The owner, 2026-10-16: "We should also make sure the agent is communicating
its engineering judgements to the user. User's judgement is the one that
matters." With the switch on, the geotech agent's prompt says so and asks an
answer to end with its judgment calls, the ``calc`` helper reports the calls
it made, and the references helper returns a reference's several methods or
values as cited options. With it off the build is what it was: every prompt,
helper prompt and tool description.
"""

import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from funhouse_agent import review_flags  # noqa: E402
from funhouse_agent.deep import agent as agent_mod  # noqa: E402
from funhouse_agent.deep.limits import (  # noqa: E402
    DEFAULT_REFERENCES_MAX_MODEL_CALLS)
from funhouse_agent.deep.prompt import (  # noqa: E402
    build_domain_prompt, judgment_calls_section)
from funhouse_agent.dispatch import ANALYSIS_MODULES  # noqa: E402
from funhouse_agent.reviewer import (  # noqa: E402
    CONSULTANT_FRAMING, REVIEWER_DEEP_PROMPT)


class _Quiet(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "quiet"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="done"))])


def _no_switches(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)


def _built(monkeypatch, on: bool, env=None, **kw):
    """Build the agent with the switch on or off (and any other switches
    in ``env``); return what create_deep_agent was handed and the compiled
    agent."""
    _no_switches(monkeypatch)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    if on:
        monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "1")
    captured = {}
    real = agent_mod.create_deep_agent

    def spy(**k):
        captured.update(k)
        return real(**k)
    monkeypatch.setattr(agent_mod, "create_deep_agent", spy)
    agent = agent_mod.build_deep_agent(model=_Quiet(), **kw)
    monkeypatch.setattr(agent_mod, "create_deep_agent", real)
    return captured, agent


def _spec(captured, name):
    return next(s for s in captured["subagents"] if s.get("name") == name)


def _references_prompt(options: bool) -> str:
    return (CONSULTANT_FRAMING + agent_mod._REFERENCES_CONCISION
            + (agent_mod._REFERENCES_OPTIONS if options else "")
            + agent_mod._REFERENCES_BUDGET.format(
                n=DEFAULT_REFERENCES_MAX_MODEL_CALLS))


def _surface(captured, agent) -> dict:
    """Every tool and helper description the model is shown: the main
    agent's tools (``task`` included, which lists the helpers), each
    helper's description and each helper's tools."""
    out = {}
    for name, t in agent.nodes["tools"].bound.tools_by_name.items():
        out[("main", name)] = (t.description,
                               t.tool_call_schema.model_json_schema())
    for s in captured["subagents"]:
        out[("helper", s.get("name"))] = s.get("description")
        for t in s.get("tools") or []:
            name = getattr(t, "name", None)
            if name:
                out[(s.get("name"), name)] = t.description
    return out


# ---------------------------------------------------------------------------
# the switch and its arm
# ---------------------------------------------------------------------------

def test_the_switch_is_off_by_default_and_has_an_arm(monkeypatch):
    _no_switches(monkeypatch)
    assert review_flags.judgment_calls() is False
    assert review_flags.JUDGMENT_CALLS_ENV == "GEOTECH_JUDGMENT_CALLS"
    assert review_flags.JUDGMENT_CALLS_ENV in review_flags.ALL_ENVS
    arm = review_flags.ARMS["judgment_calls"]
    assert arm == {review_flags.JUDGMENT_CALLS_ENV: "1"}
    with review_flags.switches(arm):
        assert review_flags.judgment_calls() is True
        assert "GEOTECH_JUDGMENT_CALLS=1" in review_flags.describe()
    assert review_flags.judgment_calls() is False
    for value in ("1", "true", "on", "yes"):
        monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, value)
        assert review_flags.judgment_calls() is True
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "0")
    assert review_flags.judgment_calls() is False


# ---------------------------------------------------------------------------
# the switch off: the build as it was
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("calc", [False, True])
@pytest.mark.parametrize("reference_mode", ["anytime", "off"])
def test_switch_off_leaves_every_prompt_as_it_was(monkeypatch, calc,
                                                  reference_mode):
    captured, _ = _built(monkeypatch, False, reference_mode=reference_mode,
                         enable_calc_subagent=calc)
    want = build_domain_prompt(ANALYSIS_MODULES, memory_enabled=False)
    if calc:
        want += "\n\n" + agent_mod._CALC_DELEGATION_NUDGE
    assert captured["system_prompt"] == want
    names = {s.get("name") for s in captured["subagents"]}
    if reference_mode == "anytime":
        assert _spec(captured, "references")["system_prompt"] == \
            _references_prompt(False)
        assert _spec(captured, "reviewer")["system_prompt"] == \
            REVIEWER_DEEP_PROMPT
    else:
        assert not {"references", "reviewer"} & names
    if calc:
        assert _spec(captured, "calc")["system_prompt"] == (
            agent_mod._CALC_PREAMBLE + agent_mod._CALC_FRAMING)
    for s in captured["subagents"]:
        assert "judgment call" not in (s.get("system_prompt") or "").lower()


def test_the_switch_changes_no_tool_or_helper_description(monkeypatch):
    """Only prompts change: every tool the main agent and each helper are
    shown (``task``'s list of helpers included) is the same on and off."""
    off = _surface(*_built(monkeypatch, False, enable_calc_subagent=True))
    on = _surface(*_built(monkeypatch, True, enable_calc_subagent=True))
    assert on == off
    assert ("main", "task") in off and ("helper", "calc") in off


# ---------------------------------------------------------------------------
# the switch on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("calc", [False, True])
def test_switch_on_ends_the_geotech_prompt_with_the_judgment_calls(
        monkeypatch, calc):
    captured, agent = _built(monkeypatch, True, enable_calc_subagent=calc,
                             extra_system_prompt="HOST BLOCK")
    want = (build_domain_prompt(ANALYSIS_MODULES, memory_enabled=False)
            + "\n\n" + judgment_calls_section(references=True)
            + "\n\nHOST BLOCK")
    if calc:
        want += "\n\n" + agent_mod._CALC_DELEGATION_NUDGE
    assert captured["system_prompt"] == want
    assert "GEOTECH_JUDGMENT_CALLS=1" in agent.geotech_build["switches"]


def test_switch_on_calc_helper_reports_its_calls(monkeypatch):
    off, _ = _built(monkeypatch, False, enable_calc_subagent=True)
    on, _ = _built(monkeypatch, True, enable_calc_subagent=True)
    assert _spec(on, "calc")["system_prompt"] == (
        _spec(off, "calc")["system_prompt"] + agent_mod._CALC_JUDGMENT_CALLS)
    text = agent_mod._CALC_JUDGMENT_CALLS
    for part in ("the user's", '"Judgment calls"', "why", "reference",
                 "alternative", "what would change the result"):
        assert part in text, part


def test_switch_on_references_return_cited_options(monkeypatch):
    captured, _ = _built(monkeypatch, True)
    assert _spec(captured, "references")["system_prompt"] == \
        _references_prompt(True)
    opts = agent_mod._REFERENCES_OPTIONS
    assert "cited option" in opts and "when it applies" in opts
    assert "say which fits the question as asked" in opts
    assert "only what was asked" in opts


def test_switch_on_leaves_the_reviewer_and_general_helpers(monkeypatch):
    off, _ = _built(monkeypatch, False, enable_calc_subagent=True)
    on, _ = _built(monkeypatch, True, enable_calc_subagent=True)
    for name in ("reviewer", "general-purpose"):
        assert _spec(on, name)["system_prompt"] == \
            _spec(off, name)["system_prompt"], name


def test_without_the_references_helper_the_block_does_not_name_it(
        monkeypatch):
    captured, _ = _built(monkeypatch, True, reference_mode="off")
    assert captured["system_prompt"].endswith(
        "\n\n" + judgment_calls_section(references=False))
    assert "`references` helper" not in captured["system_prompt"]
    assert "`references` helper" in judgment_calls_section(references=True)


def test_the_figure_helper_is_unchanged(monkeypatch):
    from funhouse_agent.deep.figure_helper import FIGURE_HELPER_NAME

    def figure(on):
        captured, _ = _built(monkeypatch, on,
                             env={review_flags.FIGURE_HELPER_ENV: "1"})
        spec = _spec(captured, FIGURE_HELPER_NAME)
        return (spec["system_prompt"], spec["description"],
                [(t.name, t.description) for t in spec["tools"]])
    assert figure(True) == figure(False)


def test_the_document_review_page_is_unchanged(monkeypatch):
    from webapp.profiles import DOCUMENT_REVIEW

    def prompts(on):
        captured, agent = _built(monkeypatch, on,
                                 **DOCUMENT_REVIEW.build_kwargs())
        return (captured["system_prompt"],
                {s.get("name"): s.get("system_prompt")
                 for s in captured["subagents"]},
                _surface(captured, agent))
    off, on = prompts(False), prompts(True)
    assert on == off
    assert "judgment call" not in off[0].lower()


# ---------------------------------------------------------------------------
# the text itself
# ---------------------------------------------------------------------------

def test_the_block_is_general_and_short():
    """No example from the incidents behind it (owner, 2026-10-16:
    geometry-specific judgment is the agent's and its references', never
    hard-coded), and it stays short."""
    block = judgment_calls_section(references=True)
    flat = " ".join(block.split())
    low = flat.lower()
    for word in ("slope", "circle", "circular", "krig", "weak layer",
                 "basal heave", "2.5", "1.5", "wrong tool"):
        assert word not in low, word
    for part in ("the user's", "as an experienced engineer would and give "
                 "your answer", "accept or override", "method or analysis",
                 "whether the default analysis suits the geometry and "
                 "loading", "factor of safety", "load factor", "drained or",
                 "which stratum governs", "search or mesh", "ambiguous data",
                 '"Judgment calls"', "why,", "alternative set aside",
                 "what would change the answer (a number only from a tool "
                 "run)", "inputs you assumed",
                 "An answer that rests on none needs no list."):
        assert part in flat, part
    assert len(block) < 1000
    assert len(agent_mod._CALC_JUDGMENT_CALLS) < 600
    assert len(agent_mod._REFERENCES_OPTIONS) < 250

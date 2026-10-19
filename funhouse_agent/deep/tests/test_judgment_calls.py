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
    captured, agent = _built(monkeypatch, False,
                             reference_mode=reference_mode,
                             enable_calc_subagent=calc)
    assert agent.geotech_judgment_consult is False      # GM-174 never acts
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
        # GM-101's budget line is unswitched (it states the budget).
        assert _spec(captured, "calc")["system_prompt"] == (
            agent_mod._CALC_PREAMBLE + agent_mod._CALC_FRAMING
            + agent_mod._CALC_BUDGET.format(
                n=agent_mod.DEFAULT_CALC_MAX_MODEL_CALLS))
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
    # the turn-end check (GM-174) acts on this build
    assert agent.geotech_judgment_consult is True


def test_switch_on_calc_helper_reports_its_calls(monkeypatch):
    off, _ = _built(monkeypatch, False, enable_calc_subagent=True)
    on, _ = _built(monkeypatch, True, enable_calc_subagent=True)
    assert _spec(on, "calc")["system_prompt"] == (
        _spec(off, "calc")["system_prompt"] + agent_mod._CALC_JUDGMENT_CALLS)
    text = agent_mod._CALC_JUDGMENT_CALLS
    for part in ("the user's", '"Judgment calls"', "why", "basis",
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
    captured, agent = _built(monkeypatch, True, reference_mode="off")
    assert captured["system_prompt"].endswith(
        "\n\n" + judgment_calls_section(references=False))
    # no helper to consult: the turn-end check (GM-174) never acts
    assert agent.geotech_judgment_consult is False
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
    _c, agent = _built(monkeypatch, True, **DOCUMENT_REVIEW.build_kwargs())
    assert agent.geotech_judgment_consult is False      # GM-174 never acts


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
                 '"Judgment calls"',
                 "why, citing its source by name and the section or page it "
                 "gives",
                 "alternative set aside", "what would change the answer",
                 "(a number only from a tool run)", "inputs you assumed",
                 "An answer that rests on no call needs no list."):
        assert part in flat, part
    # GM-155: 963 characters before; the consult as a step, a tool's
    # defaults and the one bounded run cost about 700 more (w13: 1,651;
    # the step came in by rewording, at 1,666 with "the section or page
    # it gives")
    assert len(block) < 1680
    assert len(judgment_calls_section(references=False)) < 1530
    assert len(agent_mod._CALC_JUDGMENT_CALLS) < 800
    assert len(agent_mod._REFERENCES_OPTIONS) < 400


# ---------------------------------------------------------------------------
# GM-155: grounded calls, given apart from assumed, numbers from tools
# ---------------------------------------------------------------------------

def test_the_consult_is_a_step_before_the_list():
    """Live wave w12: no ON answer cited a reference for any call. w13,
    with the consult offered as one source among three: the references
    helper was asked 0 times in 4 ON runs; F61j left a `choices` basis
    uncited and F51j said "found no reference" without looking. Now the
    consult is a step before the list, one task covering every call with
    no basis in hand (the cost lever); a basis is cited by name and the
    section or page it gives (no page made up); "no reference found" only
    where that consult found none."""
    flat = " ".join(judgment_calls_section(references=True).split())
    for part in ("Before the list, ask the `references` helper once, in one "
                 "task, about every call with no basis in a tool's "
                 "`choices` or a reference already looked up.",
                 "why, citing its source by name and the section or page "
                 "it gives (a `choices` basis or a reference looked up);",
                 '"rests on judgment, no reference found" only where that '
                 "consult found none;"):
        assert part in flat, part
    # the step comes before the list it feeds
    assert flat.index("Before the list") < flat.index('"Judgment calls"')
    assert "otherwise say plainly" not in flat
    alone = " ".join(judgment_calls_section(references=False).split())
    assert ("(a `choices` basis or a reference looked up); otherwise say "
            "plainly that it rests on judgment, with no reference cited;") \
        in alone
    assert "`references` helper" not in alone and "consult" not in alone
    calc = agent_mod._CALC_JUDGMENT_CALLS
    # the helper hands up the calls it has no basis for, for that consult
    assert ("the basis the tool's `choices` give, by name and the section "
            'or page it gives, else "no basis in hand"') in calc
    assert "rests on judgment" not in calc
    assert "several choices at once" in agent_mod._REFERENCES_OPTIONS


def test_a_tools_defaults_are_calls():
    """w12 F51j: K0, named only in the tool's notes, fell out of the ON
    list where the OFF answer had stated it."""
    flat = " ".join(judgment_calls_section(references=True).split())
    assert ("A tool's defaults are calls too: its `choices` not marked "
            "given, its `defaults_used` where the answer turns on them, and "
            "what its notes say it assumed.") in flat
    # only those that matter: the bearing tool has 9 defaults, the wall 13
    assert ("a tool's defaults and choices included where the result turns "
            "on them") in agent_mod._CALC_JUDGMENT_CALLS


def test_the_list_runs_one_alternative_and_drops_no_other():
    """w12 F61j ran fewer alternatives than OFF and said what would change
    the answer instead; F18 OFF and F18j quoted sensitivities no tool gave.
    One run for the call the answer turns on most, the rest by direction,
    and the run adds to the alternatives, never replaces them."""
    flat = " ".join(judgment_calls_section(references=True).split())
    for part in ("for the call the answer turns on most, run the "
                 "alternative (one tool call; no second helper task or "
                 "deliverable) and give its number",
                 "for the others, say which way the answer would move",
                 "That run adds to the alternatives you would run anyway, "
                 "never replaces them.",
                 "Leave out rounding and presentation."):
        assert part in flat, part
    assert "Leave out rounding and presentation." in \
        agent_mod._CALC_JUDGMENT_CALLS


@pytest.mark.parametrize("calc", [False, True])
def test_switch_off_carries_no_gm155_switched_text(monkeypatch, calc):
    """With the switch off, none of the switched GM-155 text reaches any
    prompt (the general fixes below are in both arms)."""
    captured, _ = _built(monkeypatch, False, enable_calc_subagent=calc)
    texts = [captured["system_prompt"]] + [
        s.get("system_prompt") or "" for s in captured["subagents"]]
    for t in texts:
        for part in ("A tool's defaults are calls", "no reference cited",
                     "Before the list, ask", "no basis in hand",
                     "the section or page it gives",
                     "several choices at once",
                     "turns on most"):
            assert part not in t, part


@pytest.mark.parametrize("on", [False, True])
def test_the_general_fixes_are_in_both_arms(monkeypatch, on):
    """GM-155's unswitched fixes, listed: the base prompt's confidence
    sentence asks for a tool-run number or a direction (both arms invented
    sensitivity percentages in w12); a brief keeps the user's inputs apart
    from the agent's assumptions, and so does the calc helper's
    deliverable (F55 OFF and F55j put an assumed concrete unit weight
    under the user's inputs; F18j's brief stated an assumed water table as
    found). w13: F18j's brief still put c = 0, which "sand" only implies,
    under GIVEN, and its calc helper wrote every assumption's basis as
    "stated by the engineer"."""
    from funhouse_agent.system_prompt import REACT_PREFIX
    captured, _ = _built(monkeypatch, on, enable_calc_subagent=True)
    main = " ".join(captured["system_prompt"].split())
    flat_prefix = " ".join(REACT_PREFIX.replace("\\\n", "").split())
    for part in ("Say how much an assumption moves a result only with a "
                 "number from a tool run or a reference looked up in this "
                 "conversation",
                 "say which way it moves the result, not by how much"):
        assert part in flat_prefix and part in main, part
    # no invitation to rerun in the OFF arm
    assert "rerun" not in flat_prefix
    assert ("Keep what the user gave apart from what you assumed: a value "
            "the user's words only imply is assumed. An assumed value goes "
            "in the brief under assumptions, with its engineering basis, "
            "never among the user's inputs") in main
    calc = " ".join(_spec(captured, "calc")["system_prompt"].split())
    # the rule, not the w13 cases the next wave measures (F18, F51)
    for t in (main, calc):
        for case in ("c = 0 for", "no water table where none"):
            assert case not in t, case
    for part in ("GIVEN OR ASSUMED: a deliverable keeps the inputs",
                 "A value the user's words only imply is assumed.",
                 "The delegation's assumptions and choices are the "
                 "delegating agent's, not the user's",
                 '"assumed, not given", never "stated by the engineer"',
                 "Never print an assumption as given.",
                 "takes `assumptions`"):
        assert part in calc, part

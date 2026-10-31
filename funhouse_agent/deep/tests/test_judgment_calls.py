"""The agent states its judgment calls (BACKLOG GM-116, behind
``GEOTECH_JUDGMENT_CALLS``, on by default since 2026-10-20: GM-226).

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
    DEFAULT_REFERENCES_MAX_MODEL_CALLS, JUDGMENT_REFERENCES_MAX_MODEL_CALLS,
    JUDGMENT_REFERENCES_WARN_REMAINING, ModelCallBudgetMiddleware)
from funhouse_agent.deep.prompt import (  # noqa: E402
    build_domain_prompt, judgment_calls_section)
from funhouse_agent.dispatch import ANALYSIS_MODULES  # noqa: E402
from funhouse_agent.reviewer import (  # noqa: E402
    CONSULTANT_ROLE, REVIEWER_DEEP_PROMPT)


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
    """Every switch off: unset, and those on by default set to 0 (GM-226,
    GM-227), so a test sees this switch's effect alone."""
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    for env in review_flags.DEFAULT_ON_ENVS:
        monkeypatch.setenv(env, "0")


def _built(monkeypatch, on: bool, env=None, **kw):
    """Build the agent with the switch on or off (and any other switches
    in ``env``); return what create_deep_agent was handed and the compiled
    agent."""
    _no_switches(monkeypatch)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "1" if on else "0")
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


def _references_prompt(options: bool, consult: bool = False) -> str:
    return (CONSULTANT_ROLE + agent_mod._REFERENCES_CONCISION
            + (agent_mod._REFERENCES_OPTIONS if options else "")
            + (agent_mod._REFERENCES_JUDGMENT_CALLS if consult else "")
            + agent_mod._REFERENCES_BUDGET.format(
                n=(JUDGMENT_REFERENCES_MAX_MODEL_CALLS if consult
                   else DEFAULT_REFERENCES_MAX_MODEL_CALLS)))


def _budget(spec):
    """A helper's model-call budget as (calls, warned from), or None."""
    for m in spec.get("middleware") or []:
        if isinstance(m, ModelCallBudgetMiddleware):
            return m.run_limit, m.warn_remaining
    return None


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

def test_the_switch_is_on_by_default_and_has_arms(monkeypatch):
    """GM-226 (owner, OWN-65, 2026-10-20): on unless ``0`` (or false, no,
    off); ``describe`` says so; the arms put it on and take it away."""
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    assert review_flags.judgment_calls() is True
    assert review_flags.JUDGMENT_CALLS_ENV == "GEOTECH_JUDGMENT_CALLS"
    assert review_flags.JUDGMENT_CALLS_ENV in review_flags.ALL_ENVS
    assert review_flags.JUDGMENT_CALLS_ENV in review_flags.DEFAULT_ON_ENVS
    assert "GEOTECH_JUDGMENT_CALLS=1 (default)" in review_flags.describe()
    arm = review_flags.ARMS["judgment_calls"]
    assert arm == {review_flags.JUDGMENT_CALLS_ENV: "1"}
    off = review_flags.ARMS["judgment_calls_off"]
    assert off == {review_flags.JUDGMENT_CALLS_ENV: "0"}
    with review_flags.switches(arm):
        assert review_flags.judgment_calls() is True
        assert "GEOTECH_JUDGMENT_CALLS=1" in review_flags.describe()
    with review_flags.switches(off):
        assert review_flags.judgment_calls() is False
        assert "GEOTECH_JUDGMENT_CALLS=0" in review_flags.describe()
    with review_flags.switches(review_flags.ARMS["baseline"]):
        assert review_flags.judgment_calls() is True
    assert review_flags.judgment_calls() is True
    for value in ("1", "true", "on", "yes", "", " "):
        monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, value)
        assert review_flags.judgment_calls() is True, value
    for value in ("0", "false", "no", "off", " OFF "):
        monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, value)
        assert review_flags.judgment_calls() is False, value


def test_only_the_three_ruled_switches_are_on_by_default(monkeypatch):
    """GM-226/GM-227: the owner ruled three switches on; every other review
    switch still waits for the suite and is OFF with its variable unset,
    and an empty value counts as unset."""
    assert set(review_flags.DEFAULT_ON_ENVS) == {
        "GEOTECH_JUDGMENT_CALLS", "GEOTECH_FIGURE_HELPER",
        "GEOTECH_PLOT_CODE"}
    readers = {
        review_flags.AGENT_ENV: lambda: (review_flags.lean_agent()
                                         or review_flags.minimal_agent()),
        review_flags.VISION_TEXT_ENV: review_flags.vision_text_context,
        review_flags.VISION_STRUCTURED_ENV: review_flags.vision_structured,
        review_flags.VISION_INLINE_ENV: review_flags.vision_inline,
        review_flags.SWEEP_ENV: review_flags.sweep,
        review_flags.FINDINGS_ENV: review_flags.findings,
        review_flags.OVERVIEW_ENV: review_flags.overview,
        review_flags.GEOMETRY_ENV: review_flags.geometry,
        review_flags.DIGEST_ENV: review_flags.digest,
        review_flags.COVERAGE_ENV: review_flags.coverage,
        review_flags.CHECKLIST_ENV: review_flags.checklist,
        review_flags.REPORT_INGEST_ENV: review_flags.report_ingest,
        review_flags.VISION_TURN_RETRY_ENV: review_flags.vision_turn_retry,
        review_flags.KEPT_READINGS_ENV:
            review_flags.kept_readings_on_review_page,
        review_flags.CITED_PAGES_ENV:
            lambda: review_flags.cited_pages() != "off",
        review_flags.CITED_DOCUMENTS_ENV:
            lambda: review_flags.cited_documents() != "off",
        review_flags.STEP_CAP_ANSWER_ENV: review_flags.step_cap_answer,
        review_flags.FIGURE_HELPER_ENV: review_flags.figure_helper,
        review_flags.PLOT_CODE_ENV: review_flags.plot_code,
        review_flags.JUDGMENT_CALLS_ENV: review_flags.judgment_calls,
    }
    assert set(readers) == set(review_flags.ALL_ENVS)
    for unset in (None, "", "  "):
        for env in review_flags.ALL_ENVS:
            if unset is None:
                monkeypatch.delenv(env, raising=False)
            else:
                monkeypatch.setenv(env, unset)
        for env, read in readers.items():
            assert bool(read()) is (env in review_flags.DEFAULT_ON_ENVS), \
                (env, unset)
    described = review_flags.describe()
    for env in review_flags.ALL_ENVS:
        assert (f"{env}=1 (default)" in described) is (
            env in review_flags.DEFAULT_ON_ENVS), env


def test_the_default_build_states_its_judgment_calls(monkeypatch):
    """No variable set at all: the geotech build is the switch-on build,
    and its record says the switch is on by default."""
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    captured = {}
    real = agent_mod.create_deep_agent

    def spy(**k):
        captured.update(k)
        return real(**k)
    monkeypatch.setattr(agent_mod, "create_deep_agent", spy)
    agent = agent_mod.build_deep_agent(model=_Quiet())
    assert ("\n\n" + judgment_calls_section(references=True)) in \
        captured["system_prompt"]
    assert agent.geotech_judgment_consult is True
    assert "GEOTECH_JUDGMENT_CALLS=1 (default)" in \
        agent.geotech_build["switches"]
    assert _spec(captured, "references")["system_prompt"] == \
        _references_prompt(True, consult=True)


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
        # GM-187's tighter budget and its warning are behind the switch
        assert _budget(_spec(captured, "references")) == (
            DEFAULT_REFERENCES_MAX_MODEL_CALLS, 0)
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
        # GM-181 / GM-155 w13b fix 10: the brief's GIVEN block and the
        # consult's citations, after the unswitched nudge
        want += "\n\n" + (agent_mod._CALC_DELEGATION_NUDGE
                          + agent_mod._CALC_DELEGATION_GIVEN
                          + agent_mod._CALC_DELEGATION_CITATIONS)
    assert captured["system_prompt"] == want
    assert "GEOTECH_JUDGMENT_CALLS=1" in agent.geotech_build["switches"]
    # the turn-end check (GM-174) acts on this build
    assert agent.geotech_judgment_consult is True


def test_switch_on_calc_helper_reports_its_calls(monkeypatch):
    off, _ = _built(monkeypatch, False, enable_calc_subagent=True)
    on, _ = _built(monkeypatch, True, enable_calc_subagent=True)
    assert _spec(on, "calc")["system_prompt"] == (
        _spec(off, "calc")["system_prompt"] + agent_mod._CALC_JUDGMENT_CALLS
        + agent_mod._CALC_GIVEN_AND_CITATIONS)
    text = agent_mod._CALC_JUDGMENT_CALLS
    for part in ("the user's", '"Judgment calls"', "why", "basis",
                 "alternative", "what would change the result"):
        assert part in text, part


def test_switch_on_references_return_cited_options(monkeypatch):
    captured, _ = _built(monkeypatch, True)
    assert _spec(captured, "references")["system_prompt"] == \
        _references_prompt(True, consult=True)
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
    # it gives"); GM-187's one line per call, 90 more (1,756)
    assert len(block) < 1770
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
                 "`choices` or a reference already looked up: list those "
                 "calls one per line and ask only for each one's source or "
                 '"no reference found".',
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
                     "turns on most",
                     # GM-181 and w13b fix 10
                     "GIVEN block", "as `given`", "*_package `given`",
                     "pass its citations", "print each citation"):
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


# ---------------------------------------------------------------------------
# GM-181: the brief fills a package's `given`; w13b fix 10: a bespoke PDF
# carries the consult's citations
# ---------------------------------------------------------------------------

def test_the_brief_carries_the_users_words_for_given(monkeypatch):
    """GM-167 prints under Given Inputs only what a package call names in
    `given`. w13b F18j: the brief put c = 0, which "sand" only implies,
    beside the user's inputs, and the package call named nothing. With
    the switch on the nudge asks for a GIVEN block in the user's own words
    and `given` from it, and the calc helper passes it; a value worked out
    from the user's words, or chosen, is an assumption with its basis."""
    captured, _ = _built(monkeypatch, True, enable_calc_subagent=True)
    main = " ".join(captured["system_prompt"].split())
    for part in ("a GIVEN block quotes the user's own words for each input "
                 "they stated",
                 "pass those inputs as `given` ({input name: the user's "
                 "words})",
                 "a value you derived from those words, or chose, goes "
                 "under assumptions with its basis"):
        assert part in main, part
    calc = " ".join(_spec(captured, "calc")["system_prompt"].split())
    for part in ("pass a *_package `given` naming each input the delegation "
                 "says the user gave",
                 "a value worked out from those words, or chosen, goes in "
                 "`assumptions` with its basis"):
        assert part in calc, part
    # the unswitched nudge comes first, whole
    assert captured["system_prompt"].endswith(agent_mod.calc_delegation_nudge(
        judgment_calls=True, references=True))


def test_the_brief_carries_the_consults_citations(monkeypatch):
    """w13b F55j: the consult's citations reached the main agent and not
    the rebuild brief, so the bespoke PDF printed none of them. The main
    agent consults before it delegates such a deliverable and passes the
    citations in the brief; the helper prints each beside the choice it
    supports. Without a references helper there is nothing to pass."""
    captured, _ = _built(monkeypatch, True, enable_calc_subagent=True)
    main = " ".join(captured["system_prompt"].split())
    assert ("ask the `references` helper before you delegate it, and pass "
            "its citations in the brief, each beside the method or "
            "assumption it supports") in main
    calc = " ".join(_spec(captured, "calc")["system_prompt"].split())
    assert ("print each citation the delegation passes beside the method "
            "or assumption it supports") in calc
    assert "(html_to_pdf)" in calc
    alone, _ = _built(monkeypatch, True, enable_calc_subagent=True,
                      reference_mode="off")
    assert alone["system_prompt"].endswith(agent_mod.calc_delegation_nudge(
        judgment_calls=True, references=False))
    assert "pass its citations" not in alone["system_prompt"]
    assert "GIVEN block" in alone["system_prompt"]


def test_the_given_and_citation_lines_are_general_and_short():
    """No case from the measured runs (the owner's no-overfitting rule),
    and short."""
    texts = (agent_mod._CALC_DELEGATION_GIVEN,
             agent_mod._CALC_DELEGATION_CITATIONS,
             agent_mod._CALC_GIVEN_AND_CITATIONS)
    for t in texts:
        low = " ".join(t.split()).lower()
        for word in ("sand", "c = 0", "cohesion", "vesic", "water table",
                     "rankine", "gec", "ufc", "wall", "footing", "4.5"):
            assert word not in low, word
    assert len(agent_mod._CALC_DELEGATION_GIVEN) < 320
    assert len(agent_mod._CALC_DELEGATION_CITATIONS) < 280
    assert len(agent_mod._CALC_GIVEN_AND_CITATIONS) < 480
    assert agent_mod.calc_delegation_nudge() == \
        agent_mod._CALC_DELEGATION_NUDGE


# ---------------------------------------------------------------------------
# GM-187: the consult's cost (live wave w14: $0.20-0.53 a turn)
# ---------------------------------------------------------------------------

def test_switch_on_the_judgment_consult_has_a_tighter_budget(monkeypatch):
    """Where the main agent is asked to consult about its calls, the
    references helper gets at most 6 calls, is told two calls ahead, and
    is asked for one line per call with the two lookups named."""
    captured, agent = _built(monkeypatch, True)
    assert agent.geotech_judgment_consult is True
    spec = _spec(captured, "references")
    assert _budget(spec) == (JUDGMENT_REFERENCES_MAX_MODEL_CALLS,
                             JUDGMENT_REFERENCES_WARN_REMAINING)
    assert f"You have {JUDGMENT_REFERENCES_MAX_MODEL_CALLS} model calls" \
        in spec["system_prompt"]
    # never above the caller's own budget, and no budget stays none
    small, _ = _built(monkeypatch, True, references_max_model_calls=4)
    assert _budget(_spec(small, "references")) == (
        4, JUDGMENT_REFERENCES_WARN_REMAINING)
    none, _ = _built(monkeypatch, True, references_max_model_calls=0)
    assert _budget(_spec(none, "references")) is None
    # the reviewer's budget is its own
    assert _budget(_spec(captured, "reviewer")) == _budget(
        _spec(_built(monkeypatch, False)[0], "reviewer"))


def test_switch_on_a_build_asking_no_consult_keeps_the_old_budget(
        monkeypatch):
    """A page that brings its own prompt asks for no judgment-call consult
    (GM-174 never acts there): its references helper keeps the options rule
    of GM-116 and the full budget."""
    captured, agent = _built(monkeypatch, True, system_prompt="OWN PAGE")
    assert agent.geotech_judgment_consult is False
    spec = _spec(captured, "references")
    assert spec["system_prompt"] == _references_prompt(True)
    assert _budget(spec) == (DEFAULT_REFERENCES_MAX_MODEL_CALLS, 0)


def test_the_judgment_consult_text_is_general_and_short():
    text = " ".join(agent_mod._REFERENCES_JUDGMENT_CALLS.split())
    for part in ("answer each in one line", '"no reference found"',
                 "`reference_search`", "`reference_get`",
                 "no need to list the modules or methods first"):
        assert part in text, part
    low = text.lower()
    for word in ("sand", "bearing", "wall", "strut", "gec", "ufc", "dm7",
                 "slope", "footing", "excavation"):
        assert word not in low, word
    assert len(agent_mod._REFERENCES_JUDGMENT_CALLS) < 500

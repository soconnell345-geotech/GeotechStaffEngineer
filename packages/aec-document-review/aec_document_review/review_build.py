"""The Document Review page builds itself (INF-31, the app split's step 8).

:func:`build_review_page_agent` is the page's agent: the ``minimal`` build
behind ``GEOTECH_REVIEW_AGENT=minimal`` (:mod:`funhouse_agent.deep.
minimal_agent`), and otherwise today's build: the vision kit with the
page's own descriptions, ``calculate``, the figure helper, the page's
reading helper in place of deepagents' stock one, a model-call budget
whose last call answers, and the scratch filesystem hidden. Extracted from
``build_deep_agent`` (whose ``review_page=True`` hands its callers here
until step 15); what both builds do the same way is
:mod:`aec_agent_core.deep.assembly`. This file goes to Document Review at
step 13; it imports nothing of the geotechnical toolkit.

Since the copy into this package (step 13, BACKLOG INF-33) nothing reaches
GeotechStaffEngineer: the reference library's chart lines, which the page
used to hand the chart stack as the geotech page does, are gone (the one
deliberate fingerprint change of the split): ``plot_data`` has no
``reference_chart`` and ``plot_code`` no ``chart_lines()``.
"""

from typing import Callable, Optional

from aec_agent_core.deep import assembly
from aec_agent_core.deep.assembly import (_AUTO_SUMMARIZATION_TRIGGER,
                                          _DEFAULT_SUMMARIZATION_KEEP)
from aec_agent_core.deep.limits import (DEFAULT_MAX_RESULT_CHARS,
                                        ModelCallBudgetMiddleware)
from aec_agent_core.deep.tool_guards import HideScratchFilesystem


# ---------------------------------------------------------------------------
# The Document Review page on the default build: three fixes the former
# lean build had (removed in DR-39) (BACKLOG DR-01 - DR-03; the review of 2026-09-26 and
# check 1's trace review, causes C3 and C8)
# ---------------------------------------------------------------------------

#: Turns the page's model-call budget off on this build (the step cap then
#: ends a long turn with GraphRecursionError, as before 5.34): a way back on a
#: live host without a release.
_BUDGET_OFF = ("0", "off", "none", "no", "false")

#: Graph steps per model call on this build, with room to spare: one model
#: call is the model node, the tools node and the middleware's before- and
#: after-model nodes. Measured offline on deepagents 0.6.8: about 3 without
#: the budget (33 calls in 100 steps; 17 in the app's 50) and at most 5 with
#: it (40 calls in 200 steps); 8 leaves room for newer deepagents.
REVIEW_PAGE_STEPS_PER_CALL = 8

#: What the primary is told the reading helper is (the ``task`` tool lists it).
REVIEW_READER_DESCRIPTION = (
    "A reading helper with your reading and looking tools and your reading "
    "rules. Hand it a self-contained reading job and get back a compact list "
    "of findings with citations - e.g. 'open <source>, read sheets 12-20 and "
    "list every note that sets a concrete strength, with the sheet'. It sees "
    "NOTHING of this conversation: put everything it needs in the "
    "description (the source name, the pages, what to find, what to "
    "return). It cannot write files or markups.")


#: Tools the geotech page has and this page cannot use: they render a
#: reference library figure and a worked example's printed source page, and
#: the review page has neither the library nor the examples (DR-04; the
#: review of 2026-09-26). ``log_grid`` reads a
#: boring or test-pit log as its grid: a geotechnical report's page, which
#: is not this page's document (BACKLOG RI-34). Since INF-30 step 4 the page
#: is built without the first two (:data:`GEOTECH_VISION_TOOLS`, which GSE
#: adds to the vision kit) and drops only ``log_grid``, a planlens tool.
GEOTECH_ONLY_TOOLS = ("read_reference_figure", "view_worked_example_source",
                      "log_grid")


def _review_page_descriptions() -> dict:
    """The page's own wording where the geotech wording is wrong here:
    ``render_region`` sends the agent to the ``drawing_ir`` module for
    coordinates, and this page has no modules (DR-01, check 1 C3)."""
    from aec_document_review.review_agent import REVIEW_DESCRIPTIONS
    return {"render_region": REVIEW_DESCRIPTIONS["render_region"]}


def _review_page_rewrites() -> dict:
    """The page's phrase swaps, so this build's examples name no kind
    of geotechnical report either (BACKLOG RI-34)."""
    from aec_document_review.review_agent import REVIEW_REWRITES
    return dict(REVIEW_REWRITES)


#: The stock helper's name in the agent library: a helper declared with it
#: takes the stock one's place.
READER_NAME = "general-purpose"


def _review_reader_subagent(primary_tools,
                            show_images=None) -> Optional[dict]:
    """deepagents' ``general-purpose`` helper, re-declared for the review
    page with the review's reading rules, the read-only tools and a budget
    whose last call reports (DR-02, check 1 C8: the stock helper had a
    286-character prompt and returned whole-page boxes as anchors).

    The name stays ``general-purpose`` so it REPLACES the stock helper. Its
    tools are the primary's reading and looking tools, the reading helper's set
    (``review_agent.READER_TOOLS``); nothing that writes a file or a markup.

    The helper is declared whole here, by its name. It used to start from
    the agent library's own spec, imported from a private name: had that
    name moved in an upgrade, this returned nothing and the page silently
    got the stock helper, with every tool and none of the reading rules
    (BACKLOG INF-71).
    ``show_images`` is the middleware that shows the helper an image it asked
    to be shown (``show_me``, BACKLOG DR-38); the helper holds the primary's
    looking tools, so it needs its own.
    """
    from aec_document_review.review_prompt import DOCUMENT_REVIEW_READER_PROMPT
    from aec_document_review.review_agent import (
        READER_EXHAUSTED, READER_MAX_MODEL_CALLS, READER_NUDGE, READER_TOOLS)
    wanted = set(READER_TOOLS)
    return {
        "name": READER_NAME,
        "description": REVIEW_READER_DESCRIPTION,
        "system_prompt": DOCUMENT_REVIEW_READER_PROMPT,
        "tools": [t for t in primary_tools
                  if getattr(t, "name", None) in wanted],
        "middleware": [*([show_images] if show_images is not None else []),
                       ModelCallBudgetMiddleware(
                           READER_MAX_MODEL_CALLS,
                           final_turn_nudge=READER_NUDGE,
                           exhausted_message=READER_EXHAUSTED),
                       HideScratchFilesystem()],
    }


def _show_images(engine):
    """The middleware that shows an agent the images its ``show_me`` looks
    stored, or ``None`` when the switch is off (``GEOTECH_VISION_SHOW_ME``,
    BACKLOG DR-38). One per agent that holds the looking tools."""
    from aec_agent_core import review_flags
    if not review_flags.show_me():
        return None
    from aec_agent_core.deep.inline_images import InlineImageMiddleware
    return InlineImageMiddleware(engine=engine)


def review_page_budget() -> Optional[int]:
    """The review page's model calls per request on this build
    (``GEOTECH_REVIEW_MAX_MODEL_CALLS``, default 40), or
    ``None`` when that variable is ``off``."""
    import os
    from aec_document_review.review_agent import (MAX_MODEL_CALLS_ENV,
                                                  max_model_calls)
    raw = os.environ.get(MAX_MODEL_CALLS_ENV, "").strip().lower()
    if raw in _BUDGET_OFF:
        return None
    return max_model_calls()


def build_review_page_agent(
    model,
    *,
    engine=None,
    attachments=None,
    save_fn: Optional[Callable] = None,
    extra_tools=None,
    extra_system_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    markup_author: Optional[str] = None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    reference_result_chars: Optional[int] = None,
    subagent_extra_tools=None,
    subagent_extra_system_prompt: Optional[str] = None,
    store=None,
    checkpointer=None,
    enable_memory: bool = False,
    enable_summarization: bool = False,
    summarization_model=None,
    summarization_trigger=_AUTO_SUMMARIZATION_TRIGGER,
    summarization_keep=_DEFAULT_SUMMARIZATION_KEEP,
    working_dir: Optional[str] = None,
    subagent_models: Optional[dict] = None,
    allowed_agents=(),
    reference_mode: str = "off",
    enable_calc_subagent: bool = False,
    review_page: bool = True,
    geotech_parts: Optional[Callable] = None,
    create: Optional[Callable] = None,
    **kwargs,
):
    """The Document Review page's agent.

    Takes the keyword arguments the page hands ``build_deep_agent``
    (``aec_document_review/review_page.py``): no analysis modules (``allowed_agents``
    empty), no reference helpers (``reference_mode="off"``), no calc
    helper, its own ``system_prompt``. The settings of the geotech page's
    Behavior panel that reach every build (the reference budget) are
    accepted and have nothing to act on here, as before.

    ``geotech_parts(engine, attachments, enable_memory) -> dict`` is how
    ``build_deep_agent(review_page=True)`` keeps an old caller's request
    for geotechnical parts (modules, reference or calc helpers, the domain
    prompt): their tools, prompt and helpers, made by the geotech builder
    and placed where that build placed them (until step 15). Without it,
    asking for them is refused. ``create`` compiles the agent (default
    ``deepagents.create_deep_agent``).
    """
    from aec_agent_core.deep.warm_imports import warm_pyplot
    warm_pyplot()
    # No standard chart lines are registered here (the app split's one
    # deliberate fingerprint change, step 13): plot_data has no
    # reference_chart and plot_code no chart_lines().
    from aec_agent_core import review_flags
    if review_flags.minimal_agent():
        from aec_document_review.minimal_agent import build_minimal_agent
        return build_minimal_agent(
            model, engine=engine, attachments=attachments,
            save_fn=save_fn, markup_author=markup_author,
            max_result_chars=max_result_chars,
            checkpointer=checkpointer, store=store,
            working_dir=working_dir)
    if geotech_parts is None and (allowed_agents or reference_mode != "off"
                                  or enable_calc_subagent):
        raise ValueError(
            "The Document Review page's build has no analysis modules, "
            "reference helpers or calc helper (allowed_agents=(), "
            "reference_mode='off', enable_calc_subagent=False); "
            "build_deep_agent builds an agent with them.")
    for name in _GEOTECH_ONLY_ARGUMENTS:
        kwargs.pop(name, None)

    # A store implies persistent memory: there is no point handing in a store
    # without routing /memories/ to it.
    if store is not None:
        enable_memory = True

    # Default-wrap the model for vision when no explicit engine is given and
    # the model is a chat-model object (not a "provider:model" string).
    if engine is None and not isinstance(model, str):
        from aec_agent_core.deep.vision_engine import LangChainVisionEngine
        engine = LangChainVisionEngine(model)

    # ONE shared attachments dict for the whole agent, exposed on the
    # returned agent so files added after the build reach every tool.
    attachments = {} if attachments is None else attachments
    parts = (geotech_parts(engine, attachments, enable_memory)
             if geotech_parts is not None else {})

    from aec_agent_core import review_flags as _flags
    from aec_agent_core.deep.calculate_tool import make_calculate_tool
    from aec_agent_core.deep.vision_kit import make_vision_tools
    tools = list(parts.get("core_tools") or []) + make_vision_tools(
        engine=engine,
        attachments=attachments,
        save_fn=save_fn,
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
        markup_author=markup_author,
        description_overrides=_review_page_descriptions(),
        description_rewrites=_review_page_rewrites(),
        plot_reference_charts=False,
        added_tools=None,
        # the looking tools take show_me behind its switch (DR-38)
        show_choice=_flags.show_me(),
        # behind the figure switches the main agent's own redraw can retire
        # a figure any helper drew this turn (GM-151)
        plot_data_replaces=_flags.figure_helper() or _flags.plot_code(),
    ) + [make_calculate_tool()]
    # kept_readings (TA-28) is the geotech page's (its review-page switch
    # went with BACKLOG DR-39); log_grid is a geotechnical report's
    # (GEOTECH_ONLY_TOOLS).
    _drop = ("log_grid", "kept_readings")
    tools = [t for t in tools if getattr(t, "name", None) not in _drop]
    if extra_tools:
        tools = list(tools) + list(extra_tools)

    # ``system_prompt`` REPLACES the geotechnical domain prompt: the page is
    # a different agent over the same tools. ``extra_system_prompt`` still
    # appends.
    if system_prompt is None:
        system_prompt = parts.get("domain_prompt")
    if system_prompt is None:
        from aec_document_review.review_prompt import build_document_review_prompt
        system_prompt = build_document_review_prompt(
            memory_enabled=enable_memory)
    if extra_system_prompt:
        system_prompt = system_prompt + "\n\n" + extra_system_prompt
    if parts.get("calc_nudge"):
        system_prompt = system_prompt + "\n\n" + parts["calc_nudge"]
    # Every primary tool refuses an argument it does not take, by name
    # (brief 5, N3). See funhouse_agent/deep/strict_args.py.
    from aec_agent_core.deep.strict_args import strict_tools
    tools = strict_tools(tools)

    subagents = list(parts.get("subagents_before_figure") or [])
    figure_tasks = None
    if _flags.figure_helper() or _flags.plot_code():
        # The figure helper (OWN-61 option B, GM-92), behind
        # GEOTECH_FIGURE_HELPER and GEOTECH_PLOT_CODE, on by default (GM-227).
        from aec_agent_core.deep.figure_helper import (FIGURE_PRIMARY_NOTE,
                                                       FigureTasks,
                                                       build_figure_subagent)
        _fig = build_figure_subagent(
            engine=engine, attachments=attachments, save_fn=save_fn,
            max_result_chars=max_result_chars,
            reference_result_chars=reference_result_chars,
            extra_tools=subagent_extra_tools,
            extra_system_prompt=subagent_extra_system_prompt)
        if _fig is not None:
            # RI-34: the task tool names no kind of geotechnical report here
            from aec_document_review.review_agent import (
                REVIEW_HELPER_REWRITES)
            for old, new in REVIEW_HELPER_REWRITES["figure"]:
                _fig["description"] = _fig["description"].replace(old, new)
            subagents.append(_fig)
            # The main agent hands figures to it and does not look again
            # at what it looked at (live wave w9, GM-107).
            system_prompt = system_prompt + "\n\n" + FIGURE_PRIMARY_NOTE
            figure_tasks = FigureTasks(with_code=any(
                getattr(t, "name", None) == "plot_code"
                for t in _fig["tools"]))
    subagents += list(parts.get("subagents_after_figure") or [])
    # The page's reading helper replaces deepagents' stock one, so it
    # carries the reading rules (DR-02); appended here so the guards reach
    # it like every other helper.
    reader = _review_reader_subagent(tools, show_images=_show_images(engine))
    if reader is not None:
        subagents.append(reader)

    allowed = parts.get("allowed_agents") or ()
    helper_models = assembly.guard_helpers(
        subagents, tools, allowed_agents=allowed,
        call_scope=parts.get("call_scope"),
        note_types=parts.get("note_types") or (),
        unknown_tool_hint=parts.get("unknown_tool_hint"),
        subagent_models=subagent_models)

    # deepagents' scratch-filesystem tools see none of the user's files and
    # answered "not found" for them (wave 2b, F41): kept off this page's
    # menu and prompt. Then a model-call budget whose last call answers from
    # what was gathered, so a long turn ends with findings and what was not
    # checked, never a GraphRecursionError (DR-03).
    # First, so it reads the tool results as the turn left them: an image a
    # look asked to be shown goes in front of the model at its next call.
    page_middleware = [*([m] if (m := _show_images(engine)) is not None
                         else []),
                       HideScratchFilesystem()]
    page_budget = review_page_budget()
    if page_budget:
        from aec_document_review.review_agent import EXHAUSTED, FINAL_NUDGE
        page_middleware.append(ModelCallBudgetMiddleware(
            page_budget, final_turn_nudge=FINAL_NUDGE,
            exhausted_message=EXHAUSTED))
    create_kwargs = assembly.create_kwargs_for(
        kwargs, model=model, store=store, checkpointer=checkpointer,
        enable_memory=enable_memory,
        enable_summarization=enable_summarization,
        summarization_model=summarization_model,
        summarization_trigger=summarization_trigger,
        summarization_keep=summarization_keep,
        figure_tasks=figure_tasks, allowed_agents=allowed,
        unknown_tool_hint=parts.get("unknown_tool_hint"),
        page_middleware=page_middleware)
    agent = assembly.compile_agent(
        create or assembly.create_deep_agent, model=model, tools=tools,
        system_prompt=system_prompt, subagents=subagents,
        create_kwargs=create_kwargs)
    return assembly.record_build(
        agent, attachments=attachments, model=model, tools=tools,
        subagents=subagents, system_prompt=system_prompt,
        helper_models=helper_models, judgment_consult=False,
        min_recursion_limit=(REVIEW_PAGE_STEPS_PER_CALL * page_budget + 30
                             if page_budget else None))


#: ``build_deep_agent`` arguments the page has nothing to act on (they ask
#: for the geotech page's helpers, which the page never builds); accepted
#: and dropped, never handed to deepagents.
_GEOTECH_ONLY_ARGUMENTS = (
    "references_max_model_calls", "calc_extra_tools",
    "calc_extra_system_prompt", "calc_max_model_calls",
    "enable_setup_agent", "setup_store", "setup_render_dir",
    "enable_report_ingest", "report_ingest_out_dir", "report_ingest_budgets",
    "report_ingest_db", "report_ingest_max_model_calls",
    "enable_report_library", "library_root", "library_db",
    "library_max_tool_calls",
)


__all__ = ["build_review_page_agent", "review_page_budget",
           "REVIEW_PAGE_STEPS_PER_CALL", "REVIEW_READER_DESCRIPTION",
           "GEOTECH_ONLY_TOOLS"]

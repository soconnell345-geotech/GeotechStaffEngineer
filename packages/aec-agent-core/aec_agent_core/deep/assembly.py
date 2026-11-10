"""The assembly both pages' agents share (INF-31, the app split's step 8).

Everything ``build_deep_agent`` (the geotech page) and
:func:`funhouse_agent.review_build.build_review_page_agent` (the Document
Review page) do the same way once each has its tools, its prompt and its
helpers: the guards round every helper (the record of what it read, the
busy-model ladder, the tool-error and output-limit guards, the
scratch-filesystem guard), deepagents' general-purpose helper re-declared
with the same guards and a budget, a helper's own model, the persistent ``/memories/`` backend and the summarizer, the
main agent's middleware chain, the compile (with the planning middleware
attached where deepagents no longer adds it), and the build record a host
reads off the compiled agent. Moved here from :mod:`funhouse_agent.deep.
agent`, whose names for these are the same objects; it goes to the core at
step 11 and imports nothing of the geotechnical toolkit (a build that has
modules hands in its unknown-tool hint and scopes and its helpers' note
middleware).
"""

from typing import Callable, Optional

from deepagents import create_deep_agent  # noqa: F401 - the default compile
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.middleware import SummarizationMiddleware

# Planning middleware (the ``write_todos`` tool). deepagents auto-attached it
# through 0.7.1; 0.7.2+ dropped the auto-attach, while our domain prompt still
# nudges the model to plan with write_todos — on the newer versions the model
# then loops calling a nonexistent tool until the recursion cap (observed live
# 2026-08: EVERY question failed with GraphRecursionError on deepagents
# 0.7.11). Import it from whichever home exists so we can attach it EXPLICITLY.
try:
    from deepagents.middleware import TodoListMiddleware as _TodoMiddleware
except ImportError:                                       # deepagents >= 0.7.2
    try:
        from langchain.agents.middleware import (
            TodoListMiddleware as _TodoMiddleware)
    except ImportError:                                   # no planning available
        _TodoMiddleware = None

try:  # the spec deepagents auto-adds; re-declared below to carry the guard
    from deepagents.middleware.subagents import (
        GENERAL_PURPOSE_SUBAGENT as _GENERAL_PURPOSE_SPEC,
    )
except Exception:  # noqa: BLE001 - layout differs: leave deepagents' default
    _GENERAL_PURPOSE_SPEC = None

from aec_agent_core.deep.limits import (
    HELPER_EXHAUSTED,
    HELPER_FINAL_NUDGE,
    UNRUNNABLE_TOOLS,
    HideTools,
    ModelCallBudgetMiddleware,
    helper_max_model_calls,
)
from aec_agent_core.deep.busy_retry import AgentBusyRetry
from aec_agent_core.deep.scratch_guard import ScratchFilesystemGuard
from aec_agent_core.deep.delivered_log import DeliveredToolResults
from aec_agent_core.deep.keep_work import HelperRecorder, KeepWork
from aec_agent_core.deep.result_map import LargeResultKeyMap
from aec_agent_core.deep.tool_guards import (OutputLimitGuard, ToolErrorGuard,
                                             auth_retry)


#: Added to deepagents' stock general-purpose helper prompt (field session
#: 2026-10-14, BACKLOG TA-21: two helpers that held extracted data returned
#: apologies of 366 and 268 characters; RI-31: data saved to a file is named,
#: not typed out again).
HELPER_PARTIAL_RULE = (
    "\n\nIf you cannot finish -- an error, a tool that refuses your data, "
    "more to read than you can -- your final message still gives everything "
    "you did gather: the data itself, in full (or, for data you saved with "
    "save_file, the file's name), then what is missing and why. "
    "Never answer with only an apology."
)


# ---------------------------------------------------------------------------
# Phase-3 capability wiring (persistent memory + summarization)
# ---------------------------------------------------------------------------

#: Filesystem route prefix that the agent's persistent ``/memories/`` files live
#: under. Everything else stays in ephemeral per-thread state.
MEMORIES_ROUTE = "/memories/"

#: The durable project-context note the agent is nudged to maintain. It is
#: declared as a deepagents ``memory=`` source so its contents are loaded into
#: the system prompt at the start of every thread (cross-session), and it lives
#: under ``/memories/`` so it is also writable/persistent via the store.
MEMORIES_AGENTS_FILE = f"{MEMORIES_ROUTE}AGENTS.md"


def _memories_namespace(_runtime) -> tuple:
    """Namespace factory for the ``/memories/`` :class:`StoreBackend`.

    Returns a fixed ``("memories",)`` namespace so every thread that shares the
    same store reads and writes the SAME persistent memory bucket. A fixed
    namespace (rather than the deprecated legacy ``assistant_id`` detection) is
    what makes ``/memories/`` durable *across* sessions: thread 1 can write a
    fact and thread 2 can read it back.
    """
    return ("memories",)


def build_memory_backend(store=None):
    """Build the :class:`CompositeBackend` that routes ``/memories/`` to a store.

    Parameters
    ----------
    store : BaseStore, optional
        A LangGraph store. When given, it is bound directly onto the
        :class:`StoreBackend` so the backend works both inside graph execution
        AND when called directly (e.g. the cross-session round-trip test). When
        ``None``, the :class:`StoreBackend` resolves the store at call time via
        ``get_store()`` — i.e. the ``store=`` handed to ``create_deep_agent``.

    Returns
    -------
    CompositeBackend
        ``default=StateBackend()`` (ephemeral scratch files, per-thread) with a
        single route ``{"/memories/": StoreBackend(...)}`` (persistent,
        cross-thread). Normal scratch paths (``/notes.md`` etc.) stay in state;
        only paths under ``/memories/`` hit the store.
    """
    return CompositeBackend(
        default=StateBackend(),
        routes={
            MEMORIES_ROUTE: StoreBackend(store=store, namespace=_memories_namespace),
        },
    )


class GeotechSummarizationMiddleware(SummarizationMiddleware):
    """Our explicitly-configured summarization middleware.

    ``create_deep_agent`` ALREADY auto-attaches a deepagents
    ``SummarizationMiddleware`` (with model-aware default thresholds), and
    LangChain's ``create_agent`` rejects two middleware that report the same
    ``.name``. The deepagents impl reports ``name == "SummarizationMiddleware"``
    for the exact base class but falls back to ``type(self).__name__`` for
    subclasses — so subclassing is the supported way to add a *second*,
    custom-tuned summarizer without colliding with the built-in one. Both share
    the ``_summarization_event`` state key and the same backend, so they
    interoperate; ours fires at whichever trigger we configure.
    """


#: Sentinel meaning "pick a trigger based on the model" — see
#: :func:`_resolve_summarization_trigger`.
_AUTO_SUMMARIZATION_TRIGGER = None
_DEFAULT_SUMMARIZATION_KEEP = ("messages", 8)

#: Fraction-of-context trigger used when the model publishes its input window.
_SUMMARIZATION_FRACTION = 0.8
#: Absolute-token fallback for models with no published context window.
_SUMMARIZATION_TOKENS_FALLBACK = 170_000


def _model_has_token_profile(model) -> bool:
    """True if ``model`` exposes ``max_input_tokens`` in its profile.

    Fraction-based summarization triggers (``("fraction", f)``) require the
    model to publish ``max_input_tokens``; without it the LangChain summarizer
    raises at construction. Real chat models (e.g. ``ChatAnthropic``) publish a
    profile; fakes/strings do not.
    """
    profile = getattr(model, "profile", None)
    return (
        isinstance(profile, dict)
        and isinstance(profile.get("max_input_tokens"), int)
    )


def _resolve_summarization_trigger(trigger, model):
    """Resolve the summarization trigger for ``model``.

    Honors an explicit ``trigger``. For the ``_AUTO_SUMMARIZATION_TRIGGER``
    sentinel, returns a fraction-of-context trigger when the model publishes its
    input window, otherwise an absolute-token fallback — mirroring deepagents'
    own model-aware defaulting so a profile-less model (string spec or a fake
    test model) never blows up on a fractional trigger.
    """
    if trigger is not _AUTO_SUMMARIZATION_TRIGGER:
        return trigger
    if _model_has_token_profile(model):
        return ("fraction", _SUMMARIZATION_FRACTION)
    return ("tokens", _SUMMARIZATION_TOKENS_FALLBACK)


def build_summarization_middleware(
    model,
    *,
    backend=None,
    trigger=_AUTO_SUMMARIZATION_TRIGGER,
    keep=_DEFAULT_SUMMARIZATION_KEEP,
):
    """Build the custom-tuned :class:`GeotechSummarizationMiddleware`.

    Parameters
    ----------
    model : BaseChatModel
        Model that writes the summaries. (A ``"provider:model"`` string is
        resolved by deepagents internally.)
    backend : BackendProtocol, optional
        Backend the summarizer offloads evicted history to (so it can be
        re-read via ``read_file``). Defaults to an ephemeral
        :class:`StateBackend` when not given.
    trigger : ContextSize, optional
        When to summarize; see :func:`_resolve_summarization_trigger`. Defaults
        to the model-aware auto trigger.
    keep : ContextSize, optional
        How much recent history to preserve verbatim. Defaults to
        ``("messages", 8)``.

    Returns
    -------
    GeotechSummarizationMiddleware
        A distinctly-named summarizer that can be appended to
        ``create_deep_agent(middleware=...)`` without colliding with the
        built-in one.
    """
    summ_backend = backend if backend is not None else StateBackend()
    return GeotechSummarizationMiddleware(
        model=model,
        backend=summ_backend,
        trigger=_resolve_summarization_trigger(trigger, model),
        keep=keep,
    )


def _merge_tools(*groups):
    """Concatenate tool lists, keeping the first tool of each name."""
    out, seen = [], set()
    for group in groups:
        for t in group or ():
            name = getattr(t, "name", None)
            if name is not None:
                if name in seen:
                    continue
                seen.add(name)
            out.append(t)
    return out or None


def _save_tool(tools):
    """The ``save_file`` tool among ``tools``, or ``None``: a helper's
    scratch guard saves its ``.json`` ``write_file`` calls with it (BACKLOG
    RI-31)."""
    for t in tools or ():
        if getattr(t, "name", None) == "save_file":
            return t
    return None


def _join_prompts(*parts):
    return "\n\n".join(p for p in parts if p) or None


# ---------------------------------------------------------------------------
# The helpers' guards
# ---------------------------------------------------------------------------

def guard_helpers(subagents: list, tools, *, allowed_agents=None,
                  call_scope: Optional[dict] = None, note_types=(),
                  unknown_tool_hint: Optional[Callable] = None,
                  subagent_models: Optional[dict] = None) -> dict:
    """Put the guards on every helper of a build, in place, and add
    deepagents' general-purpose helper with the same guards where the build
    has not re-declared it; give a helper its own model. Returns the helpers on a model of their own (name ->
    model), for the build record.

    ``tools`` are the main agent's (a helper without "tools" has them).
    ``call_scope`` maps a helper's name to the modules it reaches, for the
    unknown-tool hint, ``unknown_tool_hint(scope)`` (a module method called
    as if it were a tool is answered with the call_agent form; live smoke
    wave 1, A15g), which a build with modules hands in; ``allowed_agents``
    is the general-purpose helper's scope. ``note_types`` are middleware classes
    a helper carries whose notes go just inside DeliveredToolResults (the
    calc helper's ``FigureLooks``).
    """
    call_scope = call_scope or {}
    # Scratch-filesystem guard on every agent in the build (field feedback
    # 2026-09-15, N1/N12): deepagents' ls/read_file/grep/glob/write_file/
    # edit_file see only scratch space and answered "not found" for real
    # files. Each named sub-agent gets its own instance; deepagents'
    # general-purpose sub-agent is re-declared with its stock description and
    # prompt so it carries the guard too (naming a spec "general-purpose" is
    # how deepagents lets a caller replace it; without "tools" it inherits
    # the primary's tools, as the stock one does).
    # An agent with call_agent also gets the unknown-tool hint: a module
    # method called as if it were a tool is answered with the call_agent form
    # (live smoke wave 1, A15g), scoped to the modules that agent reaches.
    for spec in subagents:
        if "runnable" in spec:
            # A compiled helper (report ingest, the report library) is run
            # by deepagents AS PROVIDED: middleware on its spec reaches
            # nothing (BACKLOG RI-02). Its ceilings are enforced inside its
            # own graph, and its answer comes back as the primary's `task`
            # result, which the primary's own DeliveredToolResults,
            # ToolErrorGuard and LargeResultKeyMap already wrap.
            continue
        # DeliveredToolResults first: it logs what the guards below made of
        # each result, as the model receives it (live smoke wave 1, G11).
        # ToolErrorGuard next: a tool that raises is a tool error, not the
        # end of the turn (wave 2b, C3); OutputLimitGuard last: a reply cut
        # at the output limit is caught before anything acts on it (C5).
        # A helper's own result notes (the calc helper's FigureLooks) go
        # just inside DeliveredToolResults, so the activity log shows them.
        notes = [m for m in spec.get("middleware") or []
                 if note_types and isinstance(m, tuple(note_types))]
        if notes:
            spec["middleware"] = [m for m in spec["middleware"]
                                  if m not in notes]
        # A helper's .json write_file is saved on the real disk with its
        # own save_file, where write_diggs can read it (RI-31); a helper
        # without "tools" has the primary's.
        extra_mw = [DeliveredToolResults(), *notes, ToolErrorGuard(),
                    LargeResultKeyMap(),
                    ScratchFilesystemGuard(_save_tool(
                        spec["tools"] if "tools" in spec else tools))]
        if not any(isinstance(m, HideTools) and UNRUNNABLE_TOOLS <= m.hidden
                   for m in spec.get("middleware") or []):
            # deepagents' shell and delete cannot run on this backend
            # (GM-176); a helper that already hides them keeps its own.
            extra_mw.append(HideTools(UNRUNNABLE_TOOLS))
        # What the helper reads and writes, recorded as it happens, so
        # a helper that stops can hand it back as a file (TA-21).
        extra_mw.insert(0, HelperRecorder())
        retry = auth_retry()
        if retry is not None:
            extra_mw.insert(0, retry)
        # A busy model is waited for on a ladder, not the end of the
        # helper (TA-20; field session 2026-10-14).
        extra_mw.insert(0, AgentBusyRetry())
        scope = call_scope.get(spec.get("name"))
        if scope and unknown_tool_hint is not None:
            extra_mw.append(unknown_tool_hint(scope))
        extra_mw.append(OutputLimitGuard())
        spec["middleware"] = list(spec.get("middleware") or []) + extra_mw
    if (_GENERAL_PURPOSE_SPEC is not None
            and not any(s.get("name") == "general-purpose" for s in subagents)):
        # A model-call budget whose last call answers from what the helper
        # has (TA-22): the field session's helpers ran with none.
        _gp_budget = helper_max_model_calls()
        subagents.append({**_GENERAL_PURPOSE_SPEC,
                          "system_prompt": (
                              _GENERAL_PURPOSE_SPEC.get("system_prompt", "")
                              + HELPER_PARTIAL_RULE),
                          "middleware": ([ModelCallBudgetMiddleware(
                                              _gp_budget,
                                              final_turn_nudge=(
                                                  HELPER_FINAL_NUDGE),
                                              exhausted_message=(
                                                  HELPER_EXHAUSTED))]
                                         if _gp_budget else [])
                          + [AgentBusyRetry()]
                          + [m for m in (auth_retry(),) if m is not None]
                          + [HelperRecorder(),
                             DeliveredToolResults(),
                             ToolErrorGuard(),
                             LargeResultKeyMap(),
                             ScratchFilesystemGuard(_save_tool(tools)),
                             HideTools(UNRUNNABLE_TOOLS)]       # GM-176
                          + ([unknown_tool_hint(allowed_agents)]
                             if allowed_agents
                             and unknown_tool_hint is not None else [])
                          + [OutputLimitGuard()]})
    # A helper's own model (INF-41): set on its spec, which deepagents
    # builds the helper with in place of the primary's model.
    helper_models = {}
    for spec in subagents:
        name = spec.get("name") if isinstance(spec, dict) else None
        chosen = (subagent_models or {}).get(name)
        if chosen is None or "runnable" in spec:
            continue
        spec["model"] = chosen
        from aec_agent_core.deep.busy_retry import engine_facts
        helper_models[str(name)] = (
            chosen if isinstance(chosen, str)
            else engine_facts(chosen).get("model") or type(chosen).__name__)
    return helper_models


# ---------------------------------------------------------------------------
# The main agent: its middleware, the compile and the build record
# ---------------------------------------------------------------------------

def create_kwargs_for(kwargs: dict, *, model, store=None, checkpointer=None,
                      enable_memory: bool = False,
                      enable_summarization: bool = False,
                      summarization_model=None,
                      summarization_trigger=_AUTO_SUMMARIZATION_TRIGGER,
                      summarization_keep=_DEFAULT_SUMMARIZATION_KEEP,
                      figure_tasks=None, allowed_agents=None,
                      unknown_tool_hint: Optional[Callable] = None,
                      page_middleware=()) -> dict:
    """``create_deep_agent``'s keyword arguments for a build's main agent:
    ``kwargs`` (the caller's own), the persistent memory, and the
    middleware chain. ``page_middleware`` is the page's own, after the
    unknown-tool hint (the review page hides the
    scratch filesystem and keeps its model-call budget there; the geotech
    page hides the shell and delete)."""
    # ----- Phase-3: persistent /memories/ backend + memory= source -----
    create_kwargs = dict(kwargs)
    backend = None
    if enable_memory:
        backend = build_memory_backend(store=store)
        # Register the durable AGENTS.md as a deepagents memory source so its
        # contents load into the system prompt at the start of every thread.
        # (`memory=` may already be in kwargs; respect an explicit override.)
        create_kwargs.setdefault("memory", [MEMORIES_AGENTS_FILE])
        create_kwargs["backend"] = backend

    # ----- Phase-3: explicit summarization middleware -----
    middleware = list(create_kwargs.pop("middleware", []) or [])
    if enable_summarization:
        middleware.append(
            build_summarization_middleware(
                summarization_model if summarization_model is not None else model,
                backend=backend,
                trigger=summarization_trigger,
                keep=summarization_keep,
            )
        )
    # A busy model (rate limit, overload) is waited for on a ladder --
    # retry-after, else 15/30/60/120 s -- within a bound per turn, never a
    # spent budget (TA-20; field session 2026-10-14: one refusal ended a
    # 51-minute turn).
    middleware.append(AgentBusyRetry())
    # A model call refused at sign-in (a 401) is asked once more before the
    # turn ends on it (BACKLOG TA-01; field session 2026-09-09).
    retry = auth_retry()
    if retry is not None:
        middleware.append(retry)
    # First of the app's tool middleware: logs each result as the model
    # receives it, after the guards below (live smoke wave 1, G11).
    middleware.append(DeliveredToolResults())
    # A tool that raises is a tool error the model reads, not the end of the
    # turn (live smoke wave 2b, C3).
    middleware.append(ToolErrorGuard())
    # A large JSON result opens with a map of its keys, so the preview of
    # one moved to scratch says where each key is (TA-30).
    middleware.append(LargeResultKeyMap())
    # Just inside it: a `task` helper that stops hands back what it read as
    # a file, at most GEOTECH_MAX_HELPERS helpers run at once, and very
    # large results are kept whole for a turn that errors (TA-19, TA-21,
    # TA-22; field session 2026-10-14).
    middleware.append(KeepWork())
    if figure_tasks is not None:
        # Inside KeepWork, whose record of the figure helper it reads.
        middleware.append(figure_tasks)
    middleware.append(ScratchFilesystemGuard())
    if allowed_agents and unknown_tool_hint is not None:
        middleware.append(unknown_tool_hint(allowed_agents))
    middleware.extend(page_middleware or ())
    # Last: a reply cut at the output-token limit is caught before anything
    # acts on it -- a cut tool call is not run, cut text is continued
    # (wave 2b, C5).
    middleware.append(OutputLimitGuard())
    create_kwargs["middleware"] = middleware

    if store is not None:
        create_kwargs["store"] = store
    if checkpointer is not None:
        create_kwargs["checkpointer"] = checkpointer
    return create_kwargs


def compile_agent(create: Callable, *, model, tools, system_prompt,
                  subagents, create_kwargs: dict):
    """Compile the main agent with ``create`` (``create_deep_agent``; a
    builder passes its own name for it, so a test that patches that name
    still sees the call).

    Planning guarantee: the domain prompt tells the model to plan with
    ``write_todos``, so that tool MUST exist. deepagents <= 0.7.1 attaches
    the todo middleware automatically; newer versions don't — attach it
    explicitly and rebuild (cheap, offline) when it is missing. Checking the
    COMPILED agent (rather than version-sniffing) stays correct on both old
    and new deepagents without ever double-attaching.

    A deployment's instructions (``GEOTECH_INSTRUCTIONS``, REF-64 (B)) are
    appended here, last, to the main prompt and each helper's, once both
    pages have their final prompt and complete helper list; unset, nothing
    changes. What was appended rides on the agent as ``geotech_appended``
    for :func:`record_build`."""
    from aec_agent_core.deep import instructions as _instructions
    system_prompt, subagents, appended = _instructions.apply(
        system_prompt, subagents)
    agent = create(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        subagents=subagents,
        **create_kwargs,
    )
    if _TodoMiddleware is not None:
        try:
            _tool_names = set(agent.nodes["tools"].bound.tools_by_name)
        except Exception:
            _tool_names = None
        if _tool_names is not None and "write_todos" not in _tool_names:
            create_kwargs["middleware"] = (
                list(create_kwargs.get("middleware", []) or [])
                + [_TodoMiddleware()])
            agent = create(
                model=model,
                tools=tools,
                system_prompt=system_prompt,
                subagents=subagents,
                **create_kwargs,
            )
    if appended:
        try:
            agent.geotech_appended = appended
            # the main prompt as compiled, its block included, for
            # record_build's prompt_chars
            agent.geotech_prompt_chars = len(system_prompt or "")
        except Exception:                              # noqa: BLE001
            pass
    return agent


def record_build(agent, *, attachments, model, tools, subagents,
                 system_prompt, helper_models: dict,
                 judgment_consult: bool = False,
                 min_recursion_limit: Optional[int] = None):
    """Set what a host reads off the compiled agent; returns it."""
    # Expose the shared attachments dict so a UI (DeepNotebookChat FileUpload)
    # can add files that reach the already-built tools. Best-effort: if the
    # compiled object rejects attribute assignment, the caller can still pass
    # the same dict it handed to the builder.
    try:
        agent.geotech_attachments = attachments
        # The chat model, so a host can record which client served a turn
        # (BACKLOG EV-14; webapp.core.engine_record).
        agent.geotech_model = model
        # What this build offers, for the turn's record: the tools the
        # model is shown, its helpers and the switches on (EV-20; wave w8:
        # whether plot_code was offered had to be read from the code).
        try:
            _offered = sorted(agent.nodes["tools"].bound.tools_by_name)
        except Exception:                              # noqa: BLE001
            _offered = sorted(getattr(t, "name", str(t)) for t in tools)
        from aec_agent_core import review_flags as _rf
        agent.geotech_build = {
            "tools": _offered,
            "helpers": [str(s.get("name")) for s in subagents
                        if isinstance(s, dict)],
            "switches": _rf.describe(),
            "prompt_chars": len(system_prompt or ""),
        }
        appended = getattr(agent, "geotech_appended", None)
        if isinstance(appended, dict) and appended:
            # a deployment's instructions, per role (REF-64 (B)); the
            # prompt's size is then the compiled one, the main block in it
            agent.geotech_build["appended"] = appended
            chars = getattr(agent, "geotech_prompt_chars", None)
            if isinstance(chars, int):
                agent.geotech_build["prompt_chars"] = chars
        if helper_models:
            # the helpers on a model of their own (INF-41)
            agent.geotech_build["helper_models"] = helper_models
        # The turn-end judgment-call check (GM-174) reads this in
        # webapp.core.stream_turn (funhouse_agent.judgment_check.mode_for).
        agent.geotech_judgment_consult = judgment_consult
        if min_recursion_limit:
            # The host's step cap is raised to fit the budget, so the budget,
            # not a GraphRecursionError, ends a long request (webapp
            # core.stream_turn reads this, as for the lean build).
            agent.geotech_min_recursion_limit = min_recursion_limit
    except Exception:
        pass
    return agent


__all__ = [
    "MEMORIES_ROUTE", "MEMORIES_AGENTS_FILE", "HELPER_PARTIAL_RULE",
    "build_memory_backend", "build_summarization_middleware",
    "GeotechSummarizationMiddleware", "guard_helpers", "create_kwargs_for",
    "compile_agent", "record_build",
]

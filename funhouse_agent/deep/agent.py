"""``build_deep_agent`` — the v5.0 deepagents port of GeotechAgent.

Wires the v1 tool surface and sub-agent design onto
``deepagents.create_deep_agent`` (deepagents 0.6.8):

* **Primary agent** — scoped to ``ANALYSIS_MODULES`` (its direct tool surface
  stays small, mirroring v1). It reaches the reference modules ONLY by
  delegating to the ``references`` sub-agent via the deepagents-native
  ``task`` tool — that is the v5 replacement for v1's ``consult_references``.
* **references sub-agent** — the reference librarian. Scoped to
  ``REFERENCE_MODULES``, prompted with ``reviewer.CONSULTANT_FRAMING``, and
  given the ``read_reference_figure`` vision tool so it can read values off
  design charts (DM7/GEC/UFC/micropile/FEMA).
* **reviewer sub-agent** — checks the primary's work against the references.
  Scoped to ``REFERENCE_MODULES``, prompted with
  ``reviewer.REVIEWER_SYSTEM_PROMPT``.

The real ``create_deep_agent`` signature (deepagents 0.6.8) targeted here::

    create_deep_agent(model=None, tools=None, *, system_prompt=None,
                      middleware=(), subagents=None, ...)

Sub-agents follow the ``SubAgent`` TypedDict
(``{name, description, system_prompt, tools?, model?, ...}``).
"""

from typing import Callable, Optional

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.middleware import SummarizationMiddleware

from funhouse_agent.dispatch import ANALYSIS_MODULES, REFERENCE_MODULES
from funhouse_agent.reviewer import CONSULTANT_FRAMING, REVIEWER_SYSTEM_PROMPT

from funhouse_agent.deep.limits import (
    DEFAULT_REFERENCES_MAX_MODEL_CALLS,
    ModelCallBudgetMiddleware,
)
from funhouse_agent.deep.prompt import build_domain_prompt
from funhouse_agent.deep.setup_agent import build_setup_subagent
from funhouse_agent.deep.tools import (
    DEFAULT_MAX_RESULT_CHARS,
    make_core_tools,
    make_vision_tools,
)
from funhouse_agent.deep.vision_engine import LangChainVisionEngine


#: Concision instruction appended to the references sub-agent's framing. The
#: sub-agent's reference-text dumps are the main token-bloat source, so it is
#: told to return only the asked-for value(s) plus the citation rather than
#: pasting long chapter passages — this shrinks both its returned text and what
#: the primary then re-reads each round.
_REFERENCES_CONCISION = (
    "\n\nBe concise. Return ONLY the specific value(s)/provision asked for "
    "with the exact citation (reference + section/table/figure). Do NOT paste "
    "long chapter-text passages or restate full sections — a few sentences "
    "plus the citation is ideal."
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


# ---------------------------------------------------------------------------
# Sub-agent descriptions (what the primary uses to decide when to delegate)
# ---------------------------------------------------------------------------

_REFERENCES_DESCRIPTION = (
    "The geotechnical reference librarian. Delegate to it whenever you need a "
    "code provision, a recommended method, a parameter range, a required "
    "factor of safety, or a value read off a design chart. It searches the "
    "design references (NAVFAC DM7, the FHWA GEC series, UFC, micropile, FEMA, "
    "NOAA — chapter text, tables, equations, and figure charts), cites the "
    "specific reference and section/table/figure number, and reads chart "
    "values off the actual figures. You do NOT have the reference modules "
    "directly — delegating to this sub-agent is the only way to reach them."
)

_REVIEWER_DESCRIPTION = (
    "A senior geotechnical reviewer. Delegate to it AFTER you have produced a "
    "draft answer that involved calculations, to check the methodology, "
    "parameter ranges, required factors of safety, and any missing code "
    "provisions against the published references (DM7, GECs, UFCs, FEMA). It "
    "returns a structured PASS / FLAG / REVISE review with citations; it does "
    "not run computations itself."
)


def build_references_subagent(
    engine=None,
    attachments=None,
    save_fn: Optional[Callable] = None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    reference_result_chars: Optional[int] = None,
    max_model_calls: Optional[int] = DEFAULT_REFERENCES_MAX_MODEL_CALLS,
) -> dict:
    """Build the ``references`` sub-agent spec (reference librarian).

    Scoped to ``REFERENCE_MODULES`` core tools plus the ``read_reference_figure``
    vision tool (so it can read values off design charts). The vision tool
    needs an engine to actually call the model; offline it returns a clear
    "vision not available" error rather than raising.

    Parameters
    ----------
    engine, attachments, save_fn
        Forwarded to :func:`~funhouse_agent.deep.tools.make_vision_tools` for
        the figure read-off tool.
    max_result_chars : int, optional
        Tool-result size cap forwarded to ``make_core_tools`` /
        ``make_vision_tools`` (default ``8000``). This bounds what is re-sent
        each round. The sub-agent's framing is also extended with a concision
        instruction so it returns only the asked-for value(s) plus the
        citation.
    reference_result_chars : int, optional
        LARGER cap for reference reads (``call_agent`` on a reference module +
        ``read_reference_figure``) — the reference text is the payload.
        Defaults to ``DEFAULT_REFERENCE_RESULT_CHARS`` (16000); see
        :func:`~funhouse_agent.deep.tools.make_core_tools`.
    max_model_calls : int, optional
        The v5.1 round budget — THE key cost lever (26/76 reference questions
        burned 69% of all tokens in rc7 because the sub-agent re-sent its
        accumulated context across unbounded internal model calls). Attaches a
        :class:`~funhouse_agent.deep.limits.ModelCallBudgetMiddleware` to the
        sub-agent so each consult gets at most this many model calls; the last
        budgeted call is forced to summarize-and-answer from what was already
        gathered (graceful — a final answer is always returned, never an
        error). Defaults to ``8``; ``None``/``0`` disables the budget.
    """
    tools = make_core_tools(
        allowed_agents=REFERENCE_MODULES,
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
    ) + make_vision_tools(
        engine=engine,
        attachments=attachments,
        save_fn=save_fn,
        include={"read_reference_figure"},
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
    )
    spec = {
        "name": "references",
        "description": _REFERENCES_DESCRIPTION,
        "system_prompt": CONSULTANT_FRAMING + _REFERENCES_CONCISION,
        "tools": tools,
    }
    if max_model_calls:
        # deepagents appends SubAgent-spec middleware to the sub-agent's
        # default stack (see create_deep_agent's subagent processing).
        spec["middleware"] = [ModelCallBudgetMiddleware(max_model_calls)]
    return spec


# ---------------------------------------------------------------------------
# A2: calc sub-agent — context isolation for tool-heavy calculation work
# ---------------------------------------------------------------------------
# Measured (module_work/A2_CONTEXT_DESIGN.md): the ~10k/turn of calc-package /
# method-dump / reference output is what makes a persist+replay chat grow ~84%
# more expensive over 10 turns. Delegating the tool-heavy calc to a sub-agent
# keeps that bulky trace in the sub-agent's own context; only a compact result
# (values + units + method + saved-file path) returns to the main thread — which
# is what replays forward. NO DATA LOSS: the sub-agent saves the full payload to
# a file, so the detail stays retrievable on demand.

#: The calc sub-agent's model-call budget (a calc chain — multiple methods +
#: a package — runs longer than a reference lookup, so a bit above the 8 that
#: bounds the references consultant). ``None``/``0`` disables the budget.
DEFAULT_CALC_MAX_MODEL_CALLS = 16

_CALC_DESCRIPTION = (
    "The calculation engine. Delegate TOOL-HEAVY calculation work to it: running "
    "an analysis method (bearing capacity, settlement, slope stability, FEM, "
    "liquefaction, downdrag, …), a parameter sweep, or building a calc package. "
    "It runs the method(s), returns the KEY numeric results (governing value(s) + "
    "units + the method used + any FoS/utilization) and the saved calc-package / "
    "plot path, and keeps the bulky intermediate tool output in its OWN context "
    "instead of yours — so a long chat does not carry every calc dump forward."
)

_CALC_FRAMING = (
    "\n\nYou are the calculation engine for a geotechnical agent. Run the "
    "requested analysis method(s) with the given inputs, then reply with a "
    "COMPACT result the delegating agent can carry forward WITHOUT re-running the "
    "calc: the governing value(s) with units, the method/standard used, the key "
    "inputs, and any factor of safety / utilization. Do NOT paste the full method "
    "dump or calc-package text into your reply.\n"
    "NO DATA LOSS: whenever you produce a large result (a calc package, a full "
    "method dump, a big table, a plot), SAVE it to a file — pass an output_path to "
    "the tool, or use save_file — and include the saved path in your reply, so the "
    "full detail can be re-read on demand. Never discard a result you were asked "
    "to compute: summarize it and point to the saved file."
)

#: Appended to the PRIMARY agent's system prompt when the calc sub-agent is on,
#: nudging it to delegate tool-heavy calc rather than run those tools inline.
_CALC_DELEGATION_NUDGE = (
    "CONTEXT DISCIPLINE: a `calc` sub-agent is available. For any tool-heavy "
    "calculation — running an analysis method, a parameter sweep, or building a "
    "calc package — DELEGATE it to `calc` instead of running those tools yourself, "
    "so the bulky intermediate output stays out of this conversation. `calc` "
    "returns the key values + units + method and the saved file path; carry those "
    "forward and do the reasoning and final synthesis here. (A prior calc's values "
    "stay in this conversation, so re-use them rather than recomputing.)"
)


def build_calc_subagent(
    engine=None,
    attachments=None,
    save_fn: Optional[Callable] = None,
    allowed_agents=None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    reference_result_chars: Optional[int] = None,
    max_model_calls: Optional[int] = DEFAULT_CALC_MAX_MODEL_CALLS,
) -> dict:
    """Build the ``calc`` sub-agent spec (A2 context isolation).

    Scoped to the analysis modules (``allowed_agents``, default
    :data:`ANALYSIS_MODULES`) plus ``save_file`` so it can persist the full
    payload. Mirrors :func:`build_references_subagent`: a concise, no-data-loss
    framing plus a :class:`~funhouse_agent.deep.limits.ModelCallBudgetMiddleware`
    budget. The sub-agent runs the tool-heavy calc in its OWN context and returns
    only a compact result (values + units + method + saved-file path) to the
    delegating agent.
    """
    allowed = frozenset(ANALYSIS_MODULES if allowed_agents is None
                        else allowed_agents)
    tools = make_core_tools(
        allowed_agents=allowed,
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
    ) + make_vision_tools(
        engine=engine,
        attachments=attachments,
        save_fn=save_fn,
        include={"save_file"},
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
    )
    spec = {
        "name": "calc",
        "description": _CALC_DESCRIPTION,
        "system_prompt": CONSULTANT_FRAMING + _CALC_FRAMING,
        "tools": tools,
    }
    if max_model_calls:
        spec["middleware"] = [ModelCallBudgetMiddleware(max_model_calls)]
    return spec


def build_reviewer_subagent(
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    reference_result_chars: Optional[int] = None,
) -> dict:
    """Build the ``reviewer`` sub-agent spec.

    Scoped to ``REFERENCE_MODULES`` core tools (reference lookup only — the
    reviewer checks work, it does not recompute).

    Parameters
    ----------
    max_result_chars : int, optional
        Tool-result size cap forwarded to ``make_core_tools`` (default
        ``8000``), so the reviewer's reference lookups are bounded too.
    reference_result_chars : int, optional
        Larger cap for the reviewer's reference reads; see
        :func:`build_references_subagent`.
    """
    return {
        "name": "reviewer",
        "description": _REVIEWER_DESCRIPTION,
        "system_prompt": REVIEWER_SYSTEM_PROMPT,
        "tools": make_core_tools(
            allowed_agents=REFERENCE_MODULES,
            max_result_chars=max_result_chars,
            reference_result_chars=reference_result_chars,
        ),
    }


def build_primary_tools(
    allowed_agents=None,
    engine=None,
    attachments=None,
    save_fn: Optional[Callable] = None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    reference_result_chars: Optional[int] = None,
) -> list:
    """Build the primary agent's tool list.

    Core dispatch tools scoped to ``allowed_agents`` (defaults to
    ``ANALYSIS_MODULES``) plus the vision/file-output tools. The reference
    modules are intentionally absent — the primary reaches them via the
    ``references`` sub-agent (the ``task`` tool).

    Parameters
    ----------
    max_result_chars : int, optional
        Tool-result size cap forwarded to ``make_core_tools`` /
        ``make_vision_tools`` (default ``8000``, mirroring v1).
    reference_result_chars : int, optional
        Larger cap for reference reads (here only ``read_reference_figure``,
        since the primary's ``call_agent`` scope excludes the reference
        modules); see :func:`~funhouse_agent.deep.tools.make_core_tools`.
    """
    if allowed_agents is None:
        allowed_agents = ANALYSIS_MODULES
    return make_core_tools(
        allowed_agents=allowed_agents,
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
    ) + make_vision_tools(
        engine=engine,
        attachments=attachments,
        save_fn=save_fn,
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
    )


def build_deep_agent(
    model,
    *,
    allowed_agents=None,
    reference_mode: str = "anytime",
    extra_system_prompt: Optional[str] = None,
    save_fn: Optional[Callable] = None,
    engine=None,
    attachments=None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    reference_result_chars: Optional[int] = None,
    references_max_model_calls: Optional[int] = DEFAULT_REFERENCES_MAX_MODEL_CALLS,
    enable_calc_subagent: bool = False,
    calc_max_model_calls: Optional[int] = DEFAULT_CALC_MAX_MODEL_CALLS,
    enable_setup_agent: bool = False,
    setup_store=None,
    setup_render_dir: Optional[str] = None,
    store=None,
    checkpointer=None,
    enable_memory: bool = False,
    enable_summarization: bool = False,
    summarization_model=None,
    summarization_trigger=_AUTO_SUMMARIZATION_TRIGGER,
    summarization_keep=_DEFAULT_SUMMARIZATION_KEEP,
    **kwargs,
):
    """Construct the deepagents port of the geotech agent.

    By default this agent already has — courtesy of deepagents — *planning*
    (the ``write_todos`` tool, via ``TodoListMiddleware``) and an in-thread
    *scratch filesystem* (``ls`` / ``read_file`` / ``write_file`` /
    ``edit_file``, via ``FilesystemMiddleware`` on an ephemeral
    :class:`StateBackend`). Those are ON with no configuration. The Phase-3
    parameters below add the *cross-session* and *durability* capabilities.

    Parameters
    ----------
    model : str | BaseChatModel
        The chat model for ``create_deep_agent`` (e.g. a
        ``langchain_anthropic.ChatAnthropic`` instance, a ``"provider:model"``
        string, or a fake chat model for offline tests).
    allowed_agents : iterable of str, optional
        Direct-call scope for the PRIMARY agent. Defaults to
        ``ANALYSIS_MODULES`` (keeps the primary's tool surface small; reference
        access is delegated to the ``references`` sub-agent). Pass an explicit
        set to override.
    reference_mode : str
        Retained from v1 for call-site compatibility. In v5 reference access is
        always available through the ``references`` sub-agent (the ``task``
        tool), so this currently only gates whether that sub-agent is attached:
        ``"off"`` omits it, anything else attaches it. Defaults to
        ``"anytime"``.
    extra_system_prompt : str, optional
        An extra block appended to the domain system prompt. Lets a caller
        specialize the agent without forking the prompt machinery — e.g. the
        seismic reviewer (``funhouse_agent.reviewers.make_seismic_reviewer_deep``)
        injects its review-mode checklist here. Default ``None`` leaves the
        prompt unchanged.
    save_fn : callable, optional
        ``(path, content) -> saved_path`` for ``save_file``. Defaults to local
        filesystem write.
    engine : GenAIEngine, optional
        Vision engine for ``analyze_image`` / ``analyze_pdf_page`` /
        ``read_reference_figure``. If ``None`` AND ``model`` is a chat-model
        object (not a ``"provider:model"`` string), it is default-wrapped with
        :class:`~funhouse_agent.deep.vision_engine.LangChainVisionEngine` so the
        vision tools route through the SAME model object — one model drives both
        text and vision, no separate Anthropic/OpenAI SDK client needed. Pass an
        explicit ``engine`` (a Funhouse engine / ``ClaudeEngine`` /
        ``NativeToolEngine``) to override; it is used unchanged. ``None`` with a
        string ``model`` leaves vision unwired (those tools return a clear error
        if invoked).
    attachments : dict, optional
        ``{key: bytes}`` of attached files for the vision tools.
    max_result_chars : int, optional
        Single knob capping the size of EVERY tool result fed back to the model
        — the primary's tools AND both sub-agents' (references, reviewer). It is
        forwarded to ``build_primary_tools`` and both sub-agent builders.
        Defaults to ``8000`` (mirrors v1's ``GeotechAgent._max_result_chars``);
        set ``0`` to disable truncation. This is the token-bloat guard: the
        references sub-agent's large reference-text dumps would otherwise be
        re-sent on every round and compound the input-token cost.
    reference_result_chars : int, optional
        LARGER cap for reference reads (``call_agent`` on a reference module
        inside the references/reviewer sub-agents, plus
        ``read_reference_figure``), where the reference text IS the payload.
        Defaults to 16000 (never below ``max_result_chars``); numeric/calc
        ``call_agent`` results are uncapped (they are tiny). Disabled along
        with everything else when ``max_result_chars <= 0``.
    references_max_model_calls : int, optional
        Per-consult model-call budget for the ``references`` sub-agent — the
        v5.1 single biggest cost lever (rc7: 26/76 reference questions burned
        69% of all tokens via unbounded internal rounds). The last budgeted
        call is forced to summarize-and-answer from what was gathered, so a
        final answer is ALWAYS returned (never an error). Defaults to ``8``;
        ``None``/``0`` disables the budget.
    enable_calc_subagent : bool
        Attach the ``calc`` sub-agent (A2 context isolation) and nudge the primary
        to delegate tool-heavy calculation to it, so the bulky calc-package /
        method-dump trace stays in the sub-agent's context instead of the main
        conversation (which is what a persist+replay chat re-sends each turn).
        The sub-agent returns a compact result (values + units + method + saved
        file path) and saves the full payload to a file (no data loss). OFF by
        default (additive / default-preserving — the library and the eval suite
        are unchanged); the web app turns it ON per conversation.
    calc_max_model_calls : int, optional
        Per-delegation model-call budget for the ``calc`` sub-agent (as
        ``references_max_model_calls`` is for references). Defaults to ``16``;
        ``None``/``0`` disables the budget.
    enable_setup_agent : bool
        Attach the ``model_setup`` sub-agent (staged LE/FEM model building
        with human confirmation gates and echo-back renders — see
        :mod:`funhouse_agent.deep.setup_agent`). OFF by default: model setup
        is an opt-in, session-scoped workflow whose tools carry mutable
        per-build state (the ProjectStore), so hosts that only ask
        calculation questions never pay for the extra tool surface.
    setup_store : ProjectStore, optional
        The shared project state for the setup sub-agent (created fresh when
        omitted). Pass your own to inspect ``setup_store.project`` from the
        host. Only used when ``enable_setup_agent`` is on.
    setup_render_dir : str, optional
        Directory the setup sub-agent writes echo-back PNGs to (default
        ``model_setup_renders/``). Only used when ``enable_setup_agent`` is
        on.
    store : BaseStore, optional
        A LangGraph store (e.g. ``InMemoryStore`` / a Postgres store) for
        cross-session / persistent memory. Passing a store turns ``enable_memory``
        on automatically — files the agent writes under ``/memories/`` are routed
        to this store and therefore survive across threads, while ordinary scratch
        files stay ephemeral in per-thread state. Forwarded to
        ``create_deep_agent(store=...)``.
    checkpointer : Checkpointer, optional
        A LangGraph checkpointer (e.g. ``InMemorySaver`` / a Postgres saver) for
        durable, resumable threads. With a checkpointer, an interrupted run can be
        resumed and a thread's full state (messages, todos, scratch files) is
        persisted under its ``thread_id``. Forwarded to
        ``create_deep_agent(checkpointer=...)``.
    enable_memory : bool
        Turn on the persistent ``/memories/`` filesystem route. When on (or when a
        ``store`` is given), the filesystem backend becomes
        ``CompositeBackend(default=StateBackend(), routes={"/memories/":
        StoreBackend()})`` and a deepagents ``memory=[/memories/AGENTS.md]`` source
        is registered so any durable project context the agent has saved is loaded
        into the system prompt at the start of every thread. Requires a ``store``
        at runtime for the route to actually persist (``create_deep_agent`` wires
        the store into ``StoreBackend`` via ``get_store()``). Defaults to ``False``.
    enable_summarization : bool
        Append a second, explicitly-configured
        :class:`GeotechSummarizationMiddleware` so long sessions auto-compact at
        our chosen threshold. NOTE: deepagents already attaches a summarizer with
        model-aware defaults; this adds a custom-tuned one alongside it (they share
        state and cannot collide because the subclass reports a distinct name).
        Defaults to ``False``.
    summarization_model : str | BaseChatModel, optional
        Model used to *write* the summaries when ``enable_summarization`` is on.
        Defaults to the primary ``model``. (If ``model`` is a ``"provider:model"``
        string, that string is used as-is.)
    summarization_trigger : ContextSize, optional
        When to summarize. Defaults to auto: ``("fraction", 0.8)`` (compact at
        ~80% of the model's input window) when the model publishes its window,
        else an absolute ``("tokens", 170000)`` fallback for profile-less models
        (string specs / fakes) so construction never fails.
    summarization_keep : ContextSize
        How much recent history to keep verbatim after a summarization. Defaults
        to ``("messages", 8)`` so the latest exchanges survive intact.
    **kwargs
        Forwarded verbatim to ``create_deep_agent`` (e.g. ``name``, ``debug``,
        ``permissions``).

    Returns
    -------
    CompiledStateGraph
        The compiled deep agent.

    Notes
    -----
    With NO Phase-3 arguments this builds exactly as before: no ``store``, no
    ``checkpointer``, the default ephemeral ``StateBackend`` filesystem, and only
    the deepagents-default summarizer. Planning (``write_todos``) and the scratch
    filesystem tools are present regardless because deepagents attaches them by
    default.
    """
    if allowed_agents is None:
        allowed_agents = ANALYSIS_MODULES

    # A store implies persistent memory: there is no point handing in a store
    # without routing /memories/ to it.
    if store is not None:
        enable_memory = True

    # Default-wrap the model for vision when no explicit engine is given and the
    # model is a chat-model object (not a "provider:model" string). This lets a
    # single model object satisfy both text and vision; an explicit engine
    # (Funhouse engine / ClaudeEngine / NativeToolEngine) is used unchanged.
    if engine is None and not isinstance(model, str):
        engine = LangChainVisionEngine(model)

    # ONE shared attachments dict for the whole agent: the primary vision tools
    # AND the references sub-agent close over the SAME object, and it is exposed
    # on the returned agent (below) so a notebook FileUpload can add files after
    # the build reaches every tool. (Previously a None default made each
    # make_vision_tools create its own empty dict — added files reached nothing.)
    attachments = {} if attachments is None else attachments

    tools = build_primary_tools(
        allowed_agents=allowed_agents,
        engine=engine,
        attachments=attachments,
        save_fn=save_fn,
        max_result_chars=max_result_chars,
        reference_result_chars=reference_result_chars,
    )

    system_prompt = build_domain_prompt(allowed_agents, memory_enabled=enable_memory)
    if extra_system_prompt:
        # Appended AFTER the domain prompt (mirrors GeotechAgent.system_prompt_extra)
        # so a caller can re-cast the agent — e.g. the seismic reviewer's review-mode
        # checklist (funhouse_agent.reviewers.make_seismic_reviewer_deep). Default
        # None leaves the prompt unchanged.
        system_prompt = system_prompt + "\n\n" + extra_system_prompt
    if enable_calc_subagent:
        # A2: nudge the primary to delegate tool-heavy calc to the `calc`
        # sub-agent so the bulky trace stays out of the main conversation.
        system_prompt = system_prompt + "\n\n" + _CALC_DELEGATION_NUDGE

    subagents = []
    if reference_mode != "off":
        subagents.append(
            build_references_subagent(
                engine=engine,
                attachments=attachments,
                save_fn=save_fn,
                max_result_chars=max_result_chars,
                reference_result_chars=reference_result_chars,
                max_model_calls=references_max_model_calls,
            )
        )
        subagents.append(
            build_reviewer_subagent(
                max_result_chars=max_result_chars,
                reference_result_chars=reference_result_chars,
            )
        )
    if enable_calc_subagent:                                   # A2 context isolation
        subagents.append(
            build_calc_subagent(
                engine=engine,
                attachments=attachments,
                save_fn=save_fn,
                allowed_agents=allowed_agents,
                max_result_chars=max_result_chars,
                reference_result_chars=reference_result_chars,
                max_model_calls=calc_max_model_calls,
            )
        )
    if enable_setup_agent:
        subagents.append(
            build_setup_subagent(
                store=setup_store,
                render_dir=setup_render_dir,
                max_result_chars=max_result_chars,
            )
        )

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
    if middleware:
        create_kwargs["middleware"] = middleware

    if store is not None:
        create_kwargs["store"] = store
    if checkpointer is not None:
        create_kwargs["checkpointer"] = checkpointer

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        subagents=subagents,
        **create_kwargs,
    )
    # Expose the shared attachments dict so a UI (DeepNotebookChat FileUpload)
    # can add files that reach the already-built tools. Best-effort: if the
    # compiled object rejects attribute assignment, the caller can still pass
    # the same dict it handed to build_deep_agent.
    try:
        agent.geotech_attachments = attachments
    except Exception:
        pass
    return agent


def run_memory_demo(model, store, checkpointer=None, *, verbose: bool = True) -> bool:
    """Prove cross-session ``/memories/`` persistence end-to-end (REAL model).

    Builds a memory-enabled agent on ``model`` + ``store`` and runs TWO separate
    threads:

    * **Thread 1** is told a durable fact and asked to save it under
      ``/memories/`` (e.g. ``write_file('/memories/AGENTS.md', ...)``).
    * **Thread 2** — a *new* ``thread_id`` sharing the SAME ``store`` — is asked
      to recall the fact. Because ``/memories/`` is store-backed, thread 2 can
      read what thread 1 wrote even though the two threads share no message
      history.

    .. warning::
       This makes REAL LLM API calls. It is NOT part of the offline suite and is
       never executed by it — it exists so a notebook / live run can confirm the
       agent actually *uses* ``/memories/`` across sessions (the offline tests
       only prove the backend round-trip and the wiring).

    Parameters
    ----------
    model : BaseChatModel
        A real tool-calling chat model.
    store : BaseStore
        A LangGraph store shared across both threads (e.g. ``InMemoryStore``).
    checkpointer : Checkpointer, optional
        Optional durable checkpointer for the threads.
    verbose : bool
        Print the recall answer.

    Returns
    -------
    bool
        ``True`` if thread 2's answer references the fact saved in thread 1.
    """
    agent = build_deep_agent(
        model=model, store=store, checkpointer=checkpointer, enable_memory=True,
    )
    fact = "The project design groundwater table is at 3.5 m below grade."
    keyword = "3.5"

    agent.invoke(
        {"messages": [{"role": "user", "content": (
            f"Remember this for the whole project and save it to your memory: {fact}"
        )}]},
        config={"configurable": {"thread_id": "demo-thread-1"}},
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": (
            "What is the project design groundwater table depth? "
            "Check your memory if needed."
        )}]},
        config={"configurable": {"thread_id": "demo-thread-2"}},
    )

    messages = result.get("messages", []) if isinstance(result, dict) else []
    final = ""
    for m in reversed(messages):
        content = getattr(m, "content", "")
        text = content if isinstance(content, str) else "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in (content or [])
        )
        if text.strip():
            final = text
            break

    recalled = keyword in final
    if verbose:
        print("Thread-2 recall answer:", final)
        print("RESULT:", "PASS" if recalled else "FAIL")
    return recalled


__all__ = [
    "build_deep_agent",
    "build_primary_tools",
    "build_references_subagent",
    "build_calc_subagent",
    "build_reviewer_subagent",
    "build_setup_subagent",
    "build_memory_backend",
    "build_summarization_middleware",
    "run_memory_demo",
    "GeotechSummarizationMiddleware",
    "MEMORIES_ROUTE",
    "MEMORIES_AGENTS_FILE",
]

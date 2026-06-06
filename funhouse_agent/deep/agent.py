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

from funhouse_agent.dispatch import ANALYSIS_MODULES, REFERENCE_MODULES
from funhouse_agent.reviewer import CONSULTANT_FRAMING, REVIEWER_SYSTEM_PROMPT

from funhouse_agent.deep.prompt import build_domain_prompt
from funhouse_agent.deep.tools import make_core_tools, make_vision_tools
from funhouse_agent.deep.vision_engine import LangChainVisionEngine


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
) -> dict:
    """Build the ``references`` sub-agent spec (reference librarian).

    Scoped to ``REFERENCE_MODULES`` core tools plus the ``read_reference_figure``
    vision tool (so it can read values off design charts). The vision tool
    needs an engine to actually call the model; offline it returns a clear
    "vision not available" error rather than raising.
    """
    tools = make_core_tools(allowed_agents=REFERENCE_MODULES) + make_vision_tools(
        engine=engine,
        attachments=attachments,
        save_fn=save_fn,
        include={"read_reference_figure"},
    )
    return {
        "name": "references",
        "description": _REFERENCES_DESCRIPTION,
        "system_prompt": CONSULTANT_FRAMING,
        "tools": tools,
    }


def build_reviewer_subagent() -> dict:
    """Build the ``reviewer`` sub-agent spec.

    Scoped to ``REFERENCE_MODULES`` core tools (reference lookup only — the
    reviewer checks work, it does not recompute).
    """
    return {
        "name": "reviewer",
        "description": _REVIEWER_DESCRIPTION,
        "system_prompt": REVIEWER_SYSTEM_PROMPT,
        "tools": make_core_tools(allowed_agents=REFERENCE_MODULES),
    }


def build_primary_tools(
    allowed_agents=None,
    engine=None,
    attachments=None,
    save_fn: Optional[Callable] = None,
) -> list:
    """Build the primary agent's tool list.

    Core dispatch tools scoped to ``allowed_agents`` (defaults to
    ``ANALYSIS_MODULES``) plus the vision/file-output tools. The reference
    modules are intentionally absent — the primary reaches them via the
    ``references`` sub-agent (the ``task`` tool).
    """
    if allowed_agents is None:
        allowed_agents = ANALYSIS_MODULES
    return make_core_tools(allowed_agents=allowed_agents) + make_vision_tools(
        engine=engine, attachments=attachments, save_fn=save_fn,
    )


def build_deep_agent(
    model,
    *,
    allowed_agents=None,
    reference_mode: str = "anytime",
    save_fn: Optional[Callable] = None,
    engine=None,
    attachments=None,
    **kwargs,
):
    """Construct the deepagents port of the geotech agent.

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
    **kwargs
        Forwarded verbatim to ``create_deep_agent`` (e.g. ``checkpointer``,
        ``name``, ``debug``).

    Returns
    -------
    CompiledStateGraph
        The compiled deep agent.
    """
    if allowed_agents is None:
        allowed_agents = ANALYSIS_MODULES

    # Default-wrap the model for vision when no explicit engine is given and the
    # model is a chat-model object (not a "provider:model" string). This lets a
    # single model object satisfy both text and vision; an explicit engine
    # (Funhouse engine / ClaudeEngine / NativeToolEngine) is used unchanged.
    if engine is None and not isinstance(model, str):
        engine = LangChainVisionEngine(model)

    tools = build_primary_tools(
        allowed_agents=allowed_agents,
        engine=engine,
        attachments=attachments,
        save_fn=save_fn,
    )

    system_prompt = build_domain_prompt(allowed_agents)

    subagents = []
    if reference_mode != "off":
        subagents.append(
            build_references_subagent(
                engine=engine, attachments=attachments, save_fn=save_fn,
            )
        )
        subagents.append(build_reviewer_subagent())

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        subagents=subagents,
        **kwargs,
    )


__all__ = [
    "build_deep_agent",
    "build_primary_tools",
    "build_references_subagent",
    "build_reviewer_subagent",
]

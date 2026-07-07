"""Narrow reviewer agents — scoped, interactive geotechnical reviewers.

A narrow reviewer is a :class:`~funhouse_agent.agent.GeotechAgent` whose direct
tool surface is scoped (via ``allowed_agents``) to ONE geotechnical domain, and
whose system prompt is re-cast into "review mode" with a domain checklist. The
owner can chat with it independently:

    from funhouse_agent import make_seismic_reviewer, NativeToolEngine
    rev = make_seismic_reviewer(NativeToolEngine(fh_prompter))
    print(rev.ask("Review this liquefaction calc: ...").answer)

The FIRST reviewer is the **seismic reviewer** (v5.4 item D6). It is one of TWO
thin surfaces over a single shared playbook; the other is the Claude Code agent
``.claude/agents/seismic-reviewer.md``. Both pull their checklist from
``funhouse_agent/review_checklists.py`` so they cannot drift.

Scope sets live in ``funhouse_agent/dispatch.py`` (``SEISMIC_MODULES`` and
``SEISMIC_REFERENCES``); the checklist/preamble lives in
``funhouse_agent/review_checklists.py`` (``SEISMIC_REVIEWER_PREAMBLE``).
"""

from typing import Optional

from funhouse_agent.agent import GeotechAgent
from funhouse_agent.dispatch import SEISMIC_MODULES, SEISMIC_REFERENCES
from funhouse_agent.review_checklists import SEISMIC_REVIEWER_PREAMBLE

#: The seismic reviewer's full direct-call scope: the seismic analysis modules
#: it may RE-RUN to verify a number, plus the seismic reference modules it cites
#: from. (reference_db / figure_db inside SEISMIC_REFERENCES still search the
#: whole library, so seismic content in any reference stays reachable.)
SEISMIC_REVIEWER_SCOPE = frozenset(SEISMIC_MODULES | SEISMIC_REFERENCES)


def make_seismic_reviewer(genai_engine, *, extra_modules=None, **kwargs):
    """Build the seismic geotechnical reviewer (Funhouse scoped sub-agent).

    Returns a :class:`~funhouse_agent.agent.GeotechAgent` scoped to the seismic
    analysis + reference modules and prompted in review mode with the shared
    seismic checklist. The reviewer can re-run the seismic tools to verify a
    calc and cite the seismic references directly.

    Parameters
    ----------
    genai_engine : GenAIEngine
        AI backend (e.g. ``NativeToolEngine(fh_prompter)``, ``ClaudeEngine()``).
    extra_modules : iterable of str, optional
        Additional module names to add to the reviewer's direct scope (e.g. a
        module you want it to also be able to run). Rarely needed.
    **kwargs
        Forwarded to ``GeotechAgent`` (``max_rounds``, ``temperature``,
        ``verbose``, ``save_fn``, ``on_tool_call`` …). ``allowed_agents``,
        ``reference_mode``, and ``system_prompt_extra`` are managed here and
        should not be overridden.

    Notes
    -----
    ``reference_mode="off"`` is set deliberately: the seismic references are
    already in the reviewer's direct scope, so the ``consult_references`` tool
    (which reaches ALL 21 references via a full-library sub-agent) would only
    dilute the seismic scoping. The reviewer looks references up directly.
    """
    allowed = set(SEISMIC_REVIEWER_SCOPE)
    if extra_modules:
        allowed |= set(extra_modules)
    # References are directly in scope → no consult sub-agent (it would reach the
    # whole reference library and undo the seismic scoping).
    kwargs.setdefault("reference_mode", "off")
    return GeotechAgent(
        genai_engine=genai_engine,
        allowed_agents=frozenset(allowed),
        system_prompt_extra=SEISMIC_REVIEWER_PREAMBLE,
        **kwargs,
    )


def make_seismic_reviewer_deep(model, *, extra_modules=None, **kwargs):
    """Build the seismic reviewer as a **deepagents** agent (v5.0 deep loop).

    The deep variant of :func:`make_seismic_reviewer`: same seismic scope and
    same review-mode checklist, but built on
    ``funhouse_agent.deep.build_deep_agent`` (planning + scratch filesystem +
    optional persistent memory). Returns a compiled deep-agent graph
    (``.invoke({"messages": [...]}, config=...)``), not a ``GeotechAgent``.

    Requires the optional ``[deep]`` extra (``deepagents``); the import is
    deferred so importing ``funhouse_agent`` never pulls in deepagents.

    Parameters
    ----------
    model : str | BaseChatModel
        Chat model for ``build_deep_agent`` (a LangChain chat model, a
        ``"provider:model"`` string, or a fake model for offline tests).
    extra_modules : iterable of str, optional
        Additional module names to add to the reviewer's direct scope.
    **kwargs
        Forwarded to ``build_deep_agent``. ``allowed_agents``,
        ``reference_mode``, and ``extra_system_prompt`` are managed here.
    """
    from funhouse_agent.deep.agent import build_deep_agent

    allowed = set(SEISMIC_REVIEWER_SCOPE)
    if extra_modules:
        allowed |= set(extra_modules)
    # reference_mode="off" → do not attach the whole-library references/reviewer
    # sub-agents; the seismic references are already in the primary's scope.
    kwargs.setdefault("reference_mode", "off")
    return build_deep_agent(
        model=model,
        allowed_agents=frozenset(allowed),
        extra_system_prompt=SEISMIC_REVIEWER_PREAMBLE,
        **kwargs,
    )


__all__ = [
    "make_seismic_reviewer",
    "make_seismic_reviewer_deep",
    "SEISMIC_REVIEWER_SCOPE",
]

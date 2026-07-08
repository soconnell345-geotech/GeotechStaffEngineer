"""Narrow reviewer agents — scoped, interactive geotechnical reviewers.

A narrow reviewer is a :class:`~funhouse_agent.agent.GeotechAgent` whose direct
tool surface is scoped (via ``allowed_agents``) to ONE geotechnical domain, and
whose system prompt is re-cast into "review mode" with a domain checklist. The
owner can chat with it independently:

    from funhouse_agent import make_seismic_reviewer, NativeToolEngine
    rev = make_seismic_reviewer(NativeToolEngine(fh_prompter))
    print(rev.ask("Review this liquefaction calc: ...").answer)

The family (v5.4 D6 + F8): **seismic** (the template), **foundations**,
**earth-retention**, and **slope / FEM**. Each is one of TWO thin surfaces over
a single shared playbook; the other is the Claude Code agent
``.claude/agents/<domain>-reviewer.md``. Both pull their checklist from
``funhouse_agent/review_checklists.py`` so they cannot drift.

Scope sets live in ``funhouse_agent/dispatch.py``
(``<DOMAIN>_MODULES`` / ``<DOMAIN>_REFERENCES``); the checklist/preamble lives in
``funhouse_agent/review_checklists.py`` (``<DOMAIN>_REVIEWER_PREAMBLE``).
"""

from funhouse_agent.agent import GeotechAgent
from funhouse_agent.dispatch import (
    SEISMIC_MODULES, SEISMIC_REFERENCES,
    FOUNDATIONS_MODULES, FOUNDATIONS_REFERENCES,
    EARTH_RETENTION_MODULES, EARTH_RETENTION_REFERENCES,
    SLOPE_FEM_MODULES, SLOPE_FEM_REFERENCES,
)
from funhouse_agent.review_checklists import (
    SEISMIC_REVIEWER_PREAMBLE,
    FOUNDATIONS_REVIEWER_PREAMBLE,
    EARTH_RETENTION_REVIEWER_PREAMBLE,
    SLOPE_FEM_REVIEWER_PREAMBLE,
)

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


# ---------------------------------------------------------------------------
# F8 reviewer family — foundations / earth-retention / slope-FEM
# ---------------------------------------------------------------------------
# Same two-surface pattern as the seismic reviewer (D6): a scoped Funhouse
# sub-agent + a Claude Code agent def, both pulling one shared checklist.

#: Foundations reviewer scope: shallow + deep foundation and ground-improvement
#: analysis modules, plus the foundation reference modules it cites from.
FOUNDATIONS_REVIEWER_SCOPE = frozenset(FOUNDATIONS_MODULES | FOUNDATIONS_REFERENCES)

#: Earth-retention reviewer scope: wall / excavation-support / M-O modules + refs.
EARTH_RETENTION_REVIEWER_SCOPE = frozenset(
    EARTH_RETENTION_MODULES | EARTH_RETENTION_REFERENCES)

#: Slope / FEM reviewer scope: slope-stability / FEM / reliability / ingest + refs.
SLOPE_FEM_REVIEWER_SCOPE = frozenset(SLOPE_FEM_MODULES | SLOPE_FEM_REFERENCES)


def _make_reviewer(scope, preamble, genai_engine, extra_modules=None, **kwargs):
    """Build a scoped Funhouse review sub-agent (shared F8 body).

    ``reference_mode="off"`` is deliberate: the domain references are already in
    the reviewer's direct scope, so the whole-library ``consult_references`` tool
    would only dilute the scoping. Set it explicitly to override.
    """
    allowed = set(scope)
    if extra_modules:
        allowed |= set(extra_modules)
    kwargs.setdefault("reference_mode", "off")
    return GeotechAgent(
        genai_engine=genai_engine,
        allowed_agents=frozenset(allowed),
        system_prompt_extra=preamble,
        **kwargs,
    )


def _make_reviewer_deep(scope, preamble, model, extra_modules=None, **kwargs):
    """Build a scoped **deepagents** review agent (shared F8 body).

    Requires the optional ``[deep]`` extra; the import is deferred so importing
    ``funhouse_agent`` never pulls in deepagents.
    """
    from funhouse_agent.deep.agent import build_deep_agent

    allowed = set(scope)
    if extra_modules:
        allowed |= set(extra_modules)
    kwargs.setdefault("reference_mode", "off")
    return build_deep_agent(
        model=model,
        allowed_agents=frozenset(allowed),
        extra_system_prompt=preamble,
        **kwargs,
    )


def make_foundations_reviewer(genai_engine, *, extra_modules=None, **kwargs):
    """Build the foundations reviewer (Funhouse scoped sub-agent).

    Scoped to the shallow + deep foundation and ground-improvement modules and
    prompted in review mode with the shared foundations checklist (GWT-in-wedge
    bearing, method-per-soil settlement/pile capacity, pile-group sign
    convention, drivability, downdrag neutral plane). See
    :func:`make_seismic_reviewer` for the parameter contract.
    """
    return _make_reviewer(FOUNDATIONS_REVIEWER_SCOPE,
                          FOUNDATIONS_REVIEWER_PREAMBLE, genai_engine,
                          extra_modules=extra_modules, **kwargs)


def make_foundations_reviewer_deep(model, *, extra_modules=None, **kwargs):
    """Deepagents variant of :func:`make_foundations_reviewer`."""
    return _make_reviewer_deep(FOUNDATIONS_REVIEWER_SCOPE,
                               FOUNDATIONS_REVIEWER_PREAMBLE, model,
                               extra_modules=extra_modules, **kwargs)


def make_earth_retention_reviewer(genai_engine, *, extra_modules=None, **kwargs):
    """Build the earth-retention reviewer (Funhouse scoped sub-agent).

    Scoped to the retaining-wall / excavation-support / seismic-earth-pressure
    modules (seismic_geotech = Mononobe-Okabe only) and prompted with the shared
    earth-retention checklist (Ka/K0/Kp state, single-FOS embedment basis,
    apparent-pressure envelopes, MSE LRFD CDRs, battered-wall M-O KAE→Ka).
    """
    return _make_reviewer(EARTH_RETENTION_REVIEWER_SCOPE,
                          EARTH_RETENTION_REVIEWER_PREAMBLE, genai_engine,
                          extra_modules=extra_modules, **kwargs)


def make_earth_retention_reviewer_deep(model, *, extra_modules=None, **kwargs):
    """Deepagents variant of :func:`make_earth_retention_reviewer`."""
    return _make_reviewer_deep(EARTH_RETENTION_REVIEWER_SCOPE,
                               EARTH_RETENTION_REVIEWER_PREAMBLE, model,
                               extra_modules=extra_modules, **kwargs)


def make_slope_fem_reviewer(genai_engine, *, extra_modules=None, **kwargs):
    """Build the slope / FEM reviewer (Funhouse scoped sub-agent).

    Scoped to the slope-stability / continuum-FEM / reliability / geometry-ingest
    modules and prompted with the shared slope-FEM checklist (LE method ordering,
    noncircular search rejection diagnostics, SRM-vs-LE + mesh convergence,
    T6-not-CST, reliability COV basis, ingest scale/provenance).
    """
    return _make_reviewer(SLOPE_FEM_REVIEWER_SCOPE,
                          SLOPE_FEM_REVIEWER_PREAMBLE, genai_engine,
                          extra_modules=extra_modules, **kwargs)


def make_slope_fem_reviewer_deep(model, *, extra_modules=None, **kwargs):
    """Deepagents variant of :func:`make_slope_fem_reviewer`."""
    return _make_reviewer_deep(SLOPE_FEM_REVIEWER_SCOPE,
                               SLOPE_FEM_REVIEWER_PREAMBLE, model,
                               extra_modules=extra_modules, **kwargs)


__all__ = [
    "make_seismic_reviewer",
    "make_seismic_reviewer_deep",
    "SEISMIC_REVIEWER_SCOPE",
    "make_foundations_reviewer",
    "make_foundations_reviewer_deep",
    "FOUNDATIONS_REVIEWER_SCOPE",
    "make_earth_retention_reviewer",
    "make_earth_retention_reviewer_deep",
    "EARTH_RETENTION_REVIEWER_SCOPE",
    "make_slope_fem_reviewer",
    "make_slope_fem_reviewer_deep",
    "SLOPE_FEM_REVIEWER_SCOPE",
]

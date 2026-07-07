"""Tests for funhouse_agent.reviewers — narrow reviewer agents (seismic first).

All tests use mock engines / a fake chat model — no API key, no network.

Run from the worktree root with the venv python::

    .venv/Scripts/python.exe -m pytest funhouse_agent/tests/test_reviewers.py -v
"""

import json

import pytest

from funhouse_agent.agent import GeotechAgent
from funhouse_agent.reviewers import (
    make_seismic_reviewer,
    make_seismic_reviewer_deep,
    SEISMIC_REVIEWER_SCOPE,
)
from funhouse_agent.dispatch import (
    ANALYSIS_MODULES, REFERENCE_MODULES,
    SEISMIC_MODULES, SEISMIC_REFERENCES,
    list_agents, list_methods, call_agent,
)
from funhouse_agent.adapters import MODULE_REGISTRY
from funhouse_agent.review_checklists import (
    SEISMIC_CHECKLIST, SEISMIC_REVIEWER_PREAMBLE,
)


# ---------------------------------------------------------------------------
# Mock engine (text / ReAct path — no native_tool_calling attribute)
# ---------------------------------------------------------------------------

class MockEngine:
    """Returns canned responses in sequence; records the system prompts seen."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0
        self.chat_calls = []

    def chat(self, user, system="", temperature=0):
        self.chat_calls.append({"user": user, "system": system})
        if self._i < len(self._responses):
            resp = self._responses[self._i]
            self._i += 1
            return resp
        return "No more responses"


# ---------------------------------------------------------------------------
# The scope sets themselves
# ---------------------------------------------------------------------------

class TestSeismicScopeSets:
    def test_modules_subset_of_analysis(self):
        assert SEISMIC_MODULES <= ANALYSIS_MODULES

    def test_references_subset_of_references(self):
        assert SEISMIC_REFERENCES <= REFERENCE_MODULES

    def test_every_scope_name_is_registered(self):
        for name in SEISMIC_REVIEWER_SCOPE:
            assert name in MODULE_REGISTRY, f"{name} not in MODULE_REGISTRY"

    def test_scope_is_union(self):
        assert SEISMIC_REVIEWER_SCOPE == (SEISMIC_MODULES | SEISMIC_REFERENCES)

    def test_core_seismic_modules_present(self):
        for name in ("seismic_geotech", "liquefaction", "liquepy",
                     "slope_stability", "opensees", "pystrata",
                     "seismic_signals", "hvsrpy", "swprocess", "fem2d"):
            assert name in SEISMIC_MODULES

    def test_seismic_modules_count(self):
        # 9 core seismic-native/-adjacent + swprocess (MASW -> Vs30 -> site class).
        assert len(SEISMIC_MODULES) == 10

    def test_core_seismic_references_present(self):
        for name in ("fema_p2082", "gec11", "gec7", "gec5", "dm7",
                     "reference_db", "figure_db"):
            assert name in SEISMIC_REFERENCES

    def test_non_seismic_modules_excluded(self):
        # Static / general-purpose modules must NOT leak into the seismic scope.
        for name in ("bearing_capacity", "settlement", "sheet_pile",
                     "reliability", "salib", "gstools", "subsurface"):
            assert name not in SEISMIC_REVIEWER_SCOPE

    def test_non_seismic_references_excluded(self):
        # The library's UFC docs are not seismic.
        for name in ("ufc_backfill", "ufc_expansive", "ufc_pavement",
                     "gec10", "gec6"):
            assert name not in SEISMIC_REFERENCES


# ---------------------------------------------------------------------------
# make_seismic_reviewer — construction + scoping + prompt
# ---------------------------------------------------------------------------

class TestMakeSeismicReviewer:
    def test_builds_geotech_agent(self):
        rev = make_seismic_reviewer(MockEngine([]))
        assert isinstance(rev, GeotechAgent)

    def test_scope_is_exactly_seismic(self):
        rev = make_seismic_reviewer(MockEngine([]))
        assert set(rev._allowed_agents) == set(SEISMIC_REVIEWER_SCOPE)

    def test_reference_mode_off(self):
        # References are directly in scope → no whole-library consult sub-agent.
        rev = make_seismic_reviewer(MockEngine([]))
        assert rev._reference_mode == "off"

    def test_list_agents_shows_only_seismic(self):
        rev = make_seismic_reviewer(MockEngine([]))
        visible = list_agents(allowed_agents=rev._allowed_agents)
        assert set(visible.keys()) == set(SEISMIC_REVIEWER_SCOPE)
        assert "bearing_capacity" not in visible
        assert "seismic_geotech" in visible

    def test_system_prompt_has_checklist_markers(self):
        rev = make_seismic_reviewer(MockEngine([]))
        sp = rev._system_prompt
        assert "YOU ARE IN SEISMIC REVIEW MODE" in sp
        assert "Seismic Review Checklist" in sp
        assert "Mononobe-Okabe" in sp
        assert "CSR/CRR" in sp
        # The scoped module CATALOG (the "| module | ... |" table) lists a
        # seismic module and NOT a static one — scoping trims the catalog. (The
        # fixed ReAct prefix still names bearing_capacity in its examples, so we
        # check the table rows, not the whole prompt.)
        assert "| seismic_geotech |" in sp
        assert "| bearing_capacity |" not in sp

    def test_extra_modules_widen_scope(self):
        rev = make_seismic_reviewer(MockEngine([]), extra_modules={"downdrag"})
        assert "downdrag" in rev._allowed_agents
        assert set(SEISMIC_REVIEWER_SCOPE) <= set(rev._allowed_agents)

    def test_kwargs_forwarded(self):
        rev = make_seismic_reviewer(MockEngine([]), max_rounds=3, temperature=0.0)
        assert rev._max_rounds == 3
        assert rev._temperature == 0.0

    def test_non_seismic_module_refused_by_scoping(self):
        rev = make_seismic_reviewer(MockEngine([]))
        result = call_agent("bearing_capacity", "bearing_capacity_analysis", {},
                            allowed_agents=rev._allowed_agents)
        assert "error" in result
        assert "Unknown module" in result["error"]

    def test_seismic_module_allowed_by_scoping(self):
        rev = make_seismic_reviewer(MockEngine([]))
        result = list_methods("seismic_geotech", allowed_agents=rev._allowed_agents)
        assert "error" not in result

    def test_seismic_reference_allowed_by_scoping(self):
        rev = make_seismic_reviewer(MockEngine([]))
        result = list_methods("fema_p2082", allowed_agents=rev._allowed_agents)
        assert "error" not in result

    def test_round_trip_ask(self):
        # A canned final answer (no tool call) exercises ask() end-to-end.
        rev = make_seismic_reviewer(
            MockEngine(["PASS — the M-O calc collapses to Ka at kh=0."])
        )
        result = rev.ask("Review this M-O calc: KAE=0.31 at kh=0.")
        assert "PASS" in result.answer
        # The review-mode preamble was actually handed to the engine.
        assert any("SEISMIC REVIEW MODE" in c["system"]
                   for c in rev._engine.chat_calls)


# ---------------------------------------------------------------------------
# Shared checklist source of truth
# ---------------------------------------------------------------------------

class TestSharedChecklist:
    def test_preamble_embeds_checklist(self):
        # The preamble injected into the agent contains the shared checklist —
        # so the .md playbook (which pastes SEISMIC_CHECKLIST) and the factory
        # cannot drift on the checklist body.
        assert SEISMIC_CHECKLIST in SEISMIC_REVIEWER_PREAMBLE

    def test_checklist_covers_key_conventions(self):
        for marker in ("kh", "sigma_v_eff", "NCEER", "Boulanger & Idriss",
                       "KAE", "Newmark", "Vs30", "MSF"):
            assert marker in SEISMIC_CHECKLIST


# ---------------------------------------------------------------------------
# Deep variant (optional [deep] stack)
# ---------------------------------------------------------------------------

class TestMakeSeismicReviewerDeep:
    def test_builds_with_fake_model(self):
        pytest.importorskip("deepagents")
        pytest.importorskip("langchain_core")
        from langchain_core.language_models.fake_chat_models import (
            GenericFakeChatModel,
        )
        from langchain_core.messages import AIMessage

        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        # A successful build proves build_deep_agent accepts both allowed_agents
        # AND extra_system_prompt (the seismic checklist) together.
        agent = make_seismic_reviewer_deep(model)
        assert agent is not None
        # Core dispatch tools are bound on the compiled primary.
        names = set(agent.nodes["tools"].bound.tools_by_name.keys())
        assert "call_agent" in names

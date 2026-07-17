"""Tests for funhouse_agent.reviewers — narrow reviewer agents (seismic first).

All tests use mock engines / a fake chat model — no API key, no network.

Run from the worktree root with the venv python::

    .venv/Scripts/python.exe -m pytest funhouse_agent/tests/test_reviewers.py -v
"""

import json
from collections import namedtuple
from pathlib import Path

import pytest

from funhouse_agent.agent import GeotechAgent
from funhouse_agent.reviewers import (
    make_seismic_reviewer,
    make_seismic_reviewer_deep,
    SEISMIC_REVIEWER_SCOPE,
    make_foundations_reviewer, make_foundations_reviewer_deep,
    FOUNDATIONS_REVIEWER_SCOPE,
    make_earth_retention_reviewer, make_earth_retention_reviewer_deep,
    EARTH_RETENTION_REVIEWER_SCOPE,
    make_slope_fem_reviewer, make_slope_fem_reviewer_deep,
    SLOPE_FEM_REVIEWER_SCOPE,
)
from funhouse_agent.dispatch import (
    ANALYSIS_MODULES, REFERENCE_MODULES,
    SEISMIC_MODULES, SEISMIC_REFERENCES,
    FOUNDATIONS_MODULES, FOUNDATIONS_REFERENCES,
    EARTH_RETENTION_MODULES, EARTH_RETENTION_REFERENCES,
    SLOPE_FEM_MODULES, SLOPE_FEM_REFERENCES,
    list_agents, list_methods, call_agent,
)
from funhouse_agent.adapters import MODULE_REGISTRY
from funhouse_agent.review_checklists import (
    SEISMIC_CHECKLIST, SEISMIC_REVIEWER_PREAMBLE,
    FOUNDATIONS_CHECKLIST, FOUNDATIONS_REVIEWER_PREAMBLE,
    EARTH_RETENTION_CHECKLIST, EARTH_RETENTION_REVIEWER_PREAMBLE,
    SLOPE_FEM_CHECKLIST, SLOPE_FEM_REVIEWER_PREAMBLE,
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


# ---------------------------------------------------------------------------
# F8 reviewer family — foundations / earth-retention / slope-FEM
# ---------------------------------------------------------------------------

_AGENTS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "agents"

_Rev = namedtuple("_Rev", [
    "name", "make", "make_deep", "scope", "modules", "references",
    "checklist", "preamble", "header", "in_scope", "out_scope",
    "markers", "md_file",
])

class TestPavementSpecialist:
    """The pavement specialist is DESIGN-mode (no checklist) but rides the
    same scoped-agent machinery as the reviewer family."""

    def _imports(self):
        from funhouse_agent.dispatch import (PAVEMENT_MODULES,
                                             PAVEMENT_REFERENCES)
        from funhouse_agent.review_checklists import \
            PAVEMENT_SPECIALIST_PREAMBLE
        from funhouse_agent.reviewers import (PAVEMENT_SPECIALIST_SCOPE,
                                              make_pavement_specialist)
        return (PAVEMENT_MODULES, PAVEMENT_REFERENCES,
                PAVEMENT_SPECIALIST_PREAMBLE, PAVEMENT_SPECIALIST_SCOPE,
                make_pavement_specialist)

    def test_scope_subsets_and_union(self):
        mods, refs, _, scope, _ = self._imports()
        assert mods <= ANALYSIS_MODULES
        assert refs <= REFERENCE_MODULES
        assert scope == (mods | refs)
        assert "pavement_design" in mods and "calc_package" in mods
        assert "aashto_1993" in refs
        for name in scope:
            assert name in MODULE_REGISTRY, f"{name} not in MODULE_REGISTRY"

    def test_builds_scoped_design_mode_agent(self):
        _, _, preamble, scope, make = self._imports()
        agent = make(MockEngine([]))
        assert isinstance(agent, GeotechAgent)
        assert set(agent._allowed_agents) == set(scope)
        assert agent._reference_mode == "off"
        assert "PAVEMENT DESIGN SPECIALIST" in preamble
        assert "US-CUSTOMARY" in preamble
        assert "UFC 3-250-01" in preamble  # both design bases in scope
        assert "both and compare" in preamble
        # Design mode, not review mode.
        assert "REVIEW MODE" not in preamble

    def test_webapp_builder_registered(self):
        import webapp.core as core
        assert core.AGENT_TYPES.get("pavement") == "Pavement design specialist"
        assert core._REVIEWER_BUILDERS.get("pavement") == \
            "make_pavement_specialist_deep"
        from funhouse_agent import reviewers
        assert hasattr(reviewers, "make_pavement_specialist_deep")


_F8 = [
    _Rev("foundations", make_foundations_reviewer, make_foundations_reviewer_deep,
         FOUNDATIONS_REVIEWER_SCOPE, FOUNDATIONS_MODULES, FOUNDATIONS_REFERENCES,
         FOUNDATIONS_CHECKLIST, FOUNDATIONS_REVIEWER_PREAMBLE,
         "FOUNDATIONS REVIEW MODE", "bearing_capacity", "slope_stability",
         ("GWT", "Converse-Labarre", "neutral plane", "Nordlund"),
         "foundations-reviewer.md"),
    _Rev("earth_retention", make_earth_retention_reviewer,
         make_earth_retention_reviewer_deep,
         EARTH_RETENTION_REVIEWER_SCOPE, EARTH_RETENTION_MODULES,
         EARTH_RETENTION_REFERENCES,
         EARTH_RETENTION_CHECKLIST, EARTH_RETENTION_REVIEWER_PREAMBLE,
         "EARTH-RETENTION REVIEW MODE", "retaining_walls", "bearing_capacity",
         ("APPARENT", "KAE", "Mononobe-Okabe", "embedment", "MSE"),
         "earth-retention-reviewer.md"),
    _Rev("slope_fem", make_slope_fem_reviewer, make_slope_fem_reviewer_deep,
         SLOPE_FEM_REVIEWER_SCOPE, SLOPE_FEM_MODULES, SLOPE_FEM_REFERENCES,
         SLOPE_FEM_CHECKLIST, SLOPE_FEM_REVIEWER_PREAMBLE,
         "SLOPE / FEM REVIEW MODE", "slope_stability", "bearing_capacity",
         ("SRM", "mesh", "OMS", "rejection", "CST"),
         "slope-fem-reviewer.md"),
]


@pytest.mark.parametrize("r", _F8, ids=[r.name for r in _F8])
class TestF8ReviewerFamily:
    def test_scope_subsets(self, r):
        assert r.modules <= ANALYSIS_MODULES
        assert r.references <= REFERENCE_MODULES

    def test_scope_is_union(self, r):
        assert r.scope == (r.modules | r.references)

    def test_every_scope_name_registered(self, r):
        for name in r.scope:
            assert name in MODULE_REGISTRY, f"{name} not in MODULE_REGISTRY"
        # geotech_common is a shared library, not an agent — must NOT be scoped.
        assert "geotech_common" not in r.scope

    def test_builds_geotech_agent_scoped(self, r):
        rev = r.make(MockEngine([]))
        assert isinstance(rev, GeotechAgent)
        assert set(rev._allowed_agents) == set(r.scope)

    def test_reference_mode_off(self, r):
        assert r.make(MockEngine([]))._reference_mode == "off"

    def test_list_agents_shows_only_domain(self, r):
        rev = r.make(MockEngine([]))
        visible = list_agents(allowed_agents=rev._allowed_agents)
        assert set(visible.keys()) == set(r.scope)
        assert r.in_scope in visible
        assert r.out_scope not in visible

    def test_prompt_has_header_and_checklist(self, r):
        sp = r.make(MockEngine([]))._system_prompt
        assert r.header in sp
        assert r.checklist in sp
        # scoping trims the catalog table to the in-scope module, not the out one
        assert f"| {r.in_scope} |" in sp
        assert f"| {r.out_scope} |" not in sp

    def test_out_of_scope_module_refused(self, r):
        rev = r.make(MockEngine([]))
        result = call_agent(r.out_scope, "whatever", {},
                            allowed_agents=rev._allowed_agents)
        assert "error" in result and "Unknown module" in result["error"]

    def test_in_scope_module_allowed(self, r):
        rev = r.make(MockEngine([]))
        assert "error" not in list_methods(r.in_scope,
                                           allowed_agents=rev._allowed_agents)

    def test_extra_modules_widen_scope(self, r):
        rev = r.make(MockEngine([]), extra_modules={"calc_package"})
        assert "calc_package" in rev._allowed_agents
        assert set(r.scope) <= set(rev._allowed_agents)

    def test_kwargs_forwarded(self, r):
        rev = r.make(MockEngine([]), max_rounds=2)
        assert rev._max_rounds == 2

    def test_preamble_embeds_checklist(self, r):
        assert r.checklist in r.preamble

    def test_checklist_covers_domain_markers(self, r):
        for m in r.markers:
            assert m in r.checklist, f"{r.name}: missing marker {m!r}"

    def test_md_agent_def_pastes_checklist_verbatim(self, r):
        """The Claude Code agent def must carry the SAME checklist (sync)."""
        md = (_AGENTS_DIR / r.md_file).read_text(encoding="utf-8")
        assert r.checklist in md, f"{r.md_file} checklist drifted from the module"
        assert r.make.__name__ in md   # sync pointer names the factory
        assert "review_checklists.py" in md

    def test_round_trip_ask(self, r):
        rev = r.make(MockEngine([f"PASS — {r.name} review, no issues."]))
        result = rev.ask("Review this calc: ...")
        assert "PASS" in result.answer
        assert any(r.header in c["system"] for c in rev._engine.chat_calls)


class TestF8ScopeDistinctness:
    def test_scopes_are_distinct(self):
        scopes = [FOUNDATIONS_REVIEWER_SCOPE, EARTH_RETENTION_REVIEWER_SCOPE,
                  SLOPE_FEM_REVIEWER_SCOPE, SEISMIC_REVIEWER_SCOPE]
        # analysis-module cores are disjoint enough to be genuinely different
        assert FOUNDATIONS_MODULES != EARTH_RETENTION_MODULES != SLOPE_FEM_MODULES
        assert len({frozenset(s) for s in scopes}) == 4

    def test_checklists_are_distinct(self):
        cks = {FOUNDATIONS_CHECKLIST, EARTH_RETENTION_CHECKLIST,
               SLOPE_FEM_CHECKLIST, SEISMIC_CHECKLIST}
        assert len(cks) == 4

    def test_seismic_geotech_shared_between_seismic_and_earth_retention(self):
        # M-O lives in seismic_geotech; earth-retention reviews M-O, seismic
        # reviews the rest — both legitimately include the module.
        assert "seismic_geotech" in EARTH_RETENTION_MODULES
        assert "seismic_geotech" in SEISMIC_MODULES


@pytest.mark.parametrize("r", _F8, ids=[r.name for r in _F8])
class TestF8ReviewerDeep:
    def test_builds_with_fake_model(self, r):
        pytest.importorskip("deepagents")
        pytest.importorskip("langchain_core")
        from langchain_core.language_models.fake_chat_models import (
            GenericFakeChatModel,
        )
        from langchain_core.messages import AIMessage

        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        agent = r.make_deep(model)
        assert agent is not None
        names = set(agent.nodes["tools"].bound.tools_by_name.keys())
        assert "call_agent" in names

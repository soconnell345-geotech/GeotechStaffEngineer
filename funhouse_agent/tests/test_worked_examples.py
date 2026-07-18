"""Worked-examples corpus tests — schema, search, dispatch surface, and the
mechanical guarantee: EVERY packaged entry's dispatch calls run clean and its
computed_result keys refer to real outputs. All offline (module calls only)."""

import pytest

from funhouse_agent import worked_examples as we
from funhouse_agent.dispatch import call_agent


def _entries():
    entries = we.load_examples()
    assert entries, "worked_examples.json missing or empty"
    return entries


def test_registry_loads_and_validates():
    for e in _entries():
        assert we.validate_entry(e) == [], (e.get("id"), we.validate_entry(e))


def test_ids_unique():
    ids = [e["id"] for e in _entries()]
    assert len(ids) == len(set(ids))


def test_domains_are_real_modules():
    from funhouse_agent.adapters import MODULE_REGISTRY
    for e in _entries():
        assert e["domain"] in MODULE_REGISTRY, e["id"]


@pytest.mark.slow
@pytest.mark.parametrize("entry", we.load_examples(),
                         ids=[e.get("id", "?") for e in we.load_examples()])
def test_every_entry_runs_clean(entry):
    """The corpus's core promise: the exemplar calculations actually run.
    ~1 min total (the slope searches dominate); deselect with -m 'not slow'."""
    assert we.verify_example(entry) == []


def test_search_finds_relevant_entries():
    hits = we.search_examples("rapid drawdown dam stability")
    assert hits and hits[0]["domain"] == "slope_stability"
    hits = we.search_examples("flexible pavement structural number ESAL")
    assert hits and hits[0]["domain"] == "pavement_design"
    assert we.search_examples("") == []
    assert we.search_examples("qzxv unmatchable gibberish zzz") == []


def test_dispatch_surface():
    r = call_agent("worked_examples", "find_worked_examples",
                   {"topic": "MSE wall external stability"})
    assert r.get("n_matches", 0) >= 1
    top = r["examples"][0]
    assert set(top) == {"id", "title", "domain", "source", "published_result"}
    full = call_agent("worked_examples", "get_worked_example",
                      {"example_id": top["id"]})
    assert full.get("dispatch_calls") and full.get("report_notes")
    missing = call_agent("worked_examples", "get_worked_example",
                         {"example_id": "WE-NOPE-99"})
    assert "error" in missing and "Known ids" in missing["error"]


def test_prompt_mentions_worked_examples():
    from funhouse_agent.deep.prompt import build_domain_prompt
    p = build_domain_prompt()
    assert "find_worked_examples" in p

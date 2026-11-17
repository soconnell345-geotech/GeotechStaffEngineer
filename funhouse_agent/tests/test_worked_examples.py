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
    assert "view_worked_example_source" in p


# ---------------------------------------------------------------------------
# Source-page retrieval (view_worked_example_source): the sample-calc twin of
# read_reference_figure — owner drops the public-source PDF in
# geotech-references/docs/, entries catalogue source_doc + 1-based pages.
# ---------------------------------------------------------------------------

import json as _json
import os as _os

_DOCS = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))),
    "geotech-references", "docs")


class _StubVisionEngine:
    def analyze_image(self, image_bytes, prompt):
        assert isinstance(image_bytes, (bytes, bytearray)) and len(image_bytes) > 1000
        self.last_prompt = prompt
        return "STUB-ANALYSIS"


def _seeded_entries():
    return [e for e in we.load_examples() if e.get("source_doc")]


def test_seeded_entries_resolve_and_pages_sane():
    entries = _seeded_entries()
    assert len(entries) >= 10
    if not _os.path.isdir(_DOCS):
        pytest.skip("refs docs folder absent (wheel install)")
    for e in entries:
        p = we.resolve_source_pdf(e)
        assert p.is_file(), e["id"]
        # EXACT name, not just "opens here": Windows matches filenames without
        # regard to capitals, the Databricks driver (Linux) does not. WE-PILE-1
        # said "GEC 12 vol 3.pdf" for the file "GEC 12 Vol 3.pdf" until
        # 2026-09-15 and would have failed only on the cluster.
        assert e["source_doc"] in _os.listdir(_DOCS), e["id"]
        assert all(pg >= 1 for pg in e["source_pdf_pages"]), e["id"]


def test_view_worked_example_source_renders_and_analyzes():
    if not _os.path.isdir(_DOCS):
        pytest.skip("refs docs folder absent (wheel install)")
    from funhouse_agent.vision_tools import dispatch_extended_tool
    eng = _StubVisionEngine()
    out = _json.loads(dispatch_extended_tool(
        "view_worked_example_source",
        {"example_id": "WE-PAVE-3", "prompt": "What does Table G-1 show?"},
        eng, {}))
    assert out.get("analysis") == "STUB-ANALYSIS", out
    assert out["pdf_page"] == out["catalogued_pages"][0]
    # review F5: the 0-based index is labelled, never a bare "page"
    assert out["pdf_page_index"] == out["pdf_page"] - 1 and "page" not in out
    assert "WE-PAVE-3" in eng.last_prompt or "UFC" in eng.last_prompt


def test_view_worked_example_source_errors():
    from funhouse_agent.vision_tools import dispatch_extended_tool
    eng = _StubVisionEngine()
    out = _json.loads(dispatch_extended_tool(
        "view_worked_example_source", {"example_id": "WE-NOPE-1"}, eng, {}))
    assert "Unknown worked example" in out.get("error", "")
    # an entry with NO catalogued source pages gives an actionable message
    no_src = next((e for e in we.load_examples()
                   if not e.get("source_doc")), None)
    if no_src is not None:
        out = _json.loads(dispatch_extended_tool(
            "view_worked_example_source", {"example_id": no_src["id"]},
            eng, {}))
        assert "no catalogued source pages" in out.get("error", "")


def test_deep_tools_include_view_worked_example():
    from funhouse_agent.deep.tools import make_vision_tools
    tools = make_vision_tools(engine=None)
    names = {t.name for t in tools}
    assert "view_worked_example_source" in names


def test_get_worked_example_accepts_id_alias():
    eid = we.load_examples()[0]["id"]
    full = call_agent("worked_examples", "get_worked_example", {"id": eid})
    assert full.get("id") == eid


def test_get_worked_example_wrong_id_says_did_you_mean():
    eid = we.load_examples()[0]["id"]
    res = call_agent("worked_examples", "get_worked_example",
                     {"example_id": eid.lower() + "x"})
    assert "Did you mean" in res["error"]
    assert eid in res["error"].split("Known ids")[0]


def test_we_sf_1_names_the_input_that_reproduces_the_print():
    """WE-SF-1's printed q_ult has no depth factors: the record says so and
    names depth_factor_method 'none', and its 'none' calls give the
    record's own no-depth-factor values."""
    from funhouse_agent.dispatch import call_agent
    (entry,) = [e for e in we.load_examples() if e["id"] == "WE-SF-1"]
    assert "depth_factor_method 'none'" in entry["report_notes"]
    plain = [c for c in entry["dispatch_calls"]
             if c["params"].get("depth_factor_method") == "none"]
    assert len(plain) == 2
    got = [call_agent(c["module"], c["method"], c["params"])["q_ultimate_kPa"]
           for c in plain]
    cr = entry["computed_result"]
    assert got[0] == pytest.approx(cr["q_ultimate_dry_no_depth_factors_kPa"],
                                   abs=0.1)
    assert got[1] == pytest.approx(cr["q_ultimate_gwt_no_depth_factors_kPa"],
                                   abs=0.1)


def test_depth_factors_are_a_stated_choice():
    """An embedded footing's result names the depth factors it applied, with
    their values, and offers leaving them out (made-up footing)."""
    from funhouse_agent.dispatch import call_agent
    params = {"width": 2.0, "depth": 1.0, "shape": "strip", "cohesion": 5.0,
              "friction_angle": 30.0, "unit_weight": 18.0}
    res = call_agent("bearing_capacity", "bearing_capacity_analysis", params)
    (depth,) = [c for c in res["choices"] if c["what"] == "depth factors"]
    assert f"{res['dc']:.3f}" in depth["chosen"] and res["dc"] > 1.0
    assert any("'none'" in a for a in depth["alternatives"])
    none = call_agent("bearing_capacity", "bearing_capacity_analysis",
                      {**params, "depth_factor_method": "none"})
    assert none["dc"] == none["dq"] == none["dgamma"] == 1.0
    assert none["q_ultimate_kPa"] < res["q_ultimate_kPa"]

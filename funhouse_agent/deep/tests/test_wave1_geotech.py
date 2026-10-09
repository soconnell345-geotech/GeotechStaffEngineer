"""Live smoke wave 1, geotech questions: the tool-surface fixes.

* G1  -- catalog and documentation results fit the cap WHOLE (never cut
  mid-JSON); every method's every parameter stays visible.
* G3  -- ``calculate`` evaluates a stated formula; the calc engine reaches
  the reference modules' computing functions; the prompts say so.
* G5  -- the references consult: a budget of 10, stated to it, and no
  scratch-filesystem tools on its menu.
* G7  -- one METHOD_INFO style with explicit ``required`` flags.
* G11 -- the activity log records what the model RECEIVED.
* G12 -- a guessed ``describe_method`` name that can mean one method resolves.
* G13 -- text-tool names resolve per reference module; long method lists
  are capped.
"""

import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from funhouse_agent import dispatch
from funhouse_agent.adapters import MODULE_REGISTRY
from funhouse_agent.deep.tools import DEFAULT_MAX_RESULT_CHARS, make_core_tools
from funhouse_agent.dispatch import ANALYSIS_MODULES, REFERENCE_MODULES


def _tools(scope=None, cap=DEFAULT_MAX_RESULT_CHARS):
    return {t.name: t for t in make_core_tools(allowed_agents=scope,
                                               max_result_chars=cap)}


# ---------------------------------------------------------------------------
# G1
# ---------------------------------------------------------------------------

def test_every_method_doc_fits_whole_and_lists_every_parameter():
    """The gate: no METHOD_INFO entry is ever cut. Every module, every
    method: the describe result is valid JSON within the cap and names every
    documented parameter with its required flag and allowed values."""
    describe = _tools()["describe_method"]
    over = []
    for module in sorted(MODULE_REGISTRY):
        mod = dispatch._load_adapter(module)
        for method, info in mod.METHOD_INFO.items():
            out = describe.invoke({"agent_name": module, "method": method})
            doc = json.loads(out)                      # never cut mid-JSON
            assert "...[truncated" not in out
            params = info.get("parameters") or {}
            assert set(params) <= set(doc.get("parameters") or {}), \
                (module, method)
            for name, spec in params.items():
                if isinstance(spec, dict) and "allowed_values" in spec:
                    assert doc["parameters"][name]["allowed_values"] == \
                        spec["allowed_values"], (module, method, name)
            if len(out) > DEFAULT_MAX_RESULT_CHARS:
                over.append((module, method, len(out)))
    assert not over, f"docs over the cap even as a skeleton: {over}"


def test_the_rapid_drawdown_tail_parameter_is_visible():
    """RDD-3: the 1,078 characters cut from rapid_drawdown_fos held the
    parameter the model then had to guess."""
    out = _tools()["describe_method"].invoke(
        {"agent_name": "slope_stability", "method": "rapid_drawdown_fos"})
    doc = json.loads(out)
    assert "stage3_effective_normal" in doc["parameters"]
    assert "_shortened" in doc and len(out) <= DEFAULT_MAX_RESULT_CHARS


def test_a_huge_method_list_is_fitted_by_category():
    out = _tools(REFERENCE_MODULES)["list_methods"].invoke(
        {"agent_name": "dm7"})
    data = json.loads(out)
    assert len(out) <= DEFAULT_MAX_RESULT_CHARS
    assert "category=" in data["_shortened"]
    assert sum(data["categories"].values()) == len(
        [m for m, i in dispatch._load_adapter("dm7").METHOD_INFO.items()
         if not i.get("alias_of")])
    # and one category lists in full
    cat = sorted(data["categories"])[0]
    one = json.loads(_tools(REFERENCE_MODULES)["list_methods"].invoke(
        {"agent_name": "dm7", "category": cat}))
    assert len(one[cat]) == data["categories"][cat]


# ---------------------------------------------------------------------------
# G3
# ---------------------------------------------------------------------------

def test_calculate_evaluates_a_stated_formula_safely():
    from funhouse_agent.deep.calculate_tool import evaluate, make_calculate_tool
    out = evaluate("(qt - sigma_v0) / Nkt",
                   {"qt": 1250, "sigma_v0": 95, "Nkt": 14})
    assert out["value"] == pytest.approx(82.5)
    assert evaluate("tan(radians(45 - phi/2))**2", {"phi": 30})["value"] == \
        pytest.approx(1 / 3)
    assert evaluate("2.5e-3 * 1000")["value"] == pytest.approx(2.5)
    for bad in ("__import__('os')", "x.real", "[1, 2]", "'a' * 3",
                "9**9**9**9", "1/0"):
        assert "error" in evaluate(bad, {"x": 1}), bad
    assert "'b'" in evaluate("a + b", {"a": 1})["error"]
    tool = make_calculate_tool()
    assert json.loads(tool.invoke({"expression": "2*pi"}))["value"] == \
        pytest.approx(6.283185, rel=1e-6)
    assert "formula" in tool.description


def test_calculate_is_on_every_agent_that_may_need_it():
    from funhouse_agent.deep.agent import (build_calc_subagent,
                                           build_primary_tools,
                                           build_references_subagent)
    assert "calculate" in {t.name for t in build_primary_tools()}
    assert "calculate" in {t.name for t in build_primary_tools(
        allowed_agents=())}                                # review page
    assert "calculate" in {t.name for t in build_calc_subagent()["tools"]}
    assert "calculate" in {t.name for t in
                           build_references_subagent()["tools"]}


def test_the_calc_engine_reaches_reference_equations():
    from funhouse_agent.deep.agent import build_calc_subagent
    call = {t.name: t for t in build_calc_subagent()["tools"]}["call_agent"]
    out = json.loads(call.invoke({"agent_name": "gec10",
                                  "method": "__nope__", "parameters": {}}))
    assert "Unknown module" not in out["error"]


def test_the_prompts_point_at_calculate_not_at_no_numbers():
    from funhouse_agent.reviewer import CONSULTANT_FRAMING
    from funhouse_agent.system_prompt import build_system_prompt
    prompt = build_system_prompt(ANALYSIS_MODULES)
    assert "`calculate` tool" in prompt
    assert "arithmetic in your head" in prompt
    assert "calculate" in CONSULTANT_FRAMING
    assert "reference lookup only" not in CONSULTANT_FRAMING


# ---------------------------------------------------------------------------
# G5
# ---------------------------------------------------------------------------

def test_the_references_consult_budget_and_menu():
    from funhouse_agent.deep.agent import build_references_subagent
    from funhouse_agent.deep.limits import (DEFAULT_REFERENCES_MAX_MODEL_CALLS,
                                            HideTools, REFERENCE_HIDDEN_TOOLS)
    assert DEFAULT_REFERENCES_MAX_MODEL_CALLS == 10
    spec = build_references_subagent()
    assert "You have 10 model calls" in spec["system_prompt"]
    hide = [m for m in spec["middleware"] if isinstance(m, HideTools)]
    assert hide and {"ls", "glob", "grep"} <= hide[0].hidden
    assert "read_file" not in REFERENCE_HIDDEN_TOOLS     # evicted results

    class _Req:
        tools = [type("T", (), {"name": n})() for n in
                 ("ls", "grep", "read_file", "call_agent")]

        def override(self, tools):
            r = _Req()
            r.tools = tools
            return r

    seen = []
    hide[0].wrap_model_call(_Req(), lambda r: seen.extend(
        t.name for t in r.tools))
    assert seen == ["read_file", "call_agent"]


# ---------------------------------------------------------------------------
# G7
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["pystrata", "liquepy", "opensees",
                                    "seismic_signals", "liquefaction"])
def test_every_parameter_says_whether_it_is_required(module):
    mod = dispatch._load_adapter(module)
    for method, info in mod.METHOD_INFO.items():
        for name, spec in info["parameters"].items():
            assert isinstance(spec.get("required"), bool), (method, name)
            assert spec.get("description"), (method, name)


def test_the_flags_match_what_the_code_demands():
    req = lambda m, meth: {p for p, s in dispatch._load_adapter(m)  # noqa
                           .METHOD_INFO[meth]["parameters"].items()
                           if s["required"]}
    assert req("liquepy", "spt_liquefaction") == {
        "depth", "N160", "FC", "gamma", "amax_g", "gwt_depth"}
    assert req("pystrata", "eql_site_response") == {"layers"}
    layers = dispatch._load_adapter("pystrata").METHOD_INFO[
        "eql_site_response"]["parameters"]["layers"]["description"]
    assert "plas_index" in layers and "damping" in layers
    fc = dispatch._load_adapter("liquefaction").METHOD_INFO[
        "liquefaction_analysis"]["parameters"]["FC"]
    assert fc["required_for"] == "SPT input"
    assert fc["description"].startswith("REQUIRED for SPT input")
    one = dispatch.call_agent("liquefaction", "liquefaction_analysis",
                              {"depth": [2.0], "q_c": [5000], "f_s": [50]})
    assert "at least 2 readings" in one["error"]
    assert "IndexError" not in one["error"]


# ---------------------------------------------------------------------------
# G12 / G13
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module,guess,real", [
    ("slope_stability", "infinite_slope", "infinite_slope_fos"),
    ("soe", "basal_heave", "check_basal_heave"),
    ("liquefaction", "triggering", "liquefaction_analysis"),
])
def test_a_guess_that_means_one_method_is_described(module, guess, real):
    doc = json.loads(_tools()["describe_method"].invoke(
        {"agent_name": module, "method": guess}))
    assert real in doc["_note"] and "parameters" in doc


def test_text_tool_names_resolve_per_reference_module():
    ref = REFERENCE_MODULES
    out = dispatch.call_agent("gec9", "text_search",
                              {"query": "wick drain"}, allowed_agents=ref)
    assert "error" not in out
    doc = dispatch.describe_method("micropile", "search", allowed_agents=ref)
    assert "search_sections" in doc["_note"]
    dm7 = dispatch.call_agent("dm7", "search_sections", {"query": "x"},
                              allowed_agents=ref)
    assert "reference_db" in dm7["error"]
    assert dm7["redirect"]["reference"] == ["dm7_1", "dm7_2"]


def test_a_long_method_list_is_capped_in_an_unknown_method_error():
    out = dispatch.call_agent("dm7", "no_such_method_xyz", {},
                              allowed_agents=REFERENCE_MODULES)
    assert "list_methods('dm7', category=" in out["error"]
    assert len(out["error"]) < 2500


# ---------------------------------------------------------------------------
# G11
# ---------------------------------------------------------------------------

def test_the_activity_log_records_what_the_model_received(tmp_path):
    from funhouse_agent.deep.agent import build_deep_agent
    from funhouse_agent.deep.scratch_guard import EMPTY_SCRATCH_NOTE
    from webapp.activity_log import ActivityLogger, load

    replies = [("", [("grep", {"pattern": "Nkt"}),
                     ("worked_examples_find", {"topic": "downdrag"})]),
               ("done", [])]

    class Scripted(BaseChatModel):
        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            text, calls = replies.pop(0)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=text, tool_calls=[
                    {"name": n, "args": a, "id": f"c{i}"}
                    for i, (n, a) in enumerate(calls)]))])

    agent = build_deep_agent(Scripted())
    log = ActivityLogger(str(tmp_path), turn=1)
    agent.invoke({"messages": [{"role": "user", "content": "q"}]},
                 config={"callbacks": [log]})
    recs = load(str(tmp_path))
    delivered = [r for r in recs if r["event"] == "tool_delivered"]
    refused = [r for r in recs if r["event"] == "tool_refused"]
    assert any(r["name"] == "grep" and EMPTY_SCRATCH_NOTE[:30] in r["text"]
               for r in delivered), recs
    assert any(r["name"] == "worked_examples_find"
               and "is not a valid tool" in r["text"] for r in refused), recs

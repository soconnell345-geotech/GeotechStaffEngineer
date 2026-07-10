"""Offline tests for the v5.0 deepagents port (NO API key, NO network).

Covers:
  (a) make_core_tools — dispatch correctness + allowed_agents scoping.
  (b) build_deep_agent(model=<fake>) — constructs and exposes the expected
      tools and the two sub-agents, with no real LLM call.

Run from the worktree root with the venv python::

    cd <worktree>
    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_deep_tools_offline.py -v
"""

import json
import re

import pytest

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from funhouse_agent.dispatch import ANALYSIS_MODULES, REFERENCE_MODULES
from funhouse_agent.deep.agent import (
    build_deep_agent,
    build_references_subagent,
    build_reviewer_subagent,
)
from funhouse_agent.deep.tools import make_core_tools, make_vision_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_by_name(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not in {[t.name for t in tools]}")


def _invoke(tool, **kwargs):
    """Invoke a StructuredTool with kwargs and parse its JSON-string result."""
    out = tool.invoke(kwargs)
    assert isinstance(out, str), f"{tool.name} must return a JSON string"
    return json.loads(out)


def _fake_model():
    """A LangChain fake chat model — never calls a real LLM."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))


def _compiled_tool_names(agent):
    """Tool names bound into a compiled deep agent's ToolNode."""
    return set(agent.nodes["tools"].bound.tools_by_name.keys())


def _listed_subagents(agent):
    """Sub-agent names advertised in the deepagents ``task`` tool description.

    deepagents lists delegable sub-agents as ``- <name>:`` lines (always
    including the built-in ``general-purpose``). This is the robust way to
    confirm our named sub-agents were wired, since the compiled graph hides
    them behind the single ``task`` tool.
    """
    desc = agent.nodes["tools"].bound.tools_by_name["task"].description
    return set(re.findall(r"- ([a-z0-9_-]+):", desc))


# ---------------------------------------------------------------------------
# (a) make_core_tools — dispatch correctness
# ---------------------------------------------------------------------------

def test_make_core_tools_shape():
    tools = make_core_tools()
    names = {t.name for t in tools}
    assert names == {"list_agents", "list_methods", "describe_method",
                     "call_agent"}
    # Each returns a JSON string (verified by the dispatch tests below).


def test_list_agents_returns_catalog():
    tools = make_core_tools()
    catalog = _invoke(_tool_by_name(tools, "list_agents"))
    assert isinstance(catalog, dict)
    # The full registry is the union of analysis + reference modules.
    assert "bearing_capacity" in catalog
    assert len(catalog) >= len(ANALYSIS_MODULES)


def test_call_agent_runs_real_calculation():
    """call_agent dispatches a real bearing-capacity analysis (SI params)."""
    tools = make_core_tools()

    # Confirm the method exists via list_methods (don't hardcode blindly).
    methods = _invoke(
        _tool_by_name(tools, "list_methods"), agent_name="bearing_capacity"
    )
    flat = {m for cat in methods.values() for m in cat}
    assert "bearing_capacity_analysis" in flat

    result = _invoke(
        _tool_by_name(tools, "call_agent"),
        agent_name="bearing_capacity",
        method="bearing_capacity_analysis",
        parameters={
            "width": 2.0,
            "length": 2.0,
            "depth": 1.5,
            "shape": "square",
            "friction_angle": 30.0,
            "cohesion": 5.0,
            "unit_weight": 18.0,
        },
    )
    assert "error" not in result, result
    # Expected result keys from a real bearing-capacity run.
    for key in ("q_ultimate_kPa", "q_allowable_kPa", "factor_of_safety",
                "Nc", "Nq", "Ngamma"):
        assert key in result, f"missing {key} in {list(result)}"
    assert result["q_ultimate_kPa"] > 0


def test_call_agent_auto_nests_flattened_params():
    """LLMs flatten params to top level — the tool must auto-nest them
    (mirrors native_tools.dispatch_native_tool)."""
    tools = make_core_tools()
    call = _tool_by_name(tools, "call_agent")
    # Pass method params at the TOP level (no parameters dict).
    out = call.invoke({
        "agent_name": "bearing_capacity",
        "method": "bearing_capacity_analysis",
        "width": 2.0,
        "length": 2.0,
        "depth": 1.5,
        "shape": "square",
        "friction_angle": 30.0,
        "cohesion": 5.0,
        "unit_weight": 18.0,
    })
    result = json.loads(out)
    assert "error" not in result, result
    assert result["q_ultimate_kPa"] > 0


# ---------------------------------------------------------------------------
# (a) allowed_agents scoping
# ---------------------------------------------------------------------------

def test_analysis_scope_refuses_reference_module():
    """An ANALYSIS_MODULES-scoped tool set must refuse a reference module."""
    tools = make_core_tools(allowed_agents=ANALYSIS_MODULES)
    result = _invoke(
        _tool_by_name(tools, "call_agent"),
        agent_name="dm7",
        method="anything",
        parameters={},
    )
    assert "error" in result
    assert "Unknown module" in result["error"]


def test_reference_scope_allows_reference_module():
    """A REFERENCE_MODULES-scoped tool set must SEE the reference module.

    The module resolves (so the error is about the bogus METHOD, not an
    unknown module), proving the module itself is visible/allowed.
    """
    # Disable truncation here: this test parses the JSON error, and the
    # "Unknown method" payload lists every available method (> the default
    # 8000-char cap), which would otherwise be truncated mid-JSON.
    tools = make_core_tools(
        allowed_agents=REFERENCE_MODULES, max_result_chars=0
    )

    # list_agents now shows the reference modules.
    catalog = _invoke(_tool_by_name(tools, "list_agents"))
    assert "dm7" in catalog
    assert "bearing_capacity" not in catalog  # analysis module is hidden

    result = _invoke(
        _tool_by_name(tools, "call_agent"),
        agent_name="dm7",
        method="__definitely_not_a_method__",
        parameters={},
    )
    assert "error" in result
    # Module is allowed → error is about the method, NOT "Unknown module".
    assert "Unknown module" not in result["error"]
    assert "Unknown method" in result["error"]


def test_analysis_scope_hides_reference_in_catalog():
    tools = make_core_tools(allowed_agents=ANALYSIS_MODULES)
    catalog = _invoke(_tool_by_name(tools, "list_agents"))
    assert "bearing_capacity" in catalog
    assert "dm7" not in catalog


# ---------------------------------------------------------------------------
# Vision tools (offline — no engine wired)
# ---------------------------------------------------------------------------

def test_vision_tools_build_and_error_without_engine():
    tools = make_vision_tools(engine=None)
    names = {t.name for t in tools}
    assert names == {"list_files", "read_pdf_text", "analyze_image",
                     "analyze_pdf_page", "read_reference_figure", "save_file"}

    # read_reference_figure without args → clear error (no raise).
    out = _invoke(
        _tool_by_name(tools, "read_reference_figure"),
        reference="", figure_number="",
    )
    assert "error" in out


def test_read_pdf_text_offline(tmp_path):
    """read_pdf_text is a pure-PyMuPDF tool: works with engine=None on a real
    path, extracts the text layer, and flags a scanned page."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Report page zero. SPT N=15.")
    # scanned page (image only, no text layer)
    tmp = fitz.open()
    tmp.new_page().insert_text((72, 72), "x")
    pix = tmp[0].get_pixmap()
    sp = doc.new_page()
    sp.insert_image(sp.rect, pixmap=pix)
    tmp.close()
    path = tmp_path / "r.pdf"
    path.write_bytes(doc.tobytes())
    doc.close()

    tools = make_vision_tools(engine=None)
    out = _invoke(_tool_by_name(tools, "read_pdf_text"),
                  source=str(path), pages="0-1")
    assert out["source_type"] == "path"
    assert "SPT N=15" in out["pages"][0]["text"]
    assert out["pages"][1]["has_text_layer"] is False
    assert out["scanned_pages"] == [1]


def test_list_files_offline(tmp_path):
    """list_files is a pure filesystem tool: works with engine=None on a real
    directory, sorts dirs first, and recurses only when asked."""
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4 ...")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "logs.csv").write_text("a,b\n")

    tools = make_vision_tools(engine=None)
    out = _invoke(_tool_by_name(tools, "list_files"), path=str(tmp_path))
    names = {e["name"]: e["type"] for e in out["entries"]}
    assert names == {"report.pdf": "file", "sub": "dir"}
    assert out["entries"][0]["type"] == "dir"  # dirs first

    deep = _invoke(_tool_by_name(tools, "list_files"),
                   path=str(tmp_path), depth=1)
    assert "sub/logs.csv" in {e["name"] for e in deep["entries"]}


def test_save_file_works_offline(tmp_path):
    tools = make_vision_tools(engine=None)
    target = tmp_path / "out" / "data.csv"
    out = _invoke(
        _tool_by_name(tools, "save_file"),
        path=str(target), content="x,y\n1,2\n",
    )
    assert "saved" in out, out
    assert target.read_text() == "x,y\n1,2\n"


# ---------------------------------------------------------------------------
# (b) build_deep_agent — offline construction with a fake model
# ---------------------------------------------------------------------------

def test_build_deep_agent_constructs():
    agent = build_deep_agent(model=_fake_model())
    # CompiledStateGraph from create_deep_agent.
    assert type(agent).__name__ == "CompiledStateGraph"


def test_primary_exposes_expected_tools():
    agent = build_deep_agent(model=_fake_model())
    names = _compiled_tool_names(agent)
    # Our domain tools are bound on the primary.
    for expected in ("call_agent", "list_agents", "list_methods",
                     "describe_method", "save_file"):
        assert expected in names, f"{expected} missing from {sorted(names)}"
    # deepagents-native subagent delegation tool (the v5 replacement for
    # v1's consult_references) is present.
    assert "task" in names


def test_primary_lists_both_subagents():
    """The references + reviewer sub-agents are delegable via the task tool."""
    agent = build_deep_agent(model=_fake_model())
    listed = _listed_subagents(agent)
    assert "references" in listed
    assert "reviewer" in listed
    # deepagents always provides its built-in general-purpose sub-agent.
    assert "general-purpose" in listed


def test_subagents_configured():
    """The references + reviewer sub-agents are built with the right scope."""
    refs = build_references_subagent()
    rev = build_reviewer_subagent()

    assert refs["name"] == "references"
    assert rev["name"] == "reviewer"

    # references prompt is the consultant framing (plus an appended concision
    # instruction); reviewer is the reviewer system prompt unchanged.
    from funhouse_agent.reviewer import (
        CONSULTANT_FRAMING, REVIEWER_SYSTEM_PROMPT,
    )
    assert refs["system_prompt"].startswith(CONSULTANT_FRAMING)
    assert "concise" in refs["system_prompt"].lower()
    assert rev["system_prompt"] == REVIEWER_SYSTEM_PROMPT

    # references has the figure read-off vision tool + core tools.
    ref_tool_names = {t.name for t in refs["tools"]}
    assert "read_reference_figure" in ref_tool_names
    assert "call_agent" in ref_tool_names

    # Both sub-agents are scoped to reference modules: call_agent refuses an
    # analysis module and accepts a reference module.
    for spec in (refs, rev):
        call = _tool_by_name(spec["tools"], "call_agent")
        analysis_attempt = json.loads(
            call.invoke({"agent_name": "bearing_capacity",
                         "method": "x", "parameters": {}})
        )
        assert "Unknown module" in analysis_attempt["error"]
        # The dm7 "Unknown method" payload lists every method (> the default
        # 8000-char cap), so it may come back truncated (not valid JSON).
        # Assert on the raw string: the module is allowed iff the error is NOT
        # "Unknown module".
        ref_raw = call.invoke({"agent_name": "dm7",
                               "method": "__nope__", "parameters": {}})
        assert "Unknown module" not in ref_raw
        assert "Unknown method" in ref_raw


def test_reference_mode_off_omits_named_subagents():
    """reference_mode='off' omits our references + reviewer sub-agents.

    NOTE: deepagents ALWAYS binds the ``task`` tool and its built-in
    ``general-purpose`` sub-agent — that cannot be turned off via the public
    API — so we assert on which NAMED sub-agents are advertised, not on the
    presence of the task tool.
    """
    agent = build_deep_agent(model=_fake_model(), reference_mode="off")
    listed = _listed_subagents(agent)
    assert "references" not in listed
    assert "reviewer" not in listed
    assert listed == {"general-purpose"}
    # Core domain tools still present.
    assert "call_agent" in _compiled_tool_names(agent)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

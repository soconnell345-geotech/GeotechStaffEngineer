"""Offline tests for the MODEL-SETUP sub-agent (NO API key, NO network).

Covers:
  (a) build_setup_subagent spec shape + the staged-protocol system prompt;
  (b) build_deep_agent wiring: OFF by default, attached with
      enable_setup_agent=True (advertised in the deepagents ``task`` tool),
      shared ProjectStore reachable from the host;
  (c) a SCRIPTED-model run through a compiled agent bound to the setup
      tools that walks the full staged flow:
        create from template → premature project_run REFUSED →
        render → request_confirmation (chat fallback: needs_user) →
        [user replies] → confirm geometry → cov_lookup → patch materials →
        confirm materials (with an edit) → confirm water_loads →
        append the LE analysis → project_run SUCCEEDS,
      asserting the gate ordering end-to-end through the ReAct loop.

Run from the worktree root with the venv python::

    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_deep_setup_agent_offline.py -v
"""

import json
import re

import pytest

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from funhouse_agent.deep.setup_agent import (
    SETUP_AGENT_DESCRIPTION,
    SETUP_SYSTEM_PROMPT,
    build_setup_subagent,
)
from funhouse_agent.deep.setup_tools import ProjectStore, make_setup_tools


class ScriptedToolModel(GenericFakeChatModel):
    """GenericFakeChatModel that tolerates bind_tools (plays its script
    regardless of the bound tool schemas — the agent loop executes whatever
    tool_calls the scripted AIMessages carry)."""

    def bind_tools(self, tools, **kwargs):  # noqa: D102 - see class docstring
        return self


EXPECTED_TOOLS = {
    "project_new", "dxf_discover", "project_show", "project_validate",
    "project_patch", "project_render", "cov_lookup",
    "request_confirmation", "project_run",
}


# ===========================================================================
# (a) Spec + prompt
# ===========================================================================

def test_build_setup_subagent_spec():
    store = ProjectStore()
    spec = build_setup_subagent(store=store, render_dir="rdir")
    assert spec["name"] == "model_setup"
    assert spec["description"] == SETUP_AGENT_DESCRIPTION
    assert spec["system_prompt"] == SETUP_SYSTEM_PROMPT
    assert {t.name for t in spec["tools"]} == EXPECTED_TOOLS
    # The shared store is reachable from every tool (host introspection).
    assert all(t.metadata["store"] is store for t in spec["tools"])
    assert store.render_dir == "rdir"


def test_prompt_encodes_the_staged_protocol():
    p = SETUP_SYSTEM_PROMPT
    # The design inversion.
    assert "never claim" in p.lower() or "never claims" in p.lower()
    assert "echo-back" in p
    # The stages, in order.
    for stage in ("GEOMETRY", "STRATIGRAPHY / MATERIALS",
                  "WATER / LOADS / REINFORCEMENT", "ANALYSIS PLAN", "RUN"):
        assert stage in p
    assert p.index("GEOMETRY") < p.index("STRATIGRAPHY / MATERIALS") \
        < p.index("WATER / LOADS / REINFORCEMENT") \
        < p.index("ANALYSIS PLAN") < p.index("RUN")
    # The non-negotiables.
    assert "request_confirmation" in p
    assert "dxf_discover" in p
    assert "NEVER guess" in p
    assert "assumption" in p.lower()
    assert "cov_lookup" in p
    assert "REFUSES" in p


# ===========================================================================
# (b) build_deep_agent wiring
# ===========================================================================

def _fake_model():
    return GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))


def _listed_subagents(agent):
    desc = agent.nodes["tools"].bound.tools_by_name["task"].description
    return set(re.findall(r"- ([a-z0-9_-]+):", desc))


def test_setup_agent_off_by_default():
    from funhouse_agent.deep.agent import build_deep_agent
    agent = build_deep_agent(model=_fake_model())
    assert "model_setup" not in _listed_subagents(agent)


def test_setup_agent_attached_when_enabled(tmp_path):
    from funhouse_agent.deep.agent import build_deep_agent
    store = ProjectStore()
    agent = build_deep_agent(
        model=_fake_model(), enable_setup_agent=True, setup_store=store,
        setup_render_dir=str(tmp_path))
    listed = _listed_subagents(agent)
    assert "model_setup" in listed
    assert "references" in listed  # existing sub-agents unaffected
    assert store.render_dir == str(tmp_path)


# ===========================================================================
# (c) Scripted staged flow through a compiled agent
# ===========================================================================

def _tc(name, args, cid):
    return {"name": name, "args": args, "id": f"call_{cid}"}


def _scripted_setup_agent(tmp_path):
    """A compiled deep agent whose tools ARE the setup tools and whose model
    is a SCRIPT: it emits exactly the staged tool-call sequence (including
    one premature project_run that must be refused)."""
    from deepagents import create_deep_agent

    store = ProjectStore()
    tools = make_setup_tools(store, render_dir=str(tmp_path))

    turn1 = [
        AIMessage(content="", tool_calls=[_tc("project_new", {
            "template": "simple_slope",
            "params": {"H": 6.0, "slope_ratio": 2.0},
            "name": "Scripted slope"}, 1)]),
        # Premature run — the gate must refuse this.
        AIMessage(content="", tool_calls=[_tc("project_run", {}, 2)]),
        AIMessage(content="", tool_calls=[_tc("project_render", {}, 3)]),
        AIMessage(content="", tool_calls=[_tc("request_confirmation", {
            "stage": "geometry",
            "summary_markdown": "6 m slope at 2H:1V — confirm the render."},
            4)]),
        AIMessage(content=(
            "Please confirm the geometry (see echo-back PNG and vertex "
            "table above).")),
    ]
    turn2 = [
        AIMessage(content="", tool_calls=[_tc("request_confirmation", {
            "stage": "geometry",
            "summary_markdown": "6 m slope at 2H:1V — confirm the render.",
            "user_response": {"approved": True, "edits": {}}}, 5)]),
        AIMessage(content="", tool_calls=[_tc("cov_lookup", {
            "property": "phi", "soil_type": "sand"}, 6)]),
        AIMessage(content="", tool_calls=[
            _tc("project_patch", {"path": "stratigraphy[0].material.phi",
                                  "value": 32.0}, 7),
            _tc("project_patch", {"path": "stratigraphy[0].material.c_prime",
                                  "value": 0.0}, 8),
            _tc("project_patch", {
                "path": "stratigraphy[0].material.probabilistic.phi",
                "value": {"cov": 0.079, "dist": "lognormal",
                          "source": "ISSMGE-TC304 (2021) Table 1.3 "
                                    "(sand, site-specific)"}}, 9),
        ]),
        AIMessage(content="", tool_calls=[_tc("request_confirmation", {
            "stage": "materials",
            "summary_markdown": ("Sand: phi=32 (assumed, TC304 mean COV "
                                 "7.9%), c'=0, gamma=18 (template default)"),
            "form_schema": [{"name": "stratigraphy[0].material.gamma",
                             "label": "Unit weight", "type": "number",
                             "unit": "kN/m3", "default": 18.0}],
            "user_response": {"approved": True,
                              "edits": {"stratigraphy[0].material.gamma":
                                        19.0}}}, 10)]),
        AIMessage(content="", tool_calls=[_tc("request_confirmation", {
            "stage": "water_loads",
            "summary_markdown": "No GWT, no surcharge, kh=0 — correct?",
            "user_response": {"approved": True, "edits": {}}}, 11)]),
        AIMessage(content="", tool_calls=[_tc("project_patch", {
            "path": "analyses", "op": "append",
            "value": {"type": "le", "name": "LE", "method": "bishop",
                      "n_slices": 20, "search": {"nx": 4, "ny": 4}}}, 12)]),
        AIMessage(content="", tool_calls=[_tc("project_run", {}, 13)]),
        AIMessage(content="Done. The Bishop search FOS is reported above."),
    ]
    model = ScriptedToolModel(messages=iter(turn1 + turn2))
    agent = create_deep_agent(model=model, tools=tools,
                              system_prompt=SETUP_SYSTEM_PROMPT)
    return agent, store


def _tool_messages(result, name):
    return [m for m in result["messages"]
            if getattr(m, "name", None) == name
            and type(m).__name__ == "ToolMessage"]


def test_scripted_staged_flow(tmp_path):
    pytest.importorskip("matplotlib")
    agent, store = _scripted_setup_agent(tmp_path)

    # ---- turn 1: create → premature run refused → render → gate question
    r1 = agent.invoke({"messages": [{
        "role": "user",
        "content": "Set up a 6 m high 2H:1V sand slope and run Bishop."}]})

    refused = json.loads(_tool_messages(r1, "project_run")[0].content)
    assert refused["status"] == "refused"
    assert refused["missing_confirmations"] == [
        "geometry", "materials", "water_loads"]

    rendered = json.loads(_tool_messages(r1, "project_render")[0].content)
    assert rendered["status"] == "rendered"
    assert "S1" in rendered["vertex_table"]

    gate1 = json.loads(_tool_messages(r1, "request_confirmation")[0].content)
    assert gate1["status"] == "needs_user"
    assert "CONFIRMATION REQUIRED: geometry" in gate1["chat_text"]
    assert store.project.confirmations.geometry is False

    # ---- turn 2: user approves → materials w/ cited COV → gates → run
    r2 = agent.invoke({"messages": r1["messages"] + [{
        "role": "user", "content": "Approved — looks right."}]})

    confs = [json.loads(m.content)
             for m in _tool_messages(r2, "request_confirmation")]
    # r2 replays turn-1's history, so the first entry is the needs_user
    # question; the three confirmations follow in stage order.
    assert confs[0]["status"] == "needs_user"
    gates = [c for c in confs if c.get("status") == "confirmed"]
    assert [c["stage"] for c in gates] == ["geometry", "materials",
                                           "water_loads"]
    # Approve-with-edits applied the user's gamma correction.
    assert gates[1]["edits_applied"] == ["stratigraphy[0].material.gamma"]
    assert store.project.stratigraphy[0].material.gamma == 19.0

    cov = json.loads(_tool_messages(r2, "cov_lookup")[0].content)
    assert cov["n_rows"] >= 1 and all(r["source"] for r in cov["rows"])
    # The cited COV landed in the project's probabilistic spec.
    assert store.project.stratigraphy[0].material.probabilistic[
        "phi"]["source"].startswith("ISSMGE-TC304")

    runs = [json.loads(m.content) for m in _tool_messages(r2, "project_run")]
    assert runs[-1]["status"] == "complete", runs[-1]
    fos = runs[-1]["results"]["LE"]["FOS"]
    assert fos is not None and 0.3 < fos < 6.0

    assert store.project.confirmations.all_confirmed()
    # The full gate ordering is visible in the store history.
    hist = " | ".join(store.history)
    assert hist.index("project_new") < hist.index("project_render") \
        < hist.index("geometry -> approved") \
        < hist.index("materials -> approved") \
        < hist.index("water_loads -> approved") \
        < hist.index("project_run: executed")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

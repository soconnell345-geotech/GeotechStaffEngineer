"""Offline tests for the MODEL-SETUP tools (NO API key, NO network).

Covers:
  (a) project lifecycle tools (new/show/patch/render/validate) over the
      shared ProjectStore;
  (b) the confirmation-gate mechanics BOTH ways:
      - chat fallback (no graph context): needs_user payload with numbered
        questions, then a second call with user_response approves/edits;
      - the REAL langgraph interrupt round-trip on a tiny StateGraph with
        InMemorySaver: payload shape under __interrupt__, resume via
        Command(resume={approved, edits}) applies patches and sets the gate;
  (c) gate ordering: project_run refuses before the three stages are
      confirmed and succeeds after;
  (d) vision-draft quarantine: stays blocked until geometry is confirmed;
  (e) patch semantics: targeted updates, open dicts, confirmation resets,
      confirmations paths refused.

Run from the worktree root with the venv python::

    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_deep_setup_tools_offline.py -v
"""

import json

import pytest

from funhouse_agent.deep.setup_tools import (
    ProjectStore,
    apply_patch,
    make_setup_tools,
)


def _toolset(tmp_path):
    store = ProjectStore()
    tools = make_setup_tools(store, render_dir=str(tmp_path))
    return store, {t.name: t for t in tools}


def _new_template(by_name, **params):
    out = json.loads(by_name["project_new"].invoke({
        "template": "simple_slope",
        "params": params or {"H": 8.0, "slope_ratio": 2.0},
    }))
    assert out["status"] == "created", out
    return out


def _confirm(by_name, stage, edits=None):
    """Drive the chat-fallback confirmation for a stage to approval."""
    first = json.loads(by_name["request_confirmation"].invoke({
        "stage": stage, "summary_markdown": f"please confirm {stage}",
    }))
    assert first["status"] == "needs_user", first
    second = json.loads(by_name["request_confirmation"].invoke({
        "stage": stage, "summary_markdown": f"please confirm {stage}",
        "user_response": {"approved": True, "edits": edits or {}},
    }))
    assert second["status"] == "confirmed", second
    return second


# ===========================================================================
# (a) Lifecycle
# ===========================================================================

def test_project_new_from_template(tmp_path):
    store, by_name = _toolset(tmp_path)
    out = _new_template(by_name)
    assert out["summary"]["provenance"] == "template"
    assert out["summary"]["confirmations"] == {
        "geometry": False, "materials": False, "water_loads": False}
    assert store.project is not None
    assert "echo-back" in out["next"]


def test_project_new_requires_exactly_one_source(tmp_path):
    _, by_name = _toolset(tmp_path)
    out = json.loads(by_name["project_new"].invoke({}))
    assert "error" in out
    out2 = json.loads(by_name["project_new"].invoke({
        "template": "simple_slope",
        "surface_points": [[0, 5], [20, 0]],
    }))
    assert "error" in out2


def test_project_new_unknown_template_lists_available(tmp_path):
    _, by_name = _toolset(tmp_path)
    out = json.loads(by_name["project_new"].invoke({"template": "volcano"}))
    assert "error" in out and "simple_slope" in out["available"]


def test_project_new_dxf_without_mapping_returns_inventory(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new("R2010")
    doc.layers.add("TOPO")
    doc.modelspace().add_lwpolyline([(0, 5), (20, 0)],
                                    dxfattribs={"layer": "TOPO"})
    path = tmp_path / "s.dxf"
    doc.saveas(path)

    _, by_name = _toolset(tmp_path)
    out = json.loads(by_name["project_new"].invoke({"dxf_path": str(path)}))
    assert out["status"] == "layer_mapping_required"
    assert "ASK THE USER" in out["message"]
    names = {ly["name"] for ly in out["inventory"]["layers"]}
    assert "TOPO" in names
    # With the user-confirmed mapping it builds.
    out2 = json.loads(by_name["project_new"].invoke({
        "dxf_path": str(path),
        "layer_mapping": {"surface": "TOPO"},
    }))
    assert out2["status"] == "created"
    assert out2["summary"]["provenance"] == "dxf"


def test_show_validate_render(tmp_path):
    store, by_name = _toolset(tmp_path)
    _new_template(by_name)

    shown = json.loads(by_name["project_show"].invoke({}))
    assert shown["project"]["meta"]["schema_version"] == 1
    assert shown["validation"]["ok"] is True

    section = json.loads(by_name["project_show"].invoke(
        {"section": "stratigraphy"}))
    assert "stratigraphy" in section and "project" not in section

    val = json.loads(by_name["project_validate"].invoke({}))
    assert val["ok"] is True and "summary" in val

    pytest.importorskip("matplotlib")
    rendered = json.loads(by_name["project_render"].invoke({}))
    assert rendered["status"] == "rendered"
    assert rendered["image_path"].endswith(".png")
    assert "S1" in rendered["vertex_table"]
    assert store.last_image_path == rendered["image_path"]


def test_tools_refuse_without_project(tmp_path):
    _, by_name = _toolset(tmp_path)
    for name in ("project_show", "project_validate", "project_render",
                 "project_run"):
        out = json.loads(by_name[name].invoke({}))
        assert "error" in out and "project_new" in out["error"]


def test_cov_lookup_returns_cited_rows(tmp_path):
    _, by_name = _toolset(tmp_path)
    out = json.loads(by_name["cov_lookup"].invoke(
        {"property": "phi", "soil_type": "sand"}))
    assert out["n_rows"] >= 1
    assert all(r["source"] for r in out["rows"])
    assert "PERCENT" in out["note"]


# ===========================================================================
# (e) Patch semantics
# ===========================================================================

def test_patch_sets_nested_value(tmp_path):
    store, by_name = _toolset(tmp_path)
    _new_template(by_name)
    out = json.loads(by_name["project_patch"].invoke({
        "path": "stratigraphy[0].material.phi", "value": 31.0}))
    assert out["status"] == "patched"
    assert store.project.stratigraphy[0].material.phi == 31.0


def test_patch_append_analysis_dispatches_type(tmp_path):
    from geo_project.schema import LEAnalysis
    store, by_name = _toolset(tmp_path)
    _new_template(by_name)
    out = json.loads(by_name["project_patch"].invoke({
        "path": "analyses", "op": "append",
        "value": {"type": "le", "method": "bishop",
                  "search": {"nx": 4, "ny": 4}}}))
    assert out["status"] == "patched"
    assert isinstance(store.project.analyses[0], LEAnalysis)
    assert store.project.analyses[0].search.nx == 4


def test_patch_open_dicts_accept_new_keys(tmp_path):
    store, by_name = _toolset(tmp_path)
    _new_template(by_name)
    out = json.loads(by_name["project_patch"].invoke({
        "path": "geometry.layer_boundaries.till_top",
        "value": [[0, -2.0], [40, -2.0]]}))
    assert out["status"] == "patched"
    assert store.project.geometry.layer_boundaries["till_top"] == \
        [(0.0, -2.0), (40.0, -2.0)]
    out2 = json.loads(by_name["project_patch"].invoke({
        "path": "stratigraphy[0].material.probabilistic.phi",
        "value": {"cov": 0.1, "dist": "lognormal",
                  "source": "Duncan (2000) Table 1"}}))
    assert out2["status"] == "patched"


def test_patch_rejects_unknown_field_and_bad_index(tmp_path):
    _, by_name = _toolset(tmp_path)
    _new_template(by_name)
    out = json.loads(by_name["project_patch"].invoke({
        "path": "stratigraphy[0].material.friction", "value": 30.0}))
    assert "error" in out
    out2 = json.loads(by_name["project_patch"].invoke({
        "path": "stratigraphy[7].material.phi", "value": 30.0}))
    assert "error" in out2


def test_patch_refuses_confirmations(tmp_path):
    _, by_name = _toolset(tmp_path)
    _new_template(by_name)
    out = json.loads(by_name["project_patch"].invoke({
        "path": "confirmations.geometry", "value": True}))
    assert "error" in out and "request_confirmation" in out["error"]


def test_patch_resets_stage_confirmation(tmp_path):
    store, by_name = _toolset(tmp_path)
    _new_template(by_name)
    _confirm(by_name, "geometry")
    assert store.project.confirmations.geometry is True
    out = json.loads(by_name["project_patch"].invoke({
        "path": "geometry.surface_points",
        "value": [[0, 8], [12, 8], [28, 0], [40, 0]]}))
    assert out["confirmation_reset"] == "geometry"
    assert store.project.confirmations.geometry is False
    # Materials patch resets materials, not geometry.
    _confirm(by_name, "geometry")
    _confirm(by_name, "materials")
    out2 = json.loads(by_name["project_patch"].invoke({
        "path": "stratigraphy[0].material.gamma", "value": 20.0}))
    assert out2["confirmation_reset"] == "materials"
    assert store.project.confirmations.geometry is True


def test_apply_patch_delete(tmp_path):
    store, by_name = _toolset(tmp_path)
    _new_template(by_name)
    by_name["project_patch"].invoke({
        "path": "loads.surcharges", "op": "append",
        "value": {"q": 10.0}})
    assert len(store.project.loads.surcharges) == 1
    out = json.loads(by_name["project_patch"].invoke({
        "path": "loads.surcharges[0]", "op": "delete"}))
    assert out["status"] == "patched"
    assert store.project.loads.surcharges == []


# ===========================================================================
# (b) Confirmation mechanics — chat fallback
# ===========================================================================

def test_fallback_needs_user_payload(tmp_path):
    _, by_name = _toolset(tmp_path)
    _new_template(by_name)
    out = json.loads(by_name["request_confirmation"].invoke({
        "stage": "geometry",
        "summary_markdown": "8 m slope at 2H:1V",
        "form_schema": [{"name": "stratigraphy[0].bottom_elevation",
                         "label": "Section bottom", "type": "number",
                         "unit": "m", "default": -8.0}],
    }))
    assert out["status"] == "needs_user"
    text = out["chat_text"]
    assert "CONFIRMATION REQUIRED: geometry" in text
    assert "1. Section bottom (m)" in text
    assert "2. Approve this stage" in text
    assert "user_response" in out["instructions"]


def test_fallback_approve_with_edits_sets_gate(tmp_path):
    store, by_name = _toolset(tmp_path)
    _new_template(by_name)
    second = _confirm(by_name, "geometry",
                      edits={"stratigraphy[0].bottom_elevation": -10.0})
    assert second["edits_applied"] == ["stratigraphy[0].bottom_elevation"]
    assert store.project.stratigraphy[0].bottom_elevation == -10.0
    assert store.project.confirmations.geometry is True


def test_fallback_rejection_does_not_set_gate(tmp_path):
    store, by_name = _toolset(tmp_path)
    _new_template(by_name)
    out = json.loads(by_name["request_confirmation"].invoke({
        "stage": "geometry", "summary_markdown": "x",
        "user_response": {"approved": False,
                          "edits": {"meta.name": "Renamed"}},
    }))
    assert out["status"] == "rejected"
    assert store.project.confirmations.geometry is False
    assert store.project.meta.name == "Renamed"  # edits still applied


def test_failed_edit_blocks_gate(tmp_path):
    store, by_name = _toolset(tmp_path)
    _new_template(by_name)
    out = json.loads(by_name["request_confirmation"].invoke({
        "stage": "geometry", "summary_markdown": "x",
        "user_response": {"approved": True,
                          "edits": {"confirmations.materials": True}},
    }))
    assert out["status"] == "edits_failed"
    assert store.project.confirmations.geometry is False
    assert store.project.confirmations.materials is False


def test_unknown_stage_rejected(tmp_path):
    _, by_name = _toolset(tmp_path)
    _new_template(by_name)
    out = json.loads(by_name["request_confirmation"].invoke({
        "stage": "vibes", "summary_markdown": "x"}))
    assert "error" in out


def test_analysis_plan_stage_logs_but_sets_no_gate(tmp_path):
    store, by_name = _toolset(tmp_path)
    _new_template(by_name)
    out = json.loads(by_name["request_confirmation"].invoke({
        "stage": "analysis_plan", "summary_markdown": "Bishop search",
        "user_response": {"approved": True},
    }))
    assert out["status"] == "confirmed"
    assert store.project.confirmations.missing() == [
        "geometry", "materials", "water_loads"]
    assert any(a.field == "analyses" for a in store.project.assumptions)


# ===========================================================================
# (b) Confirmation mechanics — REAL langgraph interrupt round-trip
# ===========================================================================

def test_interrupt_payload_shape_and_resume_with_edits(tmp_path):
    pytest.importorskip("matplotlib")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command
    from typing_extensions import TypedDict

    store, by_name = _toolset(tmp_path)
    _new_template(by_name)
    by_name["project_render"].invoke({})  # so the payload carries the image

    rc = by_name["request_confirmation"]

    class S(TypedDict):
        result: str

    def confirm_node(state: S):
        out = rc.invoke({
            "stage": "geometry",
            "summary_markdown": "Confirm the echoed section.",
            "form_schema": [{
                "name": "stratigraphy[0].material.gamma",
                "label": "Unit weight", "type": "number",
                "unit": "kN/m3", "default": 18.0,
            }],
        })
        return {"result": out}

    g = StateGraph(S)
    g.add_node("confirm", confirm_node)
    g.add_edge(START, "confirm")
    g.add_edge("confirm", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-confirm"}}

    # 1) The run pauses with the structured payload under __interrupt__.
    paused = app.invoke({"result": ""}, config=cfg)
    assert "__interrupt__" in paused
    payload = paused["__interrupt__"][0].value
    assert payload["type"] == "model_setup_confirmation"
    assert payload["stage"] == "geometry"
    assert payload["image_path"] == store.last_image_path
    assert payload["form_schema"][0]["type"] == "number"
    assert payload["resume_contract"] == {
        "approved": "bool", "edits": "{patch_path: value}"}
    assert store.project.confirmations.geometry is False  # still paused

    # 2) Resume with approve + an edit: patch applies, gate sets.
    done = app.invoke(
        Command(resume={"approved": True,
                        "edits": {"stratigraphy[0].material.gamma": 19.5}}),
        config=cfg)
    out = json.loads(done["result"])
    assert out["status"] == "confirmed"
    assert out["edits_applied"] == ["stratigraphy[0].material.gamma"]
    assert store.project.stratigraphy[0].material.gamma == 19.5
    assert store.project.confirmations.geometry is True


# ===========================================================================
# (c) Gate ordering — project_run refuses early, succeeds after
# ===========================================================================

def _complete_materials(by_name):
    by_name["project_patch"].invoke({
        "path": "stratigraphy[0].material.phi", "value": 30.0})
    by_name["project_patch"].invoke({
        "path": "stratigraphy[0].material.c_prime", "value": 5.0})
    by_name["project_patch"].invoke({
        "path": "stratigraphy[0].material.gamma", "value": 19.0})


def test_project_run_gate_ordering(tmp_path):
    store, by_name = _toolset(tmp_path)
    _new_template(by_name, H=6.0, slope_ratio=2.0)

    # 0 of 3 stages: refused, names all three.
    out = json.loads(by_name["project_run"].invoke({}))
    assert out["status"] == "refused"
    assert out["missing_confirmations"] == [
        "geometry", "materials", "water_loads"]

    _confirm(by_name, "geometry")
    out = json.loads(by_name["project_run"].invoke({}))
    assert out["status"] == "refused"
    assert out["missing_confirmations"] == ["materials", "water_loads"]

    _complete_materials(by_name)
    _confirm(by_name, "materials")
    _confirm(by_name, "water_loads")

    # All gates set but no analyses yet.
    out = json.loads(by_name["project_run"].invoke({}))
    assert out["status"] == "refused" and "No analyses" in out["error"]

    by_name["project_patch"].invoke({
        "path": "analyses", "op": "append",
        "value": {"type": "le", "name": "LE", "method": "bishop",
                  "n_slices": 20, "search": {"nx": 4, "ny": 4}}})
    out = json.loads(by_name["project_run"].invoke({}))
    assert out["status"] == "complete", out
    fos = out["results"]["LE"]["FOS"]
    assert fos is not None and 0.3 < fos < 6.0
    # Raw objects were stripped from the JSON payload.
    assert "_result" not in out["results"]["LE"]


def test_project_run_refused_on_validation_errors(tmp_path):
    store, by_name = _toolset(tmp_path)
    _new_template(by_name)
    _confirm(by_name, "geometry")
    _confirm(by_name, "materials")   # materials still EMPTY
    _confirm(by_name, "water_loads")
    by_name["project_patch"].invoke({
        "path": "analyses", "op": "append",
        "value": {"type": "le", "method": "bishop"}})
    out = json.loads(by_name["project_run"].invoke({}))
    assert out["status"] == "refused"
    codes = {i["code"] for i in out["validation"]["issues"]}
    assert "MAT002" in codes or "MAT001" in codes


# ===========================================================================
# (d) Vision-draft quarantine
# ===========================================================================

def test_vision_draft_blocked_until_geometry_confirmed(tmp_path):
    store, by_name = _toolset(tmp_path)
    out = json.loads(by_name["project_new"].invoke({
        "vision_draft": {
            "surface_points": [[0, 8], [10, 8], [26, 0], [36, 0]],
        },
        "section_bottom": -8.0,
    }))
    assert out["status"] == "created"
    assert out["summary"]["provenance"] == "vision_draft"
    # The draft itself fails validation with the blocking GEOM007.
    codes = {i["code"] for i in out["validation"]["issues"]}
    assert "GEOM007" in codes

    _complete_materials(by_name)
    _confirm(by_name, "materials")
    _confirm(by_name, "water_loads")
    by_name["project_patch"].invoke({
        "path": "analyses", "op": "append",
        "value": {"type": "le", "method": "bishop",
                  "search": {"nx": 3, "ny": 3}}})

    # Geometry unconfirmed → refused on the missing gate.
    out = json.loads(by_name["project_run"].invoke({}))
    assert out["status"] == "refused"
    assert out["missing_confirmations"] == ["geometry"]

    # Human confirms the echoed geometry → GEOM007 clears, run proceeds.
    _confirm(by_name, "geometry")
    val = json.loads(by_name["project_validate"].invoke({}))
    assert "GEOM007" not in {i["code"] for i in val["issues"]}
    out = json.loads(by_name["project_run"].invoke({}))
    assert out["status"] == "complete", out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

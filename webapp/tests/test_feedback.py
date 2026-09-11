"""Feedback capture (owner feedback 2026-09-11): the writer, the agent tool,
the build_agent wiring (primary AND calc sub-agent), and the sidebar box."""

import json
import os

import pytest

pytest.importorskip("langchain_core")

from webapp import feedback


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------

def test_record_writes_jsonl_and_md(tmp_path):
    conv = tmp_path / "conv"
    e = feedback.record(str(conv), source="user", kind="user_feedback",
                        summary="  The report   had no figure ",
                        details="Asked for a profile sketch, got a table.",
                        context={"turn": 3})
    assert e["summary"] == "The report had no figure"
    assert e["source"] == "user" and e["kind"] == "user_feedback"
    assert e["context"]["turn"] == 3
    assert "app_version" in e["context"]
    rows = feedback.load(str(conv))
    assert len(rows) == 1 and rows[0]["summary"] == e["summary"]
    md = (conv / feedback.MD_NAME).read_text(encoding="utf-8")
    assert md.startswith("# Feedback recorded in this conversation")
    assert "**The report had no figure**" in md
    assert "Asked for a profile sketch" in md
    assert "turn=3" in md


def test_record_appends_and_header_written_once(tmp_path):
    conv = str(tmp_path)
    feedback.record(conv, source="agent", kind="capability_gap", summary="one")
    feedback.record(conv, source="agent", kind="tool_error", summary="two")
    assert [r["summary"] for r in feedback.load(conv)] == ["one", "two"]
    md = open(os.path.join(conv, feedback.MD_NAME), encoding="utf-8").read()
    assert md.count("# Feedback recorded") == 1
    assert md.index("**one**") < md.index("**two**")


def test_unknown_kind_and_source_are_normalised(tmp_path):
    e = feedback.record(str(tmp_path), source="robot", kind="Missing-Plot",
                        summary="x")
    assert e["kind"] == "other" and e["source"] == "agent"
    e2 = feedback.record(str(tmp_path), source="AGENT", kind="capability gap",
                         summary="y")
    assert e2["kind"] == "capability_gap" and e2["source"] == "agent"


def test_blank_summary_rejected(tmp_path):
    with pytest.raises(ValueError):
        feedback.record(str(tmp_path), source="user", kind="other",
                        summary="   ")


def test_details_capped(tmp_path):
    e = feedback.record(str(tmp_path), source="user", kind="other",
                        summary="s", details="x" * 20000)
    assert len(e["details"]) == feedback.MAX_DETAILS_CHARS


def test_load_missing_and_malformed(tmp_path):
    assert feedback.load(str(tmp_path / "nope")) == []
    p = tmp_path / feedback.JSONL_NAME
    p.write_text('{"summary": "ok"}\nnot json\n\n', encoding="utf-8")
    assert [r["summary"] for r in feedback.load(str(tmp_path))] == ["ok"]


# ---------------------------------------------------------------------------
# agent tool
# ---------------------------------------------------------------------------

def test_tool_records_with_context_and_returns_string(tmp_path):
    tool = feedback.make_record_feedback_tool(
        str(tmp_path), context_fn=lambda: {"turn": 2, "model": "m"})
    assert tool.name == "record_feedback"
    out = tool.invoke({"kind": "capability_gap",
                       "summary": "no tool draws a p-y curve",
                       "details": "lateral_pile has no plot method"})
    assert isinstance(out, str)
    assert "recorded" in out.lower() and "continue" in out.lower()
    rows = feedback.load(str(tmp_path))
    assert rows[0]["kind"] == "capability_gap"
    assert rows[0]["source"] == "agent"
    assert rows[0]["context"]["turn"] == 2


def test_tool_never_raises(tmp_path):
    # unwritable target: a FILE where the directory should be
    blocker = tmp_path / "conv"
    blocker.write_text("i am a file", encoding="utf-8")
    tool = feedback.make_record_feedback_tool(str(blocker))
    out = tool.invoke({"kind": "other", "summary": "x"})
    assert isinstance(out, str) and "could not be recorded" in out.lower()
    # blank summary is the tool's own ValueError → readable string
    tool2 = feedback.make_record_feedback_tool(str(tmp_path))
    out2 = tool2.invoke({"kind": "other", "summary": ""})
    assert "could not be recorded" in out2.lower()


def test_context_fn_failure_is_swallowed(tmp_path):
    def boom():
        raise RuntimeError("no context")
    tool = feedback.make_record_feedback_tool(str(tmp_path), context_fn=boom)
    out = tool.invoke({"kind": "other", "summary": "still works"})
    assert "recorded" in out.lower()
    assert feedback.load(str(tmp_path))[0]["summary"] == "still works"


def test_tools_for_shape():
    tools, prompt = feedback.tools_for("/nowhere")
    assert [t.name for t in tools] == ["record_feedback"]
    assert "record_feedback" in prompt and "capability_gap" in prompt


# ---------------------------------------------------------------------------
# build_agent wiring: primary AND calc sub-agent get the tool
# ---------------------------------------------------------------------------

def test_calc_subagent_spec_accepts_extra_tools():
    from funhouse_agent.deep.agent import build_calc_subagent
    tool = feedback.make_record_feedback_tool("/nowhere")
    spec = build_calc_subagent(extra_tools=[tool],
                               extra_system_prompt=feedback.FEEDBACK_PROMPT)
    assert "record_feedback" in [t.name for t in spec["tools"]]
    assert spec["system_prompt"].endswith(feedback.FEEDBACK_PROMPT)
    # default build unchanged
    base = build_calc_subagent()
    assert "record_feedback" not in [t.name for t in base["tools"]]


def test_build_agent_hands_tool_to_primary_and_calc(monkeypatch, tmp_path):
    """``core.build_agent`` must route record_feedback to BOTH the primary
    (extra_tools) and the calc sub-agent (calc_extra_tools), bound to the
    conversation dir = parent of temp_dir."""
    import webapp.core as core
    captured = {}

    def fake_build_deep_agent(model, **kw):
        captured.update(kw)
        return object()

    import funhouse_agent.deep.agent as deep_agent
    monkeypatch.setattr(deep_agent, "build_deep_agent", fake_build_deep_agent)
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    conv = core.conversation_dir("t123")
    files = os.path.join(conv, "files")
    os.makedirs(files)
    core.build_agent(object(), {}, files, [])
    prim = [t.name for t in captured.get("extra_tools") or []]
    calc = [t.name for t in captured.get("calc_extra_tools") or []]
    assert "record_feedback" in prim
    assert "record_feedback" in calc
    assert "record_feedback" in captured.get("extra_system_prompt", "")
    assert "record_feedback" in captured.get("calc_extra_system_prompt", "")
    # the tool is bound to THIS conversation's record dir
    tool = next(t for t in captured["calc_extra_tools"]
                if t.name == "record_feedback")
    tool.invoke({"kind": "capability_gap", "summary": "bound to conv"})
    assert os.path.isfile(os.path.join(conv, feedback.JSONL_NAME))
    rows = feedback.load(conv)
    assert rows[0]["context"]["thread_id"] == "t123"


# ---------------------------------------------------------------------------
# sidebar box (AppTest)
# ---------------------------------------------------------------------------

def test_sidebar_feedback_form_saves_with_conversation(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import webapp.core as core
    import webapp.engine_config as engine_config
    from webapp.engine_config import EngineResolution

    app = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "app.py")
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.setattr(
        engine_config, "resolve_engine",
        lambda *_a, **_k: EngineResolution(model=object(), source="anthropic",
                                           model_name="fake", message=""))
    monkeypatch.setattr(core, "build_agent", lambda *_a, **_k: object())
    at = AppTest.from_file(app, default_timeout=30).run()
    assert not at.exception
    boxes = [t for t in at.sidebar.text_area
             if t.key and str(t.key).startswith("feedback_text_")]
    assert len(boxes) == 1, "sidebar Feedback box missing"
    boxes[0].set_value("The calc package had no figures.\nPlease add them.")
    btn = next(b for b in at.sidebar.button
               if "save feedback" in str(b.label).lower())
    btn.click().run()
    assert not at.exception
    tid = at.session_state["thread_id"]
    rows = feedback.load(core.conversation_dir(tid))
    assert len(rows) == 1
    assert rows[0]["source"] == "user"
    assert rows[0]["summary"] == "The calc package had no figures."
    assert "Please add them." in rows[0]["details"]
    assert rows[0]["context"]["thread_id"] == tid

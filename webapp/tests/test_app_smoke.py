"""Headless AppTest coverage for webapp/app.py — the A3 turn-loop wiring.

Proves end-to-end (what the core.py unit tests can't reach — the app.py wiring
itself) that:
  * the app boots clean,
  * a normal turn persists the user + assistant transcript and clears the
    mid-turn partial checkpoint,
  * a mid-stream exception is caught IN-CHAT (transcript KEPT + assistant entry
    marked with the error, both persisted) instead of a red-box state wipe.

Uses ``streamlit.testing.v1.AppTest`` with the engine + agent + stream mocked,
so no API key or LLM call is made. The persistent-data root is redirected to a
tmp dir via ``GEOTECH_WEBAPP_DATA``.
"""

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import webapp.core as core
import webapp.engine_config as engine_config
from webapp.engine_config import EngineResolution

_APP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _fake_engine(*_a, **_k):
    # model is not None => EngineResolution.ok is True => the agent gets built.
    return EngineResolution(model=object(), source="anthropic",
                            model_name="fake-model", message="")


def _stream_ok(_agent, _messages, _thread_id, **_kw):
    yield {"kind": "token", "text": "Hello "}
    yield {"kind": "token", "text": "world."}
    yield {"kind": "turn_done", "answer": "Hello world.", "turn_tokens": 5}


def _stream_boom(_agent, _messages, _thread_id, **_kw):
    yield {"kind": "token", "text": "partial answer "}
    raise RuntimeError("stream blew up")


def _mk_at(monkeypatch, tmp_path, stream_fn):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.setattr(engine_config, "resolve_engine", _fake_engine)
    monkeypatch.setattr(core, "build_agent", lambda *_a, **_k: object())
    monkeypatch.setattr(core, "build_reviewer_agent",
                        lambda kind, *_a, **_k: f"reviewer:{kind}")
    monkeypatch.setattr(core, "stream_turn", stream_fn)
    return AppTest.from_file(_APP, default_timeout=30)


def test_app_boots_clean(monkeypatch, tmp_path):
    at = _mk_at(monkeypatch, tmp_path, _stream_ok).run()
    assert not at.exception
    assert at.session_state["initialized"] is True
    assert at.session_state["transcript"] == []


def test_normal_turn_persists_and_clears_partial(monkeypatch, tmp_path):
    at = _mk_at(monkeypatch, tmp_path, _stream_ok).run()
    at.chat_input[0].set_value("What is bearing capacity?").run()
    assert not at.exception
    tr = at.session_state["transcript"]
    assert [e["role"] for e in tr] == ["user", "assistant"]
    assert tr[1]["text"] == "Hello world."
    assert "error" not in tr[1]
    tid = at.session_state["thread_id"]
    # durable on disk, and the mid-turn partial checkpoint was cleared
    assert [e["role"] for e in core.load_transcript(tid)] == ["user", "assistant"]
    assert not os.path.isfile(core.partial_path(tid))


def test_stream_exception_keeps_conversation(monkeypatch, tmp_path):
    at = _mk_at(monkeypatch, tmp_path, _stream_boom).run()
    at.chat_input[0].set_value("crash please").run()
    # THE headline A3 assertion: a mid-stream crash does NOT surface as an
    # uncaught exception (no red-box state wipe).
    assert not at.exception
    tr = at.session_state["transcript"]
    assert [e["role"] for e in tr] == ["user", "assistant"]   # both KEPT
    assert "stream blew up" in tr[1].get("error", "")
    # the error was surfaced in-chat (handled), not as a crash
    assert any("stream blew up" in e.value for e in at.error)
    # and the interrupted turn is durably persisted with its error marker
    tid = at.session_state["thread_id"]
    reloaded = core.load_transcript(tid)
    assert [e["role"] for e in reloaded] == ["user", "assistant"]
    assert reloaded[1].get("error")


def _stream_saves_external(_agent, _messages, _thread_id, **_kw):
    # Simulate a plot / calc-package adapter saving into the host working folder
    # — the dir the app advertises via GEOTECH_DEFAULT_OUTPUT_DIR (which is what
    # funhouse_agent._fileio.default_output_dir() returns for a bare output_path).
    d = os.environ.get("GEOTECH_DEFAULT_OUTPUT_DIR") or ""
    if d:
        with open(os.path.join(d, "plot.html"), "w", encoding="utf-8") as fh:
            fh.write("<html>fig</html>")
    yield {"kind": "token", "text": "saved a plot"}
    yield {"kind": "turn_done", "answer": "saved a plot", "turn_tokens": 3}


def test_external_working_folder_artifact_bridged_into_files(monkeypatch, tmp_path):
    """A6/A4 wiring: a save into an EXTERNAL working folder is copied INTO the
    conversation files/ dir, so it persists and renders as a durable card
    (portable relative ref) instead of being lost or left as a non-portable
    absolute path. Regression guard for the turn-loop bridge call."""
    external = tmp_path / "project_out"
    external.mkdir()
    at = _mk_at(monkeypatch, tmp_path, _stream_saves_external).run()
    tid = at.session_state["thread_id"]
    # Point the agent's saves OUT via the real sidebar control (driving the
    # widget so the app persists it, rather than editing meta out of band).
    wd_widget = next(w for w in at.text_input if w.key == f"workdir_{tid}")
    wd_widget.set_value(str(external)).run()
    at.chat_input[0].set_value("make a plot").run()
    assert not at.exception

    files_dir = core.conversation_files_dir(tid)
    # the external save was bridged into files/ (durable with the conversation)
    assert os.path.isfile(os.path.join(files_dir, "plot.html"))
    # the assistant turn records it as an artifact resolving under files/
    arts = (at.session_state["transcript"][-1].get("artifacts") or [])
    assert any(os.path.basename(a) == "plot.html" for a in arts)
    ref = core.load_transcript(tid)[-1]["artifacts"][0]
    assert os.path.abspath(ref).startswith(os.path.abspath(files_dir))
    assert core.describe_artifact(ref).exists is True


def test_agent_picker_builds_reviewer_and_persists(monkeypatch, tmp_path):
    """A5e: choosing a reviewer in the sidebar rebuilds via the reviewer builder,
    records it in the conversation's behavior, and persists it to meta on the
    turn (so a resume via behavior_from_meta restores the agent type)."""
    at = _mk_at(monkeypatch, tmp_path, _stream_ok).run()
    tid = at.session_state["thread_id"]
    sel = next(w for w in at.selectbox if w.key == f"agent_{tid}")
    sel.set_value("seismic").run()
    assert not at.exception
    assert at.session_state["behavior"]["agent_type"] == "seismic"
    assert at.session_state["agent"] == "reviewer:seismic"     # reviewer branch
    at.chat_input[0].set_value("review this liquefaction calc").run()
    assert core.behavior_from_meta(
        core.load_meta(tid))["agent_type"] == "seismic"        # durable → resumes


def test_local_tracer_writes_turn_trace(monkeypatch, tmp_path):
    """A7: with GEOTECH_TRACE=1 a turn writes one compact JSONL trace line to the
    conversation dir (duration + tokens + tool calls)."""
    monkeypatch.setenv("GEOTECH_TRACE", "1")
    at = _mk_at(monkeypatch, tmp_path, _stream_ok).run()
    tid = at.session_state["thread_id"]
    at.chat_input[0].set_value("what is bearing capacity?").run()
    assert not at.exception
    recent = core.load_recent_traces(tid, n=1)
    assert recent and "duration_s" in recent[-1]
    assert recent[-1]["turn_tokens"] == 5          # from _stream_ok's turn_done
    assert recent[-1]["error"] is None

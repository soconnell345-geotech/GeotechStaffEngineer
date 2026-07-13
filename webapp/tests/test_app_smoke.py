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


def _stream_ok(_agent, _messages, _thread_id):
    yield {"kind": "token", "text": "Hello "}
    yield {"kind": "token", "text": "world."}
    yield {"kind": "turn_done", "answer": "Hello world.", "turn_tokens": 5}


def _stream_boom(_agent, _messages, _thread_id):
    yield {"kind": "token", "text": "partial answer "}
    raise RuntimeError("stream blew up")


def _mk_at(monkeypatch, tmp_path, stream_fn):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.setattr(engine_config, "resolve_engine", _fake_engine)
    monkeypatch.setattr(core, "build_agent", lambda *_a, **_k: object())
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

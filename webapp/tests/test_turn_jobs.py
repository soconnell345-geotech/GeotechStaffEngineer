"""Tests for detached turn execution (webapp/turn_jobs.py).

The whole point of the module: a turn survives the death of the thing that
started it. A fake stream stands in for the agent; persistence goes to a
real (tmp) conversation directory via the normal core functions.
"""

import os
import time

import pytest

import webapp.core as core
import webapp.turn_jobs as tj


@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    # each test gets a clean job registry
    tj._JOBS.clear()
    yield


def _fake_stream(events):
    def stream_turn(agent, messages, thread_id, recursion_limit=None):
        yield from events
    return stream_turn


def _mk_conv(tid="TJ1"):
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    os.makedirs(files, exist_ok=True)
    return tid, files


def _ctx(tid, files, transcript, artifacts, prompt="bearing capacity?"):
    return {
        "prompt": prompt, "temp_dir": files,
        "before": core.snapshot_dir(files), "staged_inputs": set(),
        "working_dir": files, "before_wd": None,
        "artifacts": artifacts, "artifacts_before_len": len(artifacts),
        "transcript": transcript, "trace_on": False,
        "model": "test-model", "behavior": core.default_behavior(),
    }


def _wait_done(job, timeout=10.0):
    t0 = time.time()
    while not job.done and time.time() - t0 < timeout:
        time.sleep(0.02)
    assert job.done, "worker did not finish in time"


EVENTS_OK = [
    {"kind": "tool_call", "text": "call_agent bearing_capacity"},
    {"kind": "token", "text": "The ultimate "},
    {"kind": "token", "text": "capacity is 1159 kPa."},
    {"kind": "turn_done", "answer": "The ultimate capacity is 1159 kPa.",
     "turn_tokens": 321},
]


class TestWorkerPersistsWithoutFollower:
    def test_turn_completes_and_persists_with_no_ui_attached(self, monkeypatch):
        """The core guarantee: nobody follows (socket dead) — work still lands."""
        monkeypatch.setattr(core, "stream_turn", _fake_stream(EVENTS_OK))
        tid, files = _mk_conv()
        transcript = [{"role": "user", "text": "bearing capacity?"}]
        messages = [{"role": "user", "content": "bearing capacity?"}]
        core.begin_partial(tid, "bearing capacity?")

        job = tj.start_turn_job(agent=object(), messages=messages,
                                thread_id=tid, recursion_limit=50,
                                ctx=_ctx(tid, files, transcript, []))
        _wait_done(job)

        assert job.result["final"] == "The ultimate capacity is 1159 kPa."
        assert job.result["turn_tokens"] == 321
        assert job.result["error"] is None
        # persisted to disk by the WORKER, not by any UI thread:
        saved = core.load_transcript(tid)
        assert saved[-1]["role"] == "assistant"
        assert "1159" in saved[-1]["text"]
        assert core.load_messages(tid)[-1]["content"].endswith("1159 kPa.")
        assert core.recover_partial(tid) is None      # partial cleared
        # in-memory session objects were updated in place too
        assert messages[-1]["role"] == "assistant"
        assert transcript[-1]["role"] == "assistant"

    def test_error_turn_is_persisted_with_friendly_error(self, monkeypatch):
        def boom(agent, messages, thread_id, recursion_limit=None):
            yield {"kind": "token", "text": "partial "}
            raise RuntimeError("engine exploded")
        monkeypatch.setattr(core, "stream_turn", boom)
        tid, files = _mk_conv("TJERR")
        transcript = [{"role": "user", "text": "q"}]
        job = tj.start_turn_job(object(), [{"role": "user", "content": "q"}],
                                tid, 50, _ctx(tid, files, transcript, []))
        _wait_done(job)
        assert "engine exploded" in (job.result["error"] or "")
        saved = core.load_transcript(tid)
        assert saved[-1].get("error")
        assert saved[-1]["text"].startswith("partial")


class TestFollowSemantics:
    def test_late_follower_replays_everything(self, monkeypatch):
        """A follower attaching AFTER completion still sees every event —
        the post-reconnect resume path."""
        monkeypatch.setattr(core, "stream_turn", _fake_stream(EVENTS_OK))
        tid, files = _mk_conv("TJF1")
        job = tj.start_turn_job(object(), [{"role": "user", "content": "q"}],
                                tid, 50, _ctx(tid, files, [], []))
        _wait_done(job)
        kinds = [e["kind"] for e in job.follow(poll_s=0.01)]
        assert kinds == ["tool_call", "token", "token", "turn_done"]

    def test_live_follower_sees_stream(self, monkeypatch):
        def slow(agent, messages, thread_id, recursion_limit=None):
            for e in EVENTS_OK:
                time.sleep(0.05)
                yield e
        monkeypatch.setattr(core, "stream_turn", slow)
        tid, files = _mk_conv("TJF2")
        job = tj.start_turn_job(object(), [{"role": "user", "content": "q"}],
                                tid, 50, _ctx(tid, files, [], []))
        seen = [e["kind"] for e in job.follow(poll_s=0.02)]
        assert seen[-1] == "turn_done" and len(seen) == 4


class TestRegistry:
    def test_one_active_job_per_conversation(self, monkeypatch):
        def never_ends(agent, messages, thread_id, recursion_limit=None):
            for _ in range(50):
                time.sleep(0.05)
                yield {"kind": "token", "text": "."}
        monkeypatch.setattr(core, "stream_turn", never_ends)
        tid, files = _mk_conv("TJR1")
        j1 = tj.start_turn_job(object(), [], tid, 50, _ctx(tid, files, [], []))
        j2 = tj.start_turn_job(object(), [], tid, 50, _ctx(tid, files, [], []))
        assert j1 is j2
        assert tj.get_turn_job(tid) is j1

    def test_consumed_job_is_invisible(self, monkeypatch):
        monkeypatch.setattr(core, "stream_turn", _fake_stream(EVENTS_OK))
        tid, files = _mk_conv("TJR2")
        job = tj.start_turn_job(object(), [], tid, 50, _ctx(tid, files, [], []))
        _wait_done(job)
        assert tj.get_turn_job(tid) is job
        job.consumed = True
        assert tj.get_turn_job(tid) is None
        # ...and a new turn can start afterwards
        j2 = tj.start_turn_job(object(), [], tid, 50, _ctx(tid, files, [], []))
        assert j2 is not job

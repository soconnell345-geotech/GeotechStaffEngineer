"""What the record says that it could not before (BACKLOG EV-06, EV-19).

The 2026-10-07 check1 trace review (TRACE_REVIEW §6) and the 2026-10-15
kriging session's (§10) hit the same gaps: how many graph steps a turn used
against its cap; what a vision side call was asked, at what detail, on what
image; which prompt and tools a run was given; the notice the model got for
a result moved to scratch; an app restart and a restore from the mirror; a
page rerun's crash; and which turn wrote which file. Each is now a field or
an event in ``activity.jsonl`` (or ``run.json``); these tests pin them with
fake engines, and check that no new field carries anything shaped like a
key.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
import zipfile

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

import webapp.core as core
import webapp.turn_jobs as tj
from webapp import activity_log as al


@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    monkeypatch.delenv("GEOTECH_TOKENS_PER_MINUTE", raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    tj._JOBS.clear()
    yield


class Scripted(BaseChatModel):
    """Plays ``turns[i]`` in user turn ``i``: one tool call a step, then
    ``answer``."""

    turns: list = []
    answer: str = "Done."

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        humans = [i for i, m in enumerate(messages) if m.type == "human"]
        turn = len(humans) - 1
        step = sum(1 for m in messages[humans[-1] + 1:] if m.type == "ai")
        calls = self.turns[turn] if turn < len(self.turns) else []
        if step < len(calls):
            name, args = calls[step]
            msg = AIMessage(content="", tool_calls=[
                {"name": name, "args": args,
                 "id": f"c{turn}_{step}_{len(messages)}"}])
        else:
            msg = AIMessage(content=self.answer)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _agent(model, tools=()):
    from funhouse_agent.deep.agent import build_deep_agent
    return build_deep_agent(model=model, reference_mode="off",
                            extra_tools=list(tools))


class Conversation:
    """A conversation run turn by turn through the app's own turn job."""

    def __init__(self, agent, tid):
        self.agent, self.tid = agent, tid
        core.ensure_conversation(tid)
        self.files = core.conversation_files_dir(tid)
        self.transcript: list = []
        self.messages: list = []
        self.artifacts: list = []

    def say(self, prompt, limit=40):
        self.transcript.append({"role": "user", "text": prompt})
        self.messages.append({"role": "user", "content": prompt})
        ctx = {"prompt": prompt, "temp_dir": self.files,
               "before": core.snapshot_dir(self.files),
               "before_mtimes": core.snapshot_mtimes(self.files),
               "staged_inputs": set(), "working_dir": self.files,
               "before_wd": None, "artifacts": self.artifacts,
               "artifacts_before_len": len(self.artifacts),
               "transcript": self.transcript, "trace_on": False,
               "model": "m", "behavior": core.default_behavior()}
        job = tj.start_turn_job(self.agent, self.messages, self.tid, limit,
                                ctx)
        t0 = time.time()
        while not job.done and time.time() - t0 < 60:
            time.sleep(0.05)
        assert job.done
        tj._JOBS.clear()
        time.sleep(0.02)                 # files written in the same tick
        return job

    def records(self, turn=None):
        return al.load(core.conversation_dir(self.tid), turn)


def _ping_tool():
    def ping(n: int) -> str:
        """Answer a ping."""
        return f"pong {n}"
    return StructuredTool.from_function(ping, name="ping")


# ---------------------------------------------------------------------------
# turn_end: graph steps against the cap; the prompt and tools offered
# ---------------------------------------------------------------------------

def test_turn_end_says_the_steps_used_and_the_cap():
    model = Scripted(turns=[[("ping", {"n": i}) for i in range(3)]])
    convo = Conversation(_agent(model, [_ping_tool()]), "S1")
    convo.say("ping three times")
    (end,) = [r for r in convo.records() if r["event"] == "turn_end"]
    # three tool rounds and the answer: model, tools x3, model -- plus the
    # agent's own middleware nodes; well under the cap of 40
    assert isinstance(end["graph_steps"], int)
    assert 8 <= end["graph_steps"] < 40
    assert end["step_cap"] == 40
    assert "step_allowance" not in end
    sp = end["system_prompt"]
    assert re.fullmatch(r"[0-9a-f]{16}", sp["sha"]) and sp["chars"] > 1000
    # this client keeps its tools on itself: the build's list stands in
    assert end["tools_offered_from"] == "build"
    assert "ping" in end["tools_offered"] and "task" in end["tools_offered"]


def test_a_turn_stopped_at_its_cap_says_it_reached_it():
    from webapp import activity_log
    model = Scripted(turns=[[("ping", {"n": i}) for i in range(200)]])
    agent = _agent(model, [_ping_tool()])
    folder = os.path.join(core.data_root(), "capped")
    log = activity_log.ActivityLogger(folder, turn=1)
    log.turn_start(prompt="ping forever")
    with pytest.raises(Exception) as caught:
        for _item in core.stream_turn(agent, [{"role": "user",
                                               "content": "ping forever"}],
                                      "capped", recursion_limit=12,
                                      callbacks=[log]):
            pass
    assert core.is_step_cap(caught.value)
    log.turn_end(error="step cap")
    (end,) = [r for r in al.load(folder) if r["event"] == "turn_end"]
    assert end["step_cap"] == 12
    assert 10 <= end["graph_steps"] <= 12


def test_the_tools_a_call_was_sent_are_the_tools_offered(monkeypatch):
    """A client that sends its tools in the request (the Prompter route):
    ``tools_offered`` is what the primary's calls carried."""
    pytest.importorskip("openai")
    from webapp.tests.test_record_ev14 import _prompter_model, _stream_handler
    sent: list = []
    model = _prompter_model(monkeypatch, _stream_handler(sent))
    convo = Conversation(_agent(model), "S2")
    convo.say("q")
    (end,) = [r for r in convo.records() if r["event"] == "turn_end"]
    assert end["tools_offered_from"] == "calls"
    names = sorted(t["function"]["name"] for t in sent[0]["tools"])
    assert end["tools_offered"] == names
    start = next(r for r in convo.records() if r["event"] == "turn_start")
    # the tools' wording, as one fingerprint, on the engine record
    assert re.fullmatch(r"[0-9a-f]{16}", start["engine"]["tools_sha"])


def test_the_tool_list_is_written_when_it_changes(tmp_path):
    """The names go on a turn's end when the list differs from the last one
    this process wrote there, the sha on every turn; a restore (the log
    replaced) brings back the process's first line and the list."""
    conv = str(tmp_path / "tools")

    def turn(n, tools):
        log = al.ActivityLogger(conv, turn=n)
        log.turn_start(prompt="q", engine={"build": {"tools": tools}})
        log.turn_end()
        return [r for r in al.load(conv, n) if r["event"] == "turn_end"][0]

    one, two = turn(1, ["b", "a"]), turn(2, ["a", "b"])
    assert one["tools_offered"] == ["a", "b"]
    assert one["tools_offered_from"] == two["tools_offered_from"] == "build"
    assert "tools_offered" not in two
    assert two["tools_offered_sha"] == one["tools_offered_sha"]
    three = turn(3, ["a", "b", "c"])
    assert three["tools_offered"] == ["a", "b", "c"]
    assert three["tools_offered_sha"] != one["tools_offered_sha"]
    al.note_restored(conv, source="zip", files=1)
    four = turn(4, ["a", "b", "c"])
    assert four["tools_offered"] == ["a", "b", "c"]
    events = [r["event"] for r in al.load(conv)]
    assert events.count("process_start") == 2
    assert events[events.index("restored") - 1] == "process_start"


def test_the_tools_fingerprint_moves_with_a_description():
    a = _agent(Scripted(), [_ping_tool()])
    fa = core.tools_fingerprint(a)
    assert "ping" in fa["names"] and re.fullmatch(r"[0-9a-f]{16}", fa["sha"])
    assert core.tools_fingerprint(_agent(Scripted(), [_ping_tool()])) == fa

    def ping(n: int) -> str:
        """Answer a ping, louder."""
        return f"PONG {n}"
    b = _agent(Scripted(), [StructuredTool.from_function(ping, name="ping")])
    fb = core.tools_fingerprint(b)
    assert fb["names"] == fa["names"] and fb["sha"] != fa["sha"]


# ---------------------------------------------------------------------------
# model_end of a vision side call: what it was asked, and on what
# ---------------------------------------------------------------------------

class _Seen(BaseChatModel):
    @property
    def _llm_type(self):
        return "seen"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="a reading"))])


def _png(w=64, h=48):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def _look(tmp_path, prompt, detail):
    from funhouse_agent.deep.vision_engine import LangChainVisionEngine
    from funhouse_agent.vision_tools import ask_vision
    engine = LangChainVisionEngine(_Seen(), detail=detail)

    def look(page: int) -> str:
        """Look at one region of a page."""
        return ask_vision(engine, _png(), prompt, "region",
                          where={"page": page})

    tool = StructuredTool.from_function(look, name="render_region")
    log = al.ActivityLogger(str(tmp_path), turn=1)
    tool.invoke({"page": 7}, config={"callbacks": [log]})
    (end,) = [r for r in al.load(str(tmp_path)) if r["event"] == "model_end"]
    return end


def test_a_side_calls_model_end_says_what_it_was_asked(tmp_path):
    end = _look(tmp_path, "Read the boring number.", "original")
    v = end["vision"]
    assert (v["kind"], v["page"]) == ("region", 7)
    assert v["image_px"] == [64, 48]
    assert v["detail"] == "original" and v["detail_sent"] == "original"
    assert v["prompt"] == "Read the boring number."
    assert v["prompt_chars"] == len("Read the boring number.")
    assert v["images"] == 1 and "prompt_sha" not in v
    assert "base64" not in json.dumps(end)             # never the image


def test_a_long_side_prompt_keeps_its_start_and_a_hash(tmp_path):
    prompt = ("Find every callout on this sheet. " * 30).strip()  # 1,019
    end = _look(tmp_path, prompt, "")
    v = end["vision"]
    assert v["prompt"] == prompt[:al.SIDE_PROMPT_CHARS]
    assert v["prompt_chars"] == len(prompt)
    assert v["prompt_sha"] == al.fingerprint(prompt)
    assert "detail_sent" not in v                      # none was sent


# ---------------------------------------------------------------------------
# a result moved to scratch: the notice the model received
# ---------------------------------------------------------------------------

def test_a_result_moved_to_scratch_is_logged_as_the_model_got_it(tmp_path):
    from funhouse_agent._fileio import working_dir_bound
    from funhouse_agent.deep.tests.test_keep_work_offline import (
        Played, _big_read_tool)
    agent = _agent(Played(briefs=["HELPER answer: logs A"], reads=1),
                   [_big_read_tool(rows=2000)])           # ~100,000 chars
    log = al.ActivityLogger(str(tmp_path / "c"), turn=1)
    with working_dir_bound(str(tmp_path)):
        agent.invoke({"messages": [{"role": "user", "content": "PRIMARY"}]},
                     config={"recursion_limit": 80, "callbacks": [log]})
    recs = al.load(str(tmp_path / "c"))
    (moved,) = [r for r in recs if r["event"] == "tool_delivered"
                and r.get("stage") == "graph"]
    assert moved["name"] == "big_read" and moved["agent"] == "general-purpose"
    assert moved["moved_to"].endswith("/large_tool_results/"
                                      + moved["tool_call_id"])
    assert moved["text"].startswith("Tool result too large")
    assert moved["chars"] < 5_000 < 80_000 < moved["chars_before"]
    # the tool's own output is still there, whole beside the log
    (own,) = [r for r in recs if r["event"] == "tool_end"
              and r["name"] == "big_read"]
    assert own["chars"] == moved["chars_before"]
    # a result no layer changed is not logged twice
    assert not [r for r in recs if r["event"] == "tool_delivered"
                and r["name"] == "task"]


# ---------------------------------------------------------------------------
# turn_files: which turn wrote which file
# ---------------------------------------------------------------------------

def test_each_turn_lists_the_files_it_wrote_with_size_and_hash():
    folder: dict = {}

    def write_note(text: str) -> str:
        """Write the note."""
        with open(os.path.join(folder["files"], "note.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(text)
        return "saved note.md"

    model = Scripted(turns=[[("write_note", {"text": "first"})],
                            [("write_note", {"text": "second, longer"})]])
    convo = Conversation(_agent(model, [StructuredTool.from_function(
        write_note, name="write_note")]), "S3")
    folder["files"] = convo.files
    convo.say("write it")
    convo.say("write it again")
    one = [r for r in convo.records(1) if r["event"] == "turn_files"]
    two = [r for r in convo.records(2) if r["event"] == "turn_files"]
    assert len(one) == len(two) == 1
    (f1,), (f2,) = one[0]["files"], two[0]["files"]
    assert (f1["path"], f1["how"], f1["bytes"]) == ("files/note.md", "new", 5)
    assert (f2["path"], f2["how"], f2["bytes"]) == (
        "files/note.md", "rewritten", 14)
    assert f1["sha"] != f2["sha"] and re.fullmatch(r"[0-9a-f]{16}", f2["sha"])
    # written after the turn's end, inside its slice of the log
    events = [r["event"] for r in convo.records(2)]
    assert events.index("turn_files") > events.index("turn_end")


def test_a_file_over_the_hash_limit_is_listed_by_size_and_time(
        tmp_path, monkeypatch):
    monkeypatch.setattr(al, "HASH_MAX_BYTES", 10)
    big, small = tmp_path / "scan.pdf", tmp_path / "memo.md"
    big.write_bytes(b"x" * 20)
    small.write_bytes(b"memo")
    fb, fs = al.file_facts(str(big), str(tmp_path)), \
        al.file_facts(str(small), str(tmp_path))
    assert fb["bytes"] == 20 and "sha" not in fb
    assert fb["mtime"] == round(os.stat(big).st_mtime, 3)
    assert fs["sha"] == al.fingerprint(b"memo") and "mtime" not in fs


# ---------------------------------------------------------------------------
# a process start and a restore
# ---------------------------------------------------------------------------

def test_the_first_turn_a_process_runs_in_a_conversation_says_so(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    for folder in (a, a, b):
        log = al.ActivityLogger(folder, turn=1)
        log.turn_start(prompt="q")
        log.turn_end()
    starts_a = [r for r in al.load(a) if r["event"] == "process_start"]
    starts_b = [r for r in al.load(b) if r["event"] == "process_start"]
    assert len(starts_a) == len(starts_b) == 1      # once per folder
    p = starts_a[0]
    assert p["pid"] == os.getpid() and p["agent"] == "app"
    assert p["process"] == starts_b[0]["process"]    # the same process
    assert re.fullmatch(r"[0-9a-f]{12}", p["process"])
    assert p["process_started"] <= time.time()
    assert [r["event"] for r in al.load(a)][:2] == ["process_start",
                                                    "turn_start"]


def _zip(tid="abc123"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        top = "a_footing_2026-10-20"
        zf.writestr(f"{top}/meta.json", json.dumps({"thread_id": tid,
                                                    "title": "A footing"}))
        zf.writestr(f"{top}/activity.jsonl", '{"event": "x"}\n')
        zf.writestr(f"{top}/files/memo.md", "memo")
    return buf.getvalue()


def test_a_restore_is_on_the_next_turns_record(tmp_path):
    pytest.importorskip("streamlit")
    from webapp import conversation_restore
    root = str(tmp_path / "rebuilt")
    res = conversation_restore.restore(_zip(), root=root)
    assert res.status == "restored", res.message
    conv = core.conversation_dir("abc123", root)
    # the restored record is the zip's copy until a turn adds to it
    with open(al.activity_path(conv), encoding="utf-8") as fh:
        assert fh.read() == '{"event": "x"}\n'
    log = al.ActivityLogger(conv, turn=2)
    log.turn_start(prompt="next")
    recs = al.load(conv)
    events = [r.get("event") for r in recs]
    assert events == ["x", "process_start", "restored", "turn_start"]
    restored = recs[2]
    assert restored["source"] == "zip" and restored["files"] == 3
    assert restored["agent"] == "app" and restored["replaced"] is False
    assert restored["ts"] <= recs[3]["ts"]
    # held once: the turn after does not repeat it
    al.ActivityLogger(conv, turn=3).turn_start(prompt="again")
    assert [r.get("event") for r in al.load(conv)].count("restored") == 1


def test_a_restore_from_the_mirror_is_held_for_the_next_turn(tmp_path):
    from webapp import sharepoint_store as sp
    from webapp.tests.test_wave2a_bookkeeping import RemoteFM
    root = str(tmp_path / "host")
    tid = "EV19REST0001"
    core.ensure_conversation(tid, title="Kriging", root=root)
    core.append_transcript(tid, {"role": "user", "text": "q1"}, root=root)
    fm = RemoteFM()
    store = sp.SharePointStore(file_manager=fm)
    store.mirror_conversation(tid, root=root)
    folder = store.session_folder(tid, root=root)
    other = str(tmp_path / "after_restart")
    res = store.restore_conversation(folder.rsplit("/", 1)[-1], root=other)
    assert res["status"] == "restored", res
    conv = core.conversation_dir(tid, other)
    al.ActivityLogger(conv, turn=2).turn_start(prompt="next")
    (held,) = [r for r in al.load(conv) if r["event"] == "restored"]
    assert held["source"] == "sharepoint"
    assert held["files"] == res["downloaded"] and held["bytes"] > 0
    assert held["errors"] == 0


# ---------------------------------------------------------------------------
# a page rerun that raised
# ---------------------------------------------------------------------------

SECRET = "sk-test-NEVER-RECORDED-0123456789"


def test_a_page_error_is_kept_in_the_open_conversation(tmp_path):
    from webapp import page_errors
    conv = str(tmp_path / "conv")
    os.makedirs(conv)
    page_errors.install()
    page_errors.install()                              # once per process
    logger = logging.getLogger(page_errors.LOGGER_NAME)
    assert sum(isinstance(h, page_errors.PageErrorRecorder)
               for h in logger.handlers) == 1
    page_errors.bind(conv)
    try:
        def draw_cards():
            raise ValueError(
                "There are multiple elements with the same key "
                f"'dlcard_C:\\Users\\someone\\conv\\files\\README.md' "
                f"(Authorization: Bearer {SECRET})")
        draw_cards()
    except ValueError as exc:
        logger.error("Uncaught app execution", exc_info=exc)
    finally:
        page_errors.bind(None)
    (rec,) = [r for r in al.load(conv) if r["event"] == "page_error"]
    assert rec["error"].startswith("ValueError: There are multiple")
    assert "README.md" in rec["error"]
    assert rec["where"].endswith("in draw_cards")
    assert "Traceback" in rec["traceback"]
    whole = json.dumps(rec)
    assert SECRET not in whole and "someone" not in whole
    # with no conversation bound, nothing is written anywhere
    logger.error("Uncaught app execution", exc_info=ValueError("x"))
    assert len([r for r in al.load(conv) if r["event"] == "page_error"]) == 1


def _page(tmp_path, body):
    """Run ``body`` as a page under the entry scripts' recording, by
    Streamlit; the conversation folder it binds."""
    from streamlit.testing.v1 import AppTest
    conv = str(tmp_path / "conv2").replace("\\", "/")
    os.makedirs(conv)
    script = ("import streamlit as st\n"
              "from webapp import page_errors\n"
              "with page_errors.recording():\n"
              f"    page_errors.bind({conv!r})\n"
              f"    {body}\n")
    at = AppTest.from_string(script, default_timeout=30)
    at.run()
    return at, conv


def test_the_entry_scripts_route_reaches_the_record(tmp_path):
    """The real route: Streamlit runs the page, the page raises; Streamlit
    still shows the error (raised on unchanged) and the record keeps it."""
    pytest.importorskip("streamlit")
    at, conv = _page(tmp_path, "raise RuntimeError('the page broke')")
    assert at.exception                          # the page shows it as before
    (rec,) = [r for r in al.load(conv) if r["event"] == "page_error"]
    assert rec["error"] == "RuntimeError: the page broke"
    assert rec["agent"] == "app"


def test_a_stop_or_rerun_is_not_an_error(tmp_path):
    pytest.importorskip("streamlit")
    at, conv = _page(tmp_path, "st.stop()")
    assert not at.exception
    assert not [r for r in al.load(conv) if r["event"] == "page_error"]
    # a rerun the page asks for runs the page again, unrecorded
    at, conv = _page(tmp_path / "rerun",
                     "st.session_state.n = st.session_state.get('n', 0) + 1"
                     "\n    if st.session_state.n < 2:\n        st.rerun()"
                     "\n    st.markdown('ran %d' % st.session_state.n)")
    assert not at.exception
    assert [m.value for m in at.markdown] == ["ran 2"]
    assert not [r for r in al.load(conv) if r["event"] == "page_error"]


def _frames(exc):
    import traceback
    return [(f.name, f.lineno) for f in
            traceback.extract_tb(exc.__traceback__)][1:]


def test_the_entry_wrapper_raises_on_exactly_what_the_page_raised(tmp_path):
    """``recording()`` around the page: Streamlit's stop and rerun signals
    (BaseException, not Exception) and an interrupt pass through as the
    same object with the same traceback and are not recorded; an error is
    recorded, then raised on as the same object with the same traceback."""
    pytest.importorskip("streamlit")
    from streamlit.runtime.scriptrunner_utils.exceptions import (
        RerunException, StopException)
    from webapp import page_errors
    conv = str(tmp_path / "wrap")
    os.makedirs(conv)

    def page(exc):
        def draw():
            raise exc
        draw()

    def run(make, wrapped):
        exc = make()
        try:
            if wrapped:
                with page_errors.recording():
                    page_errors.bind(conv)
                    page(exc)
            else:
                page(exc)
        except BaseException as caught:            # noqa: BLE001
            assert caught is exc
            return caught
        finally:
            page_errors.bind(None)
        raise AssertionError("nothing raised")

    for make in (StopException, lambda: RerunException(None),
                 KeyboardInterrupt, lambda: ValueError("drawn twice")):
        plain, wrapped = run(make, False), run(make, True)
        assert _frames(plain) == _frames(wrapped)
        assert (wrapped.__context__, wrapped.__cause__) == (None, None)
    (rec,) = [r for r in al.load(conv) if r["event"] == "page_error"]
    assert rec["error"] == "ValueError: drawn twice"


def test_a_page_error_on_a_linux_host_keeps_no_user_path_or_link(tmp_path):
    from webapp import page_errors
    conv = str(tmp_path / "linux")
    os.makedirs(conv)
    exc = ValueError(
        "There are multiple elements with the same key 'dlcard_/Workspace/"
        "Users/first.last@example.org/geotech_app/conversations/abc/files/"
        "README.md' (upload to https://contoso.sharepoint.com/sites/GeoTeam/"
        "Shared%20Documents/Project7/report.pdf failed; dbfs:/Volumes/team/"
        "first.last/x.csv; 12 kN/m/m)")
    assert page_errors.record(exc, conv)
    (rec,) = [r for r in al.load(conv) if r["event"] == "page_error"]
    text = json.dumps(rec, ensure_ascii=False)
    for gone in ("first.last", "example.org", "Workspace", "GeoTeam",
                 "Project7", "Volumes"):
        assert gone not in text, gone
    assert "'dlcard_README.md'" in rec["error"]
    assert "https://contoso.sharepoint.com/…" in rec["error"]
    assert "dbfs:x.csv" in rec["error"] and "kN/m/m" in rec["error"]


def test_an_error_seen_by_both_routes_is_recorded_once(tmp_path):
    from webapp import page_errors
    conv = str(tmp_path / "once")
    exc = KeyError("dlcard")
    assert not page_errors.record(exc, conv)     # no folder: nothing made
    assert not os.path.exists(conv)
    os.makedirs(conv)
    assert page_errors.record(exc, conv)
    assert not page_errors.record(exc, conv)
    page_errors.bind(conv)
    try:
        logging.getLogger(page_errors.LOGGER_NAME).error(
            "Uncaught app execution", exc_info=exc)
    finally:
        page_errors.bind(None)
    assert len([r for r in al.load(conv) if r["event"] == "page_error"]) == 1


# ---------------------------------------------------------------------------
# run.json: the fingerprint of what the run was given
# ---------------------------------------------------------------------------

def test_run_json_carries_the_prompt_and_tools_fingerprint(tmp_path,
                                                          monkeypatch):
    pytest.importorskip("planlens.tools")
    pytest.importorskip("pymupdf")
    from funhouse_agent import review_flags
    from funhouse_agent.review_eval.runner import run_task
    from funhouse_agent.review_eval.tasks import OPEN_TASKS, Task
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    docs = next(t for t in OPEN_TASKS if t.id == "fixture-markups").documents
    task = Task(id="ev06-fp", question="What is on page 1?",
                documents=list(docs), category="extract", doc_type="report",
                auto_checks=False, checks=[])
    res = run_task(task, Scripted(turns=[[]], answer="A title block."),
                   arm="baseline", arm_env={}, docs_dir=None,
                   run_dir=str(tmp_path / "run"))
    assert res["error"] is None, res.get("traceback")
    fp = res["fingerprint"]
    assert re.fullmatch(r"[0-9a-f]{16}", fp["system_prompt"]["sha"])
    assert fp["system_prompt"]["chars"] > 1000
    assert re.fullmatch(r"[0-9a-f]{16}", fp["tools"]["sha"])
    assert "open_document" in fp["tools"]["names"]
    end = next(r for r in al.load(str(tmp_path / "run"))
               if r["event"] == "turn_end")
    assert end["system_prompt"] == fp["system_prompt"]
    assert end["tools_offered"] == fp["tools"]["names"]


# ---------------------------------------------------------------------------
# no new field carries a key
# ---------------------------------------------------------------------------

#: The fields and events this train added.
NEW_FIELDS = ("graph_steps", "graph_steps_by_pass", "step_cap",
              "step_allowance", "system_prompt", "tools_offered",
              "tools_offered_sha", "tools_offered_from", "vision", "stage",
              "moved_to",
              "chars_before", "files", "n_files", "process", "pid",
              "process_started", "app_version", "source", "bytes",
              "errors", "replaced", "skipped", "partial", "where",
              "traceback", "error")
NEW_EVENTS = ("process_start", "restored", "page_error", "turn_files")

#: Shaped like a key: the masking rule of funhouse_agent.error_text, and
#: the words that name one.
KEYLIKE = re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_*\-]{6,}|\bBearer\s+\S+|"
                     r"\b[A-Za-z0-9_\-]{32,}\b|api[-_]?key|authorization",
                     re.IGNORECASE)


def _keylike(rec):
    out = []
    for key in NEW_FIELDS:
        if key in rec:
            text = json.dumps(rec[key], ensure_ascii=False)
            out += [m.group(0) for m in KEYLIKE.finditer(text)]
    if rec.get("event") == "turn_start":
        out += KEYLIKE.findall(str((rec.get("engine") or {})
                                   .get("tools_sha")))
    return out


def test_no_record_with_a_new_field_carries_a_key(tmp_path, monkeypatch):
    pytest.importorskip("openai")
    pytest.importorskip("streamlit")
    from webapp import conversation_restore, page_errors
    from webapp.tests.test_record_ev14 import (SECRET as KEY,
                                               _prompter_model,
                                               _stream_handler)
    # a turn on a client holding a key, with a restore before it
    root = str(tmp_path / "keys")
    res = conversation_restore.restore(_zip("keyconv1"), root=root)
    assert res.status == "restored"
    conv = core.conversation_dir("keyconv1", root)
    model = _prompter_model(monkeypatch, _stream_handler([]))
    agent = _agent(model)
    log = al.ActivityLogger(conv, turn=1)
    log.turn_start(prompt="q", engine=core.engine_record(agent))
    for _item in core.stream_turn(agent, [{"role": "user", "content": "q"}],
                                  "keyconv1", recursion_limit=30,
                                  callbacks=[log]):
        pass
    log.turn_end()
    with open(os.path.join(conv, "files", "memo.md"), "w") as fh:
        fh.write(f"api_key={KEY}")                       # a file with a key
    log.turn_files([al.file_facts(os.path.join(conv, "files", "memo.md"),
                                  conv)])
    # a vision side call and a page error whose text holds a key
    end = _look(tmp_path / "vision", f"Read it. Bearer {KEY}", "original")
    with pytest.raises(RuntimeError):
        with page_errors.recording():
            page_errors.bind(conv)
            raise RuntimeError(f"refused: api-key {KEY} "
                               "Authorization: Bearer x")
    page_errors.bind(None)
    recs = al.load(conv)
    seen = {r.get("event") for r in recs}
    assert set(NEW_EVENTS) <= seen
    assert any("graph_steps" in r for r in recs)
    for rec in recs:
        if rec.get("event") in NEW_EVENTS or any(k in rec for k in (
                "graph_steps", "system_prompt", "tools_offered")):
            assert KEY not in json.dumps(rec), rec.get("event")
            if rec.get("event") != "page_error":
                assert not _keylike(rec), (rec.get("event"), _keylike(rec))
    # the page error's own text is masked: the key and the bearer token go
    (err,) = [r for r in recs if r["event"] == "page_error"]
    assert KEY not in json.dumps(err) and "Bearer x" not in json.dumps(err)
    # a key in a side call's prompt is masked in its excerpt
    assert "vision" in end and KEY not in json.dumps(end)
    assert not _keylike({"vision": end["vision"]})

"""The per-turn note of files the conversation already holds.

Field session 2026-10-06: the agent downloaded a report from SharePoint in
turn 4 and worked from it in turns 4 and 5. In turn 7 -- after a SharePoint
failure in turn 6 -- it told the user it had only ever had the original
attachment and that its earlier extraction was "not verified"; in turn 11 it
said it had "not actually opened and read the full appendix PDF yet". The
file had been in the working folder throughout. The app replays earlier
turns as the user's messages and the model's final answers only, so no tool
result -- the download included -- reaches a later turn.

``core.working_files_note`` builds a short note of what is on disk, from the
attachments index, the transcript and the downloads ledger, and
``core.with_turn_note`` puts it in front of the CURRENT turn's message only.
Synthetic names throughout.
"""

import json
import os
import time

import pytest

import webapp.activity_log as al
import webapp.core as core
import webapp.turn_jobs as tj


@pytest.fixture
def conv(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    tj._JOBS.clear()
    tid = "WFN1"
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    os.makedirs(files, exist_ok=True)
    return tid, files, core.conversation_dir(tid)


def _put(files, name, data=b"x"):
    path = os.path.join(files, name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _field_session(files, conv_dir):
    """A conversation shaped like the field session: a 52-page attachment in
    turn 1, the full report fetched from SharePoint in turn 4, files made in
    turns 4 and 5."""
    _put(files, "Report_main_text.pdf", b"%PDF main" * 10)
    core.save_attachments_index(os.path.basename(conv_dir),
                                ["Report_main_text.pdf"])
    full = _put(files, "Report_full.pdf", b"%PDF full" * 100)
    core.record_download(conv_dir, "Shared Documents/General/APP/uploaded "
                         "references/Report Vol I_2026.pdf", full, 900)
    _put(files, "extraction_notes.txt", b"notes")
    _put(files, "partial.xml", b"<x/>")
    _put(files, "stratigraphy.csv", b"a,b")
    return [
        {"role": "attach", "text": "Report_main_text.pdf (90 bytes)"},
        {"role": "user", "text": "summarize"},
        {"role": "assistant", "text": "...", "artifacts": []},
        {"role": "user", "text": "extract borings"},
        {"role": "assistant", "text": "...",
         "artifacts": [os.path.join(files, "extraction_notes.txt")]},
        {"role": "user", "text": "liquefaction?"},
        {"role": "assistant", "text": "..."},
        {"role": "user", "text": "here is the link"},
        {"role": "assistant", "text": "...", "inputs": ["Report_full.pdf"],
         "artifacts": [os.path.join(files, "partial.xml")]},
        {"role": "user", "text": "make a package"},
        {"role": "assistant", "text": "...", "inputs": ["Report_full.pdf"],
         "artifacts": [os.path.join(files, "stratigraphy.csv")]},
    ]


def test_the_note_names_every_file_and_where_it_came_from(conv):
    tid, files, conv_dir = conv
    transcript = _field_session(files, conv_dir)
    note = core.working_files_note(files, transcript)
    assert note.startswith("[System note] Files already in this "
                           "conversation's working folder")
    assert "do not tell the user a file is unavailable" in note
    lines = note.splitlines()[1:]
    by_name = {ln.split("'")[1]: ln for ln in lines}
    assert set(by_name) == {"Report_main_text.pdf", "Report_full.pdf",
                            "extraction_notes.txt", "partial.xml",
                            "stratigraphy.csv"}
    assert "attached by the user" in by_name["Report_main_text.pdf"]
    full = by_name["Report_full.pdf"]
    assert "fetched to read from SharePoint" in full and "in turn 4" in full
    assert "uploaded references/Report Vol I_2026.pdf" in full
    # by its name in the working folder, never the server path (A6)
    assert full.startswith("- 'Report_full.pdf' ")
    assert files not in note and conv_dir not in note
    assert "by these names" in note
    assert "you produced it in turn 2" in by_name["extraction_notes.txt"]
    assert "in turn 5" in by_name["stratigraphy.csv"]


def test_nothing_on_disk_means_no_note(conv):
    tid, files, conv_dir = conv
    assert core.working_files_note(files, []) == ""
    # a transcript naming files that are gone says nothing about them
    gone = [{"role": "user", "text": "q"},
            {"role": "assistant", "text": "a",
             "artifacts": [os.path.join(files, "deleted.csv")],
             "inputs": ["deleted.pdf"]}]
    assert core.working_files_note(files, gone) == ""


def test_this_turns_own_upload_is_not_announced_twice(conv):
    tid, files, conv_dir = conv
    path = _put(files, "fresh_upload.pdf")
    core.save_attachments_index(tid, ["fresh_upload.pdf"])
    pending = core.attachment_note([core.Attachment(
        key="fresh_upload.pdf", path=path, size=1)])
    assert core.working_files_note(files, [], exclude_text=pending) == ""
    assert "fresh_upload.pdf" in core.working_files_note(files, [])


def test_the_oldest_are_dropped_past_the_limit(conv):
    tid, files, conv_dir = conv
    tr = [{"role": "user", "text": "q"}]
    for i in range(5):
        _put(files, f"f{i}.csv")
        tr.append({"role": "assistant", "text": "a",
                   "artifacts": [os.path.join(files, f"f{i}.csv")]})
    note = core.working_files_note(files, tr, limit=3)
    assert "The 2 oldest are not listed" in note
    assert "'f4.csv'" in note and "'f0.csv'" not in note


def test_the_ledger_keeps_one_entry_per_remote(conv):
    tid, files, conv_dir = conv
    a = _put(files, "a.pdf")
    b = _put(files, "b.pdf")
    core.record_download(conv_dir, "Shared Documents/X/r.pdf", a, 1)
    core.record_download(conv_dir, "shared documents/x/R.PDF", b, 1)
    led = core.load_downloads(conv_dir)
    assert len(led) == 1 and led[0]["local"] == os.path.abspath(b)
    with open(os.path.join(conv_dir, core.DOWNLOADS_LEDGER), "w") as fh:
        fh.write("not json")
    assert core.load_downloads(conv_dir) == []          # never raises


class TestWithTurnNote:
    def test_the_note_goes_in_front_of_the_last_user_message_only(self):
        history = [{"role": "user", "content": "first"},
                   {"role": "assistant", "content": "answer"},
                   {"role": "user", "content": "second"}]
        out = core.with_turn_note(history, "[System note] files")
        assert out[-1]["content"] == "[System note] files\n\nsecond"
        assert out[0]["content"] == "first"
        assert history[-1]["content"] == "second"      # caller's list untouched

    def test_no_note_or_no_user_message_changes_nothing(self):
        history = [{"role": "user", "content": "q"},
                   {"role": "assistant", "content": "a"}]
        assert core.with_turn_note(history, "") == history
        assert core.with_turn_note(history, "note") == history

    def test_stream_turn_shows_the_note_to_the_agent_and_saves_none(self):
        seen = {}

        class Agent:
            def stream(self, inp, config=None, stream_mode=None):
                seen["messages"] = inp["messages"]
                return iter(())

        history = [{"role": "user", "content": "bulk up the lab table"}]
        list(core.stream_turn(Agent(), history, "tid",
                              turn_note="[System note] report.pdf is here"))
        assert seen["messages"][-1]["content"].startswith(
            "[System note] report.pdf is here")
        assert history == [{"role": "user",
                            "content": "bulk up the lab table"}]


def test_a_turn_job_carries_the_note_and_records_it(conv, monkeypatch):
    tid, files, conv_dir = conv
    seen = {}

    def fake_stream(agent, messages, thread_id, recursion_limit=None,
                    callbacks=None, turn_note=None):
        seen["note"] = turn_note
        seen["last"] = messages[-1]["content"]
        yield {"kind": "turn_done", "answer": "done", "turn_tokens": 1}

    monkeypatch.setattr(core, "stream_turn", fake_stream)
    note = "[System note] Files already in this conversation's working folder"
    transcript = [{"role": "user", "text": "q"}]
    messages = [{"role": "user", "content": "q"}]
    ctx = {"prompt": "q", "temp_dir": files,
           "before": core.snapshot_dir(files), "staged_inputs": set(),
           "working_dir": files, "before_wd": None, "artifacts": [],
           "artifacts_before_len": 0, "transcript": transcript,
           "trace_on": True, "model": "m", "behavior": core.default_behavior(),
           "turn_note": note}
    job = tj.start_turn_job(object(), messages, tid, None, ctx)
    t0 = time.time()
    while not job.done and time.time() - t0 < 10:
        time.sleep(0.02)
    assert job.done and seen["note"] == note
    assert seen["last"] == "q"                  # the note is added downstream
    recs = al.load(conv_dir)
    assert recs[0]["event"] == "turn_start" and recs[0]["context_note"] == note
    saved = json.load(open(os.path.join(conv_dir, "messages.json")))
    assert note not in json.dumps(saved)        # never in the saved history


# -- the real shell (AppTest): the note reaches the agent on a later turn ----

def test_the_shell_tells_a_later_turn_about_an_earlier_download(
        monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import webapp.engine_config as engine_config
    from webapp.engine_config import EngineResolution

    seen = []

    def stream(_agent, messages, _tid, **kw):
        seen.append({"note": kw.get("turn_note"),
                     "last": messages[-1]["content"]})
        yield {"kind": "turn_done", "answer": "ok", "turn_tokens": 1}

    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.setenv("GEOTECH_UPLOAD_MODE", "http")
    for e in ("GEOTECH_APP_PROFILE", "DEV_IDENTITY", "GEOTECH_USER_EMAIL"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setattr(engine_config, "resolve_engine",
                        lambda *a, **k: EngineResolution(
                            model=object(), source="prompter",
                            model_name="fake", message=""))
    monkeypatch.setattr(core, "build_agent", lambda *a, **k: object())
    monkeypatch.setattr(core, "stream_turn", stream)
    app = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "app.py")
    at = AppTest.from_file(app, default_timeout=30).run()
    at.chat_input[0].set_value("first question").run()
    assert not at.exception and seen[-1]["note"] in (None, "")

    # In that turn the agent fetched a report from SharePoint (what the
    # download tool leaves behind: the file and the ledger entry).
    files = at.session_state["temp_dir"]
    conv_dir = os.path.dirname(os.path.abspath(files))
    local = _put(files, "Report_full.pdf", b"%PDF " * 50)
    core.record_download(conv_dir, "Shared Documents/X/Report Vol I.pdf",
                         local, 250)

    at.chat_input[0].set_value("bulk up the lab table").run()
    assert not at.exception
    note = seen[-1]["note"]
    assert note and "'Report_full.pdf'" in note
    assert "from SharePoint 'Shared Documents/X/Report Vol I.pdf'" in note
    assert seen[-1]["last"] == "bulk up the lab table"   # history unchanged
    saved = core.load_messages(at.session_state["thread_id"])
    assert all("[System note] Files already" not in m["content"]
               for m in saved)

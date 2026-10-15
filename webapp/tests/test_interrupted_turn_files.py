"""An interrupted turn's files reach the next turn's note (BACKLOG TA-28).

Field session 2026-10-14: turn 2 wrote a DIGGS file with the better grid and
was then interrupted (the app was killed mid-turn). On the next open its
checkpoint was folded in (TA-19), but the recovered entry named no files,
so turn 3's note of the conversation's files -- built from the uploads, the
answers and the downloads -- left it out. Every turn-3 helper's
``list_files`` showed the file; nobody opened it, and turn 3 built a third
grid while the better one sat in the folder.

Now the recovered entry carries the files the turn wrote (since it started),
the note says that turn never answered, and every other file in the folder
is listed too, so nothing in it is invisible to the next turn. Synthetic
names throughout.
"""

from __future__ import annotations

import json
import os
import time

import pytest

import webapp.core as core
from webapp import turn_record as tr


@pytest.fixture
def conv(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    tid = "TA28"
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    os.makedirs(files, exist_ok=True)
    return tid, files, core.conversation_dir(tid)


def _put(files, name, data=b"x", age_s=0.0):
    path = os.path.join(files, *name.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    if age_s:
        t = time.time() - age_s
        os.utime(path, (t, t))
    return path


def _rows(note):
    return {ln.split("'")[1]: ln for ln in note.splitlines()
            if ln.startswith("- '")}


def test_the_checkpoint_keeps_when_the_turn_started(conv):
    tid, _files, _conv = conv
    core.begin_partial(tid, "Can you continue?")
    started = json.load(open(core.partial_path(tid)))["started"]
    core.checkpoint_partial(tid, "so far")
    data = json.load(open(core.partial_path(tid)))
    assert data["started"] == started and data["text"] == "so far"


def test_an_interrupted_turns_files_reach_the_next_turns_note(conv):
    tid, files, conv_dir = conv
    # Turn 1 (answered): the report attached, a test-pit file produced.
    _put(files, "site_report.pdf", b"%PDF " * 20, age_s=600)
    core.save_attachments_index(tid, ["site_report.pdf"])
    tp1 = _put(files, "tp1_records.json", b"[]", age_s=500)
    core.append_transcript(tid, {"role": "user", "text": "Extract to DIGGS."})
    core.append_transcript(tid, {"role": "assistant", "text": "Done; TP-1.",
                                 "artifacts": [tp1]})
    # Turn 2 starts, fetches a report, writes a DIGGS file and a helper's
    # records -- and the process is killed before it answers.
    core.append_transcript(tid, {"role": "user", "text": "Are you able to "
                                                         "continue?"})
    core.begin_partial(tid, "Are you able to continue?")
    rec = tr.TurnRecord(question={"role": "user",
                                  "content": "Are you able to continue?"},
                        checkpoint_path=core.partial_messages_path(tid))
    from langchain_core.messages import AIMessage, ToolMessage
    rec.add([AIMessage(content="", id="a1", tool_calls=[
        {"name": "log_grid", "id": "g1", "args": {"pages": "28-43"}}]),
        ToolMessage(content="row\n" * 20_000, tool_call_id="g1",
                    name="log_grid")])
    core.checkpoint_partial(tid, "")
    fetched = _put(files, "older_report.pdf", b"%PDF old")
    core.record_download(conv_dir, "Shared Documents/X/older report.pdf",
                         fetched, 8)
    diggs = _put(files, "site_grid_v2.diggs.xml", b"<Diggs/>")
    records = _put(files, "records/spt_records.json", b"[{}]")
    _put(files, ".scratch/contact_sheet.png", b"png")      # tool scratch
    _put(files, "digest/abc/index.sqlite", b"cache")       # a cache

    got = core.recover_partial(tid)

    assert sorted(got["artifacts"]) == sorted([diggs, records])
    assert "the 2 files it saved before it stopped are below" in got["text"]
    assert "its 1 finished tool result is kept" in got["text"]
    core.append_transcript(tid, got)              # as the app does on open
    transcript = core.load_transcript(tid)
    assert sorted(transcript[-1]["artifacts"]) == sorted([diggs, records])
    assert sorted(core.artifacts_from_transcript(transcript)) == \
        sorted([tp1, diggs, records])                 # the turn's cards

    note = core.working_files_note(files, transcript)
    rows = _rows(note)
    row = rows["site_grid_v2.diggs.xml"]
    assert "you produced it in turn 2" in row
    assert "interrupted before it answered" in row
    assert "no answer of yours names this file" in row
    assert "in turn 2" in rows["records/spt_records.json"]
    assert "attached by the user" in rows["site_report.pdf"]
    assert "you produced it in turn 1" in rows["tp1_records.json"]
    assert "interrupted" not in rows["tp1_records.json"]
    assert "fetched to read from SharePoint" in rows["older_report.pdf"]
    # the fold's own file (the turn's largest result) is listed, as kept
    kept = [n for n in rows if n.startswith("kept/turn2_log-grid")]
    assert kept and "kept whole from a turn that stopped" in rows[kept[0]]
    # never the scratch or a cache
    assert "contact_sheet" not in note and "index.sqlite" not in note


def test_an_older_record_without_the_start_still_shows_the_file(conv):
    """A checkpoint from before this change has no start time: nothing is
    attributed to the turn, and the folder's other files are listed."""
    tid, files, conv_dir = conv
    core.append_transcript(tid, {"role": "user", "text": "q"})
    with open(core.partial_path(tid), "w", encoding="utf-8") as fh:
        json.dump({"prompt": "q", "text": ""}, fh)
    _put(files, "site_grid_v2.diggs.xml", b"<Diggs/>")
    got = core.recover_partial(tid)
    assert "artifacts" not in got
    core.append_transcript(tid, got)
    rows = _rows(core.working_files_note(files, core.load_transcript(tid)))
    assert "no answer of yours names it" in rows["site_grid_v2.diggs.xml"]


def test_the_folders_other_files_are_listed_and_capped(conv):
    tid, files, conv_dir = conv
    _put(files, "site_report.pdf", age_s=900)
    core.save_attachments_index(tid, ["site_report.pdf"])
    for i in range(11):
        _put(files, f"orphan_{i:02d}.csv", age_s=800 - i * 10)
    _put(files, "kept/helper_older-logs_ab12.md", b"# kept", age_s=10)
    _put(files, "half.json.part", b"{")                  # mid-write
    note = core.working_files_note(files, [])
    rows = _rows(note)
    # the named file is never pushed out by the others
    assert "attached by the user" in rows["site_report.pdf"]
    unnamed = [n for n in rows if n != "site_report.pdf"]
    assert len(unnamed) == core.UNNAMED_FILES_NOTE_MAX
    assert "orphan_00.csv" not in rows and "orphan_01.csv" not in rows
    assert "The 2 oldest are not listed" in note
    assert "what a helper that stopped had gathered" in \
        rows["kept/helper_older-logs_ab12.md"]
    assert "half.json.part" not in note
    # the list ends with the newest
    assert list(rows)[-1] == "kept/helper_older-logs_ab12.md"

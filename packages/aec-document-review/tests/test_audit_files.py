"""Audit of the files a tester keeps (2026-11-18): every conversation the
page offers to download comes back from its zip, and an upload is never
dropped without a word."""

import json
import os

import pytest

pytest.importorskip("streamlit")

import aec_document_review.core as core
from aec_document_review import conversation_export, conversation_restore


def test_a_conversation_with_no_turn_yet_comes_back_from_its_zip(
        monkeypatch, tmp_path):
    # Files uploaded, no question asked yet: no activity log exists, and the
    # page offers the download all the same.
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "here"))
    tid = core.new_thread_id()
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    os.makedirs(files, exist_ok=True)
    core.stage_uploads({}, files, [("site_plan.pdf", b"%PDF-1.4 plan")])
    core.save_attachments_index(tid, ["site_plan.pdf"])
    core.append_transcript(tid, {"role": "attach",
                                 "text": "site_plan.pdf (13 bytes)"})
    x = conversation_export.build(tid)
    other = str(tmp_path / "rebuilt")
    res = conversation_restore.restore(x.data, root=other)
    assert res.status == "restored", res.message
    got = {}
    core.load_attachments(res.thread_id, got, other)
    assert got == {"site_plan.pdf": b"%PDF-1.4 plan"}
    assert core.load_transcript(res.thread_id, other) == \
        core.load_transcript(tid)

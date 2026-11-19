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


def test_a_second_file_under_a_name_already_held_is_kept_beside_it():
    held = {"report.pdf": b"%PDF rev A"}
    # the uploader hands its files back on every rerun: the same bytes
    # again are not a new upload
    assert core.uploads_to_stage([("report.pdf", b"%PDF rev A")], held) == []
    # a revised file under the same name is: it is staged beside the first
    got = core.uploads_to_stage([("report.pdf", b"%PDF rev B")], held)
    assert got == [("report_2.pdf", b"%PDF rev B")]
    held.update({core.sanitize_key(n): d for n, d in got})
    # ... once only, however often the uploader hands it back
    assert core.uploads_to_stage([("report.pdf", b"%PDF rev B")], held) == []
    # two names that read alike once their letters are made safe
    held = {}
    first = core.uploads_to_stage([("Prüfung.pdf", b"%PDF 1")], held)
    held.update({core.sanitize_key(n): d for n, d in first})
    second = core.uploads_to_stage([("Präfung.pdf", b"%PDF 2")], held)
    assert [core.sanitize_key(n) for n, _d in first + second] == \
        ["Pr_fung.pdf", "Pr_fung_2.pdf"]
    # two such files in one batch
    batch = core.uploads_to_stage([("a b.pdf", b"1"), ("a_b.pdf", b"2")], {})
    assert [core.sanitize_key(n) for n, _d in batch] == \
        ["a_b.pdf", "a_b_2.pdf"]


def test_the_same_bytes_under_another_name_are_still_an_upload():
    held = {"report.pdf": b"%PDF rev A"}
    assert core.uploads_to_stage([("copy.pdf", b"%PDF rev A")], held) == \
        [("copy.pdf", b"%PDF rev A")]

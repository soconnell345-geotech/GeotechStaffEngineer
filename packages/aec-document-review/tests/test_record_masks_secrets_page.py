"""Document Review keeps and shows the text of an error with its secrets
masked (a repair, no switch): a failed turn's error (shown, and kept in the
transcript, the history note and the activity log), the connection
diagnostics' report, and a SharePoint tool's failure message.

An error with no secret in it is kept and shown as before. Every value is
a made-up shape (``fake``, ``example.test``), never a live key.
"""

from __future__ import annotations

import os
import time

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

import aec_document_review.core as core
import aec_document_review.turn_jobs as tj
from aec_agent_core import activity_log as al
from aec_agent_core import error_text

FAKE_TOKEN = "Fake0aB1cD2eF3gH4iJ5kL6mN7oP8qR9sT0u"
SIGNED = ("https://files.example.test/sites/a/d.pdf?download=1"
          "&tempauth=v1.FakeTempauthValue000")
SECRET_TEXT = (f"upstream refused the call for url: {SIGNED}; sent Bearer "
               f"FakeBearerValue000 with key {FAKE_TOKEN} and "
               "sk-fake-00000000-not-real")
GONE = ("FakeTempauthValue000", "FakeBearerValue000", FAKE_TOKEN, "sk-fake")
ORDINARY = ("page 12 of 40 could not be read from "
            "'Site_Investigation_Report_Rev2_Final_20261118.pdf'; "
            "pages 3-7; see https://docs.example.test/errors?page=2#limits")


@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    tj._JOBS.clear()
    yield
    tj._JOBS.clear()


class Refusing(BaseChatModel):
    """A model whose every call raises ``message``."""

    message: str = ""

    @property
    def _llm_type(self):
        return "refusing"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise ValueError(self.message)


def _failed_turn(tid: str, message: str):
    from aec_document_review.review_build import build_review_page_agent
    agent = build_review_page_agent(model=Refusing(message=message),
                                    extra_tools=[], reference_mode="off")
    prompt = "What does the report say?"
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    os.makedirs(files, exist_ok=True)
    transcript = [{"role": "user", "text": prompt}]
    core.append_transcript(tid, transcript[-1])
    ctx = {"prompt": prompt, "temp_dir": files,
           "before": core.snapshot_dir(files), "staged_inputs": set(),
           "working_dir": files, "before_wd": None, "artifacts": [],
           "artifacts_before_len": 0, "transcript": transcript,
           "trace_on": True, "model": "test-model",
           "behavior": core.default_behavior()}
    job = tj.start_turn_job(agent, [{"role": "user", "content": prompt}],
                            tid, 60, ctx)
    t0 = time.time()
    while not job.done and time.time() - t0 < 30:
        time.sleep(0.02)
    assert job.done
    return job


def _kept_texts(tid: str) -> str:
    folder = core.conversation_dir(tid)
    out = []
    for name in ("transcript.jsonl", "messages.json", "trace.jsonl",
                 "activity.jsonl"):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                out.append(fh.read())
    return "\n".join(out)


def test_a_failed_turns_error_is_shown_and_kept_masked():
    job = _failed_turn("MASK1", SECRET_TEXT)
    shown = job.result["error"]
    assert shown and "upstream refused the call" in shown
    assert "tempauth=…" in shown
    for secret in GONE:
        assert secret not in shown, secret
    (entry,) = [e for e in core.load_transcript("MASK1")
                if e.get("role") == "assistant"]
    assert entry["error"] == shown
    recs = al.load(core.conversation_dir("MASK1"))
    (end,) = [r for r in recs if r["event"] == "turn_end"]
    assert end["error"] == shown
    assert any(r["event"] == "model_error" for r in recs)
    kept = _kept_texts("MASK1")
    for secret in GONE:
        assert secret not in kept, secret


def test_a_failed_turns_error_with_no_secret_is_as_before(monkeypatch):
    job = _failed_turn("MASK2", ORDINARY)
    shown = job.result["error"]
    assert f"ValueError: {ORDINARY}" in shown
    (end,) = [r for r in al.load(core.conversation_dir("MASK2"))
              if r["event"] == "turn_end"]
    assert end["error"] == shown
    (model,) = [r for r in al.load(core.conversation_dir("MASK2"))
                if r["event"] == "model_error"]
    assert model["error"] == f"ValueError: {ORDINARY}"


def test_the_diagnostics_report_masks_what_the_record_masks(monkeypatch):
    from aec_document_review import diagnostics as diag
    text = diag.format_report([diag._check(
        "plain request (invoke)", diag.FAIL,
        f"PermissionDenied: {SECRET_TEXT}\n"
        "Set GEOTECH_PROMPTER_DISABLE_STREAMING=1 if it repeats.")])
    for secret in GONE:
        assert secret not in text, secret
    assert "GEOTECH_PROMPTER_DISABLE_STREAMING=1" in text
    assert "plain request (invoke): FAIL" in text
    # with no secret in it, the report is what it was before the masking
    checks = [diag._check("plain request (invoke)", diag.FAIL,
                          f"ValueError: {ORDINARY}"),
              diag._check("versions", diag.PASS, "core 1.2.0")]
    now = diag.format_report(checks)
    monkeypatch.setattr(error_text, "mask_secrets", lambda t: t)
    assert diag.format_report(checks) == now


def test_a_sharepoint_failure_message_is_masked(monkeypatch):
    from aec_document_review import sharepoint_log, sharepoint_tools
    logged = []
    monkeypatch.setattr(sharepoint_log, "record",
                        lambda step, message, path="", **kw:
                        logged.append(message))
    exc = RuntimeError(f"403 Client Error: Forbidden for url: {SIGNED}")
    said = sharepoint_tools._failure("download", exc, "Docs/d.pdf")
    assert "FakeTempauthValue000" not in said and "tempauth=…" in said
    assert said.startswith("SharePoint download failed")
    plain = RuntimeError("404 Client Error: Not Found for url: "
                         "https://files.example.test/sites/a/Docs/"
                         "Site_Investigation_Report_Rev2_Final_20261118.pdf")
    assert sharepoint_tools._failure("list", plain) == \
        sharepoint_tools._failure_text("list", plain)

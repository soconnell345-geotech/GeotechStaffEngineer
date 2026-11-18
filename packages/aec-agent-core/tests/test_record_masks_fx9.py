"""Finding 9 of ``module_work/record_masks/REVIEW.md``: error text the run
record keeps ONE LEVEL DOWN, where it masks nothing by name, is masked where
it is produced -- the vision probe's failure (``vision.probe_fallback``, the
profile's ``error`` and ``summary``), the photograph-page check's ``why``
(``cited_pages`` -> ``photo_pages.why``) and the deployment instructions'
read errors (``turn_start.engine.build.appended.error`` / ``file_errors``).

Every value here is made up (``Made0Up...``, ``example.test``), never a live
key of any vendor. A text with no secret comes out character for character.
"""

from __future__ import annotations

import pytest

from aec_agent_core import cited_pages, error_text, vision_probe, vision_view
from aec_agent_core.deep import instructions, vision_engine
from aec_agent_core.error_text import TEXT_NOT_CHECKED, masked_error_text

#: A made-up value: letters and digits, mixed, 21 characters.
VALUE = "Made0UpValue1For2Test"
SECRET = (f"refused: https://files.example.test/a/sheet.pdf?sig={VALUE}&se=2 "
          f"with api_key={VALUE}")
PLAIN = "refused: https://files.example.test/a/sheet.pdf?page=3 (request 0f3a)"


# ---------------------------------------------------------------------------
# the helper beside the masker
# ---------------------------------------------------------------------------

def test_the_helper_keeps_a_text_with_no_secret_as_it_came():
    assert masked_error_text(PLAIN) == PLAIN
    assert masked_error_text("") == ""
    assert masked_error_text(None) == ""
    assert masked_error_text(PLAIN, 20) == PLAIN[:20]


def test_the_helper_masks_and_then_cuts():
    out = masked_error_text(SECRET)
    assert VALUE not in out and "sig=…" in out and "api_key=…" in out
    # Cut after the masking: a cut through the value leaves no part of it.
    cut = masked_error_text(f"api_key={VALUE}", len("api_key=") + 8)
    assert cut == "api_key=…"


def test_the_helper_never_hands_on_a_text_it_could_not_check(monkeypatch):
    def boom(text):
        raise RuntimeError("the masker failed")
    monkeypatch.setattr(error_text, "_mask", boom)
    assert masked_error_text(SECRET) == TEXT_NOT_CHECKED
    assert masked_error_text(SECRET, 300) == TEXT_NOT_CHECKED
    assert masked_error_text("") == ""


# ---------------------------------------------------------------------------
# the vision probe's failure
# ---------------------------------------------------------------------------

class Refused(Exception):
    """A made-up client error that quotes the request it was refused."""


class Busy409(Exception):
    status_code = 409


class RefusingModel:
    model_name = "made-up-vision-alias"
    openai_api_base = "https://models.example.test/v1"

    def __init__(self, error, text):
        self.error, self.text = error, text

    def invoke(self, messages):
        raise self.error(self.text)


@pytest.fixture()
def probing(monkeypatch):
    for env in (vision_view.BUDGET_ENV, vision_view.DETAIL_ENV,
                vision_view.CHART_BUDGET_ENV, vision_view.MAX_PX_ENV,
                vision_engine.BUSY_TRIES_ENV):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv(vision_probe.PROBE_ENV, "1")
    monkeypatch.setattr(vision_engine, "_sleep", lambda s: None)
    vision_probe.clear_cache()
    yield
    vision_probe.clear_cache()


def _everywhere(prof) -> str:
    """Every text a failed profile hands on: its ``error``, its summary
    (``run.json``'s ``vision_profile.summary``), its dict
    (``profiles[].error``) and the fallback note (``vision.probe_fallback``
    on a side call's record, and the line a vision result carries)."""
    return "\n".join([prof.error or "", prof.summary(),
                      str(prof.to_dict()),
                      vision_view.probe_fallback_note(prof) or ""])


@pytest.mark.parametrize("error", [Refused, Busy409])
def test_a_probe_failure_keeps_no_secret(probing, error):
    pytest.importorskip("pymupdf")
    prof = vision_probe.probe(RefusingModel(error, SECRET))
    assert prof.failed and prof.error
    said = _everywhere(prof)
    assert VALUE not in said
    assert "sig=…" in said and "files.example.test" in said


def test_a_probe_failure_with_no_secret_reads_as_before(probing):
    pytest.importorskip("pymupdf")
    prof = vision_probe.probe(RefusingModel(Refused, PLAIN))
    assert prof.error == f"Refused: {PLAIN}"
    long = "x" * 400
    prof = vision_probe.probe(RefusingModel(Refused, long))
    assert prof.error == f"Refused: {long}"[:300]


def test_a_busy_probe_with_no_secret_reads_as_before(probing):
    pytest.importorskip("pymupdf")
    prof = vision_probe.probe(RefusingModel(Busy409, PLAIN))
    assert prof.failed
    assert prof.error.endswith(f"Busy409: {PLAIN}")
    assert prof.error.startswith("probe call '")


def test_a_probe_that_raised_outright_keeps_no_secret(probing, monkeypatch):
    def boom(model):
        raise Refused(SECRET)
    monkeypatch.setattr(vision_probe, "probe", boom)
    prof = vision_probe.profile_for(RefusingModel(Refused, SECRET))
    assert prof.failed and VALUE not in _everywhere(prof)
    vision_probe.clear_cache()

    def plain(model):
        raise Refused(PLAIN)
    monkeypatch.setattr(vision_probe, "probe", plain)
    prof = vision_probe.profile_for(RefusingModel(Refused, PLAIN))
    assert prof.error == f"Refused: {PLAIN}"


# ---------------------------------------------------------------------------
# the photograph-page check's ``why``
# ---------------------------------------------------------------------------

def _photo_check(monkeypatch, text):
    def boom(*args, **kwargs):
        raise Refused(text)
    monkeypatch.setattr(cited_pages, "photo_coverage", boom)
    events = []
    check = cited_pages.TurnPageCheck(
        record_folder="made-up-folder", log=events.append, mode="off",
        photo_mode="record")
    return check, events


def test_a_failed_photo_check_keeps_no_secret(monkeypatch):
    check, events = _photo_check(monkeypatch, SECRET)
    got = check._photos()
    assert got["checked"] is False
    assert VALUE not in got["why"] and "sig=…" in got["why"]
    check._record({}, "recorded", got)
    assert events and VALUE not in str(events)
    assert events[0][cited_pages.PHOTO_KEY]["why"] == got["why"]


def test_a_failed_photo_check_with_no_secret_reads_as_before(monkeypatch):
    check, _ = _photo_check(monkeypatch, PLAIN)
    assert check._photos() == {
        "checked": False, "why": f"the check failed: Refused: {PLAIN}"}


# ---------------------------------------------------------------------------
# the deployment instructions' read errors
# ---------------------------------------------------------------------------

def test_an_instructions_folder_error_keeps_no_secret(monkeypatch, tmp_path):
    def boom(path):
        raise Refused(SECRET)
    monkeypatch.setattr(instructions.os, "listdir", boom)
    ins = instructions.Instructions(str(tmp_path))
    rec = ins.record({})
    assert VALUE not in rec["error"] and "sig=…" in rec["error"]
    assert VALUE not in str(instructions.describe(ins))


def test_an_instructions_file_error_keeps_no_secret(monkeypatch, tmp_path):
    (tmp_path / "all.md").write_text("made-up text", encoding="utf-8")

    def boom(path):
        raise Refused(SECRET)
    monkeypatch.setattr(instructions, "_read", boom)
    rec = instructions.Instructions(str(tmp_path)).record({})
    assert VALUE not in str(rec) and "sig=…" in rec["file_errors"]["all.md"]


def test_an_instructions_error_with_no_secret_reads_as_before(
        monkeypatch, tmp_path):
    (tmp_path / "all.md").write_text("made-up text", encoding="utf-8")

    def boom(path):
        raise Refused(PLAIN)
    monkeypatch.setattr(instructions, "_read", boom)
    rec = instructions.Instructions(str(tmp_path)).record({})
    assert rec["file_errors"] == {"all.md": f"Refused: {PLAIN}"}
    assert instructions._reason(Refused("")) == "Refused"
    assert instructions._reason(
        FileNotFoundError(2, "No such file or directory", "/made/up")) == (
        "FileNotFoundError: No such file or directory")

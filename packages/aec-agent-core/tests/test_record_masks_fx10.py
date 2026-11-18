"""Finding 10 of ``module_work/record_masks/REVIEW.md``: the review suite's
result (``run.json``, ``results.json``) keeps no secret in the ``detail`` of
a check that raised, nor in the vision profile's ``summary`` and its
``profiles[].error`` -- masked where they are made, and, for a result saved
before that, when it is read back (:func:`runner._masked_errors`).

Every value here is made up (``Made0Up...``, ``example.test``), never a live
key of any vendor. A text with no secret comes out character for character.
"""

from __future__ import annotations

import json

import pytest

from aec_agent_core import vision_probe
from aec_agent_core.review_eval import checks, runner

#: A made-up value: letters and digits, mixed, 21 characters.
VALUE = "Made0UpValue1For2Test"
SECRET = (f"refused: https://files.example.test/a/sheet.pdf?sig={VALUE}&se=2 "
          f"with api_key={VALUE}")
PLAIN = "refused: https://files.example.test/a/sheet.pdf?page=3 (request 0f3a)"


class Refused(Exception):
    """A made-up error that quotes the request it was refused."""


@pytest.fixture()
def raising_check(monkeypatch):
    """A made-up check type that raises the text it is given."""
    def check_made_up(answer, text="", **_):
        raise Refused(text)
    monkeypatch.setitem(checks.CHECKS, "made_up_raises", check_made_up)
    return "made_up_raises"


def test_a_check_that_raised_keeps_no_secret(raising_check):
    got = checks.run_check({"type": raising_check, "text": SECRET}, "x")
    assert not got["passed"]
    assert VALUE not in got["detail"]
    assert got["detail"].startswith("check error: Refused: refused: ")
    assert "sig=…" in got["detail"] and "api_key=…" in got["detail"]


def test_a_check_that_raised_with_no_secret_reads_as_before(raising_check):
    got = checks.run_check({"type": raising_check, "text": PLAIN}, "x")
    assert got == {"type": raising_check, "label": raising_check,
                   "passed": False, "info": False,
                   "detail": f"check error: Refused: {PLAIN}"}
    # A check that does not raise: its detail is the check's own words.
    said = checks.run_check(
        {"type": "contains_all", "terms": ["api_key=" + VALUE]},
        "the answer quotes api_key=" + VALUE)
    assert said["passed"]


def test_a_measured_profiles_summary_keeps_no_secret(monkeypatch):
    # The summary and the dict are made from the profile's error, which the
    # probe masks when it sets it (finding 9); the suite writes both.
    prof = vision_probe.VisionProfile(
        error=vision_probe._error_text(f"Refused: {SECRET}"), failed=True)
    monkeypatch.setattr(vision_probe, "enabled", lambda: True)
    monkeypatch.setattr(vision_probe, "cached_profile", lambda model: prof)
    got = runner._vision_profile(object())
    assert VALUE not in json.dumps(got) and "sig=…" in got["summary"]
    monkeypatch.setattr(vision_probe, "cached_profile", lambda model: None)
    monkeypatch.setattr(vision_probe, "cached_profiles", lambda: [prof, prof])
    got = runner._vision_profile(object())
    assert VALUE not in json.dumps(got)
    assert "sig=…" in got["profiles"][1]["error"]


def _old_result(text):
    """A result as a run before the masking saved it."""
    return {
        "task": "a", "arm": "baseline", "answer": f"it said {text}",
        "error": None,
        "turns": [{"error": None, "answer": "b"}],
        "checks": [
            {"type": "made_up", "label": "one", "passed": False,
             "info": False, "detail": f"check error: Refused: {text}"},
            {"type": "contains_all", "label": "two", "passed": True,
             "info": False, "detail": f"found: {text}"}],
        "vision_profile": {
            "summary": f"an unknown model: image budget not determined "
                       f"(Refused: {text}) -- probe FAILED",
            "error": f"Refused: {text}",
            "profiles": [{"error": f"Refused: {text}", "failed": True},
                         {"error": None, "failed": False}]},
        "score": {"checks_passed": 1, "checks_total": 2, "passed": False}}


def test_an_old_result_read_back_keeps_no_secret_in_its_error_texts():
    old = _old_result(SECRET)
    out = runner._masked_errors(json.loads(json.dumps(old)))
    assert set(out) == set(old)
    assert VALUE not in out["checks"][0]["detail"]
    assert "sig=…" in out["checks"][0]["detail"]
    profile = out["vision_profile"]
    assert VALUE not in json.dumps(profile, ensure_ascii=False)
    assert "sig=…" in profile["summary"]
    assert "sig=…" in profile["profiles"][0]["error"]
    assert profile["profiles"][1] == {"error": None, "failed": False}
    # Not an error's text: the answer, and what a check that ran found.
    assert out["answer"] == old["answer"]
    assert out["checks"][1] == old["checks"][1]
    assert out["score"] == old["score"]


def test_an_old_result_with_no_secret_is_read_back_as_it_was():
    old = _old_result(PLAIN)
    assert runner._masked_errors(json.loads(json.dumps(old))) == old
    assert runner._masked_errors({"checks": None, "vision_profile": None}) \
        == {"checks": None, "vision_profile": None}
    assert runner._masked_errors({"vision_profile": {"probe": "off"}}) \
        == {"vision_profile": {"probe": "off"}}


def test_a_saved_result_is_rewritten_only_when_masking_changed_it(tmp_path):
    saved = tmp_path / "run.json"
    text = json.dumps(_old_result(PLAIN))        # not the runner's layout
    saved.write_text(text, encoding="utf-8")
    got = runner._masked_saved(json.loads(text), saved)
    assert got == _old_result(PLAIN)
    assert saved.read_text(encoding="utf-8") == text
    text = json.dumps(_old_result(SECRET))
    saved.write_text(text, encoding="utf-8")
    got = runner._masked_saved(json.loads(text), saved)
    on_disk = json.loads(saved.read_text(encoding="utf-8"))
    assert on_disk == got
    assert VALUE not in on_disk["checks"][0]["detail"]
    assert VALUE not in json.dumps(on_disk["vision_profile"])

"""Finding 10 of ``module_work/record_masks/REVIEW.md``, through the suite:
a ``run.json`` saved before the masking is not carried into ``results.json``
(and is not written again) with its error texts raw, on a re-score and on a
plain re-run that finds the task done already.

Every value here is made up, never a live key of any vendor.
"""

from __future__ import annotations

import json

import pytest

#: A made-up value: letters and digits, mixed, 21 characters.
VALUE = "Made0UpValue1For2Test"
SECRET = (f"refused: https://files.example.test/a/sheet.pdf?sig={VALUE}&se=2 "
          f"with api_key={VALUE}")
PLAIN = "refused: https://files.example.test/a/sheet.pdf?page=3 (request 0f3a)"
TASK = "calc-bearing-consistency"
ANSWER = (r"Yes. \(q_{ult} = 1{,}195.3\) kPa (printed p. 3), FS = 3.0 "
          r"(p. 1), so \(q_{all} = 398.4\) kPa, matching p. 4.")


class _NoModel:
    model = "none"

    def invoke(self, *a, **k):
        raise AssertionError("a saved run must not call the model")

    bind_tools = invoke


def _saved(tmp_path, text):
    run_dir = tmp_path / "runs" / "baseline" / TASK
    run_dir.mkdir(parents=True)
    old = {"task": TASK, "arm": "baseline", "answer": ANSWER, "files": [],
           "tool_calls": [], "error": None,
           "turns": [{"error": f"RuntimeError: {text}", "answer": ANSWER}],
           "checks": [{"type": "contains_all", "label": "one",
                       "passed": False, "info": False,
                       "detail": f"check error: RuntimeError: {text}"}],
           "vision_profile": {
               "summary": f"an unknown model: image budget not determined "
                          f"(RuntimeError: {text}) -- probe FAILED",
               "error": f"RuntimeError: {text}",
               "profiles": [{"error": f"RuntimeError: {text}"}]},
           "score": {"checks_passed": 0, "checks_total": 1, "passed": False}}
    path = run_dir / "run.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    return path, old


def _suite(tmp_path, **kw):
    from aec_document_review.review_eval import score_review_suite
    return score_review_suite(model=_NoModel(), out_dir=tmp_path,
                              arms=("baseline",), ids=[TASK], verbose=False,
                              **kw)


@pytest.mark.parametrize("rescore", [True, False])
def test_an_old_run_is_not_carried_on_with_its_secrets(tmp_path, rescore):
    path, _ = _saved(tmp_path, SECRET)
    out = _suite(tmp_path, rescore=rescore)
    run = out["results"]["baseline"][TASK]
    assert VALUE not in json.dumps(run)
    assert VALUE not in path.read_text(encoding="utf-8")
    assert "sig=…" in run["vision_profile"]["summary"]
    assert "sig=…" in run["turns"][0]["error"]
    for kept in tmp_path.glob("*.json"):
        assert VALUE not in kept.read_text(encoding="utf-8"), kept.name


def test_an_old_run_with_no_secret_is_left_as_it_was(tmp_path):
    path, old = _saved(tmp_path, PLAIN)
    before = path.read_text(encoding="utf-8")
    out = _suite(tmp_path)
    assert out["results"]["baseline"][TASK] == old
    assert path.read_text(encoding="utf-8") == before

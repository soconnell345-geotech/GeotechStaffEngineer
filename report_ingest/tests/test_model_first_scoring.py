"""Model-first reading of one hand-truthed item, scored as today's readers
are (BACKLOG RI-87 piece 9). No model is called: a scripted engine answers.

What is checked: the placement counts the plan's 90 % gate reads, that a
call which raised leaves an error the next run reads again (and no calls in
the arm), that an answer in prose is a scored error with its call recorded,
that the arm's run-file facts say what ran, the run-file check that keeps
one ``out_dir`` to one arm, and that today's path does not import this
module. Every value here is invented.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from report_ingest import model_first_scoring as mfs
from report_ingest.tests.fake_engine import FakeEngine

pymupdf = pytest.importorskip("pymupdf")


@pytest.fixture(scope="module")
def synthetic_pdf():
    from report_ingest.tests.narrative_fixtures import build_narrative_report

    return build_narrative_report().pdf


@pytest.fixture()
def doc(synthetic_pdf):
    from planlens.document import open_document

    opened = open_document(synthetic_pdf, name="made_up")
    yield opened
    opened.close()


def _engine(turns):
    engine = FakeEngine(turns, name="fake-first")
    engine.accepts_pdf = True
    return engine


CALC_TRUTH = {"id": "lateral_pile__R00_p20", "report": "R00",
              "kind": "lateral_pile", "pages": [20],
              "inputs": [{"name": "Pile-head load", "value": 120.0,
                          "unit": "kN", "page": 20}], "results": []}


def test_the_placement_counts_are_what_the_gate_reads():
    found = [{"counts": {"placed": 3, "moved": 1, "not_found": 1,
                         "no_text": 2},
              "items": [{"page_level": True}, {"page_level": False}, {}]},
             {"counts": {"placed": 1}, "items": []}, None]
    got = mfs.placement_counts(found)
    assert (got["values"], got["with_text"], got["at_a_spot"],
            got["page_level"]) == (8, 6, 4, 1)
    assert got["counts"]["not_found"] == 1 and got["counts"]["several"] == 0
    assert mfs.placement_counts([])["values"] == 0


def test_a_calculation_read_model_first_names_its_arm(doc):
    pytest.importorskip("report_ingest.locate")
    from report_ingest.model_first import READ_PROMPT_VERSION
    from report_ingest.tests import test_graph_model_first as tgm

    engine = _engine([tgm._calc()])
    before, after, arm = mfs.first_calc(CALC_TRUTH, doc, engine,
                                        report_id="R00")
    assert engine.n_calls == 1
    assert (arm["model_first"], arm["route"], arm["prompt_version"]) == (
        "on", "pdf", READ_PROMPT_VERSION)
    assert [c["pages"] for c in arm["calls"]] == [[20]]
    assert arm["placement"]["values"] >= 1
    assert after.error is None and after.n_pages == 1
    assert after.record == after.model_record
    assert after.scores["inputs"].found == 1
    assert before.stage != after.stage


def test_a_call_that_raised_is_an_error_with_no_calls_in_the_arm(doc):
    engine = _engine([{"raise": RuntimeError("made-up refusal")}])
    before, after, arm = mfs.first_calc(CALC_TRUTH, doc, engine,
                                        report_id="R00")
    assert "made-up refusal" in after.error
    assert "calls" not in arm and arm["model_first"] == "on"
    # A failed call leaves the before column scored.
    assert before.error is None
    # The cluster's retry reads the error off the saved run file.
    from report_ingest.cluster_scoring import _saved_failure
    assert _saved_failure({"after": after.to_dict()})


def test_an_answer_in_prose_is_scored_as_an_error_with_its_call(doc):
    engine = _engine([{"text": "Made-up words and no answer."}])
    truth = {"id": "R00_p7", "pages": [7], "depth_unit": "m", "fields": {},
             "layers": [{"top": 0.0}], "samples": [], "water": []}
    _before, after, arm = mfs.first_log(truth, doc, engine, report_id="R00")
    assert after.error
    assert [c["answered"] for c in arm["calls"]] == [False]


def test_a_route_the_engine_cannot_send_is_refused(doc):
    engine = FakeEngine([], name="fake-no-pdf")
    with pytest.raises(ValueError):
        mfs.first_calc(CALC_TRUTH, doc, engine, page_route="pdf")
    assert engine.n_calls == 0


def test_one_out_dir_holds_one_arm(tmp_path):
    from report_ingest.cluster_scoring import _arm_clash, _FirstArm

    on, off = _FirstArm("on"), _FirstArm("off")
    path = tmp_path / "item.json"
    assert _arm_clash(path, on) is None                   # nothing saved
    path.write_text(json.dumps({"after": {}}), encoding="utf-8")
    assert _arm_clash(path, off) is None                  # saved before RI-87
    assert "read with model first off" in _arm_clash(path, on)
    path.write_text(json.dumps({"model_first": "on"}), encoding="utf-8")
    assert _arm_clash(path, on) is None
    assert "this run is off" in _arm_clash(path, off)
    path.write_text("{not json", encoding="utf-8")
    assert _arm_clash(path, on) is None


def test_today_s_path_does_not_import_this_module():
    code = ("import sys; import report_ingest.cluster_scoring, "
            "report_ingest.graph; "
            "print('report_ingest.model_first_scoring' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "False"

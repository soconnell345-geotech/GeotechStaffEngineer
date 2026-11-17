"""RI-99 (5): a summary row's ``sample_id`` described, behind a switch.

Today's lab answer schema describes a test's own ``sample_id`` ("as
printed") and gives a summary row's none. ``GEOTECH_INGEST_LAB_SAMPLE_ID_
AS_PRINTED`` (off) gives the row the test's own description on the
model-first schema. Off, the schema and the saved prompt version are what
they were; on, the lab answers are saved under a version of their own.
Made-up values only.
"""

from __future__ import annotations

import copy

import pymupdf
import pytest

from report_ingest import model_first as mf
from report_ingest.engine import strict_schema


def _without_row_description(schema):
    out = copy.deepcopy(schema)
    said = out["$defs"]["ReadRow"]["properties"]["sample_id"].pop(
        "description")
    return out, said


def test_off_the_lab_schema_and_its_version_are_today_s(monkeypatch):
    monkeypatch.delenv(mf.LAB_SAMPLE_ID_AS_PRINTED_ENV, raising=False)
    assert mf.lab_sample_id_as_printed_enabled() is False
    assert mf.answer_schema("lab") is mf.LabSheetReading
    row = strict_schema(mf.LabSheetReading)["$defs"]["ReadRow"]
    assert "description" not in row["properties"]["sample_id"]
    assert mf.prompt_version("lab") == mf.READ_PROMPT_VERSION


def test_on_the_row_s_sample_id_is_described_and_nothing_else_moves():
    off = strict_schema(mf.answer_schema("lab"))
    with mf.use_lab_sample_id_as_printed():
        on = strict_schema(mf.answer_schema("lab"))
        plain = mf.answer_schema("lab").model_json_schema()
        said = mf.instruction("lab", "R00", [4])
    rest, description = _without_row_description(on)
    assert rest == off
    assert description == "as printed" == strict_schema(
        mf.LabSheetReading)["$defs"]["ReadTest"]["properties"][
            "sample_id"]["description"]
    # Same class names and docstrings: only the one description is new.
    rest, _ = _without_row_description(plain)
    assert rest == mf.LabSheetReading.model_json_schema()
    assert said == mf.instruction("lab", "R00", [4])     # no words change


@pytest.mark.parametrize("stage", ["logs", "lab", "calc", "narrative"])
def test_on_only_the_lab_answers_are_saved_under_their_own_version(stage):
    from report_ingest.graph import _still_current
    from report_ingest.model_first_pages import PageGroup

    parts = [PageGroup(stage, [7])]
    off = {"pages": [7], "route": "pdf",
           "prompt_version": mf.prompt_version(stage)}
    assert off["prompt_version"] == mf.READ_PROMPT_VERSION
    with mf.use_lab_sample_id_as_printed():
        on = {**off, "prompt_version": mf.prompt_version(stage)}
        assert (on["prompt_version"] == mf.READ_PROMPT_VERSION + "+sid") \
            is (stage == "lab")
        assert _still_current(on, parts, "pdf", stage)
        assert _still_current(off, parts, "pdf", stage) is (stage != "lab")
    assert _still_current(on, parts, "pdf", stage) is (stage != "lab")


def test_the_switch_reads_the_environment(monkeypatch):
    monkeypatch.setenv(mf.LAB_SAMPLE_ID_AS_PRINTED_ENV, "1")
    assert mf.lab_sample_id_as_printed_enabled() is True
    assert mf.answer_schema("lab") is not mf.LabSheetReading
    with mf.use_lab_sample_id_as_printed(False):
        assert mf.answer_schema("lab") is mf.LabSheetReading
    monkeypatch.setenv(mf.LAB_SAMPLE_ID_AS_PRINTED_ENV, "")
    assert mf.lab_sample_id_as_printed_enabled() is False


def test_a_lab_window_read_with_the_switch_on_builds_the_same_record():
    """The reader offers the switched-on schema, the answer is built as
    today's is, and the result says which version it was read under."""
    from planlens.document import open_document
    from report_ingest import model_first_read as mfr
    from report_ingest.model_first_pages import PageGroup
    from report_ingest.tests.fake_engine import FakeEngine

    made = pymupdf.open()
    for n in range(3):
        made.new_page().insert_text((72, 72), f"MADE-UP PAGE {n}")
    data = made.tobytes()
    made.close()
    answer = {
        "depth_unit": "ft",
        "tests": [{"kind": "summary_table", "test_type": "Made-up summary",
                   "rows": [{"investigation_id": "QX-2", "sample_id": "ST 4",
                             "depth_top": 6.0, "wc": 21.0}],
                   "prov": {"page": 1}}]}

    def read():
        doc = open_document(data, name="made_up")
        engine = FakeEngine([{"final": mf.answer_schema(
            "lab").model_validate(answer)}])
        engine.accepts_pdf = True
        try:
            got = mfr.read_lab(doc, [PageGroup("lab", [1], key="lab-1")],
                               engine, report_id="R00")[0]
        finally:
            doc.close()
        return got, engine.calls[0]

    off, off_call = read()
    with mf.use_lab_sample_id_as_printed():
        on, on_call = read()
    assert off["prompt_version"] == mf.READ_PROMPT_VERSION
    assert on["prompt_version"] == mf.READ_PROMPT_VERSION + "+sid"
    assert on["tests"] == off["tests"]
    assert on["tests"][0]["result"]["rows"][0]["sample_id"] == "ST 4"
    assert on_call["messages"][0]["content"][1]["text"] \
        == off_call["messages"][0]["content"][1]["text"]

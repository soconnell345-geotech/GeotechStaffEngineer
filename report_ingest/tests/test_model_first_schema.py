"""Model-first ingest (BACKLOG RI-87), piece 4, offline: the answer schemas
with what plan section 8.5 adds, the instructions, and where the new fields
go in the record.

The first test holds today's ingest still: the four readers' answer schemas,
as the model is sent them, are master's to the byte. Made-up values only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest

from report_ingest import lab_reader, log_reader
from report_ingest import model_first as mf
from report_ingest.calc_reader import CalcReading
from report_ingest.engine import strict_schema
from report_ingest.narrative_reader import NarrativeReading

#: sha256 of ``json.dumps(strict_schema(cls), ensure_ascii=False)`` for each
#: of today's reading classes, taken on master 32b9645 (2026-10-25). The same
#: on Python 3.12, 3.13 and 3.14 and pydantic 2.13.4 and 2.14.1.
MASTER_SCHEMAS = {
    "LogReading":
        "00ec2299c92445c601ce6de84a80e3e1da06101a3bd25f8994136cb0ba8e2d8f",
    "LabSheetReading":
        "6857fd4e7c8052efeb5099ee7137394cbb748e50f131d5d4ac3cd5c1c536f8b7",
    "CalcReading":
        "b9fd730d9dc880832b692852fb998394d918e03121d74d59b88a19ea827d29c6",
    "NarrativeReading":
        "aa12cef0562e3ccc60e83bbee64b033b228539b94b343255006bcdbc9557dd43",
}

#: sha256 of what the model-first readers ask (each stage's instruction as
#: sent, its vocabulary lists filled in, and the logs' and lab's answer
#: schemas), by READ_PROMPT_VERSION.
MODEL_FIRST_ASKS = {
    "mf-1": "99aafd0dbd196bf8961ea25054e9c8646f15c40c711de3788de477f1c151246d",
}

TODAY = {"LogReading": log_reader.LogReading,
         "LabSheetReading": lab_reader.LabSheetReading,
         "CalcReading": CalcReading, "NarrativeReading": NarrativeReading}

OPEN_WINDOW = (-math.inf, math.inf)


def _digest(value):
    text = json.dumps(value, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _asks():
    return _digest({"instructions": {stage: mf.instruction(stage, "R00", [0])
                                     for stage in mf.INSTRUCTIONS},
                    "logs": strict_schema(mf.LogReading),
                    "lab": strict_schema(mf.LabSheetReading)})


# ---------------------------------------------------------------------------
# today's schemas do not move
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(MASTER_SCHEMAS))
def test_today_s_reader_schemas_are_master_s_to_the_byte(name):
    # Imported model_first above: its subclasses leave their parents alone.
    assert _digest(strict_schema(TODAY[name])) == MASTER_SCHEMAS[name], (
        f"{name}'s answer schema is not master's. It is what the model is "
        f"sent on today's path, so a change to it is model-facing and is "
        f"measured before it merges; model-first's additions go in "
        f"report_ingest.model_first (RI-87 section 8.5). If no reader class "
        f"changed, a pydantic release changed the JSON it writes: check, "
        f"then take the digest again.")


def test_the_plain_baseline_reads_the_package_s_measured_instructions():
    from module_work.report_ingest_harness import run_plain_baseline as rpb

    assert rpb.INSTRUCTIONS is mf.PLAIN_INSTRUCTIONS
    assert rpb.instruction("calc", "R00", [4]) == mf.instruction(
        "calc", "R00", [4], texts=mf.PLAIN_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# the model-first schemas: today's plus the additions, nothing else
# ---------------------------------------------------------------------------

def _nodes(schema):
    """The schema's classes by title: the root and every $defs entry."""
    out = {schema["title"]: schema}
    for node in (schema.get("$defs") or {}).values():
        out[node["title"]] = node
    return out


def _without_additions(new, old):
    """``new`` with the added fields taken out and today's description put
    back on a field whose description alone changed."""
    new = copy.deepcopy(new)
    new.get("$defs", {}).pop("HeaderField", None)
    old_nodes = _nodes(old)
    for title, node in _nodes(new).items():
        for name in mf.ADDED_FIELDS.get(title, ()):
            node["properties"].pop(name)
            node["required"].remove(name)
        if title == "ReadSample":
            node["properties"]["recovery_percent"] = copy.deepcopy(
                old_nodes[title]["properties"]["recovery_percent"])
    return new


@pytest.mark.parametrize("new, old", [
    (mf.LogReading, log_reader.LogReading),
    (mf.LabSheetReading, lab_reader.LabSheetReading),
])
def test_a_model_first_schema_is_today_s_plus_the_additions(new, old):
    new_schema, old_schema = strict_schema(new), strict_schema(old)
    assert new.__name__ == old.__name__          # the answer tool names it
    # Same names, same docstrings, same order: only the additions differ.
    assert json.dumps(_without_additions(new_schema, old_schema)) == \
        json.dumps(old_schema)


def test_the_new_fields_come_last_in_their_classes():
    for cls, name in ((mf.LogReading, "LogReading"),
                      (mf.ReadSample, "ReadSample"),
                      (mf.ReadRow, "ReadRow")):
        added = mf.ADDED_FIELDS[name]
        assert tuple(cls.model_fields)[-len(added):] == added
    sample = list(mf.ReadSample.model_fields)
    assert sample.index("recovery_percent") == \
        list(log_reader.ReadSample.model_fields).index("recovery_percent")


def test_every_new_description_says_what_goes_there_as_printed():
    nodes = {**_nodes(strict_schema(mf.LogReading)),
             **_nodes(strict_schema(mf.LabSheetReading))}
    said = []
    for title, names in mf.ADDED_FIELDS.items():
        for name in names:
            text = nodes[title]["properties"][name]["description"]
            said.append(text)
            assert "printed" in text, (title, name, text)
    said.append(nodes["ReadSample"]["properties"]["recovery_percent"]
                ["description"])
    for name in ("name", "value"):
        said.append(nodes["HeaderField"]["properties"][name]["description"])
    assert all("as printed" in text or "if not printed" in text
               for text in said)
    assert not any("exact" in text.lower() for text in said)


def test_today_s_classes_refuse_the_new_fields_and_the_new_ones_take_them():
    answer = _log_answer()
    reading = mf.LogReading.model_validate(answer)
    assert reading.project_number == "EX-0000"
    assert reading.samples[0].recovery_length == 14.0
    with pytest.raises(Exception):
        log_reader.LogReading.model_validate(answer)
    assert isinstance(reading, log_reader.LogReading)


def test_each_stage_has_its_answer_schema():
    assert mf.answer_schema("logs") is mf.LogReading
    assert mf.answer_schema("lab") is mf.LabSheetReading
    assert mf.answer_schema("calc") is CalcReading
    assert mf.answer_schema("narrative") is NarrativeReading
    with pytest.raises(ValueError):
        mf.answer_schema("soundings")


# ---------------------------------------------------------------------------
# the instructions
# ---------------------------------------------------------------------------

def test_only_the_logs_instruction_differs_from_the_plain_one():
    assert set(mf.INSTRUCTIONS) == set(mf.PLAIN_INSTRUCTIONS)
    for stage in ("lab", "calc", "narrative"):
        assert mf.INSTRUCTIONS[stage] == mf.PLAIN_INSTRUCTIONS[stage]
    logs = mf.INSTRUCTIONS["logs"]
    assert logs != mf.PLAIN_INSTRUCTIONS["logs"]
    for asked in ("the project name and number", "the client",
                  "any other field the header prints", "its recovery"):
        assert asked in logs
    assert not any("exact" in text.lower()
                   for text in mf.INSTRUCTIONS.values())


def test_what_the_readers_ask_is_versioned():
    digest = _asks()
    assert MODEL_FIRST_ASKS.get(mf.READ_PROMPT_VERSION) == digest, (
        f"what the model-first readers ask changed: raise "
        f"READ_PROMPT_VERSION and record {digest!r} under it")


def test_an_instruction_names_its_pages_on_either_route():
    by_pdf = mf.instruction("logs", "R00", [7, 8])
    assert "report R00, pages 7, 8" in by_pdf
    assert "PDF page 1 is page 7, PDF page 2 is page 8" in by_pdf
    by_pictures = mf.instruction("logs", "R00", [7, 8], route="pictures")
    assert "headed by its page index" in by_pictures
    assert "PDF page" not in by_pictures
    lab = mf.instruction("lab", "R00", [3])
    assert "summary_table" in lab and "{" not in lab


# ---------------------------------------------------------------------------
# where the new fields go
# ---------------------------------------------------------------------------

def _prov(page=4):
    return {"page": page, "bbox": None, "from_image": False, "note": ""}


def _sample(sample_id, top, **extra):
    return {"sample_id": sample_id, "top": top, "bottom": top + 1.5,
            "kind": "spt", "prov": _prov(), **extra}


def _log_answer(**changes):
    answer = {
        "investigation_id": "B-7", "depth_unit": "ft",
        "project": "Sample Project", "project_number": "EX-0000",
        "client": "Sample Client",
        "header_fields": [{"name": "Weather:", "value": "Overcast"},
                          {"name": "Project No.", "value": "EX-0000"}],
        "samples": [_sample("S-1", 2.5, recovery_length=14.0,
                            recovery_length_unit="in"),
                    _sample("S-2", 5.0, recovery_percent=80.0)],
    }
    answer.update(changes)
    return answer


def _built(answer, window=OPEN_WINDOW):
    reading = mf.LogReading.model_validate(answer)
    builder = log_reader._Builder(reading, window, [4], "R00")
    return reading, builder.build()


def test_the_project_fields_go_where_the_log_scorer_reads_them():
    from report_ingest.log_scoring import _record_fields

    reading, built = _built(_log_answer())
    assert built.fields == {}                   # today's builder ignores them
    inv, unresolved = mf.apply_log_additions(reading, built)
    assert unresolved == [] and built.fields == {}
    assert inv.fields["project"] == "Sample Project"
    assert inv.fields["project_number"] == "EX-0000"
    assert inv.fields["client"] == "Sample Client"
    # A header field goes in under its label as printed; one with the same
    # label and value as a field already there is that field.
    assert inv.fields["Weather"] == "Overcast"
    assert "Project No." in inv.fields          # another label: kept
    seen = _record_fields(inv)
    assert seen["project_number"] == "EX-0000"
    assert seen["client"] == "Sample Client"


def test_a_header_field_under_its_printed_label_is_scored():
    from report_ingest.log_scoring import score_record

    answer = _log_answer(header_fields=[
        {"name": "Foreman:", "value": "A. Sample"}])
    inv, _ = mf.apply_log_additions(*_built(answer))
    truth = {"id": "R00_p4", "depth_unit": "ft",
             "fields": {"foreman": "A. Sample", "project_number": "EX-0000",
                        "client": "Sample Client"}}
    assert score_record(truth, [inv]).scores["fields"].found == 3


def test_a_header_label_given_twice_with_two_values_keeps_both():
    answer = _log_answer(header_fields=[
        {"name": "Weather", "value": "Overcast"},
        {"name": "weather:", "value": "Overcast"},
        {"name": "WEATHER", "value": "Light rain"}])
    inv, _ = mf.apply_log_additions(*_built(answer))
    assert inv.fields["Weather"] == "Overcast"
    assert inv.fields["WEATHER (2)"] == "Light rain"
    assert len([k for k in inv.fields if "weather" in k.lower()]) == 2


def test_a_numbered_label_counts_as_its_label():
    answer = _log_answer(header_fields=[
        {"name": "Weather", "value": "Overcast"},
        {"name": "Weather", "value": "Light rain"},
        {"name": "Weather", "value": "Light rain"},
        {"name": "Weather (2)", "value": "Overcast"},
        {"name": "Weather (2)", "value": "Fog"}])
    inv, _ = mf.apply_log_additions(*_built(answer))
    weather = {k: v for k, v in inv.fields.items() if "weather" in k.lower()}
    assert weather == {"Weather": "Overcast", "Weather (2)": "Light rain",
                       "Weather (3)": "Fog"}


def test_a_recovery_length_goes_into_the_sample_s_recovery():
    from report_ingest.log_scoring import _recovery_numbers

    reading, built = _built(_log_answer())
    inv, _ = mf.apply_log_additions(reading, built)
    first, second = inv.samples
    assert (first.recovery.value, first.recovery.unit) == (14.0, "in")
    assert first.recovery_percent is None
    assert second.recovery is None and second.recovery_percent == 80.0
    assert built.samples[0].recovery is None    # the input is not changed
    assert 14.0 in _recovery_numbers(first, "recovery_percent")


def test_a_recovery_length_with_no_unit_keeps_the_unit_unknown():
    answer = _log_answer(samples=[_sample("S-1", 2.5, recovery_length=0.3)])
    inv, _ = mf.apply_log_additions(*_built(answer))
    assert (inv.samples[0].recovery.value, inv.samples[0].recovery.unit) == \
        (0.3, "")


def test_a_negative_recovery_length_stays_out_and_is_listed():
    answer = _log_answer(samples=[_sample("S-1", 2.5, recovery_length=-6.0,
                                          recovery_length_unit="in")])
    inv, unresolved = mf.apply_log_additions(*_built(answer))
    assert inv.samples[0].recovery is None
    assert unresolved[0]["value"] == -6.0
    assert unresolved[0]["refused_by"] == "python"


def test_a_sample_the_builder_refused_does_not_shift_the_others():
    answer = _log_answer(samples=[
        _sample("S-1", 2.5, recovery_length=10.0, recovery_length_unit="in"),
        _sample("S-2", 45.0, recovery_length=11.0, recovery_length_unit="in"),
        _sample("S-3", 7.5, recovery_length=12.0, recovery_length_unit="in")])
    reading, built = _built(answer, window=(0.0, 30.0))
    assert [s.sample_id for s in built.samples] == ["S-1", "S-3"]
    inv, unresolved = mf.apply_log_additions(reading, built)
    assert [s.recovery.value for s in inv.samples] == [10.0, 12.0]
    # The refused sample's recovery is listed, not lost without a word.
    assert [(u["what"], u["value"], u["why"]) for u in unresolved] == [
        ("sample S-2 recovery", 11.0, "its sample is not in the record")]


def test_a_sample_is_found_wherever_it_sits_in_the_record():
    answer = _log_answer(samples=[
        _sample("S-1", 2.5, recovery_length=10.0, recovery_length_unit="in"),
        _sample("S-2", 5.0, recovery_length=11.0, recovery_length_unit="in")])
    reading, built = _built(answer)
    built.samples.reverse()
    inv, unresolved = mf.apply_log_additions(reading, built)
    assert {s.sample_id: s.recovery.value for s in inv.samples} == \
        {"S-1": 10.0, "S-2": 11.0}
    assert unresolved == []


def test_a_recovery_length_in_percent_is_a_percent_recovery():
    answer = _log_answer(samples=[
        _sample("S-1", 2.5, recovery_length=64.0, recovery_length_unit="%"),
        _sample("S-2", 5.0, recovery_length=70.0, recovery_length_unit="%",
                recovery_percent=70.0),
        _sample("S-3", 7.5, recovery_length=55.0, recovery_length_unit="%",
                recovery_percent=60.0),
        _sample("S-4", 10.0, recovery_length=120.0,
                recovery_length_unit="percent")])
    inv, unresolved = mf.apply_log_additions(*_built(answer))
    assert [s.recovery_percent for s in inv.samples] == [64.0, 70.0, 60.0,
                                                         None]
    assert all(s.recovery is None for s in inv.samples)
    assert [(u["what"], u["value"]) for u in unresolved] == [
        ("sample S-3 recovery", 55.0), ("sample S-4 recovery", 120.0)]


def _lab_answer(rows):
    return {"depth_unit": "ft", "tests": [{
        "kind": "summary_table", "description": "SUMMARY OF LABORATORY TESTS",
        "rows": rows, "prov": _prov(9)}]}


def _row(hole, depth, **extra):
    return {"investigation_id": hole, "sample_id": "S-1", "depth_top": depth,
            "wc": 21.0, **extra}


def _lab_built(answer):
    reading = mf.LabSheetReading.model_validate(answer)
    return reading, lab_reader._Builder(reading, [9], "R00", []).build()


def test_a_row_s_bottom_elevation_goes_into_its_other_columns():
    answer = _lab_answer([
        _row("B-7", 5.0,
             elevation_top={"value": 104.0, "unit": "ft"},
             elevation_bottom={"value": 102.5, "unit": "ft"}),
        _row("B-8", 10.0)])
    reading, tests = _lab_built(answer)
    out, unresolved = mf.apply_lab_additions(reading, tests)
    rows = out[0].result.rows
    assert rows[0].other == [("elevation bottom (ft)", "102.5")]
    assert rows[1].other == []
    assert unresolved == [] and tests[0].result.rows[0].other == []


def test_the_lab_scorer_finds_a_bottom_elevation_there():
    from report_ingest.lab_scoring import score_record

    answer = _lab_answer([_row("B-7", 5.0,
                               elevation_bottom={"value": 102.5,
                                                 "unit": "ft"})])
    truth = {"id": "summary_table__R00_p9", "kind": "summary_table",
             "depth_unit": "ft",
             "rows": [{"investigation_id": "B-7", "depth_top": 5.0,
                       "wc_pct": 21.0, "elev_bottom": 102.5}]}
    reading, tests = _lab_built(answer)
    before = score_record(truth, tests).scores["index"]
    after = score_record(truth, mf.apply_lab_additions(reading, tests)[0]
                         ).scores["index"]
    assert (before.found, after.found) == (1, 2)
    assert after.total == 2


def test_two_rows_of_one_sample_out_of_order_do_not_swap_values():
    answer = _lab_answer([
        _row("B-7", 5.0, elevation_bottom={"value": 102.5, "unit": "ft"}),
        _row("B-7", 10.0, elevation_bottom={"value": 97.5, "unit": "ft"})])
    reading, tests = _lab_built(answer)
    tests[0].result.rows.reverse()
    out, unresolved = mf.apply_lab_additions(reading, tests)
    assert all(row.other == [] for row in out[0].result.rows)
    assert [u["value"] for u in unresolved] == [102.5, 97.5]


def test_a_bottom_elevation_the_row_already_prints_is_not_added_twice():
    answer = _lab_answer([_row(
        "B-7", 5.0, elevation_bottom={"value": 102.5, "unit": "ft"},
        other=[{"name": "Elev. bottom (ft)", "value": "102.5"}])])
    out, unresolved = mf.apply_lab_additions(*_lab_built(answer))
    assert out[0].result.rows[0].other == [("Elev. bottom (ft)", "102.5")]
    assert unresolved == []


def test_a_bottom_elevation_with_no_row_to_go_in_is_listed():
    answer = _lab_answer([_row("B-7", 5.0,
                               elevation_bottom={"value": 102.5,
                                                 "unit": "ft"})])
    reading, _tests = _lab_built(answer)
    out, unresolved = mf.apply_lab_additions(reading, [])
    assert out == []
    assert unresolved[0]["value"] == 102.5
    assert "could not be found" in unresolved[0]["why"]

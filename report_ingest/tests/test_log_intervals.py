"""Every drawn line an interval, written changes as notes, stratum labels as
printed (BACKLOG RI-103 parts 1 and 2; :mod:`report_ingest.log_intervals`).

All of it sits behind ``GEOTECH_INGEST_LOG_INTERVALS``, off by default, on
the model-first log path. These tests check the switch (off: the instruction
and the answer schema are today's to the byte; on: a version suffix of its
own), the record (no interval merged, notes on their interval, the line kind
and the stratum label as printed, a label carried down marked so), the
consumers with a switched-on record, and the scorer's interval groups.

Every log, hole name, depth, description and label here is made up.
"""

from __future__ import annotations

import json

import pytest

from report_ingest import log_intervals as li
from report_ingest import model_first as mf
from report_ingest.log_intervals import use_log_intervals
from report_ingest.model import (
    Investigation, Layer, LayerNote, Provenance, Quantity, ReportRecord,
)

pymupdf = pytest.importorskip("pymupdf", reason="the log pages are drawn")


def _ft(value: float, page: int = 1) -> Quantity:
    return Quantity(value=value, unit="ft",
                    prov=Provenance(page=page, method="model"))


# ---------------------------------------------------------------------------
# a made-up log, drawn: a depth scale, a soil column cut by lines (one of
# them dashed), a stratum column and a legend that defines the dashed line
# ---------------------------------------------------------------------------

#: (top ft, base ft, description, symbol, how its top line is drawn)
DRAWN = (
    (0.0, 4.0, "MADE-UP FILL, brown sandy clay", "CL", "solid"),
    (4.0, 9.0, "MADE-UP lean CLAY, gray, stiff", "CL", "solid"),
    (9.0, 14.0, "MADE-UP lean CLAY, gray, stiff", "CL", "dashed"),
    (14.0, 20.0, "MADE-UP silty SAND, dense", "SM", "solid"),
)


def _draw_log() -> bytes:
    made = pymupdf.open()
    made.new_page().insert_text((72, 72), "MADE-UP COVER PAGE")
    page = made.new_page()
    page.insert_text((72, 60), "LOG OF BORING QX-7", fontsize=12)
    page.insert_text((72, 78), "DEPTH (ft)", fontsize=8)
    page.insert_text((150, 78), "STRATUM", fontsize=8)
    page.insert_text((230, 78), "DESCRIPTION", fontsize=8)
    y0, per_ft = 100.0, 25.0
    for foot in range(0, 21, 2):
        page.insert_text((80, y0 + foot * per_ft + 3), str(foot), fontsize=8)
    for top, _base, words, symbol, drawn in DRAWN:
        y = y0 + top * per_ft
        page.draw_line((140, y), (540, y),
                       dashes="[4 3] 0" if drawn == "dashed" else None)
        page.insert_text((230, y + 12), f"{words} ({symbol})", fontsize=8)
    page.insert_text((150, y0 + 12), "FILL", fontsize=8)
    page.insert_text((230, y0 + 6.0 * per_ft), "wet below 7 ft", fontsize=8)
    page.draw_line((140, y0 + 20 * per_ft), (540, y0 + 20 * per_ft))
    page.insert_text((72, 640), "LEGEND: a dashed line is an approximate "
                                "boundary; the transition may be gradual.",
                     fontsize=7)
    made.new_page().insert_text((72, 72), "MADE-UP PAGE AFTER THE LOG")
    data = made.tobytes()
    made.close()
    return data


def _layer(top, bottom, description, uscs="", **more):
    return {"top": top, "bottom": bottom, "description": description,
            "uscs": uscs, "prov": {"page": 1}, **more}


def _answer(layers, **over):
    data = {"investigation_id": "QX-7", "kind": "boring", "depth_unit": "ft",
            "elevation": 212.0, "elevation_unit": "ft", "total_depth": 20.0,
            "layers": layers}
    data.update(over)
    return li.LogReading.model_validate(data)


def _read(answer, on: bool = True):
    """The model-first log path on the drawn log, the model's answer given."""
    from planlens.document import open_document
    from report_ingest import model_first_read as mfr
    from report_ingest.model_first_pages import PageGroup
    from report_ingest.tests.fake_engine import FakeEngine

    doc = open_document(_draw_log(), name="made_up")
    engine = FakeEngine([{"final": answer}])
    engine.accepts_pdf = True
    try:
        with use_log_intervals(on):
            got = mfr.read_logs(doc, [PageGroup("logs", [1], key="log-1")],
                                engine, report_id="R00", route="auto")
    finally:
        doc.close()
    return got[0], engine


def _record_of(blob) -> Investigation:
    return Investigation.model_validate(blob["investigation"])


def _line_at_every_change():
    """The drawn log as an answer: a line at every change, two adjacent
    intervals alike in words and symbol, a written change with a depth and
    one without, a tag, and the legend's word for the dashed line."""
    return _answer([
        _layer(0.0, 4.0, DRAWN[0][2], "CL", top_elevation=212.0,
               stratum_label="FILL", stratum_label_source="STRATUM"),
        _layer(4.0, 9.0, DRAWN[1][2], "CL", top_elevation=208.0,
               top_line="solid", top_line_full_width=True,
               notes=[{"depth": 7.0, "text": "wet below 7 ft"},
                      {"text": "as above, but brown"}]),
        _layer(9.0, 14.0, DRAWN[2][2], "CL", top_line="approximate",
               top_line_full_width=False),
        _layer(14.0, 20.0, DRAWN[3][2], "SM", top_line="solid",
               top_line_full_width=True),
    ])


# ---------------------------------------------------------------------------
# the switch: the three tests of the round's common list, and the version
# ---------------------------------------------------------------------------

class TestTheSwitch:
    def test_it_is_off_by_default_and_reads_the_environment(
            self, monkeypatch):
        monkeypatch.delenv(li.LOG_INTERVALS_ENV, raising=False)
        assert li.LOG_INTERVALS_ENV == "GEOTECH_INGEST_LOG_INTERVALS"
        assert li.log_intervals_enabled() is False
        monkeypatch.setenv(li.LOG_INTERVALS_ENV, "1")
        assert li.log_intervals_enabled() is True
        with use_log_intervals(False):
            assert li.log_intervals_enabled() is False

    def test_off_the_instruction_and_schema_are_today_s_to_the_byte(
            self, monkeypatch):
        monkeypatch.delenv(li.LOG_INTERVALS_ENV, raising=False)
        said = mf.instruction("logs", "R00", [3])
        assert said == mf.INSTRUCTIONS["logs"].format(
            where="report R00, page 3",
            pages_note=mf.page_map_sentence([3], route="pdf", whole=False))
        assert li.MODEL_FIRST_PARAGRAPH not in said
        assert mf.answer_schema("logs") is mf.LogReading
        assert not set(li.ADDED_LAYER_FIELDS) & set(
            mf.LogReading.model_json_schema()["$defs"]["ReadLayer"]
            ["properties"])
        assert mf.prompt_version("logs") == mf.READ_PROMPT_VERSION

    def test_on_the_instruction_gains_one_paragraph_and_nothing_else(self):
        off = mf.instruction("logs", "R00", [3])
        with use_log_intervals():
            on = mf.instruction("logs", "R00", [3])
            plain = mf.instruction("logs", "R00", [3],
                                   texts=mf.PLAIN_INSTRUCTIONS)
            others = {stage: mf.instruction(stage, "R00", [3])
                      for stage in ("lab", "calc")}
        assert on == off.replace("say which unit. ", "say which unit. "
                                 + li.MODEL_FIRST_PARAGRAPH + " ", 1)
        assert on != off
        assert li.MODEL_FIRST_PARAGRAPH not in plain  # the measured baseline
        assert others == {stage: mf.instruction(stage, "R00", [3])
                          for stage in ("lab", "calc")}

    def test_the_paragraph_says_what_a_line_and_a_written_change_are(self):
        said = " ".join(li.MODEL_FIRST_PARAGRAPH.split())
        for words in ("between two lines the log draws",
                      "one layer for every interval, from one drawn line "
                      "to the next,",
                      "never join two intervals",
                      "a written change is never a layer",
                      "a drawn line is never a note",
                      "never from the look of the line alone",
                      "only where the log prints one", "marked carried",
                      "only on the interval it is printed against"):
            assert words in said, words

    def test_on_the_schema_is_today_s_plus_the_interval_fields(self):
        with use_log_intervals():
            schema = mf.answer_schema("logs")
            other = {s: mf.answer_schema(s) for s in ("lab", "calc")}
        assert schema is li.LogReading
        assert schema.__name__ == "LogReading"       # the answer tool's name
        assert other == {s: mf.answer_schema(s) for s in ("lab", "calc")}
        new = schema.model_json_schema()
        old = mf.LogReading.model_json_schema()
        assert new["title"] == old["title"] == "LogReading"
        layer = new["$defs"]["ReadLayer"]
        assert list(layer["properties"]) == (
            list(old["$defs"]["ReadLayer"]["properties"])
            + list(li.ADDED_LAYER_FIELDS))           # the new fields last
        assert layer["required"] == old["$defs"]["ReadLayer"]["required"]
        # Everything but the layer, its note and the layers list is today's.
        for name, was in old["$defs"].items():
            if name != "ReadLayer":
                assert new["$defs"][name] == was, name
        assert set(new["$defs"]) - set(old["$defs"]) == {"ReadIntervalNote"}
        for name, was in old["properties"].items():
            if name != "layers":
                assert new["properties"][name] == was, name
        for name in old["$defs"]["ReadLayer"]["properties"]:
            assert layer["properties"][name] == \
                old["$defs"]["ReadLayer"]["properties"][name]

    def test_the_schema_passes_the_strict_mode_rewrite(self):
        from report_ingest.engine import STRICT_UNSUPPORTED, strict_schema

        def keys(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    yield key
                    yield from keys(value)
            elif isinstance(node, list):
                for item in node:
                    yield from keys(item)

        strict = strict_schema(li.LogReading)
        assert not set(keys(strict)) & set(STRICT_UNSUPPORTED)

    def test_on_the_version_carries_a_suffix_of_its_own(self):
        from report_ingest.water_status import use_water_not_measured

        assert mf.prompt_version("logs") == mf.READ_PROMPT_VERSION
        with use_log_intervals():
            assert mf.prompt_version("logs") == mf.READ_PROMPT_VERSION + "+li"
            for stage in ("lab", "calc", "narrative"):
                assert mf.prompt_version(stage) == mf.READ_PROMPT_VERSION
            with use_water_not_measured():
                assert mf.prompt_version("logs") == \
                    mf.READ_PROMPT_VERSION + "+wnm+li"

    @pytest.mark.parametrize("stage", ["logs", "lab", "calc", "narrative"])
    def test_a_saved_answer_resumes_only_under_its_own_switch(self, stage):
        from report_ingest.graph import _still_current
        from report_ingest.model_first_pages import PageGroup

        parts = [PageGroup(stage, [7])]
        off = {"pages": [7], "route": "pdf",
               "prompt_version": mf.prompt_version(stage)}
        assert _still_current(off, parts, "pdf", stage)
        with use_log_intervals():
            on = {**off, "prompt_version": mf.prompt_version(stage)}
            assert _still_current(on, parts, "pdf", stage)
            assert _still_current(off, parts, "pdf", stage) is (
                stage != "logs")
        assert _still_current(on, parts, "pdf", stage) is (stage != "logs")

    def test_the_call_offers_the_interval_schema_and_saves_its_version(self):
        blob, engine = _read(_line_at_every_change())
        assert blob["prompt_version"] == mf.READ_PROMPT_VERSION + "+li"
        said = str(engine.calls[0]["messages"])
        assert "INTERVAL" in said and "marked carried" in said

    def test_off_the_same_page_is_read_as_today(self):
        answer = mf.LogReading.model_validate({
            "investigation_id": "QX-7", "depth_unit": "ft",
            "total_depth": 20.0,
            "layers": [_layer(0.0, 4.0, DRAWN[0][2], "CL")]})
        blob, engine = _read(answer, on=False)
        assert blob["prompt_version"] == mf.READ_PROMPT_VERSION
        assert "INTERVAL" not in str(engine.calls[0]["messages"])
        assert not set(li.ADDED_LAYER_FIELDS) & set(
            blob["investigation"]["layers"][0])

    def test_the_readme_lists_the_switch(self):
        from pathlib import Path

        import report_ingest

        readme = (Path(report_ingest.__file__).parent / "README.md").read_text(
            encoding="utf-8")
        assert li.LOG_INTERVALS_ENV in readme


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

class TestIntervals:
    def test_every_interval_is_kept_and_none_merged(self):
        """A line at every change of description: four intervals in, four
        out, in order -- the two adjacent ones alike in words and symbol
        among them."""
        blob, _engine = _read(_line_at_every_change())
        record = _record_of(blob)
        assert [(ly.top.value, ly.bottom.value) for ly in record.layers] == [
            (0.0, 4.0), (4.0, 9.0), (9.0, 14.0), (14.0, 20.0)]
        assert record.layers[1].description == record.layers[2].description
        assert record.layers[1].uscs == record.layers[2].uscs == "CL"
        assert [ly.description for ly in record.layers] == [
            d[2] for d in DRAWN]

    def test_an_interval_keeps_the_elevation_the_log_prints(self):
        record = _record_of(_read(_line_at_every_change())[0])
        first, second, third = record.layers[:3]
        assert (first.top_elevation.value, first.top_elevation.unit) == (
            212.0, "ft")
        assert second.top_elevation.value == 208.0
        assert third.top_elevation is None          # not printed: not made

    def test_the_reconciler_keeps_like_intervals_apart(self):
        from report_ingest.reconciler import reconcile

        record = _record_of(_read(_line_at_every_change())[0])
        out = reconcile(ReportRecord(investigations=[record]))
        assert [ly.top.value for ly in out.investigations[0].layers] == [
            0.0, 4.0, 9.0, 14.0]

    def test_a_log_read_in_two_calls_keeps_every_interval(self):
        from report_ingest.model_first_read import _join_logs

        with use_log_intervals():
            first, _ = li.apply_interval_additions(
                _answer([]), Investigation(
                    investigation_id="QX-7", depth_unit="ft", pages=[1],
                    layers=[Layer(top=_ft(0), bottom=_ft(4),
                                  description="MADE-UP CLAY", uscs="CL")]))
        later = Investigation(
            investigation_id="QX-7", depth_unit="ft", pages=[2],
            layers=[Layer(top=_ft(4, 2), bottom=_ft(9, 2),
                          description="MADE-UP CLAY", uscs="CL",
                          notes=[LayerNote(text="as above, but brown")])])
        joined, _rows = _join_logs([first, later])
        assert [(ly.top.value, len(ly.notes)) for ly in joined.layers] == [
            (0.0, 0), (4.0, 1)]


class TestNotes:
    def test_a_written_change_is_a_note_on_its_interval(self):
        record = _record_of(_read(_line_at_every_change())[0])
        assert len(record.layers) == 4              # not a fifth interval
        with_depth, without = record.layers[1].notes
        assert with_depth.text == "wet below 7 ft"
        assert (with_depth.depth.value, with_depth.depth.unit) == (7.0, "ft")
        assert with_depth.depth.prov.page == 1      # a Quantity with its page
        assert with_depth.prov.page == 1
        assert (without.text, without.depth) == ("as above, but brown", None)
        assert [ly.notes for ly in record.layers[2:]] == [[], []]

    def test_a_note_s_depth_is_in_the_log_s_own_unit(self):
        answer = _answer([_layer(0.0, 3.0, "MADE-UP SAND",
                                 notes=[{"depth": 1.5,
                                         "text": "becoming gray at 1.5 m"}])],
                         depth_unit="m", total_depth=3.0)
        record = _record_of(_read(answer)[0])
        assert record.layers[0].notes[0].depth.unit == "m"

    def test_a_negative_note_depth_is_left_off_and_listed(self):
        answer = _answer([_layer(0.0, 4.0, "MADE-UP SAND",
                                 notes=[{"depth": -2.0, "text": "wet"}])])
        blob, _ = _read(answer)
        note = _record_of(blob).layers[0].notes[0]
        assert (note.text, note.depth) == ("wet", None)
        listed = [r for r in blob["unresolved"]
                  if "note 'wet' depth" in r.get("what", "")]
        assert listed and listed[0]["value"] == -2.0

    def test_a_note_outside_its_interval_is_kept_and_flagged(self):
        answer = _answer([_layer(0.0, 4.0, "MADE-UP SAND",
                                 notes=[{"depth": 7.0,
                                         "text": "wet below 7 ft"}])])
        blob, _ = _read(answer)
        note = _record_of(blob).layers[0].notes[0]
        assert note.depth.value == 7.0
        assert any(r.get("flag") == "interval_note_depth"
                   for r in blob["unresolved"])

    def test_a_note_with_no_words_is_left_out_and_listed(self):
        answer = _answer([_layer(0.0, 4.0, "MADE-UP SAND",
                                 notes=[{"depth": 2.0, "text": "  "}])])
        blob, _ = _read(answer)
        assert _record_of(blob).layers[0].notes == []
        assert any("no words" in r.get("why", "") for r in blob["unresolved"])


class TestLines:
    def test_the_legend_s_word_for_a_dashed_line_is_kept(self):
        record = _record_of(_read(_line_at_every_change())[0])
        assert [(ly.top_line, ly.top_line_full_width)
                for ly in record.layers] == [
            ("", None), ("solid", True), ("approximate", False),
            ("solid", True)]

    def test_a_log_that_says_nothing_of_its_lines_leaves_them_empty(self):
        answer = _answer([_layer(0.0, 4.0, "MADE-UP SAND"),
                          _layer(4.0, 9.0, "MADE-UP CLAY")])
        saved = _read(answer)[0]["investigation"]["layers"]
        for layer in saved:
            assert "top_line" not in layer
            assert "top_line_full_width" not in layer


class TestStratumLabels:
    def test_a_stratum_column_s_label_is_kept_with_its_source(self):
        answer = _answer([
            _layer(0.0, 4.0, "MADE-UP SAND", stratum_label=" Unit  A1 ",
                   stratum_label_source="STRATUM"),
            _layer(4.0, 9.0, "MADE-UP TILL, gray", stratum_label="TILL",
                   stratum_label_source="in the description"),
            _layer(9.0, 14.0, "MADE-UP CLAY")])
        record = _record_of(_read(answer)[0])
        assert [(ly.stratum_label, ly.stratum_label_source,
                 ly.stratum_label_carried) for ly in record.layers] == [
            ("Unit A1", "STRATUM", False),
            ("TILL", "in the description", False), ("", "", False)]

    def test_a_label_carried_over_three_intervals_is_marked_carried(self):
        answer = _answer([
            _layer(0.0, 4.0, "MADE-UP SAND, brown", stratum_label="2b",
                   stratum_label_source="MODEL LAYER"),
            _layer(4.0, 9.0, "MADE-UP SAND, gray", stratum_label="2b",
                   stratum_label_source="MODEL LAYER",
                   stratum_label_carried=True),
            _layer(9.0, 14.0, "MADE-UP SAND, wet", stratum_label="2b",
                   stratum_label_source="MODEL LAYER",
                   stratum_label_carried=True),
            _layer(14.0, 20.0, "MADE-UP CLAY")])
        blob, _ = _read(answer)
        record = _record_of(blob)
        assert [(ly.stratum_label, ly.stratum_label_carried)
                for ly in record.layers] == [
            ("2b", False), ("2b", True), ("2b", True), ("", False)]
        assert not any(r.get("flag") == "label_carried"
                       for r in blob["unresolved"])

    def test_a_label_whose_extent_is_not_plain_stays_on_its_interval(self):
        """The reading gives it only to the interval it is printed against;
        the record gives it to no other."""
        answer = _answer([
            _layer(0.0, 4.0, "MADE-UP SAND", stratum_label="Qal",
                   stratum_label_source="GEOLOGIC UNIT"),
            _layer(4.0, 9.0, "MADE-UP SAND, gray"),
            _layer(9.0, 14.0, "MADE-UP CLAY")])
        record = _record_of(_read(answer)[0])
        assert [ly.stratum_label for ly in record.layers] == ["Qal", "", ""]

    def test_a_carried_mark_with_no_label_above_is_listed(self):
        answer = _answer([
            _layer(0.0, 4.0, "MADE-UP SAND"),
            _layer(4.0, 9.0, "MADE-UP CLAY", stratum_label="Qal",
                   stratum_label_carried=True)])
        blob, _ = _read(answer)
        assert _record_of(blob).layers[1].stratum_label_carried is True
        assert [r["read"] for r in blob["unresolved"]
                if r.get("flag") == "label_carried"] == ["Qal"]

    def test_a_source_or_a_carried_mark_with_no_label_is_dropped(self):
        answer = _answer([_layer(0.0, 4.0, "MADE-UP SAND",
                                 stratum_label_source="STRATUM",
                                 stratum_label_carried=True)])
        saved = _read(answer)[0]["investigation"]["layers"][0]
        assert not {"stratum_label", "stratum_label_source",
                    "stratum_label_carried"} & set(saved)


class TestTheBuilderItReuses:
    def test_a_layer_the_builder_keeps_out_does_not_shift_the_others(self):
        """A layer with an impossible top is kept out whole by today's
        builder (listed, as read); the intervals after it keep their own
        notes and labels."""
        answer = _answer([
            _layer(0.0, 4.0, "MADE-UP SAND", stratum_label="A"),
            _layer(-30.0, 9.0, "MADE-UP MISREAD", stratum_label="B"),
            _layer(9.0, 14.0, "MADE-UP CLAY", stratum_label="C",
                   notes=[{"depth": 10.0, "text": "wet below 10 ft"}])])
        blob, _ = _read(answer)
        record = _record_of(blob)
        assert [(ly.description, ly.stratum_label, len(ly.notes))
                for ly in record.layers] == [
            ("MADE-UP SAND", "A", 0), ("MADE-UP CLAY", "C", 1)]
        assert any("MADE-UP MISREAD" in json.dumps(r)
                   for r in blob["unresolved"])


class TestTheSavedRecord:
    def test_a_layer_with_none_of_it_is_written_as_before(self):
        layer = Layer(top=_ft(0), bottom=_ft(4), description="MADE-UP SAND")
        assert list(layer.model_dump(mode="json")) == [
            "top", "bottom", "description", "uscs", "consistency", "color",
            "moisture", "prov"]

    def test_a_layer_with_all_of_it_loads_back_the_same(self):
        layer = Layer(
            top=_ft(4), bottom=_ft(9), description="MADE-UP CLAY",
            top_elevation=Quantity(value=208.0, unit="ft"),
            notes=[LayerNote(depth=_ft(7), text="wet below 7 ft"),
                   LayerNote(text="as above, but brown")],
            top_line="dashed", top_line_full_width=False,
            stratum_label="2b", stratum_label_source="MODEL LAYER",
            stratum_label_carried=True)
        saved = json.loads(layer.model_dump_json())
        assert saved["top_line_full_width"] is False   # False is a reading
        assert "depth" not in saved["notes"][1]
        assert Layer.model_validate(saved) == layer

    def test_a_record_saved_before_the_fields_loads(self):
        old = {"top": {"value": 0.0, "unit": "ft", "prov": None},
               "bottom": None, "description": "MADE-UP SAND", "uscs": "",
               "consistency": "", "color": "", "moisture": "", "prov": None}
        assert Layer.model_validate(old).model_dump(mode="json") == old


class TestAContinuationSheet:
    def test_an_interval_printed_again_keeps_what_the_second_adds(self):
        from report_ingest.reconciler import _fold_into

        first = Investigation(
            investigation_id="QX-7", depth_unit="ft", pages=[1],
            layers=[Layer(top=_ft(0), bottom=_ft(4), description="MADE-UP A"),
                    Layer(top=_ft(4), description="MADE-UP CLAY, gray")])
        later = Investigation(
            investigation_id="QX-7", depth_unit="ft", pages=[2],
            layers=[Layer(top=_ft(4, 2), bottom=_ft(9, 2),
                          description="MADE-UP CLAY, gray",
                          stratum_label="2b",
                          notes=[LayerNote(depth=_ft(7, 2),
                                           text="wet below 7 ft")]),
                    Layer(top=_ft(9, 2), bottom=_ft(14, 2),
                          description="MADE-UP CLAY, gray")])
        _fold_into(first, later)
        assert [ly.top.value for ly in first.layers] == [0.0, 4.0, 9.0]
        again = first.layers[1]
        assert again.stratum_label == "2b"
        assert [n.text for n in again.notes] == ["wet below 7 ft"]
        # The next interval has the same words and is still its own.
        assert first.layers[2].description == again.description

    def test_two_printings_with_different_notes_fold_to_one_layer(self):
        """Review finding 1: each printing's own notes made the two copies
        differ, so the hole got two layers at one top."""
        from report_ingest.reconciler import _fold_into

        first = Investigation(
            investigation_id="QX-7", depth_unit="ft", pages=[1],
            layers=[Layer(top=_ft(4), bottom=_ft(9),
                          description="MADE-UP CLAY, gray",
                          notes=[LayerNote(depth=_ft(7),
                                           text="wet below 7 ft")])])
        later = Investigation(
            investigation_id="QX-7", depth_unit="ft", pages=[2],
            layers=[Layer(top=_ft(4, 2), bottom=_ft(9, 2),
                          description="MADE-UP CLAY, gray",
                          notes=[LayerNote(depth=_ft(7, 2),
                                           text="wet below 7 ft"),
                                 LayerNote(text="as above, but brown")])])
        _fold_into(first, later)
        assert [ly.top.value for ly in first.layers] == [4.0]
        assert [(n.depth.value if n.depth else None, n.text)
                for n in first.layers[0].notes] == [
            (7.0, "wet below 7 ft"), (None, "as above, but brown")]

    def test_a_label_printed_on_the_first_printing_keeps_its_mark(self):
        """Review finding 1: the label, its source and its carried mark go
        together, and only where the first printing has no label."""
        from report_ingest.reconciler import _fold_into

        first = Investigation(
            investigation_id="QX-7", depth_unit="ft", pages=[1],
            layers=[Layer(top=_ft(4), bottom=_ft(9),
                          description="MADE-UP CLAY, gray",
                          stratum_label="Unit 2",
                          stratum_label_source="STRATUM")])
        later = Investigation(
            investigation_id="QX-7", depth_unit="ft", pages=[2],
            layers=[Layer(top=_ft(4, 2), bottom=_ft(9, 2),
                          description="MADE-UP CLAY, gray",
                          stratum_label="Unit 2", stratum_label_source="UNIT",
                          stratum_label_carried=True)])
        _fold_into(first, later)
        assert len(first.layers) == 1
        kept = first.layers[0]
        assert (kept.stratum_label, kept.stratum_label_source,
                kept.stratum_label_carried) == ("Unit 2", "STRATUM", False)

    def test_a_record_with_none_of_the_fields_folds_as_before(self):
        """Pinned on the base commit's reconciler (50ef609), run on the same
        two sheets: the same layers, the same samples, the same count."""
        from report_ingest.model import Sample
        from report_ingest.reconciler import _fold_into

        first = Investigation(
            investigation_id="QX-7", depth_unit="ft", pages=[1],
            layers=[Layer(top=_ft(0), bottom=_ft(4),
                          description="MADE-UP SAND", uscs="SP"),
                    Layer(top=_ft(4), description="MADE-UP CLAY, gray",
                          uscs="CL")],
            samples=[Sample(sample_id="S-2", top=_ft(5), bottom=_ft(6.5))])
        later = Investigation(
            investigation_id="QX-7", depth_unit="ft", pages=[2],
            layers=[Layer(top=_ft(4, 2), bottom=_ft(9, 2),
                          description="MADE-UP CLAY, gray", uscs="CL",
                          consistency="stiff"),
                    Layer(top=_ft(4, 2), bottom=_ft(9, 2),
                          description="MADE-UP CLAY, gray", uscs="CH"),
                    Layer(top=_ft(9, 2), bottom=_ft(14, 2),
                          description="MADE-UP CLAY, gray", uscs="CL")],
            samples=[Sample(sample_id="S-2", top=_ft(5, 2),
                            bottom=_ft(6.5, 2)),
                     Sample(sample_id="S-3", top=_ft(10, 2),
                            bottom=_ft(11.5, 2))])
        said = _fold_into(first, later)
        assert said == {"restated": 1, "total_depth": None}
        assert [(ly.top.value, ly.bottom.value if ly.bottom else None,
                 ly.description, ly.uscs, ly.consistency, ly.top.prov.page)
                for ly in first.layers] == [
            (0.0, 4.0, "MADE-UP SAND", "SP", "", 1),
            (4.0, None, "MADE-UP CLAY, gray", "CL", "", 1),
            (4.0, 9.0, "MADE-UP CLAY, gray", "CH", "", 2),
            (9.0, 14.0, "MADE-UP CLAY, gray", "CL", "", 2)]
        assert [s.sample_id for s in first.samples] == ["S-2", "S-3"]


class TestAHoleJoinedAfterReading:
    @staticmethod
    def _parts():
        def layer(top, bottom, page, **more):
            return Layer(top=_ft(top, page), bottom=_ft(bottom, page),
                         description="MADE-UP CLAY",
                         prov=Provenance(page=page, method="model"), **more)

        upper = Investigation(
            investigation_id="QX-5", depth_unit="ft", pages=[4],
            layers=[layer(0, 6, 4, stratum_label="Unit B",
                          stratum_label_source="STRATUM")])
        lower = Investigation(
            investigation_id="QX-5", depth_unit="ft", pages=[5],
            layers=[layer(6, 12, 5, stratum_label="Unit B",
                          stratum_label_source="STRATUM",
                          stratum_label_carried=True),
                    layer(12, 18, 5, stratum_label="Qal",
                          stratum_label_source="STRATUM",
                          stratum_label_carried=True)])
        return upper, lower

    def _joined_entries(self):
        from report_ingest.graph import _join_by_hole
        from report_ingest.tests.record_fixtures import build_record

        upper, lower = self._parts()
        # What the reader lists for each part, read apart.
        unresolved = (li.carried_label_entries(upper)
                      + li.carried_label_entries(lower))
        record = build_record(investigations=[upper, lower])
        _join_by_hole(record, 0, unresolved, [], None)
        assert [ly.top.value for ly in record.investigations[0].layers] == [
            0.0, 6.0, 12.0]
        return [(row["what"], row["read"]) for row in unresolved
                if row.get("flag") == "label_carried"]

    def test_a_carried_entry_the_join_makes_untrue_is_dropped(self):
        """Review finding 3: the label printed on the part above is found
        once the parts are one log; the entry with nothing above stays."""
        with use_log_intervals():
            got = self._joined_entries()
        assert got == [("interval at 12 ft: stratum label", "Qal")]

    def test_off_the_entries_are_left_as_they_were(self, monkeypatch):
        monkeypatch.delenv(li.LOG_INTERVALS_ENV, raising=False)
        assert self._joined_entries() == [
            ("interval at 6 ft: stratum label", "Unit B"),
            ("interval at 12 ft: stratum label", "Qal")]


# ---------------------------------------------------------------------------
# the consumers, with a switched-on record
# ---------------------------------------------------------------------------

@pytest.fixture()
def switched_on() -> ReportRecord:
    """A record whose one boring was read under the switch: more, shorter
    intervals, with every new field present somewhere."""
    from report_ingest.tests.record_fixtures import boring, build_record, q

    hole = boring()
    page = hole.pages[0]
    hole.layers = [
        Layer(top=q(0.0, "ft", page), bottom=q(2.0, "ft", page),
              description="Brown sandy lean CLAY, stiff, moist", uscs="CL",
              top_elevation=q(102.5, "ft", page),
              stratum_label="FILL", stratum_label_source="STRATUM"),
        Layer(top=q(2.0, "ft", page), bottom=q(5.0, "ft", page),
              description="Brown sandy lean CLAY, stiff, moist", uscs="CL",
              top_line="dashed", top_line_full_width=False,
              stratum_label="FILL", stratum_label_source="STRATUM",
              stratum_label_carried=True,
              notes=[LayerNote(depth=q(4.0, "ft", page),
                               text="becoming soft at 4 ft"),
                     LayerNote(text="as above, but mottled")]),
        Layer(top=q(5.0, "ft", page), bottom=q(8.0, "ft", page),
              description="Brown sandy lean CLAY, stiff, moist", uscs="CL"),
        Layer(top=q(8.0, "ft", page), bottom=q(30.0, "ft", page),
              description="Grey silty SAND, medium dense", uscs="SM",
              top_line="solid", top_line_full_width=True),
    ]
    return build_record(investigations=[hole])


class TestConsumers:
    def test_the_reconciler_runs_and_keeps_every_interval(self, switched_on):
        from report_ingest.reconciler import reconcile

        out = reconcile(switched_on)
        layers = out.investigations[0].layers
        assert [ly.top.value for ly in layers] == [0.0, 2.0, 5.0, 8.0]
        assert [n.text for n in layers[1].notes] == [
            "becoming soft at 4 ft", "as above, but mottled"]
        assert layers[1].stratum_label_carried is True

    def test_the_summary_and_the_library_page_are_written(
            self, switched_on, tmp_path):
        from report_ingest.reconciler import reconcile
        from report_ingest.writers import (library_page, record_key,
                                           summary_markdown, write_outputs)

        record = reconcile(switched_on)
        summary = summary_markdown(record)
        assert "B-1" in summary
        page = library_page(record, record_key(record))
        assert page.strip()
        written = write_outputs(record, tmp_path)
        saved = json.loads(next(
            tmp_path.glob("*record.json")).read_text(encoding="utf-8"))
        assert written
        layer = saved["investigations"][0]["layers"][1]
        assert layer["notes"][0]["text"] == "becoming soft at 4 ft"
        assert layer["top_line_full_width"] is False
        assert ReportRecord.model_validate(saved).investigations[0].layers \
            == record.investigations[0].layers

    def test_the_summary_is_the_same_with_and_without_the_new_fields(
            self, switched_on):
        """The summary shows none of the new fields in this branch (the
        report tables export is another branch's): taking them off the
        record changes nothing it writes."""
        from report_ingest.model import LAYER_INTERVAL_FIELDS
        from report_ingest.writers import summary_markdown

        bare = switched_on.model_copy(deep=True)
        for layer in bare.investigations[0].layers:
            for name in LAYER_INTERVAL_FIELDS:
                setattr(layer, name,
                        Layer.model_fields[name].get_default(
                            call_default_factory=True))
        assert summary_markdown(switched_on) == summary_markdown(bare)

    def test_the_diggs_file_validates_and_round_trips(self, switched_on):
        from report_ingest.diggs_writer import (diggs_roundtrip_gate,
                                                diggs_schema_gate,
                                                write_diggs)

        xml = write_diggs(switched_on)
        ok, errors = diggs_schema_gate(xml)
        assert ok, errors
        ok, diffs = diggs_roundtrip_gate(xml, switched_on)
        assert ok, diffs
        # Four intervals in, four lithology observations out; and the new
        # fields are left out of the file in this branch.
        assert xml.count("<LithologyObservation") == 4
        for words in ("becoming soft", "mottled", "dashed", "FILL"):
            assert words not in xml

    def test_the_library_s_search_words_hold_a_note_and_a_label(
            self, switched_on):
        from report_ingest.library import _chunks

        words = " ".join(str(part) for chunk in _chunks(switched_on, "")
                         for part in chunk if isinstance(part, str))
        assert "becoming soft at 4 ft" in words and "FILL" in words

    def test_the_question_list_reads_soft_in_a_note(self, switched_on):
        """Every interval of the made-up hole is described as stiff; the
        only 'soft' is the change written inside one."""
        from report_ingest import question_list as ql

        source = ql._Source(record=switched_on, bound={}, printed={})
        answer = ql._soft(source, None)
        assert answer.status == "answered" and answer.value is True
        for layer in switched_on.investigations[0].layers:
            layer.notes = []
        assert ql._soft(source, None).status == "not said"


# ---------------------------------------------------------------------------
# the scorer and the answer sheets
# ---------------------------------------------------------------------------

def _sheet(**layer_more):
    return {"id": "R00_QX-7", "depth_unit": "ft", "layers": [
        {"top": 0, "bottom": 4, "description": "MADE-UP SAND", "uscs": "SP"},
        {"top": 4, "bottom": 9, "description": "MADE-UP CLAY", "uscs": "CL",
         **layer_more}]}


def _scored_hole(**second_more) -> Investigation:
    return Investigation(
        investigation_id="QX-7", depth_unit="ft", pages=[1],
        layers=[Layer(top=_ft(0), bottom=_ft(4), description="MADE-UP SAND",
                      uscs="SP", stratum_label="Qal"),
                Layer(top=_ft(4), bottom=_ft(9), description="MADE-UP CLAY",
                      uscs="CL", **second_more)])


class TestTheScorer:
    def test_a_sheet_without_the_new_keys_scores_exactly_as_today(self):
        """Pinned: the same scores whether or not the record holds notes,
        lines and labels, no interval group, and today's metrics only."""
        from report_ingest.log_scoring import METRICS, score_record

        plain = score_record(_sheet(), [_scored_hole()])
        full = score_record(_sheet(), [_scored_hole(
            notes=[LayerNote(depth=_ft(7), text="wet below 7 ft")],
            top_line="dashed", top_line_full_width=False,
            stratum_label="Qal", stratum_label_carried=True)])
        assert plain.to_dict()["scores"] == full.to_dict()["scores"] == {
            "layer_top": {"found": 2, "total": 2, "rate": 1.0, "misses": []},
            "uscs": {"found": 2, "total": 2, "rate": 1.0, "misses": []}}
        assert set(plain.scores) <= set(METRICS)
        assert plain.total.to_dict() == full.total.to_dict()

    def test_notes_are_scored_by_depth_and_words(self):
        from report_ingest.log_scoring import score_record

        sheet = _sheet(notes=[{"depth": 7, "text": "wet below 7 ft"},
                              {"text": "as above, but brown"},
                              {"depth": 8, "text": "trace gravel"}])
        hole = _scored_hole(notes=[
            LayerNote(depth=_ft(7.5), text="Wet below 7 ft."),  # in tolerance
            LayerNote(depth=_ft(5), text="as above, but brown"),  # a depth
            LayerNote(depth=_ft(5), text="trace gravel")])        # too far
        got = score_record(sheet, [hole]).scores["interval_note"]
        assert (got.found, got.total) == (1, 3)
        hole = _scored_hole(notes=[
            LayerNote(depth=_ft(7), text="wet below 7 ft"),
            LayerNote(text="as above, but brown"),
            LayerNote(depth=_ft(8), text="trace gravel")])
        got = score_record(sheet, [hole]).scores["interval_note"]
        assert (got.found, got.total) == (3, 3)

    def test_a_note_made_into_an_interval_is_a_miss(self):
        from report_ingest.log_scoring import score_record

        sheet = _sheet(notes=[{"depth": 7, "text": "wet below 7 ft"}])
        hole = _scored_hole()
        hole.layers[1].bottom = _ft(7)
        hole.layers.append(Layer(top=_ft(7), bottom=_ft(9),
                                 description="wet below 7 ft"))
        got = score_record(sheet, [hole]).scores["interval_note"]
        assert (got.found, got.total) == (0, 1)

    def test_a_note_s_words_are_equal_not_one_inside_the_other(self):
        """Review finding 4: equal after case, spacing and punctuation."""
        from report_ingest.log_scoring import score_record

        sheet = _sheet(notes=[{"text": "wet below 7 ft"}])
        short = score_record(sheet, [_scored_hole(
            notes=[LayerNote(text="wet")])]).scores["interval_note"]
        assert (short.found, short.total) == (0, 1)
        same = score_record(sheet, [_scored_hole(
            notes=[LayerNote(text="Wet below 7 ft.")])]).scores[
            "interval_note"]
        assert (same.found, same.total) == (1, 1)

    def test_a_label_is_exact_after_whitespace_and_case(self):
        from report_ingest.log_scoring import score_record

        def label(sheet_label, record_label):
            got = score_record(_sheet(label=sheet_label),
                               [_scored_hole(stratum_label=record_label)])
            return got.scores["stratum_label"].found

        assert label("Unit 2b", " unit  2B ") == 1
        assert label("Unit 2b", "Unit 2") == 0      # no partial credit
        assert label("", "") == 1                   # the log prints none
        assert label("", "Qal") == 0                # one made up

    def test_a_carried_mark_the_other_way_misses_on_the_mark_only(self):
        from report_ingest.log_scoring import score_record

        sheet = _sheet(label="Qal", label_source="GEOLOGIC UNIT",
                       label_carried=True)
        printed = score_record(sheet, [_scored_hole(
            stratum_label="Qal", stratum_label_source="Geologic Unit")])
        assert printed.scores["stratum_label"].found == 1
        assert printed.scores["label_source"].found == 1
        assert (printed.scores["label_carried"].found,
                printed.scores["label_carried"].total) == (0, 1)
        carried = score_record(sheet, [_scored_hole(
            stratum_label="Qal", stratum_label_carried=True)])
        assert carried.scores["label_carried"].found == 1
        # The reverse: the sheet says printed, the record carries it down.
        reverse = score_record(_sheet(label="Qal"), [_scored_hole(
            stratum_label="Qal", stratum_label_carried=True)])
        assert reverse.scores["stratum_label"].found == 1
        assert reverse.scores["label_carried"].found == 0

    def test_a_line_kind_is_scored_and_a_guess_is_a_miss(self):
        from report_ingest.log_scoring import score_record

        sheet = _sheet(line="dashed", line_full_width=False)
        got = score_record(sheet, [_scored_hole(
            top_line="Dashed", top_line_full_width=False)])
        assert got.scores["line_kind"].found == 1
        assert got.scores["line_width"].found == 1
        # The sheet says the log says nothing of its lines.
        got = score_record(_sheet(line=""), [_scored_hole(top_line="solid")])
        assert got.scores["line_kind"].found == 0

    def test_the_groups_are_their_own_and_stay_out_of_the_total(self):
        from report_ingest.log_scoring import (INTERVAL_METRICS, METRICS,
                                               score_record)

        sheet = _sheet(notes=[{"depth": 7, "text": "wet below 7 ft"}],
                       line="dashed", line_full_width=True, label="Qal",
                       label_source="UNIT", label_carried=True)
        got = score_record(sheet, [_scored_hole()])
        assert set(got.scores) - set(METRICS) == set(INTERVAL_METRICS) - {
            "label_source", "label_carried"}        # the label was missed
        assert not set(INTERVAL_METRICS) & set(METRICS)
        assert got.total.to_dict() == score_record(
            _sheet(), [_scored_hole()]).total.to_dict()

    def test_a_saved_run_is_rescored_with_the_groups(self):
        from report_ingest.log_scoring import (dump_investigations,
                                               rescore_saved)

        hole = _scored_hole(stratum_label="Qal")
        after = {"record": dump_investigations([hole])}
        got = rescore_saved(_sheet(label="Qal"), after)
        assert got["scores"]["stratum_label"]["found"] == 1
        assert got["overall"] == {"found": 4, "total": 4, "rate": 1.0,
                                  "misses": []}

    def test_the_results_table_keeps_overall_and_lists_the_groups(self):
        from report_ingest.cluster_scoring import _render_logs, _score_logs
        from report_ingest.log_scoring import score_record

        def row(sheet):
            after = score_record(sheet, [_scored_hole(stratum_label="Qal")])
            return {"log_id": sheet["id"], "set": "open",
                    "before": after.to_dict(), "after": after.to_dict(),
                    "cost": {}, "seconds": 0.0}

        def rendered(sheet):
            logs = _score_logs({"a": row(sheet)}, {}, "made-up-model",
                               ["R00"])
            return "\n".join(_render_logs(logs))

        plain, grouped = rendered(_sheet()), rendered(_sheet(label="Qal"))
        overall = [line for line in plain.splitlines()
                   if line.startswith("OVERALL")]
        assert overall and overall == [
            line for line in grouped.splitlines()
            if line.startswith("OVERALL")]
        assert "stratum_label" in grouped and "not in OVERALL" in grouped
        assert "stratum_label" not in plain


class TestTheAnswerSheet:
    def test_a_sheet_s_layer_becomes_the_record_s_fields(self):
        harness = pytest.importorskip(
            "module_work.report_ingest_harness.truth_records")
        layers = harness._layers([
            {"top": 0, "bottom": 4, "description": "MADE-UP SAND"},
            {"top": 4, "bottom": 9, "description": "MADE-UP CLAY",
             "notes": [{"depth": 7, "text": "wet below 7 ft"},
                       {"text": "as above, but brown"}],
             "line": "dashed", "line_full_width": False,
             "label": "2b", "label_source": "MODEL LAYER",
             "label_carried": True}], "ft")
        assert list(layers[0].model_dump(mode="json")) == [
            "top", "bottom", "description", "uscs", "consistency", "color",
            "moisture", "prov"]                      # as it always was
        second = layers[1]
        assert [(n.text, n.depth.value if n.depth else None)
                for n in second.notes] == [("wet below 7 ft", 7.0),
                                           ("as above, but brown", None)]
        assert (second.top_line, second.top_line_full_width) == (
            "dashed", False)
        assert (second.stratum_label, second.stratum_label_source,
                second.stratum_label_carried) == ("2b", "MODEL LAYER", True)

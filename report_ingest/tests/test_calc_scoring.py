"""The calculation scorer's arithmetic, on hand-made cases.

No document and no model: a truth file is a dict and a record is a
:class:`~report_ingest.model.Calculation`, so every rule the scorecard rests
on -- what counts as the same label, what counts as the same number, what a
``null`` program means, what happens to a result the reader filed as an
input -- is asserted directly.
"""

from __future__ import annotations

import pytest

from report_ingest.calc_scoring import (
    METRICS, MODEL_ONLY, NAME_RATIO, VALUE_TOL, Expected, expectations_for,
    pages_of, report_of, same_name, same_text, same_value, score_floor,
    score_record,
)
from report_ingest.model import (
    G, Calculation, NamedQuantity, Quantity, calc_si,
)

# Every id, label, number and page here is made up: none comes off a report.


def _truth(**over):
    data = {
        "id": "settlement__R91_p6",
        "kind": "settlement",
        "pages": [6, 7, 8],
        "program": None,
        "method": "Elastic settlement by sublayers",
        "subject": "Pump house mat",
        "inputs": [{"name": "Mat Width B (ft)", "value": 31.5,
                    "unit": "ft", "page": 8}],
        "results": [{"name": "Total Settlement Under Mat (in)",
                     "value": 0.63, "unit": "in", "page": 8}],
    }
    data.update(over)
    return data


def _calc(**over) -> Calculation:
    data = dict(kind="settlement", program=None,
                method="Elastic settlement by sublayers",
                subject="Pump house mat", inputs=[], results=[],
                pages=[6, 7, 8])
    data.update(over)
    return Calculation(**data)


def _named(name, value=None, unit="", text=""):
    if value is None:
        return NamedQuantity(name=name, text=text)
    return NamedQuantity(name=name, value=Quantity(value=value, unit=unit))


# ---------------------------------------------------------------------------
# reading a truth file
# ---------------------------------------------------------------------------

class TestReadingATruthFile:

    def test_the_pages_are_the_run_the_file_states(self):
        assert pages_of(_truth()) == [6, 7, 8]

    def test_a_file_that_states_no_pages_falls_back_to_its_own_id(self):
        assert pages_of({"id": "pavement__R92_p9"}) == [9]

    def test_a_first_page_and_a_count_are_a_run(self):
        assert pages_of({"id": "x", "first_page": 5, "n_pages": 3}) == \
            [5, 6, 7]

    def test_a_range_written_as_a_string_is_a_run(self):
        assert pages_of({"id": "x", "pages": "5-8"}) == [5, 6, 7, 8]

    def test_the_report_comes_off_the_file_or_off_the_id(self):
        assert report_of(_truth()) == "R91"
        assert report_of({"id": "pavement__R92_p9"}) == "R92"
        assert report_of({"id": "nothing"}) == ""

    def test_the_two_lists_come_back_typed(self):
        inputs, results = expectations_for(_truth())
        assert [e.name for e in inputs] == ["Mat Width B (ft)"]
        assert results[0].value == 0.63 and results[0].unit == "in"
        assert results[0].shown == "0.63 in"

    def test_a_value_printed_as_a_word_carries_text_and_no_number(self):
        (row,) = expectations_for(
            {"results": [{"name": "Soil Profile Class", "text": "C"}]})[1]
        assert row.value is None and row.text == "C"
        assert row.quantity is None and row.shown == "C"


# ---------------------------------------------------------------------------
# comparing
# ---------------------------------------------------------------------------

class TestComparing:

    def test_a_label_matches_fuzzily(self):
        assert same_name("Total Settlement Under Mat (in)",
                         "Total Settlement Under Mat (in), last row")
        assert same_name("Maximum bending moment", "Maximum Bending Moment")
        assert not same_name("Mat Width B (ft)", "Depth to rock")

    def test_a_one_letter_answer_must_be_the_others_tail(self):
        """A partial ratio is the wrong tool here: E scores 100 against Site
        Class C because the letter is somewhere inside it."""
        assert same_text("C", "Site Class C")
        assert same_text("Site Class C", "C")
        assert not same_text("E", "Site Class C")
        assert same_text("OK", "OK")

    def test_a_value_matches_within_two_percent(self):
        expect = Expected(name="x", value=100.0, unit="kPa")
        assert same_value(expect, _named("x", 101.5, "kPa"))
        assert not same_value(expect, _named("x", 110.0, "kPa"))

    def test_a_value_matches_within_the_last_printed_digit(self):
        """Two per cent of 0.63 is 0.013 and half the last printed digit
        0.005; the looser of the two is used, so 0.635 passes and 0.70 does
        not."""
        expect = Expected(name="x", value=0.63, unit="in")
        assert same_value(expect, _named("x", 0.635, "in"))
        assert not same_value(expect, _named("x", 0.70, "in"))

    def test_two_units_are_compared_in_si(self):
        expect = Expected(name="x", value=1000.0, unit="psf")
        assert same_value(expect, _named("x", 47.88, "kPa"))
        assert not same_value(expect, _named("x", 1000.0, "kPa"))

    def test_a_unit_neither_side_converts_is_compared_as_printed(self):
        expect = Expected(name="x", value=612.4, unit="kN-m")
        assert same_value(expect, _named("x", 612.4, "kN-m"))
        assert not same_value(expect, _named("x", 700.0, "kN-m"))

    def test_a_number_the_reader_kept_as_words_is_still_found(self):
        expect = Expected(name="x", value=0.63, unit="in")
        assert same_value(expect, _named("x", text="0.63 inches"))

    # -- RI-74 (1): the word order let go, Greek letters by name --

    def test_a_label_in_another_word_order_is_the_same_label(self):
        assert same_name("Ultimate, case 2, Clay Layer",
                         "Ultimate Bearing Capacity (kPa), Clay Layer, Case 2")
        assert same_name("Clay Layer unit weight",
                         "Unit Weight, Clay Layer (kN/m3)")

    def test_a_greek_letter_matches_its_name(self):
        assert same_name("Ngamma", "Nγ")
        assert same_name("rho", "ρ")
        assert same_name("Increment, delta P, upper wall",
                         "Upper Wall - Increment, ΔP")
        assert same_name("Phi", "φ (deg)")

    def test_word_order_does_not_join_two_different_labels(self):
        """Letting the order go is not letting the words go: a label the
        other shares one word with is still another label."""
        assert not same_name("Allowable, case 1", "Ultimate, case 1")
        assert not same_name("Nc, lower block", "Nc (Sand Fill)")
        assert not same_name("Clay Layer cohesion", "Sand Fill unit weight")

    def test_a_reordered_label_for_another_case_is_another_label(self):
        """Every word of one label among the other's, AND the same numbers
        in both: the words two labels share must not outweigh the case
        number that differs (a token-set ratio of 90 passed both)."""
        assert same_name("Allowable, case 1, Silt",
                         "Allowable Bearing Capacity (kPa), Silt, Case 1")
        assert not same_name("Allowable, case 2, Silt",
                             "Allowable Bearing Capacity (kPa), Silt, Case 1")
        assert not same_name("Width (B), case 1", "Case 2 Width (B)")
        assert not same_name("Case 2 Width (B)", "Width (B), case 1")

    # -- 2026-10-24: a number in one unit is not a number in another kind --

    def test_two_units_of_different_kinds_never_agree(self):
        """750 mm and 0.75 % are both 0.75 in SI; one is a diameter and
        the other a reinforcement ratio."""
        expect = Expected(name="reinforcement ratio", value=0.75, unit="%")
        assert not same_value(expect, _named("reinforcement ratio", 750.0,
                                             "mm"))
        expect = Expected(name="strip load", value=37.5, unit="kN/m2")
        assert not same_value(expect, _named("strip load", 37.5, "kN/m3"))

    def test_g_on_a_calculation_is_gravity(self):
        """The record's unit table reads g as grams; a seismic coefficient
        printed in g is an acceleration."""
        expect = Expected(name="kh", value=0.15, unit="g")
        assert same_value(expect, _named("kh", 0.15, "g"))
        assert same_value(expect, _named("kh", 1.471, "m/s2"))
        assert not same_value(expect, _named("kh", 0.15, "kg"))
        assert calc_si(Quantity(value=0.15, unit="g")) == pytest.approx(
            (0.15 * G, "m/s2"))


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

class TestScoringTheRecord:

    def test_a_perfect_reading_scores_everything(self):
        score = score_record(_truth(), _calc(
            inputs=[_named("Mat Width B (ft)", 31.5, "ft")],
            results=[_named("Total Settlement Under Mat (in)", 0.63,
                            "in")]))

        assert score.total.found == score.total.total == 6
        for metric in METRICS:
            assert score.scores[metric].rate == 1.0

    def test_the_wrong_kind_is_the_only_thing_that_fails(self):
        score = score_record(_truth(), _calc(
            kind="shallow_foundation_bearing",
            inputs=[_named("Mat Width B (ft)", 31.5, "ft")],
            results=[_named("Total Settlement Under Mat (in)", 0.63,
                            "in")]))

        assert score.scores["kind"].found == 0
        assert score.kind_read == "shallow_foundation_bearing"
        assert score.total.found == 5

    def test_a_null_program_means_the_reader_must_name_none(self):
        named = score_record(_truth(), _calc(program="GeoSuite 4"))
        silent = score_record(_truth(), _calc(program=None))

        assert named.scores["program"].found == 0
        assert silent.scores["program"].found == 1

    def test_a_program_matches_fuzzily(self):
        truth = _truth(program="SLOPECALC 3.21")
        assert score_record(truth, _calc(
            program="SLOPECALC 3.21")).scores["program"].found == 1
        assert score_record(truth, _calc(
            program="PaveDesign")).scores["program"].found == 0

    def test_an_alternate_subject_is_accepted(self):
        truth = _truth(_alternates={"subject": ["the north wing mat"]})
        score = score_record(truth, _calc(subject="the north wing mat"))

        assert score.scores["subject"].found == 1

    def test_a_result_filed_as_an_input_is_found_and_counted_as_misplaced(
            self):
        score = score_record(_truth(), _calc(
            inputs=[_named("Mat Width B (ft)", 31.5, "ft"),
                    _named("Total Settlement Under Mat (in)", 0.63,
                           "in")]))

        assert score.scores["results"].found == 1
        assert score.misplaced == 1
        assert "found in the other list" in "".join(
            score.scores["results"].misses) or not \
            score.scores["results"].misses

    def test_a_value_with_the_right_label_and_the_wrong_number_is_a_miss(
            self):
        score = score_record(_truth(), _calc(
            results=[_named("Total Settlement Under Mat (in)", 1.63,
                            "in")]))

        assert score.scores["results"].found == 0
        assert "Total Settlement Under Mat (in) = 0.63 in [p8]" in \
            score.scores["results"].misses

    def test_a_metric_the_truth_does_not_state_is_not_asked(self):
        score = score_record(_truth(method="", subject=""), _calc())

        assert "method" not in score.scores
        assert "subject" not in score.scores


# ---------------------------------------------------------------------------
# the floor
# ---------------------------------------------------------------------------

class TestScoringTheFloor:

    def test_the_floor_is_not_asked_the_three_it_cannot_answer(self):
        floor = Calculation(kind="other", program=None, inputs=[
            _named("Mat Width B (ft)", 31.5, "ft")])

        score = score_floor(_truth(), floor)

        for metric in MODEL_ONLY:
            assert metric not in score.scores
        assert score.scores["inputs"].found == 1
        assert score.scores["program"].found == 1

    def test_the_floor_finds_a_result_it_filed_as_an_input(self):
        """Which is the only place it CAN file one: a pattern cannot tell an
        input from a result, and the scorer says so rather than marking it
        wrong."""
        floor = Calculation(kind="other", inputs=[
            _named("Total Settlement Under Mat (in), last row", 0.63,
                   "in")])

        score = score_floor(_truth(), floor)

        assert score.scores["results"].found == 1
        assert score.misplaced == 1

    def test_a_floor_that_named_a_program_the_pages_do_not_is_wrong(self):
        floor = Calculation(kind="other", program="LPILE")

        assert score_floor(_truth(), floor).scores["program"].found == 0


# ---------------------------------------------------------------------------
# a saved run, re-scored with no model
# ---------------------------------------------------------------------------

class TestASavedRun:

    def _after(self):
        import json
        merged = _calc(inputs=[_named("Mat Width B (ft)", 31.5, "ft")],
                       results=[_named("Total Settlement Under Mat (in)",
                                       0.63, "in")])
        model = _calc(inputs=[_named("Mat Width B (ft)", 31.5, "ft")])
        floor = Calculation(kind="other", inputs=[
            _named("Total Settlement Under Mat (in)", 0.63, "in")])
        after = score_record(_truth(), merged)
        after.record = merged.model_dump(mode="json")
        after.model_record = model.model_dump(mode="json")
        after.floor_record = floor.model_dump(mode="json")
        return json.loads(json.dumps(after.to_dict()))

    def test_the_misses_and_the_records_travel_in_the_run_file(self):
        blob = self._after()
        assert blob["record"]["kind"] == "settlement"
        assert blob["model_record"]["results"] == []
        assert blob["scores"]["results"]["misses"] == []

    def test_it_rescores_with_the_current_checks_and_no_model(self):
        from report_ingest.calc_scoring import rescore_saved
        blob = self._after()
        # pretend the run was scored by an older, stricter check
        blob["scores"]["results"] = {"found": 0, "total": 1, "rate": 0.0,
                                     "misses": ["Total ... = 0.63 in"]}
        new = rescore_saved(_truth(), blob)
        assert new["scores"]["results"]["found"] == 1
        assert new["overall"]["found"] == new["overall"]["total"] == 6
        assert new["model_alone"]["scores"]["results"]["found"] == 0
        assert new["floor_alone"]["scores"]["results"]["found"] == 1
        assert new["floor_alone"]["misplaced"] == 1
        assert "kind" not in new["floor_alone"]["scores"]

    def test_an_old_run_file_without_records_is_not_rescored(self):
        from report_ingest.calc_scoring import rescore_saved
        assert rescore_saved(_truth(), {"scores": {}}) is None

    def test_the_floor_comes_back_out_of_its_record(self):
        from report_ingest.calc_reader import (
            CalcFloor, FloorValue, floor_from_record,
        )
        floor = CalcFloor(program="PROG 2.1", title="A footing", pages=[4, 5],
                          values=[
            FloorValue(name="Width (ft)", value=6.5, unit="ft", page=4,
                       bbox=(10.0, 20.0, 90.0, 30.0), method="text",
                       confidence=0.7),
            FloorValue(name="Checked", text="AB", page=5, method="tables",
                       confidence=0.85),
            FloorValue(name="Load", value=46.5, unit="kips", page=5,
                       method="quantities", confidence=0.65)])

        back = floor_from_record(floor.as_calculation("SYN"))

        assert back.program == floor.program and back.pages == [4, 5]
        assert back.values == floor.values

    def test_it_merges_again_with_the_current_merge(self):
        """A saved run whose merge let the floor's label fragment stand is
        merged again: the reader's number replaces it (RI-74 (2) (a))."""
        import json
        from report_ingest.calc_reader import CalcFloor, FloorValue
        from report_ingest.calc_scoring import rescore_saved
        truth = _truth(inputs=[{"name": "Mat Width B (ft)",
                                "value": 31.5, "unit": "ft", "page": 8}])
        floor = CalcFloor(pages=[6, 7, 8], values=[FloorValue(
            name="Mat Width B (ft)", text="B =", page=8,
            bbox=(10.0, 20.0, 90.0, 30.0), method="text", confidence=0.7)])
        model = _calc(inputs=[_named("Mat Width B (ft)", 31.5, "ft")])
        stale = _calc(inputs=[_named("Mat Width B (ft)", text="B =")])
        after = score_record(truth, stale)
        after.record = stale.model_dump(mode="json")
        after.model_record = model.model_dump(mode="json")
        after.floor_record = floor.as_calculation().model_dump(mode="json")
        blob = json.loads(json.dumps(after.to_dict()))

        assert rescore_saved(truth, blob)["scores"]["inputs"]["found"] == 0
        new = rescore_saved(truth, blob, remerge=True)
        assert new["remerged"] is True
        assert new["scores"]["inputs"]["found"] == 1
        (row,) = new["record"]["inputs"]
        assert row["value"]["value"] == 31.5
        assert new["disagreements"] == 1 and new["reconciled"] == 0

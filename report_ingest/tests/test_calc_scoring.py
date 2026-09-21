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
from report_ingest.model import Calculation, NamedQuantity, Quantity


def _truth(**over):
    data = {
        "id": "settlement__R18_p41",
        "kind": "settlement",
        "pages": [41, 42, 43],
        "program": None,
        "method": "Schmertmann strain influence",
        "subject": "NOB spread footing",
        "inputs": [{"name": "Footing Width B (ft)", "value": 25.8,
                    "unit": "ft", "page": 43}],
        "results": [{"name": "Total Cumulative Settlement (inches)",
                     "value": 0.74, "unit": "in", "page": 43}],
    }
    data.update(over)
    return data


def _calc(**over) -> Calculation:
    data = dict(kind="settlement", program=None,
                method="Schmertmann strain influence",
                subject="NOB spread footing", inputs=[], results=[],
                pages=[41, 42, 43])
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
        assert pages_of(_truth()) == [41, 42, 43]

    def test_a_file_that_states_no_pages_falls_back_to_its_own_id(self):
        assert pages_of({"id": "pavement__R29_p127"}) == [127]

    def test_a_first_page_and_a_count_are_a_run(self):
        assert pages_of({"id": "x", "first_page": 5, "n_pages": 3}) == \
            [5, 6, 7]

    def test_a_range_written_as_a_string_is_a_run(self):
        assert pages_of({"id": "x", "pages": "5-8"}) == [5, 6, 7, 8]

    def test_the_report_comes_off_the_file_or_off_the_id(self):
        assert report_of(_truth()) == "R18"
        assert report_of({"id": "pavement__R29_p127"}) == "R29"
        assert report_of({"id": "nothing"}) == ""

    def test_the_two_lists_come_back_typed(self):
        inputs, results = expectations_for(_truth())
        assert [e.name for e in inputs] == ["Footing Width B (ft)"]
        assert results[0].value == 0.74 and results[0].unit == "in"
        assert results[0].shown == "0.74 in"

    def test_a_value_printed_as_a_word_carries_text_and_no_number(self):
        (row,) = expectations_for(
            {"results": [{"name": "Seismic Site Class", "text": "C"}]})[1]
        assert row.value is None and row.text == "C"
        assert row.quantity is None and row.shown == "C"


# ---------------------------------------------------------------------------
# comparing
# ---------------------------------------------------------------------------

class TestComparing:

    def test_a_label_matches_fuzzily(self):
        assert same_name("Total Cumulative Settlement (inches)",
                         "Total Cumulative Settlement (inches), last row")
        assert same_name("Maximum bending moment", "Maximum Bending Moment")
        assert not same_name("Footing Width B (ft)", "Depth to Water Table")

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
        """Two per cent of 0.74 is 0.015, which would let 0.75 pass. The
        printed precision is the looser of the two here and it is used."""
        expect = Expected(name="x", value=0.74, unit="in")
        assert same_value(expect, _named("x", 0.745, "in"))
        assert not same_value(expect, _named("x", 0.80, "in"))

    def test_two_units_are_compared_in_si(self):
        expect = Expected(name="x", value=1000.0, unit="psf")
        assert same_value(expect, _named("x", 47.88, "kPa"))
        assert not same_value(expect, _named("x", 1000.0, "kPa"))

    def test_a_unit_neither_side_converts_is_compared_as_printed(self):
        expect = Expected(name="x", value=612.4, unit="kN-m")
        assert same_value(expect, _named("x", 612.4, "kN-m"))
        assert not same_value(expect, _named("x", 700.0, "kN-m"))

    def test_a_number_the_reader_kept_as_words_is_still_found(self):
        expect = Expected(name="x", value=0.74, unit="in")
        assert same_value(expect, _named("x", text="0.74 inches"))


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

class TestScoringTheRecord:

    def test_a_perfect_reading_scores_everything(self):
        score = score_record(_truth(), _calc(
            inputs=[_named("Footing Width B (ft)", 25.8, "ft")],
            results=[_named("Total Cumulative Settlement (inches)", 0.74,
                            "in")]))

        assert score.total.found == score.total.total == 6
        for metric in METRICS:
            assert score.scores[metric].rate == 1.0

    def test_the_wrong_kind_is_the_only_thing_that_fails(self):
        score = score_record(_truth(), _calc(
            kind="shallow_foundation_bearing",
            inputs=[_named("Footing Width B (ft)", 25.8, "ft")],
            results=[_named("Total Cumulative Settlement (inches)", 0.74,
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
        truth = _truth(program="SLIDEINTERPRET 6.039")
        assert score_record(truth, _calc(
            program="SLIDEINTERPRET 6.039")).scores["program"].found == 1
        assert score_record(truth, _calc(
            program="WinPAS")).scores["program"].found == 0

    def test_an_alternate_subject_is_accepted(self):
        truth = _truth(_alternates={"subject": ["the north wing mat"]})
        score = score_record(truth, _calc(subject="the north wing mat"))

        assert score.scores["subject"].found == 1

    def test_a_result_filed_as_an_input_is_found_and_counted_as_misplaced(
            self):
        score = score_record(_truth(), _calc(
            inputs=[_named("Footing Width B (ft)", 25.8, "ft"),
                    _named("Total Cumulative Settlement (inches)", 0.74,
                           "in")]))

        assert score.scores["results"].found == 1
        assert score.misplaced == 1
        assert "found in the other list" in "".join(
            score.scores["results"].misses) or not \
            score.scores["results"].misses

    def test_a_value_with_the_right_label_and_the_wrong_number_is_a_miss(
            self):
        score = score_record(_truth(), _calc(
            results=[_named("Total Cumulative Settlement (inches)", 1.74,
                            "in")]))

        assert score.scores["results"].found == 0
        assert "Total Cumulative Settlement (inches) = 0.74 in [p43]" in \
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
            _named("Footing Width B (ft)", 25.8, "ft")])

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
            _named("Total Cumulative Settlement (inches), last row", 0.74,
                   "in")])

        score = score_floor(_truth(), floor)

        assert score.scores["results"].found == 1
        assert score.misplaced == 1

    def test_a_floor_that_named_a_program_the_pages_do_not_is_wrong(self):
        floor = Calculation(kind="other", program="LPILE")

        assert score_floor(_truth(), floor).scores["program"].found == 0

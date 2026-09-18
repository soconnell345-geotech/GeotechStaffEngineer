"""Scoring one log, before and after, on synthetic truth and records.

No corpus, no model. What is checked is the MATCHING, which is where a
scorer goes wrong quietly: a tolerance that is really a different tolerance,
an N value credited from drives that are not at that depth, a field that
cannot be found because an empty canonical key got there first.
"""

from __future__ import annotations

import pytest

from report_ingest.log_scoring import (
    LAYER_TOL_M, METRICS, SAMPLE_TOL_M, LogScore, Score, score_record,
    truth_investigations,
)
from report_ingest.model import (
    DrillingDetails, Investigation, Layer, Quantity, SPT, Sample, WaterLevel,
)


def _ft(value: float) -> Quantity:
    return Quantity(value=value, unit="ft")


TRUTH = {
    "id": "R99_p10",
    "investigation_id": "B-1",
    "pages": [10],
    "depth_unit": "ft",
    "fields": {"boring_id": "B-1", "hammer": "Automatic SPT Hammer",
               "method": "Hollow Stem Auger", "sheet": "1 of 1",
               "total_depth": 25.0},
    "layers": [
        {"top": 0.0, "bottom": 5.0, "description": "SANDY LEAN CLAY",
         "uscs": "CL"},
        {"top": 5.0, "bottom": 25.0, "description": "SILTY SAND",
         "uscs": "SM"},
    ],
    "samples": [
        {"id": "1", "top": 2.5, "bottom": 4.0, "type": "spt",
         "blows": [5, 9, 12], "n": 21, "recovery": 67, "wc": 18,
         "duw": 112},
        {"id": "2", "top": 20.0, "bottom": 21.0, "type": "spt",
         "blows": [12, 30, '50/5"'], "n": None, "recovery": None},
    ],
    "water": [{"depth": 10.0, "when": "while drilling"}],
}


def _perfect() -> Investigation:
    """A record that says exactly what the truth says."""
    return Investigation(
        investigation_id="B-1", depth_unit="ft",
        total_depth=_ft(25.0), sheet="1 of 1",
        drilling=DrillingDetails(hammer_type="Automatic SPT Hammer",
                                 method="Hollow Stem Auger"),
        layers=[
            Layer(top=_ft(0.0), bottom=_ft(5.0), uscs="CL",
                  description="SANDY LEAN CLAY"),
            Layer(top=_ft(5.0), bottom=_ft(25.0), uscs="SM",
                  description="SILTY SAND"),
        ],
        samples=[
            Sample(sample_id="1", top=_ft(2.5), bottom=_ft(4.0), kind="spt",
                   recovery_percent=67.0, water_content=18.0,
                   dry_unit_weight=Quantity(value=112.0, unit="pcf")),
            Sample(sample_id="2", top=_ft(20.0), bottom=_ft(21.0),
                   kind="spt"),
        ],
        spt=[
            SPT(depth_top=_ft(2.5), depth_bottom=_ft(4.0), blows=[5, 9, 12],
                n=21),
            SPT(depth_top=_ft(20.0), blows=[12, 30, '50/5"'], refusal=True),
        ],
        water=[WaterLevel(depth=_ft(10.0), when="while_drilling")])


class TestAPerfectRecord:
    def test_it_scores_everything_the_truth_states(self):
        score = score_record(TRUTH, [_perfect()])
        assert score.total.found == score.total.total
        assert score.total.total == 18

    def test_every_metric_the_truth_supports_is_scored(self):
        score = score_record(TRUTH, [_perfect()])
        scored = {m for m in METRICS
                  if m in score.scores and score.scores[m].total}
        assert scored == {"n_value", "blows", "sample_depth", "layer_top",
                          "uscs", "water", "recovery", "index", "fields"}

    def test_a_field_the_log_printed_under_its_own_name_is_found(self):
        """The bug this pins: an empty canonical key used to take the slot.

        ``sheet`` lives on the investigation, and a record that carries it in
        ``fields`` instead used to score a miss because the empty
        ``inv.sheet`` had already claimed the name.
        """
        record = _perfect()
        record.sheet = ""
        record.fields = {"sheet": "1 of 1"}
        score = score_record(TRUTH, [record])
        assert not score.scores["fields"].misses


class TestWhatItCatches:
    def test_an_empty_record_scores_nothing_and_does_not_crash(self):
        score = score_record(TRUTH, [])
        assert score.total.found == 0
        assert score.total.total == 18

    def test_a_depth_outside_the_tolerance_is_a_miss(self):
        record = _perfect()
        record.layers[1].top = _ft(5.0 + (LAYER_TOL_M + 0.2) * 3.2808399)
        score = score_record(TRUTH, [record])
        assert score.scores["layer_top"].found == 1
        assert score.scores["layer_top"].total == 2

    def test_a_depth_inside_the_tolerance_is_a_hit(self):
        record = _perfect()
        record.layers[1].top = _ft(5.0 + (LAYER_TOL_M - 0.05) * 3.2808399)
        score = score_record(TRUTH, [record])
        assert score.scores["layer_top"].found == 2

    def test_a_wrong_n_value_is_a_miss_even_at_the_right_depth(self):
        record = _perfect()
        record.spt[0].n = 22
        score = score_record(TRUTH, [record])
        assert score.scores["n_value"].found == 0
        assert "N=21" in score.scores["n_value"].misses[0]

    def test_a_wrong_uscs_symbol_is_a_miss(self):
        record = _perfect()
        record.layers[0].uscs = "CH"
        score = score_record(TRUTH, [record])
        assert score.scores["uscs"].found == 1

    def test_a_water_level_off_by_more_than_the_tolerance_is_a_miss(self):
        record = _perfect()
        record.water[0].depth = _ft(12.0)
        score = score_record(TRUTH, [record])
        assert score.scores["water"].found == 0

    def test_an_index_value_on_the_wrong_sample_is_a_miss(self):
        record = _perfect()
        record.samples[0].water_content = None
        record.samples[1].water_content = 18.0
        score = score_record(TRUTH, [record])
        assert "wc=18" in " ".join(score.scores["index"].misses)


class TestTheRulesItInherits:
    def test_an_n_the_log_never_printed_is_credited_from_its_drives(self):
        """Half the templates print only the drives, and nothing computes.

        The truth states N=21 and the record's drives are 5-9-12, whose
        second and third are 21. The reader was RIGHT to leave n null, so
        this counts as found -- otherwise the metric would score a design
        decision rather than a defect.
        """
        record = _perfect()
        record.spt[0].n = None
        score = score_record(TRUTH, [record])
        assert score.scores["n_value"].found == 1

    def test_that_credit_needs_the_drives_at_the_right_depth(self):
        record = _perfect()
        record.spt[0].n = None
        record.spt[0].depth_top = _ft(14.0)
        score = score_record(TRUTH, [record])
        assert score.scores["n_value"].found == 0

    def test_a_blow_record_is_found_however_the_form_printed_it(self):
        record = _perfect()
        record.spt[0].blows = [5, 9, 12]
        assert score_record(TRUTH, [record]).scores["blows"].found == 2
        # ... and out of order, because a stack of cells asserts no order
        record.spt[0].blows = [9, 5, 12]
        assert score_record(TRUTH, [record]).scores["blows"].found == 2
        record.spt[0].blows = [4, 4, 4]
        assert score_record(TRUTH, [record]).scores["blows"].found == 1

    def test_a_refusal_is_matched_on_the_numbers_in_it(self):
        record = _perfect()
        record.spt[1].blows = [12, 30, "50/5 in"]
        score = score_record(TRUTH, [record])
        assert score.scores["blows"].found == 2

    def test_a_sample_is_an_interval_so_a_value_at_its_middle_counts(self):
        record = _perfect()
        record.samples[0].top = _ft(3.8)          # inside 2.5-4.0 + tolerance
        record.spt[0].depth_top = _ft(3.8)
        score = score_record(TRUTH, [record])
        assert score.scores["sample_depth"].found == 2
        assert score.scores["n_value"].found == 1

    def test_depths_are_compared_in_metres_whatever_the_log_prints(self):
        """The same log in metres scores the same as in feet."""
        metric_truth = dict(TRUTH)
        metric_truth["depth_unit"] = "m"
        metric_truth["layers"] = [
            {"top": 0.0, "bottom": 1.524, "description": "CLAY",
             "uscs": "CL"}]
        metric_truth["samples"] = []
        metric_truth["water"] = []
        metric_truth["fields"] = {}
        record = Investigation(
            investigation_id="B-1", depth_unit="m",
            layers=[Layer(top=Quantity(value=0.0, unit="m"),
                          bottom=Quantity(value=1.524, unit="m"),
                          uscs="CL", description="CLAY")])
        score = score_record(metric_truth, [record])
        assert score.scores["layer_top"].found == 1
        assert score.scores["uscs"].found == 1


class TestSeveralBoringsOnOneSheet:
    def test_a_tabular_sheet_is_flattened_and_scored_as_one(self):
        truth = {
            "id": "R99_p1", "pages": [1], "depth_unit": "m",
            "fields": {}, "layers": [], "samples": [], "water": [],
            "investigations": [
                {"investigation_id": "S-1",
                 "layers": [{"top": 0.0, "bottom": 2.0, "description": "FILL",
                             "uscs": "SM"}],
                 "samples": [], "water": []},
                {"investigation_id": "S-2",
                 "layers": [{"top": 0.0, "bottom": 3.0, "description": "FILL",
                             "uscs": "SM"}],
                 "samples": [], "water": []},
            ]}
        assert len(truth_investigations(truth)) == 2
        records = [
            Investigation(investigation_id="S-1", depth_unit="m",
                          layers=[Layer(top=Quantity(value=0.0, unit="m"),
                                        bottom=Quantity(value=2.0, unit="m"),
                                        uscs="SM", description="FILL")]),
            Investigation(investigation_id="S-2", depth_unit="m",
                          layers=[Layer(top=Quantity(value=0.0, unit="m"),
                                        bottom=Quantity(value=3.0, unit="m"),
                                        uscs="SM", description="FILL")]),
        ]
        score = score_record(truth, records)
        assert score.scores["layer_top"].total == 2
        assert score.scores["layer_top"].found == 2


class TestTheScoreObject:
    def test_scores_add_up(self):
        a, b = Score(), Score()
        a.add(True), a.add(False, "a miss")
        b.add(True)
        a += b
        assert (a.found, a.total) == (2, 3)
        assert a.rate == pytest.approx(2 / 3)
        assert a.misses == ["a miss"]

    def test_an_empty_score_has_no_rate_rather_than_a_zero(self):
        assert Score().rate is None

    def test_a_log_score_serialises_for_the_run_file(self):
        score = LogScore(log_id="R99_p10", report="R99", stage="record")
        score.score("n_value").add(True)
        blob = score.to_dict()
        assert blob["overall"] == {"found": 1, "total": 1, "rate": 1.0}
        assert blob["scores"]["n_value"]["found"] == 1
        import json
        json.dumps(blob)

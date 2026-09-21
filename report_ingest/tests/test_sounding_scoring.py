"""The sounding scorer's arithmetic, on records built by hand.

Nothing here opens a PDF or calls a model: the point is the tolerances and
what they let through, which are the only thing a scorecard's numbers mean.
"""

from __future__ import annotations

import pytest

from report_ingest.model import (
    CPTData, CPTPoint, DCPData, DCPPoint, Investigation, PitDimensions,
    Quantity,
)
from report_ingest.sounding_scoring import (
    CHANNEL_TOL, DEPTH_TOL, METRICS, PIT_METRICS, SERIES_METRICS, U2_TOL,
    kind_of, pages_of, report_of, score_pit, score_record, score_sounding,
    truth_series,
)


def q(value, unit):
    return Quantity(value=value, unit=unit)


def cpt(points, unit="m", **kwargs):
    return Investigation(
        investigation_id=kwargs.pop("investigation_id", "CPT-1"),
        kind="cpt", depth_unit=unit,
        cpt=CPTData(points=points, **kwargs))


def dcp(points, unit="m", **kwargs):
    return Investigation(
        investigation_id=kwargs.pop("investigation_id", "DP-1"),
        kind="dcp", depth_unit=unit,
        dcp=DCPData(points=points, **kwargs))


def point(depth, qc=None, fs=None, u2=None, unit="m"):
    return CPTPoint(depth=q(depth, unit),
                    qc=None if qc is None else q(qc, "MPa"),
                    fs=None if fs is None else q(fs, "MPa"),
                    u2=None if u2 is None else q(u2, "kPa"))


TRUTH = {
    "id": "cpt__R20_p47", "report": "R20", "kind": "cpt", "pages": [47],
    "depth_unit": "m", "qc_unit": "MPa", "fs_unit": "MPa", "u2_unit": "kPa",
    "investigation_id": "CPT-1",
    "series": [
        {"depth": 0.5, "qc": 10.0, "fs": 0.10, "u2": 40.0},
        {"depth": 1.0, "qc": 20.0, "fs": 0.20, "u2": 80.0},
    ],
}


def _rate(score, metric):
    got = score.scores.get(metric)
    return None if got is None else got.rate


# ---------------------------------------------------------------------------
# reading a truth file
# ---------------------------------------------------------------------------

class TestReadingATruthFile:

    def test_the_kind_comes_off_the_file_or_its_own_id(self):
        assert kind_of({"kind": "dcp"}) == "dcp"
        assert kind_of({"id": "test_pit__R23_p91"}) == "test_pit"
        assert kind_of({"id": "dcp__R28_p89"}) == "dcp"

    def test_the_pages_come_off_the_file_or_its_own_id(self):
        assert pages_of({"pages": [91, 92]}) == [91, 92]
        assert pages_of({"id": "cpt__R20_p47"}) == [47]

    def test_the_report_comes_off_the_file_or_its_own_id(self):
        assert report_of({"report": "R20"}) == "R20"
        assert report_of({"id": "dcp__R29_p63"}) == "R29"

    def test_a_file_with_no_series_reads_as_an_empty_one(self):
        assert truth_series({}) == []


# ---------------------------------------------------------------------------
# a cone sounding
# ---------------------------------------------------------------------------

class TestScoringACone:

    def test_an_exact_record_scores_everything(self):
        score = score_sounding(
            TRUTH, [cpt([point(0.5, 10.0, 0.10, 40.0),
                         point(1.0, 20.0, 0.20, 80.0)])], "record")
        assert _rate(score, "depth") == 1.0
        assert _rate(score, "qc") == 1.0
        assert _rate(score, "fs") == 1.0
        assert _rate(score, "u2") == 1.0

    def test_a_tip_resistance_within_five_per_cent_is_found(self):
        inside = 10.0 * (1 + CHANNEL_TOL * 0.9)
        truth = dict(TRUTH, series=[{"depth": 0.5, "qc": 10.0}])
        score = score_sounding(truth, [cpt([point(0.5, inside)])], "record")
        assert _rate(score, "qc") == 1.0

    def test_a_tip_resistance_past_five_per_cent_is_a_miss(self):
        outside = 10.0 * (1 + CHANNEL_TOL * 3)
        score = score_sounding(
            TRUTH, [cpt([point(0.5, outside, 0.10, 40.0)])], "record")
        assert score.scores["qc"].found == 0

    def test_one_axis_tick_widens_the_tolerance(self):
        # On a coarse axis five per cent is finer than the paper can state.
        truth = dict(TRUTH, axis_ticks={"qc": 5.0},
                     series=[{"depth": 0.5, "qc": 10.0}])
        score = score_sounding(truth, [cpt([point(0.5, 14.0)])], "record")
        assert _rate(score, "qc") == 1.0

    def test_pore_pressure_gets_its_own_looser_tolerance(self):
        assert U2_TOL > CHANNEL_TOL
        inside = 40.0 * (1 + U2_TOL * 0.9)
        truth = dict(TRUTH, series=[{"depth": 0.5, "u2": 40.0}])
        score = score_sounding(truth, [cpt([point(0.5, u2=inside)])],
                               "record")
        assert _rate(score, "u2") == 1.0

    def test_a_reading_at_the_wrong_depth_is_not_that_reading(self):
        score = score_sounding(
            TRUTH, [cpt([point(0.5 + DEPTH_TOL * 3, 10.0, 0.10, 40.0)])],
            "record")
        assert score.scores["depth"].found == 0
        assert score.scores["qc"].found == 0

    def test_a_depth_the_truth_does_not_carry_is_not_counted(self):
        # The truth is a SAMPLE of the sounding; a reader may carry more.
        score = score_sounding(
            TRUTH, [cpt([point(0.5, 10.0, 0.10, 40.0),
                         point(0.75, 15.0, 0.15, 60.0),
                         point(1.0, 20.0, 0.20, 80.0)])], "record")
        assert score.scores["depth"].total == 2
        assert _rate(score, "depth") == 1.0

    def test_a_record_with_nothing_in_it_misses_everything(self):
        score = score_sounding(TRUTH, [], "record")
        assert score.scores["depth"].found == 0
        assert score.scores["qc"].found == 0
        assert score.scores["qc"].total == 2

    def test_a_unit_the_table_cannot_convert_still_compares(self):
        # A dynamic resistance in daN/cm2 converts to nothing; the truth and
        # the record are then compared in the sheet's own unit.
        truth = {"id": "dcp__RXX_p1", "kind": "dcp", "depth_unit": "m",
                 "index_unit": "daN/cm2",
                 "series": [{"depth": 1.0, "index": 44.8}]}
        record = dcp([DCPPoint(depth=q(1.0, "m"),
                               index=q(44.8, "daN/cm2"))])
        score = score_sounding(truth, [record], "record")
        assert _rate(score, "index") == 1.0


# ---------------------------------------------------------------------------
# a dynamic probe
# ---------------------------------------------------------------------------

DCP_TRUTH = {
    "id": "dcp__R28_p112", "report": "R28", "kind": "dcp", "pages": [112],
    "depth_unit": "m", "index_unit": "MPa", "investigation_id": "DP-1",
    "series": [{"depth": 0.2, "blows": 21, "index": 7.3},
               {"depth": 0.4, "blows": 23, "index": 8.0}],
}


class TestScoringADynamicProbe:

    def test_an_exact_record_scores_everything(self):
        record = dcp([DCPPoint(depth=q(0.2, "m"), blows=21.0,
                               index=q(7.3, "MPa")),
                      DCPPoint(depth=q(0.4, "m"), blows=23.0,
                               index=q(8.0, "MPa"))])
        score = score_sounding(DCP_TRUTH, [record], "record")
        assert _rate(score, "blows") == 1.0
        assert _rate(score, "index") == 1.0

    def test_a_blow_count_is_exact_and_has_no_tolerance(self):
        # Nineteen is not twenty. There is no tolerance on counting.
        truth = dict(DCP_TRUTH, series=[{"depth": 0.2, "blows": 21}])
        record = dcp([DCPPoint(depth=q(0.2, "m"), blows=20.0)])
        score = score_sounding(truth, [record], "record")
        assert score.scores["blows"].found == 0

    def test_a_fractional_blow_count_is_kept_as_printed(self):
        truth = dict(DCP_TRUTH, series=[{"depth": 1.0, "blows": 0.5}])
        record = dcp([DCPPoint(depth=q(1.0, "m"), blows=0.5)])
        assert _rate(score_sounding(truth, [record], "record"),
                     "blows") == 1.0

    def test_an_index_gets_the_channel_tolerance(self):
        truth = dict(DCP_TRUTH,
                     series=[{"depth": 0.2, "blows": 21, "index": 7.3}])
        record = dcp([DCPPoint(depth=q(0.2, "m"), blows=21.0,
                               index=q(7.3 * 1.03, "MPa"))])
        score = score_sounding(truth, [record], "record")
        assert _rate(score, "index") == 1.0


# ---------------------------------------------------------------------------
# a test pit
# ---------------------------------------------------------------------------

PIT_TRUTH = {
    "id": "test_pit__RXX_p1", "report": "RXX", "kind": "test_pit",
    "pages": [1], "depth_unit": "m",
    "layers": [{"top": 0.0, "bottom": 0.4, "description": "TOPSOIL",
                "uscs": None},
               {"top": 0.4, "bottom": 1.3, "description": "CLAYEY SAND",
                "uscs": "SC"}],
    "samples": [{"id": "S-1", "top": 0.5, "bottom": 0.6}],
    "water": [{"depth": 1.0}],
    "fields": {"test_pit_id": "TP-1", "total_depth": 1.30},
    "dimensions": {"width": 90, "unit": "cm"},
}


def _pit(**kwargs):
    from report_ingest.model import Layer, Sample, WaterLevel
    inv = Investigation(investigation_id="TP-1", kind="test_pit",
                        depth_unit="m", total_depth=q(1.30, "m"))
    inv.layers = [Layer(top=q(0.0, "m"), bottom=q(0.4, "m"),
                        description="TOPSOIL"),
                  Layer(top=q(0.4, "m"), bottom=q(1.3, "m"),
                        description="CLAYEY SAND", uscs="SC")]
    inv.samples = [Sample(sample_id="S-1", top=q(0.5, "m"),
                          bottom=q(0.6, "m"), kind="bulk")]
    inv.water = [WaterLevel(depth=q(1.0, "m"), when="at_completion")]
    for key, value in kwargs.items():
        setattr(inv, key, value)
    return inv


class TestScoringAPit:

    def test_a_pit_is_scored_by_the_log_scorers_metrics(self):
        score = score_pit(PIT_TRUTH, [_pit()], "record")
        assert _rate(score, "layer_top") == 1.0
        assert _rate(score, "uscs") == 1.0
        assert _rate(score, "sample_depth") == 1.0
        assert _rate(score, "water") == 1.0

    def test_the_dimensions_are_the_metric_a_pit_adds(self):
        assert "dimensions" in PIT_METRICS
        assert "dimensions" not in SERIES_METRICS
        score = score_pit(
            PIT_TRUTH,
            [_pit(pit=PitDimensions(width=q(90.0, "cm")))], "record")
        assert _rate(score, "dimensions") == 1.0

    def test_a_dimension_in_another_unit_still_matches(self):
        score = score_pit(
            PIT_TRUTH, [_pit(pit=PitDimensions(width=q(0.9, "m")))],
            "record")
        assert _rate(score, "dimensions") == 1.0

    def test_a_pit_with_no_dimensions_misses_the_metric(self):
        score = score_pit(PIT_TRUTH, [_pit()], "record")
        assert score.scores["dimensions"].found == 0
        assert score.scores["dimensions"].total == 1

    def test_a_truth_that_asks_no_dimension_asks_nothing(self):
        truth = dict(PIT_TRUTH)
        truth.pop("dimensions")
        score = score_pit(truth, [_pit()], "record")
        assert "dimensions" not in score.scores


# ---------------------------------------------------------------------------
# the shape of a score
# ---------------------------------------------------------------------------

class TestTheScore:

    def test_score_record_dispatches_on_the_kind(self):
        assert score_record(PIT_TRUTH, [_pit()]).kind == "test_pit"
        assert score_record(TRUTH, [cpt([])]).kind == "cpt"

    def test_the_total_adds_every_metric_once(self):
        score = score_sounding(
            TRUTH, [cpt([point(0.5, 10.0, 0.10, 40.0)])], "record")
        total = score.total
        assert total.total == sum(score.scores[m].total for m in METRICS
                                  if m in score.scores)

    def test_a_score_serialises_for_a_run_file(self):
        blob = score_sounding(TRUTH, [cpt([])], "record").to_dict()
        assert blob["kind"] == "cpt"
        assert blob["stage"] == "record"
        assert "overall" in blob and "scores" in blob

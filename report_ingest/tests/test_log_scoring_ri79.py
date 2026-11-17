"""The log scorer's conventions for a paired public answer (BACKLOG RI-79).

Made-up records and made-up answer sheets only. Each test of items 1 to 4
states a miss that was the scorer's own, not a reader's: a blow record
joined in one string read as its first count, a refusal's penetration read
as a count, a tube's blows credited from another test's increments, and an
N worked out by a rule nothing held prints for that number of increments.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from report_ingest.log_scoring import (
    METRICS, N_RULES, SET_ASIDE, _blow_entries, _blow_numbers, rescore_run,
    rescore_saved, score_grid, score_record,
)
from report_ingest.model import Investigation, Quantity, SPT, Sample


def _m(value: float) -> Quantity:
    return Quantity(value=value, unit="m")


def _truth(*samples, **more):
    return {"id": "P98_p0", "investigation_id": "BH7", "pages": [0],
            "depth_unit": "m", "fields": {}, "layers": [],
            "samples": list(samples), "water": [], **more}


def _record(*spt, samples=(), **more):
    return Investigation(investigation_id="BH7", depth_unit="m",
                         samples=list(samples), spt=list(spt), **more)


# -- 1. increments joined in one string ---------------------------------------

@pytest.mark.parametrize("entry, counts", [
    ("3,6", [3, 6]), ("3, 6, 8", [3, 6, 8]), ("3/6/8", [3, 6, 8]),
    ("3-6-8", [3, 6, 8]), ("1,2/2,3,3,4", [1, 2, 2, 3, 3, 4]),
    ("12,15,20", [12, 15, 20]), ("7", [7]), ("WOH", []),
])
def test_every_increment_of_a_joined_record_is_read(entry, counts):
    assert _blow_numbers([entry]) == [float(c) for c in counts]


def test_a_record_written_as_one_string_matches_its_increments():
    """The record keeps "3,6,8" as the log printed it; the answer has the
    three counts. Read as its first number only, the record held 3."""
    truth = _truth({"top": 4.0, "bottom": 4.45, "blows": [3, 6, 8],
                    "n": None})
    for printed in ("3,6,8", "3, 6, 8", "3/6/8", "3-6-8"):
        record = _record(SPT(depth_top=_m(4.0), blows=[printed]))
        assert score_record(truth, [record]).scores["blows"].found == 1, \
            printed
    wrong = _record(SPT(depth_top=_m(4.0), blows=["3,6,9"]))
    assert score_record(truth, [wrong]).scores["blows"].found == 0


def test_an_answer_written_as_one_string_is_read_whole_too():
    truth = _truth({"top": 4.0, "bottom": 4.45, "blows": ["3,6,8"],
                    "n": None})
    record = _record(SPT(depth_top=_m(4.0), blows=[3, 6, 8]))
    assert score_record(truth, [record]).scores["blows"].found == 1
    short = _record(SPT(depth_top=_m(4.0), blows=[3]))
    assert score_record(truth, [short]).scores["blows"].found == 0


# -- 2. a count over a penetration ----------------------------------------------

@pytest.mark.parametrize("entry, count, pen, unit", [
    ("50/20", 50, 20, ""), ('50/5"', 50, 5, "in"), ("50/20mm", 50, 20, "mm"),
    ("50/5 in", 50, 5, "in"), ("50 for 20 mm", 50, 20, "mm"),
    ("50 blows for 20mm", 50, 20, "mm"), ("50/0.1m", 50, 0.1, "m"),
])
def test_a_count_over_a_penetration_is_one_count(entry, count, pen, unit):
    assert _blow_entries([entry]) == [(float(count), float(pen), unit)]
    assert _blow_numbers([entry]) == [float(count)]


def test_a_refusals_penetration_is_not_a_blow_count():
    """The answer carries the counts (10, 15, 50); the record keeps the last
    drive as the log printed it, 50 blows over 20 mm. With the 20 read as a
    fourth count the record never held the answer's three."""
    truth = _truth({"top": 6.0, "bottom": None, "blows": [10, 15, 50],
                    "n": None, "refusal": True})
    for last in ("50/20", "50/20mm", '50/1"'):
        record = _record(SPT(depth_top=_m(6.0), blows=[10, 15, last],
                             refusal=True))
        assert _blow_numbers(record.spt[0].blows) == [10.0, 15.0, 50.0]
        assert score_record(truth, [record]).scores["blows"].found == 1


def test_a_penetration_is_compared_where_both_sides_state_one():
    truth = _truth({"top": 6.0, "bottom": None,
                    "blows": [10, 15, "50/20mm"], "n": None})
    same = _record(SPT(depth_top=_m(6.0), blows=[10, 15, "50/20 mm"]))
    other = _record(SPT(depth_top=_m(6.0), blows=[10, 15, "50/60mm"]))
    assert score_record(truth, [same]).scores["blows"].found == 1
    assert score_record(truth, [other]).scores["blows"].found == 0


# -- 3. blows on a driven tube sample -------------------------------------------

TUBE = {"id": "U2", "top": 2.0, "bottom": 2.45, "type": "U", "blows": [8],
        "blows_kind": "sampler", "n": None}


def test_a_tubes_blows_are_found_on_a_drive_record_at_its_depth():
    record = _record(
        SPT(depth_top=_m(2.0), depth_bottom=_m(2.45), blows=["8 blows"],
            sample_id="U2"),
        samples=[Sample(sample_id="U2", top=_m(2.0), bottom=_m(2.45))])
    assert score_record(_truth(TUBE), [record]).scores["blows"].found == 1


def test_a_tubes_blows_are_not_found_among_another_tests_increments():
    """A penetration test follows the tube, inside the tube's window, and
    one of its six increments is an 8. That is not the tube's blow count."""
    record = _record(
        SPT(depth_top=_m(2.45), blows=[1, 2, 2, 3, 8, 4], n=17),
        samples=[Sample(sample_id="U2", top=_m(2.0), bottom=_m(2.45),
                        note="8 blows")])
    blows = score_record(_truth(TUBE), [record]).scores["blows"]
    assert (blows.found, blows.total) == (0, 1)
    assert "sampler blows" in blows.misses[0]


def test_a_sheet_without_the_mark_is_scored_as_before():
    plain = {k: v for k, v in TUBE.items() if k != "blows_kind"}
    record = _record(SPT(depth_top=_m(2.45), blows=[1, 2, 2, 3, 8, 4], n=17))
    assert score_record(_truth(plain), [record]).scores["blows"].found == 1


# -- 4. N from the increments -----------------------------------------------------

def test_the_rule_held_is_three_increments_and_the_occasional_fourth():
    """p. 3-51's 2nd + 3rd, and for a fourth drive the same (Fig. 3-20,
    p. 3-52, the record's own reading; the lead's call of 2026-11-13)."""
    assert N_RULES == {3: (1, 2), 4: (1, 2)}


def test_four_increments_with_no_printed_n_are_credited_as_on_master():
    truth = _truth({"top": 3.0, "bottom": 3.6, "blows": [5, 9, 12, 14],
                    "n": 21})
    record = _record(SPT(depth_top=_m(3.0), blows=[5, 9, 12, 14]))
    score = score_record(truth, [record])
    assert (score.scores["n_value"].found,
            score.scores["n_value"].total) == (1, 1)
    assert score.set_aside == {}


def test_five_increments_with_no_printed_n_are_still_set_aside():
    truth = _truth({"top": 3.0, "bottom": 3.75,
                    "blows": [5, 9, 12, 14, 16], "n": 21})
    record = _record(SPT(depth_top=_m(3.0), blows=[5, 9, 12, 14, 16]))
    score = score_record(truth, [record])
    assert "n_value" not in score.scores
    assert score.set_aside == {"n_no_rule": ["N=21 at 3.0 m (5 increments)"]}


def test_three_increments_with_no_printed_n_are_credited_by_the_rule():
    truth = _truth({"top": 3.0, "bottom": 3.45, "blows": [5, 9, 12],
                    "n": 21})
    record = _record(SPT(depth_top=_m(3.0), blows=[5, 9, 12]))
    score = score_record(truth, [record])
    assert (score.scores["n_value"].found, score.set_aside) == (1, {})


SIX = {"top": 3.0, "bottom": 3.45, "blows": [1, 2, 2, 3, 3, 4], "n": 12}


def test_six_increments_with_no_printed_n_are_set_aside_not_missed():
    """No page held says how six increments make an N, so nothing is worked
    out: the answer's N is counted in a group of its own, in no total."""
    record = _record(SPT(depth_top=_m(3.0), blows=["1,2/2,3,3,4"]))
    score = score_record(_truth(SIX), [record])
    assert "n_value" not in score.scores
    assert score.set_aside == {"n_no_rule": ["N=12 at 3.0 m (6 increments)"]}
    assert score.scores["blows"].found == 1
    assert score.total.total == sum(
        s.total for name, s in score.scores.items() if name in METRICS)
    blob = score.to_dict()["set_aside"]["n_no_rule"]
    assert blob["count"] == 1 and blob["what"] == SET_ASIDE["n_no_rule"]


def test_a_six_increment_n_is_never_credited_by_the_three_increment_rule():
    """2 + 2 is not this record's N whatever the answer says."""
    truth = _truth({**SIX, "n": 4})
    record = _record(SPT(depth_top=_m(3.0), blows=[1, 2, 2, 3, 3, 4]))
    score = score_record(truth, [record])
    assert "n_value" not in score.scores and score.set_aside["n_no_rule"]


def test_a_printed_n_on_six_increments_is_scored_as_ever():
    right = _record(SPT(depth_top=_m(3.0), blows=[1, 2, 2, 3, 3, 4], n=12))
    wrong = _record(SPT(depth_top=_m(3.0), blows=[1, 2, 2, 3, 3, 4], n=11))
    for record, found in ((right, 1), (wrong, 0)):
        score = score_record(_truth(SIX), [record])
        assert (score.scores["n_value"].found,
                score.scores["n_value"].total) == (found, 1)
        assert not score.set_aside


def test_an_n_with_no_drives_at_that_depth_is_still_a_miss():
    for record in (_record(),
                   _record(SPT(depth_top=_m(3.0), blows=[9, 9, 9, 9, 9, 9]))):
        score = score_record(_truth(SIX), [record])
        assert (score.scores["n_value"].found,
                score.scores["n_value"].total) == (0, 1)
        assert not score.set_aside


# -- 6. a struck header field -------------------------------------------------------

def test_a_struck_field_is_in_no_total_and_an_unstruck_one_is_scored():
    truth = _truth(fields={"client": "A Client", "project": "Made-up Road"},
                   fields_from_data=["client", "project"],
                   fields_struck=["client"])
    record = _record(fields={"project": "Made-up Road"})
    score = score_record(truth, [record])
    assert (score.scores["fields"].found, score.scores["fields"].total) == (
        1, 1)
    assert score.set_aside == {"fields_struck": ["client='A Client'"]}
    truth["fields_struck"] = []
    unstruck = score_record(truth, [record])
    assert (unstruck.scores["fields"].found,
            unstruck.scores["fields"].total) == (1, 2)


# -- the grid scorer's own branches (items 4 and 6) -------------------------------

def test_the_grid_scorer_sets_aside_an_n_with_no_rule_and_a_struck_field():
    """The grid path: six increments standing in a blows column with no N
    printed set the answer's N aside, and a struck field is in no total."""
    cell = SimpleNamespace(depth=3.0, depth_top=None, depth_bottom=None,
                           column_id="c1", numbers=[1, 2, 2, 3, 3, 4],
                           bbox=(0.0, 0.0, 1.0, 1.0))
    columns = {"c1": SimpleNamespace(names=("blows",))}
    grid = SimpleNamespace(rows=[cell], unit="m", column=columns.get,
                           fields={"project": "Made-up Road"}, layers=[],
                           water=[], warnings=[])
    truth = _truth(SIX, fields={"client": "A Client",
                                "project": "Made-up Road"},
                   fields_struck=["client"])
    score = score_grid(truth, grid)
    assert "n_value" not in score.scores
    assert score.set_aside["n_no_rule"] == ["N=12 at 3.0 m (6 increments)"]
    assert score.scores["blows"].found == 1
    assert (score.scores["fields"].found, score.scores["fields"].total) == (
        1, 1)
    assert score.set_aside["fields_struck"] == ["client='A Client'"]


# -- a saved run, re-scored with no model call ----------------------------------------

def test_a_saved_run_is_rescored_from_its_files(tmp_path):
    truth = _truth(SIX, {"top": 6.0, "bottom": None, "blows": [10, 15, 50],
                         "n": None})
    record = _record(SPT(depth_top=_m(3.0), blows=["1,2/2,3,3,4"]),
                     SPT(depth_top=_m(6.0), blows=[10, 15, "50/20"]))
    dumped = [record.model_dump(mode="json")]
    # As a run before this change saved it: both blow records and the N
    # scored as missed.
    old = {"scores": {"blows": {"found": 0, "total": 2},
                      "n_value": {"found": 0, "total": 1},
                      "sample_depth": {"found": 2, "total": 2}},
           "record": dumped, "model_record": dumped}
    logs, sheets = tmp_path / "out" / "logs", tmp_path / "truth"
    logs.mkdir(parents=True)
    sheets.mkdir()
    (logs / "P98_p0.json").write_text(
        json.dumps({"log_id": "P98_p0", "after": old}), encoding="utf-8")
    (logs / "P98_p1.json").write_text(
        json.dumps({"log_id": "P98_p1", "after": old}), encoding="utf-8")
    (sheets / "P98_p0.json").write_text(json.dumps(truth), encoding="utf-8")

    got = rescore_run(logs, sheets)
    assert got["run_files"] == 1
    assert got["skipped"] == {"P98_p1": "no answer sheet"}
    assert got["saved"]["blows"] == [0, 2]
    assert got["rescored"]["blows"] == [2, 2]
    assert got["saved"]["overall"] == [2, 5]
    assert got["rescored"]["overall"] == [4, 4]
    assert got["set_aside"] == {"n_no_rule": 1}
    # Nothing is written unless asked.
    assert json.loads((logs / "P98_p0.json").read_text())["after"] == old

    rescore_run(logs, sheets, write=True)
    blob = json.loads((logs / "P98_p0.json").read_text())
    assert blob["after_as_run"] == old
    assert blob["after"]["set_aside"]["n_no_rule"]["count"] == 1
    assert blob["after"]["model_alone"]["set_aside"]
    # A second pass still reports the run as it was first saved.
    assert rescore_run(logs, sheets, write=True)["saved"]["blows"] == [0, 2]


def test_a_stale_set_aside_does_not_outlive_a_rescore():
    truth = _truth({"top": 3.0, "bottom": 3.45, "blows": [5, 9, 12],
                    "n": 21})
    record = _record(SPT(depth_top=_m(3.0), blows=[5, 9, 12]))
    after = {"record": [record.model_dump(mode="json")],
             "set_aside": {"n_no_rule": {"count": 1}}}
    assert "set_aside" not in rescore_saved(truth, after)


def test_the_results_say_what_was_set_aside():
    from report_ingest.cluster_scoring import _render_logs, _score_logs

    record = _record(SPT(depth_top=_m(3.0), blows=["1,2/2,3,3,4"]))
    after = score_record(_truth(SIX), [record]).to_dict()
    row = {"log_id": "P98_p0", "set": "blind", "cost": {}, "seconds": 1.0,
           "before": {"scores": {}, "overall": {"found": 0, "total": 0}},
           "after": after}
    text = "\n".join(_render_logs(_score_logs({"P98_p0": row}, {}, "m", [])))
    assert "Set aside from `after`, in no total: 1 -- N not printed" in text
    # ... and nothing is said where nothing was.
    row["after"] = score_record(
        _truth(SIX), [_record(SPT(depth_top=_m(3.0), n=12,
                                  blows=[1, 2, 2, 3, 3, 4]))]).to_dict()
    text = "\n".join(_render_logs(_score_logs({"P98_p0": row}, {}, "m", [])))
    assert "Set aside" not in text

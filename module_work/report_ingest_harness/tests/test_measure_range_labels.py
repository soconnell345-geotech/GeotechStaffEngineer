"""Offline tests of ``measure_range_labels`` (BACKLOG RI-73), made-up labels.

Run::

    .venv/Scripts/python -m pytest module_work/report_ingest_harness/tests/test_measure_range_labels.py -q
"""

from __future__ import annotations

from module_work.report_ingest_harness import measure_range_labels as mrl


HAND = {"R99": {0: "narrative", 1: "divider", 2: "lab_test", 3: "lab_test",
                4: "lab_test", 5: "divider", 6: "boring_log",
                7: "boring_log"}}
GIVEN = {"R99": {0: "narrative", 1: "lab_test", 2: "figure", 3: "figure",
                 4: "lab_test", 5: "divider", 6: "divider", 7: "other"}}


def test_dividers_are_left_out_on_either_side():
    kept = mrl.without_dividers(HAND["R99"], GIVEN["R99"])
    assert sorted(kept) == [0, 2, 3, 4, 7]


def test_strict_and_harmful_are_scored_with_and_without_dividers():
    scored = mrl.score_sets(HAND, {"made-up": GIVEN})["made-up"]
    # Wrong: 1 (divider->lab_test, harmful), 2-3 (lab_test->figure,
    # harmful), 6 (boring_log->divider, harmful), 7 (boring_log->other,
    # harmful). Right: 0, 4, 5.
    assert scored["right"] == 3 and scored["harmful"] == 5
    # Without dividers: pages 0, 2, 3, 4, 7; right 0 and 4.
    assert scored["pages_nodiv"] == 5 and scored["right_nodiv"] == 2
    assert scored["harmful_nodiv"] == 3
    lines = "\n".join(mrl.report_lines({"made-up": scored}))
    assert "HARMFUL, no dividers" in lines and "strict, no dividers" in lines


def test_the_harmful_runs_are_listed_largest_first():
    runs = mrl.harmful_runs(HAND, GIVEN)
    assert runs[0] == ("R99", 2, 3, "lab_test", "figure",
                       "should be read, is not")
    assert ("R99", 1, 1, "divider", "lab_test",
            "read, should not be") in runs
    assert len(runs) == 4



def test_a_runs_column_is_named_by_the_folder_or_by_name(tmp_path):
    folder = tmp_path / "plain"
    assert mrl.runs_column(f"made-up={folder}") == ("made-up", folder)
    assert mrl.runs_column(str(folder)) == ("plain", folder)

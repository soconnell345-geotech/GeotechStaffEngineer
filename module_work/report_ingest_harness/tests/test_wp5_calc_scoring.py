"""The WP5 CALCULATION scorecard's own arithmetic, with no corpus and no model.

The scores are built by hand here, so the whole of the report renders for
real: the two columns, the three metrics the floor is not asked, the per-kind
and per-run tables, and the line that says there is no blind set.

Not to be confused with ``test_wp5_scoring``, which is the vision-labels
experiment that carried the same work-package number.
"""

from __future__ import annotations

import pytest

from module_work.report_ingest_harness import measure_wp5_calc as m
from report_ingest.calc_scoring import CalcScore, Score


def _score(calc_id, report, kind, stage, found, **over):
    score = CalcScore(calc_id=calc_id, report=report, kind=kind, stage=stage)
    for metric, (f, t) in found.items():
        score.scores[metric] = Score(found=f, total=t)
    for name, value in over.items():
        setattr(score, name, value)
    return score


def _pair(calc_id, report, kind, **over):
    before = _score(calc_id, report, kind, "floor",
                    {"program": (1, 1), "inputs": (9, 13),
                     "results": (1, 4)}, n_pages=3, floor_values=48)
    after = _score(calc_id, report, kind, "record",
                   {"kind": (1, 1), "program": (1, 1), "method": (1, 1),
                    "subject": (1, 1), "inputs": (12, 13),
                    "results": (4, 4)},
                   model_calls=1, tool_calls=0, unresolved=0, misplaced=1,
                   disagreements=1, kept=34, added=6, reconciled=8,
                   cost={"input_tokens": 21000, "output_tokens": 1400,
                         "dollars": 0.0, "seconds": 19.0})
    for name, value in over.items():
        setattr(after, name, value)
    return before, after


ROWS = [_pair("settlement__R18_p41", "R18", "settlement"),
        _pair("pavement__R29_p127", "R29", "pavement")]


class TestTheScorecard:

    def test_the_two_columns_are_both_printed(self):
        text = m.report(ROWS, openset=("R18", "R29"))

        assert "metric" in text and "before" in text and "after" in text
        assert "inputs" in text and "results" in text

    def test_the_three_the_floor_cannot_answer_have_no_before(self):
        text = m.report(ROWS, openset=("R18", "R29"))
        seen = set()
        for line in text.splitlines():
            if "runs" in line and "pages" in line:
                break                            # the per-kind table starts
            if line.startswith(("kind", "method", "subject")):
                assert line.split()[1] == "-", line
                seen.add(line.split()[0])

        assert seen == {"kind", "method", "subject"}

    def test_the_before_overall_leaves_those_three_out(self):
        """Or it would count six metrics in one column and three in the
        other and call the difference the model."""
        text = m.report(ROWS, openset=("R18", "R29"))
        overall = next(ln for ln in text.splitlines()
                       if ln.startswith("OVERALL"))

        assert "22/36" in overall            # program + inputs + results
        assert "40/42" in overall            # all six, after

    def test_every_run_has_a_line_with_its_id_and_its_pages(self):
        text = m.report(ROWS, openset=("R18", "R29"))

        assert "settlement__R18_p41" in text
        assert "pavement__R29_p127" in text
        assert "floor" in text and "misp" in text

    def test_the_per_kind_table_appears_when_there_is_more_than_one(self):
        text = m.report(ROWS, openset=("R18", "R29"))

        assert "kind" in text and "runs" in text and "pages" in text

    def test_the_merge_counts_are_reported(self):
        text = m.report(ROWS, openset=("R18", "R29"))

        assert "merge: 2 disagreement(s), 68 kept from the floor" in text
        assert "12 added by the reader, 16 reconciled" in text

    def test_a_reader_that_failed_is_loud_and_out_of_every_number(self):
        before, after = _pair("slope_stability__R23_p386", "R23",
                              "slope_stability")
        after.error = "BadRequestError: too many images"
        rows = ROWS + [(before, after)]

        text = m.report(rows, openset=("R18", "R29", "R23"))

        assert "THE READER FAILED ON 1 OF 3 RUN(S)" in text
        assert "READER FAILED" in text
        assert "too many images" in text

    def test_the_floor_alone_prints_no_after_column(self):
        rows = [(before, None) for before, _after in ROWS]

        text = m.report(rows, openset=("R18", "R29"))

        assert "merge:" not in text
        assert "cost:" not in text

    def test_the_blind_group_appears_only_when_something_is_blind(self):
        assert "blind" not in m.report(ROWS, openset=("R18", "R29"))
        assert "blind --" in m.report(ROWS, openset=("R18",))

    def test_the_detail_lists_the_misses(self):
        before, after = _pair("settlement__R16_p162", "R16", "settlement")
        after.scores["results"].misses = ["rho = 25.8 mm [p164]"]

        text = m.report([(before, after)], openset=("R16",), detail=True)

        assert "misses:" in text
        assert "settlement__R16_p162 record results: rho = 25.8 mm" in text


class TestTheTruthFolder:

    def test_the_truth_folder_is_under_the_gitignored_raw_tree(self):
        assert "raw" in m.TRUTH_DIR.parts
        assert m.TRUTH_DIR.name == "calc"

    def test_a_missing_truth_folder_says_where_it_should_be(self,
                                                             monkeypatch,
                                                             tmp_path):
        monkeypatch.setattr(m, "TRUTH_DIR", tmp_path / "nowhere")

        with pytest.raises(FileNotFoundError, match="gitignored"):
            m.load_truth()

    def test_the_open_set_comes_off_open_txt(self, monkeypatch, tmp_path):
        (tmp_path / "OPEN.txt").write_text("R18\nR29\n", encoding="utf-8")
        monkeypatch.setattr(m, "TRUTH_DIR", tmp_path)

        assert m.open_reports() == ("R18", "R29")

    def test_no_open_txt_means_no_open_set(self, monkeypatch, tmp_path):
        monkeypatch.setattr(m, "TRUTH_DIR", tmp_path)

        assert m.open_reports() == ()

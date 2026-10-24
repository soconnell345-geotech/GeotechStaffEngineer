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


# Every id and value here is made up: none comes off a report.
ROWS = [_pair("settlement__R91_p6", "R91", "settlement"),
        _pair("pavement__R92_p9", "R92", "pavement")]


class TestTheScorecard:

    def test_the_two_columns_are_both_printed(self):
        text = m.report(ROWS, openset=("R91", "R92"))

        assert "metric" in text and "before" in text and "after" in text
        assert "inputs" in text and "results" in text

    def test_the_three_the_floor_cannot_answer_have_no_before(self):
        text = m.report(ROWS, openset=("R91", "R92"))
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
        text = m.report(ROWS, openset=("R91", "R92"))
        overall = next(ln for ln in text.splitlines()
                       if ln.startswith("OVERALL"))

        assert "22/36" in overall            # program + inputs + results
        assert "40/42" in overall            # all six, after

    def test_every_run_has_a_line_with_its_id_and_its_pages(self):
        text = m.report(ROWS, openset=("R91", "R92"))

        assert "settlement__R91_p6" in text
        assert "pavement__R92_p9" in text
        assert "floor" in text and "misp" in text

    def test_the_per_kind_table_appears_when_there_is_more_than_one(self):
        text = m.report(ROWS, openset=("R91", "R92"))

        assert "kind" in text and "runs" in text and "pages" in text

    def test_the_merge_counts_are_reported(self):
        text = m.report(ROWS, openset=("R91", "R92"))

        assert "merge: 2 disagreement(s), 68 kept from the floor" in text
        assert "12 added by the reader, 16 reconciled" in text

    def test_a_reader_that_failed_is_loud_and_out_of_every_number(self):
        before, after = _pair("slope_stability__R93_p4", "R93",
                              "slope_stability")
        after.error = "BadRequestError: too many images"
        rows = ROWS + [(before, after)]

        text = m.report(rows, openset=("R91", "R92", "R93"))

        assert "THE READER FAILED ON 1 OF 3 RUN(S)" in text
        assert "READER FAILED" in text
        assert "too many images" in text

    def test_the_floor_alone_prints_no_after_column(self):
        rows = [(before, None) for before, _after in ROWS]

        text = m.report(rows, openset=("R91", "R92"))

        assert "merge:" not in text
        assert "cost:" not in text

    def test_the_blind_group_appears_only_when_something_is_blind(self):
        assert "blind" not in m.report(ROWS, openset=("R91", "R92"))
        assert "blind --" in m.report(ROWS, openset=("R91",))

    def test_the_detail_lists_the_misses(self):
        before, after = _pair("settlement__R94_p3", "R94", "settlement")
        after.scores["results"].misses = ["rho = 17.6 mm [p5]"]

        text = m.report([(before, after)], openset=("R94",), detail=True)

        assert "misses:" in text
        assert "settlement__R94_p3 record results: rho = 17.6 mm" in text


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
        (tmp_path / "OPEN.txt").write_text("R91\nR92\n", encoding="utf-8")
        monkeypatch.setattr(m, "TRUTH_DIR", tmp_path)

        assert m.open_reports() == ("R91", "R92")

    def test_no_open_txt_means_no_open_set(self, monkeypatch, tmp_path):
        monkeypatch.setattr(m, "TRUTH_DIR", tmp_path)

        assert m.open_reports() == ()


class TestRescoringSavedRuns:
    """``--rescore``: saved run files scored again, no model call."""

    def _run_file(self, folder, truth, merged, model, floor):
        import json
        from report_ingest.calc_scoring import score_record
        after = score_record(truth, merged)
        after.record = merged.model_dump(mode="json")
        after.model_record = model.model_dump(mode="json")
        after.floor_record = floor.model_dump(mode="json")
        (folder / "calc").mkdir(parents=True, exist_ok=True)
        (folder / "calc" / f"{truth['id']}.json").write_text(
            json.dumps({"calc_id": truth["id"], "after": after.to_dict()}),
            encoding="utf-8")

    def test_a_folder_is_scored_again_and_merged_again(self, tmp_path):
        from report_ingest.calc_reader import CalcFloor, FloorValue
        from report_ingest.model import Calculation, NamedQuantity, Quantity
        truth = {"id": "settlement__R99_p7", "kind": "settlement",
                 "pages": [7], "program": None,
                 "inputs": [{"name": "Pad width (m)", "value": 2.4,
                             "unit": "m", "page": 7}]}
        floor = CalcFloor(pages=[7], values=[FloorValue(
            name="Pad width (m)", text="B =", page=7,
            bbox=(10.0, 20.0, 90.0, 30.0), method="text", confidence=0.7)])
        model = Calculation(kind="settlement", pages=[7], inputs=[
            NamedQuantity(name="Pad width (m)",
                          value=Quantity(value=2.4, unit="m"))])
        stale = Calculation(kind="settlement", pages=[7], inputs=[
            NamedQuantity(name="Pad width (m)", text="B =")])
        self._run_file(tmp_path / "arm", truth, stale, model,
                       floor.as_calculation())
        (tmp_path / "arm" / "calc" / "other__R98_p1.json").write_text(
            '{"after": {}}', encoding="utf-8")

        again = m.rescore_dirs([truth], [tmp_path / "arm"], remerge=True)
        as_saved = m.rescore_dirs([truth], [tmp_path / "arm"])

        # the synthetic file carries no saved model-alone score: 0
        assert "| settlement__R99_p7 | 2 | 3 | 0 | 3 | 3 |" in again
        assert "| settlement__R99_p7 | 2 | 2 | 0 | 3 | 3 |" in as_saved
        assert "other__R98_p1 | not re-scorable" in again
        assert "(merged again)" in again and "(merged again)" not in as_saved

    def test_remerge_alone_is_refused(self, capsys):
        assert m.main(["--remerge"]) == 1
        assert "--rescore" in capsys.readouterr().err

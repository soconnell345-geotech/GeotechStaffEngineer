"""The WP7 library scorecard runs, and its truth is true.

Two things. The measurement itself runs end to end with no model and no
network, over the synthetic library the fixtures build. And the twenty
questions' EXPECTED answers are checked against the fixture records rather
than trusted: a hand-written truth that has drifted from the records would
otherwise show up as a retrieval failure and be chased in the wrong place.
"""

from __future__ import annotations

import json

import pytest

from module_work.report_ingest_harness.measure_wp7_library import (
    Score, main, measure, questions, report,
)
from report_ingest.library import Library, NARRATIVE_FIELDS
from report_ingest.library_agent import TOOL_NAMES
from report_ingest.tests.library_fixtures import REPORT_IDS, build_library


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    folder = tmp_path_factory.mktemp("wp7")
    build_library(folder)
    lib = Library(str(folder))
    yield lib
    lib.close()


class TestTheQuestions:

    def test_there_are_twenty_with_distinct_ids(self):
        asked = questions()

        assert len(asked) == 20
        assert len({question.id for question in asked}) == 20

    def test_every_query_is_one_the_sub_agent_can_make(self):
        for question in questions():
            assert question.query[0] in TOOL_NAMES

    def test_every_expected_report_is_in_the_library(self, library):
        known = {row["report"] for row in library.list_reports()["reports"]}

        for question in questions():
            assert question.reports <= known, question.id

    def test_every_expected_page_belongs_to_a_report_it_expects(self):
        for question in questions():
            for report_id, page in question.pages:
                assert report_id in question.reports, question.id
                assert isinstance(page, int) and page >= 0

    def test_every_field_a_question_names_is_a_narrative_field(self):
        for question in questions():
            name, arguments = question.query
            if name == "compare":
                assert arguments["field"] in NARRATIVE_FIELDS, question.id
            for field in arguments.get("fields", ()):
                assert field in NARRATIVE_FIELDS, question.id

    def test_the_questions_are_written_as_questions(self):
        for question in questions():
            assert question.question.endswith("?"), question.id
            assert len(question.question.split()) >= 5, question.id


class TestTheMeasurement:

    def test_it_runs_over_the_whole_set(self, library):
        measured = measure(library, questions())

        assert measured["n_questions"] == 20
        assert len(measured["questions"]) == 20

    def test_the_chosen_query_finds_every_report_the_truth_expects(self,
                                                                   library):
        """Recall of 1.000 is the bar: the truth was written off the records.

        Precision is NOT asserted at 1.000 -- two of the questions are
        answered by SEARCH, and a second report that genuinely mentions the
        words is a real false positive of retrieval rather than a defect of
        the truth.
        """
        measured = measure(library, questions())

        assert measured["chosen"]["reports"]["recall"] == 1.0
        assert measured["chosen"]["reports"]["precision"] > 0.9

    def test_the_chosen_query_cites_every_page_the_truth_expects(self,
                                                                 library):
        measured = measure(library, questions())

        assert measured["chosen"]["pages"]["recall"] == 1.0
        assert measured["chosen"]["pages"]["precision"] > 0.9

    def test_no_query_errors(self, library):
        measured = measure(library, questions())

        assert [row["id"] for row in measured["questions"] if row["error"]] \
            == []

    def test_search_alone_is_measured_apart_and_is_worse(self, library):
        """The floor is a floor: it is reported, not asserted as good."""
        measured = measure(library, questions())

        assert measured["n_unsearchable"] >= 1
        assert measured["search"]["reports"]["recall"] > 0.5
        assert measured["search"]["pages"]["precision"] < \
            measured["chosen"]["pages"]["precision"]

    def test_the_report_names_what_the_chosen_query_got_wrong(self, library):
        measured = measure(library, questions())
        text = report(measured, library.library_stats())

        assert "# WP7" in text
        assert "## The chosen query" in text
        assert "## Search alone" in text
        assert text.count("| Q") == 20


class TestScore:

    def test_an_empty_expectation_scores_as_nothing_rather_than_as_perfect(
            self):
        score = Score()
        score.add(set(), set())

        assert score.precision is None and score.recall is None

    def test_precision_and_recall_are_what_they_say(self):
        score = Score()
        score.add({"a", "b", "c"}, {"a", "b", "d"})

        assert score.precision == pytest.approx(2 / 3)
        assert score.recall == pytest.approx(2 / 3)
        assert score.f1 == pytest.approx(2 / 3)


class TestTheCommandLine:

    def test_it_runs_and_prints(self, capsys, tmp_path):
        assert main(["--keep", str(tmp_path / "lib")]) == 0

        out = capsys.readouterr().out
        assert "# WP7" in out and "precision" in out

    def test_only_narrows_the_set(self, capsys, tmp_path):
        assert main(["--only", "Q01", "Q02", "--keep",
                     str(tmp_path / "lib")]) == 0

        assert "2 hand-written question(s)" in capsys.readouterr().out

    def test_an_id_that_is_not_there_is_said_rather_than_ignored(
            self, capsys, tmp_path):
        assert main(["--only", "Q99", "--keep", str(tmp_path / "lib")]) == 2

    def test_the_json_is_written_and_carries_the_numbers(self, tmp_path):
        out = tmp_path / "wp7.json"
        main(["--json", str(out), "--keep", str(tmp_path / "lib")])

        blob = json.loads(out.read_text(encoding="utf-8"))
        assert blob["n_questions"] == 20
        assert blob["chosen"]["reports"]["recall"] == 1.0
        assert blob["library"]["reports"] == 7
        assert "_scores" not in blob


class TestTheExampleQuestionsFile:
    """The shape the cluster half reads, kept readable without the corpus."""

    @pytest.fixture()
    def example(self):
        import pathlib

        path = (pathlib.Path(__file__).parents[1]
                / "library_questions.EXAMPLE.json")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_it_has_five_questions(self, example):
        assert len(example["questions"]) == 5

    def test_every_question_has_an_id_a_question_and_its_reports(self,
                                                                 example):
        for row in example["questions"]:
            assert row["id"] and row["question"].endswith("?")
            assert isinstance(row["reports"], list)

    def test_the_reports_it_names_are_the_synthetic_ones(self, example):
        known = set(REPORT_IDS) | {"L02-bound1"}
        for row in example["questions"]:
            assert set(row["reports"]) <= known, row["id"]
            for report_id, page in row.get("pages", ()):
                assert report_id in known and isinstance(page, int)

    def test_one_of_them_is_deliberately_unanswerable(self, example):
        unanswerable = [row for row in example["questions"]
                        if row.get("must_say_no")]

        assert len(unanswerable) == 1
        assert unanswerable[0]["reports"] == []

    def test_it_says_what_it_is_and_how_it_is_scored(self, example):
        assert "library_questions.json" in example["_what_this_is"]
        assert "citations" in example["_how_it_is_scored"]

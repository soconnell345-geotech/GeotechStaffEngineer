"""A folder of ingested reports, asked questions across all of them.

What the library has to survive: a folder restored from somewhere with no
database beside it, a report re-written after the index was built, a record
that will not load, and a report bound inside another one that has to stay a
report of its own rather than being folded into its parent. Those are the
tests, with one for each query function on the synthetic six.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time

import pytest

from report_ingest.library import (
    NARRATIVE_FIELDS, REVIEW_KINDS, Library, LibraryError,
)
from report_ingest.tests.library_fixtures import build_library


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """One synthetic library, built once: nothing here writes into it."""
    folder = tmp_path_factory.mktemp("library")
    build_library(folder)
    return str(folder)


@pytest.fixture()
def library(root):
    lib = Library(root)
    yield lib
    lib.close()


@pytest.fixture()
def fresh(tmp_path):
    """A library of its own, for the tests that change the folder."""
    folder = tmp_path / "library"
    build_library(folder)
    lib = Library(str(folder))
    yield lib
    lib.close()


class TestTheIndex:

    def test_a_folder_with_no_database_is_indexed_from_the_records(
            self, tmp_path):
        folder = tmp_path / "restored"
        build_library(folder)
        os.remove(folder / "reports.db")

        library = Library(str(folder))
        stats = library.library_stats()
        library.close()

        assert stats["reports"] == 7        # the six, plus the bound child
        assert stats["indexed_chunks"] > 100

    def test_a_second_question_does_not_rebuild(self, fresh):
        fresh.library_stats()

        assert fresh.refresh() is False

    def test_a_record_written_since_the_build_makes_it_stale(self, fresh):
        fresh.library_stats()
        record = os.path.join(fresh.root, "L01", "report.record.json")
        later = time.time() + 10
        os.utime(record, (later, later))

        assert fresh.refresh() is True

    def test_a_force_rebuilds_whatever_the_stamp_says(self, fresh):
        fresh.library_stats()

        assert fresh.refresh(force=True) is True

    def test_the_writers_own_rows_are_rebuilt_too(self, tmp_path):
        """The index is derived: the ``reports`` table comes back as well."""
        folder = tmp_path / "restored"
        build_library(folder)
        os.remove(folder / "reports.db")

        library = Library(str(folder))
        library.library_stats()
        library.close()

        connection = sqlite3.connect(str(folder / "reports.db"))
        rows = connection.execute(
            "SELECT report_id, n_investigations FROM reports "
            "ORDER BY report_id").fetchall()
        connection.close()
        assert ("L01", 2) in rows
        assert ("L05", 1) in rows

    def test_a_record_that_will_not_load_is_a_row_saying_so(self, tmp_path):
        folder = tmp_path / "library"
        build_library(folder)
        (folder / "L06" / "report.record.json").write_text(
            "{not json", encoding="utf-8")

        library = Library(str(folder))
        stats = library.library_stats()
        library.close()

        assert stats["reports"] == 6        # the broken one is not counted
        assert [row["report"] for row in stats["could_not_be_read"]] == ["L06"]

    def test_a_root_that_is_not_a_folder_says_so(self, tmp_path):
        library = Library(str(tmp_path / "nowhere"))

        with pytest.raises(LibraryError):
            library.refresh()

    def test_the_keys_are_the_ones_the_writers_wrote(self, root):
        """A rebuild must not give a report a second identity.

        The library page carries the key the ingest computed off the source
        file's own bytes. Recomputing without the PDF would fall back to the
        record's identity and split one report into two rows on the next
        re-ingest.
        """
        library = Library(root)
        rows = {row["report"]: row["key"]
                for row in library.list_reports()["reports"]}
        library.close()

        page = os.path.join(root, "L01", "report.page.md")
        with open(page, encoding="utf-8") as handle:
            head = handle.read(600)
        assert f'id: "{rows["L01"]}"' in head


class TestABoundReport:

    def test_it_is_a_report_of_the_library_in_its_own_right(self, library):
        rows = {row["report"]: row for row in library.list_reports()["reports"]}

        assert "L02-bound1" in rows
        assert rows["L02-bound1"]["firm"] == "Atlas Ground Engineering"
        assert rows["L02-bound1"]["explorations"] == 1

    def test_it_names_the_report_it_was_bound_inside(self, library):
        rows = {row["report"]: row for row in library.list_reports()["reports"]}

        assert rows["L02-bound1"]["bound_inside"] == "L02"
        assert "bound_inside" not in rows["L02"]

    def test_its_boring_is_its_own_and_not_its_parents(self, library):
        parent = library.explorations("L02")
        child = library.explorations("L02-bound1")

        assert [inv["exploration"] for inv in parent["explorations"]] == \
            ["B-1", "B-2"]
        assert [inv["exploration"] for inv in child["explorations"]] == \
            ["BH-101"]

    def test_the_parent_is_listed_as_holding_one(self, library):
        rows = library.list_reports(has_kind="bound")["reports"]

        assert [row["report"] for row in rows] == ["L02"]


class TestFind:

    def test_a_hit_carries_the_report_and_the_pages(self, library):
        found = library.find("severely corrosive to buried concrete", k=3)

        (first, *_) = found["hits"]
        assert first["report"] == "L05"
        assert first["pages"] == [4]
        assert first["found_by"] == "fts"
        assert "corrosive" in first["snippet"]

    def test_it_searches_the_written_pages_as_well_as_the_records(
            self, library):
        found = library.find("utility corridor", k=8)

        assert {hit["section"] for hit in found["hits"]} & {"page", "summary"}

    def test_a_section_narrows_it(self, library):
        found = library.find("boring", k=6, section="exploration")

        assert {hit["section"] for hit in found["hits"]} == {"exploration"}

    def test_one_report_narrows_it(self, library):
        found = library.find("clay", k=6, report_id="L02")

        assert {hit["report"] for hit in found["hits"]} == {"L02"}

    def test_a_report_that_is_not_there_is_said_so(self, library):
        found = library.find("clay", report_id="L99")

        assert "no report" in found["error"]
        assert found["hits"] == []

    def test_an_empty_query_is_an_answer_not_a_crash(self, library):
        assert library.find("")["error"]

    def test_punctuation_the_index_cannot_parse_does_not_raise(self, library):
        """A question is typed as a question, quotation marks and all."""
        found = library.find('what is the "site class" (and why)?', k=4)

        assert found["hits"]

    def test_the_fuzzy_pass_catches_a_name_spelled_a_second_way(self,
                                                               library):
        pytest.importorskip("rapidfuzz")
        found = library.find("Meridien Geotecnical", k=4)

        assert "L02" in {hit["report"] for hit in found["hits"]}
        assert "fuzzy" in {hit["found_by"] for hit in found["hits"]}

    def test_nothing_matching_comes_back_empty_rather_than_wrong(self,
                                                                library):
        found = library.find("zzzqqxx", k=4)

        assert found["hits"] == []


class TestWhereIs:

    def test_it_answers_with_places(self, library):
        answer = library.where_is("allowable bearing pressure of 3,000 psf",
                                  k=4)

        places = [(row["report"], row["page"]) for row in answer["locations"]]
        assert ("L01", 4) in places

    def test_one_report_and_page_is_listed_once(self, library):
        answer = library.where_is("Site Class", k=8)

        places = [(row["report"], row["page"]) for row in answer["locations"]]
        assert len(places) == len(set(places))


class TestListReports:

    def test_with_no_filter_it_is_the_whole_library(self, library):
        answer = library.list_reports()

        assert answer["n"] == 7
        assert [row["report"] for row in answer["reports"]] == [
            "L01", "L02", "L02-bound1", "L03", "L04", "L05", "L06"]

    @pytest.mark.parametrize("filters,expected", [
        ({"post": "Vale Harbour"}, ["L02", "L02-bound1", "L05"]),
        ({"firm": "Meridian"}, ["L02", "L05"]),
        ({"phase": "Design-build"}, ["L01", "L04"]),
        ({"property_type": "New embassy"}, ["L01"]),
        ({"document_type": "recommendation letter"}, ["L04"]),
        ({"has_kind": "cpt"}, ["L03"]),
        ({"has_kind": "test_pit"}, ["L01", "L04"]),
        ({"has_kind": "calculations"}, ["L01", "L02", "L04"]),
        ({"date_from": "2026-01-01"}, ["L01", "L04"]),
        ({"date_to": "2015-01-01"}, ["L02-bound1"]),
        ({"date_from": "2024-01-01", "date_to": "2025-12-31"},
         ["L02", "L03"]),
    ])
    def test_each_filter(self, library, filters, expected):
        answer = library.list_reports(**filters)

        assert [row["report"] for row in answer["reports"]] == expected

    def test_filters_combine(self, library):
        answer = library.list_reports(post="Vale Harbour", firm="Meridian")

        assert [row["report"] for row in answer["reports"]] == ["L02", "L05"]

    def test_nothing_matching_is_an_empty_list_with_the_filters_named(
            self, library):
        answer = library.list_reports(post="Nowhere At All")

        assert answer["reports"] == []
        assert answer["filters"]["post"] == "Nowhere At All"


class TestFacts:

    def test_the_fields_asked_for_come_back_with_their_pages(self, library):
        answer = library.facts("L02", ["siteClass", "recommendedFoundations"])

        rows = {row["field"]: row for row in answer["fields"]}
        assert rows["siteClass"]["value"] == "Site Class C"
        assert rows["siteClass"]["pages"] == [4]
        assert rows["siteClass"]["quote"] == "Site Class C"
        assert rows["recommendedFoundations"]["value"] == \
            "driven piles; pile-supported mat"

    def test_with_no_fields_it_is_everything_the_report_answered(self,
                                                                library):
        answer = library.facts("L01")

        names = {row["field"] for row in answer["fields"]}
        assert "postName" in names and "liquefactionPotential" in names
        assert names <= set(NARRATIVE_FIELDS)

    def test_a_field_the_report_did_not_answer_is_named_not_returned_empty(
            self, library):
        answer = library.facts("L06", ["siteClass", "postName"])

        assert answer["fields"] == []
        assert answer["not_answered"] == ["siteClass", "postName"]

    def test_a_name_that_is_not_a_field_is_said_so(self, library):
        answer = library.facts("L01", ["siteClass", "soilColour"])

        assert answer["not_a_field"] == ["soilColour"]
        assert [row["field"] for row in answer["fields"]] == ["siteClass"]

    def test_it_carries_the_counts_and_the_qa_summary(self, library):
        answer = library.facts("L02")

        assert answer["counts"]["investigations"] == 2
        assert answer["qa"]["by_kind"] == {"count_mismatch": 1,
                                           "disagreement": 1}
        assert answer["qa"]["needs_review"] == 2

    def test_a_report_that_is_not_there_lists_the_ones_that_are(self,
                                                               library):
        answer = library.facts("L99")

        assert "no report" in answer["error"]
        assert "L01" in answer["reports_in_the_library"]

    def test_a_report_can_be_named_by_its_library_key(self, library):
        key = library.list_reports()["reports"][0]["key"]

        assert library.facts(key)["report"] == "L01"


class TestCompare:

    def test_one_field_across_the_library(self, library):
        answer = library.compare("siteClass")

        rows = {row["report"]: row["value"] for row in answer["rows"]}
        assert rows == {"L01": "Site Class D", "L02": "Site Class C",
                        "L02-bound1": "Site Class D", "L05": "Site Class E"}
        assert dict(answer["rows"][0])["pages"] == [4]

    def test_the_reports_that_did_not_answer_are_named(self, library):
        answer = library.compare("siteClass")

        assert answer["missing"] == ["L03", "L04", "L06"]

    def test_it_can_be_narrowed_to_some_reports(self, library):
        answer = library.compare("liquefactionPotential", ["L01", "L02"])

        assert [row["report"] for row in answer["rows"]] == ["L01", "L02"]

    def test_a_report_that_is_not_there_is_named(self, library):
        answer = library.compare("siteClass", ["L01", "L99"])

        assert answer["not_in_the_library"] == ["L99"]

    def test_a_field_that_is_not_a_field_lists_the_ones_that_are(self,
                                                                library):
        answer = library.compare("soilColour")

        assert "not one of the 37" in answer["error"]
        assert len(answer["fields"]) == 37


class TestExplorations:

    def test_a_boring_comes_with_its_depths_and_its_samples(self, library):
        answer = library.explorations("L01", "boring")

        (boring,) = answer["explorations"]
        assert boring["exploration"] == "B-1"
        assert boring["total_depth"] == "30 ft"
        assert boring["ground_elevation"] == "102.5 ft"
        assert boring["n_layers"] == 2 and boring["n_samples"] == 3
        assert [drive["n"] for drive in boring["spt"]] == [12, 15, 26]
        assert boring["water"] == [{"depth": "12 ft",
                                    "when": "while_drilling"}]
        assert boring["pages"] == [7, 8]

    def test_a_test_pit_carries_its_plan_size(self, library):
        answer = library.explorations("L04", "test_pit")

        assert answer["explorations"][0]["pit"]["width"] == "0.9 m"

    def test_a_sounding_says_how_many_readings_it_holds(self, library):
        cone = library.explorations("L03", "cpt")["explorations"][0]
        probe = library.explorations("L04", "dcp")["explorations"][0]

        assert cone["cpt_points"] == 8
        assert probe["dcp_points"] == 11

    def test_no_kind_is_every_exploration(self, library):
        answer = library.explorations("L04")

        assert answer["n"] == 3

    def test_a_kind_the_report_has_none_of_is_an_empty_answer(self, library):
        answer = library.explorations("L01", "cpt")

        assert answer["explorations"] == [] and answer["n"] == 0

    def test_a_report_that_is_not_there_says_so(self, library):
        assert "no report" in library.explorations("L99")["error"]


class TestLabSummary:

    def test_the_tests_come_with_their_values_and_pages(self, library):
        answer = library.lab_summary("L01", "atterberg")

        (test,) = answer["tests"]
        assert test["exploration"] == "B-1" and test["sample"] == "S-1"
        assert test["standard"] == "ASTM D4318"
        assert test["values"]["ll"] == 38.0
        assert test["pages"] == [12]

    def test_it_counts_the_kinds(self, library):
        answer = library.lab_summary("L01")

        assert answer["by_kind"] == {"atterberg": 1, "gradation": 1,
                                     "summary_table": 1}

    def test_a_report_with_no_laboratory_testing_is_an_empty_answer(self,
                                                                    library):
        assert library.lab_summary("L03")["tests"] == []


class TestCalculations:

    def test_what_the_report_worked_out(self, library):
        answer = library.calculations("L04")

        (calc,) = answer["calculations"]
        assert calc["works_out"] == "slope_stability"
        assert calc["program"] == "SLIDE2 9.0"
        assert calc["method"] == "Bishop simplified"
        assert "FS static: 1.43" in calc["results"]
        assert calc["pages"] == [20, 21]

    def test_a_calculation_that_names_no_program_says_not_stated(self,
                                                                 library):
        (calc,) = library.calculations("L01")["calculations"]

        assert calc["program"] == "not stated"

    def test_a_report_with_none_is_an_empty_answer(self, library):
        assert library.calculations("L03")["calculations"] == []


class TestDisagreements:

    def test_the_whole_library_is_what_a_person_should_look_at(self, library):
        answer = library.disagreements()

        assert answer["reports"] == ["L01", "L02", "L03", "L04"]
        assert {entry["kind"] for entry in answer["entries"]} <= \
            set(REVIEW_KINDS)

    def test_a_note_and_a_skipped_are_not_in_it(self, library):
        kinds = {entry["kind"] for entry in library.disagreements()["entries"]}

        assert "note" not in kinds and "skipped" not in kinds

    def test_a_conflict_carries_both_values_and_both_pages(self, library):
        (entry,) = library.disagreements("L01")["entries"]

        assert entry["kind"] == "conflict"
        assert entry["values"] == ["38", "41"]
        assert entry["pages"] == [12, 14]

    def test_a_report_with_nothing_flagged_is_an_empty_answer(self, library):
        assert library.disagreements("L05")["entries"] == []


class TestLibraryStats:

    def test_what_the_library_holds(self, library):
        stats = library.library_stats()

        assert stats["reports"] == 7
        assert stats["bound_inside_another"] == 1
        assert stats["totals"]["explorations"] == 12
        assert stats["totals"]["calculations"] == 4
        assert stats["years"] == {"first": 2014, "last": 2026}

    def test_it_counts_the_firms_and_the_posts(self, library):
        stats = library.library_stats()

        assert stats["firms"]["Meridian Geotechnical"] == 2
        assert stats["posts"]["Vale Harbour"] == 3

    def test_it_counts_the_kinds_a_report_holds(self, library):
        stats = library.library_stats()

        assert stats["kinds"]["boring"] == 5
        assert stats["kinds"]["bound"] == 1


class TestTheRowCap:

    def test_every_answer_stops_at_max_rows_and_says_it_did(self, root):
        library = Library(root, max_rows=2)
        try:
            listed = library.list_reports()
            compared = library.compare("siteClass")
            flagged = library.disagreements()
        finally:
            library.close()

        assert len(listed["reports"]) == 2 and listed["truncated"] is True
        assert len(compared["rows"]) == 2 and compared["truncated"] is True
        assert len(flagged["entries"]) == 2 and flagged["truncated"] is True


def test_the_library_module_imports_neither_anthropic_nor_langgraph():
    """The query layer stays cheap: it is imported to answer one question.

    ``anthropic`` is optional and not installed on the cluster, and LangGraph
    is imported only when the sub-agent's graph is actually built.
    """
    import subprocess
    import sys

    code = ("import sys; import report_ingest.library, "
            "report_ingest.library_agent; "
            "print('anthropic' in sys.modules, 'langgraph' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "False False"


def test_a_library_can_be_opened_over_a_folder_the_ingest_just_wrote(
        tmp_path):
    """The shape ``IngestSummary.library_root`` points at.

    One report's own output folder IS a library of one, because the database
    sits beside its record. A report read this turn is therefore queryable
    this turn.
    """
    from report_ingest.tests.library_fixtures import record_l01
    from report_ingest.writers import write_outputs

    out = tmp_path / "one_report"
    write_outputs(record_l01(), out, write_diggs_file=False)

    library = Library(str(out))
    try:
        assert library.library_stats()["reports"] == 1
        assert library.facts("L01", ["postName"])["fields"][0]["value"] == \
            "Elmridge"
    finally:
        library.close()


def test_the_index_survives_a_report_being_added(tmp_path):
    from report_ingest.tests.library_fixtures import record_l01
    from report_ingest.writers import write_outputs

    folder = tmp_path / "library"
    build_library(folder)
    library = Library(str(folder))
    try:
        assert library.library_stats()["reports"] == 7
        record = record_l01()
        record.document.report_id = "L07"
        record.general.projectName = "Marlow Point Compound"
        record.general.projectNumber = "26-777"
        write_outputs(record, folder / "L07",
                      db_path=folder / "reports.db", write_diggs_file=False)
        later = time.time() + 10
        os.utime(folder / "L07" / "report.record.json", (later, later))

        assert library.refresh() is True
        assert library.library_stats()["reports"] == 8
        assert library.facts("L07")["report"] == "L07"
    finally:
        library.close()


def test_the_records_are_never_written_back_to(tmp_path):
    """Asking a question changes nothing on disk except the index."""
    folder = tmp_path / "library"
    build_library(folder)
    record = folder / "L01" / "report.record.json"
    before = json.loads(record.read_text(encoding="utf-8"))
    stamp = os.path.getmtime(record)

    library = Library(str(folder))
    try:
        library.find("clay")
        library.facts("L01")
        library.explorations("L01")
    finally:
        library.close()

    assert json.loads(record.read_text(encoding="utf-8")) == before
    assert os.path.getmtime(record) == stamp

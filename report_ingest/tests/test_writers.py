"""The four outputs and the library index, from a synthetic record.

A record goes in and five files come out; these tests are about what is IN
them. The ones that matter most: that the DIGGS file passes both of its gates
and that the verdicts are in the record rather than only in a return value,
that a page's tags are earned by something the record holds, and that
re-ingesting a report updates its library row instead of adding a second one.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from report_ingest.model import (
    BoundReport, DocumentFacts, GeneralFacts, NaturalHazardFacts,
    ParentReport, ReportRecord,
)
from report_ingest.reconciler import reconcile
from report_ingest.tests.record_fixtures import (
    atterberg, boring, build_record, general_facts, gradation, page_context,
    summary_table,
)
from report_ingest.writers import (
    DB_SCHEMA_VERSION, confidence_of, front_matter, key_parameters,
    library_page, open_library, parent_key, record_key, status_of,
    summary_markdown, tags_for, tier_of, write_outputs,
)


@pytest.fixture()
def record():
    return reconcile(build_record())


@pytest.fixture()
def written(record, tmp_path):
    return write_outputs(record, tmp_path)


class TestWhatIsWritten:

    def test_all_five_outputs_land(self, written, tmp_path):
        names = sorted(p.name for p in tmp_path.iterdir())
        assert names == ["report.diggs.xml", "report.page.md",
                         "report.record.json", "report.summary.md",
                         "reports.db"]

    def test_the_record_round_trips_through_its_own_json(self, written):
        blob = json.loads(open(written.record, encoding="utf-8").read())
        again = ReportRecord.model_validate(blob)

        assert again.schema_version == "4.0"
        assert again.general.boringCount == 1
        assert again.investigations[0].total_depth.unit == "ft"
        assert again.lab_tests[0].result.ll == 38.0

    def test_both_diggs_gates_pass_and_say_so_in_the_record(self, record,
                                                            written):
        assert written.schema_ok is True
        assert written.roundtrip_ok is True
        verdicts = {e.where: e for e in record.qa if e.where.startswith("diggs")}
        assert set(verdicts) == {"diggs.schema", "diggs.roundtrip"}
        assert all(e.kind == "note" for e in verdicts.values())

    def test_writing_twice_leaves_one_verdict_not_two(self, record, tmp_path):
        write_outputs(record, tmp_path)
        write_outputs(record, tmp_path)

        assert len([e for e in record.qa if e.where == "diggs.schema"]) == 1

    def test_a_record_with_nothing_to_write_says_so_instead(self, tmp_path):
        record = ReportRecord(general=general_facts())
        written = write_outputs(record, tmp_path)

        assert written.diggs == ""
        assert written.schema_ok is None
        assert any(e.where == "diggs" and "no DIGGS file was written"
                   in e.detail for e in record.qa)
        assert (tmp_path / "report.page.md").is_file()


class TestTheSummaryPage:

    def test_every_answered_question_appears_with_its_pages(self, record):
        text = summary_markdown(record)

        assert "| `boringCount` | 1 | p4 |" in text
        assert "| `siteClass` | Site Class D | p4 |" in text
        assert "## The general questions" in text
        assert "## The natural-hazard questions" in text

    def test_a_question_the_report_did_not_answer_is_not_in_the_table(
            self, record):
        text = summary_markdown(record)

        assert "`postName`" not in text
        assert "`primeContractor`" not in text

    def test_the_bearing_values_are_a_table_of_their_own(self, record):
        text = summary_markdown(record)

        assert "## Bearing recommendations, as values" in text
        assert "| 3000 psf | spread footing |" in text

    def test_the_counts_and_the_qa_list_are_there(self, record, written):
        text = summary_markdown(record)

        assert "## What was extracted" in text
        assert "| investigations | 2 |" in text
        assert "the DIGGS file is valid against the bundled 2.6 schema" in text

    def test_a_record_with_no_qa_says_so(self, tmp_path):
        record = build_record()
        text = summary_markdown(record)

        assert "_Nothing was skipped, partial or in conflict._" in text

    def test_a_pipe_in_an_answer_does_not_break_the_table(self):
        record = build_record(general=GeneralFacts(
            projectName="A | B", documentType="geotechnical report"))
        text = summary_markdown(record)

        assert "A \\| B" in text


class TestTheLibraryPage:

    def test_the_front_matter_carries_the_library_s_fields(self, record):
        meta = front_matter(record, "abc123", source="/tmp/R01.pdf")

        assert meta["id"] == "abc123"
        assert meta["title"] == "Rosewood Terrace Development - " \
                                "geotechnical report"
        assert meta["authors"] == "Soil & Rock Consulting Engineers"
        assert meta["year"] == 2026
        assert meta["doc_type"] == "geotechnical report"
        assert meta["tier"] == "deep"
        assert meta["status"] == "summarized"
        assert meta["confidence"] in ("high", "medium", "low")
        assert meta["original_path"] == "/tmp/R01.pdf"

    def test_the_tags_are_earned_by_something_in_the_record(self, record):
        tags = tags_for(record)

        assert tags["disciplines"] == ["Geotechnical", "Seismic/Earthquake"]
        assert "Site Investigation/Drilling" in tags["topics"]
        assert "Shallow Foundations" in tags["topics"]   # from the foundation
        assert "Liquefaction" in tags["topics"]
        assert "SPT" in tags["methods"]                  # from the drives
        assert "Atterberg Limits" in tags["methods"]
        assert set(tags["materials"]) == {"Clay", "Sand"}  # from the USCS
        assert "ASTM" in tags["standards_referenced"]

    def test_a_record_with_no_hazards_is_not_tagged_seismic(self):
        record = build_record(natural_hazards=NaturalHazardFacts())

        assert tags_for(record)["disciplines"] == ["Geotechnical"]
        assert "Liquefaction" not in tags_for(record)["topics"]

    def test_the_page_carries_the_sections_as_tables(self, record,
                                                      written):
        page = library_page(record, "abc123")

        assert page.startswith("---\n")
        assert "## Key takeaways" in page and "## Key parameters" in page
        assert "## Explorations" in page
        assert "| B-1 | boring | 30 ft |" in page
        assert "## Laboratory testing" in page
        assert "| atterberg | B-1 | 2 ft |" in page
        assert "## Quality assurance" in page

    def test_the_key_parameters_are_the_numbers_a_reader_came_for(self, record):
        parameters = key_parameters(record)

        assert any("3000 psf" in line for line in parameters)
        assert any("Site class D" in line for line in parameters)
        assert any("SPT N from 12 to 26" in line for line in parameters)
        assert any("Groundwater as shallow as 12 ft" in line
                   for line in parameters)

    def test_an_appendix_only_file_is_reference_tier_and_loses_the_bullets(
            self):
        general = general_facts()
        general.documentType = "report appendix or figure(s)"
        record = build_record(general=general)

        assert tier_of(record) == "reference"
        page = library_page(record, "abc123")
        assert "## Key takeaways" not in page
        assert "## Explorations" in page

    def test_status_and_confidence_follow_what_went_wrong(self):
        record = reconcile(build_record(), **page_context())
        assert status_of(record) == "needs_ocr"      # a page had no text, no DI

        record = build_record(general=GeneralFacts(projectName="x"))
        assert confidence_of(record) == "low"        # barely anything answered

        record = reconcile(build_record(lab_tests=[
            atterberg(ll=38.0), summary_table(ll_for_s1=41.0)]))
        assert confidence_of(record) == "medium"     # a conflict


class TestTheLibraryIndex:

    def test_one_row_with_the_page_s_own_fields(self, record, written):
        connection = sqlite3.connect(written.db)
        connection.row_factory = sqlite3.Row
        row = dict(connection.execute("SELECT * FROM reports").fetchone())
        connection.close()

        assert row["id"] == written.key
        assert row["report_id"] == "SYN"
        assert row["year"] == 2026
        assert row["n_investigations"] == 2 and row["n_lab_tests"] == 3
        assert json.loads(row["topics"])
        assert row["record_path"].endswith("report.record.json")
        assert row["schema_version"] == "4.0"

    def test_the_schema_version_is_recorded(self, written):
        connection = sqlite3.connect(written.db)
        value = connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        connection.close()

        assert value[0] == DB_SCHEMA_VERSION

    def test_re_ingesting_updates_the_row_rather_than_adding_one(
            self, record, tmp_path):
        # The same SOURCE FILE, read again after the record changed: one row,
        # updated. The key is the file's, so a re-read of the same PDF can
        # never leave two rows behind.
        pdf = tmp_path / "R01.pdf"
        pdf.write_bytes(b"%PDF-1.7\n" + b"x" * 5000)
        first = write_outputs(record, tmp_path, source=pdf)
        record.general.projectName = "Rosewood Terrace, Phase 2"
        second = write_outputs(record, tmp_path, source=pdf)

        connection = sqlite3.connect(second.db)
        rows = connection.execute("SELECT title FROM reports").fetchall()
        connection.close()
        assert first.key == second.key
        assert len(rows) == 1
        assert "Phase 2" in rows[0][0]

    def test_two_reports_can_share_one_library(self, record, tmp_path):
        write_outputs(record, tmp_path / "a", db_path=tmp_path / "lib.db")
        other = build_record(general=GeneralFacts(
            projectName="Another Site", documentType="report addendum"),
            investigations=[boring("C-1")], lab_tests=[])
        other.document.report_id = "SYN2"
        write_outputs(other, tmp_path / "b", db_path=tmp_path / "lib.db")

        connection = sqlite3.connect(tmp_path / "lib.db")
        rows = connection.execute(
            "SELECT report_id FROM reports ORDER BY report_id").fetchall()
        connection.close()
        assert [r[0] for r in rows] == ["SYN", "SYN2"]


class TestTheKey:

    def test_the_same_file_keys_the_same_report(self, record, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.7\n" + b"x" * 5000)

        assert record_key(record, pdf) == record_key(record, pdf)

    def test_a_different_file_keys_a_different_report(self, record, tmp_path):
        one, two = tmp_path / "a.pdf", tmp_path / "b.pdf"
        one.write_bytes(b"%PDF-1.7\n" + b"x" * 5000)
        two.write_bytes(b"%PDF-1.7\n" + b"y" * 5000)

        assert record_key(record, one) != record_key(record, two)

    def test_with_no_file_the_record_keys_itself(self, record):
        again = build_record()

        assert record_key(record) == record_key(again)
        other = build_record(general=GeneralFacts(projectName="Elsewhere"))
        assert record_key(record) != record_key(other)


class TestReportsBoundInsideThisOne:
    """The parent's own section, and the library row the child gets.

    The record here is hand-built rather than ingested: these are the
    writers, and what they have to get right is that a reader of the summary
    cannot mistake the bound report's explorations for this report's, and
    that two records off ONE file do not collide on one library row.
    """

    def _with_child(self, **over):
        row = BoundReport(
            bound_id="bound1", report_id="SYN.bound1",
            title="Former Owner Site Study", firm="Older Firm & Partners",
            date="11 June 2019", kind="appended_prior_report",
            document_type="geotechnical report",
            pages="15-18", first_page=15, last_page=18, n_pages=4,
            said_by=["planlens", "triage"],
            counts={"investigations": 1, "lab_tests": 2, "samples": 7,
                    "spt": 7, "qa": 4},
            folder="bound/bound1",
            record_path="bound/bound1/report.record.json")
        for name, value in over.items():
            setattr(row, name, value)
        record = build_record()
        record.bound_documents = [row]
        return record

    def _child_record(self):
        record = build_record()
        record.document = DocumentFacts(report_id="SYN.bound1", n_pages=4,
                                        workflow="standard")
        record.parent = ParentReport(
            report_id="SYN", bound_id="bound1", pages="15-18",
            first_page=15, last_page=18, n_pages=4,
            record_path="../../report.record.json")
        return record

    def test_the_section_names_it_and_says_what_it_holds(self):
        text = summary_markdown(self._with_child())

        assert "## Reports bound inside this one" in text
        assert "Former Owner Site Study · Older Firm & Partners · 11 June 2019" \
            in text
        assert "an earlier report, appended whole" in text
        assert "| 15-18 |" in text
        assert "1 exploration, 2 lab tests, 7 samples, 7 driven records" in text
        assert "`bound/bound1`" in text

    def test_it_says_the_counts_above_do_not_include_it(self):
        text = summary_markdown(self._with_child())

        assert "is in none of the counts above" in text

    def test_a_document_that_was_not_read_says_so_instead(self):
        text = summary_markdown(self._with_child(read=False))

        assert "_not read; see the QA section_" in text

    def test_a_record_with_nothing_bound_has_no_section(self):
        assert "Reports bound inside this one" not in \
            summary_markdown(build_record())

    def test_the_childs_own_summary_says_where_it_came_from(self):
        text = summary_markdown(self._child_record())

        assert "bound inside another one (`SYN`)" in text
        assert "at pages 15-18 of that file" in text
        assert "Every page number below is a page of that same file" in text

    def test_the_child_gets_its_own_key_off_the_same_file(self, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.7\n" + b"x" * 5000)
        parent, child = build_record(), self._child_record()

        assert record_key(child, pdf) != record_key(parent, pdf)
        assert parent_key(child, pdf) == record_key(parent, pdf)
        # And it is stable: the same file and the same handle, twice.
        assert record_key(child, pdf) == record_key(child, pdf)

    def test_two_bound_documents_of_one_file_key_apart(self, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.7\n" + b"x" * 5000)
        one, two = self._child_record(), self._child_record()
        two.parent.bound_id = "bound2"

        assert record_key(one, pdf) != record_key(two, pdf)

    def test_the_library_row_points_at_the_parents_row(self, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.7\n" + b"x" * 5000)
        out = tmp_path / "out"
        write_outputs(build_record(), out, source=pdf,
                      db_path=tmp_path / "reports.db")
        write_outputs(self._child_record(), out / "bound" / "bound1",
                      source=pdf, db_path=tmp_path / "reports.db")

        connection = sqlite3.connect(tmp_path / "reports.db")
        rows = {r[0]: (r[1], r[2]) for r in connection.execute(
            "SELECT report_id, id, parent FROM reports")}
        connection.close()
        assert rows["SYN"][1] == ""
        assert rows["SYN.bound1"][1] == rows["SYN"][0]

    def test_an_older_library_grows_the_parent_column(self, tmp_path):
        # A library written before this train has no `parent`; opening it
        # must add the column rather than refuse the row.
        from report_ingest.writers import _SCHEMA

        db = tmp_path / "old.db"
        before = "\n".join(line for line in _SCHEMA.splitlines()
                           if "parent" not in line)
        assert "parent" not in before
        connection = sqlite3.connect(db)
        connection.executescript(before)
        connection.execute("INSERT INTO reports (id, report_id) "
                           "VALUES ('abc', 'R01')")
        connection.commit()
        connection.close()

        connection = open_library(db)
        try:
            names = {r["name"] for r in
                     connection.execute("PRAGMA table_info(reports)")}
            kept = [tuple(row) for row in connection.execute(
                "SELECT report_id, parent FROM reports")]
        finally:
            connection.close()
        assert "parent" in names
        assert kept == [("R01", None)]

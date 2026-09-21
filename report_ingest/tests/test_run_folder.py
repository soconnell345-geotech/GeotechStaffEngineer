"""A folder of reports into one library: resuming, failing, and not doubling.

The folder run is the report-library use, and what it has to survive is a run
of three hundred reports that gets interrupted, a file that will not read, and
being started again the next morning. Those are the tests.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from report_ingest.run_folder import load_di_result, run_folder
from report_ingest.tests.fake_engine import FakeEngine
from report_ingest.tests.narrative_fixtures import build_narrative_report
from report_ingest.tests.test_graph import full_script

pytest.importorskip("planlens.document.roles")


def _variant(text: str) -> bytes:
    """The synthetic report with one more line on its cover: a DIFFERENT file."""
    import fitz

    doc = fitz.open(stream=build_narrative_report().pdf, filetype="pdf")
    doc[0].insert_text((90, 700), text, fontsize=10)
    pdf = doc.tobytes()
    doc.close()
    return pdf


@pytest.fixture()
def folder(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "R01.pdf").write_bytes(_variant("Volume 1"))
    (reports / "R02.pdf").write_bytes(_variant("Volume 2"))
    return reports


def _run(folder, out, **kwargs):
    kwargs.setdefault("log", lambda *_a: None)
    return run_folder(folder, lambda: FakeEngine(full_script()),
                      out_dir=out, **kwargs)


class TestAFolderRun:

    def test_every_report_lands_in_one_library(self, folder, tmp_path):
        run = _run(folder, tmp_path / "out")

        assert [row.report_id for row in run.reports] == ["R01", "R02"]
        assert run.failures == []
        connection = sqlite3.connect(run.db)
        rows = connection.execute(
            "SELECT report_id, n_investigations FROM reports "
            "ORDER BY report_id").fetchall()
        connection.close()
        # FOUR rows from two files: each carries a report bound inside it
        # (the synthetic report's pages 15-18), and a bound document is a
        # record and a library row of its own. Its one boring is ITS one
        # boring -- the two on its parent's row do not include it.
        assert rows == [("R01", 2), ("R01.bound1", 1),
                        ("R02", 2), ("R02.bound1", 1)]

    def test_a_bound_document_points_at_the_report_it_came_out_of(
            self, folder, tmp_path):
        run = _run(folder, tmp_path / "out")

        connection = sqlite3.connect(run.db)
        rows = dict(connection.execute(
            "SELECT report_id, parent FROM reports").fetchall())
        ids = dict(connection.execute(
            "SELECT report_id, id FROM reports").fetchall())
        connection.close()
        assert rows["R01"] == "" and rows["R02"] == ""
        assert rows["R01.bound1"] == ids["R01"]
        assert rows["R02.bound1"] == ids["R02"]
        assert (tmp_path / "out" / "R01" / "bound" / "bound1"
                / "report.record.json").is_file()

    def test_each_report_has_its_own_folder_of_outputs(self, folder,
                                                       tmp_path):
        _run(folder, tmp_path / "out")

        for name in ("R01", "R02"):
            here = tmp_path / "out" / name
            assert (here / "report.record.json").is_file()
            assert (here / "report.page.md").is_file()
            assert (here / "report.diggs.xml").is_file()

    def test_the_index_says_what_the_run_came_to(self, folder, tmp_path):
        _run(folder, tmp_path / "out")

        text = (tmp_path / "out" / "INDEX.md").read_text(encoding="utf-8")
        assert "2 report(s), 0 failed" in text
        assert "| R01 | 22 | standard | 2 | 2 | 1 |" in text
        assert "`Bound in` counts reports bound inside that one" in text
        blob = json.loads(
            (tmp_path / "out" / "index.json").read_text(encoding="utf-8"))
        assert blob["n_reports"] == 2 and blob["n_failed"] == 0

    def test_report_ids_limits_the_run(self, folder, tmp_path):
        run = _run(folder, tmp_path / "out", report_ids=["R02"])

        assert [row.report_id for row in run.reports] == ["R02"]


class TestSurvival:

    def test_a_second_run_skips_what_is_finished(self, folder, tmp_path):
        _run(folder, tmp_path / "out")
        again = run_folder(folder, lambda: FakeEngine([]),
                           out_dir=tmp_path / "out", log=lambda *_a: None)

        assert [row.skipped for row in again.reports] == [True, True]
        assert [row.investigations for row in again.reports] == [2, 2]
        assert again.failures == []

    def test_one_bad_report_does_not_stop_the_folder(self, folder, tmp_path):
        (folder / "R00.pdf").write_bytes(b"not a PDF at all")
        run = _run(folder, tmp_path / "out")

        assert [row.report_id for row in run.reports] == ["R00", "R01", "R02"]
        assert [row.report_id for row in run.failures] == ["R00"]
        assert run.reports[1].ok and run.reports[2].ok
        text = (tmp_path / "out" / "INDEX.md").read_text(encoding="utf-8")
        assert "## What failed" in text and "`R00`" in text

    def test_the_same_file_twice_is_one_document_and_says_so(self, tmp_path):
        reports = tmp_path / "reports"
        reports.mkdir()
        pdf = build_narrative_report().pdf
        (reports / "R01.pdf").write_bytes(pdf)
        (reports / "R01_copy.pdf").write_bytes(pdf)

        run = _run(reports, tmp_path / "out")

        assert run.reports[1].duplicate_of == "R01"
        connection = sqlite3.connect(run.db)
        (count,) = connection.execute(
            "SELECT count(*) FROM reports").fetchone()
        connection.close()
        # Two rows, not four: the file's own record and the one report bound
        # inside it. The copy is the same bytes, so it is the same document
        # and the same two rows, written twice.
        assert count == 2
        text = (tmp_path / "out" / "INDEX.md").read_text(encoding="utf-8")
        assert "## The same document twice" in text


class TestTheDiResults:

    def test_a_missing_folder_is_none_not_an_error(self, tmp_path):
        assert load_di_result(None, "R01") is None
        assert load_di_result(tmp_path, "R01") is None

    def test_an_unreadable_result_is_none(self, tmp_path):
        (tmp_path / "R01.json").write_text("{not json", encoding="utf-8")

        assert load_di_result(tmp_path, "R01") is None

    def test_both_spellings_are_read(self, tmp_path, monkeypatch):
        seen = {}

        class FakeLayout:
            def __init__(self, result):
                seen["result"] = result

        import planlens.document.azure_di as azure
        monkeypatch.setattr(azure, "AzureLayout", FakeLayout)

        (tmp_path / "DI_data_R07.json").write_text(
            json.dumps({"pages": []}), encoding="utf-8")
        assert isinstance(load_di_result(tmp_path, "R07"), FakeLayout)
        assert seen["result"] == {"pages": []}

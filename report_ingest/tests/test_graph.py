"""The whole ingest end to end, on the synthetic report, with a fake engine.

One PDF goes in and five files come out, and no model is involved: every pass
takes an engine and the engine here replays a script. What the tests are for
is the LOOP -- that every work item reaches the right reader, that the
workflow triage chose changes what runs, that a second run over the same
folder costs nothing, and that one item failing does not take the report down
with it.
"""

from __future__ import annotations

import json

import pytest

from report_ingest.graph import Budgets, ingest_report, output_paths
from report_ingest.label_review import LabelChange, ReviewFindings
from report_ingest.lab_reader import LabSheetReading, ReadTest
from report_ingest.lab_reader import ReadProv as LabProv
from report_ingest.log_reader import LogReading, ReadLayer, ReadSample, ReadSPT
from report_ingest.log_reader import ReadProv as LogProv
from report_ingest.model import ReportRecord
from report_ingest.narrative_reader import NarrativeReading, ReadCitation
from report_ingest.tests.fake_engine import FakeEngine
from report_ingest.tests.narrative_fixtures import build_narrative_report
from report_ingest.triage import TriageFindings

pytest.importorskip("planlens.document.roles")


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("corpus") / "SYN.pdf"
    path.write_bytes(build_narrative_report().pdf)
    return path


def triage_turn(workflow: str = "standard", **over):
    data = dict(document_type="geotechnical report", bound_together=[],
                has_narrative=True, has_logs=True, has_lab=True,
                has_calcs=True, languages=["English"],
                toc_agreement="matched", anomalies=[], workflow=workflow,
                rationale="A whole report with narrative, logs and lab data.")
    data.update(over)
    return {"final": TriageFindings(**data)}


def review_turns(changes=()):
    """The review loop: one turn with nothing to look at, then the findings."""
    return [{"text": "The rule labels look right from the ledger."},
            {"final": ReviewFindings(changes=list(changes), structure=[],
                                     unresolved=[], notes="")}]


def narrative_turn():
    return {"final": NarrativeReading(
        documentType="geotechnical report",
        quickSummary="A geotechnical investigation for a residential "
                     "development, recommending spread footings.",
        projectName="Rosewood Terrace Development",
        geotechnicalEngineerFirm="Soil & Rock Consulting Engineers",
        boringCount=4, testPitCount=3,
        recommendedFoundations=["spread footings"],
        bearingCapacity=["3,000 psf allowable for spread footings"],
        siteClass="Site Class D", seismicCodeUsed="ASCE 7-16",
        liquefactionPotential="low", reportDate="14 March 2026",
        citations=[ReadCitation(field="boringCount", page=4,
                                quote="Four borings and three test pits")])}


def log_turn(name: str = "B-1", kind: str = "boring"):
    prov = LogProv(page=7, bbox=(72.0, 90.0, 500.0, 110.0))
    return {"final": LogReading(
        investigation_id=name, kind=kind, depth_unit="m", units_known=True,
        total_depth=31.5,
        layers=[ReadLayer(top=0.0, bottom=31.5, uscs="CL",
                          description="Brown sandy lean CLAY, stiff, moist",
                          prov=prov)],
        samples=[ReadSample(sample_id="1", top=1.5, bottom=3.0, kind="spt",
                            prov=prov)],
        spt=[ReadSPT(depth_top=1.5, depth_bottom=3.0, blows=["3", "5", "7"],
                     n=12, sample_id="1", prov=prov)],
        pages_read=[7])}


def lab_turn(kind: str = "atterberg", page: int = 12):
    prov = LabProv(page=page, bbox=(72.0, 100.0, 400.0, 130.0))
    values = dict(ll=38.0, pl=19.0, pi=19.0) if kind == "atterberg" \
        else dict(fines_percent=54.0)
    return {"final": LabSheetReading(
        depth_unit="m",
        tests=[ReadTest(kind=kind, investigation_id="B-1", sample_id="S-3",
                        depth_top=4.5, prov=prov, **values)],
        pages_read=[page])}


def full_script():
    """Every call one standard run of the synthetic report makes."""
    return ([triage_turn()] + review_turns() + [narrative_turn()]
            + [log_turn("B-1"), log_turn("TP-1", "test_pit")]
            + [lab_turn("atterberg", 12), lab_turn("gradation", 13)])


class TestAStandardRun:

    def test_every_output_is_written_and_the_record_validates(self, pdf,
                                                              tmp_path):
        engine = FakeEngine(full_script())
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN")

        paths = output_paths(tmp_path)
        assert set(paths) == {"record", "summary", "page", "diggs", "db"}
        again = ReportRecord.model_validate(
            json.loads(open(paths["record"], encoding="utf-8").read()))
        assert again.schema_version == record.schema_version
        assert engine.n_calls == len(full_script())

    def test_the_diggs_gates_are_green_and_recorded(self, pdf, tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN")

        verdicts = {e.where: e for e in record.qa
                    if e.where.startswith("diggs")}
        assert verdicts["diggs.schema"].kind == "note"
        assert verdicts["diggs.roundtrip"].kind == "note"

    def test_each_item_reached_its_own_reader(self, pdf, tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN")

        assert [i.investigation_id for i in record.investigations] == \
            ["B-1", "TP-1"]
        assert [i.kind for i in record.investigations] == \
            ["boring", "test_pit"]
        assert sorted(t.kind for t in record.lab_tests) == \
            ["atterberg", "gradation"]
        assert record.general.boringCount == 4
        assert record.natural_hazards.siteClassNormalized == "D"

    def test_the_document_block_carries_the_run(self, pdf, tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN")

        assert record.document.report_id == "SYN"
        assert record.document.n_pages == 22
        assert record.document.workflow == "standard"
        assert record.document.model_calls == len(full_script())
        assert record.document.input_tokens > 0
        assert record.document.seconds >= 0

    def test_the_calc_pages_are_recorded_as_not_read(self, pdf, tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN")

        (entry,) = [e for e in record.qa
                    if e.where == "items.calculation"]
        assert "work package 5" in entry.detail
        assert entry.pages == [20, 21]

    def test_the_appended_report_is_recorded_as_not_read(self, pdf, tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN")

        (entry,) = [e for e in record.qa
                    if e.where == "items.appended_report"]
        assert entry.pages == [15, 16, 17, 18]

    def test_the_project_block_comes_off_the_narrative(self, pdf, tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN")

        assert record.project.name == "Rosewood Terrace Development"

    def test_the_reconciler_ran(self, pdf, tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN")

        # The narrative says four borings and one log was read: a mismatch,
        # recorded rather than corrected.
        (entry,) = [e for e in record.qa if e.kind == "count_mismatch"
                    and e.where == "general.boringCount"]
        assert entry.values == ["narrative 4", "found 1"]
        assert record.narrative.found_counts["borings"] == 1


class TestResuming:

    def test_a_second_run_over_the_same_folder_costs_nothing(self, pdf,
                                                             tmp_path):
        first = ingest_report(pdf, FakeEngine(full_script()),
                              out_dir=tmp_path, report_id="SYN")
        empty = FakeEngine([])
        second = ingest_report(pdf, empty, out_dir=tmp_path, report_id="SYN")

        assert empty.n_calls == 0
        assert [i.investigation_id for i in second.investigations] == \
            [i.investigation_id for i in first.investigations]
        assert second.general.boringCount == first.general.boringCount

    def test_resume_false_reads_it_all_again(self, pdf, tmp_path):
        ingest_report(pdf, FakeEngine(full_script()), out_dir=tmp_path,
                      report_id="SYN")
        engine = FakeEngine(full_script())
        ingest_report(pdf, engine, out_dir=tmp_path, report_id="SYN",
                      resume=False)

        assert engine.n_calls == len(full_script())

    def test_every_item_left_its_own_file(self, pdf, tmp_path):
        ingest_report(pdf, FakeEngine(full_script()), out_dir=tmp_path,
                      report_id="SYN")

        files = sorted(p.name for p in (tmp_path / "items").iterdir())
        assert len(files) == 5               # narrative, two logs, two sheets
        assert (tmp_path / "triage.json").is_file()
        assert (tmp_path / "review.json").is_file()


class TestTheWorkflows:

    def test_needs_person_stops_after_triage(self, pdf, tmp_path):
        engine = FakeEngine([triage_turn("needs_person")])
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN")

        assert engine.n_calls == 1
        assert record.investigations == [] and record.lab_tests == []
        (entry,) = [e for e in record.qa if e.where == "workflow"]
        assert "nothing was read" in entry.detail
        assert record.document.workflow == "needs_person"

    def test_appendix_only_skips_the_narrative_reader(self, pdf, tmp_path):
        script = ([triage_turn("appendix_only")] + review_turns()
                  + [log_turn("B-1"), log_turn("TP-1", "test_pit"),
                     lab_turn("atterberg", 12), lab_turn("gradation", 13)])
        engine = FakeEngine(script)
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN")

        assert engine.n_calls == len(script)
        assert record.general.answered() == []
        (entry,) = [e for e in record.qa if e.where == "items.narrative"]
        assert "appendix or figure material" in entry.detail
        assert len(record.investigations) == 2

    def test_a_flagged_anomaly_becomes_a_qa_note(self, pdf, tmp_path):
        script = ([triage_turn(anomalies=["the contents list names an "
                                          "appendix E that is not here"])]
                  + review_turns() + [narrative_turn(), log_turn("B-1"),
                                      log_turn("TP-1", "test_pit"),
                                      lab_turn("atterberg", 12),
                                      lab_turn("gradation", 13)])
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN")

        assert any(e.where == "triage" and "appendix E" in e.detail
                   for e in record.qa)

    def test_bound_together_documents_are_flagged(self, pdf, tmp_path):
        script = ([triage_turn(
            "multi_document",
            bound_together=[{"kind": "appended_prior_report",
                             "pages": "15-18", "title": "Former Owner Study"}])]
            + review_turns() + [narrative_turn(), log_turn("B-1"),
                                log_turn("TP-1", "test_pit"),
                                lab_turn("atterberg", 12),
                                lab_turn("gradation", 13)])
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN")

        (entry,) = [e for e in record.qa
                    if e.where == "triage.bound_together"]
        assert "15-18" in entry.values[0]


class TestTheBudgets:

    def test_the_passes_can_be_turned_off(self, pdf, tmp_path):
        script = [narrative_turn(), log_turn("B-1"),
                  log_turn("TP-1", "test_pit"), lab_turn("atterberg", 12),
                  lab_turn("gradation", 13)]
        engine = FakeEngine(script)
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN",
                               budgets=Budgets(triage=False, review=False))

        assert engine.n_calls == len(script)
        assert record.document.workflow == "standard"
        assert not (tmp_path / "triage.json").exists()

    def test_max_items_stops_the_run_early(self, pdf, tmp_path):
        engine = FakeEngine([triage_turn()] + review_turns()
                            + [narrative_turn(), log_turn("B-1")])
        record = ingest_report(pdf, engine, out_dir=tmp_path, report_id="SYN",
                               budgets=Budgets(max_items=5))

        assert len(record.investigations) == 1
        assert record.lab_tests == []

    def test_a_reader_that_fails_does_not_take_the_report_down(self, pdf,
                                                               tmp_path):
        # The first log's turn is prose rather than a reading, so the log
        # reader raises; everything after it must still be read.
        script = ([triage_turn()] + review_turns() + [narrative_turn()]
                  + [{"text": "I cannot read this page."},
                     log_turn("TP-1", "test_pit"),
                     lab_turn("atterberg", 12), lab_turn("gradation", 13)])
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN",
                               budgets=Budgets(log=1))

        assert [i.investigation_id for i in record.investigations] == ["TP-1"]
        assert len(record.lab_tests) == 2
        assert any("could not be read" in e.detail for e in record.qa)


class TestTheLabelReviewChangesWhatIsRead:

    def test_a_relabelled_page_changes_the_work_items(self, pdf, tmp_path):
        # The review moves the photograph page into the laboratory appendix;
        # a third laboratory sheet then has to be read.
        script = ([triage_turn()]
                  + review_turns([LabelChange(
                      page=10, from_label="photos", to_label="lab_test",
                      reason="the page is a results sheet, not photographs",
                      evidence="render_page")])
                  + [narrative_turn(), log_turn("B-1"),
                     log_turn("TP-1", "test_pit"),
                     lab_turn("moisture_content", 10),
                     lab_turn("atterberg", 12), lab_turn("gradation", 13)])
        engine = FakeEngine(script)
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN")

        assert engine.n_calls == len(script)
        assert len(record.lab_tests) == 3

"""The whole ingest end to end, on the synthetic report, with a fake engine.

One PDF goes in and five files come out, and no model is involved: every pass
takes an engine and the engine here replays a script. What the tests are for
is the LOOP -- that every work item reaches the right reader, that the
workflow triage chose changes what runs, that a second run over the same
folder costs nothing, and that one item failing does not take the report down
with it.
"""

from __future__ import annotations

import functools
import json

import pytest

from report_ingest.bound import BoundIdentity
from report_ingest.graph import Budgets, ingest_report, output_paths
from report_ingest.label_review import LabelChange, ReviewFindings
from report_ingest.calc_reader import CalcReading, ReadValue
from report_ingest.calc_reader import ReadProv as CalcProv
from report_ingest.lab_reader import LabSheetReading, ReadTest
from report_ingest.lab_reader import ReadProv as LabProv
from report_ingest.log_reader import LogReading, ReadLayer, ReadSample, ReadSPT
from report_ingest.log_reader import ReadProv as LogProv
from report_ingest.model import ReportRecord
from report_ingest.narrative_reader import NarrativeReading, ReadCitation
from report_ingest.tests.fake_engine import FakeEngine
from report_ingest.tests.narrative_fixtures import build_narrative_report
from report_ingest.triage import TriageFindings
from report_ingest.vision_labels import (
    SHEET_PAGES, VisionPageAnswer, VisionSheetAnswer,
)

pytest.importorskip("planlens.document.roles")


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("corpus") / "SYN.pdf"
    path.write_bytes(build_narrative_report().pdf)
    return path


@functools.lru_cache(maxsize=1)
def rules_labels():
    """What planlens' rules call each page of the synthetic report.

    The vision voter's script is built from these, so a test can say "the
    two voters agree" or "they split on page 10" without hard-coding
    eighteen labels that planlens is free to improve.
    """
    from planlens.document import open_document
    from planlens.document.roles import page_roles

    doc = open_document(build_narrative_report().pdf, name="SYN")
    try:
        return {r.page: r.role for r in page_roles(doc)}
    finally:
        doc.close()


def _rules_confidence(page: int) -> float:
    """planlens' own confidence for one page of the synthetic report."""
    from planlens.document import open_document
    from planlens.document.roles import page_roles

    doc = open_document(build_narrative_report().pdf, name="SYN")
    try:
        return next(r.confidence for r in page_roles(doc) if r.page == page)
    finally:
        doc.close()


def vision_turns(overrides=None, confidence: float = 0.8):
    """The vision voter's replies: one per contact sheet of six pages.

    With no overrides it agrees with the rules on every page, which is the
    ordinary case and the one where the expensive review never runs.
    """
    labels = dict(rules_labels())
    labels.update(overrides or {})
    pages = sorted(labels)
    turns = []
    for start in range(0, len(pages), SHEET_PAGES):
        chunk = pages[start:start + SHEET_PAGES]
        turns.append({"final": VisionSheetAnswer(pages=[
            VisionPageAnswer(page=page, label=labels[page],
                             confidence=confidence,
                             reason="what the page looks like")
            for page in chunk])})
    return turns


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


def calc_turn(page: int = 20):
    prov = CalcProv(page=page, bbox=(72.0, 30.0, 400.0, 60.0))
    return {"final": CalcReading(
        kind="lateral_pile", program="LPILE", program_version="2022",
        method="p-y analysis", subject="test pile",
        inputs=[ReadValue(name="Pile-head load", value=120.0, unit="kN",
                          prov=prov)],
        results=[ReadValue(name="Maximum moment", value=202.95, unit="kN",
                           prov=prov),
                 ReadValue(name="Maximum deflection", value=1.155, unit="mm",
                           prov=prov)],
        summary="A lateral pile analysis printing moment, shear and "
                "deflection against depth.",
        pages_read=[page])}


def reader_turns():
    """The six readers a standard run of the synthetic report calls."""
    return ([narrative_turn()]
            + [log_turn("B-1"), log_turn("TP-1", "test_pit")]
            + [lab_turn("atterberg", 12), lab_turn("gradation", 13)]
            + [calc_turn()])


def identity_turn(**over):
    """The one call a bound document's own front matter costs.

    planlens' rules call pages 15-18 of the synthetic report a nested
    document all by themselves, so EVERY default run of it finds one bound
    report whether or not triage mentions one. That is the production
    behaviour and these scripts carry it.
    """
    data = dict(title="Former Owner Site Study", firm="Older Firm & Partners",
                date="11 June 2019", document_type="geotechnical report",
                kind="appended_prior_report", same_site="yes")
    data.update(over)
    return {"final": BoundIdentity(**data)}


def bound_narrative_turn():
    """The bound report's OWN narrative: a different project, on purpose.

    Nothing of it may reach the parent's record and nothing of the parent's
    may reach its, so every field a test looks at is spelled differently
    from :func:`narrative_turn`.
    """
    return {"final": NarrativeReading(
        documentType="geotechnical report",
        quickSummary="An earlier investigation of the same site by another "
                     "firm, reproduced in this report as an appendix.",
        projectName="Rosewood Terrace Preliminary Study",
        geotechnicalEngineerFirm="Older Firm & Partners",
        boringCount=1, reportDate="11 June 2019",
        citations=[ReadCitation(field="boringCount", page=16,
                                quote="One boring was advanced")])}


def bound_reader_turns():
    """The two readers the bound report's own pages call: prose and BH-1."""
    return [bound_narrative_turn(), log_turn("BH-1")]


def main_script():
    """What the READER tier is asked for in a standard run.

    The cluster stage gives the vision voter its own engine on a cheaper
    tier, so the two scripts are separate there: this one and
    :func:`vision_turns`.
    """
    return ([triage_turn()] + [identity_turn()] + reader_turns()
            + bound_reader_turns())


def full_script():
    """Every call one standard run of the synthetic report makes.

    Triage, the vision voter agreeing with the rules on every page, the one
    identity call for the report bound in at pages 15-18, the five readers,
    and that bound report's own two. NO LABEL REVIEW: the voters agreed
    everywhere, and in the default ``review_mode="disagreements"`` a report
    with no split pages never pays for the review at all.
    """
    return ([triage_turn()] + vision_turns() + [identity_turn()]
            + reader_turns() + bound_reader_turns())


def unbound_script():
    """The same run with ``ingest_bound=False``: no child, no identity call."""
    return [triage_turn()] + vision_turns() + reader_turns()


def rules_only_script():
    """The same run with ``label_policy="rules"``: no vision, one review."""
    return ([triage_turn()] + review_turns() + [identity_turn()]
            + reader_turns() + bound_reader_turns())


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

    def test_the_calculation_item_reaches_the_calculation_reader(self, pdf,
                                                                 tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN")

        (calc,) = record.calculations
        assert calc.kind == "lateral_pile"
        assert calc.program == "LPILE 2022"
        assert calc.pages == [20, 21]
        assert calc.result("Maximum moment").value.value == 202.95
        assert not [e for e in record.qa if e.where == "items.calculation"]
        assert record.counts()["calculations"] == 1

    def test_the_calculations_reach_the_summary_and_the_library_page(
            self, pdf, tmp_path):
        ingest_report(pdf, FakeEngine(full_script()), out_dir=tmp_path,
                      report_id="SYN")

        summary = (tmp_path / "report.summary.md").read_text(encoding="utf-8")
        page = (tmp_path / "report.page.md").read_text(encoding="utf-8")
        assert "## Calculations" in summary and "## Calculations" in page
        assert "lateral pile" in summary and "LPILE 2022" in summary
        assert "Maximum moment 202.95 kN" in summary

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
        # narrative, two logs, two sheets, one calculation
        assert len(files) == 6
        assert (tmp_path / "triage.json").is_file()
        assert (tmp_path / "vision.json").is_file()
        assert (tmp_path / "labels.json").is_file()
        # The voters agreed on every page, so nobody paid for the review.
        assert not (tmp_path / "review.json").exists()


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

    def test_a_workflow_spelt_another_way_is_still_acted_on(self, pdf,
                                                            tmp_path):
        # RI-20: the graph branches on the exact string, so "Needs a person"
        # used to run every reader with nothing said.
        engine = FakeEngine([triage_turn("Needs a person")])
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN")

        assert engine.n_calls == 1
        assert record.document.workflow == "needs_person"

    def test_a_workflow_off_the_list_runs_standard_and_says_so(self, pdf,
                                                               tmp_path):
        script = full_script()
        script[0] = triage_turn("unsure")
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN")

        assert record.document.workflow == "standard"
        (entry,) = [e for e in record.qa if e.where == "workflow"]
        assert entry.kind == "note" and "'unsure'" in entry.detail
        # and a resumed run reads the same answer back off triage.json
        again = ingest_report(pdf, FakeEngine([]), out_dir=tmp_path,
                              report_id="SYN")
        assert again.document.workflow == "standard"
        assert [e.detail for e in again.qa if e.where == "workflow"] == \
            [entry.detail]

    def test_appendix_only_skips_the_narrative_reader(self, pdf, tmp_path):
        script = ([triage_turn("appendix_only")] + vision_turns()
                  + [identity_turn()]
                  + [log_turn("B-1"), log_turn("TP-1", "test_pit"),
                     lab_turn("atterberg", 12), lab_turn("gradation", 13),
                     calc_turn()]
                  + bound_reader_turns())
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
                  + vision_turns() + [identity_turn()] + reader_turns()
                  + bound_reader_turns())
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN")

        assert any(e.where == "triage" and "appendix E" in e.detail
                   for e in record.qa)

    def test_bound_together_documents_are_flagged(self, pdf, tmp_path):
        script = ([triage_turn(
            "multi_document",
            bound_together=[{"kind": "appended_prior_report",
                             "pages": "15-18", "title": "Former Owner Study"}])]
            + vision_turns() + [identity_turn()] + reader_turns()
            + bound_reader_turns())
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN")

        (entry,) = [e for e in record.qa
                    if e.where == "triage.bound_together"]
        assert "15-18" in entry.values[0]


class TestTheBudgets:

    def test_the_passes_can_be_turned_off(self, pdf, tmp_path):
        script = ([identity_turn()] + reader_turns()
                  + bound_reader_turns())
        engine = FakeEngine(script)
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN", label_policy="rules",
                               budgets=Budgets(triage=False, review=False))

        assert engine.n_calls == len(script)
        assert record.document.workflow == "standard"
        assert record.document.label_policy == "rules"
        assert record.document.review_mode == "none"
        assert not (tmp_path / "triage.json").exists()
        assert not (tmp_path / "vision.json").exists()

    def test_rules_only_reproduces_the_old_label_path(self, pdf, tmp_path):
        """``label_policy="rules"`` with the review over the whole report."""
        script = rules_only_script()
        engine = FakeEngine(script)
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN", label_policy="rules",
                               review_mode="all")

        assert engine.n_calls == len(script)
        assert not (tmp_path / "vision.json").exists()
        assert (tmp_path / "review.json").is_file()
        assert record.document.label_split_pages == 0
        assert [v.voter for v in record.page_labels[0].voters] == ["rules"]

    def test_max_items_stops_the_run_early(self, pdf, tmp_path):
        engine = FakeEngine([triage_turn()] + vision_turns()
                            + [identity_turn()]
                            + [narrative_turn(), log_turn("B-1")]
                            + bound_reader_turns())
        record = ingest_report(pdf, engine, out_dir=tmp_path, report_id="SYN",
                               budgets=Budgets(max_items=5))

        assert len(record.investigations) == 1
        assert record.lab_tests == []

    def test_a_reader_that_fails_does_not_take_the_report_down(self, pdf,
                                                               tmp_path):
        # The first log's turn is prose rather than a reading, so the log
        # reader raises; everything after it must still be read.
        script = ([triage_turn()] + vision_turns() + [identity_turn()]
                  + [narrative_turn()]
                  + [{"text": "I cannot read this page."},
                     log_turn("TP-1", "test_pit"),
                     lab_turn("atterberg", 12), lab_turn("gradation", 13)]
                  + bound_reader_turns())
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN",
                               budgets=Budgets(log=1))

        assert [i.investigation_id for i in record.investigations] == ["TP-1"]
        assert len(record.lab_tests) == 2
        assert any("could not be read" in e.detail for e in record.qa)


class TestTheLabelReviewChangesWhatIsRead:

    def test_a_relabelled_page_changes_the_work_items(self, pdf, tmp_path):
        # The voters SPLIT on page 10 -- the rules call it photos, the
        # picture calls it a figure -- so the review is shown it, and moves
        # it into the laboratory appendix; a third sheet then has to be read.
        script = ([triage_turn()] + vision_turns({10: "figure"})
                  + review_turns([LabelChange(
                      page=10, from_label="figure", to_label="lab_test",
                      reason="the page is a results sheet, not photographs",
                      evidence="render_page")])
                  + [identity_turn()]
                  + [narrative_turn(), log_turn("B-1"),
                     log_turn("TP-1", "test_pit"),
                     lab_turn("moisture_content", 10),
                     lab_turn("atterberg", 12), lab_turn("gradation", 13),
                     calc_turn()]
                  + bound_reader_turns())
        engine = FakeEngine(script)
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN")

        assert engine.n_calls == len(script)
        assert len(record.lab_tests) == 3
        page = next(p for p in record.page_labels if p.page == 10)
        assert page.label == "lab_test" and page.settled_by == "review"


class TestThePageVote:
    """Three cheap voters label every page, and the record says who said what."""

    def test_the_record_carries_a_label_per_page_with_its_voters(self, pdf,
                                                                 tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN")

        assert len(record.page_labels) == 22
        page = record.page_labels[7]
        assert page.model_dump() == {
            "page": 7,
            "label": rules_labels()[7],
            "confidence": page.confidence,
            "agreed": True,
            "policy": "structural",
            "settled_by": "vote",
            "voters": [
                {"voter": "rules", "label": rules_labels()[7],
                 "confidence": page.voters[0].confidence, "family": ""},
                {"voter": "vision", "label": rules_labels()[7],
                 "confidence": 0.8, "family": ""},
            ],
        }
        assert 0.0 < page.confidence <= 1.0

    def test_the_document_block_says_how_the_labels_were_settled(self, pdf,
                                                                 tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN")

        assert record.document.label_policy == "structural"
        assert record.document.review_mode == "disagreements"
        assert record.document.label_split_pages == 0
        assert record.document.review_changed == 0

    def test_a_split_the_review_does_not_settle_becomes_a_qa_entry(
            self, pdf, tmp_path):
        # The voters split on page 10 and the review changes nothing.
        script = ([triage_turn()] + vision_turns({10: "figure"})
                  + review_turns() + reader_turns())
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN")

        (entry,) = [e for e in record.qa if e.kind == "label_disagreement"]
        assert entry.pages == [10]
        assert entry.values == [
            f"rules {rules_labels()[10]} (%.2f)" % _rules_confidence(10),
            "vision figure (0.80)"]
        assert "did not agree" in entry.detail
        page = next(p for p in record.page_labels if p.page == 10)
        assert page.agreed is False and page.label == "figure"

    def test_a_review_that_cannot_answer_does_not_fail_the_report(
            self, pdf, tmp_path):
        """Foundry, 2026-10-04: a long report's review ran out of output
        room (stop_reason 'length') and R21 failed whole. Now the review is
        asked again and, when nothing comes back, the vote's labels stand
        and the QA says why the splits were not settled."""
        out_of_room = {"text": "", "stop_reason": "length"}
        script = ([triage_turn()] + vision_turns({10: "figure"})
                  + [{"text": "looked"}, out_of_room, out_of_room]
                  + reader_turns())
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN")

        (note,) = [e for e in record.qa if e.where == "labels.review"]
        assert note.kind == "partial" and "'length'" in note.detail
        assert 10 in note.pages
        (entry,) = [e for e in record.qa if e.kind == "label_disagreement"]
        assert entry.pages == [10]
        page = next(p for p in record.page_labels if p.page == 10)
        assert page.label == "figure" and page.agreed is False
        # nothing is saved, so a resumed run asks again
        assert not (tmp_path / "review.json").exists()
        assert record.investigations          # the readers still ran

    def test_the_vote_can_change_a_label_with_no_review_at_all(self, pdf,
                                                              tmp_path):
        script = [triage_turn()] + vision_turns({10: "figure"}) \
            + reader_turns()
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN", review_mode="none")

        page = next(p for p in record.page_labels if p.page == 10)
        assert page.label == "figure" and page.settled_by == "vote"
        assert record.document.review_mode == "none"
        assert not (tmp_path / "review.json").exists()
        # The split is still recorded, so a reviewer of the record sees it.
        assert record.document.label_split_pages == 1
        assert any(e.kind == "label_disagreement" for e in record.qa)


class TestWhatTheReviewIsShown:

    def _brief(self, engine):
        """The words the review's first call carried."""
        for call in engine.calls:
            if call["tools"] and "read_page" in call["tools"]:
                return call["messages"][0]["content"][0]["text"]
        raise AssertionError("the review never ran")

    def test_disagreements_mode_sends_the_split_pages_and_their_neighbours(
            self, pdf, tmp_path):
        script = ([triage_turn()] + vision_turns({10: "figure"})
                  + review_turns() + reader_turns())
        engine = FakeEngine(script)
        ingest_report(pdf, engine, out_dir=tmp_path, report_id="SYN")

        brief = self._brief(engine)
        head, ledger = brief.split("THE PAGE LEDGER:")
        assert sorted({int(line[1:4]) for line in ledger.splitlines()
                       if line.startswith("p0")}) == [8, 9, 10, 11, 12]
        assert "you are NOT being asked about all of them" in head
        assert "THE PAGES THE LABELLERS SPLIT ON" in head
        # The split page is named, with what each voter said about it.
        (line,) = [row for row in head.splitlines() if row.startswith("p010")]
        assert "vision says figure" in line and "rules says" in line

    def test_the_budget_scales_with_the_split_and_not_the_report(self, pdf,
                                                                 tmp_path):
        from report_ingest.label_review import budget_for, budget_for_split

        script = ([triage_turn()] + vision_turns({10: "figure"})
                  + review_turns() + reader_turns())
        engine = FakeEngine(script)
        ingest_report(pdf, engine, out_dir=tmp_path, report_id="SYN")

        saved = json.loads((tmp_path / "review.json")
                           .read_text(encoding="utf-8"))
        assert saved["budget"] == budget_for_split(5) == 20
        assert saved["budget"] < budget_for(22)
        assert saved["asked_pages"] == [8, 9, 10, 11, 12]

    def test_all_mode_shows_the_whole_report_and_the_old_budget(self, pdf,
                                                               tmp_path):
        from report_ingest.label_review import budget_for

        script = [triage_turn()] + vision_turns() + review_turns() \
            + reader_turns()
        engine = FakeEngine(script)
        ingest_report(pdf, engine, out_dir=tmp_path, report_id="SYN",
                      review_mode="all")

        brief = self._brief(engine)
        assert "NOT being asked about all of them" not in brief
        saved = json.loads((tmp_path / "review.json")
                           .read_text(encoding="utf-8"))
        assert saved["budget"] == budget_for(22) == 60
        assert saved["asked_pages"] == []

    def test_a_change_on_a_page_the_voters_agreed_on_is_applied_and_flagged(
            self, pdf, tmp_path):
        """The agent wandered. It may be right, so the change stands -- loudly."""
        script = ([triage_turn()] + vision_turns({10: "figure"})
                  + review_turns([LabelChange(
                      page=3, from_label="narrative", to_label="toc",
                      reason="it is a contents list",
                      evidence="read_page")])
                  + reader_turns())
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN")

        page = next(p for p in record.page_labels if p.page == 3)
        assert page.label == "toc" and page.settled_by == "review"
        (note,) = [e for e in record.qa
                   if e.where == "labels.page3" and e.kind == "note"]
        assert "not one of the pages the review was asked about" in note.detail


class TestTheTrustPolicy:

    def test_without_a_table_it_falls_back_to_structural_and_says_so(
            self, pdf, tmp_path, capsys):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN",
                               label_policy="trust")

        assert record.document.label_policy == "structural"
        printed = capsys.readouterr().out
        assert "falls back to 'structural'" in printed
        saved = json.loads((tmp_path / "labels.json")
                           .read_text(encoding="utf-8"))
        assert "trust" in saved["note"] and saved["policy"] == "structural"

    def test_with_a_table_it_is_the_policy_that_ran(self, pdf, tmp_path):
        table = tmp_path / "trust_table.json"
        table.write_text(json.dumps({"learned_on": ["R36"], "table": {
            "narrative": {"pages": 9, "rules": 8, "vision": 1,
                          "winner": "rules"}}}), encoding="utf-8")
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path / "run", report_id="SYN",
                               label_policy="trust", trust_table=table)

        assert record.document.label_policy == "trust"
        assert all(p.policy == "trust" for p in record.page_labels)


class TestTheRunFile:

    def test_labels_json_records_the_mode_the_splits_and_the_cost(self, pdf,
                                                                  tmp_path):
        script = ([triage_turn()] + vision_turns({10: "figure"})
                  + review_turns([LabelChange(
                      page=10, from_label="figure", to_label="lab_test",
                      reason="a results sheet", evidence="render_page")])
                  + [identity_turn()]
                  + [narrative_turn(), log_turn("B-1"),
                     log_turn("TP-1", "test_pit"),
                     lab_turn("moisture_content", 10),
                     lab_turn("atterberg", 12), lab_turn("gradation", 13),
                     calc_turn()]
                  + bound_reader_turns())
        ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                      report_id="SYN")

        saved = json.loads((tmp_path / "labels.json")
                           .read_text(encoding="utf-8"))
        assert saved["policy"] == "structural"
        assert saved["review_mode"] == "disagreements"
        assert saved["vision_mode"] == "sheet"
        assert saved["split_pages"] == [10] and saved["n_split"] == 1
        assert saved["n_agreed"] == 21
        assert saved["review_changed"] == [10]
        assert saved["n_review_changed_off_split"] == 0
        assert saved["labels"]["10"] == "lab_test"
        assert saved["cost"]["vision_paid"] is True
        assert saved["cost"]["review_paid"] is True
        assert saved["cost"]["vision"]["calls"] == 4      # 22 pages, 6 a sheet
        assert saved["cost"]["review"]["calls"] == 2

    def test_a_resumed_run_keeps_the_numbers_and_pays_nothing(self, pdf,
                                                              tmp_path):
        ingest_report(pdf, FakeEngine(full_script()), out_dir=tmp_path,
                      report_id="SYN")
        empty = FakeEngine([])
        ingest_report(pdf, empty, out_dir=tmp_path, report_id="SYN")

        assert empty.n_calls == 0
        saved = json.loads((tmp_path / "labels.json")
                           .read_text(encoding="utf-8"))
        assert saved["cost"]["vision_paid"] is False
        assert saved["cost"]["vision"]["calls"] == 4


class TestASavedReviewIsHeldToTheSameRule:
    """A review.json saved before BACKLOG RI-15 may hold changes made on the
    ledger alone; a resumed run applies only what a fresh one would."""

    def test_a_saved_change_that_names_no_look_is_left_out(self):
        from report_ingest.graph import _review_changes

        blob = {"changes": [
            {"page": 3, "from": "other", "to": "figure", "evidence": "ledger"},
            {"page": 4, "from": "other", "to": "plan",
             "evidence": "render_page"},
            {"page": 5, "from": "other", "to": "toc", "evidence": "outline"},
            {"page": 6, "from": "other", "to": "photos"},
            {"page": 7, "from": "other", "to": "plan", "evidence": ""},
        ]}
        assert [r["page"] for r in _review_changes(blob, [])] == [4]

    def test_a_reused_label_run_is_not_second_guessed(self):
        from report_ingest.graph import _review_changes
        from report_ingest.label_vote import combine

        choices = [combine("other", "figure", page=3)]
        rows = _review_changes({"final_labels": {"3": "figure"}}, choices)
        assert [(r["page"], r["evidence"]) for r in rows] == [(3, "reused")]


class TestAReportBoundInsideAReport:
    """pages 15-18 of the synthetic report are another firm's whole study.

    Before this train they were four pages listed in a QA entry and never
    read. Now they are a record of their own, and what every test here is
    really checking is that NOTHING crosses the boundary in either
    direction.
    """

    def test_the_parent_lists_the_child_with_the_right_pages(self, pdf,
                                                             tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN")

        (child,) = record.bound_documents
        assert child.bound_id == "bound1"
        assert child.pages == "15-18"
        assert (child.first_page, child.last_page, child.n_pages) == (15, 18, 4)
        assert child.title == "Former Owner Site Study"
        assert child.firm == "Older Firm & Partners"
        assert child.kind == "appended_prior_report"
        assert child.report_id == "SYN.bound1"
        assert child.folder == "bound/bound1"
        assert child.read is True
        # The rules alone claim those pages; triage said nothing here.
        assert child.said_by == ["planlens"]

    def test_the_child_holds_its_own_borings_and_the_parent_holds_none(
            self, pdf, tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN")

        assert [i.investigation_id for i in record.investigations] == \
            ["B-1", "TP-1"]
        child = ReportRecord.model_validate(json.loads(
            (tmp_path / "bound" / "bound1" / "report.record.json")
            .read_text(encoding="utf-8")))
        assert [i.investigation_id for i in child.investigations] == ["BH-1"]
        assert child.document.report_id == "SYN.bound1"
        assert child.document.n_pages == 4
        # And the child's narrative is its own, not its parent's.
        assert child.general.projectName == "Rosewood Terrace Preliminary Study"
        assert record.general.projectName == "Rosewood Terrace Development"

    def test_the_child_record_points_back_at_its_parent(self, pdf, tmp_path):
        ingest_report(pdf, FakeEngine(full_script()), out_dir=tmp_path,
                      report_id="SYN")

        child = ReportRecord.model_validate(json.loads(
            (tmp_path / "bound" / "bound1" / "report.record.json")
            .read_text(encoding="utf-8")))
        assert child.parent is not None
        assert child.parent.report_id == "SYN"
        assert child.parent.bound_id == "bound1"
        assert child.parent.pages == "15-18"
        assert child.parent.n_pages == 4
        # Its page numbers are the PARENT file's: one PDF, one numbering.
        assert [p.page for p in child.page_labels] == [15, 16, 17, 18]
        assert child.investigations[0].pages == [18]

    def test_the_parents_summary_carries_the_section(self, pdf, tmp_path):
        ingest_report(pdf, FakeEngine(full_script()), out_dir=tmp_path,
                      report_id="SYN")

        text = (tmp_path / "report.summary.md").read_text(encoding="utf-8")
        assert "## Reports bound inside this one" in text
        assert "Former Owner Site Study" in text
        assert "an earlier report, appended whole" in text
        assert "| 15-18 |" in text
        assert "`bound/bound1`" in text
        assert "is in none of the counts above" in text

    def test_both_diggs_files_are_written_and_pass_their_gates(self, pdf,
                                                              tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN")

        child = ReportRecord.model_validate(json.loads(
            (tmp_path / "bound" / "bound1" / "report.record.json")
            .read_text(encoding="utf-8")))
        for holder in (record, child):
            verdicts = {e.where: e.kind for e in holder.qa
                        if e.where.startswith("diggs")}
            assert verdicts["diggs.schema"] == "note"
            assert verdicts["diggs.roundtrip"] == "note"
        assert (tmp_path / "bound" / "bound1" / "report.diggs.xml").is_file()
        # The parent's DIGGS holds the parent's explorations and no more:
        # DIGGS has no way to say "this one belongs to another report".
        xml = (tmp_path / "report.diggs.xml").read_text(encoding="utf-8")
        assert "B-1" in xml and "BH-1" not in xml

    def test_the_library_has_a_row_for_each_with_the_parent_column(
            self, pdf, tmp_path):
        import sqlite3

        ingest_report(pdf, FakeEngine(full_script()), out_dir=tmp_path,
                      report_id="SYN")

        connection = sqlite3.connect(tmp_path / "reports.db")
        rows = {r[0]: (r[1], r[2]) for r in connection.execute(
            "SELECT report_id, id, parent FROM reports")}
        connection.close()
        assert set(rows) == {"SYN", "SYN.bound1"}
        assert rows["SYN"][1] == ""
        assert rows["SYN.bound1"][1] == rows["SYN"][0]

    def test_the_parents_qa_says_where_the_child_went(self, pdf, tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN")

        (entry,) = [e for e in record.qa if e.where == "bound.bound1"]
        assert "read into their own record" in entry.detail
        assert entry.pages == [15, 18]
        (item,) = [e for e in record.qa
                   if e.where == "items.appended_report"]
        assert item.kind == "note" and item.pages == [15, 16, 17, 18]

    def test_ingest_bound_false_reproduces_the_old_behaviour(self, pdf,
                                                             tmp_path):
        script = unbound_script()
        engine = FakeEngine(script)
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN", ingest_bound=False)

        assert engine.n_calls == len(script)
        assert record.bound_documents == []
        assert not (tmp_path / "bound").exists()
        (entry,) = [e for e in record.qa
                    if e.where == "items.appended_report"]
        assert entry.kind == "skipped"
        assert entry.pages == [15, 16, 17, 18]

    def test_a_resumed_run_re_uses_the_childs_items(self, pdf, tmp_path):
        ingest_report(pdf, FakeEngine(full_script()), out_dir=tmp_path,
                      report_id="SYN")
        files = sorted(p.name for p in
                       (tmp_path / "bound" / "bound1" / "items").iterdir())
        assert len(files) == 2                 # its narrative and its log
        assert (tmp_path / "bound" / "bound1" / "identity.json").is_file()

        empty = FakeEngine([])
        again = ingest_report(pdf, empty, out_dir=tmp_path, report_id="SYN")

        assert empty.n_calls == 0
        (child,) = again.bound_documents
        assert child.counts["investigations"] == 1
        assert child.title == "Former Owner Site Study"

    def test_the_narrative_reader_is_told_which_pages_are_not_its_report(
            self, pdf, tmp_path):
        engine = FakeEngine(full_script())
        ingest_report(pdf, engine, out_dir=tmp_path, report_id="SYN")

        briefs = [call for call in engine.calls
                  if "REPORTS BOUND INSIDE THIS ONE" in str(call["messages"])]
        assert briefs, "the parent's narrative brief never named the child"
        text = str(briefs[0]["messages"])
        assert "pages 15-18" in text
        assert "Former Owner Site Study" in text
        assert "previousInvestigationCount" in text

    def test_the_childs_reader_is_not_shown_its_parents_pages(self, pdf,
                                                              tmp_path):
        engine = FakeEngine(full_script())
        ingest_report(pdf, engine, out_dir=tmp_path, report_id="SYN")

        blob = json.loads((tmp_path / "bound" / "bound1" / "items"
                           / "item_3.json").read_text(encoding="utf-8"))
        assert set(blob["pages"]) <= {15, 16, 17, 18}

    def test_a_short_run_is_not_a_document_and_stays_in_the_parent(
            self, pdf, tmp_path, monkeypatch):
        import report_ingest.bound as bound_module

        monkeypatch.setattr(bound_module, "MIN_BOUND_PAGES", 5)
        script = unbound_script()
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN")

        assert record.bound_documents == []
        (entry,) = [e for e in record.qa if e.where == "bound.short_run"]
        assert entry.pages == [15, 16, 17, 18]
        assert "under the 5-page floor" in entry.detail

    def test_triage_and_the_rules_disagreeing_gives_the_union_and_a_note(
            self, pdf, tmp_path):
        # Triage says the bound report starts at its appendix tab, page 14;
        # the rules say it starts at its own cover, page 15. The record
        # takes 14-18 so that no page of it is left with its parent.
        script = ([triage_turn(
            "multi_document",
            bound_together=[{"kind": "appended_prior_report",
                             "pages": "14-17",
                             "title": "Former Owner Study"}])]
            + vision_turns() + [identity_turn()] + reader_turns()
            + bound_reader_turns())
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN")

        (child,) = record.bound_documents
        assert child.pages == "14-18"
        assert child.said_by == ["planlens", "triage"]
        (note,) = [e for e in record.qa if e.where == "bound.extent"]
        assert "takes the union" in note.detail
        assert note.values == ["triage 14-17", "planlens 15-18"]


# ---------------------------------------------------------------------------
# BACKLOG RI-14: the pipeline's structure
# ---------------------------------------------------------------------------

def _narrative_calls(engine):
    return [call for call in engine.calls
            if call["output_format"] is NarrativeReading]


def _brief_text(call):
    return "\n".join(str(block.get("text", ""))
                     for message in call["messages"]
                     for block in message["content"]
                     if isinstance(block, dict))


class TestTheNarrativeIsReadOnce:
    """The vision voter calls page 3 a figure, so the rules' one narrative
    item (pages 2-4) becomes two, pages 2 and 4, with a figure between."""

    def _script(self):
        return ([triage_turn()] + vision_turns({3: "figure"})
                + [identity_turn()] + reader_turns() + bound_reader_turns())

    def test_two_narrative_items_cost_one_read_over_both(self, pdf,
                                                         tmp_path):
        engine = FakeEngine(self._script())
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN", review_mode="none")

        assert engine.n_calls == len(self._script())
        parent, child = _narrative_calls(engine)
        text = _brief_text(parent)
        assert "=== page 2 ===" in text and "=== page 4 ===" in text
        assert {2, 4} <= set(record.narrative.pages)
        (note,) = [e for e in record.qa if e.where == "items.narrative"]
        assert "read once" in note.detail and note.pages == [2, 4]
        assert (tmp_path / "items" / "item_2-item_4.json").is_file()
        assert record.general.projectName == "Rosewood Terrace Development"

    def test_its_unsettled_lines_are_carried_once(self, pdf, tmp_path):
        record = ingest_report(pdf, FakeEngine(self._script()),
                               out_dir=tmp_path, report_id="SYN",
                               review_mode="none")

        # Only boringCount carries a citation, so every other answer is an
        # unsettled line of the ONE reading -- once, not once per item.
        rows = [e for e in record.qa if e.where == "projectName"]
        assert len(rows) == 1 and "no citation" in rows[0].detail

    def test_a_resumed_run_reads_it_off_disk(self, pdf, tmp_path):
        first = ingest_report(pdf, FakeEngine(self._script()),
                              out_dir=tmp_path, report_id="SYN",
                              review_mode="none")
        empty = FakeEngine([])
        again = ingest_report(pdf, empty, out_dir=tmp_path, report_id="SYN",
                              review_mode="none")

        assert empty.n_calls == 0
        assert again.general == first.general


class TestABoundRangeTheLabelsDidNotDraw:
    """Triage says pages 12-17 are another report; the rules call only 15-18
    ``appended_report``, and pages 12-13 are laboratory sheets as far as the
    labels go. The bound document is 12-18 and those sheets are ITS: read
    once, into its record, and never into the parent's as well."""

    def _script(self):
        parent = [narrative_turn(), log_turn("B-1"),
                  log_turn("TP-1", "test_pit"), calc_turn()]
        child = [lab_turn("atterberg", 12), lab_turn("gradation", 13),
                 log_turn("BH-1")]
        return ([triage_turn("multi_document", bound_together=[
            {"kind": "appended_prior_report", "pages": "12-17",
             "title": "Former Owner Study"}])]
            + vision_turns() + [identity_turn()] + parent + child)

    def _child(self, tmp_path):
        return ReportRecord.model_validate(json.loads(
            (tmp_path / "bound" / "bound1" / "report.record.json")
            .read_text(encoding="utf-8")))

    def test_its_sheets_are_read_into_the_child_alone(self, pdf, tmp_path):
        engine = FakeEngine(self._script())
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN")

        assert engine.n_calls == len(self._script())
        assert record.bound_documents[0].pages == "12-18"
        assert record.lab_tests == []
        assert sorted(t.kind for t in self._child(tmp_path).lab_tests) == \
            ["atterberg", "gradation"]
        (note,) = [e for e in record.qa if e.where == "bound.items"]
        assert note.values == ["item_7 lab_test 12", "item_8 lab_test 13"]
        assert note.pages == [12, 13]

    def test_the_parents_reconciler_does_not_call_them_unread(self, pdf,
                                                              tmp_path):
        record = ingest_report(pdf, FakeEngine(self._script()),
                               out_dir=tmp_path, report_id="SYN")

        assert not [e for e in record.qa if e.where.startswith("pages.")
                    and set(e.pages) & {12, 13}]

    def test_an_item_that_straddles_the_edge_keeps_its_own_pages(self):
        from planlens.document.roles import Item
        from report_ingest.graph import _outside_bound

        qa = []
        items = [Item(id="item_1", kind="lab_test", pages=[11, 12, 13],
                      evidence={"first_page": 11}),
                 Item(id="item_2", kind="boring_log", pages=[14, 15]),
                 Item(id="item_3", kind="appended_report", pages=[15, 16]),
                 Item(id="item_4", kind="narrative", pages=[16])]
        kept, handed = _outside_bound(items, {12, 13, 14, 15, 16}, qa)

        assert [(i.id, i.pages) for i in kept] == [
            ("item_1", [11]), ("item_3", [15, 16]), ("item_4", [16])]
        assert kept[0].evidence["first_page"] == 11
        assert [(i.id, pages) for i, pages in handed] == [
            ("item_1", [12, 13]), ("item_2", [14, 15])]
        assert qa[0].values == ["item_1 lab_test 12-13",
                                "item_2 boring_log 14-15"]
        assert "handed to that report's own pass" in qa[0].detail
        # The narrative goes too when the switch says so.
        kept, _ = _outside_bound(items, {16}, [], narrative_too=True)
        assert [i.id for i in kept] == ["item_1", "item_2", "item_3"]

    def test_a_handed_page_the_child_did_not_read_is_skipped_here(
            self, pdf, tmp_path, monkeypatch):
        import report_ingest.graph as graph

        real = graph._read_lab

        def read_lab(doc, pages, *args, **kwargs):
            if 13 in pages:
                raise RuntimeError("the sheet would not read")
            return real(doc, pages, *args, **kwargs)

        monkeypatch.setattr(graph, "_read_lab", read_lab)
        script = self._script()
        del script[-2]                       # no call for the sheet at 13
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN")

        (entry,) = [e for e in record.qa if e.kind == "skipped"
                    and e.where == "pages.lab_test"]
        assert entry.pages == [13] and entry.values == ["item_8"]
        assert "bound1" in entry.detail and "handed" in entry.detail
        assert [t.kind for t in self._child(tmp_path).lab_tests] == \
            ["atterberg"]

    def test_a_child_that_could_not_be_read_leaves_its_pages_skipped(
            self, pdf, tmp_path, monkeypatch):
        import report_ingest.graph as graph

        def broken(*args, **kwargs):
            raise RuntimeError("the bound pass broke")

        monkeypatch.setattr(graph, "_one_bound", broken)
        script = self._script()[:-3]          # no child readers run
        record = ingest_report(pdf, FakeEngine(script), out_dir=tmp_path,
                               report_id="SYN")

        assert record.lab_tests == []
        rows = [e for e in record.qa if e.kind == "skipped"
                and e.where == "pages.lab_test"]
        assert [(e.values, e.pages) for e in rows] == [
            (["item_7"], [12]), (["item_8"], [13])]
        (run,) = [e for e in record.qa
                  if e.where == "items.appended_report"]
        assert run.kind == "skipped" and "bound1" in run.detail


class _ByFormat(FakeEngine):
    """Answers each call off a queue kept per structured answer, so a test
    of WHO reads a page need not know the order the passes run in.

    A queue is a list, taken in order, or a callable of how many calls of
    that answer came before.
    """

    def __init__(self, answers):
        super().__init__([])
        self.answers = {name: (source if callable(source) else list(source))
                        for name, source in answers.items()}
        self.formats = []

    def complete(self, messages, *, output_format=None, **kwargs):
        name = getattr(output_format, "__name__", str(output_format))
        before = self.formats.count(name)
        self.formats.append(name)
        source = self.answers[name]
        self.turns.append(source(before) if callable(source)
                          else source.pop(0))
        return super().complete(messages, output_format=output_format,
                                **kwargs)


def _answers(triage, vision=None, review=()):
    """Every pass answered; the readers the same way whatever they read."""
    first, findings = review_turns(review)
    return {
        "TriageFindings": [triage],
        "VisionSheetAnswer": vision_turns(vision),
        "None": lambda n: first,
        "ReviewFindings": [findings],
        "BoundIdentity": lambda n: identity_turn(),
        "NarrativeReading": lambda n: (narrative_turn() if n == 0
                                       else bound_narrative_turn()),
        "LogReading": lambda n: log_turn("B-1"),
        "LabSheetReading": lambda n: lab_turn("atterberg", 12),
        "CalcReading": lambda n: calc_turn(),
    }


def _pages_read_by(monkeypatch, name):
    """The pages each call of one reader is given, in order."""
    import report_ingest.graph as graph

    seen = []
    real = getattr(graph, name)

    def wrapped(doc, pages, *args, **kwargs):
        seen.append(sorted(int(p) for p in pages))
        return real(doc, pages, *args, **kwargs)

    monkeypatch.setattr(graph, name, wrapped)
    return seen


def _bound_record(record, tmp_path, pages):
    (row,) = [b for b in record.bound_documents if b.pages == pages]
    return ReportRecord.model_validate_json(
        (tmp_path / row.record_path).read_text(encoding="utf-8"))


class TestTheBoundPassKeepsTheFilesReaderLabels:
    """The file's own pass hands a bound document the logs and sheets on its
    pages; that document's pass re-runs the rules over its window and has no
    label review, so it keeps the label the file settled on for every page a
    reader reads (BACKLOG RI-14 review)."""

    def test_a_page_the_review_moved_is_read_once_by_the_child(
            self, pdf, tmp_path, monkeypatch):
        # The voters split on page 10 (the rules say photos), the review
        # makes it a laboratory sheet, and triage says 7-13 is another
        # report: the bound pass, left to itself, calls it a figure.
        change = LabelChange(page=10, from_label="figure",
                             to_label="lab_test", reason="a results sheet",
                             evidence="render_page")
        engine = _ByFormat(_answers(
            triage_turn("multi_document", bound_together=[
                {"kind": "data_report", "pages": "7-13", "title": "x"}]),
            vision={10: "figure"}, review=[change]))
        lab = _pages_read_by(monkeypatch, "_read_lab")
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN")

        assert sum(pages.count(10) for pages in lab) == 1
        assert not [t for t in record.lab_tests if 10 in t.pages]
        child = _bound_record(record, tmp_path, "7-13")
        assert [t for t in child.lab_tests if 10 in t.pages]
        (label,) = [pl for pl in child.page_labels if pl.page == 10]
        assert (label.label, label.settled_by) == ("lab_test", "parent")
        (note,) = [e for e in child.qa if e.where == "bound.labels"]
        assert note.pages == [10] and note.values[0].endswith("-> lab_test")
        assert not [e for e in record.qa if e.kind == "skipped"
                    and e.where.startswith("pages.")]

    def test_an_item_across_the_edge_is_read_on_each_side_once(
            self, pdf, tmp_path, monkeypatch):
        # Triage puts the edge between the two sheets of the boring log at
        # 7-8: page 7 stays this report's and page 8 is handed over.
        engine = _ByFormat(_answers(triage_turn(
            "multi_document", bound_together=[
                {"kind": "data_report", "pages": "8-17", "title": "x"}])))
        logs = _pages_read_by(monkeypatch, "_read_log")
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN", review_mode="none")

        assert [7] in logs and [8] in logs
        assert sum(pages.count(7) + pages.count(8) for pages in logs) == 2
        assert [i.pages for i in record.investigations] == [[7]]
        # Its reading is filed apart from one of the whole item, so a folder
        # written before the edge was drawn is not taken for this one.
        assert sorted(p.name for p in (tmp_path / "items").glob("*-own.json"))
        assert not [e for e in record.qa if e.kind == "skipped"
                    and e.where.startswith("pages.")]


class TestTheNarrativeAndTheBoundPages:
    """``narrative_skips_bound``: OFF by default, because it changes what the
    narrative reader is shown."""

    def test_by_default_the_parents_reader_is_shown_them_and_told(self, pdf,
                                                                  tmp_path):
        engine = FakeEngine(full_script())
        ingest_report(pdf, engine, out_dir=tmp_path, report_id="SYN")

        text = _brief_text(_narrative_calls(engine)[0])
        assert "=== page 16 ===" in text
        assert "REPORTS BOUND INSIDE THIS ONE" in text

    def test_on_it_is_not_shown_them_and_is_still_told(self, pdf, tmp_path):
        engine = FakeEngine(full_script())
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN", narrative_skips_bound=True)

        assert engine.n_calls == len(full_script())
        parent, child = _narrative_calls(engine)
        text = _brief_text(parent)
        for page in (15, 16, 17, 18):
            assert f"=== page {page} ===" not in text
        assert "REPORTS BOUND INSIDE THIS ONE" in text
        assert not set(record.narrative.pages) & {15, 16, 17, 18}
        # The child is read as before, on its own pages.
        assert "=== page 16 ===" in _brief_text(child)

    def test_a_reading_with_it_on_is_not_taken_for_one_with_it_off(
            self, pdf, tmp_path):
        ingest_report(pdf, FakeEngine(full_script()), out_dir=tmp_path,
                      report_id="SYN")
        engine = FakeEngine([narrative_turn()])
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN", narrative_skips_bound=True)

        # Only the parent's narrative is read again: everything else is the
        # same reading as before and comes off disk.
        assert engine.n_calls == 1
        assert (tmp_path / "items" / "item_2.json").is_file()
        assert (tmp_path / "items" / "item_2-skips-bound.json").is_file()
        assert not set(record.narrative.pages) & {15, 16, 17, 18}


class TestTheExplorationAnswersAndTheLogs:
    """The narrative says four borings and three pits and names none; the
    logs read are B-1 and TP-1 and no cone."""

    def test_zeros_makes_an_unanswered_count_with_no_logs_zero(
            self, pdf, tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN",
                               counts_from_logs="zeros")

        general = record.general
        assert (general.boringCount, general.testPitCount) == (4, 3)
        assert general.cptCount == 0
        # Only the zeros: a list or a count off the logs is not filled in.
        assert general.boringDictionary is None
        assert general.testPitDictionary is None
        (note,) = [e for e in record.qa if e.where == "general.explorations"]
        assert "cptCount is 0" in note.detail
        assert note.values == ["cptCount: not stated -> 0"]
        # The difference over the stated count is still on the record.
        (entry,) = [e for e in record.qa if e.kind == "count_mismatch"
                    and e.where == "general.boringCount"]
        assert entry.values == ["narrative 4", "found 1"]

    def test_fill_takes_every_open_answer_off_the_logs(self, pdf, tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN",
                               counts_from_logs="fill")

        general = record.general
        assert (general.boringCount, general.testPitCount) == (4, 3)
        assert general.cptCount == 0
        assert general.boringDictionary == ["B-1"]
        assert general.testPitDictionary == ["TP-1"]
        (note,) = [e for e in record.qa if e.where == "general.explorations"]
        assert "came from the logs read" in note.detail

    def test_settle_overrules_a_stated_count_and_keeps_what_it_said(
            self, pdf, tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN",
                               counts_from_logs="settle")

        assert (record.general.boringCount,
                record.general.testPitCount) == (1, 1)
        assert record.narrative.stated_counts["borings"] == 4
        (entry,) = [e for e in record.qa if e.kind == "count_mismatch"
                    and e.where == "general.boringCount"]
        assert entry.values == ["narrative 4", "found 1"]
        (note,) = [e for e in record.qa if e.where == "general.explorations"]
        assert "boringCount: 4 -> 1" in note.values

    @pytest.mark.parametrize("mode", [{}, {"counts_from_logs": "off"}],
                             ids=["default", "off"])
    def test_off_the_default_leaves_the_narratives_answers_alone(
            self, pdf, tmp_path, mode):
        # Off until it is measured on real reads (the owner's rule).
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN", **mode)

        assert record.general.cptCount is None
        assert record.general.boringDictionary is None
        assert not [e for e in record.qa
                    if e.where == "general.explorations"]

    def test_a_bound_report_is_settled_against_its_own_logs(self, pdf,
                                                           tmp_path):
        record = ingest_report(pdf, FakeEngine(full_script()),
                               out_dir=tmp_path, report_id="SYN",
                               counts_from_logs="fill")

        child = ReportRecord.model_validate(json.loads(
            (tmp_path / "bound" / "bound1" / "report.record.json")
            .read_text(encoding="utf-8")))
        assert child.general.boringDictionary == ["BH-1"]
        assert child.general.testPitCount == 0
        assert record.general.boringDictionary == ["B-1"]

    def test_an_unknown_mode_is_refused(self, pdf, tmp_path):
        with pytest.raises(ValueError, match="counts_from_logs"):
            ingest_report(pdf, FakeEngine([]), out_dir=tmp_path,
                          counts_from_logs="always")


def _silent_narrative():
    """The narrative, saying nothing of borings or pits."""
    turn = narrative_turn()["final"]
    return {"final": turn.model_copy(update={"boringCount": None,
                                             "testPitCount": None})}


class TestOnlyTheLogsReadVouchForAnAnswer:
    """A log that exists and was not read -- it failed, or the item ceiling
    cut it -- is a pit or a boring the logs read do not account for, so no 0
    and no answer off the logs for that kind (BACKLOG RI-14 review)."""

    def _script(self):
        return ([triage_turn()] + vision_turns()
                + [identity_turn(), _silent_narrative(), log_turn("B-1"),
                   lab_turn("atterberg", 12), lab_turn("gradation", 13),
                   calc_turn()]
                + bound_reader_turns())

    @pytest.fixture
    def pit_fails(self, monkeypatch):
        import report_ingest.graph as graph

        real = graph._read_log

        def read_log(doc, pages, *args, **kwargs):
            if 9 in pages:
                raise RuntimeError("the pit log would not read")
            return real(doc, pages, *args, **kwargs)

        monkeypatch.setattr(graph, "_read_log", read_log)

    def test_zeros_does_not_zero_a_kind_whose_log_failed(
            self, pdf, tmp_path, pit_fails):
        record = ingest_report(pdf, FakeEngine(self._script()),
                               out_dir=tmp_path, report_id="SYN",
                               counts_from_logs="zeros")

        assert record.general.testPitCount is None
        assert record.general.cptCount == 0          # no cone item at all
        (note,) = [e for e in record.qa if e.where == "general.explorations"]
        assert note.values == [
            "cptCount: not stated -> 0",
            "testPitCount: held (the file has 1 test_pit_log item(s), "
            "1 not read)"]

    def test_fill_takes_nothing_for_a_kind_with_an_unread_log(
            self, pdf, tmp_path, pit_fails):
        record = ingest_report(pdf, FakeEngine(self._script()),
                               out_dir=tmp_path, report_id="SYN",
                               counts_from_logs="fill")

        general = record.general
        assert (general.boringCount, general.boringDictionary) == \
            (1, ["B-1"])
        assert general.testPitCount is None
        assert general.testPitDictionary is None

    def test_zeros_does_not_zero_a_kind_the_ceiling_cut(self, pdf, tmp_path):
        # Four items: the front matter to the boring log; the pit log is
        # the fifth.
        engine = _ByFormat({**_answers(triage_turn()),
                            "NarrativeReading": lambda n: (
                                _silent_narrative() if n == 0
                                else bound_narrative_turn())})
        record = ingest_report(pdf, engine, out_dir=tmp_path,
                               report_id="SYN", counts_from_logs="zeros",
                               budgets=Budgets(max_items=4))

        assert [i.investigation_id for i in record.investigations] == ["B-1"]
        assert record.general.testPitCount is None
        assert record.general.cptCount == 0


# ---------------------------------------------------------------------------
# a sounding is not a log
# ---------------------------------------------------------------------------

class TestTheSoundingReaders:

    def test_a_cone_page_and_a_probe_page_go_to_their_own_readers(self):
        from report_ingest.graph import ITEM_READERS
        assert ITEM_READERS["cpt_log"] == "cpt"
        assert ITEM_READERS["dcp_log"] == "dcp"
        # A pit IS a log: a ruler, a description column and samples.
        assert ITEM_READERS["test_pit_log"] == "log"

    def test_the_budget_has_a_ceiling_for_them(self):
        from report_ingest.graph import Budgets
        assert Budgets().sounding >= 1

    def test_a_sounding_item_becomes_an_investigation(self, tmp_path):
        from planlens.document import open_document
        from report_ingest.graph import Budgets, _read_sounding
        from report_ingest.model import Investigation
        from report_ingest.sounding_reader import SoundingReading
        from report_ingest.tests.fake_engine import FakeEngine
        from report_ingest.tests.sounding_fixtures import build_tabulated_cpt

        gt = build_tabulated_cpt()
        path = tmp_path / "cpt.pdf"
        path.write_bytes(gt.pdf)
        doc = open_document(str(path))
        try:
            engine = FakeEngine([{"final": SoundingReading(
                investigation_id="CPT-4", kind="cpt", depth_unit="m",
                qc_unit="MPa", points=[])}])

            class _Item:
                id = "item_1"
                title = "CPT-4"

            blob = _read_sounding(doc, [0], engine, Budgets(), _Item(),
                                  "RXX", "cpt")
            record = Investigation.model_validate(blob["investigation"])
            assert record.kind == "cpt"
            assert record.cpt.n_points == len(gt.series)
        finally:
            doc.close()


class TestTriageEngine:
    """The triage can run on its own engine (a cheaper tier). Until 5.32 the
    graph took one engine, so the ingest stage's ``triage_model`` was silently
    the main model (Foundry run, 2026-10-01)."""

    def test_triage_goes_to_its_own_engine_and_nothing_else_does(
            self, pdf, tmp_path):
        script = full_script()
        triage = FakeEngine(script[:1])
        main = FakeEngine(script[1:])
        ingest_report(pdf, main, triage_engine=triage, out_dir=tmp_path,
                      report_id="SYN")
        assert triage.n_calls == 1
        assert main.n_calls == len(script) - 1

    def test_without_one_the_main_engine_triages(self, pdf, tmp_path):
        main = FakeEngine(full_script())
        ingest_report(pdf, main, out_dir=tmp_path, report_id="SYN")
        assert main.n_calls == len(full_script())


class TestTheMinuteBudget:
    """``Budgets.tokens_per_minute`` (BACKLOG RI-127) puts the process's
    pacer for it on the run's engines for the length of the run, and unset
    puts nothing anywhere. The budgets are made up."""

    @staticmethod
    def _engine(name="made-up-model"):
        from report_ingest.engine import ClaudeEngine

        return ClaudeEngine(name, client=object())    # never called here

    def test_unset_paces_nothing(self):
        from report_ingest import graph

        engine = self._engine()
        assert Budgets().tokens_per_minute is None
        assert graph._pace(Budgets(), engine, graph._Labelling(), None,
                           None) == []
        assert engine.pacer is None

    def test_set_every_engine_of_the_run_shares_one_pacer(self):
        import types

        from report_ingest import graph
        from report_ingest.engine import shared_pacer

        tally = graph._Tally()
        main, vision, reader, staged = (self._engine(n) for n in (
            "made-up-a", "made-up-b", "made-up-c", "made-up-d"))
        labelling = graph._Labelling(vision_engine=tally.wrap(vision))
        first_run = types.SimpleNamespace(engines={"logs": tally.wrap(
            staged)})
        put = graph._pace(Budgets(tokens_per_minute=123_459), tally.wrap(main),
                          labelling, {"log": tally.wrap(reader)}, first_run)
        pacer = shared_pacer(123_459)
        assert put == [main, vision, reader, staged]
        assert all(e.pacer is pacer for e in put)

    def test_the_run_carries_it_and_takes_it_off_at_the_end(
            self, pdf, tmp_path, monkeypatch):
        from report_ingest import graph
        from report_ingest.engine import shared_pacer

        seen = {}

        def stop(doc, engine, *args, **kwargs):
            seen["pacer"] = engine.pacer        # through the run's wrapper
            raise RuntimeError("made-up stop")

        monkeypatch.setattr(graph, "_run", stop)
        engine = self._engine()
        with pytest.raises(RuntimeError, match="made-up stop"):
            ingest_report(pdf, engine, out_dir=tmp_path, report_id="SYN",
                          budgets=Budgets(tokens_per_minute=123_461,
                                          large_request_fraction=0.25))
        assert seen["pacer"] is shared_pacer(123_461, large_fraction=0.25)
        assert engine.pacer is None, "taken off when the run ended"

    def test_an_engine_s_own_pacer_is_kept(self, pdf, tmp_path, monkeypatch):
        from report_ingest import graph
        from report_ingest.engine import RequestPacer

        own = RequestPacer(54_321)
        engine = self._engine()
        engine.pacer = own

        def stop(*args, **kwargs):
            raise RuntimeError("made-up stop")

        monkeypatch.setattr(graph, "_run", stop)
        with pytest.raises(RuntimeError):
            ingest_report(pdf, engine, out_dir=tmp_path, report_id="SYN",
                          budgets=Budgets(tokens_per_minute=123_463))
        assert engine.pacer is own

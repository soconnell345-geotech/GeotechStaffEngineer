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

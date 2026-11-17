"""What a report names that a partial copy drops (BACKLOG RI-185).

Made-up pages and records only: explorations the text names with no log,
laboratory results the text says are in the document with none held, and
appendices a contents page lists that the file plainly lacks.
"""

from __future__ import annotations

import pytest

from report_ingest.completeness import (
    completeness_qa, lab_said_not_held, listed_appendices_not_held,
    named_not_held_qa, named_not_logged,
)
from report_ingest.model import Investigation, ReportRecord


def record(*holes: str, lab: int = 0) -> ReportRecord:
    out = ReportRecord()
    out.investigations = [Investigation(investigation_id=h) for h in holes]
    if lab:
        from report_ingest.model import LabTest, OtherResult
        out.lab_tests = [LabTest(kind="other", result=OtherResult())
                         for _ in range(lab)]
    return out


NARRATIVE = (
    "FIELD EXPLORATION\n"
    "Eight made-up borings, B-1 through B-8, were drilled at the made-up\n"
    "site to depths of 6 to 12 m. Borings B-1 through B-8 were logged by\n"
    "a made-up engineer.")
LOG = "LOG OF BORING {name}\nMADE-UP CONSULTANTS\n0.0 Made-up brown clay"


def with_logs(*names: str):
    """A narrative page and one log page per name: texts and labels."""
    texts = [NARRATIVE] + [LOG.format(name=n) for n in names]
    labels = {0: "narrative", **{p: "boring_log"
                                 for p in range(1, len(texts))}}
    return texts, labels


# ---------------------------------------------------------------------------
# 1. explorations named, with no log
# ---------------------------------------------------------------------------

class TestNamedNotLogged:

    def test_named_holes_the_record_and_the_file_lack_are_flagged(self):
        held = ["B-1", "B-2", "B-3", "B-4", "B-5"]
        texts, labels = with_logs(*held)
        lacking = named_not_logged(texts, labels, record(*held))

        assert lacking == {"B-6": [0], "B-7": [0], "B-8": [0]}

    def test_none_of_the_named_holes_held_is_flagged_too(self):
        # The copy lacks every log: the series is the exploration word's,
        # counted because the text says the logs are in this document.
        said = NARRATIVE + ("\nThe made-up boring logs are presented in "
                            "Appendix A.")
        lacking = named_not_logged([said], {0: "narrative"}, record())

        assert sorted(lacking) == [f"B-{n}" for n in range(1, 9)]
        # Without that sentence the holes may be logged nowhere at all.
        assert named_not_logged([NARRATIVE], {0: "narrative"},
                                record()) == {}

    def test_a_whole_report_is_not_flagged(self):
        held = [f"B-{n}" for n in range(1, 9)]
        texts, labels = with_logs(*held)
        assert named_not_logged(texts, labels, record(*held)) == {}

    def test_a_padded_or_prefixed_name_on_the_log_is_the_same_hole(self):
        held = [f"B-{n:02d}" for n in range(1, 9)]
        texts, labels = with_logs(*held)
        assert named_not_logged(texts, labels, record(*held)) == {}

    def test_a_single_mention_is_not_flagged(self):
        held = [f"B-{n}" for n in range(1, 9)]
        texts, labels = with_logs(*held)
        texts[0] += "\nA made-up boring B-9 is shown on the made-up plan."
        assert named_not_logged(texts, labels, record(*held)) == {}

    def test_a_hole_the_text_says_was_not_drilled_is_not_missing(self):
        held = [f"B-{n}" for n in range(1, 8)]
        texts, labels = with_logs(*held)
        texts[0] += ("\nBoring B-8 was not drilled because of made-up "
                     "overhead lines.")
        assert named_not_logged(texts, labels, record(*held)) == {}

    @pytest.mark.parametrize("said", [
        "Boring B-8 was abandoned at 1 m on a made-up obstruction.",
        "Borings B-7 and B-8 were drilled by others for the made-up owner.",
        "Borings B-7 and B-8 are from a previous investigation of the "
        "made-up site.",
        "Borings B-7 and B-8 were drilled for the 1998 investigation.",
    ])
    def test_abandoned_by_others_and_earlier_holes_are_not_missing(
            self, said):
        held = [f"B-{n}" for n in range(1, 7)]
        texts, labels = with_logs(*held)
        texts[0] = ("Six made-up borings, B-1 through B-6, were drilled. "
                    "Borings B-1 through B-6 were logged in the field. "
                    + said + " " + said.replace("made-up", "invented"))
        assert named_not_logged(texts, labels, record(*held)) == {}

    def test_a_bound_report_s_holes_are_its_own_record_s(self):
        # B-7 and B-8 are named twice and printed on the bound pages.
        held = [f"B-{n}" for n in range(1, 7)]
        texts, labels = with_logs(*held)
        texts += [LOG.format(name="B-7"), LOG.format(name="B-8")]
        labels.update({7: "boring_log", 8: "boring_log"})
        assert named_not_logged(texts, labels, record(*held),
                                bound=[7, 8]) == {}

    def test_a_scanned_bound_document_may_hold_them(self):
        held = [f"B-{n}" for n in range(1, 7)]
        texts, labels = with_logs(*held)
        texts += ["", ""]
        labels.update({7: "appended_report", 8: "appended_report"})
        assert named_not_logged(texts, labels, record(*held),
                                bound=[7, 8]) == {}

    def test_holes_the_reader_may_have_named_otherwise_are_not_missing(self):
        # Two scanned logs read under names the text does not use: the two
        # holes the text names may be those.
        held = [f"B-{n}" for n in range(1, 7)]
        texts, labels = with_logs(*held)
        texts += ["", ""]
        labels.update({7: "boring_log", 8: "boring_log"})
        assert named_not_logged(
            texts, labels, record(*held, "8-1", "8-3")) == {}
        # Scanned logs nothing was read from may hold them as well.
        assert named_not_logged(texts, labels, record(*held)) == {}
        # One scan cannot hold two holes.
        assert sorted(named_not_logged(texts[:-1], labels,
                                       record(*held))) == ["B-7", "B-8"]

    def test_a_figure_s_or_a_sample_s_name_is_not_a_hole_s(self):
        held = [f"B-{n}" for n in range(1, 9)]
        texts, labels = with_logs(*held)
        texts[0] += ("\nSee Figure B-9 and Appendix B-10. Figures B-9 and "
                     "B-11 show made-up sections; Sample B-12 and Sample "
                     "B-12 again. The made-up D50 was 0.2 mm; D50 again.")
        assert named_not_logged(texts, labels, record(*held)) == {}

    def test_logs_said_to_be_under_separate_cover_are_not_flagged(self):
        texts = [NARRATIVE + "\nThe made-up logs are provided under "
                 "separate cover."]
        assert named_not_logged(texts, {0: "narrative"}, record()) == {}

    def test_test_pits_and_soundings_are_named_the_same_way(self):
        texts = ["Made-up test pits TP-1, TP-2 and TP-3 were excavated and "
                 "soundings CPT-1 and CPT-2 advanced. Test pits TP-1, TP-2 "
                 "and TP-3 met made-up fill; CPT-1 and CPT-2 met refusal.",
                 "TEST PIT LOG TP-1\nMade-up fill"]
        labels = {0: "narrative", 1: "test_pit_log"}
        lacking = named_not_logged(texts, labels, record("TP-1"))

        # Only a series the record holds counts: it holds no sounding.
        assert sorted(lacking) == ["TP-2", "TP-3"]
        assert sorted(named_not_logged(
            texts, labels, record("TP-1", "CPT-3"))) == [
            "CPT-1", "CPT-2", "TP-2", "TP-3"]

    @pytest.mark.parametrize("said", [
        "A made-up monitoring well, MW-1, was installed in boring B-3. "
        "Water in MW-1 stood at 3 m.",
        "Made-up piezometers VW-1 and VW-2 were set in borings B-2 and "
        "B-5. Readings from VW-1 and VW-2 are in Table 3.",
        "Made-up hand augers HA-1 through HA-3 were advanced. Table 1 "
        "lists the soils in HA-1 through HA-3.",
    ])
    def test_a_series_the_record_does_not_hold_is_not_counted(self, said):
        held = [f"B-{n}" for n in range(1, 9)]
        texts, labels = with_logs(*held)
        texts[0] += "\n" + said
        assert named_not_logged(texts, labels, record(*held)) == {}

    @pytest.mark.parametrize("said", [
        "Two additional borings, B-9 and B-10, should be drilled at the "
        "made-up wall. Borings B-9 and B-10 are needed for final design.",
        "Borings B-9 and B-10 will be drilled once access is granted. "
        "Logs of B-9 and B-10 will follow.",
        "Borings B-9 and B-10 were drilled in 2011 for the made-up "
        "original building. Water in B-9 and B-10 was at 3 m in 2011.",
        "Existing borings B-9 and B-10 met the same clay; the existing "
        "B-9 and B-10 logs are on file.",
    ])
    def test_future_dated_and_existing_holes_are_not_missing(self, said):
        held = [f"B-{n}" for n in range(1, 9)]
        texts, labels = with_logs(*held)
        texts[0] += "\n" + said
        assert named_not_logged(texts, labels, record(*held)) == {}

    def test_an_aashto_class_is_not_a_hole(self):
        held = ["A-1", "A-2", "A-3", "A-4"]
        texts = ["Made-up auger holes A-1 through A-4 met fill. Auger "
                 "holes A-1 through A-4 were logged. The fill is A-2-4 "
                 "and the clay A-7-6; the A-7-6 clay is soft and the "
                 "A-2-4 fill loose."] + [LOG.format(name=n) for n in held]
        labels = {0: "narrative", **{p: "boring_log" for p in range(1, 5)}}
        assert named_not_logged(texts, labels, record(*held)) == {}

    def test_log_pages_naming_no_recorded_hole_may_hold_the_missing(self):
        # The headers print "B 7" and "BORING NO. 8", which the text's
        # name finder does not read: those pages may be B-7's and B-8's.
        held = [f"B-{n}" for n in range(1, 7)]
        texts, labels = with_logs(*held)
        texts += ["LOG OF BORING B 7\nMADE-UP CONSULTANTS\n0.0 Made-up clay",
                  "LOG OF BORING NO. 8\nMADE-UP CONSULTANTS\n0.0 Made-up clay"]
        labels.update({7: "boring_log", 8: "boring_log"})
        assert named_not_logged(texts, labels, record(*held)) == {}
        # One such page cannot hold two holes.
        assert sorted(named_not_logged(texts[:-1], labels,
                                       record(*held))) == ["B-7", "B-8"]

    def test_with_the_document_the_log_headers_are_read(self):
        # The log reader's header rule reads "B 7" off the page's header
        # band; the hole is printed, so it is not missing.
        class Line:
            def __init__(self, text, top):
                self.text, self.bbox = text, (40.0, top, 300.0, top + 10)
                self.rotation = 0

        class Page:
            def __init__(self, lines):
                self.lines, self.height = lines, 792.0

        class Doc:
            def __init__(self, heads):
                self.heads = heads

            def page(self, n, tables=True):
                return Page([Line(self.heads.get(n, ""), 30.0),
                             Line("0.0 Made-up clay", 400.0)])

        held = [f"B-{n}" for n in range(1, 8)]
        texts, labels = with_logs(*held)
        # Its text names a recorded hole in a remark, so read as text the
        # page is taken for B-3's and B-8 is flagged.
        texts.append("LOG OF BORING B 8\nOffset 3 m from B-3\n0.0 Made-up "
                     "clay")
        labels[8] = "boring_log"
        assert sorted(named_not_logged(texts, labels,
                                       record(*held))) == ["B-8"]
        heads = {p: f"Boring No.: B-{p}" for p in range(1, 8)}
        heads[8] = "Boring No.: B 8"
        assert named_not_logged(texts, labels, record(*held),
                                doc=Doc(heads)) == {}


# ---------------------------------------------------------------------------
# 2. laboratory results named, with none held
# ---------------------------------------------------------------------------

LAB_SAID = ("Made-up sieve analyses and Atterberg limits were run on "
            "selected samples. The laboratory test results are presented "
            "in Appendix C.")
CONTENTS = ("TABLE OF CONTENTS\n"
            "Appendix A - Site Plan and Figures ......... 12\n"
            "Appendix B - Boring Logs ................... 14\n"
            "Appendix C - Laboratory Test Results ....... 30")


class TestLabSaidNotHeld:

    def test_results_said_to_be_included_with_none_held_are_flagged(self):
        texts = [CONTENTS, LAB_SAID]
        labels = {0: "toc", 1: "narrative"}
        said = lab_said_not_held(texts, labels, record("B-1"))

        assert [page for page, _s in said] == [0, 1]
        assert said[0][1].startswith("Appendix C - Laboratory Test Results")
        assert said[1][1] == ("The laboratory test results are presented "
                              "in Appendix C.")

    def test_a_record_with_a_lab_test_is_not_flagged(self):
        texts, labels = [CONTENTS, LAB_SAID], {0: "toc", 1: "narrative"}
        assert lab_said_not_held(texts, labels, record("B-1", lab=1)) == []

    def test_a_file_with_a_laboratory_page_is_not_flagged(self):
        texts = [CONTENTS, LAB_SAID, "Made-up plot of made-up points"]
        labels = {0: "toc", 1: "narrative", 2: "lab_test"}
        assert lab_said_not_held(texts, labels, record("B-1")) == []
        # A page labelled otherwise that leads with a sheet's words.
        texts[2] = "GRAIN SIZE DISTRIBUTION\nMade-up sample S-2"
        labels[2] = "other"
        assert lab_said_not_held(texts, labels, record("B-1")) == []

    def test_a_single_mention_is_not_flagged(self):
        assert lab_said_not_held([LAB_SAID], {0: "narrative"},
                                 record("B-1")) == []

    @pytest.mark.parametrize("said", [
        "Laboratory testing is recommended on the made-up clay, and the "
        "results should be presented in Appendix C.",
        "Laboratory test results will be included in Appendix C of the "
        "final made-up report.",
        "The laboratory test results are provided under separate cover; "
        "they are not attached in Appendix C.",
        "Laboratory tests by others on the made-up fill are included in "
        "Appendix C of the previous report.",
        "The laboratory program is described in Appendix C.",
        "The results of the laboratory tests are shown on the boring logs.",
    ])
    def test_planned_or_elsewhere_results_are_not_flagged(self, said):
        texts = [said, said.replace("made-up", "invented") + " "]
        labels = {0: "narrative", 1: "narrative"}
        assert lab_said_not_held(texts, labels, record("B-1")) == []

    def test_results_on_the_logs_are_the_logs(self):
        texts = ["TABLE OF CONTENTS\nAppendix A - Boring Logs and "
                 "Laboratory Test Results\nAppendix B - Photographs",
                 "The results of the made-up laboratory moisture content "
                 "tests are presented in Appendix A on the boring logs."]
        labels = {0: "toc", 1: "narrative"}
        assert lab_said_not_held(texts, labels, record("B-1")) == []

    def test_a_sheet_labelled_other_behind_letterhead_lines_is_one(self):
        sheet = ("MADE-UP LABS INC\n12 Made-up Road\nProject: Made-up\n"
                 "Client: Made-up\nDate: made-up\nBoring: B-1\nDepth: 2 m\n"
                 "Technician: X\nPARTICLE SIZE ANALYSIS")
        texts = [CONTENTS, LAB_SAID, sheet]
        labels = {0: "toc", 1: "narrative", 2: "other"}
        assert lab_said_not_held(texts, labels, record("B-1")) == []
        # A scanned page labelled other may be a sheet as well.
        texts[2] = ""
        assert lab_said_not_held(texts, labels, record("B-1")) == []
        # A page of the report's own text is not searched past its lead.
        labels[2] = "narrative"
        texts[2] = "Made-up text."
        assert len(lab_said_not_held(texts, labels, record("B-1"))) == 2


# ---------------------------------------------------------------------------
# 3. appendices in the contents with no pages
# ---------------------------------------------------------------------------

def a_report(*, lab_pages: bool, lab_title: str = ""):
    """Contents listing A to C; A and B held, C as the arguments say."""
    texts = [CONTENTS, "Made-up narrative of the made-up site.",
             "APPENDIX A\nSITE PLAN AND FIGURES", "Made-up site plan",
             "APPENDIX B\nBORING LOGS", LOG.format(name="B-1")]
    labels = {0: "toc", 1: "narrative", 2: "divider", 3: "plan",
              4: "divider", 5: "boring_log"}
    if lab_title:
        labels[len(texts)] = "other"
        texts.append(lab_title)
    if lab_pages:
        labels[len(texts)] = "lab_test"
        texts.append("Made-up plasticity chart")
    return texts, labels


class TestListedAppendicesNotHeld:

    def test_a_listed_appendix_the_file_lacks_is_named(self):
        texts, labels = a_report(lab_pages=False)
        assert listed_appendices_not_held(texts, labels) == [
            ("C", "Laboratory Test Results", 0)]

    def test_a_whole_report_is_not_flagged(self):
        texts, labels = a_report(lab_pages=True)
        texts.insert(6, "APPENDIX C\nLABORATORY TEST RESULTS")
        labels = {**labels, 6: "divider", 7: "lab_test"}
        assert listed_appendices_not_held(texts, labels) == []

    def test_pages_of_its_content_after_the_last_one_held_hold_it(self):
        # No divider for Appendix C, but laboratory sheets follow B.
        texts, labels = a_report(lab_pages=True)
        assert listed_appendices_not_held(texts, labels) == []

    def test_a_slightly_different_title_holds_it(self):
        texts, labels = a_report(
            lab_pages=False,
            lab_title="LABORATORY TESTING RESULT SHEETS\nMade-up sheet")
        assert listed_appendices_not_held(texts, labels) == []

    def test_no_appendix_held_is_ri_184_s_to_say(self):
        texts, labels = [CONTENTS, "Made-up narrative."], {0: "toc",
                                                           1: "narrative"}
        assert listed_appendices_not_held(texts, labels) == []

    def test_an_appendix_not_used_or_of_unknown_content_is_left_alone(self):
        texts, labels = a_report(lab_pages=True)
        texts[0] += ("\nAppendix D - Not Used"
                     "\nAppendix E - Calculations (under separate cover)"
                     "\nAppendix F - Important Made-up Information")
        assert listed_appendices_not_held(texts, labels) == []

    @pytest.mark.parametrize("said", [
        "The made-up laboratory results (Appendix C) will be sent under "
        "separate cover.",
        "Appendix C is not used in this made-up report.",
    ])
    def test_an_appendix_the_text_says_is_not_here_is_left_alone(self,
                                                                 said):
        texts, labels = a_report(lab_pages=False)
        texts[1] += " " + said
        assert listed_appendices_not_held(texts, labels) == []
        # A sentence that only names it does not excuse it.
        texts[1] = "Made-up results are presented in Appendix C."
        assert listed_appendices_not_held(texts, labels) == [
            ("C", "Laboratory Test Results", 0)]

    def test_one_listed_appendix_is_not_a_list(self):
        texts = ["CONTENTS\nAppendix A - Boring Logs", "Made-up narrative."]
        assert listed_appendices_not_held(
            texts, {0: "toc", 1: "narrative"}) == []


# ---------------------------------------------------------------------------
# the entry
# ---------------------------------------------------------------------------

class TestTheEntry:

    def test_one_open_partial_entry_names_all_three(self):
        held = ["B-1", "B-2", "B-3", "B-4", "B-5", "B-6"]
        texts = [CONTENTS, NARRATIVE + "\n" + LAB_SAID,
                 "APPENDIX A\nSITE PLAN AND FIGURES", "Made-up site plan",
                 "APPENDIX B\nBORING LOGS"] + \
            [LOG.format(name=n) for n in held]
        labels = {0: "toc", 1: "narrative", 2: "divider", 3: "plan",
                  4: "divider", **{p: "boring_log" for p in range(5, 11)}}
        (entry,) = named_not_held_qa(texts, labels, record(*held))

        assert (entry.kind, entry.rank) == ("partial", "open")
        assert entry.where == "document.completeness"
        assert entry.detail == (
            "this copy looks incomplete: the text names the explorations "
            "B-7 and B-8 (page 2), and the record holds no log for them; "
            "the text says laboratory results are in this document "
            "(pages 1-2), naming sieve or grain size and Atterberg limits, "
            "and the record holds no laboratory page or test; the contents "
            "(page 1) list Appendix C, which the file does not hold")
        assert entry.values == [
            "exploration B-7: named on page 2",
            "exploration B-8: named on page 2",
            'laboratory results, page 1: "Appendix C - Laboratory Test '
            'Results ....... 30"',
            'laboratory results, page 2: "The laboratory test results are '
            'presented in Appendix C."',
            'Appendix C: "Laboratory Test Results", listed on page 1']
        assert entry.pages == [0, 1]

    def test_a_complete_report_with_every_kind_gets_none(self):
        held = [f"B-{n}" for n in range(1, 9)]
        texts = [CONTENTS, NARRATIVE + "\n" + LAB_SAID,
                 "APPENDIX A\nSITE PLAN AND FIGURES", "Made-up site plan",
                 "APPENDIX B\nBORING LOGS"] + \
            [LOG.format(name=n) for n in held] + \
            ["APPENDIX C\nLABORATORY TEST RESULTS",
             "ATTERBERG LIMITS\nMade-up sample S-1"]
        labels = {0: "toc", 1: "narrative", 2: "divider", 3: "plan",
                  4: "divider", **{p: "boring_log" for p in range(5, 13)},
                  13: "divider", 14: "lab_test"}
        assert named_not_held_qa(texts, labels,
                                 record(*held, lab=1)) == []
        assert completeness_qa(texts, labels) == []

    def test_without_labels_or_a_record_nothing_is_said(self):
        assert named_not_held_qa([NARRATIVE], None, record()) == []
        assert named_not_held_qa([NARRATIVE], {0: "narrative"}, None) == []


def test_the_graph_hands_the_check_the_record_and_the_bound_pages(
        tmp_path, monkeypatch):
    pytest.importorskip("planlens.document.roles")
    from report_ingest import graph
    from report_ingest.graph import ingest_report
    from report_ingest.model import QAEntry
    from report_ingest.tests import test_graph as tg
    from report_ingest.tests.fake_engine import FakeEngine
    from report_ingest.tests.narrative_fixtures import build_narrative_report

    pdf = tmp_path / "SYN.pdf"
    pdf.write_bytes(build_narrative_report().pdf)
    seen = {}

    def note(texts, labels, record, bound=(), doc=None):
        seen.update(texts=list(texts), labels=dict(labels), record=record,
                    bound=list(bound), doc=doc)
        return [QAEntry(kind="partial", rank="open",
                        where="document.completeness",
                        detail="this copy looks incomplete: made up names")]

    monkeypatch.setattr(graph, "named_not_held_qa", note)
    out = ingest_report(pdf, FakeEngine(tg.full_script()),
                        out_dir=tmp_path / "out", report_id="SYN",
                        model_first="off")

    assert seen["record"] is out
    assert seen["bound"] == sorted(
        p for p, label in seen["labels"].items()
        if label == "appended_report")
    assert len(seen["texts"]) == len(seen["labels"])
    assert [q.detail for q in out.qa
            if q.where == "document.completeness"
            and "made up names" in q.detail]
    assert seen["doc"] is not None


def test_a_completeness_check_that_fails_does_not_end_the_ingest(
        tmp_path, monkeypatch):
    pytest.importorskip("planlens.document.roles")
    from report_ingest import graph
    from report_ingest.graph import ingest_report
    from report_ingest.tests import test_graph as tg
    from report_ingest.tests.fake_engine import FakeEngine
    from report_ingest.tests.narrative_fixtures import build_narrative_report

    pdf = tmp_path / "SYN.pdf"
    pdf.write_bytes(build_narrative_report().pdf)

    def fails(*args, **kwargs):
        raise KeyError("made-up")

    monkeypatch.setattr(graph, "named_not_held_qa", fails)
    out = ingest_report(pdf, FakeEngine(tg.full_script()),
                        out_dir=tmp_path / "out", report_id="SYN",
                        model_first="off")

    assert out.document.n_pages > 0
    (entry,) = [q for q in out.qa if q.where == "document.completeness"
                and "did not run" in q.detail]
    assert entry.kind == "note"
    assert "(KeyError: " in entry.detail
    assert any((tmp_path / "out").rglob("report.record.json"))


def test_two_pages_apart_are_named_as_two():
    from report_ingest.completeness import _span
    assert _span([2, 39]) == "pages 3 and 40"
    assert _span([2, 3]) == "pages 3-4"
    assert _span([2, 5, 39]) == "pages 3-40"

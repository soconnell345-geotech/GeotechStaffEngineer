"""Is this copy of the report the whole of it? (BACKLOG RI-184)

Made-up page texts only: a footer numbering that runs short, and figures,
tables and appendices the text cites that no page holds.
"""

from __future__ import annotations

import pytest

from report_ingest.completeness import (
    cited_not_held, completeness_qa, short_runs,
)


def footer(x: int, y: int, body: str = "Made-up narrative text.") -> str:
    return f"{body}\nMADE-UP CONSULTANTS\nPage {x} of {y}"


# ---------------------------------------------------------------------------
# a printed numbering that runs short
# ---------------------------------------------------------------------------

class TestShortRuns:

    def test_a_document_lacking_its_last_page_is_found(self):
        # A letter's first page prints no footer and holds its page 1.
        texts = ["Dear Made-up Client, the made-up letter begins."] + \
            [footer(x, 6) for x in range(2, 6)]
        (run,) = short_runs(texts)

        assert (run.of, run.missing) == (6, (6,))
        assert run.pages == (1, 2, 3, 4) and run.printed == (2, 3, 4, 5)

    def test_a_whole_document_is_not(self):
        texts = ["Made-up cover"] + [footer(x, 4) for x in range(2, 5)] + \
            ["Made-up appendix cover sheet"]
        assert short_runs(texts) == []

    def test_a_figure_between_two_numbered_pages_holds_nothing_missing(self):
        texts = [footer(1, 4), "FIGURE 1 MADE-UP SITE PLAN", footer(3, 4),
                 footer(4, 4)]
        assert short_runs(texts) == []

    def test_numbers_skipped_with_no_page_between_are_missing(self):
        texts = [footer(1, 5), footer(2, 5), footer(5, 5)]
        (run,) = short_runs(texts)

        assert run.missing == (3, 4)

    def test_a_run_that_starts_late_after_other_pages_lacks_its_first(self):
        texts = ["Made-up laboratory sheet one, with its made-up results.",
                 "Made-up laboratory sheet two, with more made-up results."] \
            + [footer(x, 7) for x in (3, 4, 5, 6, 7)]
        (run,) = short_runs(texts)

        assert run.missing == (1, 2)

    def test_a_scanned_page_with_no_text_may_hold_a_missing_page(self):
        texts = [footer(1, 3), footer(2, 3), "   "]
        assert short_runs(texts) == []

    def test_a_second_numbering_on_a_longer_run_s_pages_is_not_counted(self):
        # A certificate's own "of 12" on three of a whole run's pages,
        # number for number.
        texts = [footer(x, 9) + (f"\nPage {x} of 12" if x in (4, 5, 6) else "")
                 for x in range(1, 10)]
        assert short_runs(texts) == []

    def test_a_calculation_package_s_numbering_is_not_checked(self):
        texts = ["Made-up calculation sheet one, for the made-up wall.",
                 "Made-up calculation sheet two, for the made-up wall.",
                 footer(3, 4), footer(4, 4)]
        assert [r.of for r in short_runs(texts)] == [4]
        labels = {page: "calculation" for page in range(4)}
        assert short_runs(texts, labels) == []


# ---------------------------------------------------------------------------
# what the text cites that the file does not hold
# ---------------------------------------------------------------------------

class TestCitedNotHeld:

    def test_figures_and_appendices_the_file_lacks_are_named(self):
        texts = ["The made-up site is shown on Figures 1 through 3; the "
                 "logs are in Appendix A and the laboratory results in "
                 "Appendices B and C.",
                 "FIGURE 2\nMADE-UP BORING LOCATION PLAN",
                 "APPENDIX B\nMADE-UP LABORATORY RESULTS"]
        missing = cited_not_held(texts)

        assert missing == {"figure": {"1": [0], "3": [0]},
                           "appendix": {"A": [0], "C": [0]}}

    def test_a_contents_page_holds_nothing(self):
        contents = "Figure 3 Made-up Profile\nFigure 4 Made-up Section"
        texts = ["See Figure 4 for the made-up section.", contents,
                 "FIGURE 3\nMADE-UP PROFILE"]
        # Unlabelled, two titles on a page may be the figures' own captions.
        assert cited_not_held(texts) == {}
        labels = {0: "narrative", 1: "toc", 2: "figure"}
        assert cited_not_held(texts, labels) == {"figure": {"4": [0, 1]}}

    def test_a_page_leading_with_many_captions_is_a_list_of_them(self):
        listed = "\n".join(f"Figure {n} Made-up Sheet {n}" for n in range(1, 5))
        texts = ["See Figure 4.", listed]
        assert cited_not_held(texts) == {
            "figure": {"1": [1], "2": [1], "3": [1], "4": [0, 1]}}

    def test_a_code_s_figure_or_table_is_not_the_report_s(self):
        texts = ["The made-up site class follows ASCE 7 Table 31.4-2, and "
                 "the made-up groundwater rose to the Water Table\n7.5 feet."]
        assert cited_not_held(texts) == {}

    def test_only_the_report_s_own_pages_cite_when_labels_say(self):
        texts = ["The made-up results are discussed below.",
                 "A made-up chart (Fig. 8.6) on a textbook's page."]
        labels = {0: "narrative", 1: "calculation"}
        assert cited_not_held(texts, labels) == {}
        assert "figure" in cited_not_held(texts)

    def test_a_picture_page_may_hold_a_figure_the_text_cannot_name(self):
        texts = ["See Figures 1 and 2.", "FIGURE NO.:\nMADE-UP TITLE BLOCK",
                 ""]
        labels = {0: "narrative", 1: "figure", 2: "plan"}
        assert cited_not_held(texts, labels) == {}
        # One picture page cannot be both figures.
        labels[2] = "narrative"
        assert cited_not_held(texts, labels) == {"figure": {"1": [0],
                                                            "2": [0]}}


# ---------------------------------------------------------------------------
# the QA entries
# ---------------------------------------------------------------------------

class TestTheNote:

    def test_a_short_copy_gets_open_partial_notes_naming_what_it_lacks(self):
        texts = (["Dear Made-up Client. See Figures 1-3 and Table 2."]
                 + [footer(x, 8) for x in range(2, 7)])
        first, second = completeness_qa(texts)

        assert (first.kind, first.rank) == ("partial", "open")
        assert first.where == "document.completeness"
        assert first.detail == (
            "this copy looks incomplete: the document numbered 'Page X of 8' "
            "on pages 2-6 lacks its pages 7-8 of 8; the file has no page to "
            "hold them")
        assert first.values == ["Page 7 of 8", "Page 8 of 8"]
        assert first.pages == [1, 2, 3, 4, 5]
        assert (second.kind, second.rank) == ("partial", "open")
        assert second.detail == (
            "this copy looks incomplete: the text cites Figures 1-3 and "
            "Table 2, which the file does not hold")
        assert second.values == ["figure 1: cited on page 1",
                                 "figure 2: cited on page 1",
                                 "figure 3: cited on page 1",
                                 "table 2: cited on page 1"]
        assert second.pages == [0]

    def test_a_whole_copy_gets_none(self):
        texts = ["See Figure 1.", "FIGURE 1\nMADE-UP SITE PLAN",
                 footer(1, 2), footer(2, 2)]
        assert completeness_qa(texts) == []


# ---------------------------------------------------------------------------
# the graph records it
# ---------------------------------------------------------------------------

def test_the_graph_puts_the_note_on_the_record(tmp_path, monkeypatch):
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

    def note(texts, labels):
        seen.update(texts=list(texts), labels=dict(labels))
        return [QAEntry(kind="partial", rank="open",
                        where="document.completeness",
                        detail="this copy looks incomplete: made up")]

    monkeypatch.setattr(graph, "completeness_qa", note)
    record = ingest_report(pdf, FakeEngine(tg.full_script()),
                           out_dir=tmp_path / "out", report_id="SYN")

    assert len(seen["texts"]) == record.document.n_pages
    assert seen["labels"] and all(isinstance(k, int) for k in seen["labels"])
    assert [e.detail for e in record.qa
            if e.where == "document.completeness"] == [
        "this copy looks incomplete: made up"]

"""Tests for the WP0 corpus harness.

Two halves, on purpose:

- The **label mapping** is pure data and always runs. Every machine and CI can
  check that the vocabulary is total over the raw strings and lands only in
  :data:`LABELS`; a mapping hole would silently drop pages out of every later
  score.
- The **loaders** need the private corpus, which is gitignored and lives on the
  owner's machine only, so they ``skip`` cleanly when it is absent. They are
  exercised on R36, a public report.

Run::

    .venv/Scripts/python -m pytest module_work/report_ingest_harness/tests -q
"""

from __future__ import annotations

import pytest

from module_work.report_ingest_harness import corpus, labels

# -- the label mapping: synthetic, always runs ------------------------------

#: Every ``page_type`` string the hand-label spreadsheet used on 2026-09-16.
#: Pinned here so a mapping that stops covering the sheet fails loudly rather
#: than at scoring time.
KNOWN_RAW = (
    "main report narrative", "boring log", "test pit log", "cpt log",
    "dcp log", "lab testing", "subsurface drainage testing", "calculation",
    "appended report", "field photos", "appendix cover page",
    "figures or tables cover page", "figures or tables cover sheet",
    "figures or reports cover page", "cover page", "cover letter",
    "table of contents", "figure", "test location plan",
    "subsurface profile", "informational appendix content",
    "other or unknown", "other appendix table",
)


def test_mapping_is_total_over_the_known_raw_strings():
    missing = [raw for raw in KNOWN_RAW if raw not in labels.RAW_TO_LABEL]
    assert missing == []
    assert len(labels.RAW_TO_LABEL) == len(KNOWN_RAW) == 23


def test_mapping_lands_only_in_the_label_vocabulary():
    stray = sorted(set(labels.RAW_TO_LABEL.values()) - set(labels.LABELS))
    assert stray == []


def test_every_label_is_reachable():
    # A label nothing maps to is either a mapping hole or dead vocabulary.
    unused = sorted(set(labels.LABELS) - set(labels.RAW_TO_LABEL.values()))
    assert unused == []


def test_labels_are_unique_and_include_the_two_added_ones():
    assert len(set(labels.LABELS)) == len(labels.LABELS) == 18
    assert "field_test" in labels.LABELS
    assert "letter" in labels.LABELS


def test_to_label_normalises_case_and_spacing():
    assert labels.to_label("  Boring   Log ") == "boring_log"
    assert labels.to_label("LAB TESTING") == "lab_test"


def test_unknown_raw_string_raises_and_names_itself():
    with pytest.raises(labels.UnknownPageType) as excinfo:
        labels.to_label("interpretive dance log")
    assert "interpretive dance log" in str(excinfo.value)


def test_page_label_is_zero_based():
    pl = labels.PageLabel(page0=0, raw="cover page", label="cover")
    assert pl.page0 == 0


# -- the loaders: need the private corpus ----------------------------------

needs_raw = pytest.mark.skipif(
    not corpus.raw_available(),
    reason="the report-ingest corpus is gitignored and not on this machine")

PUBLIC_ID = "R36"          # a public report, so it is safe to name here


@needs_raw
def test_manifest_lists_the_whole_corpus():
    reports = corpus.list_reports()
    assert [r.id for r in reports] == [f"R{n:02d}" for n in range(1, 39)]
    assert sum(r.pages for r in reports) == 7829


@needs_raw
def test_report_info_never_prints_the_private_name():
    info = corpus.report(PUBLIC_ID)
    assert info.private_name                      # it IS loaded
    assert info.private_name not in repr(info)    # and never displayed
    assert info.is_public


@needs_raw
def test_pdf_path_is_named_by_id_only():
    assert corpus.pdf_path(PUBLIC_ID).name == f"{PUBLIC_ID}.pdf"


@needs_raw
def test_missing_id_is_an_error_not_a_guess():
    with pytest.raises(KeyError):
        corpus.report("R99")


@needs_raw
def test_truncated_di_result_reads_as_unavailable():
    # R17's export is truncated; the harness must say "no DI", not raise.
    assert corpus.has_di("R17") is False
    assert corpus.di_page_count("R17") is None
    assert corpus.load_di("R17") is None


@needs_raw
def test_open_report_without_di_reads_the_pdf_text_layer():
    info = corpus.report(PUBLIC_ID)
    with corpus.open_report(PUBLIC_ID, di="none") as doc:
        summaries = doc.page_map()
    assert len(summaries) == info.pages
    assert {s.page for s in summaries} == set(range(info.pages))


@needs_raw
def test_open_report_auto_falls_back_when_there_is_no_di_result():
    assert not corpus.has_di(PUBLIC_ID)
    with pytest.warns(RuntimeWarning, match="no usable Azure DI result"):
        doc = corpus.open_report(PUBLIC_ID, di="auto")
    with doc:
        counts = corpus.text_source_counts(doc)
    assert counts == {"pdf_text": corpus.report(PUBLIC_ID).pages}


@needs_raw
def test_bad_di_mode_is_rejected():
    with pytest.raises(ValueError, match="di must be one of"):
        corpus.open_report(PUBLIC_ID, di="sometimes")


@needs_raw
def test_pages_of_narrows_a_layout_to_the_pages_it_is_given():
    class FakeLayout:
        name = "azure_di"

        def covers(self, index):
            return index < 10

        def extract(self, page, index, words=False):
            return ("lines", index, words)

    narrowed = corpus._PagesOf(FakeLayout(), [1, 3, 99])
    assert narrowed.pages == [1, 3]        # 99 is outside the layout
    assert narrowed.covers(1) and not narrowed.covers(2)
    assert narrowed.extract(None, 3, words=True) == ("lines", 3, True)


@needs_raw
@pytest.mark.skipif(not labels.labels_available(),
                    reason="the hand-label spreadsheet is not on this machine")
def test_every_mapped_sheet_has_one_row_per_page():
    for match in labels.sheet_map():
        if match.rid is None:
            continue
        assert match.rows_match, match.describe()


@needs_raw
@pytest.mark.skipif(not labels.labels_available(),
                    reason="the hand-label spreadsheet is not on this machine")
def test_labels_for_a_mapped_report_are_in_page_order_and_mapped():
    rid = labels.mapped_ids()[0]
    page_labels = labels.labels_for(rid)
    assert len(page_labels) == corpus.report(rid).pages
    assert [p.page0 for p in page_labels] == list(range(len(page_labels)))
    assert all(p.label in labels.LABELS for p in page_labels)


@needs_raw
@pytest.mark.skipif(not labels.labels_available(),
                    reason="the hand-label spreadsheet is not on this machine")
def test_label_counts_cover_every_mapped_page():
    counts = labels.label_counts()
    assert set(counts) == set(labels.LABELS)
    expected = sum(corpus.report(rid).pages for rid in labels.mapped_ids())
    assert sum(counts.values()) == expected


# -- the WP1 scorer ---------------------------------------------------------
#
# Synthetic: the scorer is arithmetic over (hand label, predicted role) pairs
# and needs no corpus to be checked.

from module_work.report_ingest_harness import measure_wp1_labels as wp1  # noqa: E402


def _scored(pairs):
    s = wp1.Scores()
    for i, (hand, pred) in enumerate(pairs):
        s.add("R99", hand, pred, i, "text", "a heading", {"rule": "test"})
    return s


def test_rates_are_precision_recall_f1_and_support():
    s = _scored([("boring_log", "boring_log")] * 8
                + [("boring_log", "lab_test")] * 2      # 2 false negatives
                + [("narrative", "boring_log")] * 2)    # 2 false positives
    precision, recall, f1, support = s.rates("boring_log")
    assert support == 10
    assert precision == pytest.approx(8 / 10)
    assert recall == pytest.approx(0.8)
    assert f1 == pytest.approx(0.8)


def test_a_label_nothing_predicted_scores_zero_not_an_error():
    s = _scored([("narrative", "narrative")])
    assert s.rates("cpt_log") == (0.0, 0.0, 0.0, 0)


def test_the_gate_needs_both_rates_on_every_gated_role():
    perfect = [(g, g) for g in wp1.GATED for _ in range(10)]
    assert _scored(perfect).passes()
    # One gated role at 0.8 recall fails the whole gate.
    slipped = list(perfect) + [("lab_test", "other")] * 3
    assert not _scored(slipped).passes()


def test_only_gated_disagreements_are_collected_as_misses():
    s = _scored([("figure", "plan"),                 # neither is gated
                 ("narrative", "figure"),            # hand is gated
                 ("other", "calculation"),           # prediction is gated
                 ("lab_test", "lab_test")])          # agreement
    assert [(m["hand"], m["pred"]) for m in s.misses] == [
        ("narrative", "figure"), ("other", "calculation")]


def test_accuracy_counts_every_page():
    s = _scored([("narrative", "narrative"), ("narrative", "other"),
                 ("lab_test", "lab_test"), ("lab_test", "lab_test")])
    assert s.n_pages == 4
    assert s.accuracy == pytest.approx(0.75)
    assert s.per_report["R99"] == [3, 4]


def test_the_ledger_section_never_carries_a_page_heading():
    """The heading is the firm's title block; the ledger is public."""
    s = _scored([("narrative", "boring_log")])
    rounds = [{"round": 1, "date": "2026-09-16", "note": "n",
               "accuracy": 0.5, "gated": {g: [1.0, 1.0] for g in wp1.GATED}}]
    text = "\n".join(wp1.build_report(s, rounds, "n", wp1.GATED))
    assert "a heading" not in text
    assert "rule=" in text and "hand=narrative" in text

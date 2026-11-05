"""BACKLOG RI-94: a bound document is told to the narrative BY ITS KIND.

Only an ``appended_prior_report`` is a previous investigation. A ``volume``
or a ``data_report`` bound in is THIS investigation's own: a Volume 1 data
report bound behind its Volume 2 holds the report's own borings, so they
are counted with the report's and it is no previous investigation. Every
bound range here is made up.
"""

from __future__ import annotations

import pytest

from report_ingest.bound import (
    BOUND_KINDS, CURRENT_KINDS, bound_kind, is_current_volume,
    narrative_block,
)

PRIOR = {"pages": "40-61", "title": "Made-up Earlier Study",
         "firm": "Made-up Older Firm", "date": "2011",
         "kind": "appended_prior_report"}
VOLUME = {"pages": "62-90", "title": "Made-up Volume 1 Data Report",
          "firm": "Made-up Firm", "date": "2024", "kind": "volume"}
DATA = {"pages": "91-120", "title": "Made-up Field Data", "kind":
        "data_report"}
OTHER = {"pages": "121-130", "title": "Made-up Specification",
         "kind": "other"}

PREVIOUS_WORDS = "IS a previous investigation of this site"
CURRENT_WORDS = "ARE this report's"


class TestTheKind:

    def test_only_volumes_and_data_reports_are_this_investigation(self):
        assert set(CURRENT_KINDS) == {"volume", "data_report"}
        assert set(CURRENT_KINDS) < set(BOUND_KINDS)
        assert is_current_volume(VOLUME) and is_current_volume(DATA)
        assert not is_current_volume(PRIOR)
        assert not is_current_volume(OTHER)

    def test_a_row_with_no_kind_is_what_triage_answers_by_default(self):
        assert bound_kind({"pages": "4-9"}) == "appended_prior_report"
        assert bound_kind({"pages": "4-9", "kind": "made-up"}) == "other"


class TestWhatTheNarrativeIsToldByKind:

    def test_a_prior_report_is_a_previous_investigation(self):
        text = narrative_block([PRIOR])

        assert ("pages 40-61: Made-up Earlier Study - Made-up Older Firm - "
                "2011") in text
        assert PREVIOUS_WORDS in text
        assert "previousInvestigationCount" in text
        assert CURRENT_WORDS not in text

    @pytest.mark.parametrize("row", [VOLUME, DATA])
    def test_a_volume_of_this_investigation_is_counted_with_it(self, row):
        text = narrative_block([row])

        assert "of THIS investigation)" in text
        assert CURRENT_WORDS in text
        assert "boringCount" in text and "boringDictionary" in text
        assert "are NOT previous investigations" in text
        # The old wording, every bound document a previous investigation,
        # is not said of a current volume.
        assert PREVIOUS_WORDS not in text
        assert "do NOT count their borings" not in text

    def test_another_document_is_neither(self):
        text = narrative_block([OTHER])

        assert "(another document, not a previous investigation)" in text
        assert "unless this report's prose" in text
        assert PREVIOUS_WORDS not in text
        assert CURRENT_WORDS not in text

    def test_prior_reports_alone_get_the_words_they_always_got(self):
        """Every bound document an earlier investigation: master's text,
        byte for byte, so today's saved readings stand."""
        text = narrative_block([PRIOR, {**PRIOR, "pages": "131-140",
                                        "kind": ""}])
        assert text.splitlines()[-1].startswith(
            "So: do NOT count their borings, test pits or laboratory tests")
        assert text.splitlines()[-1].endswith(
            "does not by itself make this one an outside report.")
        assert "(" not in "".join(text.splitlines()[1:3])

    def test_a_file_with_every_kind_says_each_rule_of_its_own_pages(self):
        text = narrative_block([PRIOR, VOLUME, DATA, OTHER])

        assert "previous investigations (pages 40-61)" in text
        assert "THIS investigation (pages 62-90, 91-120)" in text
        assert "other documents bound in (pages 121-130)" in text
        assert text.count("outsideProject") == 1

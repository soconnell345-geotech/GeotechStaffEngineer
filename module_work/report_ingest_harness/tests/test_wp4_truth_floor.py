"""The floor under the WP4 scorecard: a flawless reading must score 100 %.

No model. Each hand answer is turned straight into the record's own
``GeneralFacts`` and ``NaturalHazardFacts`` -- a reader that got everything
right -- and scored against the file it came from. Every report must come back
at full recall and full precision.

WHY THIS IS THE TEST THAT MATTERS. A scorer is a measuring instrument, and an
instrument that reads low on a perfect specimen makes a good reader look bad
and a prompt change look like an improvement. This one has already caught the
real thing: the first draft of the record model GUESSED the vocabularies for
``propertyType``, ``projectPhase`` and ``liquefactionPotential`` and guessed
every one of them wrong, so the model REFUSED the hand's own answers and the
scorer marked seven fields wrong on all eight reports. The values in
``report_ingest.model`` are now the ones the hand actually uses, and this file
is what stops that happening again.

It also proves the truth files themselves load: that every field name is one
of the owner's, that every value is one the record can hold, and that
``_alternates`` and ``_skip`` are shaped the way the scorer reads them.

It SKIPS when the gitignored raw folder is absent, which is everywhere but the
owner's machine. ``report_ingest/tests/test_narrative_scoring.py`` walks the
same rules on synthetic answers, so CI still covers them.

PRIVACY. Report IDs and field names. No assertion, message or skip reason here
names a project, a firm, a place or a file.
"""

from __future__ import annotations

import json

import pytest

from module_work.report_ingest_harness import measure_wp4_narrative as m
from report_ingest.model import (
    GENERAL_FIELDS, GeneralFacts, NATURAL_HAZARD_FIELDS, NaturalHazardFacts,
    SUMMARY_FIELDS, SUMMARY_WORD_LIMITS,
)
from report_ingest.narrative_scoring import (
    alternates_for, score_narrative, skipped_fields,
)

pytestmark = pytest.mark.skipif(
    not m.truth_files(),
    reason="the hand answers are gitignored and are not on this machine")


def _truths():
    return m.load_truth()


def _ids():
    return [t["id"] for t in _truths()]


def _perfect(truth):
    """The reading a reader would return if it read the report perfectly."""
    skip = skipped_fields(truth)
    general = {name: value for name, value in (truth.get("general") or {}).items()
               if name in GENERAL_FIELDS and value is not None
               and name not in skip}
    hazards = {name: value
               for name, value in (truth.get("natural_hazards") or {}).items()
               if name in NATURAL_HAZARD_FIELDS and value is not None
               and name not in skip}
    return GeneralFacts(**general), NaturalHazardFacts(**hazards)


@pytest.mark.parametrize("truth", _truths(), ids=_ids())
def test_a_flawless_reading_scores_everything(truth):
    general, hazards = _perfect(truth)
    score = score_narrative(truth, general, hazards, report=truth["id"])

    assert score.recall.found == score.recall.total, score.recall.misses[:5]
    assert score.precision.found == score.precision.total, \
        score.precision.misses[:5]
    assert score.agreement.found == score.agreement.total
    assert score.recall.total >= 20, "a hand answer this thin is a mistake"


@pytest.mark.parametrize("truth", _truths(), ids=_ids())
def test_every_hand_answer_is_a_value_the_record_can_hold(truth):
    """The record model must not refuse the owner's own vocabulary."""
    _perfect(truth)          # constructing it IS the assertion


@pytest.mark.parametrize("truth", _truths(), ids=_ids())
def test_the_field_names_are_the_owners_and_nothing_else(truth):
    known = set(GENERAL_FIELDS) | set(NATURAL_HAZARD_FIELDS)
    for section in ("general", "natural_hazards"):
        unknown = sorted(set(truth.get(section) or {}) - known)
        assert not unknown, f"{section} names fields the schema has not: " \
                            f"{unknown}"


@pytest.mark.parametrize("truth", _truths(), ids=_ids())
def test_the_alternates_and_skips_are_shaped_as_the_scorer_reads_them(truth):
    known = set(GENERAL_FIELDS) | set(NATURAL_HAZARD_FIELDS)
    for name in (truth.get("_alternates") or {}):
        assert name in known, f"_alternates names an unknown field: {name}"
        assert alternates_for(truth, name), \
            f"_alternates[{name}] is empty; drop the key instead"
    for name in skipped_fields(truth):
        assert name in known, f"_skip names an unknown field: {name}"


@pytest.mark.parametrize("truth", _truths(), ids=_ids())
def test_every_summary_the_hand_wrote_is_scored_for_presence(truth):
    general, hazards = _perfect(truth)
    score = score_narrative(truth, general, hazards, report=truth["id"])

    written = [name for name in SUMMARY_FIELDS
               if (truth.get("general") or {}).get(name)
               or (truth.get("natural_hazards") or {}).get(name)]
    assert score.summaries_present.total == len(
        [name for name in written if name not in skipped_fields(truth)])
    assert score.summaries_present.found == score.summaries_present.total


def test_the_word_limits_are_reported_against_the_hands_own_summaries():
    """The hand is the best case; where it runs long, so will a reader.

    Not a failure of anything here: it is the measurement that says whether
    the plan's word limits are livable. It is asserted only that the hand
    stays inside twice the limit, so a genuinely runaway answer still fails.
    """
    over = []
    for truth in _truths():
        for section in ("general", "natural_hazards"):
            for name, value in (truth.get(section) or {}).items():
                if name not in SUMMARY_FIELDS or not value:
                    continue
                words = len(str(value).split())
                limit = SUMMARY_WORD_LIMITS[name]
                if words > limit:
                    over.append((truth["id"], name, words, limit))
                assert words <= limit * 2, \
                    f"{truth['id']} {name}: {words} words for a {limit}-word " \
                    f"field is not a summary"
    # Recorded in the test's own output rather than asserted away.
    print(f"\nhand summaries over their limit: {over}")

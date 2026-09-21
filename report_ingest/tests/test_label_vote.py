"""The combiner the production graph and the scoring stage both call.

No document, no model, no planlens: the whole of :mod:`report_ingest.label_vote`
is arithmetic over hand-made voters, which is why it can be pinned exactly.
The point of the file is that ONE copy of each policy exists -- the last class
below sets the stage's own spelling against the graph's on the same votes and
demands the same answer.
"""

from __future__ import annotations

import json

import pytest

from report_ingest.label_vote import (
    DEFAULT_POLICY, LOG_LABELS, POLICIES, STRUCTURAL_RULES_WIN, PageChoice,
    Voter, combine, load_trust_table, neighbourhood, split_pages,
)


def rules(label, confidence=0.9):
    return Voter("rules", label, confidence)


def vision(label, confidence=0.9):
    return Voter("vision", label, confidence)


def template(family="Northgate Soils", confidence=0.95):
    return Voter("template", "", confidence, family)


TRUST = {
    # The rules are believed where they put a page in 'calculation', and
    # vision is believed where they put it in 'figure'.
    "calculation": {"pages": 40, "rules": 30, "vision": 10,
                    "winner": "rules"},
    "figure": {"pages": 20, "rules": 4, "vision": 16, "winner": "vision"},
}


class TestThePolicies:

    def test_structural_keeps_the_labels_the_rules_own(self):
        for label in STRUCTURAL_RULES_WIN:
            choice = combine(rules(label), vision("narrative"),
                             policy="structural")
            assert choice.label == label

    def test_structural_takes_vision_everywhere_else(self):
        choice = combine(rules("narrative"), vision("plan"),
                         policy="structural")
        assert choice.label == "plan"
        assert "structural" in choice.why

    def test_confidence_believes_whichever_said_so_more_firmly(self):
        assert combine(rules("narrative", 0.4), vision("plan", 0.9),
                       policy="confidence").label == "plan"
        assert combine(rules("narrative", 0.9), vision("plan", 0.4),
                       policy="confidence").label == "narrative"

    def test_confidence_ties_to_the_rules(self):
        choice = combine(rules("narrative", 0.7), vision("plan", 0.7),
                         policy="confidence")
        assert choice.label == "narrative"

    def test_trust_follows_the_table_per_label_class(self):
        assert combine(rules("calculation"), vision("narrative"),
                       policy="trust", trust_table=TRUST).label == \
            "calculation"
        assert combine(rules("figure"), vision("boring_log"),
                       policy="trust", trust_table=TRUST).label == \
            "boring_log"

    def test_trust_falls_to_vision_on_a_class_the_table_never_saw(self):
        """A class the in-sample reports never split on is not a rules win."""
        choice = combine(rules("photos"), vision("plan"), policy="trust",
                         trust_table=TRUST)
        assert choice.label == "plan"

    def test_rules_is_the_old_behaviour_and_consults_nothing(self):
        choice = combine(rules("narrative"), vision("plan"), policy="rules",
                         trust_table=TRUST)
        assert choice.label == "narrative" and choice.why == "rules"

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(ValueError, match="unknown label policy"):
            combine(rules("narrative"), vision("plan"), policy="vibes")

    def test_every_policy_answers_something_from_the_vocabulary(self):
        for policy in POLICIES:
            choice = combine(rules("toc"), vision("cover"), policy=policy,
                             trust_table=TRUST)
            assert choice.label in ("toc", "cover")
            assert choice.policy == policy

    def test_the_default_is_structural(self):
        assert DEFAULT_POLICY == "structural"
        assert combine(rules("narrative"), vision("plan")).label == "plan"


class TestTheAgreedFlag:

    def test_two_voters_saying_the_same_thing_agree(self):
        choice = combine(rules("plan", 0.6), vision("plan", 0.8))
        assert choice.agreed is True
        # Agreement does not lower either voter's own certainty.
        assert choice.confidence == pytest.approx(0.8)

    def test_two_voters_splitting_do_not(self):
        choice = combine(rules("plan", 0.6), vision("figure", 0.8))
        assert choice.agreed is False
        assert choice.label == "figure"
        assert choice.confidence == pytest.approx(0.8)

    def test_one_voter_agrees_with_itself(self):
        """A vision pass that did not run leaves nothing to disagree with."""
        choice = combine(rules("plan", 0.6))
        assert choice.agreed is True and choice.label == "plan"
        assert [v.name for v in choice.voters] == ["rules"]

    def test_a_missing_confidence_is_zero_and_not_a_crash(self):
        choice = combine(rules("plan", None), vision("plan", None))
        assert choice.agreed is True and choice.confidence == 0.0

    def test_a_confidence_out_of_range_is_clipped(self):
        choice = combine(rules("plan", 3.5), vision("figure", -1.0))
        assert 0.0 <= choice.confidence <= 1.0


class TestTheTemplateVoter:

    def test_it_pulls_a_split_towards_the_log_class(self):
        """The rules see a log, the picture sees a graph; the form decides."""
        choice = combine(rules("boring_log", 0.5), vision("figure", 0.9),
                         template(), policy="structural")
        assert choice.label == "boring_log"
        assert "template" in choice.why and choice.agreed is False

    def test_a_page_both_voters_call_a_log_still_agrees(self):
        choice = combine(rules("cpt_log"), vision("cpt_log"), template())
        assert choice.agreed is True and choice.label == "cpt_log"

    def test_a_form_nobody_else_calls_a_log_is_a_disagreement(self):
        """Nothing to break the tie towards, so the page wants a second look."""
        choice = combine(rules("lab_test"), vision("lab_test"), template(),
                         policy="structural")
        assert choice.label == "lab_test"
        assert choice.agreed is False

    def test_it_never_fires_under_the_rules_policy(self):
        choice = combine(rules("figure"), vision("boring_log"), template(),
                         policy="rules")
        assert choice.label == "figure" and choice.agreed is False

    def test_its_vote_is_a_family_and_not_a_label(self):
        choice = combine(rules("boring_log"), vision("figure"), template())
        (row,) = [v for v in choice.to_dict()["voters"]
                  if v["voter"] == "template"]
        assert row["label"] == "" and row["family"] == "Northgate Soils"
        assert row["confidence"] == pytest.approx(0.95)

    def test_the_four_log_labels_are_the_class_it_votes_for(self):
        assert set(LOG_LABELS) == {"boring_log", "test_pit_log", "cpt_log",
                                   "dcp_log"}


class TestWhatTheGraphAsksNext:

    def test_split_pages_are_the_ones_that_did_not_agree(self):
        choices = [combine(rules("plan"), vision("plan"), page=0),
                   combine(rules("plan"), vision("figure"), page=1),
                   combine(rules("toc"), vision("cover"), page=2)]
        assert split_pages(choices) == [1, 2]

    def test_a_neighbourhood_is_the_page_and_two_either_side(self):
        assert neighbourhood([10], n_pages=40) == [8, 9, 10, 11, 12]

    def test_it_never_runs_off_the_end_of_the_document(self):
        assert neighbourhood([0, 21], n_pages=22) == [0, 1, 2, 19, 20, 21]

    def test_overlapping_neighbourhoods_are_one_set(self):
        assert neighbourhood([10, 11], n_pages=40) == [8, 9, 10, 11, 12, 13]


class TestTheTrustTable:

    def test_a_dict_is_taken_as_it_stands(self):
        assert load_trust_table(TRUST)["calculation"]["winner"] == "rules"

    def test_the_vote_stage_s_own_file_shape_is_read(self, tmp_path):
        path = tmp_path / "trust_table.json"
        path.write_text(json.dumps({"learned_on": ["R36"], "table": TRUST}),
                        encoding="utf-8")
        assert load_trust_table(str(path))["figure"]["winner"] == "vision"

    def test_a_missing_file_is_an_empty_table_and_not_an_error(self, tmp_path):
        assert load_trust_table(str(tmp_path / "nothing.json")) == {}
        assert load_trust_table(None) == {}

    def test_a_cell_with_no_winner_is_dropped(self):
        assert load_trust_table({"plan": {"pages": 3}}) == {}


class TestTheStageAndTheGraphAgree:
    """One copy of the arithmetic, checked from both of its callers."""

    def test_every_policy_gives_the_same_label_on_the_same_votes(self):
        from report_ingest.vote import PageVote
        from report_ingest.vote import combine as stage_combine

        cases = [
            ("narrative", 0.8, "plan", 0.9),
            ("calculation", 0.55, "narrative", 0.95),
            ("figure", 0.9, "boring_log", 0.4),
            ("appended_report", 0.3, "lab_test", 0.99),
            ("other", 0.7, "photos", 0.7),
        ]
        for page, (rule, rc, seen, vc) in enumerate(cases):
            vote = PageVote(report="R36", page=page, rules=rule, vision=seen,
                            rules_confidence=rc, vision_confidence=vc)
            for policy in ("trust", "structural", "confidence"):
                assert stage_combine(vote, policy, TRUST) == combine(
                    rules(rule, rc), vision(seen, vc), policy=policy,
                    trust_table=TRUST, page=page).label

    def test_the_stage_scores_three_policies_and_the_graph_knows_four(self):
        from report_ingest import vote as stage

        assert set(stage.POLICIES) < set(POLICIES)
        assert "rules" in POLICIES and "rules" not in stage.POLICIES

    def test_a_page_choice_round_trips_through_json(self):
        choice = combine(rules("boring_log", 0.5), vision("figure", 0.9),
                         template(), policy="structural", page=7)
        again = json.loads(json.dumps(choice.to_dict()))
        assert again["page"] == 7 and again["agreed"] is False
        assert [v["voter"] for v in again["voters"]] == [
            "rules", "vision", "template"]
        assert isinstance(choice, PageChoice)

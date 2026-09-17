"""Pass 0c offline: the brief, the tools, the budget, the changes applied.

Every test runs the real loop over planlens' synthetic report with a scripted
engine, so what is exercised is the pass, not a mock of it.
"""

from __future__ import annotations

import pytest

from report_ingest.label_review import (
    LABEL_DEFINITIONS, LabelChange, ReviewFindings, StructureEntry,
    UnresolvedPage, budget_for, review_labels,
)
from report_ingest.tests.fake_engine import FakeEngine, ScriptExhausted


@pytest.fixture(scope="module")
def synthetic():
    from planlens.document import open_document
    from planlens.testing import build_synthetic_report

    gt = build_synthetic_report()
    doc = open_document(gt.pdf, name="synthetic")
    try:
        yield doc, gt
    finally:
        doc.close()


@pytest.fixture(scope="module")
def roles(synthetic):
    from planlens.document.roles import page_roles
    doc, _ = synthetic
    return page_roles(doc)


def _findings(changes=(), structure=(), unresolved=(), notes=""):
    return ReviewFindings(
        changes=[LabelChange(**c) for c in changes],
        structure=[StructureEntry(**s) for s in structure],
        unresolved=[UnresolvedPage(**u) for u in unresolved],
        notes=notes)


def _script(*turns, findings=None):
    """A loop script: the turns, then the model stopping, then its findings.

    The loop always makes one more call than the script's tool turns (the
    model answering without tools), and :func:`review_labels` then asks for
    the findings in a separate structured call. Spelling that out here keeps
    every test honest about how many calls the pass really makes.
    """
    return list(turns) + [{"text": "done looking"},
                          {"final": findings if findings is not None
                           else _findings()}]


def _change(page, to_label, from_label="other", reason="it says so",
            evidence="read_page"):
    return {"page": page, "from_label": from_label, "to_label": to_label,
            "reason": reason, "evidence": evidence}


# -- the vocabulary ---------------------------------------------------------

def test_the_vocabulary_is_exactly_planlens_roles():
    from planlens.document.roles import ROLES
    assert set(LABEL_DEFINITIONS) == set(ROLES), (
        "a label the review can return but planlens cannot produce, or the "
        "other way round, would score as a miss on every page")


def test_every_label_carries_a_definition_that_separates_it():
    for name, text in LABEL_DEFINITIONS.items():
        assert len(text) > 20, f"{name} needs a definition, not a restatement"


def test_the_exploration_logs_say_they_may_be_plotted_not_tabulated():
    # The cost checkpoint's largest error class: an exploration's own results
    # plotted as a chart, called "figure". Fifteen of the review's nineteen
    # wrong changes were this or the neighbouring sounding confusion.
    for name in ("boring_log", "test_pit_log", "cpt_log", "dcp_log"):
        text = LABEL_DEFINITIONS[name].lower()
        assert "chart" in text or "plotted" in text, (
            f"{name} must say it can be a plot, not only a form")
        assert "identifier" in text, (
            f"{name} must say it belongs to ONE named exploration")
    figure = LABEL_DEFINITIONS["figure"].lower()
    assert "exploration" in figure, (
        "the figure definition must rule out an exploration's own results")


def test_the_review_rule_sends_a_plotted_log_to_its_exploration():
    from report_ingest.label_review import REVIEW_SYSTEM

    # The prompt is hard-wrapped, so compare on collapsed whitespace.
    text = " ".join(REVIEW_SYSTEM.lower().split())
    assert "never figure" in text
    assert "blows per increment" in text, (
        "the rule must say how to tell a DCP from a CPT, which is where four "
        "of the checkpoint's wrong changes went")


# -- the budget -------------------------------------------------------------

def test_the_budget_is_a_floor_of_sixty_and_a_quarter_of_the_pages():
    assert budget_for(20) == 60
    assert budget_for(240) == 60
    assert budget_for(456) == 114
    assert budget_for(729) == 182


def test_the_loop_stops_spending_when_the_budget_is_gone(synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine(_script(*([{"tools": [("read_page", {"page": 7}),
                                    ("read_page", {"page": 8})]}] * 3)))
    review = review_labels(doc, roles, profile={}, budget=3, engine=engine)
    assert review.budget == 3
    assert review.tool_calls == 3, "never more calls than the budget"
    assert review.stopped_on_budget is True
    refused = [r for r in engine.tool_results() if r["is_error"]]
    assert any("budget spent" in r["content"] for r in refused)


def test_a_review_that_spends_nothing_is_not_flagged_as_out_of_budget(
        synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine(_script())
    review = review_labels(doc, roles, profile={}, engine=engine)
    assert review.tool_calls == 0
    assert review.stopped_on_budget is False
    assert review.model_calls == 2, "one loop turn, then the findings call"


def test_the_loop_cannot_spin_for_ever_on_a_model_that_never_stops(synthetic,
                                                                   roles):
    doc, _ = synthetic
    # The model keeps calling tools and never answers. The loop is cut at
    # max_model_calls and the findings are asked for anyway.
    engine = FakeEngine([{"tools": [("outline", {})]}] * 3
                        + [{"final": _findings()}])
    review = review_labels(doc, roles, profile={}, engine=engine,
                           max_model_calls=3)
    assert review.model_calls == 4, "three loop turns, then the findings call"
    assert review.tool_calls == 3


# -- the tools --------------------------------------------------------------

def test_read_page_returns_the_page_as_the_readers_see_it(synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine(_script({"tools": [("read_page", {"page": 7})]}))
    review_labels(doc, roles, profile={}, engine=engine)
    result = engine.tool_results()[0]
    assert result["is_error"] is False
    assert "=== page 7" in result["content"]
    assert "BORING" in result["content"].upper()


def test_render_page_and_contact_sheet_come_back_as_pictures(synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine(_script({"tools": [
        ("render_page", {"page": 5}), ("contact_sheet", {"start_page": 0})]}))
    review_labels(doc, roles, profile={}, engine=engine)
    for result in engine.tool_results():
        kinds = [b["type"] for b in result["content"]]
        assert kinds == ["text", "image"]
        assert result["content"][1]["png"][:4] == b"\x89PNG"


def test_a_page_outside_the_document_is_an_answer_not_a_crash(synthetic,
                                                              roles):
    doc, _ = synthetic
    engine = FakeEngine(_script({"tools": [("read_page", {"page": 9999})]}))
    review = review_labels(doc, roles, profile={}, engine=engine)
    result = engine.tool_results()[0]
    assert result["is_error"] is True
    assert "outside this document" in result["content"]
    assert review.tool_calls == 1, "a mistake still costs a call"


def test_an_unknown_tool_is_reported_back_with_the_real_names(synthetic,
                                                              roles):
    doc, _ = synthetic
    engine = FakeEngine(_script({"tools": [("read_the_whole_thing", {})]}))
    review_labels(doc, roles, profile={}, engine=engine)
    result = engine.tool_results()[0]
    assert result["is_error"] is True
    assert "read_page" in result["content"]


def test_a_contact_sheet_snaps_to_its_sheet_boundary(synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine(_script(
        {"tools": [("contact_sheet", {"start_page": 13})]}))
    review_labels(doc, roles, profile={}, engine=engine)
    assert "pages 0 to " in engine.tool_results()[0]["content"][0]["text"]


# -- the brief ---------------------------------------------------------------

def test_the_brief_names_the_vocabulary_the_weak_spots_and_the_budget(
        synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine(_script())
    review_labels(doc, roles, profile={"workflow": "standard"}, engine=engine)
    sent = engine.calls[0]["messages"][0]["content"][0]["text"]
    assert "THE LABEL VOCABULARY" in sent
    assert "THE RULES' WEAK SPOTS: CHECK THESE FIRST" in sent
    assert "WHAT TRIAGE DECIDED THIS DOCUMENT IS" in sent
    assert "tool-call budget is 60" in sent
    for label in LABEL_DEFINITIONS:
        assert label in sent


def test_the_tools_are_offered_on_the_loop_but_not_on_the_findings_call(
        synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine(_script({"tools": [("outline", {})]}))
    review_labels(doc, roles, profile={}, engine=engine)
    assert set(engine.calls[0]["tools"]) == {"read_page", "render_page",
                                             "contact_sheet", "outline"}
    assert engine.calls[-1]["tools"] == []
    assert engine.calls[-1]["output_format"] is ReviewFindings


# -- what comes out ----------------------------------------------------------

def test_final_labels_are_the_rules_with_the_changes_applied(synthetic, roles):
    doc, gt = synthetic
    engine = FakeEngine(_script(findings=_findings(
        changes=[_change(10, "figure", from_label="photos")])))
    review = review_labels(doc, roles, profile={}, engine=engine)
    assert len(review.final_labels) == gt.n_pages
    assert review.final_labels[10] == "figure"
    unchanged = [p for p in review.final_labels if p != 10]
    assert all(review.final_labels[p] == review.rules_labels[p]
               for p in unchanged)


def test_a_change_records_the_label_the_rules_actually_gave(synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine(_script(findings=_findings(
        changes=[_change(10, "figure", from_label="narrative")])))
    review = review_labels(doc, roles, profile={}, engine=engine)
    # The model misremembered the old label; the rules are authoritative.
    assert review.changes[0]["from"] == review.rules_labels[10]
    assert review.changes[0]["to"] == "figure"
    assert review.changes[0]["evidence"] == "read_page"


def test_a_change_to_a_label_outside_the_vocabulary_is_refused(synthetic,
                                                               roles):
    doc, _ = synthetic
    engine = FakeEngine(_script(findings=_findings(
        changes=[_change(10, "boring_log_continuation")])))
    review = review_labels(doc, roles, profile={}, engine=engine)
    assert review.changes == []
    assert "not in the vocabulary" in review.rejected_changes[0]["rejected"]
    assert review.final_labels[10] == review.rules_labels[10]


def test_a_change_to_a_page_that_does_not_exist_is_refused(synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine(_script(
        findings=_findings(changes=[_change(900, "figure")])))
    review = review_labels(doc, roles, profile={}, engine=engine)
    assert review.changes == []
    assert "no such page" in review.rejected_changes[0]["rejected"]


def test_a_change_that_changes_nothing_is_refused(synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine(_script())
    review = review_labels(doc, roles, profile={}, engine=engine)
    settled = review.rules_labels[7]
    engine2 = FakeEngine(_script(findings=_findings(
        changes=[_change(7, settled)])))
    review2 = review_labels(doc, roles, profile={}, engine=engine2)
    assert review2.changes == []
    assert "already has that label" in review2.rejected_changes[0]["rejected"]


def test_the_same_page_is_only_changed_once(synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine(_script(findings=_findings(
        changes=[_change(10, "figure"), _change(10, "plan")])))
    review = review_labels(doc, roles, profile={}, engine=engine)
    assert review.final_labels[10] == "figure"
    assert len(review.changes) == 1
    assert "already changed once" in review.rejected_changes[0]["rejected"]


def test_structure_and_unresolved_come_through_and_serialize(synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine(_script(findings=_findings(
        structure=[{"title": "Appendix A - Boring Logs", "pages": "6-10"}],
        unresolved=[{"page": 13, "why": "the sheet names no test"}],
        notes="one appendix mixes logs and photographs")))
    review = review_labels(doc, roles, profile={}, engine=engine)
    blob = review.to_dict()
    assert blob["structure"][0]["pages"] == "6-10"
    assert blob["unresolved"][0]["page"] == 13
    assert blob["notes"].startswith("one appendix")
    assert set(blob["final_labels"]) == {str(p) for p in review.final_labels}


def test_the_review_reports_what_it_cost_on_this_report_alone(synthetic,
                                                              roles):
    doc, _ = synthetic
    engine = FakeEngine(_script({"tools": [("outline", {})]}))
    review = review_labels(doc, roles, profile={}, engine=engine)
    assert review.cost["calls"] == 3
    assert review.cost["input_tokens"] == 3000
    assert review.model == "fake-model"


def test_no_structured_findings_is_an_error_not_silent_agreement(synthetic,
                                                                 roles):
    doc, _ = synthetic
    engine = FakeEngine([{"text": "the labels all look fine to me"},
                         {"text": "still fine"}])
    with pytest.raises(RuntimeError, match="no structured findings"):
        review_labels(doc, roles, profile={}, engine=engine)


def test_the_script_runs_out_rather_than_inventing_a_turn(synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine([{"tools": [("outline", {})]}])
    with pytest.raises(ScriptExhausted):
        review_labels(doc, roles, profile={}, engine=engine)

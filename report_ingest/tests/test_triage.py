"""Pass 0b offline: the counted facts, the prompt it builds, the profile.

The document is planlens' own synthetic report (a cover, a contents list,
narrative, a figure, three appendices with logs, lab sheets and calculation
printouts, and a prior report bound in whole), so the facts these tests
assert are facts about a document whose every page is known.
"""

from __future__ import annotations

import pytest

from report_ingest.tests.fake_engine import FakeEngine
from report_ingest.triage import (
    DOCUMENT_TYPES, TOC_AGREEMENTS, WORKFLOWS, TriageFindings,
    document_facts, front_matter_text, outline_text, triage,
)


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


def _findings(**over):
    base = dict(
        document_type="geotechnical report", bound_together=[],
        has_narrative=True, has_logs=True, has_lab=True, has_calcs=True,
        languages=["English"], toc_agreement="matched", anomalies=[],
        workflow="standard", rationale="a whole report with every part.")
    base.update(over)
    return TriageFindings(**base)


# -- the facts Python counts ------------------------------------------------

def test_facts_count_the_document_rather_than_describe_it(synthetic, roles):
    doc, gt = synthetic
    facts = document_facts(doc, roles)
    assert facts.n_pages == gt.n_pages
    # A vector-text synthetic report: nothing scanned, nothing unreadable.
    assert facts.scan_fraction == 0.0
    assert facts.unreliable_text_fraction == 0.0
    assert facts.di_pages == 0
    assert sum(facts.role_counts.values()) == gt.n_pages
    assert sum(facts.kind_counts.values()) == gt.n_pages


def test_a_report_bound_inside_another_restarts_the_page_numbering(synthetic,
                                                                   roles):
    doc, _ = synthetic
    facts = document_facts(doc, roles)
    # The prior report bound in at page 15 starts its own "Page 1 of n".
    assert facts.page_numbering_restarts, (
        "the nested report's own numbering should read as a restart")
    assert all(0 <= p < facts.n_pages for p in facts.page_numbering_restarts)


def test_the_facts_prompt_states_every_number_it_measured(synthetic, roles):
    doc, _ = synthetic
    text = document_facts(doc, roles).as_prompt()
    for phrase in ("pages:", "scanned fraction:", "unreliable-text fraction:",
                   "printed page numbering restarts", "page kinds:",
                   "rule labels:"):
        assert phrase in text


def test_low_confidence_pages_are_listed_for_the_review_to_open(synthetic,
                                                               roles):
    doc, _ = synthetic
    facts = document_facts(doc, roles)
    assert all(0 <= p < facts.n_pages for p in facts.low_confidence_pages)
    assert facts.to_dict()["n_low_confidence_pages"] == len(
        facts.low_confidence_pages)


# -- what the model is given ------------------------------------------------

def test_front_matter_stops_at_the_end_of_the_contents(synthetic, roles):
    doc, _ = synthetic
    text = front_matter_text(doc, roles)
    assert "=== page 0" in text
    # The narrative begins at page 2; the front matter must not run into it.
    assert "=== page 4" not in text


def test_front_matter_is_cut_at_its_limit_and_says_so(synthetic, roles):
    doc, _ = synthetic
    text = front_matter_text(doc, roles, max_chars=200)
    assert "cut at the size limit" in text
    assert len(text) < 400


def test_the_outline_is_lines_a_model_reads_not_json(synthetic):
    doc, _ = synthetic
    from planlens.document.roles import document_outline
    text = outline_text(document_outline(doc))
    assert "DIVIDERS AND TABS" in text
    assert "APPENDIX A" in text.upper()
    assert not text.lstrip().startswith("{")


def test_triage_sends_the_ledger_the_outline_and_one_contact_sheet(synthetic,
                                                                   roles):
    doc, _ = synthetic
    engine = FakeEngine([{"final": _findings()}])
    triage(doc, roles, engine=engine)
    call = engine.calls[0]
    assert call["n_images"] == 1, "the first contact sheet goes with the call"
    assert call["output_format"] is TriageFindings
    sent = call["messages"][0]["content"][0]["text"]
    assert "FACTS ALREADY COUNTED FROM THE FILE" in sent
    assert "PAGE LEDGER, ONE LINE PER PAGE" in sent
    assert "FRONT MATTER, VERBATIM" in sent
    assert "p000" in sent and "p021" in sent


def test_a_long_ledger_is_cut_and_the_cut_is_declared(synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine([{"final": _findings()}])
    triage(doc, roles, engine=engine, max_ledger_chars=200)
    sent = engine.calls[0]["messages"][0]["content"][0]["text"]
    assert "ledger cut at the size limit" in sent


# -- the profile that comes out ---------------------------------------------

def test_the_profile_takes_its_numbers_from_python_not_the_model(synthetic,
                                                                 roles):
    doc, _ = synthetic
    facts = document_facts(doc, roles)
    engine = FakeEngine([{"final": _findings()}])
    profile = triage(doc, roles, engine=engine, facts=facts)
    assert profile.scan_fraction == facts.scan_fraction
    assert profile.unreliable_text_fraction == facts.unreliable_text_fraction
    assert profile.page_numbering_restarts == facts.page_numbering_restarts
    assert profile.facts is facts


def test_the_profile_carries_the_model_and_what_the_call_cost(synthetic,
                                                             roles):
    doc, _ = synthetic
    engine = FakeEngine([{"final": _findings()}], name="test-model")
    profile = triage(doc, roles, engine=engine)
    assert profile.model == "test-model"
    assert profile.cost["calls"] == 1
    assert profile.cost["input_tokens"] == 1000


def test_a_bound_together_document_survives_into_the_profile(synthetic, roles):
    doc, _ = synthetic
    engine = FakeEngine([{"final": _findings(
        workflow="multi_document",
        bound_together=[{"kind": "appended_prior_report", "pages": "14-18",
                         "title": "Former Owner Site Study"}])}])
    profile = triage(doc, roles, engine=engine)
    assert profile.workflow == "multi_document"
    assert profile.bound_together[0]["kind"] == "appended_prior_report"
    assert profile.summary_row()["bound_together"] == 1
    assert profile.to_dict()["bound_together"][0]["pages"] == "14-18"


def test_no_structured_answer_is_an_error_not_an_empty_profile(synthetic,
                                                              roles):
    doc, _ = synthetic
    engine = FakeEngine([{"text": "I would rather describe it in prose."}])
    with pytest.raises(RuntimeError, match="no structured answer"):
        triage(doc, roles, engine=engine)


# -- the schema the model is held to ----------------------------------------

def test_the_enumerations_are_the_owners_words():
    assert "geotechnical report" in DOCUMENT_TYPES
    assert "report appendix or figure(s)" in DOCUMENT_TYPES
    assert set(WORKFLOWS) == {"standard", "appendix_only", "partial",
                              "multi_document", "scanned", "needs_person"}
    assert set(TOC_AGREEMENTS) == {"matched", "partial", "none", "no_toc"}


def test_every_enumeration_is_named_in_the_schema_the_model_sees():
    schema = TriageFindings.model_json_schema()
    props = schema["properties"]
    for value in DOCUMENT_TYPES:
        assert value in props["document_type"]["description"]
    for value in WORKFLOWS:
        assert value in props["workflow"]["description"]
    for value in TOC_AGREEMENTS:
        assert value in props["toc_agreement"]["description"]


def test_the_findings_model_asks_for_no_counted_number():
    # scan_fraction and friends are measured, never requested: a model that
    # is asked to count 455 ledger lines will be nearly right, which is the
    # worst thing a scorecard field can be.
    fields = set(TriageFindings.model_fields)
    assert not fields & {"scan_fraction", "unreliable_text_fraction",
                         "page_numbering_restarts", "n_pages"}

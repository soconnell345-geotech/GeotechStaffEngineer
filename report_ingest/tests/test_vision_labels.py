"""The vision-first label pass offline: both modes, the gates, the budget.

Every test runs the real pass over planlens' synthetic report with a scripted
engine, so what is exercised is the pass -- the pictures it renders, the
prompt it builds, the shape it asks for and what Python refuses -- and not a
mock of it. No model is called and no credential is read.
"""

from __future__ import annotations

import pytest

from report_ingest.engine import strict_schema
from report_ingest.label_review import LABEL_DEFINITIONS
from report_ingest.tests.fake_engine import FakeEngine, ScriptExhausted
from report_ingest.vision_labels import (
    DEFAULT_DPI, MODES, VISION_LABELS, VisionPageAnswer, VisionSheetAnswer,
    classify_pages_by_vision, vision_system,
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


def _answer(page, label="narrative", confidence=0.9, reason="it says so"):
    return VisionPageAnswer(page=page, label=label, confidence=confidence,
                            reason=reason)


class _Loose:
    """An answer object that is NOT schema-validated.

    A strict schema stops a compliant provider returning a label outside the
    vocabulary or a confidence outside 0 to 1, and the Python gates exist for
    the provider that is not compliant. This is how a test reaches them.
    """

    def __init__(self, page, label, confidence=0.5, reason="because"):
        self.page, self.label = page, label
        self.confidence, self.reason = confidence, reason


class _LooseSheet:
    def __init__(self, entries):
        self.pages = list(entries)


def _page_script(pages, **over):
    return [{"final": _answer(p, **over)} for p in pages]


# -- the vocabulary and the prompt -------------------------------------------

def test_the_vocabulary_is_the_label_reviews_and_planlens_own():
    from planlens.document.roles import ROLES

    assert set(VISION_LABELS) == set(ROLES), (
        "a label this pass can return that planlens cannot produce, or the "
        "other way round, would read as a miss on every page carrying it")
    assert len(VISION_LABELS) == 18


def test_the_system_prompt_carries_every_definition_and_the_plotted_log_rule():
    for mode in MODES:
        text = vision_system(mode)
        for name, definition in LABEL_DEFINITIONS.items():
            assert name in text
            assert definition.split(",")[0][:30] in text
        flat = " ".join(text.lower().split())
        assert "never figure" in flat, (
            "an exploration's own results plotted as a chart is the largest "
            "error class the review measured; the rule has to be here too")
        assert "blows per increment" in flat


def test_the_sheet_prompt_says_one_answer_per_page_and_no_others():
    flat = " ".join(vision_system("sheet").lower().split())
    assert "every page on the sheet" in flat
    assert "not on the sheet" in flat
    page = " ".join(vision_system("page").lower().split())
    assert "one page at a time" in page


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown mode"):
        vision_system("thumbnail")


# -- the schema strict mode demands -------------------------------------------

@pytest.mark.parametrize("model", [VisionPageAnswer, VisionSheetAnswer])
def test_every_object_in_the_schema_is_closed_and_fully_required(model):
    """Prompter sends ``response_format`` with ``strict: true``, which refuses
    a schema whose objects allow extra properties or leave one optional."""
    schema = strict_schema(model)
    seen = 0

    def walk(node):
        nonlocal seen
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" or "properties" in node:
            seen += 1
            assert node.get("additionalProperties") is False
            assert sorted(node.get("required") or []) == sorted(
                node.get("properties") or {})
        for key, value in node.items():
            if key in ("properties", "$defs"):
                for child in value.values():
                    walk(child)
            elif key in ("items", "anyOf", "allOf", "oneOf"):
                walk(value)

    walk(schema)
    assert seen >= 1


def test_the_label_field_is_the_eighteen_and_nothing_else():
    field = strict_schema(VisionPageAnswer)["properties"]["label"]
    assert field["enum"] == list(VISION_LABELS)
    assert field["type"] == "string"


def test_the_schema_declares_no_keyword_strict_mode_would_refuse():
    """A minimum, a maximum or a maxLength gets the whole call rejected, and
    a rejected call returns nothing at all. Confidence is clipped in Python
    instead."""
    schema = strict_schema(VisionPageAnswer)
    for name in ("confidence", "reason", "page"):
        field = schema["properties"][name]
        assert not ({"minimum", "maximum", "maxLength", "minLength",
                     "exclusiveMinimum", "exclusiveMaximum"} & set(field))


# -- page mode ----------------------------------------------------------------

def test_page_mode_is_one_call_per_page_each_with_its_own_picture(synthetic):
    doc, gt = synthetic
    engine = FakeEngine(_page_script(range(gt.n_pages)))
    out = classify_pages_by_vision(doc, engine)

    assert engine.n_calls == gt.n_pages
    assert out.model_calls == gt.n_pages
    assert len(out.labels) == gt.n_pages
    assert out.pages_asked == gt.n_pages
    assert all(call["n_images"] == 1 for call in engine.calls)
    assert all(call["output_format"] is VisionPageAnswer
               for call in engine.calls)
    assert out.label_map[0] == "narrative"


def test_only_the_pages_asked_for_are_looked_at(synthetic):
    doc, _ = synthetic
    engine = FakeEngine(_page_script([1, 2, 3]))
    out = classify_pages_by_vision(doc, engine, pages="1-3")

    assert engine.n_calls == 3
    assert sorted(out.label_map) == [1, 2, 3]


def test_the_page_is_rendered_at_the_dpi_asked_for(synthetic):
    doc, _ = synthetic
    for dpi in (DEFAULT_DPI, 72.0):
        engine = FakeEngine(_page_script([4]))
        classify_pages_by_vision(doc, engine, pages=[4], dpi=dpi)
        body = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert f"rendered at {dpi:.0f} dpi" in body
        assert "page 4 of this document" in body


def test_the_default_dpi_is_the_one_a_vision_model_keeps():
    # A letter page at 100 dpi is 850 x 1100, just above the 768 px short
    # side a 4.1-class stack scales to; 72 dpi would hand it less than it is
    # willing to look at, and 200 dpi four times the bytes for no more pixels.
    assert DEFAULT_DPI == 100.0
    assert 8.5 * DEFAULT_DPI >= 768


def test_the_pass_never_sees_a_tool_and_asks_for_a_shape_every_time(synthetic):
    doc, _ = synthetic
    engine = FakeEngine(_page_script([0]))
    classify_pages_by_vision(doc, engine, pages=[0])

    assert engine.calls[0]["tools"] == []
    assert engine.calls[0]["output_format"] is VisionPageAnswer


# -- sheet mode ---------------------------------------------------------------

def test_sheet_mode_labels_every_page_on_the_sheet_in_one_call(synthetic):
    doc, _ = synthetic
    shown = [0, 1, 2, 3, 4, 5]
    engine = FakeEngine([{"final": VisionSheetAnswer(
        pages=[_answer(p, label="figure") for p in shown])}])
    out = classify_pages_by_vision(doc, engine, pages=shown, mode="sheet")

    assert engine.n_calls == 1, "six pages, one call"
    assert engine.calls[0]["n_images"] == 1
    assert engine.calls[0]["output_format"] is VisionSheetAnswer
    assert sorted(out.label_map) == shown
    assert set(out.label_map.values()) == {"figure"}
    assert out.unresolved == []


def test_a_sheet_carries_the_page_indexes_it_shows(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([{"final": VisionSheetAnswer(
        pages=[_answer(p) for p in (2, 3)])}])
    classify_pages_by_vision(doc, engine, pages=[2, 3], mode="sheet")
    body = engine.calls[0]["messages"][0]["content"][0]["text"]

    assert "contact sheet of pages 2, 3" in body
    assert "labelled" in body, "the legend tells the model how to read a tile"


def test_the_pages_are_split_into_sheets_of_the_size_asked_for(synthetic):
    doc, _ = synthetic
    shown = list(range(6))
    engine = FakeEngine([
        {"final": VisionSheetAnswer(pages=[_answer(p) for p in shown[:2]])},
        {"final": VisionSheetAnswer(pages=[_answer(p) for p in shown[2:4]])},
        {"final": VisionSheetAnswer(pages=[_answer(p) for p in shown[4:]])},
    ])
    out = classify_pages_by_vision(doc, engine, pages=shown, mode="sheet",
                                   sheet_pages=2)

    assert engine.n_calls == 3
    assert len(out.labels) == 6


def test_a_page_the_reply_left_out_is_unresolved_and_never_guessed(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([{"final": VisionSheetAnswer(
        pages=[_answer(p) for p in (0, 2)])}])
    out = classify_pages_by_vision(doc, engine, pages=[0, 1, 2], mode="sheet")

    assert sorted(out.label_map) == [0, 2]
    assert [u["page"] for u in out.unresolved] == [1]
    assert "left this page out" in out.unresolved[0]["why"]


def test_a_page_the_reply_invented_is_dropped_with_a_note(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([{"final": VisionSheetAnswer(
        pages=[_answer(0), _answer(1), _answer(99)])}])
    out = classify_pages_by_vision(doc, engine, pages=[0, 1], mode="sheet")

    assert sorted(out.label_map) == [0, 1]
    assert any(q["page"] == 99 and "not on this sheet" in q["note"]
               for q in out.qa)


def test_a_page_answered_twice_keeps_the_first_answer(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([{"final": _LooseSheet([
        _Loose(0, "plan"), _Loose(0, "profile")])}])
    out = classify_pages_by_vision(doc, engine, pages=[0], mode="sheet")

    assert out.label_map == {0: "plan"}
    assert any("twice" in q["note"] for q in out.qa)


# -- the outline flag ----------------------------------------------------------

def test_the_outline_rides_on_every_call_when_it_is_asked_for(synthetic):
    doc, _ = synthetic
    engine = FakeEngine(_page_script([0, 1]))
    classify_pages_by_vision(doc, engine, pages=[0, 1], outline_context=True)

    for call in engine.calls:
        body = call["messages"][0]["content"][0]["text"]
        assert "WHAT THIS DOCUMENT PRINTS ABOUT ITSELF" in body


def test_pure_vision_is_the_default_and_carries_no_outline(synthetic):
    doc, _ = synthetic
    engine = FakeEngine(_page_script([0]))
    out = classify_pages_by_vision(doc, engine, pages=[0])
    body = engine.calls[0]["messages"][0]["content"][0]["text"]

    assert "WHAT THIS DOCUMENT PRINTS ABOUT ITSELF" not in body
    assert out.outline_context is False


def test_the_outline_is_cut_at_the_size_limit_because_it_is_resent(synthetic):
    doc, _ = synthetic
    engine = FakeEngine(_page_script([0]))
    classify_pages_by_vision(doc, engine, pages=[0], outline_context=True,
                             max_context_chars=40)
    body = engine.calls[0]["messages"][0]["content"][0]["text"]

    assert "outline cut at the size limit" in body


# -- the Python gates ----------------------------------------------------------

def test_a_label_outside_the_vocabulary_becomes_other_with_a_note(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([{"final": _Loose(0, "boring_log_continuation")}])
    out = classify_pages_by_vision(doc, engine, pages=[0])

    assert out.label_map == {0: "other"}
    assert out.qa[0]["page"] == 0
    assert "not in the vocabulary" in out.qa[0]["note"]


@pytest.mark.parametrize("given,kept", [(1.4, 1.0), (-0.2, 0.0), (0.7, 0.7)])
def test_a_confidence_outside_zero_to_one_is_clipped(synthetic, given, kept):
    doc, _ = synthetic
    engine = FakeEngine([{"final": _Loose(0, "plan", confidence=given)}])
    out = classify_pages_by_vision(doc, engine, pages=[0])

    assert out.labels[0].confidence == pytest.approx(kept)
    clipped = [q for q in out.qa if "clipped" in q["note"]]
    assert bool(clipped) is (given != kept)


def test_a_confidence_that_is_not_a_number_is_read_as_zero(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([{"final": _Loose(0, "plan", confidence="very")}])
    out = classify_pages_by_vision(doc, engine, pages=[0])

    assert out.labels[0].confidence == 0.0
    assert any("not a number" in q["note"] for q in out.qa)


def test_an_answer_for_the_wrong_page_is_filed_where_it_was_asked(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([{"final": _answer(11, label="plan")}])
    out = classify_pages_by_vision(doc, engine, pages=[3])

    assert out.label_map == {3: "plan"}, "the page rendered is the page meant"
    assert any(q["page"] == 3 and "named page 11" in q["note"] for q in out.qa)


def test_no_structured_answer_is_an_unresolved_page_not_a_guess(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([{"text": "it looks like a boring log to me"}])
    out = classify_pages_by_vision(doc, engine, pages=[7])

    assert out.labels == []
    assert out.unresolved[0]["page"] == 7
    assert "no structured answer" in out.unresolved[0]["why"]


def test_a_sheet_with_no_structured_answer_loses_only_its_own_pages(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([
        {"text": "I cannot read this sheet"},
        {"final": VisionSheetAnswer(pages=[_answer(p) for p in (2, 3)])},
    ])
    out = classify_pages_by_vision(doc, engine, pages=[0, 1, 2, 3],
                                   mode="sheet", sheet_pages=2)

    assert sorted(out.label_map) == [2, 3]
    assert [u["page"] for u in out.unresolved] == [0, 1]


# -- the budget ----------------------------------------------------------------

def test_the_budget_caps_the_calls_and_the_rest_are_unresolved(synthetic):
    doc, _ = synthetic
    engine = FakeEngine(_page_script([0, 1]))
    out = classify_pages_by_vision(doc, engine, pages=[0, 1, 2, 3], budget=2)

    assert engine.n_calls == 2, "the budget is a ceiling on model calls"
    assert sorted(out.label_map) == [0, 1]
    assert [u["page"] for u in out.unresolved] == [2, 3]
    assert "budget of 2" in out.unresolved[0]["why"]
    assert out.stopped_on_budget is True
    assert out.budget == 2


def test_a_run_inside_its_budget_is_not_flagged_as_stopped(synthetic):
    doc, _ = synthetic
    engine = FakeEngine(_page_script([0, 1]))
    out = classify_pages_by_vision(doc, engine, pages=[0, 1], budget=9)

    assert out.stopped_on_budget is False
    assert out.unresolved == []


def test_the_budget_counts_calls_not_pages_in_sheet_mode(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([{"final": VisionSheetAnswer(
        pages=[_answer(p) for p in (0, 1)])}])
    out = classify_pages_by_vision(doc, engine, pages=[0, 1, 2, 3],
                                   mode="sheet", sheet_pages=2, budget=1)

    assert engine.n_calls == 1
    assert sorted(out.label_map) == [0, 1]
    assert [u["page"] for u in out.unresolved] == [2, 3]


def test_the_script_runs_out_rather_than_inventing_a_turn(synthetic):
    doc, _ = synthetic
    engine = FakeEngine(_page_script([0]))
    with pytest.raises(ScriptExhausted):
        classify_pages_by_vision(doc, engine, pages=[0, 1])


# -- what it cost ---------------------------------------------------------------

def test_the_pass_reports_what_it_cost_on_this_document_alone(synthetic):
    doc, _ = synthetic
    engine = FakeEngine(_page_script([0, 1, 2]))
    out = classify_pages_by_vision(doc, engine, pages=[0, 1, 2])

    assert out.cost["calls"] == 3
    assert out.cost["input_tokens"] == 3000       # the fake engine's 1000 each
    assert out.cost["output_tokens"] == 300
    assert out.model == "fake-model"
    assert out.cost["dollars"] == 0.0, "an unpriced model costs no dollars"


def test_the_result_serialises_to_something_a_run_file_can_hold(synthetic):
    import json

    doc, _ = synthetic
    engine = FakeEngine(
        [{"final": _answer(0, label="cover", confidence=0.83)}])
    out = classify_pages_by_vision(doc, engine, pages=[0], mode="page")
    blob = out.to_dict()

    assert json.loads(json.dumps(blob))["labels"] == {"0": "cover"}
    assert blob["detail"][0]["confidence"] == 0.83
    assert blob["mode"] == "page" and blob["dpi"] == 100.0
    assert blob["pages_asked"] == 1 and blob["model_calls"] == 1


def test_an_unknown_mode_is_refused_before_a_page_is_rendered(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([])
    with pytest.raises(ValueError, match="unknown mode"):
        classify_pages_by_vision(doc, engine, mode="contact")
    assert engine.n_calls == 0

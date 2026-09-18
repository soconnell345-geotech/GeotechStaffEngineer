"""The vision-first label pass offline: all three modes, the gates, the budget.

Every test runs the real pass over planlens' synthetic report with a scripted
engine, so what is exercised is the pass -- the pictures it renders, the
prompt it builds, the shape it asks for and what Python refuses -- and not a
mock of it. No model is called and no credential is read.
"""

from __future__ import annotations

import pytest

from report_ingest.engine import strict_schema, text_block
from report_ingest.label_review import LABEL_DEFINITIONS
from report_ingest.tests.fake_engine import FakeEngine, ScriptExhausted
from report_ingest.vision_labels import (
    DEFAULT_DPI, DOCUMENT_WINDOW, MAX_IMAGES_PER_CALL, MODES, STRIP_SHEETS_MAX,
    VISION_LABELS, VisionPageAnswer, VisionSheetAnswer,
    classify_pages_by_vision, document_windows, stamp_page_number,
    vision_system,
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


# -- document mode ------------------------------------------------------------

def _sheet(pages, label="narrative"):
    return {"final": VisionSheetAnswer(
        pages=[_answer(p, label=label) for p in pages])}


def _png_size(png):
    import io

    from PIL import Image

    return Image.open(io.BytesIO(png)).size


def test_the_document_prompt_says_what_the_thumbnails_are_for():
    flat = " ".join(vision_system("document").lower().split())
    assert "thumbnail contact sheets of the whole report" in flat
    assert "orientation" in flat
    assert "do not answer for a page you have only seen as a thumbnail" in flat
    assert "one entry per full-size page" in flat
    assert "stamped 'p. n'" in flat
    # The dividers and the contents list are the evidence the whole mode
    # exists to put in front of the model.
    assert "divider or fly sheet" in flat and "contents list" in flat
    # And the vocabulary and the rules are the same ones, not a second
    # wording of them.
    for name, definition in LABEL_DEFINITIONS.items():
        assert name in vision_system("document")
        assert definition.split(",")[0][:30] in vision_system("document")


def test_the_document_prompt_does_not_claim_the_page_stands_alone():
    """Page mode tells the model there is no neighbouring context. In
    document mode there is, and using it is the point."""
    alone = "no neighbouring context"
    assert alone in vision_system("page")
    assert alone in vision_system("sheet")
    assert alone not in vision_system("document")
    assert "in the light of the pages around it" in vision_system("document")


def test_the_measured_image_cap_is_the_default():
    # Measured on the owner's cluster on 2026-09-18: 50 went through and a
    # 51st came back "Too many images in request: 51, maximum allowed: 50".
    assert MAX_IMAGES_PER_CALL == 50
    assert DOCUMENT_WINDOW + STRIP_SHEETS_MAX <= MAX_IMAGES_PER_CALL


def test_a_call_never_carries_more_images_than_the_cap(synthetic):
    doc, gt = synthetic
    cap = 14
    engine = FakeEngine([_sheet(w) for w in
                         document_windows(list(range(gt.n_pages)), 13, 3)])
    classify_pages_by_vision(doc, engine, mode="document",
                             images_per_call=cap, window=36)

    assert engine.n_calls >= 2, "the cap must have forced more than one window"
    for call in engine.calls:
        # n_images counts the strip AND the full-size pages: the cap is on
        # the request, not on the pages.
        assert call["n_images"] <= cap
        assert len(call["image_blocks"]) == call["n_images"]
    assert max(c["n_images"] for c in engine.calls) == cap, (
        "the window should fill the room the strip leaves")


def test_the_strip_and_the_window_share_the_cap(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([_sheet(range(6))])
    out = classify_pages_by_vision(doc, engine, pages="0-5", mode="document",
                                   window=6, images_per_call=50)

    strip = out.cost["strip_sheets"]
    assert strip == 1, "22 pages is one contact sheet of 48"
    assert engine.calls[0]["n_images"] == strip + 6


def test_the_windows_tile_every_page_and_overlap():
    pages = list(range(151))
    windows = document_windows(pages, 36, 3)

    assert [w[0] for w in windows][:3] == [0, 33, 66], "the step is 36 - 3"
    covered = sorted({p for w in windows for p in w})
    assert covered == pages, "every page is in some window"
    for before, after in zip(windows, windows[1:]):
        shared = set(before) & set(after)
        assert len(shared) == 3, "consecutive windows share the overlap"


def test_a_short_last_window_is_left_short_rather_than_padded():
    windows = document_windows(list(range(10)), 4, 0)
    assert windows == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]], (
        "the tail is cheap, not wrong: a short window is fewer images, and "
        "padding it backwards would re-send four pages to gain two")
    assert sorted({p for w in windows for p in w}) == list(range(10))


def test_pages_that_are_not_a_run_keep_their_gaps():
    assert document_windows([0, 1, 5, 9], 2, 0) == [[0, 1], [5, 9]]


def test_every_full_size_page_is_stamped_with_the_index_it_is_scored_by(
        synthetic):
    """A report restarts its printed numbering in every appendix, so the
    number the model must answer with is drawn on the picture."""
    doc, _ = synthetic
    engine = FakeEngine([_sheet(range(4))])
    out = classify_pages_by_vision(doc, engine, pages="0-3", mode="document",
                                   window=4)

    blocks = engine.calls[0]["image_blocks"]
    pages = blocks[out.cost["strip_sheets"]:]
    assert len(pages) == 4
    for n, block in enumerate(pages):
        plain, _info = doc.render(n, dpi=DEFAULT_DPI)
        assert block["png"] != plain, f"page {n} was sent unstamped"
        assert _png_size(block["png"]) == _png_size(plain), (
            "the stamp is drawn ON the page, not added as a margin: the "
            "render's size is what the model is charged for")


def test_the_stamp_keeps_the_page_size_at_any_dpi(synthetic):
    doc, _ = synthetic
    for dpi in (72.0, 100.0, 200.0):
        plain, _info = doc.render(2, dpi=dpi)
        stamped = stamp_page_number(plain, 2)
        assert _png_size(stamped) == _png_size(plain)
        assert stamped != plain


def test_the_labels_decided_so_far_arrive_from_the_second_window_on(
        synthetic):
    doc, _ = synthetic
    engine = FakeEngine([_sheet([0, 1, 2, 3], label="cover"),
                         _sheet([2, 3, 4, 5], label="toc")])
    classify_pages_by_vision(doc, engine, pages="0-5", mode="document",
                             window=4, overlap=2)

    first = engine.calls[0]["messages"][0]["content"][0]["text"]
    second = engine.calls[1]["messages"][0]["content"][0]["text"]
    assert "LABELS DECIDED SO FAR" not in first, (
        "nothing has been decided before the first window")
    assert "LABELS DECIDED SO FAR" in second
    # As RUNS, which is both shorter and the shape of the evidence: a run of
    # one label is what an appendix looks like.
    assert "0-1: cover" in second
    assert "2" not in second.split("LABELS DECIDED SO FAR")[1].split(
        "This document has")[0].replace("0-1: cover", ""), (
        "only pages BEFORE the window are listed")


def test_the_outline_rides_on_a_document_call_only_when_asked(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([_sheet(range(4))])
    classify_pages_by_vision(doc, engine, pages="0-3", mode="document",
                             window=4)
    assert "WHAT THIS DOCUMENT PRINTS ABOUT ITSELF" not in \
        engine.calls[0]["messages"][0]["content"][0]["text"]

    engine = FakeEngine([_sheet(range(4))])
    classify_pages_by_vision(doc, engine, pages="0-3", mode="document",
                             window=4, outline_context=True)
    assert "WHAT THIS DOCUMENT PRINTS ABOUT ITSELF" in \
        engine.calls[0]["messages"][0]["content"][0]["text"]


def test_each_strip_sheet_is_introduced_by_the_pages_it_shows(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([_sheet(range(4))])
    classify_pages_by_vision(doc, engine, pages="0-3", mode="document",
                             window=4)

    texts = [b["text"] for b in engine.calls[0]["messages"][0]["content"]
             if b["type"] == "text"]
    assert any(t.startswith("Thumbnails of pages 0-21") for t in texts), (
        "a sheet must never arrive as an unlabelled picture")
    assert any("full-size pages now follow" in t for t in texts)


def test_an_answer_for_a_page_outside_the_window_is_dropped_with_a_note(
        synthetic):
    doc, _ = synthetic
    engine = FakeEngine([{"final": VisionSheetAnswer(
        pages=[_answer(0), _answer(1), _answer(77)])}])
    out = classify_pages_by_vision(doc, engine, pages="0-1", mode="document",
                                   window=4)

    assert sorted(out.label_map) == [0, 1]
    assert any(q["page"] == 77 and "not a full-size page of this window"
               in q["note"] for q in out.qa)


def test_a_page_the_model_skipped_is_unresolved_and_never_guessed(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([{"final": VisionSheetAnswer(
        pages=[_answer(0), _answer(2)])}])
    out = classify_pages_by_vision(doc, engine, pages="0-2", mode="document",
                                   window=4)

    assert sorted(out.label_map) == [0, 2]
    assert [u["page"] for u in out.unresolved] == [1]
    assert "left this page out of the window" in out.unresolved[0]["why"]


def test_a_page_skipped_in_one_window_is_rescued_by_the_next(synthetic):
    """This is what the overlap is FOR."""
    doc, _ = synthetic
    engine = FakeEngine([
        {"final": VisionSheetAnswer(pages=[_answer(p) for p in (0, 1)])},
        {"final": VisionSheetAnswer(
            pages=[_answer(p, label="plan") for p in (2, 3, 4, 5)])},
    ])
    out = classify_pages_by_vision(doc, engine, pages="0-5", mode="document",
                                   window=4, overlap=2)

    assert out.unresolved == []
    assert out.label_map[2] == "plan", "the second window answered it"
    assert out.label_map[3] == "plan"


def test_a_page_answered_twice_across_windows_keeps_the_later_answer(
        synthetic):
    """The later window was made with more of the report already decided,
    so its answer is the better-informed one."""
    doc, _ = synthetic
    engine = FakeEngine([
        {"final": VisionSheetAnswer(
            pages=[_answer(p, label="figure") for p in (0, 1, 2, 3)])},
        {"final": VisionSheetAnswer(
            pages=[_answer(p, label="profile") for p in (2, 3, 4, 5)])},
    ])
    out = classify_pages_by_vision(doc, engine, pages="0-5", mode="document",
                                   window=4, overlap=2)

    assert out.label_map[0] == "figure" and out.label_map[1] == "figure"
    assert out.label_map[2] == "profile", "the later window wins"
    assert out.label_map[3] == "profile"
    assert out.label_map[4] == "profile"
    assert len(out.labels) == 6, "one row a page, not one a window"


def test_a_page_answered_twice_inside_one_window_keeps_the_first(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([{"final": _LooseSheet([
        _Loose(0, "plan"), _Loose(0, "profile")])}])
    out = classify_pages_by_vision(doc, engine, pages=[0], mode="document",
                                   window=4)

    assert out.label_map == {0: "plan"}
    assert any("twice within one window" in q["note"] for q in out.qa)


def test_a_window_with_no_structured_answer_loses_only_its_own_pages(
        synthetic):
    doc, _ = synthetic
    engine = FakeEngine([
        {"text": "I cannot read these"},
        {"final": VisionSheetAnswer(pages=[_answer(p) for p in (4, 5, 6, 7)])},
    ])
    out = classify_pages_by_vision(doc, engine, pages="0-7", mode="document",
                                   window=4, overlap=0)

    assert sorted(out.label_map) == [4, 5, 6, 7]
    assert [u["page"] for u in out.unresolved] == [0, 1, 2, 3]
    assert "no structured answer" in out.unresolved[0]["why"]


def test_the_budget_stops_the_run_and_the_rest_is_unresolved(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([_sheet([0, 1, 2, 3])])
    out = classify_pages_by_vision(doc, engine, pages="0-11", mode="document",
                                   window=4, overlap=0, budget=1)

    assert engine.n_calls == 1, "the budget is a ceiling on model calls"
    assert sorted(out.label_map) == [0, 1, 2, 3]
    assert [u["page"] for u in out.unresolved] == list(range(4, 12))
    assert "budget of 1" in out.unresolved[0]["why"]
    assert out.stopped_on_budget is True


def test_the_gates_apply_to_a_document_answer_too(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([{"final": _LooseSheet([
        _Loose(0, "boring_log_continuation"), _Loose(1, "plan",
                                                     confidence=1.9)])}])
    out = classify_pages_by_vision(doc, engine, pages="0-1", mode="document",
                                   window=4)

    assert out.label_map[0] == "other"
    assert any("not in the vocabulary" in q["note"] for q in out.qa)
    assert out.label_map[1] == "plan"
    assert [row.confidence for row in out.labels if row.page == 1] == [1.0]


def test_the_cost_records_the_windows_the_strip_and_the_mode(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([_sheet([0, 1, 2, 3]), _sheet([4, 5, 6, 7])])
    out = classify_pages_by_vision(doc, engine, pages="0-7", mode="document",
                                   window=4, overlap=0, detail="low")

    assert out.cost["mode"] == "document"
    assert out.cost["windows"] == 2
    assert out.cost["strip_sheets"] == 1
    assert out.cost["window_pages"] == 4
    assert out.cost["images_per_call"] == MAX_IMAGES_PER_CALL
    assert out.cost["detail"] == "low"
    assert out.cost["calls"] == 2
    assert out.cost["input_tokens"] == 2000
    assert out.mode == "document"


def test_the_document_result_serialises_to_a_run_file(synthetic):
    import json

    doc, _ = synthetic
    engine = FakeEngine([_sheet([0, 1], label="cover")])
    out = classify_pages_by_vision(doc, engine, pages="0-1", mode="document",
                                   window=4)
    blob = json.loads(json.dumps(out.to_dict()))

    assert blob["labels"] == {"0": "cover", "1": "cover"}
    assert blob["mode"] == "document"
    assert blob["cost"]["windows"] == 1


# -- the strip when the report is too long to show all of itself --------------

def test_a_report_too_long_to_show_sends_the_sheets_nearest_the_window():
    from report_ingest.vision_labels import _strip_for

    sheets = [(b"png", {"pages": list(range(i * 48, i * 48 + 48))})
              for i in range(16)]                 # 768 pages, 16 sheets
    near = _strip_for(sheets, list(range(384, 420)), 12)

    assert len(near) == 12
    shown = [s[1]["pages"][0] for s in near]
    assert shown == sorted(shown), "the sheets stay in page order"
    # The window sits at page ~400, which is sheet 8; the twelve kept must
    # bracket it rather than start at the front of the report.
    assert 384 in [s[1]["pages"][0] for s in near]
    assert shown[0] > 0, "the far end of a long report is what gets dropped"


def test_a_report_that_fits_sends_all_of_itself():
    from report_ingest.vision_labels import _strip_for

    sheets = [(b"a", {"pages": [0]}), (b"b", {"pages": [48]})]
    assert _strip_for(sheets, [0, 1], 12) == sheets
    assert _strip_for(sheets, [0, 1], 0) == []


# -- the detail key ------------------------------------------------------------

def test_the_detail_reaches_every_picture_in_every_mode(synthetic):
    doc, _ = synthetic
    for mode, script in (("page", _page_script([0])),
                         ("sheet", [_sheet([0])]),
                         ("document", [_sheet([0])])):
        engine = FakeEngine(script)
        classify_pages_by_vision(doc, engine, pages=[0], mode=mode,
                                 detail="low", window=4)
        blocks = engine.calls[0]["image_blocks"]
        assert blocks, mode
        assert {b.get("detail") for b in blocks} == {"low"}, mode


def test_no_detail_is_the_default_and_leaves_the_request_as_it_was(synthetic):
    doc, _ = synthetic
    engine = FakeEngine(_page_script([0]))
    classify_pages_by_vision(doc, engine, pages=[0])
    assert engine.calls[0]["image_blocks"][0].get("detail") is None


def test_an_unknown_detail_is_refused_before_a_page_is_rendered(synthetic):
    doc, _ = synthetic
    engine = FakeEngine([])
    with pytest.raises(ValueError, match="unknown detail"):
        classify_pages_by_vision(doc, engine, detail="medium")
    assert engine.n_calls == 0


def test_the_detail_reaches_the_provider_payload():
    """The point of the key is the request it produces."""
    from report_ingest.engine import PrompterEngine, image_block, user

    messages = [user(text_block("look"), image_block(b"\x89PNG", "low"))]
    payload = PrompterEngine._to_provider(messages, "sys")
    parts = payload[-1]["content"]
    picture = [p for p in parts if p["type"] == "image_url"][0]

    assert picture["image_url"]["detail"] == "low"
    assert picture["image_url"]["url"].startswith("data:image/png;base64,")


def test_a_picture_with_no_detail_sends_no_detail_key():
    from report_ingest.engine import PrompterEngine, image_block, user

    payload = PrompterEngine._to_provider(
        [user(text_block("look"), image_block(b"\x89PNG"))], None)
    picture = [p for p in payload[-1]["content"]
               if p["type"] == "image_url"][0]

    assert "detail" not in picture["image_url"]


def test_the_claude_engine_ignores_a_detail_it_has_no_key_for():
    from report_ingest.engine import ClaudeEngine, image_block

    block = ClaudeEngine._to_provider_block(image_block(b"\x89PNG", "low"))
    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/png"
    assert "detail" not in block

"""The review prompts look first and treat text as evidence (owner, 2026-10-01).

Field session 2026-10-01: asked for every tag on an 85-sheet drawing set,
the agent searched the text layer first (the lettering was drawn as lines),
looked at the sheets the searches and thumbnails suggested, and missed whole
sheets. The owner: "Systematic visual enumeration is much more flexible; even
if it's more expensive, we don't care much here. The tools can be a fallback
or supporting evidence, but the LLM shouldn't use them in lieu of vision."
These pins keep the ordering from drifting back.
"""

from funhouse_agent.deep import prompt as P
from funhouse_agent.deep.review_agent import INLINE_DESCRIPTIONS, REVIEW_DESCRIPTIONS


def _flat(text):
    """One space between words: the prompts wrap at 79 columns."""
    return " ".join(text.split())


REVIEW_PROMPTS = {
    "legacy": _flat(P.DOCUMENT_REVIEW_PROMPT),
    "lean": _flat(P.DOCUMENT_REVIEW_PROMPT_LEAN),
    "reader": _flat(P.DOCUMENT_REVIEW_READER_PROMPT),
}


def test_review_prompts_look_at_every_page_in_scope():
    for name, text in REVIEW_PROMPTS.items():
        assert "LOOK at every one" in text, name
        assert "never a filter" in text, name
        assert "cost is never a reason to look at fewer pages" in text, name
        assert "Then text, then your eyes" not in text, name


def test_review_prompts_make_coverage_part_of_every_answer():
    for name, text in REVIEW_PROMPTS.items():
        assert "Say what you covered" in text, name
        assert "never \"not in the document\" unless you looked at" in text, name


def test_markups_come_from_tool_results_not_memory():
    for name in ("legacy", "lean"):
        text = REVIEW_PROMPTS[name]
        # 2026-10-07: "the look that found it" was the whole-page look, and
        # every ring placed from one missed; anchor on the zoom instead.
        assert ("`view` and `image_box` of the zoomed look in which the "
                "thing is legible") in text
        assert "only says where to zoom" in text
        assert "of the look that found it" not in text
        assert "never by a location from memory" in text
        assert "misplaced" in text


def test_the_geotech_page_reads_text_pages_and_looks_at_pictures():
    text = _flat(P.build_domain_prompt())
    assert "Text first, then your eyes" not in text
    assert "never decides which pages get looked at" in text


def test_page_tools_are_for_every_page_not_a_fallback():
    for table in (REVIEW_DESCRIPTIONS, INLINE_DESCRIPTIONS):
        desc = _flat(table["analyze_pdf_page"])
        assert "EVERY page" in desc and "not only when a" in desc

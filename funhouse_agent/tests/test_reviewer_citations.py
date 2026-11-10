"""Field feedback 2026-09-15 (SOE review re-run), N13.

The reviewer sub-agent's four reference lookups all came back empty -- two of
them a list_methods category filter that matched nothing and returned {} --
and it then cited "FHWA GEC-12 (Design and Construction of Deep Excavations)"
about ten times. GEC-12 is Driven Piles.
"""

from funhouse_agent.dispatch import list_methods
from funhouse_agent.reviewer import CONSULTANT_FRAMING, REVIEWER_SYSTEM_PROMPT


def test_reviewer_may_cite_only_what_its_tools_returned():
    assert "Cite ONLY a reference" in REVIEWER_SYSTEM_PROMPT
    assert "GEC-12 is Driven Piles" in REVIEWER_SYSTEM_PROMPT
    assert "only ones a tool returned in this review" in REVIEWER_SYSTEM_PROMPT


def test_the_reviewer_has_no_lookup_count_in_its_rules():
    """2026-11-10 reference review, item 7: the reviewer has a model-call
    budget (DEFAULT_REVIEWER_MAX_MODEL_CALLS); the old "3-4 reference
    lookups maximum" rule was left beside it, in both prompts."""
    from funhouse_agent.reviewer import REVIEWER_DEEP_PROMPT
    for prompt in (REVIEWER_SYSTEM_PROMPT, REVIEWER_DEEP_PROMPT):
        assert "lookups maximum" not in prompt
        assert "Limit yourself" not in prompt
        assert "4. Do NOT repeat the full analysis" in prompt


def test_references_consult_may_not_cite_from_memory():
    assert "never from memory" in CONSULTANT_FRAMING
    assert CONSULTANT_FRAMING.endswith("Question: ")


def test_not_found_is_only_for_a_lookup_that_ran_and_found_nothing():
    """2026-11-10 reference review, item 5: "not found" had covered lookups
    that errored or were never made."""
    assert ('Say "not found" only for a lookup that ran and returned '
            "nothing, and name that lookup") in CONSULTANT_FRAMING
    assert "say it failed or was not made" in CONSULTANT_FRAMING


def test_category_that_matches_nothing_lists_the_real_categories():
    # Never the bare {} that read as "nothing on that": the note says the
    # category matched nothing, the real categories are named, and methods
    # are still listed (the 2026-10 change: the methods instead of an error).
    out = list_methods("gec7", category="earth retaining structures")
    assert out != {} and out["available_categories"]
    assert "not one of its categories" in out["note"] \
        or "is not a 'gec7' category" in out["note"]
    assert out["methods"]
    good = out["available_categories"][0]
    assert "error" not in list_methods("gec7", category=good)
    assert "note" not in list_methods("gec7", category=good)
    assert "error" not in list_methods("gec7")

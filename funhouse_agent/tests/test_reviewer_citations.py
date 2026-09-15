"""Field feedback 2026-09-15 (Nairobi SOE re-run), N13.

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


def test_references_consult_may_not_cite_from_memory():
    assert "rather than citing from memory" in CONSULTANT_FRAMING
    assert CONSULTANT_FRAMING.endswith("Question: ")


def test_category_that_matches_nothing_lists_the_real_categories():
    out = list_methods("gec7", category="earth retaining structures")
    assert "error" in out and out["available_categories"]
    good = out["available_categories"][0]
    assert "error" not in list_methods("gec7", category=good)
    assert "error" not in list_methods("gec7")

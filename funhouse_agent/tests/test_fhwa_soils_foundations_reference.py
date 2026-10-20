"""FHWA NHI-06-088/089 (Soils and Foundations) through the reference tools.

REF-29: the app's tools cite this manual (the slope search's Figs. 6-14 to
6-16 and 6-19 and p. 6-38, the sheet-pile Fig. 10-10), and a references
consult could not find it. These check each lookup a consult makes: a text
search, a section by number, a figure search and a figure by number, all
through ``dispatch.call_agent`` as the consult sub-agent calls them.
"""

import json

import pytest

pytest.importorskip("geotech_references.fhwa_soils_foundations")

from funhouse_agent import dispatch  # noqa: E402
from funhouse_agent.cited_documents import module_documents  # noqa: E402

REF = "fhwa_soils_foundations"


def _call(agent, method, params):
    out = dispatch.call_agent(agent, method, params)
    json.dumps(out)
    assert "error" not in out, out
    return out


def test_registered_as_a_reference_module():
    assert REF in dispatch.REFERENCE_MODULES
    assert REF not in dispatch.ANALYSIS_MODULES
    assert module_documents(REF) == {"fhwa-nhi-06-088", "fhwa-nhi-06-089"}


def test_search_finds_janbu_charts():
    out = _call("reference_db", "reference_search",
                {"query": "Janbu stability charts", "reference": REF})
    hits = out["result"]
    hits = hits["hits"] if isinstance(hits, dict) else hits
    assert "6.6.3" in [h["section_id"] for h in hits[:5]]
    out = _call(REF, "search_sections", {"query": "Nordlund"})
    hits = out["result"]
    hits = hits["hits"] if isinstance(hits, dict) else hits
    assert "9.5.1.1" in [h["section_id"] for h in hits[:5]]


def test_section_lookup_cites_the_printed_page():
    out = _call(REF, "retrieve_section", {"section_id": "6.6.3"})
    sec = out["result"] if "result" in out else out
    assert sec["title"] == "Janbu’s Stability Charts"
    # the slope search cites NHI-06-088 p. 6-38 for circles above the toe
    assert "[p. 6-38]" in sec["body"]
    assert "a circle passing above the toe of the slope may be more critical" \
        in sec["body"]
    out = _call("reference_db", "reference_get",
                {"reference": REF, "section_id": "10.5.5"})
    sec = out["result"] if "result" in out else out
    assert sec["title"].startswith("Step 7")
    assert "[p. 10-43]" in sec["body"]


def test_figure_lookup_cited_figures():
    for num, page in (("6-14", "6-27"), ("6-15", "6-28"), ("6-16", "6-34"),
                      ("6-19", "6-39"), ("10-10", "10-22")):
        out = _call("figure_db", "figure_get",
                    {"reference": REF, "figure_number": num})
        fig = out["result"] if "result" in out else out
        assert fig["printed_page"] == page, num
    out = _call("figure_db", "figure_search",
                {"query": "Janbu stability chart", "reference": REF})
    hits = out["result"]
    hits = hits["hits"] if isinstance(hits, dict) else hits
    assert {"6-16", "6-19"} & {h["figure_number"] for h in hits[:5]}

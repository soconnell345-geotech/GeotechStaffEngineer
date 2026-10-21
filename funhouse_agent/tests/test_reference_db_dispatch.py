"""Tests for the reference_db adapter through funhouse_agent.dispatch."""

from __future__ import annotations

import pytest

from funhouse_agent.dispatch import call_agent, list_agents


def test_reference_db_registered():
    assert "reference_db" in list_agents()


def test_reference_search_returns_summary_only():
    res = call_agent("reference_db", "reference_search", {
        "query": "primary consolidation settlement",
        "reference": "dm7_1",
        "limit": 3,
    })
    hits = res.get("result", res)
    assert isinstance(hits, list) and hits, "expected hits list"
    h = hits[0]
    assert h["reference"] == "dm7_1"
    assert "summary" in h
    assert "body" not in h  # search hits are summary-only


def test_reference_get_returns_full_body_ufc_form():
    res = call_agent("reference_db", "reference_get", {
        "reference": "dm7_1",
        "section_id": "5-5.2",
    })
    assert res.get("section_id") == "5-5.2"
    assert "body" in res
    assert isinstance(res["body"], str) and res["body"]


def test_reference_query_select_groupby():
    res = call_agent("reference_db", "reference_query", {
        "sql": "SELECT reference, COUNT(*) AS n FROM sections "
               "WHERE reference IN ('dm7_1','dm7_2') GROUP BY reference",
    })
    rows = res.get("result", res)
    assert isinstance(rows, list)
    counts = {r["reference"]: r["n"] for r in rows}
    assert counts.get("dm7_1", 0) >= 400
    assert counts.get("dm7_2", 0) >= 400


def test_reference_query_blocks_writes():
    res = call_agent("reference_db", "reference_query", {
        "sql": "DELETE FROM sections",
    })
    rows = res.get("result", res)
    assert "error" in rows[0]


def test_list_indexed_references():
    res = call_agent("reference_db", "list_indexed_references", {})
    inv = res.get("result", res)
    assert isinstance(inv, list)
    refs = {r["reference"] for r in inv}
    assert "dm7_1" in refs
    assert "dm7_2" in refs


@pytest.mark.parametrize("name", ["GEC-10", "GEC 10", "gec10", "GEC_10",
                                  "NHI-18-024", "FHWA-NHI-18-024"])
def test_reference_filter_accepts_printed_names(name):
    """REF-41: filtering on "GEC-10" returned an empty list silently."""
    res = call_agent("reference_db", "reference_search", {
        "query": "drilled shaft side resistance", "reference": name,
        "limit": 3})
    hits = res.get("result", res)
    assert hits and all(h["reference"] == "gec_10" for h in hits)


def test_reference_filter_unknown_name_gets_did_you_mean():
    res = call_agent("reference_db", "reference_search", {
        "query": "settlement", "reference": "GEC-99"})
    row = res.get("result", res)[0]
    assert "error" in row and "gec_9" in row["did_you_mean"]
    got = call_agent("reference_db", "reference_get", {
        "reference": "GEC 10", "section_id": "10.3.5"})
    assert "error" not in got or "no reference named" not in got["error"]
    bad = call_agent("reference_db", "reference_get", {
        "reference": "nonsense", "section_id": "1"})
    assert bad.get("did_you_mean")

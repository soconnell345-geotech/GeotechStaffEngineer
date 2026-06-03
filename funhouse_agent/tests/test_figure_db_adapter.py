"""Offline tests for the figure_db adapter's starve-the-shortcut signpost.

No API key, no PDF render — these exercise only the FTS5 catalog search and the
``read_value`` next-step instruction the adapter attaches to each hit. The point
of the signpost: a value that lives in a chart must be read off the rendered
image via ``read_reference_figure``, never inferred from the caption or memory.
The search result identifies the figure and carries no values to cheat from.
"""

from __future__ import annotations

import pytest

from funhouse_agent.adapters import figure_db_adapter


# Raw (unwrapped) adapter function — returns the list of hit dicts directly,
# bypassing the JSON-cleaning wrapper so we can assert on the hit shape.
_figure_search = figure_db_adapter.METHOD_REGISTRY["figure_search"].__wrapped__


def _catalog_available() -> bool:
    try:
        from geotech_references import _figures_db
        return bool(_figures_db.list_indexed_figures())
    except Exception:
        return False


catalog_required = pytest.mark.skipif(
    not _catalog_available(),
    reason="geotech_references figure catalog not installed",
)


@catalog_required
def test_log_spiral_query_finds_fig_4_12():
    """Sanity: the known DM 7.2 log-spiral chart is retrievable by concept."""
    hits = _figure_search("log spiral passive earth pressure", reference="dm7_2")
    assert hits, "expected at least one hit for the log-spiral query"
    assert "4-12" in {h.get("figure_number") for h in hits}


@catalog_required
def test_every_hit_carries_read_value_signpost():
    hits = _figure_search("log spiral", reference="dm7_2")
    assert hits
    for hit in hits:
        assert "read_value" in hit, f"hit missing read_value: {hit}"
        rv = hit["read_value"]
        # Concrete next call, prefilled with THIS hit's identifiers.
        assert "read_reference_figure" in rv
        assert f"reference='{hit['reference']}'" in rv
        assert f"figure_number='{hit['figure_number']}'" in rv
        # Explicit prohibition on the from-caption / from-memory shortcut.
        assert "do not" in rv.lower()


@catalog_required
def test_hit_carries_no_value_to_cheat_from():
    """The search result identifies the figure but exposes no chart values:
    no description/page-content fields a model could read a number off of."""
    hits = _figure_search("log spiral", reference="dm7_2")
    assert hits
    leak_keys = {"description", "text", "body", "content", "value", "values"}
    for hit in hits:
        assert not (leak_keys & set(hit)), f"unexpected value-bearing key in {hit}"


@catalog_required
def test_tolerates_guessed_cap_parameter():
    """A model that passes ``top_k`` (or similar) instead of ``limit`` must not
    crash the call — it should be honored as the result cap."""
    hits = _figure_search("earth pressure", reference="dm7_2", top_k=3)
    assert hits, "expected hits despite the aliased cap parameter"
    assert len(hits) <= 3
    # An unrelated unknown kwarg is tolerated (ignored), not fatal.
    hits2 = _figure_search("earth pressure", reference="dm7_2", foo="bar")
    assert isinstance(hits2, list)


@catalog_required
def test_error_hits_are_not_annotated():
    """A bad FTS query returns an error dict, which must not be signposted."""
    hits = _figure_search('"unterminated phrase', reference="dm7_2")
    for hit in hits:
        if "error" in hit:
            assert "read_value" not in hit

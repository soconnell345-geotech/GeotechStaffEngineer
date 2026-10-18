"""Every reference tool result fits the reference cap as valid JSON (REF-26).

The deep agent's ``call_agent`` tool sends at most
``DEFAULT_REFERENCE_RESULT_CHARS`` (16,000) characters of a reference
module's result and cuts the rest, leaving broken JSON and losing everything
after the cut. Before REF-26: ``list_chapters`` for three references
(ufc_concrete_practice 27,820), ``search_sections`` (708,876 for "a"),
``load_chapter`` (206,140), ``reference_search`` at 50 hits (36,424),
``reference_query`` (258,786) and ``figure_search`` at 50 (68,859) and one
``figure_get`` (30,444). Each now fits by its shape. This walks every
reference adapter's methods with the worst arguments there are, through
``dispatch.call_agent`` and ``json.dumps`` exactly as the tool does, and
fails on any result over the cap or not valid JSON.

Skips where geotech-references is absent or older than REF-26.
"""

from __future__ import annotations

import inspect
import json

import pytest

retrieval = pytest.importorskip("geotech_references._retrieval")
if "max_chars" not in inspect.signature(retrieval.search_sections).parameters:
    pytest.skip("geotech-references older than REF-26", allow_module_level=True)

from geotech_references import _figures_db, _retrieval_db  # noqa: E402

from funhouse_agent import dispatch  # noqa: E402
from funhouse_agent.adapters._reference_common import (  # noqa: E402
    RESULT_BUDGET_CHARS,
)
from funhouse_agent.deep.tools import DEFAULT_REFERENCE_RESULT_CHARS  # noqa: E402

CAP = DEFAULT_REFERENCE_RESULT_CHARS

#: The broadest queries there are: every section matches "a" / "the".
BROAD = ("a", "the", "of", "shall", "pile", "design", "pile capacity")

#: Methods walked with their own arguments below; the rest of a reference
#: adapter is walked when it needs no argument.
_WALKED = {"list_chapters", "retrieve_section", "search_sections",
           "load_chapter", "reference_search", "reference_get",
           "reference_query", "figure_search", "figure_get"}


def _call(agent: str, method: str, params: dict) -> dict:
    """The result as the call_agent tool would send it, checked."""
    text = json.dumps(dispatch.call_agent(agent, method, params), default=str)
    where = f"{agent}.{method}({params})"
    assert len(text) <= CAP, f"{where}: {len(text):,} characters"
    return json.loads(text)


def _text_agents() -> list[str]:
    return [name for name in sorted(dispatch.REFERENCE_MODULES)
            if "list_chapters" in dispatch._load_adapter(name).METHOD_REGISTRY]


def test_the_budget_leaves_room_under_the_cap():
    assert RESULT_BUDGET_CHARS <= CAP - 1500


@pytest.mark.parametrize("agent", _text_agents())
def test_text_retrieval_fits(agent):
    whole = _call(agent, "list_chapters", {})
    listing = whole["result"]
    # every chapter's sections when they fit, else the chapters and a note
    assert listing and all(("sections" in c) != ("note" in whole)
                           for c in listing)
    ids = []
    for c in listing:
        one = _call(agent, "list_chapters", {"chapter": c["chapter"]})
        ids += [s["section_id"] for o in one["result"]
                for s in o.get("sections", [])]
        _call(agent, "load_chapter", {"chapter": c["chapter"]})
    assert ids
    for sid in dict.fromkeys(ids):
        _call(agent, "retrieve_section", {"section_id": sid})
        _call(agent, "retrieve_section", {"section_id": sid,
                                          "include_subsections": True})
    missing = _call(agent, "retrieve_section", {"section_id": ids[0] + ".99"})
    assert "error" in missing
    for q in BROAD:
        out = _call(agent, "search_sections", {"query": q, "limit": 50})
        assert out.get("shown", 0) == len(out["result"])
        _call(agent, "search_sections", {"query": q, "limit": 50,
                                         "offset": 10})


def _largest_sections(n: int = 40) -> list[tuple[str, str]]:
    conn = _retrieval_db._ro_connect()
    try:
        rows = conn.execute(
            "SELECT reference, section_id FROM sections ORDER BY "
            "length(body) + length(coalesce(key_points, '')) "
            "+ length(coalesce(applicability, '')) "
            "+ length(coalesce(equations_json, '')) "
            "+ length(coalesce(figures_json, '')) "
            "+ length(coalesce(tables_json, '')) DESC LIMIT ?",
            (n,)).fetchall()
        parents = conn.execute(
            "SELECT p.reference, p.section_id FROM sections p JOIN sections c"
            " ON c.parent_rowid = p.rowid GROUP BY p.rowid"
            " ORDER BY count(*) DESC LIMIT ?", (n,)).fetchall()
    finally:
        conn.close()
    return [tuple(r) for r in rows] + [tuple(r) for r in parents]


def test_reference_db_fits():
    _retrieval_db._ensure_db()
    for ref, sid in _largest_sections():
        _call("reference_db", "reference_get",
              {"reference": ref, "section_id": sid})
        _call("reference_db", "reference_get",
              {"reference": ref, "section_id": sid,
               "include_subsections": True})
    for q in BROAD:
        out = _call("reference_db", "reference_search",
                    {"query": q, "limit": 50})
        assert out["result"] and out["shown"] == len(out["result"])
        _call("reference_db", "reference_search",
              {"query": q, "limit": 50, "offset": 40})
    for sql in ("SELECT * FROM sections",
                "SELECT body FROM sections ORDER BY length(body) DESC",
                "SELECT * FROM sections ORDER BY length(body) DESC LIMIT 1"):
        out = _call("reference_db", "reference_query", {"sql": sql})
        assert out["result"]
    _call("reference_db", "list_indexed_references", {})


def test_figure_db_fits():
    _figures_db._ensure_db()
    conn = _figures_db._ro_connect()
    try:
        figs = conn.execute(
            "SELECT reference, figure_number FROM figures ORDER BY "
            "length(coalesce(caption, '')) "
            "+ length(coalesce(description, '')) DESC LIMIT 25").fetchall()
    finally:
        conn.close()
    for ref, number in figs:
        _call("figure_db", "figure_get",
              {"reference": ref, "figure_number": number})
    for q in BROAD + ("chart", "pressure coefficient"):
        out = _call("figure_db", "figure_search", {"query": q, "limit": 50})
        assert all("read_value" in h for h in out["result"])
    _call("figure_db", "list_indexed_figures", {})


@pytest.mark.parametrize("agent", sorted(dispatch.REFERENCE_MODULES))
def test_every_method_without_arguments_fits(agent):
    mod = dispatch._load_adapter(agent)
    for method, info in mod.METHOD_INFO.items():
        if (method in _WALKED or info.get("alias_of")
                or method.startswith("plot")):
            continue
        required = [k for k, v in (info.get("parameters") or {}).items()
                    if isinstance(v, dict) and v.get("required")]
        if not required:
            _call(agent, method, {})
        elif (agent in ("reference_db", "figure_db")
              or info.get("category") == "Text Retrieval"):
            # a text or search method this file does not walk with its
            # worst arguments yet: walk it above before it ships
            pytest.fail(f"{agent}.{method} takes {required} and is not "
                        "walked by this test")

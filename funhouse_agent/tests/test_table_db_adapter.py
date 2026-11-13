"""The table_db adapter and the table page reader (REF-86 / REF-92).

``table_db`` searches the library's printed-table catalogs and gives every
hit ``how_to_read``, the next step in this app's own tool names; the
``read_reference_table`` reader renders a page-only table's page(s) and
reads them with vision. The catalog tests use the library's real catalogs
(table numbers only; no cell value of a printed table is copied here); the
read tests use a made-up PDF and a made-up table, so they run with no
reference PDF on the machine.
"""

from __future__ import annotations

import json
import sys

import pytest

tables_db = pytest.importorskip("geotech_references._tables_db")

from funhouse_agent import dispatch, reference_docs  # noqa: E402
from funhouse_agent.adapters import table_db_adapter  # noqa: E402
from funhouse_agent.adapters._reference_common import (  # noqa: E402
    RESULT_BUDGET_CHARS,
)

_search = table_db_adapter.METHOD_REGISTRY["table_search"].__wrapped__
SCOPE = dispatch.REFERENCE_MODULES


def _call(method: str, params: dict) -> dict:
    return json.loads(json.dumps(dispatch.call_agent(
        "table_db", method, params, allowed_agents=SCOPE), default=str))


@pytest.fixture(autouse=True)
def _unpinned():
    reference_docs.pin_chart_reads(None)
    yield
    reference_docs.pin_chart_reads(None)


def _first_page_table() -> dict:
    tables_db._ensure_db()
    conn = tables_db._ro_connect()
    try:
        row = conn.execute(
            "SELECT reference, table_number, pdf_page_index, printed_page "
            "FROM tables WHERE value_source = 'page' LIMIT 1").fetchone()
    finally:
        conn.close()
    return dict(row)


# ---------------------------------------------------------------------------
# The signpost
# ---------------------------------------------------------------------------

def test_every_hit_names_its_next_step_in_this_apps_tools():
    hits = _search("bond strength", limit=20)
    assert hits
    kinds = set()
    for hit in hits:
        step = hit["how_to_read"]
        kinds.add(hit["values"])
        assert "lookup_reference" not in step
        if hit["values"] == "text" and hit["values_in"]:
            assert (f"call_agent('reference_db', 'reference_get', "
                    f"{{'reference': '{hit['reference']}', 'section_id': "
                    f"'{hit['values_in']}'}})") in step
        elif hit["values"] == "lookup" and hit["values_in"]:
            module, method = table_db_adapter.lookup_target(
                hit["reference"], hit["values_in"])
            # one method, or each of several, named with the module's call
            assert f"call_agent('{module}'," in step and method in step
    assert len(kinds) >= 2


def test_a_page_table_names_the_reader_only_where_pages_can_be_read():
    row = _first_page_table()
    params = {"reference": row["reference"],
              "table_number": row["table_number"],
              "pdf_page_index": row["pdf_page_index"]}
    reference_docs.pin_chart_reads(True)
    step = _call("table_get", params)["how_to_read"]
    assert (f"read_reference_table(reference='{row['reference']}', "
            f"table_number='{row['table_number']}', "
            f"pdf_page_index={row['pdf_page_index']},") in step
    assert "Do NOT give a value" in step
    reference_docs.pin_chart_reads(False)
    step = _call("table_get", params)["how_to_read"]
    assert "read_reference_table" not in step
    assert step == reference_docs.no_table_read(
        row["table_number"], row["printed_page"], row["reference"])


def test_a_multi_page_table_says_how_many_pages_a_read_sends():
    from funhouse_agent.reference_vision import TABLE_PAGES_MAX
    tables_db._ensure_db()
    conn = tables_db._ro_connect()
    try:
        row = conn.execute(
            "SELECT reference, table_number, pdf_page_index FROM tables "
            "WHERE value_source = 'page' AND json_array_length(pages) > ? "
            "LIMIT 1", (TABLE_PAGES_MAX,)).fetchone()
    finally:
        conn.close()
    if row is None:
        pytest.skip("no page table over more pages than one read sends")
    reference_docs.pin_chart_reads(True)
    step = _call("table_get", {"reference": row[0], "table_number": row[1],
                               "pdf_page_index": row[2]})["how_to_read"]
    assert f"one read sends up to {TABLE_PAGES_MAX} of them" in step


def test_errors_are_returned_as_errors_and_not_signposted():
    out = _call("table_search", {"query": "pile", "reference": "dm7"})
    assert out["result"] and "error" in out["result"][0]
    assert "how_to_read" not in out["result"][0]
    assert "dm7_1" in out["result"][0]["error"]
    out = _call("table_get", {"reference": "gec_10",
                              "table_number": "99-99"})
    assert "error" in out and "how_to_read" not in out


def test_result_cap_aliases_are_read_as_limit():
    for alias in ("top_k", "max_results", "n"):
        out = _call("table_search", {"query": "pile", alias: 3})
        assert "error" not in out, out
        assert 1 <= len(out["result"]) <= 3


def test_a_wide_search_fits_and_says_where_to_go_on():
    for query in ("pile", "soil", "table", "a"):
        out = dispatch.call_agent("table_db", "table_search",
                                  {"query": query, "limit": 50},
                                  allowed_agents=SCOPE)
        text = json.dumps(out, default=str)
        assert len(text) <= RESULT_BUDGET_CHARS, (query, len(text))
        if out.get("left_out"):
            nxt = out["next_offset"]
            assert nxt == len(out["result"])
            assert f"offset={nxt}" in out["note"]


def test_list_indexed_tables_counts_every_kind():
    listed = _call("list_indexed_tables", {})["result"]
    assert listed
    for ref in listed:
        assert ref["n_tables"] == ref["n_text"] + ref["n_lookup"] + \
            ref["n_page"]


def test_an_older_library_gives_an_empty_adapter_and_no_table_reader(
        monkeypatch):
    import geotech_references
    monkeypatch.setitem(sys.modules, "geotech_references._tables_db", None)
    monkeypatch.delattr(geotech_references, "_tables_db", raising=False)
    assert table_db_adapter._build() == ({}, {})
    assert reference_docs.table_catalogs_available() is False
    from funhouse_agent.deep.tools import reference_page_readers
    from funhouse_agent.reviewer import consultant_role
    assert reference_page_readers() == {"read_reference_figure"}
    role = consultant_role(True)
    assert "table_search" not in role and "read_reference_table" not in role


def test_the_librarian_is_told_what_it_can_read_of_a_table():
    from funhouse_agent.reviewer import consultant_role
    assert "read_reference_table call for a table printed only on its " \
           "page" in consultant_role(True, True)
    without = consultant_role(False, True)
    assert "read_reference_table" not in without
    assert "cannot be read here: cite the table and its printed page" \
        in without


# ---------------------------------------------------------------------------
# The page reader, on a made-up table
# ---------------------------------------------------------------------------

class _Reader:
    """A vision engine that answers every read with ``answer``."""

    accepts_jpeg = False

    def __init__(self, answer="Table 9-9 (made up): row A, column B = 1.5"):
        self.answer = answer
        self.prompts = []

    def vision_profile(self):
        return None

    def analyze_image(self, image, prompt):
        self.prompts.append(prompt)
        return self.answer


@pytest.fixture
def made_up_table(monkeypatch, tmp_path):
    """A six-page made-up PDF holding a made-up Table 9-9 over all six."""
    pymupdf = pytest.importorskip("pymupdf")
    from funhouse_agent import chart_reading
    doc = pymupdf.open()
    for i in range(6):
        page = doc.new_page()
        page.insert_text((72, 72), f"Table 9-9 A made-up table, sheet {i + 1}")
    path = tmp_path / "made_up.pdf"
    path.write_bytes(doc.tobytes())
    rec = {"reference": "made_up", "table_number": "9-9",
           "caption": "A made-up table", "pdf_path": str(path),
           "pdf_page_index": 0, "pages": [0, 1, 2, 3, 4, 5],
           "printed_page": "9-1", "page_estimated": False,
           "values": "page", "values_in": None}

    def table_get(ref, num, pdf_page_index=None, **kw):
        return dict(rec)

    def resolve_table_pdf(ref, num, pdf_page_index=None):
        return str(path), (0 if pdf_page_index is None else pdf_page_index)

    monkeypatch.setattr(tables_db, "table_get", table_get)
    monkeypatch.setattr(tables_db, "resolve_table_pdf", resolve_table_pdf)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    yield rec
    chart_reading.close_documents()


def _read(engine, **args):
    from funhouse_agent.vision_tools import dispatch_extended_tool
    return json.loads(dispatch_extended_tool(
        tool_name="read_reference_figure",
        arguments={"reference": "made_up", "table_number": "9-9",
                   "prompt": "read row A, column B", **args},
        engine=engine, attachments={}))


def test_a_read_sends_at_most_four_pages_and_names_the_rest(made_up_table):
    from funhouse_agent.reference_vision import TABLE_PAGES_MAX
    eng = _Reader()
    out = _read(eng, pdf_page_index=1)
    assert "error" not in out, out
    assert len(eng.prompts) == TABLE_PAGES_MAX == 4
    assert [p["pdf_page_index"] for p in out["pages_read"]] == [1, 2, 3, 4]
    assert out["pages_not_read"] == [5]
    assert "pdf_page_index=5" in out["pages_left_note"]
    for prompt in eng.prompts:
        assert "FIRST confirm you are looking at Table 9-9" in prompt
        assert "printed over 6 pages" in prompt
        assert "Request: read row A, column B" in prompt
        # a table read carries no chart wording
        assert "curve" not in prompt and "axis" not in prompt
    for page in out["pages_read"]:
        assert page["analysis"].startswith("Table 9-9")
        assert page["view"] and page["view_px"]
    assert out["table_number"] == "9-9"
    assert "second_reading" not in out
    assert len(json.dumps(out)) <= 16_000


def test_a_read_that_finds_no_table_says_no_value_was_read(made_up_table):
    from funhouse_agent.reference_vision import NOT_ON_PAGE
    eng = _Reader(answer=f"{NOT_ON_PAGE} The page shows body text.")
    out = _read(eng, pdf_page_index=5)
    assert len(eng.prompts) == 1
    assert "no value was read" in out["second_reading"]


def test_long_answers_are_cut_evenly_to_fit(made_up_table):
    from funhouse_agent.reference_vision import TABLE_RESULT_CHARS
    eng = _Reader(answer="Table 9-9 " + "made-up value 1.0; " * 900)
    out = _read(eng)
    assert len(json.dumps(out)) <= TABLE_RESULT_CHARS
    assert "shortened" in out["cut"]


def test_the_deep_tool_reaches_the_reader(made_up_table):
    from funhouse_agent.deep.tools import make_vision_tools
    eng = _Reader()
    tools = make_vision_tools(engine=eng, attachments={},
                              include={"read_reference_table"})
    tool = {t.name: t for t in tools}["read_reference_table"]
    out = json.loads(tool.invoke({"reference": "made_up",
                                  "table_number": "9-9",
                                  "pdf_page_index": 4,
                                  "prompt": "read row A"}))
    assert [p["pdf_page_index"] for p in out["pages_read"]] == [4, 5]
    assert len(eng.prompts) == 2


def test_the_lookup_report_names_a_table_read():
    from funhouse_agent.deep import lookup_report
    assert "read_reference_table" in lookup_report.LOOKUP_TOOLS

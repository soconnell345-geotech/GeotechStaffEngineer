"""The references helper can reach every printed table it is told it has.

The owner's gate for the reference release (BACKLOG REF-92): "the
references subagent has access to everything it's told it has". The
library's table catalogs (``geotech_references._tables_db``, REF-86) say
where each printed table's values are, and a fetched section's
``tables_held`` names a tool for each listed table. These tests run over
the library's real catalogs (its package data, read at run time; nothing
of a printed table is copied here beyond its number) and check, through
the app's own dispatch exactly as the helper calls it:

* every ``"text"`` row's section is returned by ``reference_db.
  reference_get``;
* every ``"lookup"`` row's function is a method of the app module
  serving that reference, by that name, and so is every function a row
  names in ``implemented_in``, except one that takes Python functions as
  inputs (the library marks it ``needs_python_callables``; no JSON call
  can pass it, so the module leaves it out, and the library's own tests
  keep it off every lookup row);
* every ``"page"`` row's ``how_to_read`` is a call the helper has (the
  ``read_reference_table`` call, with a page the library resolves), or,
  where its page cannot be read on this host, the plain statement that the
  values are only on that printed page, naming no reading tool;
* on a host with no reference PDFs the helper's tools and prompt name no
  table reader, and its search and fetch results say the page cannot be
  read here;
* a fetched section still carries ``figures_held``, ``tables_held`` and
  ``not_held``, its note's lookup clause is followed by the tool of this
  app that runs the function, and it stays within the reference cap.
"""

from __future__ import annotations

import importlib
import json
import re

import pytest

tables_db = pytest.importorskip("geotech_references._tables_db")

from funhouse_agent import dispatch, reference_docs  # noqa: E402
from funhouse_agent.adapters import table_db_adapter  # noqa: E402
from funhouse_agent.deep.tools import DEFAULT_REFERENCE_RESULT_CHARS  # noqa: E402

CAP = DEFAULT_REFERENCE_RESULT_CHARS
SCOPE = dispatch.REFERENCE_MODULES

#: Library sections that grew past 14,000 characters with the reach keys
#: (brief, 2026-11-13): each must still fit the cap through the app.
GREW = (("gec_10", "A.2.b"), ("fhwa_pavements", "5.4.3.a"),
        ("gec_11", "4.4.6.c"), ("gec_11", "9.2.6"), ("gec_11", "D.2"))

_READ_CALL = re.compile(
    r"read_reference_table\(reference='([^']*)', table_number='([^']*)', "
    r"pdf_page_index=(\d+),")


def _rows(kind=None) -> list:
    tables_db._ensure_db()
    conn = tables_db._ro_connect()
    try:
        sql = "SELECT * FROM tables"
        args = ()
        if kind:
            sql += " WHERE value_source = ?"
            args = (kind,)
        rows = [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()
    for r in rows:
        r["values"] = r["value_source"]
        r["pages"] = json.loads(r["pages"] or "[]") or (
            [r["pdf_page_index"]] if r["pdf_page_index"] is not None else [])
        r["implemented_in"] = json.loads(r["implemented_in"] or "[]")
    return rows


def _call(agent: str, method: str, params: dict) -> dict:
    """The result as the call_agent tool sends it: under the cap, JSON."""
    text = json.dumps(dispatch.call_agent(agent, method, params,
                                          allowed_agents=SCOPE), default=str)
    assert len(text) <= CAP, f"{agent}.{method}({params}): {len(text):,}"
    return json.loads(text)


def _get(row) -> dict:
    return _call("table_db", "table_get", {
        "reference": row["reference"], "table_number": row["table_number"],
        "pdf_page_index": row["pdf_page_index"]})


@pytest.fixture(autouse=True)
def _unpinned():
    """Each test looks at this host as it is unless it pins."""
    reference_docs.pin_chart_reads(None)
    yield
    reference_docs.pin_chart_reads(None)


@pytest.fixture
def no_pdf_host(tmp_path, monkeypatch):
    """A host with an empty docs folder, an empty cache, no packs and no
    fetcher; page reads looked for, not pinned."""
    docs = tmp_path / "docs"
    cache = tmp_path / "cache"
    docs.mkdir()
    cache.mkdir()
    monkeypatch.setenv("GEOTECH_REFERENCES_DOCS", str(docs))
    monkeypatch.setenv(reference_docs.CACHE_ENV, str(cache))
    monkeypatch.delenv("GEOTECH_REFERENCES_PACKS", raising=False)
    saved = reference_docs._FETCHER
    reference_docs.register_fetcher(None)
    yield
    reference_docs.register_fetcher(saved)


# ---------------------------------------------------------------------------
# The catalogs as the app reads them
# ---------------------------------------------------------------------------

def test_the_catalogs_hold_every_kind():
    rows = _rows()
    kinds = {k: sum(r["values"] == k for r in rows)
             for k in ("text", "lookup", "page")}
    assert sum(kinds.values()) == len(rows) > 1000
    assert all(kinds.values()), kinds
    listed = _call("table_db", "list_indexed_tables", {})["result"]
    assert sum(r["n_tables"] for r in listed) == len(rows)


def test_every_text_row_section_is_fetched_by_reference_get():
    failed = []
    rows = _rows("text")
    for row in rows:
        if not row["values_in"]:
            failed.append((row["reference"], row["table_number"],
                           "no section named"))
            continue
        out = _call("reference_db", "reference_get", {
            "reference": row["reference"], "section_id": row["values_in"]})
        if "error" in out or out.get("section_id") != row["values_in"]:
            failed.append((row["reference"], row["table_number"],
                           row["values_in"], str(out.get("error"))[:80]))
    assert rows and not failed, f"{len(failed)} of {len(rows)}: {failed[:10]}"


def _named_functions():
    """Every (reference, row, "module.function") a row names: its lookup
    and its implemented_in."""
    for row in _rows():
        names = list(row["implemented_in"])
        if row["values"] == "lookup" and row["values_in"]:
            names.insert(0, row["values_in"])
        for full in dict.fromkeys(names):
            yield row, full


def _takes_python_functions(reference: str, full: str) -> bool:
    """Whether the library function ``full`` ("module.function") of
    ``reference`` takes Python functions as inputs."""
    mod, _, name = f"geotech_references.{reference}.{full}".rpartition(".")
    fn = getattr(importlib.import_module(mod), name)
    return bool(getattr(fn, "needs_python_callables", False))


def test_every_named_function_is_a_method_of_the_reference_module():
    """A lookup row's function can be called through the app: the module
    serving the reference offers it by its name. So can every
    implemented_in function, except one that takes Python functions as
    inputs (no JSON call can pass them; such a function is never a lookup
    row's)."""
    missing = []
    checked = 0
    for row, full in _named_functions():
        checked += 1
        module, method = table_db_adapter.lookup_target(row["reference"],
                                                        full)
        if module is not None:
            assert method in dispatch._load_adapter(module).METHOD_REGISTRY
            continue
        is_lookup = row["values"] == "lookup" and full == row["values_in"]
        if is_lookup or not _takes_python_functions(row["reference"], full):
            missing.append((row["reference"], row["table_number"], full))
    assert checked > 300
    assert not missing, missing[:10]


def test_every_lookup_row_names_the_call_that_runs_it():
    failed = []
    for row in _rows("lookup"):
        step = _get(row)["how_to_read"]
        module, method = table_db_adapter.lookup_target(row["reference"],
                                                        row["values_in"])
        if module is None or f"call_agent('{module}', '{method}'" not in step:
            failed.append((row["reference"], row["table_number"], step[:90]))
    assert not failed, failed[:10]


def _check_page_step(row, step: str, readable_here: bool) -> str:
    """'' when ``step`` is a true next step for this page row, else why."""
    m = _READ_CALL.search(step)
    if m:
        if not readable_here:
            return "names the reader on a host that cannot read the page"
        ref, number, page = m.group(1), m.group(2), int(m.group(3))
        if (ref, number) != (row["reference"], row["table_number"]) \
                or page not in row["pages"]:
            return f"call does not name this table's page: {m.group(0)}"
        try:
            tables_db.resolve_table_pdf(ref, number, pdf_page_index=page)
        except FileNotFoundError:
            # not on this machine: readable only through a fetcher's
            # listing or the cache, which table_readable checked
            if not reference_docs.table_readable(ref, number, page):
                return "names the reader for a PDF not here"
        except KeyError as e:
            return f"the library cannot resolve it: {e}"
        return ""
    if "read_reference_table" in step or "read_reference_figure" in step:
        return "names a reader outside the call form"
    if step.endswith(reference_docs.no_table_read(
            row["table_number"], row["printed_page"], row["reference"])):
        if reference_docs.table_readable(row["reference"],
                                         row["table_number"],
                                         row["pdf_page_index"]):
            return "says not readable where the page can be read"
        return ""
    return "neither the read call nor the not-readable statement"


def test_every_page_row_gets_a_next_step_the_helper_has():
    """Over every page row, the next step from table_get is the read
    call, with a page of the table the library resolves, where this host
    reads the page; else the plain statement that the page cannot be read
    here."""
    charts = reference_docs.chart_reads_available()
    rows = _rows("page")
    failed = []
    for row in rows:
        step = _get(row)["how_to_read"]
        why = _check_page_step(row, step, charts)
        if why:
            failed.append((row["reference"], row["table_number"], why))
    assert len(rows) > 1000
    assert not failed, f"{len(failed)} of {len(rows)}: {failed[:10]}"
    if charts:
        # the helper has the tool the steps name
        from funhouse_agent.deep.agent import build_references_subagent
        names = {getattr(t, "name", None)
                 for t in build_references_subagent()["tools"]}
        assert "read_reference_table" in names


def test_search_hits_carry_the_same_next_steps():
    for query in ("bond strength", "resistance factors", "pile", "soil"):
        out = _call("table_db", "table_search", {"query": query, "limit": 50})
        hits = out["result"]
        assert hits, query
        for hit in hits:
            assert hit["values"] in ("text", "lookup", "page")
            step = hit["how_to_read"]
            assert "lookup_reference" not in step
            if hit["values"] == "text" and hit["values_in"]:
                assert "'reference_db', 'reference_get'" in step
            elif hit["values"] == "page":
                assert ("read_reference_table(" in step) or \
                    "cannot read that page" in step


# ---------------------------------------------------------------------------
# A host with no reference PDFs
# ---------------------------------------------------------------------------

def _one_page_row_per_reference() -> list:
    seen = {}
    for row in _rows("page"):
        seen.setdefault(row["reference"], row)
    return list(seen.values())


def test_a_host_without_the_pdfs_is_told_no_table_page_can_be_read(
        no_pdf_host):
    from funhouse_agent.agent import _build_agent_system_prompt
    from funhouse_agent.deep.agent import (build_primary_tools,
                                           build_references_subagent,
                                           references_description)

    assert reference_docs.chart_reads_available() is False
    rows = _one_page_row_per_reference()
    assert len(rows) > 20
    for row in rows:
        step = _get(row)["how_to_read"]
        assert "read_reference_table" not in step
        assert step == reference_docs.no_table_read(
            row["table_number"], row["printed_page"], row["reference"])
        assert f"printed only on printed page {row['printed_page']}" in step \
            or row["printed_page"] in (None, "")
        assert "this host cannot read that page" in step
    hits = _call("table_db", "table_search",
                 {"query": "bond strength", "limit": 10})["result"]
    assert any(h["values"] == "page" for h in hits)
    for hit in hits:
        assert "read_reference_table" not in hit["how_to_read"]
    # the helper's tools, prompt and description
    spec = build_references_subagent()
    names = {getattr(t, "name", None) for t in spec["tools"]}
    assert not names & {"read_reference_table", "read_reference_figure"}
    assert "read_reference_table" not in spec["system_prompt"]
    assert "a table whose values are printed only on its page cannot be " \
           "read here" in spec["system_prompt"]
    assert "Nor can it read a table whose values are printed only on its " \
           "page" in spec["description"]
    assert "read_reference_table" not in references_description(False)
    primary = {getattr(t, "name", None) for t in build_primary_tools()}
    assert "read_reference_table" not in primary
    v1 = _build_agent_system_prompt(dispatch.REFERENCE_MODULES)
    assert "read_reference_table" not in v1


def test_a_host_with_the_pdfs_offers_the_table_reader():
    from funhouse_agent.agent import _build_agent_system_prompt
    from funhouse_agent.deep.agent import (build_primary_tools,
                                           build_references_subagent)

    reference_docs.pin_chart_reads(True)
    spec = build_references_subagent()
    names = {getattr(t, "name", None) for t in spec["tools"]}
    assert {"read_reference_table", "read_reference_figure"} <= names
    assert "read_reference_table call for a table printed only on its " \
           "page" in spec["system_prompt"]
    assert "off its printed page" in spec["description"]
    primary = {getattr(t, "name", None) for t in build_primary_tools()}
    assert "read_reference_table" in primary
    v1 = _build_agent_system_prompt(dispatch.REFERENCE_MODULES)
    assert "### 13. read_reference_table" in v1


# ---------------------------------------------------------------------------
# What a fetched section says it can reach
# ---------------------------------------------------------------------------

def _sections_with_tables_held() -> list:
    from geotech_references import _retrieval_db
    _retrieval_db._ensure_db()
    conn = _retrieval_db._ro_connect()
    try:
        rows = conn.execute(
            "SELECT s.reference, s.section_id, r.reach_json FROM "
            "section_reach r JOIN sections s ON s.rowid = r.sec_rowid"
        ).fetchall()
    finally:
        conn.close()
    return [(r[0], r[1]) for r in rows if '"tables_held"' in (r[2] or "")]


def test_fetched_sections_carry_the_reach_keys_in_this_apps_words():
    from geotech_references import _reach
    from funhouse_agent.adapters.reference_db_adapter import APP_LOOKUP_TOOL
    clause = _reach.TABLE_ANSWERS["lookup"]
    sections = _sections_with_tables_held()
    assert len(sections) > 100
    # a fetch near the size budget goes without the library's note (its
    # keys kept); every note that has the lookup clause names the tool
    lookups = pages = named = 0
    for ref, sid in sections:
        out = _call("reference_db", "reference_get",
                    {"reference": ref, "section_id": sid})
        assert "error" not in out, (ref, sid, out.get("error"))
        held = out.get("tables_held")
        assert held, (ref, sid)
        note = out.get("note") or ""
        # no tool name this app lacks; the lookup clause names the app's
        assert "lookup_reference" not in json.dumps(out), (ref, sid)
        assert "table_get" not in note, (ref, sid)
        assert note.count(clause) == note.count(clause + APP_LOOKUP_TOOL)
        named += clause in note
        for answer in held.values():
            text = str(answer)
            if text.startswith("lookup: "):
                lookups += 1
                fn = text[len("lookup: "):].split(" ")[0]
                module, _m = table_db_adapter.lookup_target(ref, fn)
                assert module, (ref, sid, text)
            elif text.startswith("page: "):
                pages += 1
    assert lookups and pages and named


def test_the_sections_that_grew_still_fit_and_keep_their_keys():
    for ref, sid in GREW:
        out = _call("reference_db", "reference_get",
                    {"reference": ref, "section_id": sid})
        assert "error" not in out, (ref, sid, out.get("error"))
        assert {"figures_held", "tables_held", "not_held"} & set(out), \
            (ref, sid)


def test_a_no_pdf_host_is_told_no_page_can_be_read_in_the_note(no_pdf_host):
    from funhouse_agent.adapters.reference_db_adapter import NO_PAGE_READS
    from geotech_references import _reach
    for ref, sid in _sections_with_tables_held()[:200]:
        out = _call("reference_db", "reference_get",
                    {"reference": ref, "section_id": sid})
        note = out.get("note") or ""
        if any(str(a).startswith("page: ")
               for a in (out.get("tables_held") or {}).values()) and note:
            assert _reach.PAGE_NOTE not in note
            assert NO_PAGE_READS in note
            return
    pytest.fail("no fetched section with a page answer and a note")


# ---------------------------------------------------------------------------
# The v1 text agent: a table read is a tool it accepts where it is told of it
# ---------------------------------------------------------------------------

_V1_CALL = ('The table is printed only on its page.\n<tool_call>\n'
            '{"tool_name": "read_reference_table", "reference": "dm7_2", '
            '"table_number": "6-24", "pdf_page_index": 465, '
            '"prompt": "Read the first row"}\n</tool_call>')


class _Scripted:
    """A text engine that asks for one table read, then answers."""
    native_tool_calling = False

    def __init__(self):
        self.replies = [_V1_CALL, "Final answer."]

    def chat(self, user, system="", temperature=0):
        return self.replies.pop(0) if self.replies else "Final answer."


def _v1_ask(monkeypatch):
    """The v1 text agent's turn, its page reads recorded, not sent."""
    from funhouse_agent import agent as v1
    sent = []

    def fake_dispatch(name, args, engine, attachments, save_fn=None):
        sent.append((name, dict(args)))
        return json.dumps({"table_number": args.get("table_number"),
                           "answer": "a made-up read"})

    monkeypatch.setattr(v1, "dispatch_extended_tool", fake_dispatch)
    agent = v1.GeotechAgent(genai_engine=_Scripted(), reference_mode="off",
                            max_rounds=3)
    return agent, agent.ask("What does the table hold?"), sent


def test_the_v1_text_agent_reads_a_table_page_where_pdfs_are(monkeypatch):
    reference_docs.pin_chart_reads(True)
    agent, result, sent = _v1_ask(monkeypatch)
    assert "### 13. read_reference_table" in agent._system_prompt
    assert not [e for e in result.errors if e["type"] == "parse"], \
        result.errors
    assert [c["tool_name"] for c in result.tool_calls] == \
        ["read_reference_table"]
    # sent on under the figure reader's name, whose hook reads the table
    assert [(n, a.get("table_number")) for n, a in sent] == \
        [("read_reference_figure", "6-24")]


def test_the_v1_text_agent_has_no_table_reader_without_pdfs(monkeypatch,
                                                            no_pdf_host):
    agent, result, sent = _v1_ask(monkeypatch)
    assert "read_reference_table" not in agent._system_prompt
    assert any("Invalid tool_name 'read_reference_table'" in e["message"]
               for e in result.errors if e["type"] == "parse"), result.errors
    assert sent == [] and result.tool_calls == []

"""BACKLOG REF-114: the reference library's table reader in the core.

(1) ``vision_tools.APP_READER_TOOLS``, the readers an app's reference
library serves through a hook, named the figure reader and the worked
examples' page reader but no table reader, so an app had to send a table
read under the figure reader's name. It now names ``read_reference_table``,
served where the app fills the hook ``vision_tools.read_reference_table``
and an unknown tool where it does not, exactly as before.

(2) ``cited_documents`` did not count a ``read_reference_table`` call as a
reference source: a table read by the main agent left the document it
rendered "not looked up". It now counts as a figure read does, by the
reference the call names, whatever its text prints.

With ``GEOTECH_CITED_DOCUMENTS`` unset (off, the default) nothing the model
or the user is handed changes; only the turn's recorded check does. All
values here are made up; no model is called.
"""

from __future__ import annotations

import json

import pytest

from aec_agent_core import cited_documents as cd
from aec_agent_core import hooks, review_flags
from aec_agent_core import vision_tools as vt
from aec_agent_core.cited_pages import record_bound


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(review_flags.CITED_DOCUMENTS_ENV, raising=False)
    with hooks._LOCK:
        saved = (dict(hooks._HOOKS), dict(hooks._RESOLVED))
        hooks._HOOKS.clear()
        hooks._RESOLVED.clear()
    cd.clear_cache()
    yield
    cd.clear_cache()
    with hooks._LOCK:
        hooks._HOOKS.clear()
        hooks._RESOLVED.clear()
        hooks._HOOKS.update(saved[0])
        hooks._RESOLVED.update(saved[1])


# ---------------------------------------------------------------------------
# (1) The core's list of app readers names the table reader
# ---------------------------------------------------------------------------

def test_the_core_lists_the_table_reader_among_the_app_readers():
    assert "read_reference_table" in vt.APP_READER_TOOLS
    # the two it listed before stay
    assert {"read_reference_figure", "view_worked_example_source"} <= \
        set(vt.APP_READER_TOOLS)


def test_an_app_serves_the_table_reader_by_its_own_name():
    seen = []
    hooks.set_hook("vision_tools.read_reference_table",
                   lambda args, engine: (seen.append((args, engine))
                                         or json.dumps({"table": "read"})))
    args = {"reference": "ref_a", "table_number": "9-9", "prompt": "row X"}
    out = vt.dispatch_extended_tool("read_reference_table", dict(args),
                                    "an engine", {})
    assert json.loads(out) == {"table": "read"}
    assert seen == [(args, "an engine")]


def test_without_its_hook_the_table_reader_is_unknown_as_before():
    """An app that fills only the figure reader's hook (the geotech page
    today sends its table reads under that name) sees what it saw before:
    a call by the table reader's name is an unknown tool and never reaches
    the figure reader; a call by the figure reader's name still does."""
    calls = []
    hooks.set_hook("vision_tools.read_reference_figure",
                   lambda args, engine: (calls.append(dict(args))
                                         or json.dumps({"figure": "read"})))
    args = {"reference": "ref_a", "table_number": "9-9"}
    out = json.loads(vt.dispatch_extended_tool(
        "read_reference_table", dict(args), None, {}))
    assert out == {"error": "Unknown extended tool: read_reference_table"}
    assert calls == []
    out = json.loads(vt.dispatch_extended_tool(
        "read_reference_figure", dict(args), None, {}))
    assert out == {"figure": "read"}
    assert calls == [args]


# ---------------------------------------------------------------------------
# (2) A table read is a reference source
# ---------------------------------------------------------------------------

PROMPT = ("Check the bearing resistance factor for the made-up footing B-0, "
          "2.0 m wide.")
ANSWER = "The resistance factor is 1.23 (UFC 3-220-20, Table 6-11)."
DOCX = {"title": "Footing B-0 note",
        "markdown": "Resistance factor 1.23 per UFC 3-220-20 Table 6-11."}


def _conv(tmp_path):
    conv = tmp_path / "conv"
    (conv / "files").mkdir(parents=True)
    return conv


def _write_turn(conv, calls, prompt=PROMPT):
    """One turn as the activity log records it."""
    with open(conv / "activity.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"turn": 1, "event": "turn_start",
                             "agent": "primary", "prompt": prompt}) + "\n")
        for i, call in enumerate(calls):
            rid = f"r1_{i}_{call['name']}"
            fh.write(json.dumps({"turn": 1, "event": "tool_start",
                                 "agent": call.get("agent", "primary"),
                                 "name": call["name"], "run_id": rid,
                                 "args": call.get("args", {})}) + "\n")
            fh.write(json.dumps({"turn": 1, "event": "tool_end",
                                 "agent": call.get("agent", "primary"),
                                 "name": call["name"], "run_id": rid,
                                 "result": json.dumps(call["result"])})
                     + "\n")


def _read(reader):
    """A main-agent read of a page of the library's DM 7.2 (``dm7_2``,
    UFC 3-220-20) whose text names no document: only the call's
    ``reference`` says what was read."""
    key = "table_number" if reader == "read_reference_table" \
        else "figure_number"
    return {"name": reader,
            "args": {"reference": "dm7_2", key: "6-11",
                     "prompt": "the value in row A, column B",
                     "pdf_page_index": 7},
            "result": {key: "6-11", "pages_read": [8],
                       "reading": "Row A, column B: 1.23"}}


def _not_looked_up(res):
    return [r["cited"] for r in res.get("not_looked_up") or []]


def test_a_table_read_is_a_reference_read():
    assert "read_reference_table" in cd.REFERENCE_TOOLS
    assert cd.source_kind("read_reference_table", {}) == cd.REFERENCES
    assert cd.source_kind("read_reference_table", {}, "primary") == \
        cd.source_kind("read_reference_figure", {}, "primary")


def test_without_a_read_the_citation_is_not_looked_up(tmp_path):
    conv = _conv(tmp_path)
    _write_turn(conv, [])
    res = cd.check_text(ANSWER, record_folder=str(conv), user_words=[PROMPT])
    assert res["checked"] is True
    assert _not_looked_up(res) == ["UFC 3-220-20"]


@pytest.mark.parametrize("reader", ["read_reference_figure",
                                    "read_reference_table"])
def test_a_page_read_of_a_reference_looks_its_document_up(tmp_path, reader):
    """The table reader counts exactly as the figure reader does: by the
    reference the call names, though its text prints no document."""
    conv = _conv(tmp_path)
    _write_turn(conv, [_read(reader)])
    res = cd.check_text(ANSWER, record_folder=str(conv), user_words=[PROMPT])
    assert _not_looked_up(res) == []
    assert {"cited": "UFC 3-220-20", "document": "ufc-3-220-20",
            "by": "a references result"} in res["looked_up"]


def test_a_table_read_of_another_reference_does_not_look_this_one_up(
        tmp_path):
    conv = _conv(tmp_path)
    read = _read("read_reference_table")
    read["args"]["reference"] = "gec_11"
    _write_turn(conv, [read])
    res = cd.check_text(ANSWER, record_folder=str(conv), user_words=[PROMPT])
    assert _not_looked_up(res) == ["UFC 3-220-20"]


# ---------------------------------------------------------------------------
# Where it shows: only in cited-documents mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("calls", [[], [_read("read_reference_table")]],
                         ids=["no read", "table read"])
def test_mode_off_hands_the_model_and_the_user_the_same(tmp_path, calls):
    """Off (unset): the answer goes out as written, the model is given no
    note, and a writer's result is what the writer returned, whether or not
    a table read stands in the record."""
    assert review_flags.cited_documents() == "off"
    conv = _conv(tmp_path)
    _write_turn(conv, calls)
    logged = []
    check = cd.TurnCitationCheck(record_folder=str(conv),
                                 user_words=[PROMPT], log=logged.append)
    assert check.mode == "off"
    assert check.feedback(ANSWER) is None
    assert check.finish(ANSWER) == ANSWER
    written = json.dumps({"saved": "files/note.docx", "status": "ok"})
    with record_bound(str(conv)):
        assert vt._citations_checked("write_docx", dict(DOCX),
                                     written) == written
        assert cd.check_deliverable(DOCX.values(), "write_docx") == {}
    # only the record says what was checked, and it is the repair
    assert [e["action"] for e in logged] == ["none"]
    assert _not_looked_up(logged[0]) == ([] if calls else ["UFC 3-220-20"])


def test_note_mode_no_longer_flags_a_table_read(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "note")
    conv = _conv(tmp_path)
    _write_turn(conv, [])
    flagged = cd.TurnCitationCheck(record_folder=str(conv),
                                   user_words=[PROMPT]).finish(ANSWER)
    assert flagged != ANSWER and "UFC 3-220-20" in flagged
    cd.clear_cache()
    conv2 = _conv(tmp_path / "two")
    _write_turn(conv2, [_read("read_reference_table")])
    check = cd.TurnCitationCheck(record_folder=str(conv2),
                                 user_words=[PROMPT])
    assert check.finish(ANSWER) == ANSWER
    with record_bound(str(conv2)):
        assert cd.check_deliverable(DOCX.values(), "write_docx") == {}


def test_retry_mode_gives_no_note_after_a_table_read(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "retry")
    conv = _conv(tmp_path)
    _write_turn(conv, [_read("read_reference_table")])
    check = cd.TurnCitationCheck(record_folder=str(conv),
                                 user_words=[PROMPT])
    assert check.feedback(ANSWER) is None
    assert check.finish(ANSWER) == ANSWER

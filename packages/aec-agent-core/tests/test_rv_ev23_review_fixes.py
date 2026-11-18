"""The review of BACKLOG EV-23 (``module_work/ev23/REVIEW.md``): one test
per finding it fixed in ``aec_agent_core.cited_documents``.

With ``GEOTECH_CITED_DOCUMENTS`` unset (off, the default) nothing the model
or the user is handed changes; only the turn's recorded check does. All
values here are made up; no model is called.
"""

from __future__ import annotations

import json

import pytest

from aec_agent_core import cited_documents as cd
from aec_agent_core import hooks, review_flags


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
        hooks._HOOKS.update(saved[0])
        hooks._RESOLVED.clear()
        hooks._RESOLVED.update(saved[1])


PROMPT = "Check the resistance factor of the made-up footing B-0."


def _conv(tmp_path, calls):
    """One turn as the activity log records it."""
    conv = tmp_path / "conv"
    conv.mkdir()
    with open(conv / "activity.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"turn": 1, "event": "turn_start",
                             "agent": "primary", "prompt": PROMPT}) + "\n")
        for i, call in enumerate(calls):
            rid = f"r1_{i}_{call['name']}"
            fh.write(json.dumps({"turn": 1, "event": "tool_start",
                                 "agent": "primary", "name": call["name"],
                                 "run_id": rid,
                                 "args": call.get("args", {})}) + "\n")
            fh.write(json.dumps({"turn": 1, "event": "tool_end",
                                 "agent": "primary", "name": call["name"],
                                 "run_id": rid,
                                 "result": json.dumps(call["result"])})
                     + "\n")
    return conv


def _check(conv, answer):
    return cd.check_text(answer, record_folder=str(conv), user_words=[PROMPT])


def _flagged(res):
    return [r["cited"] for r in res.get("not_looked_up") or []]


# ---------------------------------------------------------------------------
# A table read whose every page failed is no source
# ---------------------------------------------------------------------------

TABLE_ANSWER = "The resistance factor is 1.23 (GEC-11, Table 9-9)."


def _table_read(rows):
    return {"name": "read_reference_table",
            "args": {"reference": "gec_11", "table_number": "9-9"},
            "result": {"reference": "gec_11", "table_number": "9-9",
                       "caption": "Table 9-9. Made-up resistance factors",
                       "pages_read": rows, "note": "read the rows named"}}


def test_a_table_read_whose_every_page_failed_is_no_source(tmp_path):
    conv = _conv(tmp_path, [_table_read([
        {"pdf_page": 41, "pdf_page_index": 40, "error": "the reader failed"},
        {"pdf_page": 42, "pdf_page_index": 41, "error": "the reader failed"},
    ])])
    res = _check(conv, TABLE_ANSWER)
    assert _flagged(res) == ["GEC-11"]
    assert not res.get("looked_up")


def test_a_table_read_with_one_page_read_is_still_a_source(tmp_path):
    conv = _conv(tmp_path, [_table_read([
        {"pdf_page": 41, "pdf_page_index": 40, "error": "the reader failed"},
        {"pdf_page": 42, "pdf_page_index": 41,
         "analysis": "Row B-0: 1.23"},
    ])])
    res = _check(conv, TABLE_ANSWER)
    assert _flagged(res) == []
    assert [r["cited"] for r in res["looked_up"]] == ["GEC-11"]

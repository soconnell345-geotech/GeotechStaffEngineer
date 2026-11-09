"""The page parsers of :mod:`funhouse_agent.coverage`: which pages a tool
call read or looked at, from the call itself. :mod:`funhouse_agent.cited_pages`
and the review suite's checks read a conversation's tool calls with them.

(The coverage ledger, the gate and ``document_coverage`` behind
``GEOTECH_COVERAGE``, which this file also pinned, were removed in BACKLOG
DR-40: ``module_work/dr40/CHECK.md``.) Offline: no model, no network.
"""

import json

from funhouse_agent import coverage as C


# ---------------------------------------------------------------------------
# What a call read
# ---------------------------------------------------------------------------

def test_pages_come_from_the_result_and_a_failed_call_reads_nothing():
    ref, got = C.pages_from_call(
        "read_document", {"handle": "doc_x", "pages": "0-40"},
        json.dumps({"handle": "doc_x", "pages_returned": "3-5,9",
                    "next": {"pages": "10-40"}}))
    assert ref == ("handle", "doc_x")
    assert got == [(3, "text"), (4, "text"), (5, "text"), (9, "text")]
    ref, got = C.pages_from_call(
        "read_pdf_text", {"source": "r.pdf"},
        json.dumps({"pages": [{"page": 2, "has_text_layer": True},
                              {"page": 3, "has_text_layer": False}]}))
    assert got == [(2, "text"), (3, "text_empty")]
    assert C.pages_from_call("analyze_pdf_page",
                             {"attachment_key": "r.pdf", "page": 7},
                             '{"page": 7}')[1] == [(7, "look")]
    assert C.pages_from_call("render_region",
                             {"attachment_key": "r.pdf", "page": 7},
                             '{"error": "no vision engine"}') == (None, [])
    _ref, got = C.pages_from_call(
        "sweep_pages", {"source": "r.pdf"},
        json.dumps({"pages_checked": "0-3",
                    "unanswered": [{"page": 2, "why": "x"}]}))
    assert got == [(0, "sweep"), (1, "sweep"), (3, "sweep")]
    # A search is not a read; nor are thumbnails.
    assert C.pages_from_call("search_document", {"handle": "h"},
                             '{"hits": [{"page": 4}]}') == (None, [])


def test_parse_and_compact_pages():
    assert C.parse_pages("2-4, 7,9-9") == [2, 3, 4, 7, 9]
    assert C.parse_pages([1, "3-4"], n_pages=4) == [1, 3]
    assert C.parse_pages(None) == [] and C.parse_pages(True) == []
    assert C.compact([9, 2, 3, 4, 7]) == "2-4,7,9"
    assert C.compact([0, 1], offset=1) == "1-2"


# N6: a page whose rows log_grid returned was read; measure is not a read.

def _grid(pages, rows, nxt=None):
    out = {"handle": "doc_g", "pages": pages,
           "rows": [{"page": p, "text": "x"} for p in rows]}
    if nxt is not None:
        out["next_offset"] = nxt
    return json.dumps(out)


def test_log_grid_rows_returned_in_full_are_reads():
    def got(args, result):
        ref, pages = C.pages_from_call("log_grid", args, result)
        assert ref in (None, ("handle", "doc_g"))
        return [p for p, how in pages if how == "text"]

    span = [7, 8, 9, 10, 12, 13, 14]
    a = {"handle": "doc_g", "pages": "7-10,12-14"}
    # windows of rows, in page order; a page may go on into the next window
    assert got(a, _grid(span, [7] * 24, 24)) == []
    assert got(dict(a, offset=24), _grid(span, [7] * 6 + [8] * 17, 47)) == [7]
    # a page whose rows end exactly at a window's end completes in the next
    assert got(dict(a, offset=47), _grid(span, [9] * 24, 71)) == [7, 8]
    assert got(dict(a, offset=71), _grid(span, [9] * 6 + [10] * 20)) == \
        [7, 8, 9, 10]
    # one call that returns every row
    assert got({"handle": "doc_g", "pages": "9"}, _grid([9], [9] * 30)) == [9]
    # no rows (rows=false; a scan's grid with no text): no page's content
    assert got({"handle": "doc_g", "pages": "12-14"},
               _grid([12, 13, 14], [])) == []
    assert got(a, json.dumps({"handle": "doc_g", "pages": span,
                              "n_rows": 60})) == []
    assert got(a, json.dumps({"error": "no such handle"})) == []
    # measure answers one question about a page: not a read
    assert C.pages_from_call("measure", {"source": "doc_g", "page": 0},
                             json.dumps({"page": 0, "value": {}})) == \
        (None, [])

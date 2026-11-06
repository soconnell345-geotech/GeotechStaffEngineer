"""find_like keeps to one budget for the whole search (Foundry brief 5, H2).

A loose example (a callout boxed with its leader) matched 978 places over 24
sheets, about 41 a page: under the per-page flood guard, so every candidate
was read — 49 contact sheets, 795 s, for 3 true instances of 28 counted.
Now the whole call keeps to ``GEOTECH_FIND_LIKE_BUDGET_S`` (default 300 s):
no sheet is sent once it is spent, and the result says the search was cut
short, how many candidates were read and which pages were not.

Replayed offline on a fake clock: each sheet read "takes" 16.2 s, the
brief-5 run's 795 s over 49 sheets.

Since BACKLOG PL-29 (b) the search is planlens.review's and the budget an
argument it is handed; planlens tests the search itself
(``planlens/review/tests/test_find_like.py``: ``test_the_default_budget``,
``test_a_loose_example_stops_at_the_budget_and_says_so``,
``test_with_no_budget_every_sheet_is_read``,
``test_a_search_inside_the_budget_is_unchanged``, ``test_page_ranges``,
which were this file's). What stays here is the binding: the app's setting,
read on every call, reaches the search through the agent build's host.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("pymupdf")
tf = pytest.importorskip("planlens.testing.tag_fixtures")
pytest.importorskip("planlens.document.findlike")

from funhouse_agent import document_tools  # noqa: E402
from funhouse_agent import find_like as fl  # noqa: E402

N_PAGES, PER_PAGE = 24, 41          # 984 candidates, as in the brief-5 run
SECONDS_PER_READ = 795 / 49


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class _Reader:
    """Reads every cell; the first of each page's cells as the target."""

    def __init__(self, clock):
        self.clock = clock
        self.calls = 0

    def analyze_image(self, png, prompt=""):
        self.calls += 1
        self.clock.t += SECONDS_PER_READ
        ids = [int(x) for x in png.decode().split(",")]
        return "\n".join(f"#{i} | {'GCE' if i % PER_PAGE == 1 else 'GCG'} "
                         f"| callout" for i in ids)


@pytest.fixture
def loose_example(monkeypatch):
    from planlens.document import Document

    hits = [SimpleNamespace(page=p, bbox=(10.0 + k, 10.0, 15.0 + k, 14.0),
                            context="callout", score=0.6, points_to=None)
            for p in range(N_PAGES) for k in range(PER_PAGE)]

    def search(self, page, bbox, pages=None, **kw):
        return {"hits": hits, "pages": list(range(N_PAGES)), "warnings": [],
                "example": {"page": 0, "bbox": list(bbox), "height_pt": 18.7}}

    def sheets(self, cands, per_sheet=20):
        out = []
        for s in range(0, len(cands), per_sheet):
            ids = list(range(s + 1, min(s + per_sheet, len(cands)) + 1))
            out.append((",".join(map(str, ids)).encode(), ids))
        return out

    monkeypatch.setattr(Document, "find_like", search)
    monkeypatch.setattr(Document, "like_sheets", sheets)
    monkeypatch.setattr(fl, "VERIFY_WORKERS", 1)
    clock = _Clock()
    monkeypatch.setattr(fl, "_clock", clock)
    return clock


EXAMPLE = [592.8, 364.3, 639.7, 382.7]


def _run(clock):
    """One find_like through the binding, as an agent build calls it."""
    engine = _Reader(clock)
    pdf = tf.build_synthetic_tag_set(n_pages=2).pdf
    raw = document_tools.find_like(
        {"attachment_key": "set.pdf", "page": 0, "bbox": EXAMPLE,
         "text": "GCE"}, {"set.pdf": pdf},
        host=document_tools.host_for(engine))
    return json.loads(raw), engine


def test_the_setting_is_read_on_every_call(monkeypatch):
    host = document_tools.host_for(None)
    monkeypatch.delenv(document_tools.FIND_LIKE_BUDGET_ENV, raising=False)
    assert fl.budget_s(host.find_like_budget()) == fl.DEFAULT_BUDGET_S < 795
    monkeypatch.setenv(document_tools.FIND_LIKE_BUDGET_ENV, "0")
    assert fl.budget_s(host.find_like_budget()) is None
    monkeypatch.setenv(document_tools.FIND_LIKE_BUDGET_ENV, "120")
    assert fl.budget_s(host.find_like_budget()) == 120.0


def test_unset_the_search_stops_at_the_default_budget(loose_example,
                                                      monkeypatch):
    monkeypatch.delenv(document_tools.FIND_LIKE_BUDGET_ENV, raising=False)
    out, engine = _run(loose_example)
    total = N_PAGES * PER_PAGE
    # 300 s at 16.2 s a sheet: 19 sheets sent, not all 50
    assert engine.calls == 19 < -(-total // fl.PER_SHEET)
    assert out["status"] == "cut_short"
    assert out["candidates_read"] + out["not_read"] == total


def test_zero_reads_every_sheet(loose_example, monkeypatch):
    monkeypatch.setenv(document_tools.FIND_LIKE_BUDGET_ENV, "0")
    out, engine = _run(loose_example)
    assert engine.calls == -(-N_PAGES * PER_PAGE // fl.PER_SHEET)
    assert "status" not in out and "not_read" not in out


def test_a_large_setting_leaves_the_search_unchanged(loose_example,
                                                     monkeypatch):
    monkeypatch.setenv(document_tools.FIND_LIKE_BUDGET_ENV, "100000")
    out, engine = _run(loose_example)
    assert "status" not in out and "note" not in out
    assert out["instances"] == N_PAGES

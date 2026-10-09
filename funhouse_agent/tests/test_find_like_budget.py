"""find_like keeps to one budget for the whole search (Foundry brief 5, H2).

A loose example (a callout boxed with its leader) matched 978 places over 24
sheets, about 41 a page: under the per-page flood guard, so every candidate
was read — 49 contact sheets, 795 s, for 3 true instances of 28 counted.
Now the whole call keeps to ``GEOTECH_FIND_LIKE_BUDGET_S`` (default 300 s):
no sheet is sent once it is spent, and the result says the search was cut
short, how many candidates were read and which pages were not.

Replayed offline on a fake clock: each sheet read "takes" 16.2 s, the
brief-5 run's 795 s over 49 sheets.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fitz")
tf = pytest.importorskip("planlens.testing.tag_fixtures")
pytest.importorskip("planlens.document.findlike")

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


def _run(clock):
    engine = _Reader(clock)
    pdf = tf.build_synthetic_tag_set(n_pages=2).pdf
    out = fl.find_like(pdf, 0, [592.8, 364.3, 639.7, 382.7], engine,
                       text="GCE")
    return out, engine


def test_the_default_budget(monkeypatch):
    monkeypatch.delenv(fl.BUDGET_ENV, raising=False)
    assert fl.budget_s() == fl.DEFAULT_BUDGET_S < 795
    monkeypatch.setenv(fl.BUDGET_ENV, "0")
    assert fl.budget_s() is None


def test_a_loose_example_stops_at_the_budget_and_says_so(loose_example,
                                                         monkeypatch):
    monkeypatch.delenv(fl.BUDGET_ENV, raising=False)
    out, engine = _run(loose_example)
    total = N_PAGES * PER_PAGE
    n_sheets = -(-total // fl.PER_SHEET)
    # 300 s at 16.2 s a sheet: 19 sheets sent, not all 50
    assert engine.calls == 19 < n_sheets
    assert out["status"] == "cut_short"
    assert out["candidates"] == total
    assert out["candidates_read"] == 19 * fl.PER_SHEET
    assert out["candidates_read"] + out["not_read"] == total
    assert "CUT SHORT" in out["note"] and "NOT read" in out["note"]
    assert f'pages="{out["not_read_pages"]}"' in out["note"]
    # The pages not read are named as a pages= string the tool takes back.
    first_unread_page = (19 * fl.PER_SHEET) // PER_PAGE
    assert out["not_read_pages"].startswith(str(first_unread_page))
    assert out["not_read_pages"].endswith(str(N_PAGES - 1))
    # Counts cover only what was read: one confirmed GCE per page read.
    assert out["instances"] == len({(i - 1) // PER_PAGE
                                    for i in range(1, 19 * 20 + 1)
                                    if i % PER_PAGE == 1})
    # No unread candidate is listed as "uncertain" (that list stays short).
    assert len(out["uncertain"]) <= 20


def test_with_no_budget_every_sheet_is_read(loose_example, monkeypatch):
    monkeypatch.setenv(fl.BUDGET_ENV, "0")
    out, engine = _run(loose_example)
    assert engine.calls == -(-N_PAGES * PER_PAGE // fl.PER_SHEET)
    assert "status" not in out and "not_read" not in out


def test_a_search_inside_the_budget_is_unchanged(loose_example, monkeypatch):
    monkeypatch.setenv(fl.BUDGET_ENV, "100000")
    out, engine = _run(loose_example)
    assert "status" not in out and "note" not in out
    assert out["instances"] == N_PAGES


def test_page_ranges():
    assert fl._page_ranges([5, 3, 4, 9, 11, 10]) == "3-5,9-11"
    assert fl._page_ranges([7]) == "7"

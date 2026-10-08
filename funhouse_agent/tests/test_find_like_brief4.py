"""find_like and an example crossed by linework — Foundry brief 4, replayed.

Two live runs (GPT-5.6 Sol, 2026-10-07) passed as their example the callout
whose lettering sits on a heavy grid line. The search returned 400 candidates
(the cap) and 367 (no warning), and the tool read every one: 18-20 vision
calls and 180-289 s each. A third run's clean example gave 43 and one call.
The replay of TRACE_REVIEW §3.5 reproduces those counts offline in seconds.

Two fixes, tested here end to end on the numpy matcher (the one FIPS hosts
use): planlens leaves linework running through the example box out of the
template, and the tool reads nothing when a page still floods
(:data:`funhouse_agent.find_like.FLOOD_PER_PAGE`) — it says to box a cleaner
copy instead.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

fitz = pytest.importorskip("fitz")
tf = pytest.importorskip("planlens.testing.tag_fixtures")
pytest.importorskip("planlens.document.findlike")

from funhouse_agent import find_like as fl  # noqa: E402
from funhouse_agent import vision_probe, vision_view  # noqa: E402
from funhouse_agent.vision_tools import _dispatch_find_like  # noqa: E402

#: The example the part C run passed: a 59 pt zoom's view and the tag's box.
T1_VIEW = [419.5, 98.8, 548.2, 169.5]
T1_IMAGE_BOX = [549, 525, 628, 591]


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    for e in (vision_view.BUDGET_ENV, vision_view.DETAIL_ENV,
              vision_view.CHART_BUDGET_ENV, vision_view.POLICY_ENV,
              vision_view.MAX_PX_ENV):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv(vision_probe.PROBE_ENV, "0")
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("PLANLENS_FINDLIKE_BACKEND", "numpy")
    vision_probe.clear_cache()


class Counts:
    """Reads every cell as the target (what it reads does not matter here;
    how many sheets it is asked to read does)."""

    def __init__(self):
        self.calls = 0

    def analyze_image(self, image_bytes, prompt=""):
        self.calls += 1
        return "\n".join(f"#{i} | GCE | callout" for i in range(1, 400))


def test_the_grid_line_example_is_read_in_a_few_sheets_not_twenty():
    gt = tf.build_synthetic_tag_set()
    engine = Counts()
    out = json.loads(_dispatch_find_like(
        {"attachment_key": "t", "page": 0, "view": T1_VIEW,
         "image_box": T1_IMAGE_BOX, "text": "GCE", "pages": "0"},
        engine, {"t": gt.pdf}))
    assert "error" not in out and "status" not in out, out
    assert "linework_left_out" in out["example"]
    # 367 candidates read in 19 sheets before; now a few dozen in a few
    assert 1 <= engine.calls <= 4, engine.calls
    callouts = [t for t in gt.tags if t.page == 0 and t.text == "GCE"
                and t.kind == "callout"]
    for t in callouts:
        cx, cy = (t.bbox[0] + t.bbox[2]) / 2, (t.bbox[1] + t.bbox[3]) / 2
        assert any(f["bbox"][0] - 2 <= cx <= f["bbox"][2] + 2
                   and f["bbox"][1] - 2 <= cy <= f["bbox"][3] + 2
                   for f in out["found"]), t


def test_a_flooding_example_is_not_read(monkeypatch):
    """Whatever still floods a page is not read: no contact sheets, no
    vision calls, no count of instances — the agent is told to box a copy
    not crossed by linework."""
    from planlens.document import Document

    def flooded(self, page, bbox, pages=None, **kw):
        hits = [SimpleNamespace(page=0, bbox=(i, 10.0, i + 5.0, 14.0),
                                context="unanchored", score=0.6,
                                points_to=None)
                for i in range(250)]
        hits.append(SimpleNamespace(page=1, bbox=(5.0, 5.0, 9.0, 9.0),
                                    context="callout", score=0.9,
                                    points_to=(20.0, 20.0)))
        return {"hits": hits, "pages": [0, 1], "warnings": [],
                "example": {"page": 0, "bbox": list(bbox), "height_pt": 4.0}}

    monkeypatch.setattr(Document, "find_like", flooded)
    monkeypatch.setattr(Document, "like_sheets",
                        lambda self, *a, **k: pytest.fail("sheets drawn"))
    gt = tf.build_synthetic_tag_set(n_pages=2)
    engine = Counts()
    out = json.loads(_dispatch_find_like(
        {"attachment_key": "t", "page": 0, "bbox": [490, 136, 501, 141],
         "text": "GCE"}, engine, {"t": gt.pdf}))
    assert engine.calls == 0
    assert out["status"] == "example_matches_linework"
    assert out["candidates"] == 251
    assert out["candidates_by_page"] == {"0": 250, "1": 1}
    assert "instances" not in out
    assert "NOT a count of instances" in out["note"]
    assert "not crossed by linework" in out["note"]
    assert "page(s) 0" in out["note"]


def test_flooded_pages():
    hits = ([SimpleNamespace(page=3)] * (fl.FLOOD_PER_PAGE + 1)
            + [SimpleNamespace(page=4)] * fl.FLOOD_PER_PAGE)
    assert fl.flooded_pages(hits) == {3: fl.FLOOD_PER_PAGE + 1}
    assert fl.flooded_pages(hits[-10:]) == {}

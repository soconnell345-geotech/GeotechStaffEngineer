"""A log sheet with no depth ruler is judged on the hole's other sheets
(BACKLOG RI-199).

A boring log runs to several sheets and planlens may fit no depth ruler on
the first or the last. Those sheets are continuations of the hole: a depth on
the first lies between the ground and the bottom of the next ruled sheet, a
depth on the last between the top of the last ruled sheet and the hole's
printed total depth. Before RI-199 every depth was judged against the union
of the RULED sheets only, so good values on the ruler-less ends were dropped
(today's readers, ``checks="refuse"``) or flagged ``out_of_range`` at
confidence 0.5 (model first, ``checks="flag"``).

Every hole, depth and page here is invented.
"""

from __future__ import annotations

import pytest

from report_ingest.log_reader import (DEPTH_SLACK, LogReading, _Builder,
                                      _depth_window, _ruled_windows,
                                      _sheet_windows)

#: A letter page's height in points.
HEIGHT = 792.0


class _Ruler:
    """A made-up ruler: ``top`` ft at the page's top edge, ten feet down
    the page."""

    step = 1.0

    def __init__(self, top: float) -> None:
        self.top = top

    def value_at(self, y: float) -> float:
        return self.top + 10.0 * y / HEIGHT


class _Grid:
    def __init__(self, rulers):
        self.rulers = rulers


class _Doc:
    class _Summary:
        height = HEIGHT

    def summary(self, page):
        return self._Summary()


#: Four sheets of one made-up hole; rulers found on the middle two only
#: (10-20 ft and 20-30 ft), none on the first (0-10) or the last (30-40).
PAGES = [10, 11, 12, 13]
GRID = _Grid({11: _Ruler(10.0), 12: _Ruler(20.0)})
SLACK = DEPTH_SLACK * 1.0


def _reading(rows, total=38.0):
    """A made-up boring: ``rows`` is ``[(page, top), ...]``, one sample
    each."""
    return LogReading.model_validate({
        "investigation_id": "QX-9", "kind": "boring", "depth_unit": "ft",
        "units_known": True, "total_depth": total,
        "samples": [{"sample_id": f"S-{n + 1}", "top": top,
                     "bottom": top + 1.5, "kind": "spt",
                     "prov": {"page": page, "bbox": [10, 20, 90, 30]}}
                    for n, (page, top) in enumerate(rows)],
        "pages_read": PAGES})


def _build(reading, checks, continued=True):
    ruled = _ruled_windows(_Doc(), GRID, PAGES)
    kw = {}
    if continued:
        kw = {"sheets": _sheet_windows(ruled, PAGES, 38.0, SLACK),
              "ruled_pages": list(ruled)}
    builder = _Builder(reading, _depth_window(_Doc(), GRID, PAGES), PAGES,
                       "R-made-up", slack=SLACK, checks=checks, **kw)
    return builder, builder.build()


class TestSheetWindows:
    def test_ruled_sheets_keep_their_own_window(self):
        ruled = _ruled_windows(_Doc(), GRID, PAGES)
        assert ruled == {11: (9.5, 20.5), 12: (19.5, 30.5)}
        sheets = _sheet_windows(ruled, PAGES, 38.0, SLACK)
        assert sheets[11] == ruled[11] and sheets[12] == ruled[12]

    def test_the_first_sheet_runs_from_the_ground_to_the_next_ruler(self):
        sheets = _sheet_windows(_ruled_windows(_Doc(), GRID, PAGES), PAGES,
                                38.0, SLACK)
        assert sheets[10] == (-SLACK, 20.5)

    def test_the_last_sheet_runs_to_the_hole_s_total_depth(self):
        sheets = _sheet_windows(_ruled_windows(_Doc(), GRID, PAGES), PAGES,
                                38.0, SLACK)
        assert sheets[13] == (19.5, 38.0 + SLACK)

    def test_with_no_total_depth_the_last_ruler_is_carried_on(self):
        sheets = _sheet_windows(_ruled_windows(_Doc(), GRID, PAGES), PAGES,
                                None, SLACK)
        assert sheets[13] == (19.5, 30.5 + 11.0)

    def test_a_sheet_between_two_rulers_lies_between_them(self):
        grid = _Grid({10: _Ruler(0.0), 12: _Ruler(20.0)})
        ruled = _ruled_windows(_Doc(), grid, [10, 11, 12])
        assert _sheet_windows(ruled, [10, 11, 12])[11] == (-0.5, 30.5)

    def test_no_ruler_anywhere_gives_nothing_to_continue(self):
        assert _sheet_windows({}, PAGES, 38.0, SLACK) == {}


@pytest.mark.parametrize("checks", ["refuse", "flag"])
class TestTheBuilder:
    ENDS = [(10, 2.0), (10, 7.0), (11, 15.0), (13, 33.0), (13, 36.5)]

    def test_values_on_the_ruler_less_ends_are_kept_as_good(self, checks):
        builder, inv = _build(_reading(self.ENDS), checks)
        assert [s.top.value for s in inv.samples] == [2.0, 7.0, 15.0, 33.0,
                                                      36.5]
        assert not [r for r in builder.unresolved
                    if "outside" in r.get("why", "")
                   and r["what"].endswith(" top")]
        if checks == "flag":
            assert all(s.prov.confidence > 0.5 for s in inv.samples)

    def test_before_ri199_they_were_lost(self, checks):
        """The old gate, the union of the ruled sheets (9.5 to 30.5 ft),
        on the same reading: the ends fall outside it."""
        builder, inv = _build(_reading(self.ENDS), checks, continued=False)
        outside = [r for r in builder.unresolved
                   if "outside" in r.get("why", "")
                   and r["what"].endswith(" top")]
        assert len(outside) == 4
        if checks == "refuse":
            assert [s.top.value for s in inv.samples] == [15.0]

    def test_a_value_truly_outside_its_sheet_still_flags(self, checks):
        # 26 ft cannot be on the first sheet: the next ruled sheet ends at
        # 20.5 ft. 12 ft cannot be on the last: the last ruled sheet starts
        # at 19.5 ft. 41 ft is past the hole's 38 ft total depth.
        builder, inv = _build(_reading([(10, 26.0), (13, 12.0), (13, 41.0),
                                        (11, 25.0)]), checks)
        outside = [r for r in builder.unresolved
                   if "outside" in r.get("why", "")
                   and r["what"].endswith(" top")]
        assert sorted(r["value"] for r in outside) == [12.0, 25.0, 26.0,
                                                       41.0]
        own = [r for r in outside if r["value"] == 25.0][0]
        assert "the scale on this log reads (9.5 to 20.5)" in own["why"]
        other = [r for r in outside if r["value"] == 26.0][0]
        assert "other sheets leave for page 10" in other["why"]
        if checks == "refuse":
            assert inv.samples == []
        else:
            assert all(s.prov.confidence == 0.5 for s in inv.samples)


# ---------------------------------------------------------------------------
# model first: the window spans the whole hole, across its calls
# ---------------------------------------------------------------------------

def test_model_first_judges_a_ruler_less_call_on_the_hole_s_other_sheets(
        monkeypatch):
    """Two calls of one hole: the first on the ruled sheets 11-12, the
    second on the ruler-less last sheet 13 alone. Its 33 ft sample is good;
    its 12 ft sample cannot be on that sheet."""
    import report_ingest.visual_scales as vs
    from report_ingest import model_first as mf
    from report_ingest import model_first_read as mfr
    from report_ingest.model_first_pages import PageGroup
    from report_ingest.tests.fake_engine import FakeEngine

    monkeypatch.setattr(vs, "log_grid_for", lambda d, pages, *a, **k: (
        _Grid({p: GRID.rulers[p] for p in pages if p in GRID.rulers}), {}))

    def log(rows):
        return mf.LogReading.model_validate({
            "investigation_id": "QX-9", "kind": "boring", "depth_unit": "ft",
            "total_depth": 38.0,
            "samples": [{"sample_id": f"S-{top:g}", "top": top,
                         "bottom": top + 1.5, "kind": "spt",
                         "prov": {"page": page}} for page, top in rows]})

    engine = FakeEngine([{"final": log([(11, 15.0), (12, 25.0)])},
                         {"final": log([(13, 33.0), (13, 12.0)])}])
    engine.accepts_pdf = False

    class Doc(_Doc):
        n_pages = 20

    monkeypatch.setattr(mfr._Caller, "read", lambda self, pages, spent: (
        [(list(pages), engine.turns.pop(0)["final"])], []), raising=True)
    groups = [PageGroup("logs", [11, 12], key="QX-9"),
              PageGroup("logs", [13], key="QX-9")]
    blob = mfr.read_logs(Doc(), groups, engine, report_id="R-made-up")[0]
    tops = {s["top"]["value"]: s["prov"]["confidence"]
            for s in blob["investigation"]["samples"]}
    assert set(tops) == {15.0, 25.0, 33.0, 12.0}
    outside = [r for r in blob["unresolved"]
               if "outside" in r.get("why", "")
                   and r["what"].endswith(" top")]
    assert [r["value"] for r in outside] == [12.0]
    assert tops[33.0] > 0.5 and tops[12.0] == 0.5

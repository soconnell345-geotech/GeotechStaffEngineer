"""BACKLOG RI-157 (narrowed 2026-10-29): a two-page probe or cone sheet whose
page 2 starts where page 1 stops is ONE record, not two.

The reconciler's continuation fold joins them as it joins a log's sheets:
one name (or a name adding only "continued"), the next page, and the
series carrying on within a stated tolerance. Two soundings of one name
whose depths overlap or leave a larger gap stay two. Every name, depth and
count here is made up.
"""

from __future__ import annotations

from typing import List, Sequence

from report_ingest.model import (CPTData, CPTPoint, DCPData, DCPPoint,
                                 Investigation, Quantity, ReportRecord)
from report_ingest.reconciler import (SOUNDING_JOIN_TOL_M,
                                      _fold_continuations, reconcile)


def _m(value: float) -> Quantity:
    return Quantity(value=round(value, 3), unit="m")


def _probe(name: str, page: int, depths: Sequence[float],
           blows: int = 4, sheet: str = "") -> Investigation:
    return Investigation(
        investigation_id=name, kind="dcp", depth_unit="m", pages=[page],
        sheet=sheet,
        dcp=DCPData(points=[DCPPoint(depth=_m(d), blows=blows + n % 3)
                            for n, d in enumerate(depths)],
                    increment=_m(0.1), test_type="DPL"))


def _steps(top: float, bottom: float, step: float = 0.1) -> List[float]:
    n = int(round((bottom - top) / step))
    return [top + step * k for k in range(1, n + 1)]


def _record(*invs: Investigation) -> ReportRecord:
    return ReportRecord(investigations=list(invs))


def test_a_probe_continued_on_the_next_page_is_one_record():
    first = _probe("DP-3", 4, _steps(0.0, 3.0))
    later = _probe("DP-3", 5, _steps(3.0, 5.2))
    record = _record(first, later)
    qa = []
    _fold_continuations(record, qa)
    assert len(record.investigations) == 1
    joined = record.investigations[0]
    assert joined.pages == [4, 5]
    depths = [p.depth.value for p in joined.dcp.points]
    assert len(depths) == 52 and depths[0] == 0.1 and depths[-1] == 5.2
    assert depths == sorted(depths)
    assert len(qa) == 1 and qa[0].where == "DP-3.sheets"
    assert "series starts at 3.10 m" in qa[0].detail


def test_a_title_adding_only_continued_is_the_same_probe():
    first = _probe("DP-03", 4, _steps(0.0, 2.0))
    later = _probe("DP-3 (continued)", 5, _steps(2.0, 4.0))
    record = _record(first, later)
    _fold_continuations(record, [])
    assert [inv.investigation_id for inv in record.investigations] == \
        ["DP-03"]
    assert len(record.investigations[0].dcp.points) == 40


def test_a_reading_printed_on_both_pages_is_kept_once():
    first = _probe("DP-3", 4, _steps(0.0, 2.0))
    later = _probe("DP-3", 5, [2.0] + _steps(2.0, 3.0))
    record = _record(first, later)
    qa = []
    _fold_continuations(record, qa)
    assert len(record.investigations) == 1
    assert len(record.investigations[0].dcp.points) == 30
    assert any("1 reading(s) printed again" in v for v in qa[0].values)


def test_a_gap_larger_than_the_tolerance_is_not_joined():
    first = _probe("DP-3", 4, _steps(0.0, 3.0))
    # Page 2 starts 0.5 m below where page 1 stops: more than the larger
    # of the stated floor and one and a half of the 0.1 m spacing.
    later = _probe("DP-3", 5, _steps(3.5, 5.0))
    assert 0.5 > max(SOUNDING_JOIN_TOL_M, 0.15)
    record = _record(first, later)
    _fold_continuations(record, [])
    assert len(record.investigations) == 2


def test_twins_of_one_name_starting_again_at_the_surface_are_not_joined():
    first = _probe("DP-3", 4, _steps(0.0, 3.0))
    later = _probe("DP-3", 5, _steps(0.0, 2.5))
    record = _record(first, later)
    _fold_continuations(record, [])
    assert len(record.investigations) == 2


def test_overlapping_depths_are_not_joined():
    first = _probe("DP-3", 4, _steps(0.0, 3.0))
    later = _probe("DP-3", 5, _steps(2.0, 4.0))
    record = _record(first, later)
    _fold_continuations(record, [])
    assert len(record.investigations) == 2


def test_another_name_or_a_page_between_is_not_joined():
    record = _record(_probe("DP-3", 4, _steps(0.0, 3.0)),
                     _probe("DP-4", 5, _steps(3.0, 5.0)))
    _fold_continuations(record, [])
    assert len(record.investigations) == 2
    record = _record(_probe("DP-3", 4, _steps(0.0, 3.0)),
                     _probe("DP-3", 6, _steps(3.0, 5.0)))
    _fold_continuations(record, [])
    assert len(record.investigations) == 2


def test_a_later_page_printing_sheet_one_is_not_joined():
    record = _record(_probe("DP-3", 4, _steps(0.0, 3.0), sheet="1 of 1"),
                     _probe("DP-3", 5, _steps(3.0, 5.0), sheet="1 of 1"))
    _fold_continuations(record, [])
    assert len(record.investigations) == 2


def test_a_cone_sounding_continued_is_one_record():
    def cone(page, depths):
        return Investigation(
            investigation_id="CPT-2", kind="cpt", depth_unit="m",
            pages=[page],
            cpt=CPTData(points=[CPTPoint(depth=_m(d),
                                         qc=Quantity(value=3.5, unit="MPa"))
                                for d in depths]))
    record = _record(cone(8, _steps(0.0, 10.0, 0.05)),
                     cone(9, _steps(10.0, 18.0, 0.05)))
    _fold_continuations(record, [])
    assert len(record.investigations) == 1
    assert record.investigations[0].cpt.points[-1].depth.value == 18.0


def test_the_count_check_counts_a_joined_sounding_once():
    record = _record(_probe("CP-1", 2, _steps(0.0, 3.0)),
                     _probe("CP-1", 3, _steps(3.0, 6.0)))
    for inv in record.investigations:     # cones, for the counted kind
        inv.kind = "cpt"
        inv.cpt = CPTData(points=[CPTPoint(depth=p.depth)
                                  for p in inv.dcp.points])
        inv.dcp = None
    record.general.cptCount = 1
    reconcile(record)
    assert len(record.investigations) == 1
    assert not [q for q in record.qa if q.kind == "count_mismatch"]

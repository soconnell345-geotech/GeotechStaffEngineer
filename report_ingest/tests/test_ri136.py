"""RI-136: a printed N of '-' (no test, no count) is no printed N: not
written, not looked for, so the read-back gate agrees. Made-up values."""

from __future__ import annotations

import pytest

from report_ingest.diggs_writer import diggs_roundtrip_gate, write_diggs
from report_ingest.model import Investigation, Layer, Quantity, SPT


def _m(value):
    return Quantity(value=value, unit="m")


def _hole(**spt):
    base = dict(depth_top=_m(3.0), depth_bottom=_m(3.45))
    base.update(spt)
    return Investigation(
        investigation_id="B-1", depth_unit="m", total_depth=_m(12.0),
        layers=[Layer(top=_m(0.0), bottom=_m(12.0),
                      description="SILTY SAND", uscs="SM")],
        spt=[SPT(**base)])


@pytest.mark.parametrize("dash", ["-", "–", "—", " - "])
def test_a_dash_round_trips_with_no_diffs(dash):
    hole = _hole(blows=[11, 12, 13], n=25, n_printed=dash)
    xml = write_diggs([hole])
    assert diggs_roundtrip_gate(xml, [hole]) == (True, [])


def test_a_dash_with_nothing_else_round_trips():
    hole = _hole(blows=[], n=None, n_printed="-")
    assert diggs_roundtrip_gate(write_diggs([hole]), [hole])[1] == []


def test_a_real_change_still_gives_exactly_one_diff():
    hole = _hole(blows=[11, 12, 13], n=25, n_printed="-")
    other = hole.model_copy(deep=True)
    other.spt[0].n = 26
    ok, diffs = diggs_roundtrip_gate(write_diggs([hole]), [other])
    assert not ok and len(diffs) == 1


def test_a_refusal_text_is_still_written_and_checked():
    hole = _hole(blows=[50], n=60, n_printed=">50", refusal=True)
    xml = write_diggs([hole])
    assert diggs_roundtrip_gate(xml, [hole])[0]
    other = hole.model_copy(deep=True)
    other.spt[0].n_printed = "R"
    ok, diffs = diggs_roundtrip_gate(xml, [other])
    assert not ok and len(diffs) == 1

"""RI-111 (Unicode dashes in a unit) and RI-112 (every N-value column in the
DIGGS read-back gate). Made-up values only."""

from __future__ import annotations

import pytest

from report_ingest.diggs_writer import diggs_roundtrip_gate, write_diggs
from report_ingest.model import (
    Investigation, Layer, Quantity, SPT, to_si,
)


class TestDashLookAlikes:
    """A unit spelt with a Unicode dash reads as the ASCII hyphen."""

    @pytest.mark.parametrize("dash", [
        "‐", "‑", "‒", "–", "−"])
    @pytest.mark.parametrize("ascii_unit", ["kN-m", "kN-m^2", "m-kN"])
    def test_each_dash_converts_like_the_ascii_hyphen(self, dash, ascii_unit):
        want = to_si(3.0, ascii_unit)
        assert want is not None
        assert to_si(3.0, ascii_unit.replace("-", dash)) == want

    def test_a_non_unit_stays_unconverted(self):
        assert to_si(3.0, "lay‐ers") is None


def _m(value: float) -> Quantity:
    return Quantity(value=value, unit="m")


def _hole(**spt) -> Investigation:
    base = dict(depth_top=_m(3.0), depth_bottom=_m(3.45))
    base.update(spt)
    return Investigation(
        investigation_id="B-1", depth_unit="m", total_depth=_m(12.0),
        layers=[Layer(top=_m(0.0), bottom=_m(12.0),
                      description="SILTY SAND", uscs="SM")],
        spt=[SPT(**base)])


def _changed(hole, **edit):
    other = hole.model_copy(deep=True)
    for key, value in edit.items():
        setattr(other.spt[0], key, value)
    return other


class TestEveryNValueColumn:
    """The gate compares each N-value column the writer writes, by name."""

    def test_the_unchanged_record_passes(self):
        hole = _hole(blows=[11, 12, 13], n=25, n_printed="N=25")
        assert diggs_roundtrip_gate(write_diggs([hole]), [hole]) == (True, [])

    def test_a_changed_number_column_is_named(self):
        hole = _hole(blows=[11, 12, 13], n=25, n_printed="N=25")
        ok, diffs = diggs_roundtrip_gate(write_diggs([hole]),
                                         [_changed(hole, n=26)])
        assert not ok
        assert len(diffs) == 1 and "'N-Value'" in diffs[0]

    def test_a_changed_text_column_is_named(self):
        hole = _hole(blows=[11, 12, 13], n=25, n_printed="N=25")
        ok, diffs = diggs_roundtrip_gate(
            write_diggs([hole]), [_changed(hole, n_printed="N=26")])
        assert not ok
        assert len(diffs) == 1 and "'N-Value (as printed)'" in diffs[0]

    def test_a_changed_blow_count_column_is_named(self):
        hole = _hole(blows=[11, 12, 13])
        ok, diffs = diggs_roundtrip_gate(
            write_diggs([hole]), [_changed(hole, blows=[11, 14, 13])])
        assert not ok
        assert len(diffs) == 1 and "Blow count 2" in diffs[0]

    def test_a_refusal_text_beside_a_number_is_compared(self):
        hole = _hole(blows=[50], n=60, n_printed=">50", refusal=True)
        xml = write_diggs([hole])
        assert diggs_roundtrip_gate(xml, [hole])[0]
        ok, diffs = diggs_roundtrip_gate(xml, [_changed(hole, n_printed="R")])
        assert not ok
        assert len(diffs) == 1 and "N printed as 'R'" in diffs[0]


class TestCalcReaderReadsADashInAUnit:
    @pytest.mark.parametrize("dash", ["‐", "‑", "‒",
                                      "–", "−"])
    def test_a_pair_line_with_a_unicode_dash_unit_matches(self, dash):
        from report_ingest.calc_reader import _PAIR_RE
        match = _PAIR_RE.match(f"M = 125.5 kN{dash}m")
        assert match is not None
        assert match.group("unit") == f"kN{dash}m"

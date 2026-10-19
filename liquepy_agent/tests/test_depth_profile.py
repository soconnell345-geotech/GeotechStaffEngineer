"""
The CPT depth profile a tool result carries (BACKLOG GM-201):
``liquepy_agent.cpt_liquefaction.depth_profile``. One row per reading for a
short CPT; a long one in depth bands that never mix readings above and
below FS_liq = 1.0 and keep each band's settlement.

Run with: pytest liquepy_agent/tests/test_depth_profile.py -v
"""

import numpy as np
import pytest

from liquepy_agent.cpt_liquefaction import (
    PROFILE_MAX_ROWS, depth_profile,
)
from liquepy_agent.liquepy_utils import has_liquepy


def _settlement(z, eps):
    """The readings' settlement: each reading over the depth halfway to
    its neighbours."""
    edges = np.r_[max(0.0, z[0] - 0.5 * (z[1] - z[0])),
                  0.5 * (z[:-1] + z[1:]),
                  z[-1] + 0.5 * (z[-1] - z[-2])]
    return float(np.sum(eps * np.diff(edges)))


def _rows_settlement(rows):
    return sum(r["volumetric_strain"] * (r["bottom_m"] - r["top_m"])
               for r in rows)


class TestOneRowPerReading:

    def test_each_reading_stands_for_halfway_to_its_neighbours(self):
        rows, note = depth_profile([1.0, 2.0, 3.0], [1.4, 0.8, 0.6],
                                   [0.004, 0.02, 0.03])
        assert note is None
        assert [(r["top_m"], r["bottom_m"]) for r in rows] == [
            (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]
        assert [r["FOS_liq"] for r in rows] == [1.4, 0.8, 0.6]
        assert [r["volumetric_strain"] for r in rows] == [0.004, 0.02, 0.03]

    def test_the_first_reading_does_not_reach_above_the_ground(self):
        rows, _ = depth_profile([0.2, 1.0], [2.0, 2.0], [0.0, 0.0])
        assert rows[0]["top_m"] == 0.0

    def test_fs_just_above_one_never_reads_as_one(self):
        rows, _ = depth_profile([1.0, 2.0], [1.0004, 0.9996], [0.0, 0.03])
        assert rows[0]["FOS_liq"] > 1.0
        assert rows[1]["FOS_liq"] <= 1.0

    def test_no_negative_strain(self):
        rows, _ = depth_profile([1.0, 2.0], [2.0, 2.0], [-1e-7, 0.0])
        assert rows[0]["volumetric_strain"] == 0.0
        assert str(rows[0]["volumetric_strain"]) == "0.0"

    @pytest.mark.parametrize("args, match", [
        (([1.0], [0.8], [0.02]), "at least two"),
        (([1.0, 2.0], [0.8], [0.02, 0.02]), "same length"),
    ])
    def test_bad_input_refused(self, args, match):
        with pytest.raises(ValueError, match=match):
            depth_profile(*args)


class TestBands:
    """A 20 m CPT read every 2 cm: 1000 readings."""

    z = np.arange(1, 1001) * 0.02

    def test_a_layered_profile_keeps_its_liquefied_depths(self):
        fs = np.where((self.z > 5.0) & (self.z < 9.0), 0.7, 1.6)
        eps = np.where(fs < 1.0, 0.025, 0.002)
        rows, note = depth_profile(self.z, fs, eps)
        assert len(rows) == PROFILE_MAX_ROWS
        # contiguous, from the first reading's top to the last's bottom
        assert all(a["bottom_m"] == b["top_m"]
                   for a, b in zip(rows, rows[1:]))
        assert rows[0]["top_m"] == 0.01 and rows[-1]["bottom_m"] == 20.01
        liq = [r for r in rows if r["FOS_liq"] <= 1.0]
        # the readings 5.02-8.98 m stand for 5.01-8.99 m
        assert (liq[0]["top_m"], liq[-1]["bottom_m"]) == (5.01, 8.99)
        assert all(r["volumetric_strain"] == 0.025 for r in liq)
        assert all(r["FOS_liq"] == 1.6 for r in rows if r not in liq)
        assert _rows_settlement(rows) == pytest.approx(
            _settlement(self.z, eps), rel=1e-3)
        assert "1000 readings summarised in 40 depth bands" in note
        assert "No band holds readings on both sides of 1.0" in note

    def test_the_lowest_reading_on_its_side_of_one(self):
        fs = np.where(self.z < 10.0, 1.2 + self.z / 100, 0.9 - self.z / 100)
        rows, _ = depth_profile(self.z, fs, np.zeros_like(self.z))
        above = [r for r in rows if r["bottom_m"] <= 10.0]
        # FS rises with depth above 10 m: each band's is its top reading's
        for r in above:
            first = self.z[self.z > r["top_m"]][0]
            assert r["FOS_liq"] == pytest.approx(1.2 + first / 100, abs=1e-3)
        below = [r for r in rows if r["top_m"] >= 10.0]
        # and falls below it: each band's is its bottom reading's
        for r in below:
            last = self.z[self.z < r["bottom_m"]][-1]
            assert r["FOS_liq"] == pytest.approx(0.9 - last / 100, abs=1e-3)

    def test_more_crossings_than_rows_are_counted_and_said(self):
        rng = np.random.default_rng(3)
        fs = 1.0 + rng.normal(0.0, 0.15, self.z.size)
        eps = np.where(fs <= 1.0, 0.03, 0.005)
        rows, note = depth_profile(self.z, fs, eps)
        assert len(rows) == PROFILE_MAX_ROWS
        assert "cross 1.0 more often than there are rows" in note
        assert "Every reading with FS_liq <= 1.0 is in a liquefied band" \
            in note
        # no reading at or below 1.0 is hidden in a band above it
        for z, f in zip(self.z, fs):
            if f <= 1.0:
                assert any(r["FOS_liq"] <= 1.0 for r in rows
                           if r["top_m"] <= z <= r["bottom_m"]), z
        assert any(r["FOS_liq"] > 1.0 for r in rows)
        assert _rows_settlement(rows) == pytest.approx(
            _settlement(self.z, eps), rel=1e-3)

    def test_a_thin_liquefied_lens_is_never_lost(self):
        # 1000 readings, sixty one-reading lenses at FS 0.9 in FS 1.5 soil:
        # more crossings than rows, and each lens 2 cm thick
        fs = np.full(self.z.size, 1.5)
        fs[np.arange(10, 1000, 16)[:60]] = 0.9
        rows, note = depth_profile(self.z, fs, np.where(fs < 1, 0.02, 0.0))
        assert len(rows) == PROFILE_MAX_ROWS
        for z in self.z[fs < 1.0]:
            assert any(r["FOS_liq"] <= 1.0 for r in rows
                       if r["top_m"] <= z <= r["bottom_m"]), z
        assert "take in" in note

    def test_a_short_lens_keeps_its_own_depths(self):
        # one 0.3 m lens in a long CPT: its own band, its readings' depths
        fs = np.where((self.z > 12.0) & (self.z < 12.3), 0.8, 1.6)
        rows, note = depth_profile(self.z, fs, np.where(fs < 1, 0.02, 0.0))
        liq = [r for r in rows if r["FOS_liq"] <= 1.0]
        assert [(r["top_m"], r["bottom_m"]) for r in liq] == [(12.01, 12.29)]
        assert "No band holds readings on both sides of 1.0" in note

    def test_the_row_limit_is_a_parameter(self):
        fs = np.where(self.z > 10.0, 0.7, 1.6)
        rows, _ = depth_profile(self.z, fs, np.zeros_like(self.z),
                                max_rows=10)
        assert len(rows) == 10


@pytest.mark.skipif(not has_liquepy(), reason="liquepy not installed")
def test_from_the_cpt_analysis():
    from liquepy_agent import analyze_cpt_liquefaction
    z = np.arange(1, 301) * 0.05
    q_c = np.where((z > 4.0) & (z < 8.0), 2500.0, 15000.0)
    r = analyze_cpt_liquefaction(z, q_c, 0.005 * q_c, gwl=1.0, pga=0.3)
    rows, note = depth_profile(r.depth, r.factor_of_safety,
                               r.volumetric_strain)
    assert len(rows) == PROFILE_MAX_ROWS and note
    assert _rows_settlement(rows) == pytest.approx(
        r.max_settlement_mm / 1000.0, rel=0.05)
    above = [x for x in rows if x["bottom_m"] <= 1.0]
    assert above and all(x["FOS_liq"] >= 2.0 for x in above)

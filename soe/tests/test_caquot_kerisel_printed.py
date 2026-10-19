"""Caquot-Kerisel Kp0 and R(delta/phi) pinned to the printed charts (GM-21).

Sources (each read off the printed page, 2026-10-19):

* CT  -- Caltrans Trenching and Shoring Manual (July 2025), Figure 4-20
  "Passive Earth Pressure Coefficient (Caquot and Kerisel, 1948)", p. 4-30:
  the Kp chart (curves for delta/phi = 1; beta/phi = 0 is the level backfill)
  and the printed table of the reduction factor R for -delta/phi = 0 to 0.7.
* NHI -- FHWA NHI-06-089 Soils and Foundations Vol. II (2006), Figure 10-10,
  p. 10-22: the same chart and an identical R table.
* DM  -- UFC 3-220-20 (DM 7.2, 2025), Figure 4-13, p. 193: log-spiral Kp after
  Kerisel and Absi (1990), drawn for delta/phi = 0.66 only.

Kp0 chart reads were taken at phi +/- 0.2 deg on a 300 dpi render and checked
against Kp_Rankine / R(0), which a smooth wall must reproduce.
"""

import pytest

from soe.earth_pressure import caquot_kerisel_Kp, rankine_Kp

# phi: (module Kp0, CT Fig 4-20 p. 4-30 chart read, beta/phi = 0, delta = phi)
KP0_PRINTED = {
    10.0: (1.65, 1.65),
    15.0: (2.20, 2.20),
    20.0: (3.00, 3.03),
    25.0: (4.30, 4.31),
    30.0: (6.30, 6.40),   # 6.30: the manual's Example 8-1 read (p. 8-12)
    35.0: (10.2, 10.17),
    40.0: (17.7, 17.77),
    45.0: (33.5, 33.6),
}

# R printed on CT Fig 4-20 (p. 4-30) = NHI Fig 10-10 (p. 10-22)
R_PRINTED = {
    20.0: {0.5: 0.862, 0.7: 0.939, 0.3: 0.787},
    25.0: {0.5: 0.808, 0.7: 0.912, 0.3: 0.711},
    30.0: {0.5: 0.746, 0.7: 0.878, 0.3: 0.627},
    35.0: {0.5: 0.674, 0.7: 0.836, 0.3: 0.536},
    40.0: {0.5: 0.592, 0.7: 0.783, 0.3: 0.439},
    45.0: {0.5: 0.500, 0.7: 0.718, 0.3: 0.339},
}
R0_PRINTED = {10.0: 0.864, 15.0: 0.775, 20.0: 0.678, 25.0: 0.574,
              30.0: 0.467, 35.0: 0.362, 40.0: 0.262, 45.0: 0.174}

# DM 7.2 Fig 4-13 (p. 193), beta = 0, delta/phi = 0.66 (Kerisel and Absi 1990)
DM_066 = {20.0: 2.75, 25.0: 3.70, 30.0: 5.30, 35.0: 8.0, 40.0: 12.0}


@pytest.mark.parametrize("phi", sorted(KP0_PRINTED))
def test_kp0_matches_printed_chart(phi):
    module, chart = KP0_PRINTED[phi]
    assert caquot_kerisel_Kp(phi) == pytest.approx(module, abs=1e-9)
    # reading accuracy on the log chart: 2 %
    assert module == pytest.approx(chart, rel=0.02)


@pytest.mark.parametrize("phi", sorted(R0_PRINTED))
def test_printed_R0_times_kp0_is_rankine(phi):
    """The printed delta/phi = 0 column times the chart Kp0 is the Rankine Kp
    (a smooth wall's log spiral is a plane): within 2 %."""
    assert R0_PRINTED[phi] * KP0_PRINTED[phi][0] == pytest.approx(
        rankine_Kp(phi), rel=0.02)
    assert caquot_kerisel_Kp(phi, delta_deg=0.0) == pytest.approx(
        rankine_Kp(phi), rel=1e-9)


@pytest.mark.parametrize("phi", sorted(R_PRINTED))
@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_R_matches_printed_table(phi, ratio):
    kp0 = caquot_kerisel_Kp(phi)
    R = caquot_kerisel_Kp(phi, delta_deg=ratio * phi) / kp0
    assert R == pytest.approx(R_PRINTED[phi][ratio], abs=1e-9)


def test_full_friction_is_R_one():
    for phi in (20.0, 30.0, 45.0):
        assert caquot_kerisel_Kp(phi, delta_deg=phi) == caquot_kerisel_Kp(phi)


def test_caltrans_example_8_1_unchanged():
    """CT Example 8-1 (p. 8-12): Kp = 0.746 x 6.3 = 4.7 at phi 30, delta/phi 0.5."""
    assert caquot_kerisel_Kp(30.0, delta_deg=15.0) == pytest.approx(4.70, abs=0.01)


@pytest.mark.parametrize("phi", sorted(DM_066))
def test_dm72_kerisel_absi_cross_read(phi):
    """DM 7.2 Fig 4-13 is Kerisel and Absi (1990), not the 1948 chart; they agree
    within 5 % up to phi 35 deg and the 1948 values run up to ~10 % higher at
    40 deg. Pinned as a cross-read, not as the module's source."""
    kp = caquot_kerisel_Kp(phi, delta_deg=0.66 * phi)
    tol = 0.05 if phi <= 35.0 else 0.11
    assert kp == pytest.approx(DM_066[phi], rel=tol)
    assert kp >= DM_066[phi] * 0.97

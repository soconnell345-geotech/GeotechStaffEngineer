"""The log-spiral passive Kp of each printed source, pinned to the printed
pages (GM-183).

* CT  -- Caltrans Trenching and Shoring Manual (2025): Fig. 4-20 (p. 4-30),
  the Caquot and Kerisel (1948) chart with its R table; Matrix 4-1 and the
  sloping-ground example on p. 4-29.
* DM  -- UFC 3-220-20 (DM 7.2, 2025): Fig. 4-13 (p. 193), Kerisel and Absi
  (1990) at delta/phi' = 0.66; Eq. 4-12 (p. 191).

Both figures are printed only as charts; the module's tables are pixel reads
of the rendered pages. The pins below are what the manuals print (the
example, Matrix 4-1) and reads of the charts at angles between the tabulated
nodes, which check the interpolation.
"""

import math

import pytest

from geotech_common import log_spiral as LS


def _rankine_Kp(phi):
    return math.tan(math.radians(45 + phi / 2)) ** 2


# --- Caquot-Kerisel 1948 (CT Fig. 4-20) ------------------------------------

def test_caltrans_sloping_ground_example_p_4_29():
    """CT p. 4-29: phi = 32, beta = 27 (beta/phi = 0.84), delta = 14 deg:
    'Kp ~= 19' read between the +0.8 and +1 curves at delta = phi; R =
    0.679 for delta/phi = 0.44; Kp' = 0.679 x 19 = 13."""
    kp0 = LS.caquot_kerisel_1948_Kp0(32.0, 27.0)
    assert kp0 == pytest.approx(19.0, abs=0.5)
    kp = LS.caquot_kerisel_1948_Kp(32.0, 14.0, beta_deg=27.0)
    assert round(kp) == 13
    assert kp / kp0 == pytest.approx(0.679, abs=0.003)


# Matrix 4-1 (CT p. 4-29): R for phi 30 / 32 / 35 at delta/phi 0.5, 0.44, 0.4
# (the phi 32 row is the manual's own interpolation of the printed table)
MATRIX_4_1 = {30.0: (0.746, 0.710, 0.686), 32.0: (0.717, 0.679, 0.653),
              35.0: (0.674, 0.631, 0.603)}


@pytest.mark.parametrize("phi", sorted(MATRIX_4_1))
def test_R_matches_caltrans_matrix_4_1(phi):
    for ratio, printed in zip((0.5, 0.44, 0.4), MATRIX_4_1[phi]):
        kp0 = LS.caquot_kerisel_1948_Kp(phi)
        R = LS.caquot_kerisel_1948_Kp(phi, ratio * phi) / kp0
        assert R == pytest.approx(printed, abs=0.0015), (phi, ratio)


# Reads of CT Fig. 4-20 at angles that are not nodes of the module's table:
# (beta/phi, phi): chart read
CT_BETWEEN = {(0.0, 32.0): 7.57, (1.0, 32.0): 21.06, (0.8, 32.0): 18.73}


@pytest.mark.parametrize("key", sorted(CT_BETWEEN))
def test_ct_between_nodes_follows_the_chart(key):
    ratio, phi = key
    kp = LS.caquot_kerisel_1948_Kp0(phi, ratio * phi)
    assert kp == pytest.approx(CT_BETWEEN[key], rel=0.01)


def test_ct_level_ground_between_nodes_moved_to_the_chart():
    """GM-183: log Kp is interpolated linearly in phi, as the chart is drawn
    on a log scale. Linear Kp between the 5 deg nodes read 7.86 at phi 32
    and 13.95 at 37.5; the chart reads 7.57 and 13.25."""
    assert LS.caquot_kerisel_1948_Kp0(32.0) == pytest.approx(7.57, rel=0.01)
    assert LS.caquot_kerisel_1948_Kp0(37.5) == pytest.approx(13.25, rel=0.01)


def test_ct_curves_are_ordered_by_slope_and_friction():
    rows = sorted(LS._CK_ROWS)
    for phi in (10.0, 20.0, 30.0, 40.0):
        values = [LS.caquot_kerisel_1948_Kp0(phi, r * phi) if r else
                  LS.caquot_kerisel_1948_Kp0(phi)
                  for r in rows if not (r > 0.6 and phi > 40)]
        assert values == sorted(values), phi
    for r in rows:
        nodes = sorted(LS._CK_ROWS[r])
        vals = [LS._CK_ROWS[r][p] for p in nodes]
        assert vals == sorted(vals), r


def test_ct_level_row_keeps_the_gm21_nodes_and_smooth_wall():
    gm21 = {10: 1.65, 15: 2.20, 20: 3.00, 25: 4.30, 30: 6.30, 35: 10.2,
            40: 17.7, 45: 33.5}
    for phi, kp in gm21.items():
        assert LS.caquot_kerisel_1948_Kp(phi) == pytest.approx(kp, abs=1e-9)
        assert LS.caquot_kerisel_1948_Kp(phi, 0.0) == pytest.approx(
            _rankine_Kp(phi), rel=1e-9)


def test_ct_sloping_ground_uses_the_printed_R0_column():
    phi, beta = 30.0, 12.0
    kp0 = LS.caquot_kerisel_1948_Kp0(phi, beta)
    assert LS.caquot_kerisel_1948_Kp(phi, 0.0, beta_deg=beta) \
        == pytest.approx(0.467 * kp0, rel=1e-9)


@pytest.mark.parametrize("phi, beta", [(8.0, 4.0), (30.0, 31.0),
                                       (30.0, -28.0), (42.0, 42.0)])
def test_ct_off_the_chart_is_refused(phi, beta):
    with pytest.raises(ValueError):
        LS.caquot_kerisel_1948_Kp0(phi, beta)


# --- UFC 3-220-20 Fig. 4-13 (Kerisel and Absi 1990, delta/phi' = 0.66) ------

# The GM-21 read of the same chart, beta = 0
DM_066 = {20.0: 2.75, 25.0: 3.70, 30.0: 5.30, 35.0: 8.0, 40.0: 12.0}
# Curves crossing printed grid lines, (beta/phi, phi): Kp
DM_GRID = {(0.8, 20.0): 4.0, (0.4, 20.0): 3.4, (0.2, 25.0): 4.3,
           (-0.4, 20.0): 2.1, (-0.2, 30.0): 4.2, (-0.4, 30.0): 3.2,
           (-0.6, 40.0): 3.4}


@pytest.mark.parametrize("phi", sorted(DM_066))
def test_fig_4_13_level_ground_matches_the_gm21_read(phi):
    assert LS.dm7_2_fig_4_13_Kp(phi) == pytest.approx(DM_066[phi], rel=0.01)


@pytest.mark.parametrize("key", sorted(DM_GRID))
def test_fig_4_13_sloping_ground_reads(key):
    ratio, phi = key
    assert LS.dm7_2_fig_4_13_Kp(phi, ratio * phi) == pytest.approx(
        DM_GRID[key], rel=0.02)


def test_fig_4_13_takes_its_own_wall_friction_only():
    assert LS.dm7_2_fig_4_13_Kp(30.0, 0.0, 19.8) == LS.dm7_2_fig_4_13_Kp(30.0)
    assert LS.dm7_2_fig_4_13_Kp(30.0, 0.0, 20.0) == LS.dm7_2_fig_4_13_Kp(30.0)
    with pytest.raises(ValueError, match="0.66 only"):
        LS.dm7_2_fig_4_13_Kp(30.0, 0.0, 30.0)
    with pytest.raises(ValueError, match="0.66 only"):
        LS.dm7_2_fig_4_13_Kp(30.0, 0.0, 0.0)


@pytest.mark.parametrize("phi, beta", [(35.0, 35.0), (45.0, 9.0),
                                       (30.0, -21.0), (30.0, 33.0)])
def test_fig_4_13_off_the_chart_is_refused(phi, beta):
    with pytest.raises(ValueError):
        LS.dm7_2_fig_4_13_Kp(phi, beta)


def test_fig_4_13_is_below_the_1948_chart():
    """UFC footnote 21: the 1990 values replace the delta/phi = 1 charts with
    reduction ratios; at delta/phi 0.66 on level ground the 1948 chart is the
    higher, by up to about 10 % at phi 40."""
    for phi in (25.0, 30.0, 35.0, 40.0):
        ck = LS.caquot_kerisel_1948_Kp(phi, 0.66 * phi)
        assert ck >= LS.dm7_2_fig_4_13_Kp(phi) * 0.99


# --- UFC 3-220-20 Eq. 4-12 ---------------------------------------------------

def test_eq_4_12_values():
    # ln Kp = ln[(1 + sin)/(1 - sin)] [1.443 (delta/phi) sin + 1]
    assert LS.dm7_2_eq_4_12_Kp(30.0, 0.0) == pytest.approx(3.0, rel=1e-9)
    assert LS.dm7_2_eq_4_12_Kp(30.0, 0.66 * 30.0) == pytest.approx(
        5.062, abs=0.001)
    assert LS.dm7_2_eq_4_12_Kp(40.0, 40.0) == pytest.approx(18.94, abs=0.01)
    for phi in (20.0, 30.0, 40.0):
        assert LS.dm7_2_eq_4_12_Kp(phi, 0.0) == pytest.approx(
            _rankine_Kp(phi), rel=1e-9)


@pytest.mark.parametrize("phi", sorted(DM_066))
def test_eq_4_12_approximates_fig_4_13(phi):
    """p. 191: 'the value of Kp can be approximated as' Eq. 4-12; at
    delta/phi' = 0.66 it runs 2 to 7 % below the Fig. 4-13 curve."""
    eq = LS.dm7_2_eq_4_12_Kp(phi, 0.66 * phi)
    chart = LS.dm7_2_fig_4_13_Kp(phi)
    assert 0.93 * chart <= eq <= chart


def test_eq_4_12_needs_delta_and_level_ground():
    with pytest.raises(ValueError, match="delta_deg"):
        LS.dm7_2_eq_4_12_Kp(30.0, None)
    with pytest.raises(ValueError, match="level ground"):
        LS.log_spiral_Kp(30.0, 20.0, 10.0, source="dm7_2_eq_4_12")


def test_dispatch_by_source():
    assert LS.log_spiral_Kp(30.0, 15.0) == LS.caquot_kerisel_1948_Kp(30.0, 15.0)
    assert LS.log_spiral_Kp(30.0, None, 6.0, source="dm7_2_fig_4_13") \
        == LS.dm7_2_fig_4_13_Kp(30.0, 6.0)
    with pytest.raises(ValueError, match="Unknown"):
        LS.log_spiral_Kp(30.0, source="dm7_2")

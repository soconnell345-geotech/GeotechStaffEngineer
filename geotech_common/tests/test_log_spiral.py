"""The log-spiral passive Kp of each printed source, pinned to the printed
pages (GM-183).

* CT  -- Caltrans Trenching and Shoring Manual (2025): Fig. 4-20 (p. 4-30),
  the Caquot and Kerisel (1948) chart with its R table; Matrix 4-1 and the
  sloping-ground example on p. 4-29.
* DM  -- UFC 3-220-20 (DM 7.2, 2025): Figs. 4-12 (p. 192) and 4-13 (p. 193),
  Kerisel and Absi (1990) at delta/phi' = 0.66; Eq. 4-12 (p. 191); the
  default route between them by the wall's geometry (GM-232).

The figures are printed only as charts; the module's tables are pixel reads
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
    # the default (GM-232): UFC 3-220-20 by geometry, Eq. 4-12 here
    assert LS.DEFAULT_SOURCE == "ufc_3_220_20"
    assert LS.log_spiral_Kp(30.0, 15.0) == LS.dm7_2_eq_4_12_Kp(30.0, 15.0)
    assert LS.log_spiral_Kp(30.0, 15.0, source="caquot_kerisel_1948")         == LS.caquot_kerisel_1948_Kp(30.0, 15.0)
    assert LS.log_spiral_Kp(30.0, None, source="dm7_2_fig_4_12",
                            theta_deg=10.0) == LS.dm7_2_fig_4_12_Kp(30.0, 10.0)
    assert LS.log_spiral_Kp(30.0, None, 6.0, source="dm7_2_fig_4_13") \
        == LS.dm7_2_fig_4_13_Kp(30.0, 6.0)
    with pytest.raises(ValueError, match="Unknown"):
        LS.log_spiral_Kp(30.0, source="dm7_2")


# --- UFC 3-220-20 Fig. 4-12 (p. 192) and the default route (GM-232) ---------

# Where each curve crosses a printed grid line, read on the page with only
# the two grid lines either side (theta, phi', Kp).
_FIG_4_12_CROSSINGS = [
    (-30, 15.0, 2.60), (-30, 20.0, 4.00), (-20, 21.31, 4.0), (-20, 25.0, 5.40),
    (-10, 15.0, 2.30), (-10, 25.0, 4.50), (0, 15.0, 2.05), (0, 25.0, 3.70),
    (10, 16.90, 2.0), (10, 40.0, 8.50), (20, 30.0, 3.10), (20, 37.73, 5.0),
    (30, 30.0, 2.30), (30, 35.30, 3.0), (40, 34.99, 2.0), (40, 30.0, 1.62),
]


@pytest.mark.parametrize("theta, phi, kp", _FIG_4_12_CROSSINGS)
def test_fig_4_12_reads_its_printed_grid(theta, phi, kp):
    assert LS.dm7_2_fig_4_12_Kp(phi, theta) == pytest.approx(kp, rel=0.006)


def test_fig_4_12_theta_0_is_fig_4_13_level_ground():
    """The two charts' vertical-face, level-ground curves, traced apart,
    agree within 1 %."""
    for phi in range(5, 41):
        assert LS.dm7_2_fig_4_12_Kp(phi, 0.0) == pytest.approx(
            LS.dm7_2_fig_4_13_Kp(phi, 0.0), rel=0.01)


def test_fig_4_12_falls_as_the_face_overhangs():
    for phi in (20.0, 30.0):
        kps = [LS.dm7_2_fig_4_12_Kp(phi, t) for t in range(-30, 41, 5)]
        assert all(b < a for a, b in zip(kps, kps[1:]))


@pytest.mark.parametrize("args, words", [
    ((30.0, 10.0, 15.0), "0.66 only"),
    ((30.0, -35.0), "-30"),
    ((30.0, 45.0), r"\+40"),
    ((35.0, -30.0), "off the chart"),          # Kp past 15
    ((10.0, 40.0), "off the chart"),           # the curve starts at phi' 15
    ((46.0, 0.0), "0 to 45"),
])
def test_fig_4_12_refuses_what_it_does_not_print(args, words):
    with pytest.raises(ValueError, match=words):
        LS.dm7_2_fig_4_12_Kp(*args)


@pytest.mark.parametrize("kwargs, route, kp, delta", [
    ({}, "dm7_2_eq_4_12", 5.062, 19.8),
    ({"delta_deg": 15.0}, "dm7_2_eq_4_12", 4.459, 15.0),
    ({"delta_deg": 30.0}, "dm7_2_eq_4_12", 6.633, 30.0),
    ({"theta_deg": 10.0}, "dm7_2_fig_4_12", 4.10, 19.8),
    ({"theta_deg": -20.0}, "dm7_2_fig_4_12", 8.41, 19.8),
    ({"beta_deg": -12.0}, "dm7_2_fig_4_13", 3.20, 19.8),
])
def test_the_default_routes_by_geometry(kwargs, route, kp, delta):
    r = LS.ufc_3_220_20_route(30.0, **kwargs)
    assert r["route"] == route
    assert r["Kp"] == pytest.approx(kp, rel=0.01)
    assert r["delta_deg"] == pytest.approx(delta)
    assert LS.log_spiral_Kp(30.0, **kwargs) == r["Kp"]


@pytest.mark.parametrize("kwargs, words", [
    # UFC prints no battered face with sloping ground
    ({"theta_deg": 10.0, "beta_deg": 5.0}, "prints no log-spiral Kp"),
    ({"theta_deg": 10.0, "beta_deg": 5.0}, r"Caquot and Kerisel \(1948\)"),
    # nor either chart at another wall friction
    ({"theta_deg": 10.0, "delta_deg": 15.0}, "0.66 only"),
    ({"beta_deg": 10.0, "delta_deg": 15.0}, "'caquot_kerisel_1948'"),
    # off Fig. 4-13: the 1948 chart named
    ({"beta_deg": 30.0, "phi_deg": 35.0}, "'caquot_kerisel_1948'"),
    ({"theta_deg": -30.0, "phi_deg": 35.0}, "off the chart"),
])
def test_the_default_says_where_ufc_prints_nothing(kwargs, words):
    kwargs = {"phi_deg": 30.0, **kwargs}
    with pytest.raises(ValueError, match=words):
        LS.ufc_3_220_20_route(**kwargs)


def test_battered_face_with_sloping_ground_names_where_to_read_kp():
    """GM-232: the refusal names the sources UFC p. 190 points to and which
    leaving-out overstates Kp."""
    with pytest.raises(ValueError) as e:
        LS.ufc_3_220_20_route(30.0, theta_deg=10.0, beta_deg=5.0)
    words = str(e.value)
    assert "Alqarawi et al. (2021)" in words
    assert "theta > 0" in words and "overstates Kp" in words


def test_above_the_charts_is_above_0_66():
    assert not LS.above_the_charts(30.0, 19.8)
    assert not LS.above_the_charts(30.0, 20.0)
    assert LS.above_the_charts(30.0, 21.0)
    assert not LS.above_the_charts(0.0, 0.0)


def test_wall_friction_note_says_what_each_source_takes():
    ufc = LS.wall_friction_note("ufc_3_220_20")
    ck = LS.wall_friction_note("caquot_kerisel_1948")
    for note in (ufc, ck):
        assert "engineer's call" in note and "Table 4-1 (p. 184" in note
        assert "(p. 260)" in note and "(p. 249)" in note
    assert "0.66, for which" in ufc and "delta = phi'" in ck
    assert "smooth" in LS.wall_friction_note("ufc_3_220_20",
                                             left_out="A smooth wall.")
    # "" where the caller has said what is taken
    said = LS.wall_friction_note("ufc_3_220_20", left_out="")
    assert "Left out" not in said and "  " not in said


def test_level_front_refuses_a_left_out_delta_where_none_is_printed():
    """The GM-242 review: in a wall's design the 1948 chart and Eq. 4-12
    by name need delta given (the Caltrans procedure begins 'Given delta',
    p. 4-29); before, the chart took delta = phi', its largest Kp. The UFC
    sources by geometry or chart take 0.66 phi'."""
    for source in ("caquot_kerisel_1948", "dm7_2_eq_4_12"):
        with pytest.raises(ValueError, match=r"needs the wall friction "
                                             r"given \(toe_delta\)"):
            LS.level_front_Kp(33.0, None, source, name="toe_delta")
        assert "refused" in LS.level_front_left_out(source)
        given = LS.level_front_Kp(33.0, 15.0, source)
        assert given["delta_deg"] == 15.0 and given["delta_given"]
    for source in ("ufc_3_220_20", "dm7_2_fig_4_12", "dm7_2_fig_4_13"):
        r = LS.level_front_Kp(33.0, None, source)
        assert r["delta_deg"] == pytest.approx(0.66 * 33.0)
        assert not r["delta_given"]
        assert "0.66" in LS.level_front_left_out(source)
    # phi' = 0 needs no wall friction with any source
    assert LS.level_front_Kp(0.0, None, "caquot_kerisel_1948")["Kp"] == 1.0


def test_vertical_wall_sources_refuse_a_battered_face():
    for source in ("caquot_kerisel_1948", "dm7_2_eq_4_12", "dm7_2_fig_4_13"):
        with pytest.raises(ValueError, match="vertical wall face"):
            LS.log_spiral_Kp(30.0, 19.8, source=source, theta_deg=10.0)
    with pytest.raises(ValueError, match="level ground"):
        LS.log_spiral_Kp(30.0, None, 5.0, source="dm7_2_fig_4_12")


def test_level_ground_eq_4_12_against_the_1948_chart():
    """The release note's comparison (GM-232): a vertical wall, level
    ground, Eq. 4-12 and the Caquot-Kerisel (1948) chart at phi' 30, 35,
    40 and delta/phi' 0.5, 0.66 and 1."""
    eq = {(p, r): round(LS.dm7_2_eq_4_12_Kp(p, r * p), 2)
          for p in (30, 35, 40) for r in (0.5, 0.66, 1.0)}
    assert eq == {(30, 0.5): 4.46, (30, 0.66): 5.06, (30, 1.0): 6.63,
                  (35, 0.5): 6.33, (35, 0.66): 7.53, (35, 1.0): 10.87,
                  (40, 0.5): 9.33, (40, 0.66): 11.7, (40, 1.0): 18.94}
    ck = {(p, r): round(LS.caquot_kerisel_1948_Kp(p, r * p), 2)
          for p in (30, 35, 40) for r in (0.5, 0.66, 1.0)}
    # Eq. 4-12 reads 5 to 11 % below the 1948 chart at delta/phi' 0.5 and
    # 0.66, 5 to 7 % above it at full wall friction
    assert ck == {(30, 0.5): 4.7, (30, 0.66): 5.36, (30, 1.0): 6.3,
                  (35, 0.5): 6.87, (35, 0.66): 8.18, (35, 1.0): 10.2,
                  (40, 0.5): 10.48, (40, 0.66): 13.14, (40, 1.0): 17.7}


def test_other_sources_give_each_value_or_why_not():
    alts = LS.other_sources(30.0, 19.8, used=("ufc_3_220_20",
                                              "dm7_2_eq_4_12"))
    assert alts == [
        "Caquot-Kerisel 1948, Caltrans T&S Fig. 4-20 (log_spiral_source "
        "'caquot_kerisel_1948'): Kp 5.36",
        "UFC 3-220-20 Fig. 4-12 (log_spiral_source 'dm7_2_fig_4_12'): "
        "Kp 5.3",
        "UFC 3-220-20 Fig. 4-13 (log_spiral_source 'dm7_2_fig_4_13'): "
        "Kp 5.3"]
    battered = LS.other_sources(30.0, 19.8, theta_deg=10.0,
                                used=("ufc_3_220_20", "dm7_2_fig_4_12"))
    assert all("none here" in a for a in battered)

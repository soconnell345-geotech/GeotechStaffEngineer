"""Log-spiral passive earth-pressure coefficient Kp, as the references print it.

The default, ``ufc_3_220_20`` (BACKLOG GM-232, owner OWN-66), takes the UFC
3-220-20 (DM 7.2, 2025) source the wall's geometry calls for, as its p. 191
describes them:

* a vertical face and level ground (theta = 0, beta = 0): Eq. 4-12, "for
  conditions with theta = 0 deg and beta = 0 deg", at any wall friction;
* a battered face with level ground (theta not 0): Fig. 4-12, "for walls with
  a sloping face and delta/phi' = 0.66";
* a vertical face with sloping ground (beta not 0): Fig. 4-13, "a similar
  chart for vertical-faced walls having a sloping backfill".

theta is the wall face's angle from vertical as UFC Figs. 4-7 and 4-12 draw
it: positive when the face slopes down into the soil it bears on, so the soil
overhangs it (Kp falls), negative the other way (Kp rises); theta = 90 deg
minus Coulomb's wall angle from horizontal. UFC prints nothing for a battered
face with sloping ground together, nor for a battered face or sloping ground
at a wall friction other than 0.66 phi': the route refuses those and says so,
as each chart here refuses a case off its page, and names the Caquot-Kerisel
(1948) chart (``caquot_kerisel_1948``), which takes a vertical wall with
sloping ground at any wall friction. Each printed source can also be named
directly:

``caquot_kerisel_1948``
    The Caquot and Kerisel (1948) chart as printed in the Caltrans Trenching
    and Shoring Manual (2025), Figure 4-20, p. 4-30 (the same chart: FHWA
    NHI-06-089 Figure 10-10, p. 10-22): Kp at delta = phi for a vertical
    wall, one curve per ground slope beta/phi from -0.9 to +1, times the
    reduction factor R(delta/phi) printed on the same figure. The manual's
    example on p. 4-29 reads it for sloping ground: phi = 32, beta = 27
    (beta/phi = 0.84), "Kp ~= 19" at delta = phi, R = 0.679 at delta/phi =
    0.44, Kp' = 0.679 x 19 = 13. The 1948 chart is for a vertical wall.
``dm7_2_eq_4_12``
    UFC 3-220-20 Eq. 4-12, p. 191, "for conditions with theta = 0 deg and
    beta = 0 deg": ln Kp = ln[(1 + sin phi')/(1 - sin phi')] x
    [1.443 (delta/phi') sin phi' + 1], for any wall friction ("while
    delta/phi' = 0.66 is a common assumption, this type of design may require
    Kp for a different value of wall friction").
``dm7_2_fig_4_12``
    UFC 3-220-20 Figure 4-12, p. 192, after Kerisel and Absi (1990): a
    battered face with level ground (beta = 0), one curve per theta from -30
    to +40 deg, drawn for delta/phi' = 0.66 only; the chart stops at Kp = 15,
    and its theta = +40 deg curve starts at Kp = 1 at phi' = 15 deg.
``dm7_2_fig_4_13``
    UFC 3-220-20 (DM 7.2, 2025) Figure 4-13, p. 193, after Kerisel and Absi
    (1990): a vertical wall with sloping ground, beta/phi' from -0.6 to +1,
    drawn for delta/phi' = 0.66 and theta = 0 only; the chart stops at
    Kp = 20. Footnote 21 (p. 191): the 1982 DM 7.2 charts "used delta/phi'
    = 1 along with reduction ratios", which "were averages that introduced
    substantial inaccuracy for some cases"; the 1948 chart takes that form.

At delta/phi' = 0.66 Eq. 4-12 reads 2 to 7 % below the theta = 0 curve of
Fig. 4-12 and the beta = 0 curve of Fig. 4-13 over phi' 10 to 40 deg (5.06
against 5.30 at phi' 30); the route follows p. 191 and takes Eq. 4-12 for theta = 0 and beta = 0.
UFC 3-220-20 p. 260 (tied-back walls) allows delta/phi' of 0.5 to 1.0 and
says "KP should be obtained from Figure 4-12", which is drawn for 0.66 only
(:data:`TIED_BACK_NOTE`).

The three charts are printed only as graphs. Their values here were read
off the rendered pages (Fig. 4-13 and the 1948 chart 2026-10-19, Fig. 4-12
2026-10-20): each curve traced pixel by pixel on the scanned figure,
calibrated on its printed grid lines, and the reads drawn back over the
figure to check they sit on the curves. Where a curve's label covers it, the
curve is fitted between the traced part and where it meets the frame of the
chart. Values between the tabulated nodes are interpolated with log Kp
linear in phi (and in beta/phi, or in theta), as the charts are drawn on a
log scale: linear interpolation in Kp between 5 degree nodes overstated the
chart by up to 6 % (level ground) and 17 % (beta = phi).

UFC 3-220-20 Table 4-2 (p. 194) is not used (GM-212, :data:`TABLE_4_2_NOTE`):
its row headed delta/phi' = 0.5 prints Fig. 4-13's values for delta/phi' =
0.66 (5.3, 8.0, 12, 20 at phi' = 30, 35, 40, 45), with Coulomb's Kp at
delta = 0.66 phi' beside them (6.1, 9.8, 18, 44). Its delta/phi' = 1 row
reads 1 to 5 % below Eq. 4-12 (6.5, 10.5, 18, 35 against 6.63, 10.87,
18.94, 35.21).

Above delta/phi' = 0.66 Eq. 4-12 is UFC's only log-spiral Kp, and UFC gives
it as an approximation with no range of delta/phi' stated; for phi' 30 to
45 deg it passes the 1948 chart between delta/phi' 0.87 and 0.95 and reads
up to 7 % above it at delta = phi' (:data:`HIGH_FRICTION_NOTE`). The wall
friction is the engineer's to set (:func:`wall_friction_note`); left out it
is 0.66 phi' with the UFC sources and phi with the 1948 chart, except in a
wall's design (:func:`level_front_Kp`), where the 1948 chart and Eq. 4-12
by name refuse it.
"""

from __future__ import annotations

import math

SOURCES = ("ufc_3_220_20", "caquot_kerisel_1948", "dm7_2_eq_4_12",
           "dm7_2_fig_4_12", "dm7_2_fig_4_13")
DEFAULT_SOURCE = "ufc_3_220_20"
#: The UFC 3-220-20 sources the default chooses between, by geometry.
UFC_ROUTES = ("dm7_2_eq_4_12", "dm7_2_fig_4_12", "dm7_2_fig_4_13")

#: UFC 3-220-20 p. 260 on tied-back walls, for a result's choices (GM-232).
TIED_BACK_QUOTE = (
    "UFC 3-220-20 p. 260 (tied-back walls): 'A delta/phi' ratio equal to "
    "0.5 to 1.0 can be used in the passive resistance calculation. Table "
    "4-1 should be consulted for the value of delta. KP should be obtained "
    "from Figure 4-12.'")
TIED_BACK_NOTE = (
    TIED_BACK_QUOTE + " Fig. 4-12 is drawn for delta/phi' = 0.66 only; for "
    "a vertical face with level ground its theta = 0 curve is the case Eq. "
    "4-12 (p. 191) gives at any wall friction, so the route takes Eq. 4-12 "
    "at the wall friction given (at 0.66 it reads 2 to 7 % below the "
    "chart; log_spiral_source 'dm7_2_fig_4_12' reads the chart)")
#: UFC 3-220-20 p. 238 on its anchored bulkhead design scenarios (Fig.
#: 4-37), for an anchored wall's choices (GM-232).
ANCHORED_BULKHEAD_QUOTE = (
    "UFC 3-220-20 p. 238, on its anchored bulkhead design scenarios (Fig. "
    "4-37): 'The effects of interface friction between the wall and soil "
    "are only considered for passive pressure in granular soils. Wall "
    "friction is ignored in all other cases'")

#: Where Eq. 4-12 is used above the charts' wall friction (GM-232), for a
#: result's choices: UFC bounds it nowhere, and it reads above the 1948
#: chart and UFC's own Table 4-2 at full wall friction.
HIGH_FRICTION_NOTE = (
    "Above delta/phi' = 0.66 UFC 3-220-20 draws no log-spiral chart, and it "
    "gives Eq. 4-12 as an approximation ('can be approximated as', p. 191) "
    "with no range of delta/phi' stated. Eq. 4-12 rises with wall friction "
    "faster than the Caquot-Kerisel (1948) chart: for phi' 30 to 45 deg it "
    "passes the chart between delta/phi' 0.87 and 0.95, and at delta = "
    "phi' it gives 6.63, 10.9 and 18.9 at phi' 30, 35 and 40 deg against "
    "6.30, 10.2 and 17.7 on the 1948 chart and 6.5, 10.5 and 18 in UFC "
    "Table 4-2 (p. 194), up to 7 % more passive resistance than the chart; "
    "below phi' 30 deg it stays under the chart. The alternatives give the "
    "1948 chart's Kp for this case")


def above_the_charts(phi_deg: float, delta_deg: float) -> bool:
    """Whether the wall friction is above the 0.66 phi' UFC's charts are
    drawn for, where Eq. 4-12 alone gives a UFC Kp
    (:data:`HIGH_FRICTION_NOTE`)."""
    return phi_deg > 0 and delta_deg is not None and (
        delta_deg / phi_deg > DM_DELTA_RATIO + _DM_DELTA_TOL)


def wall_friction_note(source: str, left_out: str = None) -> str:
    """The wall friction delta as a judgment call, for a result's choices
    (GM-232): where UFC 3-220-20 gives a value, and what is taken when it
    is left out with ``source`` (``left_out``, a sentence, where the caller
    takes something else; "" where the caller has said it). The same words
    whether delta was given or not."""
    if left_out is None and source == "caquot_kerisel_1948":
        left_out = (
            "Left out with the Caquot-Kerisel (1948) chart, the tool takes "
            "delta = phi', the chart's own Kp (reduction factor R = 1) and "
            "the most wall friction there can be.")
    elif left_out is None:
        left_out = ("Left out, the tool takes delta/phi' = 0.66, for which "
                    "UFC Figs. 4-12 and 4-13 are drawn.")
    return (
        "The wall friction angle delta is the engineer's call: set it from "
        "the wall-soil interface, UFC 3-220-20 Table 4-1 (p. 184, interface "
        "friction angles by wall material and soil); p. 191 calls "
        "delta/phi' = 0.66 'a common assumption'. "
        + (left_out + " " if left_out else "") + "UFC "
        "gives a range for some walls: delta/phi' of 0.5 to 1.0 on the "
        "passive side of a tied-back wall (p. 260), and for soldier piles in "
        "coarse-grained soil KP 'determined without wall friction' (p. 249)")

#: The level-ground comparison (GM-232), for a result's choices.
EQ_VS_CHARTS_NOTE = (
    "UFC 3-220-20 p. 191 gives Eq. 4-12 to approximate Kp 'for conditions "
    "with theta = 0 deg and beta = 0 deg', at any wall friction, and the "
    "route takes it there. At delta/phi' = 0.66 it reads 2 to 7 % below the "
    "theta = 0 curve of Fig. 4-12 and the beta = 0 curve of Fig. 4-13 over "
    "phi' 10 to 40 deg (5.06 against 5.30 at phi' 30); log_spiral_source "
    "'dm7_2_fig_4_12' or 'dm7_2_fig_4_13' reads the charts")

#: Why UFC 3-220-20 Table 4-2 is not one of the sources (GM-212), for a
#: result's choices and the tools' descriptions. Read on the rendered page
#: (p. 194) beside Fig. 4-13 (p. 193) and Eq. 4-12 (p. 191).
TABLE_4_2_NOTE = (
    "UFC 3-220-20 Table 4-2 (p. 194) is not used: the log-spiral Kp in its "
    "row headed delta/phi' = 0.5, 5.3, 8.0, 12 and 20 at phi' = 30, 35, 40 "
    "and 45, are Fig. 4-13's values, drawn for delta/phi' = 0.66 (5.30, "
    "8.00, 12.0, 20), and the Coulomb Kp printed beside them, 6.1, 9.8, 18 "
    "and 44, are Coulomb's at delta = 0.66 phi' (at 0.5 it gives 5.0, 7.4, "
    "11.8 and 21.4). Eq. 4-12 at delta/phi' = 0.5 gives 4.46, 6.33, 9.33 "
    "and 14.3. The tool reads Fig. 4-13 at the 0.66 it is drawn for and "
    "Eq. 4-12 at the wall friction given")

#: What each source is, in the references' words, for a result's choices.
SOURCE_BASIS = {
    "ufc_3_220_20": (
        "UFC 3-220-20 (DM 7.2), the source the wall's geometry calls for "
        "(pp. 191-193): Eq. 4-12 for a vertical face and level ground at any "
        "wall friction, Fig. 4-12 for a battered face with level ground and "
        "Fig. 4-13 for a vertical face with sloping ground, both drawn for "
        "delta/phi' = 0.66"),
    "caquot_kerisel_1948": (
        "Caquot and Kerisel (1948) log-spiral chart, Kp at delta = phi times "
        "the reduction factor R(delta/phi), as printed in the Caltrans "
        "Trenching and Shoring Manual Fig. 4-20 (p. 4-30) and FHWA "
        "NHI-06-089 Fig. 10-10, for a vertical wall"),
    "dm7_2_fig_4_12": (
        "UFC 3-220-20 (DM 7.2) Fig. 4-12 (p. 192), log spiral after Kerisel "
        "and Absi (1990), a wall face at theta -30 to +40 deg from vertical "
        "with level ground, drawn for delta/phi' = 0.66"),
    "dm7_2_fig_4_13": (
        "UFC 3-220-20 (DM 7.2) Fig. 4-13 (p. 193), log spiral after Kerisel "
        "and Absi (1990), vertical wall with sloping ground, drawn for "
        "delta/phi' = 0.66"),
    "dm7_2_eq_4_12": (
        "UFC 3-220-20 (DM 7.2) Eq. 4-12 (p. 191), which UFC gives to "
        "approximate the log-spiral Kp for theta = 0 and beta = 0 at any "
        "wall friction"),
}

# ---------------------------------------------------------------------------
# Caquot and Kerisel (1948): Caltrans T&S Fig. 4-20, p. 4-30
# ---------------------------------------------------------------------------

# Kp at delta = phi, by beta/phi, at phi nodes (deg). The level row (beta =
# 0) keeps the GM-21 reads at 0 and every 5 deg (phi = 30 keeps 6.30, the
# value the manual's Example 8-1 reads, p. 8-12) and adds the 2.5 deg reads;
# above 45 deg it is clamped at 33.5, as before. The sloping rows are read
# from 10 deg (below it the curves crowd into their labels) to the top of the
# chart (Kp = 100); a row stops where its curve leaves the chart.
_CK_ROWS = {
    1.0: {10: 1.94, 12.5: 2.36, 15: 2.90, 17.5: 3.66, 20: 4.68, 22.5: 6.14,
          25: 8.20, 27.5: 11.2, 30: 15.7, 32.5: 22.8, 35: 34.6, 37.5: 54.9,
          40: 93.5},
    0.8: {10: 1.89, 12.5: 2.29, 15: 2.82, 17.5: 3.55, 20: 4.53, 22.5: 5.87,
          25: 7.77, 27.5: 10.4, 30: 14.3, 32.5: 20.1, 35: 29.4, 37.5: 44.2,
          40: 70.1},
    0.6: {10: 1.85, 12.5: 2.21, 15: 2.67, 17.5: 3.29, 20: 4.13, 22.5: 5.23,
          25: 6.77, 27.5: 8.89, 30: 12.0, 32.5: 16.5, 35: 23.5, 37.5: 34.3,
          40: 52.4, 42.5: 81.1},
    0.4: {10: 1.79, 12.5: 2.11, 15: 2.51, 17.5: 3.03, 20: 3.72, 22.5: 4.61,
          25: 5.78, 27.5: 7.41, 30: 9.63, 32.5: 12.8, 35: 17.7, 37.5: 25.0,
          40: 36.7, 42.5: 55.3, 45: 86.0},
    0.2: {10: 1.74, 12.5: 2.03, 15: 2.37, 17.5: 2.80, 20: 3.35, 22.5: 4.07,
          25: 4.94, 27.5: 6.12, 30: 7.75, 32.5: 9.99, 35: 13.4, 37.5: 18.4,
          40: 26.2, 42.5: 37.6, 45: 54.2},
    0.0: {0: 1.00, 10: 1.65, 12.5: 1.92, 15: 2.20, 17.5: 2.57, 20: 3.00,
          22.5: 3.59, 25: 4.30, 27.5: 5.17, 30: 6.30, 32.5: 7.93, 35: 10.2,
          37.5: 13.3, 40: 17.7, 42.5: 24.3, 45: 33.5},
    -0.2: {10: 1.59, 12.5: 1.79, 15: 2.03, 17.5: 2.31, 20: 2.66, 22.5: 3.08,
           25: 3.60, 27.5: 4.24, 30: 5.01, 32.5: 6.05, 35: 7.48, 37.5: 9.33,
           40: 12.0, 42.5: 15.5, 45: 20.2},
    -0.4: {10: 1.49, 12.5: 1.66, 15: 1.85, 17.5: 2.06, 20: 2.31, 22.5: 2.60,
           25: 2.94, 27.5: 3.36, 30: 3.86, 32.5: 4.46, 35: 5.24, 37.5: 6.26,
           40: 7.66, 42.5: 9.37, 45: 11.6},
    -0.6: {10: 1.37, 12.5: 1.49, 15: 1.62, 17.5: 1.77, 20: 1.93, 22.5: 2.12,
           25: 2.32, 27.5: 2.55, 30: 2.81, 32.5: 3.13, 35: 3.52, 37.5: 4.00,
           40: 4.57, 42.5: 5.24, 45: 6.05},
    -0.8: {10: 1.23, 12.5: 1.30, 15: 1.36, 17.5: 1.44, 20: 1.52, 22.5: 1.60,
           25: 1.69, 27.5: 1.78, 30: 1.88, 32.5: 1.99, 35: 2.11, 37.5: 2.24,
           40: 2.39, 42.5: 2.55, 45: 2.75},
    -0.9: {10: 1.16, 12.5: 1.20, 15: 1.24, 17.5: 1.28, 20: 1.32, 22.5: 1.36,
           25: 1.40, 27.5: 1.44, 30: 1.47, 32.5: 1.51, 35: 1.55, 37.5: 1.59,
           40: 1.63, 42.5: 1.67, 45: 1.71},
}

# Wall-friction reduction factor R = Kp(delta)/Kp(delta = phi), the table
# printed on Caltrans T&S Fig. 4-20 (p. 4-30) and, identically, FHWA
# NHI-06-089 Fig. 10-10 (p. 10-22): phi 10-45 deg, delta/phi 0 to 0.7
# (Matrix 4-1, p. 4-29, quotes its phi 30/32/35 rows). One table serves every
# ground slope (the manual's p. 4-29 example applies it at beta/phi = 0.84).
# R = 1.0 at delta = phi by definition; between 0.7 and 1.0 it is
# interpolated linearly.
_CK_R_PHI = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
_CK_R_RATIO = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
_CK_R = {
    10.0: {0.0: 0.864, 0.1: 0.881, 0.2: 0.898, 0.3: 0.912, 0.4: 0.929, 0.5: 0.946, 0.6: 0.962, 0.7: 0.978, 1.0: 1.000},
    15.0: {0.0: 0.775, 0.1: 0.803, 0.2: 0.830, 0.3: 0.854, 0.4: 0.881, 0.5: 0.907, 0.6: 0.934, 0.7: 0.961, 1.0: 1.000},
    20.0: {0.0: 0.678, 0.1: 0.716, 0.2: 0.752, 0.3: 0.787, 0.4: 0.824, 0.5: 0.862, 0.6: 0.901, 0.7: 0.939, 1.0: 1.000},
    25.0: {0.0: 0.574, 0.1: 0.620, 0.2: 0.666, 0.3: 0.711, 0.4: 0.759, 0.5: 0.808, 0.6: 0.860, 0.7: 0.912, 1.0: 1.000},
    30.0: {0.0: 0.467, 0.1: 0.520, 0.2: 0.574, 0.3: 0.627, 0.4: 0.686, 0.5: 0.746, 0.6: 0.811, 0.7: 0.878, 1.0: 1.000},
    35.0: {0.0: 0.362, 0.1: 0.417, 0.2: 0.475, 0.3: 0.536, 0.4: 0.603, 0.5: 0.674, 0.6: 0.752, 0.7: 0.836, 1.0: 1.000},
    40.0: {0.0: 0.262, 0.1: 0.316, 0.2: 0.375, 0.3: 0.439, 0.4: 0.512, 0.5: 0.592, 0.6: 0.682, 0.7: 0.783, 1.0: 1.000},
    45.0: {0.0: 0.174, 0.1: 0.221, 0.2: 0.276, 0.3: 0.339, 0.4: 0.414, 0.5: 0.500, 0.6: 0.600, 0.7: 0.718, 1.0: 1.000},
}

# ---------------------------------------------------------------------------
# Kerisel and Absi (1990): UFC 3-220-20 Fig. 4-13, p. 193
# ---------------------------------------------------------------------------

# Kp at delta/phi' = 0.66, theta = 0, by beta/phi', at phi' nodes (deg). Every
# curve starts at Kp = 1 at phi' = 0; at phi' = 5 the beta = phi' and 0.8 phi'
# curves run together and are left out. A row whose curve leaves the top of
# the chart (Kp = 20) ends there, at the phi' where it does. Cross-read: the
# beta = 0 row agrees within 0.3 % with the GM-21 read of the same chart
# (2.75, 3.70, 5.30, 8.0, 12.0 at 20 to 40 deg).
_DM_TOP = 20.0
_DM_ROWS = {
    1.0: {0: 1.0, 7.5: 1.62, 10: 1.91, 12.5: 2.29, 15: 2.80, 17.5: 3.46,
          20: 4.30, 22.5: 5.45, 25: 7.01, 27.5: 9.27, 30: 12.5, 32.5: 17.4,
          33.46: _DM_TOP},
    0.8: {0: 1.0, 7.5: 1.59, 10: 1.86, 12.5: 2.20, 15: 2.65, 17.5: 3.25,
          20: 4.00, 22.5: 4.99, 25: 6.31, 27.5: 8.23, 30: 10.9, 32.5: 14.8,
          34.81: _DM_TOP},
    0.6: {0: 1.0, 5: 1.34, 7.5: 1.54, 10: 1.79, 12.5: 2.10, 15: 2.50,
          17.5: 3.03, 20: 3.70, 22.5: 4.58, 25: 5.71, 27.5: 7.24, 30: 9.31,
          32.5: 12.3, 35: 16.7, 36.39: _DM_TOP},
    0.4: {0: 1.0, 5: 1.31, 7.5: 1.51, 10: 1.73, 12.5: 2.00, 15: 2.35,
          17.5: 2.82, 20: 3.40, 22.5: 4.12, 25: 5.00, 27.5: 6.21, 30: 7.81,
          32.5: 9.98, 35: 13.0, 37.5: 17.5, 38.51: _DM_TOP},
    0.2: {0: 1.0, 5: 1.29, 7.5: 1.46, 10: 1.66, 12.5: 1.90, 15: 2.20,
          17.5: 2.62, 20: 3.10, 22.5: 3.64, 25: 4.30, 27.5: 5.26, 30: 6.50,
          32.5: 8.01, 35: 10.0, 37.5: 13.0, 40: 17.5, 41.05: _DM_TOP},
    0.0: {0: 1.0, 5: 1.26, 7.5: 1.41, 10: 1.59, 12.5: 1.80, 15: 2.05,
          17.5: 2.37, 20: 2.75, 22.5: 3.18, 25: 3.70, 27.5: 4.39, 30: 5.30,
          32.5: 6.52, 35: 8.00, 37.5: 9.74, 40: 12.0, 42.5: 15.2,
          45: _DM_TOP},
    -0.2: {0: 1.0, 5: 1.23, 7.5: 1.37, 10: 1.52, 12.5: 1.70, 15: 1.90,
           17.5: 2.14, 20: 2.40, 22.5: 2.72, 25: 3.10, 27.5: 3.60, 30: 4.20,
           32.5: 4.88, 35: 5.70, 37.5: 6.78, 40: 8.20, 42.5: 10.1, 45: 12.5},
    -0.4: {0: 1.0, 5: 1.19, 7.5: 1.30, 10: 1.42, 12.5: 1.56, 15: 1.72,
           17.5: 1.90, 20: 2.10, 22.5: 2.31, 25: 2.55, 27.5: 2.85, 30: 3.20,
           32.5: 3.61, 35: 4.10, 37.5: 4.70, 40: 5.40, 42.5: 6.21, 45: 7.10},
    -0.6: {0: 1.0, 5: 1.15, 7.5: 1.24, 10: 1.33, 12.5: 1.42, 15: 1.52,
           17.5: 1.63, 20: 1.76, 22.5: 1.90, 25: 2.05, 27.5: 2.22, 30: 2.40,
           32.5: 2.58, 35: 2.80, 37.5: 3.08, 40: 3.40, 42.5: 3.74, 45: 4.06},
}
DM_DELTA_RATIO = 0.66      # the one wall friction Figs. 4-12, 4-13 are drawn for
_DM_DELTA_TOL = 0.01

# ---------------------------------------------------------------------------
# Kerisel and Absi (1990): UFC 3-220-20 Fig. 4-12, p. 192
# ---------------------------------------------------------------------------

# Kp at delta/phi' = 0.66, beta = 0, by theta (deg), at phi' nodes (deg).
# Traced 2026-10-20 off the figure's raster (1430 x 1869 px, about 222 dpi)
# on its own frame and grid lines (residuals under 0.4 px). Each curve's
# label is set in the curve, so the curve was traced on both sides of it and
# fitted across; the theta = -30 to 0 curves end where they meet the top of
# the chart (Kp = 15), the others at phi' = 45. Checked against the printed
# grid lines where 45 crossings were read with only the two grid lines
# either side of each: within 0.3 %, and 0.6 % on the theta = -20 curve
# next to its label. Inside a label no curve is printed: a separate fit
# across the theta = -30 label (phi' 30) and the -20 label (phi' 32.5)
# reads 1.3 and 1.6 % above the nodes there (review, 2026-10-20), which is
# how well the chart can be read in those spans. The theta = 0 row agrees
# with the beta = 0 row of Fig. 4-13 (traced separately, above) within 1 %
# everywhere.
_FIG_4_12_TOP = 15.0
_FIG_4_12_ROWS = {
    -30: {0: 1.0, 2.5: 1.13, 5: 1.30, 7.5: 1.50, 10: 1.76, 12.5: 2.13,
          15: 2.60, 17.5: 3.22, 20: 4.00, 22.5: 5.00, 25: 6.31, 27.5: 8.13,
          30: 10.5, 32.5: 14.0, 33.09: _FIG_4_12_TOP},
    -20: {0: 1.0, 2.5: 1.13, 5: 1.30, 7.5: 1.50, 10: 1.76, 12.5: 2.06,
          15: 2.45, 17.5: 2.96, 20: 3.60, 22.5: 4.41, 25: 5.40, 27.5: 6.70,
          30: 8.41, 32.5: 10.6, 35: 14.0, 35.62: _FIG_4_12_TOP},
    -10: {0: 1.0, 2.5: 1.13, 5: 1.28, 7.5: 1.47, 10: 1.70, 12.5: 1.98,
          15: 2.30, 17.5: 2.66, 20: 3.10, 22.5: 3.72, 25: 4.50, 27.5: 5.43,
          30: 6.60, 32.5: 8.27, 35: 10.5, 37.5: 13.5, 38.54: _FIG_4_12_TOP},
    0: {0: 1.0, 2.5: 1.13, 5: 1.25, 7.5: 1.41, 10: 1.59, 12.5: 1.80,
        15: 2.05, 17.5: 2.37, 20: 2.75, 22.5: 3.18, 25: 3.70, 27.5: 4.39,
        30: 5.30, 32.5: 6.52, 35: 8.00, 37.5: 9.74, 40: 12.0,
        42.26: _FIG_4_12_TOP},
    10: {0: 1.0, 2.5: 1.09, 5: 1.20, 7.5: 1.32, 10: 1.47, 12.5: 1.65,
         15: 1.84, 17.5: 2.06, 20: 2.30, 22.5: 2.61, 25: 3.00, 27.5: 3.50,
         30: 4.10, 32.5: 4.81, 35: 5.70, 37.5: 6.91, 40: 8.50, 42.5: 10.5,
         45: 13.1},
    20: {0: 1.0, 2.5: 1.05, 5: 1.12, 7.5: 1.21, 10: 1.31, 12.5: 1.44,
         15: 1.58, 17.5: 1.74, 20: 1.93, 22.5: 2.17, 25: 2.45, 27.5: 2.75,
         30: 3.10, 32.5: 3.59, 35: 4.20, 37.5: 4.92, 40: 5.80, 42.5: 6.95,
         45: 8.42},
    30: {0: 1.0, 2.5: 1.02, 5: 1.04, 7.5: 1.08, 10: 1.14, 12.5: 1.22,
         15: 1.32, 17.5: 1.44, 20: 1.57, 22.5: 1.73, 25: 1.90, 27.5: 2.08,
         30: 2.30, 32.5: 2.59, 35: 2.95, 37.5: 3.38, 40: 3.90, 42.5: 4.52,
         45: 5.33},
    40: {15: 1.0, 17.5: 1.08, 20: 1.16, 22.5: 1.25, 25: 1.36, 27.5: 1.48,
         30: 1.62, 32.5: 1.79, 35: 2.00, 37.5: 2.25, 40: 2.55, 42.5: 2.88,
         45: 3.31},
}
FIG_4_12_THETA = (-30.0, 40.0)     # the curves drawn, theta from vertical


def _rankine_Kp(phi_deg: float) -> float:
    return math.tan(math.radians(45.0 + phi_deg / 2.0)) ** 2


def _lininterp(x, xs, ys):
    """Clamped piecewise-linear interpolation."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def _row_at(row: dict, phi: float, clamp: bool = False):
    """log Kp linear in phi between a row's nodes; None outside them (off
    the chart), unless ``clamp``."""
    xs = sorted(row)
    if phi < xs[0] - 1e-9 or phi > xs[-1] + 1e-9:
        if not clamp:
            return None
        phi = min(max(phi, xs[0]), xs[-1])
    logs = [math.log(row[x]) for x in xs]
    return math.exp(_lininterp(phi, xs, logs))


def _across_rows(rows: dict, phi: float, ratio: float, what: str,
                 label: str = "beta/phi"):
    """Kp between the two rows that bracket ``ratio`` (beta/phi, or theta
    for Fig. 4-12, named by ``label``), log Kp linear in it."""
    keys = sorted(rows)
    off = (f"{what}: phi {phi:g} deg with {label} {ratio:.3g} is off the "
           "chart")
    for k in keys:
        if abs(ratio - k) < 1e-9:
            value = _row_at(rows[k], phi)
            if value is None:
                raise ValueError(off)
            return value
    lo = max(k for k in keys if k < ratio)
    hi = min(k for k in keys if k > ratio)
    a, b = _row_at(rows[lo], phi), _row_at(rows[hi], phi)
    if a is None or b is None:
        raise ValueError(off)
    t = (ratio - lo) / (hi - lo)
    return math.exp(math.log(a) + t * (math.log(b) - math.log(a)))


def caquot_kerisel_1948_Kp0(phi_deg: float, beta_deg: float = 0.0) -> float:
    """Kp at delta = phi off the Caquot-Kerisel chart (Caltrans T&S Fig.
    4-20): the level curve for beta = 0 (clamped at 45 deg), else between
    the beta/phi curves, -0.9 to +1, phi 10 to 45 deg."""
    if not beta_deg:
        return _row_at(_CK_ROWS[0.0], phi_deg, clamp=True)
    what = "Caquot-Kerisel (Caltrans T&S Fig. 4-20)"
    if phi_deg < 10.0 or phi_deg > 45.0:
        raise ValueError(f"{what}: the sloping-ground curves are read for "
                         f"phi 10 to 45 deg, got {phi_deg:g}")
    ratio = beta_deg / phi_deg
    if ratio < -0.9 - 1e-9 or ratio > 1.0 + 1e-9:
        raise ValueError(f"{what}: the chart's curves run from beta/phi = "
                         f"-0.9 to +1, got beta {beta_deg:g} deg on phi "
                         f"{phi_deg:g} deg ({ratio:.3g})")
    return _across_rows(_CK_ROWS, phi_deg, min(max(ratio, -0.9), 1.0), what)


def _ck_R(phi_deg: float, ratio: float) -> float:
    """R off the printed table, bilinear in (phi, delta/phi), clamped in
    phi; ``ratio`` from 0 to 1."""
    phi_c = min(max(phi_deg, _CK_R_PHI[0]), _CK_R_PHI[-1])
    row = [_lininterp(ratio, _CK_R_RATIO, [_CK_R[p][k] for k in _CK_R_RATIO])
           for p in _CK_R_PHI]
    return _lininterp(phi_c, _CK_R_PHI, row)


def caquot_kerisel_1948_Kp(phi_deg: float, delta_deg: float = None,
                           beta_deg: float = 0.0,
                           Kp_initial: float = None) -> float:
    """Caquot-Kerisel (1948) log-spiral Kp' = R x Kp0 (Caltrans T&S Sec. 4-6,
    Fig. 4-20). Kp0 is the chart's Kp at delta = phi for the ground slope
    beta (or ``Kp_initial``); R the printed reduction factor for delta/phi.
    ``delta_deg`` None means delta = phi (R = 1). On level ground, between
    delta = 0 and the 0.1 column R runs to the Rankine anchor R(0) =
    Kp_rankine/Kp0, so a smooth wall returns the Rankine Kp (the printed R(0)
    times the chart Kp0 gives it within 2 %); on sloping ground the printed
    R(0) column is used."""
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    Kp0 = (Kp_initial if Kp_initial is not None
           else caquot_kerisel_1948_Kp0(phi_deg, beta_deg))
    if delta_deg is None or phi_deg <= 0:
        return Kp0
    ratio = min(max(delta_deg / phi_deg, 0.0), 1.0)
    if ratio >= 1.0:
        return Kp0
    lowest = _CK_R_RATIO[1]  # 0.1
    if ratio >= lowest or beta_deg:
        return Kp0 * _ck_R(phi_deg, ratio)
    R0 = _rankine_Kp(phi_deg) / Kp0
    return Kp0 * (R0 + (ratio / lowest) * (_ck_R(phi_deg, lowest) - R0))


def dm7_2_eq_4_12_Kp(phi_deg: float, delta_deg: float) -> float:
    """UFC 3-220-20 Eq. 4-12 (p. 191), theta = 0 and beta = 0:
    ln Kp = ln[(1 + sin phi')/(1 - sin phi')] [1.443 (delta/phi') sin phi'
    + 1]."""
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    if delta_deg is None:
        raise ValueError("UFC 3-220-20 Eq. 4-12 needs the wall friction "
                         "angle delta given (delta_deg); p. 191: 'while "
                         "delta/phi' = 0.66 is a common assumption, this "
                         "type of design may require Kp for a different "
                         "value of wall friction'")
    if phi_deg == 0:
        return 1.0
    if delta_deg < 0 or delta_deg > phi_deg + 1e-9:
        raise ValueError(f"UFC 3-220-20 Eq. 4-12: delta must be 0 to phi, "
                         f"got {delta_deg:g} on phi {phi_deg:g}")
    s = math.sin(math.radians(phi_deg))
    return math.exp(math.log((1 + s) / (1 - s))
                    * (1.443 * (delta_deg / phi_deg) * s + 1.0))


def dm7_2_fig_4_13_Kp(phi_deg: float, beta_deg: float = 0.0,
                      delta_deg: float = None) -> float:
    """Kp off UFC 3-220-20 Fig. 4-13 (p. 193; Kerisel and Absi 1990): theta
    = 0, delta/phi' = 0.66 (``delta_deg`` None means 0.66 phi'; another
    value is refused), beta/phi' from -0.6 to +1, Kp up to 20."""
    what = "UFC 3-220-20 Fig. 4-13"
    if phi_deg < 0 or phi_deg > 45:
        raise ValueError(f"{what}: phi' runs 0 to 45 deg, got {phi_deg:g}")
    if phi_deg == 0:
        if beta_deg:
            raise ValueError(f"{what}: a slope on phi' = 0 is off the chart")
        return 1.0
    if delta_deg is not None and abs(delta_deg / phi_deg
                                     - DM_DELTA_RATIO) > _DM_DELTA_TOL:
        raise ValueError(
            f"{what} is drawn for delta/phi' = 0.66 only, got delta "
            f"{delta_deg:g} on phi' {phi_deg:g} ({delta_deg / phi_deg:.3g}); "
            "UFC 3-220-20 Eq. 4-12 takes another wall friction on level "
            "ground, and the Caquot-Kerisel chart any wall friction")
    ratio = (beta_deg or 0.0) / phi_deg
    if ratio < -0.6 - 1e-9 or ratio > 1.0 + 1e-9:
        raise ValueError(f"{what}: the chart's curves run from beta/phi' = "
                         f"-0.6 to +1, got beta {beta_deg:g} deg on phi' "
                         f"{phi_deg:g} deg ({ratio:.3g})")
    value = _across_rows(_DM_ROWS, phi_deg, min(max(ratio, -0.6), 1.0), what)
    if value > _DM_TOP * (1 + 1e-9):
        raise ValueError(f"{what}: Kp above 20 is off the chart")
    return value


def dm7_2_fig_4_12_Kp(phi_deg: float, theta_deg: float = 0.0,
                      delta_deg: float = None) -> float:
    """Kp off UFC 3-220-20 Fig. 4-12 (p. 192; Kerisel and Absi 1990): level
    ground (beta = 0), delta/phi' = 0.66 (``delta_deg`` None means 0.66
    phi'; another value is refused), the wall face at theta from -30 to +40
    deg from vertical (see the module docstring for its sign), Kp up to
    15."""
    what = "UFC 3-220-20 Fig. 4-12"
    if phi_deg < 0 or phi_deg > 45:
        raise ValueError(f"{what}: phi' runs 0 to 45 deg, got {phi_deg:g}")
    theta = float(theta_deg or 0.0)
    lo, hi = FIG_4_12_THETA
    if theta < lo - 1e-9 or theta > hi + 1e-9:
        raise ValueError(f"{what}: its curves run from theta = {lo:g} to "
                         f"{hi:+g} deg, got theta {theta:g} deg")
    if phi_deg > 0 and delta_deg is not None and abs(
            delta_deg / phi_deg - DM_DELTA_RATIO) > _DM_DELTA_TOL:
        raise ValueError(
            f"{what} is drawn for delta/phi' = 0.66 only, got delta "
            f"{delta_deg:g} on phi' {phi_deg:g} ({delta_deg / phi_deg:.3g})")
    if phi_deg == 0 and theta <= 30.0:
        return 1.0
    value = _across_rows(_FIG_4_12_ROWS, phi_deg, theta, what,
                         label="theta")
    if value > _FIG_4_12_TOP * (1 + 1e-9):
        raise ValueError(f"{what}: Kp above 15 is off the chart")
    return value


def _vertical_wall_only(source: str, theta_deg: float):
    if theta_deg:
        raise ValueError(
            f"log_spiral_source {source!r} is for a vertical wall face "
            f"(theta = 0), got theta {theta_deg:g} deg; UFC 3-220-20 Fig. "
            "4-12 (log_spiral_source 'dm7_2_fig_4_12', or the default "
            "'ufc_3_220_20') takes a battered face with level ground")


def _ratio(phi_deg: float, delta_deg: float) -> float:
    return delta_deg / phi_deg if phi_deg > 0 else 0.0


def ufc_3_220_20_route(phi_deg: float, delta_deg: float = None,
                       beta_deg: float = 0.0, theta_deg: float = 0.0) -> dict:
    """The UFC 3-220-20 log-spiral Kp, from the source the wall's geometry
    calls for (pp. 191-193; the module docstring).

    ``delta_deg`` None means delta/phi' = 0.66, "a common assumption" (p.
    191) and the one the charts are drawn for. Returns ``route`` (the source
    used, one of :data:`UFC_ROUTES`), ``Kp``, ``delta_deg`` (the wall
    friction used), ``delta_given`` and ``where`` (the equation or figure,
    its page and the case, in words). A case UFC does not print is refused
    with a ValueError that says so and names what covers it."""
    beta = float(beta_deg or 0.0)
    theta = float(theta_deg or 0.0)
    given = delta_deg is not None
    delta = float(delta_deg) if given else DM_DELTA_RATIO * phi_deg
    ratio = _ratio(phi_deg, delta)
    off_066 = phi_deg > 0 and abs(ratio - DM_DELTA_RATIO) > _DM_DELTA_TOL
    ck = ("the Caquot-Kerisel (1948) chart (log_spiral_source "
          "'caquot_kerisel_1948', Caltrans Trenching and Shoring Manual Fig. "
          "4-20: a vertical wall, beta/phi -0.9 to +1, any wall friction)")
    if theta and beta:
        raise ValueError(
            "UFC 3-220-20 prints no log-spiral Kp for a battered face "
            f"(theta {theta:g} deg) with sloping ground (beta {beta:g} "
            "deg) together: Fig. 4-12 (p. 192) is for a battered face with "
            "level ground, Fig. 4-13 (p. 193) for a vertical face with "
            "sloping ground, Eq. 4-12 (p. 191) for a vertical face with "
            "level ground. The Caquot-Kerisel (1948) chart here "
            "(log_spiral_source 'caquot_kerisel_1948') is for a vertical "
            "wall too. UFC p. 190 points to the tables of Caquot and "
            "Kerisel (1948) for other wall geometries and to the log-spiral "
            "equations of Alqarawi et al. (2021); neither is held here. "
            "Read Kp from one of them, or run with theta_deg 0 or beta_deg "
            "0 and say which was left out: leaving out a face that overhangs "
            "the soil (theta > 0) or ground falling away from the wall "
            "(beta < 0) overstates Kp")
    if theta:
        if off_066:
            raise ValueError(
                f"UFC 3-220-20 Fig. 4-12 (p. 192), its log-spiral chart for "
                f"a battered face (theta {theta:g} deg), is drawn for "
                f"delta/phi' = 0.66 only; got delta {delta:g} on phi' "
                f"{phi_deg:g} ({ratio:.3g}). UFC prints no battered-face Kp "
                "at another wall friction (p. 260 allows 0.5 to 1.0 for "
                "tied-back walls and still points to Fig. 4-12). The "
                "Caquot-Kerisel (1948) chart here is for a vertical wall. "
                "Give delta_deg = 0.66 phi' to read Fig. 4-12")
        try:
            Kp = dm7_2_fig_4_12_Kp(phi_deg, theta, delta)
        except ValueError as exc:
            raise ValueError(f"{exc}; UFC 3-220-20 prints nothing beyond "
                             "it, and the Caquot-Kerisel (1948) chart here "
                             "is for a vertical wall") from None
        return {"route": "dm7_2_fig_4_12", "Kp": Kp, "delta_deg": delta,
                "delta_given": given,
                "where": (f"UFC 3-220-20 Fig. 4-12 (p. 192): a battered "
                          f"face (theta {theta:g} deg from vertical) with "
                          "level ground, read at delta/phi' = 0.66, the "
                          "wall friction it is drawn for")}
    if beta:
        if off_066:
            raise ValueError(
                f"UFC 3-220-20 Fig. 4-13 (p. 193), its log-spiral chart for "
                f"sloping ground (beta {beta:g} deg), is drawn for "
                f"delta/phi' = 0.66 only; got delta {delta:g} on phi' "
                f"{phi_deg:g} ({ratio:.3g}). UFC prints no sloping-ground Kp "
                f"at another wall friction; {ck} covers it. Or give "
                "delta_deg = 0.66 phi' to read Fig. 4-13")
        try:
            Kp = dm7_2_fig_4_13_Kp(phi_deg, beta, delta)
        except ValueError as exc:
            raise ValueError(f"{exc}; UFC 3-220-20 prints nothing beyond "
                             f"it, and {ck} may cover it") from None
        return {"route": "dm7_2_fig_4_13", "Kp": Kp, "delta_deg": delta,
                "delta_given": given,
                "where": (f"UFC 3-220-20 Fig. 4-13 (p. 193): a vertical "
                          f"face with sloping ground (beta {beta:g} deg, "
                          f"beta/phi' {beta / phi_deg:.2f}), read at "
                          "delta/phi' = 0.66, the wall friction it is "
                          "drawn for")}
    Kp = dm7_2_eq_4_12_Kp(phi_deg, delta)
    return {"route": "dm7_2_eq_4_12", "Kp": Kp, "delta_deg": delta,
            "delta_given": given,
            "where": (f"UFC 3-220-20 Eq. 4-12 (p. 191), the approximation "
                      f"UFC gives for a vertical face and level ground "
                      f"(theta = 0, beta = 0), at delta/phi' {ratio:.2f}")}


def log_spiral_Kp(phi_deg: float, delta_deg: float = None,
                  beta_deg: float = 0.0,
                  source: str = DEFAULT_SOURCE,
                  theta_deg: float = 0.0) -> float:
    """The log-spiral passive Kp from ``source`` (see :data:`SOURCES`);
    ``theta_deg`` is the wall face's angle from vertical (the module
    docstring), taken by ``ufc_3_220_20`` and ``dm7_2_fig_4_12``."""
    if source == "ufc_3_220_20":
        return ufc_3_220_20_route(phi_deg, delta_deg, beta_deg,
                                  theta_deg)["Kp"]
    if source == "dm7_2_fig_4_12":
        if beta_deg:
            raise ValueError(
                "UFC 3-220-20 Fig. 4-12 is for level ground (beta = 0); its "
                "Fig. 4-13 takes a slope on a vertical face")
        return dm7_2_fig_4_12_Kp(phi_deg, theta_deg, delta_deg)
    if source in ("caquot_kerisel_1948", "dm7_2_fig_4_13", "dm7_2_eq_4_12"):
        _vertical_wall_only(source, theta_deg)
    if source == "caquot_kerisel_1948":
        return caquot_kerisel_1948_Kp(phi_deg, delta_deg, beta_deg=beta_deg)
    if source == "dm7_2_fig_4_13":
        return dm7_2_fig_4_13_Kp(phi_deg, beta_deg, delta_deg)
    if source == "dm7_2_eq_4_12":
        if beta_deg:
            raise ValueError(
                "UFC 3-220-20 Eq. 4-12 is for level ground (beta = 0); its "
                "Fig. 4-13 or the Caquot-Kerisel chart takes a slope")
        return dm7_2_eq_4_12_Kp(phi_deg, delta_deg)
    raise ValueError(f"Unknown log-spiral source {source!r}; one of "
                     f"{list(SOURCES)}")


#: Why the Kp of a wall's design refuses a left-out wall friction with
#: these sources (GM-242 review), naming the input to give.
LEVEL_FRONT_NEEDS_DELTA = {
    "caquot_kerisel_1948": (
        "log_spiral_source 'caquot_kerisel_1948' needs the wall friction "
        "given ({name}): the Caltrans T&S Manual's procedure for the chart "
        "(Sec. 4-6, p. 4-29) begins 'Given delta, beta, and phi', and its "
        "Example 8-1 takes 15 deg on phi 30. Or leave log_spiral_source "
        "out: UFC 3-220-20 takes delta/phi' = 0.66 when it is left out."),
    "dm7_2_eq_4_12": (
        "log_spiral_source 'dm7_2_eq_4_12' (UFC 3-220-20 Eq. 4-12) needs "
        "the wall friction given ({name}); p. 191: 'while delta/phi' = 0.66 "
        "is a common assumption, this type of design may require Kp for a "
        "different value of wall friction'. Or leave log_spiral_source out "
        "(UFC 3-220-20 by geometry, delta/phi' = 0.66 when left out)."),
}


def level_front_left_out(source: str) -> str:
    """What :func:`level_front_Kp` takes for a left-out wall friction with
    ``source``, as the ``left_out`` sentence of :func:`wall_friction_note`
    (GM-242 review)."""
    if source in LEVEL_FRONT_NEEDS_DELTA:
        return (f"With log_spiral_source '{source}' it must be given; left "
                f"out, it is refused.")
    return ("Left out, the tool takes delta/phi' = 0.66, for which UFC "
            "Figs. 4-12 and 4-13 are drawn.")


def level_front_Kp(phi_deg: float, delta_deg: float = None,
                   source: str = DEFAULT_SOURCE,
                   name: str = "passive_wall_friction_deg") -> dict:
    """The log-spiral Kp in front of a vertical wall face with level
    ground, the passive side below an excavation or a retaining wall's toe
    (GM-242): ``Kp``, ``delta_deg`` (the wall friction used), ``delta_given``
    and ``route`` (the UFC 3-220-20 source the default took, else None).
    ``delta_deg`` None takes delta/phi' = 0.66 (UFC 3-220-20 p. 191, 'a
    common assumption', the wall friction its charts are drawn for) with
    the UFC sources by geometry or chart; the Caquot-Kerisel (1948) chart
    and Eq. 4-12 by name refuse it (:data:`LEVEL_FRONT_NEEDS_DELTA`, naming
    the input ``name``), as earth_pressure_coefficient refuses Eq. 4-12:
    a wall's embedment or sliding check is not left at the chart's delta =
    phi'. phi' = 0 gives Kp = 1 with no wall friction."""
    given = delta_deg is not None
    if phi_deg <= 0:
        return {"Kp": 1.0, "delta_deg": 0.0, "delta_given": given,
                "route": None}
    if source == DEFAULT_SOURCE:
        r = ufc_3_220_20_route(phi_deg, delta_deg)
        return {"Kp": r["Kp"], "delta_deg": r["delta_deg"],
                "delta_given": given, "route": r["route"]}
    if not given:
        if source in LEVEL_FRONT_NEEDS_DELTA:
            raise ValueError(LEVEL_FRONT_NEEDS_DELTA[source].format(
                name=name))
        delta_deg = DM_DELTA_RATIO * phi_deg
    return {"Kp": log_spiral_Kp(phi_deg, delta_deg, 0.0, source=source),
            "delta_deg": float(delta_deg), "delta_given": given,
            "route": None}


#: What each source is called in a result's alternatives.
_SOURCE_NAME = {
    "ufc_3_220_20": "UFC 3-220-20 by geometry",
    "caquot_kerisel_1948": "Caquot-Kerisel 1948, Caltrans T&S Fig. 4-20",
    "dm7_2_eq_4_12": "UFC 3-220-20 Eq. 4-12",
    "dm7_2_fig_4_12": "UFC 3-220-20 Fig. 4-12",
    "dm7_2_fig_4_13": "UFC 3-220-20 Fig. 4-13",
}


def source_Kp(key: str, phi_deg: float, delta_deg: float,
              beta_deg: float = 0.0, theta_deg: float = 0.0):
    """``(Kp, None)`` from source ``key`` for the case, or ``(None, why)``
    in a few words when the source gives none; ``delta_deg`` is the wall
    friction used."""
    off_066 = phi_deg > 0 and abs(_ratio(phi_deg, delta_deg)
                                  - DM_DELTA_RATIO) > _DM_DELTA_TOL
    why_not = {
        "caquot_kerisel_1948": theta_deg and "for a vertical wall only",
        "dm7_2_eq_4_12": (theta_deg or beta_deg) and (
            "for theta = 0 and beta = 0 only"),
        "dm7_2_fig_4_12": (beta_deg and "for level ground only") or (
            off_066 and "drawn for delta/phi' = 0.66 only"),
        "dm7_2_fig_4_13": (theta_deg and "for a vertical face only") or (
            off_066 and "drawn for delta/phi' = 0.66 only"),
    }.get(key)
    if why_not:
        return None, why_not
    try:
        return log_spiral_Kp(phi_deg, delta_deg, beta_deg, source=key,
                             theta_deg=theta_deg), None
    except ValueError:
        return None, ("UFC prints none for this case"
                      if key == "ufc_3_220_20" else "off its chart")


def source_label(key: str) -> str:
    """How a result's alternatives name source ``key``."""
    return f"{_SOURCE_NAME[key]} (log_spiral_source '{key}')"


def other_sources(phi_deg: float, delta_deg: float, beta_deg: float = 0.0,
                  theta_deg: float = 0.0, used=()) -> list:
    """Each source not in ``used`` with its Kp for the same case, or why it
    does not give one, for a result's alternatives: "<name> (log_spiral_source
    '<key>'): Kp 5.43" or "...: none here, <reason>". ``delta_deg`` is the
    wall friction the result used."""
    out = []
    for key in SOURCES:
        if key in used:
            continue
        value, reason = source_Kp(key, phi_deg, delta_deg, beta_deg,
                                  theta_deg)
        out.append(f"{source_label(key)}: none here, {reason}" if reason
                   else f"{source_label(key)}: Kp {value:.3g}")
    return out

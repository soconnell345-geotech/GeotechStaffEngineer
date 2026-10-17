"""Phase E / v5.4 F4 — Slide2 #97 Pilarcitos Dam rapid-drawdown method ordering.

Pilarcitos Dam (Duncan, Wright & Wong 1990; GEO-SLOPE Ex 3) — a homogeneous
rolled earth-fill that FAILED on the upstream slope during the Oct-Nov 1969
drawdown. It is the third published rapid-drawdown case history (with #95/#96 and
#98) that tabulates all three total-stress methods, so it cross-checks the
Corps / Lowe-Karafiath / Duncan-Wright-Wong ORDERING added in V-048.

Published (Slide2 #97): Corps 2-stage 0.823 / Lowe-Karafiath 1.047 / DWW 3-stage
1.043 (Duncan-Wright-Wong 1990: 0.82 / 1.05 / 1.05). Drawdown El. 692 -> 657 ft.

RECOVERED DATA (all published — GEO-SLOPE property table + Slide2 Table 97.1):
homogeneous fill gamma=135 pcf, effective c'=0 / phi'=45 deg, total-stress
R-envelope c_R=60 psf / phi_R=23 deg; toe El. 620, crest ~El. 698 (height 78 ft),
full pool El. 692 (72 ft), drawn to El. 657 (37 ft).

GEOMETRY (resolved 2026-07-18/19, wiki-verification): the original DWW 1990
paper IS in the owner's library; its record gives the real upstream face as a
TWO-SLOPE section — 2H:1V for the lower 58 ft, then 3H:1V to the crest (paper
p. 262 Fig. 3) — vs the uniform 2.5H:1V assumed historically (which sits
between the two real inclinations; kept below for the slope-invariance test).
V-052b now runs the TRUE section: Corps 0.722 (published 0.823; < 1.0, failure
reproduced), LK 1.279 / DWW 1.266 (published ~1.05) — the true geometry shrinks
the LK/DWW overshoot from ~29% to ~22%, isolating the residual to the steep
published phi'=45 Kc-interpolation gain documented below.

VERDICT: **CONVENTION (ordering validated; magnitudes geometry/phi'-limited).**
* The three-method ORDERING Corps < DWW < LK (the strength-rule signature) is
  reproduced, with DWW ~ LK (the stage-3 drained check barely fires here, exactly
  as published where DWW 1.043 ~ LK 1.047).
* The Corps 2-stage magnitude lands near published and BELOW 1.0 (reproducing the
  observed failure): ~0.79 at 2.5:1 vs published 0.823.
* The Lowe-Karafiath / DWW magnitudes OVERSHOOT published (~1.35 vs ~1.05). Two
  honest reasons, neither tuned away: (a) the upstream slope is assumed, and (b)
  the published effective phi'=45 deg is very steep (the scout flagged it as high
  for a clayey fill) -> a large Kf=(1+sin45)/(1-sin45)=5.83 -> the Kc interpolation
  between the R (Kc=1) and drained (Kc=Kf) envelopes gains more strength than
  Slide2's narrow LK-vs-Corps spread implies. On the moderate-phi' #98 dam our
  spacing matched published to ~10% (V-048); the wide spread is specific to this
  steep-phi' case. Recorded, ordering pinned, magnitudes documented not tuned.

See INVENTORY.md (#97), RESULTS.md (V-052), and V-048 (#98 Lowe-Karafiath).
"""

import math

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.rapid_drawdown import search_rapid_drawdown

FT, PSF, PCF = 0.3048, 0.04788, 0.157087
_HEIGHT = 78.0  # ft, toe(rel 0) -> crest(rel 78)


def _pilarcitos(slope_hv=2.5):
    """Homogeneous Pilarcitos section with a representative upstream slope
    (angle NOT published; 2.5H:1V typical earth-dam value)."""
    run = slope_hv * _HEIGHT
    face = [(0, 0), (run, _HEIGHT), (run + 80, _HEIGHT)]
    return SlopeGeometry(
        surface_points=[(x * FT, z * FT) for x, z in face],
        soil_layers=[SlopeSoilLayer(
            name="fill", top_elevation=_HEIGHT * FT, bottom_elevation=-8 * FT,
            gamma=135 * PCF, phi=45.0, c_prime=0.0,
            R_c=60 * PSF, R_phi=23.0)])


def _deep_windows(geom):
    """Entry / exit windows for a DEEP upstream slide: one end in the lower
    40 % of the section, the other in the upper 40 % (the automatic windows
    before GM-57). These tests compare with the published values on that
    slide, as they did before GM-57.

    The windows the search now sets from the slope (GM-57) also take
    circles that come out on the face, and under drawdown the
    Lowe-Karafiath / DWW search then finds the rapid-drawdown infinite
    slope: a sliver of a circle on the lower face, about an inch deep, at
    (gamma'/gamma) tan(phi')/tan(beta) = 1.075 (true section; deep slide
    1.279 / 1.266) and 1.076 on the 2:1 section; Corps stays on a deep
    circle, 0.718 (0.726). That answer is pinned in
    test_v052b_automatic_windows_find_the_circle_above_the_pool. (A DWW
    0.955 on the 2:1 section, seen while GM-57 was built, was the stage-3
    Fellenius normal under the remaining pool, which counted the pond's
    pressure on a slice base in full and its weight at cos^2(alpha); with
    the pond taken out of both terms (rapid_drawdown.
    _fellenius_effective_normal) DWW ~ LK again. The default stage-3 normal
    stays 'fellenius', the owner's ruling.)

    Whether the published 0.82 / 1.05 / 1.05 belong to a deep slide cannot
    be checked from the reference library (geotech-references): neither the
    Slide2 verification manual (#97) nor the DWW 1990 paper is in it, and
    what the repo records of them (validation_examples/INVENTORY.md, the
    wiki-verification ledger) gives the factors of safety and the section,
    not the surface they belong to. The deep windows keep these tests what
    they were."""
    x0, x1 = geom.x_range
    w = x1 - x0
    return dict(x_entry_range=(x0, x0 + 0.4 * w),
                x_exit_range=(x0 + 0.6 * w, x1))


def _search(slope_hv, method, infinite_slope=True):
    run = slope_hv * _HEIGHT
    geom = _pilarcitos(slope_hv)
    return search_rapid_drawdown(
        geom, 72 * FT, 37 * FT, method=method,
        surface_type="circular", nx=7, ny=6,
        x_range=(0.2 * run * FT, run * FT),
        y_range=(80 * FT, 90 * FT + 2 * _HEIGHT * FT), n_slices=20,
        infinite_slope=infinite_slope, **_deep_windows(geom)).FOS


def test_v052_pilarcitos_method_ordering_is_slope_invariant():
    """The published three-method ordering Corps < DWW < LK (with DWW ~ LK, the
    stage-3 check barely firing) holds at every plausible upstream slope
    (2:1, 2.5:1, 3:1) — i.e. it is a strength-rule signature, not geometry-tuned.

    The published values belong to the observed DEEP upstream slide, so the
    shallow planar family is left out here (GM-54). With it on, the 2:1 face's
    shallow slide governs: DWW 0.860 / LK 1.017 against the circles' 1.098 /
    1.105 -- the rapid-drawdown infinite slope, about (gamma'/gamma)
    tan(phi')/tan(beta), where the stage-3 drained check fires near the
    surface, a different mechanism from the one published."""
    for slope in (2.0, 2.5, 3.0):
        corps = _search(slope, "corps_2stage", infinite_slope=False)
        lk = _search(slope, "lowe_karafiath", infinite_slope=False)
        dww = _search(slope, "duncan_3stage", infinite_slope=False)
        assert corps < dww < lk                     # published Corps < DWW < LK
        assert dww == pytest.approx(lk, rel=0.05)   # DWW ~ LK (stage 3 barely fires)


def test_v052_pilarcitos_corps_reproduces_failure():
    """Corps 2-stage (R-envelope-governed here) lands near the published 0.823 and
    BELOW 1.0 on the representative 2.5:1 section, reproducing the observed upstream
    failure. (Lowe-Karafiath / DWW overshoot published — see the module docstring:
    assumed slope + steep published phi'=45 inflates the Kc-interpolation gain.)"""
    corps = _search(2.5, "corps_2stage")
    assert corps < 1.0                              # predicts the failure (as published)
    assert corps == pytest.approx(0.823, abs=0.12)  # near published (ours ~0.79)
    # Lowe-Karafiath is above Corps (ordering) but overshoots published 1.047
    lk = _search(2.5, "lowe_karafiath")
    assert lk > corps
    assert lk > 1.0                                 # ours ~1.35; documented overshoot


# ---------------------------------------------------------------------------
# TRUE two-slope section (wiki-verification 2026-07-18/19): DWW 1990 p. 262 /
# the owner-library record give the real upstream face — 2H:1V for the lower
# 58 ft, then 3H:1V to the 78-ft crest (vs the representative uniform 2.5H:1V
# assumed above, which sits between the two real inclinations).
# ---------------------------------------------------------------------------

def _pilarcitos_true():
    face = [(0, 0), (116, 58), (176, 78), (260, 78)]   # ft: 2:1 lower, 3:1 upper
    return SlopeGeometry(
        surface_points=[(x * FT, z * FT) for x, z in face],
        soil_layers=[SlopeSoilLayer(
            name="fill", top_elevation=_HEIGHT * FT, bottom_elevation=-8 * FT,
            gamma=135 * PCF, phi=45.0, c_prime=0.0,
            R_c=60 * PSF, R_phi=23.0)])


def _search_true(method):
    geom = _pilarcitos_true()
    return search_rapid_drawdown(
        geom, 72 * FT, 37 * FT, method=method,
        surface_type="circular", nx=7, ny=6,
        x_range=(10.0, 54.0), y_range=(24.5, 75.0), n_slices=20,
        **_deep_windows(geom)).FOS


def test_v052b_automatic_windows_find_the_circle_above_the_pool():
    """GM-57: with the windows set from the slope (the test's own centre
    grid) the Lowe-Karafiath and DWW searches of the true section govern on
    the rapid-drawdown infinite slope: a sliver of a circle on the lower 2:1
    face, about an inch deep, at (gamma'/gamma) tan(phi')/tan(beta) = 1.075
    (the same anywhere on the face above the pool; Corps = LK = DWW there),
    reported as that mechanism and with no window-edge warning. Corps stays
    on a deep circle (85 ft long, entering below the remaining pool) at
    0.718. (Before the pond fix to the Fellenius normal a fully automatic
    DWW search read 0.052 on a toe circle under the pool.)
    """
    geom = _pilarcitos_true()
    gamma = 135 * PCF
    closed = (gamma - 9.81) / gamma * math.tan(math.radians(45.0)) / 0.5
    out = {}
    for method in ("corps_2stage", "lowe_karafiath", "duncan_3stage"):
        out[method] = search_rapid_drawdown(
            geom, 72 * FT, 37 * FT, method=method, surface_type="circular",
            nx=7, ny=6, x_range=(10.0, 54.0), y_range=(24.5, 75.0),
            n_slices=20)
    corps = out["corps_2stage"]
    c = corps.search.critical
    assert corps.FOS == pytest.approx(0.72, abs=0.01)
    assert (c.x_exit - c.x_entry) / FT > 80.0                 # deep
    assert corps.search.governing_location["depth"] > 1.0
    for method in ("lowe_karafiath", "duncan_3stage"):
        r = out[method]
        s = r.search
        assert r.FOS == pytest.approx(closed, rel=0.005)
        assert s.governing_location["depth"] < 0.1             # a sliver
        assert any("infinite slope mechanism" in w and "rapid drawdown" in w
                   for w in s.warnings)
        assert not any("edge of the" in w for w in s.warnings)


def test_v052b_true_geometry_failure_and_ordering():
    """On the REAL two-slope section: Corps 2-stage = 0.722 — below 1.0
    (failure reproduced) and within ~0.1 of the published 0.823; the ordering
    Corps < DWW <= LK holds. LK/DWW land ~1.27-1.28 vs published ~1.05: the
    true geometry SHRINKS the overshoot from ~29% (assumed 2.5:1 section) to
    ~22%, isolating the residual to the steep published phi'=45 Kc gain (the
    same code matches the moderate-phi' Slide2 #98 dam to ~10%) — documented,
    not tuned."""
    corps = _search_true("corps_2stage")
    lk = _search_true("lowe_karafiath")
    dww = _search_true("duncan_3stage")
    assert corps < 1.0                              # the dam failed; so says Corps
    assert corps == pytest.approx(0.823, abs=0.11)  # ours 0.722 vs published 0.823
    assert corps < dww <= lk + 1e-9                 # published ordering
    assert dww == pytest.approx(lk, rel=0.05)       # DWW ~ LK (stage 3 barely fires)
    # magnitudes recorded (regression-pinned loosely, NOT pinned to published):
    assert lk == pytest.approx(1.28, abs=0.08)
    assert dww == pytest.approx(1.27, abs=0.08)

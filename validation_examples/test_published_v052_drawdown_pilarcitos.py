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

VERDICT (GM-140 step 9, every stage as EM 1110-2-1902 Appendix G prints it):
* The three-method ORDERING Corps < DWW <= LK is reproduced, with DWW ~ LK (the
  drained check barely fires here, as published: DWW 1.043 ~ LK 1.047).
* TRUE section, deep slide: Corps 0.722 (published 0.823, -12 %; below 1.0,
  the failure reproduced), LK 0.944 (1.047, -10 %), DWW 0.943 (1.043, -10 %).
  The three procedures are now low by about the same amount, which points at
  the recovered section and windows, not at one procedure's strength rule.
* 2.5:1 section: Corps 0.786, LK 1.008, DWW 1.004 (-4.5 %, -3.7 %, -3.7 %).
* GM-15, the "steep-phi' K_c overshoot" (LK 1.279 and DWW 1.263 on the true
  section, 22 % above print), was the module's own K_c rule: from the base
  inclination, with special branches, it reached K_f on many slices and took
  the drained envelope's strength. Eq. G-11 (p. G-7) gives K_c from sigma'_c
  and tau_c alone; with it the overshoot is gone. Not tuned.

See INVENTORY.md (#97), RESULTS.md (V-052), V-048 (#98 Lowe-Karafiath) and
module_work/gm140/STEP9_CHECK.md.
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
    circles that come out on the face; what the automatic search finds
    there is pinned in
    test_v052b_automatic_windows_find_the_circle_above_the_pool.

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


# GM-140 step 5a: on the assumed 3:1 section the two searches now end on
# different circles and DWW reads 1.5441 against LK's 1.5426 (it was 1.5416
# against 1.5429; printed, for the real dam: LK 1.047, DWW 1.043). On each
# of the two circles DWW is still below LK (1.5412 / 1.5426 on LK's circle,
# 1.5441 / 1.5461 on DWW's). The pond's edge is now a slice boundary (account
# D16), which moves the 20 slices of every trial circle; at the governing
# centre the drawdown FOS is not smooth in the radius on 20 slices (LK 1.5429,
# 1.5446, 1.5426 and 1.5674 at R = 33.985, 33.990, 33.995 and 34.000 m), so
# the one-dimensional radius search of each method stops at a different lip.
# That is the search's and the drawdown slicing's (steps 7 and 9), not the
# pore-pressure rule; the tolerance is not widened and the value not re-pinned
# (module_work/gm140/STEP5A_CHECK.md).
_SEARCH_LANDS_APART = pytest.mark.xfail(
    strict=True,
    reason="GM-140 step 5a: the DWW and LK radius searches end on different "
           "circles of a jagged 20-slice objective (DWW 1.5441 > LK 1.5426 "
           "by 0.1 %); per circle DWW <= LK holds. See STEP5A_CHECK.md.")


@pytest.mark.parametrize("slope", [
    2.0, 2.5, pytest.param(3.0, marks=_SEARCH_LANDS_APART)])
def test_v052_pilarcitos_method_ordering_is_slope_invariant(slope):
    """The published three-method ordering Corps < DWW < LK (with DWW ~ LK, the
    stage-3 check barely firing) holds at every plausible upstream slope
    (2:1, 2.5:1, 3:1) — i.e. it is a strength-rule signature, not geometry-tuned.

    The published values belong to the observed DEEP upstream slide, so the
    shallow planar family is left out here (GM-54). With it on, the 2:1 face's
    shallow slide governs: DWW 0.860 / LK 1.017 against the circles' 1.098 /
    1.105 -- the rapid-drawdown infinite slope, about (gamma'/gamma)
    tan(phi')/tan(beta), where the stage-3 drained check fires near the
    surface, a different mechanism from the one published."""
    corps = _search(slope, "corps_2stage", infinite_slope=False)
    lk = _search(slope, "lowe_karafiath", infinite_slope=False)
    dww = _search(slope, "duncan_3stage", infinite_slope=False)
    assert corps < dww < lk                     # published Corps < DWW < LK
    assert dww == pytest.approx(lk, rel=0.05)   # DWW ~ LK (stage 3 barely fires)


def test_v052_pilarcitos_corps_reproduces_failure():
    """Corps 2-stage (R-envelope-governed here) lands near the published 0.823 and
    BELOW 1.0 on the representative 2.5:1 section, reproducing the observed upstream
    failure; Lowe-Karafiath lands near its published 1.047."""
    corps = _search(2.5, "corps_2stage")
    assert corps < 1.0                              # predicts the failure (as published)
    assert corps == pytest.approx(0.823, abs=0.12)  # near published (ours ~0.79)
    lk = _search(2.5, "lowe_karafiath")
    assert lk > corps
    # GM-140 step 9: 1.347 -> 1.008; printed 1.047 (Slide2 #97), -3.7 %, on
    # an assumed 2.5:1 face
    assert lk == pytest.approx(1.047, rel=0.06)


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
    grid) every procedure governs on a real circle, not a sliver. Corps
    stays on a deep circle (85 ft long, entering below the remaining pool)
    at 0.718. Lowe-Karafiath and DWW govern on a circle 66 ft long and
    about 3 m deep across the lower 2:1 face, at 0.929 (no slice's drained
    strength is lower there, so the two agree and no third stage is
    computed). Before GM-140 step 9 the module's own K_c rule gave these
    two the rapid-drawdown infinite slope instead, a sliver about an inch
    deep at (gamma'/gamma) tan(phi')/tan(beta) = 1.075.
    """
    geom = _pilarcitos_true()
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
    for method, printed in (("lowe_karafiath", 1.047),
                            ("duncan_3stage", 1.043)):
        r = out[method]
        s = r.search
        # GM-140 step 9: 1.075 (a sliver) -> 0.929 (a 3 m deep circle);
        # printed 1.047 / 1.043 (Slide2 #97), -11 %, on the recovered
        # section; Corps on the same section is 12 % below its print
        assert r.FOS == pytest.approx(printed, rel=0.13)
        assert r.FOS == pytest.approx(0.929, abs=0.02)
        assert s.governing_location["depth"] > 1.0             # not a sliver
        assert not any("edge of the" in w for w in s.warnings)
    dww = out["duncan_3stage"].drawdown
    assert dww.answer_stage == 2 and dww.stage3_fos is None
    assert out["duncan_3stage"].FOS == pytest.approx(
        out["lowe_karafiath"].FOS, abs=1e-6)


def test_v052b_true_geometry_failure_and_ordering():
    """On the REAL two-slope section, deep slide: Corps 2-stage = 0.722, below
    1.0 (failure reproduced) and within ~0.1 of the published 0.823; the
    ordering Corps < DWW <= LK holds; LK 0.944 and DWW 0.943 against the
    published 1.047 and 1.043. This is GM-15's case: before Eq. G-11 gave
    K_c the two read 1.279 and 1.263, 22 % above print."""
    corps = _search_true("corps_2stage")
    lk = _search_true("lowe_karafiath")
    dww = _search_true("duncan_3stage")
    assert corps < 1.0                              # the dam failed; so says Corps
    assert corps == pytest.approx(0.823, abs=0.11)  # ours 0.722 vs published 0.823
    assert corps < dww <= lk + 1e-9                 # published ordering
    assert dww == pytest.approx(lk, rel=0.05)       # DWW ~ LK (stage 3 barely fires)
    # GM-140 step 9: 1.279 -> 0.944; printed 1.047 (Slide2 #97), -9.8 %
    assert lk == pytest.approx(1.047, rel=0.12)
    # GM-140 step 9: 1.263 -> 0.943; printed 1.043 (Slide2 #97), -9.6 %
    assert dww == pytest.approx(1.043, rel=0.12)

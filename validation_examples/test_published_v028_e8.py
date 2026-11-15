"""ACADS 4 (Slide2 Verification Manual problem #9; V-028, E8): the weak
seam with the problem's two distributed loads, against print.

The problem (Giam & Donald 1989; Slide2 manual pp. 50-54): a slope with a
1 m weak seam (c' = 0, phi' = 10 deg) in a soil of c' = 28.5 kPa, phi' = 20
deg, both 18.84 kN/m3 (Table 9.1), a piezometric line (Table 9.3) and two
distributed loads (Table 9.2): 20 kPa on the bench, x = 23 to 43, and 20
rising to 40 kPa on the crest, x = 70 to 80. Slide2 searches it with a
block search: a polyline in the weak layer and projection angles from the
weak layer to the slope surface. Printed, no optimisation (Table 9.4,
manual p. 52): Spencer 0.760, GLE 0.720, Janbu corrected 0.734. With
optimisation (Table 9.5): Spencer 0.707. Referee 0.78 [Giam].

VALIDATION (GM-140 step 1 fix round, 2026-11-13). A fixed family of 81
three-part surfaces is declared below: a wedge down from the bench side,
along the seam's mid-line, a wedge up to the crest. Each is scored by
Spencer, by Morgenstern-Price with the half-sine function and by Janbu's
simplified method with the f0 correction, with the loads on. The lowest
solved value of each method is within 3 % of its printed value: Spencer
0.753 (print 0.760, -0.9 %), Morgenstern-Price 0.722 (print 0.720 by GLE,
+0.3 %), Janbu corrected 0.728 (print 0.734, -0.8 %), all three on (42.91,
28.0), (47.0, 26.09), (76.0, 34.25), (79.66, 40.0). No search routine of
the module is used, and nothing is fitted: the family was declared before
it was scored.

What cannot be checked:
- the ground, the seam and the base are read off Figure 9.1, which is not
  in the repository (the problem text prints the soils, the loads and the
  piezometric line only);
- the printed critical surfaces are in Figures 9.2 to 9.4 only, so the
  comparison is of each method's lowest value, not of one surface;
- the bench is entered at el. 28.0 where Table 9.2 puts the bench load at
  y = 27.75;
- the crest load, printed as 20 rising to 40 kPa, is entered as its mean,
  30 kPa, over x = 70 to 80.
The piezometric line is Table 9.3 as printed.

The module's weak-layer search (GM-140 step 7a: a wedge along each layer,
then the refinement) reaches Table 9.5 refined; its wedge stage alone does
not reach Table 9.4 at the printed inclinations (strict xfails, below).

E8 itself is the capability: ``SlopeGeometry.surcharges``, a list of
(pressure, x_start, x_end) zones summed on top of the single
``surcharge``/``surcharge_x_range`` pair. Its tests further down measure
the zones on the surface kept in test_published_v028_slope.py, with
Janbu's simplified method (corrected), which solves there; that surface
is a record, not a critical surface.

See slope_stability/geometry.py (surcharges / surcharge_at), VALIDATION.md
B11 and test_published_v028_slope.py.
"""

import math

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import PolylineSlipSurface
from slope_stability.slices import build_slices
from slope_stability.gle import gle_fos, janbu_fos
from slope_stability.analysis import search_critical_surface

_SURF = [(20.0, 28.0), (43.0, 28.0), (68.0, 40.0), (84.0, 40.0)]
_SEAM_TOP = [(20.0, 19.0), (84.0, 37.0)]
_SEAM_BOT = [(20.0, 18.0), (84.0, 36.0)]
_PIEZO = [(20.0, 27.75), (43.0, 27.75), (49.0, 29.8), (60.0, 34.0),
          (66.0, 35.8), (74.0, 37.6), (80.0, 38.4), (84.0, 38.4)]
# Table 9.2 external loads: bench 20 kPa (x 23-43); crest 20->40 kPa (x 70-80),
# represented as a uniform 30 kPa (the mean of the published ramp).
_ZONES = [(20.0, 23.0, 43.0), (30.0, 70.0, 80.0)]
# The surface kept in test_published_v028_slope.py (a record, not a
# critical surface).
_SURFACE = [(41.987, 28.0), (49.095, 26.656), (56.203, 28.249),
            (63.312, 30.304), (70.42, 35.555), (77.528, 40.0)]


def _geom(surcharges=None):
    return SlopeGeometry(
        surface_points=_SURF,
        soil_layers=[
            SlopeSoilLayer(name="upper", top_elevation=40.0, bottom_elevation=15.0,
                           gamma=18.84, phi=20.0, c_prime=28.5,
                           bottom_boundary_points=_SEAM_TOP),
            SlopeSoilLayer(name="weak", top_elevation=40.0, bottom_elevation=15.0,
                           gamma=18.84, phi=10.0, c_prime=0.0,
                           bottom_boundary_points=_SEAM_BOT),
            SlopeSoilLayer(name="lower", top_elevation=40.0, bottom_elevation=15.0,
                           gamma=18.84, phi=20.0, c_prime=28.5),
        ],
        gwt_points=_PIEZO, surcharges=surcharges)


# ════════════════════════════════════════════════════════════════════════════
# The validation: a declared family of surfaces along the seam, loads on
# ════════════════════════════════════════════════════════════════════════════
# Each surface: from the ground on the bench side down to the seam's
# mid-line, along the mid-line, and up to the ground on the crest side.
_SEAM_REACHED_AT = (45.0, 47.0, 49.0)       # x where the bench wedge meets it
_SEAM_LEFT_AT = (72.0, 76.0, 80.0)          # x where the crest wedge leaves it
_BENCH_WEDGE_DEG = (25.0, 30.0, 35.0)       # above the horizontal
_CREST_WEDGE_DEG = (50.0, 57.5, 65.0)
_FAMILY_SLICES = 30

# Printed, no optimisation: Slide2 Verification Manual problem #9, Table
# 9.4, manual p. 52.
_PRINTED = {"spencer": 0.760, "morgenstern_price": 0.720, "janbu": 0.734}


def _seam_mid(x):
    """The seam's mid-line: el. 18.5 at x = 20, rising 18 m in 64 m."""
    return 18.5 + (x - 20.0) * 18.0 / 64.0


def _ground(x):
    if x <= 43.0:
        return 28.0
    if x >= 68.0:
        return 40.0
    return 28.0 + (x - 43.0) * 12.0 / 25.0


def _wedge_to_ground(x0, angle_deg, direction):
    """Where a straight leg from the seam's mid-line at x0, rising at
    angle_deg above the horizontal toward the bench (direction -1) or the
    crest (+1), meets the ground (by bisection on the distance in x)."""
    z0, t = _seam_mid(x0), math.tan(math.radians(angle_deg))
    lo, hi = 0.0, 60.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if z0 + t * mid < _ground(x0 + direction * mid):
            lo = mid
        else:
            hi = mid
    return (x0 + direction * hi, _ground(x0 + direction * hi))


def _family():
    for x1 in _SEAM_REACHED_AT:
        for x2 in _SEAM_LEFT_AT:
            for bench in _BENCH_WEDGE_DEG:
                for crest in _CREST_WEDGE_DEG:
                    yield [_wedge_to_ground(x1, bench, -1), (x1, _seam_mid(x1)),
                           (x2, _seam_mid(x2)), _wedge_to_ground(x2, crest, +1)]


@pytest.fixture(scope="module")
def family_lowest():
    """(number of surfaces scored, {method: (lowest solved F, surface)})
    over the family, Table 9.2's loads on. Spencer is the GLE with a
    constant interslice function, Morgenstern-Price the GLE with the
    half-sine; a value counts only where the solution converged (Janbu:
    where it returned one)."""
    geom = _geom(surcharges=_ZONES)
    lowest, n = {}, 0
    for pts in _family():
        slip = PolylineSlipSurface(points=pts)
        slices = build_slices(geom, slip, _FAMILY_SLICES)
        n += 1
        spencer = gle_fos(slices, slip, f_interslice="constant")
        mp = gle_fos(slices, slip, f_interslice="half_sine")
        fj, _uncorrected, _f0 = janbu_fos(slices, slip)
        for name, value, solved in (
                ("spencer", spencer.fos, spencer.converged),
                ("morgenstern_price", mp.fos, mp.converged),
                ("janbu", fj, fj < 900.0)):
            if solved and (name not in lowest or value < lowest[name][0]):
                lowest[name] = (value, pts)
    return n, lowest


def test_v028_family_is_the_81_declared_surfaces(family_lowest):
    n, lowest = family_lowest
    assert n == 81
    assert set(lowest) == set(_PRINTED)


def test_v028_spencer_against_print(family_lowest):
    # printed 0.760: Slide2 Verification Manual problem #9, Table 9.4
    # (no optimisation), manual p. 52
    assert family_lowest[1]["spencer"][0] == pytest.approx(0.760, rel=0.03)


def test_v028_morgenstern_price_against_print(family_lowest):
    # printed 0.720 (GLE): Slide2 Verification Manual problem #9, Table
    # 9.4 (no optimisation), manual p. 52
    assert family_lowest[1]["morgenstern_price"][0] == pytest.approx(
        0.720, rel=0.03)


def test_v028_janbu_corrected_against_print(family_lowest):
    # printed 0.734 (Janbu corrected): Slide2 Verification Manual problem
    # #9, Table 9.4 (no optimisation), manual p. 52
    assert family_lowest[1]["janbu"][0] == pytest.approx(0.734, rel=0.03)


_WEDGE_STAGE_SHORT = (
    "GM-140 step 7a: the weak-layer search's wedge stage (refine=False), "
    "a wedge along each layer with the wedges at EM 1110-2-1902 C-1g(2)'s "
    "printed inclinations (45 -/+ phi'_D/2, pp. C-8 to C-10), reads Spencer "
    "0.804 against Table 9.4's 0.760 (+5.8 %) and above the declared "
    "family's 0.748. The family's lowest surface has its bench wedge at 25 "
    "deg; C-1g(2)'s passive inclination on the trial's F is 31 to 33 deg, "
    "and on the same base ends (47, 76) the printed angles give 0.79. So "
    "the shortfall is the fixed inclinations, not the base ends: Slide2's "
    "block search varies the projection angles, and EM Fig. C-7b shows "
    "searched inclinations, which the brief's item 4 does not include (a "
    "call left for the owner in STEP7A_CHECK.md). Nothing tuned; the "
    "refined search reaches Table 9.5 (test above).")


@pytest.fixture(scope="module")
def weak_layer_searches():
    """The weak-layer search, Spencer, Table 9.2's loads on, without and
    with the refinement (GM-140 step 7a)."""
    return {refine: search_critical_surface(
        _geom(surcharges=_ZONES), surface_type="weak_layer", method="spencer",
        n_slices=30, refine=refine, x_entry_range=(20.0, 50.0),
        x_exit_range=(58.0, 84.0)) for refine in (False, True)}


def test_v028_weak_layer_search_refined_against_table_9_5(
        weak_layer_searches):
    """The refined search (EM C-1g(3), p. C-10) against Slide2's result
    with optimisation. Its surface carries the printed checks' flags,
    which the result reports beside it (the declared family's surfaces
    carry them too)."""
    res = weak_layer_searches[True]
    assert res.critical is not None
    # GM-140 step 7a: 0.830 (flagged) -> 0.697; printed 0.707 (Spencer,
    # Slide2 Verification Manual problem #9, Table 9.5, with optimisation,
    # manual p. 52), -1.4 %
    assert res.critical.FOS == pytest.approx(0.707, rel=0.03)
    assert res.critical_flags          # reported with its flags
    assert res.critical.FOS <= weak_layer_searches[False].critical.FOS


@pytest.mark.xfail(strict=True, reason=_WEDGE_STAGE_SHORT)
def test_v028_weak_layer_wedge_stage_against_table_9_4(weak_layer_searches):
    res = weak_layer_searches[False]
    # printed 0.760 (Spencer): Slide2 Verification Manual problem #9,
    # Table 9.4 (no optimisation), manual p. 52
    assert res.critical.FOS == pytest.approx(0.760, rel=0.03)


@pytest.mark.xfail(strict=True, reason=_WEDGE_STAGE_SHORT)
def test_v028_weak_layer_wedge_stage_reaches_the_declared_family(
        weak_layer_searches, family_lowest):
    """The wedge stage reaches the family declared above (the brief's
    coverage assertion), within the slicing difference: the family's best
    surface at 30 against 60 slices."""
    f_fam, pts = family_lowest[1]["spencer"]
    slip = PolylineSlipSurface(points=pts)
    g = _geom(surcharges=_ZONES)
    f60 = gle_fos(build_slices(g, slip, 60), slip,
                  f_interslice="constant").fos
    assert (weak_layer_searches[False].critical.FOS
            <= f_fam + abs(f60 - f_fam) + 1e-6)


# ════════════════════════════════════════════════════════════════════════════
# E8, the capability: several surcharge zones (a record on the kept surface)
# ════════════════════════════════════════════════════════════════════════════

def _fos(surcharges=None, single=0.0):
    """Janbu's simplified method with the f0 correction on the kept
    surface (Spencer and GLE do not converge on it)."""
    slip = PolylineSlipSurface(points=_SURFACE)
    g = _geom(surcharges)
    if single:
        g.surcharge = single
        g.surcharge_x_range = (70.0, 80.0)
    return janbu_fos(build_slices(g, slip, 50), slip)[0]


def test_v028e8_no_surcharge_matches_v028():
    """Default (no surcharges) reproduces the record on the kept surface."""
    assert _fos() == pytest.approx(1.073, abs=0.005)


def test_v028e8_two_zones_lower_fos():
    """The two ACADS-4 surcharge zones lower the FOS on the kept surface
    (the crest-side zone adds driving weight; the bench zone sits over the
    toe and resists)."""
    f0, f2 = _fos(), _fos(_ZONES)
    assert f2 == pytest.approx(1.034, abs=0.005)     # ours (a record)
    assert f2 < f0
    assert _fos([_ZONES[1]]) == pytest.approx(1.023, abs=0.005)
    assert _fos([_ZONES[0]]) > f0


def test_v028e8_zones_and_single_surcharge_coexist():
    """The zones sum on TOP of the legacy single surcharge; a zone the surface
    does not cross does not change the FOS (no slices under it)."""
    # a zone entirely upslope of the sliding mass (x 10-18) is inert
    assert _fos([(50.0, 10.0, 18.0)]) == pytest.approx(_fos(), abs=1e-9)
    # single surcharge + a zone both apply (sum): heavier load, lower FOS,
    # the same as one zone of the summed pressure
    both = _fos([(30.0, 70.0, 80.0)], single=15.0)
    assert both < _fos([(30.0, 70.0, 80.0)])
    assert both == pytest.approx(_fos([(45.0, 70.0, 80.0)]), abs=1e-9)


def test_v028e8_bad_zone_rejected():
    with pytest.raises(ValueError, match="x_start < x_end"):
        _geom([(20.0, 43.0, 23.0)])
    with pytest.raises(ValueError, match="non-negative"):
        _geom([(-5.0, 23.0, 43.0)])

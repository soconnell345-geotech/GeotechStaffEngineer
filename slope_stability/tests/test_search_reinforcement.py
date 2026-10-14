"""GM-53 / GM-54: the critical-surface searches score reinforcement, and the
rapid-drawdown search runs the toe-circle and infinite-slope families.

GM-53. `search._compute_fos` used to score every trial surface WITHOUT its
nails / anchors / geosynthetics / stabilizing piles (only `analyze_slope`
resolved them), so a reinforced slope searched to its unreinforced FOS.
Slide2 #86 (Duncan & Wright 2005 Fig 7.28: c' = 0, phi' = 37 deg fill, five
800 lb/ft tiebacks, 20 ft long, at el 4-20 ft; "global failure" windows):
published Bishop 1.629, Spencer 1.620, GLE 1.622; D&W 1.61.

GM-54. `search_rapid_drawdown` now scores the same supplementary families as
`search_critical_surface`, with the drawdown FOS.
"""

import math
import warnings

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.reinforcement import Anchor, Geosynthetic
from slope_stability.slip_surface import CircularSlipSurface
from slope_stability.analysis import (
    analyze_slope, search_critical_surface, rapid_drawdown_fos,
)
from slope_stability.rapid_drawdown import search_rapid_drawdown
from slope_stability.search import _compute_fos, _FOS_MAX, trial_reinforcement
from slope_stability.search_supplements import (
    search_toe_circles, CIRCULAR, TOE_CIRCLE, INFINITE_SLOPE, HELD_NOTE,
)


def _acads(T=None):
    g = SlopeGeometry(
        surface_points=[(20.0, 25.0), (30.0, 25.0), (50.0, 35.0), (70.0, 35.0)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=35.0,
                                    bottom_elevation=20.0, gamma=20.0,
                                    phi=19.6, c_prime=3.0)])
    if T:
        g.geosynthetics = [Geosynthetic(elevation=29.0, T_allow=T)]
    return g


def _slide86(tiebacks=True):
    g = SlopeGeometry(
        surface_points=[(0, 0), (30, 24), (75, 24)],
        soil_layers=[SlopeSoilLayer(name="fill", top_elevation=24.0,
                                    bottom_elevation=0.0, gamma=130.0,
                                    phi=37.0, c_prime=0.0)])
    if tiebacks:
        g.anchors = [Anchor(x_head=z / 0.8, z_head=z, length=20.0,
                            T_allow=800.0, inclination=0.0)
                     for z in (4, 8, 12, 16, 20)]
    return g


def _slide85(anchor=True):
    """Slide2 #85 (D&W Fig 6.34): saturated clay c = 350 psf, gamma = 98 pcf
    on a rigid base at el 10; one 9,000 lb/ft support at mid-height."""
    g = SlopeGeometry(
        surface_points=[(15, 10), (25, 30), (57, 30)],
        soil_layers=[SlopeSoilLayer(name="clay", top_elevation=30,
                                    bottom_elevation=10, gamma=98.0,
                                    cu=350.0, analysis_mode="undrained")])
    if anchor:
        g.anchors = [Anchor(x_head=20.0, z_head=20.0, length=37.0,
                            T_allow=9000.0, inclination=0.0)]
    return g


# ---------------------------------------------------------------------------
# GM-53: reinforcement in every trial surface
# ---------------------------------------------------------------------------

def test_unreinforced_geometry_resolves_nothing():
    """No reinforcement -> nothing resolved, every solve as before."""
    assert trial_reinforcement(_acads(), CircularSlipSurface(40, 45, 21)) is None


def test_reinforced_search_matches_analyze_slope():
    """The searched FOS is the reinforced one: it equals analyze_slope on the
    reported surface, and is above the unreinforced search (0.985)."""
    g = _acads(200.0)
    res = search_critical_surface(g, method="bishop")
    c = res.critical
    again = analyze_slope(g, method="bishop", xc=c.xc, yc=c.yc,
                          radius=c.radius).FOS
    assert c.FOS == pytest.approx(again, rel=1e-9)
    plain = search_critical_surface(_acads(), method="bishop").critical.FOS
    assert c.FOS > plain + 0.1


def test_slide86_tiebacks_reproduce_published():
    """Slide2 #86 with its five tiebacks, global-failure windows: Bishop
    1.612 against Slide2 1.629 and Duncan & Wright 1.61; the critical surface
    crosses all five tiebacks, and the shallow planar slide (0.942 without
    supports) no longer governs."""
    res = search_critical_surface(_slide86(), method="bishop",
                                  x_entry_range=(0.0, 4.0),
                                  x_exit_range=(30.0, 75.0))
    c = res.critical
    assert c.FOS == pytest.approx(1.612, abs=0.01)                 # ours
    assert c.FOS == pytest.approx(1.61, rel=0.02)                  # D&W
    assert c.FOS == pytest.approx(1.629, rel=0.02)                 # Slide2
    assert len(c.reinforcements) == 5
    assert res.governing_surface != INFINITE_SLOPE
    assert res.candidates.get(INFINITE_SLOPE, _FOS_MAX) > 1.6
    chk = res.infinite_slope_checks[0]
    assert chk["FOS_dry_closed_form"] == pytest.approx(0.942, abs=0.001)
    assert chk["note"] == HELD_NOTE            # tiebacks hold the face
    assert not any("infinite slope may govern" in w for w in res.warnings)


def test_reinforced_rigorous_nonconvergence_is_rejected():
    """On a reinforced circle where the rigorous GLE does not converge,
    spencer_fos falls back to the legacy solver, which ignores the
    reinforcement; a search must reject that surface, not score it
    unreinforced. (Slide2 #85 anchored toe circle: analyze_slope's Spencer
    falls back to 0.891, the unreinforced value.)"""
    g = _slide85()
    slip = CircularSlipSurface(15.012, 24.112, 14.112)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fallback = analyze_slope(g, method="spencer", slip_surface=slip).FOS
    assert fallback == pytest.approx(0.891, abs=0.01)
    assert _compute_fos(g, slip, "spencer", 30) == _FOS_MAX
    assert _compute_fos(g, slip, "bishop", 30) < _FOS_MAX


def test_toe_circles_stay_above_a_rigid_base():
    """A toe on the rigid base: every toe circle may at most touch the base
    (build_slices tolerates a below-base sliver; the toe family does not)."""
    out = search_toe_circles(_slide85(anchor=False), method="bishop")
    assert out["FOS"] < _FOS_MAX
    assert out["yc"] - out["radius"] >= 10.0 - 1e-4


def test_rapid_drawdown_counts_reinforcement():
    """rapid_drawdown_fos applies the reinforcement crossing the surface at
    every stage, as analyze_slope does."""
    def dam(T=None):
        g = SlopeGeometry(
            surface_points=[(-10, 0), (0, 0), (30, 15), (40, 15)],
            soil_layers=[SlopeSoilLayer(
                name="fill", top_elevation=15, bottom_elevation=-5,
                gamma=19, gamma_sat=20, phi=28, c_prime=8,
                R_c=10, R_phi=16)])
        if T:
            g.geosynthetics = [Geosynthetic(elevation=6.0, T_allow=T)]
        return g
    circle = dict(xc=10.0, yc=25.0, radius=27.0)
    plain = rapid_drawdown_fos(dam(), 13.0, 3.0, **circle)
    reinforced = rapid_drawdown_fos(dam(80.0), 13.0, 3.0, **circle)
    assert plain.stage3_converged and reinforced.stage3_converged
    assert plain.FOS == pytest.approx(1.006, abs=0.005)
    assert reinforced.FOS == pytest.approx(1.043, abs=0.005)


# ---------------------------------------------------------------------------
# GM-54: the rapid-drawdown search runs the supplementary families
# ---------------------------------------------------------------------------

def _shell():
    """A free-draining c' = 0 shell face (R_phi None) at 1.25H:1V."""
    return SlopeGeometry(
        surface_points=[(0, 0), (30, 24), (75, 24)],
        soil_layers=[SlopeSoilLayer(name="shell", top_elevation=24.0,
                                    bottom_elevation=-10.0, gamma=19.0,
                                    gamma_sat=20.0, phi=37.0, c_prime=0.0)])


def test_rapid_drawdown_search_runs_the_families():
    kw = dict(method="corps_2stage", surface_type="circular", nx=4, ny=4,
              n_slices=20)
    off = search_rapid_drawdown(_shell(), 20.0, 5.0, toe_circles=False,
                                infinite_slope=False, **kw)
    assert off.search.governing_surface == CIRCULAR
    assert off.FOS > 1.3
    on = search_rapid_drawdown(_shell(), 20.0, 5.0, **kw)
    cand = on.search.candidates
    closed = math.tan(math.radians(37.0)) / 0.8
    assert cand[INFINITE_SLOPE] == pytest.approx(closed, rel=0.005)
    assert on.FOS <= cand[INFINITE_SLOPE] + 1e-9
    assert on.search.governing_surface in (TOE_CIRCLE, INFINITE_SLOPE)
    assert on.drawdown.FOS == pytest.approx(on.FOS, abs=1e-6)
    assert "Governing surface" in on.summary()

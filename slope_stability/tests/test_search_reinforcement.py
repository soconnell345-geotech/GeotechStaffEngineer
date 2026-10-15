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

GM-59. A support force acts at the base of the slice it crosses, in every
equation the method satisfies (Wright & Duncan 1991): Bishop counts its
moment and its vertical component; active (Method A) or passive (Method B,
divided by F) in every method; reverse-curvature circles closed by a crack
(default) or discarded. GM-60. Spencer / GLE with large point loads: the
direct march and a robust lambda search. Pinned to Slide2 #85.
"""

import math
import warnings

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.reinforcement import Anchor, Geosynthetic
from slope_stability.nails import SoilNail
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


# ---------------------------------------------------------------------------
# GM-59: a support force in each method; active and passive; reverse curvature
# ---------------------------------------------------------------------------
#
# Slide2 #85 (Duncan & Wright 2005 Fig 6.34), Table 85.2 / 85.3, circular:
#   active  Bishop 1.531  Spencer 1.884  GLE 1.575   (D&W 1.51)
#   passive Bishop 1.324  Spencer 1.872  GLE 1.378   (D&W 1.32)
# Slide2's printed GLE (active) circle: centre (15.446, 37.624), R 27.594.

_SLIDE2_85_CIRCLE = dict(xc=15.446, yc=37.624, radius=27.594)


def _slide85c(convention="active", **geom_kw):
    """`_slide85` with the support's convention and geometry options."""
    return SlopeGeometry(
        surface_points=[(15, 10), (25, 30), (57, 30)],
        soil_layers=[SlopeSoilLayer(name="clay", top_elevation=30,
                                    bottom_elevation=10, gamma=98.0,
                                    cu=350.0, analysis_mode="undrained")],
        anchors=[Anchor(x_head=20.0, z_head=20.0, length=37.0,
                        T_allow=9000.0, inclination=0.0,
                        support_convention=convention)],
        **geom_kw)


def test_slide85_rigorous_methods_converge_on_slide2_circle():
    """GM-60 pin: Spencer and GLE now converge on Slide2's own #85 circle
    (before, both fell back to the unreinforced legacy value 0.913). With
    phi = 0 every method that satisfies moment equilibrium gives the same F
    on a circle, so all three equal Bishop: 1.564 against Slide2 GLE 1.575
    and D&W 1.51."""
    vals = {m: analyze_slope(_slide85c(), method=m, **_SLIDE2_85_CIRCLE).FOS
            for m in ("bishop", "spencer", "gle")}
    assert vals["bishop"] == pytest.approx(1.564, abs=0.002)
    assert vals["spencer"] == pytest.approx(vals["bishop"], abs=1e-3)
    assert vals["gle"] == pytest.approx(vals["bishop"], abs=1e-3)
    for v in vals.values():
        assert 1.51 * 0.99 <= v <= 1.884         # the published spread
    assert vals["gle"] == pytest.approx(1.575, rel=0.01)


def test_slide85_gle_state_is_in_equilibrium_slice_by_slice():
    """GM-60: the rigorous solution with the 9,000 lb/ft point load closes
    every slice's horizontal and vertical equilibrium (the old lagged
    iteration's 'converged' state on these circles missed horizontal
    equilibrium by up to 199 lb/ft on a slice, at the wrong lambda)."""
    from slope_stability.gle import gle_fos
    from slope_stability.reinforcement import compute_reinforcement_forces
    from slope_stability.slices import build_slices
    g = _slide85c()
    slip = CircularSlipSurface(**_SLIDE2_85_CIRCLE)
    x_in, x_out = slip.find_entry_exit(g)
    slices = build_slices(g, slip, 30)
    forces = compute_reinforcement_forces(g, slip, x_in, x_out)
    for f_name, lam in (("constant", 1.005), ("half_sine", 1.465)):
        r = gle_fos(slices, slip, f_interslice=f_name, reinf_forces=forces)
        assert r.converged
        assert r.lam == pytest.approx(lam, abs=0.01)
        E, X = r.interslice_E, r.interslice_X
        for i, s in enumerate(slices):
            N, S, a = r.base_normal[i], r.shear_mobilized[i], s.alpha
            rh = sum(f.T * f.dir_x for f in forces
                     if s.x_left <= f.x <= s.x_right)
            assert (E[i + 1] - E[i]) == pytest.approx(
                -N * math.sin(a) + S * math.cos(a) + rh, abs=0.05)
            assert N * math.cos(a) + S * math.sin(a) == pytest.approx(
                s.weight - (X[i] - X[i + 1]), abs=0.05)


def test_slide85_passive_bishop_matches_published():
    """Method B (passive): the support is divided by F with the soil
    strength. On Slide2's active circle Bishop gives 1.329 (D&W 1.32, Slide2
    passive Bishop 1.324); active gives 1.564 there."""
    passive = analyze_slope(_slide85c("passive"), method="bishop",
                            **_SLIDE2_85_CIRCLE).FOS
    assert passive == pytest.approx(1.329, abs=0.002)
    assert passive == pytest.approx(1.32, rel=0.01)


def test_slide85_bishop_search_reverse_curvature_and_short_arm():
    """The default Bishop search finds 1.304 on a reverse-curvature toe
    circle (centre 6 ft below the crest, closed by a 5.9 ft dry crack, as
    Slide2's default closes such circles); the support passes 4.1 ft below
    its centre, so Bishop counts 29 % of it. The result says both, and that
    Spencer has no solution there. Discarding reverse-curvature circles
    (Slide2's other choice) gives 1.503 active / 1.305 passive against D&W
    1.51 / 1.32 and Slide2 1.531 / 1.324."""
    res = search_critical_surface(_slide85c(), method="bishop")
    assert res.critical.FOS == pytest.approx(1.304, abs=0.005)
    text = " ".join(res.warnings)
    assert "reverse-curvature" in text and "5.89 m deep" in text
    assert "29 % of their force counts" in text
    assert "finds no equilibrium solution" in text
    for convention, fos, dw in (("active", 1.503, 1.51),
                                ("passive", 1.305, 1.32)):
        r = search_critical_surface(
            _slide85c(convention, reverse_curvature="discard"),
            method="bishop")
        assert r.critical.FOS == pytest.approx(fos, abs=0.005)
        assert r.critical.FOS == pytest.approx(dw, rel=0.015)
        assert r.critical.yc >= 30.0                 # centre above the crest
        assert not any("reverse-curvature" in w for w in r.warnings)


def test_reverse_curvature_option_is_checked():
    with pytest.raises(ValueError, match="reverse_curvature"):
        _slide85c(reverse_curvature="keep")
    slip = CircularSlipSurface(15.012, 24.112, 14.112)
    crack = slip.reverse_curvature_crack(_slide85c())
    assert crack["depth"] == pytest.approx(5.888, abs=0.01)
    assert CircularSlipSurface(**_SLIDE2_85_CIRCLE).reverse_curvature_crack(
        _slide85c()) is None


def test_slide85_toe_family_rescoring_finds_spencer_circles():
    """GM-60: with Spencer the toe-circle family scans with Bishop and
    re-scores the best circles; the best Bishop toe circles of anchored #85
    have no Spencer solution, so it now passes over them (before: nothing
    found)."""
    from slope_stability.search_supplements import search_toe_circles
    out = search_toe_circles(_slide85c(), method="spencer")
    assert out["FOS"] < _FOS_MAX
    assert out["FOS"] == pytest.approx(1.584, abs=0.01)
    assert 1.51 <= out["FOS"] <= 1.884               # the published spread


def _phi_slope():
    """A c'-phi' slope with a downward-inclined anchor (the vertical
    component matters only where phi > 0)."""
    g = SlopeGeometry(
        surface_points=[(0.0, 10.0), (20.0, 10.0), (40.0, 20.0), (70.0, 20.0)],
        soil_layers=[SlopeSoilLayer(name="soil", top_elevation=20.0,
                                    bottom_elevation=-15.0, gamma=19.0,
                                    phi=25.0, c_prime=8.0)])
    slip = CircularSlipSurface(30.0, 32.0, 26.0)
    return g, slip


def _hand_bishop(slices, slip, forces, passive=False, vertical=True):
    """Bishop's simplified method written out (Duncan, Wright & Brandon Ch.
    6), with each support at the base of the slice it crosses (Wright &
    Duncan 1991): its moment about the centre and (``vertical``) its
    vertical component in that slice's vertical equilibrium (W -> W - Fz)."""
    F = 1.5
    for _ in range(200):
        div = F if passive else 1.0
        drive = sum((s.weight + s.surcharge_force) * (s.x_mid - slip.xc)
                    for s in slices) / slip.radius
        m_t = sum(f.T * ((f.x - slip.xc) * f.dir_z - (f.z - slip.yc)
                         * abs(f.dir_x)) for f in forces) / slip.radius
        res = 0.0
        for s in slices:
            fz = sum(f.T * f.dir_z for f in forces
                     if s.x_left <= f.x <= s.x_right) / div if vertical else 0.0
            tp = math.tan(math.radians(s.phi))
            m_a = math.cos(s.alpha) + math.sin(s.alpha) * tp / F
            res += (s.c * s.width + max(s.weight - fz - s.pore_pressure
                                        * s.width, 0.0) * tp) / m_a
        F_new = ((res + m_t) / drive) if passive else (res / (drive - m_t))
        if abs(F_new - F) < 1e-10:
            return F_new
        F = F_new
    return F


@pytest.mark.parametrize("convention", ["active", "passive"])
def test_bishop_counts_the_vertical_component(convention):
    """Wright & Duncan (1991), conclusion 6: Bishop's simplified method
    includes the vertical component of a support in the vertical equilibrium
    of the slice it crosses. A 30 deg anchor in phi' = 25 deg soil: the
    module equals the method written out, and differs from the moment-only
    value."""
    from slope_stability.methods import bishop_fos
    from slope_stability.reinforcement import compute_reinforcement_forces
    from slope_stability.slices import build_slices
    g, slip = _phi_slope()
    g.anchors = [Anchor(x_head=34.0, z_head=17.0, length=25.0, T_allow=150.0,
                        inclination=30.0, support_convention=convention)]
    x_in, x_out = slip.find_entry_exit(g)
    slices = build_slices(g, slip, 40)
    forces = compute_reinforcement_forces(g, slip, x_in, x_out)
    assert len(forces) == 1 and forces[0].dir_z < -0.4
    passive = convention == "passive"
    ours = bishop_fos(slices, slip, tol=1e-10, reinf_forces=forces)
    hand = _hand_bishop(slices, slip, forces, passive=passive)
    assert ours == pytest.approx(hand, rel=1e-6)
    # the downward component presses the base: more friction than the
    # moment alone gives
    moment_only = _hand_bishop(slices, slip, forces, passive=passive,
                               vertical=False)
    assert ours > moment_only * 1.002


@pytest.mark.parametrize("method", ["fellenius", "bishop", "janbu",
                                    "spencer", "gle"])
def test_passive_below_active_in_every_method(method):
    """Every method honours the convention, for every support kind: the
    same force divided by F (passive, Method B) raises F less than applied
    unfactored (active, Method A)."""
    g, slip = _phi_slope()
    out = {}
    for convention in ("active", "passive"):
        g.nails = [SoilNail(x_head=30.0, z_head=15.0, length=22.0,
                            bond_stress=120.0, spacing_h=1.5,
                            support_convention=convention)]
        out[convention] = analyze_slope(g, slip_surface=slip, method=method,
                                        n_slices=40).FOS
    g.nails = None
    plain = analyze_slope(g, slip_surface=slip, method=method,
                          n_slices=40).FOS
    assert plain < out["passive"] < out["active"]


def test_support_convention_is_checked_on_every_support():
    with pytest.raises(ValueError, match="support_convention"):
        Anchor(x_head=0, z_head=0, length=5, T_allow=10,
               support_convention="method_b")
    with pytest.raises(ValueError, match="support_convention"):
        Geosynthetic(elevation=1.0, T_allow=10, support_convention="x")
    with pytest.raises(ValueError, match="support_convention"):
        SoilNail(x_head=0, z_head=0, length=5, support_convention="x")

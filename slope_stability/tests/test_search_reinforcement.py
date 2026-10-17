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
import re
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
    unreinforced. (Slide2 #85 anchored toe circle: spencer_fos falls back
    to 0.938, the unreinforced value (0.891 before GM-134); analyze_slope
    reports no FOS, GM-81.)"""
    from slope_stability.methods import spencer_fos
    from slope_stability.slices import build_slices
    g = _slide85()
    slip = CircularSlipSurface(15.012, 24.112, 14.112)
    forces = trial_reinforcement(g, slip)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fallback, _ = spencer_fos(build_slices(g, slip, 30), slip,
                                  reinf_forces=forces)
    assert fallback == pytest.approx(0.938, abs=0.01)
    assert analyze_slope(g, method="spencer", slip_surface=slip).FOS is None
    assert _compute_fos(g, slip, "spencer", 30) == _FOS_MAX
    assert _compute_fos(g, slip, "bishop", 30) < _FOS_MAX


def test_toe_circles_stay_above_a_rigid_base():
    """A toe on the rigid base: every toe circle may at most touch the base
    (build_slices tolerates a below-base sliver; the toe family does not).
    The arc's lowest point inside the sliding mass: the best circle, since
    GM-134, is centred left of the toe and rises from it."""
    g = _slide85(anchor=False)
    out = search_toe_circles(g, method="bishop")
    assert out["FOS"] < _FOS_MAX
    slip = CircularSlipSurface(out["xc"], out["yc"], out["radius"])
    x_en, x_ex = slip.find_entry_exit(g)
    x_low = min(max(out["xc"], x_en), x_ex)
    assert slip.slip_elevation_at(x_low) >= 10.0 - 1e-4


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
    """The circle search alone reaches a shallow circle on the face at the
    closed form, 0.944 (above 1.3 with windows from the model width, before
    GM-57); with the families on, a toe circle governs."""
    kw = dict(method="corps_2stage", surface_type="circular", nx=4, ny=4,
              n_slices=20)
    off = search_rapid_drawdown(_shell(), 20.0, 5.0, toe_circles=False,
                                infinite_slope=False, **kw)
    closed = math.tan(math.radians(37.0)) / 0.8
    assert off.search.governing_surface == CIRCULAR
    assert off.FOS == pytest.approx(closed, rel=0.005)
    on = search_rapid_drawdown(_shell(), 20.0, 5.0, **kw)
    cand = on.search.candidates
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
    on a circle, so all three equal Bishop: 1.566 (1.568 on 960 slices)
    against Slide2 GLE 1.575 and D&W 1.51. Since GM-134 each slice's base
    is its chord, whose normal passes a hair off the centre: the GLE takes
    that arm and Bishop does not, so they agree within 0.1 % (exactly
    before, at 1.564)."""
    vals = {m: analyze_slope(_slide85c(), method=m, **_SLIDE2_85_CIRCLE).FOS
            for m in ("bishop", "spencer", "gle")}
    assert vals["bishop"] == pytest.approx(1.566, abs=0.002)
    assert vals["spencer"] == pytest.approx(vals["bishop"], rel=1e-3)
    assert vals["gle"] == pytest.approx(vals["bishop"], rel=1e-3)
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
    for f_name, lam in (("constant", 0.990), ("half_sine", 1.452)):
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
    """The default Bishop search finds 1.372 (1.374 on 960 slices) on a
    reverse-curvature toe circle (centre 6 ft below the crest, closed by a
    5.9 ft dry crack, as Slide2's default closes such circles); the support
    passes 4.1 ft below its centre, so Bishop counts 29 % of it. The result
    says both, and that Spencer has no solution there. Discarding
    reverse-curvature circles (Slide2's other choice) gives 1.511 active /
    1.316 passive at 30 slices and 1.514 / 1.318 at 240 on the same
    circles, against D&W 1.51 / 1.32 (Slide2 1.531 / 1.324): active the
    main search's circle (centre (17.3, 32.5), R 22.5), passive a circle
    passing above the toe (GM-122; centre (17.0, 33.0), R 23.0). Before
    GM-57 the automatic exit window, from x = 40.2, kept the search off
    such circles.

    Those 'discard' values check m_alpha on every slice, as Slide2 does
    (``m_alpha_rule='every_slice'``). By default slices with phi' = 0 are
    not checked (GM-127): 1.508 / 1.316 at 30 slices, 1.512 / 1.318 at 240.

    Before GM-134 each slice of a circle took the circle's angle at its
    midpoint, and the base of the slice where the arc turns vertical,
    width / cos(a), fell short of the arc: the toe circle read 1.304 on 30
    slices (1.325 on 60, 1.362 on 960), the 'discard' searches 1.467 /
    1.296, and by default a circle centred at the crest's level whose last
    slice had m_alpha 0.195 kept 1.432 / 1.274."""
    res = search_critical_surface(_slide85c(), method="bishop")
    assert res.critical.FOS == pytest.approx(1.372, abs=0.005)
    text = " ".join(res.warnings)
    assert "reverse-curvature" in text and "5.89 m deep" in text
    assert "29 % of their force counts" in text
    assert "finds no equilibrium solution" in text
    # its slices are phi = 0, so m_alpha is not checked (GM-127); its FOS
    # is 1.373 on 60 slices, within 1 %, so nothing is warned of
    assert res.n_rejected_m_alpha == 0
    assert "On 60 slices" not in text
    for convention, rule, fos, fine_fos in (
            ("active", "every_slice", 1.511, 1.51),     # D&W 1.51
            ("passive", "every_slice", 1.316, 1.32),    # D&W 1.32
            ("active", "frictional", 1.508, 1.512),
            ("passive", "frictional", 1.316, 1.318)):
        g = _slide85c(convention, reverse_curvature="discard")
        r = search_critical_surface(g, method="bishop", m_alpha_rule=rule)
        c = r.critical
        assert c.FOS == pytest.approx(fos, abs=0.005)
        assert c.yc >= 30.0 - 0.02                   # centre above the crest
        fine = analyze_slope(g, method="bishop", n_slices=240, xc=c.xc,
                             yc=c.yc, radius=c.radius).FOS
        assert fine == pytest.approx(fine_fos, rel=0.005)
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
    found). 1.564 (1.584 before GM-134)."""
    from slope_stability.search_supplements import search_toe_circles
    out = search_toe_circles(_slide85c(), method="spencer")
    assert out["FOS"] < _FOS_MAX
    assert out["FOS"] == pytest.approx(1.564, abs=0.01)
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


# ---------------------------------------------------------------------------
# GM-81: warnings on a single surface; no silent fallback
# ---------------------------------------------------------------------------

# The reverse-curvature toe circle the default #85 Bishop search finds
_TOE_85 = dict(xc=15.012, yc=24.112, radius=14.112)


@pytest.mark.parametrize("method, label", [("spencer", "Spencer"),
                                           ("gle", "GLE")])
def test_no_solution_with_supports_reports_no_fos(method, label):
    """Anchored #85 toe circle: Spencer and GLE find no equilibrium solution.
    Before GM-81 analyze_slope returned the legacy solver's 0.891 (Spencer),
    the slope WITHOUT its 9,000 lb/ft support, with only a Python warning.
    Now FOS is None, the result says so, and nothing is warned in Python."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no Python warning at all
        r = analyze_slope(_slide85c(), method=method, **_TOE_85)
    assert r.FOS is None and r.is_stable is None
    assert r.method == label
    assert r.method_solved is False and r.method_used is None
    text = " ".join(r.warnings)
    assert "finds no equilibrium solution on this surface" in text
    assert "No factor of safety is reported" in text
    assert "ignores the 1 support(s)" in text
    d = r.to_dict()
    assert d["FOS"] is None and d["is_stable"] is None
    assert d["method_solved"] is False and d["method_used"] is None
    assert d["warnings"] == r.warnings
    assert "no equilibrium solution" in r.summary()


def test_no_solution_in_compare_methods_is_left_out():
    """compare_methods on the same circle: Bishop solves (1.372); Spencer and
    Morgenstern-Price have no solution, so their fields stay None (before:
    the unreinforced legacy values), named in methods_without_solution."""
    r = analyze_slope(_slide85c(), method="bishop", compare_methods=True,
                      **_TOE_85)
    assert r.FOS == pytest.approx(1.372, abs=0.002) and r.method_solved
    assert r.FOS_spencer is None and r.FOS_morgenstern_price is None
    assert r.methods_without_solution == ["Spencer", "Morgenstern-Price"]
    assert sum("(method comparison) finds no equilibrium solution" in w
               for w in r.warnings) == 2


def test_no_supports_labels_the_legacy_value():
    """Without supports the legacy approximate solver's value stands in, and
    is labelled: on #85's clay (phi = 0) the GLE finds no lambda where the
    moment and force factors of safety agree on Slide2's circle; the legacy
    solver gives theta = 0 (-0.01 deg since GM-134, each slice's base its
    chord), Bishop's value (as every moment-equilibrium method gives on a
    phi = 0 circle)."""
    from slope_stability.methods import spencer_fos
    from slope_stability.slices import build_slices
    g = _slide85(anchor=False)
    r = analyze_slope(g, method="spencer", **_SLIDE2_85_CIRCLE)
    slip = CircularSlipSurface(**_SLIDE2_85_CIRCLE)
    before, _ = spencer_fos(build_slices(g, slip, 30), slip)
    assert r.FOS == before                          # the same value as before
    assert r.FOS == pytest.approx(
        analyze_slope(g, method="bishop", **_SLIDE2_85_CIRCLE).FOS, abs=1e-4)
    assert r.method_solved is False
    assert r.method_used == "legacy approximate Spencer solver"
    assert r.methods_without_solution == ["Spencer"]
    (w,) = r.warnings
    assert "finds no equilibrium solution on this surface" in w
    assert f"FOS = {r.FOS:.3f} is the legacy approximate solver's value" in w
    assert re.search(r"theta = -?0\.0[01] deg", w)
    assert r.to_dict()["method_used"] == "legacy approximate Spencer solver"


def test_single_surface_reverse_curvature_and_short_arm():
    """The warnings a search gives its governing circle, now on the circle
    itself: Bishop on the #85 toe circle (1.372) is closed by a 5.89 ft dry
    crack, and counts 29 % of the support; Spencer has no solution there.
    reverse_curvature='discard' does not stop analyze_slope evaluating it."""
    for geom_kw in ({}, {"reverse_curvature": "discard"}):
        r = analyze_slope(_slide85c(**geom_kw), method="bishop", **_TOE_85)
        assert r.FOS == pytest.approx(1.372, abs=0.002)
        assert r.method_solved and r.method_used == "" and r.produced_by == \
            "Bishop"
        rc, arm = r.warnings
        assert "reverse-curvature" in rc and "5.89 m deep" in rc
        assert "a circle given to analyze_slope is evaluated either way" in rc
        assert "crossing this circle" in arm
        assert "29 % of their force counts" in arm
        assert "finds no equilibrium solution" in arm
    assert CircularSlipSurface(**_TOE_85).reverse_curvature_crack(
        _slide85c())["depth"] == pytest.approx(5.888, abs=0.01)


def test_short_arm_gives_spencer_when_it_solves():
    """A horizontal anchor crossing a Bishop circle near its centre's level:
    Bishop counts 21 % of it, and the warning gives Spencer's FOS on the
    same circle, as analyze_slope(method='spencer') finds it."""
    g, _ = _phi_slope()
    g.anchors = [Anchor(x_head=34.0, z_head=17.0, length=25.0,
                        T_allow=150.0, inclination=0.0)]
    circle = dict(xc=26.0, yc=20.0, radius=14.0)
    r = analyze_slope(g, method="bishop", **circle)
    spencer = analyze_slope(g, method="spencer", **circle)
    assert spencer.method_solved
    (w,) = r.warnings
    assert "21 % of their force counts" in w
    assert f"gives {spencer.FOS:.3f} on this circle" in w


@pytest.mark.parametrize("method", ["bishop", "spencer", "gle", "fellenius",
                                    "janbu"])
def test_normal_case_unchanged(method):
    """A circle with no supports, not reverse-curvature, where the method
    solves: no warnings, method_solved, the same FOS as the method's own
    function, and no new key in to_dict but method_solved / method_used."""
    from slope_stability import methods as M
    from slope_stability.gle import janbu_fos
    from slope_stability.slices import build_slices
    g = _acads()
    slip = CircularSlipSurface(40.0, 45.0, 21.0)
    r = analyze_slope(g, method=method, slip_surface=slip)
    sl = build_slices(g, slip, 30)
    own = {"bishop": lambda: M.bishop_fos(sl, slip),
           "spencer": lambda: M.spencer_fos(sl, slip)[0],
           "gle": lambda: M.morgenstern_price_fos(sl, slip)[0],
           "fellenius": lambda: M.fellenius_fos(sl, slip),
           "janbu": lambda: janbu_fos(sl, slip, tol=1e-5)[0]}[method]()
    assert r.FOS == own
    assert r.method_solved and r.warnings == []
    assert r.methods_without_solution == []
    d = r.to_dict()
    assert "warnings" not in d and "methods_without_solution" not in d
    assert d["method_solved"] is True and d["method_used"] == r.method


def test_search_labels_a_legacy_critical_circle():
    """A rigorous search scores a circle no support crosses with the legacy
    solver when the GLE has no solution (unchanged); the governing circle
    now says so. #85 without its support, Spencer: every circle of the
    main search (since GM-134 the best toe circle, centred left of the toe,
    has a Spencer solution, 0.910, so the families are left off here)."""
    res = search_critical_surface(_slide85(anchor=False), method="spencer",
                                  nx=4, ny=4, toe_circles=False)
    c = res.critical
    assert c.method_solved is False
    assert c.method_used == "legacy approximate Spencer solver"
    assert any("finds no equilibrium solution on the critical circle" in w
               and f"{c.FOS:.3f}" in w for w in res.warnings)
    assert res.to_dict()["critical"]["method_solved"] is False

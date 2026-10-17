"""GM-127: Slide2's m_alpha < 0.2 check on every scored slip surface.

m_alpha = cos(a) (1 + tan(a) tan(phi) / F) is the denominator of the normal
force on a slice base in Bishop's simplified method (Bishop 1955, Eq. 12),
in Janbu's simplified method and in the GLE behind Spencer,
Morgenstern-Price and GLE (Fredlund & Krahn 1977). Slide2 reports a surface
with m_alpha < 0.2 on any slice on the final iteration as invalid (its error
-112; "Check malpha < 0.2", after Whitman & Bailey 1967: below 0.2 the FOS
may be incorrect or misleading). Every search here leaves such a surface
out and counts it (``n_rejected_m_alpha``, ``choices``); a surface given to
`analyze_slope` is analysed and a warning says so. Slices with phi' = 0 are
not checked by default (``m_alpha_rule='frictional'``): there m_alpha =
cos(a) does not change the FOS, and leaving such a surface out only raised
the minimum (a vertical cut in undrained clay, 0.800 -> 0.933 against
Taylor's 0.851). ``m_alpha_rule='every_slice'`` checks every slice, as
Slide2 does.

The w11 surface (wave w11, F61): a differential-evolution surface 32 m deep
in sandstone under a 10 m cut, FOS 1.57 with m_alpha -0.37 on 10 of its 40
slices, reported as critical before.
"""
import math
import types

import pytest

from slope_stability.analysis import analyze_slope, search_critical_surface
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.methods import M_ALPHA_MIN, m_alpha_min, m_alpha_values
from slope_stability.probabilistic import fosm_fos
from slope_stability.rapid_drawdown import rapid_drawdown_fos
from slope_stability.results import SearchResult
from slope_stability.search import _FOS_MAX, _compute_fos, search_de
from slope_stability.search_supplements import (
    finer_slices_warning, m_alpha_confirmation, m_alpha_none_warning,
)
from slope_stability.slices import build_slices
from slope_stability.slip_surface import (
    CircularSlipSurface, PolylineSlipSurface,
)

FT, PCF, PSF = 0.3048, 0.157087464, 0.04788026

_W11_PTS = [(28.578, 8.578), (33.145, 6.709), (37.712, -22.17),
            (42.279, -20.072), (46.846, 10.0)]


def _w11():
    """A 10 m cut at 1H:1V: clay (c' 3, phi' 22) over sandstone."""
    return SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (30, 10), (60, 10)],
        soil_layers=[
            SlopeSoilLayer("clay", top_elevation=10, bottom_elevation=6.5,
                           gamma=18.5, phi=22, c_prime=3),
            SlopeSoilLayer("sandstone", top_elevation=6.5,
                           bottom_elevation=-30, gamma=23, phi=40,
                           c_prime=200)])


def _duncan_ex4():
    """Duncan Example 4: 3:1, 20 ft of sand (phi 38) over clay (su 500 psf)."""
    H = 20 * FT
    crest = 20 * FT
    toe = crest + 3 * H
    return SlopeGeometry(
        surface_points=[(0, H), (crest, H), (toe, 0), (toe + 30 * FT, 0)],
        soil_layers=[
            SlopeSoilLayer("Sand", top_elevation=H, bottom_elevation=0.0,
                           gamma=120 * PCF, phi=38.0, c_prime=0.0),
            SlopeSoilLayer("Clay", top_elevation=0.0, bottom_elevation=-2 * H,
                           gamma=115 * PCF, gamma_sat=115 * PCF + 1.0,
                           cu=500 * PSF, analysis_mode="undrained")])


def _vertical_cut():
    """A vertical 5 m cut in undrained clay, cu 20 kPa, 18 kN/m3."""
    return SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (20.001, 5.0), (45, 5.0)],
        soil_layers=[SlopeSoilLayer("clay", 5.0, -10.0, 18.0, cu=20.0,
                                    analysis_mode="undrained")])


def _weak_over_rock():
    return SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (30, 10), (60, 10)],
        soil_layers=[
            SlopeSoilLayer("weak", top_elevation=10, bottom_elevation=6,
                           gamma=18, phi=26, c_prime=2),
            SlopeSoilLayer("rock", top_elevation=6, bottom_elevation=-10,
                           gamma=21, phi=40, c_prime=150)])


# ---------------------------------------------------------------------------
# m_alpha itself
# ---------------------------------------------------------------------------

def test_m_alpha_is_cos_a_for_phi_zero_and_does_not_turn_with_the_slope():
    """phi = 0: m_alpha = cos(a) whatever F. A slope falling the other way
    gives the same values (the base angle is taken positive in the
    direction of sliding, as each method takes it)."""
    g = _duncan_ex4()
    slip = CircularSlipSurface(16.0, 12.0, 14.0)
    sl = build_slices(g, slip, 30)
    clay = [s for s in sl if s.phi == 0.0]
    vals = m_alpha_values(clay, 1.3, slip, "gle")
    assert vals == pytest.approx([math.cos(s.alpha) for s in clay])
    mirrored = SlopeGeometry(
        surface_points=[(-x, z) for x, z in reversed(g.surface_points)],
        soil_layers=g.soil_layers)
    mslip = CircularSlipSurface(-16.0, 12.0, 14.0)
    msl = build_slices(mirrored, mslip, 30)
    for which in ("bishop", "gle"):
        for rule in ("frictional", "every_slice"):
            assert m_alpha_min(msl, 1.3, mslip, which, rule) == \
                pytest.approx(m_alpha_min(sl, 1.3, slip, which, rule),
                              abs=1e-9)
    assert M_ALPHA_MIN == 0.2


def test_phi_zero_slices_are_not_checked_by_default():
    """The least m_alpha leaves out phi' = 0 slices unless every slice is
    asked for; with none left, 1.0."""
    g = _duncan_ex4()
    slip = CircularSlipSurface(16.0, 12.0, 14.0)
    sl = build_slices(g, slip, 30)
    sand = [s for s in sl if s.phi > 0]
    assert 0 < len(sand) < len(sl)
    assert m_alpha_min(sl, 1.3, slip, "gle") == pytest.approx(
        min(m_alpha_values(sand, 1.3, slip, "gle")))
    assert m_alpha_min(sl, 1.3, slip, "gle", "every_slice") == pytest.approx(
        min(m_alpha_values(sl, 1.3, slip, "gle")))
    clay = [s for s in sl if s.phi == 0.0]
    assert m_alpha_min(clay, 1.3, slip, "gle") == 1.0


# ---------------------------------------------------------------------------
# phi' = 0: a vertical cut in undrained clay, against Taylor
# ---------------------------------------------------------------------------

def test_vertical_cut_in_undrained_clay_against_taylor():
    """Taylor's stability number for a vertical face, 3.83, gives F = 3.83
    x 20 / (18 x 5) = 0.851. The default search finds 0.800 on a toe
    circle closed by a crack, whose steep last slice has m_alpha = cos(a)
    0.18 (phi = 0): not checked, so the circle stays. Checking every
    slice, as Slide2 does, leaves it out and reads 0.933, 9.6 % above
    Taylor on the unsafe side; the choice says which was used."""
    taylor = 3.83 * 20.0 / (18.0 * 5.0)
    res = search_critical_surface(_vertical_cut(), method="bishop")
    assert res.critical.FOS == pytest.approx(0.800, abs=0.005)
    assert res.critical.FOS <= taylor
    assert res.n_rejected_m_alpha == 0
    (ch,) = [c for c in res.choices
             if c["what"] == "slip surfaces left out for m_alpha"]
    assert ch["how"] == "default" and "phi' > 0" in ch["chosen"]
    assert any("every_slice" in a for a in ch["alternatives"])
    every = search_critical_surface(_vertical_cut(), method="bishop",
                                    m_alpha_rule="every_slice")
    assert every.critical.FOS == pytest.approx(0.933, abs=0.005)
    assert every.n_rejected_m_alpha > 0
    (ch,) = [c for c in every.choices
             if c["what"] == "slip surfaces left out for m_alpha"]
    assert ch["how"] == "given" and "any slice" in ch["chosen"]
    c = res.critical
    one = analyze_slope(_vertical_cut(), method="bishop", xc=c.xc, yc=c.yc,
                        radius=c.radius)
    assert not any("m_alpha" in w for w in one.warnings)
    one = analyze_slope(_vertical_cut(), method="bishop", xc=c.xc, yc=c.yc,
                        radius=c.radius, m_alpha_rule="every_slice")
    assert any("is 0.18 on the slice" in w for w in one.warnings)
    with pytest.raises(ValueError, match="m_alpha_rule"):
        search_critical_surface(_vertical_cut(), m_alpha_rule="all")


# ---------------------------------------------------------------------------
# a surface with negative m_alpha: left out of a search, warned on alone
# ---------------------------------------------------------------------------

def test_w11_surface_is_left_out_of_a_search():
    """`_compute_fos`, every search's scorer, leaves the w11 surface out;
    with the check off (a fixed surface) it scores 1.57."""
    geom = _w11()
    slip = PolylineSlipSurface(points=_W11_PTS)
    info = {}
    assert _compute_fos(geom, slip, "spencer", 40, info=info) == _FOS_MAX
    assert info["m_alpha_min"] == pytest.approx(-0.372, abs=0.005)
    assert _compute_fos(geom, slip, "spencer", 40,
                        m_alpha_check=False) == pytest.approx(1.570,
                                                              abs=0.005)


def test_w11_surface_given_to_analyze_slope_is_analysed_with_a_warning():
    """A single surface is analysed and said to be one a search leaves out:
    where m_alpha is lowest, on how many slices, and why it matters."""
    r = analyze_slope(_w11(), slip_surface=PolylineSlipSurface(
        points=_W11_PTS), method="spencer", n_slices=40)
    assert r.FOS == pytest.approx(1.570, abs=0.005)
    (w,) = [w for w in r.warnings if "m_alpha" in w]
    assert "-0.37" in w and "10 of 40 slices" in w
    assert "against the sliding" in w and "passive zone" in w
    assert "Whitman & Bailey 1967" in w and "error -112" in w
    assert "'Check malpha < 0.2' setting on" in w
    assert "given to analyze_slope, is analysed" in w
    assert "Rankine passive angle, 45 deg - phi'/2" in w


def test_a_surface_with_m_alpha_above_the_limit_has_no_warning():
    g = _duncan_ex4()
    r = analyze_slope(g, method="bishop", n_slices=30, xc=16.764,
                      yc=9.521, radius=17.108)
    assert not any("m_alpha" in w for w in r.warnings)


def test_ordinary_method_of_slices_has_no_m_alpha():
    r = analyze_slope(_w11(), slip_surface=PolylineSlipSurface(
        points=_W11_PTS), method="fellenius", n_slices=40)
    assert not any("m_alpha" in w for w in r.warnings)
    assert _compute_fos(_w11(), PolylineSlipSurface(points=_W11_PTS),
                        "fellenius", 40) < _FOS_MAX


def test_de_search_leaves_out_and_counts_negative_m_alpha_surfaces():
    """Differential evolution on the weak layer over rock, its windows on
    the face: the surfaces diving into the rock have m_alpha < 0.2 and are
    left out and counted; any surface reported has m_alpha >= 0.2."""
    geom = _weak_over_rock()
    res = search_de(geom, x_entry_range=(25.5, 28.0),
                    x_exit_range=(29.0, 40.0), n_points=7, method="spencer",
                    seed=1, maxiter=4, popsize=6, n_seed_trials=40)
    assert res.n_rejected_m_alpha > 0
    assert res.to_dict()["n_rejected"]["m_alpha"] == res.n_rejected_m_alpha
    c = res.critical
    if c is not None:
        slip = PolylineSlipSurface(points=list(c.slip_points))
        assert m_alpha_min(build_slices(geom, slip, 30), c.FOS, slip,
                           "gle") >= M_ALPHA_MIN


def test_de_reports_the_seeding_search_when_it_has_no_surface(monkeypatch):
    """When the evolution itself ends on no admissible surface, the best
    surface of the random search that seeded it is reported, and said to
    be (not only when the time budget cut it short)."""
    import numpy as np
    import scipy.optimize
    from scipy.optimize import OptimizeResult

    def nothing(fun, bounds, **kw):
        return OptimizeResult(x=np.array([lo for lo, _ in bounds]),
                              fun=_FOS_MAX, nfev=0)

    monkeypatch.setattr(scipy.optimize, "differential_evolution", nothing)
    acads = SlopeGeometry(
        surface_points=[(20.0, 25.0), (30.0, 25.0), (50.0, 35.0),
                        (70.0, 35.0)],
        soil_layers=[SlopeSoilLayer("soil", top_elevation=35.0,
                                    bottom_elevation=20.0, gamma=20.0,
                                    phi=19.6, c_prime=3.0)])
    res = search_de(acads, x_entry_range=(20, 32), x_exit_range=(46, 68),
                    n_points=5, method="spencer", seed=1, maxiter=2,
                    popsize=4, n_seed_trials=30)
    assert res.critical is not None and res.critical.FOS < _FOS_MAX
    (w,) = res.warnings
    assert "random search that seeded it (30 trials)" in w
    assert f"{res.critical.FOS:.3f}" in w


@pytest.mark.slow
def test_w11_de_search_runs_its_whole_budget():
    """The w11 DE search (windows 20-30 / 28-60, 5 points, seed 3): surfaces
    left out for m_alpha are scored by how far m_alpha falls below 0.2,
    not a flat 999.9, so the evolution runs all 31 generations (2,325
    surfaces) instead of stopping after about 8 (600). It reports 6.85 on
    a surface through the sandstone with m_alpha >= 0.2, where it reported
    1.570 at 32 m depth with m_alpha -0.37 before GM-127."""
    geom = _w11()
    res = search_critical_surface(
        geom, surface_type="noncircular_de", method="spencer",
        x_entry_range=(20, 30), x_exit_range=(28, 60), n_slices=40,
        n_points=5, seed=3)
    assert res.n_surfaces_evaluated == 2325
    assert res.n_rejected_m_alpha > 0
    c = res.critical
    assert c.FOS == pytest.approx(6.847, abs=0.01)
    slip = PolylineSlipSurface(points=list(c.slip_points))
    assert m_alpha_min(build_slices(geom, slip, 40), c.FOS, slip,
                       "gle") >= M_ALPHA_MIN


# ---------------------------------------------------------------------------
# Duncan Example 4 with Janbu: the count and the choice
# ---------------------------------------------------------------------------

def test_duncan_ex4_janbu_search_on_30_slices():
    """With Janbu the default search read Duncan Ex 4 at 1.076 on 30 slices
    on a circle whose steepest slice, in the sand, had m_alpha 0.007,
    against 1.213 on 120 slices. That circle is left out now: 1.199 on 30
    slices, 1.238 on 60 (+3.3 %, which a warning says). The count and the
    two slicings are in the result."""
    res = search_critical_surface(_duncan_ex4(), method="janbu", n_slices=30)
    assert res.critical.FOS == pytest.approx(1.199, abs=0.01)
    assert res.n_rejected_m_alpha > 0
    d = res.to_dict()
    assert d["n_rejected"]["m_alpha"] == res.n_rejected_m_alpha
    (ch,) = [c for c in res.choices
             if c["what"] == "slip surfaces left out for m_alpha"]
    assert ch["chosen"].startswith(f"{res.n_rejected_m_alpha} left out")
    assert "error -112" in ch["basis"]
    assert "on 30 slices" in ch["basis"] and "on 60 slices" in ch["basis"]
    assert "phi' = 0 are not checked" in ch["basis"]
    assert "Left out, m_alpha < 0.2" in res.summary()
    assert any(w.startswith("On 60 slices, twice the search's 30") and
               "1.238" in w for w in res.warnings)


def test_fellenius_search_says_nothing_is_left_out():
    res = search_critical_surface(_duncan_ex4(), method="fellenius",
                                  n_slices=30, toe_circles=False,
                                  infinite_slope=False, refine=False)
    assert res.n_rejected_m_alpha == 0
    (ch,) = [c for c in res.choices
             if c["what"] == "slip surfaces left out for m_alpha"]
    assert ch["chosen"] == "none"
    assert "no m_alpha" in ch["basis"]


def test_finer_slices_warning_and_none_found_warning():
    """On twice the slices: a FOS that moves more than 1 % is warned on,
    with m_alpha when it falls below 0.2 there; a method with no solution
    there is warned on; a slicing that cannot be built is not. A search
    that found nothing says what it left out."""
    base = {"n_slices": 30, "n_slices_fine": 60, "fos": 1.10,
            "m_alpha_min": 0.21}
    w = finer_slices_warning(dict(base, fos_fine=1.12, m_alpha_min_fine=0.15))
    assert w.startswith("On 60 slices, twice the search's 30")
    assert "1.120 against 1.100 (+1.8 %)" in w and "0.15" in w
    assert "Rankine passive angle" in w
    w = finer_slices_warning(dict(base, fos_fine=1.12, m_alpha_min_fine=0.4))
    assert "+1.8 %" in w and "more slices" in w and "m_alpha" not in w
    assert finer_slices_warning(dict(base, fos_fine=1.105,
                                     m_alpha_min_fine=0.15)) is None
    w = finer_slices_warning(dict(base, fos_fine=None,
                                  refused_fine="no_solution"))
    assert "finds no solution" in w
    assert finer_slices_warning(dict(base, fos_fine=None,
                                     refused_fine="geometry")) is None
    res = SearchResult(n_rejected_m_alpha=8, n_rejected_nonconverged=5)
    w = m_alpha_none_warning(res)
    assert w.startswith("No surface is reported") and "8 were left out" in w
    assert "5 had no converged solution" in w
    assert "surface_type 'circular' or 'entry_exit'" in w
    assert m_alpha_none_warning(SearchResult()) is None


def test_twice_the_slices_keeps_the_reason():
    """Slide2 #85 with its support: on the circle a Spencer search finds,
    Spencer has no solution on 60 slices, which is warned on. Taylor's
    deep circle on 60 slices dips 1 slice under the base: the choice says
    it is not admissible there, and nothing is warned of."""
    from slope_stability.reinforcement import Anchor
    s85 = SlopeGeometry(
        surface_points=[(15, 10), (25, 30), (57, 30)],
        soil_layers=[SlopeSoilLayer("clay", 30, 10, 98.0, cu=350.0,
                                    analysis_mode="undrained")],
        anchors=[Anchor(x_head=20.0, z_head=20.0, length=37.0,
                        T_allow=9000.0, inclination=0.0)])
    circle = types.SimpleNamespace(is_circular=True, xc=15.0, yc=32.0,
                                   radius=22.0)
    conf = m_alpha_confirmation(s85, circle, "spencer", 30)
    assert conf["fos"] == pytest.approx(1.542, abs=0.002)
    assert conf["fos_fine"] is None and conf["refused_fine"] == "no_solution"
    assert "finds no solution" in finer_slices_warning(conf)
    H = 40 * FT
    taylor = SlopeGeometry(
        surface_points=[(0, H), (4 * H, H), (6 * H, 0), (10 * H, 0)],
        soil_layers=[SlopeSoilLayer("clay", H, -H, 20.0, cu=0.12 * 20.0 * H,
                                    analysis_mode="undrained")])
    res = search_critical_surface(taylor, method="bishop")
    (ch,) = [c for c in res.choices
             if c["what"] == "slip surfaces left out for m_alpha"]
    assert "on 60 slices it is not admissible" in ch["basis"]
    assert "below the bottom of the deepest soil layer" in ch["basis"]
    assert not any("On 60 slices" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# fixed-surface analyses keep every surface
# ---------------------------------------------------------------------------

def test_fixed_surface_fosm_keeps_a_low_m_alpha_surface():
    """FOSM on one given surface is not a search: no realisation is left out
    for its m_alpha."""
    res = fosm_fos(_w11(), {"phi:sandstone": {"mean": 40.0, "cov": 0.05}},
                   slip_surface=PolylineSlipSurface(points=_W11_PTS),
                   method="spencer", n_slices=40)
    assert res.fos_mlv == pytest.approx(1.570, abs=0.005)
    assert res.fos_mlv < _FOS_MAX


def test_rapid_drawdown_reports_m_alpha_min():
    """The drawn-down stage's least m_alpha, each slice at the strength it
    was solved with (the drawdown search leaves out a surface below 0.2)."""
    g = SlopeGeometry(
        surface_points=[(0.0, 0.0), (305 * FT, 110 * FT),
                        (380 * FT, 110 * FT)],
        soil_layers=[SlopeSoilLayer(
            name="fill", top_elevation=110 * FT, bottom_elevation=-5.0,
            gamma=135 * PCF, phi=30.0, c_prime=0.0, R_c=1200 * PSF,
            R_phi=16.0)])
    r = rapid_drawdown_fos(g, 110 * FT, 24 * FT, xc=169.5 * FT,
                           yc=210 * FT, radius=210 * FT, n_slices=50)
    assert M_ALPHA_MIN <= r.m_alpha_min <= 1.0
    assert not any("m_alpha" in w for w in r.warnings)

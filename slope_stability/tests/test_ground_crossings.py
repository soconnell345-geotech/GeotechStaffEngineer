"""One sliding mass per trial surface (GM-122) and entry / exit windows given
the wrong way round (GM-124).

GM-122. ``find_entry_exit`` took a circle from its first crossing of the
ground to its last. A circle that leaves the ground and comes back into it
was then scored as ONE mass: the stretch in the air between its crossings
became slices (dropped one by one as having no height), so the slice count,
the reported entry and the layers along the surface were wrong. Measured on
the w10 geometry below (a 10 m 1H:1V cut, 3.5 m of weak clay over rock):

- the circle (10, 39.5) R 40 crosses the ground at x 3.70, 16.30, 21.06 and
  37.01: a sliver under the level ground in front of the toe, the air, then
  the slope. Scored as one mass: Bishop 13.55 on 34 of 40 slices. The slope
  mass on its own: Spencer 7.40.
- the default search's critical circle (0, 50) R 50.92 (Bishop 1.143 on 9
  of 40 slices) is below the ground at the profile's left end, rises out of
  it at x 9.64 and comes back in at 26.55 on the face. The answer named an
  entry at x 9.64 and sandstone along a surface that never reaches it.

A slip surface is the part of the trial surface below the ground surface
between where it enters the soil and where it leaves it: the base of one
sliding mass whose equilibrium the method of slices takes (Duncan, Wright &
Brandon 2014, Ch. 6). Such a surface is refused (``GroundCrossingError``)
and the searches count the refusals (``n_rejected_ground_crossings``).

Refusing them showed what they had hidden: the default search scored no
circle that comes out on the face. Its automatic windows meet at the middle
of the face, so on w10 every circle it scored entered below z = 5, and the
answer went from 1.14 (across the air) to 6.31 (a toe circle through the
sandstone).
The circles passing above the toe (FHWA NHI-06-088 p. 6-38) are now a
supplementary family of their own (``search_supplements``): 0.936, from the
face where the clay's base comes out.
"""

import re

import pytest

from slope_stability.analysis import analyze_slope, search_critical_surface
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.rapid_drawdown import search_rapid_drawdown
from slope_stability.search import (
    _ordered_windows, grid_search, search_entry_exit, search_weak_layer_biased,
)
from slope_stability.methods import m_alpha_min
from slope_stability.search_supplements import search_toe_circles
from slope_stability.slices import build_slices
from slope_stability.slip_surface import (
    _TOUCH_FRACTION, CircularSlipSurface, GroundCrossingError,
    PolylineSlipSurface,
)


def _w10():
    """F61 (wave w10): ground (0, 0), (20, 0), (30, 10), (60, 10); the top
    3.5 m a weak residual clay over sandstone, no water."""
    return SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (30, 10), (60, 10)],
        soil_layers=[
            SlopeSoilLayer("Residual clay", 10, 6.5, 18.5, phi=22, c_prime=3),
            SlopeSoilLayer("Sandstone", 6.5, -40, 23, phi=40, c_prime=200),
        ])


def _clay85():
    """Slide2 #85's clay face without its support: a 20 m face at 2V:1H of
    undrained clay, su 350 kPa."""
    return SlopeGeometry(
        surface_points=[(15, 10), (25, 30), (57, 30)],
        soil_layers=[SlopeSoilLayer("clay", 30, 10, 98.0, cu=350.0,
                                    analysis_mode="undrained")])


# ---------------------------------------------------------------------------
# GM-122: the rule on one surface
# ---------------------------------------------------------------------------

def test_a_circle_crossing_the_ground_four_times_is_refused():
    slip = CircularSlipSurface(10.0, 39.5, 40.0)
    with pytest.raises(GroundCrossingError) as err:
        slip.find_entry_exit(_w10())
    (a0, b0), (a1, b1) = err.value.stretches
    assert (a0, b0) == pytest.approx((3.695, 16.305), abs=0.01)
    assert (a1, b1) == pytest.approx((21.059, 37.014), abs=0.01)
    assert "2 separate stretches" in str(err.value)
    assert "sliding mass" in str(err.value)
    # a ValueError, so every caller that refuses a bad surface refuses it
    assert isinstance(err.value, ValueError)
    with pytest.raises(ValueError):
        build_slices(_w10(), slip, 40)
    with pytest.raises(ValueError):
        analyze_slope(_w10(), slip_surface=slip, method="bishop")


def test_a_circle_below_the_ground_at_the_profile_end_and_on_the_face_is_refused():
    """F61 [9]: the default search's critical circle before GM-122."""
    slip = CircularSlipSurface(0.0, 50.0, 50.92)
    with pytest.raises(GroundCrossingError) as err:
        slip.find_entry_exit(_w10())
    (a0, b0), (a1, b1) = err.value.stretches
    assert a0 == 0.0 and b0 == pytest.approx(9.636, abs=0.01)
    assert (a1, b1) == pytest.approx((26.549, 31.509), abs=0.01)
    assert "the end of the profile" in str(err.value)


def test_a_circle_bounding_one_mass_is_scored_as_before():
    """The w10 critical circle in the clay (entry / exit search): one stretch
    below the ground, every slice kept."""
    g = _w10()
    slip = CircularSlipSurface(26.06, 12.8, 5.87)
    x_en, x_ex = slip.find_entry_exit(g)
    assert 26.5 < x_en < 27.5 and 30.5 < x_ex < 31.5
    assert len(build_slices(g, slip, 40)) == 40
    r = analyze_slope(g, slip_surface=slip, method="bishop", n_slices=40)
    assert r.FOS == pytest.approx(1.014, abs=0.01)


def test_a_stretch_that_only_touches_the_ground_is_left_out():
    """V-042's pinned circle passes 1.3 cm over the toe and dips 0.2 mm under
    the level ground in front of it (x 4.30 to 4.44). That touch carries no
    mass: the surface is the face mass, from x 5.01 to 18.22. Taken from the
    first crossing to the last it lost the slice at the toe (28 of 30 slices
    kept): Bishop 1.325 against 1.336 now (Hassan & Wolff 1.334, Slide 1.340).
    """
    g = SlopeGeometry(
        surface_points=[(0, 5), (5, 5), (15, 15), (20, 15)],
        soil_layers=[SlopeSoilLayer("mat1", 15, 0.0, 18.0, phi=30.0,
                                    c_prime=18.0, ru=0.2)])
    slip = CircularSlipSurface(4.3736565339749305, 19.588888034671577,
                               14.589059771501798)
    x_en, x_ex = slip.find_entry_exit(g)
    assert x_en == pytest.approx(5.014, abs=0.01)
    assert x_ex == pytest.approx(18.222, abs=0.01)
    assert len(build_slices(g, slip, 30)) == 30
    r = analyze_slope(g, slip_surface=slip, method="bishop", n_slices=30)
    assert r.FOS == pytest.approx(1.336, abs=0.002)


@pytest.mark.parametrize("fraction, one_mass", [(0.995, True), (1.005, False)])
def test_a_touch_just_under_the_threshold_is_left_out_and_just_over_refused(
        fraction, one_mass):
    """A polyline 2 m deep under the crest that also dips under the level
    ground in front of the toe, with the slope between them in the air. A
    dip just under 1 % of the 2 m (``_TOUCH_FRACTION``) is a touch and is
    left out; just over it is a second mass and the surface is refused."""
    g = _w10()
    dip = _TOUCH_FRACTION * 2.0 * fraction
    slip = PolylineSlipSurface([(2.0, 0.0), (3.0, -dip), (4.0, 0.0),
                                (25.0, 12.0), (40.0, 10.0), (45.0, 8.0),
                                (50.0, 10.0)])
    if one_mass:
        assert slip.find_entry_exit(g) == pytest.approx((40.0, 50.0))
        assert len(build_slices(g, slip, 20)) == 20
    else:
        with pytest.raises(GroundCrossingError) as err:
            slip.find_entry_exit(g)
        (a0, b0), (a1, b1) = err.value.stretches
        assert 2.0 <= a0 < b0 <= 4.0 and 40.0 <= a1 < b1 <= 50.0


def test_a_polyline_that_rises_into_the_air_and_comes_back_is_refused():
    """F61 [10]: the weak-layer search's governing polyline dipped under the
    level ground in front of the toe, rose over the toe into the air and
    came back into the face."""
    pts = [(2.567, 0.0), (16.022, -0.3), (29.478, 8.432), (42.933, 9.585),
           (56.388, 10.0)]
    with pytest.raises(GroundCrossingError) as err:
        PolylineSlipSurface(pts).find_entry_exit(_w10())
    assert len(err.value.stretches) == 2


def test_a_polyline_with_its_ends_on_the_ground_is_one_mass():
    g = _w10()
    pts = [(24.0, 4.0), (27.0, 5.0), (31.0, 7.5), (36.0, 10.0)]
    x_en, x_ex = PolylineSlipSurface(pts).find_entry_exit(g)
    assert (x_en, x_ex) == (24.0, 36.0)
    assert len(build_slices(g, PolylineSlipSurface(pts), 20)) == 20


# ---------------------------------------------------------------------------
# GM-122: the searches refuse and count
# ---------------------------------------------------------------------------

def test_the_default_search_reports_one_mass_and_counts_refusals():
    g = _w10()
    r = search_critical_surface(g, method="bishop", n_slices=40)
    assert r.n_rejected_ground_crossings > 0
    assert r.to_dict()["n_rejected"]["ground_crossings"] == \
        r.n_rejected_ground_crossings
    assert "not one sliding mass" in r.summary()
    c = r.critical
    slip = CircularSlipSurface(c.xc, c.yc, c.radius)
    slip.find_entry_exit(g)                    # one stretch: no error
    assert len(build_slices(g, slip, 40)) == 40
    # the reported ends are the mass's own: no end out in the level ground
    # in front of the toe (F61 reported x 9.64 there)
    assert c.x_entry >= 19.9


def test_the_default_search_covers_circles_passing_above_the_toe():
    """The circle that governs w10 comes out on the face above the toe,
    where no circle the main search scores can (its automatic windows meet
    at the middle of the face): 0.938, its lower end where the clay's base
    comes out (z 6.5), every slice base in the clay. The main search and
    the toe circles give 6.44 through the sandstone (6.33 and 6.31 before
    GM-134, when a slice across the clay's base took one material for its
    whole base; 6.44 on 3,840 slices either way)."""
    g = _w10()
    r = search_critical_surface(g, method="bishop")
    c = r.critical
    assert r.governing_surface == "circle_above_toe"
    assert c.FOS == pytest.approx(0.938, abs=0.01)
    assert r.candidates["circular"] > 6.0
    assert r.candidates["toe_circle"] > 6.0
    assert 26.4 <= c.x_entry <= 26.7 and 30.0 <= c.x_exit <= 31.5
    slip = CircularSlipSurface(c.xc, c.yc, c.radius)
    assert {g.layer_at_point(s.x_mid, s.z_base).name
            for s in build_slices(g, slip, 30)} == {"Residual clay"}
    assert any(w.startswith("A circle passing above the toe governs")
               and re.search(r"6\.5\d m above the toe", w)
               and "which meet at the middle of the face" in w
               for w in r.warnings)
    # what was covered, in the result
    (face,) = r.search_ranges["faces_covered"]
    assert face["toe"] == [20, 0] and face["crest"] == [30, 10]
    assert face["lower_end_heights_above_toe"][0] == 0.0
    assert max(face["lower_end_heights_above_toe"]) == pytest.approx(0.9)
    fams = next(ch for ch in r.choices
                if ch["what"] == "surface families scored")
    assert "circle passing above the toe" in fams["chosen"]
    assert "faces_covered" in fams["basis"]
    # every family leaves out m_alpha < 0.2 (GM-127): said once, for all
    ma = next(ch for ch in r.choices
              if ch["what"] == "slip surfaces left out for m_alpha")
    assert "-112" in ma["basis"]
    d = r.to_dict()
    assert d["governing_surface"] == "circle_above_toe"
    assert "faces_covered" in d["search_ranges"]


def test_circles_passing_above_the_toe_are_not_degenerate():
    """The family keeps its centres above the ground at both ends (no
    reverse-curvature circle closed by a crack) and, like Slide2 (error
    -112), refuses a circle with m_alpha = cos(a) (1 + tan(a) tan(phi) / F)
    below 0.2 on a slice, where Bishop's method can give a spurious low
    FOS (since GM-127 every family does, in `_compute_fos`, by one rule).
    Duncan Ex 4 (3:1, 6.1 m of sand over clay, su 23.9 kPa): a
    reverse-curvature circle from the face with steep slices in the sand
    read 0.17 on 30 slices and 1.93 on 80 (Spencer 1.53).

    Slices with phi' = 0 are not checked by default (GM-127 review): there
    m_alpha = cos(a) does not change the FOS. Slide2 #85's steep clay face
    without its support: 0.911 on a circle whose least m_alpha, cos(a) on a
    phi = 0 slice, is 0.37; checking every slice, as Slide2 does, gives
    0.911 too. Before GM-134 the family kept a circle at 0.906 whose last
    slice, near vertical, had m_alpha 0.19, and which reads 0.947 on 960
    slices: the base of a slice at the circle's midpoint angle fell short
    of the steep arc (0.910 checking every slice)."""
    H = 6.096
    g = SlopeGeometry(
        surface_points=[(0, H), (H, H), (4 * H, 0), (4 * H + 9.144, 0)],
        soil_layers=[
            SlopeSoilLayer("Sand", H, 0.0, 18.85, phi=38.0, c_prime=0.0),
            SlopeSoilLayer("Clay", 0.0, -2 * H, 18.06, gamma_sat=19.06,
                           cu=23.94, analysis_mode="undrained"),
        ])
    r = search_critical_surface(g, nx=8, ny=8)
    assert r.candidates["circle_above_toe"] > 1.2
    # candidates are rounded to 4 places
    assert r.critical.FOS == pytest.approx(r.candidates["circular"], abs=5e-5)
    clay = _clay85()
    a = search_toe_circles(clay, method="bishop")["above_toe"]
    slip = CircularSlipSurface(a["xc"], a["yc"], a["radius"])
    slices = build_slices(clay, slip, 30)
    assert m_alpha_min(slices, a["FOS"], slip, "bishop") == 1.0
    assert m_alpha_min(slices, a["FOS"], slip, "bishop",
                       "every_slice") == pytest.approx(0.37, abs=0.01)
    assert slip.reverse_curvature_crack(clay) is None
    assert a["FOS"] == pytest.approx(0.911, abs=0.003)
    every = search_critical_surface(clay, method="bishop",
                                    m_alpha_rule="every_slice")
    assert every.candidates["circle_above_toe"] == pytest.approx(
        0.911, abs=0.003)


def test_janbu_screens_the_circles_passing_above_the_toe_with_bishop():
    """The family's circles can enter the face steeply, where Janbu's solve
    falls with the slice count. Slide2 #85's clay face without its support,
    default search: scanned with Janbu itself the family reported 0.697 on
    a circle Bishop and Spencer both read 1.22, Janbu 0.86 on 20 slices,
    0.70 on 30 and no solution on 40 or more. Janbu now scans the family
    with Bishop and re-scores the best three with Janbu: 1.033, 1.036 on
    120 slices (the toe circle's 1.024 governs, as before the family).
    Since GM-127 m_alpha is not checked on phi = 0 slices by default, so
    the Bishop scan ranks first circles centred at the crest's level with
    a near-vertical last slice, on which Janbu has no solution: the family
    reads 1.138 (1.142 on 120 slices), and 1.033 with
    ``m_alpha_rule='every_slice'``."""
    clay = _clay85()
    r = search_critical_surface(clay, method="janbu")
    assert r.candidates["circle_above_toe"] >= 1.0
    assert r.critical.FOS >= 1.0
    c = r.critical
    fine = analyze_slope(clay, method="janbu", n_slices=120,
                         xc=c.xc, yc=c.yc, radius=c.radius).FOS
    assert c.FOS == pytest.approx(fine, rel=0.02)
    a = search_toe_circles(clay, method="janbu")["above_toe"]
    assert a["FOS"] == pytest.approx(r.candidates["circle_above_toe"],
                                     abs=1e-3)
    fine = analyze_slope(clay, method="janbu", n_slices=120, xc=a["xc"],
                         yc=a["yc"], radius=a["radius"]).FOS
    assert a["FOS"] == pytest.approx(fine, rel=0.02)
    slip = CircularSlipSurface(a["xc"], a["yc"], a["radius"])
    assert m_alpha_min(build_slices(clay, slip, 30), a["FOS"], slip,
                       "bishop") >= 0.2


def test_janbu_and_spencer_find_the_circle_passing_above_the_toe():
    """w10 with the default search under Janbu, and the family alone under
    Spencer: the same circle from the face as Bishop's (0.936), each method's
    own FOS, close to its 120-slice value."""
    g = _w10()
    r = search_critical_surface(g, method="janbu")
    c = r.critical
    assert r.governing_surface == "circle_above_toe"
    assert c.FOS == pytest.approx(0.938, abs=0.005)
    assert 26.4 <= c.x_entry <= 26.7 and 30.0 <= c.x_exit <= 31.5
    fine = analyze_slope(g, method="janbu", n_slices=120,
                         xc=c.xc, yc=c.yc, radius=c.radius).FOS
    assert c.FOS == pytest.approx(fine, rel=0.02)
    a = search_toe_circles(g, method="spencer")["above_toe"]
    assert a["FOS"] == pytest.approx(0.935, abs=0.005)
    assert a["height"] == pytest.approx(0.65, abs=0.02)
    fine = analyze_slope(g, method="spencer", n_slices=120, xc=a["xc"],
                         yc=a["yc"], radius=a["radius"]).FOS
    assert a["FOS"] == pytest.approx(fine, rel=0.01)


def test_circles_passing_above_the_toe_keep_to_given_windows():
    """Caller-given windows bound the family like the toe circles: with the
    entry kept below the middle of the face nothing from above it is
    reported. toe_circles=False leaves both circle families out, and says
    so."""
    g = _w10()
    r = search_critical_surface(g, method="bishop",
                                x_entry_range=(0, 25), x_exit_range=(25, 60))
    assert r.governing_surface != "circle_above_toe"
    assert r.critical.x_entry <= 25.0
    off = search_critical_surface(g, method="bishop", toe_circles=False)
    assert "circle_above_toe" not in off.candidates
    assert "faces_covered" not in off.search_ranges
    fams = next(ch for ch in off.choices
                if ch["what"] == "surface families scored")
    # a circular search without them: tried by turning them on, not by "a
    # circular search" (GM-125)
    assert ("not tried: circle passing above the toe, by a search with "
            "them (toe_circles=True)") in fams["alternatives"]


def test_a_noncircular_search_counts_refused_polylines():
    r = search_weak_layer_biased(_w10(), (0, 25), (25, 60), n_trials=150,
                                 seed=1, n_slices=20)
    assert r.n_rejected_ground_crossings > 0


def test_a_search_inside_another_adds_to_its_count():
    """search_critical_surface counts its main search's refusals and its
    supplementary families' too."""
    g = _w10()
    inner = grid_search(g, (0, 60), (11, 50), 6, 6, n_slices=30,
                        x_entry_range=(0, 25), x_exit_range=(25, 60))
    outer = search_critical_surface(g, x_range=(0, 60), y_range=(11, 50),
                                    nx=6, ny=6, n_slices=30, refine=False,
                                    x_entry_range=(0, 25),
                                    x_exit_range=(25, 60))
    assert inner.n_rejected_ground_crossings > 0
    assert outer.n_rejected_ground_crossings >= inner.n_rejected_ground_crossings


# ---------------------------------------------------------------------------
# GM-124: entry / exit windows given the wrong way round
# ---------------------------------------------------------------------------

def test_ordered_windows():
    assert _ordered_windows(None, None) == (None, None, None)
    assert _ordered_windows((20, 29), (29.5, 45)) == ((20, 29), (29.5, 45),
                                                      None)
    en, ex, note = _ordered_windows((30, 50), (20, 28))
    assert (en, ex) == ((20, 28), (30, 50))
    assert "swapped" in note and "low-x end" in note
    en, ex, note = _ordered_windows((29, 20), (30, 50))
    assert en == (20, 29) and "high bound first" in note
    # overlapping windows are left alone
    assert _ordered_windows((20, 40), (15, 35))[2] is None


def test_reversed_windows_are_swapped_and_said():
    """F61 [13]/[14] gave the entry window right of the exit window; the
    entry/exit search scored nothing and said nothing."""
    g = _w10()
    kw = dict(surface_type="entry_exit", method="bishop", nx=6, ny=6,
              n_slices=30)
    right = search_critical_surface(g, x_entry_range=(20, 29),
                                    x_exit_range=(29.5, 45), **kw)
    wrong = search_critical_surface(g, x_entry_range=(29.5, 45),
                                    x_exit_range=(20, 29), **kw)
    assert wrong.critical is not None
    assert wrong.critical.FOS == pytest.approx(right.critical.FOS, abs=1e-9)
    assert "swapped" in wrong.warnings[0]
    assert not any("swapped" in w for w in right.warnings)
    assert list(wrong.search_ranges["x_entry_range"]) == [20, 29]


def test_reversed_windows_in_the_direct_searches():
    g = _w10()
    r = search_entry_exit(g, (30, 50), (20, 28), n_entry=4, n_exit=4)
    assert r.critical is not None and "swapped" in r.warnings[0]
    r = grid_search(g, (15, 40), (11, 30), 4, 4, x_entry_range=(30, 50),
                    x_exit_range=(20, 28))
    assert "swapped" in r.warnings[0]


def test_an_entry_exit_search_with_no_pair_says_so():
    r = search_entry_exit(_w10(), (25, 26), (24.8, 25.6), n_entry=3, n_exit=3)
    assert r.critical is None
    assert "No circle was scored" in r.warnings[-1]


def test_reversed_windows_in_the_rapid_drawdown_search():
    ft = 0.3048
    lay = SlopeSoilLayer(name="fill", top_elevation=110 * ft,
                         bottom_elevation=-5.0, gamma=21.2, phi=30.0,
                         c_prime=0.0, R_c=57.5, R_phi=16.0)
    dam = SlopeGeometry(surface_points=[(0.0, 0.0), (305 * ft, 110 * ft),
                                        (380 * ft, 110 * ft)],
                        soil_layers=[lay])
    r = search_rapid_drawdown(dam, 110 * ft, 24 * ft, surface_type="entry_exit",
                              x_entry_range=(90, 116), x_exit_range=(0, 40),
                              nx=2, ny=2, n_slices=20, toe_circles=False,
                              infinite_slope=False)
    assert r.search.critical is not None
    assert "swapped" in r.search.warnings[0]

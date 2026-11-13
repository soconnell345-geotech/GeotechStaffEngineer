"""GM-140 step 4: tension cracks and reverse-curvature circles as printed.

Made-up slopes. Each test's docstring states the printed rule and its page
(EM 1110-2-1902, 31 Oct 2003; Slide v6 Tutorial 16).
"""

import math
import warnings

import pytest

from slope_stability import analyze_slope, search_critical_surface
from slope_stability import cracks
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slices import build_slices
from slope_stability.slip_surface import CircularSlipSurface

_CU, _GAMMA = 40.0, 18.0
_CIRCLE = dict(xc=22.0, yc=18.0, radius=19.0)


def _clay(**kw):
    return SlopeSoilLayer("clay", top_elevation=10.0, bottom_elevation=-8.0,
                          gamma=_GAMMA, cu=_CU, analysis_mode="undrained",
                          **kw)


def _geom(mirror=False, layers=None, **kw):
    """A 10 m, 2H:1V slope: crest at low x, or at high x when mirrored."""
    pts = [(0.0, 10.0), (10.0, 10.0), (30.0, 0.0), (45.0, 0.0)]
    if mirror:
        pts = [(45.0 - x, z) for x, z in reversed(pts)]
    return SlopeGeometry(surface_points=pts, soil_layers=layers or [_clay()],
                         **kw)


def _mirrored(circle):
    return dict(xc=45.0 - circle["xc"], yc=circle["yc"],
                radius=circle["radius"])


# ---------------------------------------------------------------------------
# The side: the upper end of the surface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mirror", [False, True])
def test_crack_lands_at_the_crest_end_either_way(mirror):
    """EM 2-2f(3), p. 2-6: the crack is "near the crest of the slope" and
    the soil upslope of it is ignored. A slope rising towards low x and
    one rising towards high x both get the crack at the crest end with no
    side given, and the same factor of safety."""
    circle = _mirrored(_CIRCLE) if mirror else _CIRCLE
    r = analyze_slope(_geom(mirror, tension_crack_depth=2.0),
                      method="bishop", include_slice_data=True, **circle)
    assert r.tension_crack["side"] == ("exit" if mirror else "entry")
    # the ground at the crack is the crest's, and the mass ends inside it
    assert r.tension_crack["ground_z_m"] == pytest.approx(10.0)
    xs = [s.x_mid for s in r.slice_data]
    if mirror:
        assert max(xs) < r.x_exit - 1.0
    else:
        assert min(xs) > r.x_entry + 1.0
    assert r.FOS == pytest.approx(1.5733, abs=2e-4)


def test_the_two_slopes_agree_and_a_given_side_is_kept():
    """The mirrored slope reads the same; 'entry' / 'exit' still give the
    end (here the toe end, a crack no print supports, as asked)."""
    a = analyze_slope(_geom(tension_crack_depth=2.0), **_CIRCLE).FOS
    b = analyze_slope(_geom(True, tension_crack_depth=2.0),
                      **_mirrored(_CIRCLE)).FOS
    assert a == pytest.approx(b, rel=1e-9)
    r = analyze_slope(_geom(tension_crack_depth=2.0,
                            tension_crack_side="exit"), **_CIRCLE)
    assert r.tension_crack["side"] == "exit"
    assert r.FOS != pytest.approx(a, abs=1e-3)
    with pytest.raises(ValueError, match="tension_crack_side"):
        _geom(tension_crack_side="toe")


# ---------------------------------------------------------------------------
# The model: the surface ends at the crack
# ---------------------------------------------------------------------------

def test_default_leaves_the_wedge_out_as_arithmetic_gives():
    """EM 2-2f(3), p. 2-6: the soil upslope from the crack "is then ignored
    in the stability computations ... by terminating the slices near the
    crest of the slope with a slice having a vertical boundary"; C-10c,
    p. C-36: the slip surface is terminated where it reaches the bottom of
    the crack. With a 2 m crack on a level crest at z = 10 the circle
    (22, 18), R = 19 reaches z = 8 at x = 22 - sqrt(19^2 - 10^2) = 5.844,
    so the mass runs from there to the exit and weighs gamma times the area
    between the ground and the arc over that stretch."""
    d = 2.0
    g = _geom(tension_crack_depth=d)
    slip = CircularSlipSurface(**_CIRCLE)
    slices = build_slices(g, slip, 400)
    x_crack = 22.0 - math.sqrt(19.0 ** 2 - (18.0 - (10.0 - d)) ** 2)
    assert slices[0].x_left == pytest.approx(x_crack, abs=1e-6)
    assert slices[0].z_top - (10.0 - d) == pytest.approx(d, abs=0.02)
    assert not any(s.in_tension_crack for s in slices)
    x_exit = slip.find_entry_exit(g)[1]
    assert slices[-1].x_right == pytest.approx(x_exit, abs=1e-6)
    # area by the midpoint rule on a fine grid, by hand of the same shapes
    n = 20000
    area = 0.0
    for i in range(n):
        x = x_crack + (x_exit - x_crack) * (i + 0.5) / n
        z_arc = 18.0 - math.sqrt(19.0 ** 2 - (x - 22.0) ** 2)
        area += (g.ground_elevation_at(x) - z_arc) * (x_exit - x_crack) / n
    assert sum(s.weight for s in slices) == pytest.approx(
        _GAMMA * area, rel=2e-4)


def test_strength_model_stays_as_a_stated_option():
    """The module's model before GM-140 (the cracked soil kept as driving
    weight without strength; no printed source) is still there when asked
    for, and reads lower than the printed model."""
    printed = analyze_slope(_geom(tension_crack_depth=3.0), **_CIRCLE)
    old = analyze_slope(_geom(tension_crack_depth=3.0,
                              tension_crack_model="strength"),
                        include_slice_data=True, **_CIRCLE)
    assert printed.tension_crack["model"] == "truncation"
    assert old.tension_crack["model"] == "strength"
    assert any(s.in_tension_crack for s in old.slice_data)
    assert old.FOS < printed.FOS
    assert printed.FOS == pytest.approx(1.5601, abs=2e-4)
    assert old.FOS == pytest.approx(1.4634, abs=2e-4)
    assert "ends at the bottom of the crack" in " ".join(printed.warnings)


def test_crack_water_acts_on_the_vertical_boundary():
    """EM 2-2f(3), p. 2-6: water in the crack is a force on the boundary of
    the slice next to it: 0.5 gamma_w z_w^2 at a third of its depth. A
    water depth above the crack's own fills it."""
    g = _geom(tension_crack_depth=3.0, tension_crack_water_depth=9.0)
    assert g.tension_crack_water_depth == 3.0
    slices = build_slices(g, CircularSlipSurface(**_CIRCLE), 40)
    assert slices[0].crack_water_force == pytest.approx(
        0.5 * 9.81 * 3.0 ** 2, rel=1e-9)
    assert slices[0].crack_water_z == pytest.approx(7.0 + 1.0)
    assert all(s.crack_water_force == 0 for s in slices[1:])
    dry = analyze_slope(_geom(tension_crack_depth=3.0), **_CIRCLE).FOS
    wet = analyze_slope(g, **_CIRCLE).FOS
    assert wet < dry


# ---------------------------------------------------------------------------
# The depth: Rankine's with the developed strengths
# ---------------------------------------------------------------------------

def test_rankine_depth_by_hand():
    """EM Eq. C-36, p. C-36: d = 2 c_D / (gamma tan(45 - phi_D / 2)), with
    c_D = c / F (Eq. 2-5) and phi_D = arctan(tan(phi) / F) (Eq. 2-6),
    p. 2-9. phi = 0: 2 x (40 / 1.25) / 18 = 3.556 m. Eq. 2-4, p. 2-9,
    z_t = (2 c_D / gamma) tan(45 + phi_D / 2), is the same depth."""
    assert cracks.rankine_crack_depth(40.0, 0.0, 18.0, 1.25) == \
        pytest.approx(2.0 * 32.0 / 18.0)
    c, phi, gamma, f = 10.0, 20.0, 19.0, 1.3
    phi_d = math.atan(math.tan(math.radians(phi)) / f)
    eq_2_4 = 2.0 * (c / f) / gamma * math.tan(math.pi / 4 + phi_d / 2)
    assert cracks.rankine_crack_depth(c, phi, gamma, f) == \
        pytest.approx(eq_2_4)
    assert cracks.rankine_crack_depth(0.0, 30.0, 19.0, 1.3) == 0.0


@pytest.mark.parametrize("method", ["fellenius", "bishop", "janbu",
                                    "spencer", "morgenstern_price", "gle"])
def test_rankine_depth_is_at_the_surfaces_own_fos_phi_zero(method):
    """F is the factor of safety of the surface itself, the solution
    repeated until the depth stops changing: for a phi = 0 clay the depth
    the result reports is 2 (c / F) / gamma at the F it reports, with
    every method."""
    r = analyze_slope(_geom(tension_crack_depth_basis="rankine"),
                      method=method, **_CIRCLE)
    t = r.tension_crack
    assert t["depth_basis"] == "rankine" and t["converged"]
    assert t["depth_m"] == pytest.approx(
        2.0 * (_CU / r.FOS) / _GAMMA, abs=2 * cracks.DEPTH_TOL)
    # the same surface with that depth given reads the same
    given = analyze_slope(_geom(tension_crack_depth=t["depth_m"]),
                          method=method, **_CIRCLE)
    assert given.FOS == pytest.approx(r.FOS, abs=2e-4)


def test_rankine_depth_for_a_c_phi_soil_at_the_converged_fos():
    """Eq. C-36 with c = 10 kPa, phi = 20 deg, gamma = 19 kN/m3 at the
    converged F of the surface (Spencer), by hand from the result's F."""
    soil = SlopeSoilLayer("s", top_elevation=10.0, bottom_elevation=-8.0,
                          gamma=19.0, phi=20.0, c_prime=10.0)
    r = analyze_slope(_geom(layers=[soil],
                            tension_crack_depth_basis="rankine"),
                      method="spencer", **_CIRCLE)
    phi_d = math.atan(math.tan(math.radians(20.0)) / r.FOS)
    by_hand = 2.0 * (10.0 / r.FOS) / (19.0 * math.tan(math.pi / 4
                                                     - phi_d / 2))
    assert r.tension_crack["depth_m"] == pytest.approx(
        by_hand, abs=2 * cracks.DEPTH_TOL)
    assert r.tension_crack["depth_m"] == pytest.approx(0.879, abs=0.005)
    assert "Eq. C-36" in " ".join(r.warnings)


def test_rankine_depth_stops_at_the_bottom_of_its_soil():
    """EM C-10c, p. C-36: where Eq. C-36 gives a depth greater than the
    height of the embankment, the crack is as deep as the embankment and
    does not extend into the foundation. A 1 m clay cap over sand: Eq.
    C-36 gives more than 1 m, the crack stops at 1 m, and the result says
    so."""
    cap = SlopeSoilLayer("cap", top_elevation=10.0, bottom_elevation=9.0,
                         gamma=18.0, cu=40.0, analysis_mode="undrained")
    sand = SlopeSoilLayer("sand", top_elevation=9.0, bottom_elevation=-8.0,
                          gamma=19.0, phi=34.0, c_prime=0.0)
    r = analyze_slope(_geom(layers=[cap, sand],
                            tension_crack_depth_basis="rankine"), **_CIRCLE)
    t = r.tension_crack
    assert t["depth_m"] == pytest.approx(1.0)
    assert t["stopped_at_soil_bottom"] is True and t["soil"] == "cap"
    assert "stops there" in " ".join(r.warnings)


def test_no_cohesion_no_crack():
    """Eq. C-36 with c = 0 is 0: a sand has no crack."""
    sand = SlopeSoilLayer("sand", top_elevation=10.0, bottom_elevation=-8.0,
                          gamma=19.0, phi=34.0, c_prime=0.0)
    plain = analyze_slope(_geom(layers=[sand]), **_CIRCLE)
    r = analyze_slope(_geom(layers=[sand],
                            tension_crack_depth_basis="rankine"), **_CIRCLE)
    assert r.FOS == pytest.approx(plain.FOS, rel=1e-12)
    assert r.tension_crack is None


def test_drained_soil_with_water_in_the_cracked_depth_uses_the_scan():
    """Eq. C-36 is printed "for total stresses and homogeneous soil" (EM
    C-10c(2), p. C-36); for effective stresses 2-2f(6), p. 2-9, prints no
    equation. A drained soil with the water table inside the cracked depth
    is scanned for the depth of minimum F instead, and the result says
    why."""
    soil = SlopeSoilLayer("s", top_elevation=10.0, bottom_elevation=-8.0,
                          gamma=19.0, phi=20.0, c_prime=10.0)
    g = _geom(layers=[soil], tension_crack_depth_basis="rankine",
              gwt_points=[(0.0, 9.8), (10.0, 9.8), (30.0, 0.0), (45.0, 0.0)])
    r = analyze_slope(g, method="spencer", **_CIRCLE)
    t = r.tension_crack
    assert t["depth_basis"] == "min_fos"
    assert "water table is inside the cracked depth" in t["rankine_not_used"]
    assert "Rankine's depth was not used" in " ".join(r.warnings)


def test_a_given_depth_and_a_basis_are_not_mixed():
    with pytest.raises(ValueError, match="tension_crack_depth"):
        _geom(tension_crack_depth=2.0, tension_crack_depth_basis="rankine")
    with pytest.raises(ValueError, match="tension_crack_depth_basis"):
        _geom(tension_crack_depth_basis="craig")


def test_slices_are_not_cut_on_a_silent_basis():
    """A crack whose depth depends on the factor of safety has no slices
    until a solve fixes it: `build_slices` refuses, and says how."""
    with pytest.raises(cracks.CrackDepthNeeded, match="solve_with_crack"):
        build_slices(_geom(tension_crack_depth_basis="rankine"),
                     CircularSlipSurface(**_CIRCLE), 30)


# ---------------------------------------------------------------------------
# The depth: the scan for minimum factor of safety
# ---------------------------------------------------------------------------

def test_scan_returns_the_depth_of_minimum_fos():
    """EM 2-2f(4), p. 2-6, and C-10c(1), p. C-36: assume a range of depths
    and take the one of minimum factor of safety. The hand table (Bishop,
    this circle, each depth given): 0 m 1.663, 1 m 1.609, 2 m 1.573,
    2.5 m 1.563, 3 m 1.560, 4 m 1.576, 5 m 1.631. The minimum is near 3 m;
    the scan returns a depth between 2.5 and 4 m and the F of the
    table's lowest row to 0.001, and it is close to Rankine's depth with the
    developed strengths (2.85 m), as the manual's two ways should be."""
    table = {d: analyze_slope(_geom(tension_crack_depth=d), **_CIRCLE).FOS
             for d in (1.0, 2.0, 2.5, 3.0, 4.0, 5.0)}
    assert table[3.0] == pytest.approx(1.5601, abs=2e-4)
    assert min(table, key=table.get) == 3.0
    r = analyze_slope(_geom(tension_crack_depth_basis="min_fos"), **_CIRCLE)
    t = r.tension_crack
    assert t["depth_basis"] == "min_fos"
    assert 2.5 < t["depth_m"] < 4.0
    # the scan steps by a twentieth of its range (0.52 m here), so its
    # minimum is the table's to within that step's effect on F
    assert r.FOS == pytest.approx(table[3.0], abs=1e-3)
    assert r.FOS == min(row["FOS"] for row in t["scan"]) or \
        r.FOS == pytest.approx(min(row["FOS"] for row in t["scan"]),
                               abs=1e-4)
    rankine = analyze_slope(_geom(tension_crack_depth_basis="rankine"),
                            **_CIRCLE)
    assert rankine.tension_crack["depth_m"] == pytest.approx(
        t["depth_m"], abs=0.5)
    assert rankine.FOS == pytest.approx(r.FOS, abs=0.002)


# ---------------------------------------------------------------------------
# Reverse-curvature circles (OWN-103)
# ---------------------------------------------------------------------------

#: centre 0.5 m below the crest: the arc is vertical at x = 6, z = 9.5
_SHALLOW = dict(xc=20.0, yc=9.5, radius=14.0)
#: centre 5 m below the crest
_DEEP = dict(xc=20.0, yc=5.0, radius=12.5)


def test_shallow_overhang_is_closed_at_the_printed_depth():
    """OWN-103 with EM C-10c, p. C-36: with a crack asked for (here
    Rankine's), a reverse-curvature circle whose overhang lies within the
    crack's depth is closed by that crack. The centre is 0.5 m below the
    crest; Rankine's depth with the developed strengths is deeper, so the
    circle is closed there, at 2 (c / F) / gamma, not at the centre's
    level. The crack is reported where the slices end (the step 4 review's
    D3)."""
    r = analyze_slope(_geom(tension_crack_depth_basis="rankine"),
                      method="bishop", include_slice_data=True, **_SHALLOW)
    t = r.tension_crack
    assert t["reverse_curvature"] is True
    assert t["overhang_depth_m"] == pytest.approx(0.5, abs=1e-6)
    assert t["depth_basis"] == "rankine"
    assert t["depth_m"] == pytest.approx(2.0 * (_CU / r.FOS) / _GAMMA,
                                         abs=2 * cracks.DEPTH_TOL)
    assert t["depth_m"] > 0.5
    # the mass ends where the arc reaches the bottom of the crack
    x_crack = 20.0 - math.sqrt(14.0 ** 2 - (9.5 - t["base_z_m"]) ** 2)
    assert r.slice_data[0].x_mid > x_crack
    assert t["x_m"] == pytest.approx(x_crack, abs=1e-3)
    first = r.slice_data[0]
    assert first.x_mid - 0.5 * first.width == pytest.approx(x_crack,
                                                            abs=1e-3)
    assert t["ground_z_m"] == pytest.approx(10.0)
    assert "reverse-curvature" in " ".join(r.warnings)


def test_deep_overhang_is_discarded_with_the_reason():
    """OWN-103: a circle whose overhanging part runs deeper than the crack
    is discarded. The centre is 5 m below the crest and Rankine's depth is
    under 5 m: no factor of safety, and the error says both depths."""
    with pytest.raises(cracks.ReverseCurvatureDiscarded) as exc:
        analyze_slope(_geom(tension_crack_depth_basis="rankine"),
                      method="bishop", **_DEEP)
    msg = str(exc.value)
    assert "5.00 m below the ground" in msg and "Eq. C-36" in msg
    out = exc.value.outcome
    assert out.discarded and out.overhang == pytest.approx(5.0, abs=1e-6)
    assert out.depth < 5.0
    assert out.refusal == cracks.DEEPER_THAN_CRACK
    # 'discard' discards the shallow one too
    with pytest.raises(cracks.ReverseCurvatureDiscarded, match="as asked"):
        analyze_slope(_geom(reverse_curvature="discard"), **_SHALLOW)


def test_a_given_crack_closes_the_circle_when_it_is_deep_enough():
    """The crack's depth is the one given, when one is: a 6 m crack closes
    the deep circle, a 2 m one does not."""
    r = analyze_slope(_geom(tension_crack_depth=6.0), **_DEEP)
    assert r.tension_crack["depth_basis"] == "given"
    assert r.tension_crack["depth_m"] == pytest.approx(6.0)
    with pytest.raises(cracks.ReverseCurvatureDiscarded,
                       match="given crack depth"):
        analyze_slope(_geom(tension_crack_depth=2.0), **_DEEP)


def test_no_crack_at_the_lower_end():
    """A crack is at the upper end of a surface (EM 2-2f(3), p. 2-6): a
    circle overhanging at its lower end has no printed crack to close it."""
    g = SlopeGeometry(
        surface_points=[(0.0, 10.0), (10.0, 10.0), (12.0, 0.0), (40.0, 0.0)],
        soil_layers=[_clay()])
    # centre below the toe's level, the arc vertical under the toe side
    slip = CircularSlipSurface(xc=14.0, yc=-1.0, radius=10.0)
    ends = slip.find_entry_exit(g)
    rev = cracks.reverse_ends(g, slip, *ends)
    assert {e["side"] for e in rev} == {"exit", "entry"}
    with pytest.raises(cracks.ReverseCurvatureDiscarded, match="lower end"):
        analyze_slope(g, slip_surface=slip)


def _steep():
    """A 10 m clay slope at 2V:1H, where reverse-curvature circles are
    among the low ones."""
    return dict(
        surface_points=[(0.0, 0.0), (10.0, 0.0), (15.0, 10.0), (45.0, 10.0)],
        soil_layers=[SlopeSoilLayer("clay", top_elevation=10.0,
                                    bottom_elevation=-6.0, gamma=18.0,
                                    cu=45.0, analysis_mode="undrained")])


@pytest.mark.parametrize("method", ["bishop", "spencer"])
def test_search_counts_and_locates_the_circles_it_discards(method):
    """F1 (the orchestrator's ruling on the step 4 review): with no crack
    asked for, every reverse-curvature circle is discarded, counted under
    its own reason and located; no crack appears in the result, and the
    choices say that naming a crack admits those within its depth.
    'discard all' is the stated option and reads the same here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = search_critical_surface(SlopeGeometry(**_steep()),
                                      method=method)
        all_out = search_critical_surface(
            SlopeGeometry(reverse_curvature="discard", **_steep()),
            method=method)
    assert res.n_discarded_reverse_curvature > 0
    where = res.crack_refused[cracks.NO_CRACK_ASKED]
    assert set(res.crack_refused) <= set(cracks.REVERSE_REFUSALS)
    assert sum(v["n"] for v in res.crack_refused.values()) == \
        res.n_discarded_reverse_curvature
    assert where["x_end_range_m"][0] >= 10.0            # above the toe
    assert where["first"][0]["overhang_depth_m"] > 0
    assert "naming a crack" in where["what"]
    assert res.to_dict()["crack_refused"][cracks.NO_CRACK_ASKED]["n"] == \
        where["n"]
    what = {c["what"]: c for c in res.choices}
    rc = what["reverse-curvature circles"]
    assert f"({res.n_discarded_reverse_curvature} discarded)" in \
        rc["chosen"]
    assert "no tension crack was asked for" in rc["chosen"]
    assert "naming a crack admits" in rc["basis"]
    assert "tension crack" not in what
    crit = res.critical
    assert crit.tension_crack is None and crit.tension_crack_depth == 0
    assert not any(w.startswith("Tension crack")
                   for w in res.warnings + crit.warnings)
    assert set(all_out.crack_refused) == {cracks.DISCARD_ASKED}
    assert all_out.n_discarded_reverse_curvature == \
        res.n_discarded_reverse_curvature
    assert all_out.critical.FOS == pytest.approx(res.critical.FOS, abs=1e-9)
    assert all_out.critical.tension_crack is None


def test_scan_option_for_a_reverse_curvature_circle():
    """The scan as the crack asked for (EM C-10c(1)): the shallow circle is
    closed at the depth of minimum F, the first minimum going down."""
    r = analyze_slope(_geom(tension_crack_depth_basis="min_fos"),
                      **_SHALLOW)
    rk = analyze_slope(_geom(tension_crack_depth_basis="rankine"),
                       **_SHALLOW)
    assert r.tension_crack["depth_basis"] == "min_fos"
    assert r.tension_crack["depth_m"] > r.tension_crack["overhang_depth_m"]
    scan = r.tension_crack["scan"]
    assert r.FOS == pytest.approx(min(row["FOS"] for row in scan), abs=1e-4)
    assert scan[-1]["FOS"] > r.FOS          # it rises past the minimum
    assert r.FOS <= rk.FOS + 1e-6


# ---------------------------------------------------------------------------
# The crack is a condition of the analysis (the step 4 review's F1)
# ---------------------------------------------------------------------------

def _f1_clay(**kw):
    """The reviewer's made-up clay: a 2V:1H face 8 m high, cu 30 kPa,
    gamma 18 kN/m3."""
    lay = SlopeSoilLayer("clay", top_elevation=8.0, bottom_elevation=-10.0,
                         gamma=18.0, cu=30.0, analysis_mode="undrained")
    return SlopeGeometry(surface_points=[(0.0, 0.0), (10.0, 0.0),
                                         (14.0, 8.0), (40.0, 8.0)],
                         soil_layers=[lay], **kw)


def _through_toe(yc, xc=12.0):
    return dict(xc=xc, yc=yc, radius=math.hypot(xc - 10.0, yc))


@pytest.mark.parametrize("kw", [dict(tension_crack_depth=3.0),
                                dict(tension_crack_depth_basis="rankine"),
                                dict(tension_crack_depth_basis="min_fos")],
                         ids=["given", "rankine", "min_fos"])
def test_f_is_continuous_across_the_crests_level_with_a_crack(kw):
    """With a crack asked for, every surface ends at the crack's bottom,
    and a circle whose centre passes the crest's level does not change
    its crack: two toe circles with their centres 2 cm above and 2 cm
    below the crest's level read within 1 %; the lower one is a
    reverse-curvature circle closed by the same crack."""
    up = analyze_slope(_f1_clay(**kw), method="bishop", **_through_toe(8.02))
    low = analyze_slope(_f1_clay(**kw), method="bishop",
                        **_through_toe(7.98))
    assert low.FOS == pytest.approx(up.FOS, rel=0.01)
    assert up.tension_crack and not up.tension_crack.get("reverse_curvature")
    assert low.tension_crack["reverse_curvature"] is True
    assert low.tension_crack["depth_basis"] == up.tension_crack["depth_basis"]
    assert low.tension_crack["depth_m"] == pytest.approx(
        up.tension_crack["depth_m"], rel=0.01)


def test_no_crack_asked_the_reverse_circle_is_no_surface():
    """With no crack asked for, the circle 2 cm above the crest's level has
    its factor of safety and no crack; the one 2 cm below is a
    reverse-curvature circle, not an admissible surface: refused, and the
    error says that naming a crack admits it. Before the fix it was given
    a 3.57 m Rankine crack and read 18 % lower than its neighbour."""
    up = analyze_slope(_f1_clay(), method="bishop", **_through_toe(8.02))
    assert up.FOS > 1.0 and up.tension_crack is None
    assert up.tension_crack_depth == 0
    assert not any("ension crack" in w for w in up.warnings)
    with pytest.raises(cracks.ReverseCurvatureDiscarded) as exc:
        analyze_slope(_f1_clay(), method="bishop", **_through_toe(7.98))
    assert exc.value.outcome.refusal == cracks.NO_CRACK_ASKED
    assert "naming a crack" in str(exc.value)
    assert "0.02 m below the ground" in str(exc.value)


def test_no_crack_search_on_the_reviewers_clay_counts_the_discards():
    """The default search on that clay, no crack asked for: the
    reverse-curvature circles are discarded and counted under their
    reason, and no crack appears in the result."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = search_critical_surface(_f1_clay(), method="bishop")
    assert res.crack_refused[cracks.NO_CRACK_ASKED]["n"] > 0
    assert res.critical.tension_crack is None
    assert res.critical.tension_crack_depth == 0
    assert not any("ension crack" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# A surface wholly within the cracked depth (D1); counts by reason (D5)
# ---------------------------------------------------------------------------

def test_a_surface_wholly_within_the_crack_is_no_surface():
    """D1: a surface that nowhere passes below the crack's depth, measured
    below the ground above it, lies wholly within the cracked depth: no
    slip surface, whatever the crack model. A small circle on the face, at
    most about 1.3 m below the ground, with a 2 m crack given."""
    small = dict(xc=21.0, yc=8.0, radius=4.0)
    slip = CircularSlipSurface(**small)
    g = _geom(tension_crack_depth=2.0, tension_crack_water_depth=2.0)
    d_max = cracks.depth_below_ground(g, slip, *slip.find_entry_exit(g))
    assert 0.5 < d_max < 2.0
    for model in ("truncation", "strength"):
        with pytest.raises(cracks.CrackRuleRefused) as exc:
            analyze_slope(_geom(tension_crack_depth=2.0,
                                tension_crack_water_depth=2.0,
                                tension_crack_model=model), **small)
        assert exc.value.outcome.refusal == cracks.WITHIN_CRACK
        assert "wholly within the cracked depth" in str(exc.value)
    # a crack shallower than the surface's depth keeps it
    r = analyze_slope(_geom(tension_crack_depth=0.5 * d_max), **small)
    assert r.FOS > 0 and r.tension_crack["depth_m"] == pytest.approx(
        0.5 * d_max, abs=1e-3)


def test_depth_below_ground_by_hand():
    """The greatest depth of a circle below straight ground: where the arc
    runs parallel to the ground, R - (perpendicular distance from the
    centre to the ground line), on a 2H:1V face."""
    g = _geom()
    slip = CircularSlipSurface(xc=21.0, yc=8.0, radius=4.0)
    # the face z = 10 - (x - 10) / 2 -> x + 2 z - 30 = 0
    dist = abs(21.0 + 2 * 8.0 - 30.0) / math.sqrt(5.0)
    vertical = (4.0 - dist) * math.sqrt(5.0) / 2.0   # depth, not distance
    assert cracks.depth_below_ground(g, slip, *slip.find_entry_exit(g)) \
        == pytest.approx(vertical, rel=1e-9)


def test_refusals_are_counted_by_reason():
    """D5: each reason the crack rule refuses a surface is counted on its
    own, in the result and in the choices, and only the reverse-curvature
    reasons count as reverse-curvature discards."""
    from slope_stability.results import SearchResult

    plan = cracks.CrackPlan(side="entry", x_end=5.0, z_ground=10.0,
                            basis="given")

    @cracks.counts_reverse_discards
    def run():
        for key, n in ((cracks.WITHIN_CRACK, 3), (cracks.SCAN_NO_MINIMUM, 1),
                       (cracks.DEEPER_THAN_CRACK, 2),
                       (cracks.NO_CRACK_ASKED, 4)):
            for _ in range(n):
                cracks.count_discard(
                    None, cracks._discarded(_geom(), plan, refusal=key,
                                            reason=key))
        res = SearchResult()
        res.choices = cracks.crack_choices(_geom(), cracks.discards_so_far())
        return res

    res = run()
    assert {k: v["n"] for k, v in res.crack_refused.items()} == {
        cracks.WITHIN_CRACK: 3, cracks.SCAN_NO_MINIMUM: 1,
        cracks.DEEPER_THAN_CRACK: 2, cracks.NO_CRACK_ASKED: 4}
    assert res.n_discarded_reverse_curvature == 6
    for k, v in res.crack_refused.items():
        assert v["what"] == cracks.REFUSAL_TEXT[k]
    what = {c["what"]: c for c in res.choices}
    assert "(6 discarded)" in what["reverse-curvature circles"]["chosen"]
    other = what["surfaces the crack rule refused"]["chosen"]
    assert other.startswith("3 surfaces that nowhere pass below")
    assert "1 surfaces on which the scan" in other


# ---------------------------------------------------------------------------
# Every caller
# ---------------------------------------------------------------------------

def test_search_with_a_rankine_crack_reports_it():
    """In a search every trial surface takes its own crack; the governing
    one reports it, and the choices say how the depth was set."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = search_critical_surface(
            _geom(tension_crack_depth_basis="rankine"), method="bishop")
        plain = search_critical_surface(_geom(), method="bishop")
    t = res.critical.tension_crack
    assert t["depth_basis"] == "rankine"
    assert t["depth_m"] == pytest.approx(
        2.0 * (_CU / res.critical.FOS) / _GAMMA, abs=0.01)
    assert res.critical.FOS < plain.critical.FOS
    what = {c["what"]: c for c in res.choices}
    assert what["tension crack depth"]["how"] == "given"
    assert "Eq. C-36" in what["tension crack depth"]["basis"]
    assert what["tension crack"]["how"] == "default"
    assert what["tension crack side"]["chosen"].startswith("the end of")
    again = analyze_slope(_geom(tension_crack_depth_basis="rankine"),
                          method="bishop", xc=res.critical.xc,
                          yc=res.critical.yc, radius=res.critical.radius)
    assert again.FOS == pytest.approx(res.critical.FOS, abs=2e-3)


def test_noncircular_surface_takes_the_same_crack():
    from slope_stability.slip_surface import PolylineSlipSurface
    slip = PolylineSlipSurface([(4.0, 10.0), (9.0, 2.0), (20.0, -2.0),
                                (33.0, 0.0)])
    plain = analyze_slope(_geom(), method="spencer", slip_surface=slip)
    r = analyze_slope(_geom(tension_crack_depth_basis="rankine"),
                      method="spencer", slip_surface=slip)
    assert r.tension_crack["side"] == "entry"
    assert r.tension_crack["depth_m"] == pytest.approx(
        2.0 * (_CU / r.FOS) / _GAMMA, abs=2 * cracks.DEPTH_TOL)
    assert r.FOS != pytest.approx(plain.FOS, abs=1e-4)


def test_yield_acceleration_takes_the_crack_at_f_equal_one():
    """At the yield coefficient F = 1, so c_D = c: Rankine's depth there is
    2 c / gamma, the depth the bisection is run with."""
    from slope_stability.newmark import yield_acceleration
    rk = yield_acceleration(_geom(tension_crack_depth_basis="rankine"),
                            method="bishop", **_CIRCLE)
    given = yield_acceleration(
        _geom(tension_crack_depth=2.0 * _CU / _GAMMA), method="bishop",
        **_CIRCLE)
    assert rk.ky == pytest.approx(given.ky, abs=1e-6)
    none = yield_acceleration(_geom(), method="bishop", **_CIRCLE)
    assert rk.ky != pytest.approx(none.ky, abs=1e-4)


def test_probabilistic_runs_take_each_runs_own_crack():
    from slope_stability.probabilistic import fosm_fos
    out = fosm_fos(_geom(tension_crack_depth_basis="rankine"),
                   {"cu": {"mean": 40.0, "cov": 0.2}},
                   method="bishop", **_CIRCLE)
    base = analyze_slope(_geom(tension_crack_depth_basis="rankine"),
                         method="bishop", **_CIRCLE)
    assert out.fos_mlv == pytest.approx(base.FOS, abs=2e-3)


def test_rapid_drawdown_takes_the_crack_through_its_stages():
    from slope_stability.analysis import rapid_drawdown_fos
    lay = SlopeSoilLayer("s", top_elevation=10.0, bottom_elevation=-8.0,
                         gamma=19.0, phi=28.0, c_prime=8.0)
    kw = dict(drawdown_from_elevation=8.0, drawdown_to_elevation=1.0,
              method="corps_2stage", **_CIRCLE)
    plain = rapid_drawdown_fos(_geom(layers=[lay]), **kw)
    cracked = rapid_drawdown_fos(_geom(layers=[lay], tension_crack_depth=1.5),
                                 **kw)
    assert cracked.FOS != pytest.approx(plain.FOS, abs=1e-4)
    assert cracked.n_slices < plain.n_slices or cracked.FOS > 0

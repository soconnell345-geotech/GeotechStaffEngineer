"""GM-140 step 8: supports as printed.

The rule. A support enters a solution as a known force, the way the print
gives it: by Method A or Method B, along the reinforcement's own direction,
and no larger than the reinforcement can carry where the slip surface
crosses it (``slope_stability.supports``).

Print: Duncan, Wright and Brandon (2014), "Soil Strength and Slope
Stability", 2nd ed., Sec. 6.5.3 (pp. 91 to 93) and chapter 8 (pp. 159 to
165); FHWA GEC 11 Sec. 9.1 (p. 9-1), Eq. 9-9 and p. 9-14; FHWA GEC 7
Eq. 5.2 (p. 105). The two benchmarks are the book's own (Figs. 8.1 and
8.4); every other value here is made up.
"""

import math

import pytest

from slope_stability.analysis import (analyze_slope, compare_methods_table,
                                      search_critical_surface)
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.methods import bishop_fos
from slope_stability.nails import SoilNail
from slope_stability.reinforcement import (
    Anchor, Geosynthetic, StabilizingPile, compute_reinforcement_forces,
    slice_support_forces)
from slope_stability.slices import build_slices
from slope_stability.slip_surface import (CircularSlipSurface,
                                          PolylineSlipSurface)
from slope_stability.supports import (
    level_crossings, line_crossings, resolve_supports, slice_index_for,
    stress_integral)

FT = 0.3048
PCF = 0.157087          # kN/m3 per lb/ft3
PSF = 0.04788026        # kPa per lb/ft2
LB_FT = 0.0145939       # kN/m per lb/ft


def _slope(**kw):
    """A made-up 2H:1V slope 10 m high in a c'-phi' soil, crest on +x."""
    return SlopeGeometry(
        surface_points=[(0, 10), (20, 10), (40, 20), (70, 20)],
        soil_layers=[SlopeSoilLayer(name="soil", top_elevation=20.0,
                                    bottom_elevation=-15.0, gamma=19.0,
                                    phi=25.0, c_prime=8.0)], **kw)


_CIRCLE = CircularSlipSurface(xc=30.0, yc=32.0, radius=26.0)


def _resolve(geom, slip=_CIRCLE):
    return resolve_supports(geom, slip, *slip.find_entry_exit(geom))


# ---------------------------------------------------------------------------
# The two printed benchmarks
# ---------------------------------------------------------------------------

def _fig_8_1(convention):
    """Duncan, Wright and Brandon (2014) Fig. 8.1, pp. 160 to 161: a slope
    20 ft high at 2 vertical on 1 horizontal; gamma = 100 pcf, phi = 0,
    c = 500 psf; a firm layer whose top is at the toe's level; one
    reinforcing force of 10,000 lb/ft acting horizontally 10 ft above the
    toe. Entered in SI (the factors above)."""
    h = 20 * FT
    return h, SlopeGeometry(
        surface_points=[(-15, 0), (0, 0), (h / 2, h), (40, h)],
        soil_layers=[
            SlopeSoilLayer("clay", h, 0.0, 100 * PCF, cu=500 * PSF,
                           analysis_mode="undrained"),
            # the firm layer: only its top matters, the circles are
            # tangent to it (its strength is made up)
            SlopeSoilLayer("firm", 0.0, -5.0, 100 * PCF, cu=5000.0,
                           analysis_mode="undrained")],
        geosynthetics=[Geosynthetic(elevation=10 * FT,
                                    T_allow=10000 * LB_FT,
                                    support_convention=convention)])


def _tangent_minimum(convention):
    """The lowest Bishop value over circles tangent to the top of the firm
    layer (centre height = radius), on a 0.1 m grid of centre x and
    radius, and its circle."""
    h, geom = _fig_8_1(convention)
    best = None
    for i in range(0, 21):
        xc = 0.1 * i
        for j in range(0, 46):
            r = h + 0.02 + 0.1 * j
            try:
                res = analyze_slope(
                    geom, slip_surface=CircularSlipSurface(xc=xc, yc=r,
                                                           radius=r),
                    method="bishop", n_slices=40)
            except ValueError:
                continue
            if res.FOS is not None and (best is None or res.FOS < best[0]):
                best = (res.FOS, xc, r, res)
    return h, best


def test_fig_8_1_method_a():
    """Printed: F = 2.19 by Method A, within 1 or 2 % (p. 161). The
    critical circle exits slightly above the toe and has its centre at
    about the crest's level, the limit before reverse curvature (the
    brief's hand check of the moment equation gives 2.21 there)."""
    h, (fos, xc, r, res) = _tangent_minimum("active")
    # GM-140 step 8: ours 2.219; printed 2.19 (Duncan, Wright and Brandon
    # 2014, Fig. 8.1, p. 161): +1.3 %
    assert fos == pytest.approx(2.19, rel=0.02)
    assert fos == pytest.approx(2.219, abs=0.005)
    assert r == pytest.approx(h, abs=0.11)          # centre at crest level
    assert 0.0 < res.x_entry < 0.1                  # exits above the toe
    assert len(res.reinforcements) == 1
    assert res.reinforcements[0]["T_kN_per_m"] == pytest.approx(145.94,
                                                                abs=0.01)
    c = [c for c in res.choices if c["what"].startswith("how the support")]
    assert "Method A" in c[0]["chosen"] and "allowable" in c[0]["chosen"]
    assert "Eqs. 8.1 and 8.2" in c[0]["basis"]


def test_fig_8_1_method_b():
    """Printed: F = 1.72 by Method B, within 1 or 2 % (p. 161)."""
    h, (fos, xc, r, res) = _tangent_minimum("passive")
    # GM-140 step 8: ours 1.722; printed 1.72 (same figure): +0.1 %
    assert fos == pytest.approx(1.72, rel=0.02)
    assert fos == pytest.approx(1.722, abs=0.005)
    assert 0.0 < res.x_entry < 0.1
    c = [c for c in res.choices if c["what"].startswith("how the support")]
    assert "Method B" in c[0]["chosen"] and "ultimate" in c[0]["chosen"]
    assert "Eqs. 8.3 and 8.4" in c[0]["basis"]


def _fig_8_4(convention):
    """Duncan, Wright and Brandon (2014) Fig. 8.4, pp. 164 to 165 (Slide2
    verification #86): a fill on rock 24 ft high at 1.25 horizontal to 1
    vertical; gamma = 130 pcf, c' = 0, phi' = 37 deg; six layers of
    reinforcement starting at the bottom of the slope, 4 ft apart, each
    20 ft long, each with 800 lb/ft that decreases linearly to zero over
    the last 4 ft of embedded length (Fig. 8.4a: the full force from the
    face to 16 ft along the layer). In ft, lb and pcf as printed: nothing
    here depends on the unit system (no water)."""
    g = SlopeGeometry(
        surface_points=[(0, 0), (30, 24), (75, 24)],
        soil_layers=[SlopeSoilLayer(name="fill", top_elevation=24.0,
                                    bottom_elevation=0.0, gamma=130.0,
                                    phi=37.0, c_prime=0.0)])
    g.geosynthetics = [
        Geosynthetic(elevation=float(z), T_allow=800.0, x_start=z / 0.8,
                     x_end=z / 0.8 + 20.0, development_length=4.0,
                     anchored_end="start", support_convention=convention)
        for z in (0, 4, 8, 12, 16, 20)]
    return g


@pytest.mark.parametrize("convention, printed, ours", [
    # GM-140 step 8: ours 1.607; printed 1.62 (Fig. 8.4, p. 165): -0.8 %
    ("active", 1.62, 1.607),
    # ours 1.448; printed 1.46: -0.8 %
    ("passive", 1.46, 1.448)])
def test_fig_8_4_spencer(convention, printed, ours):
    """Printed: Spencer's procedure, circles, minimum F = 1.62 by Method A
    and 1.46 by Method B."""
    res = search_critical_surface(_fig_8_4(convention), method="spencer",
                                  x_entry_range=(0.0, 4.0),
                                  x_exit_range=(30.0, 75.0))
    c = res.critical
    assert c.FOS == pytest.approx(printed, rel=0.02)
    assert c.FOS == pytest.approx(ours, abs=0.01)
    # the five layers above the base are crossed at their full force (the
    # sixth lies on the rock, under every circle)
    full = [r for r in c.reinforcements if r["T_kN_per_m"] == 800.0]
    assert len(full) == 5
    what = [x["chosen"] for x in res.choices
            if x["what"].startswith("how the support")]
    assert ("Method A" if convention == "active" else "Method B") in what[0]


# ---------------------------------------------------------------------------
# What a layer can carry at the crossing (D34)
# ---------------------------------------------------------------------------

def test_layer_crossed_near_its_end_carries_its_pullout():
    """Duncan, Wright and Brandon (2014) Fig. 8.2 and Eq. 8.8 (pp. 161 to
    163): the capacity is the lesser of T_lim and the pullout measured
    from each end, T_po = 2 gamma z alpha F* L_e under a constant depth of
    fill; Eq. 8.10 and Method A: divided by F_R."""
    g = _slope(geosynthetics=[Geosynthetic(
        elevation=14.0, T_allow=200.0, x_end=50.0, F_star=0.4, alpha=0.8,
        pullout_factor=1.5)])
    (f,) = _resolve(g).forces
    x_c = 30.0 + math.sqrt(26.0 ** 2 - 18.0 ** 2)       # the crossing
    assert f.x == pytest.approx(x_c, abs=1e-9)
    L_e = 50.0 - x_c                                    # 1.24 m, level crest
    hand = 2 * 19.0 * 6.0 * 0.8 * 0.4 * L_e / 1.5       # Eq. 8.8 / F_R
    assert f.T == pytest.approx(hand, rel=1e-9)
    assert f.T < 200.0 and f.controlled_by == "pullout"
    # without the pullout inputs the full force is applied and said
    g0 = _slope(geosynthetics=[Geosynthetic(elevation=14.0, T_allow=200.0,
                                            x_end=50.0)])
    out = _resolve(g0)
    assert out.forces[0].T == 200.0
    assert "pullout resistance was not checked" in out.notes[0]
    r = analyze_slope(g0, slip_surface=_CIRCLE, method="bishop")
    assert any("pullout resistance was not checked" in w
               for w in r.warnings)
    # Method B: the ultimate pullout, not divided by F_R
    gb = _slope(geosynthetics=[Geosynthetic(
        elevation=14.0, T_allow=200.0, x_end=50.0, F_star=0.4, alpha=0.8,
        support_convention="passive")])
    assert _resolve(gb).forces[0].T == pytest.approx(hand * 1.5, rel=1e-9)


def test_pullout_beneath_the_slope_face_is_eq_8_9():
    """Eq. 8.9: T_po = gamma tan(beta) alpha F* L_e^2 beneath the slope
    face. The made-up face rises 1 on 2 from (20, 10)."""
    g = _slope()
    L = 3.0
    got = 2 * 0.8 * 0.4 * stress_integral(g, 12.0, 24.0, 24.0 + L)
    assert got == pytest.approx(19.0 * 0.5 * 0.8 * 0.4 * L ** 2, rel=1e-12)


def test_pullout_from_the_front_end_limits_too():
    """Fig. 8.2: pullout is measured from each end. A layer starting just
    in front of the crossing is held by that short length."""
    x_c = 30.0 + math.sqrt(26.0 ** 2 - 18.0 ** 2)
    g = _slope(geosynthetics=[Geosynthetic(
        elevation=14.0, T_allow=200.0, x_start=x_c - 0.5, x_end=65.0,
        F_star=0.4, alpha=0.8, pullout_factor=1.5)])
    (f,) = _resolve(g).forces
    assert f.controlled_by == "pullout_front"
    assert f.T == pytest.approx(2 * 19.0 * 6.0 * 0.8 * 0.4 * 0.5 / 1.5,
                                rel=1e-9)
    # an end fixed to a facing is not a pullout end
    g.geosynthetics[0].anchored_end = "start"
    assert _resolve(g).forces[0].T == 200.0


def test_gec11_counts_nothing_with_less_than_1_m_behind():
    """GEC 11 p. 9-14: "Only count reinforcement that extends 3 ft (1 m)
    beyond the surface"; Eq. 9-9: pullout behind the surface, with the
    coverage ratio."""
    kw = dict(elevation=14.0, T_allow=200.0, F_star=0.4, alpha=0.8,
              pullout_factor=1.5, pullout_basis="gec11")
    out = _resolve(_slope(geosynthetics=[Geosynthetic(x_end=49.0, **kw)]))
    assert out.forces == []
    assert "less than the 1 m that counts" in out.not_counted[0]["reason"]
    x_c = 30.0 + math.sqrt(26.0 ** 2 - 18.0 ** 2)
    (f,) = _resolve(_slope(geosynthetics=[Geosynthetic(
        x_end=50.0, coverage_ratio=0.5, **kw)])).forces
    assert f.T == pytest.approx(
        2 * 19.0 * 6.0 * 0.8 * 0.4 * 0.5 * (50.0 - x_c) / 1.5, rel=1e-9)


def test_pullout_inputs_are_asked_for_not_filled_in():
    """F*, alpha and F_R are findings about the fill and the product
    (Tables 8.3 and 8.4; GEC 11 Table 3-6): none has a default."""
    with pytest.raises(ValueError, match="both F_star and alpha"):
        Geosynthetic(elevation=14.0, T_allow=200.0, F_star=0.4)
    g = _slope(geosynthetics=[Geosynthetic(
        elevation=14.0, T_allow=200.0, x_end=50.0, F_star=0.4, alpha=0.8)])
    with pytest.raises(ValueError, match="pullout_factor"):
        _resolve(g)
    with pytest.raises(ValueError, match="pullout_basis"):
        Geosynthetic(elevation=14.0, T_allow=200.0, pullout_basis="x")


# ---------------------------------------------------------------------------
# A nail (D35)
# ---------------------------------------------------------------------------

def test_nail_limited_by_its_front_term():
    """GEC 7 Eq. 5.2 (p. 105): Tmax = min(R_T, R_F, R_PO); in front of the
    surface the load is limited by the facing with the pullout of the
    length between the facing and the surface (Fig. 5.7)."""
    def nail(**kw):
        return SoilNail(x_head=38.0, z_head=19.0, length=30.0,
                        inclination=15.0, bond_stress=100.0, spacing_h=1.5,
                        bar_diameter=50.0, **kw)
    plain = _resolve(_slope(nails=[nail()]))
    (f0,) = plain.forces
    assert f0.controlled_by in ("tensile", "pullout")
    assert "facing_capacity" in plain.notes[0]         # not checked: said
    out = _resolve(_slope(nails=[nail(facing_capacity=20.0)]))
    (f,) = out.forces
    t = math.hypot(f.x - 38.0, f.z - 19.0)             # length in front
    rate = 100.0 * math.pi * 0.150
    assert f.controlled_by == "facing"
    assert f.T == pytest.approx((20.0 + rate * t) / 1.5, rel=1e-9)
    assert f.T < f0.T and out.notes == []


# ---------------------------------------------------------------------------
# Bishop's simplified method with known forces (Eq. 6.78)
# ---------------------------------------------------------------------------

def _bishop_by_hand(slices, slip, forces, tol=1e-12):
    """Eq. 6.78 with Eqs. 6.70 and 6.75, Method A, written out:
    F = SUM[(c' b + (W - F_v - u b) tan phi') / m_alpha]
        / (SUM W sin(alpha) - M_n / r)."""
    r = slip.radius
    drive = sum((s.weight + s.surcharge_force) * (s.x_mid - slip.xc) / r
                for s in slices)
    fv = {}
    m_n = 0.0
    for f in forces:
        # T acts at psi from the horizontal where it crosses (Eq. 6.70):
        # moment about the centre, counterclockwise positive
        tx, tz = f.T * f.dir_x, f.T * f.dir_z
        m_n += (f.x - slip.xc) * tz - (f.z - slip.yc) * tx
        i = next(k for k, s in enumerate(slices)
                 if s.x_left <= f.x <= s.x_right)
        fv[i] = fv.get(i, 0.0) + tz                    # F_v, upward
    fos = 1.5
    for _ in range(200):
        num = 0.0
        for i, s in enumerate(slices):
            tan_phi = math.tan(math.radians(s.phi))
            w = s.weight + s.surcharge_force - fv.get(i, 0.0)
            m_a = math.cos(s.alpha) + math.sin(s.alpha) * tan_phi / fos
            num += (s.c * s.width
                    + (w - s.pore_pressure * s.width) * tan_phi) / m_a
        new = num / (drive - m_n / r)
        if abs(new - fos) < tol:
            return new
        fos = new
    raise AssertionError("no convergence")


def test_horizontal_support_adds_nothing_to_vertical_equilibrium():
    """p. 93: horizontal reinforcement enters moment equilibrium only."""
    g = _slope(geosynthetics=[Geosynthetic(elevation=14.0, T_allow=120.0)])
    x0, x1 = _CIRCLE.find_entry_exit(g)
    slices = build_slices(g, _CIRCLE, 40)
    forces = compute_reinforcement_forces(g, _CIRCLE, x0, x1)
    per_slice = slice_support_forces(forces, slices, passive=False,
                                     d_sign=1.0)
    assert [fz for _fx, fz in per_slice.values()] == [0.0]
    assert bishop_fos(slices, _CIRCLE, tol=1e-12, reinf_forces=forces) \
        == pytest.approx(_bishop_by_hand(slices, _CIRCLE, forces),
                         rel=1e-9)


def test_inclined_support_adds_t_sin_psi():
    """p. 93 and Eq. 6.75: inclined reinforcement enters both equations;
    its vertical part, T sin(psi), is always included."""
    psi = 20.0
    g = _slope(anchors=[Anchor(x_head=30.0, z_head=15.0, length=30.0,
                               T_allow=120.0, inclination=psi)])
    x0, x1 = _CIRCLE.find_entry_exit(g)
    slices = build_slices(g, _CIRCLE, 40)
    forces = compute_reinforcement_forces(g, _CIRCLE, x0, x1)
    per_slice = slice_support_forces(forces, slices, passive=False,
                                     d_sign=1.0)
    ((_fx, fz),) = per_slice.values()
    # the pull on the mass points down along the anchor: F_v upward is
    # -T sin(psi)
    assert fz == pytest.approx(-120.0 * math.sin(math.radians(psi)))
    with_v = bishop_fos(slices, _CIRCLE, tol=1e-12, reinf_forces=forces)
    assert with_v == pytest.approx(_bishop_by_hand(slices, _CIRCLE, forces),
                                   rel=1e-9)
    forces[0].dir_x, forces[0].dir_z = 1.0, 0.0        # the moment only
    assert with_v != pytest.approx(
        _bishop_by_hand(slices, _CIRCLE, forces), rel=1e-4)


def test_oms_is_left_out_of_the_comparison_with_its_reason():
    """No Ordinary Method of Slices equation with known forces is printed
    (p. 93): the method is refused where supports cross the surface and
    the comparison table leaves it out and says why."""
    g = _slope(geosynthetics=[Geosynthetic(elevation=14.0, T_allow=120.0)])
    table = compare_methods_table(g, slip_surface=_CIRCLE)
    assert not any("Fellenius" in r["method"] for r in table["rows"])
    assert "Fellenius" in table["methods_without_solution"]
    assert any("no reference held prints the Ordinary Method of Slices"
               in w for w in table["warnings"])
    # where no support crosses, the method is solved as before
    far = _slope(geosynthetics=[Geosynthetic(elevation=14.0, T_allow=120.0,
                                             x_start=60.0)])
    t2 = compare_methods_table(far, slip_surface=_CIRCLE)
    assert any("Fellenius" in r["method"] for r in t2["rows"])


# ---------------------------------------------------------------------------
# Direction (Sec. 8.6) and a pull that assists the sliding (OWN-59)
# ---------------------------------------------------------------------------

def test_force_acts_toward_the_anchorage_on_either_hand():
    """GM-82: a geosynthetic's force on the sliding mass points along the
    layer toward its anchorage, on a slope facing either way."""
    g = _slope(geosynthetics=[Geosynthetic(elevation=14.0, T_allow=50.0)])
    (f,) = _resolve(g).forces
    assert (f.dir_x, f.dir_z, f.side) == (1.0, 0.0, -1.0)
    mirror = SlopeGeometry(
        surface_points=[(-70, 20), (-40, 20), (-20, 10), (0, 10)],
        soil_layers=g.soil_layers,
        geosynthetics=[Geosynthetic(elevation=14.0, T_allow=50.0)])
    slip = CircularSlipSurface(xc=-30.0, yc=32.0, radius=26.0)
    (fm,) = _resolve(mirror, slip).forces
    assert (fm.dir_x, fm.dir_z, fm.side) == (-1.0, 0.0, 1.0)
    assert fm.x == pytest.approx(-f.x)
    a = analyze_slope(g, slip_surface=_CIRCLE, method="spencer").FOS
    b = analyze_slope(mirror, slip_surface=slip, method="spencer").FOS
    assert a == pytest.approx(b, rel=1e-6)


def test_force_parallel_to_the_slip_surface_is_the_stated_option():
    """Sec. 8.6, pp. 163 to 164: the reinforcement's own orientation is
    the default (the more conservative); parallel to the slip surface is
    the other extreme."""
    own = _slope(geosynthetics=[Geosynthetic(elevation=14.0,
                                             T_allow=120.0)])
    par = _slope(geosynthetics=[Geosynthetic(
        elevation=14.0, T_allow=120.0,
        force_direction="along_slip_surface")])
    (f,) = _resolve(par).forces
    alpha = math.atan2(f.x - 30.0, 32.0 - 14.0)         # the arc's slope
    assert (f.dir_x, f.dir_z) == pytest.approx((math.cos(alpha),
                                                math.sin(alpha)))
    r_own = analyze_slope(own, slip_surface=_CIRCLE, method="spencer")
    r_par = analyze_slope(par, slip_surface=_CIRCLE, method="spencer")
    assert r_par.FOS > r_own.FOS                        # the other extreme
    c = [c for c in r_own.choices if c["what"].startswith("direction")]
    assert c[0]["how"] == "default" and "Sec. 8.6" in c[0]["basis"]
    with pytest.raises(ValueError, match="force_direction"):
        Anchor(x_head=0, z_head=0, length=5, T_allow=10,
               force_direction="normal")


def test_support_pulling_with_the_slide_counts_nothing_and_is_named():
    """OWN-59: a support whose pull assists the sliding is slack by
    default; applying it as it stands is the stated option. A nail at 70
    degrees crosses the arc where it slopes at 31.7 degrees: its pull
    along the surface points down the slope."""
    def geom(**kw):
        return _slope(nails=[SoilNail(
            x_head=40.0, z_head=20.0, length=20.0, inclination=70.0,
            bond_stress=100.0, spacing_h=1.5)], **kw)
    plain = analyze_slope(_slope(), slip_surface=_CIRCLE, method="spencer")
    slack = analyze_slope(geom(), slip_surface=_CIRCLE, method="spencer")
    assert slack.FOS == pytest.approx(plain.FOS, rel=1e-9)
    assert slack.reinforcements is None
    (n,) = slack.supports_not_counted
    assert n["kind"] == "nail" and "slack" in n["reason"]
    assert any("count nothing" in w for w in slack.warnings)
    assert slack.to_dict()["supports_not_counted"] == [n]
    c = [c for c in slack.choices if "assists the sliding" in c["what"]]
    assert c[0]["chosen"].startswith("counts nothing")
    assert c[0]["how"] == "default"
    given = analyze_slope(geom(assisting_support="as_given"),
                          slip_surface=_CIRCLE, method="spencer")
    assert given.supports_not_counted is None
    assert len(given.reinforcements) == 1
    assert given.FOS != pytest.approx(plain.FOS, rel=1e-5)
    with pytest.raises(ValueError, match="assisting_support"):
        _slope(assisting_support="x")


def test_layer_crossed_twice_is_slack_at_its_toe_side_crossing():
    """A layer below the toe's level is crossed twice: at the crest side
    it holds the mass; at the toe side its pull points with the slide."""
    g = _slope(geosynthetics=[Geosynthetic(elevation=8.0, T_allow=100.0)])
    out = _resolve(g)
    (f,) = out.forces
    assert f.x == pytest.approx(40.0) and f.dir_x == 1.0
    (n,) = out.not_counted
    assert n["x_m"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# The crossing, found exactly; the legacy force; the slice boundary
# ---------------------------------------------------------------------------

def test_crossing_next_to_a_circles_vertical_end_is_found():
    """A line is crossed with the circle itself. Sampled at 200 points
    (before), an anchor reaching the arc within one step of its vertical
    end was missed: here the arc's last 0.4 mm."""
    g = SlopeGeometry(
        surface_points=[(0, 0), (10, 0), (20, 10), (60, 10)],
        soil_layers=[SlopeSoilLayer("clay", 10.0, -10.0, 18.0, cu=30.0,
                                    analysis_mode="undrained")])
    slip = CircularSlipSurface(xc=18.0, yc=10.0, radius=10.0,
                               x_start=10.0 + 1e-9, x_end=28.0 - 1e-9)
    z = 9.9
    x_c = 18.0 + math.sqrt(100.0 - 0.1 ** 2)            # 27.9995
    g.anchors = [Anchor(x_head=19.95, z_head=z, length=8.05,
                        T_allow=50.0, inclination=0.0)]
    hits = line_crossings(slip, 10.0, 28.0, 19.95, z, 1.0, 0.0, 8.05)
    assert [h[1] for h in hits] == pytest.approx([x_c], abs=1e-9)
    # a polyline is crossed segment by segment
    poly = PolylineSlipSurface([(10, 0), (16, -2), (24, 2), (30, 10)])
    (h,) = line_crossings(poly, 10.0, 30.0, 20.0, 9.0, 1.0, -0.5 / 1.0,
                          30.0)
    assert h[2] == pytest.approx(2 + (h[1] - 24) * 8 / 6)
    assert level_crossings(poly, 10.0, 30.0, 6.0) == [
        (pytest.approx(27.0), 1.0)]
    assert level_crossings(poly, 10.0, 30.0, -1.0) == [
        (pytest.approx(13.0), -1.0), (pytest.approx(18.0), 1.0)]


def test_legacy_force_without_an_elevation_is_refused():
    """D37: a force with no elevation has no line of action (Eq. 6.70);
    it was left out without notice."""
    with pytest.raises(ValueError, match="reinforcement_elevation"):
        _slope(reinforcement_force=80.0)
    g = _slope(reinforcement_force=80.0, reinforcement_elevation=14.0)
    (f,) = _resolve(g).forces
    assert f.T == 80.0 and f.kind == "geosynthetic"


def test_support_on_a_slice_boundary_acts_on_one_slice():
    """GM-353: a layer on a material boundary meets the slip surface on a
    slice boundary. It acts on the slice that holds the part of it inside
    the sliding mass, whatever the rounding: the same factor of safety at
    the radius and 1e-8 m either side."""
    bounds = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    assert slice_index_for(bounds, 1.0, -1.0) == 0
    assert slice_index_for(bounds, 1.0, +1.0) == 1
    assert slice_index_for(bounds, 1.0 + 1e-9, -1.0) == 0
    assert slice_index_for(bounds, 1.0 - 1e-9, +1.0) == 1
    assert slice_index_for(bounds, 1.0, 0.0) == 0
    assert slice_index_for(bounds, 1.5, +1.0) == 1
    assert slice_index_for(bounds, 3.0, +1.0) == 2
    assert slice_index_for(bounds, 4.0, 0.0) is None

    g = SlopeGeometry(
        surface_points=[(0, 10), (20, 10), (40, 20), (70, 20)],
        soil_layers=[
            SlopeSoilLayer(name="fill", top_elevation=20.0,
                           bottom_elevation=14.0, gamma=19.0, phi=32.0,
                           c_prime=0.0),
            SlopeSoilLayer(name="clay", top_elevation=14.0,
                           bottom_elevation=-15.0, gamma=17.0, phi=22.0,
                           c_prime=12.0)],
        geosynthetics=[Geosynthetic(elevation=14.0, T_allow=150.0)])
    values = []
    for dr in (-1e-8, 0.0, 1e-8):
        slip = CircularSlipSurface(xc=30.0, yc=32.0, radius=26.0 + dr)
        slices = build_slices(g, slip, 30)
        (f,) = compute_reinforcement_forces(g, slip,
                                            *slip.find_entry_exit(g))
        # the crossing is a slice boundary (the layer top is a slice edge)
        assert min(abs(s.x_right - f.x) for s in slices) < 1e-6
        values.append(analyze_slope(g, slip_surface=slip, method="spencer",
                                    n_slices=30).FOS)
    assert values[0] == pytest.approx(values[1], abs=1e-6)
    assert values[2] == pytest.approx(values[1], abs=1e-6)


def test_a_search_states_the_method_once():
    """Every result with supports names Method A or Method B and the
    measure of force (Eqs. 8.1 to 8.4): a search says it once."""
    g = _slope(geosynthetics=[Geosynthetic(elevation=14.0, T_allow=60.0)])
    res = search_critical_surface(g, method="bishop", nx=4, ny=4)
    said = [c for c in res.choices
            if c["what"].startswith("how the support")]
    assert len(said) == 1 and "Method A" in said[0]["chosen"]
    assert not any(c["what"].startswith("how the support")
                   for c in (res.critical.choices or []))
    assert not any(c["what"].startswith("how the support") for c in
                   analyze_slope(_slope(), slip_surface=_CIRCLE).choices)


def test_stabilizing_pile_force_is_against_the_sliding():
    g = _slope(stabilizing_piles=[StabilizingPile(x=30.0,
                                                  shear_capacity=100.0)])
    (f,) = _resolve(g).forces
    assert (f.dir_x, f.dir_z) == (1.0, 0.0)

"""GM-140 step 3: the printed acceptability checks.

The rule. Once a method has solved a surface and the surface is one sliding
mass, every further test is a printed acceptability check whose finding is
reported with the surface: which check, which slice, what value. A search
keeps a flagged surface by default and says so; refusing on a check is a
stated option. Nothing is left out silently, and circles and polylines meet
the same checks.

Every geometry here is made up. Each test's docstring gives the printed
rule and its page.
"""
import math

import pytest

from slope_stability.acceptability import (
    CHECKS, C10B_REMEDIES, surface_findings, flags,
)
from slope_stability.analysis import analyze_slope, search_critical_surface
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.methods import (
    bishop_fos, fellenius_fos, negative_term,
)
from slope_stability.reinforcement import Anchor
from slope_stability.slices import build_slices
from slope_stability.slip_surface import (
    CircularSlipSurface, PolylineSlipSurface,
)
from slope_stability.solution import solve_method


def _slope(c=5.0, phi=30.0, bottom=-10.0, **kw):
    """A made-up 10 m slope at 1H:1V in one soil."""
    return SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (30, 10), (60, 10)],
        soil_layers=[SlopeSoilLayer("fill", top_elevation=10,
                                    bottom_elevation=bottom, gamma=19,
                                    phi=phi, c_prime=c)], **kw)


def _finding(result, check):
    got = [f for f in result.acceptability if f["check"] == check]
    assert got, f"no {check} finding in {result.acceptability}"
    return got[0]


# ---------------------------------------------------------------------------
# Item 1: m_alpha below 0.2 (OWN-102)
# ---------------------------------------------------------------------------

def test_a_search_keeps_and_flags_a_surface_with_low_m_alpha():
    """Whitman and Bailey (1967, p. 487, Eq. 5): the user is warned if the
    denominator becomes small, "for example, if cos(theta) + tan(phi)
    sin(theta) / F < 0.2 ... the user must examine the result in detail".
    A warning, not a rejection (OWN-102): a Bishop search keeps the five
    such circles it meets on this slope, flags them and counts them by
    family. With the refusal option it leaves the same five out, which is
    the count the module gave before this step."""
    geom = _slope(c=1.0, phi=36.0, bottom=-12.0)
    kept = search_critical_surface(geom, method="bishop")
    assert kept.n_rejected_m_alpha == 0
    by_family = {r["family"]: r for r in kept.trial_flags}
    assert by_family["circle_above_toe"]["by_check"] == {"low_m_alpha": 5}
    assert by_family["circle_above_toe"]["n_flagged"] == 5
    assert by_family["circular"]["n_flagged"] == 0

    refused = search_critical_surface(geom, method="bishop",
                                      low_m_alpha="refuse")
    assert refused.n_rejected_m_alpha == 5          # the earlier count
    assert sum(r["n_flagged"] for r in refused.trial_flags) == 0
    # the five are far above the minimum, so the headline is the same
    assert kept.critical.FOS == pytest.approx(refused.critical.FOS,
                                              abs=1e-9)
    # and the search scored five more trials than it refused before
    n_kept = sum(r["n_scored"] for r in kept.trial_flags)
    n_ref = sum(r["n_scored"] for r in refused.trial_flags)
    assert n_kept == n_ref + 5

    # choices says what was done, either way
    names = {c["what"]: c for c in kept.choices}
    assert "kept and flagged: 5" in str(
        names["slip surfaces with m_alpha below 0.2"]["chosen"])
    names = {c["what"]: c for c in refused.choices}
    assert "5 left out" in str(
        names["slip surfaces left out for m_alpha"]["chosen"])


def test_a_flagged_circle_names_the_slice_and_gives_both_values():
    """One of the circles the search above keeps. The finding names the
    slice and its m_alpha (Whitman and Bailey 1967, p. 487), and Bishop's
    value is reported beside the Ordinary Method of Slices value for the
    same circle (Duncan 1996, p. 580)."""
    geom = _slope(c=1.0, phi=36.0, bottom=-12.0)
    res = analyze_slope(geom, xc=42.928, yc=10.081, radius=15.072,
                        method="bishop")
    assert res.flags == ["low_m_alpha"]
    f = _finding(res, "low_m_alpha")
    assert [s["slice"] for s in f["slices"]] == [30]
    assert f["slices"][0]["m_alpha"] < 0.2
    assert f["slices"][0]["against_sliding"] is False   # a steep entry
    assert "Whitman and Bailey" in f["source"]
    # both values, Bishop's own the higher here: it stands
    cmp_ = _finding(res, "bishop_against_oms")
    assert cmp_["compared"] and cmp_["taken"] == "bishop"
    assert res.FOS_bishop == pytest.approx(res.FOS)
    assert res.FOS_fellenius < res.FOS_bishop
    assert any("below 0.2 on slice 30" in w for w in res.warnings)
    d = res.to_dict()
    assert d["flags"] == ["low_m_alpha"]
    assert d["acceptability"][0]["check"] == "low_m_alpha"


_STEEP_EXIT = [(17, 0), (18, -2.5), (23, -1), (34, 10)]


def test_a_steep_exit_polyline_is_analysed_and_names_em_c10b_remedies():
    """EM 1110-2-1902 C-10b (pp. C-34 to C-36) prints four remedies for a
    steep exit, in order: flatten the toe of the surface (Eq. C-32),
    change the side-force inclination, use the Ordinary Method of Slices,
    a passive strength as cohesion with phi = 0. The warning names them;
    none is applied: the polyline is analysed as given, and it meets the
    same m_alpha check a circle does."""
    geom = _slope(c=1.0, phi=36.0, bottom=-12.0)
    slip = PolylineSlipSurface(points=_STEEP_EXIT)
    # (GM-344: Morgenstern-Price; Spencer's method, asked for here before,
    # has no solution on this polyline for lambda from 0 to 1.5, and the
    # value it read was not a root)
    res = analyze_slope(geom, slip_surface=slip,
                        method="morgenstern_price")
    assert res.FOS is not None
    assert "low_m_alpha" in res.flags
    f = _finding(res, "low_m_alpha")
    assert all(s["against_sliding"] for s in f["slices"])
    assert f["remedies"].startswith("EM 1110-2-1902 C-10b")
    w = [w for w in res.warnings if "below 0.2" in w][0]
    for words in ("flatten the toe", "Eq. C-32", "side-force inclination",
                  "Ordinary Method of Slices", "cohesion with phi = 0",
                  "None is applied"):
        assert words in w
    # the surface is the one given: nothing was changed on it
    assert res.slip_points == _STEEP_EXIT


def test_every_exported_search_reports_its_governing_surface_s_flags():
    """The step 3 review's defect 2: the governing surface's flags, the
    warning and the lowest surface with no flag come from one place
    (`search_supplements.report_acceptability`), for every exported search
    that returns a SearchResult, not only `search_critical_surface`."""
    import slope_stability as ss
    for name in ("grid_search", "search_noncircular", "search_pso",
                 "search_weak_layer_biased", "search_entry_exit",
                 "search_de"):
        assert getattr(getattr(ss, name), "reports_acceptability", False), \
            name


def test_search_noncircular_called_directly_reports_the_flags():
    """ACADS 1(a) (printed 1.00), a 200-trial random Spencer search called
    directly: its lowest solved surface (1.092 when measured) has tensile
    interslice forces and the line of thrust outside the mass. The result
    says so (``critical_flags``, the warning) and gives the lowest scored
    surface with no flag beside it (1.096); before, a direct call returned
    it with ``critical_flags == []``. (Until GM-344 the lowest was a
    zig-zag at 0.633 with a base in tension as well: a crossing at a
    negative lambda, which the scan no longer looks for.)"""
    from slope_stability.search import search_noncircular
    g = SlopeGeometry(
        surface_points=[(20.0, 25.0), (30.0, 25.0), (50.0, 35.0),
                        (70.0, 35.0)],
        soil_layers=[SlopeSoilLayer(name="soil", top_elevation=35.0,
                                    bottom_elevation=20.0, gamma=20.0,
                                    phi=19.6, c_prime=3.0)])
    res = search_noncircular(
        g, x_entry_range=(20.0, 32.0), x_exit_range=(46.0, 68.0),
        n_trials=200, n_points=6, n_slices=25, seed=42, method="spencer")
    assert {"side_tension", "thrust_outside"} <= set(res.critical_flags)
    # GM-344: 0.633 -> 1.092; printed 1.00 (ACADS 1(a)), +9.2 %
    assert res.critical.FOS == pytest.approx(1.092, abs=0.002)
    assert res.critical.acceptability
    low = res.lowest_unflagged
    assert low["FOS"] is not None and low["FOS"] > res.critical.FOS
    assert low["family"] == "noncircular"
    (w,) = [w for w in res.warnings if "acceptability checks" in w]
    assert f"FOS {res.critical.FOS:.3f}" in w
    assert f"FOS {low['FOS']:.3f}" in w
    assert [r["family"] for r in res.trial_flags] == ["noncircular"]


def test_phi_zero_slices_are_exempt_by_the_modules_own_reasoning():
    """No printed source exempts phi' = 0 slices from the m_alpha check;
    the module's reasoning is that there m_alpha does not depend on F and
    the strength does not depend on the normal force. The exemption stays
    the default, stated as the module's own; 'every_slice' is the option
    (OWN-102). The circle's centre is 8 m below the crest where it ends, a
    reverse-curvature circle: since GM-140 step 4 it is a surface only with
    a crack asked for at least that deep, so an 8 m crack is given, and the
    surface ends where its arc turns vertical."""
    geom = SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (30, 10), (60, 10)],
        soil_layers=[SlopeSoilLayer("clay", top_elevation=10,
                                    bottom_elevation=-10, gamma=18,
                                    cu=40.0, analysis_mode="undrained")],
        tension_crack_depth=8.0)
    kw = dict(xc=20.0, yc=2.0, radius=10.0, method="bishop")
    default = analyze_slope(geom, **kw)
    every = analyze_slope(geom, m_alpha_rule="every_slice", **kw)
    assert "low_m_alpha" not in default.flags
    assert "low_m_alpha" in every.flags
    assert default.FOS == pytest.approx(every.FOS)


# ---------------------------------------------------------------------------
# Item 1: Bishop against the Ordinary Method of Slices (Duncan 1996, p. 580)
# ---------------------------------------------------------------------------

def test_bishop_below_oms_takes_the_oms_value_and_says_so():
    """Duncan (1996, p. 580): calculate the Ordinary Method of Slices
    value for each circle when Bishop's method is used; where Bishop's F is
    below it, Bishop's solution has numerical problems and the Ordinary
    Method of Slices value is the better answer. On this small circle
    Bishop's value is the lower: the Ordinary Method of Slices value is
    taken, both are reported and the result says so."""
    geom = _slope()
    slip = CircularSlipSurface(18, 12, 10)
    slices = build_slices(geom, slip, 30)
    b, o = bishop_fos(slices, slip), fellenius_fos(slices, slip)
    assert b < o - 1e-4                         # the premise
    res = analyze_slope(geom, xc=18, yc=12, radius=10, method="bishop")
    assert res.FOS == pytest.approx(o)
    assert res.FOS_bishop == pytest.approx(b)
    assert res.FOS_fellenius == pytest.approx(o)
    assert "Ordinary Method of Slices" in res.method_used
    assert "Duncan 1996, p. 580" in res.method_used
    assert res.flags == ["bishop_below_oms"]
    f = _finding(res, "bishop_below_oms")
    assert f["taken"] == "oms" and f["FOS_bishop"] < f["FOS_oms"]
    assert any("better answer" in w for w in res.warnings)
    # the comparison column of another method's result is Bishop's own
    other = analyze_slope(geom, xc=18, yc=12, radius=10, method="janbu",
                          compare_methods=True)
    assert other.FOS_bishop == pytest.approx(b)


def test_bishop_equal_to_oms_on_a_phi_zero_circle_is_not_flagged():
    """EM 1110-2-1902 C-2b (p. C-16): the Ordinary Method of Slices equals
    the rigorous methods when phi = 0. The two are then the same sum and
    differ by rounding only, which is not "Bishop's below": the comparison
    is made to the tolerance Bishop's iteration is solved to."""
    geom = SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (30, 10), (60, 10)],
        soil_layers=[SlopeSoilLayer("clay", top_elevation=10,
                                    bottom_elevation=-10, gamma=18,
                                    cu=40.0, analysis_mode="undrained")])
    res = analyze_slope(geom, xc=22, yc=16, radius=16, method="bishop")
    assert res.FOS == pytest.approx(res.FOS_fellenius, rel=1e-9)
    assert "bishop_below_oms" not in res.flags
    assert res.method_used == ""
    assert _finding(res, "bishop_against_oms")["taken"] == "bishop"


def test_a_circle_with_supports_says_the_comparison_was_not_made():
    """OWN-102's limit (2026-11-13): Duncan, Wright and Brandon (2014,
    p. 93) print no Ordinary Method of Slices equation with known forces,
    so on a circle a support crosses Bishop's value is not compared with
    the Ordinary Method of Slices value, and the result says so and why."""
    geom = _slope(anchors=[Anchor(x_head=25, z_head=5, length=20,
                                  T_allow=50.0)])
    res = analyze_slope(geom, xc=24, yc=14, radius=15, method="bishop")
    assert res.reinforcements                       # the anchor crosses
    f = _finding(res, "bishop_against_oms")
    assert f["compared"] is False and f["flag"] is False
    assert "supports cross the circle" in f["reason"]
    assert "p. 93" in f["reason"]
    assert res.FOS_fellenius is None
    assert res.method_used == ""
    assert any("was not compared" in w for w in res.warnings)


def test_a_seismic_coefficient_says_the_comparison_was_not_made():
    """The same limit (OWN-102, 2026-11-13) for a seismic coefficient."""
    res = analyze_slope(_slope(kh=0.1), xc=24, yc=14, radius=15,
                        method="bishop")
    f = _finding(res, "bishop_against_oms")
    assert f["compared"] is False
    assert "seismic" in f["reason"]


def test_a_bishop_search_reports_both_values_on_its_critical_circle():
    """Duncan (1996, p. 580), in a search: the Ordinary Method of Slices
    value is computed for every Bishop circle; the governing circle carries
    both, and ``choices`` says on how many circles it was taken."""
    res = search_critical_surface(_slope(), method="bishop", nx=4, ny=4)
    c = res.critical
    assert c.FOS_bishop is not None and c.FOS_fellenius is not None
    assert c.FOS == pytest.approx(max(c.FOS_bishop, c.FOS_fellenius),
                                  rel=1e-6)
    names = {ch["what"]: ch for ch in res.choices}
    entry = names["Bishop's value against the Ordinary Method of Slices "
                  "value"]
    assert "Duncan (1996, p. 580)" in entry["basis"]


# ---------------------------------------------------------------------------
# Item 2: the frictional term as printed (D21)
# ---------------------------------------------------------------------------

def _artesian():
    """The made-up slope with a pressure head 6 m above the toe's level
    under the toe side (x < 24): u b exceeds W on the slices there."""
    pts = [(x, z, 9.81 * (6.0 - z) if x < 24 else 0.0)
           for x in range(0, 61, 2) for z in (-10, -5, 0, 5, 10)]
    return _slope(pore_pressure_points=pts)


def test_bishop_uses_w_minus_ub_as_printed_and_names_the_slices():
    """Bishop (1955) Eq. 13, p. 10 writes the frictional term as
    W(1 - B) = W - u b with no lower limit; Whitman and Bailey (1967,
    p. 487) ask that the user be warned if the numerator becomes negative
    for any slice. By default the term is used as printed and the slices
    are named; setting it to zero is the stated option. The two values
    differ as the arithmetic says: each satisfies Eq. 13 with its own
    term."""
    geom = _artesian()
    slip = CircularSlipSurface(24, 14, 15)
    slices = build_slices(geom, slip, 30)
    printed = analyze_slope(geom, xc=24, yc=14, radius=15, method="bishop")
    zero = analyze_slope(geom, xc=24, yc=14, radius=15, method="bishop",
                         negative_friction_term="zero")
    f = _finding(printed, "negative_frictional_term")
    named = [s["slice"] for s in f["slices"]]
    by_hand = [i + 1 for i, s in enumerate(slices)
               if s.weight + s.surcharge_force
               - s.pore_pressure * s.width < 0]
    assert named == by_hand and named
    assert f["rule"] == "as_printed"
    assert _finding(zero, "negative_frictional_term")["rule"] == "zero"
    assert [s["slice"] for s in _finding(
        zero, "negative_frictional_term")["slices"]] == named
    assert printed.FOS_bishop < zero.FOS_bishop

    def eq13(F, clamp):
        num = den = 0.0
        for s in slices:
            W = s.weight + s.surcharge_force
            term = W - s.pore_pressure * s.width
            if clamp:
                term = max(term, 0.0)
            tp = math.tan(math.radians(s.phi))
            # alpha is positive in the sliding direction here (the slope
            # rises to the right and slides to the left)
            m = math.cos(s.alpha) + math.sin(s.alpha) * tp / F
            num += (s.c * s.width + term * tp) / m
            den += W * (s.x_mid - slip.xc) / slip.radius
        return num / abs(den)

    assert eq13(printed.FOS_bishop, False) == pytest.approx(
        printed.FOS_bishop, abs=2e-4)
    assert eq13(zero.FOS_bishop, True) == pytest.approx(
        zero.FOS_bishop, abs=2e-4)


def test_the_oms_uses_w_cos_alpha_minus_ul_as_printed():
    """Bishop (1955) Eq. 9, p. 10: the Ordinary Method of Slices' normal
    force is N' = W cos(alpha) - u l, with no lower limit. As printed and
    set to zero differ by exactly the negative terms' friction over the
    driving sum."""
    geom = _artesian()
    slip = CircularSlipSurface(24, 14, 15)
    slices = build_slices(geom, slip, 30)
    f_printed = fellenius_fos(slices, slip)
    with negative_term("zero"):
        f_zero = fellenius_fos(slices, slip)
    driving = abs(sum((s.weight + s.surcharge_force)
                      * (s.x_mid - slip.xc) / slip.radius for s in slices))
    lost = sum(min((s.weight + s.surcharge_force) * math.cos(s.alpha)
                   - s.pore_pressure * s.base_length, 0.0)
               * math.tan(math.radians(s.phi)) for s in slices)
    assert lost < 0
    assert f_printed - f_zero == pytest.approx(lost / driving, rel=1e-9)
    res = analyze_slope(geom, xc=24, yc=14, radius=15, method="fellenius")
    assert res.FOS == pytest.approx(f_printed)
    assert "negative_frictional_term" in res.flags
    assert "Eq. 9" in _finding(res, "negative_frictional_term")["text"]


def test_an_unknown_option_value_is_refused():
    geom = _slope()
    with pytest.raises(ValueError, match="negative_friction_term"):
        analyze_slope(geom, xc=24, yc=14, radius=15,
                      negative_friction_term="clamp")
    with pytest.raises(ValueError, match="negative_friction_term"):
        search_critical_surface(geom, negative_friction_term="clamp")
    with pytest.raises(ValueError, match="low_m_alpha"):
        search_critical_surface(geom, low_m_alpha="reject")


# ---------------------------------------------------------------------------
# Item 3: compression and the line of thrust (D22, D50)
# ---------------------------------------------------------------------------

def test_spencer_reports_tension_on_bases_and_sides_at_the_crest():
    """EM 1110-2-1902 C-10a (pp. C-33 to C-34): calculated normal and
    interslice forces should be examined; tensile forces should not exist
    on the sides or bottom of slices. A Spencer circle in a soil with
    cohesion is in tension where it enters at the crest: the result names
    the slices and the slice sides. The tension is reported; no crack is
    added (that is EM C-10c's remedy, another step)."""
    geom = _slope()
    res = analyze_slope(geom, xc=20, yc=10, radius=10, method="spencer",
                        include_slice_data=True)
    assert res.FOS is not None
    assert "base_tension" in res.flags and "side_tension" in res.flags
    base = _finding(res, "base_tension")
    side = _finding(res, "side_tension")
    n = res.n_slices
    # at the crest end: the last slices and sides (high x)
    assert max(s["slice"] for s in base["slices"]) == n
    assert all(s["N_eff_kN_per_m"] < 0 for s in base["slices"])
    assert all(s["E_kN_per_m"] < 0 for s in side["sides"])
    assert all(s["x"] > 25.0 for s in side["sides"])
    # the same slices the slice table shows in tension
    table = [i + 1 for i, s in enumerate(res.slice_data) if s.N_eff_kN < 0]
    assert [s["slice"] for s in base["slices"]] == table
    assert "C-10a" in base["source"] and "C-10a" in side["source"]
    # the equations as printed there, and the option named (GM-344)
    assert base["rule"] == "as_printed"
    assert "base_in_tension='no_friction'" in base["text"]
    assert res.tension_crack_depth == 0.0       # nothing was added


def test_a_line_of_thrust_outside_the_mass_is_said_and_not_hidden():
    """EM 1110-2-1902 C-10a (pp. C-33 to C-34): the line of thrust should
    be within the sliding mass. The figure's line is clipped to the mass
    for display (D50); the result keeps the line as solved, says at which
    slice sides it left the mass, and the figure's legend says so."""
    geom = _slope()
    res = analyze_slope(geom, xc=18, yc=8, radius=10, method="spencer",
                        include_slice_data=True)
    assert "thrust_outside" in res.flags
    f = _finding(res, "thrust_outside")
    raw = dict(res.thrust_line_unclipped)
    shown = dict(res.thrust_line)
    for s in f["sides"]:
        lo, hi = sorted((s["z_slip"], s["z_ground"]))
        assert s["z_thrust"] < lo or s["z_thrust"] > hi
        x = min(raw, key=lambda v: abs(v - s["x"]))
        assert lo - 1e-6 <= shown[x] <= hi + 1e-6       # drawn clipped
        assert not (lo <= raw[x] <= hi)                 # as solved
    from slope_stability.plotting import _thrust_label
    assert "outside the sliding mass" in _thrust_label(res)
    assert f"{len(f['sides'])} slice side" in _thrust_label(res)


def test_circles_and_polylines_meet_the_same_checks():
    """The rule: circles and polylines meet the same checks (EM
    1110-2-1902 C-10a). The same arc, given as a circle and as the polyline
    through its slice corners, is put through the same function and is
    found in tension at the crest either way."""
    geom = _slope()
    circ = CircularSlipSurface(20, 10, 10)
    sl = build_slices(geom, circ, 30)
    pts = [(sl[0].x_left, geom.ground_elevation_at(sl[0].x_left))]
    pts += [(s.x_right, circ.slip_elevation_at(s.x_right)) for s in sl[:-1]]
    pts += [(sl[-1].x_right, geom.ground_elevation_at(sl[-1].x_right))]
    poly = PolylineSlipSurface(points=pts)
    sl_p = build_slices(geom, poly, 30)
    a = solve_method(sl, circ, "spencer")
    b = solve_method(sl_p, poly, "spencer")
    assert a.solved and b.solved
    fa = flags(surface_findings(geom, circ, sl, a))
    fb = flags(surface_findings(geom, poly, sl_p, b))
    # (the two solves take moments about different axes, so their states
    # are not identical; the tension at the crest is found on both)
    assert "base_tension" in fa and "base_tension" in fb
    assert "side_tension" in fa and "side_tension" in fb
    assert set(fa) <= set(CHECKS) and set(fb) <= set(CHECKS)


# ---------------------------------------------------------------------------
# Item 4: in a search
# ---------------------------------------------------------------------------

def test_a_search_gives_the_flags_of_its_critical_surface():
    """EM 1110-2-1902 C-10a and UFC 3-220-10 pp. 361 to 362: the engineer
    examines the forces of the critical surface. A Spencer search of a
    slope with cohesion ends on a circle in tension at the crest: the
    result gives the flags it carries, how many trials carried each flag
    by family, and the lowest factor of safety among the scored surfaces
    that carry none, beside the headline. The headline stays the lowest
    solved surface."""
    res = search_critical_surface(_slope(), method="spencer", nx=4, ny=4)
    assert res.critical_flags
    assert res.critical.flags == res.critical_flags
    assert set(res.critical_flags) <= set(CHECKS)
    for rec in res.trial_flags:
        assert rec["n_flagged"] <= rec["n_scored"]
        assert set(rec["by_check"]) <= set(CHECKS)
    low = res.lowest_unflagged
    assert low is not None
    if low["FOS"] is None:
        assert "without a flag" in low["note"]
    else:
        assert low["FOS"] >= res.critical.FOS - 1e-9
        assert low["method"] == "spencer"
    w = [w for w in res.warnings if "printed acceptability checks" in w]
    assert len(w) == 1 and "kept and reported, flagged" in w[0]
    d = res.to_dict()
    assert d["critical_flags"] == res.critical_flags
    assert "lowest_unflagged" in d and "trial_flags" in d
    assert d["critical"]["flags"] == res.critical_flags


def test_an_unflagged_critical_surface_has_no_second_number():
    """A search whose governing surface meets every check reports no
    lowest unflagged surface: the headline is it."""
    res = search_critical_surface(_slope(c=1.0, phi=36.0, bottom=-12.0),
                                  method="bishop")
    assert res.critical_flags == []
    assert res.lowest_unflagged is None
    assert "lowest_unflagged" not in res.to_dict()


# ---------------------------------------------------------------------------
# Item 5: the noncircular screens are counted, each by name
# ---------------------------------------------------------------------------

def test_noncircular_screen_refusals_are_counted_by_screen():
    """The noncircular screens (slice count, span, agreement between
    moment axes and between slicings, a FOS below 1.5 on a jagged surface)
    are not found in print: EM 1110-2-1902 C-10a prints the checks on
    forces and the line of thrust instead. Those kept are counted in the
    result one by one, with how many of the surfaces they refused the
    printed checks also flag, and named in ``choices`` as the module's
    own."""
    from slope_stability.search import NONCIRCULAR_SCREENS
    res = search_critical_surface(
        _slope(), method="spencer", surface_type="noncircular",
        n_trials=150, seed=4, x_entry_range=(8, 22), x_exit_range=(28, 45))
    sr = res.screen_refusals
    assert sr and set(sr) <= set(NONCIRCULAR_SCREENS)
    geometry = sum(sr.get(k, {"n": 0})["n"] for k in ("slice_count", "span"))
    assert geometry == res.n_rejected_geometry
    assert sr.get("jagged_below_1_5", {"n": 0})["n"] == res.n_rejected_jagged
    for k in ("one_moment_axis", "one_slicing", "jagged_below_1_5"):
        if k in sr:
            assert 0 <= sr[k]["n_flagged"] <= sr[k]["n"]
    names = {c["what"]: c for c in res.choices}
    entry = names["noncircular trial surfaces refused by the module's "
                  "screens"]
    assert "not found in print" in entry["basis"]


# ---------------------------------------------------------------------------
# Item 6: one below-base test (D19)
# ---------------------------------------------------------------------------

def test_the_toe_circle_family_uses_the_common_below_base_test(monkeypatch):
    """D19 (GM-128's rule): the toe-circle family had a below-base test of
    its own; it now asks the one test every family uses. With that test
    made to refuse everything, the family scores no circle at all (before,
    its own test let circles through to the scorer)."""
    import slope_stability.search as search
    from slope_stability.search_supplements import search_toe_circles
    monkeypatch.setattr(search, "_dips_below_base",
                        lambda geom, slip, touch=None: True)
    out = search_toe_circles(_slope(), method="bishop")
    assert out["n_evaluated"] == 0
    assert out["FOS"] >= 999.0


# ---------------------------------------------------------------------------
# Every caller
# ---------------------------------------------------------------------------

def test_the_drawdown_search_takes_the_option_and_refuses_an_unknown_one():
    from slope_stability.rapid_drawdown import search_rapid_drawdown
    geom = SlopeGeometry(
        surface_points=[(0, 0), (40, 16), (60, 16)],
        soil_layers=[SlopeSoilLayer("fill", top_elevation=16,
                                    bottom_elevation=-4, gamma=20,
                                    phi=32, c_prime=4, R_c=20, R_phi=18)])
    with pytest.raises(ValueError, match="low_m_alpha"):
        search_rapid_drawdown(geom, 14, 4, low_m_alpha="reject")


def test_the_remedies_are_em_c10bs_four_in_the_printed_order():
    """EM 1110-2-1902 C-10b (pp. C-34 to C-36), the order printed."""
    order = [C10B_REMEDIES.index(w) for w in (
        "flatten the toe", "side-force inclination",
        "use the Ordinary Method of Slices", "passive strength")]
    assert order == sorted(order)
    assert "Eq. C-32" in C10B_REMEDIES
    assert "-(45 deg - phi'/2)" in C10B_REMEDIES

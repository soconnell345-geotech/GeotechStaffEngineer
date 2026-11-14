"""
The GLE solver: a root, or no number (GM-140; BACKLOG GM-344).

The solver behind Spencer's method, Morgenstern-Price and GLE reports a
factor of safety only where ONE state meets both the force and the moment
equation at one lambda. A change of order of the two factors of safety
between two branches of solutions is a jump, not a crossing. A base in
tension is treated as printed by default; the module's device (the slice
taken frictionless) is an option applied in every equation.

Every value here is made up. The roots are checked against a second
computation written apart from ``gle.py``
(``module_work/gm140/gle_root_spencer_script.py``: Spencer's two equations
in their Q form, no E, no X, no lambda, no fitted axis).
"""

import importlib.util
import math
import os

import pytest

from slope_stability import gle
from slope_stability.analysis import analyze_slope, search_critical_surface
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.gle import (
    BASE_TENSION_RULES, base_tension, base_tension_in_force, gle_fos,
)
from slope_stability.slices import build_slices
from slope_stability.slip_surface import (
    CircularSlipSurface, PolylineSlipSurface,
)
from slope_stability.solution import solve_method

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "..", "..", "module_work", "gm140",
                       "gle_root_spencer_script.py")


@pytest.fixture(scope="module")
def own():
    """The second computation of Spencer's equations."""
    spec = importlib.util.spec_from_file_location("gle_root_own", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BETA = math.degrees(math.atan(0.5))             # a 2H:1V face
GAMMA, PHI, GAMMA_W = 19.5, 34.0, 9.81
#: (gamma' / gamma_sat) tan(phi') / tan(beta): the infinite slope with
#: seepage parallel to the face, c' = 0 (FHWA NHI-06-088 Sec. 6.3.2,
#: Eq. 6-10, p. 6-8)
CLOSED_WET = (GAMMA - GAMMA_W) / GAMMA * math.tan(math.radians(PHI)) / 0.5
#: tan(phi') / tan(beta): the same slope dry (the same section)
CLOSED_DRY = math.tan(math.radians(PHI)) / 0.5


def _spencer(own, water, rule, n_slices=60):
    geom, slip, slices = own.case(water, n_slices)
    with base_tension(rule):
        sol = solve_method(slices, slip, "spencer")
    return geom, slip, slices, sol


def _system(own, water, rule, f_name="constant"):
    _, _, slices = own.case(water)
    ns, _ = gle._normalize(slices)
    with base_tension(rule):
        return gle._GLESystem(ns, gle._fit_axis_point(ns), f_name, 1e-5), \
            slices


def _circle_case():
    """An ordinary made-up circle whose crest slice is in tension."""
    geom = SlopeGeometry(
        surface_points=[(-10, 0), (0, 0), (20, 10), (40, 10)],
        soil_layers=[SlopeSoilLayer(
            name="fill", top_elevation=10, bottom_elevation=-20, gamma=19,
            phi=25, c_prime=10, ru=0.25)])
    slip = CircularSlipSurface(6, 18, 19.5)
    return geom, slip, build_slices(geom, slip, 30)


# ---------------------------------------------------------------------------
# The case: r_u is Eq. 6-10's pore pressure
# ---------------------------------------------------------------------------

def test_the_r_u_of_the_case_is_the_pore_pressure_of_eq_6_10(own):
    """FHWA NHI-06-088 Sec. 6.3.2 (p. 6-8) is the infinite slope with seepage
    parallel to the face, where the pore pressure at depth z (vertical) is
    gamma_w z cos^2(beta). The case gives it as r_u = (gamma_w / gamma)
    cos^2(beta), so there is no water line and no pond: on every slice of
    the plane u = r_u gamma z is that pressure."""
    _, _, slices = own.case("ru")
    cos2 = math.cos(math.radians(BETA)) ** 2
    for s in slices[1:-1]:                  # the plane, 2 m below the face
        z = (s.weight / s.width) / GAMMA
        assert z == pytest.approx(2.0, abs=1e-6)
        assert s.pore_pressure == pytest.approx(GAMMA_W * z * cos2,
                                                rel=1e-9)


# ---------------------------------------------------------------------------
# Ruling 1: a reported factor of safety is a root
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("water", ["ru", "dry"])
@pytest.mark.parametrize("rule", BASE_TENSION_RULES)
def test_spencer_reports_a_root_of_spencers_equations(own, water, rule):
    """Spencer (1967, Eqs. 7 and 8, p. 15): F and theta meet force and
    moment equilibrium together. Whatever the solver reports is a root of
    the second computation's two equations, and within 0.1 % of that
    computation's own root.

    Before (master 9c73795 and 3d553a8, 2026-11-14) the r_u case read
    1.0887 at 31.13 deg, reported as converged (0.8901 at 25.41 deg with
    gle_fos's default tol of 1e-4): not a root (the old test accepted the
    two factors of safety within 1e-3 of each other, which the fitted
    moment axis 15 km away gives at every lambda, and reported their
    mean). The record said 0.6065 at 18.00 deg; neither master tree
    reproduces that (the review's defect 4)."""
    _, _, slices, sol = _spencer(own, water, rule)
    assert sol.solved
    d = own.from_module(slices)
    device = rule == "no_friction"
    near = own._newton(d, sol.fos, math.radians(sol.param), device)
    assert near is not None
    assert own.is_root(d, near[0], near[1], device)
    assert sol.fos == pytest.approx(near[0], rel=1e-3)
    assert sol.param == pytest.approx(math.degrees(near[1]), abs=0.05)


def test_what_master_reported_is_not_a_root(own):
    """The numbers master (9c73795 and 3d553a8) reported for the r_u case
    (Spencer 1.0887 at 31.13 deg through analyze_slope and solve_method;
    0.8901 at 25.41 deg from gle_fos at its default tol), each
    "converged", do not meet Spencer's equations: the force closure and
    the moment residual there are far from zero (about 3e-3 and 7e-3 of
    their scales at 1.0887)."""
    _, _, slices = own.case("ru")
    d = own.from_module(slices)
    for f_master, theta in ((1.0887, 31.128), (0.8901, 25.408)):
        assert not own.is_root(d, f_master, math.radians(theta), False,
                               rel=1e-4)
    _, _, _, sol = _spencer(own, "ru", "as_printed")
    assert abs(sol.fos - 1.0887) > 0.01 and abs(sol.fos - 0.8901) > 0.01


def test_the_reported_factor_is_the_state_s_own(own):
    """One state at one lambda: the factor reported is the one that closes
    the forces of that state, and the same state is in moment equilibrium
    (no mean of two factors)."""
    _, slip, slices = own.case("ru")
    res = gle_fos(slices, slip, f_interslice="constant", tol=1e-5)
    assert res.converged
    assert res.fos == res.fos_force
    assert abs(res.fos_moment - res.fos_force) < 1e-5
    # the forces close: no interslice force at either end
    assert abs(res.interslice_E[0]) < 1e-6
    assert abs(res.interslice_E[-1]) < 1e-3
    # and the base forces reported are that state's (sum of forces zero)
    fx = fz = 0.0
    for s, N, S in zip(slices, res.base_normal, res.shear_mobilized):
        fx += -N * math.sin(s.alpha) + S * math.cos(s.alpha)
        fz += N * math.cos(s.alpha) + S * math.sin(s.alpha) - s.weight
    assert abs(fx) < 1e-3 and abs(fz) < 1e-3


def test_a_sign_change_between_two_branches_is_not_a_bracket(own):
    """Ruling 1. On the r_u case the force equation has two solutions at
    lambda = 0.4 and 0.45, one each side of the toe slice's pole
    F = tan(phi') tan(theta - alpha_toe). The net moment is positive on
    the upper one at 0.4 and negative on the lower one at 0.45: a change
    of sign, but between two branches. Refining it closes on no state in
    moment equilibrium, and no root is returned."""
    sys_, _ = _system(own, "ru", "as_printed")
    up = sys_._state(0.40, 0.80)
    low = sys_._state(0.45, 0.64)
    assert up[1] == pytest.approx(0.7986, abs=2e-4)
    assert low[1] == pytest.approx(0.6457, abs=2e-4)
    assert up[7] != low[7]                  # different sides of a pole
    assert up[6] * low[6] < 0               # the net moments differ in sign
    lam, st = sys_.refine(0.40, up, 0.45, low)
    assert st is None
    # and neither state passes for a root, though F_m - F_f is far below
    # the 1e-3 the solver used to accept (the moment axis is 15 km away)
    assert abs(up[0] - up[1]) < 1e-3 and abs(low[0] - low[1]) < 1e-3
    assert not sys_.agrees(up) and not sys_.agrees(low)


def test_no_crossing_in_the_range_is_no_number_with_its_reason(own):
    """Ruling 1 and step 2's rule: where the two factors of safety do not
    cross in the lambda range there is no number, and the reason says
    so."""
    _, slip, slices = own.case("ru")
    res = gle_fos(slices, slip, f_interslice="constant", tol=1e-5,
                  lambda_bounds=(-0.2, 0.2))
    assert not res.converged
    (w,) = [w for w in res.warnings if w.startswith("no F_m = F_f crossing")]
    # R15: the scan went upward from lambda = 0 on its branch, to the bound
    assert ("the scan went upward from lambda = 0 on the branch of "
            "solutions that starts there to lambda = 0.2") in w
    from slope_stability.solution import gle_failure_reason
    assert "factors of safety that agree" in gle_failure_reason(res, None)
    assert "the scan went upward from lambda = 0" in gle_failure_reason(
        res, None)


def test_the_scan_stops_where_its_branch_ends(own, monkeypatch):
    """The review's question 4 (ruling 2 as written): where the branch
    followed from lambda = 0 has no state at a lambda, the branch ends
    there and there is no number, the reason saying where. The scan does
    not step over that lambda to whatever state with the same pole
    signature lies beyond (before, it did: GM-60). Made here by taking
    away every state from lambda 0.45 to 0.55 on the r_u case, whose
    crossing on that branch is at lambda 0.495. Fix round 2: the step
    from 0.4 (a state) to 0.5 (none) is bisected until the two are less
    than 1e-4 apart, so the reason says the branch ends between 0.4499
    and 0.4500 (it said between 0.4 and 0.5)."""
    real = gle._GLESystem._state
    asked = []

    def cut(self, lam, F_near, side=None):
        asked.append(lam)
        if 0.45 <= lam <= 0.55:
            return None
        return real(self, lam, F_near, side)

    monkeypatch.setattr(gle._GLESystem, "_state", cut)
    _, slip, slices = own.case("ru")
    res = gle_fos(slices, slip, f_interslice="constant", tol=1e-5)
    assert not res.converged
    assert max(asked) <= 0.5 + 1e-9         # nothing beyond the end
    (w,) = [w for w in res.warnings if w.startswith("no F_m = F_f crossing")]
    assert ("stopped where that branch ends, between lambda = 0.4499 and "
            "0.4500 (no state at 0.4500)") in w
    from slope_stability.solution import gle_failure_reason
    assert "between lambda = 0.4499 and 0.4500" in gle_failure_reason(
        res, None)


# ---------------------------------------------------------------------------
# Check (a) and ruling 2: which root
# ---------------------------------------------------------------------------

def test_the_branch_from_lambda_0_ends_at_the_locked_toe_wedge(own):
    """Ruling 2: the first crossing met from lambda = 0 on the branch of
    solutions that starts there. On the r_u case that branch stays above
    the toe slice's pole all the way, and its crossing is 26.32 deg,
    F 0.918, the toe wedge carrying about 1,530 kN/m: a root of Spencer's
    equations as printed (the second computation finds it under both
    settings; no base is in tension on it). The toe slice's own
    denominator there is 0.0176 (the review's script: 0.0176)."""
    for rule in BASE_TENSION_RULES:
        _, _, slices, sol = _spencer(own, "ru", rule)
        assert sol.fos == pytest.approx(0.9182, abs=5e-4)
        assert sol.param == pytest.approx(26.319, abs=0.01)
        assert max(sol.gle.interslice_E) == pytest.approx(1533.0, rel=0.01)
        assert not sol.gle.jumps
        den = sol.gle.slice_denominator
        assert den[0] == pytest.approx(0.0176, abs=5e-4)
        assert min(den[1:]) > 0.2
        assert sol.gle.side_force_theta_deg[0] == pytest.approx(26.319,
                                                                abs=0.01)
        assert sol.gle.side_force_E[0] == pytest.approx(1533.0, rel=0.01)


def test_check_a_spencer_is_reported_with_the_finding_on_slice_1(own):
    """Check (a) as the orchestrator's R6 rewrites it. Spencer 0.918 at
    26.32 deg is reported under both settings, 37 % above the closed form
    0.6704 (FHWA NHI-06-088 Sec. 6.3.2, p. 6-8), and never without a
    finding: the denominator of the toe slice's own equations,
    cos(alpha - theta) + sin(alpha - theta) tan(phi') / F, is 0.0176,
    below Whitman and Bailey's 0.2 (1967, p. 487). The finding gives the
    slice, the value and the forces on it, cites EM 1110-2-1902 C-10b (a
    steep exit) and names its remedies as printed, Morgenstern-Price with
    the half-sine being the module's form of a change of side-force
    inclination."""
    geom, slip, _ = own.case("ru")
    for rule in BASE_TENSION_RULES:
        res = analyze_slope(geom, slip_surface=slip, method="spencer",
                            n_slices=60, base_in_tension=rule)
        assert res.FOS == pytest.approx(0.9182, abs=5e-4)
        assert res.theta_spencer == pytest.approx(26.319, abs=0.01)
        assert res.FOS / CLOSED_WET == pytest.approx(1.37, abs=0.005)
        assert "low_m_alpha" in res.flags
        (f,) = [f for f in res.acceptability if f["check"] == "low_m_alpha"]
        assert f["denominator"] == "side_force_inclination"
        assert [s["slice"] for s in f["slices"]] == [1]
        s1 = f["slices"][0]
        assert s1["m_alpha"] == pytest.approx(0.0176, abs=5e-4)
        assert s1["against_sliding"] is True
        assert s1["theta_deg"] == pytest.approx(26.32, abs=0.01)
        assert s1["E_side_kN_per_m"] == pytest.approx(1533.0, rel=0.01)
        assert s1["N_eff_kN_per_m"] == pytest.approx(1381.0, rel=0.01)
        assert "Whitman and Bailey 1967, p. 487" in f["text"]
        assert "EM 1110-2-1902 C-10b, p. C-34" in f["text"]
        assert f["remedies"].startswith("EM 1110-2-1902 C-10b")
        assert ("Morgenstern-Price with the half-sine side-force function "
                "is the module's form of it") in f["remedies"]
        assert any("0.0176 on slice 1" in w for w in res.warnings)


def test_check_a_morgenstern_price_half_sine_within_2_percent(own):
    """Check (a): Morgenstern-Price with the half-sine (no side shear at
    the ends, so the toe wedge does not lock) within 2 % of the closed
    form 0.6704: 0.6779, +1.1 % (the review's script: 0.67790)."""
    geom, slip, _ = own.case("ru")
    res = analyze_slope(geom, slip_surface=slip, method="morgenstern_price",
                        n_slices=60)
    assert res.FOS == pytest.approx(CLOSED_WET, rel=0.02)
    assert res.FOS == pytest.approx(0.6779, abs=5e-4)


def test_the_other_roots_are_roots_of_the_solver_s_equations_too(own):
    """Check (a), the second computation's arithmetic: the two roots below
    the toe slice's pole. NOT what the solver reports (it reports 0.918,
    the root on the branch from lambda = 0, above): they lie on the branch
    below that pole, which does not start at lambda = 0. The solver's
    equations and the second computation's are the same under both
    settings: started on that branch by hand, the solver's own scan finds
    them.

    As printed: 26.20 deg, F 0.6505, slice 1's N' about -111 kN/m and the
    first side in tension. With the slice taken frictionless: 26.563 deg
    (the face's own angle), F 0.6704, the closed form of FHWA NHI-06-088
    Eq. 6-10 (p. 6-8)."""
    for rule, start, f_root, theta in (
            ("as_printed", 0.64, 0.6505, 26.200),
            ("no_friction", 0.67, 0.6704, 26.563)):
        sys_, slices = _system(own, "ru", rule)
        st = sys_._state(0.40, start)
        found, jumps, _ = sys_.first_crossing(0.40, st, -1.5, 1.5)
        assert found is not None
        lam, state = found
        assert state[1] == pytest.approx(f_root, abs=2e-4)
        assert math.degrees(math.atan(lam)) == pytest.approx(theta,
                                                             abs=0.01)
        d = own.from_module(slices)
        assert own.is_root(d, state[1], math.atan(lam),
                           rule == "no_friction", rel=1e-6)
        n1 = state[4][0] - sys_.ns[0].u * sys_.ns[0].length
        if rule == "as_printed":
            assert n1 == pytest.approx(-110.6, abs=0.5)
            assert state[2][1] < 0                  # side 1 in tension
        else:
            assert state[1] == pytest.approx(CLOSED_WET, rel=5e-3)
            assert state[5][0] == 0.0               # slice 1 frictionless


def test_the_finding_does_not_fire_on_an_ordinary_circle():
    """R8: on the review's ordinary made-up circle (ground (-20, 0) (0, 0)
    (20, 10) (50, 10); c' 10, phi' 25, gamma 19, r_u 0.25; centre (5, 20)
    through the toe; 30 slices) no slice's own denominator is near 0.2:
    as printed the least is 0.654 on slice 1 (the review's script: 0.654);
    with the crest slice taken frictionless it is that slice's cos(alpha),
    0.516. Spencer reports it with no low_m_alpha finding, under both
    settings."""
    geom = SlopeGeometry(
        surface_points=[(-20, 0), (0, 0), (20, 10), (50, 10)],
        soil_layers=[SlopeSoilLayer(name="soil", top_elevation=10,
                                    bottom_elevation=-20, gamma=19.0,
                                    phi=25.0, c_prime=10.0, ru=0.25)])
    radius = math.hypot(5.0, 20.0)
    slip = CircularSlipSurface(5.0, 20.0, radius)
    slices = build_slices(geom, slip, 30)
    for rule in BASE_TENSION_RULES:
        res = analyze_slope(geom, xc=5.0, yc=20.0, radius=radius,
                            method="spencer", n_slices=30,
                            base_in_tension=rule)
        assert res.FOS is not None
        assert "low_m_alpha" not in res.flags
        with base_tension(rule):
            sol = solve_method(slices, slip, "spencer")
        least = min(sol.gle.slice_denominator)
        assert least == pytest.approx(
            0.654 if rule == "as_printed" else 0.516, abs=0.005)


def test_the_denominator_is_m_alpha_for_horizontal_side_forces(own):
    """For horizontal side forces (lambda = 0, theta = 0) the slice's own
    denominator is m_alpha, so Bishop's and Janbu's checks do not move:
    each slice's value is cos(alpha - theta) + sin(alpha - theta)
    tan(phi') / F at its own theta."""
    geom, slip, slices = _circle_case()
    sol = solve_method(slices, slip, "spencer")
    th = [math.radians(t) for t in sol.gle.side_force_theta_deg]
    assert th[-1] == 0.0                    # the end of the mass
    for s, t, d in zip(slices, th, sol.gle.slice_denominator):
        tp = math.tan(math.radians(s.phi))
        assert d == pytest.approx(math.cos(s.alpha - t)
                                  + math.sin(s.alpha - t) * tp / sol.fos,
                                  rel=1e-9)
    from slope_stability.methods import m_alpha_values
    m = m_alpha_values(slices, sol.fos, slip, "gle")
    assert sol.gle.slice_denominator[-1] == pytest.approx(m[-1], rel=1e-12)


def _fill_slope(c):
    """Step 2's made-up 10 m fill slope (phi' 22 deg, gamma 19)."""
    return SlopeGeometry(
        surface_points=[(0, 0), (10, 0), (16, 10), (34, 10)],
        soil_layers=[SlopeSoilLayer(name="fill", top_elevation=10.0,
                                    bottom_elevation=-6.0, gamma=19.0,
                                    phi=22.0, c_prime=c)])


@pytest.mark.parametrize("c", [30.0, 38.0])
def test_a_cohesive_crest_in_tension_with_no_root_names_c10c(c):
    """R7 and R13: on step 2's fill circle (9.5, 11.0, 11.2) at 30 and
    38 kPa the review's own script finds no crossing on the branch from
    lambda = 0 for lambda from 0 to 1.5 (its only roots there are at
    negative theta, with slice denominators below 0.01), so the solver's
    no number stands, under both settings: the method's own behaviour
    (step 2's rule). The reason says the scan went upward from lambda = 0
    on that branch, and names EM 1110-2-1902 C-10c and step 4's
    tension_crack_depth."""
    for rule in BASE_TENSION_RULES:
        res = analyze_slope(_fill_slope(c), xc=9.5, yc=11.0, radius=11.2,
                            method="spencer", n_slices=30,
                            base_in_tension=rule)
        assert res.FOS is None and not res.method_solved
        (w,) = [w for w in res.warnings if "no equilibrium solution" in w]
        assert "the scan went upward from lambda = 0" in w
        assert "EM 1110-2-1902 C-10c (p. C-36)" in w
        assert "tension_crack_depth" in w


def test_a_slice_taken_frictionless_is_exempt_as_phi_0_slices_are():
    """GM-344 fix round: with base_in_tension='no_friction' the crest
    slice of step 2's fill circle at 10 kPa (base inclined 79 deg, in
    tension) is taken frictionless, and its denominator in the solve is
    cos(alpha - theta), 0.189: like a slice with phi' = 0 it does not
    depend on F and the strength is c l whatever N', so the default rule
    does not look at it (the module's own reasoning for phi' = 0 slices);
    m_alpha_rule='every_slice' does. As printed the slice's denominator is
    0.60 and nothing is flagged."""
    kw = dict(xc=9.5, yc=11.0, radius=11.2, method="spencer", n_slices=30)
    res = analyze_slope(_fill_slope(10.0), base_in_tension="no_friction",
                        **kw)
    assert res.FOS == pytest.approx(0.9688, abs=0.0005)
    assert "low_m_alpha" not in res.flags
    every = analyze_slope(_fill_slope(10.0), base_in_tension="no_friction",
                          m_alpha_rule="every_slice", **kw)
    (f,) = [x for x in every.acceptability if x["check"] == "low_m_alpha"]
    assert [s["slice"] for s in f["slices"]] == [30]
    assert f["slices"][0]["m_alpha"] == pytest.approx(0.189, abs=0.001)
    printed = analyze_slope(_fill_slope(10.0), **kw)
    assert printed.FOS == pytest.approx(0.9620, abs=0.0005)
    assert "low_m_alpha" not in printed.flags


def test_the_case_dry(own):
    """Check (c). Dry, the closed form is tan(phi') / tan(beta) = 1.349
    (FHWA NHI-06-088 Sec. 6.3.2, p. 6-8). Spencer's own solution on this
    surface, under both settings, is 26.365 deg, F 1.3805 (the second
    computation's root, to 0.1 %): 2.3 % above the closed form, which is
    the passive wedge of the 2 m toe ramp (about 160 kN/m on a driving
    force of about 6,200 kN/m), not a fault of the solve. No base is in
    tension, so the two settings give one answer."""
    got = {}
    for rule in BASE_TENSION_RULES:
        geom, slip, slices, sol = _spencer(own, "dry", rule)
        got[rule] = sol.fos
        # printed 1.349 (FHWA NHI-06-088 Sec. 6.3.2); +2.3 %, the toe ramp
        assert sol.fos == pytest.approx(1.3805, abs=5e-4)
        assert sol.fos / CLOSED_DRY == pytest.approx(1.0234, abs=1e-3)
        assert sol.param == pytest.approx(26.365, abs=0.01)
        assert min(sol.gle.base_normal_eff) > 0
    assert got["as_printed"] == got["no_friction"]


# ---------------------------------------------------------------------------
# Ruling 3: a base in tension
# ---------------------------------------------------------------------------

def test_as_printed_is_the_default():
    assert base_tension_in_force() == "as_printed"
    assert BASE_TENSION_RULES == ("as_printed", "no_friction")
    with pytest.raises(ValueError, match="base_in_tension"):
        with base_tension("clamp"):
            pass


def test_a_crest_slice_in_tension_under_each_rule(own):
    """An ordinary circle whose last slice is in tension. As printed
    (Spencer 1967, pp. 14 to 15: no rule for a negative N') the friction
    term is negative there: 17.891 deg, F 1.3742. With the slice taken
    frictionless in every equation: 17.836 deg, F 1.3756. Each is the
    second computation's root under the same setting."""
    geom, slip, slices = _circle_case()
    d = own.from_module(slices)
    for rule, f_root, theta in (("as_printed", 1.3742, 17.891),
                                ("no_friction", 1.3756, 17.836)):
        with base_tension(rule):
            sol = solve_method(slices, slip, "spencer")
        assert sol.fos == pytest.approx(f_root, abs=2e-4)
        assert sol.param == pytest.approx(theta, abs=0.01)
        device = rule == "no_friction"
        assert own.is_root(d, sol.fos, math.radians(sol.param), device,
                           rel=1e-5)
        assert sol.gle.base_tension_rule == rule
        assert sol.gle.frictionless[-1] is device
        assert not any(sol.gle.frictionless[:-1])
        if device:
            # frictionless: the slice's mobilised shear is its cohesion
            s = slices[-1]
            assert sol.gle.shear_mobilized[-1] == pytest.approx(
                s.c * s.base_length / sol.fos, rel=1e-9)
        else:
            assert sol.gle.base_normal_eff[-1] < 0


def test_the_device_is_in_every_equation(own):
    """With the slice taken frictionless, its normal force comes from
    vertical equilibrium with m_alpha = cos(alpha), the same in the march,
    in the moment sum and in the force sum (before, the friction stayed in
    m_alpha and in the march's vertical equation)."""
    geom, slip, slices = _circle_case()
    ns, _ = gle._normalize(slices)
    with base_tension("no_friction"):
        sys_ = gle._GLESystem(ns, (slip.xc, slip.yc), "constant", 1e-5)
    st = sys_._state(0.3, 1.4)
    E, V, N, tps = st[2], st[3], st[4], st[5]
    i = len(ns) - 1
    s = ns[i]
    assert tps[i] == 0.0
    dV = V[i] - V[i + 1]
    n_vertical = (s.W - dV - s.c * s.length * math.sin(s.alpha) / st[1]) \
        / math.cos(s.alpha)
    assert N[i] == pytest.approx(n_vertical, rel=1e-9)
    assert sys_._normal_eff(s, st[1], dV) == pytest.approx(
        (n_vertical, 0.0), rel=1e-9)


def test_the_option_reaches_analyze_slope_and_the_finding_names_it():
    geom, slip, _ = _circle_case()
    kw = dict(xc=6, yc=18, radius=19.5, method="spencer", n_slices=30)
    a = analyze_slope(geom, **kw)
    b = analyze_slope(geom, base_in_tension="no_friction", **kw)
    assert a.FOS == pytest.approx(1.3742, abs=2e-4)
    assert b.FOS == pytest.approx(1.3756, abs=2e-4)
    assert base_tension_in_force() == "as_printed"      # reset after
    fa = [f for f in a.acceptability if f["check"] == "base_tension"]
    assert fa and fa[0]["rule"] == "as_printed"
    assert "base_in_tension='no_friction'" in fa[0]["text"]
    with pytest.raises(ValueError, match="base_in_tension"):
        analyze_slope(geom, base_in_tension="clamp", **kw)


def test_janbu_follows_the_same_rule():
    """Janbu's simplified method is the force equation of the same system
    at lambda = 0, so the rule for a base in tension is one rule for it
    too."""
    geom, slip, slices = _circle_case()
    out = {}
    for rule in BASE_TENSION_RULES:
        with base_tension(rule):
            out[rule] = solve_method(slices, slip, "janbu").fos_uncorrected
    assert out["as_printed"] != out["no_friction"]
    assert out["as_printed"] == pytest.approx(out["no_friction"], rel=0.01)


def test_a_search_says_the_rule_and_which_root():
    geom, _, _ = _circle_case()
    res = search_critical_surface(geom, method="spencer", nx=3, ny=3,
                                  n_slices=12, toe_circles=False,
                                  infinite_slope=False, refine=False)
    what = {c["what"]: c for c in res.choices}
    rule = what["friction on a slice base in tension (a negative effective "
                "normal force)"]
    assert rule["how"] == "default"
    assert "base_in_tension='no_friction'" in rule["alternatives"][0]
    root = what["which solution is reported where the method's two "
                "equations have more than one"]
    assert "first crossing" in root["chosen"]
    assert "not found in print" in root["basis"]
    with pytest.raises(ValueError, match="base_in_tension"):
        search_critical_surface(geom, method="spencer",
                                base_in_tension="clamp")
    res_b = search_critical_surface(geom, method="bishop", nx=3, ny=3,
                                    n_slices=12, toe_circles=False,
                                    infinite_slope=False, refine=False)
    assert not any("base in tension" in c["what"] for c in res_b.choices)


# ---------------------------------------------------------------------------
# Fix round 2: the end of the branch narrowed (F1); a crossing below 0
# named where one surface is analysed (F2)
# ---------------------------------------------------------------------------

def _acads1a():
    """ACADS 1(a) (the benchmark's geometry and soil, as
    `test_acceptability` builds it for its 200-trial Spencer search)."""
    return SlopeGeometry(
        surface_points=[(20.0, 25.0), (30.0, 25.0), (50.0, 35.0),
                        (70.0, 35.0)],
        soil_layers=[SlopeSoilLayer(name="soil", top_elevation=35.0,
                                    bottom_elevation=20.0, gamma=20.0,
                                    phi=19.6, c_prime=3.0)])


#: Three trial surfaces of ACADS 1(a)'s 200-trial random Spencer search
#: (seed 42, 6 points; windows 20 to 32 and 46 to 68), as the search
#: generates them.
ACADS1A_TRIALS = {
    97: [(29.968772226687243, 25.0),
         (33.556334532794764, 21.340723589200287),
         (37.14389683890228, 26.312533799125266),
         (40.7314591450098, 25.762745740430244),
         (44.31902145111732, 25.28547516430836),
         (47.90658375722484, 33.95329187861242)],
    16: [(31.95345656288116, 25.97672828144058),
         (37.090868368741134, 21.062515645791105),
         (42.2282801746011, 22.43438636297264),
         (47.36569198046107, 33.04720275165329),
         (52.50310378632105, 25.130616348317467),
         (57.64051559218102, 35.0)],
    167: [(26.53359033778574, 25.0),
          (34.729379399817596, 24.666657230912183),
          (42.92516846184945, 27.054315780732026),
          (51.120957523881316, 32.03248869190597),
          (59.31674658591317, 32.91192365186557),
          (67.51253564794503, 35.0)],
}


def _trial(i):
    slip = PolylineSlipSurface(points=ACADS1A_TRIALS[i])
    return slip, build_slices(_acads1a(), slip, 25)


@pytest.mark.parametrize("trial, f_root, theta",
                         [(97, 13.9859, 23.2500), (16, 8.0514, 5.9556)])
@pytest.mark.parametrize("rule", BASE_TENSION_RULES)
def test_a_crossing_in_the_last_step_before_the_branch_ends_is_found(
        own, trial, f_root, theta, rule):
    """Fix round 2 (F1). On trials 97 and 16 of ACADS 1(a)'s search the
    branch from lambda = 0 ends just past a crossing. The scan went in
    one step from the last lambda with a state (0.4; 0.1) to one without
    (0.5; 0.2) and never looked between, so Spencer had no number. That
    step is now bisected to 1e-4, each state tested as the steps' are:
    Spencer reads the review's script's roots, 13.9859 at 23.25 deg and
    8.0514 at 5.9556 deg, under both settings (no base is in tension
    there), each a root of the second computation's equations."""
    slip, slices = _trial(trial)
    with base_tension(rule):
        sol = solve_method(slices, slip, "spencer")
    assert sol.solved
    assert sol.fos == pytest.approx(f_root, rel=1e-3)
    assert sol.param == pytest.approx(theta, abs=0.01)
    d = own.from_module(slices)
    device = rule == "no_friction"
    near = own._newton(d, sol.fos, math.radians(sol.param), device)
    assert near is not None
    assert own.is_root(d, near[0], near[1], device)
    assert sol.fos == pytest.approx(near[0], rel=1e-3)
    assert own.is_root(d, sol.fos, math.radians(sol.param), device,
                       rel=1e-6)


def test_trial_167_still_has_no_number():
    """Fix round 2 (F1). On trial 167 the review's script finds no
    crossing on the branch from theta = 0 (to 60 deg; its roots there are
    at F 0.02 to 0.16, with bases in tension and slices below their
    poles): no number, under both settings, the scan having gone to
    1.5."""
    slip, slices = _trial(167)
    for rule in BASE_TENSION_RULES:
        with base_tension(rule):
            sol = solve_method(slices, slip, "spencer")
        assert not sol.solved
        assert "that starts there to lambda = 1.5" in sol.reason


def test_a_crossing_below_0_is_named_where_one_surface_is_analysed():
    """Fix round 2 (F2; the orchestrator's ruling 2, last clause). Step
    2's fill circle at 30 kPa by Morgenstern-Price (half-sine), as
    printed: no crossing upward from lambda = 0 to 1.5. Analysed as one
    surface, the branch from lambda = 0 is scanned down to -1.5 too (the
    step and end rule of the upward scan); it meets a crossing at lambda
    -0.0342, F 1.6089, the crest slice's base and six sides in tension,
    which the reason names with its flags and does not report. Spencer on
    the same circle meets none below 0 (its branch from 0 ends near
    lambda -0.33) and names none. The made-up planar case has its
    crossing upward, so nothing is scanned below 0 there."""
    kw = dict(xc=9.5, yc=11.0, radius=11.2, n_slices=30)
    res = analyze_slope(_fill_slope(30.0), method="morgenstern_price", **kw)
    assert res.FOS is None
    (w,) = [w for w in res.warnings if "no equilibrium solution" in w]
    assert ("below lambda = 0 the same branch meets a crossing, at lambda "
            "= -0.0342 with F = 1.6089, where 1 slice base(s) (slice 30) "
            "and 6 slice side(s) (side 24, 25, 26, 27, 28 and 29) are in "
            "tension") in w
    assert "/ F, is 0.4352 on slice 30" in w
    assert ("it is not reported, because below lambda = 0 the interslice "
            "shear acts against the sense the sliding gives it") in w
    sp = analyze_slope(_fill_slope(30.0), method="spencer", **kw)
    (w,) = [w for w in sp.warnings if "no equilibrium solution" in w]
    assert sp.FOS is None and "below lambda = 0" not in w
    # off by default in the solver a search's trials go through
    slip = CircularSlipSurface(9.5, 11.0, 11.2)
    slices = build_slices(_fill_slope(30.0), slip, 30)
    sol = solve_method(slices, slip, "morgenstern_price")
    assert not sol.solved and "below lambda = 0" not in sol.reason
    with gle.single_surface():
        sol = solve_method(slices, slip, "morgenstern_price")
    assert "at lambda = -0.0342 with F = 1.6089" in sol.reason


def test_a_search_makes_no_scan_below_0(monkeypatch):
    """Fix round 2 (F2): never inside a search's trials (run time).
    ACADS 1(a)'s 200-trial random Spencer search leaves trials without a
    solution; the scan below 0 is made on none of them, and the search's
    value stands (1.0916). Trial 167 analysed alone (no number) is
    scanned once."""
    calls = []
    real = gle._GLESystem.crossing_below

    def counted(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(gle._GLESystem, "crossing_below", counted)
    r = search_critical_surface(
        _acads1a(), x_entry_range=(20.0, 32.0), x_exit_range=(46.0, 68.0),
        surface_type="noncircular", n_trials=200, n_points=6, n_slices=25,
        seed=42, method="spencer")
    # The minimum of a 200-trial random polyline sample: a pin on the search's repeatability,
    # not a benchmark value. ACADS 1(a)'s benchmark is the circle: Spencer 0.9838 against the
    # printed 1.00 (Bishop 0.9849).
    assert r.critical.FOS == pytest.approx(1.0916, abs=5e-4)
    assert r.no_solution[0]["not_converged"] > 0
    assert calls == []
    one = analyze_slope(_acads1a(), method="spencer", n_slices=25,
                        slip_surface=PolylineSlipSurface(
                            points=ACADS1A_TRIALS[167]))
    assert one.FOS is None
    assert len(calls) == 1

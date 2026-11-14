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
    """FHWA NHI-06-088 Eq. 6-10 (p. 6-8) is the infinite slope with seepage
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

    Before (master, 2026-11-14) the r_u case read 0.6065 at 18.00 deg,
    reported as converged: not a root (the bisection had closed on a jump
    between two branches, the two factors of safety differing by less than
    1e-3 because the fitted moment axis lies 15 km away, and their mean
    was reported)."""
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
    """The number master reported for the r_u case (Spencer 0.6065 at
    18.00 deg, converged) does not meet Spencer's equations: the moment
    residual there is far from zero, and Newton's method from it leaves
    for a root elsewhere."""
    _, _, slices = own.case("ru")
    d = own.from_module(slices)
    assert not own.is_root(d, 0.6065, math.radians(17.997), False,
                           rel=1e-4)
    _, _, _, sol = _spencer(own, "ru", "as_printed")
    assert abs(sol.fos - 0.6065) > 0.01


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
    assert any(w.startswith("no F_m = F_f crossing") for w in res.warnings)
    from slope_stability.solution import gle_failure_reason
    assert "factors of safety that agree" in gle_failure_reason(res, None)


def test_a_branch_that_ends_is_named_in_the_reason():
    """The reason carries where the branch followed has no state."""
    from slope_stability.solution import gle_failure_reason
    res = gle.GLEResult()
    res.warnings.append(
        "no F_m = F_f crossing found in lambda range; the branch of "
        "solutions followed from lambda = 0 has no state at lambda = 0.7")
    assert "no state at lambda = 0.7" in gle_failure_reason(res, None)


# ---------------------------------------------------------------------------
# Check (a) and ruling 2: which root
# ---------------------------------------------------------------------------

def test_the_branch_from_lambda_0_ends_at_the_locked_toe_wedge(own):
    """Ruling 2: the first crossing met from lambda = 0 on the branch of
    solutions that starts there. On the r_u case that branch stays above
    the toe slice's pole all the way, and its crossing is 26.32 deg,
    F 0.918, the toe wedge carrying about 1,530 kN/m: a root of Spencer's
    equations as printed (the second computation finds it under both
    settings). It is NOT the root the brief's check (a) expected; see
    ``module_work/gm140/GLE_ROOT_CHECK.md``, first lines."""
    for rule in BASE_TENSION_RULES:
        _, _, slices, sol = _spencer(own, "ru", rule)
        assert sol.fos == pytest.approx(0.9182, abs=5e-4)
        assert sol.param == pytest.approx(26.319, abs=0.01)
        assert max(sol.gle.interslice_E) == pytest.approx(1533.0, rel=0.01)
        assert not sol.gle.jumps


def test_the_other_roots_are_roots_of_the_solver_s_equations_too(own):
    """The solver's equations and the second computation's are the same
    under both settings: started on the branch below the toe slice's pole,
    the solver's own scan finds the second computation's other roots.

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


@pytest.mark.xfail(strict=True, reason=(
    "The brief's check (a), as written: it took 26.20 deg, F 0.6505 to be "
    "the root on the branch from lambda = 0. That branch's root is "
    "26.32 deg, F 0.918 (the toe wedge locked); 0.6505 lies on the branch "
    "below the toe slice's pole. GLE_ROOT_CHECK.md, first lines."))
def test_check_a_as_printed_as_the_brief_wrote_it(own):
    geom, slip, slices, sol = _spencer(own, "ru", "as_printed")
    assert sol.fos == pytest.approx(0.6505, abs=5e-4)
    assert sol.param == pytest.approx(26.20, abs=0.01)


@pytest.mark.xfail(strict=True, reason=(
    "The brief's check (a), as written: with the option it expected the "
    "closed form 0.6704 at theta = beta. The branch from lambda = 0 has "
    "no base in tension, so the option changes nothing on it: 26.32 deg, "
    "F 0.918. GLE_ROOT_CHECK.md, first lines."))
def test_check_a_with_the_option_as_the_brief_wrote_it(own):
    geom, slip, slices, sol = _spencer(own, "ru", "no_friction")
    assert sol.fos == pytest.approx(CLOSED_WET, rel=5e-3)
    assert sol.param == pytest.approx(BETA, abs=0.1)


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

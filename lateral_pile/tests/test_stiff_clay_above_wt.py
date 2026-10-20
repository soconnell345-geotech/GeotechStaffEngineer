"""Stiff clay above the water table (backlog GM-202): Welch & Reese (1972)
as printed in the COM624P manual (FHWA-SA-91-048), pp. 336-340, Eqs.
3.27-3.29 and Figs. 3.9-3.10, and in FHWA GEC 9 (FHWA-HIF-18-031, 2018)
App. A.3, pp. 224-226, Eqs. A-19 to A-21. Before GM-202 loading='cyclic'
gave the static curve whatever the number of cycles: it capped p at the
value reached at 16 y50, which the static curve already does.

The static curve is pinned to COM624P Example 1 (printed pp. 101-126, a
14HP89 in uniform stiff clay above the water table, static loading): its
printed p-y curves (pp. 121-122) and its head deflections and moments
(p. 126). Neither COM624P nor GEC 9 prints a cyclic example, so the cyclic
curve is pinned to the printed equations by hand and to the deflection
Fig. 3.10 labels as the start of the plateau."""

import math

import pytest

from lateral_pile import LateralPileAnalysis, Pile, SoilLayer
from lateral_pile.calc_steps import _extract_model_params
from lateral_pile.py_curves import StiffClayAboveWT

IN, PSI, PCI, KIP = 0.0254, 6.894757, 271.4471, 4.448222
LB_IN = 4.448222e-3 / IN            # kN/m in one lb/in
B = 14.7 * IN                       # 14HP89 flange width (COM624P p. 101)
Y50 = 2.5 * 0.007 * 14.7            # in, Eq. 3.10: 0.25725 in


def _clay(**kw):
    """COM624P Example 1's clay: c 14 psi, gamma 0.069 lb/in^3, eps50
    0.007 (p. 101)."""
    return StiffClayAboveWT(c=14.0 * PSI, gamma=0.069 * PCI, eps50=0.007,
                            **kw)


def _p_lb_in(model, y_in, z_in):
    return model.get_p(y_in * IN, z_in * IN, B) / LB_IN


# ---------------------------------------------------------------------------
# The static curve: COM624P Example 1's printed p-y curves (pp. 121-122).
# The program prints y at i x 16 y50 / 15 (0.2744 in) and 5.145 in.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z_in, rows", [
    (20.0, {1: 395.168, 3: 520.070, 8: 664.590, 15: 777.686}),
    (60.0, {1: 558.061, 4: 789.217, 11: 1016.318, 15: 1098.258}),
    (100.0, {1: 720.954, 6: 1128.354, 11: 1312.973, 15: 1418.830}),
    (150.0, {1: 924.570, 2: 1099.506, 12: 1720.819, 15: 1819.545}),
])
def test_static_curve_is_com624p_example_1(z_in, rows):
    clay = _clay()
    for i, p in rows.items():
        assert _p_lb_in(clay, i * 16 * Y50 / 15, z_in) == pytest.approx(
            p, rel=2e-5)
    # the plateau past 16 y50 (printed at 5.145 in)
    assert _p_lb_in(clay, 5.145, z_in) == pytest.approx(rows[15], rel=2e-5)


@pytest.mark.parametrize("v_kip, y_top_in, m_max_in_kip", [
    (20.0, 0.145, 661.0),     # printed summary table, p. 126
    (40.0, 0.623, 1740.0),
])
def test_com624p_example_1_head_deflection_and_moment(v_kip, y_top_in,
                                                      m_max_in_kip):
    """A 14HP89, strong axis (I 904 in^4, E 29,000 ksi), 300 in long, head
    at the ground, free, 39 kips axial, 120 increments (pp. 101-119)."""
    pile = Pile(length=300 * IN, diameter=B, E=29e6 * PSI,
                moment_of_inertia=904 * IN ** 4)
    r = LateralPileAnalysis(pile, [SoilLayer(0.0, 300 * IN, _clay())]).solve(
        Vt=v_kip * KIP, Q=39.0 * KIP, n_elements=120)
    assert r.converged
    assert r.y_top / IN == pytest.approx(y_top_in, rel=0.01)
    assert r.max_moment / (KIP * IN) == pytest.approx(m_max_in_kip, rel=0.01)


# ---------------------------------------------------------------------------
# The cyclic curve by the printed steps (COM624P p. 338, GEC 9 p. 226):
# for several p/pu, y_s off the static curve (Eq. 3.27), C = 9.6 (p/pu)^4
# (Eq. 3.28), y_c = y_s + y50 C log N (Eq. 3.29).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2.0, 10.0, 100.0, 5000.0])
@pytest.mark.parametrize("ratio", [0.1, 0.35, 0.5, 0.8, 0.95])
def test_cyclic_curve_follows_the_printed_steps(n, ratio):
    clay = _clay(loading="cyclic", n_cycles=n)
    z = 60.0
    pu = clay.get_pu(z * IN, B) / LB_IN
    y_s = Y50 * (ratio / 0.5) ** 4                       # Eq. 3.27 for y
    c_rep = 9.6 * ratio ** 4                             # Eq. 3.28
    y_c = y_s + Y50 * c_rep * math.log10(n)              # Eq. 3.29
    assert _p_lb_in(clay, y_c, z) == pytest.approx(ratio * pu, rel=1e-9)


def test_cyclic_hand_calculation():
    """COM624P Example 1's clay at 60 in, N = 100: pu = 1098.258 lb/in
    (p. 121), y50 = 0.25725 in. At p/pu = 0.5: y_s = y50, C = 9.6 x
    0.5^4 = 0.6, y_c = 0.25725 + 0.25725 x 0.6 x 2 = 0.56595 in, so the
    cyclic curve carries 549.13 lb/in at 0.56595 in, where the static one
    carries 0.5 x 1098.258 x (0.56595 / 0.25725)^0.25 = 668.77 lb/in."""
    static, cyclic = _clay(), _clay(loading="cyclic", n_cycles=100)
    assert cyclic.cyclic_factor == pytest.approx(2.2)
    assert _p_lb_in(cyclic, 0.56595, 60.0) == pytest.approx(549.129, rel=1e-5)
    assert _p_lb_in(static, 0.56595, 60.0) == pytest.approx(668.77, rel=1e-5)


@pytest.mark.parametrize("n", [10.0, 100.0, 1000.0])
def test_cyclic_curve_reaches_pu_where_figure_3_10_says(n):
    """Fig. 3.10: p = pu from y = 16 y50 + 9.6 y50 log N."""
    clay = _clay(loading="cyclic", n_cycles=n)
    pu = clay.get_pu(60 * IN, B) / LB_IN
    y_u = 16 * Y50 + 9.6 * Y50 * math.log10(n)
    assert _p_lb_in(clay, y_u, 60.0) == pytest.approx(pu, rel=1e-12)
    assert _p_lb_in(clay, 2 * y_u, 60.0) == pytest.approx(pu, rel=1e-12)
    assert _p_lb_in(clay, 0.99 * y_u, 60.0) < pu
    # the static curve is already at pu there: the old cyclic curve was
    assert _p_lb_in(_clay(), 16 * Y50, 60.0) == pytest.approx(pu, rel=1e-12)


def test_one_cycle_is_the_static_curve():
    static, one = _clay(), _clay(loading="cyclic", n_cycles=1)
    for y_in in (0.01, 0.2, 1.0, 4.0, 6.0):
        for z_in in (0.0, 20.0, 150.0):
            assert _p_lb_in(one, y_in, z_in) == pytest.approx(
                _p_lb_in(static, y_in, z_in), rel=1e-12)


def test_more_cycles_soften_the_curve():
    ps = [_p_lb_in(_clay(loading="cyclic", n_cycles=n), 1.0, 60.0)
          for n in (10, 100, 1000)]
    assert _p_lb_in(_clay(), 1.0, 60.0) > ps[0] > ps[1] > ps[2]


def test_cyclic_needs_the_number_of_cycles():
    with pytest.raises(ValueError, match="n_cycles"):
        _clay(loading="cyclic")
    with pytest.raises(ValueError, match="cyclic' only"):
        _clay(n_cycles=100)
    with pytest.raises(ValueError, match=">= 1"):
        _clay(loading="cyclic", n_cycles=0.5)


def test_get_py_curve_reaches_the_cyclic_plateau():
    clay = _clay(loading="cyclic", n_cycles=1000)
    y, p = clay.get_py_curve(z=60 * IN, b=B)
    pu = clay.get_pu(60 * IN, B)
    assert p[-1] == pytest.approx(pu)
    assert y[-1] == pytest.approx(25 * 2.5 * 0.007 * B * 2.8)


# ---------------------------------------------------------------------------
# The analysis: cyclic loads now deflect the pile more, and say N.
# ---------------------------------------------------------------------------

def _ex1(clay, v_kip=20.0):
    pile = Pile(length=300 * IN, diameter=B, E=29e6 * PSI,
                moment_of_inertia=904 * IN ** 4)
    return LateralPileAnalysis(pile, [SoilLayer(0.0, 300 * IN, clay)]).solve(
        Vt=v_kip * KIP, Q=39.0 * KIP, n_elements=120)


def test_cyclic_loading_deflects_more_with_more_cycles():
    static = _ex1(_clay())
    tens, hundreds = (_ex1(_clay(loading="cyclic", n_cycles=n))
                      for n in (10, 100))
    assert static.y_top < tens.y_top < hundreds.y_top
    assert static.max_moment < tens.max_moment < hundreds.max_moment


def test_choices_state_the_number_of_cycles():
    r = _ex1(_clay(loading="cyclic", n_cycles=100))
    n = next(c for c in r.choices if c["what"].startswith("number of load"))
    assert n["chosen"] == "100 (y50 x 2.2)"
    assert n["how"] == "given"
    assert "Eqs. 3.28-3.29" in n["basis"] and "no default" in n["basis"]
    assert "N = 1: the static curve, the stiffest" in n["alternatives"]
    assert not any(a.startswith("N = 100:") for a in n["alternatives"])
    loading = r.choices[-1]
    assert loading["what"] == "loading" and loading["chosen"] == (
        "cyclic, every layer")
    assert not any(c["what"].startswith("number of load")
                   for c in _ex1(_clay()).choices)


def test_the_calc_tables_print_the_number_of_cycles():
    assert "N=100 cycles" in _extract_model_params(
        _clay(loading="cyclic", n_cycles=100))
    assert "cycles" not in _extract_model_params(_clay())

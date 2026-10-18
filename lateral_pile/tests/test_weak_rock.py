"""Weak-rock p-y curves as printed (backlog GM-159): Reese (1997) in FHWA
GEC 9 (FHWA-HIF-18-031, 2018), Appendix A.5, pp. 230-232, Eqs. A-33 to
A-42, and GEC 10 (FHWA-NHI-18-024, 2018) Section 9.3.3.4.4, pp. 9-31 to
9-33. Before GM-159 alpha_r was fixed at 1.0 whatever the RQD (p_ur three
times the printed value at RQD 100) and the curve was 0.25 E_m y to p_ur."""

import math

import pytest

from lateral_pile import LateralPileAnalysis, Pile, SoilLayer
from lateral_pile.py_curves import SoftClayMatlock, WeakRock

QU, EM, B = 2000.0, 300000.0, 1.2       # kPa, kPa, m


# ---------------------------------------------------------------------------
# p_ur by hand: GEC 9 p. 231, Eq. A-33 (z_r <= 3b) and A-34 (z_r >= 3b);
# alpha_r = 0.33 at RQD 100 %, rising linearly to 1.0 at RQD 0.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rqd, z, pur", [
    # RQD 100: alpha_r = 1/3
    (100.0, 1.8, 2480.0),   # (1/3)(2000)(1.2)(1 + 1.4 x 1.8/1.2) = 800 x 3.1
    (100.0, 6.0, 4160.0),   # 5.2 (1/3)(2000)(1.2), z_r = 6.0 > 3b = 3.6
    # RQD 40: alpha_r = 1 - (2/3)(0.40) = 11/15
    (40.0, 1.8, 5456.0),    # (11/15)(2000)(1.2)(3.1)
    (40.0, 6.0, 9152.0),    # 5.2 (11/15)(2000)(1.2)
])
def test_p_ur_by_hand(rqd, z, pur):
    rock = WeakRock(qu=QU, Er=EM, RQD=rqd)
    assert rock.get_pu(z, B) == pytest.approx(pur, rel=1e-12)


def test_alpha_r_follows_rqd():
    assert WeakRock(qu=QU, Er=EM).alpha_r == pytest.approx(1 / 3)
    assert WeakRock(qu=QU, Er=EM, RQD=0).alpha_r == pytest.approx(1.0)
    assert WeakRock(qu=QU, Er=EM, RQD=50).alpha_r == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="RQD"):
        WeakRock(qu=QU, Er=EM, RQD=120)


def test_shallow_and_deep_p_ur_meet_at_three_diameters():
    rock = WeakRock(qu=QU, Er=EM, RQD=70)
    assert rock.get_pu(3 * B, B) == pytest.approx(rock.get_pu(3 * B + 1e-9, B))


def test_depth_is_measured_below_the_rock_surface():
    """z_r is the depth below the rock surface (GEC 9 p. 231)."""
    buried = WeakRock(qu=QU, Er=EM, rock_top=3.0)
    assert buried.get_pu(4.8, B) == pytest.approx(2480.0)
    assert buried.get_pu(3.0, B) == pytest.approx(800.0)   # alpha_r qu b


# ---------------------------------------------------------------------------
# The curve: GEC 9 p. 231-232, Eqs. A-35 to A-42
# ---------------------------------------------------------------------------

def test_initial_slope_k_ir_by_depth():
    """K_ir = k_ir E_m; k_ir = 100 + 400 z_r/(3b) to 3b, 500 below."""
    rock = WeakRock(qu=QU, Er=EM)
    assert rock.get_Kir(0.0, B) == pytest.approx(100 * EM)
    assert rock.get_Kir(1.8, B) == pytest.approx(300 * EM)   # 100 + 400(1.8)/3.6
    assert rock.get_Kir(3.6, B) == pytest.approx(500 * EM)
    assert rock.get_Kir(9.0, B) == pytest.approx(500 * EM)


def test_three_segments_of_the_curve():
    rock = WeakRock(qu=QU, Er=EM)                # krm 0.0005
    z, pur, Kir = 1.8, 2480.0, 300 * EM
    y_rm = 0.0005 * B                            # A-41
    y_A = (pur / (2 * y_rm ** 0.25 * Kir)) ** (4 / 3)   # A-42
    # first segment, p = K_ir y (A-38)
    assert rock.get_p(0.5 * y_A, z, B) == pytest.approx(Kir * 0.5 * y_A)
    # the two segments meet at y_A
    assert Kir * y_A == pytest.approx(0.5 * pur * (y_A / y_rm) ** 0.25)
    # second segment, p = (p_ur/2)(y/y_rm)^0.25 (A-39): p_ur/2 at y_rm
    assert rock.get_p(y_rm, z, B) == pytest.approx(pur / 2)
    assert rock.get_p(4 * y_rm, z, B) == pytest.approx(pur / 2 * 4 ** 0.25)
    # third segment, p = p_ur (A-40), from 16 y_rm on
    assert rock.get_p(16 * y_rm, z, B) == pytest.approx(pur)
    assert rock.get_p(0.05, z, B) == pytest.approx(pur)
    assert rock.get_p(-0.05, z, B) == pytest.approx(-pur)


def test_larger_krm_is_the_softer_curve():
    """GEC 10 p. 9-31: the lower k_rm gives a stiffer initial response."""
    soft = WeakRock(qu=QU, Er=EM, krm=0.0005)
    stiff = WeakRock(qu=QU, Er=EM, krm=0.00005)
    assert stiff.get_p(0.001, 1.8, B) > soft.get_p(0.001, 1.8, B)


def test_the_curve_reaches_p_ur():
    ys, ps = WeakRock(qu=QU, Er=EM).get_py_curve(1.8, B)
    assert ps[0] == 0.0 and ps[-1] == pytest.approx(2480.0)
    assert all(b >= a for a, b in zip(ps, ps[1:]))


# ---------------------------------------------------------------------------
# In an analysis: the rock surface and what the result says it used
# ---------------------------------------------------------------------------

def _layers(rock_top=None):
    return [
        SoilLayer(0.0, 3.0, SoftClayMatlock(c=50.0, gamma=8.0, eps50=0.01)),
        SoilLayer(3.0, 6.0, WeakRock(qu=QU, Er=EM, RQD=50.0,
                                     rock_top=rock_top)),
        SoilLayer(6.0, 12.0, WeakRock(qu=QU, Er=EM)),
    ]


def test_the_analysis_measures_z_r_from_the_top_of_the_rock():
    layers = _layers()
    analysis = LateralPileAnalysis(Pile(length=12.0, diameter=B, E=25e6),
                                   layers)
    assert [l.py_model.rock_top for l in analysis.layers[1:]] == [3.0, 3.0]
    assert layers[1].py_model.rock_top is None     # the caller's left alone
    given = LateralPileAnalysis(Pile(length=12.0, diameter=B, E=25e6),
                                _layers(rock_top=2.5))
    # a rock surface given on the run's first layer holds for the run
    assert [l.py_model.rock_top for l in given.layers[1:]] == [2.5, 2.5]


def test_the_calc_tables_print_the_rock_surface():
    from lateral_pile.calc_steps import _extract_model_params
    analysis = LateralPileAnalysis(Pile(length=12.0, diameter=B, E=25e6),
                                   _layers())
    params = _extract_model_params(analysis.layers[2].py_model)
    assert "rock surface 3 m" in params and "k_rm=0.0005" in params


def test_choices_state_alpha_r_and_k_rm():
    r = LateralPileAnalysis(Pile(length=12.0, diameter=B, E=25e6),
                            _layers()).solve(Vt=500.0)
    by = {c["what"]: c for c in r.choices}
    assert "z_r below the rock surface at 3 m" in by["p-y curves 3-6 m"][
        "basis"]
    given = by["strength reduction alpha_r 3-6 m"]
    assert given["chosen"] == "0.667 (RQD 50 %)" and given["how"] == "given"
    default = by["strength reduction alpha_r 6-12 m"]
    assert default["chosen"] == "0.333 (RQD 100 %)"
    assert default["how"] == "default"
    assert "GEC 9 p. 231" in default["basis"]
    assert "lower bound" in default["basis"]
    assert any(a.startswith("RQD 0: alpha_r = 1.0") and "less deflection"
               in a for a in default["alternatives"])
    krm = by["k_rm 6-12 m"]
    assert krm["chosen"] == 0.0005 and krm["how"] == "default"
    assert any(a.startswith("0.00005, the stiffer end: less deflection")
               for a in krm["alternatives"])


def test_cyclic_loading_says_the_criterion_is_static():
    layers = [SoilLayer(0.0, 10.0, WeakRock(qu=QU, Er=EM, loading="cyclic"))]
    r = LateralPileAnalysis(Pile(length=10.0, diameter=B, E=25e6),
                            layers).solve(Vt=500.0)
    assert "static loading (GEC 9 p. 92)" in r.choices[0]["basis"]


# ---------------------------------------------------------------------------
# GEC 10 Figure 9-31, p. 9-33: a 6-ft by 40-ft shaft through 20 ft of soft
# clay (su = 2 psi) into weak rock (qu = 180 psi, E_m = 27,000 psi), 75
# kips at the ground: 0.116 in and 15,722 in-kip. Not printed, and assumed
# here: the shaft's gross section at f'c 4 ksi, the clay's gamma' 7.5 kN/m3
# and eps50 0.02, and RQD 0 with k_rm 0.0004, the values GEC 10 p. 9-31
# gives for this rock (Brown and Kahle 2002). The printed sensitivity rows
# set the trend: clay 10 % thicker, +22 % / +5.5 %; clay su 10 % lower,
# +2 % / +1 %; rock qu and E_m 10 % lower, +2 % / 0 %. Before GM-159 this
# case gave 0.88 in and 12,840 in-kip.
# ---------------------------------------------------------------------------

_IN, _PSI, _KIP = 0.0254, 6.894757, 4.448222


def _gec10(clay_ft=20.0, rock_factor=1.0, su_factor=1.0):
    clay = clay_ft * 12 * _IN
    length = 40 * 12 * _IN
    pile = Pile(length=length, diameter=72 * _IN,
                E=57000 * math.sqrt(4000) * _PSI)
    layers = [
        SoilLayer(0.0, clay, SoftClayMatlock(c=2 * _PSI * su_factor,
                                             gamma=7.5, eps50=0.02)),
        SoilLayer(clay, length, WeakRock(qu=180 * _PSI * rock_factor,
                                         Er=27000 * _PSI * rock_factor,
                                         RQD=0.0, krm=0.0004)),
    ]
    r = LateralPileAnalysis(pile, layers).solve(Vt=75 * _KIP,
                                                n_elements=200)
    return r.y_top / _IN, r.max_moment / (_KIP * _IN)


def test_gec10_figure_9_31_example():
    y, m = _gec10()
    assert m == pytest.approx(15722, rel=0.02)
    assert y == pytest.approx(0.116, rel=0.15)
    y_clay, m_clay = _gec10(clay_ft=22.0)
    assert y_clay / y - 1 == pytest.approx(0.22, abs=0.03)
    assert m_clay / m - 1 == pytest.approx(0.055, abs=0.01)
    y_su, m_su = _gec10(su_factor=0.9)
    assert y_su / y - 1 == pytest.approx(0.02, abs=0.01)
    assert m_su / m - 1 == pytest.approx(0.01, abs=0.005)
    y_rock, m_rock = _gec10(rock_factor=0.9)
    assert y_rock / y - 1 == pytest.approx(0.02, abs=0.01)
    assert m_rock / m - 1 == pytest.approx(0.0, abs=0.005)

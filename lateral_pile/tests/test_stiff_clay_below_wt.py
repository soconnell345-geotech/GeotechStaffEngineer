"""Stiff clay below the water table (backlog GM-208): Reese, Cox & Koop
(1975) as printed in the COM624P manual (FHWA-SA-91-048), pp. 328-335,
Eqs. 3.14-3.26, Figs. 3.6-3.8 and Tables 3.2-3.3, and in FHWA GEC 9
(FHWA-HIF-18-031, 2018) App. A.2, pp. 219-224, Eqs. A-7 to A-18. Before
GM-208 the curves were a simplified construction (a line, a square-root
parabola and a plateau at A pu, with y50 = 2.5 eps50 b and A from a
made-up table), and 'cyclic' used ks.

Neither COM624P nor GEC 9 prints an example in stiff clay below the water
table (COM624P's Examples 1-6 and GEC 9 App. B use other criteria), so
the curves are pinned to HAND CALCULATIONS from the printed equations:
c = 100 kPa, gamma' = 9 kN/m^3, eps50 = 0.005, b = 0.6 m, at x = 1.5 m
(x/b = 2.5: A_s = 0.565, A_c = 0.305 read off Fig. 3.7 / A-6),
ks = 271,400 and kc = 108,600 kN/m^3 (Table 3.2, 1,000 and 400 lb/in^3):

    y50 = eps50 b = 0.003 m                                    (3.17)
    pc  = min(2(100)(0.6) + 9(0.6)(1.5) + 2.83(100)(1.5), 11(100)(0.6))
        = min(552.6, 660) = 552.6 kN/m                   (3.14, 3.15)
    static, k x = 407,100 kN/m^2:
      the line meets 0.5 pc (y/y50)^0.5 at y = (0.5 pc)^2 / ((k x)^2 y50)
        = 0.1535 mm, p = 62.51 kN/m
      y = As y50 = 1.695 mm: 0.5 pc 0.565^0.5           = 207.68  (3.18)
      y = 6 mm:   pc [0.5 (2)^0.5 - 0.055 (2.5398)^1.25] = 293.30  (3.19)
      y = 20 mm:  pc [0.5 (3.39)^0.5 - 0.411]
                  - (0.0625/0.003) pc (0.02 - 0.01017)   = 168.44  (3.20)
      y = 50 mm:  281.60 - 0.75 pc 0.565                 =  47.44  (3.21)
    cyclic, yp = 4.1 Ac y50 = 3.7515 mm, k x = 162,900 kN/m^2:
      y = 0.45 yp: Ac pc                                  = 168.54  (3.24)
      the line meets 3.24 at y = 0.8605 mm, p = 140.18 kN/m
      y = 5 mm:  0.936 Ac pc - (0.085/0.003) pc (0.005 - 0.6 yp)
                                                          = 114.71  (3.25)
      y = 10 mm: 0.936 Ac pc - (0.102/0.003) pc yp        =  87.27  (3.26)
"""

import math

import numpy as np
import pytest

from lateral_pile import LateralPileAnalysis, Pile, SoilLayer
from lateral_pile.calc_steps import _extract_model_params
from lateral_pile.py_curves import STIFF_CLAY_K_TABLE, StiffClayBelowWT

PCI = 271.4471                      # kN/m^3 in one lb/in^3
B, X = 0.6, 1.5
KS, KC = 271_400.0, 108_600.0


def _clay(loading="static", **kw):
    k = {"ks": KS} if loading == "static" else {"kc": KC}
    k.update(kw)
    return StiffClayBelowWT(c=100.0, gamma=9.0, eps50=0.005,
                            loading=loading, **k)


def test_pc_and_y50_as_printed():
    clay = _clay()
    assert clay.get_pu(X, B) == pytest.approx(552.6)
    assert clay.get_pu(10.0, B) == pytest.approx(660.0)       # 11 c b
    # y50 = eps50 b (Eq. 3.17), not 2.5 eps50 b: the curve's top at As y50
    y50 = 0.005 * B
    As = clay._get_As(X, B)
    assert clay.get_p(As * y50, X, B) == pytest.approx(
        0.5 * 552.6 * math.sqrt(As))


def test_as_and_ac_read_off_figure_3_7():
    clay = _clay()
    # both start at 0.2 at the ground (Fig. 3.7, Fig. A-6)
    assert clay._get_As(0.0, B) == pytest.approx(0.20)
    assert clay._get_Ac(0.0, B) == pytest.approx(0.20)
    for zb, As, Ac in ((1.0, 0.435, 0.285), (2.5, 0.565, 0.305),
                       (4.5, 0.60, 0.305), (12.0, 0.60, 0.305)):
        assert clay._get_As(zb * B, B) == pytest.approx(As)
        assert clay._get_Ac(zb * B, B) == pytest.approx(Ac)
    zb = np.linspace(0.0, 12.0, 97)
    assert np.all(np.diff([clay._get_As(z * B, B) for z in zb]) >= 0)
    assert np.all(np.diff([clay._get_Ac(z * B, B) for z in zb]) >= 0)


@pytest.mark.parametrize("y, p", [
    (0.0001, 407_100 * 0.0001),     # on the initial line, (ks x) y
    (0.001695, 207.685),            # 3.18 at As y50
    (0.006, 293.298),               # 3.19
    (0.02, 168.436),                # 3.20
    (0.05, 47.439),                 # 3.21, the residual
])
def test_static_hand_calculation(y, p):
    clay = _clay()
    assert clay.get_p(y, X, B) == pytest.approx(p, rel=1e-4)
    assert clay.get_p(-y, X, B) == pytest.approx(-p, rel=1e-4)


def test_static_initial_line_meets_the_curve_where_printed():
    clay = _clay()
    y_meet = (0.5 * 552.6) ** 2 / (407_100 ** 2 * 0.003)
    assert y_meet == pytest.approx(1.5355e-4, rel=1e-3)
    for y in (0.9 * y_meet, 1.1 * y_meet):
        line, curve = 407_100 * y, 0.5 * 552.6 * math.sqrt(y / 0.003)
        assert clay.get_p(y, X, B) == pytest.approx(min(line, curve))


@pytest.mark.parametrize("y, p", [
    (0.0005, 162_900 * 0.0005),     # on the initial line, (kc x) y
    (0.00168818, 168.543),          # 3.24 at 0.45 yp: Ac pc
    (0.005, 114.714),               # 3.25
    (0.01, 87.272),                 # 3.26, the residual
])
def test_cyclic_hand_calculation(y, p):
    clay = _clay("cyclic")
    assert clay.get_p(y, X, B) == pytest.approx(p, rel=1e-4)


def test_segments_join_where_printed():
    """3.18-3.21 and 3.24-3.26 meet at As y50, 6 As y50, 18 As y50,
    0.6 yp and 1.8 yp (to the printed coefficients' three figures)."""
    y50 = 0.005 * B
    for x in (0.3, 1.5, 6.0):
        st, cy = _clay(), _clay("cyclic")
        As, Ac = st._get_As(x, B), cy._get_Ac(x, B)
        yp = 4.1 * Ac * y50
        pc = st.get_pu(x, B)
        for model, y in ((st, As * y50), (st, 6 * As * y50),
                         (st, 18 * As * y50), (cy, 0.6 * yp),
                         (cy, 1.8 * yp)):
            lo, hi = model._curve(y * (1 - 1e-9), x, B), model._curve(
                y * (1 + 1e-9), x, B)
            assert lo == pytest.approx(hi, abs=5e-4 * pc)


def test_residuals():
    """Beyond 18 As y50, pc (0.5 (6 As)^0.5 - 0.411 - 0.75 As); beyond
    1.8 yp, 0.518 Ac pc. Where As is below 0.223 the static residual is
    negative; p is held at zero."""
    st, cy = _clay(), _clay("cyclic")
    for x in (0.3, 1.5, 6.0):
        pc, As, Ac = st.get_pu(x, B), st._get_As(x, B), cy._get_Ac(x, B)
        assert st.get_p(1.0, x, B) == pytest.approx(
            pc * (0.5 * math.sqrt(6 * As) - 0.411 - 0.75 * As))
        assert cy.get_p(1.0, x, B) == pytest.approx(
            (0.936 - 0.102 * 4.1) * Ac * pc)
    assert st._get_As(0.03, B) < 0.223
    assert st.get_p(1.0, 0.03, B) == 0.0
    assert st.get_p(0.01, 0.0, B) == 0.0      # the line is zero at the ground


def test_cyclic_is_below_static():
    st, cy = _clay(), _clay("cyclic")
    for x in (0.6, 1.5, 4.0):
        ys = np.linspace(0.0005, 0.006, 12)
        assert all(cy.get_p(y, x, B) < st.get_p(y, x, B) for y in ys)


def test_each_loading_takes_its_own_k():
    with pytest.raises(ValueError, match="needs ks"):
        StiffClayBelowWT(c=100.0, gamma=9.0, eps50=0.005)
    with pytest.raises(ValueError, match="needs kc.*108,600 kN/m3"):
        StiffClayBelowWT(c=100.0, gamma=9.0, eps50=0.005, loading="cyclic")
    with pytest.raises(ValueError, match="kc .* uses ks only"):
        _clay(kc=KC)
    with pytest.raises(ValueError, match="ks .* uses kc only"):
        _clay("cyclic", ks=KS)
    with pytest.raises(ValueError, match="kc must be positive"):
        StiffClayBelowWT(c=100.0, gamma=9.0, eps50=0.005, loading="cyclic",
                         kc=0.0)
    assert _clay().k == KS and _clay("cyclic").k == KC


def test_table_3_2_in_si():
    """500 / 1,000 / 2,000 and 200 / 400 / 800 lb/in^3; 0.5-1, 1-2 and
    2-4 ton/ft^2 (one ton/ft^2 = 95.76 kPa)."""
    for (lo, hi, k), pci in zip(STIFF_CLAY_K_TABLE["ks"], (500, 1000, 2000)):
        assert k == pytest.approx(pci * PCI, rel=5e-4)
    for (lo, hi, k), pci in zip(STIFF_CLAY_K_TABLE["kc"], (200, 400, 800)):
        assert k == pytest.approx(pci * PCI, rel=1e-3)
    assert [r[:2] for r in STIFF_CLAY_K_TABLE["ks"]] == [
        (48.0, 96.0), (96.0, 192.0), (192.0, 383.0)]


def test_get_py_curve_reaches_the_residual():
    for model, y_end in ((_clay(), 18 * 0.565 * 0.003),
                         (_clay("cyclic"), 1.8 * 4.1 * 0.305 * 0.003)):
        y, p = model.get_py_curve(X, B)
        assert y[-1] > y_end
        assert p[-1] == pytest.approx(model.get_p(1.0, X, B))


def _shaft(model, V):
    pile = Pile(length=15.0, diameter=0.9, E=25e6)
    return LateralPileAnalysis(pile, [SoilLayer(0.0, 15.0, model)]).solve(
        Vt=V, n_elements=150, max_iterations=400)


def test_cyclic_deflects_more_than_static():
    st = StiffClayBelowWT(c=100.0, gamma=9.0, eps50=0.005, ks=1000 * PCI)
    cy = StiffClayBelowWT(c=100.0, gamma=9.0, eps50=0.005, kc=400 * PCI,
                          loading="cyclic")
    a, b = _shaft(st, 300.0), _shaft(cy, 300.0)
    assert a.converged and b.converged
    assert a.y_top < b.y_top and a.max_moment < b.max_moment


def test_choices_and_calc_tables_name_the_k():
    r = _shaft(_clay("cyclic"), 300.0)
    k = next(c for c in r.choices if c["what"].startswith(
        "k of the initial line, kc"))
    assert k["chosen"] == "108600 kN/m3" and k["how"] == "given"
    assert "Table 3.2" in k["basis"]
    assert "54,300 kN/m3 for 48-96 kPa" in k["alternatives"]
    assert not any("108,600" in a for a in k["alternatives"])
    assert "kc=108600" in _extract_model_params(_clay("cyclic"))
    assert "ks=271400" in _extract_model_params(_clay())

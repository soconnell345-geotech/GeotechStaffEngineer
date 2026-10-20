"""Matlock (1970) soft clay, cyclic loading, above x_r (backlog GM-208), as
printed in the COM624P manual (FHWA-SA-91-048), pp. 323-327, Eq. 3.13 and
Fig. 3.5b, and in FHWA GEC 9 (FHWA-HIF-18-031, 2018) App. A.1, Eq. A-6.
Past 3 y50 the curve falls on a straight line from 0.72 pu at 3 y50 to
0.72 pu (x / x_r) at 15 y50, and stays there; below x_r it stays at
0.72 pu. Before GM-208 it dropped straight to 0.72 pu x / x_r at 3 y50.

Neither manual prints a cyclic soft-clay example, so the curve is pinned
to a HAND CALCULATION from the printed equations: c = 20 kPa,
gamma' = 8 kN/m^3, eps50 = 0.02, J = 0.5, b = 0.6 m, at x = 2 m:

    y50 = 2.5 eps50 b = 0.03 m                                 (3.10)
    pu  = min((3 + 8(2)/20 + 0.5(2)/0.6) 20 (0.6), 9 (20)(0.6))
        = min(65.6, 108) = 65.6 kN/m                        (3.8, 3.9)
    x_r = 6 (20)(0.6) / (8 (0.6) + 0.5 (20)) = 4.8649 m        (3.12)
    y = 2 y50:  0.5 pu 2^(1/3)                        = 41.325 kN/m
    y = 3 y50:  0.72 pu                               = 47.232 kN/m
    y = 15 y50 and beyond: 0.72 pu (2 / 4.8649)       = 19.418 kN/m
    y = 9 y50:  halfway, (47.232 + 19.418) / 2        = 33.325 kN/m
"""

import pytest

from lateral_pile.py_curves import SoftClayMatlock

B, X, Y50 = 0.6, 2.0, 0.03


def _clay(loading="cyclic"):
    return SoftClayMatlock(c=20.0, gamma=8.0, eps50=0.02, loading=loading)


def test_x_r_and_pu():
    clay = _clay()
    assert clay.get_zr(B) == pytest.approx(72.0 / 14.8)
    assert clay.get_pu(X, B) == pytest.approx(65.6)


@pytest.mark.parametrize("n_y50, p", [
    (2.0, 41.325),
    (3.0, 47.232),
    (6.0, 47.232 - 0.25 * (47.232 - 19.418)),
    (9.0, 33.325),
    (15.0, 19.418),
    (25.0, 19.418),
])
def test_above_x_r_hand_calculation(n_y50, p):
    clay = _clay()
    assert clay.get_p(n_y50 * Y50, X, B) == pytest.approx(p, abs=2e-3)
    assert clay.get_p(-n_y50 * Y50, X, B) == pytest.approx(-p, abs=2e-3)


def test_below_x_r_stays_at_0_72_pu():
    clay = _clay()
    x = 6.0
    pu = clay.get_pu(x, B)
    for n_y50 in (3.0, 9.0, 15.0, 30.0):
        assert clay.get_p(n_y50 * Y50, x, B) == pytest.approx(0.72 * pu)


def test_static_curve_until_0_72_pu():
    """The cyclic curve is the static one while p is below 0.72 pu, which
    the static curve reaches at 2.99 y50; it never exceeds 0.72 pu."""
    cyc, sta = _clay(), _clay("static")
    pu = cyc.get_pu(X, B)
    for n_y50 in (0.1, 0.5, 1.0, 2.5, 2.98):
        y = n_y50 * Y50
        assert cyc.get_p(y, X, B) == pytest.approx(sta.get_p(y, X, B))
    assert cyc.get_p(2.995 * Y50, X, B) == pytest.approx(0.72 * pu)


def test_softer_than_static_and_continuous():
    cyc, sta = _clay(), _clay("static")
    for i in range(1, 601):
        y = i * 0.05 * Y50
        assert cyc.get_p(y, X, B) <= sta.get_p(y, X, B) + 1e-9
    # no step at 3 y50 or 15 y50 (before GM-208: 47.2 to 19.4 kN/m at 3 y50)
    for n_y50 in (3.0, 15.0):
        lo, hi = (cyc.get_p(n_y50 * Y50 * f, X, B) for f in (1 - 1e-9,
                                                              1 + 1e-9))
        assert lo == pytest.approx(hi, abs=1e-3)

"""When the p-y iteration stops without a solution (backlog GM-208): the
iteration that needs more steps, and the load the soil cannot resist.

A 0.9 m, 15 m shaft (E 25 GPa, free head) in stiff clay below the water
table (c 100 kPa, gamma' 9 kN/m^3, eps50 0.005; Reese, Cox & Koop as
printed in COM624P pp. 328-335):

- static (ks 271,400) at 900 kN and cyclic (kc 108,600) at 600 kN need
  133 and 126 secant iterations, past the old default of 100;
- cyclic at 900 kN has no solution. Pushing the head (displacement
  control) shows the load rising to about 890 kN at a head deflection of
  about 274 mm and falling past it, as the cyclic curves fall from Ac pc
  to 0.52 Ac pc (Eqs. 3.24-3.26). Load control ran away: at 100
  iterations the head deflection was 305 mm and climbing; at 643 it
  "converged" at 1.5e10 mm, once every secant modulus was too small to
  change the solve.
"""

import warnings

import pytest

from lateral_pile import (LateralPileAnalysis, Pile,
                          ReinforcedConcreteSection, SoilLayer)
from lateral_pile.py_curves import SoftClayMatlock, StiffClayBelowWT
from lateral_pile.solver import SoilResistanceExceeded


def _clay(loading="static"):
    k = {"ks": 271_400.0} if loading == "static" else {"kc": 108_600.0}
    return StiffClayBelowWT(c=100.0, gamma=9.0, eps50=0.005,
                            loading=loading, **k)


def _shaft(model, V, **kw):
    pile = Pile(length=15.0, diameter=0.9, E=25e6)
    return LateralPileAnalysis(pile, [SoilLayer(0.0, 15.0, model)]).solve(
        Vt=V, **kw)


@pytest.mark.parametrize("loading, V, iterations, y_mm", [
    ("static", 600.0, 43, 12.13),
    ("static", 900.0, 133, 40.79),
    ("cyclic", 600.0, 126, 63.04),
])
def test_converges_within_the_default(loading, V, iterations, y_mm):
    r = _shaft(_clay(loading), V)
    assert r.converged and r.message is None
    assert r.iterations == iterations
    assert r.y_top * 1000 == pytest.approx(y_mm, abs=0.01)


@pytest.mark.parametrize("max_iterations", [500, 2000])
def test_load_beyond_what_the_soil_can_resist(max_iterations):
    with pytest.raises(SoilResistanceExceeded) as exc:
        _shaft(_clay("cyclic"), 900.0, max_iterations=max_iterations)
    e = exc.value
    assert str(e).startswith(
        "No solution: the soil cannot resist this load.")
    assert "did not converge" not in str(e)
    assert e.load_factor * 900.0 == pytest.approx(890.0, abs=3.0)
    assert e.head_deflection == pytest.approx(0.274, abs=0.01)
    assert "about 890 kN (applied: 900 kN)" in str(e)
    # just inside the peak there is a solution
    r = _shaft(_clay("cyclic"), 880.0)
    assert r.converged and 0.2 < r.y_top < 0.274


def test_curves_that_do_not_soften():
    """Matlock soft clay, cyclic, c 20 kPa: the resistance is approached
    only as the pile moves without bound, and 450 kN is past it."""
    with pytest.raises(SoilResistanceExceeded, match="cannot resist"):
        _shaft(SoftClayMatlock(c=20.0, gamma=8.0, eps50=0.02,
                               loading="cyclic"), 450.0)


def test_an_iteration_that_stops_short_says_so():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = _shaft(_clay("cyclic"), 600.0, max_iterations=10)
    assert not r.converged
    assert "did not converge in 10 iterations" in r.message
    assert "it is the iteration that stopped, not the soil" in r.message
    assert any("raise max_iterations" in str(x.message) for x in w)
    d = r.to_dict()
    assert next(iter(d)) == "message" and d["converged"] is False
    assert r.message in r.summary()


def test_cracked_section_with_stickup_keeps_the_stickup():
    """The cracked-EI path's final solve carried no stickup, so the
    result's mesh started at the ground while it reported one."""
    rc = ReinforcedConcreteSection(diameter=0.9, fc=35000.0, fy=420000.0,
                                   n_bars=12, bar_diameter=0.02546,
                                   cover=0.075)
    pile = Pile.from_rc_section(length=15.0, rc_section=rc)
    layers = [SoilLayer(0.0, 15.0, _clay())]
    r = LateralPileAnalysis(pile, layers).solve(Vt=200.0, stickup=2.0)
    assert r.z[0] == pytest.approx(-2.0)
    assert r.y_top > r.y_ground > 0

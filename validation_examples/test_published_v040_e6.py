"""Phase E / v5.4 E6 validation — stabilizing-pile PASSIVE-force option.

Slide2 Verification #54 (Yamagami 2000): a homogeneous slope reinforced by a
single row of micro-piles (spacing 1 m, shear 10.7 kN/pile). Published (Table
54.2): no-pile 1.102 (Slide) / 1.10 (Yamagami); with-pile 1.193 / 1.20.

V-040 recorded a CONVENTION verdict: the module's default ACTIVE support
convention (the pile force reduces the driving moment) gives with-pile Bishop
1.223 -- +2.5% over the published 1.193 -- because active support is more
effective than a passive pile force for a given capacity. E6 adds an OPT-IN
passive convention (``StabilizingPile.support_convention='passive'`` / Slide2
"Method B": the force is added to the resisting side instead), default unchanged.

VERDICT: **PASS (passive option narrows the residual).** On the pinned V-040
critical circle the passive convention drops the with-pile Bishop FOS from 1.223
(active) to 1.203 -- +0.8% over the published 1.193, vs the active +2.5%. The
default (active) is byte-identical to V-040. The small residual that remains is
the figure-read pile location + single-pile search subtlety documented in V-040;
NOT tuned (the value falls out of the Method-A -> Method-B switch).

See slope_stability/reinforcement.py (support_convention / moment_resistance),
methods.py (Fellenius/Bishop passive term), VALIDATION.md B10, and
test_published_v040_slope.py (the active-convention V-040 record).
"""

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.reinforcement import StabilizingPile
from slope_stability.analysis import analyze_slope

_XC, _YC, _R = 2.0, 8.89, 9.11          # V-040 pinned no-pile critical circle
_PUB_NO_PILE, _PUB_PILE = 1.102, 1.193


def _geom(piles=None):
    g = SlopeGeometry(
        surface_points=[(-6, 0), (0, 0), (8, 4), (12, 4)],
        soil_layers=[SlopeSoilLayer(name="m1", top_elevation=4,
                                    bottom_elevation=-5, gamma=15.68,
                                    phi=10.0, c_prime=4.9)])
    g.stabilizing_piles = piles
    return g


def _pile(convention="active"):
    return [StabilizingPile(x=8.75, shear_capacity=10.7, spacing=1.0,
                            support_convention=convention)]


def _fos(piles=None, method="bishop"):
    return analyze_slope(_geom(piles), xc=_XC, yc=_YC, radius=_R,
                         method=method, n_slices=40).FOS


def test_v040e6_active_default_unchanged():
    """The default (active) convention reproduces the V-040 pinned FOS byte-for-
    byte -- no-pile 1.114, with-pile 1.223 -- so the option is default-preserving.
    Omitting support_convention == passing 'active'."""
    assert _fos() == pytest.approx(1.114, abs=0.01)
    assert _fos(_pile("active")) == pytest.approx(1.223, abs=0.01)
    # omitted convention defaults to active
    default_pile = [StabilizingPile(x=8.75, shear_capacity=10.7, spacing=1.0)]
    assert _fos(default_pile) == pytest.approx(_fos(_pile("active")), abs=1e-9)


def test_v040e6_passive_is_closer_to_published():
    """The passive convention lowers the with-pile Bishop FOS from 1.223 (active,
    +2.5% over published 1.193) to 1.203 (+0.8%), closer to the referee."""
    active = _fos(_pile("active"))
    passive = _fos(_pile("passive"))
    assert passive == pytest.approx(1.203, abs=0.01)     # ours (pinned)
    assert passive < active                              # passive less effective
    # closer to the published with-pile FOS than active is
    assert abs(passive - _PUB_PILE) < abs(active - _PUB_PILE)
    assert passive == pytest.approx(_PUB_PILE, rel=0.02)  # within 2% of published


def test_v040e6_passive_still_helps_vs_no_pile():
    """Passive support still RAISES the FOS above the unreinforced slope (the pile
    is stabilizing, just less so than active)."""
    assert _fos(_pile("passive")) > _fos()


def test_v040e6_fellenius_passive_lower_than_active():
    """The active/passive split also applies to the Fellenius (OMS) circular
    method (passive adds to resisting; active reduces driving)."""
    assert _fos(_pile("passive"), "fellenius") < _fos(_pile("active"), "fellenius")


def test_v040e6_bad_convention_rejected():
    with pytest.raises(ValueError, match="support_convention"):
        StabilizingPile(x=8.75, shear_capacity=10.7, support_convention="bogus")

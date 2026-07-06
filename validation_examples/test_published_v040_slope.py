"""Phase E / v5.3 B2d validation — stabilizing-pile slope reinforcement.

Slide2 Verification #54 (manual pp. 196-198; Yamagami 2000) — a homogeneous slope
reinforced by a single row of micro-piles. Geometry (Fig 54.1, LABELED): surface
(-6,0)-(0,0)-(8,4)-(12,4), base to z=-5; soil c'=4.9 kPa / phi'=10 / gamma=15.68.
A single row of piles at the crest, spaced 1 m, with a shear strength of 10.7 kN
per pile; circular failure surface.

Published: no-pile 1.102 (Slide) / 1.10 (Yamagami); with-pile 1.193 / 1.20.

VERDICT: **CONVENTION (pile reinforcement validated).** The pile-force INTEGRATION
is exact (see slope_stability/tests/test_reinforcement.py::TestStabilizingPile:
for a phi=0 case the reinforced FOS equals M_R/(M_D - (Fpile/spacing)*d_perp/R)
to machine precision). Here:
  * no-pile critical circle: 1.114 vs published 1.102 (+1.1%);
  * with the 10.7 kN/m pile on that surface (at the figure crest location
    x~=8.75): 1.223 vs published 1.193 (+2.5%).
The ~2.5% over-prediction of the pile benefit is the support-force convention
(the module applies the pile force ACTIVELY — reducing the driving moment — which
is slightly more effective than a passive resisting force) plus the figure-read
pile location and the single-pile search subtlety (a re-search WITH the pile finds
a pile-avoiding surface at ~1.11, so the verification applies the pile to the
recovered critical surface, as Slide reports). NOT tuned.

Slide2 #106 (Cai & Ugai 2000, Ito-Matsui pile vs spacing) is the target for the
Ito & Matsui (1975) plastic-force FORMULA itself; its cross-section (Fig 106.1) is
not in the manual extract, so the formula is unit-tested for the published spacing
TREND (force per metre falls as spacing/diameter grows — #106: FS 1.54/1.37/1.31/
1.25 for ratio 2/3/4/6) rather than a single-surface FOS. See INVENTORY (V-040).
"""

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.reinforcement import StabilizingPile
from slope_stability.analysis import analyze_slope

# Recovered critical circle (grid search) for the no-pile slope.
_XC, _YC, _R = 2.0, 8.89, 9.11


def _geom(piles=None):
    g = SlopeGeometry(
        surface_points=[(-6, 0), (0, 0), (8, 4), (12, 4)],
        soil_layers=[SlopeSoilLayer(name="m1", top_elevation=4,
                                    bottom_elevation=-5, gamma=15.68,
                                    phi=10.0, c_prime=4.9)])
    g.stabilizing_piles = piles
    return g


def _fos(method, piles=None):
    return analyze_slope(_geom(piles), xc=_XC, yc=_YC, radius=_R,
                         method=method, n_slices=40).FOS


def test_v040_no_pile_critical():
    """No-pile critical circle ~ 1.11 vs published 1.102 (+1.1%)."""
    assert _fos("bishop") == pytest.approx(1.114, abs=0.01)
    assert _fos("bishop") == pytest.approx(1.102, rel=0.03)


def test_v040_pile_raises_fos_toward_published():
    """The 10.7 kN/m micro-pile row raises the FOS to ~1.22 vs published 1.193
    (+2.5%); pile integration validated, residual = active-vs-passive convention
    + figure-read pile location."""
    pile = [StabilizingPile(x=8.75, shear_capacity=10.7, spacing=1.0)]
    with_pile = _fos("bishop", pile)
    assert with_pile == pytest.approx(1.223, abs=0.01)     # ours (pinned)
    assert with_pile > _fos("bishop")                      # pile helps
    assert with_pile == pytest.approx(1.193, rel=0.03)     # within 3% published


def test_v040_pile_scales_with_capacity():
    """A stronger pile gives a larger FOS increase (the active driving-moment
    reduction FOS = M_R/(M_D - red) grows faster than linearly in capacity)."""
    base = _fos("bishop")
    d1 = _fos("bishop", [StabilizingPile(x=8.75, shear_capacity=10.7)]) - base
    d2 = _fos("bishop", [StabilizingPile(x=8.75, shear_capacity=21.4)]) - base
    assert d2 > d1 > 0
    assert d2 > 1.5 * d1        # super-linear (active convention), not zero


def test_v040_ito_matsui_formula_spacing_trend():
    """#106 trend: the Ito-Matsui plastic lateral force per metre of wall falls
    monotonically as the pile spacing/diameter ratio grows (why #106 FS drops
    1.54 -> 1.25 for ratio 2 -> 6)."""
    from slope_stability.reinforcement import ito_matsui_lateral_force
    B = 0.8
    fpm = [ito_matsui_lateral_force(10.0, 20.0, 18.0, r * B, r * B - B,
                                    z_top=6.0, z_bot=0.0) / (r * B)
           for r in (2, 3, 4, 6)]
    assert fpm == sorted(fpm, reverse=True)
    assert all(v > 0 for v in fpm)

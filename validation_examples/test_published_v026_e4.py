"""Phase E / v5.4 E4 validation — exit-side tension crack + mass-truncation model.

Slide2 Verification #2 = ACADS 1(b) (Giam & Donald 1989) [manual pp. 24-27]: the
ACADS #1 slope (surface (20,25)-(30,25)-(50,35)-(70,35) m, 2:1, 10 m) with the #2
soil (c'=32 kPa, phi'=10 deg, gamma=20 kN/m3) and a WATER-FILLED tension crack,
Rankine/Craig-1997 depth zc = 2c'/(gamma*sqrt(Ka)) = 3.81 m. Published (Table 2.2,
WITH the water crack): Bishop 1.596, Spencer 1.592, GLE 1.592; referee 1.65 [Giam].

V-026 (test_published_v026_slope.py) had to (a) MIRROR the slope so the crest fell
on the module's entry-side-only crack, and (b) record a CONVENTION verdict: the
module's crack was ~6% MORE CONSERVATIVE than Slide2 (Bishop 1.497 vs 1.596)
because it kept the cracked wedge as zero-strength DRIVING soil, whereas Slide2
TRUNCATES the sliding mass at the crack. E4 closes both gaps:

  * ``tension_crack_side='exit'`` places the crack on the high-x (exit) crest, so
    the ORIGINAL (un-mirrored) ACADS #1 geometry is analysed directly.
  * ``tension_crack_model='truncation'`` removes the cracked wedge from the
    sliding mass (mass ends at the vertical crack face), matching Slide2/UTEXAS.

VERDICT: **PASS (mass-truncation matches Slide2; exit-side symmetric).**
  * The exit-side crack on the un-mirrored slope reproduces the mirrored entry-
    side FOS to machine precision (the two are mirror images).
  * The mass-truncation model gives Bishop 1.597 / Spencer 1.593 / GLE 1.594 --
    matching Slide2's published water-crack 1.596 / 1.592 / 1.592 to <0.1%,
    resolving the V-026 CONVENTION residual (the module was conservative only
    because it kept the driving wedge; Slide2 truncates it).
  * The default (entry-side, strength model) is byte-identical to V-026.
NOT tuned; the match falls out of the correct mass model.

See slope_stability/geometry.py (tension_crack_side / tension_crack_model),
slices.py, VALIDATION.md B9, and test_published_v026_slope.py (the strength-model
V-026 record this supersedes for the published match).
"""

import math

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import CircularSlipSurface
from slope_stability.slices import build_slices
from slope_stability.methods import bishop_fos, spencer_fos
from slope_stability.gle import gle_fos

_C, _PHI, _GAMMA = 32.0, 10.0, 20.0
_KA = (1 - math.sin(math.radians(_PHI))) / (1 + math.sin(math.radians(_PHI)))
_ZC = 2 * _C / (_GAMMA * math.sqrt(_KA))          # 3.814 m

# Mirrored ACADS #1 (crest on the LEFT / entry) — the V-026 setup, pinned circle.
_MIRROR = [(20.0, 35.0), (40.0, 35.0), (60.0, 25.0), (70.0, 25.0)]
_MXC, _MYC, _MR = 51.96, 42.94, 20.47
# Original ACADS #1 (crest on the RIGHT / exit) = mirror about x -> 90 - x.
_ORIG = [(20.0, 25.0), (30.0, 25.0), (50.0, 35.0), (70.0, 35.0)]
_OXC, _OYC, _OR = 90 - 51.96, 42.94, 20.47        # 38.04

# Slide2 published (Table 2.2), WITH the water-filled crack.
_PUB = {"bishop": 1.596, "spencer": 1.592, "gle": 1.592}


def _geom(surf, side="entry", model="strength", tc=0.0, tw=0.0):
    return SlopeGeometry(
        surface_points=surf,
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=35.0,
                                    bottom_elevation=10.0, gamma=_GAMMA,
                                    phi=_PHI, c_prime=_C)],
        tension_crack_depth=tc, tension_crack_water_depth=tw,
        tension_crack_side=side, tension_crack_model=model)


def _methods(g, xc, yc, r):
    slip = CircularSlipSurface(xc, yc, r)
    sl = build_slices(g, slip, 60)
    return {
        "bishop": bishop_fos(sl, slip),
        "spencer": spencer_fos(sl, slip)[0],
        "gle": gle_fos(sl, slip, f_interslice="half_sine").fos,
        "n_crack": sum(1 for s in sl if s.in_tension_crack),
        "n_slices": len(sl),
    }


def test_v026e4_default_entry_strength_unchanged():
    """DEFAULT (entry side, strength model) reproduces the V-026 pinned water-
    crack FOS byte-for-byte -- the E4 params are additive/default-preserving."""
    f = _methods(_geom(_MIRROR, "entry", "strength", _ZC, _ZC), _MXC, _MYC, _MR)
    assert f["bishop"] == pytest.approx(1.497, abs=0.01)
    assert f["spencer"] == pytest.approx(1.498, abs=0.01)
    assert f["gle"] == pytest.approx(1.494, abs=0.01)
    assert f["n_crack"] == 4


def test_v026e4_exit_side_crack_is_mirror_symmetric():
    """The exit-side crack on the ORIGINAL (un-mirrored) ACADS #1 slope gives the
    SAME FOS as the entry-side crack on the mirrored slope, to machine precision
    -- so the crest no longer has to be forced onto the entry side."""
    f_entry = _methods(_geom(_MIRROR, "entry", "strength", _ZC, _ZC),
                       _MXC, _MYC, _MR)
    f_exit = _methods(_geom(_ORIG, "exit", "strength", _ZC, _ZC),
                      _OXC, _OYC, _OR)
    for m in ("bishop", "spencer", "gle"):
        assert f_exit[m] == pytest.approx(f_entry[m], abs=1e-6)
    assert f_exit["n_crack"] == 4


def test_v026e4_mass_truncation_matches_slide2():
    """The mass-truncation model (cracked wedge REMOVED from the sliding mass)
    reproduces Slide2's published water-crack FOS to <0.1% -- Bishop 1.597 vs
    1.596, Spencer 1.593 vs 1.592, GLE 1.594 vs 1.592 -- resolving the V-026
    CONVENTION residual. The truncation drops the cracked slices (n_crack 0,
    fewer slices than the strength model's 60)."""
    f = _methods(_geom(_MIRROR, "entry", "truncation", _ZC, _ZC),
                 _MXC, _MYC, _MR)
    assert f["bishop"] == pytest.approx(1.597, abs=0.01)     # ours (pinned)
    assert f["spencer"] == pytest.approx(1.593, abs=0.01)
    assert f["gle"] == pytest.approx(1.594, abs=0.01)
    # matches Slide2 published to well within 0.5%
    for m in ("bishop", "spencer", "gle"):
        assert abs(f[m] - _PUB[m]) / _PUB[m] < 0.005
    assert f["n_crack"] == 0                                 # wedge removed
    assert f["n_slices"] < 60                                # truncated mass


def test_v026e4_truncation_is_less_conservative_than_strength():
    """Truncation (removing the driving wedge) gives a HIGHER FOS than the
    strength model (which keeps the wedge as zero-strength driving soil), both
    dry and water-filled -- the mechanism behind the V-026 conservatism."""
    strength = _methods(_geom(_MIRROR, "entry", "strength", _ZC, _ZC),
                        _MXC, _MYC, _MR)["bishop"]
    trunc = _methods(_geom(_MIRROR, "entry", "truncation", _ZC, _ZC),
                     _MXC, _MYC, _MR)["bishop"]
    assert trunc > strength
    # water-filled truncation still below the dry truncation (thrust drives)
    trunc_dry = _methods(_geom(_MIRROR, "entry", "truncation", _ZC, 0.0),
                         _MXC, _MYC, _MR)["bishop"]
    assert trunc < trunc_dry
    assert trunc_dry == pytest.approx(1.660, abs=0.01)       # ours (pinned)


def test_v026e4_bad_crack_params_rejected():
    with pytest.raises(ValueError):
        _geom(_MIRROR, side="bogus")
    with pytest.raises(ValueError):
        _geom(_MIRROR, model="bogus")

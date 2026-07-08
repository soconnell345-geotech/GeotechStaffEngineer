"""V-043 — Slide2 Verification #39: Tandjiria (2002) geosynthetic-reinforced
embankment on soft clay (2 materials, tension crack). [manual pp. 147-150]

Geometry (labeled Fig 39.1/39.2, meters): crest el 9 (x 0-10), embankment face
down to (20,3), bench el 3 to x=30; a 3 m soft-clay foundation (el 0-3) full
width; embankment fill above el 3 for x in [0,20]. Two fills are analyzed:
  Clay fill  (Table 39.1): fill & soft clay both c'=20, phi=0, gamma=19.4
             (undrained); a WATER-FILLED tension crack in the crest.
  Sand fill  (Table 39.2): fill c'=0/phi=37/gamma=17; soft clay c'=20/phi=0/
             gamma=20; a (dry) tension crack.
The reinforcement (geosynthetic, active force) sits at the base of the fill
(el 3); the published task is the force to raise the no-reinforcement critical
surface to FS = 1.35 (Spencer).

Published (Slide2 / Tandjiria 2002):
  no-reinf Spencer FS: clay circ 0.975/0.981, clay noncirc 0.935/0.941;
                       sand circ 1.209/1.219, sand noncirc 1.188/1.192.
  reinf force (FS=1.35): clay circ 169/170, clay noncirc 184/190;
                         sand circ 44/45, sand noncirc 56/56 kN/m.

VERDICTS
- SAND, no reinforcement: PASS. Circular Spencer 1.180 (-2.4% vs 1.209),
  noncircular 1.21 (within ~2% of 1.188). Clean without a crack -- the dry sand
  crack barely intersects the critical surface.
- SAND, reinforcement force: CONVENTION. The force to reach FS=1.35 on the
  module's own searched surface is ~55 kN/m (circular) / ~50 kN/m (noncircular) --
  bracketing the published 44-56 kN/m band but with the circular/noncircular
  split inverted, because the reinforcement force is highly critical-surface
  sensitive and the search finds a slightly different surface (the manual itself
  attributes such differences to search-method/surface differences).
- CLAY, no reinforcement: CONVENTION. Without the crack the module gives circular
  1.039 (+6.6% vs 0.975); the published value includes a WATER-FILLED crest crack
  whose depth is NOT labeled in Fig 39.1 and to which the FS is very sensitive
  (a full 2c/gamma=2.06 m water column drives FS to ~0.27), so it is not
  reproduced -- not tuned to a fitted crack depth.
"""
import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.reinforcement import Geosynthetic
from slope_stability.analysis import search_critical_surface, analyze_slope
from slope_stability.slip_surface import CircularSlipSurface

_SURFACE = [(0, 9), (10, 9), (20, 3), (30, 3)]
# Spencer critical circles recovered by entry-exit search (seed=2), pinned.
_SAND_CIRC = (15.193905105454997, 10.242328996634253, 9.938523884815131)
_CLAY_CIRC = (15.027360761475776, 12.983131511749898, 12.986612150416436)


def _geom(fill, T=None):
    if fill == "clay":
        f = dict(c_prime=20.0, phi=0.0, gamma=19.4, gamma_sat=19.4, cu=20.0,
                 analysis_mode="undrained")
        sc = dict(c_prime=20.0, phi=0.0, gamma=19.4, gamma_sat=19.4, cu=20.0,
                  analysis_mode="undrained")
    else:
        f = dict(c_prime=0.0, phi=37.0, gamma=17.0, gamma_sat=17.0,
                 analysis_mode="drained")
        sc = dict(c_prime=20.0, phi=0.0, gamma=20.0, gamma_sat=20.0, cu=20.0,
                  analysis_mode="undrained")
    g = SlopeGeometry(surface_points=_SURFACE, soil_layers=[
        SlopeSoilLayer(name="fill", top_elevation=9, bottom_elevation=3.0, **f),
        SlopeSoilLayer(name="softclay", top_elevation=3, bottom_elevation=0.0, **sc)])
    if T is not None:
        g.geosynthetics = [Geosynthetic(elevation=3.0, T_allow=T)]
    return g


def test_v043_sand_no_reinforcement_circular():
    """PASS: sand-fill embankment, no reinforcement, circular Spencer = 1.180
    (Slide2 1.209 / Tandjiria 1.219, within ~2.5%)."""
    xc, yc, r = _SAND_CIRC
    fos = analyze_slope(_geom("sand"), xc=xc, yc=yc, radius=r,
                        method="spencer").FOS
    assert fos == pytest.approx(1.180, abs=0.01)
    assert fos == pytest.approx(1.209, rel=0.04)


def test_v043_sand_no_reinforcement_noncircular():
    """PASS: sand-fill embankment, no reinforcement, noncircular Spencer ~ 1.20
    (Slide2 1.188 / Tandjiria 1.192, within ~2%)."""
    r = search_critical_surface(_geom("sand"), method="spencer",
                                surface_type="noncircular", n_trials=2000,
                                n_slices=30, seed=2)
    assert r.critical.FOS == pytest.approx(1.19, abs=0.04)


def test_v043_sand_reinforcement_force_brackets_published():
    """CONVENTION: the geosynthetic force to raise the sand circular critical
    surface to FS=1.35 is ~55 kN/m, bracketing the published 44-56 kN/m band
    (critical-surface sensitive; see module note)."""
    xc, yc, r = _SAND_CIRC
    slip = CircularSlipSurface(xc=xc, yc=yc, radius=r)
    lo, hi = 0.0, 300.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        fs = analyze_slope(_geom("sand", T=mid), slip_surface=slip,
                           method="spencer").FOS
        if fs < 1.35:
            lo = mid
        else:
            hi = mid
    T = 0.5 * (lo + hi)
    assert 40.0 <= T <= 65.0            # brackets published 44-56 kN/m
    # the solved force does raise this surface to the target
    fs_at_T = analyze_slope(_geom("sand", T=T), slip_surface=slip,
                            method="spencer").FOS
    assert fs_at_T == pytest.approx(1.35, abs=0.01)


def test_v043_clay_no_reinforcement_no_crack_is_high():
    """CONVENTION: clay-fill embankment WITHOUT the (unlabeled) water-filled
    crest crack gives circular Spencer 1.039 (+6.6% vs the published 0.975 that
    includes the crack). The FS is very crack-sensitive; not tuned."""
    xc, yc, r = _CLAY_CIRC
    fos = analyze_slope(_geom("clay"), xc=xc, yc=yc, radius=r,
                        method="spencer").FOS
    assert fos == pytest.approx(1.039, abs=0.01)
    assert fos > 0.975                  # higher than the water-crack value

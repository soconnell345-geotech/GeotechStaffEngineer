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
- SAND, no reinforcement: PASS (circular). Circular Spencer 1.210 (+0.05% vs
  1.209), 1.208-1.211 on 25 to 45 slices and 1.213 on 3,840. Before GM-134 it
  was pinned at 1.180 (-2.4%), a 30-slice value: the circle's slices were cut
  evenly, so one across the crossing of the fill's base took one material
  (1.246, 1.190, 1.180, 1.232 on 25, 29, 30, 31 slices); the crossings are
  slice boundaries now. Clean without a crack -- the dry sand crack barely
  intersects the critical surface. Noncircular: a KNOWN GAP (GM-117). Differential evolution
  reaches 1.234 (+3.8% on 1.188), above our own converged critical circle
  (1.213 on 60 slices or more) and a 41-point polyline inscribed in it
  (1.2145): the search falls short of a surface it should reach or beat.
- SAND, reinforcement force: CONVENTION. The force to reach FS=1.35 on the
  module's own searched surface is 41 kN/m (circular; 40 on 240 slices) /
  ~50 kN/m (noncircular), against the published 44-45 / 56 kN/m: the circular
  7% below, the split the same way round. The reinforcement force is highly
  critical-surface sensitive (the manual itself attributes such differences to
  search-method/surface differences). Before GM-134 the circular force was
  ~55 kN/m, from the 30-slice 1.180 above.
- CLAY, no reinforcement: CONVENTION. Without the crack the module gives circular
  1.041 (+6.8% vs 0.975); the published value includes a WATER-FILLED crest crack
  whose depth is NOT labeled in Fig 39.1 and to which the FS is very sensitive
  (a full 2c/gamma=2.06 m water column drives FS to ~0.27), so it is not
  reproduced -- not tuned to a fitted crack depth.
"""
import functools
import statistics

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.reinforcement import Geosynthetic
from slope_stability.analysis import search_critical_surface, analyze_slope
from slope_stability.slip_surface import CircularSlipSurface

_SURFACE = [(0, 9), (10, 9), (20, 3), (30, 3)]
# Spencer critical circles recovered by entry-exit search (seed=2), pinned.
# The clay circle came 3.5 mm under the firm base at el 0; its radius is cut
# to touch the base 0.1 mm above it, since a circle under the base is refused
# at any slice count (GM-134).
_SAND_CIRC = (15.193905105454997, 10.242328996634253, 9.938523884815131)
_CLAY_CIRC = (15.027360761475776, 12.983131511749898, 12.983031511749898)


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
    """PASS: sand-fill embankment, no reinforcement, circular Spencer = 1.210
    (Slide2 1.209 / Tandjiria 1.219, within 1%); 1.180 before GM-134, a
    30-slice value (module docstring)."""
    xc, yc, r = _SAND_CIRC
    fos = analyze_slope(_geom("sand"), xc=xc, yc=yc, radius=r,
                        method="spencer").FOS
    assert fos == pytest.approx(1.210, abs=0.003)
    assert fos == pytest.approx(1.209, rel=0.01)


@functools.lru_cache(maxsize=1)
def _sand_de():
    """Differential evolution on the sand fill, 30 slices, seeds 1-3: the
    noncircular minima."""
    return tuple(search_critical_surface(
        _geom("sand"), method="spencer", surface_type="noncircular_de",
        n_slices=30, seed=seed).candidates["noncircular"]
        for seed in (1, 2, 3))


def test_v043_sand_no_reinforcement_noncircular():
    """Sand-fill embankment, no reinforcement, noncircular Spencer (Slide2
    1.188 / Tandjiria 1.192): what the searches reach. The comparison with
    the published is the known gap below.

    Differential evolution on 30 slices reaches 1.234 from seeds 1, 2 and 3,
    and its surface reads 1.233-1.234 on 31, 60 and 120 slices. Before
    GM-131 cut a slice at every corner, DE reported 1.09-1.10 on surfaces
    that dip steeply into the soft clay and read 1.24-1.32 with one slice
    more: even slices across their corners. The random search (2000 trials)
    stays above DE: 1.34 and 1.46 from seeds 1 and 2 (1.19 and 1.27 before,
    the same artefact), so DE is the search compared with the published.
    (Before GM-57, DE seed 1 reported 0.717 on a surface whose rigorous
    state held about one moment axis at 30 slices only; see
    test_search_robustness.)"""
    de = list(_sand_de())
    assert de == pytest.approx([1.234] * 3, abs=0.002)
    for seed in (1, 2):
        r = search_critical_surface(_geom("sand"), method="spencer",
                                    surface_type="noncircular", n_trials=2000,
                                    n_slices=30, seed=seed)
        assert r.candidates["noncircular"] >= min(de) - 1e-9


@pytest.mark.xfail(strict=True, reason=(
    "GM-117: the noncircular search falls short. Its best, 1.234, is above "
    "our own converged critical circle (1.213 on 60 slices or more) and a "
    "41-point polyline inscribed in it (1.2145), so DE misses surfaces it "
    "should reach or beat; Slide2 1.188"))
def test_v043_sand_noncircular_reaches_the_published():
    """Known gap: the noncircular minimum within 0.04 of Slide2's 1.188 /
    Tandjiria's 1.192 (it was, at 1.19, only through even slices across the
    corners of surfaces diving into the soft clay, before GM-131)."""
    assert statistics.median(_sand_de()) == pytest.approx(1.19, abs=0.04)


def test_v043_sand_reinforcement_force_brackets_published():
    """CONVENTION: the geosynthetic force to raise the sand circular critical
    surface to FS=1.35 is 41 kN/m, 7% below the published circular 44-45
    kN/m (critical-surface sensitive; see module note). ~55 kN/m before
    GM-134."""
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
    assert T == pytest.approx(40.9, abs=1.0)            # ours (pinned)
    assert T == pytest.approx(44.0, rel=0.1)    # within 10% of published 44
    # the solved force does raise this surface to the target
    fs_at_T = analyze_slope(_geom("sand", T=T), slip_surface=slip,
                            method="spencer").FOS
    assert fs_at_T == pytest.approx(1.35, abs=0.01)


def test_v043_clay_no_reinforcement_no_crack_is_high():
    """CONVENTION: clay-fill embankment WITHOUT the (unlabeled) water-filled
    crest crack gives circular Spencer 1.041 (+6.8% vs the published 0.975 that
    includes the crack). The FS is very crack-sensitive; not tuned."""
    xc, yc, r = _CLAY_CIRC
    fos = analyze_slope(_geom("clay"), xc=xc, yc=yc, radius=r,
                        method="spencer").FOS
    assert fos == pytest.approx(1.041, abs=0.01)
    assert fos > 0.975                  # higher than the water-crack value

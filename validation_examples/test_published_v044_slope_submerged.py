"""V-044 — Slide2 Verification #70: Duncan & Wright (2005) Fig 6.27 submerged
homogeneous slope with ponded water. [manual pp. 233-236]

Geometry (labeled Fig 70.1/70.2, feet): homogeneous slope, base el 0, surface
(0,15),(30,15),(105,45),(140,45) (a 2.5:1 face from (30,15) to (105,45)); the
whole slope is SUBMERGED under ponded water. Case 1 water table at el 75 ft
(30 ft above the crest el 45); Case 2 at el 105 ft (60 ft above). Homogeneous
gamma = 128 pcf, c' = 100 psf, phi' = 20 deg. Circular auto-refine search.

Published (Tables 70.2/70.3): Bishop 1.603, Spencer 1.599, GLE 1.599 (circular);
reference FS = 1.60 (Duncan & Wright). Crucially, Case 1 and Case 2 give the SAME
factors of safety -- for a fully submerged slope the FS is independent of the
water depth above the crest (buoyancy of the submerged soil and the ponded-water
buttress balance).

VERDICT: PASS. On the searched Bishop critical circle: Bishop 1.597, Spencer
1.598, GLE 1.595 -- within 0.4% of the published 1.603/1.599/1.599 and the
reference 1.60. The module reproduces the water-level INDEPENDENCE to ~7
significant figures: the identical circle gives Bishop 1.5974 at water el 75 ft
(Case 1), 105 ft (Case 2), AND 45 ft (water exactly at the crest) -- validating
the ponded-water buttress + submerged unit-weight handling.

NOTE: a free Spencer/entry-exit search on this ponded geometry can converge to a
spurious near-zero-Spencer degenerate surface; the robust path here is the
Bishop centre-grid search (used to locate the pinned circle), consistent with
Slide's "auto refine" circular search.
"""
import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import analyze_slope

_F, _PSF, _PCF = 0.3048, 0.04788, 0.157087
_SURFACE = [(x*_F, z*_F) for x, z in [(0, 15), (30, 15), (105, 45), (140, 45)]]
_GAMMA, _C, _PHI = 128*_PCF, 100*_PSF, 20.0
# Bishop critical circle from the centre-grid search (m), pinned.
_XC, _YC, _R = 15.060705882352943, 27.93623529411765, 24.10195456819613


def _geom(water_el_ft):
    L = SlopeSoilLayer(name="mat1", top_elevation=45*_F, bottom_elevation=0.0,
        gamma=_GAMMA, gamma_sat=_GAMMA, c_prime=_C, phi=_PHI,
        analysis_mode="drained")
    we = water_el_ft * _F
    return SlopeGeometry(surface_points=_SURFACE, soil_layers=[L],
        gwt_points=[(0.0, we), (140*_F, we)])


def test_v044_submerged_fos_matches_published():
    """PASS: circular FOS on the critical circle reproduces the published
    Bishop/Spencer/GLE (1.603/1.599/1.599; ref 1.60) within ~0.4%."""
    g = _geom(75)                          # Case 1
    fb = analyze_slope(g, xc=_XC, yc=_YC, radius=_R, method="bishop", n_slices=40).FOS
    fs = analyze_slope(g, xc=_XC, yc=_YC, radius=_R, method="spencer", n_slices=40).FOS
    fg = analyze_slope(g, xc=_XC, yc=_YC, radius=_R, method="gle", n_slices=40).FOS
    assert fb == pytest.approx(1.603, rel=0.01)
    assert fs == pytest.approx(1.599, rel=0.01)
    assert fg == pytest.approx(1.599, rel=0.01)
    assert fb == pytest.approx(1.60, abs=0.02)     # Duncan-Wright reference


def test_v044_fos_independent_of_water_level():
    """PASS: for the fully submerged slope the FOS is independent of the ponded
    water depth -- Case 1 (el 75), Case 2 (el 105) and water-at-crest (el 45)
    give an identical FOS on the same circle (to machine precision)."""
    fos = [analyze_slope(_geom(w), xc=_XC, yc=_YC, radius=_R,
                         method="bishop", n_slices=40).FOS
           for w in (75, 105, 45)]
    assert fos[0] == pytest.approx(1.5974, abs=0.005)
    assert fos[1] == pytest.approx(fos[0], abs=1e-5)   # Case 2 == Case 1
    assert fos[2] == pytest.approx(fos[0], abs=1e-5)   # water-at-crest == Case 1

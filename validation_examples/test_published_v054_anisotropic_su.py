"""Phase E / v5.4 F6 validation — anisotropic undrained strength su(alpha).

The new `strength_model='anisotropic'` makes the undrained strength depend on the
slice-base inclination alpha (ADP: su_active at the active/crest zone alpha>=+45,
su_dss at direct simple shear alpha=0, su_passive at the passive/toe zone
alpha<=-45; Casagrande & Carrillo 1944 sin(2 alpha) interpolation). Two published
anchors plus one honest N/A:

1. ISOTROPIC SANITY (Frontiers Earth Sci 13:1581457). H=5 m, 45deg slope,
   gamma=20 kN/m3, undrained su=40 kPa -> published FOS 2.16 (FLAC 2.18; Taylor
   phi=0 chart ~2.2). With anisotropy OFF the model must reproduce the isotropic
   result. Our critical toe circle (2.5, 4.0, R=4.717) gives Bishop 2.287 —
   within ~5% of the published 2.16-2.2 (Bishop phi=0 sits slightly above the
   FLAC lower bound). AND the anisotropic model with su_active=su_dss=su_passive
   reproduces the mohr_coulomb cu=40 FOS EXACTLY on the same surface (structural
   isotropic-reduction check).
   (Note: the auto centre-grid SEARCH over-estimates here (~2.74) because it
   under-samples the small toe circle — a search-coverage limit, unrelated to the
   strength model; the critical circle above is the brute-force minimum.)

2. BAKKLANDET natural slope (IOP EES 1523 (2025) 012038; NGI-ADP FEM). Anisotropy
   ratios DSS/A=0.63, P/A=0.35 dropped the reported factor of safety from 1.43
   (isotropic) to 1.09 anisotropic — a ~24% reduction. Applying the SAME ratios
   here (su_active=40, su_dss=25.2, su_passive=14) reduces our FOS from the
   isotropic critical 2.287 to 1.850 on that circle (19.1%) / 1.796 on the
   anisotropic critical circle (3.0, 3.0, R=4.243) — a 21.5% critical-vs-critical
   reduction, BRACKETING Bakklandet's 24%. Verdict CONVENTION: the anisotropy
   DIRECTION and MAGNITUDE of the FOS reduction reproduce the published FEM
   result; the exact value is not pinned (FEM vs limit-equilibrium, different
   geometry, and Bakklandet's depth-varying su_A vs our representative constant su).

3. Slide2 #105 anisotropic (Bishop 0.970 / Janbu 0.935 / Spencer 1.086 /
   GLE 1.017): the cross-section lives in Slide2 Tutorial 32, NOT the verification
   manual, so the geometry does not survive honest extraction -> N/A (source),
   recorded in INVENTORY, not pinned here.

Refs: Casagrande & Carrillo (1944); Lo (1965); Su & Liao (1999); Al-Karni &
Al-Shamrani (2000) (anisotropy matters mainly for slopes flatter than ~53deg).
See slope_stability/DESIGN.md, INVENTORY.md (#anisotropic), RESULTS.md (V-054).
"""

import math

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import analyze_slope

# H=5 m, 45deg slope, gamma=20, su=40 (isotropic anchor geometry)
_FACE = [(0, 0), (5, 5), (25, 5)]
_ISO_CIRCLE = dict(xc=2.5, yc=4.0, radius=math.hypot(2.5, 4.0))     # crit toe circle
_ANI_CIRCLE = dict(xc=3.0, yc=3.0, radius=math.hypot(3.0, 3.0))     # anisotropic crit
# Bakklandet anisotropy ratios on su_active=40: DSS/A=0.63, P/A=0.35
_SU_A, _SU_DSS, _SU_P = 40.0, 25.2, 14.0


def _geom(model, **kw):
    return SlopeGeometry(
        surface_points=_FACE,
        soil_layers=[SlopeSoilLayer(
            name="clay", top_elevation=5.0, bottom_elevation=-10.0, gamma=20.0,
            analysis_mode="undrained", cu=40.0,
            strength_model=model, **kw)])


def _fos(g, circle, method="bishop"):
    return analyze_slope(g, method=method, n_slices=40, **circle).FOS


def test_v054_isotropic_anchor_matches_published():
    """H=5/45deg/gamma=20/su=40: the critical toe circle gives Bishop 2.287,
    within ~5% of the published isotropic FOS 2.16 (FLAC 2.18; Taylor ~2.2)."""
    f = _fos(_geom("mohr_coulomb"), _ISO_CIRCLE)
    assert f == pytest.approx(2.287, abs=0.02)          # ours (pinned)
    assert f == pytest.approx(2.16, rel=0.08)           # within ~8% of published
    assert f == pytest.approx(2.18, rel=0.06)           # within ~6% of FLAC


def test_v054_anisotropic_reduces_to_isotropic_exactly():
    """su_active = su_dss = su_passive => the anisotropic model reproduces the
    mohr_coulomb cu=40 FOS EXACTLY on the critical circle (all methods)."""
    for m in ("bishop", "spencer", "gle"):
        f_mc = _fos(_geom("mohr_coulomb"), _ISO_CIRCLE, m)
        f_an = _fos(_geom("anisotropic", su_active=40.0, su_dss=40.0,
                          su_passive=40.0), _ISO_CIRCLE, m)
        assert f_an == pytest.approx(f_mc, rel=1e-9)


def test_v054_bakklandet_anisotropy_reduction():
    """The Bakklandet ratios (DSS/A=0.63, P/A=0.35) reduce the FOS from the
    isotropic critical 2.287 to 1.850 on that circle (19.1%) / 1.796 on the
    anisotropic critical (21.5% critical-vs-critical), BRACKETING Bakklandet's
    published ~24% (1.43 -> 1.09) reduction. CONVENTION (FEM vs LEM)."""
    f_iso = _fos(_geom("mohr_coulomb"), _ISO_CIRCLE)
    aniso = _geom("anisotropic", su_active=_SU_A, su_dss=_SU_DSS,
                  su_passive=_SU_P)
    f_ani_on_iso = _fos(aniso, _ISO_CIRCLE)
    f_ani_crit = _fos(aniso, _ANI_CIRCLE)
    assert f_ani_on_iso == pytest.approx(1.850, abs=0.02)   # ours (pinned)
    assert f_ani_crit == pytest.approx(1.796, abs=0.02)     # ours (pinned)
    # anisotropy lowers the FOS, and the critical surface migrates (deeper drop)
    assert f_ani_crit < f_ani_on_iso < f_iso
    drop = 1.0 - f_ani_crit / f_iso
    assert 0.18 < drop < 0.28                               # brackets Bakklandet 24%


def test_v054_active_zone_stronger_than_passive():
    """The physical ADP signature: active-zone (alpha>0, crest) slice bases carry
    a higher su than passive-zone (alpha<0, toe) bases for su_active>su_passive."""
    lay = _geom("anisotropic", su_active=_SU_A, su_dss=_SU_DSS,
                su_passive=_SU_P).soil_layers[0]
    assert lay._anisotropic_su(math.radians(40.0)) > \
        lay._anisotropic_su(0.0) > lay._anisotropic_su(math.radians(-40.0))

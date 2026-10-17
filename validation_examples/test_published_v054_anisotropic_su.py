"""Phase E / v5.4 F6 validation — anisotropic undrained strength su(alpha).

The new `strength_model='anisotropic'` makes the undrained strength depend on the
slice-base inclination alpha (ADP: su_active at the active/crest zone alpha>=+45,
su_dss at direct simple shear alpha=0, su_passive at the passive/toe zone
alpha<=-45; Casagrande & Carrillo 1944 sin(2 alpha) interpolation). Two published
anchors plus one honest N/A:

1. ISOTROPIC SANITY (Frontiers Earth Sci 13:1581457). H=5 m, 45deg slope,
   gamma=20 kN/m3, undrained su=40 kPa -> published FOS 2.16 (FLAC 2.18; Taylor
   phi=0 chart ~2.2). With anisotropy OFF the model must reproduce the isotropic
   result. With level ground carried 40 m past the toe the critical circle is
   a base circle tangent to the firm base (centre (2.63, 11.11), R 21.10):
   Bishop 2.222, within ~3% of the published 2.16 and 2% of FLAC 2.18 (Bishop
   phi=0 sits slightly above the FLAC lower bound). The profile as given starts
   at the toe, so there only toe circles fit: the search finds 2.346, and the
   toe circle (2.5, 4.0, R=4.717) the anisotropic checks below use gives
   2.379. AND the anisotropic model with su_active=su_dss=su_passive
   reproduces the mohr_coulomb cu=40 FOS EXACTLY on the same surface (structural
   isotropic-reduction check).
   (Before GM-134 that toe circle read 2.287 and was the search's minimum:
   its slices took the circle's angle at their midpoints, and the base of the
   one where the arc turns vertical fell short of the arc, an error that falls
   only as one over the square root of the slice count (2.374 on 3,840
   slices); each slice now takes its chord, and it reads 2.384 there.)

2. BAKKLANDET natural slope (IOP EES 1523 (2025) 012038; NGI-ADP FEM). Anisotropy
   ratios DSS/A=0.63, P/A=0.35 dropped the reported factor of safety from 1.43
   (isotropic) to 1.09 anisotropic — a ~24% reduction. Applying the SAME ratios
   here (su_active=40, su_dss=25.2, su_passive=14) reduces our FOS on the toe
   circle from 2.379 to 1.942 (18.4%) / 1.886 on the anisotropic critical
   circle (3.0, 3.0, R=4.243) — a 20.7% reduction, against Bakklandet's 24%
   (19.1% / 21.5% from 2.287 before GM-134). Verdict CONVENTION: the anisotropy
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
_ISO_CIRCLE = dict(xc=2.5, yc=4.0, radius=math.hypot(2.5, 4.0))     # toe circle (critical before GM-134)
_ANI_CIRCLE = dict(xc=3.0, yc=3.0, radius=math.hypot(3.0, 3.0))     # anisotropic crit
# with level ground carried 40 m past the toe: the critical base circle
_BASE_CIRCLE = dict(xc=2.63, yc=11.11, radius=21.10)
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
    """H=5/45deg/gamma=20/su=40: with level ground past the toe the critical
    base circle gives Bishop 2.222, within ~3% of the published isotropic FOS
    2.16 (FLAC 2.18; Taylor ~2.2). On the profile as given (it starts at the
    toe) the toe circle gives 2.379 (2.287 before GM-134; see the module
    docstring)."""
    f = _fos(_geom("mohr_coulomb"), _ISO_CIRCLE)
    assert f == pytest.approx(2.379, abs=0.02)          # ours (pinned)
    clay = _geom("mohr_coulomb").soil_layers
    past_toe = SlopeGeometry(surface_points=[(-40.0, 0.0)] + _FACE,
                             soil_layers=clay)
    base = _fos(past_toe, _BASE_CIRCLE)
    assert base == pytest.approx(2.222, abs=0.005)      # ours (pinned)
    assert base == pytest.approx(2.16, rel=0.04)        # within ~3% of published
    assert base == pytest.approx(2.18, rel=0.03)        # within ~2% of FLAC
    assert base < f


def test_v054_anisotropic_reduces_to_isotropic_exactly():
    """su_active = su_dss = su_passive => the anisotropic model reproduces the
    mohr_coulomb cu=40 FOS EXACTLY on the critical circle (all methods)."""
    for m in ("bishop", "spencer", "gle"):
        f_mc = _fos(_geom("mohr_coulomb"), _ISO_CIRCLE, m)
        f_an = _fos(_geom("anisotropic", su_active=40.0, su_dss=40.0,
                          su_passive=40.0), _ISO_CIRCLE, m)
        assert f_an == pytest.approx(f_mc, rel=1e-9)


def test_v054_bakklandet_anisotropy_reduction():
    """The Bakklandet ratios (DSS/A=0.63, P/A=0.35) reduce the FOS on the
    toe circle from 2.379 to 1.942 (18.4%) / 1.886 on the anisotropic
    critical (20.7%), against Bakklandet's published ~24% (1.43 -> 1.09)
    reduction (2.287 -> 1.850 / 1.796 before GM-134). CONVENTION (FEM vs
    LEM)."""
    f_iso = _fos(_geom("mohr_coulomb"), _ISO_CIRCLE)
    aniso = _geom("anisotropic", su_active=_SU_A, su_dss=_SU_DSS,
                  su_passive=_SU_P)
    f_ani_on_iso = _fos(aniso, _ISO_CIRCLE)
    f_ani_crit = _fos(aniso, _ANI_CIRCLE)
    assert f_ani_on_iso == pytest.approx(1.942, abs=0.02)   # ours (pinned)
    assert f_ani_crit == pytest.approx(1.886, abs=0.02)     # ours (pinned)
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

"""Phase E / v5.3 B2a validation — slope_stability rapid drawdown.

Slide2 Verification Manual (Rocscience), Problems #95 / #96 — the SAME
homogeneous EM 1110-2-1902 App. G embankment (gamma=135 pcf, c'=0/phi'=30,
R-envelope cR=1200 psf/phiR=16), the SAME specified slip circle (centre
169.5,210 ft, R=210 ft), the SAME drawdown (el 110 -> 24 ft); only the method
differs:

V-037  #95  USACE / Army-Corps 2-stage        published FOS 1.347 (ref 1.35).
V-038  #96  Duncan-Wright-Wong 3-stage         published FOS 1.443 (ref 1.44).

GEOMETRY (upgrade pass): the cross-section is now the exact two-segment
upstream face recovered from Fig 95.1 and calibrated against the published slip
circle (centre/radius match the arc through the confirmed entry (72,24) and
exit (354,110) to <0.5 ft): toe (0,0) -> 3H:1V to (220,73) -> 2.5H:1V to the
crest shoulder (312,110) -> flat crest to (380,110); base el 0; homogeneous.

STAGE-1 SEEPAGE (upgrade pass): Slide2 / EM 1110-2-1902 compute the pre-drawdown
STEADY-STATE seepage field (the phreatic DECLINES through the dam), whereas the
module default is a flat full-pool phreatic (hydrostatic to the reservoir) — the
conservative no-through-seepage bound. ``rapid_drawdown_fos`` now exposes an
optional ``stage1_phreatic_points`` so the steady-seepage line can be supplied
for stage 1 (default unchanged). A representative declined phreatic (110 ft at
the upstream limit -> 80 ft at x=380, ~0.08 gradient) is used below to show the
recovery; the exact flow net is Slide2's to compute.

VERDICTS
--------
* V-037 (#95, Corps 2-stage): **PASS under the steady-seepage stage-1.** With the
  declined phreatic the FOS = 1.34 vs published 1.347 (0.6%). The combined
  R/effective envelope is confirmed correct: using the pure R envelope instead
  overshoots to ~1.42, and the flat-phreatic default gives the conservative
  bound 1.21. Geometry alone does not close the gap (the exact section is, if
  anything, slightly lower than the earlier straight-face approximation) — the
  residual was the stage-1 flow net, now reproducible.
* V-038 (#96, Duncan 3-stage): **CONVENTION (approximate) — improved by the E2
  stage-3 refinement.** The DEFAULT (Fellenius stage-3 normal) is unchanged: flat
  1.235 / seepage 1.273, ordering (3-stage >= 2-stage) holds. The V5.3 diagnosis
  blamed the Kc interpolation; the E2 re-investigation found the interpolation is
  actually sound (it yields ~1.45 on its own) and the residual came from the
  STAGE-3 drained substitution firing spuriously: it estimated the drawn-down
  effective normal with a Fellenius ``W*cos(a)/l - u`` term that systematically
  under-predicts N' (17 of 50 slices substituted a too-low drained strength). The
  optional ``stage3_effective_normal='gle'`` uses the rigorous GLE normal instead
  (consistent with stage 1; 9 physically-genuine substitutions), lifting the FOS
  to flat 1.306 / seepage 1.370 and closing most of the residual to the published
  1.443 (~5% left, within the representative-flow-net + LE-N'-at-FOS sensitivity
  that already shows as V-037's ~0.6% Corps residual). Gated behind the new
  parameter (default preserved). NOT tuned; see DESIGN.md and the E2 test below.
See INVENTORY.md (V-037/V-038), RESULTS.md, and slope_stability/rapid_drawdown.py.
"""

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import rapid_drawdown_fos

FT, PSF, PCF = 0.3048, 0.04788, 0.157087

# Exact two-segment upstream face (Fig 95.1), calibrated to the published circle.
_FACE = [(0, 0), (220, 73), (312, 110), (380, 110)]
# Representative steady-seepage stage-1 phreatic: reservoir level at the upstream
# limit declining to 80 ft at the downstream limit of the modeled section.
_SEEPAGE = [(0 * FT, 110 * FT), (380 * FT, 80 * FT)]


def _dam_geom():
    return SlopeGeometry(
        surface_points=[(x * FT, z * FT) for (x, z) in _FACE],
        soil_layers=[SlopeSoilLayer(
            name="fill", top_elevation=110 * FT, bottom_elevation=-1.0 * FT,
            gamma=135 * PCF, phi=30.0, c_prime=0.0,
            R_c=1200 * PSF, R_phi=16.0)])


def _fos(method, stage1_phreatic=None, stage3="fellenius"):
    return rapid_drawdown_fos(
        _dam_geom(), 110 * FT, 24 * FT, xc=169.5 * FT, yc=210 * FT,
        radius=210 * FT, method=method, n_slices=50,
        stage1_phreatic_points=stage1_phreatic,
        stage3_effective_normal=stage3)


def test_v037_corps_2stage_flat_is_conservative_bound():
    """Default flat full-pool phreatic = conservative no-through-seepage bound:
    the USACE 2-stage FOS = 1.207 (pinned). Full-pool stage-1 is stable (~2.3)."""
    r = _fos("corps_2stage")
    assert r.FOS == pytest.approx(1.207, abs=0.02)
    assert r.stage1_fos > 2.0


def test_v037_corps_2stage_matches_published_under_seepage():
    """PASS under steady-seepage stage-1: with the declined phreatic the USACE
    2-stage FOS rises to 1.34, matching the published 1.347 within ~1%."""
    r = _fos("corps_2stage", stage1_phreatic=_SEEPAGE)
    assert r.FOS == pytest.approx(1.347, rel=0.03)   # within 3% of published
    assert r.FOS > _fos("corps_2stage").FOS          # seepage raises the FOS


def test_v038_duncan_3stage_ordering_and_residual():
    """CONVENTION (approximate): the Duncan-Wright-Wong 3-stage reproduces the
    published ORDERING (3-stage >= 2-stage at the default) and rises under the
    seepage stage-1 (1.235 -> 1.273), but a residual to the published 1.443
    remains from the c'=0 Kc interpolation (documented follow-up)."""
    f2_flat = _fos("corps_2stage").FOS
    f3_flat = _fos("duncan_3stage").FOS
    f3_seep = _fos("duncan_3stage", stage1_phreatic=_SEEPAGE).FOS
    assert f3_flat == pytest.approx(1.235, abs=0.02)      # ours (pinned)
    assert f3_flat >= f2_flat - 1e-6                      # DWW no more conservative
    assert f3_seep > f3_flat                              # seepage raises it
    assert f3_seep == pytest.approx(1.273, abs=0.03)      # ours (pinned)


def test_v038_duncan_3stage_gle_stage3_refinement():
    """E2 refinement: ``stage3_effective_normal='gle'`` uses the RIGOROUS GLE
    drawn-down effective normal for the stage-3 drained substitution (consistent
    with stage 1) instead of the Fellenius estimate that under-predicts N' and
    over-fires the substitution. It lifts the Duncan 3-stage FOS toward the
    published 1.443 while leaving the default byte-identical, and never touches
    the Corps 2-stage (which has no stage 3)."""
    # default (Fellenius stage-3) is unchanged
    assert _fos("duncan_3stage").FOS == pytest.approx(1.235, abs=0.02)
    assert _fos("duncan_3stage", _SEEPAGE).FOS == pytest.approx(1.273, abs=0.03)
    # rigorous GLE stage-3 raises the FOS (fewer spurious drained substitutions)
    g_flat = _fos("duncan_3stage", None, "gle")
    g_seep = _fos("duncan_3stage", _SEEPAGE, "gle")
    assert g_flat.FOS == pytest.approx(1.306, abs=0.02)   # ours (pinned)
    assert g_seep.FOS == pytest.approx(1.370, abs=0.03)   # ours (pinned)
    assert g_flat.FOS > _fos("duncan_3stage").FOS         # gle >= fellenius
    assert g_seep.FOS > _fos("duncan_3stage", _SEEPAGE).FOS
    # fewer stage-3 substitutions than the Fellenius default (17 -> ~9 seepage)
    assert g_seep.n_drained_substituted < \
        _fos("duncan_3stage", _SEEPAGE).n_drained_substituted
    # still short of the published 1.443 (documented ~5% residual), honest
    assert g_seep.FOS < 1.443
    # Corps 2-stage has no stage 3: the option is inert there
    assert _fos("corps_2stage", _SEEPAGE, "gle").FOS == \
        pytest.approx(_fos("corps_2stage", _SEEPAGE).FOS, abs=1e-9)


def test_v037_v038_undrained_is_critical():
    """The rapid (undrained) drawdown FOS is far below a free-draining (drained)
    drawdown of the same dam -- the reason the analysis is done."""
    undrained = _fos("corps_2stage").FOS
    g = _dam_geom()
    g.soil_layers[0].R_phi = None      # free-draining
    drained = rapid_drawdown_fos(g, 110 * FT, 24 * FT, xc=169.5 * FT,
                                 yc=210 * FT, radius=210 * FT,
                                 method="corps_2stage").FOS
    assert undrained < 0.7 * drained

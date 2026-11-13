"""Validation: slope_stability rapid drawdown on a given circle.

Slide2 Verification Manual (Rocscience), Problems #95 / #96: the homogeneous
embankment of EM 1110-2-1902 Appendix G's worked example (gamma = 135 pcf,
c' = 0, phi' = 30 deg, R envelope c_R = 1,200 psf and phi_R = 16 deg), the
same given slip circle (centre 169.5, 210 ft; R = 210 ft), drawdown to El.
24 ft; only the procedure differs:

V-037  #95  Corps of Engineers 1970 two-stage     printed 1.347 (EM: 1.35).
V-038  #96  improved three-stage (Duncan, Wright  printed 1.443 (EM: 1.44).
            and Wong)

The section is the two-segment upstream face recovered from Fig 95.1: toe
(0,0), 3H:1V to (220,73), 2.5H:1V to the crest shoulder (312,110), level
crest to (380,110). The Slide2 problem text gives the maximum pool as 110 ft;
the EM's example has 103 ft.

WATER. The EM's own example assumes a level piezometric line at the maximum
pool and another at the lowered pool (G-4b, p. G-9), which is the module's
default, named in every result. (Until GM-140 step 9 this file pinned values
on a "representative" seepage line that no page prints, and drew conclusions
from them; those pins and conclusions were removed.)

VERDICTS (GM-140 step 9: every stage as Appendix G prints it)
--------
* V-038 (#96, three-stage): 1.427 at the 110 ft pool against the printed
  1.443 (-1.1 %); 1.431 at the EM's 103 ft pool against its 1.44. The final
  value cannot tell the two pools apart. Stage by stage on the EM's own
  section and slices: `slope_stability/tests/
  test_rapid_drawdown_em_appendix_g.py` (2.20 / 1.52 / 1.44 printed).
* V-037 (#95, Corps 1970): the EM computed its example of this procedure by
  the Modified Swedish Method with the side forces at a fixed 19.4 deg (G-5,
  pp. G-9 to G-15: 3.49, then 1.35). The library's force-equilibrium solution
  at that inclination gives 1.338 at the EM's pool (-0.7 % on 1.347) and
  1.304 at 110 ft (-3.2 %). The module's own answer is by Spencer's method,
  which the manual recommends: 1.207 (1.236 at 103 ft), about 10 % below the
  printed value; the result says what the manual's example used. Which
  method of slices Slide2 used for #95 and #96 is not settled.
See slope_stability/VALIDATION.md (V-037/V-038) and module_work/gm140/
STEP9_CHECK.md.
"""

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import rapid_drawdown_fos

FT, PSF, PCF = 0.3048, 0.04788, 0.157087

# The two-segment upstream face (Fig 95.1), calibrated to the printed circle.
_FACE = [(0, 0), (220, 73), (312, 110), (380, 110)]


def _dam_geom():
    return SlopeGeometry(
        surface_points=[(x * FT, z * FT) for (x, z) in _FACE],
        soil_layers=[SlopeSoilLayer(
            name="fill", top_elevation=110 * FT, bottom_elevation=-1.0 * FT,
            gamma=135 * PCF, phi=30.0, c_prime=0.0,
            R_c=1200 * PSF, R_phi=16.0)])


def _fos(method, pool_ft=110, **kw):
    return rapid_drawdown_fos(
        _dam_geom(), pool_ft * FT, 24 * FT, xc=169.5 * FT, yc=210 * FT,
        radius=210 * FT, method=method, n_slices=50, **kw)


def test_v037_corps_1970_by_the_manuals_own_method_of_slices():
    """#95 printed 1.347 (EM 1110-2-1902 p. G-15: 1.35, by the Modified
    Swedish Method with side forces at 19.4 deg). Force equilibrium at that
    fixed inclination, at the EM's 103 ft pool: 1.338, -0.7 %."""
    r = _fos("corps_2stage", pool_ft=103, stage_solver=("force", 19.4))
    assert r.FOS == pytest.approx(1.347, rel=0.01)
    assert r.stage1_fos == pytest.approx(3.49, rel=0.01)     # EM p. G-12
    assert r.answer_stage == 2 and r.stage3_fos is None


def test_v037_corps_1970_by_spencer_is_the_modules_answer():
    """The module's Corps 1970 procedure is solved by Spencer's method (the
    lead's call, GM-140 step 9); the first stage at full pool is stable and
    the result says what the manual's example used."""
    r = _fos("corps_2stage")
    # Unchanged by GM-140 step 9 (1.207). Printed 1.347 by another method
    # of slices: -10.4 %; see the test above for the printed value.
    assert r.FOS == pytest.approx(1.207, abs=0.02)
    assert r.stage1_fos > 2.0
    assert r.FOS == r.stage2_fos
    assert any("Modified Swedish" in n for n in r.notes)


def test_v038_three_stage_against_print():
    """#96 printed 1.443 (EM p. G-20: 1.44). Every stage as Appendix G
    prints it, Spencer's method, level pools."""
    r = _fos("duncan_3stage")
    # GM-140 step 9: 1.235 -> 1.427; printed 1.443 (Slide2 #96), -1.1 %
    assert r.FOS == pytest.approx(1.443, rel=0.02)
    assert r.answer_stage == 3 and r.FOS == r.stage3_fos
    assert r.stage2_fos > r.stage3_fos            # the drained check lowers it
    em_pool = _fos("duncan_3stage", pool_ft=103)
    # GM-140 step 9: 1.237 -> 1.431; printed 1.44 (EM p. G-20), -0.6 %
    assert em_pool.FOS == pytest.approx(1.44, rel=0.01)
    assert em_pool.stage1_fos == pytest.approx(2.20, rel=0.01)   # p. G-15
    assert em_pool.stage2_fos == pytest.approx(1.52, rel=0.01)   # p. G-18


def test_v038_three_stage_is_not_below_the_corps_procedure():
    """The printed ordering: the three-stage value (1.443) above the 1970
    procedure's, with the same method of slices for both."""
    assert _fos("duncan_3stage").FOS >= _fos("corps_2stage").FOS


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

"""V-028: Slide2 Verification Manual problem #9 = ACADS 4, the weak seam
(manual pp. 50-54; printed Spencer 0.760, GLE 0.720, Janbu corrected
0.734, Table 9.4, p. 52; referee 0.78).

The comparison with print is in test_published_v028_e8.py: with the
problem's loads (Table 9.2) on, the lowest of a declared family of
surfaces along the seam reads Spencer 0.753, Morgenstern-Price 0.722 and
Janbu corrected 0.728, each within 1 % of its printed value. This file
keeps three things without the loads:

1. The weights (GM-140 step 1, one material per point): every soil here
   weighs 18.84 kN/m3, so the mass on the kept surface weighs 18.84 times
   its area, 3,003 kN/m. All three layers are entered with a top at el.
   40, and before the step each was weighed from that flat top (8,768
   kN/m).
2. The kept surface, as a record of what moved when each point is weighed
   once: Janbu's simplified method (corrected) read 0.819 on it before
   and reads 1.073 now. It is the minimum of a weak-layer search run on
   the tripled weights (seed 3, 30 slices), NOT a critical surface, and
   nothing printed is compared with it. Spencer and Morgenstern-Price do
   not converge on it.
3. The SS-6 regression: the weak-layer search returns a surface that
   solves at the slice counts that used to break it. Its value is not
   pinned: see ``test_v028_weak_layer_search_against_print`` in
   test_published_v028_e8.py (a strict xfail).

What the problem text gives (module_work/slope_v53/
slide2_selected_problems.txt, manual pp. 50-54): the two soils (Table 9.1),
the loads (Table 9.2) and the piezometric line (Table 9.3). The ground
surface, the seam and the slip surface are in Figure 9.1 only, which is
not in the repository, so the geometry below (read off the figure in
2026-06) cannot be checked. Table 9.2 puts the bench load at y = 27.75,
and the piezometric line is at 27.75 along the bench, where the ground
here is at 28.0: reported, not changed. The piezometric line is entered as
Table 9.3 prints it (29.8, 34.0 and 37.6 at x = 49, 60 and 74).

See module_work/gm140/STEP1_CHECK.md, validation_examples/INVENTORY.md
(V-028) and slope_stability/VALIDATION.md B11.
"""

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import PolylineSlipSurface
from slope_stability.slices import build_slices
from slope_stability.gle import gle_fos, janbu_fos
from slope_stability.analysis import search_critical_surface


# ════════════════════════════════════════════════════════════════════════════
# V-028 — Slide2 #9 (ACADS 4): weak seam + piezometric surface, noncircular
# ════════════════════════════════════════════════════════════════════════════
# Surface (20,28),(43,28),(68,40),(84,40); base z=15; a 1 m weak seam (c=0,phi=10)
# from (20,18.5)->(84,36.5) mid; Soil1 c=28.5/phi=20 above & below; piezometric
# surface per Table 9.3. Published (noncircular): no-optimization Spencer 0.760 /
# GLE 0.720 / Janbu-c 0.734; block-search-with-optimization Spencer 0.707 / GLE
# 0.683 / Janbu 0.699; Slope-2000 GLE 0.6878; referee 0.78 [Giam].

_V028_SURF = [(20.0, 28.0), (43.0, 28.0), (68.0, 40.0), (84.0, 40.0)]
_V028_SEAM_TOP = [(20.0, 19.0), (84.0, 37.0)]
_V028_SEAM_BOT = [(20.0, 18.0), (84.0, 36.0)]
# Table 9.3 as printed
_V028_PIEZO = [(20.0, 27.75), (43.0, 27.75), (49.0, 29.8), (60.0, 34.0),
               (66.0, 35.8), (74.0, 37.6), (80.0, 38.4), (84.0, 38.4)]


def _v028_geom():
    return SlopeGeometry(
        surface_points=_V028_SURF,
        soil_layers=[
            SlopeSoilLayer(name="upper", top_elevation=40.0, bottom_elevation=15.0,
                           gamma=18.84, phi=20.0, c_prime=28.5,
                           bottom_boundary_points=_V028_SEAM_TOP),
            SlopeSoilLayer(name="weak", top_elevation=40.0, bottom_elevation=15.0,
                           gamma=18.84, phi=10.0, c_prime=0.0,
                           bottom_boundary_points=_V028_SEAM_BOT),
            SlopeSoilLayer(name="lower", top_elevation=40.0, bottom_elevation=15.0,
                           gamma=18.84, phi=20.0, c_prime=28.5),
        ],
        gwt_points=_V028_PIEZO)


# The surface the weak-layer search returned while the soil was weighed
# three times (seed=3, 30 slices). A record only: not a critical surface.
_V028_SURFACE = [(41.987, 28.0), (49.095, 26.656), (56.203, 28.249),
                 (63.312, 30.304), (70.42, 35.555), (77.528, 40.0)]


def _v028_slices(n=50):
    slip = PolylineSlipSurface(points=_V028_SURFACE)
    return build_slices(_v028_geom(), slip, n), slip


def test_v028_slices_weigh_their_area():
    """Every soil here weighs 18.84 kN/m3, so each slice weighs 18.84 h b
    and the mass 3,003 kN/m (8,768 before GM-140 step 1)."""
    sl, _slip = _v028_slices()
    for s in sl:
        assert s.weight == pytest.approx(18.84 * s.height * s.width, rel=1e-9)
    assert sum(s.weight for s in sl) == pytest.approx(3003.0, abs=1.0)


def test_v028_kept_surface_record():
    """RECORD of what moved when each point is weighed once, not a
    validation: on the kept surface Janbu's simplified method (corrected)
    reads 1.073 (0.819 while the soil was weighed three times). The
    surface is the minimum of a search run on the tripled weights and not
    a critical surface. Spencer and Morgenstern-Price (half-sine) do not
    converge on it, and no value is pinned from a solution that did not
    converge."""
    sl, slip = _v028_slices()
    fj, _uncorrected, _f0 = janbu_fos(sl, slip)
    assert fj == pytest.approx(1.073, abs=0.005)
    assert not gle_fos(sl, slip, f_interslice="constant").converged
    assert not gle_fos(sl, slip, f_interslice="half_sine").converged


def test_v028_weak_layer_search_is_not_degenerate():
    """REGRESSION for the SS-6 fix: the weak-layer noncircular search
    returns a surface with a solution, not the pre-fix degenerate
    ~0.05-0.18, at the slice counts that used to break it. The value it
    returns is not pinned here: it does not reach print (see
    ``test_v028_weak_layer_search_against_print`` in
    test_published_v028_e8.py)."""
    geom = _v028_geom()
    for n_slices in (30, 40):
        res = search_critical_surface(
            geom, surface_type="weak_layer", method="spencer",
            n_trials=700, n_points=6, n_slices=n_slices, seed=3,
            x_entry_range=(20.0, 50.0), x_exit_range=(58.0, 84.0))
        assert res.critical is not None
        assert res.critical.FOS > 0.3
        assert [o["layer"] for o in res.layer_overlaps] == ["weak", "lower"]

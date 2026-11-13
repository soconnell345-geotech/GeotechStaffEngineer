"""Phase E / v5.3 B2 round 1 validation — slope_stability.

V-028  Slide2 #9 = ACADS 4 weak-seam noncircular (referee 0.78).

NOT VALIDATING since GM-140 step 1 (one material per point, 2026-11-13).
The 0.808 pinned here before (+3.6 % on the referee) rested on the soil
being weighed three times: all three layers are entered with a top at el.
40, and each was weighed from that flat top (the sliding mass 8,768 kN/m
against 3,003 kN/m from its area). With each point weighed once the same
surface reads Janbu (corrected) 1.073; Spencer and GLE have no solution on
it (the 1.041 and 1.010 they return are the old approximate solver's
fallback, the account's D23/D30, step 2 of the rebuild); and the
weak-layer search returns 1.43, a solved Spencer value: 38 to 83 % above
the referee. The values below are a record of what
the module computes on these inputs, not a validation; the comparison with
the referee is a strict xfail.

What the problem text gives (module_work/slope_v53/
slide2_selected_problems.txt, manual pp. 50-54): the two soils (Table 9.1),
the loads (Table 9.2) and the piezometric line (Table 9.3). The ground
surface, the seam and the slip surface are in Figure 9.1 only, which is
not in the repository, so the geometry below (read off the figure in
2026-06) cannot be checked. Table 9.2 puts the bench load at y = 27.75,
and the piezometric line is at 27.75 along the bench, where the ground
here is at 28.0: reported, not changed (with the bench at 27.75 the
surface reads Janbu 1.047). The piezometric line is entered as Table
9.3 prints it (29.8, 34.0 and 37.6 at x = 49, 60 and 74; 29.86, 34.06 and
37.68 before, a difference of 0.3 % in F).

See module_work/gm140/STEP1_CHECK.md, validation_examples/INVENTORY.md
(V-028) and slope_stability/VALIDATION.md B11.
"""

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import PolylineSlipSurface
from slope_stability.slices import build_slices
from slope_stability.methods import spencer_fos
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
# three times (seed=3, 30 slices); kept as the surface the record is on.
_V028_SURFACE = [(41.987, 28.0), (49.095, 26.656), (56.203, 28.249),
                 (63.312, 30.304), (70.42, 35.555), (77.528, 40.0)]


_NOT_VALIDATING = (
    "V-028 does not validate (GM-140 step 1): with each point weighed once "
    "the module reads 1.07 (Janbu) on the seam surface and 1.43 from the "
    "weak-layer search against the referee's 0.78. The geometry was read "
    "off Figure 9.1, which is not in the repository, and cannot be checked "
    "against the problem text; no input was adjusted.")


def _v028_methods(n=50):
    geom = _v028_geom()
    slip = PolylineSlipSurface(points=_V028_SURFACE)
    sl = build_slices(geom, slip, n)
    fs, _ = spencer_fos(sl, slip)
    fg = gle_fos(sl, slip, f_interslice="half_sine").fos
    fj, _u, _f0 = janbu_fos(sl, slip)
    return sl, fs, fg, fj


def test_v028_slices_weigh_their_area():
    """Every soil here weighs 18.84 kN/m3, so each slice weighs 18.84 h b
    and the mass 3,003 kN/m (8,768 before GM-140 step 1)."""
    sl, _fs, _fg, _fj = _v028_methods()
    for s in sl:
        assert s.weight == pytest.approx(18.84 * s.height * s.width, rel=1e-9)
    assert sum(s.weight for s in sl) == pytest.approx(3003.0, abs=1.0)


def test_v028_weak_seam_surface_record():
    """RECORD, not a validation: on the seam-following surface Janbu's
    simplified method (corrected) reads 1.073. Spencer and GLE (half-sine)
    do not converge on it; what they return (1.041, 1.010) is the old
    approximate solver's fallback, to go in step 2 of the rebuild, and is
    recorded here only so a change shows."""
    sl, fs, fg, fj = _v028_methods()
    assert fj == pytest.approx(1.073, abs=0.005)
    slip = PolylineSlipSurface(points=_V028_SURFACE)
    assert not gle_fos(sl, slip, f_interslice="half_sine").converged
    assert fs == pytest.approx(1.041, abs=0.005)
    assert fg == pytest.approx(1.010, abs=0.005)


@pytest.mark.xfail(strict=True, reason=_NOT_VALIDATING)
def test_v028_weak_seam_surface_matches_referee():
    _sl, _fs, _fg, fj = _v028_methods()
    assert fj == pytest.approx(0.78, rel=0.05)           # referee 0.78


def test_v028_weak_layer_search_is_not_degenerate():
    """REGRESSION for the SS-6 fix: the weak-layer noncircular search returns
    a surface that solves, not the pre-fix degenerate ~0.05-0.18, at the
    slice counts that used to break it. It reads 1.43 (not validating; see
    the module docstring)."""
    geom = _v028_geom()
    for n_slices in (30, 40):
        res = search_critical_surface(
            geom, surface_type="weak_layer", method="spencer",
            n_trials=700, n_points=6, n_slices=n_slices, seed=3,
            x_entry_range=(20.0, 50.0), x_exit_range=(58.0, 84.0))
        assert res.critical is not None
        assert res.critical.FOS > 0.3
        assert res.critical.FOS == pytest.approx(1.43, abs=0.03)
        assert [o["layer"] for o in res.layer_overlaps] == ["weak", "lower"]

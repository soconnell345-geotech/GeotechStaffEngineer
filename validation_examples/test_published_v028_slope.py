"""Phase E / v5.3 B2 round 1 validation — slope_stability.

V-028  Slide2 #9 = ACADS 4 weak-seam noncircular (referee 0.78) — RE-VALIDATED
       after the weak-layer search degenerate-surface fix (SS-6). Previously the
       search returned spurious ~0.05-0.18 surfaces; it now finds a sane
       seam-following surface at Spencer ~0.79 (+1.5% vs the referee 0.78).

See validation_examples/INVENTORY.md (V-028) and RESULTS.md.
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
_V028_PIEZO = [(20.0, 27.75), (43.0, 27.75), (49.0, 29.86), (60.0, 34.06),
               (66.0, 35.80), (74.0, 37.68), (80.0, 38.4), (84.0, 38.4)]


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


# The weak-layer search critical surface (deterministic, seed=3, 30 slices).
_V028_SURFACE = [(41.987, 28.0), (49.095, 26.656), (56.203, 28.249),
                 (63.312, 30.304), (70.42, 35.555), (77.528, 40.0)]


def test_v028_weak_seam_pinned_surface_matches_referee():
    """PASS: on the weak-layer search's seam-following critical surface, the
    module's Spencer FOS = 0.792 -- within +1.5% of the referee 0.78 (and above
    Slide2's no-optimization 0.760 / optimized 0.707 band). GLE 0.786, Janbu
    0.804. The surface follows the weak seam (c=0, phi=10) with the inclined
    piezometric surface active."""
    geom = _v028_geom()
    slip = PolylineSlipSurface(points=_V028_SURFACE)
    sl = build_slices(geom, slip, 50)
    fs, _ = spencer_fos(sl, slip)
    fg = gle_fos(sl, slip, f_interslice="half_sine").fos
    fj, _u, _f0 = janbu_fos(sl, slip)
    assert fs == pytest.approx(0.792, abs=0.02)          # ours
    assert fs == pytest.approx(0.78, rel=0.05)           # referee 0.78
    assert fg == pytest.approx(0.786, abs=0.03)
    assert fj == pytest.approx(0.804, abs=0.03)


def test_v028_weak_layer_search_is_sane_after_fix():
    """REGRESSION for the SS-6 fix: the weak-layer noncircular search now returns
    a sane seam-following critical (~0.79), NOT the pre-fix degenerate spurious
    ~0.05-0.18 surface. Robust across the slice count that used to break it."""
    geom = _v028_geom()
    for n_slices in (30, 40):
        res = search_critical_surface(
            geom, surface_type="weak_layer", method="spencer",
            n_trials=700, n_points=6, n_slices=n_slices, seed=3,
            x_entry_range=(20.0, 50.0), x_exit_range=(58.0, 84.0))
        assert res.critical is not None
        # sane physical range near the referee, never the degenerate < 0.3
        assert 0.70 < res.critical.FOS < 0.90

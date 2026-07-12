"""Phase E / v5.4 F4 validation — additional rapid-drawdown METHODS.

The rapid-drawdown module (validated for the Corps 2-stage and Duncan-Wright-Wong
3-stage methods in V-037/V-038/V-041) gains the third classical procedure that
Slide2 / the rapid-drawdown literature tabulate:

  Lowe & Karafiath (1960), 2-stage — the SAME Kc-interpolated stage-2 undrained
  strength as Duncan-Wright-Wong, but WITHOUT the third (drained-substitution)
  stage. Duncan, Wright & Wong (1990) state their method IS Lowe & Karafiath's
  with the drained-strength check appended; because stage 3 only ever substitutes
  a *lower* strength, ``lowe_karafiath`` FOS >= ``duncan_3stage`` FOS always.

Two published anchors, both from the SAME Slide2 verification problems already in
the suite (this adds the extra published METHOD columns, no new geometry):

#95/#96 homogeneous EM 1110-2-1902 dam (EXACT geometry, cf. V-037/V-038) — the
Lowe-Karafiath stage-2 is the Duncan-Wright-Wong stage-2, so on this dam the
Lowe-Karafiath seepage FOS (1.450) lands right on the PUBLISHED Duncan-Wright-Wong
1.443. That is a strong cross-check that the Kc (anisotropic-consolidation) stage-2
is correct and that the default 3-stage's residual to 1.443 is entirely the
Fellenius stage-3 effective normal (the E2 finding), NOT the interpolation.

#98 Walter Bouldin (APPROXIMATE recovered geometry, cf. V-041) — the published
SEARCH minima are Corps 0.931, Lowe-Karafiath 1.075, Duncan-Wright-Wong 1.039
(Duncan, Wright & Wong 1990). On the recovered section the search minima are
Corps 0.837, Lowe-Karafiath 0.964, DWW 0.938 — all ~10% BELOW published (the same
coherent geometry offset documented in V-041, NOT a method defect), with the
correct published method ORDERING Lowe-Karafiath > DWW > Corps reproduced exactly.

VERDICT: **Lowe-Karafiath — PASS (mechanics/ordering) / CONVENTION (geometry).**
The stage logic is exact (it reproduces the published DWW value through stage 2 on
the exact #95/#96 dam) and the published #98 method ordering is reproduced; the #98
magnitudes inherit V-041's ~10% geometry deficit. NOT tuned.

See slope_stability/rapid_drawdown.py, DESIGN.md, INVENTORY.md (#95/#96/#98),
and V-037/V-038/V-041.
"""

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import rapid_drawdown_fos
from slope_stability.rapid_drawdown import search_rapid_drawdown

FT, PSF, PCF = 0.3048, 0.04788, 0.157087

# ── #95/#96 homogeneous dam (EXACT geometry; cf. V-037/V-038) ────────────────
_FACE = [(0, 0), (220, 73), (312, 110), (380, 110)]
_SEEPAGE = [(0 * FT, 110 * FT), (380 * FT, 80 * FT)]


def _dam_9596():
    return SlopeGeometry(
        surface_points=[(x * FT, z * FT) for x, z in _FACE],
        soil_layers=[SlopeSoilLayer(
            name="fill", top_elevation=110 * FT, bottom_elevation=-1.0 * FT,
            gamma=135 * PCF, phi=30.0, c_prime=0.0,
            R_c=1200 * PSF, R_phi=16.0)])


def _fos(method, seepage=None):
    return rapid_drawdown_fos(
        _dam_9596(), 110 * FT, 24 * FT, xc=169.5 * FT, yc=210 * FT,
        radius=210 * FT, method=method, n_slices=50,
        stage1_phreatic_points=seepage)


# ── #98 Walter Bouldin (APPROXIMATE recovered geometry; cf. V-041) ──────────
_SURF_98 = [(0, 0), (100, 40), (140, 60), (180, 60)]


def _dam_98():
    return SlopeGeometry(
        surface_points=[(x * FT, z * FT) for x, z in _SURF_98],
        soil_layers=[
            SlopeSoilLayer(name="ClayeySiltySand", top_elevation=60 * FT,
                           bottom_elevation=51 * FT, gamma=128 * PCF,
                           c_prime=240 * PSF, phi=32.7, R_c=650 * PSF, R_phi=13.0),
            SlopeSoilLayer(name="MicaceousSand", top_elevation=51 * FT,
                           bottom_elevation=30 * FT, gamma=123 * PCF,
                           c_prime=220 * PSF, phi=22.5, R_c=450 * PSF, R_phi=11.0),
            SlopeSoilLayer(name="CretaceousClay", top_elevation=30 * FT,
                           bottom_elevation=17 * FT, gamma=124 * PCF,
                           c_prime=180 * PSF, phi=19.0, R_c=180 * PSF, R_phi=13.0),
            SlopeSoilLayer(name="ClayeySandyGravel", top_elevation=17 * FT,
                           bottom_elevation=-2 * FT, gamma=125 * PCF,
                           c_prime=0.0, phi=40.0, R_phi=None),   # free-draining
        ])


def _search_98(method):
    return search_rapid_drawdown(
        _dam_98(), 47 * FT, 15 * FT, method=method, surface_type="circular",
        nx=6, ny=5, x_range=(30 * FT, 140 * FT), y_range=(60 * FT, 150 * FT),
        n_slices=20)


# ── Lowe-Karafiath stage logic on the EXACT #95/#96 dam ─────────────────────

def test_v048_lowe_karafiath_is_dww_without_stage3():
    """Lowe-Karafiath uses the Duncan-Wright-Wong Kc-interpolated stage-2 strength
    but omits stage 3, so it is NEVER more conservative than the 3-stage (stage 3
    only substitutes a lower strength). On the exact #95/#96 dam LK = 1.357 flat /
    1.450 seepage (pinned), each >= the corresponding duncan_3stage FOS."""
    lk_flat = _fos("lowe_karafiath")
    lk_seep = _fos("lowe_karafiath", _SEEPAGE)
    assert lk_flat.FOS == pytest.approx(1.357, abs=0.02)      # ours (pinned)
    assert lk_seep.FOS == pytest.approx(1.450, abs=0.02)      # ours (pinned)
    # Lowe-Karafiath (no stage 3) >= Duncan-Wright-Wong (with stage 3)
    assert lk_flat.FOS >= _fos("duncan_3stage").FOS - 1e-6
    assert lk_seep.FOS >= _fos("duncan_3stage", _SEEPAGE).FOS - 1e-6
    # It runs no stage-3 substitutions (2-stage method)
    assert lk_flat.n_drained_substituted == 0


def test_v048_lowe_karafiath_stage2_reproduces_published_dww():
    """The Lowe-Karafiath seepage FOS (1.450) lands on the PUBLISHED Duncan-Wright-
    Wong 1.443 for this dam. Since LK's stage 2 IS the DWW stage 2, this confirms
    the Kc (anisotropic-consolidation) interpolation is correct and the default
    3-stage's residual to 1.443 is the Fellenius stage-3 normal (E2), not stage 2."""
    lk_seep = _fos("lowe_karafiath", _SEEPAGE).FOS
    assert lk_seep == pytest.approx(1.443, rel=0.02)          # published DWW #96


# ── #98 published search minima: the LOWE-KARAFIATH column + ordering ────────

def test_v048_98_lowe_karafiath_search_minimum():
    """#98 Lowe-Karafiath search minimum = 0.964 vs published 1.075 (~10% low);
    the residual is V-041's approximate recovered geometry (flat layers, no riprap
    veneer), the SAME ~10% offset seen for Corps and DWW there, not the method."""
    r = _search_98("lowe_karafiath")
    assert r.FOS == pytest.approx(0.964, abs=0.03)            # ours (pinned)
    assert r.FOS == pytest.approx(1.075, rel=0.15)            # within 15% of published
    assert r.critical is not None and r.critical.is_circular


def test_v048_98_all_published_method_columns():
    """All three published #98 search-minimum columns reproduced on the recovered
    section, each ~10% low (the shared V-041 geometry offset) but in the correct
    published ORDER Lowe-Karafiath (1.075) > Duncan-Wright-Wong (1.039) > Corps
    2-stage (0.931). Ours: LK 0.964 > DWW 0.938 > Corps 0.837."""
    fos = {m: _search_98(m).FOS for m in
           ("corps_2stage", "duncan_3stage", "lowe_karafiath")}
    # ordering (method effect, geometry-independent)
    assert fos["lowe_karafiath"] > fos["duncan_3stage"] > fos["corps_2stage"]
    # each column vs its published value (within 15%; ~10% geometry-limited, V-041)
    assert fos["corps_2stage"] == pytest.approx(0.931, rel=0.15)
    assert fos["duncan_3stage"] == pytest.approx(1.039, rel=0.15)
    assert fos["lowe_karafiath"] == pytest.approx(1.075, rel=0.15)
    # our pinned magnitudes (guard against silent drift)
    assert fos["corps_2stage"] == pytest.approx(0.837, abs=0.03)
    assert fos["duncan_3stage"] == pytest.approx(0.938, abs=0.03)
    assert fos["lowe_karafiath"] == pytest.approx(0.964, abs=0.03)

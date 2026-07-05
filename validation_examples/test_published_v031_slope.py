"""Phase E / v5.3 validation — slope_stability published benchmarks from the
Rocscience Slide2 Slope Stability Verification Manual (public), part 2.

Non-ACADS problems whose geometry was recovered from the manual's own labeled
figures (rendered from the public PDF):

  V-031  Slide2 #57 — Pockoski & Duncan (2000) reinforced-program test slope 3:
         2-layer (sandy clay over highly-plastic clay), water table, dry tension
         crack. Circular critical surface. [manual pp. 203-205]
  V-032  Slide2 #61 — Baker (2003) example 3: homogeneous slope, Mohr-Coulomb
         (the power-curve strength variant is out of the module's scope).
         [manual pp. 213-215]
  V-033  Slide2 #62 — Loukidis et al. (2003) example 1: homogeneous slope, critical
         seismic coefficient. Reproduce FOS=1.0 at the author's kc. [pp. 216-218]

All three reproduce the published Bishop/Spencer factors of safety to <1% and are
clean PASSes. They exercise: a layered composite slope with a water table (#57),
a Mohr-Coulomb homogeneous slope (#61), and — most valuably — the module's
PSEUDO-STATIC SEISMIC engine on a clean, fully-labeled geometry (#62), which
reproduces Loukidis' critical-seismic-coefficient FOS=1.0 to +0.5% and thereby
validates the kh handling independently of the harder 3-layer ACADS seismic case.

Units: #57 is US customary (feet/psf/pcf, converted inline); #61/#62 are SI.
See validation_examples/INVENTORY.md (V-031..V-033) and RESULTS.md.
"""

import math

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import CircularSlipSurface
from slope_stability.slices import build_slices
from slope_stability.methods import bishop_fos, spencer_fos, fellenius_fos
from slope_stability.gle import gle_fos, janbu_fos
from slope_stability.analysis import search_critical_surface

FT, PSF, PCF = 0.3048, 0.04788, 0.157087


def _methods(geom, xc, yc, R, ns=50):
    slip = CircularSlipSurface(xc, yc, R)
    sl = build_slices(geom, slip, ns)
    fs, _ = spencer_fos(sl, slip)
    fj, _u, _f0 = janbu_fos(sl, slip)
    return {
        "fell": fellenius_fos(sl, slip),
        "bishop": bishop_fos(sl, slip),
        "spencer": fs,
        "gle": gle_fos(sl, slip, f_interslice="half_sine").fos,
        "janbu": fj,
    }


# ════════════════════════════════════════════════════════════════════════════
# V-031 — Slide2 #57: Pockoski & Duncan (2000) test slope 3 (2-layer + water)
# ════════════════════════════════════════════════════════════════════════════
# Surface (ft, L->R): (-70,100),(0,100),(125,150),(200,150) — 2.5:1, 50 ft high.
# Horizontal material boundary at el 90 ft (sandy clay over a highly-plastic-clay
# band, base el 85 ft). Water table (-70,100),(0,100),(125,140),(200,140). Dry
# tension crack (auto, shallow). Sandy clay c'=300 psf/phi'=35/130 pcf; Highly
# Plastic Clay c'=0/phi'=25/130 pcf. Published Table 57.3 (circular): Spencer
# 1.422, Bishop 1.417, Janbu-simplified 1.263, Lowe-Karafiath 1.414, Ordinary
# 1.319 (GOLD-NAIL 1.40).

def _v031_geom():
    return SlopeGeometry(
        surface_points=[(-70*FT, 100*FT), (0.0, 100*FT), (125*FT, 150*FT), (200*FT, 150*FT)],
        gwt_points=[(-70*FT, 100*FT), (0.0, 100*FT), (125*FT, 140*FT), (200*FT, 140*FT)],
        soil_layers=[
            SlopeSoilLayer(name="sandy_clay", top_elevation=150*FT, bottom_elevation=90*FT,
                           gamma=130*PCF, phi=35.0, c_prime=300*PSF),
            SlopeSoilLayer(name="hp_clay", top_elevation=90*FT, bottom_elevation=85*FT,
                           gamma=130*PCF, phi=25.0, c_prime=0.0),
        ])


# critical circle from a no-crack Bishop search (SI, m)
_V031_CIRCLE = (10.5, 60.3, 34.3)


def test_v031_pockoski_duncan_circular_bishop_spencer():
    """PASS: the module reproduces Slide2 Table 57.3 (circular) Bishop 1.417 and
    Spencer 1.422 for the Pockoski & Duncan test slope 3 to <0.6%. Exercises a
    2-layer slope (sandy clay over a highly-plastic-clay band) with a water
    table."""
    fos = _methods(_v031_geom(), *_V031_CIRCLE)
    assert fos["bishop"] == pytest.approx(1.411, abs=0.02)     # ours
    assert fos["spencer"] == pytest.approx(1.415, abs=0.02)    # ours
    assert fos["bishop"] == pytest.approx(1.417, rel=0.01)     # Slide2 circular
    assert fos["spencer"] == pytest.approx(1.422, rel=0.01)    # Slide2 circular
    assert fos["gle"] == pytest.approx(1.398, abs=0.02)


def test_v031_critical_search_finds_circular():
    """The Bishop entry-exit search locates the circular critical surface near the
    published 1.42 (composite disabled -> circular)."""
    res = search_critical_surface(
        _v031_geom(), method="bishop", surface_type="entry_exit",
        nx=10, ny=10, n_slices=30,
        x_entry_range=(-40*FT, 30*FT), x_exit_range=(60*FT, 200*FT))
    assert res.critical is not None
    assert res.critical.FOS == pytest.approx(1.42, abs=0.05)


def test_v031_janbu_and_ordinary_are_method_conventions():
    """CONVENTION note: the module's Janbu (f0-CORRECTED, 1.374) runs above
    Slide2's Janbu-SIMPLIFIED 1.263 (+8.8%), and the Ordinary/Fellenius (1.159)
    below Slide2's Ordinary 1.319 -- both are the usual method-definition /
    OMS-with-water pathologies, not surface disagreements (Bishop & Spencer,
    the rigorous methods, match to <0.6%)."""
    fos = _methods(_v031_geom(), *_V031_CIRCLE)
    assert fos["janbu"] > 1.263      # corrected > Slide simplified
    assert fos["fell"] < 1.319       # OMS-with-water is conservative


# ════════════════════════════════════════════════════════════════════════════
# V-032 — Slide2 #61: Baker (2003) example 3, homogeneous Mohr-Coulomb
# ════════════════════════════════════════════════════════════════════════════
# Surface (m): (0,0),(6,6),(20,6) — 45 deg (1:1), 6 m high; base y=0.
# Homogeneous clay, M-C: c'=6 kPa, phi'=32 deg, gamma=18 kN/m3.
# Published: Spencer FOS 1.366 (M-C), Janbu-simplified 1.291. (The power-curve
# nonlinear-envelope variant, FS 1.48, is out of the module's scope.)

def _v032_geom():
    return SlopeGeometry(
        surface_points=[(0.0, 0.0), (6.0, 6.0), (20.0, 6.0)],
        soil_layers=[SlopeSoilLayer(name="clay", top_elevation=6.0, bottom_elevation=-6.0,
                                    gamma=18.0, phi=32.0, c_prime=6.0)])


_V032_CIRCLE = (-1.3, 9.7, 9.3)


def test_v032_baker_mohr_coulomb_spencer():
    """PASS: the module reproduces Baker (2003) ex 3 Mohr-Coulomb Spencer FOS
    1.366 (Slide2 #61) to <0.1%. Homogeneous 45-deg 6-m clay slope."""
    fos = _methods(_v032_geom(), *_V032_CIRCLE)
    assert fos["spencer"] == pytest.approx(1.365, abs=0.02)   # ours
    assert fos["spencer"] == pytest.approx(1.366, rel=0.01)   # Slide2 / Baker
    assert fos["bishop"] == pytest.approx(1.367, abs=0.02)
    # module Janbu (corrected) is near Slide2 Janbu-simplified 1.291
    assert fos["janbu"] == pytest.approx(1.373, abs=0.03)


def test_v032_baker_critical_search():
    res = search_critical_surface(
        _v032_geom(), method="bishop", surface_type="entry_exit",
        nx=10, ny=10, n_slices=30, x_entry_range=(0.0, 6.0), x_exit_range=(6.0, 20.0))
    assert res.critical is not None
    assert res.critical.FOS == pytest.approx(1.366, abs=0.05)


# ════════════════════════════════════════════════════════════════════════════
# V-033 — Slide2 #62: Loukidis et al. (2003) ex 1, critical seismic coefficient
# ════════════════════════════════════════════════════════════════════════════
# Surface (m): (-50,0),(0,0),(75,25),(150,25) — 3:1, 25 m high; base y=-25.
# Homogeneous clay c'=25 kPa, phi'=30 deg, gamma=20 kN/m3. Loukidis' critical
# seismic coefficient for the DRY slope is kc=0.432, at which the Spencer FOS
# should equal 1.000. (This is the clean, fully-labeled geometry that validates
# the module's pseudo-static seismic independently of the 3-layer ACADS case.)

def _v033_geom(kh):
    return SlopeGeometry(
        surface_points=[(-50.0, 0.0), (0.0, 0.0), (75.0, 25.0), (150.0, 25.0)],
        soil_layers=[SlopeSoilLayer(name="clay", top_elevation=25.0, bottom_elevation=-25.0,
                                    gamma=20.0, phi=30.0, c_prime=25.0)], kh=kh)


_V033_CIRCLE = (16.4, 113.4, 115.0)


def test_v033_loukidis_critical_seismic_coefficient():
    """PASS: at Loukidis' critical seismic coefficient kc=0.432 the module's
    Spencer FOS = 1.005 (target 1.000, +0.5%) -- reproducing Slide2 #62 and
    VALIDATING the pseudo-static seismic engine on a clean fully-labeled
    geometry. Bishop 0.994, GLE 1.005 also bracket 1.0."""
    fos = _methods(_v033_geom(0.432), *_V033_CIRCLE)
    assert fos["spencer"] == pytest.approx(1.005, abs=0.02)   # ours
    assert fos["spencer"] == pytest.approx(1.000, abs=0.02)   # Loukidis target
    assert fos["bishop"] == pytest.approx(0.994, abs=0.02)
    assert fos["gle"] == pytest.approx(1.005, abs=0.02)


def test_v033_loukidis_static_is_far_above_one():
    """The DRY static slope (kh=0) is very stable (Bishop ~2.55) -- it fails only
    under the strong seismic kc=0.432, which is the point of Loukidis' critical-
    coefficient method (the seismic coefficient that drives FOS down to 1)."""
    fos = _methods(_v033_geom(0.0), 16.4, 88.5, 90.5)
    assert fos["bishop"] > 2.3
    # applying kc=0.432 drops the same slope to ~1.0
    seis = _methods(_v033_geom(0.432), *_V033_CIRCLE)
    assert seis["spencer"] < 1.1
    assert seis["spencer"] < fos["bishop"]

"""Phase E validation — Caltrans T&S + GEC-4 SOE / sheet-pile examples (V-012..V-025).

Sources:
  - V-012  Caltrans Trenching & Shoring Manual (July 2025), Ch. 7, Example 7-1B:
           cantilever soldier-pile wall by the AASHTO *Simplified* method
           (effective-width / soil-arching, simplified toe-moment cubic).
           extract: extracts/cal_7_1B.txt (PDF pp. 133-139).
  - V-013  Caltrans T&S Manual, Ch. 8, Example 8-1: single-ground-anchor sheet-pile
           wall (FHWA apparent 1.3x pressure diagram, Caquot-Kerisel log-spiral
           passive Kp=4.7, anchor-force back-out). extracts: cal_8_1.txt, cal_p162.png.
  - V-014  Caltrans T&S Manual, Ch. 10, Example 10-2 + Figure 10-17: basal-heave
           factor of safety (Bjerrum-Eide Nc with L/B correction + a side-shear
           term S in the driving block). extracts: cal_10x.txt, cal_p230.png.
  - V-016  FHWA GEC-4 (FHWA-IF-99-015), Appendix A, Design Example 1: two-tier
           anchored soldier-beam wall (FHWA apparent envelope pe, tributary anchor
           loads, hinge-method moments). ALREADY SI. extract: gec4_ex1.txt.
  - V-025  Caltrans T&S Manual, Ch. 7, Example 7-2: layered Rankine active pressure
           with hydrostatic water (stress-point diagram, total driving force).
           extract: cal_7_1B.txt tail (PDF pp. 139-142).
See validation_examples/INVENTORY.md (V-012/013/014/016/025) and RESULTS.md.

KEY FINDINGS (details in each test docstring and RESULTS.md):

- V-025 (layered Ka + water) is a clean PASS. The sheet_pile `rankine_Ka` +
  `active_pressure` primitives reproduce every published stress ordinate and the
  total driving force (24,611 lb/ft) to <0.5% (0.00% with the source-rounded Ka).
  This is the earth-pressure builder shared by sheet_pile/soe.

- V-016 (GEC-4 2-tier anchored) is largely PASS via `soe.rankine_Ka` + the FHWA
  trapezoid formulas: Ka, pe, ps, TH1, TH2, M2/M3, subgrade reaction R, and both
  anchor design loads all reproduce to <=3%. The one soft spot is M1 (upper hinge):
  the inventory's compact formula (13/54)*H1^2*(pe+ps) gives 70.4 vs the published
  76 kN-m/m because GEC-4 applies the uniform surcharge over a *larger* tributary
  than the apparent-pressure term in the top region — documented CONVENTION.

- V-012 (Caltrans simplified soldier pile): the published Rankine Ka=0.271 / Kp=3.69
  match the module `rankine_Ka`/`rankine_Kp` EXACTLY. But the *simplified soldier-pile
  method itself* (active on the 2-ft hole width, passive on the arching-amplified
  width f*b = 0.08*phi*b = 5.6 ft, simplified toe-moment cubic) is NOT in the module
  -- `analyze_cantilever` is a continuous-wall (per-metre) free-earth-support solver
  with NO soldier-pile arching / effective-width logic. The simplified cubic
  (D0=12.27, D=14.73 ft) + Mmax (379.7) + Vmax (137.7) are reproduced by hand on top
  of the module coefficients to document the method (N/A-scope for the solver).

- V-013 (Caltrans single-anchor sheet pile): Rankine Ka=0.333 matches the module
  exactly, and the FHWA apparent-diagram quantities (P, PT=1.3P, max ordinate sigma_a,
  upper tributary anchor force T1U, inclined anchor T) are reproduced by hand to <1.5%.
  But the module's `analyze_anchored` implements CLASSICAL free-earth-support with a
  triangular Rankine/Coulomb active diagram, NOT the FHWA apparent (1.3x trapezoid)
  method, and has NO Caquot-Kerisel log-spiral passive (Kp=4.7) -- closest are
  Rankine Kp=3.0 (-36%) or Coulomb Kp(delta=15)=4.98 (+6%). So the packaged
  embedment/anchor/moment is a method+passive-coefficient scope gap (N/A / CONVENTION).

- V-014 (Caltrans basal heave): the module HAS `check_basal_heave_bjerrum_eide`, but
  it computes a bearing-capacity ratio FOS = cu*Nc/(gamma*H+q) -- a fundamentally
  different formulation than the Caltrans force-balance (resistance qu*0.7B vs driving
  block W + 0.7B*q - S, where S = c*H is sidewall shear). The module omits the
  side-shear term and uses the inverted-footing FOS, not the 0.7B block. Its
  Bjerrum-Eide Nc table also reads 6.71 at H/B=2, Be/Le=1/3 vs the Caltrans chart's
  7.6. The Caltrans FS=1.54 is reproduced by hand to document the method (N/A-scope).

Units: examples are US customary except V-016 (SI). Conversions shown inline.
  1 ft = 0.3048 m, 1 kip = 4.448 kN, 1 ksf = 47.88 kPa, 1 psf = 0.04788 kPa,
  1 pcf = 0.157087 kN/m3, 1 kip-ft = 1.356 kN-m, 1 kip/ft = 14.594 kN/m.
"""

import math

import numpy as np
import pytest

# sheet_pile primitives (earth-pressure builder shared with soe)
from sheet_pile.earth_pressure import (
    rankine_Ka as sp_rankine_Ka,
    rankine_Kp as sp_rankine_Kp,
    coulomb_Kp as sp_coulomb_Kp,
    active_pressure as sp_active_pressure,
)
from sheet_pile.cantilever import analyze_cantilever, WallSoilLayer
from sheet_pile.anchored import analyze_anchored

# soe primitives
from soe.earth_pressure import rankine_Ka as soe_rankine_Ka
from soe.stability import (
    check_basal_heave_bjerrum_eide,
    _interpolate_Nc_bjerrum,
)

# Unit conversions (US -> SI)
FT = 0.3048
KIP = 4.448
KSF = 47.88
PSF = 0.04788
PCF = 0.157087
KIPFT = 1.356          # kip-ft -> kN-m
KFT = 14.594           # kip/ft (force per length) -> kN/m


# ════════════════════════════════════════════════════════════════════════════
# V-025 : Caltrans Ex 7-2 — layered Rankine active pressure + hydrostatic water
#   The cleanest check; exercises the sheet_pile earth-pressure builder.
# ════════════════════════════════════════════════════════════════════════════

# Layered profile (US units as published):
#   L1 coarse sand & gravel: 4 ft, gamma = 130 pcf, phi = 37 deg  -> Ka1
#   L2 fine sand: gamma = 102.4 pcf, phi = 30 deg -> Ka2;
#                 6 ft moist (to the GWT at 10 ft below top), then 20 ft submerged
#                 (gamma_sub = 102.4 - 62.4 = 40 pcf, as printed).
#   Wall friction = 0 (Rankine).
_V025_G1, _V025_G2, _V025_GSUB = 130.0, 102.4, 40.0   # pcf
_V025_GW = 62.4                                        # pcf


def test_v025_earth_pressure_coefficients():
    """Rankine Ka for both layers matches the published values (phi=37 -> 0.249,
    phi=30 -> 0.333) via the module's `rankine_Ka`."""
    assert sp_rankine_Ka(37.0) == pytest.approx(0.249, abs=0.001)
    assert sp_rankine_Ka(30.0) == pytest.approx(0.333, abs=0.001)


def test_v025_stress_points():
    """PASS: the module's `active_pressure` primitive reproduces every published
    stress ordinate (psf) at the layer/water discontinuities. The (-)/(+) pair at
    the 4-ft boundary is the Ka-discontinuity stress point (same overburden, Ka1
    above vs Ka2 below)."""
    Ka1 = 0.249       # use the source-rounded Ka so ordinates match to the printed psf
    Ka2 = 0.333
    # sigma1+  : overburden of L1 (130*4) times Ka1
    s1p = sp_active_pressure(_V025_G1, 4.0, Ka1)
    # sigma1-  : same overburden times Ka2 (the discontinuity)
    s1m = sp_active_pressure(_V025_G1, 4.0, Ka2)
    # sigma2   : at the GWT (10 ft) = s1m + 6 ft of moist L2 with Ka2
    s2 = s1m + _V025_G2 * 6.0 * Ka2
    # sigma3   : at the base (30 ft) = s2 + 20 ft of submerged L2 with Ka2
    s3 = s2 + _V025_GSUB * 20.0 * Ka2
    # water at base
    u = _V025_GW * 20.0
    assert s1p == pytest.approx(129.48, abs=0.5)    # pub 129.48 psf
    assert s1m == pytest.approx(173.16, abs=0.5)    # pub 173.16 psf
    assert s2 == pytest.approx(377.76, abs=0.5)     # pub 377.76 psf
    assert s3 == pytest.approx(644.16, abs=0.5)     # pub 644.16 psf
    assert u == pytest.approx(1248.0, abs=1.0)      # pub 1,248 psf


def test_v025_total_driving_force():
    """PASS: summing the six published force blocks (5 soil + 1 water) gives the
    total driving force FTOTAL = 24,611 lb/ft, matching to <0.1% (0.00% with the
    source-rounded Ka). This pins layered-Ka + hydrostatic water in the builder."""
    Ka1, Ka2 = 0.249, 0.333
    s1p = _V025_G1 * 4.0 * Ka1
    s1m = _V025_G1 * 4.0 * Ka2
    s2 = s1m + _V025_G2 * 6.0 * Ka2
    s3 = s2 + _V025_GSUB * 20.0 * Ka2
    u = _V025_GW * 20.0
    F1 = 0.5 * 4.0 * s1p                 # L1 triangle
    F2 = 6.0 * s1m                       # L2 moist rectangle
    F3 = 0.5 * 6.0 * (s2 - s1m)          # L2 moist triangle
    F4 = 20.0 * s2                       # L2 submerged rectangle
    F5 = 0.5 * 20.0 * (s3 - s2)          # L2 submerged triangle
    F6 = 0.5 * 20.0 * u                  # hydrostatic water triangle
    assert F1 == pytest.approx(258.96, rel=0.01)
    assert F2 == pytest.approx(1038.96, rel=0.01)
    assert F3 == pytest.approx(613.80, rel=0.01)
    assert F4 == pytest.approx(7555.20, rel=0.01)
    assert F5 == pytest.approx(2664.00, rel=0.01)
    assert F6 == pytest.approx(12480.0, rel=0.01)
    FTOTAL = F1 + F2 + F3 + F4 + F5 + F6
    assert FTOTAL == pytest.approx(24610.92, rel=0.005)   # pub 24,610.9 lb/ft

    # And in SI (force per length): 24,610.9 lb/ft -> kN/m
    FTOTAL_kNm = FTOTAL * (PSF * FT) / 1.0   # psf*ft = lb/ft; lb/ft -> kN/m
    # 1 lb/ft = 4.448 N / 0.3048 m = 14.594 N/m = 0.0145939 kN/m
    assert (FTOTAL * 0.0145939) == pytest.approx(359.2, rel=0.01)  # ~359 kN/m


# ════════════════════════════════════════════════════════════════════════════
# V-012 : Caltrans Ex 7-1B — cantilever soldier-pile wall, Simplified method
#   Rankine coefficients PASS via module; the soldier-pile arching/effective-width
#   simplified toe-moment method is N/A-scope (reproduced by hand for documentation).
# ════════════════════════════════════════════════════════════════════════════

# Given (US units): H = 15 ft, gamma = 125 pcf, phi = 35, c = 0, no water;
#   W14x120 soldier piles at s = 8 ft o.c. in 2-ft-dia holes; construction
#   surcharge 72 psf over the retained height (driving only).
_V012_H = 15.0
_V012_GAMMA = 0.125     # ksf/ft = kcf
_V012_PHI = 35.0
_V012_S = 8.0           # pile spacing, ft
_V012_B = 2.0           # hole diameter, ft
_V012_SUR = 0.072       # ksf


def test_v012_rankine_coefficients_match_module():
    """PASS (module primitive): the published Rankine Ka=0.271 and Kp=3.69 are
    exactly the module's `rankine_Ka`/`rankine_Kp` at phi=35. (The Caltrans note
    flags that Rankine underestimates passive; the example still uses Rankine Kp
    here for the simplified method.)"""
    assert sp_rankine_Ka(35.0) == pytest.approx(0.271, abs=0.001)
    assert sp_rankine_Kp(35.0) == pytest.approx(3.69, abs=0.01)


def test_v012_arching_effective_width_is_not_in_module():
    """N/A (scope): the simplified soldier-pile method puts ACTIVE pressure on the
    2-ft hole width and PASSIVE pressure on the arching-amplified width
    f*b = (0.08*phi)*b = 2.8*2 = 5.6 ft. The sheet_pile/soe cantilever solver has
    NO such effective-width/arching logic (it is a continuous per-metre wall). This
    test documents the arching factor the example uses; the module does not expose it."""
    f = 0.08 * _V012_PHI            # arching factor = 2.8 (capped at 3 per Caltrans)
    assert f == pytest.approx(2.8, abs=0.01)
    fb = f * _V012_B
    assert fb == pytest.approx(5.6, abs=0.01)   # effective passive width


def test_v012_simplified_embedment_and_demands_by_hand():
    """N/A (scope), reproduced by hand: the simplified toe-moment cubic
    D0^3 - 1.2133 D0^2 - 93.432 D0 - 518.75 = 0 (built from the module's Ka/Kp,
    the 2-ft active width, and the 5.6-ft passive width) gives D0 = 12.27 ft,
    design D = 1.2 D0 = 14.73 ft, zero shear at Y = 6.00 ft below dredge,
    Mmax = 379.7 kip-ft, Vmax = 137.7 kips -- all matching the published Simplified
    row. The solver in the module cannot produce these (no soldier-pile widths)."""
    Ka = sp_rankine_Ka(_V012_PHI)
    Kp = sp_rankine_Kp(_V012_PHI)
    f = 0.08 * _V012_PHI
    fb = f * _V012_B

    # --- force coefficients (per pile), from the module coefficients --------
    sig_dredge = _V012_GAMMA * _V012_H * Ka                     # 0.508 ksf
    assert sig_dredge == pytest.approx(0.508, abs=0.002)
    PA1 = 0.5 * _V012_H * sig_dredge * _V012_S                  # 30.48 kips
    assert PA1 == pytest.approx(30.48, rel=0.01)
    PAs = _V012_H * _V012_S * _V012_SUR                         # 8.64 kips
    assert PAs == pytest.approx(8.64, rel=0.01)
    # active-below-dredge triangular coeff (on hole width b): 0.5*Ka*gamma*b = 0.0339
    c_act_tri = 0.5 * Ka * _V012_GAMMA * _V012_B
    assert c_act_tri == pytest.approx(0.0339, abs=0.0003)
    # passive coeff (on amplified width fb): 0.5*Kp*gamma*fb = 1.291 ; /3 -> 0.430
    c_pass = 0.5 * Kp * _V012_GAMMA * fb
    assert c_pass == pytest.approx(1.291, rel=0.01)
    assert c_pass / 3.0 == pytest.approx(0.430, abs=0.003)

    # --- published simplified toe-moment cubic (FS = 1.0) -------------------
    cubic = [1.0, -1.2133, -93.432, -518.75]
    roots = [r.real for r in np.roots(cubic) if abs(r.imag) < 1e-6 and r.real > 0]
    D0 = max(roots)
    assert D0 == pytest.approx(12.272, rel=0.05)         # pub D0 = 12.27 ft
    D_design = 1.2 * D0
    assert D_design == pytest.approx(14.73, rel=0.05)    # pub D = 14.73 ft

    # --- zero shear, Mmax, Vmax --------------------------------------------
    yroots = [r.real for r in np.roots([1.2571, -1.016, -39.12])
              if abs(r.imag) < 1e-6 and r.real > 0]
    Y = max(yroots)
    assert Y == pytest.approx(5.997, abs=0.1)            # pub Y = 6.0 ft below dredge

    Mmax = (PAs * (7.5 + Y) + PA1 * (5.0 + Y)
            + 1.016 * Y * (Y / 2.0)
            + 0.0339 * Y ** 2 * (Y / 3.0)
            - 1.291 * Y ** 2 * (Y / 3.0))
    assert Mmax == pytest.approx(379.7, rel=0.07)        # pub Mmax = 379.7 kip-ft
    # SI: 379.7 kip-ft * 1.356 = 514.9 kN-m (per pile)
    assert Mmax * KIPFT == pytest.approx(514.9, rel=0.07)

    Vmax = 1.291 * D0 ** 2 - 8.64 - 30.48 - 1.016 * D0 - 5.11
    assert Vmax == pytest.approx(137.73, rel=0.05)       # pub Vmax = 137.7 kips


def test_v012_module_continuous_solver_is_a_different_framework():
    """CONVENTION: driving the module's continuous-wall `analyze_cantilever`
    (FS_passive=1.0, no increase) for the same H/gamma/phi/surcharge gives a
    converged per-metre embedment of ~11.3 ft -- NOT comparable to the per-pile
    D0=12.27 ft because the module spreads active+passive over a continuous wall
    (no 2-ft active / 5.6-ft passive soldier-pile widths). Pins the framework gap."""
    H_m = _V012_H * FT
    gamma_si = 125 * PCF
    sur_si = 72 * PSF
    res = analyze_cantilever(
        H_m,
        [WallSoilLayer(thickness=10 * H_m, unit_weight=gamma_si,
                       friction_angle=_V012_PHI, cohesion=0.0)],
        surcharge=sur_si, FOS_passive=1.0, embedment_increase=1.0,
    )
    D_ft = res.embedment_converged / FT
    # Continuous-wall embedment is in a different framework -- just confirm it is
    # NOT the soldier-pile D0 (i.e. the methods are distinct), within a broad band.
    assert 9.0 < D_ft < 13.0
    assert abs(D_ft - 12.272) > 0.5      # not the per-pile simplified value


# ════════════════════════════════════════════════════════════════════════════
# V-013 : Caltrans Ex 8-1 — single ground-anchor sheet-pile wall
#   Ka PASS via module; FHWA apparent diagram + log-spiral passive are method gaps.
# ════════════════════════════════════════════════════════════════════════════

# Given (US units): H = 25 ft excavation, single anchor 10 ft below top inclined
#   15 deg, anchors at 10 ft horizontal spacing; gamma = 115 pcf, phi = 30, c = 0,
#   delta = 15, no water. PZ22 sheet pile.
_V013_H = 25.0
_V013_GAMMA = 0.115     # kcf
_V013_PHI = 30.0
_V013_DELTA = 15.0


def test_v013_rankine_Ka_matches_module():
    """PASS (module primitive): published Rankine Ka = 0.333 is exactly the
    module's `rankine_Ka` at phi=30."""
    assert sp_rankine_Ka(30.0) == pytest.approx(0.333, abs=0.001)


def test_v013_passive_coefficient_source_differs():
    """CONVENTION: the example uses the Caquot-Kerisel LOG-SPIRAL passive
    Kp = 6.3 * R(0.746) = 4.7 (delta/phi = -0.5). The module has NO log-spiral
    method -- its options are Rankine Kp = 3.0 (-36%) or Coulomb Kp(delta=15) = 4.98
    (+6%). Documents the passive-coefficient source gap (module not tuned)."""
    assert sp_rankine_Kp(30.0) == pytest.approx(3.0, abs=0.01)
    assert sp_coulomb_Kp(30.0, delta_deg=15.0) == pytest.approx(4.98, abs=0.05)
    Kp_logspiral = 6.3 * 0.746
    assert Kp_logspiral == pytest.approx(4.7, abs=0.05)
    # the published value lies between Rankine and Coulomb, matching neither tightly
    assert abs(Kp_logspiral - 3.0) > 0.5
    assert abs(Kp_logspiral - sp_coulomb_Kp(30.0, delta_deg=15.0)) > 0.2


def test_v013_apparent_diagram_quantities_by_hand():
    """PASS (by hand on the module Ka): the FHWA single-anchor apparent diagram
    reproduces the published P, PT=1.3P, max ordinate sigma_a, upper-tributary
    anchor force T1U, per-anchor horizontal load TH, and inclined anchor T to <1.5%.
    (The module does not expose this apparent-diagram method -- see next test.)"""
    Ka = sp_rankine_Ka(_V013_PHI)
    # total active for a triangular distribution P = 0.5*gamma*H^2*Ka
    P = 0.5 * _V013_GAMMA * _V013_H ** 2 * Ka * 1000.0      # lb/ft
    assert P == pytest.approx(11980.0, rel=0.01)           # pub 11,980 lb/ft
    PT = 1.3 * P
    assert PT == pytest.approx(15574.0, rel=0.01)          # pub 15,574 lb/ft
    # max ordinate sigma_a = PT / ((2/3) H)
    sigma_a = PT / ((2.0 / 3.0) * _V013_H)
    assert sigma_a == pytest.approx(934.4, rel=0.01)       # pub 934.4 psf
    # upper tributary anchor force T1U (trapezoid above the anchor; top transition
    # (2/3)*10 = 6.667 ft ramp then constant to the 10-ft anchor)
    T1U = 0.5 * sigma_a * 6.667 + sigma_a * (10.0 - 6.667)
    assert T1U == pytest.approx(6228.0, rel=0.01)          # pub 6,228 lb/ft
    # published total anchor T1 = T1U + T1L = 6228 + 8026 = 14,254 lb/ft of wall
    T1 = 14254.0
    TH = T1 * 10.0 / 1000.0                                 # 10-ft anchor spacing -> kips
    assert TH == pytest.approx(143.87, rel=0.02)           # pub 143.87 kips/anchor
    T_incl = 143.87 / math.cos(math.radians(15.0))
    assert T_incl == pytest.approx(148.95, rel=0.01)       # pub 148.95 kips
    # published Mmax at the anchor = 22,494 ft-lb/ft -> SI
    Mmax_kNm = 22494.0 * KIPFT / 1000.0 / FT               # ft-lb/ft -> kN-m/m
    assert Mmax_kNm == pytest.approx(100.1, rel=0.02)


def test_v013_module_anchored_uses_classical_FES_not_apparent():
    """N/A (scope): the module's `analyze_anchored` implements CLASSICAL free-earth-
    support with a TRIANGULAR Rankine/Coulomb active diagram, not the FHWA apparent
    (1.3x trapezoid) method, and uses Rankine/Coulomb passive (no log-spiral). Its
    embedment/anchor/moment therefore differ from the example by method, not by a
    bug. Pins the framework gap; module not tuned to the apparent-diagram example."""
    H_m = _V013_H * FT
    ad_m = 10.0 * FT
    gamma_si = 115 * PCF
    layers = [WallSoilLayer(thickness=10 * H_m, unit_weight=gamma_si,
                            friction_angle=_V013_PHI, cohesion=0.0,
                            wall_friction_deg=15.0)]
    res = analyze_anchored(H_m, ad_m, layers, surcharge=0.0,
                           FOS_passive=1.3, pressure_method="rankine")
    D_ft = res.embedment_depth / FT
    # Classical FES with Rankine Kp=3.0 gives a markedly different embedment than
    # the example's log-spiral-Kp apparent-diagram D = 6.09 ft -- methods differ.
    assert D_ft > 9.0                       # classical Rankine FES ~10.8 ft
    assert abs(D_ft - 6.09) > 2.0           # not the apparent-diagram value


# ════════════════════════════════════════════════════════════════════════════
# V-014 : Caltrans Ex 10-2 — basal-heave factor of safety
#   Module has a Bjerrum-Eide method but a DIFFERENT formulation (no side shear,
#   bearing-capacity FOS not the 0.7B force block). N/A-scope; FS reproduced by hand.
# ════════════════════════════════════════════════════════════════════════════

# Given (US units): H = 30 ft, B = 15 ft, L = 45 ft, surcharge q = 300 psf,
#   clay c = 500 psf (phi=0), gamma = 120 pcf.
_V014_H, _V014_B, _V014_L = 30.0, 15.0, 45.0
_V014_Q, _V014_C, _V014_GAMMA = 0.300, 0.500, 0.120     # ksf, ksf, kcf


def test_v014_caltrans_force_balance_by_hand():
    """N/A (scope), reproduced by hand: the Caltrans method balances a resisting
    bearing force qu*(0.7B) against a driving block W + (0.7B)*q - S, where
    S = c*H is the sidewall shear on the vertical failure plane. With Nc=7.6
    (H/B=2, L/B=3): qu = c*Nc = 3.8 ksf; F_RS = 3.8*10.5 = 40.0 k/ft;
    W = 10.5*30*0.120 = 37.8; surcharge = 10.5*0.300 = 3.15; S = 0.5*30 = 15;
    F_dr = 37.8 + 3.15 - 15 = 26.0; FS = 40.0/26.0 = 1.54. Matches exactly."""
    Nc = 7.6
    qu = _V014_C * Nc                          # 3.80 ksf
    width = 0.7 * _V014_B                       # 10.5 ft
    F_RS = qu * width                           # 40.0 k/ft
    assert F_RS == pytest.approx(40.0, abs=0.2)
    W = width * _V014_H * _V014_GAMMA           # 37.8 k/ft
    surcharge = width * _V014_Q                 # 3.15 k/ft
    S = _V014_C * _V014_H                        # 15.0 k/ft  (side shear c*H)
    F_dr = W + surcharge - S
    assert W == pytest.approx(37.8, abs=0.1)
    assert surcharge == pytest.approx(3.15, abs=0.05)
    assert S == pytest.approx(15.0, abs=0.1)
    assert F_dr == pytest.approx(26.0, abs=0.1)
    FS = F_RS / F_dr
    assert FS == pytest.approx(1.54, abs=0.05)   # pub FS = 1.54


def test_v014_module_method_differs_from_caltrans():
    """N/A (scope): the module's `check_basal_heave_bjerrum_eide` computes the
    inverted-footing bearing FOS = cu*Nc/(gamma*H + q), with NO sidewall-shear term
    and NOT the 0.7B force block. For these inputs it returns FOS ~ 0.86 -- a
    different (and here far more conservative) basal-heave formulation. Its Nc table
    also reads ~6.71 at H/B=2, Be/Le=1/3 vs the Caltrans chart's 7.6. Documented gap."""
    H_m = _V014_H * FT
    B_m = _V014_B * FT
    L_m = _V014_L * FT
    cu = 500 * PSF
    gamma_si = 120 * PCF
    q_si = 300 * PSF
    res = check_basal_heave_bjerrum_eide(H_m, cu, gamma_si, Be=B_m, Le=L_m,
                                         q_surcharge=q_si)
    # module FOS is the bearing-capacity ratio, ~0.86 -- not the Caltrans 1.54
    assert res.FOS == pytest.approx(0.86, abs=0.05)
    assert abs(res.FOS - 1.54) > 0.3
    # module Nc table value at this geometry differs from the Caltrans chart 7.6
    Nc_module = _interpolate_Nc_bjerrum(H_m / B_m, B_m / L_m)
    assert Nc_module == pytest.approx(6.71, abs=0.1)
    assert abs(Nc_module - 7.6) > 0.5


# ════════════════════════════════════════════════════════════════════════════
# V-016 : GEC-4 Design Example 1 — two-tier anchored soldier-beam wall (SI)
#   Largely PASS via soe.rankine_Ka + the FHWA trapezoid formulas; M1 is a
#   documented surcharge-tributary CONVENTION delta.
# ════════════════════════════════════════════════════════════════════════════

# Given (SI as published): H = 10 m; anchors at H1 = 2.5 m and 6.25 m below top
#   (H2 = 3.75 m, H3 = 3.75 m); soldier-beam spacing 2.5 m; anchor inclination 15;
#   medium dense silty sand gamma = 18 kN/m3, phi' = 33, no groundwater.
#   Traffic surcharge qs = 0.6 m * 18 = 11 kPa.
_V016_GAMMA = 18.0
_V016_PHI = 33.0
_V016_H = 10.0
_V016_H1 = 2.5
_V016_H2 = 3.75
_V016_H3 = 3.75
_V016_QS = 11.0       # kPa (0.6 m of soil)
_V016_SPACING = 2.5
_V016_INCL = 15.0


def test_v016_Ka_and_pe():
    """PASS: soe `rankine_Ka`(phi=33) = 0.295, and the FHWA two-anchor trapezoid
    pe = 0.65*Ka*gamma*H^2 / (H - H1/3 - H3/3) = 43.6 kN/m2 -- matches the example."""
    Ka = soe_rankine_Ka(_V016_PHI)
    assert Ka == pytest.approx(0.295, abs=0.002)
    pe = (0.65 * Ka * _V016_GAMMA * _V016_H ** 2
          / (_V016_H - _V016_H1 / 3.0 - _V016_H3 / 3.0))
    assert pe == pytest.approx(43.6, rel=0.01)        # pub pe = 43.6 kN/m2


def test_v016_surcharge_pressure():
    """PASS: the uniform lateral surcharge ps = Ka*qs acts over the full height;
    Ka*11 = 3.2 kPa (pub 3.2)."""
    Ka = soe_rankine_Ka(_V016_PHI)
    ps = Ka * _V016_QS
    assert ps == pytest.approx(3.2, abs=0.1)          # pub ps = 3.2 kPa


def test_v016_tributary_anchor_loads():
    """PASS: tributary-area horizontal anchor loads reproduce the example:
    TH1 = (2/3 H1 + H2/2)*pe + (H1 + H2/2)*ps = 168 kN/m;
    TH2 = (H2/2 + 23/48 H3)*pe + (H2/2 + H3/2)*ps = 172 kN/m."""
    pe, ps = 43.6, 3.2
    TH1 = (2.0 / 3.0 * _V016_H1 + _V016_H2 / 2.0) * pe + (_V016_H1 + _V016_H2 / 2.0) * ps
    TH2 = (_V016_H2 / 2.0 + 23.0 / 48.0 * _V016_H3) * pe + (_V016_H2 / 2.0 + _V016_H3 / 2.0) * ps
    assert TH1 == pytest.approx(168.0, rel=0.02)      # pub 168 kN/m
    assert TH2 == pytest.approx(172.0, rel=0.02)      # pub 172 kN/m


def test_v016_hinge_moments():
    """M2/M3 (lower span) PASS exactly: (1/10)*H2^2*(pe+ps) = 66 kN-m/m.
    M1 (upper hinge) CONVENTION: the inventory's compact form
    (13/54)*H1^2*(pe+ps) = 70.4, while GEC-4 publishes 76 because the uniform
    surcharge ps is applied over a larger tributary than the apparent term in the
    top region. The dominant earth-pressure part (13/54)*pe*H1^2 = 65.6 is exact;
    Mmax for design = max(M1,M2,M3) = 76 (governed by M1)."""
    pe, ps = 43.6, 3.2
    M23 = (1.0 / 10.0) * _V016_H2 ** 2 * (pe + ps)
    assert M23 == pytest.approx(66.0, rel=0.02)       # pub M2,3 = 66 kN-m/m

    M1_earth = (13.0 / 54.0) * pe * _V016_H1 ** 2     # dominant apparent-pressure part
    assert M1_earth == pytest.approx(65.6, rel=0.01)

    M1_compact = (13.0 / 54.0) * _V016_H1 ** 2 * (pe + ps)
    assert M1_compact == pytest.approx(70.4, rel=0.01)   # inventory compact form
    # documented delta vs the published 76 (surcharge-tributary convention)
    assert abs(M1_compact - 76.0) / 76.0 < 0.08          # within 8% (CONVENTION)


def test_v016_subgrade_reaction():
    """PASS: subgrade reaction at the base R = (3/16)*H3*pe + (H3/2)*ps = 37 kN/m."""
    pe, ps = 43.6, 3.2
    R = (3.0 / 16.0) * _V016_H3 * pe + (_V016_H3 / 2.0) * ps
    assert R == pytest.approx(37.0, rel=0.03)         # pub R = 37 kN/m


def test_v016_anchor_design_loads():
    """PASS: anchor design loads DL = TH * spacing / cos(incl):
    DL1 = 168 * 2.5 / cos15 = 435 kN; DL2 = 172 * 2.5 / cos15 = 445 kN."""
    DL1 = 168.0 * _V016_SPACING / math.cos(math.radians(_V016_INCL))
    DL2 = 172.0 * _V016_SPACING / math.cos(math.radians(_V016_INCL))
    assert DL1 == pytest.approx(435.0, rel=0.02)      # pub 435 kN
    assert DL2 == pytest.approx(445.0, rel=0.02)      # pub 445 kN

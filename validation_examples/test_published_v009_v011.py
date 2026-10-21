"""Phase E validation — GEC-11 Example E4/E7 MSE wall design (V-009..V-011).

Source: FHWA GEC-11 Vol 2 (FHWA-NHI-10-025), Appendix E.
  - Example E4: segmental-precast-panel MSE wall, level backfill + LL surcharge,
    steel bar-mat reinforcement (V-009 external, V-010 internal).
  - Example E7: the same wall (Example #4) under earthquake loading —
    Mononobe-Okabe KAE + seismic sliding CDR (V-011).
See validation_examples/INVENTORY.md entries V-009..V-011 and RESULTS.md.
Extraction artifacts: extracts/g11_E4.txt, g11_E7b.txt.

KEY FINDINGS (details in each test docstring and RESULTS.md):

- V-011 (M-O regression anchor) is a clean PASS. The repo's M-O active
  coefficient gives KAE = 0.4782 vs the published 0.4785 (-0.06%, well inside
  +/-2%); PAE = 19.65 k/ft (exact); and the full seismic sliding chain
  (THF=24.64, V=67.52, R=38.98, CDR=1.58) reproduces the example exactly when
  built on the module KAE. delta=phi=30, kv=0 is handled correctly — no bug.

- V-009: the GEC-11 LRFD bookkeeping (Strength I max/min load-factor pairing,
  LL-on-resisting-side exclusion) is NOT packaged in `analyze_mse_wall` — the
  high-level MSE API reports ASD factors of safety (R/demand, no load factors).
  So the high-level external API is a documented ASD-vs-LRFD scope gap.
  HOWEVER, the module's earth-pressure + geometry + MSE *primitives*
  (`rankine_Ka`, `horizontal_force_active`, `Tmax_at_level`,
  `pullout_resistance`) reproduce every published quantity to <=2% when driven
  with the example's LRFD factors. The tests assert the primitives (PASS) and
  pin the high-level ASD-vs-LRFD gap (CONVENTION).

- V-010 (bar-mat internal): the steel bar-mat / welded-grid Kr/Ka (2.5->1.2)
  and F* (20(t/St)->10(t/St)) curves are NOW built into `Kr_Ka_ratio` /
  `F_star_metallic` (reinforcement_type "bar_mat"/"welded_grid"/"metallic_grid";
  v5.2 Q4). The V-010 tests drive the BUILT-IN curves (no hand-fed
  coefficients) into `Tmax_at_level` / `pullout_resistance` and reproduce
  Table E4-7.4 Levels 1/4/7/10 (sigma_H/Tmax <=3%, F*(L4)=0.955). The
  former ribbed-STRIP-only coverage gap is closed.

Units: example is US customary; modules are SI.
  1 ft = 0.3048 m, 1 kip = 4.448 kN, 1 ksf = 47.88 kPa, 1 pcf = 0.157087 kN/m3.
  Force per length: 1 kip/ft = 14.594 kN/m.
  Moment per length: 1 k-ft/ft = 1.356/0.3048 = 4.4488 kN-m/m.
"""

import math

import pytest

from retaining_walls.earth_pressure import rankine_Ka, horizontal_force_active
from retaining_walls.mse import (
    analyze_mse_wall, check_external_stability_lrfd, Kr_Ka_ratio, F_star_metallic,
    Tmax_at_level, pullout_resistance,
)
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.reinforcement import Reinforcement
from seismic_geotech.mononobe_okabe import mononobe_okabe_KAE

FT = 0.3048
KIP = 4.448
KSF = 47.88
PCF = 0.157087
KFT = 14.594               # 1 kip/ft (force per length) -> kN/m
KFTFT = 1.356 / 0.3048     # 1 k-ft/ft (moment per length) -> kN-m/m  (= 4.4488)

# ── Shared Example-E4 wall parameters (SI) ──────────────────────────────────
H = 25.64 * FT             # design height (He=23.64 + embed 2.0 ft)
L = 18.0 * FT              # reinforcement length (0.7H)
GAMMA = 125 * PCF          # reinforced / retained / foundation fill
PHI_R = 34.0               # reinforced fill
PHI_F = 30.0               # retained fill
PHI_FD = 30.0              # foundation
Q_LL = 0.25 * KSF          # live-load surcharge (heq = 2 ft of soil)
# GM-256: the bar mat's coverage, E4 p. E4-18: Level 4 is "4W11 + W11x0.5'",
# 4 longitudinal wires at 0.5 ft, so b = 1.5 ft between the outside wires,
# "for each panel of 5 ft width" (Rc = 0.30). The external checks and the
# internal Kr/Ka and F* pinned here do not depend on it.
_E4_MAT = {"width": 1.5 * FT, "horizontal_spacing": 5.0 * FT}


# ── V-009 : MSE external stability — unfactored loads + LRFD CDRs ────────────

def _unfactored_loads():
    """Unfactored vertical/horizontal forces and moments about Point A (toe),
    built from the module's earth-pressure primitives. Returns SI values."""
    Kaf = rankine_Ka(PHI_F)
    V1 = GAMMA * H * L                 # EV: reinforced mass weight
    Vs = Q_LL * L                      # LL: surcharge over the mass
    F1 = 0.5 * Kaf * GAMMA * H ** 2    # EH: active earth thrust
    F2 = Kaf * Q_LL * H                # LL: surcharge thrust
    MV1 = V1 * (L / 2.0)
    MVs = Vs * (L / 2.0)
    MF1 = F1 * (H / 3.0)
    MF2 = F2 * (H / 2.0)
    return dict(Kaf=Kaf, V1=V1, Vs=Vs, F1=F1, F2=F2,
                MV1=MV1, MVs=MVs, MF1=MF1, MF2=MF2)


def test_v009_earth_pressure_coefficients():
    """Ka for reinforced (phi=34) and retained (phi=30) fill match the example."""
    assert rankine_Ka(PHI_R) == pytest.approx(0.283, abs=0.001)   # Kar
    assert rankine_Ka(PHI_F) == pytest.approx(0.333, abs=0.001)   # Kaf


def test_v009_unfactored_forces_and_moments():
    """PASS: the module's `rankine_Ka` + the GEC-11 force equations reproduce all
    Table E4-4.3/4.4 unfactored forces and moments about Point A to <0.5%.
    These are the load quantities the analysis module's primitives contribute."""
    u = _unfactored_loads()
    assert u["V1"] / KFT == pytest.approx(57.69, rel=0.005)        # k/ft
    assert u["Vs"] / KFT == pytest.approx(4.50, rel=0.01)
    assert u["F1"] / KFT == pytest.approx(13.68, rel=0.01)
    assert u["F2"] / KFT == pytest.approx(2.13, rel=0.01)
    assert u["MV1"] / KFTFT == pytest.approx(519.21, rel=0.005)    # k-ft/ft
    assert u["MF1"] / KFTFT == pytest.approx(116.94, rel=0.01)
    assert u["MF2"] / KFTFT == pytest.approx(27.36, rel=0.01)


def test_v009_horizontal_force_active_primitive():
    """The module's `horizontal_force_active` returns the combined F1+F2 thrust
    and its line of action, the building block of every external check."""
    Kaf = rankine_Ka(PHI_F)
    Pa, z_Pa = horizontal_force_active(GAMMA, H, Kaf, q=Q_LL)
    u = _unfactored_loads()
    assert Pa / KFT == pytest.approx((u["F1"] + u["F2"]) / KFT, rel=1e-6)
    assert Pa / KFT == pytest.approx(15.83, rel=0.01)             # 13.68+2.13


def test_v009_sliding_CDR_lrfd():
    """PASS (LRFD bookkeeping on module primitives): sliding capacity:demand
    ratios reproduce Table E4-6.1. LL is excluded from the resisting side; the
    sliding interface uses phi_fd (foundation < reinforced). Str I max=1.85,
    min=2.08, critical (min VFm / max Hm)=1.37 — all within +/-0.05."""
    u = _unfactored_loads()
    tanfd = math.tan(math.radians(PHI_FD))
    # Strength I (max): EV=1.35, EH=1.50, LL=1.75 ; phi_sliding = 1.0
    Hm_max = 1.50 * u["F1"] + 1.75 * u["F2"]
    VN_max = (1.35 * u["V1"]) * tanfd
    assert VN_max / Hm_max == pytest.approx(1.85, abs=0.05)
    # Strength I (min): EV=1.00, EH=0.90, LL=1.75
    Hm_min = 0.90 * u["F1"] + 1.75 * u["F2"]
    VN_min = (1.00 * u["V1"]) * tanfd
    assert VN_min / Hm_min == pytest.approx(2.08, abs=0.05)
    # Critical max/min: min capacity over max demand
    assert VN_min / Hm_max == pytest.approx(1.37, abs=0.05)


def test_v009_eccentricity_lrfd():
    """PASS: limiting-eccentricity computation reproduces Table E4-6.2 (LL
    excluded). Str I max eL = 2.87 ft (< L/4 = 4.50); critical max/min eL =
    3.87 ft — within +/-0.05 ft."""
    u = _unfactored_loads()
    # Strength I (max), LL excluded from resisting
    VA = 1.35 * u["V1"]
    MRA = 1.35 * u["MV1"]
    MOA = 1.50 * u["MF1"] + 1.75 * u["MF2"]
    a = (MRA - MOA) / VA
    eL = L / 2.0 - a
    assert eL / FT == pytest.approx(2.87, abs=0.05)
    assert (L / 4.0) / FT == pytest.approx(4.50, abs=0.02)   # limit
    # Critical max/min: min resisting moment (EV=1.0), max overturning
    MRA_C = 1.00 * u["MV1"]
    VA_C = 1.00 * u["V1"]
    a_C = (MRA_C - MOA) / VA_C
    eL_C = L / 2.0 - a_C
    assert eL_C / FT == pytest.approx(3.87, abs=0.05)


def test_v009_bearing_lrfd():
    """PASS: bearing computation reproduces Table E4-6.3 (LL INCLUDED — it
    increases bearing stress). Str I max: eL=2.60 ft, B'=12.79 ft,
    sigma_v=6.70 ksf, CDR=1.57. Service I sigma_v=4.66 ksf. The critical
    max/min combo (sigma_v=5.86 ksf, CDR=1.79) reproduces to ~+/-2% — within the
    example's own 'consistent-values' rounding note."""
    u = _unfactored_loads()
    qnf_str = 10.50 * KSF
    # Strength I (max)
    SV = 1.35 * u["V1"] + 1.75 * u["Vs"]
    MRA = 1.35 * u["MV1"] + 1.75 * u["MVs"]
    MOA = 1.50 * u["MF1"] + 1.75 * u["MF2"]
    a = (MRA - MOA) / SV
    eL = L / 2.0 - a
    Bp = L - 2.0 * eL
    sigma_v = SV / Bp
    assert eL / FT == pytest.approx(2.60, abs=0.05)
    assert Bp / FT == pytest.approx(12.79, abs=0.05)
    assert sigma_v / KSF == pytest.approx(6.70, rel=0.02)
    assert qnf_str / sigma_v == pytest.approx(1.57, abs=0.05)
    # Service I (all factors 1.0, LL included)
    SVs = u["V1"] + u["Vs"]
    MRAs = u["MV1"] + u["MVs"]
    MOAs = u["MF1"] + u["MF2"]
    a_s = (MRAs - MOAs) / SVs
    eLs = L / 2.0 - a_s
    sigma_vs = SVs / (L - 2.0 * eLs)
    assert sigma_vs / KSF == pytest.approx(4.66, rel=0.02)
    # Critical max/min combo (slightly looser — see docstring)
    MRA_C = 1.00 * (u["MV1"] + u["MVs"])
    VC = 1.00 * (u["V1"] + u["Vs"])
    a_C = (MRA_C - MOA) / VC
    eL_C = L / 2.0 - a_C
    sigma_vC = VC / (L - 2.0 * eL_C)
    assert sigma_vC / KSF == pytest.approx(5.86, rel=0.025)


def test_v009_highlevel_api_reports_asd_fos_not_lrfd_cdr():
    """CONVENTION / scope: the high-level `analyze_mse_wall` returns ASD factors
    of safety (R/demand, NO load factors), not the GEC-11 LRFD CDRs. This
    pins the documented API gap — the ASD sliding FoS is the *unfactored*
    resistance/demand ratio, distinctly larger than the LRFD sliding CDR
    (1.85). The module is not tuned to the LRFD example. Like Eq. 4-12
    (p. 4-21, "the live load is excluded since it increases sliding
    stability") the resisting weight leaves the live load over the
    reinforced zone out (GM-164: it was counted, ~2.27; now ~2.10;
    live_load_resists=True gives the old free body)."""
    geom = MSEWallGeometry(wall_height=H, reinforcement_length=L,
                           reinforcement_spacing=2.5 * FT, surcharge=Q_LL)
    rebar = Reinforcement(name="bar mat", type="metallic_grid",
                          Tallowable=200.0, Fy=448000.0, thickness=0.0095,
                          **_E4_MAT)
    res = analyze_mse_wall(
        geom, gamma_backfill=GAMMA, phi_backfill=PHI_R, reinforcement=rebar,
        gamma_foundation=GAMMA, phi_foundation=PHI_FD,
        q_allowable=10.50 * KSF, phi_retained=PHI_F, gamma_retained=GAMMA,
    )
    # ASD sliding FoS uses unfactored W and unfactored thrust -> > LRFD CDR
    assert res.FOS_sliding > 1.85          # ASD, not the LRFD 1.85
    # It is an UNFACTORED R/demand ratio (W = V1 alone, the live load left
    # out as in Eq. 4-12; demand is the combined earth+surcharge thrust),
    # NOT the load-factored LRFD CDR.
    Kaf = rankine_Ka(PHI_F)
    Pa, _ = horizontal_force_active(GAMMA, H, Kaf, q=Q_LL)
    W = GAMMA * H * L
    asd_expected = (W * math.tan(math.radians(PHI_FD))) / Pa
    assert res.FOS_sliding == pytest.approx(asd_expected, rel=0.02)   # ~2.10
    old = analyze_mse_wall(
        geom, gamma_backfill=GAMMA, phi_backfill=PHI_R, reinforcement=rebar,
        gamma_foundation=GAMMA, phi_foundation=PHI_FD,
        q_allowable=10.50 * KSF, phi_retained=PHI_F, gamma_retained=GAMMA,
        live_load_resists=True,
    )
    W_old = W + Q_LL * L
    assert old.FOS_sliding == pytest.approx(
        W_old * math.tan(math.radians(PHI_FD)) / Pa, rel=0.02)    # ~2.27
    assert old.sigma_v == res.sigma_v          # bearing counts it either way


def test_v009_lrfd_external_stability_high_level_path():
    """PASS (v5.3): the NEW high-level `check_external_stability_lrfd` computes the
    full GEC-11 Example E4 external-stability CDR set — sliding, eccentricity,
    bearing — directly from the wall/soil inputs with the AASHTO/GEC-11 Strength I
    (max+min) and Service I load-factor combinations built in (no hand-assembled
    factors). Reproduces Tables E4-6.1/6.2/6.3 to the published values."""
    geom = MSEWallGeometry(wall_height=H, reinforcement_length=L,
                           reinforcement_spacing=2.5 * FT, surcharge=Q_LL)
    r = check_external_stability_lrfd(
        geom, gamma_backfill=GAMMA, phi_backfill=PHI_R, phi_foundation=PHI_FD,
        phi_retained=PHI_F, gamma_retained=GAMMA,
        bearing_resistance_strength=10.50 * KSF,
        bearing_resistance_service=7.50 * KSF,
    )

    # Sliding CDRs (Table E4-6.1): Str I max 1.85, min 2.08, critical 1.37
    s = r["sliding"]
    assert s["CDR_strength_max"] == pytest.approx(1.85, abs=0.05)
    assert s["CDR_strength_min"] == pytest.approx(2.08, abs=0.05)
    assert s["CDR_critical"] == pytest.approx(1.37, abs=0.05)
    assert s["CDR_governing"] == pytest.approx(1.37, abs=0.05)   # critical governs
    assert s["passes"]

    # Eccentricity (Table E4-6.2): Str I max eL 2.87 ft, critical 3.87 ft, limit L/4
    e = r["eccentricity"]
    assert e["eL_strength_max_m"] / FT == pytest.approx(2.87, abs=0.05)
    assert e["eL_critical_m"] / FT == pytest.approx(3.87, abs=0.05)
    assert e["e_limit_m"] / FT == pytest.approx(4.50, abs=0.02)   # L/4
    assert e["passes"]                                            # 3.87 < 4.50

    # Bearing (Table E4-6.3): Str I max eL 2.60, B' 12.79, sigma_v 6.70 ksf,
    # CDR 1.57; Service sigma_v 4.66 ksf; critical sigma_v 5.86 (source rounding).
    b = r["bearing"]
    assert b["eL_strength_max_m"] / FT == pytest.approx(2.60, abs=0.05)
    assert b["B_eff_strength_max_m"] / FT == pytest.approx(12.79, abs=0.05)
    assert b["sigma_v_strength_max_kPa"] / KSF == pytest.approx(6.70, rel=0.02)
    assert b["CDR_strength"] == pytest.approx(1.57, abs=0.05)
    assert b["sigma_v_service_kPa"] / KSF == pytest.approx(4.66, rel=0.02)
    # critical bearing: 5.75 (module) vs 5.86 (published) — within the source's
    # own "consistent-values" rounding note (+/-2.5%)
    assert b["sigma_v_critical_kPa"] / KSF == pytest.approx(5.86, rel=0.025)
    assert b["passes"]

    assert r["passes"]


def test_v009_analyze_mse_wall_lrfd_flag_attaches_cdrs():
    """PASS (v5.3): `analyze_mse_wall(lrfd_external=True)` runs BOTH the ASD FOS
    path (unchanged default) AND the LRFD external path, attaching the CDR set as
    `result.external_lrfd`. Default (flag off) leaves external_lrfd = None."""
    geom = MSEWallGeometry(wall_height=H, reinforcement_length=L,
                           reinforcement_spacing=2.5 * FT, surcharge=Q_LL)
    rebar = Reinforcement(name="bar mat", type="metallic_grid",
                          Tallowable=200.0, Fy=448000.0, thickness=0.0095,
                          **_E4_MAT)
    # default: LRFD not requested -> external_lrfd absent, ASD FOS unchanged
    res_default = analyze_mse_wall(
        geom, gamma_backfill=GAMMA, phi_backfill=PHI_R, reinforcement=rebar,
        gamma_foundation=GAMMA, phi_foundation=PHI_FD,
        q_allowable=10.50 * KSF, phi_retained=PHI_F, gamma_retained=GAMMA,
    )
    assert res_default.external_lrfd is None
    assert "external_lrfd" not in res_default.to_dict()

    # opt-in: LRFD CDRs attached, matching the standalone high-level path
    res = analyze_mse_wall(
        geom, gamma_backfill=GAMMA, phi_backfill=PHI_R, reinforcement=rebar,
        gamma_foundation=GAMMA, phi_foundation=PHI_FD,
        q_allowable=10.50 * KSF, phi_retained=PHI_F, gamma_retained=GAMMA,
        lrfd_external=True, bearing_resistance_strength=10.50 * KSF,
        bearing_resistance_service=7.50 * KSF,
    )
    lrfd = res.external_lrfd
    assert lrfd is not None
    assert lrfd["sliding"]["CDR_governing"] == pytest.approx(1.37, abs=0.05)
    assert lrfd["bearing"]["CDR_strength"] == pytest.approx(1.57, abs=0.05)
    assert res.to_dict()["external_lrfd"]["sliding"]["CDR_strength_max"] == \
        pytest.approx(1.85, abs=0.05)
    # the ASD FOS path is untouched by the flag
    assert res.FOS_sliding == pytest.approx(res_default.FOS_sliding, rel=1e-9)


# ── V-010 : MSE internal stability — bar-mat Kr/Tmax/pullout ─────────────────

def _kr_ka_barmat(z_ft):
    """Bar-mat Kr/Ka via the BUILT-IN `Kr_Ka_ratio` ("metallic_grid"):
    2.5 at Z=0 to 1.2 at Z>=20 ft (GEC-11 Fig E4-5). The curve now lives in
    the module (v5.2 Q4); this wrapper just converts ft -> m."""
    return Kr_Ka_ratio(z_ft * FT, "metallic_grid")


def _f_star_barmat(z_ft, t_over_St):
    """Bar-mat F* via the BUILT-IN `F_star_metallic` ("metallic_grid"):
    20(t/St) at Z=0 to 10(t/St) at Z>=20 ft (GEC-11 Fig E4-5). v5.2 Q4."""
    return F_star_metallic(z_ft * FT, PHI_R, "metallic_grid", t_over_St)


def test_v010_module_curves_now_include_barmat_branch():
    """PASS (coverage closed, v5.2 Q4): the module's built-in `Kr_Ka_ratio` and
    `F_star_metallic` now carry BOTH the ribbed-metallic-STRIP curves (Kr/Ka
    1.7->1.2; F* 1.2 + log Cu->tan-phi) AND the steel-bar-mat curves this
    example uses (Kr/Ka 2.5->1.2; F* 20(t/St)->10(t/St)), selected by
    reinforcement_type. The bar-mat branch matches GEC-11 Fig E4-5."""
    # ribbed strip (default / "metallic")
    assert Kr_Ka_ratio(0.0, "metallic") == pytest.approx(1.7)      # strip top
    assert Kr_Ka_ratio(0.0) == pytest.approx(1.7)                  # default == strip
    # GEC-11 Vol I Eq. 3-4: Cu 4 when unknown (F* 1.8), 2.0 at most (GM-246)
    assert F_star_metallic(0.0, PHI_R) == pytest.approx(1.8, abs=0.003)
    assert F_star_metallic(0.0, PHI_R, Cu=7.0) == pytest.approx(2.0)
    # steel bar mat / welded grid — the new branch
    assert Kr_Ka_ratio(0.0, "metallic_grid") == pytest.approx(2.5)   # bar-mat top
    assert Kr_Ka_ratio(0.0, "bar_mat") == pytest.approx(2.5)
    assert Kr_Ka_ratio(20.0 * FT, "metallic_grid") == pytest.approx(1.2)
    assert Kr_Ka_ratio(6.096, "metallic") == pytest.approx(1.2)    # strip -> 1.2 @6m
    assert F_star_metallic(0.0, PHI_R, "metallic_grid", 0.374 / 6.0) == \
        pytest.approx(1.246, abs=0.001)
    assert F_star_metallic(20.0 * FT, PHI_R, "metallic_grid", 0.374 / 6.0) == \
        pytest.approx(0.623, abs=0.001)


def test_v010_Tmax_primitive_matches_barmat_levels():
    """PASS (primitive): `Tmax_at_level` reproduces the Table E4-7.4 bar-mat
    Tmax and horizontal stress sigma_H at Levels 1/4/7/10 to <=3% when fed the
    bar-mat Kr/Ka and the EV load factor (1.35 on both soil and surcharge per
    Table E4-7.2). sigma_H follows the example's average-over-tributary-bounds
    method (Kr at z- and z+ averaged)."""
    Ka = rankine_Ka(PHI_R)
    wp = 5.0 * FT
    gamma_EV = 1.35
    # (level, Z- ft, Z+ ft, Svt ft, sigma_H ksf published, Tmax k/panel published)
    levels = [
        (1, 0.00, 3.12, 3.12, 0.40, 6.25),
        (4, 8.12, 10.62, 2.50, 1.02, 12.77),
        (7, 15.62, 18.12, 2.50, 1.26, 15.71),
        (10, 23.12, 25.64, 2.52, 1.51, 19.05),
    ]
    for lvl, zm, zp, svt_ft, sH_pub, T_pub in levels:
        def sigma_h_at(z_ft):
            KrKa = _kr_ka_barmat(z_ft)
            sv_soil = GAMMA * (z_ft * FT)
            # soil + surcharge, both with gamma_P-EV = 1.35 (Table E4-7.2)
            return (KrKa * Ka * sv_soil + KrKa * Ka * Q_LL) * gamma_EV
        sigma_h = 0.5 * (sigma_h_at(zm) + sigma_h_at(zp))
        # Tmax via the module primitive: T = sigma_h * Atrib
        Tmax = Tmax_at_level(
            z=0.0, gamma_backfill=0.0, Ka=1.0, Kr_Ka=1.0,
            Sv=(svt_ft * FT) * wp, q_surcharge=sigma_h,   # feed sigma_h as the stress
        )
        assert sigma_h / KSF == pytest.approx(sH_pub, abs=0.03), (
            f"L{lvl} sigma_H {sigma_h / KSF:.3f} vs {sH_pub}")
        assert Tmax / KIP == pytest.approx(T_pub, rel=0.03), (
            f"L{lvl} Tmax {Tmax / KIP:.2f} vs {T_pub}")


def test_v010_Tmax_at_level_pointvalue_level4():
    """PASS (primitive, direct call): driving `Tmax_at_level` with the example's
    point-value Level-4 inputs (Z=9.37 ft, Kr/Ka=1.891, Atrib=12.5 ft2,
    gamma=125 pcf, q=0.25 ksf) and the 1.35 EV factor reproduces sigma_H and
    Tmax (12.8 k/panel vs published 12.77)."""
    Ka = rankine_Ka(PHI_R)
    z_ft = 9.37
    KrKa = _kr_ka_barmat(z_ft)
    Atrib = (2.50 * FT) * (5.0 * FT)
    T_unfactored = Tmax_at_level(
        z=z_ft * FT, gamma_backfill=GAMMA, Ka=Ka, Kr_Ka=KrKa,
        Sv=Atrib, q_surcharge=Q_LL,
    )
    Tmax = 1.35 * T_unfactored      # apply the EV load factor
    assert Tmax / KIP == pytest.approx(12.77, rel=0.03)


def test_v010_pullout_resistance_primitive_level4():
    """PASS (primitive): `pullout_resistance` with C=2 (two grid surfaces, the
    example's '2b') reproduces the Level-4 nominal Pr = 23.06 k/ft and the
    factored phi_p*Pr = 20.75 k/ft (phi_pullout=0.90). F*=0.955 (bar-mat interp),
    Le = L - 0.3H = 10.31 ft, unfactored soil sigma_v (gamma_P-EV=1.0)."""
    z_ft = 9.37
    F_star = _f_star_barmat(z_ft, 0.374 / 6.0)
    assert F_star == pytest.approx(0.955, abs=0.005)
    Le = (18.0 - 0.3 * 25.64) * FT
    assert Le / FT == pytest.approx(10.31, abs=0.02)
    Pr = pullout_resistance(
        z=z_ft * FT, gamma_backfill=GAMMA, Le=Le, F_star=F_star,
        alpha_pullout=1.0, C=2.0, q_surcharge=0.0, Rc=1.0,   # unfactored soil stress
    )
    assert Pr / KFT == pytest.approx(23.06, rel=0.02)         # nominal
    assert (0.90 * Pr) / KFT == pytest.approx(20.75, rel=0.02)  # factored


def test_v010_pullout_Le_lengthens_below_midheight():
    """The Level-10 effective length lengthens below H/2 (Z>H/2 branch):
    Le = 17.24 ft (published) vs the upper-zone 10.31 ft. Confirms the
    coherent-gravity La taper used for inextensible bar mats."""
    # GEC-11: for Z > H/2, Le = L - 0.6*(H - Z)  (taper to toe).  Z=24.37 ft.
    Z_ft = 24.37
    Le_ft = 18.0 - 0.6 * (25.64 - Z_ft)
    assert Le_ft == pytest.approx(17.24, abs=0.05)


def test_v010_highlevel_path_barmat_curves_match_published():
    """PASS (high-level path, v5.2 Q4): driving `check_internal_stability` with
    a steel bar-mat reinforcement (WELDED_WIRE_GRID_W11) makes the BUILT-IN
    `Kr_Ka_ratio`/`F_star_metallic` produce the bar-mat curves automatically —
    no hand-fed coefficients. At the published E4 level depths the high-level
    Kr/Ka (2.5->1.2) and F* (20 t/St->10 t/St, St=6 in) match GEC-11 Fig E4-5:
    Kr/Ka(L4 z=9.37 ft)=1.891, F*(L1)=1.246, F*(L4)=0.955."""
    import dataclasses
    from retaining_walls.mse import check_internal_stability
    from retaining_walls.reinforcement import WELDED_WIRE_GRID_W11

    # Build a geometry at the example's 10 bar-mat levels (Z=1.87..24.37 ft) by
    # asking for ~2.5 ft uniform spacing; the curves are depth-driven so we
    # check Kr/F* at the produced level depths against the built-in bar-mat law.
    geom = MSEWallGeometry(wall_height=H, reinforcement_length=L,
                           reinforcement_spacing=2.5 * FT, surcharge=Q_LL)
    rows = check_internal_stability(geom, gamma_backfill=GAMMA,
                                    phi_backfill=PHI_R,
                                    reinforcement=dataclasses.replace(
                                        WELDED_WIRE_GRID_W11, **_E4_MAT))
    tSt = WELDED_WIRE_GRID_W11.t_over_St
    assert tSt == pytest.approx(0.374 / 6.0, abs=0.002)   # W11 t / 6-in St
    for r in rows:
        z = r["depth_m"]
        # bar-mat Kr/Ka 2.5 -> 1.2 over 0-20 ft (NOT the strip 1.7)
        assert r["Kr_Ka"] == pytest.approx(
            round(Kr_Ka_ratio(z, "metallic_grid"), 3), abs=0.002)
        # bar-mat F* 20(t/St) -> 10(t/St) (NOT the strip 2.0->tan-phi)
        assert r["F_star"] == pytest.approx(
            round(F_star_metallic(z, PHI_R, "metallic_grid", tSt), 3),
            abs=0.002)
    # the shallowest level is clearly in the bar-mat family (Kr/Ka > 2)
    assert rows[0]["Kr_Ka"] > 2.0
    # F* at the surface-most level approaches the published 20*t/St = 1.246
    assert F_star_metallic(0.0, PHI_R, "metallic_grid", tSt) == pytest.approx(
        1.246, abs=0.001)


# ── V-010 : E4's tension (rupture) check, GM-262 ─────────────────────────────

IN = 0.0254
KSI = 6894.757
#: E4 p. E4-14: wire diameters after 75 years, D75 = D - (0.47 mil/year/
#: side)(59 years)(2 sides): W11 0.374 -> 0.318 in., W15 0.437 -> 0.381 in.
#: (the page also lists "Steel loss = 0.16 mil/year/side", the zinc's rate;
#: the calculation takes 0.47, AASHTO's carbon steel rate).
E4_D75_IN = {"W11": 0.318, "W15": 0.381}
#: GEC-11 Vol II Table E4-7.4 (p. E4-18), Strength I (max): Level: (Z ft,
#: Tmax k per 5 ft panel (factored), phi_s Tn k per wire, Nt, Ng, wire).
E4_TABLE_7_4 = {
    1: (1.87, 6.25, 3.38, 1.9, 4, "W11"), 2: (4.37, 8.36, 3.38, 2.5, 3, "W11"),
    3: (6.87, 10.80, 3.38, 3.2, 4, "W11"), 4: (9.37, 12.77, 3.38, 3.8, 4, "W11"),
    5: (11.87, 14.26, 4.84, 2.9, 4, "W15"), 6: (14.37, 15.23, 4.84, 3.1, 4, "W15"),
    7: (16.87, 15.71, 4.84, 3.2, 4, "W15"), 8: (19.37, 16.03, 4.84, 3.3, 4, "W15"),
    9: (21.87, 17.10, 4.84, 3.5, 4, "W15"), 10: (24.37, 19.05, 4.84, 3.9, 4, "W15")}


def test_v010_tensile_resistance_per_wire():
    """PASS (GM-262): E4 Step 7.4 (p. E4-15), Grade 65 (Fy = 65 ksi):
    W11 Tn = 65 (0.0795 in.2) = 5.17 k, Tr = 0.65 Tn = 3.36 k; W15 Tn = 7.42
    k, Tr = 4.82 k (phi_t 0.65, Table E4-5.2, p. E4-6), from the module's
    Eq. 3-11a / Fig. 3-5 expression with one wire (b = 1 makes it Ptal,
    Eq. 3-11b). Table E4-7.4's spreadsheet prints 3.38 and 4.84 (its note,
    p. E4-18: hand and spreadsheet values differ slightly)."""
    from retaining_walls.reinforcement import steel_grid_Tal
    for wire, Tn_pub, Tr_pub, Tr_table in (("W11", 5.17, 3.36, 3.38),
                                           ("W15", 7.42, 4.82, 4.84)):
        Tn = steel_grid_Tal(1, E4_D75_IN[wire] * IN, 65.0 * KSI, 1.0) / KIP
        assert Tn == pytest.approx(Tn_pub, rel=0.002), wire
        assert 0.65 * Tn == pytest.approx(Tr_pub, rel=0.002), wire
        assert 0.65 * Tn == pytest.approx(Tr_table, rel=0.008), wire


@pytest.mark.parametrize("level", sorted(E4_TABLE_7_4))
def test_v010_tension_check_by_level(level):
    """PASS (GM-262): Table E4-7.4 (p. E4-18) at every level: Nt = Tmax /
    (phi_s Tn) wires a 5 ft panel, and the mat chosen, Ng wires, carries the
    factored Tmax (Ng phi_s Tn >= Tmax). phi_s Tn from the module; the
    printed Nt are to one decimal from 3.38 / 4.84, so within 0.07."""
    from retaining_walls.reinforcement import steel_grid_Tal
    _, Tmax, _, Nt, Ng, wire = E4_TABLE_7_4[level]
    Tr = 0.65 * steel_grid_Tal(1, E4_D75_IN[wire] * IN, 65.0 * KSI, 1.0) / KIP
    assert Tmax / Tr == pytest.approx(Nt, abs=0.07)
    assert Ng * Tr >= Tmax


def test_v010_tension_level4_through_the_built_in_mat():
    """PASS (GM-262): E4 Level 4 (Z = 9.37 ft) needs Nt = Tmax / Tr = 12.75
    / 3.36 = 3.8 wires a 5 ft panel and takes "4W11 + W11x0.5'" (pp. E4-17,
    E4-18), so its four wires carry 4 x 3.36 / 12.75 = 1.05 times the
    factored Tmax. The built-in mat (WELDED_WIRE_GRID_W11,
    this mat since GM-262) at Sh 5 ft: Tal x Rc = 4 x 0.48 Fy Ac / Sh, the
    ASD allowable (GEC-11 Vol I p. 4-77; 0.65 / 1.35 = 0.48), against the
    unfactored Tmax at the level (the point value of
    test_v010_Tmax_at_level_pointvalue_level4): FOS_rupture 1.044. Before
    GM-262 the mat's Tal was 32.0 kN/m of grid width: 0.347."""
    import dataclasses
    from retaining_walls.mse import coverage_ratio
    from retaining_walls.reinforcement import WELDED_WIRE_GRID_W11
    mat = dataclasses.replace(WELDED_WIRE_GRID_W11, **_E4_MAT)
    Tal_Rc = mat.Tallowable * coverage_ratio(mat)          # kN/m of wall
    T = Tmax_at_level(z=9.37 * FT, gamma_backfill=GAMMA, Ka=rankine_Ka(PHI_R),
                      Kr_Ka=_kr_ka_barmat(9.37), Sv=2.5 * FT, q_surcharge=Q_LL)
    assert Tal_Rc / T == pytest.approx(4 * 3.36 / 12.75, rel=0.01)
    assert Tal_Rc / T == pytest.approx(1.044, abs=0.001)
    assert 32.0 * coverage_ratio(mat) / T == pytest.approx(0.347, abs=0.001)


# ── V-010 : E4's mats per level, GM-266 ─────────────────────────────────────

#: Table E4-7.4 (p. E4-18), the rest of each row: Level: (F*, Le ft,
#: phi_p Pr k/ft, Bar Mat as printed).
E4_TABLE_7_4_MATS = {
    1: (1.188, 10.31, 5.16, "4W11+W11x0.5'"),
    2: (1.110, 10.31, 11.25, "3W11+W11x0.5'"),
    3: (1.033, 10.31, 16.47, "4W11+W11x0.5'"),
    4: (0.955, 10.31, 20.75, "4W11+W11x0.5'"),
    5: (0.438, 10.31, 12.06, "4W15+W11x1'"),
    6: (0.399, 11.24, 14.50, "4W15+W11x1'"),
    7: (0.360, 12.74, 17.41, "4W15+W11x1'"),
    8: (0.214, 14.24, 13.27, "4W15+W11x1.5'"),
    9: (0.208, 15.74, 16.12, "4W15+W11x1.5'"),
    10: (0.208, 17.24, 19.66, "4W15+W11x1.5'")}


class _E4Levels(MSEWallGeometry):
    """E4's wall with its levels at the printed Z (Table E4-7.4, p. E4-18):
    1.87 ft, then 2.5 ft apart to 24.37 ft. The module's levels sit at
    Sv (i + 1/2), which for Sv 2.5 ft puts the first at 1.25 ft."""

    @property
    def reinforcement_depths(self):
        return [E4_TABLE_7_4[k][0] * FT for k in sorted(E4_TABLE_7_4)]

    @property
    def n_reinforcement_levels(self):
        return len(E4_TABLE_7_4)


def _e4_by_level(reinforcement):
    geom = _E4Levels(wall_height=H, reinforcement_length=L,
                     reinforcement_spacing=2.5 * FT, surcharge=Q_LL)
    return analyze_mse_wall(geom, GAMMA, PHI_R, reinforcement)


def _e4_mats():
    import dataclasses
    from retaining_walls.reinforcement import (
        WELDED_WIRE_GRID_W11, BAR_MAT_E4_3W11, BAR_MAT_E4_4W15_ST1,
        BAR_MAT_E4_4W15_ST1_5)
    mats = {"4W11+W11x0.5'": WELDED_WIRE_GRID_W11,
            "3W11+W11x0.5'": BAR_MAT_E4_3W11,
            "4W15+W11x1'": BAR_MAT_E4_4W15_ST1,
            "4W15+W11x1.5'": BAR_MAT_E4_4W15_ST1_5}
    return [dataclasses.replace(mats[E4_TABLE_7_4_MATS[k][3]],
                                horizontal_spacing=5.0 * FT)
            for k in sorted(E4_TABLE_7_4_MATS)]


def test_v010_e4_mats_per_level():
    """PASS (GM-266): E4 with its mats as Table E4-7.4 prints them (p.
    E4-18), one a level: 4W11 + W11x0.5' at Levels 1, 3 and 4, 3W11 +
    W11x0.5' at Level 2, 4W15 + W11x1' at Levels 5 to 7 and 4W15 +
    W11x1.5' at Levels 8 to 10, the longitudinal wires at Sl = 0.5 ft (p.
    E4-17), W15 0.381 in. and W11 0.318 in. after 75 years (p. E4-14), Fy
    65 ksi and the W11 transverse wire t = 0.374 in. (p. E4-15), each mat
    across a 5 ft panel. Each level takes its own mat's t/St in F*: Levels
    5 to 10 print F* 0.438 to 0.208, the 1 ft and 1.5 ft transverse
    spacings. F*, Le and phi_p Pr = 0.90 Pr (per ft of mat width) match the
    table to 0.3 %. The tension check is the ASD one (Tal = 0.48 Fy Ac,
    GEC-11 Vol I p. 4-77, against the unfactored Tmax), so FOS_rupture is
    set against Ng phi_s Tn / Tmax, the LRFD ratio Table E4-7.4 implies.
    They match to 2 % at Levels 2 to 10, and Level 10 gives 1.018 against
    4 x 4.84 / 19.05 = 1.016. Level 1 is 13 % over, from the top level's
    tributary depth: E4 gives it the wall from the top to midway to Level
    2, Svt = 3.12 ft (Table E4-7.1, p. E4-11), with sigma_H averaged over
    it (p. E4-13); the module gives every level Sv = 2.5 ft and sigma_H at
    the level (3.12 / 2.5 x 0.910 = 1.136; BACKLOG GM-273)."""
    res = _e4_by_level(_e4_mats())
    rows = res.internal_results
    assert [r["reinforcement"] for r in rows] == [m.name for m in _e4_mats()]
    assert "W15" in rows[9]["reinforcement"] and "x1.5'" in rows[9][
        "reinforcement"]
    for level, r in zip(sorted(E4_TABLE_7_4_MATS), rows):
        F, Le_ft, phiPr, _ = E4_TABLE_7_4_MATS[level]
        _, Tmax, phiTn, _, Ng, _ = E4_TABLE_7_4[level]
        assert r["F_star"] == pytest.approx(F, abs=0.001), level
        assert r["Le_m"] / FT == pytest.approx(Le_ft, abs=0.03), level
        assert 0.90 * r["Pr_kN_per_m"] / r["Rc"] / KFT == pytest.approx(
            phiPr, rel=0.003), level
        if level > 1:
            assert r["FOS_rupture"] == pytest.approx(
                Ng * phiTn / Tmax, rel=0.02), level
    assert rows[9]["FOS_rupture"] == pytest.approx(1.018, abs=0.001)
    assert 4 * 4.84 / 19.05 == pytest.approx(1.016, abs=0.001)
    assert rows[0]["FOS_rupture"] / (4 * 3.38 / 6.25) == pytest.approx(
        1.127, abs=0.002)


def test_v010_e4_level10_needs_its_w15_mat():
    """PASS (GM-266): at Level 10 (Z = 24.37 ft) E4's 4W15 + W11x1.5' mat
    carries the tension, FOS_rupture 1.018, and the 4W11 mat of Levels 1
    to 4 would not: 0.709, its four W11 wires (4 x 3.38 = 13.5 k a panel)
    against the 19.05 k factored Tmax (Table E4-7.4, p. E4-18): 0.71. With
    one reinforcement for the wall (welded_wire_grid_w11) the rupture
    check fails from Level 5 down, where E4 changes to W15."""
    import dataclasses
    from retaining_walls.reinforcement import WELDED_WIRE_GRID_W11
    one = _e4_by_level(dataclasses.replace(WELDED_WIRE_GRID_W11,
                                           horizontal_spacing=5.0 * FT))
    rows = one.internal_results
    assert rows[9]["FOS_rupture"] == pytest.approx(0.709, abs=0.001)
    assert 4 * 3.38 / 19.05 == pytest.approx(0.71, abs=0.001)
    assert [r["FOS_rupture"] >= 1.0 for r in rows] == [True] * 4 + [False] * 6
    per_level = _e4_by_level(_e4_mats()).internal_results
    assert all(r["FOS_rupture"] >= 1.0 for r in per_level)


# ── V-011 : Mononobe-Okabe KAE + seismic sliding CDR (Example E7) ────────────
#   REGRESSION ANCHOR for the M-O delta-handling / battered-wall fix.

def test_v011_mononobe_okabe_KAE_regression_anchor():
    """PASS (regression anchor): the M-O active coefficient for the E7 case
    (vertical wall, level backfill, phi=30, delta=30, kh=kmax=0.206, kv=0)
    gives KAE = 0.4782 vs the published 0.4785 — -0.06%, well inside +/-2%.
    delta=phi=30 with kv=0 is the case the battered-wall M-O fix targeted; this
    pins the correct value so a future M-O sign/units regression is caught."""
    KAE = mononobe_okabe_KAE(phi_deg=30.0, delta_deg=30.0, kh=0.206,
                             kv=0.0, beta_deg=0.0, i_deg=0.0)
    assert KAE == pytest.approx(0.4785, rel=0.02)
    # tighter pin of the actual value (documentation):
    assert KAE == pytest.approx(0.4782, abs=0.0005)


def test_v011_PAE_resultant():
    """PASS: total seismic active thrust PAE = 0.5*gamma*h^2*KAE = 19.65 k/ft
    (h = 25.64 ft pressure-plane height), inclined at delta=30. Exact match."""
    KAE = mononobe_okabe_KAE(30.0, 30.0, 0.206, 0.0, 0.0, 0.0)
    PAE = 0.5 * GAMMA * H ** 2 * KAE       # kN/m
    assert PAE / KFT == pytest.approx(19.65, rel=0.01)


def test_v011_seismic_sliding_CDR():
    """PASS: the full Step-8 seismic sliding chain reproduces the example exactly
    on the module KAE. W=57.68 k/ft, kav=0.211 -> PIR=6.09; THF = PAE*cos30 +
    PIR + 0.5*qLS*H*KAE = 17.02+6.09+1.53 = 24.64; V = W + PAE*sin30 = 67.52;
    R = V*tan30 = 38.98; CDR = R/THF = 1.58 (phi_sliding=1.0)."""
    KAE = mononobe_okabe_KAE(30.0, 30.0, 0.206, 0.0, 0.0, 0.0)
    PAE_kft = (0.5 * GAMMA * H ** 2 * KAE) / KFT     # back to k/ft
    H_ft = 25.64
    qLS = 0.25                                        # ksf
    W = 25.64 * 18.0 * 0.125                          # k/ft
    kav = 0.211
    PIR = 0.5 * kav * W
    assert W == pytest.approx(57.68, abs=0.02)
    assert PIR == pytest.approx(6.09, abs=0.02)

    THF = (PAE_kft * math.cos(math.radians(30.0)) + PIR
           + 0.5 * qLS * H_ft * KAE)
    assert PAE_kft * math.cos(math.radians(30.0)) == pytest.approx(17.02, abs=0.05)
    assert 0.5 * qLS * H_ft * KAE == pytest.approx(1.53, abs=0.03)
    assert THF == pytest.approx(24.64, abs=0.05)

    V = W + PAE_kft * math.sin(math.radians(30.0))
    assert V == pytest.approx(67.52, abs=0.05)
    R = V * math.tan(math.radians(PHI_FD))            # phi_fd = 30
    assert R == pytest.approx(38.98, abs=0.05)

    CDR = R / THF
    assert CDR == pytest.approx(1.58, abs=0.03)

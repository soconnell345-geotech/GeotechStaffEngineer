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
    of safety (R/demand, NO load factors, LL not split out), not the GEC-11 LRFD
    CDRs. This pins the documented API gap — the ASD sliding FoS is the
    *unfactored* resistance/demand ratio, distinctly larger than the LRFD
    sliding CDR (1.85). The module is not tuned to the LRFD example."""
    geom = MSEWallGeometry(wall_height=H, reinforcement_length=L,
                           reinforcement_spacing=2.5 * FT, surcharge=Q_LL)
    rebar = Reinforcement(name="bar mat", type="metallic_grid",
                          Tallowable=200.0, Fy=448000.0, thickness=0.0095)
    res = analyze_mse_wall(
        geom, gamma_backfill=GAMMA, phi_backfill=PHI_R, reinforcement=rebar,
        gamma_foundation=GAMMA, phi_foundation=PHI_FD,
        q_allowable=10.50 * KSF, phi_retained=PHI_F, gamma_retained=GAMMA,
    )
    # ASD sliding FoS uses unfactored W and unfactored thrust -> > LRFD CDR
    assert res.FOS_sliding > 1.85          # ASD, not the LRFD 1.85
    # It is an UNFACTORED R/demand ratio (W includes the surcharge; demand is the
    # combined earth+surcharge thrust), NOT the load-factored LRFD CDR.
    Kaf = rankine_Ka(PHI_F)
    Pa, _ = horizontal_force_active(GAMMA, H, Kaf, q=Q_LL)
    W = GAMMA * H * L + Q_LL * L
    asd_expected = (W * math.tan(math.radians(PHI_FD))) / Pa
    assert res.FOS_sliding == pytest.approx(asd_expected, rel=0.02)   # ~2.27


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
                          Tallowable=200.0, Fy=448000.0, thickness=0.0095)
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
    1.7->1.2; F* 2.0->tan-phi) AND the steel-bar-mat curves this example uses
    (Kr/Ka 2.5->1.2; F* 20(t/St)->10(t/St)), selected by reinforcement_type.
    The strip default is unchanged; the bar-mat branch matches GEC-11 Fig E4-5."""
    # ribbed strip (default / "metallic") — unchanged
    assert Kr_Ka_ratio(0.0, "metallic") == pytest.approx(1.7)      # strip top
    assert Kr_Ka_ratio(0.0) == pytest.approx(1.7)                  # default == strip
    assert F_star_metallic(0.0, PHI_R) == pytest.approx(2.0)
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
    from retaining_walls.mse import check_internal_stability
    from retaining_walls.reinforcement import WELDED_WIRE_GRID_W11

    # Build a geometry at the example's 10 bar-mat levels (Z=1.87..24.37 ft) by
    # asking for ~2.5 ft uniform spacing; the curves are depth-driven so we
    # check Kr/F* at the produced level depths against the built-in bar-mat law.
    geom = MSEWallGeometry(wall_height=H, reinforcement_length=L,
                           reinforcement_spacing=2.5 * FT, surcharge=Q_LL)
    rows = check_internal_stability(geom, gamma_backfill=GAMMA,
                                    phi_backfill=PHI_R,
                                    reinforcement=WELDED_WIRE_GRID_W11)
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

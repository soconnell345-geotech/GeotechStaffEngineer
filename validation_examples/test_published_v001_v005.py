"""Phase E validation — GEC-12 Vol 3 Appendix D driven-pile design example (V-001..V-005).

Source: FHWA GEC-12 Vol 3 (FHWA-NHI-16-064), Appendix D, North & South Abutment
design blocks. See validation_examples/INVENTORY.md entries V-001..V-005 and
RESULTS.md. Extraction artifacts: extracts/g12_north_abut.txt, g12_south.txt,
g12_north_drv_np.txt, plus rendered figures g12v3_p60.png / g12v3_p212.png.

Pile for all entries: HP 12x74 (A = 21.8 in2). H-pile "box" convention — shaft on
box perimeter 2*(d+bf), toe on box area d*bf. The module's make_h_pile("HP12x74")
gives perimeter = 4.058 ft and tip_area = 1.029 ft2, matching the example's stated
~4.06 ft / ~1.03 ft2 box values.

Datum note: the example references depths to the bottom of footing/cap, which is
5 ft below ground surface. The pile head is AT the footing bottom, so the
analysis-module profile is modeled FROM the footing bottom (z=0 at the pile head):
the loose silty sand / medium clay layers are clipped at the footing line and the
water table is shifted up by 5 ft accordingly. (Modeling from the ground surface
instead would integrate ~5 ft of spurious skin friction above the pile head.)

Summary of verdicts (details in each test docstring and RESULTS.md):
- V-001 axial_pile Nordlund (sand): shaft PASS (-8%..+5%); toe CONVENTION — the
  high-level API uses one phi per layer, so it cannot apply the example's separate
  Layer-3 toe phi=40 (design limit); driving the toe function with phi=40 gives -5%.
- V-002 axial_pile alpha (clay): toe (9*cu) PASS; shaft CONVENTION — the module's
  simplified Tomlinson alpha-vs-cu curve gives a lower alpha for stiff clay than the
  example's DrivenPiles Tomlinson curve.
- V-003 wave_equation drivability: N/A (scope) — no Delmag D36-52 in the hammer DB
  and the simplified diesel model can't reproduce a GRLWEAP diesel bearing graph.
- V-004 downdrag neutral plane: PASS — the Fellenius NP construction reproduces
  NP, Qmax, and drag force to <1% when fed the published Table D-6 shaft.
- V-005 Meyerhof (1976) pile-group SPT settlement: PASS — pile_group's new
  meyerhof_group_settlement (SI API, m/kN -> mm) reproduces the published
  1.04 in (50-ft case) to within +/-5%.

Units: example is US customary; modules are SI.
1 ft = 0.3048 m, 1 kip = 4.448 kN, 1 ksf = 47.88 kPa, 1 pcf = 0.157087 kN/m3,
1 kip-ft = 1.356 kN-m.
"""

import math

import numpy as np
import pytest

from axial_pile import (
    make_h_pile, AxialSoilLayer, AxialSoilProfile, AxialPileAnalysis,
)
from axial_pile.nordlund import end_bearing_cohesionless
from axial_pile.tomlinson import alpha_tomlinson
from downdrag import (
    DowndragSoilLayer, DowndragSoilProfile, DowndragAnalysis,
)
from wave_equation.hammer import Hammer, list_hammers
from wave_equation.cushion import make_cushion_from_properties
from wave_equation.drivability import drivability_study

FT = 0.3048
KIP = 4.448
KSF = 47.88
PCF = 0.157087
KIPFT = 1.356


# ── V-001 : Nordlund driven H-pile in sand (North Abutment, Table D-6) ───────

def _v001_profile():
    """V-001 cohesionless profile, modeled FROM the footing bottom (pile head).

    Layers below the footing (5 ft bgs): L1 loose silty sand 0-20 ft (phi 33,
    gamma 105 pcf), L2 medium dense sand 20-45 ft (phi 36, 112 pcf), L3 dense
    gravel 45+ ft (phi 36 shaft, 125 pcf). Water table 15 ft bgs = 10 ft below
    the footing.
    """
    return AxialSoilProfile(layers=[
        AxialSoilLayer(thickness=20 * FT, soil_type="cohesionless",
                       unit_weight=105 * PCF, friction_angle=33,
                       description="L1 loose silty sand"),
        AxialSoilLayer(thickness=25 * FT, soil_type="cohesionless",
                       unit_weight=112 * PCF, friction_angle=36,
                       description="L2 medium dense sand"),
        AxialSoilLayer(thickness=52 * FT, soil_type="cohesionless",
                       unit_weight=125 * PCF, friction_angle=36,
                       description="L3 dense gravel (shaft phi)"),
    ], gwt_depth=10 * FT)


def test_v001_hpile_box_section_convention():
    """make_h_pile('HP12x74') reproduces the example's H-pile box perimeter and
    toe area (the convention that drives both shaft and toe)."""
    pile = make_h_pile("HP12x74")
    assert pile.perimeter / FT == pytest.approx(4.06, abs=0.05)      # ~4.06 ft
    assert pile.tip_area / (FT * FT) == pytest.approx(1.03, abs=0.03)  # ~1.03 ft2


def test_v001_nordlund_shaft_resistance_matches():
    """Nordlund shaft resistance vs Table D-6 (depths below footing).

    PASS within +/-15% at every depth (-8% to +5%). The module's displacement-
    pile Kd fit reproduces the published Nordlund shaft for this phi 33-36
    profile once the footing datum is used.
    """
    pile = make_h_pile("HP12x74")
    soil = _v001_profile()
    published_Rs = {35: 137.5, 50: 250.7, 60: 344.1, 70: 452.5}
    for D_ft, Rs_pub in published_Rs.items():
        r = AxialPileAnalysis(pile=pile, soil=soil,
                              pile_length=D_ft * FT, method="auto").compute()
        assert r.Q_skin / KIP == pytest.approx(Rs_pub, rel=0.15), (
            f"D={D_ft} ft: shaft {r.Q_skin / KIP:.1f} vs published {Rs_pub}"
        )


def test_v001_toe_with_design_limit_phi40_matches():
    """Toe resistance reproduces the Table D-6 Layer-3 plateau (428.1 kips) to
    within -5% when driven with the example's separate Layer-3 toe phi = 40 deg
    (the GEC-12 design-limit toe friction angle).

    The module's q_L cap (Meyerhof Fig 7-15) at phi=40 governs, giving 408.5
    kips — the basis for the published ~428 kip plateau.
    """
    pile = make_h_pile("HP12x74")
    soil = _v001_profile()
    for D_ft in (50, 60, 70):           # depths where the tip is in Layer 3
        sigma_v_tip = soil.effective_stress_at_depth(D_ft * FT)
        Rt = end_bearing_cohesionless(
            40.0, sigma_v_tip, pile.tip_area, D_ft * FT, pile.width) / KIP
        assert Rt == pytest.approx(428.1, rel=0.15), (
            f"D={D_ft} ft: toe(phi=40) {Rt:.1f} vs published plateau 428.1"
        )


def test_v001_toe_single_phi_api_is_a_convention_gap():
    """CONVENTION: the high-level AxialPileAnalysis API uses ONE phi per layer,
    so it applies phi=36 (shaft) at the toe too, giving ~301 kips in Layer 3
    (-30% vs the published 428.1, which uses the separate toe phi=40). This pins
    the documented API limitation — it is NOT tuned away."""
    pile = make_h_pile("HP12x74")
    soil = _v001_profile()
    r = AxialPileAnalysis(pile=pile, soil=soil,
                          pile_length=60 * FT, method="auto").compute()
    Rt_single_phi = r.Q_tip / KIP
    assert Rt_single_phi == pytest.approx(301.0, rel=0.05)   # module, phi=36 toe
    assert abs(Rt_single_phi - 428.1) / 428.1 > 0.15         # genuinely off vs published


# ── V-002 : Tomlinson alpha H-pile in clay (South Abutment, Table D-100) ─────

def _su_at_bgs(bgs_ft):
    """Undrained strength su (ksf) vs depth below ground surface, from the
    Tables D-96/97/98 sample-depth values (linear between samples)."""
    if bgs_ft < 25:
        return float(np.interp(bgs_ft, [1, 6, 11, 16, 21],
                               [0.65, 0.66, 0.68, 0.70, 0.72]))
    elif bgs_ft < 45:
        return float(np.interp(bgs_ft, [26, 31, 36, 41],
                               [1.79, 1.83, 1.93, 2.00]))
    else:
        return float(np.interp(bgs_ft, list(range(46, 97, 5)),
                               [3.11, 3.19, 3.30, 3.36, 3.39, 3.50,
                                3.55, 3.58, 3.60, 3.65, 3.70]))


def _v002_profile():
    """V-002 cohesive profile as 1-ft cohesive sublayers from the cap bottom
    (5 ft bgs), each with its sample-depth su. L1 medium clay (110 pcf), L2
    stiff clay (124 pcf), L3 very stiff clay (129 pcf)."""
    layers = []
    z_cap = 5.0
    while z_cap < 96:
        bgs = z_cap + 0.5
        gamma = 110 if bgs < 25 else (124 if bgs < 45 else 129)
        layers.append(AxialSoilLayer(
            thickness=1 * FT, soil_type="cohesive",
            unit_weight=gamma * PCF, cohesion=_su_at_bgs(bgs) * KSF))
        z_cap += 1
    return AxialSoilProfile(layers=layers, gwt_depth=None)


def test_v002_toe_nine_su_matches():
    """End bearing 9*cu reproduces Table D-100 toe resistance to within +/-3%
    at the depths where the tip sits cleanly inside Layer 3 (D = 70/80/90 ft)."""
    pile = make_h_pile("HP12x74")
    soil = _v002_profile()
    published_Rt = {70: 32.75, 80: 33.03, 90: 33.68}
    for D_ft, Rt_pub in published_Rt.items():
        r = AxialPileAnalysis(pile=pile, soil=soil,
                              pile_length=D_ft * FT, method="auto").compute()
        assert r.Q_tip / KIP == pytest.approx(Rt_pub, rel=0.05), (
            f"D={D_ft} ft: toe {r.Q_tip / KIP:.1f} vs published {Rt_pub}"
        )


def test_v002_alpha_curve_convention_gap():
    """CONVENTION / scope: the module's simplified Tomlinson alpha-vs-cu curve
    matches the example for SOFT (su~0.7 ksf, alpha~0.85) and VERY STIFF
    (su~3.5 ksf, alpha~0.34) clay, but gives a much lower alpha for STIFF clay
    (su~1.9 ksf: module 0.42 vs the example's DrivenPiles ~0.76). That single
    band drives the shaft under-prediction below.
    """
    # soft clay — agrees
    assert alpha_tomlinson(0.70 * KSF, "steel") == pytest.approx(0.85, abs=0.03)
    # very stiff clay — agrees
    assert alpha_tomlinson(3.45 * KSF, "steel") == pytest.approx(0.34, abs=0.03)
    # stiff clay — module is well below the example's ~0.76
    alpha_stiff = alpha_tomlinson(1.90 * KSF, "steel")
    assert alpha_stiff == pytest.approx(0.42, abs=0.03)
    assert (0.76 - alpha_stiff) > 0.15      # genuinely different alpha curve


def test_v002_shaft_underpredicts_documented():
    """Shaft resistance vs Table D-100: the module under-predicts by 18-32%
    (outside +/-15%), driven entirely by the lower stiff-clay alpha. Recorded
    as a CONVENTION gap (alpha-curve choice), module not tuned to the example.
    This test pins the observed deficit so a future alpha-curve change is
    caught."""
    pile = make_h_pile("HP12x74")
    soil = _v002_profile()
    published_Rs = {70: 318.4, 80: 369.3, 90: 420.33}
    for D_ft, Rs_pub in published_Rs.items():
        r = AxialPileAnalysis(pile=pile, soil=soil,
                              pile_length=D_ft * FT, method="auto").compute()
        ratio = (r.Q_skin / KIP) / Rs_pub
        assert 0.68 <= ratio <= 0.85, (
            f"D={D_ft} ft: shaft ratio {ratio:.2f} outside documented band"
        )


# ── V-003 : wave-equation drivability with a Delmag D36-52 diesel ────────────

def test_v003_delmag_d36_52_not_in_hammer_db():
    """N/A (scope), part 1: the wave_equation hammer database has no Delmag
    D36-52 (only the -22/-32 single-energy diesel series), so the published
    GRLWEAP setup cannot be looked up directly."""
    names = list_hammers()
    assert "Delmag D36-52" not in names
    assert not any("36-52" in n or "30-52" in n or "46-52" in n for n in names)


def test_v003_simplified_diesel_does_not_reproduce_bearing_graph():
    """N/A (scope), part 2: a hand-built D36-52 diesel (ram 7.94 kips, rated
    energy 89.3 ft-kips) with GRLWEAP-style Smith params and a generic cushion
    does NOT reproduce the published bearing-graph point. At the published
    nominal driving resistance Rndr = 746 kips the module predicts ~48 blows/ft
    (published 120 bpf — outside +/-25%) and ~48 ksi compression (published
    40.1 ksi — outside +/-10%). The module's diesel hammer is a simple
    energy/efficiency velocity conversion, not a diesel combustion + ram-cycle
    GRLWEAP model. Documented coverage gap; module not modified.
    """
    ram_kN = 7.94 * KIP
    E_kNm = 89.3 * KIPFT
    hammer = Hammer("Delmag D36-52 (hand-built)", ram_weight=ram_kN,
                    stroke=E_kNm / ram_kN, efficiency=0.80,
                    hammer_type="diesel", rated_energy=E_kNm)
    A = 21.8 * 0.0254 ** 2          # HP12x74 steel area, m2
    cushion = make_cushion_from_properties(
        area=A, thickness=0.05, elastic_modulus=2e6, cor=0.80)
    # Layer-3 Smith params: quake 0.10 in, shaft damping 0.05 s/ft, toe 0.15 s/ft
    res = drivability_study(
        hammer, cushion, pile_area=A, pile_E=200e6, pile_unit_weight=77.0,
        depths=[61 * FT], R_at_depth=[746 * KIP], skin_fractions=[0.6],
        quake_side=0.10 * 0.0254, quake_toe=0.10 * 0.0254,
        damping_side=0.05 / FT, damping_toe=0.15 / FT, helmet_weight=5.0)
    pt = res.points[0]
    blows_per_ft = pt.blow_count * FT
    comp_ksi = pt.max_comp_stress / 6895.0
    # Both the published blow count and stress are OUT of the inventory tolerance
    assert abs(blows_per_ft - 120.0) / 120.0 > 0.25      # bpf not within +/-25%
    assert abs(comp_ksi - 40.1) / 40.1 > 0.10            # stress not within +/-10%


# ── V-004 : downdrag neutral-plane location and drag force (Block 17) ────────

def _v004_calibrated_betas():
    """Per-layer beta overrides that make the downdrag module's shaft match the
    published Nordlund shaft distribution (Table D-6) at the layer boundaries,
    so the NP construction is validated decoupled from the shaft method (the
    inventory permits taking the resistance distribution from Table D-6).

    Table D-6 shaft below the footing: 51.6 kips at D=20 ft (end L1), 209.8 at
    D=45 (end L2), 344.1 at D=60. Layer shares: L1=51.6, L2=158.2, L3=134.3.
    """
    pile_perim = 4.058 * FT
    tip_area_box = 1.0294 * FT * FT

    def build(b1, b2, b3):
        return DowndragSoilProfile(layers=[
            DowndragSoilLayer(thickness=20 * FT, soil_type="cohesionless",
                              unit_weight=105 * PCF, phi=33, beta=b1),
            DowndragSoilLayer(thickness=25 * FT, soil_type="cohesionless",
                              unit_weight=112 * PCF, phi=36, beta=b2),
            DowndragSoilLayer(thickness=47 * FT, soil_type="cohesionless",
                              unit_weight=125 * PCF, phi=36, beta=b3),
        ], gwt_depth=10 * FT)

    def shaft_in_range(soil, z0, z1):
        a = DowndragAnalysis(soil=soil, pile_length=z1, pile_diameter=0.3086,
                             pile_perimeter=pile_perim, pile_area=tip_area_box,
                             pile_unit_weight=20.0, Q_dead=201 * KIP, Nt=1.0)
        r = a.compute()
        z = r.z
        dz = z[1] - z[0]
        mask = (z >= z0) & (z < z1)
        return float(np.sum(r.unit_skin_friction[mask] * pile_perim * dz) / KIP)

    def calibrate(target, z0, z1, idx, fixed):
        lo, hi = 0.05, 2.0
        mid = 0.5 * (lo + hi)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            bs = list(fixed)
            bs[idx] = mid
            if shaft_in_range(build(*bs), z0, z1) < target:
                lo = mid
            else:
                hi = mid
        return mid

    b1 = calibrate(51.6, 0, 20 * FT, 0, [0.3, 0.3, 0.3])
    b2 = calibrate(158.2, 20 * FT, 45 * FT, 1, [b1, 0.3, 0.3])
    b3 = calibrate(134.3, 45 * FT, 60 * FT, 2, [b1, b2, 0.3])
    return build, b1, b2, b3


def test_v004_neutral_plane_and_drag_force():
    """PASS: the Fellenius neutral-plane construction reproduces the published
    NP, max axial force, and drag force.

    HP12x74 driven 60 ft below the footing; sustained load Q = 201 kips; toe
    fully mobilized at 428.1 kips. Published: NP 54 ft below pile head (6 ft
    above the toe), Qmax = 486 kips, DF = 285 kips.
    """
    build, b1, b2, b3 = _v004_calibrated_betas()
    soil = build(b1, b2, b3)
    tip_area_box = 1.0294 * FT * FT
    pile_perim = 4.058 * FT
    L = 60 * FT

    # Nt set so the toe is fully mobilized at 428.1 kips (100% toe mobilization)
    sigma_v_tip = soil.effective_stress_at_depth(L)
    Nt = 428.1 * KIP / (sigma_v_tip * tip_area_box)

    r = DowndragAnalysis(
        soil=soil, pile_length=L, pile_diameter=0.3086,
        pile_perimeter=pile_perim, pile_area=tip_area_box,
        pile_unit_weight=20.0, Q_dead=201 * KIP, Nt=Nt).compute()

    NP_below_head_ft = r.neutral_plane_depth / FT     # z=0 is the pile head
    Qmax_kips = r.max_pile_load / KIP
    DF_kips = (r.max_pile_load - 201 * KIP) / KIP

    assert r.toe_resistance / KIP == pytest.approx(428.1, rel=0.01)
    assert NP_below_head_ft == pytest.approx(54.0, abs=3.0)   # +/-3 ft
    assert Qmax_kips == pytest.approx(486.0, rel=0.10)        # +/-10%
    assert DF_kips == pytest.approx(285.0, rel=0.10)          # +/-10%


# ── V-005 : Meyerhof (1976) pile-group SPT settlement (Block 15, Table D-23) ─

def test_v005_meyerhof_group_settlement_matches_published():
    """PASS: pile_group.meyerhof_group_settlement (Meyerhof 1976 SPT method)
    reproduces the published Table D-23 50-ft case to within +/-5%.

    The method packages S[in] = 4*pf[ksf]*If*sqrt(B[ft])/N160 behind an SI
    public API (B in m, load in kN, settlement returned in mm), converting
    internally. Inputs (50-ft penetration row): B=5 ft, Z=41 ft, Q=1540 kips
    (unfactored permanent), DB=5 ft into the bearing stratum, N160=59 ->
    published S = 1.04 in (~26.5 mm).
    """
    from pile_group import meyerhof_group_settlement

    # SI inputs (the published US values converted via the module's factors).
    B_m = 5.0 * FT
    Z_m = 41.0 * FT
    Q_kN = 1540.0 * KIP
    DB_m = 5.0 * FT

    r = meyerhof_group_settlement(
        group_width=B_m, group_length=Z_m, N160=59,
        load_kN=Q_kN, embedment_DB=DB_m,
    )

    # Net pressure and influence factor reproduce the published hand calc.
    assert r["pf_ksf"] == pytest.approx(7.512, abs=0.01)
    assert r["pf_kPa"] == pytest.approx(7.512 * KSF, rel=0.01)
    assert r["influence_factor"] == pytest.approx(0.9167, abs=0.001)

    # Published settlement: 1.04 in ~ 26.5 mm, +/-5%.
    assert r["settlement_in"] == pytest.approx(1.04, rel=0.05)
    assert r["settlement_mm"] == pytest.approx(26.5, rel=0.05)

    # Cross-check the SI public-API result against the raw US-form hand calc
    # (pins the internal unit conversion).
    B_ft, Z_ft, Q_kips, DB_ft = 5.0, 41.0, 1540.0, 5.0
    pf_ksf = Q_kips / (B_ft * Z_ft)
    If = 1.0 - (2.0 / 3.0 * DB_ft) / (8.0 * B_ft)
    S_in_hand = 4.0 * pf_ksf * If * math.sqrt(B_ft) / 59
    assert r["settlement_in"] == pytest.approx(S_in_hand, rel=1e-4)

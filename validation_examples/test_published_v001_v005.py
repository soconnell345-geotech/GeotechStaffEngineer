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
5 ft below ground surface. Since GM-121 (2026-10-18) the profiles are modeled
FROM THE GROUND SURFACE with the pile head at the footing (head_depth = 5 ft):
the shaft starts at the head, sigma' keeps the 5 ft of soil above it (Table
D-6's toe at 0.01 ft below the footing, 16.4 kips, needs it) and Fig. 7-17's
D runs from the ground surface (GEC-12 p. 246). Before GM-121 the profiles
started at the footing with that overburden removed.

Summary of verdicts (details in each test docstring and RESULTS.md):
- V-001 axial_pile Nordlund (sand): shaft PASS (+1.1 %..+1.2 % since GM-121's
  charts: Fig. 7-9, Tables 7-6/7-7, Fig. 7-14); toe PASS with the example's
  separate Layer-3 toe phi=40 when sigma'p is not limited (429.3 kips, +0.3 %
  vs the 428.1 plateau). With GEC-12 Step 8's 3 ksf limit on sigma'p (the
  module's default since GM-121) the toe is 371.5 kips (-13 %): the example
  (DrivenPiles) did not apply Step 8. Below the plateau the toe follows Table
  D-6 within 0.7 % at phi 33 and 40 and 2.3 % at phi 36 (GM-282: N'q read
  again off Fig. 7-16; it was 2-5 % low).
- V-002 axial_pile alpha (clay): toe (9*cu) PASS; shaft PASS within 1 % with the
  example's two Fig. 7-17 choices (smooth steel curves for the H-pile, Ca held at
  the chart's end); with the module's defaults (corrugated curve for H-piles,
  GEC-12 p. 247; Ca held at the chart's end) +13 %..+14 %, and with API alpha
  past the chart +25 %..+53 %, documented CONVENTION differences.
- V-003 wave_equation drivability: N/A (scope) — no Delmag D36-52 in the hammer DB
  and the simplified diesel model can't reproduce a GRLWEAP diesel bearing graph.
- V-004 downdrag neutral plane: PASS — the neutral plane at 100% toe
  mobilization reproduces NP, Qmax, and drag force to <2% when fed the
  published Table D-6 shaft (the pile weight in the load curve is the
  difference); with the pile weight left out, as the example does, to
  0.0 ft / +0.1% / +0.2% (since GM-01, 2026-10-14).
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
from geotech_common.numpy_compat import trapezoid
from wave_equation.hammer import Hammer, list_hammers
from wave_equation.cushion import make_cushion_from_properties
from wave_equation.drivability import drivability_study

FT = 0.3048
KIP = 4.448
KSF = 47.88
PCF = 0.157087
KIPFT = 1.356


# ── V-001 : Nordlund driven H-pile in sand (North Abutment, Table D-6) ───────

def _v001_profile(toe_phi=None):
    """V-001 cohesionless profile FROM THE GROUND SURFACE; the pile head sits at
    the footing bottom, 5 ft below grade (head_depth).

    L1 loose silty sand 0-25 ft bgs (phi 33, gamma 105 pcf), L2 medium dense
    sand 25-50 ft (phi 36, 112 pcf), L3 dense gravel 50+ ft (phi 36 shaft,
    125 pcf; toe phi 40 where the example used it). Water table 15 ft bgs.
    """
    return AxialSoilProfile(layers=[
        AxialSoilLayer(thickness=25 * FT, soil_type="cohesionless",
                       unit_weight=105 * PCF, friction_angle=33,
                       description="L1 loose silty sand"),
        AxialSoilLayer(thickness=25 * FT, soil_type="cohesionless",
                       unit_weight=112 * PCF, friction_angle=36,
                       description="L2 medium dense sand"),
        AxialSoilLayer(thickness=52 * FT, soil_type="cohesionless",
                       unit_weight=125 * PCF, friction_angle=36,
                       toe_friction_angle=toe_phi,
                       description="L3 dense gravel"),
    ], gwt_depth=15 * FT)


def _v001_run(D_ft, toe_phi=None, **kw):
    """D_ft below the footing (Table D-6's depth)."""
    return AxialPileAnalysis(pile=make_h_pile("HP12x74"),
                             soil=_v001_profile(toe_phi),
                             pile_length=D_ft * FT, head_depth=5 * FT,
                             method="auto", **kw).compute()


def test_v001_hpile_box_section_convention():
    """make_h_pile('HP12x74') reproduces the example's H-pile box perimeter and
    toe area (the convention that drives both shaft and toe)."""
    pile = make_h_pile("HP12x74")
    assert pile.perimeter / FT == pytest.approx(4.06, abs=0.05)      # ~4.06 ft
    assert pile.tip_area / (FT * FT) == pytest.approx(1.03, abs=0.03)  # ~1.03 ft2


def test_v001_nordlund_shaft_resistance_matches():
    """Nordlund shaft resistance vs Table D-6 (depths below footing).

    PASS within +1.1 % to +1.2 % at every depth since GM-121 (2026-10-18):
    delta/phi from GEC-12 Fig. 7-9 (curve f, V = 0.151 ft3/ft), K_delta from
    Tables 7-6/7-7 and C_F from Fig. 7-14 (before: a fitted K_delta and
    delta/phi 0.75 on the footing-datum profile, -8 % to +5 %).
    """
    published_Rs = {35: 137.5, 50: 250.7, 60: 344.1, 70: 452.5}
    for D_ft, Rs_pub in published_Rs.items():
        r = _v001_run(D_ft)
        assert r.Q_skin / KIP == pytest.approx(Rs_pub, rel=0.02), (
            f"D={D_ft} ft: shaft {r.Q_skin / KIP:.1f} vs published {Rs_pub}"
        )


def test_v001_overburden_above_the_footing_counts():
    """Table D-6's sigma' includes the 5 ft of soil above the footing: its toe
    at 0.01 ft below the footing is 16.4 kips, which needs sigma' > 0 there.
    The ground-surface profile with head_depth gives 16.1 kips; a profile
    started at the footing (overburden removed) gives none, and its shaft
    falls 17-27 % short of the table."""
    r = _v001_run(0.01, toe_stress_limit=None)
    assert r.Q_tip / KIP == pytest.approx(16.4, rel=0.03)
    assert r.layer_breakdown[0]["depth_top_m"] == pytest.approx(5 * FT, abs=0.01)
    footing = AxialSoilProfile(layers=[
        AxialSoilLayer(20 * FT, "cohesionless", 105 * PCF, friction_angle=33),
        AxialSoilLayer(25 * FT, "cohesionless", 112 * PCF, friction_angle=36),
        AxialSoilLayer(52 * FT, "cohesionless", 125 * PCF, friction_angle=36),
    ], gwt_depth=10 * FT)
    r = AxialPileAnalysis(pile=make_h_pile("HP12x74"), soil=footing,
                          pile_length=60 * FT).compute()
    assert r.Q_skin / KIP / 344.1 == pytest.approx(0.81, abs=0.02)


def test_v001_toe_with_design_limit_phi40_matches():
    """Toe resistance reproduces the Table D-6 Layer-3 plateau (428.1 kips) to
    within 1 % with the example's separate Layer-3 toe phi = 40 deg, when
    sigma'p is not limited: q_L (Fig. 7-15) at phi = 40 governs at 50, 60
    and 70 ft (429.3 kips; Eq. 7-9 gives 439.9 at 50 ft since GM-282, 425.9
    before)."""
    pile = make_h_pile("HP12x74")
    soil = _v001_profile(toe_phi=40)
    for D_ft in (50, 60, 70):           # depths where the tip is in Layer 3
        sigma_v_tip = soil.effective_stress_at_depth((D_ft + 5) * FT)
        Rt = end_bearing_cohesionless(
            40.0, sigma_v_tip, pile.tip_area, D_ft * FT, pile.width,
            notes=[], sigma_p_limit=None) / KIP
        assert Rt == pytest.approx(428.1, rel=0.01), (
            f"D={D_ft} ft: toe(phi=40) {Rt:.1f} vs published plateau 428.1"
        )


def test_v001_toe_single_phi_api_is_default_behaviour():
    """DEFAULT BEHAVIOUR: with NO per-layer toe phi set, the high-level
    AxialPileAnalysis API uses ONE phi per layer, so it applies phi=36
    (shaft) at the toe too: ~155 kips in Layer 3 (-64 % vs the published
    428.1, which needs the design-limit toe phi=40 that GEC-12's own
    example selects)."""
    Rt_single_phi = _v001_run(60).Q_tip / KIP
    assert Rt_single_phi == pytest.approx(154.8, rel=0.02)   # module, phi=36 toe
    assert abs(Rt_single_phi - 428.1) / 428.1 > 0.15         # off vs published


def test_v001_toe_below_the_plateau_follows_table_d6():
    """Below the q_L plateau Table D-6's toe is alpha_t N'q sigma'p x box area
    with sigma'p not limited (Eq. 7-9, Fig. 7-16): the module, with the limit
    off, is within 0.7 % of it in Layer 1 (toe phi 33) and Layer 3 (toe phi
    40) and 1.9-2.3 % low in Layer 2 (toe phi 36). GM-282 read N'q again off
    Fig. 7-16's own image: the earlier nodes gave 2.0-2.3 %, 4.6-4.9 % and
    2.7 % low."""
    published = {0.01: 16.4, 2: 23.0, 5: 32.8, 8: 42.6, 10.01: 49.2,
                 23: 118.1, 26: 126.2, 30: 136.9, 34: 147.6,
                 46: 406.4, 47: 414.1, 48: 421.8}
    for D_ft, Rp_pub in published.items():
        r = _v001_run(D_ft, toe_phi=40 if D_ft > 45 else None,
                      toe_stress_limit=None)
        tol = 0.025 if 20 < D_ft < 45 else 0.008
        assert r.Q_tip / KIP == pytest.approx(Rp_pub, rel=tol), (
            f"D={D_ft} ft: toe {r.Q_tip / KIP:.1f} vs published {Rp_pub}")


def test_v001_highlevel_api_toe_phi40_matches_plateau():
    """The high-level API with the Layer-3 toe phi=40:

    - with GEC-12 Step 8's limit on sigma'p (3 ksf, the module's default since
      GM-121) the toe is alpha_t N'q x 3 ksf x box area = 371.5 kips, -13 %
      vs the 428.1 plateau. Table D-6 (DrivenPiles) did not apply Step 8:
      sigma'v at its Layer-3 toe is 3.6-4.8 ksf and its plateau is q_L(40)
      times the box area;
    - with the limit off (toe_stress_limit None) it reproduces the plateau
      within 1 %.
    The shaft is unaffected (still phi=36)."""
    for D_ft in (50, 60, 70):           # depths where the tip is in Layer 3
        r = _v001_run(D_ft, toe_phi=40)
        assert r.Q_tip / KIP == pytest.approx(371.5, rel=0.005)
        assert r.toe_details["sigma_p_kPa"] == pytest.approx(3 * KSF, rel=1e-3)
        r = _v001_run(D_ft, toe_phi=40, toe_stress_limit=None)
        assert r.Q_tip / KIP == pytest.approx(428.1, rel=0.01), (
            f"D={D_ft} ft: high-level toe(phi=40) {r.Q_tip / KIP:.1f} "
            f"vs published plateau 428.1"
        )
    r40 = _v001_run(60, toe_phi=40)
    r36 = _v001_run(60)
    assert r40.Q_skin == pytest.approx(r36.Q_skin, rel=1e-12)
    assert r40.Q_tip > r36.Q_tip


# ── V-002 : alpha-method H-pile in clay (South Abutment, Table D-100) ─────────

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
    """V-002 cohesive profile as 1-ft cohesive sublayers from the GROUND
    SURFACE, each with its sample-depth su (Fig. 7-17's D is measured from
    the ground surface, GEC-12 p. 246). L1 medium clay (110 pcf), L2 stiff
    clay (124 pcf), L3 very stiff clay (129 pcf)."""
    layers = []
    z = 0.0
    while z < 101:
        bgs = z + 0.5
        gamma = 110 if bgs < 25 else (124 if bgs < 45 else 129)
        layers.append(AxialSoilLayer(
            thickness=1 * FT, soil_type="cohesive",
            unit_weight=gamma * PCF, cohesion=_su_at_bgs(bgs) * KSF))
        z += 1
    return AxialSoilProfile(layers=layers, gwt_depth=None)


def _v002_run(D_ft, **kw):
    """D_ft below the cap (5 ft bgs; Table D-100's depth)."""
    return AxialPileAnalysis(pile=make_h_pile("HP12x74"), soil=_v002_profile(),
                             pile_length=D_ft * FT, head_depth=5 * FT,
                             method="auto", **kw).compute()


def test_v002_toe_nine_su_matches():
    """End bearing 9*cu reproduces Table D-100 toe resistance to within +/-2%
    at the depths where the tip sits cleanly inside Layer 3 (D = 70/80/90 ft)."""
    published_Rt = {70: 32.75, 80: 33.03, 90: 33.68}
    for D_ft, Rt_pub in published_Rt.items():
        r = _v002_run(D_ft, adhesion_beyond_chart="chart_end")
        assert r.Q_tip / KIP == pytest.approx(Rt_pub, rel=0.02), (
            f"D={D_ft} ft: toe {r.Q_tip / KIP:.1f} vs published {Rt_pub}"
        )


def test_v002_alpha_from_fig_7_17():
    """Since GM-121 alpha comes from GEC-12 Fig. 7-17 by su, pile type and
    D/b. The example's stiff-clay alpha (~0.76 at su ~1.9 ksf, deep) is the
    smooth-steel D = 40b curve's 0.79; the corrugated curve, which GEC-12
    p. 247 gives for H-piles, reads 0.92 there."""
    assert alpha_tomlinson(0.70 * KSF, "steel", D_over_b=40) == \
        pytest.approx(0.99, abs=0.02)
    assert alpha_tomlinson(1.90 * KSF, "steel", D_over_b=40) == \
        pytest.approx(0.79, abs=0.02)
    assert alpha_tomlinson(1.90 * KSF, "h_pile", D_over_b=40) == \
        pytest.approx(0.92, abs=0.02)


def test_v002_shaft_matches_with_the_examples_chart_choices():
    """PASS within 1 % with the two choices the example made: the smooth
    steel curves of Fig. 7-17 for the H-pile (adhesion_curve) and Ca held at
    the curve's end where su exceeds the chart (adhesion_beyond_chart)."""
    published_Rs = {70: 318.4, 80: 369.3, 90: 420.33}
    for D_ft, Rs_pub in published_Rs.items():
        r = _v002_run(D_ft, adhesion_curve="smooth_steel",
                      adhesion_beyond_chart="chart_end")
        assert r.Q_skin / KIP == pytest.approx(Rs_pub, rel=0.015), (
            f"D={D_ft} ft: shaft {r.Q_skin / KIP:.1f} vs published {Rs_pub}"
        )


def test_v002_shaft_with_the_module_defaults_documented():
    """CONVENTION: Layer 3's su (3.1-3.7 ksf) is past the end of Fig.
    7-17's curves below about 70 ft, where GEC-12 recommends API alpha for
    highly overconsolidated clays (p. 247, alpha-method Step 1) and gives
    no su / sigma'vo that marks one. Since GM-229 (owner OWN-63) the
    default run is refused until the engineer gives that line. With a line
    of 1.0 Layer 3 (su / sigma'vo 0.32-0.41) is not highly
    overconsolidated, so Ca is held at the curve's end as the example did,
    and the shaft is 13-14 % above Table D-100, the corrugated curve's
    higher adhesion (GEC-12 p. 247 for H-piles); the as-drawn reading of
    Fig. 7-17 (GM-229) moves none of it, since no V-002 layer reads alpha
    above 1. With API alpha in every layer past the chart
    (adhesion_beyond_chart "api") it is 25-53 % above: API alpha (about
    0.7 at sigma'v ~7 ksf) nearly doubles the chart-end adhesion. The test
    pins both bands so a change to either is caught."""
    from axial_pile.capacity import JudgmentNeeded
    with pytest.raises(JudgmentNeeded, match="highly_overconsolidated"):
        _v002_run(70)
    published_Rs = {70: 318.4, 80: 369.3, 90: 420.33}
    for D_ft, Rs_pub in published_Rs.items():
        r = _v002_run(D_ft, highly_overconsolidated_su_ratio=1.0)
        ratio = (r.Q_skin / KIP) / Rs_pub
        assert 1.10 <= ratio <= 1.17, (
            f"D={D_ft} ft: shaft ratio {ratio:.2f} outside documented band"
        )
        sources = {lb.get("adhesion_source") for lb in r.layer_breakdown}
        assert "Fig. 7-17 end" in sources and "API alpha" not in sources
        assert not any(lb.get("highly_overconsolidated")
                       for lb in r.layer_breakdown)
        assert max(lb["alpha"] for lb in r.layer_breakdown) <= 1.0
        api = _v002_run(D_ft, adhesion_beyond_chart="api")
        ratio = (api.Q_skin / KIP) / Rs_pub
        assert 1.20 <= ratio <= 1.60, (
            f"D={D_ft} ft: API shaft ratio {ratio:.2f} outside the band"
        )
        assert any(lb.get("adhesion_source") == "API alpha"
                   for lb in api.layer_breakdown)


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
    (published 120 bpf — outside +/-25%). The compression stress, 48.5 ksi
    (+21%) until GM-294's review took the pile-top force without the
    helmet's inertia, is 39.0 ksi (published 40.1 ksi), on a blow count that
    is not reproduced. The module's diesel hammer is a simple
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
    # The published blow count is OUT of the inventory tolerance
    assert abs(blows_per_ft - 120.0) / 120.0 > 0.25      # bpf not within +/-25%
    assert comp_ksi == pytest.approx(39.0, abs=0.5)      # 40.1 published
    # the diesel run as a drop ram says so (GEC-12 Vol 2 p. 206)
    assert res.warnings and "p. 206" in res.warnings[0]


# ── V-004 : downdrag neutral-plane location and drag force (Block 17) ────────

def _v004_calibrated_betas():
    """Per-layer beta overrides that make the downdrag module's shaft match the
    published Nordlund shaft distribution (Table D-6) at the layer boundaries,
    so the NP construction is validated decoupled from the shaft method (the
    inventory permits taking the resistance distribution from Table D-6).

    Table D-6 shaft below the footing: 51.6 kips at D=20 ft (end L1), 209.8 at
    D=45 (end L2), 344.1 at D=60. Layer shares: L1=51.6, L2=158.2, L3=134.3.

    The shaft is integrated the way the module integrates it (since GM-01,
    2026-10-14): the reported unit skin friction read as piecewise linear
    between the nodes, a repeated depth (a layer boundary) being a jump.
    The layer boundaries are nodes, so the integral between them is the
    trapezoid sum over the nodes in between. Before, this helper summed
    each node's value over the segment below it, which matched the
    module's old drag-load rule but not its old plane search.
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
        z, fs = r.z, r.unit_skin_friction
        tol = 1e-9
        # Nodes from z0 (the node below it, when z0 is a boundary) to z1.
        i0 = int(np.searchsorted(z, z0 - tol, side="left"))
        if i0 + 1 < len(z) and abs(z[i0 + 1] - z0) < tol:
            i0 += 1
        i1 = int(np.searchsorted(z, z1 + tol, side="right"))
        return float(trapezoid(fs[i0:i1], z[i0:i1]) * pile_perim / KIP)

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


def _v004_run(**kw):
    """HP12x74 driven 60 ft below the footing; sustained load Q = 201
    kips; Nt set so the toe is fully mobilized at 428.1 kips (100% toe
    mobilization)."""
    build, b1, b2, b3 = _v004_calibrated_betas()
    soil = build(b1, b2, b3)
    tip_area_box = 1.0294 * FT * FT
    L = 60 * FT
    Nt = 428.1 * KIP / (soil.effective_stress_at_depth(L) * tip_area_box)
    return DowndragAnalysis(
        soil=soil, pile_length=L, pile_diameter=0.3086,
        pile_perimeter=4.058 * FT, pile_area=tip_area_box,
        pile_unit_weight=20.0, Q_dead=201 * KIP, Nt=Nt, **kw).compute()


def test_v004_neutral_plane_and_drag_force():
    """PASS: the Fellenius neutral-plane construction, with the pile's own
    weight left out of the load curve by default as the published Block 17
    construction (and UFC 3-220-20 Fig 6-26, GEC-12 §7.3.6.1) does,
    reproduces the published NP, max axial force and drag force: NP 54.0
    ft, Qmax 486.6 kips, DF 285.6 kips vs published 54 ft below the pile
    head (6 ft above the toe) / 486 / 285 kips.

    Achieved 2026-10-14 (GM-01) with the weight zeroed; the default since
    2026-10-15 (owner, OWN-58). The unit weight given (20 kN/m3) is not
    used unless include_pile_weight=True. Before GM-01 the same inputs
    gave NP 53.3 ft / Qmax 487.2 / DF 286.2 kips with the pile weight in.
    """
    r = _v004_run()
    assert r.include_pile_weight is False
    assert r.pile_weight_to_np == 0.0
    NP_below_head_ft = r.neutral_plane_depth / FT     # z=0 is the pile head
    Qmax_kips = r.max_pile_load / KIP
    DF_kips = (r.max_pile_load - 201 * KIP) / KIP

    assert r.toe_resistance / KIP == pytest.approx(428.1, rel=0.01)
    assert NP_below_head_ft == pytest.approx(54.00, abs=0.01)
    assert Qmax_kips == pytest.approx(486.6, abs=0.05)
    assert DF_kips == pytest.approx(285.6, abs=0.05)
    assert NP_below_head_ft == pytest.approx(54.0, abs=0.5)   # published
    assert Qmax_kips == pytest.approx(486.0, rel=0.005)
    assert DF_kips == pytest.approx(285.0, rel=0.005)
    # Load curve = resistance curve at the plane (toe + positive shaft).
    assert r.max_pile_load == pytest.approx(
        r.toe_resistance + r.positive_skin_friction, rel=1e-9)


def test_v004_with_pile_weight_included():
    """PASS: include_pile_weight=True keeps the old basis (the pile's own
    weight in the load curve, 20 kN/m3 x the box area) and reproduces the
    numbers the module gave before 2026-10-15: NP 53.61 ft, Qmax 490.1,
    DF 289.1 kips (published 54 / 486 / 285: -0.4 ft, +0.8 %, +1.4 %)."""
    r = _v004_run(include_pile_weight=True)
    assert r.include_pile_weight is True
    assert r.pile_weight_to_np / KIP == pytest.approx(7.03, abs=0.01)
    assert r.neutral_plane_depth / FT == pytest.approx(53.61, abs=0.01)
    assert r.max_pile_load / KIP == pytest.approx(490.1, abs=0.05)
    assert (r.max_pile_load - 201 * KIP) / KIP == pytest.approx(289.1,
                                                                abs=0.05)
    assert r.neutral_plane_depth / FT == pytest.approx(54.0, abs=3.0)
    assert r.max_pile_load / KIP == pytest.approx(486.0, rel=0.10)
    assert r.max_pile_load == pytest.approx(
        r.toe_resistance + r.positive_skin_friction, rel=1e-9)


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

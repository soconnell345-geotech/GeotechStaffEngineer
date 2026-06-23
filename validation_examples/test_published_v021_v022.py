"""Phase E validation — bearing_capacity + settlement (GEC-6 Ex B-1) benchmarks (V-021, V-022).

Source (FHWA GEC-6, FHWA-SA-02-054, "Shallow Foundations"):
  - V-021  Appendix B, Example 1 ("Interior Bridge Pier on Spread Footing"),
           Step 7 bearing capacity. Square footing, Df = 2.3 m, bearing in silty
           sand: phi = 35 deg, c = 0, gamma = 19.6 kN/m3 (all layers); GW at
           9.1 m below grade. Vesic/AASHTO N-factors (Nq = 33.3, Nγ = 48.0 at
           φ = 35), Vesic shape factors (sq = 1.7, s_γ = 0.6), dq = 1.0 (cohesive
           overburden), and AASHTO groundwater-correction FACTORS Cwq = 1.0,
           Cwγ = 0.5 + 0.5·[Dw/(Df + 1.5·B)] = 0.9 (evaluated at the B = 6 m
           trial and held fixed). Published: qult = 2553 + 254·B(m) kPa; with
           FS = 3, qall = 851 + 85·B → B = 3 m: 1106; 4.6 m: 1242; 6.1 m: 1369.
           Effective-area follow-on (Step 10): 4.9 m square, P = 8070 kN,
           eB = 0.077 m, eL = 0.117 m → B' = 4.75, L' = 4.67, A' = 22.2 m2,
           q_applied = 364 kPa; sliding FS = 31. extract: extracts/gec6_ex1.txt
           (PDF pp. 161-168).
  - V-022  Appendix B, Example 1, Tables B1-2 / B1-3 (PDF pp. 168-170). Hough
           (granular / SPT C'-index) settlement of the layers below the footing,
           with a 2:1 stress increase Δσv = q·B²/(B+Z)² (square footing). Per
           layer dH = H/C'·log10[(σ'vo + Δσ)/σ'vo], summed. Layers below the
           base: L2 silty sand (H = 2.1 m, Z = 1.05 m, σ'vo = 65.7 kPa, C' = 65);
           L3a (H = 4.7, Z = 4.45, 132 kPa, C' = 120); L3b (H = 3.0, Z = 8.3,
           193 kPa, C' = 102); L4 (H = 3.0, Z = 11.3, 222 kPa, C' = 110).
           Published totals (mm): B = 3 m → 21/25/28/30 at q = 240/290/335/380;
           B = 4.6 m → 28/31/34/37; B = 6.1 m → 31/35/38/41. Worked single case
           (B = 3, q = 240): per-layer 15 + 4 + 1 + 1 = 21 mm.

See validation_examples/INVENTORY.md (V-021, V-022) and RESULTS.md.

KEY FINDINGS (details in each test docstring and RESULTS.md):

- V-021 is a PASS via the module factors (the strongest bearing check in the
  library). The `bearing_capacity` Vesic path reproduces the published
  Nq = 33.30 / Nγ = 48.03 at φ = 35 EXACTLY (the example's rounded 33.3 / 48.0),
  and the Vesic shape factors reproduce sq = 1.700 / s_γ = 0.600 EXACTLY. Driving
  the example's own assembly (q·Nq·sq·Cwq + 0.5·γ·B·Nγ·s_γ·Cwγ, with dq = 1.0 and
  the AASHTO Cwγ = 0.9 held fixed at the B = 6 trial) on the module's factor
  functions reproduces qult = 2552 + 254.2·B vs the published 2553 + 254·B — the
  intercept AND slope to 4 figures. qall (FS = 3) at B = 3/4.6/6.1 m = 1105/1240/
  1368 kPa vs published 1106/1242/1369 (all <0.2%, well inside ±3%). The
  effective-area follow-on is a clean PASS through the module's `Footing`:
  the 4.9 m square with eB = 0.077, eL = 0.117 gives B' = 4.746, L' = 4.666,
  A' = 22.15 m2, q_applied = 364.4 kPa (pub 364) and sliding FS = 30.9 (pub 31).

- V-021 high-level `compute()` is a CONVENTION (depth factor + groundwater model).
  The packaged `BearingCapacityAnalysis(...).compute()` runs ~17% high (qult 3897
  vs 3315 at B = 3) because it applies (a) Vesic DEPTH factors dq ≈ 1.10-1.20 (the
  example sets dq = 1.0 because the overburden is cohesive — AASHTO GEC-6
  Table 5-4), and (b) its effective-unit-weight groundwater model (which sees the
  GW at 9.1 m as below the bearing wedge for these B, so γ_eff = γ = 19.6 with NO
  reduction) instead of the AASHTO Cwγ = 0.9 correction-factor the example uses.
  Both module choices are defensible; the example's closed form is recovered
  exactly by assembling the SAME module N/shape factors the example's way
  (dq = 1, AASHTO Cw). Module NOT tuned.

- V-022 Hough method is now a PASS via the NEW `settlement.hough` module method
  (v5.2): `hough_settlement(layers, q_net, B, L=None)` over `HoughLayer`
  objects. The Hough form dH = H/C'·log10[(σ'vo+Δσ)/σ'vo] uses a "bearing
  capacity index" C' (distinct from the module's Cc/(1+e0)) and REUSES the
  module's `approximate_2to1` for the 2:1 stress increase (square footing:
  q·B²/(B+Z)² = q·B·L/((B+z)(L+z)) at B=L). The module reproduces the published
  Table B1-2 stress fractions (0.55/0.16/0.07/0.04 at B = 3, etc.) to ≤2%, and
  the full Hough settlement Table B1-3 (all 12 q×B cells) to within ±15% /
  the source's mm rounding (the worked B = 3, q = 240 case lands at 21.5 mm =
  15.4+4.4+1.1+0.6 vs the published 21 = 15+4+1+1). The Cc/Cr e-log(p)
  consolidation path remains for cohesive soils.

Units: V-021 / V-022 inputs are already SI (kPa, m, kN/m3, degrees). No
conversions needed. The only non-SI quantity is the Hough index C' (dimensionless
bearing-capacity index, used directly).
"""

import math
import warnings

import pytest

# bearing_capacity (V-021)
from bearing_capacity import (
    Footing, SoilLayer, BearingSoilProfile, BearingCapacityAnalysis,
)
from bearing_capacity.factors import (
    bearing_capacity_Nq, bearing_capacity_Ngamma, shape_factors,
)
# settlement (V-022): the 2:1 stress-distribution primitive + Hough method
from settlement.stress_distribution import approximate_2to1
from settlement.hough import HoughLayer, hough_settlement


# ════════════════════════════════════════════════════════════════════════════
# V-021 : GEC-6 Ex B-1 Step 7 — square-footing ultimate bearing capacity
# ════════════════════════════════════════════════════════════════════════════
#
# phi = 35, c = 0, gamma = 19.6 kN/m3, Df = 2.3 m, square (B = L), GW at 9.1 m.
# Published assembly (Eqn. 5-14, cohesion term dropped):
#   qult = q·Nq·sq·dq·Cwq + 0.5·γ·B·Nγ·s_γ·Cwγ
# with q = γ·Df = 45.1 kPa, dq = 1.0 (cohesive overburden, Table 5-4),
# Cwq = 1.0 (Dw/Df capped), Cwγ = 0.5 + 0.5·[Dw/(Df+1.5·B)] = 0.9 at the B = 6
# trial (held fixed → linear qult). N/shape from Tables 5-1/5-2.

_V021_PHI = 35.0
_V021_C = 0.0
_V021_GAMMA = 19.6           # kN/m3
_V021_DF = 2.3               # m
_V021_DW = 9.1               # m (GW below grade)
_V021_Q = _V021_GAMMA * _V021_DF      # 45.08 kPa surcharge at base
_V021_CWG_FIXED = 0.9        # AASHTO Cwγ at the B = 6 m trial (example holds fixed)


def _cwg(B):
    """AASHTO Cwγ groundwater factor (GEC-6 Eqn. 5-10), capped at 1.0."""
    return min(0.5 + 0.5 * (_V021_DW / (_V021_DF + 1.5 * B)), 1.0)


def _cwq():
    """AASHTO Cwq groundwater factor (GEC-6 Eqn. 5-11), capped at 1.0."""
    return min(0.5 + 0.5 * (_V021_DW / _V021_DF), 1.0)


def test_v021_N_factors_match_vesic_aashto_table():
    """PASS (the direct-match the inventory calls out): the module's Vesic N
    factors reproduce the GEC-6 Table 5-1 values for φ = 35 — Nq = 33.30 (pub
    33.3) and Nγ = 48.03 (pub 48.0) — to 4 figures. These ARE the AASHTO/Vesic
    bearing-capacity factors; the example rounds them to 33.3 / 48.0."""
    Nq = bearing_capacity_Nq(_V021_PHI)
    Ng = bearing_capacity_Ngamma(_V021_PHI, "vesic")
    assert Nq == pytest.approx(33.30, abs=0.02)        # pub 33.3
    assert Ng == pytest.approx(48.03, abs=0.05)        # pub 48.0
    # the Vesic path is the right one: Meyerhof/Hansen Nγ differ markedly
    assert bearing_capacity_Ngamma(_V021_PHI, "meyerhof") == pytest.approx(37.15, abs=0.1)
    assert bearing_capacity_Ngamma(_V021_PHI, "hansen") == pytest.approx(33.92, abs=0.1)


def test_v021_shape_factors_match_table_5_2():
    """PASS: the module's Vesic shape factors for a square footing (B/L = 1)
    reproduce the example's sq = 1 + (B/L)·tan(φ) = 1.700 and
    s_γ = 1 − 0.4·(B/L) = 0.600 EXACTLY (GEC-6 Table 5-2)."""
    sc, sq, sg = shape_factors(_V021_PHI, 1.0, 1.0, "vesic")
    assert sq == pytest.approx(1.700, abs=0.001)       # pub 1.7
    assert sg == pytest.approx(0.600, abs=0.001)       # pub 0.6


def test_v021_groundwater_correction_factors():
    """PASS: the AASHTO Cwγ / Cwq groundwater-correction factors the example uses.
    Cwq = 0.5 + 0.5·(Dw/Df) = 2.48 → capped at 1.0 (pub "Use Cwq = 1.0"); and
    Cwγ = 0.5 + 0.5·[Dw/(Df + 1.5·B)] = 0.903 at the B = 6 m trial (pub 0.9). The
    example holds this single Cwγ fixed across footing widths to get the linear
    qult, so we pin the B = 6 value."""
    assert _cwq() == pytest.approx(1.0, abs=1e-9)
    assert _cwg(6.0) == pytest.approx(0.9, abs=0.01)   # pub 0.9 at B = 6 trial


def test_v021_qult_qall_match_published_closed_form():
    """PASS (strongest bearing check): driving the example's own assembly
    qult = q·Nq·sq·Cwq + 0.5·γ·B·Nγ·s_γ·Cwγ on the MODULE's factor functions —
    with dq = 1.0 (cohesive overburden) and the AASHTO Cwγ = 0.9 fixed — exactly
    reproduces the published closed form qult = 2553 + 254·B (our intercept
    2552.0, slope 254.2) and qall (FS = 3) = 851 + 85·B. At B = 3/4.6/6.1 m:
    qult = 3315/3721/4102, qall = 1105/1240/1368 vs published 1106/1242/1369
    (<0.2%, well inside ±3%)."""
    Nq = bearing_capacity_Nq(_V021_PHI)
    Ng = bearing_capacity_Ngamma(_V021_PHI, "vesic")
    sc, sq, sg = shape_factors(_V021_PHI, 1.0, 1.0, "vesic")
    dq = 1.0          # example: overburden is cohesive → dq set to 1.0 (Table 5-4)
    Cwq = _cwq()
    Cwg = _V021_CWG_FIXED

    # intercept (overburden term) and slope (self-weight term per metre of B)
    intercept = _V021_Q * Nq * sq * dq * Cwq
    slope = 0.5 * _V021_GAMMA * Ng * sg * Cwg
    assert intercept == pytest.approx(2553.0, rel=0.005)   # pub 2553
    assert slope == pytest.approx(254.0, rel=0.005)        # pub 254

    pub_qult = {3.0: 3315.0, 4.6: 3721.0, 6.1: 4102.0}
    pub_qall = {3.0: 1106.0, 4.6: 1242.0, 6.1: 1369.0}
    for B in (3.0, 4.6, 6.1):
        qult = intercept + slope * B
        qall = qult / 3.0
        # closed-form linear form
        assert qult == pytest.approx(2553.0 + 254.0 * B, rel=0.005)
        # vs the published per-width values
        assert qult == pytest.approx(pub_qult[B], rel=0.03)    # ±3%
        assert qall == pytest.approx(pub_qall[B], rel=0.03)


def test_v021_highlevel_compute_is_convention_depth_and_gw():
    """CONVENTION (high-level API): the packaged `BearingCapacityAnalysis.compute()`
    runs ~17% HIGH vs the published qult (3897 vs 3315 at B = 3) for two
    defensible reasons, NOT a bug:
      (1) it applies Vesic DEPTH factors dq ≈ 1.10-1.20, while the example sets
          dq = 1.0 (the overburden is cohesive, GEC-6 Table 5-4 guidance);
      (2) its groundwater model averages the effective unit weight over the
          bearing wedge — for these B the GW at 9.1 m is BELOW the wedge, so
          γ_eff = γ = 19.6 with NO reduction, whereas the example applies the
          AASHTO Cwγ = 0.9 correction factor.
    Both are valid conventions; the example's closed form is recovered exactly by
    the previous test (same module N/shape factors, example's dq = 1 + Cwγ). We
    pin that compute() has dq > 1 and the no-reduction γ_eff here, documenting the
    delta."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ft = Footing(width=3.0, depth=_V021_DF, shape="square")
        soil = BearingSoilProfile(
            layer1=SoilLayer(cohesion=_V021_C, friction_angle=_V021_PHI,
                             unit_weight=_V021_GAMMA),
            gwt_depth=_V021_DW)
        res = BearingCapacityAnalysis(footing=ft, soil=soil,
                                      factor_of_safety=3.0).compute()
    # module applies a depth factor > 1 (example uses dq = 1.0)
    assert res.dq > 1.05
    # module sees full γ (GW below the B = 3 wedge), no Cwγ-style reduction
    assert res.gamma_eff == pytest.approx(19.6, abs=0.01)
    # the combined effect is ~+17% on qult vs the published 3315 at B = 3
    pub_qult_B3 = 3315.0
    assert res.q_ultimate > pub_qult_B3
    assert (res.q_ultimate - pub_qult_B3) / pub_qult_B3 == pytest.approx(0.176, abs=0.03)
    # the module N/shape factors themselves are correct (matched above)
    assert res.Nq == pytest.approx(33.30, abs=0.02)
    assert res.Ngamma == pytest.approx(48.03, abs=0.05)
    assert res.sq == pytest.approx(1.700, abs=0.001)
    assert res.sgamma == pytest.approx(0.600, abs=0.001)


def test_v021b_effective_area_eccentric_load():
    """PASS (effective-area follow-on, Step 10): the module's `Footing` effective
    dimensions reproduce the example's eccentric-load bookkeeping. For the 4.9 m
    square footing with eB = 0.077 m, eL = 0.117 m, P = 8070 kN:
    B' = B − 2·eB = 4.746 (pub 4.75), L' = 4.666 (pub 4.67), A' = 22.15 m2
    (pub 22.2), q_applied = P/A' = 364.4 kPa (pub 364). The earlier 4.6 m trial
    likewise gives B' = 4.45, L' = 4.37, A' = 19.4, q = 416 (pub). Sliding FS =
    0.7·(W + P)/V = 30.9 (pub 31)."""
    P = 8070.0

    # 4.9 m square, eccentric (the accepted design)
    ft = Footing(width=4.9, length=4.9, shape="rectangular", depth=_V021_DF,
                 eccentricity_B=0.077, eccentricity_L=0.117)
    assert ft.B_eff == pytest.approx(4.75, abs=0.02)       # pub 4.75
    assert ft.L_eff == pytest.approx(4.67, abs=0.02)       # pub 4.67
    assert ft.A_eff == pytest.approx(22.2, abs=0.1)        # pub 22.2 m2
    assert P / ft.A_eff == pytest.approx(364.0, abs=2.0)   # pub 364 kPa

    # 4.6 m trial (rejected — q_applied too high)
    ft2 = Footing(width=4.6, length=4.6, shape="rectangular", depth=_V021_DF,
                  eccentricity_B=0.077, eccentricity_L=0.117)
    assert ft2.B_eff == pytest.approx(4.45, abs=0.02)
    assert ft2.L_eff == pytest.approx(4.37, abs=0.02)
    assert ft2.A_eff == pytest.approx(19.4, abs=0.1)
    assert P / ft2.A_eff == pytest.approx(416.0, abs=2.0)  # pub 416 kPa

    # sliding FS = 0.7·(W + P)/V (Step 11): W = 1159 kN cover+footing, V = 209 kN
    W, V = 1159.0, 209.0
    FS_sliding = 0.7 * (W + P) / V
    assert FS_sliding == pytest.approx(31.0, abs=1.0)      # pub 31


# ════════════════════════════════════════════════════════════════════════════
# V-022 : GEC-6 Ex B-1 Tables B1-2/B1-3 — Hough granular settlement (2:1 stress)
# ════════════════════════════════════════════════════════════════════════════
#
# 2:1 stress increase Δσv = q·B²/(B+Z)² (square footing). Per-layer Hough
# dH = H/C'·log10[(σ'vo + Δσ)/σ'vo], summed. C' is the Hough "bearing-capacity
# index" (dimensionless), distinct from the module's Cc/(1+e0).

# (H [m], σ'vo [kPa], C' [-]) and midpoint depth below base Z [m]
_V022_LAYERS = [
    (2.1, 65.7, 65),    # L2 silty sand
    (4.7, 132.0, 120),  # L3a well-graded sand
    (3.0, 193.0, 102),  # L3b well-graded sand (saturated)
    (3.0, 222.0, 110),  # L4 clean uniform sand
]
_V022_ZMID = [1.05, 4.45, 8.3, 11.3]   # m below footing base


def _hough_total_mm(B, q):
    """Hough settlement (mm) summed over the four layers, 2:1 square stress."""
    total = 0.0
    for (H, svo, Cp), Z in zip(_V022_LAYERS, _V022_ZMID):
        # square footing: approximate_2to1(q,B,B,Z) == q·B²/(B+Z)²
        dsig = approximate_2to1(q, B, B, Z)
        total += H / Cp * math.log10((svo + dsig) / svo) * 1000.0
    return total


def _hough_layers():
    """Build the four GEC-6 Ex B-1 granular layers as HoughLayer objects."""
    return [
        HoughLayer(thickness=H, depth_to_center=Z, sigma_v0=svo, C_prime=Cp)
        for (H, svo, Cp), Z in zip(_V022_LAYERS, _V022_ZMID)
    ]


def _hough_module_total_mm(B, q):
    """Hough settlement (mm) via the NEW settlement.hough module method."""
    return hough_settlement(_hough_layers(), q_net=q, B=B).total_mm


def test_v022_2to1_stress_increase_matches_table_b1_2():
    """PASS (primitive): the example's 2:1 stress increase Δσv = q·B²/(B+Z)² for a
    square footing IS the `settlement` module's `approximate_2to1` (which for a
    square B = L gives q·B·L/((B+z)(L+z)) = q·B²/(B+Z)²). The module reproduces
    the published Table B1-2 stress fractions Δσv/q at every (B, layer): e.g.
    B = 3 m → 0.549/0.162/0.070/0.044 (pub 0.55/0.16/0.07/0.04), B = 6.1 m →
    0.728/0.334/0.179/0.123 (pub 0.73/0.33/0.18/0.12), to ≤2%."""
    pub = {
        3.0: [0.55, 0.16, 0.07, 0.04],
        4.6: [0.66, 0.26, 0.13, 0.08],
        6.1: [0.73, 0.33, 0.18, 0.12],
    }
    for B, fracs in pub.items():
        for Z, f_pub in zip(_V022_ZMID, fracs):
            f = approximate_2to1(1.0, B, B, Z)        # Δσv/q
            assert f == pytest.approx(f_pub, abs=0.01)


def test_v022_hough_worked_case_b3_q240():
    """PASS — the NEW `settlement.hough.hough_settlement` module method reproduces
    the example's worked single case (B = 3 m, q = 240 kPa). The Hough form
    dH = H/C'·log10[(σ'vo+Δσ)/σ'vo] uses the bearing-capacity index C' (distinct
    from Cc/(1+e0)) and the module's 2:1 stress increase. Per-layer 15.4 + 4.4 +
    1.1 + 0.6 = 21.5 mm vs the published 15 + 4 + 1 + 1 = 21 mm (source rounds each
    layer to mm). Driven through the module (not inline arithmetic)."""
    res = hough_settlement(_hough_layers(), q_net=240.0, B=3.0)
    per_layer = [lyr["settlement_mm"] for lyr in res.layers]
    # published per-layer (rounded): 15 / 4 / 1 / 1
    assert per_layer[0] == pytest.approx(15.0, abs=1.0)
    assert per_layer[1] == pytest.approx(4.0, abs=1.0)
    assert per_layer[2] == pytest.approx(1.0, abs=1.0)
    assert per_layer[3] == pytest.approx(1.0, abs=1.0)
    assert res.total_mm == pytest.approx(21.0, abs=1.5)   # pub 21 mm


def test_v022_hough_full_table_b1_3():
    """PASS — the full published Table B1-3 reproduced by the NEW module method
    `settlement.hough.hough_settlement`. Every cell (settlement vs applied stress
    q ∈ {240,290,335,380} kPa and footing width B ∈ {3,4.6,6.1} m) is reproduced
    to within ±15% (inventory tolerance); the largest relative spread is ~4.6% at
    B = 4.6, q = 240 (26.7 vs 28). Driven through the module method (not inline
    arithmetic). A regression check confirms the module result equals the inline
    closed form, so the validated layer/stress handling is the same."""
    pub = {
        (3.0, 240): 21, (3.0, 290): 25, (3.0, 335): 28, (3.0, 380): 30,
        (4.6, 240): 28, (4.6, 290): 31, (4.6, 335): 34, (4.6, 380): 37,
        (6.1, 240): 31, (6.1, 290): 35, (6.1, 335): 38, (6.1, 380): 41,
    }
    for (B, q), S_pub in pub.items():
        S = _hough_module_total_mm(B, float(q))       # NEW module method
        # module result must match the validated inline closed form exactly
        assert S == pytest.approx(_hough_total_mm(B, float(q)), rel=1e-9)
        # ±15% (inventory tolerance) OR ±1.5 mm for the small rounded values
        assert (S == pytest.approx(S_pub, rel=0.15)
                or S == pytest.approx(S_pub, abs=1.5)), \
            f"B={B}, q={q}: ours={S:.1f} mm vs pub {S_pub} mm"


def test_v022_settlement_module_exposes_hough_method():
    """Coverage gap CLOSED: the `settlement` module now exposes the Hough /
    C'-index granular method (`hough_settlement`, `HoughLayer`, `HoughResult`,
    `hough_settlement_layer`). Its index C' is the Hough bearing-capacity index
    (NOT Cc/(1+e0)); the 2:1 stress primitive is reused. The Cc/Cr e-log(p)
    consolidation path remains for cohesive soils."""
    import settlement
    names = dir(settlement)
    assert "hough_settlement" in names
    assert "HoughLayer" in names
    assert "HoughResult" in names
    assert "hough_settlement_layer" in names
    # the distinct consolidation form (Cc/(1+e0)) is still present
    assert "consolidation_settlement_layer" in names
    # the shared 2:1 stress primitive is still available and reused
    assert "approximate_2to1" in names
    from settlement import stress_distribution
    assert hasattr(stress_distribution, "approximate_2to1")

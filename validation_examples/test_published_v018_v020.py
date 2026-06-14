"""Phase E validation — ground_improvement (GEC-13) published benchmarks (V-018/019/020).

Sources (FHWA GEC-13 Vol 1, Ground Modification Methods Reference Manual):
  - V-018  Ch. 5 Sec 4.3.1 (Example Problem 1) — rammed aggregate pier (RAP)
           settlement by the two-layer (upper-zone stiffness-modulus) method.
           Embankment q = 125 pcf * 20 ft = 2,500 psf over a 15-ft soft clay on
           rock; clay gamma_sat = 120 pcf, WT at surface; Cc = 0.25, eo = 0.7;
           po(mid, 7.5 ft) = (120-62.4)*7.5 = 432 psf. RAP d = 2.75 ft, square
           spacing s = 5 ft, de = 1.05*s = 5.25 ft -> Ra = Ac/(pi/4*de^2) = 0.27;
           stress-concentration ns = 6; pier stiffness modulus kg = 65 pci.
           Published: unimproved S = 22 in; improved suz = qg/kg = 0.68 ~ 0.7 in.
           extract: extracts/g13_v1_ex.txt (PDF pp. 370-373).
  - V-019  Ch. 5 Sec 4.3.2 (Example Problem 2) + Figure 5-27 — stone-column
           settlement improvement by the Priebe (1995) improvement factor.
           Embankment q = 125 pcf * 15 ft = 1,875 psf over 50 ft soft clay
           (gamma_sat = 120 pcf, WT at surface; po(mid, 25 ft) = 1,440 psf;
           Cc = 0.2, eo = 0.6) over dense sand (no settlement). Stone columns
           d = 3.0 ft, spacing s = 5.7 ft -> A/Ac = (s/d)^2 = 3.6 (source). Chart
           settlement-improvement ratio = 2.7 -> improved S = 27/2.7 = 10 in.
           extract: extracts/g13_v1_ex.txt (PDF pp. 373-375).
  - V-020  Ch. 2 Sec 4.4 — PVD / wick-drain design example (Barron-Hansbo radial
           consolidation). 20 ft NC clay (sand lenses) over rock; cv = 0.1
           ft2/day, ch = 2*cv = 0.2 ft2/day; PVD dw = 2.5 in; triangular pattern
           (de = 1.05*s); target U = 90%; ideal drains (no smear / well
           resistance). Published: t90 ~ 300 days at s ~ 8 ft, ~ 500 days at
           s ~ 10 ft ("on the order of"). extract: extracts/g13v1_pvd.txt
           (PDF p. 116).

See validation_examples/INVENTORY.md (V-018, V-019, V-020) and RESULTS.md.

KEY FINDINGS (details in each test docstring and RESULTS.md):

- V-018 unimproved is a PASS, the improved RAP settlement is N/A (scope). The
  baseline consolidation S = Cc/(1+eo)*H*log10((po+dq)/po) = 22.0 in is reproduced
  EXACTLY by both the `settlement` module (`consolidation_settlement_layer`) and
  the closed form. The ground_improvement `area_replacement_ratio(triangular)`
  reproduces the example's Ra = Ac/(pi/4*de^2) = 0.2744 to 4 figures (because the
  triangular tributary sqrt(3)/2*s^2 numerically equals pi/4*(1.05 s)^2). BUT the
  RAP upper-zone *settlement* method (top-of-pier stress qg = q*ns/(Ra*ns-Ra+1),
  then suz = qg/kg with kg the pier stiffness modulus in pci) is NOT implemented in
  the module: `analyze_aggregate_piers`/`improved_settlement` use the equal-strain
  SETTLEMENT-REDUCTION-FACTOR model SRF = 1/(1+as*(n-1)), a different method that
  needs no kg. So the published improved 0.68 in is reproduced by hand on the
  module's `area_replacement_ratio` to document the convention/coverage gap; the
  module's own SRF path is shown for contrast (it gives a much larger improved S
  because SRF for as=0.27, n=6 is 0.43, i.e. ~9.4 in, not 0.7 in -- a fundamentally
  different model).

- V-019 unimproved is a PASS; the Priebe improvement ratio is a CONVENTION. The
  baseline S = 27.2 in is reproduced exactly. The module DOES expose the genuine
  Priebe (1995) basic improvement factor (`priebe_basic_improvement_factor`). The
  published ratio 2.7 is a CHART read (Figure 5-27, after Wallays et al. 1983) and
  the answer depends on (a) the area-ratio convention and (b) the column friction
  angle. With the SOURCE's area ratio as = Ac/A = 1/3.6 = 0.277 and the module's
  default phi_col = 42.5 deg, Priebe n0 = 3.06 (Δ +13%, outside ±0.3). With the
  module's OWN geometric triangular as = 0.251 (sqrt(3)/2 tributary) and phi_col =
  42.5, n0 = 2.806 -- WITHIN 2.7±0.3. The chart value 2.7 corresponds to phi_col
  ~ 39 deg at as = 0.277. So the formula is correct; the ~13% spread is the
  area-ratio-convention + phi_col + chart-read latitude, exactly as the inventory
  flagged. Module NOT tuned to the chart.

- V-020 is a PASS (radial-only ideal Barron). With ch only (no vertical, no smear,
  no well resistance) the module's Barron-Hansbo radial solution
  (`time_for_radial_consolidation` with F(n) = ln(n) - 0.75) gives t90 = 299 days
  at s = 8 ft and 503 days at s = 10 ft -- essentially exact vs the published
  ~300/~500 days ("on the order of"). Confirms the de = 1.05*s unit cell and the
  dw plumbing. The convention is RADIAL-ONLY: adding the module's combined
  vertical+radial term (U_total = 1-(1-Uv)(1-Ur)) over-predicts U at those times
  (~93%), so the source clearly used radial-only -- documented, not a bug.

Units: all three entries are US customary, converted to SI inline.
  1 ft = 0.3048 m, 1 in = 25.4 mm, 1 pcf = 0.157087 kN/m3, 1 psf = 0.04788 kPa,
  1 psi = 6.895 kPa, ft2/day -> m2/day * 0.092903 (-> m2/year * 365).
"""

import math

import pytest

# ground_improvement (all three entries)
from ground_improvement.aggregate_piers import (
    area_replacement_ratio, priebe_basic_improvement_factor,
    settlement_reduction_factor, improved_settlement,
)
from ground_improvement.wick_drains import (
    influence_diameter, drain_function_F, time_for_radial_consolidation,
    radial_time_factor, radial_degree_of_consolidation, analyze_wick_drains,
)
# settlement (tight unimproved-consolidation anchor for V-018 / V-019)
from settlement.consolidation import (
    ConsolidationLayer, consolidation_settlement_layer,
)

# Unit conversions (US -> SI)
FT = 0.3048
IN_MM = 25.4
PCF = 0.157087               # pcf -> kN/m3
PSF = 0.04788                # psf -> kPa
PSI = 6.895                  # psi -> kPa
M2_PER_FT2 = 0.092903        # ft2 -> m2


# ════════════════════════════════════════════════════════════════════════════
# V-018 : GEC-13 Ex 1 — rammed aggregate pier settlement (two-layer method)
# ════════════════════════════════════════════════════════════════════════════
#
# Soft clay H = 15 ft on rock; embankment dq = 125 pcf * 20 ft = 2500 psf;
# clay gamma_sat = 120 pcf, WT at surface -> po(7.5 ft) = (120-62.4)*7.5 = 432 psf;
# Cc = 0.25, eo = 0.7. RAP d = 2.75 ft, s = 5 ft, de = 1.05 s = 5.25 ft,
# Ra = Ac/(pi/4 de^2) = 0.27, ns = 6, kg = 65 pci.

_V018_CC = 0.25
_V018_EO = 0.7
_V018_H_FT = 15.0
_V018_PO_PSF = (120.0 - 62.4) * 7.5          # 432 psf
_V018_DQ_PSF = 125.0 * 20.0                  # 2500 psf
_V018_D_FT = 2.75
_V018_S_FT = 5.0
_V018_NS = 6.0
_V018_KG_PCI = 65.0


def test_v018_unimproved_consolidation_pass():
    """PASS: the unimproved (no ground improvement) consolidation settlement
    S = Cc/(1+eo)*H*log10((po+dq)/po) = 22.0 in is reproduced EXACTLY, both by the
    `settlement` module (`consolidation_settlement_layer`, NC clay) and by the
    closed form -- in SI. This is the tight closed-form anchor the inventory calls
    out, independent of any ground-improvement method."""
    # closed form (consistent in any unit since it is a log ratio * H)
    S_cf_in = (_V018_CC / (1.0 + _V018_EO) * _V018_H_FT
               * math.log10((_V018_PO_PSF + _V018_DQ_PSF) / _V018_PO_PSF)) * 12.0
    assert S_cf_in == pytest.approx(22.0, abs=0.5)

    # settlement module in SI: NC clay (sigma_p = sigma_v0), thickness/depth in m,
    # stresses in kPa
    layer = ConsolidationLayer(
        thickness=_V018_H_FT * FT, depth_to_center=7.5 * FT, e0=_V018_EO,
        Cc=_V018_CC, Cr=0.0, sigma_v0=_V018_PO_PSF * PSF,
        sigma_p=_V018_PO_PSF * PSF)
    S_m = consolidation_settlement_layer(layer, delta_sigma=_V018_DQ_PSF * PSF)
    S_in = S_m / FT * 12.0
    assert S_m == pytest.approx(0.559, abs=0.005)       # m
    assert S_in == pytest.approx(22.0, abs=0.5)         # in (pub 22)
    assert S_in == pytest.approx(S_cf_in, rel=1e-6)


def test_v018_area_replacement_ratio_matches_source():
    """PASS (primitive): the module `area_replacement_ratio(triangular)`
    reproduces the example's Ra = Ac/(pi/4*de^2) with de = 1.05*s = 0.2744 to 4
    figures. The triangular tributary sqrt(3)/2*s^2 numerically equals the
    de = 1.05 s unit-cell area pi/4*(1.05 s)^2 (both 21.65 ft^2 here), so the
    module's geometric `as` and the source's de-based Ra coincide."""
    d, s = _V018_D_FT * FT, _V018_S_FT * FT
    as_tri = area_replacement_ratio(d, s, "triangular")

    # source Ra = Ac / (pi/4 * de^2), de = 1.05 s
    Ac = math.pi / 4.0 * (_V018_D_FT * FT) ** 2
    de = 1.05 * (_V018_S_FT * FT)
    Ra_src = Ac / (math.pi / 4.0 * de ** 2)

    assert Ra_src == pytest.approx(0.2744, abs=0.001)
    assert as_tri == pytest.approx(Ra_src, abs=0.001)     # module == source Ra
    assert as_tri == pytest.approx(0.27, abs=0.01)        # pub Ra (rounded 0.27)


def test_v018_improved_rap_stiffness_modulus_is_scope_gap():
    """N/A (scope): the published improved RAP settlement uses the two-layer
    upper-zone STIFFNESS-MODULUS method -- top-of-pier stress
    qg = q*ns/(Ra*ns - Ra + 1), then suz = qg/kg (kg = pier stiffness modulus,
    pci) -- which the module does NOT implement. `analyze_aggregate_piers` /
    `improved_settlement` use the equal-strain SETTLEMENT-REDUCTION-FACTOR model
    SRF = 1/(1+as*(n-1)), a different method that needs no kg. We (1) reproduce the
    published qg / suz = 0.68 in by hand on the module's `area_replacement_ratio`,
    and (2) show the module's SRF path gives a fundamentally different (much
    larger) improved settlement -- documenting the coverage gap, not a bug."""
    # hand reproduction of the GEC-13 upper-zone method, using the module's `as`
    Ra = area_replacement_ratio(_V018_D_FT * FT, _V018_S_FT * FT, "triangular")
    q_psf = _V018_DQ_PSF
    qg_psf = q_psf * _V018_NS / (Ra * _V018_NS - Ra + 1.0)
    suz_in = (qg_psf / 144.0) / _V018_KG_PCI    # psf->psi /144, /pci -> inches
    # source rounds Ra to 0.27 (qg 6383); with as=0.2743 qg=6324 -> suz 0.676
    assert qg_psf == pytest.approx(6324.0, rel=0.02)
    assert suz_in == pytest.approx(0.68, abs=0.05)         # pub ~0.7 in
    # lower zone = 0 (rock) -> total improved ~ 0.7 in
    assert (suz_in + 0.0) == pytest.approx(0.7, abs=0.05)

    # SI form of the same hand calc (qg in kPa, kg in kPa/m): suz_m = qg/kg
    qg_kPa = qg_psf * PSF
    kg_kPa_per_m = _V018_KG_PCI * PSI / (IN_MM / 1000.0)   # pci -> kPa/m
    suz_m = qg_kPa / kg_kPa_per_m
    assert suz_m / FT * 12.0 == pytest.approx(suz_in, rel=0.01)

    # CONTRAST: the module's packaged SRF improved-settlement is a DIFFERENT model
    # and does NOT reproduce 0.7 in -- it has no kg, only as and n.
    srf = settlement_reduction_factor(Ra, _V018_NS)
    S_srf_in = improved_settlement(22.0, Ra, _V018_NS)     # 22 in unreinforced
    assert srf == pytest.approx(0.429, abs=0.01)
    assert S_srf_in == pytest.approx(9.4, abs=0.5)         # ~9.4 in, NOT 0.7
    assert S_srf_in > 5.0                                  # clearly different model


# ════════════════════════════════════════════════════════════════════════════
# V-019 : GEC-13 Ex 2 — stone-column settlement improvement (Priebe)
# ════════════════════════════════════════════════════════════════════════════
#
# Soft clay H = 50 ft on dense sand; embankment dq = 125 pcf * 15 ft = 1875 psf;
# clay gamma_sat = 120 pcf, WT at surface -> po(25 ft) = 1440 psf; Cc = 0.2,
# eo = 0.6. Stone columns d = 3.0 ft, s = 5.7 ft, A/Ac = (s/d)^2 = 3.6 (source).
# Chart settlement-improvement ratio = 2.7 -> improved S = 27/2.7 = 10 in.

_V019_CC = 0.2
_V019_EO = 0.6
_V019_H_FT = 50.0
_V019_PO_PSF = (120.0 - 62.4) * 25.0         # 1440 psf
_V019_DQ_PSF = 125.0 * 15.0                  # 1875 psf
_V019_D_FT = 3.0
_V019_S_FT = 5.7
_V019_RATIO_PUB = 2.7


def test_v019_unimproved_consolidation_pass():
    """PASS: the unimproved consolidation settlement
    S = Cc/(1+eo)*H*log10((po+dq)/po) = 27.2 in is reproduced EXACTLY by the
    `settlement` module and the closed form, in SI. Tight closed-form anchor."""
    S_cf_in = (_V019_CC / (1.0 + _V019_EO) * _V019_H_FT
               * math.log10((_V019_PO_PSF + _V019_DQ_PSF) / _V019_PO_PSF)) * 12.0
    assert S_cf_in == pytest.approx(27.0, abs=0.5)

    layer = ConsolidationLayer(
        thickness=_V019_H_FT * FT, depth_to_center=25.0 * FT, e0=_V019_EO,
        Cc=_V019_CC, Cr=0.0, sigma_v0=_V019_PO_PSF * PSF,
        sigma_p=_V019_PO_PSF * PSF)
    S_m = consolidation_settlement_layer(layer, delta_sigma=_V019_DQ_PSF * PSF)
    S_in = S_m / FT * 12.0
    assert S_m == pytest.approx(0.690, abs=0.005)       # m
    assert S_in == pytest.approx(27.2, abs=0.5)         # in (pub 27)
    assert S_in == pytest.approx(S_cf_in, rel=1e-6)


def test_v019_priebe_improvement_factor_convention():
    """CONVENTION: the module exposes the genuine Priebe (1995) basic improvement
    factor `priebe_basic_improvement_factor(as, phi_col, nu_s)`. The published
    ratio 2.7 is a CHART read (Figure 5-27, after Wallays et al. 1983), and the
    factor depends on (a) the area-ratio convention and (b) the column friction
    angle:

      - SOURCE area ratio as = Ac/A = 1/(s/d)^2 = 1/3.6 = 0.277, phi_col = 42.5
        (module default) -> Priebe n0 = 3.06 (Δ +13%, OUTSIDE 2.7±0.3).
      - module's OWN geometric triangular as = 0.251 (sqrt(3)/2 tributary),
        phi_col = 42.5 -> n0 = 2.806 (WITHIN 2.7±0.3).
      - the chart 2.7 corresponds to phi_col ~ 39 deg at as = 0.277.

    The formula is correct; the spread is the area-ratio-convention + phi_col +
    chart-read latitude the inventory flagged. The module is NOT tuned to the
    chart. Improved S = 27/n0 brackets the published 10 in (8.8-9.6 in)."""
    as_src = 1.0 / (_V019_S_FT / _V019_D_FT) ** 2          # Ac/A, source convention
    assert as_src == pytest.approx(0.277, abs=0.002)

    n0_src = priebe_basic_improvement_factor(as_src, phi_column=42.5)
    assert n0_src == pytest.approx(3.06, abs=0.05)         # default phi, OUTSIDE 0.3
    assert abs(n0_src - _V019_RATIO_PUB) > 0.3

    # module's own geometric triangular as -> within the soft tolerance
    as_tri = area_replacement_ratio(_V019_D_FT * FT, _V019_S_FT * FT, "triangular")
    n0_tri = priebe_basic_improvement_factor(as_tri, phi_column=42.5)
    assert as_tri == pytest.approx(0.251, abs=0.003)
    assert n0_tri == pytest.approx(2.81, abs=0.05)
    assert n0_tri == pytest.approx(_V019_RATIO_PUB, abs=0.3)   # within 2.7±0.3

    # the chart 2.7 corresponds to a softer column phi (~39 deg) at as=0.277
    n0_phi39 = priebe_basic_improvement_factor(as_src, phi_column=39.0)
    assert n0_phi39 == pytest.approx(2.7, abs=0.15)

    # improved settlement = 27 in / n0 brackets the published 10 in
    S_improved_src = 27.0 / n0_src
    S_improved_tri = 27.0 / n0_tri
    assert S_improved_src == pytest.approx(8.8, abs=0.5)
    assert S_improved_tri == pytest.approx(9.6, abs=0.5)
    # published improved 10 in (27/2.7); ours bracket it from below
    assert 8.0 < S_improved_tri < 11.0


# ════════════════════════════════════════════════════════════════════════════
# V-020 : GEC-13 Ch. 2 PVD example — t90 vs drain spacing (Barron-Hansbo radial)
# ════════════════════════════════════════════════════════════════════════════
#
# 20 ft NC clay (sand lenses) over rock; cv = 0.1 ft2/day, ch = 0.2 ft2/day;
# dw = 2.5 in; triangular (de = 1.05 s); target U = 90%; ideal drains.
# Published: t90 ~ 300 days at s ~ 8 ft, ~ 500 days at s ~ 10 ft ("on the order of").

_V020_CH_M2_DAY = 0.2 * M2_PER_FT2          # 0.018581 m2/day
_V020_CV_M2_DAY = 0.1 * M2_PER_FT2          # 0.009290 m2/day
_V020_DW_M = 2.5 * IN_MM / 1000.0           # 0.0635 m (2.5 in)
_V020_H_FT = 20.0


def test_v020_radial_only_t90_pass():
    """PASS: with RADIAL-ONLY ideal Barron-Hansbo (no vertical drainage, no smear,
    no well resistance), the module reproduces the published t90 ~ 300 days at
    s ~ 8 ft and ~ 500 days at s ~ 10 ft to <1% (well inside ±20%). Uses the
    module primitives `influence_diameter` (de = 1.05 s), `drain_function_F`
    (F(n) = ln(n) - 0.75 for the ideal case), and `time_for_radial_consolidation`
    (t = -F/8 * ln(1-U) * de^2 / ch). Units carried in m2/day and days."""
    expected = {8.0: 300.0, 10.0: 500.0}
    got = {}
    for s_ft in (8.0, 10.0):
        s = s_ft * FT
        de = influence_diameter(s, "triangular")          # 1.05 s
        n = de / _V020_DW_M
        F_n = drain_function_F(n, smear_ratio=1.0, kh_ks_ratio=1.0)   # ln(n)-0.75
        t_days = time_for_radial_consolidation(90.0, _V020_CH_M2_DAY, de, F_n)
        got[s_ft] = t_days
        # sanity on n and F(n) (independent of unit system)
        assert n == pytest.approx({8.0: 40.3, 10.0: 50.4}[s_ft], abs=0.5)
        # ±20% on the "on the order of" published values
        assert t_days == pytest.approx(expected[s_ft], rel=0.20)

    # our values are essentially exact vs the published ~300 / ~500
    assert got[8.0] == pytest.approx(299.0, abs=5.0)
    assert got[10.0] == pytest.approx(503.0, abs=5.0)
    # spacing increase lengthens t90 (de^2 * F(n) both grow)
    assert got[10.0] > got[8.0]


def test_v020_de_unit_cell_and_dw_plumbing():
    """Pins the de = 1.05*s triangular unit cell and the dw plumbing that drive
    V-020: de(8 ft) = 8.40 ft = 2.560 m, de(10 ft) = 10.50 ft = 3.200 m;
    dw = 2.5 in = 0.0635 m; n = de/dw = 40.3 / 50.4. These are the geometry inputs
    the radial solution consumes."""
    de8 = influence_diameter(8.0 * FT, "triangular")
    de10 = influence_diameter(10.0 * FT, "triangular")
    assert de8 == pytest.approx(8.40 * FT, rel=1e-6)       # 2.560 m
    assert de10 == pytest.approx(10.50 * FT, rel=1e-6)     # 3.200 m
    assert _V020_DW_M == pytest.approx(0.0635, abs=1e-4)
    assert de8 / _V020_DW_M == pytest.approx(40.3, abs=0.3)
    assert de10 / _V020_DW_M == pytest.approx(50.4, abs=0.3)


def test_v020_combined_vertical_radial_overpredicts():
    """CONVENTION documentation: the source used RADIAL-ONLY. Adding the module's
    combined vertical+radial term U_total = 1-(1-Uv)(1-Ur) over-predicts U at the
    published times -- at s = 8 ft, t = 300 days the combined U_total ~ 93% (vs the
    target 90% from radial alone), because the 20-ft clay over rock (single
    drainage, Hdr = 20 ft) contributes ~31% vertical consolidation on its own. So
    the published t90 is the radial-only time; the combined model would give a
    SHORTER t90. Documents which convention the example used (not a bug)."""
    s = 8.0 * FT
    # run the packaged combined analysis at t = 300 days (= 300/365 yr), ideal drains
    ch_yr = _V020_CH_M2_DAY * 365.0
    cv_yr = _V020_CV_M2_DAY * 365.0
    Hdr = _V020_H_FT * FT                                  # single drainage over rock
    res = analyze_wick_drains(
        spacing=s, ch=ch_yr, cv=cv_yr, Hdr=Hdr, time=300.0 / 365.0,
        dw=_V020_DW_M, pattern="triangular",
        smear_ratio=1.0, kh_ks_ratio=1.0)
    # radial alone hits ~90% at 300 days (the published target)
    assert res.Ur_percent == pytest.approx(90.0, abs=1.5)
    # but vertical adds ~31%, so the COMBINED U_total exceeds 90%
    assert res.Uv_percent == pytest.approx(31.0, abs=3.0)
    assert res.U_total_percent > 92.0
    assert res.U_total_percent == pytest.approx(93.1, abs=1.0)

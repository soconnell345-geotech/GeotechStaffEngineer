"""V-074 to V-076: three more printed examples of the AASHTO 1993 Guide,
end to end through pavement_design (GM-298). Read from
``geotech-references/docs/aashto1993.pdf`` (a scan; pdf_page is the
0-based page index).

V-074, the effective modulus of subgrade reaction (Section 3.2.1): the
Figure 3.3 example (pdf_page 128, printed II-39: Dsb 6 in, Esb 20,000 psi,
MR 7,000 psi -> k_inf 400 pci) and the Table 3.3 worksheet (pdf_page 132,
printed II-43): granular subbase 6 in, depth to rigid foundation 5 ft,
projected slab 9 in, LS 1.0; per month MR / Esb / composite k (Fig 3.3) /
k on the rigid foundation (Fig 3.4) / u_r (Fig 3.5): 20,000 / 50,000 /
1,100 / 1,350 / 0.35 (Jan, Feb, Dec); 2,500 / 15,000 / 160 / 230 / 0.86
(Mar); 4,000 / 15,000 / 230 / 300 / 0.78 (Apr, May, Nov); 7,000 / 20,000 /
410 / 540 / 0.60 (June-Oct); sum u_r 7.25, average 0.60, effective k 540
pci, corrected for loss of support 170 pci. The guide reads the example's
point twice, 400 on the figure and 410 in the table.

V-075, Appendix D Example 1 (pdf_pages 381-384, printed D-22 to D-25):
Table D.20's growth factors [(1 + g)^n - 1]/g and Table D.21's worksheet,
13 vehicle classes, current two-way daily traffic times the 20-year, 2 %
growth factor 24.30 times 365 times each class's truck factor; design
ESAL for both directions (D-22: "the total for both directions of
travel"). Every row of column E is its own product to the ESAL, and the
column adds to 43,797,514; the total line prints 43,772,314, 25,200
(0.06 %) below its own column, a slip in the print (the text, D-22, says
"43.8 million").

V-076, Appendix H, the flexible pavement design example (pdf_pages
406-412, printed H-1 to H-7): first-year two-way traffic 2.5 x 10^6
ESAL, DD 0.50, DL 0.80 (three lanes one way), 3 % compound growth ->
W18 = 1.0 x 10^6 [(1 + g)^t - 1]/g (Fig H.1), 18.6 x 10^6 at 15 years;
24 half-month roadbed moduli (Fig H.3) -> sum u 5.446, average 0.227,
effective MR 5,700 psi; R 95 % (0.90^(1/2), two stages), So 0.35, dPSI
4.6 - 2.5 = 2.1 -> maximum initial SN 5.6; Table H.2 (swelling, VR 1.2
in, Ps 84 %, theta 0.075): trial 13 years -> dPSI_SW 0.21, dPSI_TR 1.89,
allowable 16.0 x 10^6 ESAL, 13.2 years; then with EAC 400,000 (a1 0.42),
EBS 30,000 (a2 0.14), ESB 11,000 psi (a3 0.08) and m 1.20 ("good", Table
2.4's 1.15 to 1.25), W18 16.0 x 10^6 and dPSI 1.89: SN1 3.2, D1* =
3.2/0.42 = 7.6 (8 in), SN1* 3.36; SN2 4.5, D2* = (4.5 - 3.36)/(0.14 x
1.20) = 6.8 (7 in), SN2* 1.18; D3* = (5.6 - 4.54)/(0.08 x 1.20) = 11 in.
The guide rounds D3* (11.04) to the nearest inch; the module rounds UP
(Figure 3.2 note 2 asks the value used to be at least the value
required), so at whole inches it gives 12 in.
"""

import pytest

from pavement_design import (compute_design_esals, design_flexible_pavement,
                             design_rigid_pavement,
                             estimate_performance_period, growth_factor)
from pavement_design.composite_k import composite_k_inf, effective_k

# --------------------------------------------------------------------------
# V-074: Table 3.3 and Figure 3.3
# --------------------------------------------------------------------------

_TABLE_3_3 = ([(20000, 50000)] * 2 + [(2500, 15000)] + [(4000, 15000)] * 2
              + [(7000, 20000)] * 5 + [(4000, 15000)] + [(20000, 50000)])
_SEASONS = [{"mr_psi": mr, "esb_psi": esb} for mr, esb in _TABLE_3_3]
# Per (MR, Esb): printed composite k, k on the rigid foundation, u_r.
_ROWS = {(20000, 50000): (1100, 1350, 0.35), (2500, 15000): (160, 230, 0.86),
         (4000, 15000): (230, 300, 0.78), (7000, 20000): (410, 540, 0.60)}


def test_v074_figure_3_3_example():
    assert composite_k_inf(7000, 20000, 6)["k_inf_pci"] == pytest.approx(
        400, rel=0.03)


def test_v074_table_3_3_rows():
    res = effective_k(_SEASONS, 9, dsb_in=6, dsg_ft=5, ls=1.0)
    for row in res["rows"]:
        k, k_rigid, ur = _ROWS[(row["mr_psi"], row["esb_psi"])]
        assert row["composite_k_pci"] == pytest.approx(k, rel=0.05)
        assert row["k_rigid_foundation_pci"] == pytest.approx(k_rigid,
                                                              rel=0.05)
        assert row["u_r"] == pytest.approx(ur, abs=0.02)


def test_v074_table_3_3_effective_k():
    res = effective_k(_SEASONS, 9, dsb_in=6, dsg_ft=5, ls=1.0)
    assert res["ur_sum"] == pytest.approx(7.25, rel=0.02)
    assert res["ur_avg"] == pytest.approx(0.60, abs=0.015)
    assert res["effective_k_pci"] == pytest.approx(540, rel=0.04)
    assert res["k_corrected_for_loss_of_support_pci"] == pytest.approx(
        170, rel=0.04)


def test_v074_through_the_rigid_design():
    res = design_rigid_pavement(
        w18=5.1e6, sc_psi=650, ec_psi=5e6, reliability_pct=95, so=0.29,
        delta_psi=1.7, j=3.2, cd=1.0, slab_thickness_in=9.0,
        composite_k={"seasonal": _SEASONS, "dsb_in": 6, "dsg_ft": 5,
                     "ls": 1.0})
    assert res.k_pci == pytest.approx(170, rel=0.04)
    assert res.k_basis["effective_k_pci"] == pytest.approx(540, rel=0.04)


# --------------------------------------------------------------------------
# V-075: Appendix D, Tables D.20 and D.21
# --------------------------------------------------------------------------

# (growth %, years, printed factor), Table D.20 (pdf_page 383, printed D-24).
_TABLE_D20 = [(0, 12, 12.0), (2, 20, 24.30), (4, 10, 12.01), (4, 20, 29.78),
              (5, 15, 21.58), (6, 25, 54.86), (7, 30, 94.46), (8, 5, 5.87),
              (10, 35, 271.02)]

# Table D.21 (pdf_page 384, printed D-25): class, current two-way daily
# traffic (A), truck factor (D), printed design ESAL (E).
_TABLE_D21 = [
    ("Passenger cars", 5925, 0.0008, 42041),
    ("Buses", 35, 0.6806, 211280),
    ("Panel and pickup trucks", 1135, 0.0122, 122816),
    ("Other 2-axle/4-tire trucks", 3, 0.0052, 138),
    ("2-axle/6-tire trucks", 372, 0.1890, 623597),
    ("3 or more axle trucks", 34, 0.1303, 39294),
    ("3 axle tractor semi-trailers", 19, 0.8646, 145703),
    ("4 axle tractor semi-trailers", 49, 0.6560, 285101),
    ("5+ axle tractor semi-trailers", 1880, 2.3719, 39550626),
    ("5 axle double trailers", 103, 2.3187, 2118268),
    ("3 axle truck-trailers", 208, 0.0152, 28042),
    ("4 axle truck-trailers", 305, 0.0152, 41119),
    ("5+ axle truck-trailers", 125, 0.5317, 589489),
]


@pytest.mark.parametrize("g, n, printed", _TABLE_D20)
def test_v075_table_d20_growth_factors(g, n, printed):
    assert growth_factor(g, n)["growth_factor"] == pytest.approx(
        printed, abs=0.006)


def test_v075_table_d21_worksheet():
    res = compute_design_esals(
        growth_rate_pct=2.0, design_period_yr=20,
        vehicles=[{"description": d, "daily_count": a, "truck_factor": tf}
                  for d, a, tf, _ in _TABLE_D21],
        directional_factor=0.5, lane_factor=1.0)
    # Column E adds to 43,797,514; the total line prints 43,772,314.
    assert sum(e for *_, e in _TABLE_D21) == 43797514
    # The module's GF is 24.2974 where the worksheet uses 24.30.
    assert res.w18_two_way_total == pytest.approx(43797514, rel=5e-4)
    assert res.w18_two_way_total == pytest.approx(43772314, rel=1e-3)
    for (desc, _, _, e), row in zip(_TABLE_D21, res.vehicle_breakdown):
        assert row["description"] == desc
        assert row["esals_per_year"] * res.growth_factor == pytest.approx(
            e, rel=5e-4, abs=1.0)


# --------------------------------------------------------------------------
# V-076: Appendix H, the flexible pavement design example
# --------------------------------------------------------------------------

# Figure H.3 (pdf_page 411, printed H-6), Jan 1st half to Dec 2nd half.
_FIG_H3 = ([6500, 20000, 20000, 4000] + [5000] * 6 + [6500] * 6
           + [6500, 5000, 5000, 5000, 5000, 6500, 6500, 6500])
_SWELL = {"vr_in": 1.2, "ps_pct": 84, "theta": 0.075}
_GOOD = {"drainage_quality": "good", "pct_saturation_time": "1-5%"}


def test_v076_traffic_figure_h1():
    res = compute_design_esals(
        base_year_w18_two_way=2.5e6, growth_rate_pct=3.0,
        design_period_yr=15, directional_factor=0.50, lane_factor=0.80)
    assert res.w18_design_lane == pytest.approx(18.6e6, rel=0.003)


def test_v076_effective_mr_figure_h3():
    res = design_flexible_pavement(
        w18=18.6e6, reliability_pct=95, so=0.35, delta_psi=2.1,
        monthly_mr_psi=_FIG_H3, layers=[{"layer_type": "asphalt",
                                         "a": 0.42}])
    assert res.effective_mr_detail["n_seasons"] == 24
    assert res.effective_mr_detail["uf_sum"] == pytest.approx(5.446,
                                                              rel=0.01)
    assert res.effective_mr_psi == pytest.approx(5700, rel=0.01)
    # The maximum initial SN (nomograph read to 0.1).
    assert res.sn_required == pytest.approx(5.6, abs=0.1)


def test_v076_performance_period_table_h2():
    res = estimate_performance_period(
        "flexible", delta_psi_design=2.1, base_year_w18=1.0e6,
        growth_rate_pct=3.0, swelling=_SWELL, max_performance_period_yr=15,
        initial_trial_yr=13, sn=5.6, mr_psi=5700, zr=-1.645, so=0.35)
    first = res["rows"][0]
    assert first["delta_psi_env"] == pytest.approx(0.21, abs=0.005)
    assert first["delta_psi_traffic"] == pytest.approx(1.89, abs=0.005)
    assert first["w18_cumulative"] == pytest.approx(16.0e6, rel=0.01)
    assert res["converged"]
    assert res["performance_period_yr"] == pytest.approx(13.2, abs=0.1)


def test_v076_layer_thicknesses():
    res = design_flexible_pavement(
        w18=16.0e6, reliability_pct=95, so=0.35, delta_psi=1.89,
        mr_psi=5700, thickness_increment_in=1.0,
        layers=[{"layer_type": "asphalt", "a": 0.42},
                {"layer_type": "granular_base", "a": 0.14,
                 "modulus_psi": 30000, **_GOOD},
                {"layer_type": "granular_subbase", "a": 0.08,
                 "modulus_psi": 11000, **_GOOD}])
    sn = [row["sn_required"] for row in res.sn_stack]
    assert sn == pytest.approx([3.2, 4.5, 5.6], abs=0.05)
    base, subbase = res.layers[1], res.layers[2]
    assert base["m"] == pytest.approx(1.20) and subbase["m"] == 1.20
    d = [lay["thickness_in"] for lay in res.layers]
    assert d[:2] == [8.0, 7.0]
    assert res.layers[0]["d_unrounded_in"] == pytest.approx(7.6, abs=0.1)
    assert base["d_unrounded_in"] == pytest.approx(6.8, abs=0.25)
    assert subbase["d_unrounded_in"] == pytest.approx(11.0, abs=0.15)
    # Rounded UP, where the guide takes the nearest inch (11.04 -> 11).
    assert d[2] == 12.0
    assert res.adequate

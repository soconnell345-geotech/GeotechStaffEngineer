"""GM-291: ground_improvement against GEC-13 (FHWA-NHI-16-027) as printed.

Wick drains: GEC-13 Vol. I, Ch. 2, Sec. 4.4's design example (p. 2-38):
20 ft of clay, cv 0.1 ft2/day, ch 2 cv, a 2.5 in drain, triangular;
Eq. 2-1 "indicates required times on the order of 300 days at an
approximate spacing of 8 feet or 500 days at an approximate spacing of 10
feet" for 90 percent consolidation. By hand: d_c = 1.05 s (p. 2-33), n =
d_c / d_w, F(n) = ln(n) - 0.75 (Eq. 2-2), t = d_c^2 / (8 c_h) F(n)
ln(1 / 0.1) (Eq. 2-1): 8 ft, n 40.32, F 2.947, t 299.2 days; 10 ft, n
50.4, F 3.170, t 503.0 days. The tool's defaults (a 2.5 in drain, smear
ignored, GEC-13 p. 2-35) are the example's.

Vibro-compaction: the fines screen (p. 4-65, p. 4-67), SPT N to relative
density by Table 4-8 (p. 4-74), and the probe spacing read off Fig. 4-34
(p. 4-77). The feasibility screening uses the same limits, and GEC-13 p.
5-9's undrained strength limits for aggregate columns. The adapter
refuses the D50 screen and the target SPT N, neither printed.
"""

import json
import math

import pytest

from ground_improvement.feasibility import evaluate_feasibility
from ground_improvement.vibro import (
    FIG_4_34, analyze_vibro_compaction, density_from_N,
    probe_spacing_fig_4_34, vibro_feasibility,
)
from ground_improvement.wick_drains import (
    DW_DEFAULT_M, analyze_wick_drains, drain_function_F, influence_diameter,
    smear_factor, time_for_radial_consolidation,
)

FT = 0.3048
DAY_YR = 1.0 / 365.25
CH_FT2_DAY = 2 * 0.1          # "ch of 2 times cv", cv 0.1 ft2/day
DW_FT = 2.5 / 12.0            # "an equivalent diameter of drain of 2.5 inches"


def _t90_days(spacing_ft):
    de = influence_diameter(spacing_ft, "triangular")
    return time_for_radial_consolidation(
        90.0, ch=CH_FT2_DAY, de=de, F_n=drain_function_F(de / DW_FT))


@pytest.mark.parametrize("spacing_ft, printed, by_hand", [
    (8.0, 300.0, 299.2), (10.0, 500.0, 503.0)])
def test_the_design_example_times(spacing_ft, printed, by_hand):
    """GEC-13 p. 2-38: 'on the order of 300 days at an approximate spacing
    of 8 feet or 500 days at an approximate spacing of 10 feet'."""
    t = _t90_days(spacing_ft)
    assert t == pytest.approx(by_hand, abs=0.2)
    assert t == pytest.approx(printed, rel=0.01)


def test_the_defaults_are_the_example_drain():
    """The tool's defaults run the example as printed: a 2.5 in drain and
    no smear (GEC-13 p. 2-35); at 299.2 days, U_r is 90 percent."""
    assert DW_DEFAULT_M == pytest.approx(2.5 * 0.0254)
    ch = CH_FT2_DAY * FT ** 2 / DAY_YR          # m2/year
    r = analyze_wick_drains(spacing=8 * FT, ch=ch, cv=ch / 2,
                            Hdr=20 * FT, time=_t90_days(8.0) * DAY_YR)
    assert r.Ur_percent == pytest.approx(90.0, abs=0.05)
    assert r.F_n == pytest.approx(math.log(r.spacing_ratio_n) - 0.75)
    assert r.equivalent_drain_diameter_m == pytest.approx(0.0635, abs=1e-4)


def test_the_smear_factor_as_ufc_prints_it():
    """UFC 3-220-10 Eq. 5-25, F_s = (kh/ks) ln(s); 0 with no smear
    (made-up inputs)."""
    assert smear_factor(3.0, 2.0) == pytest.approx(2.0 * math.log(3.0))
    assert smear_factor(1.0, 5.0) == 0.0
    assert drain_function_F(30.0, 3.0, 2.0) == pytest.approx(
        math.log(30.0) - 0.75 + 2.0 * math.log(3.0))


def test_fig_4_34_as_read():
    """The nodes read off GEC-13 Fig. 4-34 (p. 4-77), the curve from 90
    percent at 1.40 m to 55 percent at 2.50 m, falling throughout."""
    assert FIG_4_34[55.0] == 2.50 and FIG_4_34[90.0] == 1.40
    assert FIG_4_34[60.0] == 2.30 and FIG_4_34[80.0] == 1.67
    xs = sorted(FIG_4_34)
    assert all(FIG_4_34[a] > FIG_4_34[b] for a, b in zip(xs, xs[1:]))
    assert probe_spacing_fig_4_34(72.5) == pytest.approx(
        (FIG_4_34[70.0] + FIG_4_34[75.0]) / 2)


@pytest.mark.parametrize("dr", [50.0, 95.0])
def test_fig_4_34_is_not_extended(dr):
    with pytest.raises(ValueError, match="Fig. 4-34"):
        probe_spacing_fig_4_34(dr)


@pytest.mark.parametrize("N, name, dr", [
    (3, "very loose", (0, 15)), (4, "very loose", (0, 15)),
    (8, "loose", (15, 35)), (20, "medium dense", (35, 65)),
    (40, "dense", (65, 85)), (55, "very dense", (85, 100))])
def test_table_4_8(N, name, dr):
    """GEC-13 Table 4-8 (p. 4-74), normally consolidated sand."""
    assert density_from_N(N) == (name, dr)


@pytest.mark.parametrize("fines, feasible, words", [
    (8.0, True, "below 10 percent"),
    (11.0, True, "zone B"),
    (14.0, True, "marginal"),
    (16.0, False, "cannot be achieved")])
def test_the_fines_screen(fines, feasible, words):
    """GEC-13 p. 4-67 (zone B below 10 percent) and p. 4-65 (more than 12
    to 15 percent silt): until GM-291 the screen took 15 to 20 percent as
    marginal and refused above 20, printed nowhere."""
    ok, reasons = vibro_feasibility(fines, initial_N_spt=8)
    assert ok is feasible
    assert any(words in r for r in reasons)


def test_a_very_dense_sand_needs_no_densification():
    """N above 50 is very dense, D_r 85 to 100 percent (Table 4-8); p.
    4-75: unlikely to need D_r above 85 percent. Until GM-291 any N above
    25 was refused."""
    assert vibro_feasibility(5.0, initial_N_spt=30)[0] is True
    ok, reasons = vibro_feasibility(5.0, initial_N_spt=55)
    assert ok is False
    assert any("85 percent" in r for r in reasons)


def test_the_spacing_needs_a_target():
    r = analyze_vibro_compaction(fines_content=5.0, initial_N_spt=8)
    assert r.recommended_spacing_m == 0.0
    assert any("target_relative_density" in x for x in r.reasons)
    r = analyze_vibro_compaction(fines_content=5.0, initial_N_spt=8,
                                 target_relative_density=60.0)
    assert r.recommended_spacing_m == pytest.approx(2.30)
    assert r.to_dict()["target_relative_density_percent"] == 60.0


def test_the_feasibility_screen_uses_the_same_limits():
    def vibro(**kw):
        res = evaluate_feasibility(soil_type="loose_sand", **kw)
        out = {d["method"]: d["reason"] for d in res.not_applicable}
        return res, out.get("Vibro-Compaction")

    res, why = vibro(fines_content=14.0, N_spt=8)
    assert why is None
    assert any("marginal" in r for r in res.recommendations)
    assert "Fig. 4-34" in res.preliminary_sizing["vibro_compaction"][
        "typical_spacing"]
    assert "12 to 15 percent" in vibro(fines_content=16.0, N_spt=8)[1]
    assert "85" in vibro(fines_content=5.0, N_spt=55)[1]


def test_the_printed_typical_sizes():
    """GEC-13 p. 2-33 (PVD spacing), p. 2-35 (drain diameter), p. 5-31
    (stone column diameter and spacing)."""
    res = evaluate_feasibility(
        soil_type="soft_clay", cu_kPa=30.0, thickness_m=6.0,
        predicted_settlement_mm=120.0, allowable_settlement_mm=50.0,
        time_constraint_months=12)
    sz = res.preliminary_sizing
    assert "3 to 5 ft" in sz["wick_drains"]["typical_spacing"]
    assert "2.5 in" in sz["wick_drains"]["typical_dw"]
    assert "5 to 12 ft" in sz["aggregate_piers"]["typical_spacing"]
    assert "2.0 to 5.0" in sz["aggregate_piers"]["typical_n"]
    assert "typical_surcharge_height" not in sz["surcharge"]


@pytest.mark.parametrize("name", ["D50", "target_N_spt"])
def test_the_adapter_refuses_what_gec13_does_not_print(name):
    from funhouse_agent import dispatch as D
    out = D.call_agent("ground_improvement", "vibro_compaction",
                       {"fines_content": 5.0, "initial_N_spt": 8,
                        name: 0.3 if name == "D50" else 25})
    out = json.loads(out) if isinstance(out, str) else out
    assert "error" in out
    assert "GEC-13" in json.dumps(out)

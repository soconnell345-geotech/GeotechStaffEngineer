"""GEC-13 Fig. 5-21's settlement ratio by E_c / E_s (GM-220).

GEC-13 (FHWA-NHI-16-027) Vol. I, Fig. 5-21, p. 5-36: the settlement ratio
n' (settlement without stone columns over settlement with them) against
A/A_c. The weighted-modulus curves (2a) E_c/E_s = 10 and (2b) 5 and
Balaam's (5a) 10 flexible, (5b) 10 rigid and (5c) 5 rigid, pinned to the
points read off the printed figure (each curve traced on the scanned page,
calibrated on its grid lines and checked by overlay, 2026-10-19).
"""

import math

import pytest

from ground_improvement import analyze_aggregate_piers
from ground_improvement.calc_steps import (get_calc_steps,
                                           get_input_summary)
from ground_improvement.settlement_ratio import (
    BALAAM_FLEXIBLE_10, BALAAM_RIGID_5, BALAAM_RIGID_10,
    balaam_settlement_ratio, settlement_ratio,
    weighted_modulus_settlement_ratio)

#: Curve (2a), E_c/E_s = 10, and (2b), 5, as traced: A/A_c -> n'.
PRINTED_2A = {2.0: 5.495, 3.0: 4.039, 4.0: 3.253, 5.0: 2.776,
              6.0: 2.457, 7.0: 2.237, 8.0: 2.082}
PRINTED_2B = {1.5: 3.654, 2.0: 3.019, 2.5: 2.608, 3.0: 2.338, 3.5: 2.138}

#: Balaam's curves read on the printed grid lines A/A_c = 2 to 8.
PRINTED_5B = {2.0: 4.35, 3.0: 3.15, 4.0: 2.59, 5.0: 2.24, 6.0: 2.02,
              7.0: 1.83, 8.0: 1.65}
PRINTED_5C = {2.0: 2.67, 3.0: 2.04, 4.0: 1.71, 5.0: 1.52, 6.0: 1.46,
              7.0: 1.42, 8.0: 1.38}
PRINTED_5A = {2.0: 4.04, 3.0: 2.76, 4.0: 2.26, 5.0: 1.96, 6.0: 1.79,
              7.0: 1.65, 8.0: 1.50}
#: ... and between the grid lines, off the traced curves
TRACED_BETWEEN = {
    "5b": {2.4: 3.720, 3.75: 2.712, 4.75: 2.316, 7.25: 1.789},
    "5c": {2.4: 2.365, 3.75: 1.776, 4.75: 1.555, 7.25: 1.416},
    "5a": {2.4: 3.371, 3.75: 2.358, 4.75: 2.015, 7.25: 1.609},
}
_CURVE = {"5b": (10.0, "rigid"), "5c": (5.0, "rigid"),
          "5a": (10.0, "flexible")}


@pytest.mark.parametrize("ratio, printed", [(10.0, PRINTED_2A),
                                            (5.0, PRINTED_2B)])
def test_the_weighted_modulus_is_the_printed_curve_2(ratio, printed):
    for x, n in printed.items():
        assert weighted_modulus_settlement_ratio(1 / x, ratio) \
            == pytest.approx(n, rel=0.025), x


@pytest.mark.parametrize("key, printed", [("5b", PRINTED_5B),
                                          ("5c", PRINTED_5C),
                                          ("5a", PRINTED_5A)])
def test_balaam_reads_the_printed_curves(key, printed):
    ratio, footing = _CURVE[key]
    for x, n in printed.items():
        assert balaam_settlement_ratio(1 / x, ratio, footing) \
            == pytest.approx(n, abs=0.005), x
    for x, n in TRACED_BETWEEN[key].items():
        assert balaam_settlement_ratio(1 / x, ratio, footing) \
            == pytest.approx(n, rel=0.005), x


def test_the_curves_lie_as_printed():
    for x in (2.0, 3.0, 5.0, 8.0):
        a = 1 / x
        rigid_10 = balaam_settlement_ratio(a, 10.0, "rigid")
        # flexible below rigid, E_c/E_s = 5 below 10, Balaam below the
        # weighted modulus at the same E_c/E_s
        assert balaam_settlement_ratio(a, 10.0, "flexible") < rigid_10
        assert balaam_settlement_ratio(a, 5.0, "rigid") < rigid_10
        assert rigid_10 < weighted_modulus_settlement_ratio(a, 10.0)
        assert balaam_settlement_ratio(a, 5.0, "rigid") \
            < weighted_modulus_settlement_ratio(a, 5.0)
    # each curve falls as the columns spread out
    for table in (BALAAM_RIGID_10, BALAAM_RIGID_5, BALAAM_FLEXIBLE_10):
        ys = [table[x] for x in sorted(table)]
        assert all(b < a for a, b in zip(ys, ys[1:]))


def test_a_rigid_ratio_between_5_and_10_is_between_the_curves():
    a = 1 / 4.0
    n = balaam_settlement_ratio(a, 7.5, "rigid")
    assert n == pytest.approx((PRINTED_5C[4.0] + PRINTED_5B[4.0]) / 2)
    # a ratio given to within 1 % of a printed one is read on that curve
    assert balaam_settlement_ratio(a, 9.95, "rigid") == pytest.approx(
        PRINTED_5B[4.0])


@pytest.mark.parametrize("args, match", [
    ((1 / 4.0, 16.0, "rigid"), "outside them"),
    ((1 / 4.0, 4.0, "rigid"), "outside them"),
    ((1 / 4.0, 5.0, "flexible"), "10 only"),
    ((1 / 12.0, 10.0, "rigid"), "off the curve"),
    ((1 / 1.6, 10.0, "rigid"), "off the curve"),
    ((1 / 9.0, 10.0, "flexible"), "off the curve"),
    ((1 / 4.0, 0.8, "rigid"), "stiffer than the soil"),
])
def test_off_the_printed_curves_is_refused(args, match):
    with pytest.raises(ValueError, match=match):
        balaam_settlement_ratio(*args)


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="must be one of"):
        settlement_ratio("priebe", 0.2, 10.0)
    with pytest.raises(ValueError, match="must be one of"):
        analyze_aggregate_piers(column_diameter=0.9, spacing=2.1,
                                E_column=50000.0, settlement_method="x")


# ------------------------------------------------- analyze_aggregate_piers

_BASE = dict(column_diameter=0.9, spacing=2.1, S_unreinforced=150.0)


def test_the_default_is_the_stress_ratio_and_its_numbers_hold():
    r = analyze_aggregate_piers(**_BASE)
    a_s = math.pi * 0.9 ** 2 / 4 / (math.sqrt(3) / 2 * 2.1 ** 2)
    assert r.settlement_reduction_factor == pytest.approx(
        1 / (1 + 4 * a_s))
    assert r.settlement_improved_mm == pytest.approx(90.02, abs=0.01)
    assert r.settlement_method == "stress_ratio"
    d = r.to_dict()
    assert "settlement_method" not in d and "modular_ratio" not in d
    (c,) = [c for c in r.choices if c["what"] == "settlement method"]
    assert c["how"] == "default" and c["chosen"].startswith("stress ratio")
    assert any("weighted_modulus" in a for a in c["alternatives"])
    assert any("balaam_rigid" in a for a in c["alternatives"])
    # with no method by the moduli the column modulus changes nothing
    assert analyze_aggregate_piers(**_BASE, E_column=50000.0) \
        .settlement_improved_mm == r.settlement_improved_mm


@pytest.mark.parametrize("method, n_prime", [
    ("weighted_modulus", 1 + (10 - 1) / 6.0034),
    ("balaam_rigid", 2.02),
    ("balaam_flexible", 1.79),
])
def test_the_methods_by_the_moduli(method, n_prime):
    r = analyze_aggregate_piers(**_BASE, E_column=50000.0,
                                settlement_method=method)
    assert r.modular_ratio == 10.0
    assert r.settlement_ratio == pytest.approx(n_prime, abs=0.005)
    assert r.settlement_improved_mm == pytest.approx(150.0 / n_prime,
                                                     rel=0.003)
    assert r.settlement_reduction_factor == pytest.approx(
        1 / r.settlement_ratio)
    assert r.composite_modulus_kPa == pytest.approx(
        5000.0 * r.settlement_ratio)
    d = r.to_dict()
    assert d["settlement_method"] == method and d["modular_ratio"] == 10.0
    by = {c["what"]: c for c in r.choices}
    c = by["settlement method"]
    assert c["how"] == "given" and "Fig. 5-21" in c["basis"]
    assert "E_c/E_s = 10" in c["chosen"]
    if method != "weighted_modulus":
        assert "average or lower bound estimate" in c["basis"]
    # n sets only the improved bearing here: no choice without one
    assert "stress concentration ratio n" not in by
    assert "E_c/E_s = 10" in by["soil modulus E_s"]["basis"]
    with_q = analyze_aggregate_piers(**_BASE, E_column=50000.0,
                                     settlement_method=method,
                                     q_unreinforced=100.0)
    assert "stress concentration ratio n" in {
        c["what"] for c in with_q.choices}
    assert "Settlement ratio n'" in r.summary()


def test_the_methods_by_the_moduli_need_the_column_modulus():
    with pytest.raises(ValueError, match="give E_column"):
        analyze_aggregate_piers(**_BASE, settlement_method="balaam_rigid")


def test_the_calc_steps_print_the_settlement_ratio():
    analysis = {"method": "aggregate_piers", "E_column": 37500.0,
                "E_soil": 5000.0, "settlement_method": "balaam_rigid"}
    r = analyze_aggregate_piers(**_BASE, E_column=37500.0,
                                settlement_method="balaam_rigid")
    items = {i.name: i.value for i in get_input_summary(r, analysis)}
    assert items["E_c"] == 37500.0 and "Balaam" in items["Settlement"]
    steps = [s for sec in get_calc_steps(r, analysis) for s in sec.items]
    (n_p,) = [s for s in steps
              if getattr(s, "title", "") == "Settlement Ratio, Balaam"]
    assert n_p.result_value == f"{r.settlement_ratio:.3f}"
    assert "between (5c) and (5b)" in n_p.notes
    srf = [s.equation for s in steps if getattr(s, "result_name", "")
           == "SRF"]
    assert srf == ["SRF = 1 / n'"]

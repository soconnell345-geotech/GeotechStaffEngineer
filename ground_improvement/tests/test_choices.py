"""GM-142: the aggregate pier result states n, the moduli and the pattern
it used, and the calc package no longer prints E_column as an input."""

import pytest

from ground_improvement import analyze_aggregate_piers
from ground_improvement.calc_steps import get_input_summary


def _by_what(choices):
    return {c["what"]: c for c in choices}


def test_aggregate_pier_choices_state_n_and_its_basis():
    r = analyze_aggregate_piers(column_diameter=0.6, spacing=1.8,
                                S_unreinforced=100.0,
                                column_type="rammed_aggregate_pier")
    c = _by_what(r.choices)
    n = c["stress concentration ratio n"]
    assert n["chosen"].startswith(
        "5, taken for column_type 'rammed_aggregate_pier' when n is not "
        "given") and n["how"] == "default"
    assert n["alternatives"][0].startswith(
        "column_type 'stone_column': n 2.5")
    # GEC-13's own ranges (pp. 5-31, 5-32, 5-43, 5-45)
    assert "GEC-13" in n["basis"]
    assert "between 2.0 and 5.0" in n["basis"]
    assert "range from 4 to 45 for rigid footings" in n["basis"]
    # GM-291: GEC-13 p. 5-32's preliminary-design ratio, quoted
    assert "a ratio of 2.5 is often" in n["basis"]
    assert "a stress ratio of 5 to 10 for area ground" in n["basis"]
    assert "soft clay" not in n["basis"] and "4 to 10" not in n["basis"]
    assert any("equal-strain modular ratio" in a
               for a in n["alternatives"])
    assert not any("Eq. 5-17" in a for a in n["alternatives"])
    assert c["soil modulus E_s"]["chosen"] == "5000 kPa"
    assert "composite modulus" in c["soil modulus E_s"]["basis"]
    # GM-179: E_column changes no result, so it is not a choice
    assert "column modulus E_c" not in c
    assert c["column pattern"]["chosen"] == "triangular"
    assert r.to_dict()["choices"] == r.choices
    assert "stress concentration ratio n" in r.summary()


def test_a_given_value_is_given():
    r = analyze_aggregate_piers(column_diameter=0.6, spacing=1.8, n=7.0,
                                E_soil=8000.0, pattern="square")
    c = _by_what(r.choices)
    assert c["stress concentration ratio n"]["chosen"] == "7"
    assert c["stress concentration ratio n"]["how"] == "given"
    assert c["soil modulus E_s"]["how"] == "given"
    assert c["column pattern"]["alternatives"] == [
        "triangular (pattern 'triangular')"]


def test_the_numbers_do_not_move():
    r = analyze_aggregate_piers(column_diameter=0.6, spacing=1.8,
                                S_unreinforced=100.0, q_unreinforced=150.0,
                                column_type="rammed_aggregate_pier")
    a_s = 0.6 ** 2 * 3.141592653589793 / 4 / (3 ** 0.5 / 2 * 1.8 ** 2)
    assert r.settlement_reduction_factor == pytest.approx(
        1.0 / (1.0 + a_s * 4.0))
    assert r.composite_modulus_kPa == pytest.approx(5000.0 * (1 + a_s * 4))


def test_the_package_does_not_print_E_column_as_an_input():
    r = analyze_aggregate_piers(column_diameter=0.6, spacing=1.8,
                                column_type="stone_column")
    items = get_input_summary(r, {"method": "aggregate_piers"})
    symbols = [i.name for i in items]
    assert "E_c" not in symbols
    assert "E_s" in symbols and "n" in symbols


# GM-310 (the call for OWN-108, 2026-11-12): n by column type, no one
# default. Made-up columns.

def test_n_is_by_column_type_and_refused_with_neither():
    base = dict(column_diameter=0.6, spacing=1.8, S_unreinforced=100.0)
    a_s = 0.6 ** 2 * 3.141592653589793 / 4 / (3 ** 0.5 / 2 * 1.8 ** 2)
    with pytest.raises(ValueError) as e:
        analyze_aggregate_piers(**base)
    msg = str(e.value)
    assert "Give column_type" in msg and "'stone_column' takes n = 2.5" in msg
    assert "'rammed_aggregate_pier' takes n = 5" in msg
    assert "p. 5-32" in msg and "p. 5-43" in msg and "or give n" in msg
    stone = analyze_aggregate_piers(**base, column_type="stone_column")
    assert stone.stress_concentration_ratio == 2.5
    assert stone.settlement_reduction_factor == pytest.approx(
        1 / (1 + 1.5 * a_s))
    rap = analyze_aggregate_piers(**base,
                                  column_type="rammed_aggregate_pier")
    assert rap.stress_concentration_ratio == 5.0
    assert rap.settlement_reduction_factor == pytest.approx(
        1 / (1 + 4.0 * a_s))
    assert rap.settlement_improved_mm < stone.settlement_improved_mm
    n = _by_what(stone.choices)["stress concentration ratio n"]
    assert n["how"] == "default" and "p. 5-32" in n["chosen"]
    assert n["alternatives"][0].startswith(
        "column_type 'rammed_aggregate_pier': n 5")
    # a given n wins, and the choice says given
    both = analyze_aggregate_piers(**base, n=3.0, column_type="stone_column")
    assert both.stress_concentration_ratio == 3.0
    assert _by_what(both.choices)["stress concentration ratio n"] == {
        **_by_what(both.choices)["stress concentration ratio n"],
        "chosen": "3", "how": "given"}
    with pytest.raises(ValueError, match="column_type must be one of"):
        analyze_aggregate_piers(**base, column_type="pier")


def test_the_area_ratio_0_2_case_of_the_call():
    """The call's own numbers: at a_s = 0.2 the settlement reduction
    factor is 0.56 at n 5 and 0.77 at n 2.5."""
    from ground_improvement.aggregate_piers import (
        settlement_reduction_factor)
    assert settlement_reduction_factor(0.2, 5.0) == pytest.approx(0.556,
                                                                  abs=1e-3)
    assert settlement_reduction_factor(0.2, 2.5) == pytest.approx(0.769,
                                                                  abs=1e-3)


def test_a_fig_5_21_settlement_with_no_bearing_needs_no_n():
    r = analyze_aggregate_piers(column_diameter=0.9, spacing=2.1,
                                S_unreinforced=150.0, E_column=50000.0,
                                settlement_method="weighted_modulus")
    assert r.stress_concentration_ratio is None
    assert r.to_dict()["stress_concentration_ratio"] is None
    assert "stress concentration ratio n" not in _by_what(r.choices)
    assert "not given, not used" in r.summary()
    (row,) = [i for i in get_input_summary(r, {"method": "aggregate_piers"})
              if i.name == "n"]
    assert row.value == "not used in this method"
    with pytest.raises(ValueError, match="Give column_type"):
        analyze_aggregate_piers(column_diameter=0.9, spacing=2.1,
                                E_column=50000.0, q_unreinforced=100.0,
                                settlement_method="weighted_modulus")


def test_a_column_type_takes_n_only_where_n_is_used():
    """GM-310 review: a Fig. 5-21 settlement with no bearing uses no n, so
    a column_type takes none there; with the bearing it takes the type's
    n. A run that gives n keeps it, and the settlement is the same."""
    base = dict(column_diameter=0.9, spacing=2.1, S_unreinforced=150.0,
                E_column=50000.0, settlement_method="weighted_modulus")
    bare = analyze_aggregate_piers(**base)
    typed = analyze_aggregate_piers(**base, column_type="stone_column")
    assert typed.stress_concentration_ratio is None
    assert "stress concentration ratio n" not in _by_what(typed.choices)
    given = analyze_aggregate_piers(**base, n=4.0)
    assert given.stress_concentration_ratio == 4.0
    for r in (typed, given):
        assert r.settlement_improved_mm == pytest.approx(
            bare.settlement_improved_mm)
    bearing = analyze_aggregate_piers(**base, column_type="stone_column",
                                      q_unreinforced=100.0)
    assert bearing.stress_concentration_ratio == 2.5
    (row,) = [i for i in get_input_summary(bearing,
                                           {"method": "aggregate_piers"})
              if i.name == "n"]
    assert row.value == 2.5

"""GM-142: the aggregate pier result states n, the moduli and the pattern
it used, and the calc package no longer prints E_column as an input."""

import pytest

from ground_improvement import analyze_aggregate_piers
from ground_improvement.calc_steps import get_input_summary


def _by_what(choices):
    return {c["what"]: c for c in choices}


def test_aggregate_pier_choices_state_n_and_its_basis():
    r = analyze_aggregate_piers(column_diameter=0.6, spacing=1.8,
                                S_unreinforced=100.0)
    c = _by_what(r.choices)
    n = c["stress concentration ratio n"]
    assert n["chosen"] == "5" and n["how"] == "default"
    # GEC-13's own ranges (Sec. 4.1.2.4; p. 5-45)
    assert "GEC-13" in n["basis"]
    assert "between 2.0 and 5.0" in n["basis"]
    assert "4 to 45 for rigid footings" in n["basis"]
    assert "5 to 10 under embankments" in n["basis"]
    assert "soft clay" not in n["basis"] and "4 to 10" not in n["basis"]
    assert any("equal-strain modular ratio" in a
               for a in n["alternatives"])
    assert not any("Eq. 5-17" in a for a in n["alternatives"])
    assert c["soil modulus E_s"]["chosen"] == "5000 kPa"
    assert "composite modulus" in c["soil modulus E_s"]["basis"]
    assert "not used in the calculation" in c["column modulus E_c"]["basis"]
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
                                S_unreinforced=100.0, q_unreinforced=150.0)
    a_s = 0.6 ** 2 * 3.141592653589793 / 4 / (3 ** 0.5 / 2 * 1.8 ** 2)
    assert r.settlement_reduction_factor == pytest.approx(
        1.0 / (1.0 + a_s * 4.0))
    assert r.composite_modulus_kPa == pytest.approx(5000.0 * (1 + a_s * 4))


def test_the_package_does_not_print_E_column_as_an_input():
    r = analyze_aggregate_piers(column_diameter=0.6, spacing=1.8)
    items = get_input_summary(r, {"method": "aggregate_piers"})
    symbols = [i.name for i in items]
    assert "E_c" not in symbols
    assert "E_s" in symbols and "n" in symbols

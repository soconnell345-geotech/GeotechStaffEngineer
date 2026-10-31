"""GM-203: ``elastic_settlement`` took Iw = 1.0 for every footing, which is
no printed point of a square. Not given, Iw is now GEC-6 Table 5-12's shape
and rigidity factor C_d (p. 97, after Winterkorn & Fang 1975) for the
footing's shape, rigidity and point, and the tool says which."""

import math

import pytest

from settlement.immediate import (
    elastic_influence_factor, elastic_settlement, gec6_influence_factor)

#: GEC-6 Table 5-12 (p. 97) as printed: center, corner, middle of short
#: side, middle of long side, average.
PRINTED = {
    "circle": (1.00, 0.64, 0.64, 0.64, 0.85),
    "circle_rigid": (0.79,) * 5,
    "square": (1.12, 0.56, 0.76, 0.76, 0.95),
    "square_rigid": (0.99,) * 5,
    1.5: (1.36, 0.67, 0.89, 0.97, 1.15),
    2: (1.52, 0.76, 0.98, 1.12, 1.30),
    3: (1.78, 0.88, 1.11, 1.35, 1.52),
    5: (2.10, 1.05, 1.27, 1.68, 1.83),
    10: (2.53, 1.26, 1.49, 2.12, 2.25),
    100: (4.00, 2.00, 2.20, 3.60, 3.70),
    1000: (5.47, 2.75, 2.94, 5.03, 5.15),
    10000: (6.90, 3.50, 3.70, 6.50, 6.60),
}
POINTS = ("center", "corner", "mid_short", "mid_long", "average")


def _cd(row, point):
    if row in ("circle", "circle_rigid"):
        return gec6_influence_factor(
            "circular", "rigid" if row == "circle_rigid" else "flexible",
            point)[0]
    if row in ("square", "square_rigid"):
        return gec6_influence_factor(
            "square", "rigid" if row == "square_rigid" else "flexible",
            point)[0]
    return gec6_influence_factor("rectangular", "flexible", point, B=2.0,
                                 L=2.0 * row)[0]


@pytest.mark.parametrize("row, point, printed", [
    (row, p, v) for row, vals in PRINTED.items()
    for p, v in zip(POINTS, vals)])
def test_every_printed_cell_of_table_5_12(row, point, printed):
    assert _cd(row, point) == printed


def test_between_rows_in_log_l_over_b():
    # L/B = 7, between the printed 5 (2.10) and 10 (2.53) rows,
    # interpolated in log(L/B); the analysis's default Iw is the same
    # table value (GM-285: it was Schleicher's closed form, 2.318 here)
    v = gec6_influence_factor("rectangular", L=14.0, B=2.0)[0]
    assert 2.10 < v < 2.53
    t = math.log(7 / 5) / math.log(10 / 5)
    assert v == pytest.approx(2.10 + t * (2.53 - 2.10), abs=1e-12)
    assert elastic_influence_factor("rectangular", 14.0, 2.0) == v
    # L and B either way round
    assert gec6_influence_factor("rectangular", L=2.0, B=14.0)[0] == v


def test_the_default_is_the_centre_of_a_flexible_square():
    # before/after, q 100 kPa, B 2 m, Es 15 MPa, nu 0.3: Iw 1.0 -> 1.12,
    # 12.13 -> 13.59 mm
    assert elastic_settlement(100, 2.0, 15000) == pytest.approx(
        100 * 2.0 * 0.91 / 15000 * 1.12)
    assert elastic_settlement(100, 2.0, 15000) * 1000 == pytest.approx(
        13.59, abs=0.005)
    # a given Iw is used as given
    assert elastic_settlement(100, 2.0, 15000, Iw=1.0) * 1000 == \
        pytest.approx(12.13, abs=0.005)


@pytest.mark.parametrize("kw, match", [
    (dict(shape="strip"), "strip has no shape and rigidity factor"),
    (dict(shape="rectangular", L=6.0, rigidity="rigid"),
     "no rigid rectangle"),
    (dict(shape="rectangular"), "needs its length L"),
    (dict(shape="square", L=3.0), "Use shape 'rectangular'"),
    (dict(point="edge"), "a circle's point"),
    (dict(rigidity="stiff"), "rigidity must be"),
])
def test_the_table_refuses_what_it_does_not_print(kw, match):
    with pytest.raises(ValueError, match=match):
        elastic_settlement(100, 2.0, 15000, **kw)


def test_the_tool_states_iw_and_the_tables_other_points():
    from funhouse_agent.dispatch import call_agent
    out = call_agent("settlement", "elastic_settlement",
                     {"q_net": 200.0, "B": 2.5, "Es": 25000.0,
                      "rigidity": "rigid"})
    assert out["Iw"] == 0.99
    assert out["immediate_settlement_mm"] == pytest.approx(18.02, abs=0.01)
    iw = out["choices"][0]
    assert iw["what"] == "influence factor Iw" and iw["how"] == "module"
    assert "Table 5-12" in iw["chosen"] and "rigid" in iw["chosen"]
    assert any("0.95" in a and "average" in a for a in iw["alternatives"])
    given = call_agent("settlement", "elastic_settlement",
                       {"q_net": 100.0, "B": 2.0, "Es": 15000.0, "Iw": 1.0})
    assert given["choices"][0]["how"] == "given"
    assert given["immediate_settlement_mm"] == pytest.approx(12.13, abs=0.01)


def test_the_rigid_square_is_used_as_printed_and_the_basis_says_why():
    """GM-214: the table's rigid square (0.99) is above the flexible
    square's average (0.95), where its rigid circle (0.79) is below the
    flexible circle's (0.85). NHI-06-089 Table 8-13 prints the same 0.99
    and no other reference in the library prints a rigid square, so it is
    kept, and a square's basis says so; a circle's does not."""
    from funhouse_agent.dispatch import call_agent
    assert gec6_influence_factor("square", "rigid")[0] == 0.99
    for kw in ({"rigidity": "rigid"}, {}):
        out = call_agent("settlement", "elastic_settlement",
                         {"q_net": 200.0, "B": 2.5, "Es": 25000.0, **kw})
        basis = out["choices"][0]["basis"]
        assert "NHI-06-089 Table 8-13" in basis
        assert "above the flexible square's average (0.95)" in basis
        assert "so 0.99 is used as printed" in basis
    out = call_agent("settlement", "elastic_settlement",
                     {"q_net": 200.0, "B": 2.5, "Es": 25000.0,
                      "shape": "circular"})
    assert "0.99 is used" not in out["choices"][0]["basis"]

"""pile_group calculation tables (backlog GM-39): the values in the tables
are the result's own, and the pile-head forces balance the applied load."""

import pytest

from pile_group import (GroupLoad, GroupPile, analyze_group_6dof,
                        analyze_vertical_group_simple,
                        create_rectangular_layout)


def _by_name(result):
    return {t.name: t for t in result.get_tables()}


@pytest.fixture(scope="module")
def six_dof():
    piles = create_rectangular_layout(2, 3, 1.5, 1.5, axial_stiffness=2e5,
                                      lateral_stiffness=2e4)
    piles[0].batter_x = 10.0                    # one battered pile
    load = GroupLoad(Vx=200.0, Vy=-50.0, Vz=3000.0, My=600.0, Mx=150.0)
    return analyze_group_6dof(piles, load), load


def test_head_forces_balance_the_applied_load(six_dof):
    result, load = six_dof
    t = _by_name(result)["pile_head_forces"]
    assert t.printed
    assert sum(t.column("Head force Fx")) == pytest.approx(load.Vx)
    assert sum(t.column("Head force Fy")) == pytest.approx(load.Vy)
    assert sum(t.column("Head force Fz")) == pytest.approx(load.Vz)


def test_axial_force_is_the_results_own(six_dof):
    result, _ = six_dof
    t = _by_name(result)["pile_head_forces"]
    for p, pf in zip(t.column("Axial force P"), result.pile_forces):
        assert round(p, 1) == pf["axial_kN"]
    assert max(t.column("Axial force P")) == pytest.approx(
        result.max_compression)


def test_head_displacements_follow_the_cap(six_dof):
    result, _ = six_dof
    t = _by_name(result)["pile_head_displacements"]
    d = result.cap_displacements
    for row, pf in zip(t.rows, result.pile_forces):
        x, y = pf["x_m"], pf["y_m"]
        assert row[1] == pytest.approx(d["dx"] - d["rz"] * y)
        assert row[3] == pytest.approx(d["dz"] - d["rx"] * y + d["ry"] * x)


def test_cap_displacements_csv_only(six_dof):
    result, _ = six_dof
    t = _by_name(result)["cap_displacements"]
    assert not t.printed
    assert dict(zip(t.column("Component"), t.column("Value")))["dx"] == \
        result.cap_displacements["dx"]


def test_simple_method_writes_csv_only():
    piles = [GroupPile(x=x, y=0.0, axial_capacity_compression=1000.0)
             for x in (-1.0, 0.0, 1.0)]
    result = analyze_vertical_group_simple(piles, GroupLoad(Vz=900.0,
                                                            My=200.0))
    tables = _by_name(result)
    t = tables["pile_head_forces"]
    assert not t.printed
    assert "pile_head_displacements" not in tables
    assert t.column("Axial force P") == [pf["axial_kN"]
                                         for pf in result.pile_forces]
    assert t.column("Utilization P/capacity")[2] == \
        result.pile_forces[2]["utilization"]


def test_to_dict_is_unchanged(six_dof):
    result, _ = six_dof
    assert "pile_head_response" not in result.to_dict()

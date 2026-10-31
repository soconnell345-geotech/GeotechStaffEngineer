"""GM-148: the calc steps print the unit end bearing the analysis used
(the bell's diameter in the large-base reductions, the 80 ksf limit), and
a cohesionless side layer with no N60 says what beta it took. Since GM-244
the reductions and the limit are the named options of O'Neill and Reese
(1999), which GEC-10 (2018) does not print."""

import pytest

from drilled_shaft import (DrillShaft, DrillShaftAnalysis, ShaftSoilLayer,
                           ShaftSoilProfile)
from drilled_shaft.calc_steps import get_calc_steps


def _steps(shaft, layers, **options):
    analysis = DrillShaftAnalysis(shaft=shaft,
                                  soil=ShaftSoilProfile(layers=layers),
                                  **options)
    result = analysis.compute()
    steps = {}
    for section in get_calc_steps(result, analysis):
        if section.title == "End Bearing":
            for item in section.items:
                steps[getattr(item, "title", "")] = item
    return result, steps


def _kpa(step):
    return float(step.result_value.replace(",", ""))


def test_a_belled_shaft_in_clay_prints_the_bells_reduction():
    """A 3 m bell (> 1.90 m) on a 1.2 m shaft: F_r on the bell, not the
    shaft, and Q_tip = q_b x A_tip in the steps as in the result."""
    shaft = DrillShaft(diameter=1.2, length=20.0, bell_diameter=3.0)
    clay = [ShaftSoilLayer(thickness=30, soil_type="cohesive",
                           unit_weight=18, cu=150)]
    result, steps = _steps(shaft, clay, large_base_reduction=True)
    fr = steps["Large-Base Reduction F_r"]
    assert "D_bell = 3.000 m" in fr.substitution
    q_b = _kpa(fr)
    assert q_b * shaft.tip_area == pytest.approx(result.Q_tip, rel=1e-3)
    assert q_b < 9 * 150                     # the reduction applied


def test_the_80_ksf_limit_is_printed_where_it_governs():
    shaft = DrillShaft(diameter=1.0, length=20.0)
    stiff = [ShaftSoilLayer(thickness=30, soil_type="cohesive",
                            unit_weight=20, cu=600)]
    result, steps = _steps(shaft, stiff, clay_base_method="oneill_reese")
    unit = steps["Unit End Bearing"]
    assert _kpa(unit) == pytest.approx(3830.0)
    assert "3,830" in unit.notes
    assert _kpa(unit) * shaft.tip_area == pytest.approx(result.Q_tip,
                                                        rel=1e-3)


def test_a_belled_shaft_in_sand_reduces_by_the_bell():
    """1.27/D with D the 2.0 m bell; the shaft's 1.0 m would print no
    reduction at all."""
    shaft = DrillShaft(diameter=1.0, length=15.0, bell_diameter=2.0)
    sand = [ShaftSoilLayer(thickness=30, soil_type="cohesionless",
                           unit_weight=19, phi=34, N60=30)]
    result, steps = _steps(shaft, sand, large_base_reduction=True)
    unit = steps["Unit End Bearing (Cohesionless Tip)"]
    assert _kpa(unit) == pytest.approx(57.5 * 30 * 1.27 / 2.0, rel=1e-3)
    assert "1.27/2.000" in unit.notes
    assert _kpa(unit) * shaft.tip_area == pytest.approx(result.Q_tip,
                                                        rel=1e-3)


def test_a_side_layer_with_no_n60_states_its_beta():
    """Under the depth beta (named; GEC-10's Eq. 10-13 needs the N60)."""
    what = "beta of cohesionless side layers given no N60"
    shaft = DrillShaft(diameter=1.0, length=15.0)
    layers = [ShaftSoilLayer(thickness=8, soil_type="cohesionless",
                             unit_weight=19, phi=32),
              ShaftSoilLayer(thickness=20, soil_type="cohesionless",
                             unit_weight=19, phi=34, N60=30)]
    r = DrillShaftAnalysis(shaft=shaft, soil=ShaftSoilProfile(layers=layers),
                           beta_method="depth").compute()
    (c,) = [c for c in r.choices if c["what"] == what]
    assert c["how"] == "module" and "0-8 m" in c["chosen"]
    assert "Eq. 9-42 (p. 9-132)" in c["basis"]
    assert c in r.to_dict()["choices"]
    # with every side N60 given there is no such choice
    layers[0] = ShaftSoilLayer(thickness=8, soil_type="cohesionless",
                               unit_weight=19, phi=32, N60=10)
    r = DrillShaftAnalysis(shaft=shaft, soil=ShaftSoilProfile(layers=layers),
                           beta_method="depth").compute()
    assert [c for c in r.choices if c["what"] == what] == []


def test_the_tools_take_the_concrete_strength_for_the_rock_side(
        tmp_path, monkeypatch):
    """GM-160 refused concrete_fc, which changed nothing; since GM-284 the
    rock side resistance takes qu no higher than it (GEC-10 p. 10-21).
    Made-up values: qu 40 MPa against f'c 30 MPa."""
    import math
    from funhouse_agent import dispatch as D
    monkeypatch.chdir(tmp_path)      # the package writes its files here
    call = {"diameter": 1.0, "shaft_length": 8.0, "concrete_fc": 30000,
            "layers": [{"thickness": 3, "soil_type": "cohesive",
                        "unit_weight": 18, "cu": 100},
                       {"thickness": 10, "soil_type": "rock",
                        "unit_weight": 23, "qu": 40000}]}
    for method in ("drilled_shaft_capacity", "lrfd_capacity"):
        r = D.call_agent("drilled_shaft", method, dict(call))
        assert "error" not in r, r.get("error")
        assert r["Q_side_rock_kN"] == pytest.approx(
            math.sqrt(30000 * 101.325) * math.pi * 5.0, rel=1e-3), method
    r = D.call_agent("calc_package", "drilled_shaft_package", dict(call))
    assert "error" not in r, r.get("error")

"""lateral_pile calculation tables (backlog GM-39): the values in the
tables are the result's own arrays."""

import numpy as np
import pytest

from lateral_pile import LateralPileAnalysis, Pile
from lateral_pile.calc_steps import PY_TABLE_POINTS, get_tables
from lateral_pile.py_curves import SandAPI, SoftClayMatlock
from lateral_pile.soil import SoilLayer


@pytest.fixture(scope="module")
def solved():
    pile = Pile(length=20.0, diameter=0.6, thickness=0.0127, E=200e6)
    layers = [
        SoilLayer(top=0, bottom=5,
                  py_model=SoftClayMatlock(c=25, gamma=8, eps50=0.02)),
        SoilLayer(top=5, bottom=20,
                  py_model=SandAPI(phi=35, gamma=10, k=22000)),
    ]
    analysis = LateralPileAnalysis(pile, layers)
    return analysis.solve(Vt=150.0, Mt=20.0), analysis


def test_response_table_is_the_result_arrays(solved):
    result, _ = solved
    (t,) = result.get_tables()
    assert t.name == "pile_response" and t.n_rows == len(result.z)
    for key, arr in [("Depth z", result.z), ("Deflection y", result.deflection),
                     ("Slope S", result.slope),
                     ("Bending moment M", result.moment),
                     ("Shear force V", result.shear),
                     ("Soil reaction p", result.soil_reaction),
                     ("Soil modulus Es", result.Es)]:
        np.testing.assert_array_equal(np.array(t.column(key)), arr)


def test_response_table_holds_the_headline_values(solved):
    result, _ = solved
    (t,) = result.get_tables()
    assert t.column("Deflection y")[0] == result.y_top
    moments = t.column("Bending moment M")
    i = int(np.argmax(np.abs(moments)))
    assert abs(moments[i]) == result.max_moment
    assert t.column("Depth z")[i] == result.max_moment_depth
    assert t.columns[3].unit == "kN-m" and t.columns[1].unit == "m"


def test_cracked_section_adds_the_ei_column(solved):
    result, _ = solved
    assert "Bending stiffness EI" not in [c.name for c in
                                          result.get_tables()[0].columns]
    result.EI_profile = np.full(len(result.z), 1.0e5)
    try:
        t = result.get_tables()[0]
        assert t.column("Bending stiffness EI")[0] == 1.0e5
    finally:
        result.EI_profile = None


def test_py_curves_are_each_layer_models_own(solved):
    result, analysis = solved
    tables = {t.name: t for t in get_tables(result, analysis)}
    py = tables["py_curves"]
    assert py.n_rows == PY_TABLE_POINTS * len(analysis.layers)
    layer = analysis.layers[1]
    z_mid = 0.5 * (layer.top + layer.bottom)
    ys, ps = layer.py_model.get_py_curve(z_mid, analysis.pile.diameter,
                                         n_points=PY_TABLE_POINTS)
    rows = [r for r in py.rows if r[0] == 2]
    assert [r[2] for r in rows] == [z_mid] * PY_TABLE_POINTS
    assert [r[3] for r in rows] == [float(y) for y in ys]
    assert [r[4] for r in rows] == [float(p) for p in ps]


def test_layer_echo_is_csv_only(solved):
    result, analysis = solved
    tables = {t.name: t for t in get_tables(result, analysis)}
    layers = tables["soil_layers"]
    assert not layers.printed
    assert layers.column("p-y model") == ["SoftClayMatlock", "SandAPI"]
    assert layers.column("Bottom") == [5.0, 20.0]

"""sheet_pile calculation tables (backlog GM-39): the per-depth table is
the module's own moment integration."""

import numpy as np
import pytest

from sheet_pile import WallSoilLayer, analyze_anchored, analyze_cantilever
from sheet_pile.calc_steps import get_tables

LAYERS = [WallSoilLayer(thickness=3.0, unit_weight=18.0, friction_angle=30.0,
                        description="sand"),
          WallSoilLayer(thickness=17.0, unit_weight=19.0, friction_angle=34.0)]


@pytest.fixture(scope="module")
def cantilever():
    return analyze_cantilever(excavation_depth=4.0, soil_layers=LAYERS,
                              surcharge=10.0)


def _by_name(tables):
    return {t.name: t for t in tables}


def test_profile_table_holds_the_maximum_moment(cantilever):
    t = _by_name(cantilever.get_tables())["wall_pressures"]
    assert t.n_rows == len(cantilever.profile) == 500
    m = np.array(t.column("Bending moment M"))
    i = int(np.argmax(np.abs(m)))
    assert abs(m[i]) == cantilever.max_moment
    assert t.column("Depth z")[i] == cantilever.max_moment_depth


def test_profile_ends_at_the_toe_in_equilibrium(cantilever):
    t = _by_name(cantilever.get_tables())["wall_pressures"]
    assert t.column("Depth z")[-1] == pytest.approx(
        cantilever.total_wall_length, abs=0.05)
    # no passive pressure above the excavation level
    above = [r for r in t.rows if r[0] <= cantilever.excavation_depth]
    assert all(r[2] == 0.0 for r in above)
    # shear is the running sum of net pressure x step
    dz = cantilever.total_wall_length / t.n_rows
    v = np.cumsum(np.array(t.column("Net pressure p")) * dz)
    np.testing.assert_allclose(t.column("Shear V"), v, rtol=1e-9, atol=1e-9)


def test_summary_and_layer_echo_are_csv_only(cantilever):
    tables = _by_name(get_tables(cantilever, {"soil_layers": LAYERS}))
    s = tables["design_summary"]
    assert not s.printed
    assert dict(zip(s.column("Symbol"), s.column("Value")))["D"] == \
        cantilever.embedment_depth
    assert tables["soil_layers"].column("Bottom") == [3.0, 20.0]
    assert not tables["soil_layers"].printed


def test_anchored_wall_has_no_per_depth_table_yet():
    r = analyze_anchored(excavation_depth=6.0, anchor_depth=1.5,
                         soil_layers=LAYERS)
    names = [t.name for t in get_tables(r, {"soil_layers": LAYERS})]
    assert names == ["soil_layers"]


def test_to_dict_is_unchanged(cantilever):
    assert "profile" not in cantilever.to_dict()

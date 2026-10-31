"""GM-256: mse_wall takes the scale effect correction factor alpha of GEC-11
Vol I Table 3-6 (p. 3-17) by reinforcement type, a tested one where given,
and refuses a custom reinforcement with no type and a 'geosynthetic' with
no tested alpha. The module tests are in
retaining_walls/tests/test_mse_scale_effect_alpha.py."""

import pytest

from funhouse_agent import dispatch as D

_WALL = dict(wall_height=6.0, gamma_backfill=19.0, phi_backfill=34.0,
             reinforcement_length=5.0, reinforcement_spacing=0.6)


def _mse(**extra):
    return D.call_agent("retaining_walls", "mse_wall", {**_WALL, **extra})


def _lowest(r):
    return min(lv["FOS_pullout"] for lv in r["internal_results"])


# GM-297: F* = (2/3) tan phi (GEC-11 Vol I Eq. 3-8); 8.394, 6.296 and
# 10.493 at 0.67 tan phi before
@pytest.mark.parametrize("params, alpha, lowest", [
    ({"reinforcement_name": "geogrid_gec11_e1_gg1"}, 0.8, 8.353),
    ({"reinforcement_type": "geogrid", "reinforcement_Tallowable": 21.9},
     0.8, 8.353),
    ({"reinforcement_type": "geotextile", "reinforcement_Tallowable": 21.9},
     0.6, 6.264),
    ({"reinforcement_type": "geosynthetic", "reinforcement_Tallowable": 21.9,
      "alpha_tested": 1.0}, 1.0, 10.441),
])
def test_alpha_by_type_or_test(params, alpha, lowest):
    r = _mse(**params)
    assert "error" not in r, r.get("error")
    assert {lv["alpha"] for lv in r["internal_results"]} == {alpha}
    assert _lowest(r) == pytest.approx(lowest, abs=1e-3)


def test_defaults_used_says_the_alpha_it_took():
    """GM-266 review: alpha_tested is an alpha FROM TESTS, so defaults_used
    gives the case a left-out one stands for (Table 3-6's alpha by type,
    with its basis), never the table's value as if tested; the value each
    level took is its ``alpha``, and the alpha choice names it and the
    table's row."""
    from funhouse_agent.adapters.retaining_walls import METHOD_INFO
    case = METHOD_INFO["mse_wall"]["parameters"]["alpha_tested"][
        "when_absent"]
    for name, alpha, row in (
            ("geogrid_gec11_e1_gg1", 0.8, "extensible grids (geogrids)"),
            ("welded_wire_grid_w11", 1.0, "inextensible grids")):
        extra = ({"reinforcement_horizontal_spacing": 1.524}
                 if name.startswith("welded") else {})
        r = _mse(reinforcement_name=name, **extra)
        assert r["defaults_used"]["alpha_tested"] == (
            f"{case['case']}: {case['basis']}")
        assert {lv["alpha"] for lv in r["internal_results"]} == {alpha}
        (c,) = [c for c in r["choices"] if "alpha_tested" in c["what"]]
        assert c["chosen"].startswith(f"alpha {alpha:g}, GEC-11 Vol I "
                                      "Table 3-6's default value for " + row)
        assert c["how"] == "default"
    given = _mse(reinforcement_name="geogrid_gec11_e1_gg1", alpha_tested=0.9)
    assert "alpha_tested" not in given["defaults_used"]


def test_a_geosynthetic_with_no_alpha_or_no_type_is_refused():
    err = _mse(reinforcement_type="geosynthetic",
               reinforcement_Tallowable=21.9)["error"]
    assert "alpha = 0.8 for geogrids" in err
    assert "alpha = 0.6 for geotextiles" in err
    assert "alpha_tested" in err
    err = _mse(reinforcement_Tallowable=21.9)["error"]
    assert "needs its reinforcement_type" in err
    assert "0.8 for geogrids, 0.6 for geotextiles" in err


def test_list_reinforcement_gives_each_alpha():
    out = D.call_agent("retaining_walls", "list_reinforcement", {})
    types = out["reinforcement_types"]
    assert types["geogrid_gec11_e1_gg1"]["type"] == "geogrid"
    assert types["geogrid_gec11_e1_gg1"]["alpha"] == 0.8
    assert types["ribbed_steel_strip_50x4"]["alpha"] == 1.0


def test_the_tool_text_names_the_table():
    from funhouse_agent.adapters.retaining_walls import METHOD_INFO
    p = METHOD_INFO["mse_wall"]["parameters"]
    assert "Table 3-6" in p["alpha_tested"]["description"]
    assert "0.8 for geogrids, 0.6 for geotextiles" in p["alpha_tested"][
        "description"]
    assert "required with reinforcement_Tallowable" in p[
        "reinforcement_type"]["description"]

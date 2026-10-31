"""GM-256: the scale effect correction factor alpha of the pullout
resistance, F* alpha sigma_v Le C Rc, as GEC-11 Vol I prints it.

  - Table 3-6 (p. 3-17), "alpha Default Value": 1.0 for inextensible strips,
    1.0* for inextensible grids (bar mats and welded wire), 0.8 for
    extensible grids, 0.6 for extensible sheets; note *: "For longitudinal
    bars/wires spacing greater than 6 inches, alpha may be less than 1.0 and
    pullout tests are required"; note (ii): "Pullout testing to determine
    alpha is recommended if alpha shown in table is less than 1.0";
  - p. 3-16: "In the absence of test data, alpha = 0.8 for geogrids and
    alpha = 0.6 for geotextiles (extensible sheets) is recommended".
Before this every geosynthetic took alpha 1.0. GEC-11 Vol II Example E1
takes 0.8 for its geogrids (validation V-057, Table E1-7.5).
"""

import dataclasses

import pytest

from retaining_walls.calc_steps import get_calc_steps
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.mse import (
    analyze_mse_wall, check_internal_stability, pullout_alpha, table_alpha,
)
from retaining_walls.reinforcement import (
    GEOGRID_GEC11_E1_GG1, GEOGRID_GEC11_E1_GG2, RIBBED_STEEL_STRIP_50x4,
    Reinforcement, WELDED_WIRE_GRID_W11,
)


def _wall():
    """The 6 m geosynthetic wall of GM-256."""
    return MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0,
                           reinforcement_spacing=0.6)


def _geotextile():
    return Reinforcement("geotextile", "geotextile", Tallowable=21.9)


def _pullout(r, **kw):
    levels = check_internal_stability(_wall(), 19.0, 34.0, r, **kw)
    po = [lv["FOS_pullout"] for lv in levels]
    return min(po), max(po), min(lv["FOS_rupture"] for lv in levels)


def _alpha_choice(r):
    (c,) = [c for c in r.choices if "(alpha_tested)" in c["what"]]
    return c


def test_table_3_6_by_type():
    assert table_alpha(RIBBED_STEEL_STRIP_50x4) == 1.0
    assert table_alpha(WELDED_WIRE_GRID_W11) == 1.0
    assert table_alpha(GEOGRID_GEC11_E1_GG1) == \
        table_alpha(GEOGRID_GEC11_E1_GG2) == 0.8
    assert table_alpha(_geotextile()) == 0.6
    assert table_alpha(Reinforcement("g", "geosynthetic", 20.0)) is None
    assert pullout_alpha(GEOGRID_GEC11_E1_GG1, 0.95) == 0.95


def test_a_geogrid_wall_before_and_after():
    """FOS_pullout 10.441 to 25.663 at the alpha 1.0 taken before; 8.353 to
    20.531 at Table 3-6's 0.8 (10.493 to 25.792 and 8.394 to 20.633 before
    GM-297 took F* = (2/3) tan phi, Eq. 3-8, not 0.67 tan phi). Rupture
    does not take alpha: 0.862 on E1's GG-I since GM-271 (1.192 on the
    retired built-in geogrid's 21.9 kN/m)."""
    lo, hi, ru = _pullout(GEOGRID_GEC11_E1_GG1)
    assert (lo, hi) == (pytest.approx(8.353, abs=1e-3),
                        pytest.approx(20.531, abs=1e-3))
    assert ru == pytest.approx(0.862, abs=1e-3)
    lo1, hi1, ru1 = _pullout(GEOGRID_GEC11_E1_GG1, alpha_tested=1.0)
    assert (lo1, hi1) == (pytest.approx(10.441, abs=1e-3),
                          pytest.approx(25.663, abs=1e-3))
    assert ru1 == ru
    r = analyze_mse_wall(_wall(), 19.0, 34.0, GEOGRID_GEC11_E1_GG1)
    c = _alpha_choice(r)
    assert c["how"] == "default"
    assert c["chosen"] == ("alpha 0.8, GEC-11 Vol I Table 3-6's default "
                           "value for extensible grids (geogrids), p. 3-17")
    assert "Table 3-6 (p. 3-17)" in c["basis"]
    assert "'In the absence of test data, alpha = 0.8 for geogrids and " \
           "alpha = 0.6 for geotextiles" in c["basis"]
    assert "note (ii)" in c["basis"]
    (alt,) = c["alternatives"]
    assert alt.startswith("alpha from pullout tests (alpha_tested), which "
                          "note (ii) recommends; at alpha 1.0")
    assert "lowest 10.4 (here 8.35)" in alt


def test_a_geotextile_wall_before_and_after():
    """The same wall with a geotextile of the same strength: 10.441 to
    25.663 as a 'geosynthetic' before, 6.264 to 15.398 at Table 3-6's
    0.6 (6.296 to 15.475 before GM-297's F* = (2/3) tan phi)."""
    lo, hi, ru = _pullout(_geotextile())
    assert (lo, hi) == (pytest.approx(6.264, abs=1e-3),
                        pytest.approx(15.398, abs=1e-3))
    assert ru == pytest.approx(1.192, abs=1e-3)
    r = analyze_mse_wall(_wall(), 19.0, 34.0, _geotextile())
    assert _alpha_choice(r)["chosen"].startswith(
        "alpha 0.6, GEC-11 Vol I Table 3-6's default value for extensible "
        "sheets (geotextiles)")


def test_a_geosynthetic_of_no_stated_kind_is_refused():
    plain = Reinforcement("g", "geosynthetic", Tallowable=21.9)
    with pytest.raises(ValueError) as err:
        analyze_mse_wall(_wall(), 19.0, 34.0, plain)
    text = str(err.value)
    for words in ("alpha = 0.8 for geogrids", "alpha = 0.6 for geotextiles",
                  "'geogrid' (alpha 0.8)", "'geotextile' (alpha 0.6)",
                  "alpha_tested", "Table 3-6, p. 3-17", "p. 3-16"):
        assert words in text, words
    with pytest.raises(ValueError, match="alpha = 0.8 for geogrids"):
        check_internal_stability(_wall(), 19.0, 34.0, plain)


def test_a_tested_alpha_is_kept_and_set_beside_the_printed_ones():
    plain = Reinforcement("g", "geosynthetic", Tallowable=21.9)
    r = analyze_mse_wall(_wall(), 19.0, 34.0, plain, alpha_tested=0.7)
    assert {lv["alpha"] for lv in r.internal_results} == {0.7}
    c = _alpha_choice(r)
    assert c["how"] == "given"
    assert c["chosen"] == "alpha 0.7, from pullout tests"
    a8, a6 = c["alternatives"]
    assert a8.startswith("alpha 0.8, Table 3-6's default value for geogrids")
    assert a6.startswith("alpha 0.6, Table 3-6's default value for "
                         "geotextiles")
    grid = analyze_mse_wall(_wall(), 19.0, 34.0, GEOGRID_GEC11_E1_GG1,
                            alpha_tested=1.0)
    assert {lv["alpha"] for lv in grid.internal_results} == {1.0}
    (alt,) = _alpha_choice(grid)["alternatives"]
    assert alt.startswith("alpha 0.8, Table 3-6's default value for "
                          "extensible grids (geogrids)")


def test_steel_takes_1_and_a_steel_grid_says_note_star():
    strip = dataclasses.replace(RIBBED_STEEL_STRIP_50x4,
                                horizontal_spacing=0.75)
    r = analyze_mse_wall(_wall(), 19.0, 34.0, strip)
    assert {lv["alpha"] for lv in r.internal_results} == {1.0}
    assert not [c for c in r.choices if "(alpha_tested)" in c["what"]]
    grid = dataclasses.replace(WELDED_WIRE_GRID_W11, coverage_ratio=0.3)
    r = analyze_mse_wall(_wall(), 19.0, 34.0, grid)
    assert {lv["alpha"] for lv in r.internal_results} == {1.0}
    c = _alpha_choice(r)
    assert c["chosen"].startswith("alpha 1, GEC-11 Vol I Table 3-6's default "
                                  "value for inextensible grids (bar mats")
    assert "'For longitudinal bars/wires spacing greater than 6 inches, " \
           "alpha may be less than 1.0 and pullout tests are required'" \
           in c["basis"]
    assert "more than 6 in. (150 mm) apart" in c["alternatives"][0]


def test_the_calc_steps_name_the_table():
    r = analyze_mse_wall(_wall(), 19.0, 34.0, GEOGRID_GEC11_E1_GG1)
    analysis = {"wall_type": "mse", "geom": _wall(), "gamma_backfill": 19.0,
                "phi_backfill": 34.0, "reinforcement": GEOGRID_GEC11_E1_GG1}
    items = [it for sec in get_calc_steps(r, analysis) for it in sec.items]
    (step,) = [it for it in items
               if getattr(it, "title", "") == "Pullout Resistance"]
    assert "α = 0.8 (GEC-11 Table 3-6, p. 3-17)" in step.substitution

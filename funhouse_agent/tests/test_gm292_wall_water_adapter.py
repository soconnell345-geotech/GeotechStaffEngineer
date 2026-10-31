"""GM-292: the cantilever_wall tool and the retaining wall package take a
water table behind the wall (retaining_walls/tests/test_gm292_water_table.py
checks the numbers against the print)."""

import pytest

from funhouse_agent import dispatch as D

# made-up wall: the hand-checked one of module_work/gm292/CHECK.md
WALL = {"wall_height": 5.0, "base_width": 3.0, "toe_length": 0.5,
        "stem_thickness_top": 0.3, "stem_thickness_base": 0.5,
        "base_thickness": 0.5, "gamma_backfill": 18.0, "phi_backfill": 30.0,
        "q_allowable": 200.0}


def _wall(**kw):
    return D.call_agent("retaining_walls", "cantilever_wall", {**WALL, **kw})


def test_no_water_table_is_a_stated_default():
    r = _wall()
    stated = r["defaults_used"]["water_table_depth"]
    assert stated.startswith("no water table: the backfill drained")
    assert "NHI-06-089 Sec. 10.5.3, p. 10-41" in stated
    for key in ("gamma_sat_backfill", "base_uplift", "uplift_in_bearing"):
        assert key not in r["defaults_used"]
    assert "water_table_depth_m" not in r
    assert "water_thrust_kN_per_m" not in r["sliding_basis"]


def test_the_water_reaches_the_module():
    r = _wall(water_table_depth=2.0, gamma_sat_backfill=20.0)
    assert "error" not in r, r.get("error")
    assert r["water_table_depth_m"] == 2.0
    assert r["water_height_m"] == 3.0
    assert r["water_thrust_kN_per_m"] == pytest.approx(44.1, abs=0.05)
    assert r["uplift_kN_per_m"] == pytest.approx(44.1, abs=0.05)
    assert r["sliding_basis"]["water_thrust_kN_per_m"] == pytest.approx(
        44.1, abs=0.05)
    assert r["FOS_sliding"] == pytest.approx(0.725, abs=0.001)
    assert r["FOS_overturning"] == pytest.approx(1.77, abs=0.005)
    used = r["defaults_used"]
    assert used["base_uplift"] == "linear"
    assert used["uplift_in_bearing"] is False
    assert "water_table_depth" not in used
    assert "gamma_sat_backfill" not in used
    whats = " ".join(c["what"] for c in r["choices"])
    assert "water table behind the wall" in whats
    assert "uplift (base_uplift)" in whats


def test_the_saturated_weight_defaults_to_gamma_backfill():
    r = _wall(water_table_depth=2.0)
    assert r["defaults_used"]["gamma_sat_backfill"] == 18.0


def test_uplift_inputs_need_a_water_table():
    r = _wall(base_uplift="uniform")
    assert "error" in r and "water_table_depth" in r["error"]


def test_the_package_draws_and_prints_the_water(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = D.call_agent("calc_package", "retaining_wall_package",
                     {**WALL, "water_table_depth": 2.0,
                      "gamma_sat_backfill": 20.0, "base_uplift": "uniform"})
    assert "error" not in r, r.get("error")
    assert r["FOS_sliding"] == pytest.approx(0.58, abs=0.005)
    html = "".join(f.read_text(encoding="utf-8", errors="ignore")
                   for f in tmp_path.rglob("*.htm*"))
    assert "Water Pressure Thrust" in html
    assert "Uplift on the Base" in html
    assert "UFC 3-220-20 Fig. 5-40" in html

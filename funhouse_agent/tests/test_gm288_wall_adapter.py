"""GM-288: the cantilever_wall tool takes the readings checked against the
printed manuals (retaining_walls/tests/test_gm288_printed_check.py) and
says which defaults a run took."""

import pytest

from funhouse_agent import dispatch as D

WALL = {"wall_height": 6.0, "gamma_backfill": 19.0, "phi_backfill": 32.0,
        "q_allowable": 250.0}


def _wall(**kw):
    return D.call_agent("retaining_walls", "cantilever_wall", {**WALL, **kw})


def test_the_defaults_a_run_took():
    r = _wall()
    used = r["defaults_used"]
    assert used["gamma_concrete"] == 23.6
    assert used["FOS_sliding_required"] == 1.5
    assert used["FOS_overturning_required"] == 2.0
    assert used["bearing_pressure"] == "uniform"
    assert used["base_thickness"].startswith("H/10")
    assert r["geometry"]["base_thickness_m"] == 0.6
    assert r["geometry"]["embedment_depth_m"] == 0.6
    assert r["bearing_pressure"] == "uniform"
    assert r["FOS_bearing"] == pytest.approx(
        250.0 / r["q_bearing_kPa"], rel=1e-3)


def test_the_defaults_follow_passive_and_rock():
    r = _wall(include_passive=True, founded_on_rock=True)
    used = r["defaults_used"]
    assert used["FOS_sliding_required"] == 2.0
    assert used["FOS_overturning_required"] == 1.5
    assert used["bearing_pressure"] == "linear"
    assert r["FOS_sliding_required"] == 2.0
    assert r["eccentricity_limit_m"] == pytest.approx(3.6 / 4, abs=1e-3)


def test_the_new_inputs_reach_the_module():
    r = _wall(stem_batter="front", stem_thickness_base=0.7,
              embedment_depth=1.2, include_passive=True,
              pressure_method="coulomb", backfill_slope=10.0,
              wall_friction_deg=15.0)
    assert r["geometry"]["stem_batter"] == "front"
    assert r["geometry"]["embedment_depth_m"] == 1.2
    assert r["sliding_basis"]["thrust_inclination_deg"] == 15.0
    assert "wall_friction_deg" not in r["defaults_used"]


def test_the_undrained_extras_are_reported():
    r = _wall(phi_foundation=0.0, c_foundation=60.0, include_passive=True,
              interface_gap_possible=True)
    basis = r["sliding_basis"]
    assert basis["Pp_cohesion_kN_per_m"] == pytest.approx(
        2 * 60.0 * 1.0 * 0.6, abs=0.05)
    assert basis["base_resistance_held_to_0_4V"] is True


@pytest.mark.parametrize("kw", [
    {"key_offset": 0.5},
    {"interface_gap_possible": True},          # the drained rule
    {"wall_friction_deg": 10.0},               # Rankine
    {"backfill_slope": -5.0},
])
def test_refused(kw):
    r = _wall(**kw)
    assert "error" in r

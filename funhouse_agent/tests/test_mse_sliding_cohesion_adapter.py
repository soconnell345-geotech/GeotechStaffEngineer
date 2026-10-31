"""GM-297: mse_wall and mse_lrfd_external_stability take sliding_cohesion,
the rule for the foundation cohesion in sliding. GEC-11 Vol I Eqs. 4-12,
4-13 (p. 4-21) print R = V mu with no cohesion term, and Step 3(i) names
cu for a cohesive foundation; the tools added 2/3 c_foundation L before.
The module tests are in retaining_walls/tests/test_gm297_print.py. Made-up
walls."""

import pytest

from funhouse_agent import dispatch as D

_WALL = dict(wall_height=6.0, gamma_backfill=19.0, phi_backfill=34.0,
             reinforcement_length=4.2, reinforcement_spacing=0.6,
             reinforcement_type="geogrid", reinforcement_Tallowable=40.0)
_LRFD = dict(wall_height=6.0, gamma_backfill=19.0, phi_backfill=34.0,
             reinforcement_length=4.2)


def _slide(r):
    return [c for c in r.get("choices", [])
            if "(sliding_cohesion)" in c["what"]]


def test_mse_wall_drained_by_default_with_the_cohesion_not_counted():
    r = D.call_agent("retaining_walls", "mse_wall",
                     {**_WALL, "phi_foundation": 28.0, "c_foundation": 10.0})
    assert "error" not in r, r.get("error")
    plain = D.call_agent("retaining_walls", "mse_wall",
                         {**_WALL, "phi_foundation": 28.0})
    assert r["FOS_sliding"] == pytest.approx(plain["FOS_sliding"])
    (c,) = _slide(r)
    assert c["how"] == "default" and c["chosen"].startswith("drained")
    assert r["defaults_used"]["sliding_cohesion"] == "drained"
    # no cohesion given: the rule moves nothing and is not listed
    assert "sliding_cohesion" not in plain["defaults_used"]
    assert not _slide(plain)


def test_mse_wall_undrained_given():
    r = D.call_agent("retaining_walls", "mse_wall",
                     {**_WALL, "phi_foundation": 28.0, "c_foundation": 40.0,
                      "sliding_cohesion": "undrained"})
    assert "error" not in r, r.get("error")
    (c,) = _slide(r)
    assert c["how"] == "given" and c["chosen"].startswith("undrained")
    assert "sliding_cohesion" not in r["defaults_used"]


def test_mse_wall_refuses_the_cantilever_rule():
    r = D.call_agent("retaining_walls", "mse_wall",
                     {**_WALL, "phi_foundation": 28.0, "c_foundation": 10.0,
                      "sliding_cohesion": "friction_plus_adhesion"})
    assert "friction_plus_adhesion" in r["error"]


def test_lrfd_undrained_for_phi_foundation_0():
    r = D.call_agent("retaining_walls", "mse_lrfd_external_stability",
                     {**_LRFD, "phi_foundation": 0.0, "c_foundation": 40.0})
    assert "error" not in r, r.get("error")
    assert r["sliding"]["sliding_cohesion"] == "undrained"
    (c,) = _slide(r)
    assert c["how"] == "default"
    assert r["defaults_used"]["sliding_cohesion"] == "undrained"
    assert any(a.startswith("drained: not run here") for a in
               c["alternatives"])

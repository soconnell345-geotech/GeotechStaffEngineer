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


# -- GM-297 review: the lowest sheet (GEC-11 Vol I p. 4-21 Step 3(iii)) and
# -- EN 1997-1's limit on the undrained resistance (Eq. 6.5)

def _choice(r, name):
    return [c for c in r.get("choices", []) if f"({name})" in c["what"]]


def test_mse_wall_geogrid_slides_along_its_lowest_sheet():
    r = D.call_agent("retaining_walls", "mse_wall",
                     {**_WALL, "phi_foundation": 28.0})
    assert "error" not in r, r.get("error")
    s = r["sliding_surfaces"]
    assert s["governing"] == "lowest sheet"
    assert s["FOS_base"] == pytest.approx(2.633, abs=0.0005)
    assert r["FOS_sliding"] == s["FOS_lowest_sheet"] == pytest.approx(
        2.344, abs=0.0005)
    assert r["defaults_used"]["rho_tested"].startswith(
        "2/3 tan phi_backfill")
    (c,) = _choice(r, "rho_tested")
    assert c["how"] == "default"
    tested = D.call_agent("retaining_walls", "mse_wall",
                          {**_WALL, "phi_foundation": 28.0,
                           "rho_tested": 30.0})
    assert tested["sliding_surfaces"]["tan_rho"] == pytest.approx(0.5774,
                                                                  abs=1e-4)
    assert "rho_tested" not in tested["defaults_used"]


def test_mse_wall_steel_refuses_rho_tested():
    steel = {**_WALL, "reinforcement_type": "metallic_strip",
             "reinforcement_width": 0.05,
             "reinforcement_horizontal_spacing": 0.75}
    r = D.call_agent("retaining_walls", "mse_wall", steel)
    assert "error" not in r, r.get("error")
    assert "sliding_surfaces" not in r
    assert "rho_tested" not in r["defaults_used"]
    r = D.call_agent("retaining_walls", "mse_wall",
                     {**steel, "rho_tested": 25.0})
    assert "Leave rho_tested out" in r["error"]


def test_lrfd_lowest_sheet_depth():
    r = D.call_agent("retaining_walls", "mse_lrfd_external_stability",
                     {**_LRFD, "phi_foundation": 28.0,
                      "lowest_sheet_depth": 5.7})
    assert "error" not in r, r.get("error")
    assert r["sliding"]["surface"] == "lowest sheet"
    assert r["sliding"]["CDR_governing"] == pytest.approx(1.563, abs=0.0005)
    assert r["sliding"]["CDR_base_governing"] == pytest.approx(1.755,
                                                               abs=0.0005)
    assert r["defaults_used"]["rho_tested"].startswith(
        "2/3 tan phi_backfill")
    plain = D.call_agent("retaining_walls", "mse_lrfd_external_stability",
                         {**_LRFD, "phi_foundation": 28.0})
    assert plain["defaults_used"]["lowest_sheet_depth"].startswith(
        "sliding checked at the base only")
    assert "rho_tested" not in plain["defaults_used"]
    r = D.call_agent("retaining_walls", "mse_lrfd_external_stability",
                     {**_LRFD, "rho_tested": 30.0})
    assert "lowest_sheet_depth" in r["error"]


def test_interface_gap_possible_with_the_undrained_rule():
    steel = {**_WALL, "reinforcement_type": "metallic_strip",
             "reinforcement_width": 0.05,
             "reinforcement_horizontal_spacing": 0.75,
             "phi_foundation": 0.0, "c_foundation": 120.0}
    plain = D.call_agent("retaining_walls", "mse_wall", steel)
    gap = D.call_agent("retaining_walls", "mse_wall",
                       {**steel, "interface_gap_possible": True})
    assert "error" not in gap, gap.get("error")
    assert plain["FOS_sliding"] == pytest.approx(3.340, abs=0.0005)
    assert gap["FOS_sliding"] == pytest.approx(1.981, abs=0.0005)
    assert plain["defaults_used"]["interface_gap_possible"] is False
    (c,) = _choice(gap, "interface_gap_possible")
    assert c["how"] == "given"
    lrfd = D.call_agent("retaining_walls", "mse_lrfd_external_stability",
                        {**_LRFD, "phi_foundation": 0.0,
                         "c_foundation": 120.0,
                         "interface_gap_possible": True})
    assert lrfd["sliding"]["CDR_governing"] == pytest.approx(1.3205,
                                                             abs=0.0006)
    drained = D.call_agent("retaining_walls", "mse_wall",
                           {**_WALL, "phi_foundation": 28.0,
                            "interface_gap_possible": True})
    assert "error" in drained

"""GM-289: the group_uplift tool, the rigid-cap tools' note on a group in
net tension, and the pile group calc package's Group Uplift section. A
group not in net tension is unchanged (the GM-289 replay; here, no
group uplift key). Values are made up, but the North Abutment row,
GEC-12 Vol. III Table D-18 (839.6 kips)."""

import json

import pytest

from funhouse_agent import dispatch as D

FT = 0.3048
PCF = 0.157087464
KIP = 4.448222


def _call(agent, method, params):
    r = D.call_agent(agent, method, params)
    return json.loads(r) if isinstance(r, str) else r


def _choice(result, what):
    return next(c for c in result["choices"] if c["what"] == what)


def test_tool_north_abutment_row():
    """One row of 9 piles 1 ft wide at 5 ft (41 ft to the outer faces), toes
    at 27 ft: GEC-12 Vol. III Table D-18, 839.6 kips; each pile's factored
    tension resistance 46 kips (Table D-16's requirement), so the piles'
    sum 414 kips governs over the block's 0.50 x 839.6 = 419.8."""
    r = _call("pile_group", "group_uplift", {
        "n_rows": 1, "n_cols": 9, "spacing": 5 * FT, "pile_diameter": FT,
        "pile_length": 27 * FT, "soil_type": "cohesionless",
        "layers": [{"thickness": 10 * FT, "unit_weight": 105.0 * PCF},
                   {"thickness": 10 * FT, "unit_weight": 42.6 * PCF},
                   {"thickness": 20 * FT, "unit_weight": 49.6 * PCF}],
        "axial_capacity_tension": 46 * KIP, "uplift_load": 406 * KIP})
    assert "error" not in r, r
    assert r["block_length_m"] / FT == pytest.approx(41.0, abs=0.01)
    assert r["group_uplift_block_kN"] / KIP == pytest.approx(839.6, abs=0.1)
    assert r["group_uplift_phi"] == 0.5
    assert r["piles_tension_resistance_kN"] / KIP == pytest.approx(414.0,
                                                                    abs=0.1)
    assert r["group_uplift_governs"] == "individual piles"
    assert r["group_uplift_adequate"] is True
    assert len(r["block_layers"]) == 3
    c = _choice(r, "group uplift resistance")
    assert "the piles' sum governs" in c["chosen"]
    assert "UFC 3-220-20 Eq. 6-40" in c["alternatives"][0]
    assert "1H:4V" in _choice(r, "group uplift block")["chosen"]


def test_tool_without_the_piles_gives_the_block_and_says_so():
    r = _call("pile_group", "group_uplift", {
        "n_rows": 3, "n_cols": 3, "spacing": 1.0, "pile_diameter": 0.3,
        "pile_length": 12.0, "soil_type": "cohesive", "unit_weight": 8.0,
        "cu": 30.0})
    assert "error" not in r, r
    # B = Z = 2.3 m: 2.3 x 2.3 x 12 x 8 + 2 x 12 x 4.6 x 30 = 3819.8 kN
    assert r["group_uplift_block_kN"] == pytest.approx(3819.8, abs=0.1)
    assert r["group_uplift_block_factored_kN"] == pytest.approx(1909.9,
                                                                abs=0.1)
    assert r["group_uplift_resistance_kN"] is None
    assert "axial_capacity_tension" in r["group_uplift_note"]
    assert "axial_capacity_tension" in r["defaults_used"]


def test_tool_small_driven_group_gives_ufc_reading():
    r = _call("pile_group", "group_uplift", {
        "n_rows": 2, "n_cols": 2, "spacing": 1.0, "pile_diameter": 0.3,
        "pile_length": 8.0, "soil_type": "cohesionless", "unit_weight": 9.0})
    c = _choice(r, "resistance factor for group uplift")
    assert c["chosen"] == "phi_ug = 0.50"
    assert "fewer than 5 piles): phi_ug 0.40" in c["alternatives"][0]


def test_tool_refuses_cohesive_without_cu():
    r = _call("pile_group", "group_uplift", {
        "n_rows": 2, "n_cols": 3, "spacing": 1.0, "pile_diameter": 0.3,
        "pile_length": 8.0, "soil_type": "cohesive", "unit_weight": 9.0})
    assert "cu" in r["error"]


@pytest.mark.parametrize("method, extra", [
    ("pile_group_simple", {}),
    ("pile_group_6dof", {"axial_stiffness": 150000.0})])
def test_rigid_cap_note_only_in_net_tension(method, extra):
    p = {"n_rows": 2, "n_cols": 2, "spacing": 1.5, **extra}
    up = _call("pile_group", method, {**p, "Vz": -400.0})
    assert "group_uplift" in up["group_uplift_note"]
    # piles in tension under a moment, the group in compression: no note
    down = _call("pile_group", method, {**p, "Vz": 400.0, "My": 900.0})
    assert down["max_tension_kN"] > 0
    assert "group_uplift_note" not in down


def _package(tmp_path, monkeypatch, **params):
    monkeypatch.chdir(tmp_path)
    r = _call("calc_package", "pile_group_package", {
        "n_rows": 3, "n_cols": 3, "spacing": 1.2, "pile_diameter": 0.4,
        "format": "html", **params})
    assert r.get("status") == "success", r
    html = open(r["output_path"], encoding="utf-8").read()
    return r, html


def test_package_checks_a_group_in_net_tension(tmp_path, monkeypatch):
    r, html = _package(tmp_path, monkeypatch, Vz=-900.0,
                       axial_capacity_tension=200.0, pile_length=12.0,
                       soil_type="cohesionless", unit_weight=9.0)
    # B = Z = 2.8 m: V = 2.8^2 x 12 + 144 x 5.6 / 4 + 1728 / 12 = 439.68 m3
    assert r["group_uplift_block_kN"] == pytest.approx(3957.1, abs=0.1)
    assert r["group_uplift_resistance_kN"] == 1800.0
    assert r["group_uplift_governs"] == "individual piles"
    assert r["group_uplift_utilization"] == 0.5
    assert "Group Uplift" in html and "R_r,group" in html


def test_package_in_net_tension_says_what_it_needs(tmp_path, monkeypatch):
    r, html = _package(tmp_path, monkeypatch, Vz=-900.0)
    assert "pile_length" in r["group_uplift_note"]
    assert "group_uplift_resistance_kN" not in r
    assert "the block is not checked here" in html


def test_package_not_in_net_tension_has_no_uplift(tmp_path, monkeypatch):
    r, html = _package(tmp_path, monkeypatch, Vz=900.0, My=1200.0,
                       axial_capacity_tension=200.0)
    assert not [k for k in r if k.startswith("group_uplift")]
    assert "Group Uplift" not in html
    r, _ = _package(tmp_path, monkeypatch, Vz=900.0, pile_length=12.0)
    assert "does not apply" in r["group_uplift_note"]

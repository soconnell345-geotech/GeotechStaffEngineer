"""GM-289 and its review: the group_uplift tool, the rigid-cap tools' note
where a pile carries tension, and the pile group calc package's Group
Uplift section over the piles in tension (as GEC-12 Vol. III checks the
back row in tension of a group in compression) and over a group in net
tension. A group with no pile in tension is unchanged (the GM-289 replay;
here, no group uplift key). Values are made up, but GEC-12 Vol. III's
North Abutment (Tables D-11 to D-18)."""

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


def test_tool_small_driven_group_takes_ufc_and_gives_gec12_beside():
    """GM-310 (the call for OWN-118): 4 driven piles are a small group by
    UFC 3-220-20 Table 6-28 note A (fewer than 5), the default; GEC-12 p.
    225's fewer than 3 is given beside it. Made-up group."""
    r = _call("pile_group", "group_uplift", {
        "n_rows": 2, "n_cols": 2, "spacing": 1.0, "pile_diameter": 0.3,
        "pile_length": 8.0, "soil_type": "cohesionless", "unit_weight": 9.0})
    c = _choice(r, "resistance factor for group uplift")
    assert c["chosen"] == ("phi_ug = 0.40 (0.50 reduced by 20 % for a small "
                           "group of 4 piles)")
    assert r["group_uplift_phi"] == pytest.approx(0.40)
    assert "Table 6-28 note A" in c["basis"] and "p. 225" in c["basis"]
    (alt,) = c["alternatives"]
    assert "fewer than 3 piles" in alt and "phi_ug 0.50" in alt
    assert "the block " in alt and alt.endswith(" kN")
    said = float(alt.rsplit("the block ", 1)[1].removesuffix(" kN"))
    assert said == pytest.approx(0.5 * r["group_uplift_block_kN"], abs=0.11)
    # 5 piles or more, and 2: the two readings agree, nothing beside
    for rows, cols in ((2, 3), (1, 2)):
        r = _call("pile_group", "group_uplift", {
            "n_rows": rows, "n_cols": cols, "spacing": 1.0,
            "pile_diameter": 0.3, "pile_length": 8.0,
            "soil_type": "cohesionless", "unit_weight": 9.0})
        c = _choice(r, "resistance factor for group uplift")
        assert not c.get("alternatives")


def test_tool_refuses_cohesive_without_cu():
    r = _call("pile_group", "group_uplift", {
        "n_rows": 2, "n_cols": 3, "spacing": 1.0, "pile_diameter": 0.3,
        "pile_length": 8.0, "soil_type": "cohesive", "unit_weight": 9.0})
    assert "cu" in r["error"]


def test_tool_counts_the_piles_under_the_cap_for_a_small_group():
    """A row of 4 drilled shafts in tension of a group of 8: not a small
    group (UFC 3-220-20 Table 6-28 note A counts the group's piles, GEC-12
    p. 225 'per substructure unit')."""
    p = {"n_rows": 1, "n_cols": 4, "spacing": 1.5, "pile_diameter": 0.6,
         "pile_length": 10.0, "soil_type": "cohesionless", "unit_weight": 9.0,
         "pile_type": "drilled_shaft"}
    assert _call("pile_group", "group_uplift", p)["group_uplift_phi"] == 0.36
    r = _call("pile_group", "group_uplift", {**p, "piles_under_cap": 8})
    assert r["group_uplift_phi"] == 0.45
    assert "piles_under_cap" in _call("pile_group", "group_uplift", {
        **p, "piles_under_cap": 3})["error"]


@pytest.mark.parametrize("method, extra", [
    ("pile_group_simple", {}),
    ("pile_group_6dof", {"axial_stiffness": 150000.0})])
def test_rigid_cap_note_where_a_pile_carries_tension(method, extra):
    """GM-289 review: GEC-12 Vol. III checks the back row in tension of a
    group in compression as a block (pp. 51, 203), so any pile in tension
    gets the note; a group with none does not."""
    p = {"n_rows": 2, "n_cols": 2, "spacing": 1.5, **extra}
    up = _call("pile_group", method, {**p, "Vz": -400.0})
    assert up["group_uplift_note"].startswith("Every pile carries tension")
    assert "uplift_load 400" in up["group_uplift_note"]
    # two piles in tension under a moment, the group in compression
    down = _call("pile_group", method, {**p, "Vz": 400.0, "My": 900.0})
    note = down["group_uplift_note"]
    assert note.startswith("2 of 4 piles carry tension, 400.0 kN in all")
    assert "piles_under_cap 4" in note and "uplift_load 400" in note
    assert "net uplift" not in note
    assert "group_uplift_note" not in _call("pile_group", method,
                                            {**p, "Vz": 400.0})


def _package(tmp_path, monkeypatch, **params):
    monkeypatch.chdir(tmp_path)
    p = {"n_rows": 3, "n_cols": 3, "spacing": 1.2, "pile_diameter": 0.4,
         "format": "html", **params}
    r = _call("calc_package", "pile_group_package",
              {k: v for k, v in p.items() if v is not None})
    assert r.get("status") == "success", r
    html = open(r["output_path"], encoding="utf-8").read()
    return r, html


SAND = dict(axial_capacity_tension=200.0, pile_length=12.0,
            soil_type="cohesionless", unit_weight=9.0)


def test_package_checks_a_group_in_net_tension(tmp_path, monkeypatch):
    r, html = _package(tmp_path, monkeypatch, Vz=-900.0, **SAND)
    # B = Z = 2.8 m: V = 2.8^2 x 12 + 144 x 5.6 / 4 + 1728 / 12 = 439.68 m3
    assert r["group_uplift_block_kN"] == pytest.approx(3957.1, abs=0.1)
    assert r["group_uplift_resistance_kN"] == 1800.0
    assert r["group_uplift_governs"] == "individual piles"
    assert r["group_uplift_utilization"] == 0.5
    assert r["group_uplift_over"] == "the whole group"
    assert "Group Uplift" in html and "R_r,group" in html


def test_package_in_net_tension_says_what_it_needs(tmp_path, monkeypatch):
    r, html = _package(tmp_path, monkeypatch, Vz=-900.0)
    assert "pile_length" in r["group_uplift_note"]
    assert "group_uplift_resistance_kN" not in r
    assert "block is not checked here" in html


def test_package_checks_the_row_in_tension_of_a_group_in_compression(
        tmp_path, monkeypatch):
    """Vz 900 kN and My 1,200 kN-m: the column at x = -1.2 m carries 66.7
    kN of tension a pile, 200 kN in all. Its block, B = 0.4 m, Z = 2.8 m:
    V = 0.4 x 2.8 x 12 + 144 x 3.2 / 4 + 1728 / 12 = 272.64 m3, W =
    2,453.8 kN; the piles' sum 3 x 200 = 600 kN governs; the cap's 450 kN
    taken 3/9."""
    r, html = _package(tmp_path, monkeypatch, Vz=900.0, My=1200.0,
                       cap_weight=450.0, **SAND)
    assert r["group_uplift_over"] == "the 3 piles in tension"
    assert r["uplift_load_kN"] == pytest.approx(200.0, abs=0.2)
    assert r["group_uplift_block_kN"] == pytest.approx(2453.8 + 150.0,
                                                       abs=0.1)
    assert r["group_uplift_resistance_kN"] == 600.0
    assert r["group_uplift_utilization"] == pytest.approx(0.333, abs=0.002)
    assert "group_uplift_checks" not in r
    assert "Over the 3 piles in tension" in html
    assert "3/9 of the cap" in html


def test_package_the_reviewers_group(tmp_path, monkeypatch):
    """The GM-289 review's made-up group: 2 x 9 piles, Vz 11,000 kN, Mx
    8,800 kN-m. Each back pile passes alone (204 of 250 kN); the back row
    as a block, B = 0.3 m, Z = 12.3 m, 8 m in sand at 9 kN/m3: V = 29.52 +
    201.60 + 42.67 = 273.79 m3, 0.50 x 2,464.1 = 1,232.0 kN against its
    1,833 kN: the block governs and the row fails."""
    group = dict(n_rows=2, n_cols=9, spacing_x=1.5, spacing_y=1.2,
                 Vz=11000.0, Mx=8800.0, axial_capacity_tension=250.0)
    alone = _call("pile_group", "pile_group_simple", group)
    assert alone["max_utilization"] == pytest.approx(0.815, abs=0.001)
    r, _ = _package(tmp_path, monkeypatch, **group, spacing=None,
                    pile_diameter=0.3, pile_length=8.0,
                    soil_type="cohesionless", unit_weight=9.0)
    assert r["group_uplift_block_factored_kN"] == pytest.approx(1232.0,
                                                                abs=0.1)
    assert r["uplift_load_kN"] == pytest.approx(1833.3, abs=0.2)
    assert r["group_uplift_governs"] == "block"
    assert r["group_uplift_utilization"] == pytest.approx(1.488, abs=0.001)
    assert r["group_uplift_adequate"] is False


def test_package_north_abutment_back_row(tmp_path, monkeypatch):
    """GEC-12 Vol. III North Abutment, Group Configuration 1 (Tables D-11,
    D-12): 2 rows of 9 piles at 5 ft by 4 ft, Q 2,505 kips, Muy 6,625
    kip-ft. The rigid cap puts 403.8 kips of tension on the back row
    (Table D-13's free body, 406); its block at 27 ft is Table D-18's
    839.6 kips, 419.8 factored; with the 46 kips a pile Table D-14 asks
    for, the piles' 414 kips govern."""
    r, _ = _package(tmp_path, monkeypatch, n_rows=2, n_cols=9,
                    spacing_x=5 * FT, spacing_y=4 * FT, spacing=None,
                    pile_diameter=FT, Vz=2505 * KIP, Mx=6625 * KIP * FT,
                    axial_capacity_tension=46 * KIP, pile_length=27 * FT,
                    soil_type="cohesionless",
                    layers=[{"thickness": 10 * FT, "unit_weight": 105.0 * PCF},
                            {"thickness": 10 * FT, "unit_weight": 42.6 * PCF},
                            {"thickness": 20 * FT, "unit_weight": 49.6 * PCF}])
    assert r["group_uplift_over"] == "the 9 piles in tension"
    assert r["uplift_load_kN"] / KIP == pytest.approx(403.8, abs=0.2)
    assert r["group_uplift_block_kN"] / KIP == pytest.approx(839.6, abs=0.1)
    assert r["group_uplift_block_factored_kN"] / KIP == pytest.approx(
        419.8, abs=0.1)
    assert r["group_uplift_governs"] == "individual piles"
    assert r["group_uplift_adequate"] is True


def test_package_net_uplift_with_a_row_in_tension_checks_both(
        tmp_path, monkeypatch):
    """Vz -300 kN, My 1,200 kN-m: six piles in tension (700 kN), the group
    in net uplift (300 kN). The outer row of three at x = -1.2 m (200 kN a
    pile), their 600 kN governing: 1.0 (review 2); the six, B = 1.6 m, Z =
    2.8 m, their 1,200 kN governing: 0.583; the whole group: 300 / 1,800 =
    0.167."""
    r, html = _package(tmp_path, monkeypatch, Vz=-300.0, My=1200.0,
                       cap_weight=900.0, **SAND)
    over = {c["over"]: c for c in r["group_uplift_checks"]}
    assert list(over) == ["the outer row of 3 piles in tension",
                          "the 6 piles in tension", "the whole group"]
    assert over["the outer row of 3 piles in tension"][
        "group_uplift_utilization"] == pytest.approx(1.0, abs=0.001)
    assert over["the 6 piles in tension"]["group_uplift_utilization"] == \
        pytest.approx(0.583, abs=0.001)
    assert over["the whole group"]["group_uplift_utilization"] == \
        pytest.approx(0.167, abs=0.001)
    assert r["group_uplift_over"] == "the outer row of 3 piles in tension"
    assert r["group_uplift_governs"] == "individual piles"
    assert "300.0 kN, 3/9 of the cap&#39;s 900 kN" in html \
        or "300.0 kN, 3/9 of the cap's 900 kN" in html


def test_package_the_outer_row_where_two_rows_carry_tension(tmp_path,
                                                            monkeypatch):
    """GM-289 review 2, the review's made-up group: 4 x 9 piles, Vz 3,600
    kN, 8 m in sand at 9 kN/m3. At Mx 10,700 kN-m only the outer row
    carries tension, its block (B = 0.3 m, Z = 12.3 m) 1,232.0 kN against
    1,775 kN: 1.44. At Mx 10,900 kN-m the second row carries 0.9 kN a pile:
    the block over both rows, B = 1.5 m, Z = 12.3 m, V = 147.60 + 220.80 +
    42.67 = 411.07 m3, 0.50 x 3,699.6 = 1,849.8 kN against 1,833 kN, passes
    (0.99), while the outer row's block still fails (1.48) and governs."""
    group = dict(n_rows=4, n_cols=9, spacing_x=1.5, spacing_y=1.2,
                 spacing=None, pile_diameter=0.3, Vz=3600.0,
                 axial_capacity_tension=250.0, pile_length=8.0,
                 soil_type="cohesionless", unit_weight=9.0)
    r, _ = _package(tmp_path, monkeypatch, Mx=10700.0, **group)
    assert r["group_uplift_over"] == "the 9 piles in tension"
    assert r["group_uplift_utilization"] == pytest.approx(1.441, abs=0.001)
    r, html = _package(tmp_path, monkeypatch, Mx=10900.0, **group)
    over = {c["over"]: c for c in r["group_uplift_checks"]}
    assert list(over) == ["the outer row of 9 piles in tension",
                          "the 18 piles in tension"]
    both = over["the 18 piles in tension"]
    assert both["group_uplift_block_factored_kN"] == pytest.approx(1849.8,
                                                                   abs=0.1)
    assert both["group_uplift_utilization"] == pytest.approx(0.991,
                                                             abs=0.001)
    assert r["group_uplift_over"] == "the outer row of 9 piles in tension"
    assert r["group_uplift_block_factored_kN"] == pytest.approx(1232.0,
                                                                abs=0.1)
    assert r["group_uplift_utilization"] == pytest.approx(1.481, abs=0.001)
    assert r["group_uplift_adequate"] is False
    assert "Over the outer row of 9 piles in tension:" in html
    assert "over the outer row of 9 piles in tension: the soil" in html


@pytest.mark.parametrize("moments, says", [
    ({"Mx": 720.0, "My": 720.0}, "do not fill a rectangle"),
    ({"Mx": 480.0, "My": 480.0}, "One pile carries tension")])
def test_package_piles_in_tension_not_a_block(tmp_path, monkeypatch,
                                              moments, says):
    """3 x 3 at 1 m, Vz 900 kN: 720 kN-m both ways puts an L of three piles
    in tension around a pile in compression; 480 kN-m, one corner pile."""
    r, html = _package(tmp_path, monkeypatch, spacing=1.0, Vz=900.0,
                       **moments, **SAND)
    assert says in r["group_uplift_note"]
    assert "group_uplift_resistance_kN" not in r
    assert says in html


def test_package_with_no_pile_in_tension_has_no_uplift(tmp_path,
                                                       monkeypatch):
    p = dict(Vz=900.0, My=500.0, axial_capacity_tension=200.0)
    assert _call("pile_group", "pile_group_simple", {
        "n_rows": 3, "n_cols": 3, "spacing": 1.2, **p})["max_tension_kN"] == 0
    r, html = _package(tmp_path, monkeypatch, **p)
    assert not [k for k in r if k.startswith("group_uplift")]
    assert "Group Uplift" not in html
    r, _ = _package(tmp_path, monkeypatch, Vz=900.0, pile_length=12.0)
    assert "does not apply" in r["group_uplift_note"]

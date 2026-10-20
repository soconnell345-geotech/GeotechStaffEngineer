"""GM-191/192: the ground anchor's load transfer, factor of safety and
minimum lengths as GEC-4 (FHWA-IF-99-015) prints them, pinned to its two
design examples (App. A)."""

import math

import pytest

from soe.anchor_design import (
    ROCK_TYPES, compute_bond_length, compute_unbonded_length,
    design_ground_anchor, distance_behind_failure_surface, get_bond_stress,
    list_bond_stress_types,
)

#: GEC-4 Table 6 (p. 71) and Table 8 (p. 74), ultimate load transfer kN/m.
PRINTED = {
    "sand_gravel_loose": 145, "sand_gravel_medium": 220,
    "sand_gravel_dense": 290,
    "sand_loose": 100, "sand_medium": 145, "sand_dense": 190,
    "sand_silt_loose": 70, "sand_silt_medium": 100, "sand_silt_dense": 130,
    "silt_clay_stiff": 30, "silt_clay_hard": 60,
    "granite_basalt": 730, "dolomitic_limestone": 580,
    "soft_limestone": 440, "sandstone": 440, "slates_hard_shales": 360,
    "soft_shales": 150,
}


def test_every_row_is_a_printed_row():
    assert set(list_bond_stress_types()) == set(PRINTED)
    for name, q in PRINTED.items():
        row = get_bond_stress(name)
        assert row["load_transfer_kN_per_m"] == q, name
        table = "Table 8 (p. 74)" if name in ROCK_TYPES else "Table 6 (p. 71)"
        assert row["source"] == f"GEC-4 {table}", name
    assert set(ROCK_TYPES) == {"granite_basalt", "dolomitic_limestone",
                               "soft_limestone", "sandstone",
                               "slates_hard_shales", "soft_shales"}


@pytest.mark.parametrize("old, points_to", [
    ("gravel", "sand_gravel_medium"), ("clay_stiff", "silt_clay_stiff"),
    ("clay_hard", "silt_clay_hard"), ("rock_soft", "soft_shales"),
    ("rock_medium", "soft_limestone"), ("rock_hard", "granite_basalt")])
def test_a_removed_row_says_where_it_went(old, points_to):
    with pytest.raises(ValueError, match="removed") as err:
        design_ground_anchor(300, 3, 8, 32, soil_type=old)
    assert points_to in str(err.value)


def test_gec4_example_1_soil():
    """App. A, design example 1 (pp. A-7 to A-9, Figure A-3): H 10 m,
    phi' 33, anchors at 15 deg from 2.5 m and 6.25 m deep, DL 445 kN,
    medium dense silty sand at Table 6's 100 kN/m and FS 2.0: bond length
    (445)(2.0)/100 = 8.9 m; unbonded 5.7 m and 3.9 m, "minimum unbonded
    length for strand anchor is 4.5 m"."""
    lengths = {}
    for depth in (2.5, 6.25):
        for tendon in ("bar_32mm", "strand_15mm"):
            r = design_ground_anchor(445, depth, 10, 33,
                                     soil_type="sand_silt_medium",
                                     tendon_type=tendon)
            assert r.load_transfer_kN_per_m == 100
            assert r.FOS_bond == 2.0
            assert r.bond_length_m == pytest.approx(8.9, abs=0.005)
            lengths[depth, tendon] = r.unbonded_length_m
    assert lengths[2.5, "bar_32mm"] == pytest.approx(5.7, abs=0.05)
    assert lengths[2.5, "strand_15mm"] == pytest.approx(5.7, abs=0.05)
    assert lengths[6.25, "bar_32mm"] == pytest.approx(3.9, abs=0.05)
    assert lengths[6.25, "strand_15mm"] == 4.5


def test_gec4_example_2_rock():
    """App. A, design example 2 (pp. A-27, A-28): mudstone taken as Table
    8's hard shales and slates, 360 kN/m, FS 3.0: 7.5 m carries 900 kN,
    and DL 636 kN needs (636)(3.0)/360 = 5.3 m."""
    r = design_ground_anchor(636, 3, 8, 30, soil_type="slates_hard_shales")
    assert r.FOS_bond == 3.0
    assert r.bond_length_m == pytest.approx(5.3, abs=0.005)
    r = design_ground_anchor(900, 3, 8, 30, soil_type="slates_hard_shales")
    assert r.bond_length_m == pytest.approx(7.5, abs=0.005)


def test_a_load_per_metre_does_not_depend_on_the_hole():
    """GM-191: Table 6's kN/m is used per metre. Until then 145 was read
    as kPa over the hole's perimeter: a 150 mm hole gave 68 kN/m and a
    bond length 2.1 times Table 6's (DL 300, FS 2: 8.78 m against 4.14
    m)."""
    q = compute_bond_length(300, load_transfer_kN_per_m=145, FOS_bond=2.0,
                            min_bond_length=0)
    assert q == pytest.approx(600 / 145, abs=0.005)
    old = compute_bond_length(300, bond_stress_kPa=145,
                              drill_diameter_mm=150, FOS_bond=2.0)
    assert old == pytest.approx(600 / (145 * math.pi * 0.150), abs=0.005)
    assert old / q == pytest.approx(1 / (math.pi * 0.150), rel=1e-3)
    a = design_ground_anchor(445, 3, 8, 32, soil_type="sand_loose")
    b = design_ground_anchor(445, 3, 8, 32, soil_type="sand_loose",
                             drill_diameter_mm=300)
    assert a.bond_length_m == b.bond_length_m == pytest.approx(8.9, abs=0.005)
    assert a.drill_diameter_mm is None and a.bond_stress_kPa is None
    assert any("drill_diameter_mm is not used" in n for n in b.notes)


def test_a_given_stress_acts_on_the_perimeter():
    """GEC-4 Table 7 (p. 73) prints stresses; one given as bond_stress_kPa
    acts over the as-drilled hole (p. 72)."""
    r = design_ground_anchor(445, 3, 8, 32, bond_stress_kPa=200,
                             drill_diameter_mm=200, soil_type="soil")
    q = 200 * math.pi * 0.200
    assert r.load_transfer_kN_per_m == pytest.approx(q, abs=0.05)
    assert r.bond_length_m == pytest.approx(2 * 445 / q, abs=0.01)
    assert r.drill_diameter_mm == 200


def test_a_given_load_transfer_is_used_per_metre():
    r = design_ground_anchor(445, 3, 8, 32, load_transfer_kN_per_m=120,
                             soil_type="soil")
    assert r.bond_length_m == pytest.approx(2 * 445 / 120, abs=0.005)
    with pytest.raises(ValueError, match="not both"):
        design_ground_anchor(445, 3, 8, 32, load_transfer_kN_per_m=120,
                             bond_stress_kPa=200, soil_type="soil")


@pytest.mark.parametrize("given", [{"bond_stress_kPa": 1000},
                                   {"load_transfer_kN_per_m": 400}])
def test_a_given_load_transfer_needs_its_ground(given):
    """GM-199: the value does not say the ground (GEC-4 Table 7, p. 73:
    0.15 to 3.1 MPa in rock, 0.03 to 1.38 MPa in soil), so a given one
    with no soil_type is refused and the call asked for it. It took the
    default's soil: a 1,000 kPa rock bond stress over a 150 mm hole
    (471 kN/m), DL 1,000 kN, got FS 2.0 and 4.5 m where rock's FS 3.0
    gives 6.37 m."""
    with pytest.raises(ValueError, match="soil_type='soil' or 'rock'"):
        design_ground_anchor(1000, 3, 8, 32, **given)


def test_the_ground_alone_beside_a_given_load_transfer():
    """GM-199: soil_type 'rock' or 'soil' names the ground of a given bond
    stress or load transfer: FS 3.0 in rock (GEC-4 p. 74), 2.0 in soil
    (p. 71), and the minimum bond length of that ground (p. E-15: 3 m for
    a bar in rock, else 4.5 m)."""
    q = 1000 * math.pi * 0.150
    rock = design_ground_anchor(1000, 3, 8, 32, bond_stress_kPa=1000,
                                soil_type="rock")
    assert rock.ground == "rock" and rock.FOS_bond == 3.0
    assert rock.bond_length_m == pytest.approx(3 * 1000 / q, abs=0.005)
    soil = design_ground_anchor(1000, 3, 8, 32, bond_stress_kPa=1000,
                                soil_type="soil")
    assert soil.ground == "soil" and soil.FOS_bond == 2.0
    assert soil.bond_length_m == 4.5            # 2 x 1000 / 471 = 4.24
    bar = design_ground_anchor(300, 3, 8, 32, bond_stress_kPa=1000,
                               soil_type="rock", tendon_type="bar_32mm")
    assert bar.bond_length_m == 3.0
    # a row beside a given stress still names the ground
    assert design_ground_anchor(1000, 3, 8, 32, bond_stress_kPa=1000,
                                soil_type="sandstone").FOS_bond == 3.0
    fs = next(c for c in rock.choices
              if c["what"] == "factor of safety on bond")
    assert "soil_type 'rock'" in fs["basis"] and "GM-199" in fs["basis"]


def test_the_ground_alone_has_no_load_transfer():
    with pytest.raises(ValueError, match="names the ground alone"):
        design_ground_anchor(300, 3, 8, 32, soil_type="rock")


def test_the_factor_of_safety_follows_the_ground():
    """GEC-4 p. 71: 2.0 in soil; p. 74: 3.0 in competent rock. A given
    one wins."""
    assert design_ground_anchor(300, 3, 8, 32).FOS_bond == 2.0
    rock = design_ground_anchor(300, 3, 8, 32, soil_type="granite_basalt")
    assert rock.FOS_bond == 3.0
    weak = design_ground_anchor(300, 3, 8, 32, soil_type="soft_shales",
                                FOS_bond=2.0)
    assert weak.FOS_bond == 2.0
    fs = next(c for c in rock.choices
              if c["what"] == "factor of safety on bond")
    assert fs["how"] == "default" and "p. 74" in fs["basis"]
    assert any("2.0" in a for a in fs["alternatives"])


def test_the_minimum_bond_length():
    """GEC-4 App. E Part 3 B (p. E-15): 4.5 m for strand and bar tendons
    in soil; in rock 4.5 m for strand and 3 m for bar tendons. It was 3 m
    in every ground."""
    def bond(soil, tendon):
        return design_ground_anchor(100, 3, 8, 32, soil_type=soil,
                                    tendon_type=tendon).bond_length_m
    assert bond("sand_gravel_dense", "strand_15mm") == 4.5
    assert bond("sand_gravel_dense", "bar_32mm") == 4.5
    assert bond("granite_basalt", "strand_15mm") == 4.5
    assert bond("granite_basalt", "bar_32mm") == 3.0


def test_the_minimum_unbonded_length():
    """GEC-4 Sec. 5.3.4 (p. 67): 4.5 m for strand tendons and 3 m for bar
    tendons. A bar was held to 4.5 m."""
    bar = design_ground_anchor(300, 7.5, 8, 32, tendon_type="bar_32mm")
    strand = design_ground_anchor(300, 7.5, 8, 32)
    assert bar.unbonded_length_m == 3.0
    assert strand.unbonded_length_m == 4.5


def test_the_distance_behind_the_failure_surface():
    """GEC-4 App. E Part 3 C (p. E-15): at least 1.5 m or 20 percent of
    the height of the wall, whichever is greater."""
    assert distance_behind_failure_surface(5.0) == 1.5
    assert distance_behind_failure_surface(7.5) == 1.5
    assert distance_behind_failure_surface(15.0) == 3.0
    # a horizontal anchor at the excavation level, phi 0 (surface at 45
    # deg): the limit line is H/5 normal to the surface, H/5 x sqrt(2)
    # along the horizontal
    L = compute_unbonded_length(15.0, 15.0, 0.0, anchor_angle_deg=0.0,
                                min_free_length=0.0)
    assert L == pytest.approx(3.0 * math.sqrt(2.0), abs=1e-9)

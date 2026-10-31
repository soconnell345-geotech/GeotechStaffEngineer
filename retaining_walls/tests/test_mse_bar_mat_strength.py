"""GM-262: the tensile strength of a steel bar mat, per metre of its width.

GEC-11 Vol I (FHWA-NHI-10-024), read on the pages:
  - Eq. 3-11a (p. 3-22): Tal = Fy Ac / b, "in strength per unit
    reinforcement width"; Fig. 3-5 (p. 3-24), for metallic grids and bar
    mats: Ac = (No. of longitudinal bars) x (D*)^2 / 4 (printed with no pi),
    D* the bar's diameter corrected for corrosion loss;
  - Table 4-7 (p. 4-48): phi 0.65 for grids on a rigid facing, static;
  - p. 4-77 (checklist IX.5): "0.55 Fy for strips and 0.48Fy for bar mats".
GEC-11 Vol II Example E4 (FHWA-NHI-10-025): W11 D = 0.374 in., 0.318 in.
after 75 years (p. E4-14); pi (0.318)^2 / 4 = 0.0795 in.^2, Tn = 65 ksi x
0.0795 = 5.17 k a wire, Tr = 0.65 Tn = 3.36 k (p. E4-15); Level 4 is
"4W11 + W11x0.5'", 4 wires at 0.5 ft, one mat a 5 ft panel (p. E4-18).

Before GM-262 the built-in W11 grid's Tal was 32.0 kN/m (Fy times one
uncorroded W11 wire, read per metre of grid width since GM-256): at E4's
layout FOS_rupture 0.947 at the top level to 0.241 at the bottom, where
E4's mat passes Levels 1 to 4.
"""

import dataclasses

import pytest

from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.mse import analyze_mse_wall, check_internal_stability
from retaining_walls.reinforcement import (
    BAR_MAT_ALLOWABLE_FY, Reinforcement, WELDED_WIRE_GRID_W11,
    steel_grid_Tal,
)

FT = 0.3048
IN = 0.0254
KSI = 6894.757
KIP = 4.4482216
PCF = 0.157087
KSF = 47.88


def _e4_wall():
    """E4: H = 25.64 ft, L = 18 ft, Sv 2.5 ft, 250 psf live load."""
    return MSEWallGeometry(wall_height=25.64 * FT, reinforcement_length=18 * FT,
                           reinforcement_spacing=2.5 * FT,
                           surcharge=0.25 * KSF)


def _e4_mat():
    """The built-in mat at E4's spacing, one a 5 ft panel."""
    return dataclasses.replace(WELDED_WIRE_GRID_W11,
                               horizontal_spacing=5.0 * FT)


def test_one_corroded_W11_wire_is_E4s():
    """p. E4-15: Tn = 5.17 k a wire, Tr = 0.65 Tn = 3.36 k (pi taken, as
    E4 takes it; Fig. 3-5 prints (D*)^2 / 4)."""
    Tn = steel_grid_Tal(1, 0.318 * IN, 65.0 * KSI, 1.0)
    assert Tn / KIP == pytest.approx(5.17, rel=0.002)
    assert 0.65 * Tn / KIP == pytest.approx(3.36, rel=0.002)
    # Ac = n pi D*^2 / 4: four wires are four times one, per their b
    assert steel_grid_Tal(4, 0.318 * IN, 65.0 * KSI, 2.0) == \
        pytest.approx(2 * Tn)


def test_the_built_in_mat_states_its_bars_and_width():
    """E4's 4W11 mat: 4 wires, b = 3 x 0.5 ft = 1.5 ft between the outside
    wires; Tal = 0.48 Fy Ac / b = 4 x 11.02 kN / 0.4572 m = 96.4 kN/m."""
    g = WELDED_WIRE_GRID_W11
    assert g.width == pytest.approx(1.5 * FT)
    assert g.Fy == pytest.approx(65.0 * KSI)
    assert g.t_over_St == pytest.approx(0.374 / 6.0, rel=1e-4)
    per_wire = BAR_MAT_ALLOWABLE_FY * steel_grid_Tal(1, 0.318 * IN,
                                                     65.0 * KSI, 1.0)
    assert per_wire == pytest.approx(11.02, abs=0.01)
    assert g.Tallowable == pytest.approx(4 * per_wire / (1.5 * FT))
    assert g.Tallowable == pytest.approx(96.44, abs=0.01)
    for words in ("4W11 + W11x0.5'", "p. E4-18", "0.48 Fy", "p. 4-77",
                  "Eq. 3-11a", "0.318 in.", "p. E4-14"):
        assert words in g.basis, words


def test_the_allowable_fraction_is_E4s_phi_over_its_load_factor():
    """The module checks Tal x Rc against the unfactored Tmax at FOS 1.0;
    E4 checks phi Tn (0.65, Table E4-5.2) against Tmax times 1.35 (Table
    E4-7.4). 0.65 / 1.35 = 0.481, the 0.48 Fy of GEC-11 Vol I p. 4-77."""
    assert BAR_MAT_ALLOWABLE_FY == 0.48
    assert 0.65 / 1.35 == pytest.approx(BAR_MAT_ALLOWABLE_FY, abs=0.002)


def test_E4s_layout_before_and_after():
    """At Sh 5 ft (Rc 0.30), Tal x Rc = 4 x 11.02 kN / 1.524 m = 28.93 kN/m
    of wall (9.60 before): FOS_rupture 2.854 at the top level, 1.081 at
    8.76 ft beside E4's Level 4 (9.37 ft, where E4's four wires carry 4 x
    3.36 / 12.75 = 1.05 times its factored Tmax, p. E4-17), and 0.571 at
    the bottom, where E4 uses 4W15 mats (p. E4-18); before
    0.947, 0.359 and 0.189. Pullout is unchanged.

    GM-277: the bottom level, at 23.75 ft, takes its contributory height
    from 22.5 ft down to the base at 25.64 ft, 3.14 ft (GEC-11 Vol I p.
    4-47, Fig. 4-14), with sigma_H at its middle, 24.07 ft: Tmax 50.66
    kN/m. At Sv it was 39.84 kN/m (0.726, before 0.241), under the 41.2
    kN/m E4 prints for its own Level 10 over 2.52 ft (19.05 / 1.35 k a 5
    ft panel, Table E4-7.4, p. E4-18), the bottom 0.64 ft in no level."""
    levels = check_internal_stability(_e4_wall(), 125 * PCF, 34.0, _e4_mat())
    ru = [lv["FOS_rupture"] for lv in levels]
    assert {lv["Rc"] for lv in levels} == {0.3}
    assert levels[0]["Tal_Rc_kN_per_m"] == pytest.approx(28.93, abs=0.01)
    assert ru[0] == pytest.approx(2.854, abs=1e-3)
    assert levels[3]["depth_m"] / FT == pytest.approx(8.76, abs=0.01)
    assert ru[3] == pytest.approx(1.081, abs=1e-3)
    assert ru[-1] == pytest.approx(0.571, abs=1e-3)
    assert levels[-1]["Tmax_kN_per_m"] == pytest.approx(50.66, abs=0.01)
    assert levels[-1]["Tmax_kN_per_m"] > 19.05 / 1.35 * KIP / (5 * FT)
    old = dataclasses.replace(_e4_mat(), Tallowable=32.0)
    before = check_internal_stability(_e4_wall(), 125 * PCF, 34.0, old)
    assert [lv["FOS_rupture"] for lv in before][::3] == pytest.approx(
        [0.947, 0.359, 0.283, 0.189], abs=1e-3)
    for a, b in zip(levels, before):
        assert a["FOS_pullout"] == b["FOS_pullout"]


def _tal_choice(r):
    found = [c for c in r.choices
             if c["what"].startswith("Long-term tensile strength Tal")]
    assert len(found) <= 1
    return found[0] if found else None


def test_the_choice_says_where_the_mats_Tal_comes_from():
    """GM-262 review: the mat's Tal is its own choice (it was a sentence in
    the coverage choice's basis), with the facing it assumes and the
    levels E4 uses it at."""
    r = analyze_mse_wall(_e4_wall(), 125 * PCF, 34.0, _e4_mat())
    (c,) = [c for c in r.choices if c["what"].startswith("Coverage ratio")]
    assert c["chosen"].startswith("Rc 0.3 = b 0.4572 m / Sh 1.524 m")
    assert "Tal x Rc 28.93 kN/m of wall" in c["chosen"]
    assert "Example E4" not in c["basis"]
    tal = _tal_choice(r)
    assert tal["how"] == "module"
    assert tal["chosen"].startswith(
        "Bar mat 4W11 + W11x0.5' (GEC-11 Vol II Example E4): Tal 96.44 kN/m "
        "of mat width (44.1 kN a 0.4572 m mat), the same at every level")
    for words in ("p. E4-18", "0.48 Fy", "p. 4-77", "rigid facing",
                  "0.65 / 1.35", "4W15 mats from Level 5", "138.4 kN/m"):
        assert words in tal["basis"], words
    # GM-297: a grid takes its t/St (GEC-11 Vol I Eqs. 3-6, 3-7)
    custom = Reinforcement("a mat", "metallic_grid", 100.0, width=0.5,
                           horizontal_spacing=1.5, thickness=0.01,
                           transverse_spacing=0.15)
    r = analyze_mse_wall(_e4_wall(), 125 * PCF, 34.0, custom)
    assert _tal_choice(r)["how"] == "given"      # it fails rupture deeper
    passing = dataclasses.replace(custom, Tallowable=1000.0)
    assert _tal_choice(analyze_mse_wall(_e4_wall(), 125 * PCF, 34.0,
                                        passing)) is None


def test_the_mats_flexible_facing_reading_is_offered():
    """GEC-11 Vol I p. 3-23 and Table 4-7 note C (p. 4-48): a grid on a
    flexible facing, or continuous with it, takes the strips' 0.75, so the
    strips' 0.55 Fy (p. 4-77): Tal 96.44 x 0.55 / 0.48 = 110.5 kN/m. The
    lowest, 0.571 x 1.146 = 0.654 at the bottom level (0.832 before its
    contributory height ran to the base, GM-277)."""
    tal = _tal_choice(analyze_mse_wall(_e4_wall(), 125 * PCF, 34.0,
                                       _e4_mat()))
    (facing,) = [a for a in tal["alternatives"] if "flexible facing" in a]
    for words in ("Table 4-7 note C", "p. 3-23", "0.55 Fy",
                  "Tal 110.5 kN/m of mat width", "x 1.146",
                  "lowest 0.654"):
        assert words in facing, words


def test_a_rupture_below_one_offers_a_stronger_mat_at_those_levels():
    """At E4's layout the 4W11 mat's FOS_rupture is below 1.0 from 3.43 m
    (11.25 ft) down, 0.571 at the bottom (0.726 before GM-277 ran the
    bottom level's contributory height to the base, 3.14 ft): E4 itself
    changes to 4W15 mats
    from Level 5 (11.87 ft), where four W11 wires carry 0.95 of the
    factored Tmax (Table E4-7.4). At E4's Level 10 (24.37 ft) the module's
    4W11 mat gives 0.709 against E4's 4 x 3.38 / 19.05 = 0.710 for four
    W11 wires, and a 4W15 mat 1.018 against 4 x 4.84 / 19.05 = 1.016."""
    tal = _tal_choice(analyze_mse_wall(_e4_wall(), 125 * PCF, 34.0,
                                       _e4_mat()))
    assert ("FOS_rupture is below 1.0 at 6 levels, z = 3.43 to 7.24 m "
            "(lowest 0.571)") in tal["chosen"]
    (one,) = [a for a in tal["alternatives"]
              if "reinforcement_per_level" in a]
    assert "reinforcement_horizontal_spacing per level" in one
    # GM-266: a stronger mat at those levels, as E4 takes from Level 5
    assert "4W15 mats from Level 5 down (Table E4-7.4, p. E4-18)" in one
    w15 = steel_grid_Tal(4, 0.381 * IN, 65.0 * KSI, 1.5 * FT,
                         BAR_MAT_ALLOWABLE_FY)
    assert w15 == pytest.approx(138.4, abs=0.05)
    deep = MSEWallGeometry(wall_height=25.64 * FT,
                           reinforcement_length=18 * FT,
                           reinforcement_spacing=2.5 * FT,
                           surcharge=0.25 * KSF)
    from retaining_walls.mse import Kr_Ka_ratio, Tmax_at_level, rankine_Ka
    z = 24.37 * FT
    T = Tmax_at_level(z, 125 * PCF, rankine_Ka(34.0),
                      Kr_Ka_ratio(z, "metallic_grid"), 2.5 * FT,
                      deep.surcharge)
    Rc = 1.5 / 5.0
    assert WELDED_WIRE_GRID_W11.Tallowable * Rc / T == pytest.approx(
        4 * 3.38 / 19.05, abs=0.003)
    assert w15 * Rc / T == pytest.approx(4 * 4.84 / 19.05, abs=0.003)


def test_a_mat_that_is_not_one_is_refused():
    for args in ((0, 0.008, 448e3, 0.4), (4, 0.0, 448e3, 0.4),
                 (4, 0.008, 448e3, 0.0)):
        with pytest.raises(ValueError, match="at least one longitudinal"):
            steel_grid_Tal(*args)

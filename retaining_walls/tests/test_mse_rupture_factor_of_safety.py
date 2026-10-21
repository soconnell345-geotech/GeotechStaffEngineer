"""GM-262 review: the factor of safety the tensile rupture check requires.

GEC-11 Vol I (FHWA-NHI-10-024), read on the rendered pages:
  - p. 4-47: Tal is "the nominal long-term reinforcement design strength"
    for steel and geosynthetics alike; Eq. 3-11a (p. 3-22) Fy Ac / b,
    Eq. 3-12 (p. 3-28) Tult / RF, RF = RFID x RFCR x RFD;
  - Eq. 4-33 (p. 4-48): Tr = phi Tal; Table 4-7 (p. 4-48), static: phi
    0.75 strips, 0.65 grids on a rigid facing, 0.90 geosynthetics;
  - p. 4-40: Tmax takes gamma_P-EV = 1.35;
  - p. 4-77: "0.55 Fy for strips and 0.48Fy for bar mats".
GEC-11 Vol II Example E1 (p. E1-15, Table E1-7.3): GG-I geogrid, Tult
3,000 lb/ft, RFCR 1.85, RFD 1.15, RFID 1.3, Tal 1,085, Tr 976 lb/ft; Table
E1-7.4 (p. E1-16): Layer 4, factored Tmax 972 lb/ft, CDR 1.00.

This module checks the unfactored Tmax, so it requires Tal Rc / Tmax >=
1.35 / phi. A steel Tal is given as the 0.55 Fy or 0.48 Fy allowable
(0.75 / 1.35 and 0.65 / 1.35 already in it): 1.0. A geosynthetic's Tal is
Tult / RF: 1.35 / 0.90 = 1.5. Before the review every reinforcement took
1.0, so a geosynthetic passed rupture at two thirds of GEC-11's factored
resistance: the 6 m UX1600 wall passed with 1.192 at the bottom level.
"""

import dataclasses

import pytest

from retaining_walls.calc_steps import get_calc_steps
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.mse import (
    GEOSYNTHETIC_RUPTURE_FOS, analyze_mse_wall, check_internal_stability,
    rupture_FOS_required,
)
from retaining_walls.reinforcement import (
    BAR_MAT_ALLOWABLE_FY, GEOGRID_UX1600, GEOGRID_UX1700,
    RIBBED_STEEL_STRIP_75x4, STRIP_ALLOWABLE_FY, Reinforcement,
    WELDED_WIRE_GRID_W11,
)


def _geogrid_wall():
    """The 6 m geogrid wall of GM-256 (L 5 m, Sv 0.6 m)."""
    return MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0,
                           reinforcement_spacing=0.6)


def _tal_choice(r):
    (c,) = [c for c in r.choices
            if c["what"].startswith("Long-term tensile strength Tal")]
    return c


def test_each_type_takes_GEC11s_load_factor_over_its_phi():
    assert 1.35 / 0.90 == pytest.approx(GEOSYNTHETIC_RUPTURE_FOS)
    assert 0.75 / 1.35 == pytest.approx(STRIP_ALLOWABLE_FY, abs=0.006)
    assert 0.65 / 1.35 == pytest.approx(BAR_MAT_ALLOWABLE_FY, abs=0.002)
    for r in (RIBBED_STEEL_STRIP_75x4, WELDED_WIRE_GRID_W11):
        assert rupture_FOS_required(r) == 1.0
    for kind in ("geogrid", "geotextile", "geosynthetic"):
        assert rupture_FOS_required(Reinforcement("g", kind, 30.0)) == 1.5


def test_E1s_geogrid_check_is_this_tools_at_1_5():
    """E1's GG-I: Tal = 3,000 / (1.85 x 1.15 x 1.3) = 1,085 lb/ft, Tr =
    0.90 Tal = 976. At Layer 4, CDR = 976 / 972 = 1.00; against the
    unfactored Tmax 972 / 1.35 the same check is Tal / Tmax = 1.507, at
    least 1.5."""
    Tal = 3000 / (1.85 * 1.15 * 1.3)
    assert Tal == pytest.approx(1085, abs=1)
    assert 0.90 * Tal == pytest.approx(976, abs=1)
    cdr = 0.90 * Tal / 972
    fos = Tal / (972 / 1.35)
    assert fos == pytest.approx(GEOSYNTHETIC_RUPTURE_FOS * cdr)
    assert cdr >= 1.0 and fos >= GEOSYNTHETIC_RUPTURE_FOS


def test_the_geogrid_wall_now_fails_rupture_at_its_two_lowest_levels():
    """FOS_rupture itself is unchanged (1.192 at the bottom level); the
    check now requires 1.5, so 5.1 m (1.332) and 5.7 m (1.192) fail, where
    every level passed before. It is GEC-11's 0.90 Tal Rc >= 1.35 Tmax."""
    levels = check_internal_stability(_geogrid_wall(), 19.0, 34.0,
                                      GEOGRID_UX1600)
    assert {lv["FOS_rupture_required"] for lv in levels} == {1.5}
    assert [lv["depth_m"] for lv in levels if not lv["passes"]] == [5.1, 5.7]
    assert levels[-1]["FOS_rupture"] == pytest.approx(1.192, abs=1e-3)
    for lv in levels:
        lrfd = 0.90 * lv["Tal_Rc_kN_per_m"] >= 1.35 * lv["Tmax_kN_per_m"]
        assert (lv["FOS_rupture"] >= 1.5) == lrfd
    before = check_internal_stability(_geogrid_wall(), 19.0, 34.0,
                                      GEOGRID_UX1600, FOS_rupture=1.0)
    assert all(lv["passes"] for lv in before)
    assert {lv["FOS_rupture_required"] for lv in before} == {1.0}
    r = analyze_mse_wall(_geogrid_wall(), 19.0, 34.0, GEOGRID_UX1600)
    assert "rupture at z = 5.10, 5.70 m" in r.verdict


def test_the_geogrids_choice_states_the_1_5_and_its_basis():
    r = analyze_mse_wall(_geogrid_wall(), 19.0, 34.0, GEOGRID_UX1600)
    c = _tal_choice(r)
    assert c["how"] == "module"
    assert c["chosen"] == (
        "Geogrid UX1600 (HDPE): Tal 21.9 kN/m of reinforcement width, the "
        "same at every level; FOS_rupture required 1.5 (1.35 / phi 0.90) on "
        "this Tal, the long-term strength Tult / RF; FOS_rupture is below "
        "1.5 at 2 levels, z = 5.1 to 5.7 m (lowest 1.19)")
    for words in ("Eq. 3-12", "1,500 lb/ft", "No source is recorded",
                  "Table 4-7", "p. 4-40", "p. E1-15", "1.35 / 0.90 = 1.5",
                  "0.48 Fy"):
        assert words in c["basis"], words
    (alt,) = c["alternatives"]
    assert "a stronger grade" in alt and "Table E1-7.4" in alt
    assert "reinforcement_spacing" in alt
    assert "29.2 kN/m (about 2,000 lb/ft)" in GEOGRID_UX1700.basis


def test_a_given_geosynthetic_Tal_is_stated_even_when_it_passes():
    strong = Reinforcement("a geogrid", "geogrid", 60.0)
    r = analyze_mse_wall(_geogrid_wall(), 19.0, 34.0, strong)
    assert all(lv["passes"] for lv in r.internal_results)
    c = _tal_choice(r)
    assert c["how"] == "given"
    assert "FOS_rupture required 1.5" in c["chosen"]
    assert "below" not in c["chosen"] and not c.get("alternatives")
    steel = Reinforcement("a strip", "metallic_strip", 2000.0, width=0.05,
                          horizontal_spacing=0.5)
    r = analyze_mse_wall(_geogrid_wall(), 19.0, 34.0, steel)
    assert not [c for c in r.choices if c["what"].startswith("Long-term")]


def test_the_calc_steps_carry_the_required_factor_of_safety():
    def steps(reinf):
        result = analyze_mse_wall(_geogrid_wall(), 19.0, 34.0, reinf)
        analysis = {"wall_type": "mse", "geom": _geogrid_wall(),
                    "gamma_backfill": 19.0, "phi_backfill": 34.0,
                    "reinforcement": reinf}
        return {getattr(it, "title", ""): it
                for sec in get_calc_steps(result, analysis)
                for it in sec.items}
    grid = steps(GEOGRID_UX1600)
    assert grid["Tensile Resistance per Metre of Wall"].equation == \
        "T_al × R_c ≥ 1.5 T_max"
    assert "FOS_rupture required: 1.5 (1.35 / phi 0.90 on T_al = T_ult / " \
        "RF)" in grid["Internal Stability per Reinforcement Level"].notes
    steel = steps(dataclasses.replace(RIBBED_STEEL_STRIP_75x4,
                                      horizontal_spacing=0.75))
    assert steel["Tensile Resistance per Metre of Wall"].equation == \
        "T_al × R_c ≥ T_max"
    assert "FOS_rupture required: 1.0 (T_al the steel allowable" in \
        steel["Internal Stability per Reinforcement Level"].notes

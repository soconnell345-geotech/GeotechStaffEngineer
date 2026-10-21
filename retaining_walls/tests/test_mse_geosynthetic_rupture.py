"""GM-268: a geosynthetic's tensile rupture check at GEC-11's allowable.

GEC-11 Vol I (FHWA-NHI-10-024) checks the factored tensile resistance
Tr = phi Tal (Eq. 4-33, p. 4-48), phi 0.90 for geosynthetics under static
loading (Table 4-7, p. 4-48; p. 3-38), against Tmax factored by
gamma_P-EV = 1.35 (p. 4-40). The module checks an allowable Ta times Rc
against the unfactored Tmax at a factor of safety of 1.0: for steel Ta is
Tal itself, already 0.48 or 0.55 Fy (p. 4-77; GM-262); for a geosynthetic,
whose Tal is the long-term strength Tult / (RFID x RFCR x RFD) of Eq. 3-12
(p. 3-28), it is 0.90 / 1.35 Tal. Before GM-268 a geosynthetic's Tal was
taken itself, 1.5 times the resistance GEC-11 checks. Vol II Example E1's
CDRs are reproduced in validation_examples (V-057).
"""

import dataclasses

import pytest

from retaining_walls.calc_steps import get_calc_steps
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.mse import (
    analyze_mse_wall, check_internal_stability, rupture_allowable,
)
from retaining_walls.reinforcement import (
    GEOGRID_UX1600, GEOGRID_UX1700, GEOSYNTHETIC_ALLOWABLE_FRACTION,
    RIBBED_STEEL_STRIP_75x4, WELDED_WIRE_GRID_W11, Reinforcement,
)


def _wall():
    """The 6 m geosynthetic wall of GM-256."""
    return MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0,
                           reinforcement_spacing=0.6)


def _choice(r, start):
    found = [c for c in r.choices if c["what"].startswith(start)]
    return found[0] if found else None


def test_the_allowable_is_phi_over_the_load_factor():
    assert GEOSYNTHETIC_ALLOWABLE_FRACTION == pytest.approx(0.90 / 1.35)
    assert rupture_allowable(GEOGRID_UX1600) == pytest.approx(14.6)
    geotextile = Reinforcement("geotextile", "geotextile", Tallowable=30.0)
    assert rupture_allowable(geotextile) == pytest.approx(20.0)
    # steel's Tal is its allowable already
    assert rupture_allowable(RIBBED_STEEL_STRIP_75x4) == \
        RIBBED_STEEL_STRIP_75x4.Tallowable
    assert rupture_allowable(WELDED_WIRE_GRID_W11) == \
        WELDED_WIRE_GRID_W11.Tallowable


def test_the_geogrid_wall_before_and_after():
    """UX1600 (Tal 21.9 kN/m): FOS_rupture 22.65 at the top to 1.192 at the
    bottom before, every level passing; 15.10 to 0.795 now, failing at
    5.1 m (0.888) and 5.7 m. UX1700 (29.2): 1.589 before, 1.060 now."""
    levels = check_internal_stability(_wall(), 19.0, 34.0, GEOGRID_UX1600)
    ru = [lv["FOS_rupture"] for lv in levels]
    assert ru[0] == pytest.approx(15.1, abs=1e-3)
    assert ru[-1] == pytest.approx(0.795, abs=1e-3)
    assert [lv["depth_m"] for lv in levels if not lv["passes"]] == [5.1, 5.7]
    for lv in levels:
        assert lv["Tal_Rc_kN_per_m"] == 21.9
        assert lv["Ta_Rc_kN_per_m"] == 14.6
        assert lv["FOS_rupture"] == pytest.approx(
            14.6 / lv["Tmax_kN_per_m"], rel=5e-3)    # Tmax rounded
    ux1700 = check_internal_stability(_wall(), 19.0, 34.0, GEOGRID_UX1700)
    assert min(lv["FOS_rupture"] for lv in ux1700) == pytest.approx(
        1.060, abs=1e-3)


def test_steel_is_unchanged():
    """The 9 m strip wall of GM-256 at Sh 0.75 m: Tal x Rc is the
    resistance, as before (63.85 kN/m, FOS_rupture Tal Rc / Tmax)."""
    geom = MSEWallGeometry(wall_height=9.0, reinforcement_length=7.0,
                           reinforcement_spacing=0.75)
    strip = dataclasses.replace(RIBBED_STEEL_STRIP_75x4,
                                horizontal_spacing=0.75)
    r = analyze_mse_wall(geom, 20.0, 34.0, strip)
    for lv in r.internal_results:
        assert lv["Ta_Rc_kN_per_m"] == lv["Tal_Rc_kN_per_m"] == \
            pytest.approx(63.85, abs=0.01)
        assert lv["FOS_rupture"] == pytest.approx(
            lv["Tal_Rc_kN_per_m"] / lv["Tmax_kN_per_m"], rel=5e-3)
    assert _choice(r, "Allowable tensile strength Ta") is None


def test_the_choice_quotes_the_print_and_gives_the_other_reading():
    r = analyze_mse_wall(_wall(), 19.0, 34.0, GEOGRID_UX1600)
    c = _choice(r, "Allowable tensile strength Ta of the geosynthetic")
    assert c["how"] == "module"
    assert c["chosen"].startswith(
        "Ta = 0.90 / 1.35 Tal = 14.6 kN/m of reinforcement width (Tal "
        "21.9); Ta x Rc against the unfactored Tmax at a factor of safety "
        "of 1.0 (lowest FOS_rupture 0.795)")
    for words in ("Tr = phi Tal (Eq. 4-33, p. 4-48)",
                  "Eq. 3-12, p. 3-28", "Table 4-7, p. 4-48",
                  "'For Strength I limit state conditions, a resistance "
                  "factor (phi) equal to 0.90 is used for geosynthetic "
                  "reinforcements'", "gamma_P-EV = 1.35",
                  "Table E1-7.4, p. E1-16", "'A FS = 1.5 will be applied "
                  "to the geogrid reinforcement in stability analysis'",
                  "would be reduced twice"):
        assert words in c["basis"], words
    (alt,) = c["alternatives"]
    assert alt.startswith("Tal itself at a factor of safety of 1.0, no "
                          "resistance or load factor: every FOS_rupture x "
                          "1.5 (lowest 1.19)")
    # Rc and the resistance per metre of wall say Ta, not Tal
    cov = _choice(r, "Coverage ratio Rc")
    assert "tensile resistance Ta x Rc 14.6 kN/m of wall" in cov["chosen"]


def test_the_built_in_geogrids_say_their_tal_has_no_source():
    for grid, lbft in ((GEOGRID_UX1600, "1,500"), (GEOGRID_UX1700, "2,000")):
        assert grid.basis.startswith(
            f"No source is recorded for this Tal: {grid.Tallowable:g} kN/m "
            f"(about {lbft} lb/ft)")
        assert "Eq. 3-12 (p. 3-28)" in grid.basis
        assert "(p. 3-30)" in grid.basis
    r = analyze_mse_wall(_wall(), 19.0, 34.0, GEOGRID_UX1600)
    c = _choice(r, "Long-term tensile strength Tal")
    assert c["how"] == "module"
    assert c["basis"] == GEOGRID_UX1600.basis
    assert c["alternatives"][0].startswith("the product's own long-term "
                                           "strength Tal")
    assert c["alternatives"][1].startswith("a stronger grade, or a closer "
                                           "vertical spacing")
    assert "FOS_rupture is below 1.0 at 2 levels, z = 5.1 to 5.7 m" in \
        c["chosen"]


def test_a_custom_geosynthetic_failing_cites_eq_3_12():
    grid = Reinforcement("my grid", "geogrid", Tallowable=15.0)
    r = analyze_mse_wall(_wall(), 19.0, 34.0, grid)
    c = _choice(r, "Long-term tensile strength Tal")
    assert c["how"] == "given"
    assert c["basis"].startswith("GEC-11 Vol I Eq. 3-12 (p. 3-28): Tal = "
                                 "Tult / (RFID x RFCR x RFD)")
    assert "Eq. 3-11a" not in c["basis"]
    # a given geosynthetic that passes still gets the allowable choice
    strong = analyze_mse_wall(_wall(), 19.0, 34.0, dataclasses.replace(
        grid, Tallowable=60.0))
    assert _choice(strong, "Long-term tensile strength Tal") is None
    assert _choice(strong, "Allowable tensile strength Ta")["how"] == \
        "module"


def test_the_calc_steps_show_the_allowable():
    r = analyze_mse_wall(_wall(), 19.0, 34.0, GEOGRID_UX1600)
    analysis = {"wall_type": "mse", "geom": _wall(), "gamma_backfill": 19.0,
                "phi_backfill": 34.0, "reinforcement": GEOGRID_UX1600}
    items = [it for sec in get_calc_steps(r, analysis) for it in sec.items]
    by_title = {getattr(it, "title", ""): it for it in items}
    step = by_title["Tensile Resistance per Metre of Wall (geosynthetic)"]
    assert step.equation.startswith("T_a × R_c ≥ T_max; T_a = φ T_al / "
                                    "γ_EV-MAX = 0.90 T_al / 1.35")
    assert step.substitution.startswith("T_a = 0.90 × 21.9 / 1.35 = 14.6")
    assert step.result_value == "14.6"
    assert "Table 4-7 (p. 4-48)" in step.reference
    table = by_title["Internal Stability per Reinforcement Level"]
    assert "T_a R_c (kN/m)" in table.headers
    assert table.rows[-1][3] == "14.6"
    assert table.rows[-1][-1] == "FAIL"
    assert "T_a = 0.90 / 1.35 T_al" in table.notes

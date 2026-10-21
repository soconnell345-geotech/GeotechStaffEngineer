"""GM-246: the ribbed steel strip pullout factor F* at the top of the wall.

GEC-11 Vol I (FHWA-NHI-10-024) p. 3-20: Eq. 3-4, F* = 1.2 + log Cu at the
top of the structure, 2.0 maximum; Eq. 3-5, tan phi at 20 ft (6 m) and
below; "If the specific Cu for the wall backfill is unknown at design time,
a Cu = 4 should be assumed (i.e., F* = 1.8 at the top of the wall)". Before
GM-246 every strip took 2.0 at the top, the value only a backfill with Cu
6.3 or more reaches.
"""

import dataclasses
import math

import pytest

from retaining_walls.calc_steps import get_calc_steps
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.mse import (
    CU_UNKNOWN, F_star_metallic, F_star_strip_top, analyze_mse_wall,
    check_internal_stability, pullout_factor_choices, uses_strip_F_star,
)
from retaining_walls.reinforcement import (
    GEOGRID_UX1600, RIBBED_STEEL_STRIP_75x4, Reinforcement,
    WELDED_WIRE_GRID_W11,
)

TAN34 = math.tan(math.radians(34.0))

# GM-256: a steel strip or grid needs its horizontal spacing. The 75 mm
# strips at Sh 0.75 m (Rc 0.1); the W11 grid as GEC-11 Example E4's 4-wire
# mat, b 0.457 m, one per 1.524 m panel (Rc 0.3)
_STRIP = dataclasses.replace(RIBBED_STEEL_STRIP_75x4, horizontal_spacing=0.75)
_GRID = dataclasses.replace(WELDED_WIRE_GRID_W11, width=0.4572,
                            horizontal_spacing=1.524)


def _geom():
    return MSEWallGeometry(wall_height=9.0, reinforcement_length=7.0,
                           reinforcement_spacing=0.75)


def _choice(result):
    (c,) = [c for c in result.choices if c["what"].startswith(
        "Pullout resistance factor F*")]
    return c


@pytest.mark.parametrize("Cu, top", [
    (None, 1.802), (4.0, 1.802), (1.0, 1.2), (2.0, 1.501), (6.0, 1.978),
    (10 ** 0.8, 2.0), (7.0, 2.0), (50.0, 2.0)])
def test_the_top_value_is_1_2_plus_log_Cu_at_most_2(Cu, top):
    assert F_star_strip_top(Cu) == pytest.approx(top, abs=1e-3)
    assert F_star_metallic(0.0, 34.0, Cu=Cu) == pytest.approx(top, abs=1e-3)
    # tan phi at 20 ft (6.096 m) and below, whatever the Cu
    for z in (6.096, 8.0):
        assert F_star_metallic(z, 34.0, Cu=Cu) == pytest.approx(TAN34)


@pytest.mark.parametrize("Cu", [0.0, 0.99, -2.0, float("nan")])
def test_a_Cu_below_1_is_refused(Cu):
    with pytest.raises(ValueError, match="D60/D10, 1 or more"):
        F_star_strip_top(Cu)


def test_before_and_after_at_0_3_and_6_m():
    """The numbers the GM-246 notes give, phi 34: F* 2.000 / 1.348 / 0.695
    before, 1.802 / 1.247 / 0.692 now (Cu 4), 2.000 / 1.348 / 0.695 at Cu
    7.0."""
    now = [F_star_metallic(z, 34.0) for z in (0.0, 3.0, 6.0)]
    assert now == pytest.approx([1.802, 1.247, 0.692], abs=1e-3)
    cu7 = [F_star_metallic(z, 34.0, Cu=7.0) for z in (0.0, 3.0, 6.0)]
    assert cu7 == pytest.approx([2.0, 1.348, 0.695], abs=1e-3)


def test_which_reinforcement_takes_the_strip_curve():
    assert uses_strip_F_star(RIBBED_STEEL_STRIP_75x4)
    assert not uses_strip_F_star(WELDED_WIRE_GRID_W11)
    assert not uses_strip_F_star(GEOGRID_UX1600)
    bare_grid = Reinforcement(name="bare grid", type="metallic_grid",
                              Tallowable=100.0)
    assert uses_strip_F_star(bare_grid)


@pytest.mark.parametrize("reinforcement, words", [
    (_GRID, "20 t/St at the top to 10 t/St"),
    (GEOGRID_UX1600, "0.67 tan phi_backfill")])
def test_where_the_F_star_takes_no_Cu_a_given_Cu_is_checked_by_note_i(
        reinforcement, words):
    """GEC-11 Table 3-6 note (i) (p. 3-17) accepts every empirical F* in
    the table, strips, grids and sheets alike, only for Cu >= 4: a Cu
    given with a grid or a geosynthetic leaves its F* as it was and is
    checked against the note, not refused (GM-246 review)."""
    left_out = analyze_mse_wall(_geom(), 20.0, 34.0, reinforcement)
    # left out, nothing is said about it
    assert not [c for c in left_out.choices if "Cu_backfill" in c["what"]]
    for Cu, within in ((5.0, True), (2.5, False)):
        r = analyze_mse_wall(_geom(), 20.0, 34.0, reinforcement,
                             Cu_backfill=Cu)
        assert [lv["F_star"] for lv in r.internal_results] == \
            [lv["F_star"] for lv in left_out.internal_results]
        (c,) = [c for c in r.choices if "(Cu_backfill)" in c["what"]]
        assert c["how"] == "given" and words in c["chosen"]
        assert c["chosen"].startswith(f"Cu {Cu:g}: ")
        assert ("at least 4, within" in c["chosen"]) is within
        assert ("tests must be run" in c["chosen"]) is not within
        assert "Table 3-1 for MSE walls is used and Cu >= 4" in c["basis"]
        assert "Appendix B" in c["alternatives"][0]
    with pytest.raises(ValueError, match="D60/D10, 1 or more"):
        check_internal_stability(_geom(), 20.0, 34.0, reinforcement,
                                 Cu_backfill=0.5)


def test_a_Cu_of_4_given_reads_as_given():
    """The library call says 'given' for a Cu the call gave, though it
    equals the value taken when it is left out."""
    c = _choice(analyze_mse_wall(_geom(), 20.0, 34.0,
                                 _STRIP, Cu_backfill=4.0))
    assert c["how"] == "given"
    assert _choice(analyze_mse_wall(
        _geom(), 20.0, 34.0, _STRIP))["how"] == "default"


def test_the_choice_states_the_printed_default_and_the_other_reading():
    r = analyze_mse_wall(_geom(), 20.0, 34.0, _STRIP)
    c = _choice(r)
    assert c["how"] == "default"
    assert c["chosen"].startswith("F* 1.80 at the top of the wall (Cu 4)")
    assert "0.675 at 20 ft (6.1 m)" in c["chosen"]
    assert "a Cu = 4 should be assumed (i.e., F* = 1.8" in c["basis"]
    assert "p. 3-20" in c["basis"] and "2.0 maximum" in c["basis"]
    (alt,) = c["alternatives"]
    assert alt.startswith("a Cu of 6.3 or more, measured on the wall's "
                          "backfill: F* 2.0 at the top")
    # the top level's FOS_pullout at Cu 7 (2.0) against Cu 4 (1.8)
    top = r.internal_results[0]
    at_cap = check_internal_stability(_geom(), 20.0, 34.0,
                                      _STRIP,
                                      Cu_backfill=7.0)[0]
    assert top["F_star"] < at_cap["F_star"]
    assert (f"FOS_pullout {at_cap['FOS_pullout']:.3g} at the top level, "
            f"{top['depth_m']:g} m (here {top['FOS_pullout']:.3g})") in alt

    given = _choice(analyze_mse_wall(_geom(), 20.0, 34.0,
                                     _STRIP,
                                     Cu_backfill=7.0))
    assert given["how"] == "given"
    assert given["chosen"].startswith(
        "F* 2.00 at the top of the wall (Cu 7: 1.2 + log Cu is above the "
        "2.0 maximum)")
    assert given["alternatives"][0].startswith(
        "Cu 4, which GEC-11 Vol I p. 3-20 says to assume")
    assert "note (i)" not in given["chosen"]
    assert "Table 3-1 for MSE walls is used and Cu >= 4" in given["basis"]


def test_a_Cu_below_4_is_computed_and_flagged():
    """GEC-11 Table 3-6 note (i) (p. 3-17): the empirical F* holds for
    Table 3-1 fill with Cu >= 4; 'For fill outside these limits, tests
    must be run'."""
    c = _choice(analyze_mse_wall(_geom(), 20.0, 34.0,
                                 _STRIP, Cu_backfill=2.5))
    assert c["chosen"].startswith("F* 1.60 at the top of the wall (Cu 2.5)")
    assert "tests must be run" in c["chosen"]


def test_a_grid_with_no_bar_geometry_says_it_took_the_strip_curve():
    bare_grid = Reinforcement(name="bare grid", type="metallic_grid",
                              Tallowable=100.0, coverage_ratio=0.3)
    (c,) = pullout_factor_choices(_geom(), 20.0, 34.0, bare_grid)
    assert "no transverse-bar geometry (t/St)" in c["basis"]
    assert check_internal_stability(_geom(), 20.0, 34.0, bare_grid,
                                    Cu_backfill=7.0)[0]["F_star"] > \
        check_internal_stability(_geom(), 20.0, 34.0, bare_grid)[0]["F_star"]


def test_the_calc_steps_show_the_Cu_and_the_F_star_curve():
    def step(cu):
        r = analyze_mse_wall(_geom(), 20.0, 34.0, _STRIP,
                             Cu_backfill=cu)
        analysis = {"wall_type": "mse", "geom": _geom(),
                    "gamma_backfill": 20.0, "phi_backfill": 34.0,
                    "reinforcement": _STRIP,
                    "Cu_backfill": cu}
        found = [it for sec in get_calc_steps(r, analysis)
                 for it in sec.items
                 if str(getattr(it, "title", "")).startswith(
                     "Pullout Resistance Factor F*")]
        assert len(found) == 1
        return found[0]

    s = step(None)
    assert ("C_u = 4 (not given: GEC-11 p. 3-20 says to assume C_u = 4"
            in s.substitution)
    assert "note (i)" not in s.substitution
    assert "tests must be run" in step(2.5).substitution
    assert s.result_value.startswith("1.80") and "0.675" in s.result_value
    assert s.reference == "GEC-11 Vol I Eqs. 3-4, 3-5 (p. 3-20)"
    assert step(7.0).substitution.startswith("C_u = 7: F* = 2.00 at z = 0")
    assert CU_UNKNOWN == 4.0

"""GM-256: mse_wall takes the reinforcement's horizontal spacing (or its
coverage ratio) and F* and alpha from pullout tests; a steel strip or grid
given neither spacing nor ratio is refused. GEC-11 Vol I Eq. 3-1 (p. 3-12),
p. 3-13, Eq. 4-35 (p. 4-49), p. 4-47; the module tests are in
retaining_walls/tests/test_mse_coverage_ratio.py."""

import pytest

from funhouse_agent import dispatch as D

_WALL = dict(wall_height=9.0, gamma_backfill=20.0, phi_backfill=34.0,
             reinforcement_length=7.0, reinforcement_spacing=0.75)


def _mse(**extra):
    return D.call_agent("retaining_walls", "mse_wall", {**_WALL, **extra})


def _coverage(r):
    (c,) = [c for c in r["choices"] if c["what"].startswith("Coverage")]
    return c


@pytest.mark.parametrize("name", ["reinforcement_horizontal_spacing", "Sh",
                                  "horizontal_spacing"])
def test_a_strip_takes_its_spacing_by_any_name(name):
    r = _mse(reinforcement_name="ribbed_steel_strip_75x4", **{name: 0.75})
    assert "error" not in r, r.get("error")
    assert {lv["Rc"] for lv in r["internal_results"]} == {0.1}
    assert min(lv["FOS_pullout"] for lv in r["internal_results"]) == \
        pytest.approx(2.876, abs=1e-3)
    assert _coverage(r)["how"] == "given"
    assert "coverage_ratio" not in r["defaults_used"]


def test_a_strip_with_no_spacing_is_refused_naming_the_inputs():
    err = _mse(reinforcement_name="ribbed_steel_strip_75x4")["error"]
    for words in ("coverage_ratio", "reinforcement_horizontal_spacing",
                  "Sh alone will do", "the tool assumes none of them"):
        assert words in err, words


def test_the_coverage_ratio_given_directly():
    r = _mse(reinforcement_name="ribbed_steel_strip_75x4", Rc=0.12)
    assert {lv["Rc"] for lv in r["internal_results"]} == {0.12}
    assert _coverage(r)["chosen"].startswith("Rc 0.12, given")
    both = _mse(reinforcement_name="ribbed_steel_strip_75x4", Rc=0.12,
                Sh=0.75)
    assert "not both" in both["error"]
    # GM-256 review: a width serves only Rc = b/Sh, so beside a given Rc it
    # would be taken and not used; a geogrid's width needs its spacing
    unused = _mse(reinforcement_type="metallic_strip",
                  reinforcement_Tallowable=500.0, Rc=0.1,
                  reinforcement_width=0.05)
    assert "the width is used only for Rc" in unused["error"]
    no_sh = _mse(reinforcement_type="geogrid", reinforcement_Tallowable=30.0,
                 reinforcement_width=0.5)
    assert "needs its horizontal spacing Sh" in no_sh["error"]


def test_a_geogrid_is_a_continuous_layer_and_says_so():
    r = _mse(reinforcement_name="geogrid_ux1600")
    assert {lv["Rc"] for lv in r["internal_results"]} == {1.0}
    assert r["defaults_used"]["coverage_ratio"].startswith(
        "Rc 1, a continuous geosynthetic layer: GEC-11 Vol I p. 3-13")
    assert _coverage(r)["how"] == "default"


def test_the_built_in_grid_and_strip_keep_their_own_width():
    """GM-262: the built-in mat is GEC-11 Vol II Example E4's 4W11 mat, b
    1.5 ft (0.4572 m), and carries it as the strip carries its 75 mm."""
    grid = _mse(reinforcement_name="welded_wire_grid_w11", Sh=1.524)
    assert {lv["Rc"] for lv in grid["internal_results"]} == {0.3}
    same = _mse(reinforcement_name="welded_wire_grid_w11",
                reinforcement_width=0.4572, Sh=1.524)
    assert same["internal_results"] == grid["internal_results"]
    other = _mse(reinforcement_name="welded_wire_grid_w11",
                 reinforcement_width=0.3, Sh=1.524)
    assert "is 0.4572 m wide" in other["error"]
    other = _mse(reinforcement_name="ribbed_steel_strip_75x4",
                 reinforcement_width=0.05, Sh=0.75)
    assert "is 0.075 m wide" in other["error"]


def test_a_custom_strip_takes_its_width_and_spacing():
    r = _mse(reinforcement_type="strip", reinforcement_Tallowable=600.0,
             reinforcement_width=0.05, Sh=0.5)
    lv = r["internal_results"][0]
    assert lv["Rc"] == pytest.approx(0.1)
    assert lv["Tal_Rc_kN_per_m"] == pytest.approx(60.0)


def test_tested_F_star_and_alpha_through_the_tool():
    base = _mse(reinforcement_name="ribbed_steel_strip_75x4", Sh=0.75)
    assert base["defaults_used"]["F_star_tested"].startswith(
        "the empirical F*: GEC-11 Vol I Eqs. 3-4 to 3-8")
    r = _mse(reinforcement_name="ribbed_steel_strip_75x4", Sh=0.75,
             F_star_tested=[[0, 1.5], [6, 0.8]], alpha_tested=0.9)
    assert "F_star_tested" not in r["defaults_used"]
    assert r["internal_results"][0]["F_star"] == pytest.approx(
        1.5 - 0.7 * 0.375 / 6, abs=1e-3)
    assert {lv["alpha"] for lv in r["internal_results"]} == {0.9}
    whats = [c["what"] for c in r["choices"]]
    assert any("(F_star_tested)" in w for w in whats)
    assert any("(alpha_tested)" in w for w in whats)
    assert not any("(Cu_backfill)" in w for w in whats)


def test_list_reinforcement_says_what_each_needs():
    out = D.call_agent("retaining_walls", "list_reinforcement", {})
    strip = out["reinforcement_types"]["ribbed_steel_strip_75x4"]
    assert strip["width_m"] == 0.075
    assert strip["Tallowable_kN_per_m"] == pytest.approx(574.67, abs=0.01)
    assert strip["mse_wall_needs"].startswith(
        "reinforcement_horizontal_spacing")
    assert "Eq. 3-11a" in out["Tallowable"]
    # GM-262: the mat carries its b and says where its Tal comes from
    grid = out["reinforcement_types"]["welded_wire_grid_w11"]
    assert grid["width_m"] == 0.4572
    assert grid["Tallowable_kN_per_m"] == pytest.approx(96.44, abs=0.01)
    assert grid["mse_wall_needs"] == strip["mse_wall_needs"]
    assert "p. E4-18" in grid["Tallowable_basis"]
    assert "allowable" in out["Tallowable"]


@pytest.mark.parametrize("name", ["reinforcement_horizontal_spacing",
                                  "coverage_ratio", "F_star_tested",
                                  "alpha_tested"])
def test_the_lrfd_external_tool_refuses_them_with_the_reason(name):
    err = D.call_agent("retaining_walls", "mse_lrfd_external_stability",
                       {"wall_height": 6, "gamma_backfill": 19,
                        "phi_backfill": 34, name: 0.5})["error"]
    assert "mse_wall's pullout and rupture checks use it" in err


def test_the_tool_text_says_per_unit_width_and_names_the_equations():
    from funhouse_agent.adapters.retaining_walls import METHOD_INFO
    p = METHOD_INFO["mse_wall"]["parameters"]
    assert "per metre of the reinforcement's own width" in p[
        "reinforcement_Tallowable"]["description"]
    assert "Eq. 3-11a" in p["reinforcement_Tallowable"]["description"]
    # GM-262: the check is at FOS 1.0 on the unfactored Tmax: an allowable
    for words in ("unfactored Tmax at a factor of safety of 1.0",
                  "0.48 Fy for bar mats (p. 4-77)", "Fig. 3-5, p. 3-24"):
        assert words in p["reinforcement_Tallowable"]["description"], words
    sh = p["reinforcement_horizontal_spacing"]["description"]
    assert "Rc = b/Sh (Eq. 3-1)" in sh and "refused" in sh
    assert "default" not in p["reinforcement_width"]
    assert "Table 3-6 note (i)" in p["F_star_tested"]["description"]
    assert "p. 3-16" in p["alpha_tested"]["description"]

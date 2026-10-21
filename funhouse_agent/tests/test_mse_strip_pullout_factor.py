"""GM-246: mse_wall takes the backfill's uniformity coefficient for the
ribbed steel strip F* (GEC-11 Vol I Eq. 3-4, p. 3-20), states the Cu = 4
it takes when it is left out, and checks a Cu given with another
reinforcement against GEC-11 Table 3-6 note (i)."""

import pytest

from funhouse_agent import dispatch as D

_WALL = dict(wall_height=9.0, gamma_backfill=20.0, phi_backfill=34.0,
             reinforcement_length=7.0, reinforcement_spacing=0.75)
#: GM-256: each built-in's coverage, Rc = b/Sh (a geogrid is continuous)
_COVER = {"ribbed_steel_strip_75x4": {"reinforcement_horizontal_spacing": 0.75},
          "welded_wire_grid_w11": {"reinforcement_width": 0.4572,
                                   "reinforcement_horizontal_spacing": 1.524},
          "geogrid_ux1600": {}}
_STRIP = dict(_WALL, reinforcement_name="ribbed_steel_strip_75x4",
              **_COVER["ribbed_steel_strip_75x4"])


def _mse(**extra):
    name = extra.get("reinforcement_name", "ribbed_steel_strip_75x4")
    return D.call_agent("retaining_walls", "mse_wall",
                        {**_WALL, "reinforcement_name": name,
                         **_COVER.get(name, {}), **extra})


def _choice(r):
    (c,) = [c for c in r["choices"] if "(Cu_backfill)" in c["what"]]
    return c


def test_left_out_it_is_the_printed_default_and_listed():
    r = _mse()
    assert r["defaults_used"]["Cu_backfill"] == 4.0
    assert r["internal_results"][0]["F_star"] == pytest.approx(1.733,
                                                               abs=0.001)
    assert _choice(r)["how"] == "default"


@pytest.mark.parametrize("name", ["Cu_backfill", "Cu",
                                  "uniformity_coefficient"])
def test_given_by_any_name_it_sets_the_top_value(name):
    r = _mse(**{name: 7.0})
    assert "Cu_backfill" not in r["defaults_used"]
    assert _choice(r)["how"] == "given"
    # GEC-11 Vol II Table E3-7.3 interpolation, 2.0 at Z = 0
    assert r["internal_results"][0]["F_star"] == pytest.approx(1.918,
                                                               abs=0.001)


def test_a_custom_strip_takes_it_and_a_geogrid_does_not_list_it():
    custom = _mse(reinforcement_name="my strip", reinforcement_type="strip",
                  reinforcement_Tallowable=40.0, reinforcement_width=0.05,
                  reinforcement_horizontal_spacing=0.5)
    assert custom["defaults_used"]["Cu_backfill"] == 4.0
    grid = _mse(reinforcement_name="geogrid_ux1600")
    assert "Cu_backfill" not in grid["defaults_used"]
    assert not [c for c in grid.get("choices", [])
                if "(Cu_backfill)" in c["what"]]


@pytest.mark.parametrize("name, cu", [
    ("welded_wire_grid_w11", 5.0), ("geogrid_ux1600", 2.5)])
def test_a_grid_or_geogrid_takes_it_as_a_note_i_check(name, cu):
    """GEC-11 Table 3-6 note (i) (p. 3-17) holds every empirical F* to Cu
    >= 4, so a Cu given with a grid or a geogrid is checked, not refused
    (GM-246 review); their F* takes no Cu and is unchanged."""
    r = _mse(reinforcement_name=name, Cu=cu)
    assert "error" not in r, r.get("error")
    c = _choice(r)
    assert c["how"] == "given" and "note (i)" in c["chosen"]
    assert ("tests must be run" in c["chosen"]) is (cu < 4)
    assert [lv["F_star"] for lv in r["internal_results"]] == [
        lv["F_star"] for lv in _mse(reinforcement_name=name)[
            "internal_results"]]


@pytest.mark.parametrize("method, params, words", [
    ("mse_wall", {**_STRIP, "Cu_backfill": 0.5}, "1 or more"),
    ("mse_wall", {**_WALL, "reinforcement_name": "geogrid_ux1600",
                  "Cu_backfill": 0.5}, "1 or more"),
    ("mse_lrfd_external_stability",
     dict(wall_height=9.0, gamma_backfill=20.0, phi_backfill=34.0,
          reinforcement_length=7.0, Cu_backfill=5.0),
     "mse_wall's pullout check uses it"),
])
def test_refused_where_it_changes_nothing(method, params, words):
    err = D.call_agent("retaining_walls", method, params).get("error", "")
    assert words in err, err


def test_the_tool_text_names_the_equation_and_the_default():
    from funhouse_agent.adapters.retaining_walls import METHOD_INFO
    info = METHOD_INFO["mse_wall"]
    d = info["parameters"]["Cu_backfill"]["description"]
    assert "F* = 1.2 + log Cu" in d and "p. 3-20" in d
    assert "a Cu = 4 should be assumed" in d
    assert "F* 1.2 + log Cu (2.0 at most; Cu_backfill)" in info[
        "parameters"]["reinforcement_type"]["description"]
    assert "Cu_backfill" in info["returns"]["choices"]

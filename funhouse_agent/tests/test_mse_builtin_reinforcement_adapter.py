"""GM-267 and GM-271: the mse_wall tool's built-in reinforcements are
reinforcements GEC-11 Vol II's worked examples print, with their Tal as
printed; the names they replaced are refused, saying what to give instead.

GM-267: the built-in strip is Example E3's ribbed strip, "1.969 in. x 0.157
in. (50 mm wide x 4 mm)", Ec = 0.102 in., Fy 65 ksi (p. E3-16): 0.55 Fy Ec
= 638.6 kN/m of strip width, 31.9 kN a strip. It was 75 mm wide, the tool's
own width.

GM-271: the built-in geogrids are Example E1's GG-I to GG-III as Table
E1-7.3 (p. E1-15) prints them, Tal = Tult / RF (Vol I Eq. 3-12, p. 3-28).
The two they replaced carried one maker's product-line names and a Tal of
no recorded source.
"""

import pytest

from funhouse_agent import dispatch as D
from funhouse_agent.adapters.retaining_walls import METHOD_INFO

_LB_FT = 4.4482216 / 0.3048 / 1000.0          # lb/ft to kN/m

_WALL = {"wall_height": 6.0, "gamma_backfill": 19.0, "phi_backfill": 34.0,
         "reinforcement_length": 5.0, "reinforcement_spacing": 0.6}


def _mse(**extra):
    return D.call_agent("retaining_walls", "mse_wall", {**_WALL, **extra})


@pytest.mark.parametrize("key, tult, rf_id, tal_lb_ft", [
    ("geogrid_gec11_e1_gg1", 3000, 1.3, 1085),
    ("geogrid_gec11_e1_gg2", 6000, 1.3, 2169),
    ("geogrid_gec11_e1_gg3", 9000, 1.2, 3525)])
def test_E1s_geogrids_as_table_E1_7_3_prints_them(key, tult, rf_id,
                                                  tal_lb_ft):
    """Table E1-7.3: RFCR 1.85, RFD 1.15 for every grade; Tal 1,085, 2,169
    and 3,525 lb/ft."""
    out = D.call_agent("retaining_walls", "list_reinforcement", {})
    grid = out["reinforcement_types"][key]
    assert grid["type"] == "geogrid" and grid["alpha"] == 0.8
    tal = tult / (rf_id * 1.85 * 1.15)
    assert tal == pytest.approx(tal_lb_ft, abs=1)
    assert grid["Tallowable_kN_per_m"] == pytest.approx(tal * _LB_FT,
                                                        abs=0.01)
    for words in ("Table E1-7.3 (p. E1-15)", f"Tult {tult:,} lb/ft",
                  f"RFID {rf_id}", "Eq. 3-12, p. 3-28", "Step 7.8, p. E1-16",
                  "Table 3-6, p. 3-17", "Table E1-7.1 (p. E1-11)"):
        assert words in grid["Tallowable_basis"], words


def test_the_strip_is_E3s_and_its_strength_per_strip_is_Tal_times_b():
    out = D.call_agent("retaining_walls", "list_reinforcement", {})
    strip = out["reinforcement_types"]["ribbed_steel_strip_50x4"]
    assert strip["width_m"] == 0.05
    assert strip["Tallowable_kN_per_m"] == pytest.approx(638.6, abs=0.01)
    for words in ("50 mm wide x 4 mm", "p. E3-16", "0.102 in.", "p. 2-16"):
        assert words in strip["Tallowable_basis"], words
    r = _mse(reinforcement_name="ribbed_steel_strip_50x4",
             reinforcement_horizontal_spacing=0.75)
    assert "error" not in r, r.get("error")
    lv = r["internal_results"][0]
    assert lv["Tal_Rc_kN_per_m"] == pytest.approx(638.6 * 0.05 / 0.75,
                                                  abs=0.01)


@pytest.mark.parametrize("old, words", [
    ("ribbed_steel_strip_75x4",
     ["is no longer offered", "75 mm width was the tool's own",
      "'ribbed_steel_strip_50x4'", "638.6 kN/m of strip width",
      "31.9 kN a strip", "reinforcement_horizontal_spacing",
      "'metallic_strip'", "reinforcement_width"]),
    ("geogrid_ux1600",
     ["is no longer offered", "21.9 kN/m, had no recorded source",
      "'geogrid_gec11_e1_gg1' (Tal 15.83 kN/m)",
      "'geogrid_gec11_e1_gg2' (Tal 31.66 kN/m)",
      "or 'geogrid_gec11_e1_gg3' (Tal 51.45 kN/m)",
      "reinforcement_type 'geogrid' with reinforcement_Tallowable",
      "Tult / RF"]),
    ("GEOGRID_UX1700",
     ["is no longer offered", "29.2 kN/m, had no recorded source",
      "Table E1-7.3 (p. E1-15)"]),
])
def test_the_retired_names_are_refused_saying_what_to_give(old, words):
    r = _mse(reinforcement_name=old, reinforcement_horizontal_spacing=0.75)
    assert "internal_results" not in r
    for w in words:
        assert w in r["error"], w


def test_the_tool_text_names_no_product():
    """The tool's own text names the GEC-11 examples, not a maker's
    product line; the retired names appear only in their refusals."""
    text = repr(METHOD_INFO).lower()
    for word in ("ux1600", "ux1700", "tensar", "hdpe", "75x4", "uniaxial"):
        assert word not in text, word
    out = D.call_agent("retaining_walls", "list_reinforcement", {})
    assert sorted(out["reinforcement_types"]) == [
        "geogrid_gec11_e1_gg1", "geogrid_gec11_e1_gg2",
        "geogrid_gec11_e1_gg3", "ribbed_steel_strip_50x4",
        "welded_wire_grid_w11"]
    listed = repr(out).lower()
    for word in ("ux1600", "ux1700", "tensar", "hdpe", "75x4"):
        assert word not in listed, word
    desc = METHOD_INFO["mse_wall"]["parameters"]["reinforcement_name"][
        "description"]
    for words in ("Example E3's 50 x 4 mm ribbed strip",
                  "Example E4's 4W11 bar mat",
                  "Example E1's geogrids GG-I to GG-III"):
        assert words in desc, words

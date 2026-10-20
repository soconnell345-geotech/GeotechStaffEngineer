"""From the ``liquefaction`` tool to ``downdrag.post_liquefaction_downdrag``
(BACKLOG GM-201): the CPT route returns FS and post-liquefaction volumetric
strain by depth, inside the tool-result size cap, and the downdrag tool
takes that result as it is; the SPT routes still need the strain given."""

import json

import numpy as np
import pytest

from funhouse_agent import dispatch as D
from funhouse_agent.adapters import MODULE_REGISTRY
from liquepy_agent import has_liquepy

needs_liquepy = pytest.mark.skipif(not has_liquepy(),
                                   reason="liquepy not installed")

#: The tool-result cap the agents work to (funhouse_agent/deep/tools.py).
RESULT_CAP = 8000

_LAYERS = [
    {"thickness": 6, "soil_type": "cohesionless", "unit_weight": 19,
     "phi": 30, "beta": 0.3},
    {"thickness": 4, "soil_type": "cohesionless", "unit_weight": 19,
     "phi": 30, "beta": 0.3},
    {"thickness": 20, "soil_type": "cohesionless", "unit_weight": 20,
     "phi": 36, "beta": 0.45},
]
_STATIC = {"layers": _LAYERS, "pile_length": 20.0, "pile_diameter": 0.5,
           "pile_perimeter": 2.0, "pile_area": 0.25, "Q_dead": 600.0,
           "Nt": 30.0}


def _cpt(n, dz, loose=(6.0, 10.0)):
    """A CPT read every ``dz`` m: loose sand in ``loose``, dense around."""
    z = np.round(np.arange(1, n + 1) * dz, 4)
    q_c = np.where((z > loose[0]) & (z < loose[1]), 2500.0, 25000.0)
    return D.call_agent("liquefaction", "liquefaction_analysis", {
        "depth": z.tolist(), "q_c": q_c.tolist(),
        "f_s": (0.005 * q_c).tolist(), "gwt_depth": 1.0, "amax_g": 0.3})


@needs_liquepy
class TestTheCptRoute:

    def test_a_short_cpt_gives_a_row_per_reading(self):
        r = _cpt(20, 0.5)
        assert "error" not in r, r
        rows = r["depth_profile"]
        assert len(rows) == 20 and "depth_profile_note" not in r
        assert set(rows[0]) == {"top_m", "bottom_m", "FOS_liq",
                                "volumetric_strain"}
        assert (rows[0]["top_m"], rows[0]["bottom_m"]) == (0.25, 0.75)
        # the loose sand liquefies and carries Zhang et al.'s strain
        liq = [x for x in rows if x["FOS_liq"] <= 1.0]
        assert liq and all(x["volumetric_strain"] > 0.01 for x in liq)
        assert len(json.dumps(r)) < RESULT_CAP

    @pytest.mark.parametrize("n, dz", [(600, 0.05), (1500, 0.02)])
    def test_a_long_cpt_comes_in_bands_inside_the_cap(self, n, dz):
        r = _cpt(n, dz)
        assert "error" not in r, r
        assert len(r["depth_profile"]) == 40
        assert f"{n} readings summarised in 40 depth bands" in \
            r["depth_profile_note"]
        assert len(json.dumps(r)) < RESULT_CAP

    def test_the_result_goes_straight_into_post_liquefaction_downdrag(self):
        liq = _cpt(600, 0.05, loose=(6.0, 10.0))
        rows = [x for x in liq["depth_profile"] if x["FOS_liq"] <= 1.0]
        assert rows[0]["top_m"] == pytest.approx(6.0, abs=0.05)
        assert rows[-1]["bottom_m"] == pytest.approx(10.0, abs=0.05)
        r = D.call_agent("downdrag", "post_liquefaction_downdrag", {
            **_STATIC, "gwt_depth": 1.0, "liquefaction_result": liq})
        assert "error" not in r, r
        zones = r["liquefied_zones"]
        assert len(zones) == 1
        assert (zones[0]["top_m"], zones[0]["bottom_m"]) == (
            rows[0]["top_m"], rows[-1]["bottom_m"])
        expect = sum(x["volumetric_strain"] * (x["bottom_m"] - x["top_m"])
                     for x in rows)
        assert r["reconsolidation_settlement_mm"] == pytest.approx(
            1000 * expect, rel=1e-3)
        how = {c["what"]: c for c in r["choices"]}
        strain = how["reconsolidation volumetric strain"]
        assert strain["chosen"] == "the liquefaction result's own"
        assert "Zhang et al. (2002)" in strain["basis"]
        assert "volumetric_strain" not in r["defaults_used"]


def test_an_spt_result_still_needs_the_strain_and_says_why():
    liq = D.call_agent("liquefaction", "liquefaction_analysis", {
        "depth": [3, 5, 7], "N160": [8, 7, 30], "FC": [5, 5, 5],
        "gamma": [19, 19, 20], "amax_g": 0.3, "gwt_depth": 1.0,
        "magnitude": 7.5})
    assert "error" not in liq, liq
    r = D.call_agent("downdrag", "post_liquefaction_downdrag", {
        **_STATIC, "gwt_depth": 1.0, "liquefaction_result": liq})
    assert "no reference in the library prints an SPT method" in r["error"]
    assert "Give volumetric_strain" in r["error"]


class TestTheDescriptionsSayIt:

    def test_the_catalog_briefs(self):
        assert "post-liquefaction downdrag" in \
            MODULE_REGISTRY["downdrag"]["brief"]
        brief = MODULE_REGISTRY["liquefaction"]["brief"]
        assert "post-liquefaction downdrag" in brief
        assert "strain by depth" in brief

    def test_the_tools_own_descriptions(self):
        liq = D.describe_method("liquefaction", "liquefaction_analysis")
        text = json.dumps(liq)
        assert "post_liquefaction_downdrag" in text
        assert "depth_profile" in text
        dd = json.dumps(D.describe_method("downdrag",
                                          "post_liquefaction_downdrag"))
        assert "depth_profile" in dd
        assert "returns only LPI/LSN/LDI" not in dd


#: The largest tool-result cap the deep agent sets
#: (deep.tools.DEFAULT_VISION_RESULT_CHARS); a post-liquefaction result is
#: fitted under it (GM-207).
LARGEST_CAP = 32000


def _noisy_cpt(seed=7):
    """A real-looking noisy 25 m CPT read every 2 cm: loose sand from 3 to
    16 m between denser sand, the cone resistance scattered about each
    layer's mean, so FS_liq crosses 1.0 again and again (GM-207)."""
    rng = np.random.default_rng(seed)
    z = np.round(np.arange(1, 1251) * 0.02, 4)
    mean = np.where(z < 3.0, 8000.0, np.where(z < 16.0, 5500.0, 20000.0))
    q_c = mean * np.exp(rng.normal(0.0, 0.35, z.size))
    f_s = q_c * np.clip(0.008 + rng.normal(0.0, 0.002, z.size), 0.002, None)
    return D.call_agent("liquefaction", "liquefaction_analysis", {
        "depth": z.tolist(), "q_c": q_c.tolist(), "f_s": f_s.tolist(),
        "gwt_depth": 1.0, "amax_g": 0.3})


def _post(**params):
    return D.call_agent("downdrag", "post_liquefaction_downdrag",
                        {**_STATIC, **params})


#: The numbers a user checks the drag force and the neutral plane by.
_HEADLINE = ("neutral_plane_before_m", "neutral_plane_after_m",
             "drag_force_before_kN", "drag_force_after_kN",
             "max_pile_load_before_kN", "max_pile_load_after_kN",
             "dragload_after_kN", "pile_settlement_before_mm",
             "pile_settlement_after_mm", "reconsolidation_settlement_mm",
             "reconsolidation_below_toe_mm", "methods_range_after",
             "by_method", "side_resistance_comparison", "warnings",
             "choices", "before")


@needs_liquepy
class TestANoisyCpt:
    """GM-207: twenty thin liquefied zones gave a 44,000-character
    result; it is fitted under the cap and says what was shortened."""

    def test_the_result_fits_and_keeps_every_number_to_check(self):
        liq = _noisy_cpt()
        r = _post(gwt_depth=1.0, liquefaction_result=liq)
        full = _post(gwt_depth=1.0, liquefaction_result=liq, detail="full")
        assert "error" not in r, r
        assert len(json.dumps(full)) > LARGEST_CAP
        assert len(json.dumps(r)) < LARGEST_CAP
        assert len(full["liquefied_zones"]) >= 10
        for key in _HEADLINE:
            assert r[key] == full[key], key
        note = r["result_note"]
        assert list(r).index("result_note") == list(r).index("warnings") + 1
        assert "over the 31,000-character limit" in note
        assert "liquefied_zones as one table" in note
        assert "detail='full' returns it whole" in note
        assert "result_note" not in full

    def test_the_zone_table_holds_every_zone_number(self):
        liq = _noisy_cpt()
        table = _post(gwt_depth=1.0,
                      liquefaction_result=liq)["liquefied_zones"]
        full = _post(gwt_depth=1.0, liquefaction_result=liq, detail="full")
        records = full["liquefied_zones"]
        assert len(table["rows"]) == len(records)
        cols = table["columns"]
        for row, rec in zip(table["rows"], records):
            for col, value in zip(cols, row):
                if col == "position":
                    assert rec["position"].startswith(value)
                else:
                    assert value == rec[col], col
            # below_the_plane_before_of, read as the note says
            top = row[cols.index("top_m")]
            below = {m["name"] for m in full["by_method"]
                     if m["applies_before"] and top < 20.0
                     and top >= m["neutral_plane_before_m"] - 1e-6}
            assert below == set(rec["below_the_plane_before_of"])

    @pytest.mark.parametrize("seed", [3, 11])
    def test_other_noisy_profiles_fit(self, seed):
        r = _post(gwt_depth=1.0, liquefaction_result=_noisy_cpt(seed))
        assert "error" not in r, r
        assert len(json.dumps(r)) < LARGEST_CAP


def test_a_one_zone_result_is_whole():
    r = _post(liquefied_layers=[{"top": 6, "bottom": 10,
                                 "volumetric_strain": 0.03}])
    full = _post(liquefied_layers=[{"top": 6, "bottom": 10,
                                    "volumetric_strain": 0.03}],
                 detail="full")
    assert r == full
    assert "result_note" not in r
    assert isinstance(r["liquefied_zones"], list)
    assert set(r["liquefied_zones"][0]) >= {
        "top_m", "bottom_m", "position", "below_the_plane_before_of"}
    assert "detail" not in r.get("defaults_used", {})

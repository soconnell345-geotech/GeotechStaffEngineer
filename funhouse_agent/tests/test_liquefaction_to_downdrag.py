"""From the ``liquefaction`` tool to ``downdrag.post_liquefaction_downdrag``
(BACKLOG GM-201): the CPT route returns FS and post-liquefaction volumetric
strain by depth, inside the tool-result size cap, and the downdrag tool
takes that result as it is; the SPT routes still need the strain given."""

import json

import numpy as np
import pytest

from funhouse_agent import dispatch as D
from downdrag.liquefaction import MAX_ZONES_LISTED
from funhouse_agent.adapters import MODULE_REGISTRY
from funhouse_agent.adapters.downdrag import RESULT_BUDGET_CHARS
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
        assert len(zones["rows"]) == 1
        assert tuple(zones["rows"][0][:2]) == (
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
        # GM-207: the one methods table is by_method
        assert "neutral_plane_comparison" not in dd
        assert "returned in by_method" in dd


#: The one zone of DESIGN.md hand check HC-1.
_HC1 = [{"top": 6, "bottom": 10, "volumetric_strain": 0.03}]


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


#: The headline: the same in the result and in detail='full'.
_HEADLINE = ("neutral_plane_before_m", "neutral_plane_after_m",
             "drag_force_before_kN", "drag_force_after_kN",
             "max_pile_load_before_kN", "max_pile_load_after_kN",
             "pile_settlement_before_mm", "pile_settlement_after_mm",
             "reconsolidation_settlement_mm", "reconsolidation_below_toe_mm",
             "methods_range_after", "warnings", "choices")


def _rows(table):
    return [dict(zip(table["columns"], row)) for row in table["rows"]]


def _every_number_kept(r, full):
    """GM-207: every neutral plane, drag force and pile settlement of the
    whole result is in the result: the headline, each method before and
    after, each side-resistance option."""
    for key in _HEADLINE:
        assert r[key] == full[key], key
    methods = _rows(r["by_method"])
    assert [m["name"] for m in methods] == \
        [m["name"] for m in full["by_method"]]
    for got, rec in zip(methods, full["by_method"]):
        for state in ("before", "after"):
            for col in (f"neutral_plane_{state}_m",
                        f"drag_force_{state}_kN",
                        f"pile_settlement_{state}_mm"):
                assert got.get(col) == rec[col], (rec["name"], col)
    options = _rows(r["side_resistance_comparison"])
    assert [o["option"] for o in options] == \
        [o["option"] for o in full["side_resistance_comparison"]]
    for got, rec in zip(options, full["side_resistance_comparison"]):
        assert got.get("neutral_plane_m") == rec["neutral_plane_depth_m"]
        assert got.get("drag_force_kN") == rec["drag_force_kN"]
        assert got.get("pile_settlement_mm") == rec["pile_settlement_mm"]


class TestOneZone:
    """GM-207: one liquefied zone (HC-1) comes back under the general
    8,000-character cap, each number once, every one a user checks the drag
    force and the neutral plane by kept."""

    def test_it_is_under_the_cap_and_keeps_every_number(self):
        r = _post(liquefied_layers=_HC1)
        full = _post(liquefied_layers=_HC1, detail="full")
        assert "error" not in r, r
        assert len(json.dumps(r)) < RESULT_CAP
        assert len(json.dumps(full)) > RESULT_BUDGET_CHARS
        _every_number_kept(r, full)
        zones = _rows(r["liquefied_zones"])
        assert len(zones) == 1
        for col, value in zones[0].items():
            assert value == full["liquefied_zones"][0][col], col
        # the side-resistance options cite their sources; detail='full'
        # quotes them
        half = _rows(r["side_resistance_comparison"])[0]
        assert half["source"] == "FHWA GEC-10 §10.7, p. 10-52 (Rollins " \
            "et al. 2018)"
        assert "'Drag loads imposed by previously liquefied layers" in \
            full["side_resistance_comparison"][0]["source"]
        assert "detail" not in r.get("defaults_used", {})

    def test_the_note_says_what_only_the_whole_result_has(self):
        r = _post(liquefied_layers=_HC1)
        full = _post(liquefied_layers=_HC1, detail="full")
        assert list(r).index("result_note") == list(r).index("warnings") + 1
        note = r["result_note"]
        assert note.startswith("Every neutral plane, drag force and pile "
                               "settlement is here. detail='full' adds")
        assert "Q_np (= Q_dead + drag force)" in note
        assert "summarised" not in note
        assert "result_note" not in full
        for key in ("after", "neutral_plane_basis"):
            assert key in full and key not in r
        for key in ("z_m", "axial_load_kN", "below_toe"):
            assert key in full["after"]


@needs_liquepy
class TestANoisyCpt:
    """GM-207: a noisy CPT gave twenty thin liquefied zones and a
    44,000-character result. It now fits the same budget: the zones as
    totals, the largest in full and the rest as depth ranges."""

    def test_it_fits_the_budget_and_keeps_every_number(self):
        liq = _noisy_cpt()
        r = _post(gwt_depth=1.0, liquefaction_result=liq)
        full = _post(gwt_depth=1.0, liquefaction_result=liq, detail="full")
        assert "error" not in r, r
        assert len(json.dumps(r)) < RESULT_BUDGET_CHARS
        assert len(full["liquefied_zones"]) > MAX_ZONES_LISTED
        _every_number_kept(r, full)
        note = r["result_note"]
        assert f"the {len(full['liquefied_zones'])} liquefied zones are " \
            f"summarised. detail='full' adds each zone," in note

    def test_the_zone_summary_accounts_for_every_zone(self):
        liq = _noisy_cpt()
        summary = _post(gwt_depth=1.0,
                        liquefaction_result=liq)["liquefied_zones"]
        full = _post(gwt_depth=1.0, liquefaction_result=liq, detail="full")
        records = full["liquefied_zones"]
        assert summary["zones"] == len(records)
        assert summary["thickness_m"] == pytest.approx(
            sum(z["thickness_m"] for z in records), abs=1e-3)
        for col in ("side_resistance_before_kN", "side_resistance_after_kN"):
            assert summary[col] == pytest.approx(
                sum(z[col] for z in records), abs=0.06)
        largest = _rows(summary["largest_settlement"])
        by_settlement = sorted(records, key=lambda z: -z[
            "reconsolidation_settlement_mm"])[:3]
        assert sorted(z["top_m"] for z in largest) == sorted(
            z["top_m"] for z in by_settlement)
        for got in largest:
            rec = next(z for z in records if z["top_m"] == got["top_m"])
            for col, value in got.items():
                assert value == rec[col], col
        others = summary["others_m"].split(", ")
        assert len(others) + len(largest) == len(records)
        assert others[0] == f"{records[0]['top_m']:g}-" \
            f"{records[0]['bottom_m']:g}"
        # a zone lies below a method's plane before the earthquake where
        # its top_m is at or below that plane (the returns say so)
        for rec in records:
            below = {m["name"] for m in full["by_method"]
                     if m["applies_before"] and rec["top_m"] < 20.0
                     and rec["top_m"] >= m["neutral_plane_before_m"] - 1e-6}
            assert below == set(rec["below_the_plane_before_of"])

    @pytest.mark.parametrize("seed", [3, 11])
    def test_other_noisy_profiles_fit(self, seed):
        r = _post(gwt_depth=1.0, liquefaction_result=_noisy_cpt(seed))
        assert "error" not in r, r
        assert len(json.dumps(r)) < RESULT_BUDGET_CHARS

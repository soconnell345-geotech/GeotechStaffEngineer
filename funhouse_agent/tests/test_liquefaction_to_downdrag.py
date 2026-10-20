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

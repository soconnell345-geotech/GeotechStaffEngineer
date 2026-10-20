"""GM-203: the capacity against length no longer drops a trial length the
analysis refuses without a word (``except: pass``). Each one is listed with
its reason, in the tool's result and in the calc package's table note; a
sweep with every length refused is refused."""

import warnings

import pytest

from drilled_shaft import DrillShaft, ShaftSoilLayer, ShaftSoilProfile
from drilled_shaft.capacity import DrillShaftAnalysis

# GEC-10's rational beta needs N60 (for sigma'p) or OCR on each cohesionless
# layer: a shaft through the sand seam is refused, one stopping above it is
# not (no trial tip of these sweeps lands in the seam, which GM-142 refuses
# for the whole sweep)
_LAYERS = [
    {"thickness": 5, "soil_type": "cohesive", "unit_weight": 18, "cu": 60},
    {"thickness": 0.5, "soil_type": "cohesionless", "unit_weight": 19,
     "phi": 32},
    {"thickness": 14.5, "soil_type": "cohesive", "unit_weight": 18,
     "cu": 80},
]


def _analysis(length=20.0):
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(L["thickness"], L["soil_type"], L["unit_weight"],
                       cu=L.get("cu", 0.0), phi=L.get("phi", 0.0))
        for L in _LAYERS])
    return DrillShaftAnalysis(shaft=DrillShaft(diameter=1.0, length=length),
                              soil=soil, beta_method="rational")


def test_a_refused_length_is_listed_with_its_reason():
    skipped = []
    curve = _analysis().capacity_vs_depth(3.0, 20.0, 2, skipped=skipped)
    assert [r["depth_m"] for r in curve] == [3.0]
    assert [s["depth_m"] for s in skipped] == [20.0]
    assert "needs N60" in skipped[0]["reason"]


def test_without_the_list_a_warning_names_them():
    with pytest.warns(UserWarning, match=r"refused 1 of the 2 .*L = 20 m"):
        _analysis().capacity_vs_depth(3.0, 20.0, 2)


def test_a_sweep_with_nothing_refused_says_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        skipped = []
        assert len(_analysis(5.0).capacity_vs_depth(3.0, 5.0, 3,
                                                    skipped=skipped)) == 3
    assert skipped == []


def test_every_length_refused_is_refused():
    with pytest.raises(ValueError, match="refused 3 of the 3"):
        _analysis().capacity_vs_depth(11.0, 20.0, 3)


def test_the_tool_returns_the_skipped_depths_and_a_warning():
    from funhouse_agent.dispatch import call_agent
    out = call_agent("drilled_shaft", "capacity_vs_depth", {
        "diameter": 1.0, "shaft_length": 20.0, "beta_method": "rational",
        "depth_min": 3.0, "depth_max": 20.0, "n_points": 2,
        "layers": _LAYERS})
    assert "error" not in out, out
    assert len(out["capacity_vs_depth"]) == 1
    assert out["skipped_depths"][0]["depth_m"] == 20.0
    assert "L = 20 m" in out["warning"]
    plain = call_agent("drilled_shaft", "capacity_vs_depth", {
        "diameter": 1.0, "shaft_length": 5.0, "beta_method": "rational",
        "depth_min": 3.0, "depth_max": 5.0, "n_points": 3,
        "layers": _LAYERS})
    assert "skipped_depths" not in plain and "warning" not in plain


def test_the_calc_table_note_names_the_skipped_lengths():
    from drilled_shaft.calc_steps import get_tables

    class _Result:
        def get_tables(self):
            return []

    a = _analysis(4.0)
    table = [t for t in get_tables(_Result(), a)
             if t.name == "capacity_vs_length"][0]
    assert "FS is not applied" in table.note
    assert "The analysis refused" in table.note and "needs N60" in table.note

"""GM-318: a shaft longer than the soil profile is refused unless
``extend_last_layer`` carries the last layer on; the result then says so in
its ``warnings`` (a field it did not have) and ``choices``. Made-up
values."""

import pytest

from drilled_shaft import DrillShaft, ShaftSoilLayer, ShaftSoilProfile
from drilled_shaft.capacity import DrillShaftAnalysis
from funhouse_agent.adapters.drilled_shaft import METHOD_REGISTRY


def _analysis(length, thick=12.0, **kw):
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(5.0, "cohesive", 18.0, cu=60.0),
        ShaftSoilLayer(thick - 5.0, "cohesive", 19.0, cu=90.0),
    ])
    return DrillShaftAnalysis(shaft=DrillShaft(diameter=1.0, length=length),
                              soil=soil, **kw)


def test_shaft_longer_than_profile_refused_and_names_the_depth():
    with pytest.raises(ValueError) as e:
        _analysis(15.0)
    msg = str(e.value)
    assert "ends at 12 m" in msg and "shaft is 15 m long" in msg
    assert "nobody explored" in msg and "extend_last_layer" in msg


def test_base_at_profile_bottom_runs_without_a_warning():
    r = _analysis(12.0).compute()
    assert r.warnings == [] and "warnings" not in r.to_dict()


def test_extension_equals_a_profile_given_to_the_base():
    ext = _analysis(15.0, extend_last_layer=True).compute()
    full = _analysis(15.0, thick=15.0).compute()
    assert ext.Q_skin == pytest.approx(full.Q_skin)
    assert ext.Q_tip == pytest.approx(full.Q_tip)
    d = ext.to_dict()
    assert "last layer was taken to go on" in d["warnings"][0]
    assert d["choices"][0]["what"] == "soil below the profile"
    assert "from 12 m to the base at 15 m" in d["choices"][0]["chosen"]
    assert "Warnings:" in ext.summary()


def test_curve_lists_lengths_past_the_profile_as_skipped():
    skipped = []
    curve = _analysis(12.0).capacity_vs_depth(8.0, 16.0, 3, skipped=skipped)
    assert [r["depth_m"] for r in curve] == [8.0, 12.0]
    assert [s["depth_m"] for s in skipped] == [16.0]
    assert "ends at 12 m" in skipped[0]["reason"]


_PARAMS = {
    "diameter": 1.0,
    "layers": [
        {"thickness": 5.0, "soil_type": "cohesive", "unit_weight": 18.0,
         "cu": 60.0},
        {"thickness": 7.0, "soil_type": "cohesive", "unit_weight": 19.0,
         "cu": 90.0}],
}


@pytest.mark.parametrize("method", ["drilled_shaft_capacity",
                                    "lrfd_capacity"])
def test_tools_refuse_and_extend(method):
    run = METHOD_REGISTRY[method]
    with pytest.raises(ValueError, match="extend_last_layer"):
        run({**_PARAMS, "shaft_length": 15.0})
    out = run({**_PARAMS, "shaft_length": 15.0, "extend_last_layer": True})
    assert "nobody explored" in out["warnings"][0]
    assert out["choices"][0]["what"] == "soil below the profile"

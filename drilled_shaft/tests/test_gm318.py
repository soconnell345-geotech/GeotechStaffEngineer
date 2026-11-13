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


def test_profile_depth_given_deeper_than_the_profile_is_refused():
    """Review C1."""
    with pytest.raises(ValueError, match="profile_depth_given = 20 m"):
        _analysis(15.0, profile_depth_given=20.0)


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


def _choice(r, what):
    return next(c for c in r.choices if c["what"] == what)


def test_clay_base_names_where_the_layers_given_end():
    """Review B1: the base choice and base_steps gave the carried-on depth
    (15 m) as the end of the profile; the layers given end at 12 m."""
    ext = _analysis(15.0, extend_last_layer=True).compute()
    assert ext.base_steps["su_profile_ends_m"] == 12.0
    assert ext.base_steps["su_last_layer_carried_to_m"] == 15.0
    words = _choice(ext, "base resistance in cohesive soil")["chosen"]
    assert "the profile ends at 12 m (its last layer carried on to the " \
           "base at 15 m" in words
    assert "ends at 15 m" not in words
    plain = _analysis(12.0).compute()         # not extended: as before
    assert plain.base_steps["su_profile_ends_m"] == 12.0
    assert "su_last_layer_carried_to_m" not in plain.base_steps
    words = _choice(plain, "base resistance in cohesive soil")["chosen"]
    assert "as the profile ends at 12 m and gives nothing below" in words


def _sand(length, **kw):
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(5.0, "cohesionless", 18.0, phi=32.0, N60=15.0),
        ShaftSoilLayer(7.0, "cohesionless", 19.0, phi=34.0, N60=25.0),
    ])
    return DrillShaftAnalysis(shaft=DrillShaft(diameter=1.0, length=length),
                              soil=soil, **kw).compute()


def test_sand_base_names_where_the_layers_given_end():
    ext = _sand(14.0, extend_last_layer=True)
    assert ext.base_steps["N60_profile_ends_m"] == 12.0
    assert ext.base_steps["N60_last_layer_carried_to_m"] == 14.0
    words = _choice(ext, "base resistance in cohesionless soil")["chosen"]
    assert "the profile ends at 12 m (its last layer carried on to the " \
           "base at 14 m" in words
    assert "ends at 14 m" not in words
    plain = _sand(12.0)
    assert plain.base_steps["N60_profile_ends_m"] == 12.0
    assert "N60_last_layer_carried_to_m" not in plain.base_steps
    words = _choice(plain, "base resistance in cohesionless soil")["chosen"]
    assert "as the profile ends at 12 m" in words and "carried" not in words


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


def test_curve_with_the_extension_says_which_rows_are_assumed():
    """Review A1: a 15 m shaft in a 12 m profile, extended. The rows past
    12 m carry the warning, the result carries it once at the top with the
    choice, and each row is the run of that length alone."""
    out = METHOD_REGISTRY["capacity_vs_depth"]({
        **_PARAMS, "shaft_length": 15.0, "extend_last_layer": True,
        "depth_min": 6.0, "n_points": 4})
    curve = out["capacity_vs_depth"]
    assert [r["depth_m"] for r in curve] == [6.0, 9.0, 12.0, 15.0]
    assert [bool(r.get("warnings")) for r in curve] == [False] * 3 + [True]
    assert "go on to the base" in curve[-1]["warnings"][0]
    assert len(out["warnings"]) == 1
    top = out["warnings"][0]
    assert "ends at 12 m" in top and "(15 m)" in top
    assert "extend_last_layer" in top
    assert [c["what"] for c in out["choices"]] == ["soil below the profile"]
    assert "base at 15 m" in out["choices"][0]["chosen"]
    run = METHOD_REGISTRY["drilled_shaft_capacity"]
    for r in curve:
        alone = run({**_PARAMS, "shaft_length": float(r["depth_m"]),
                     "extend_last_layer": True})
        assert r["Q_ultimate_kN"] == pytest.approx(alone["Q_ultimate_kN"],
                                                   abs=0.06)


def test_curve_inside_the_profile_has_no_extension_words():
    out = METHOD_REGISTRY["capacity_vs_depth"]({
        **_PARAMS, "shaft_length": 12.0, "extend_last_layer": True,
        "depth_min": 6.0, "n_points": 3})
    assert "warnings" not in out and "choices" not in out
    assert not any(r.get("warnings") for r in out["capacity_vs_depth"])


@pytest.mark.parametrize("method", ["drilled_shaft_capacity",
                                    "lrfd_capacity"])
def test_tools_refuse_and_extend(method):
    run = METHOD_REGISTRY[method]
    with pytest.raises(ValueError, match="extend_last_layer"):
        run({**_PARAMS, "shaft_length": 15.0})
    out = run({**_PARAMS, "shaft_length": 15.0, "extend_last_layer": True})
    assert "nobody explored" in out["warnings"][0]
    assert out["choices"][0]["what"] == "soil below the profile"

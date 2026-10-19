"""GM-185: the ground-anchor result states its choices, and the test load
is an input (GEC-4 Sec. 7.4.2.2, p. 143: 133 % common for permanent
anchors, 120 % for temporary ones, 150 % in potentially creeping soils)."""

from funhouse_agent.dispatch import call_agent

_ANCHOR = {"design_load_kN": 300, "anchor_depth": 3, "excavation_depth": 8,
           "phi_deg": 32}


def _how(out):
    return {c["what"]: c["how"] for c in out["choices"]}


def test_the_result_states_its_choices():
    out = call_agent("soe", "design_ground_anchor", _ANCHOR)
    assert _how(out) == {
        "anchor design method": "module",
        "test load": "default",
        "design load limit (fraction of SMTS)": "default",
        "ultimate bond stress": "default",
        "factor of safety on bond": "default",
    }
    test_load = next(c for c in out["choices"] if c["what"] == "test load")
    assert "p. 143" in test_load["basis"] and "p. 147" in test_load["basis"]
    assert len(test_load["alternatives"]) == 2
    bond = next(c for c in out["choices"]
                if c["what"] == "ultimate bond stress")
    assert "not traced to a printed table" in bond["basis"]
    for name in ("test_load_factor", "max_load_pct", "soil_type",
                 "FOS_bond"):
        assert name in out["defaults_used"], name


def test_the_test_load_is_an_input():
    default = call_agent("soe", "design_ground_anchor", _ANCHOR)
    assert default["test_load_kN"] == 399.0
    assert default["tendon"]["n_strands"] == 2
    creeping = call_agent("soe", "design_ground_anchor",
                          {**_ANCHOR, "test_load_factor": 1.5})
    assert creeping["test_load_kN"] == 450.0
    assert creeping["tendon"]["n_strands"] == 3
    assert _how(creeping)["test load"] == "given"
    assert "test_load_factor" not in creeping.get("defaults_used", {})


def test_a_given_bond_stress_leaves_soil_type_out():
    out = call_agent("soe", "design_ground_anchor",
                     {**_ANCHOR, "bond_stress_kPa": 200.0})
    assert "soil_type" not in out["defaults_used"]
    assert _how(out)["ultimate bond stress"] == "given"


def test_the_hammer_efficiency_cites_gec12_volume_2():
    from funhouse_agent.adapters.wave_equation import METHOD_INFO
    words = [spec["description"] for info in METHOD_INFO.values()
             for name, spec in info.get("parameters", {}).items()
             if name == "hammer_type"]
    assert words and all("GEC-12 Vol 2" in w and "p. 205" in w
                         for w in words)

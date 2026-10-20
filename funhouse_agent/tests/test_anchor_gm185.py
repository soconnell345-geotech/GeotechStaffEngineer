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
        "ultimate load transfer": "default",
        "factor of safety on bond": "default",
        "distance behind the critical potential failure surface": "module",
        "minimum unbonded length": "module",
        "minimum bond length": "module",
    }
    test_load = next(c for c in out["choices"] if c["what"] == "test load")
    assert "p. 143" in test_load["basis"] and "p. 147" in test_load["basis"]
    assert len(test_load["alternatives"]) == 2
    # GM-191: GEC-4 Table 6's load transfer, per metre of bond length
    bond = next(c for c in out["choices"]
                if c["what"] == "ultimate load transfer")
    assert "GEC-4 Table 6 (p. 71)" in bond["basis"]
    assert out["load_transfer_kN_per_m"] == 145
    for name in ("test_load_factor", "max_load_pct", "soil_type",
                 "FOS_bond"):
        assert name in out["defaults_used"], name
    # the hole does not enter a load per metre
    assert "drill_diameter_mm" not in out["defaults_used"]


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


def test_a_given_bond_stress_acts_on_the_hole():
    """GM-191: a given stress (GEC-4 Table 7, p. 73) acts over the drill
    hole's perimeter, so the diameter is used; soil_type still names the
    ground for FOS_bond and the minimum bond length, and since GM-199 the
    call gives it."""
    out = call_agent("soe", "design_ground_anchor",
                     {**_ANCHOR, "bond_stress_kPa": 200.0,
                      "soil_type": "sand_medium"})
    assert out["defaults_used"]["drill_diameter_mm"] == 150.0
    assert "soil_type" not in out["defaults_used"]
    assert _how(out)["ultimate load transfer"] == "given"
    assert _how(out)["drill hole diameter"] == "default"
    assert out["load_transfer_kN_per_m"] == 94.2


def test_a_given_bond_stress_needs_its_ground():
    """GM-199: a rock bond stress with no soil_type took the default's
    soil, FS 2.0 and the soil minimum bond length. Now the call is refused
    and asked for the ground; given as 'rock', FS 3.0 (GEC-4 p. 74) and
    defaults_used says so."""
    out = call_agent("soe", "design_ground_anchor",
                     {**_ANCHOR, "bond_stress_kPa": 1000.0})
    assert "soil_type='soil' or 'rock'" in out["error"]
    out = call_agent("soe", "design_ground_anchor",
                     {**_ANCHOR, "bond_stress_kPa": 1000.0,
                      "soil_type": "rock"})
    assert out["FOS_bond"] == 3.0 and out["ground"] == "rock"
    assert out["defaults_used"]["FOS_bond"] == 3.0
    assert "soil_type" not in out["defaults_used"]
    assert _how(out)["factor of safety on bond"] == "default"
    out = call_agent("soe", "design_ground_anchor",
                     {**_ANCHOR, "load_transfer_kN_per_m": 120.0,
                      "soil_type": "soil"})
    assert out["FOS_bond"] == 2.0 and out["ground"] == "soil"
    assert out["defaults_used"]["FOS_bond"] == 2.0


def test_rock_takes_gec4s_factor_of_safety():
    """GM-192: GEC-4 p. 74, 3.0 in competent rock; defaults_used lists
    the value the call used."""
    out = call_agent("soe", "design_ground_anchor",
                     {**_ANCHOR, "soil_type": "granite_basalt"})
    assert out["FOS_bond"] == 3.0
    assert out["defaults_used"]["FOS_bond"] == 3.0


def test_a_removed_soil_type_says_where_it_went():
    out = call_agent("soe", "design_ground_anchor",
                     {**_ANCHOR, "soil_type": "rock_hard"})
    assert "removed" in out["error"] and "granite_basalt" in out["error"]


def test_the_hammer_efficiency_cites_gec12_volume_2():
    from funhouse_agent.adapters.wave_equation import METHOD_INFO
    words = [spec["description"] for info in METHOD_INFO.values()
             for name, spec in info.get("parameters", {}).items()
             if name == "hammer_type"]
    assert words and all("GEC-12 Vol 2" in w and "p. 205" in w
                         for w in words)

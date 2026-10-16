"""What a settlement result says it chose (backlog GM-115)."""

from settlement import ConsolidationLayer, SettlementAnalysis
from settlement.stress_distribution import stress_method_choice


def _analysis(**kw):
    return SettlementAnalysis(
        q_applied=150, q_overburden=20, B=3.0, L=3.0, Es_immediate=15000,
        consolidation_layers=[ConsolidationLayer(
            thickness=4.0, depth_to_center=4.0, e0=0.9, Cc=0.3, Cr=0.05,
            sigma_v0=60.0)], **kw)


def _by_what(result):
    return {c["what"]: c for c in result.choices}


def test_defaults_are_reported_as_defaults():
    c = _by_what(_analysis().compute())
    assert c["immediate settlement"]["how"] == "default"
    assert "Iw = 1.12" in c["immediate settlement"]["chosen"]
    stress = c["stress increase with depth"]
    assert stress["how"] == "default" and stress["chosen"].startswith("2:1")
    assert "Boussinesq (stress_method 'boussinesq')" in stress["alternatives"]
    assert "drainage path" not in c       # no cv: no time curve


def test_given_method_and_the_assumed_t1():
    r = _analysis(stress_method="boussinesq", C_alpha=0.01,
                  t_secondary=10).compute()
    c = _by_what(r)
    assert c["stress increase with depth"]["how"] == "given"
    t1 = c["end of primary consolidation t1"]
    assert t1["chosen"] == "1 year (assumed)" and t1["how"] == "default"
    e0 = c["void ratio e0 for secondary compression"]
    assert e0["chosen"] == "1" and e0["how"] == "default"
    # Hdr is not used without a cv: no drainage path is reported
    assert "drainage path" not in c
    # the assumed t1 is a warning on the result, not only a choice
    (w,) = r.warnings
    assert "assumed t1 = 1 year" in w and "arbitrary" in w
    d = r.to_dict()
    assert d["choices"] == r.choices and d["warnings"] == r.warnings
    assert "Warnings:" in r.summary()


def test_drainage_path_from_the_layers():
    r = _analysis(cv=2.0, C_alpha=0.01, t_secondary=10,
                  e0_secondary=0.9).compute()
    c = _by_what(r)
    assert c["drainage path"]["chosen"].startswith("Hdr = 2 m (double")
    assert c["drainage path"]["how"] == "module"
    assert c["end of primary consolidation t1"]["how"] == "module"
    assert c["void ratio e0 for secondary compression"]["how"] == "given"
    assert r.warnings == []


def test_no_es_says_immediate_was_not_computed():
    r = SettlementAnalysis(q_applied=100, B=2, L=2).compute()
    assert r.choices[0]["chosen"] == "not computed: no Es given"
    assert "Choices:" in r.summary()


def test_stress_method_choice_names_the_method():
    two = stress_method_choice("2to1")
    assert two["how"] == "default"
    assert two["basis"].startswith("FHWA NHI-06-088 Sec. 2.4.2")
    assert two["alternatives"] == ["Boussinesq (stress_method 'boussinesq')",
                                   "Westergaard (stress_method 'westergaard')"]
    assert stress_method_choice("westergaard")["chosen"].startswith(
        "Westergaard")

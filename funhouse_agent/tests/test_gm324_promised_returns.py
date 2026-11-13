"""Every key a tool promises is in its result (BACKLOG GM-324, GM-325).

The module audit of 2026-11-12 found tools whose ``returns`` named keys the
result never carried (the CPT tool's ``n_liquefiable``, the field
correlations' profiles) and results that carried far less than the module
computed (the site-response spectra). For the liquefaction, site-response
and signals tools this runs each one on made-up inputs and checks every
key its ``METHOD_INFO["returns"]`` names against the result. A key that is
there only in some cases is listed with the case that shows it, so each
promised key is seen at least once.
"""

import json

import numpy as np
import pytest

from funhouse_agent.dispatch import call_agent

liquepy = pytest.importorskip("liquepy")


def _call(module, method, params):
    out = call_agent(module, method, params)
    out = json.loads(out) if isinstance(out, str) else out
    assert not out.get("error"), out.get("error")
    return out


def _returns(adapter, method):
    import importlib
    mod = importlib.import_module(f"funhouse_agent.adapters.{adapter}")
    return set(mod.METHOD_INFO[method]["returns"])


_Z = [round(0.5 * i, 1) for i in range(1, 25)]
CPT = dict(depth=_Z, q_c=[4000.0 if z < 6 else 9000.0 for z in _Z],
           f_s=[30.0 if z < 6 else 60.0 for z in _Z])
SPT = dict(depth=[2.0, 5.0, 8.0, 12.0], N160=[8, 12, 18, 22],
           FC=[5, 10, 15, 8], gamma=[16.0, 18.5, 19.5, 20.5],
           amax_g=0.25, gwt_depth=3.0, m_w=7.0)


def _seen(cases):
    """The union of the result keys over the cases, and each result."""
    results = [_call(*c) for c in cases]
    return set().union(*(set(r) for r in results)), results


def test_cpt_liquefaction():
    promised = _returns("liquepy_adapter", "cpt_liquefaction")
    seen, (wet, dry) = _seen([
        ("liquepy", "cpt_liquefaction", {**CPT, "gwl": 2.0, "pga": 0.3}),
        # no reading below the water table: the case with `warnings`
        ("liquepy", "cpt_liquefaction", {**CPT, "gwl": 50.0, "pga": 0.3}),
    ])
    assert promised <= seen, sorted(promised - seen)
    # all but `warnings` in every result
    assert promised - {"warnings"} <= set(wet)
    assert wet["n_liquefiable"] > 0 and "warnings" not in wet
    assert dry["n_liquefiable"] == 0 and dry["min_fos_basis"] == "all readings"


def test_spt_liquefaction():
    promised = _returns("liquepy_adapter", "spt_liquefaction")
    out = _call("liquepy", "spt_liquefaction", SPT)
    assert promised <= set(out), sorted(promised - set(out))


def test_field_correlations():
    promised = _returns("liquepy_adapter", "field_correlations")
    z = [round(0.05 * i, 2) for i in range(4, 204)]          # 200 readings
    long = dict(depth=z, q_c=[3000.0 if x < 5 else 12000.0 for x in z],
                f_s=[40.0 if x < 5 else 60.0 for x in z], gwl=1.5)
    seen, (short, banded) = _seen([
        ("liquepy", "field_correlations", {**CPT, "gwl": 1.5}),
        ("liquepy", "field_correlations", long),   # shows profile_note
    ])
    assert promised <= seen, sorted(promised - seen)
    assert promised - {"profile_note"} <= set(short)
    assert len(short["profile"]) == len(_Z)
    assert len(banded["profile"]) == 40 and banded["profile_note"]
    for row in short["profile"]:
        assert set(row) == {"top_m", "bottom_m", "Vs_m_per_s", "Dr",
                            "su_ratio", "I_c"}


def test_unified_liquefaction_analysis():
    promised = _returns("liquefaction_adapter", "liquefaction_analysis")
    cpt = {**CPT, "gwt_depth": 2.0, "amax_g": 0.3}
    seen, (c, b, n, _dry) = _seen([
        ("liquefaction", "liquefaction_analysis", cpt),
        ("liquefaction", "liquefaction_analysis", SPT),
        ("liquefaction", "liquefaction_analysis",
         {**SPT, "method": "nceer2001"}),
        # no reading below the water table: the case with `warnings`
        ("liquefaction", "liquefaction_analysis",
         {**cpt, "gwt_depth": 50.0}),
    ])
    assert promised <= seen, sorted(promised - seen)
    cpt_only = {"depth_profile", "lpi", "lsn", "ldi_m", "min_fos_basis",
                "max_settlement_mm", "n_susceptible", "warnings",
                "n_points", "gwl_m", "pga_g", "i_c_limit"}
    assert promised - {"layer_results", "warnings"} <= set(c)
    for spt in (b, n):
        assert promised - cpt_only <= set(spt)
        assert not cpt_only & set(spt)


def test_nceer_liquefaction_evaluation():
    promised = _returns("seismic_geotech", "liquefaction_evaluation")
    out = _call("seismic_geotech", "liquefaction_evaluation",
                {"depths": SPT["depth"], **{k: SPT[k] for k in (
                    "N160", "FC", "gamma", "amax_g", "gwt_depth")}})
    assert promised <= set(out), sorted(promised - set(out))


def test_one_total_stress_for_one_gamma():
    """The same layered ``gamma`` gives the same total and effective
    stress on both SPT procedures and on seismic's own NCEER tool."""
    b = _call("liquefaction", "liquefaction_analysis", SPT)
    n = _call("liquefaction", "liquefaction_analysis",
              {**SPT, "method": "nceer2001"})
    s = _call("seismic_geotech", "liquefaction_evaluation",
              {"depths": SPT["depth"], **{k: SPT[k] for k in (
                  "N160", "FC", "gamma", "amax_g", "gwt_depth")}})
    want_v = [32.0, 87.5, 146.0, 228.0]
    want_e = [32.0, 67.9, 96.9, 139.7]
    for out in (b, n, s):
        rows = out["layer_results"]
        assert [r["sigma_v_kPa"] for r in rows] == want_v
        assert [r["sigma_v_eff_kPa"] for r in rows] == want_e
    for out in (b, n):
        stress = [c for c in out["choices"]
                  if c["what"] == "total vertical stress"]
        assert len(stress) == 1
        assert stress[0]["chosen"] == "summed through the layers above"
        # the result says where a change in unit weight is taken (GM-324
        # review: the depths were described as layer mid-depths)
        assert "taken at the upper one" in stress[0]["basis"]
    assert b["choices"][-1]["basis"] == [
        c for c in n["choices"] if c["what"] == "total vertical stress"
    ][0]["basis"]


def test_both_spt_routes_refuse_a_depth_of_zero():
    """GM-324 review: the bi2014 route refused a first depth of 0 and the
    nceer2001 route returned its FOS 99.9 placeholder for it."""
    for method in ("bi2014", "nceer2001"):
        out = call_agent("liquefaction", "liquefaction_analysis",
                         {**SPT, "depth": [0.0, 5.0, 8.0, 12.0],
                          "method": method})
        out = json.loads(out) if isinstance(out, str) else out
        assert "positive" in str(out.get("error")), (method, out)


def test_the_depth_text_says_where_to_put_depths():
    import importlib
    for adapter, method, name in (
            ("liquefaction_adapter", "liquefaction_analysis", "depth"),
            ("liquepy_adapter", "spt_liquefaction", "depth"),
            ("seismic_geotech", "liquefaction_evaluation", "depths")):
        p = importlib.import_module(
            f"funhouse_agent.adapters.{adapter}").METHOD_INFO[method][
                "parameters"][name]
        text = p.get("description") or p.get("brief")
        assert "mid-depth" not in text
        assert "where (N1)60 was measured" in text
        assert "each change in unit weight" in text


def test_the_two_modules_hold_one_rule_text():
    from liquepy_agent.spt_liquefaction import TOTAL_STRESS_RULE as a
    from seismic_geotech.liquefaction import TOTAL_STRESS_RULE as b
    assert a == b


LAYERS = [
    dict(thickness=4, Vs=160, unit_wt=17.5, soil_model="darendeli",
         plas_index=0),
    dict(thickness=8, Vs=220, unit_wt=19, soil_model="menq"),
    dict(thickness=0, Vs=760, unit_wt=22, soil_model="linear", damping=0.01),
]


@pytest.mark.parametrize("method", ["eql_site_response",
                                    "linear_site_response"])
def test_site_response(method):
    pytest.importorskip("pystrata")
    promised = _returns("pystrata_adapter", method)
    base = dict(layers=LAYERS, motion="synthetic_long", target_pga_g=0.1)
    linear_only = [dict(thickness=10, Vs=200, unit_wt=18,
                        soil_model="linear", damping=0.02), LAYERS[-1]]
    seen, (dry, wet, lin) = _seen([
        ("pystrata", method, base),              # shows `warnings`
        ("pystrata", method, {**base, "water_table_depth": 2.0, "k0": 0.6}),
        # no Darendeli or Menq layer: shows `stress_inputs_note`
        ("pystrata", method, {**base, "layers": linear_only, "k0": 0.6}),
    ])
    # depth_profile_note comes only with more than 20 sub-layers, or a
    # profile the module cannot read (pystrata_agent/tests/test_gm325.py
    # shows both)
    promised -= {"depth_profile_note"}
    assert promised <= seen, sorted(promised - seen)
    assert promised - {"warnings", "stress_inputs_note"} <= set(wet)
    assert "warnings" in dry and "warnings" not in wet
    assert wet["water_table_depth_m"] == 2.0 and wet["k0"] == 0.6
    assert lin["k0"] is None and lin["stress_inputs_note"]
    assert lin["max_shear_strain_pct"] > 0
    assert len(wet["Sa_surface_g"]) == len(wet["periods_s"]) == 11
    assert all(v > 0 for v in wet["Sa_surface_g"])
    assert len(json.dumps(wet)) < 8000


@pytest.mark.parametrize("method", ["eql_site_response",
                                    "linear_site_response"])
def test_defaults_used_lists_the_stress_inputs_only_when_used(method):
    """GM-325 re-check (R1): defaults_used listed k0 0.5 and the 'no water
    table' case on runs whose result said neither was used. Now both are
    listed only on a run that computed a Darendeli or Menq layer's
    confining stress."""
    pytest.importorskip("pystrata")
    rock = LAYERS[-1]
    runs = {
        "computed": [LAYERS[0], rock],
        "linear only": [dict(thickness=10, Vs=200, unit_wt=18,
                             soil_model="linear", damping=0.02), rock],
        "stress_mean given": [dict(LAYERS[0], stress_mean=60.0), rock],
    }
    used = {}
    for name, layers in runs.items():
        out = _call("pystrata", method,
                    dict(layers=layers, motion="synthetic_long",
                         target_pga_g=0.1))
        used[name] = out.get("defaults_used") or {}
        assert ("stress_inputs_note" in out) == (name != "computed"), name
    assert used["computed"]["k0"] == 0.5
    assert "no water table" in str(used["computed"]["water_table_depth"])
    for name in ("linear only", "stress_mean given"):
        assert not {"k0", "water_table_depth"} & set(used[name]), (
            name, used[name])


def test_intensity_measures():
    pytest.importorskip("eqsig")
    promised = _returns("seismic_signals_adapter", "intensity_measures")
    seen, (none, given, never) = _seen([
        ("seismic_signals", "intensity_measures",
         {"motion": "synthetic_pulse"}),
        ("seismic_signals", "intensity_measures",
         {"motion": "synthetic_pulse", "bracketed_threshold_g": 0.05}),
        ("seismic_signals", "intensity_measures",
         {"motion": "synthetic_pulse", "bracketed_threshold_g": 5.0}),
    ])
    assert promised <= seen, sorted(promised - seen)
    assert promised - {"bracketed_duration_note"} <= set(given)
    # could not be computed: None with the reason, never 0.0
    for out in (none, never):
        assert out["bracketed_duration_s"] is None
        assert out["bracketed_duration_note"]
    assert given["bracketed_duration_s"] > 0
    assert "bracketed_duration_note" not in given

"""GM-183 and GM-210 through the tools: the printed sources of the
log-spiral passive Kp, the sloping-ground at-rest K0, and the passive
resistance on a shear key."""

import pytest

from funhouse_agent import dispatch as D


def _k(**params):
    return D.call_agent("retaining_walls", "earth_pressure_coefficient",
                        params)


def _ck(**params):
    return _k(phi_deg=params.pop("phi_deg", 30.0), theory="caquot_kerisel",
              state="passive", **params)


def test_log_spiral_default_unchanged_and_listed():
    r = _ck()
    assert r["K"] == 6.3 and r["delta_deg"] == 30.0
    assert r["log_spiral_source"] == "caquot_kerisel_1948"
    assert r["defaults_used"]["log_spiral_source"] == "caquot_kerisel_1948"
    (c,) = r["choices"]
    assert c["chosen"] == "caquot_kerisel_1948" and c["how"] == "default"
    assert set(c["alternatives"]) == {"dm7_2_fig_4_13", "dm7_2_eq_4_12"}
    assert "Fig. 4-20" in c["basis"]


def test_caquot_kerisel_takes_sloping_ground():
    """Caltrans T&S p. 4-29: phi 32, beta 27, delta 14 gives Kp' = 13."""
    r = _ck(phi_deg=32.0, beta_deg=27.0, delta_deg=14.0)
    assert "error" not in r and round(r["K"]) == 13


def test_dm7_2_sources():
    fig = _ck(log_spiral_source="dm7_2_fig_4_13")
    assert fig["K"] == pytest.approx(5.30, abs=0.01)
    assert fig["delta_deg"] == pytest.approx(19.8)     # the chart's 0.66 phi
    assert fig["choices"][0]["how"] == "given"
    sloped = _ck(log_spiral_source="dm7_2_fig_4_13", beta_deg=-12.0)
    assert sloped["K"] == pytest.approx(3.20, rel=0.02)
    eq = _ck(log_spiral_source="dm7_2_eq_4_12", delta_deg=19.8)
    assert eq["K"] == pytest.approx(5.062, abs=0.001)


@pytest.mark.parametrize("params, words", [
    ({"log_spiral_source": "dm7_2_eq_4_12"}, "delta_deg"),
    ({"log_spiral_source": "dm7_2_eq_4_12", "delta_deg": 20.0,
      "beta_deg": 5.0}, "level ground"),
    ({"log_spiral_source": "dm7_2_fig_4_13", "delta_deg": 15.0}, "0.66 only"),
    ({"log_spiral_source": "dm7_2_fig_4_13", "phi_deg": 35.0,
      "beta_deg": 35.0}, "off the chart"),
])
def test_dm7_2_sources_refuse_what_they_do_not_print(params, words):
    r = _ck(**params)
    assert words in r["error"]


def test_source_refused_where_unused():
    r = _k(phi_deg=30.0, state="passive", log_spiral_source="dm7_2_eq_4_12")
    assert "does not use it" in r["error"]
    assert "log_spiral_source" not in (_k(phi_deg=30.0, state="passive")
                                       .get("defaults_used") or {})


def test_rankine_passive_slope_refusal_names_the_log_spiral():
    r = _k(phi_deg=30.0, state="passive", beta_deg=10.0)
    assert "theory 'caquot_kerisel' takes a slope" in r["error"]


def test_sloping_at_rest_needs_a_printed_form():
    r = _k(phi_deg=30.0, state="at_rest", beta_deg=10.0)
    assert "k0_sloping_ground" in r["error"] and "Table 4-3" in r["error"]
    ct = _k(phi_deg=30.0, state="at_rest", beta_deg=10.0,
            k0_sloping_ground="caltrans_eq_4_3_14")
    ec = _k(phi_deg=30.0, state="at_rest", beta_deg=10.0,
            k0_sloping_ground="en_1997_1_eq_9_2")
    # (1 - sin 30)(1 - sin 10) and (1 - sin 30)(1 + sin 10)
    assert ct["K"] == pytest.approx(0.4132, abs=1e-4)
    assert ec["K"] == pytest.approx(0.5868, abs=1e-4)
    assert ct["theory"] == "caltrans_eq_4_3_14"
    assert ec["choices"][0]["alternatives"] == ["caltrans_eq_4_3_14"]
    level = _k(phi_deg=30.0, state="at_rest")
    assert level["K"] == 0.5 and level["theory"] == "jaky_at_rest"


@pytest.mark.parametrize("params, words", [
    ({"beta_deg": 35.0, "k0_sloping_ground": "en_1997_1_eq_9_2"},
     "beta <= phi'"),
    ({"beta_deg": -10.0, "k0_sloping_ground": "caltrans_eq_4_3_14"},
     "rising behind the wall"),
])
def test_sloping_at_rest_limits(params, words):
    assert words in _k(phi_deg=30.0, state="at_rest", **params)["error"]


def test_k0_form_refused_off_at_rest():
    r = _k(phi_deg=30.0, state="active",
           k0_sloping_ground="caltrans_eq_4_3_14")
    assert "Leave it out" in r["error"]


def _wall(**params):
    layers = [{"thickness": 15.0, "unit_weight": 18.0,
               "friction_angle": 30.0,
               "wall_friction_deg": params.pop("delta", 0.0)}]
    return D.call_agent("sheet_pile", "cantilever_wall",
                        {"excavation_depth": 4.0, "layers": layers,
                         **params})


def test_sheet_pile_log_spiral_source():
    ck = _wall(delta=19.8, pressure_method="log_spiral")
    assert ck["defaults_used"]["log_spiral_source"] == "caquot_kerisel_1948"
    fig = _wall(delta=19.8, pressure_method="log_spiral",
                log_spiral_source="dm7_2_fig_4_13")
    assert fig["embedment_depth_m"] > ck["embedment_depth_m"]
    off = _wall(delta=0.0, pressure_method="log_spiral",
                log_spiral_source="dm7_2_fig_4_13")
    assert "layers [0]" in off["error"]
    assert "log_spiral_source" not in (_wall().get("defaults_used") or {})


def test_cantilever_wall_key_passive_on_the_key_face():
    """GM-210: Pp on the key face, base_thickness 0.6 to 1.2 m (UFC 3-220-20
    Fig. 4-29): 1/2 x 3 x 18 x (1.2^2 - 0.6^2) = 29.2 kN/m (was 9.7)."""
    r = D.call_agent("retaining_walls", "cantilever_wall", {
        "wall_height": 5.0, "gamma_backfill": 18.0, "phi_backfill": 30.0,
        "base_width": 3.0, "toe_length": 0.5, "stem_thickness_base": 0.5,
        "base_thickness": 0.6, "has_shear_key": True, "key_depth": 0.6,
        "include_passive": True})
    assert r["sliding_basis"]["Pp_kN_per_m"] == pytest.approx(29.2)

"""GM-183 and GM-210 through the tools: the printed sources of the
log-spiral passive Kp, the sloping-ground at-rest K0, and the passive
resistance on a shear key; GM-232: the UFC 3-220-20 source by the wall's
geometry as the log-spiral default; GM-243: the at-rest K0 behind a slope
is EN 1997-1 Eq. 9.2 always."""

import math

import pytest

from funhouse_agent import dispatch as D


def _k(**params):
    return D.call_agent("retaining_walls", "earth_pressure_coefficient",
                        params)


def _ck(**params):
    return _k(phi_deg=params.pop("phi_deg", 30.0), theory="caquot_kerisel",
              state="passive", **params)


def test_log_spiral_default_is_ufc_by_geometry_and_listed():
    """GM-232: a vertical face and level ground take UFC 3-220-20 Eq. 4-12
    (p. 191), at 0.66 phi' when no wall friction is given; the choices say
    so and give every other source's Kp."""
    r = _ck()
    assert r["K"] == pytest.approx(5.062, abs=0.001)
    assert r["delta_deg"] == pytest.approx(19.8)
    assert r["log_spiral_source"] == "ufc_3_220_20"
    assert r["log_spiral_route"] == "dm7_2_eq_4_12"
    assert r["defaults_used"]["log_spiral_source"] == "ufc_3_220_20"
    c, d = r["choices"]
    assert c["chosen"] == "ufc_3_220_20" and c["how"] == "default"
    assert "Eq. 4-12 (p. 191)" in c["basis"]
    assert c["alternatives"] == [
        "Caquot-Kerisel 1948, Caltrans T&S Fig. 4-20 (log_spiral_source "
        "'caquot_kerisel_1948'): Kp 5.36",
        "UFC 3-220-20 Fig. 4-12 (log_spiral_source 'dm7_2_fig_4_12'): "
        "Kp 5.3",
        "UFC 3-220-20 Fig. 4-13 (log_spiral_source 'dm7_2_fig_4_13'): "
        "Kp 5.3"]
    assert d["what"] == "Wall friction delta (delta_deg)"
    assert d["how"] == "default"
    assert d["chosen"] == "19.8 deg (delta/phi' 0.66)"
    assert "common assumption" in d["basis"]
    assert "engineer's call" in d["basis"]
    assert "Table 4-1 (p. 184" in d["basis"]
    # wall friction given: Eq. 4-12 at it, the same choice called given
    r = _ck(delta_deg=15.0)
    assert r["K"] == pytest.approx(4.459, abs=0.001)
    assert "Kp 4.7" in r["choices"][0]["alternatives"][0]
    (g,) = r["choices"][1:]
    assert g["how"] == "given" and g["basis"] == d["basis"]
    assert "delta_deg" not in r["defaults_used"]


def test_left_out_wall_friction_is_listed_at_the_value_it_took():
    """GM-232: delta_deg left out is listed in defaults_used at the wall
    friction the source took (0.66 phi' for UFC, phi for the 1948 chart),
    not the declared 0.0 the Coulomb theory takes."""
    assert _ck()["defaults_used"]["delta_deg"] == pytest.approx(19.8)
    ck = _ck(log_spiral_source="caquot_kerisel_1948")
    assert ck["K"] == pytest.approx(6.30, abs=0.01)
    assert ck["defaults_used"]["delta_deg"] == pytest.approx(30.0)
    (d,) = [c for c in ck["choices"] if c["what"].startswith("Wall friction")]
    assert d["how"] == "default" and "delta = phi'" in d["basis"]
    # Coulomb keeps its declared 0.0
    co = _k(phi_deg=30.0, theory="coulomb", state="passive")
    assert co["defaults_used"]["delta_deg"] == 0.0


def test_eq_4_12_above_the_charts_wall_friction_is_noted():
    """GM-232: above delta/phi' = 0.66 the default route takes Eq. 4-12,
    which UFC gives as an approximation with no range stated and which
    reads above the 1948 chart at full wall friction; the choice says so.
    At or below 0.66, and on the 1948 chart, no such note."""
    from geotech_common.log_spiral import HIGH_FRICTION_NOTE
    hi = _ck(delta_deg=30.0)
    assert hi["K"] == pytest.approx(6.628, abs=0.001)
    assert HIGH_FRICTION_NOTE in hi["choices"][0]["basis"]
    assert "Kp 6.3" in hi["choices"][0]["alternatives"][0]
    for kw in ({}, {"delta_deg": 15.0},
               {"delta_deg": 30.0,
                "log_spiral_source": "caquot_kerisel_1948"}):
        assert HIGH_FRICTION_NOTE not in _ck(**kw)["choices"][0]["basis"]


def test_battered_face_reads_fig_4_12():
    """GM-232: theta, the face's angle from vertical (UFC 3-220-20 Figs.
    4-7, 4-12), routes to Fig. 4-12 (p. 192) at delta/phi' = 0.66."""
    r = _ck(theta_deg=10.0)
    assert r["K"] == pytest.approx(4.10, abs=0.01)
    assert r["log_spiral_route"] == "dm7_2_fig_4_12"
    assert r["theta_deg"] == 10.0
    assert "Fig. 4-12 (p. 192)" in r["choices"][0]["basis"]
    assert _ck(theta=-10.0)["K"] == pytest.approx(6.60, abs=0.01)  # alias
    # theta = 90 - alpha: a face battered 10 deg reads lower than vertical
    assert r["K"] < _ck()["K"]


@pytest.mark.parametrize("params, words", [
    ({"theta_deg": 10.0, "beta_deg": 5.0}, "prints no log-spiral Kp"),
    ({"theta_deg": 10.0, "delta_deg": 15.0}, "0.66 only"),
    ({"beta_deg": 10.0, "delta_deg": 15.0}, "'caquot_kerisel_1948'"),
    ({"theta_deg": 10.0, "log_spiral_source": "caquot_kerisel_1948"},
     "vertical wall face"),
    ({"alpha_deg": 80.0}, "theta_deg = 90 - alpha_deg"),
])
def test_the_default_refuses_what_ufc_does_not_print(params, words):
    assert words in _ck(**params)["error"]


def test_caquot_kerisel_takes_sloping_ground():
    """Caltrans T&S p. 4-29: phi 32, beta 27, delta 14 gives Kp' = 13 on
    the 1948 chart; UFC 3-220-20 prints no sloping-ground Kp at that wall
    friction, so the default refuses and names the 1948 chart."""
    r = _ck(phi_deg=32.0, beta_deg=27.0, delta_deg=14.0,
            log_spiral_source="caquot_kerisel_1948")
    assert "error" not in r and round(r["K"]) == 13
    r = _ck(phi_deg=32.0, beta_deg=27.0, delta_deg=14.0)
    assert "'caquot_kerisel_1948'" in r["error"]
    r = _ck(phi_deg=32.0, beta_deg=27.0)            # 0.66 phi': Fig. 4-13
    assert r["log_spiral_route"] == "dm7_2_fig_4_13"


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


@pytest.mark.parametrize("phi, beta, k0", [
    # K0 (1 + sin beta), K0 = 1 - sin phi'; 26.565 deg is 2H:1V
    (30.0, 10.0, 0.5868), (30.0, 20.0, 0.6710), (30.0, 26.565, 0.7236),
    (35.0, 10.0, 0.5005), (35.0, 20.0, 0.5723), (35.0, 26.565, 0.6171),
])
def test_sloping_at_rest_is_en_1997_1_eq_9_2(phi, beta, k0):
    """GM-243 (owner OWN-67): behind ground rising from the wall the
    at-rest K0 is EN 1997-1 Eq. 9.2, asked of nobody; the choices give the
    arithmetic, the page and the sign of beta."""
    r = _k(phi_deg=phi, state="at_rest", beta_deg=beta)
    assert r["K"] == pytest.approx(k0, abs=1e-4)
    assert r["theory"] == "en_1997_1_eq_9_2"
    ocr_c, c = r["choices"]
    assert ocr_c["chosen"] == "OCR 1" and ocr_c["how"] == "default"
    assert c["chosen"] == "EN 1997-1 Eq. 9.2" and c["how"] == "module"
    assert "alternatives" not in c
    assert f"= {k0:.3f}." in c["basis"]
    assert "Eq. 9.2 (p. 104)" in c["basis"]
    assert "rising away from the back of the wall" in c["basis"]
    assert "Caltrans" not in c["basis"]


def test_level_at_rest_is_jaky():
    level = _k(phi_deg=30.0, state="at_rest")
    assert level["K"] == 0.5 and level["theory"] == "jaky_at_rest"
    (c,) = level["choices"]  # GM-245: OCR defaulted, and said so
    assert c["chosen"] == "OCR 1" and c["how"] == "default"
    assert "normally consolidated" in c["basis"]
    assert "Eq. 9.1" in c["basis"] and "(p. 104)" in c["basis"]
    assert "very high" not in c["basis"]


def test_ocr_1_is_unchanged_from_jaky():
    """GM-245: OCR 1 given equals OCR left out, level and sloping."""
    for beta in (0.0, 15.0):
        a = _k(phi_deg=32.0, state="at_rest", beta_deg=beta)
        b = _k(phi_deg=32.0, state="at_rest", beta_deg=beta, OCR=1)
        assert a["K"] == b["K"]
        assert b["choices"][0]["how"] == "given"


def test_ocr_4_doubles_level_k0():
    """EN 1997-1 Eq. 9.1: K0 = (1 - sin phi') sqrt(OCR); sqrt(4) = 2."""
    r = _k(phi_deg=30.0, state="at_rest", OCR=4)
    assert r["K"] == pytest.approx(1.0, abs=1e-4)
    assert r["theory"] == "en_1997_1_eq_9_1"
    (c,) = r["choices"]
    assert c["chosen"] == "OCR 4" and c["how"] == "given"
    assert "= 1.000." in c["basis"]
    assert "should not be used for very high values of OCR" in c["basis"]
    assert "your call" in c["basis"]


def test_ocr_with_slope_applies_eq_9_1_then_9_2():
    r = _k(phi_deg=30.0, state="at_rest", OCR=4, beta_deg=20.0)
    assert r["K"] == pytest.approx(1.0 * (1 + math.sin(math.radians(20))),
                                   abs=1e-4)
    assert r["theory"] == "en_1997_1_eq_9_2"
    assert "1.000 (1 + sin 20) = 1.342" in r["choices"][1]["basis"]


def test_ocr_below_1_refused_and_not_for_active():
    assert "at least 1" in _k(phi_deg=30.0, state="at_rest",
                              OCR=0.5)["error"]
    e = _k(phi_deg=30.0, state="active", OCR=3)["error"]
    assert "does not take" in e and "overconsolidation ratio" in e
    assert _k(phi_deg=30.0, state="active", OCR=1)["K"] == pytest.approx(
        1 / 3, abs=1e-3)


@pytest.mark.parametrize("beta, words", [
    (35.0, "beta <= phi'"),
    (-10.0, "ground falling away from the wall"),
])
def test_sloping_at_rest_limits(beta, words):
    assert words in _k(phi_deg=30.0, state="at_rest",
                       beta_deg=beta)["error"]


@pytest.mark.parametrize("state", ["at_rest", "active"])
def test_caltrans_k0_form_refused_pointing_to_eq_9_2(state):
    """GM-243: the input that chose the Caltrans T&S Eq. 4-3-14 form is
    refused, with Eq. 9.2 named; naming Eq. 9.2 itself changes nothing."""
    r = _k(phi_deg=30.0, state=state, beta_deg=10.0,
           k0_sloping_ground="caltrans_eq_4_3_14")
    assert "does not take ['k0_sloping_ground']" in r["error"]
    assert "EN 1997-1:2004 Eq. 9.2 (p. 104)" in r["error"]
    assert "Did you mean" not in r["error"]
    same = _k(phi_deg=30.0, state="at_rest", beta_deg=10.0,
              k0_sloping_ground="en_1997_1_eq_9_2")
    assert same == _k(phi_deg=30.0, state="at_rest", beta_deg=10.0)


def test_k0_form_is_not_advertised():
    from funhouse_agent.adapters.retaining_walls import METHOD_INFO
    info = METHOD_INFO["earth_pressure_coefficient"]
    assert "k0_sloping_ground" not in info["parameters"]
    words = " ".join(spec["description"]
                     for spec in info["parameters"].values())
    assert "4-3-14" not in words and "Eq. 9.2" in words


def _wall(**params):
    layers = [{"thickness": 15.0, "unit_weight": 18.0,
               "friction_angle": 30.0,
               "wall_friction_deg": params.pop("delta", 0.0)}]
    return D.call_agent("sheet_pile", "cantilever_wall",
                        {"excavation_depth": 4.0, "layers": layers,
                         **params})


def test_sheet_pile_log_spiral_source():
    ufc = _wall(delta=19.8, pressure_method="log_spiral")
    assert ufc["defaults_used"]["log_spiral_source"] == "ufc_3_220_20"
    # the source, then each layer's wall friction (GM-242)
    c, friction = ufc["choices"]
    assert friction["chosen"] == "layer 1: 19.8 deg (wall_friction_deg)"
    assert friction["how"] == "given"
    assert c["how"] == "default" and "Eq. 4-12" in c["basis"]
    assert "layer 1 (phi' 30, delta 19.8) Kp 5.06" in c["basis"]
    ck = _wall(delta=19.8, pressure_method="log_spiral",
               log_spiral_source="caquot_kerisel_1948")
    fig = _wall(delta=19.8, pressure_method="log_spiral",
                log_spiral_source="dm7_2_fig_4_13")
    # Kp 5.06 (Eq. 4-12) < 5.30 (Fig. 4-13) < 5.36 (1948)
    assert ufc["embedment_depth_m"] > fig["embedment_depth_m"]         > ck["embedment_depth_m"]
    for source in ("dm7_2_fig_4_13", "dm7_2_fig_4_12"):
        off = _wall(delta=0.0, pressure_method="log_spiral",
                    log_spiral_source=source)
        assert "layers [0]" in off["error"]
    assert "log_spiral_source" not in (_wall().get("defaults_used") or {})
    assert "choices" not in _wall()


def test_anchored_sheet_pile_states_ufc_p_260():
    layers = [{"thickness": 15.0, "unit_weight": 18.0,
               "friction_angle": 30.0, "wall_friction_deg": 15.0}]
    r = D.call_agent("sheet_pile", "anchored_wall",
                     {"excavation_depth": 5.0, "anchor_depth": 1.0,
                      "layers": layers, "pressure_method": "log_spiral"})
    c, friction = r["choices"]
    assert friction["what"].startswith("Wall friction delta")
    assert "p. 260" in c["basis"] and "Figure 4-12" in c["basis"]
    assert "Kp 4.46" in c["basis"]                 # Eq. 4-12 at 0.5
    assert any("Kp 4.7" in a for a in c["alternatives"])


def test_sheet_pile_wall_friction_rows():
    """GM-242: a layer's interface_soil names the steel sheet pile row
    (EM 1110-2-2504 Table 3-3, USS Table 4, UFC 3-220-20 Table 4-1); left
    out, the log-spiral passive side takes the clean sand row, 17 deg."""
    layers = [{"thickness": 15.0, "unit_weight": 18.0,
               "friction_angle": 32.0}]
    args = {"excavation_depth": 4.0, "pressure_method": "log_spiral"}
    left_out = D.call_agent("sheet_pile", "cantilever_wall",
                            {**args, "layers": layers})
    friction = left_out["choices"][-1]
    assert friction["chosen"] == "layer 1: 17 deg (the clean_sand row, taken)"
    assert friction["how"] == "default"
    silty = D.call_agent("sheet_pile", "cantilever_wall", {
        **args, "layers": [{**layers[0], "interface_soil": "silty_sand"}]})
    assert silty["choices"][-1]["chosen"] == (
        "layer 1: 14 deg (interface_soil 'silty_sand')")
    bad = D.call_agent("sheet_pile", "cantilever_wall", {
        **args, "layers": [{**layers[0], "interface_soil": "sand"}]})
    assert "interface_soil must be one of" in bad["error"]


def test_cantilever_wall_key_passive_over_the_key_depth():
    """GM-210: Pp over the key depth, 0.6 to 1.2 m below the top of the
    base (UFC 3-220-20 Fig. 4-29): 1/2 x 3 x 18 x (1.2^2 - 0.6^2) = 29.2
    kN/m (was 9.7)."""
    r = D.call_agent("retaining_walls", "cantilever_wall", {
        "wall_height": 5.0, "gamma_backfill": 18.0, "phi_backfill": 30.0,
        "base_width": 3.0, "toe_length": 0.5, "stem_thickness_base": 0.5,
        "base_thickness": 0.6, "has_shear_key": True, "key_depth": 0.6,
        "include_passive": True})
    assert r["sliding_basis"]["Pp_kN_per_m"] == pytest.approx(29.2)

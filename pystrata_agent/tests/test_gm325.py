"""BACKLOG GM-325: the mean effective stress behind the Darendeli and Menq
curves takes the water table and K0, the result states both, and the
spectra and the profiles with depth come back. Made-up profiles."""

import json

import numpy as np
import pytest

from pystrata_agent import has_pystrata
from pystrata_agent.eql_site_response import (
    GAMMA_W, STRESS_MEAN_FLOOR, curve_layer_stresses, layer_stress,
    stress_choices)
from pystrata_agent.results import (
    PROFILE_MAX_ROWS, REPORTED_PERIODS_S, EQLSiteResponseResult)

needs_pystrata = pytest.mark.skipif(not has_pystrata(),
                                    reason="pystrata not installed")

LAYERS = [
    dict(thickness=4, Vs=160, unit_wt=17.5, soil_model="darendeli",
         plas_index=0),
    dict(thickness=8, Vs=220, unit_wt=19, soil_model="darendeli",
         plas_index=20),
    dict(thickness=12, Vs=350, unit_wt=20, soil_model="menq"),
    dict(thickness=0, Vs=760, unit_wt=22, soil_model="linear", damping=0.01),
]


def test_no_water_table_is_total_stress():
    st = layer_stress(LAYERS, 1)
    # 17.5 x 4 + 19 x 4 at 8 m
    assert st["mid_depth_m"] == 8.0
    assert st["sigma_v_kPa"] == pytest.approx(146.0)
    assert st["u_kPa"] == 0.0
    assert st["stress_mean_kPa"] == pytest.approx(146.0 * 2.0 / 3.0)


def test_water_table_takes_the_pore_pressure_off():
    st = layer_stress(LAYERS, 1, water_table_depth=2.0)
    assert st["u_kPa"] == pytest.approx(GAMMA_W * 6.0)
    assert st["sigma_v_eff_kPa"] == pytest.approx(146.0 - GAMMA_W * 6.0)
    # sigma'm = sigma'v (1 + 2 K0) / 3, NAVFAC DM7.2 Eq. 6-22
    assert st["stress_mean_kPa"] == pytest.approx(
        (146.0 - GAMMA_W * 6.0) * (1 + 2 * 0.5) / 3)
    # a layer wholly above the water table keeps its total stress
    assert layer_stress(LAYERS, 0, water_table_depth=2.0)["u_kPa"] == 0.0
    assert layer_stress(LAYERS, 0, water_table_depth=1.0)["u_kPa"] == \
        pytest.approx(GAMMA_W * 1.0)


def test_k0_enters_the_mean_stress():
    one = layer_stress(LAYERS, 2, water_table_depth=2.0, k0=1.0)
    assert one["stress_mean_kPa"] == pytest.approx(one["sigma_v_eff_kPa"])
    half = layer_stress(LAYERS, 2, water_table_depth=2.0, k0=0.5)
    assert half["stress_mean_kPa"] == pytest.approx(
        one["sigma_v_eff_kPa"] * 2.0 / 3.0)


def test_the_floor_is_said():
    thin = [dict(thickness=0.2, Vs=150, unit_wt=17, soil_model="darendeli",
                 plas_index=0), LAYERS[-1]]
    st = layer_stress(thin, 0)
    assert st["floored"] and st["stress_mean_kPa"] == STRESS_MEAN_FLOOR
    rows = curve_layer_stresses(thin)
    assert "floor" in rows[0]["stress_mean_from"]
    _, warns = stress_choices(thin)
    assert any("set to it" in w for w in warns)


def test_rows_say_given_or_computed():
    layers = [dict(LAYERS[0], stress_mean=40.0)] + LAYERS[1:]
    rows = curve_layer_stresses(layers, water_table_depth=2.0)
    assert [r["layer"] for r in rows] == [1, 2, 3]      # no row for rock
    assert rows[0]["stress_mean_from"] == "given"
    assert rows[0]["stress_mean_kPa"] == 40.0
    assert rows[1]["stress_mean_from"] == "computed"


def test_choices_and_warnings_without_a_water_table():
    choices, warns = stress_choices(LAYERS)
    by = {c["what"]: c for c in choices}
    assert "outcrop" in by["input motion"]["chosen"]
    assert by["water table"]["how"] == "default"
    assert "effective stress = total stress" in by["water table"]["chosen"]
    k0 = by["mean effective stress for the Darendeli and Menq curves"]
    assert "K0 = 0.5" in k0["chosen"] and "DM7.2 Eq. 6-22" in k0["basis"]
    assert len(warns) == 1 and "water_table_depth" in warns[0]


def test_choices_with_a_water_table_carry_no_warning():
    choices, warns = stress_choices(LAYERS, water_table_depth=2.0, k0=0.6)
    by = {c["what"]: c for c in choices}
    assert by["water table"]["chosen"].startswith("2 m")
    assert warns == []


def test_no_curve_layer_no_stress_choice():
    lin = [dict(thickness=10, Vs=300, unit_wt=19, soil_model="linear",
                damping=0.03), LAYERS[-1]]
    choices, warns = stress_choices(lin)
    assert [c["what"] for c in choices] == ["input motion"]
    assert warns == []
    # every stress given: nothing computed, nothing to warn of
    given = [dict(LAYERS[0], stress_mean=30.0), LAYERS[-1]]
    choices, warns = stress_choices(given)
    assert [c["what"] for c in choices] == ["input motion"]
    assert warns == []


def _result(n):
    """A made-up result with n sub-layers over the half-space."""
    tops = np.arange(n + 1, dtype=float)             # last: the half-space
    T = np.logspace(-2, 1, 500)
    return EQLSiteResponseResult(
        depths=np.r_[0.0, tops[:-1] + 0.5],
        max_strain_pct=np.r_[0.0, np.linspace(0.01, 0.2, n)],
        depths_accel=tops, max_accel_g=np.linspace(0.4, 0.2, n + 1),
        # pystrata's step-plot form: the first sub-layer's value twice
        depths_vs=tops, initial_Vs=np.r_[200.0, np.linspace(200, 400, n)],
        compatible_Vs=np.r_[150.0, np.linspace(150, 300, n)],
        periods=T, Sa_surface_g=np.interp(T, [0.01, 0.5, 10], [0.4, 1.2, 0.01]),
        Sa_input_g=np.interp(T, [0.01, 0.5, 10], [0.3, 0.8, 0.01]))


def test_to_dict_carries_the_spectra():
    d = _result(6).to_dict()
    assert d["periods_s"] == list(REPORTED_PERIODS_S)
    assert len(d["Sa_surface_g"]) == len(d["Sa_input_g"]) == 11
    assert d["Sa_surface_g"][0] == pytest.approx(0.4, abs=1e-3)
    assert d["Sa_surface_max_g"] == pytest.approx(1.2, abs=0.01)
    assert d["period_at_Sa_surface_max_s"] == pytest.approx(0.5, abs=0.01)


def test_to_dict_carries_the_profile():
    d = _result(6).to_dict()
    rows = d["depth_profile"]
    assert len(rows) == 6 and "depth_profile_note" not in d
    # the strain is the sub-layer's own (its middle), not the surface point
    assert rows[0] == {"top_m": 0.0, "bottom_m": 1.0,
                       "max_strain_pct": 0.01, "max_accel_g": 0.4,
                       "initial_Vs_m_per_s": 200.0,
                       "compatible_Vs_m_per_s": 150.0}
    assert rows[-1] == {"top_m": 5.0, "bottom_m": 6.0,
                        "max_strain_pct": 0.2,
                        "max_accel_g": round(0.4 - 0.2 * 5 / 6, 4),
                        "initial_Vs_m_per_s": 400.0,
                        "compatible_Vs_m_per_s": 300.0}


def test_a_fine_profile_is_merged_and_says_so():
    r = _result(90)
    d = r.to_dict()
    rows = d["depth_profile"]
    assert len(rows) == PROFILE_MAX_ROWS
    assert "90 sub-layers merged" in d["depth_profile_note"]
    assert rows[0]["top_m"] == 0.0 and rows[-1]["bottom_m"] == 90.0
    # the largest strain of the profile survives the merge
    assert max(x["max_strain_pct"] for x in rows) == pytest.approx(
        float(r.max_strain_pct.max()), abs=1e-4)
    assert len(json.dumps(d)) < 8000


def test_empty_result_has_no_profile_or_spectrum():
    d = EQLSiteResponseResult().to_dict()
    assert "depth_profile" not in d and "periods_s" not in d
    assert d["water_table_depth_m"] is None and d["K0"] == 0.5


@needs_pystrata
class TestThroughPystrata:
    def _run(self, **kw):
        from pystrata_agent import analyze_eql_site_response
        return analyze_eql_site_response(
            LAYERS, motion="synthetic_long", target_pga_g=0.1, **kw)

    def test_profile_rows_are_the_layers_own(self):
        r = self._run()
        rows = r.to_dict()["depth_profile"]
        assert rows[0]["top_m"] == 0.0 and rows[-1]["bottom_m"] == 24.0
        for row in rows:                 # each row inside one input layer
            mid = 0.5 * (row["top_m"] + row["bottom_m"])
            want = 160.0 if mid < 4 else 220.0 if mid < 12 else 350.0
            assert row["initial_Vs_m_per_s"] == want
            assert row["compatible_Vs_m_per_s"] <= want
        assert max(x["max_strain_pct"] for x in rows) == pytest.approx(
            float(r.max_strain_pct.max()), abs=1e-4)

    def test_default_is_unchanged_and_said(self):
        r = self._run()
        d = r.to_dict()
        assert d["water_table_depth_m"] is None and d["K0"] == 0.5
        assert [x["stress_mean_kPa"] for x in d["layer_stresses"]] == \
            pytest.approx([23.33, 97.33, 228.0])
        assert len(d["warnings"]) == 1
        assert len(d["depth_profile"]) >= 3
        assert d["Sa_surface_g"][0] == pytest.approx(d["pga_surface_g"],
                                                     rel=0.02)

    def test_a_water_table_softens_the_curves(self):
        dry, wet = self._run(), self._run(water_table_depth=2.0)
        assert [x["stress_mean_kPa"] for x in wet.layer_stresses] == \
            pytest.approx([23.33, 58.09, 123.36], abs=0.01)
        assert "warnings" not in wet.to_dict()
        # lower confinement, more modulus reduction: larger strains
        assert wet.max_strain_pct.max() > dry.max_strain_pct.max()

    def test_linear_analysis_takes_them_too(self):
        from pystrata_agent import analyze_linear_site_response
        r = analyze_linear_site_response(
            LAYERS, motion="synthetic_long", water_table_depth=2.0, k0=0.8)
        d = r.to_dict()
        assert d["water_table_depth_m"] == 2.0 and d["K0"] == 0.8
        assert d["layer_stresses"][1]["stress_mean_kPa"] == pytest.approx(
            87.14 * (1 + 1.6) / 3, abs=0.01)

    def test_bad_inputs_are_refused(self):
        with pytest.raises(ValueError, match="water_table_depth"):
            self._run(water_table_depth=-1.0)
        with pytest.raises(ValueError, match="k0"):
            self._run(k0=0.0)

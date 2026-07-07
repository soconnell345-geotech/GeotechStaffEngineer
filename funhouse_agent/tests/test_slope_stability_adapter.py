"""le-modern P9 tests: slope_stability adapter — new methods + params.

Covers the modernized LE surface exposed through the funhouse adapter:
- METHOD_INFO/REGISTRY contract + allowed_values (gle/janbu, f_interslice,
  entry_exit / noncircular_de surface types, strength models)
- analyze_slope with the rigorous GLE engine (thrust line, interslice E/X)
- compare_methods_table (F&K-style one-surface/all-methods table)
- entry-exit critical-surface search
- reinforcement pass-through (nails / anchors / geosynthetics)
- SHANSEP and Hoek-Brown per-layer strength models, ponded water
- probabilistic: fosm_fos + monte_carlo_fos
- clear ValueError on bad enum values
"""

import pytest

from funhouse_agent.adapters.slope_stability import (
    METHOD_INFO, METHOD_REGISTRY,
)


# ----------------------------------------------------------------------------
# Shared geometry: 10-m-high 2H:1V slope, single drained layer, GWT below toe
# ----------------------------------------------------------------------------

def _base(**over):
    p = {
        "surface_points": [[0, 10], [20, 10], [40, 20], [70, 20]],
        "soil_layers": [{"top_elevation": 20, "bottom_elevation": -15,
                         "gamma": 18, "gamma_sat": 20,
                         "phi": 25, "c_prime": 5}],
        "gwt_points": [[0, 8], [70, 8]],
        "xc": 30.0, "yc": 32.0, "radius": 26.0,
    }
    p.update(over)
    return p


def _no_circle(p):
    return {k: v for k, v in p.items() if k not in ("xc", "yc", "radius")}


# ----------------------------------------------------------------------------
# METHOD_INFO contract
# ----------------------------------------------------------------------------

class TestMethodInfo:
    def test_keys_match_registry(self):
        assert set(METHOD_INFO.keys()) == set(METHOD_REGISTRY.keys())
        for name, info in METHOD_INFO.items():
            for field in ("category", "brief", "parameters", "returns"):
                assert field in info, f"{name} missing {field}"

    def test_expected_methods(self):
        assert set(METHOD_INFO.keys()) == {
            "analyze_slope", "search_critical_surface",
            "compare_methods_table", "infinite_slope_fos",
            "rapid_drawdown_fos", "search_rapid_drawdown",
            "yield_acceleration",
            "newmark_displacement", "newmark_jibson2007",
            "fosm_fos", "monte_carlo_fos",
        }

    def test_method_allowed_values_modernized(self):
        av = METHOD_INFO["analyze_slope"]["parameters"]["method"]["allowed_values"]
        for m in ("fellenius", "bishop", "janbu", "spencer",
                  "morgenstern_price", "gle"):
            assert m in av

    def test_f_interslice_allowed_values(self):
        av = METHOD_INFO["analyze_slope"]["parameters"]["f_interslice"]["allowed_values"]
        assert set(av) == {"constant", "half_sine", "clipped_sine",
                           "trapezoidal"}

    def test_surface_type_allowed_values(self):
        av = METHOD_INFO["search_critical_surface"]["parameters"]["surface_type"]["allowed_values"]
        for s in ("circular", "entry_exit", "noncircular", "noncircular_de"):
            assert s in av

    def test_defaults_in_allowed_values(self):
        for m, mi in METHOD_INFO.items():
            for p, pi in mi["parameters"].items():
                if "allowed_values" in pi and pi.get("default") is not None:
                    assert pi["default"] in pi["allowed_values"], f"{m}.{p}"


class TestDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        result = list_methods("slope_stability")
        names = [m for v in result.values() for m in v]
        for m in ("analyze_slope", "compare_methods_table", "fosm_fos",
                  "monte_carlo_fos"):
            assert m in names

    def test_describe_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("slope_stability", "analyze_slope")
        assert "f_interslice" in info["parameters"]
        assert "nails" in info["parameters"]


# ----------------------------------------------------------------------------
# analyze_slope: rigorous methods + outputs
# ----------------------------------------------------------------------------

class TestAnalyzeSlope:
    def test_gle_with_slice_data_and_thrust(self):
        r = METHOD_REGISTRY["analyze_slope"](_base(
            method="gle", include_slice_data=True))
        assert 1.5 < r["FOS"] < 2.5
        assert "thrust_line" in r
        row = r["slice_data"][len(r["slice_data"]) // 2]
        for key in ("N_eff_kN_per_m", "S_mob_kN_per_m", "U_base_kN_per_m",
                    "E_left_kN_per_m", "X_right_kN_per_m", "alpha_deg"):
            assert key in row

    def test_janbu_reports_corrected_and_uncorrected(self):
        r = METHOD_REGISTRY["analyze_slope"](_base(method="janbu"))
        assert r["FOS"] == pytest.approx(r["FOS_janbu_corrected"])
        assert r["FOS_janbu_uncorrected"] < r["FOS_janbu_corrected"]
        assert r["janbu_f0"] > 1.0

    def test_constant_f_reproduces_spencer(self):
        r_sp = METHOD_REGISTRY["analyze_slope"](_base(method="spencer"))
        r_mp = METHOD_REGISTRY["analyze_slope"](_base(
            method="morgenstern_price", f_interslice="constant"))
        assert r_mp["FOS"] == pytest.approx(r_sp["FOS"], rel=1e-3)

    def test_compare_methods_table(self):
        t = METHOD_REGISTRY["compare_methods_table"](_base())
        names = [row["method"] for row in t["rows"]]
        assert "Fellenius (OMS)" in names
        assert "Spencer" in names
        assert any(n.startswith("Morgenstern-Price") for n in names)
        assert "Method comparison" in t["summary"]


# ----------------------------------------------------------------------------
# Search options
# ----------------------------------------------------------------------------

class TestSearch:
    def test_entry_exit_search(self):
        r = METHOD_REGISTRY["search_critical_surface"](_no_circle(_base(
            surface_type="entry_exit",
            x_entry_range=[2, 18], x_exit_range=[42, 68],
            nx=4, ny=4, method="bishop", n_slices=20)))
        assert r["critical"]["FOS"] > 0.5
        assert r["n_surfaces_evaluated"] > 0

    def test_invalid_surface_type(self):
        with pytest.raises(ValueError, match="surface_type 'spiral'"):
            METHOD_REGISTRY["search_critical_surface"](_no_circle(_base(
                surface_type="spiral")))


# ----------------------------------------------------------------------------
# Reinforcement pass-through
# ----------------------------------------------------------------------------

class TestReinforcement:
    def test_nail_increases_fos(self):
        plain = METHOD_REGISTRY["analyze_slope"](_base(method="gle"))
        nailed = METHOD_REGISTRY["analyze_slope"](_base(
            method="gle",
            nails=[{"x_head": 30, "z_head": 15, "length": 25,
                    "bond_stress": 150, "spacing_h": 1.0}]))
        assert nailed["FOS"] > plain["FOS"]

    def test_anchor_and_geosynthetic_accepted(self):
        plain = METHOD_REGISTRY["analyze_slope"](_base(method="bishop"))
        r = METHOD_REGISTRY["analyze_slope"](_base(
            method="bishop",
            anchors=[{"x_head": 30, "z_head": 15, "length": 25,
                      "T_allow": 100}],
            geosynthetics=[{"elevation": 12.0, "T_allow": 50}]))
        assert r["FOS"] > plain["FOS"]

    def test_nail_missing_required_key(self):
        with pytest.raises(ValueError, match=r"nails\[\].*length"):
            METHOD_REGISTRY["analyze_slope"](_base(
                nails=[{"x_head": 30, "z_head": 15}]))

    def test_stabilizing_pile_shear_increases_fos(self):
        plain = METHOD_REGISTRY["analyze_slope"](_base(method="bishop"))
        piled = METHOD_REGISTRY["analyze_slope"](_base(
            method="bishop",
            stabilizing_piles=[{"x": 35.0, "shear_capacity": 120.0,
                                "spacing": 1.5}]))
        assert piled["FOS"] > plain["FOS"]

    def test_stabilizing_pile_ito_matsui_accepted(self):
        r = METHOD_REGISTRY["analyze_slope"](_base(
            method="bishop",
            stabilizing_piles=[{"x": 35.0, "ito_matsui": True,
                                "diameter": 0.6, "spacing": 1.5}]))
        assert r["FOS"] > 0 and r["reinforcements"]

    def test_stabilizing_pile_missing_x(self):
        with pytest.raises(ValueError, match=r"stabilizing_piles\[\].*x"):
            METHOD_REGISTRY["analyze_slope"](_base(
                stabilizing_piles=[{"shear_capacity": 100.0}]))

    def test_stabilizing_pile_passive_convention(self):
        """The support_convention='passive' option (Slide2 Method B) gives a
        smaller FOS gain than the default active convention (E6)."""
        active = METHOD_REGISTRY["analyze_slope"](_base(
            method="bishop",
            stabilizing_piles=[{"x": 35.0, "shear_capacity": 120.0,
                                "spacing": 1.5}]))["FOS"]
        passive = METHOD_REGISTRY["analyze_slope"](_base(
            method="bishop",
            stabilizing_piles=[{"x": 35.0, "shear_capacity": 120.0,
                                "spacing": 1.5,
                                "support_convention": "passive"}]))["FOS"]
        plain = METHOD_REGISTRY["analyze_slope"](_base(method="bishop"))["FOS"]
        assert plain < passive < active
        with pytest.raises(ValueError, match="support_convention"):
            METHOD_REGISTRY["analyze_slope"](_base(
                stabilizing_piles=[{"x": 35.0, "shear_capacity": 120.0,
                                    "support_convention": "bogus"}]))


# ----------------------------------------------------------------------------
# Strength models + ponded water
# ----------------------------------------------------------------------------

class TestStrengthModels:
    def test_shansep_ocr_raises_fos(self):
        def run(ocr):
            return METHOD_REGISTRY["analyze_slope"](_base(soil_layers=[{
                "top_elevation": 20, "bottom_elevation": -15,
                "gamma": 18, "gamma_sat": 20, "analysis_mode": "undrained",
                "strength_model": "shansep", "shansep_S": 0.25,
                "shansep_m": 0.8, "ocr": ocr}]))["FOS"]
        assert run(2.0) > run(1.0)

    def test_hoek_brown_runs(self):
        r = METHOD_REGISTRY["analyze_slope"](_base(soil_layers=[{
            "top_elevation": 20, "bottom_elevation": -15, "gamma": 24,
            "strength_model": "hoek_brown", "hb_sigci": 5000,
            "hb_gsi": 45, "hb_mi": 10}]))
        assert r["FOS"] > 1.0

    def test_invalid_strength_model(self):
        with pytest.raises(ValueError, match="strength_model 'tresca'"):
            METHOD_REGISTRY["analyze_slope"](_base(soil_layers=[{
                "top_elevation": 20, "bottom_elevation": -15, "gamma": 18,
                "strength_model": "tresca"}]))

    def test_ponded_water_buttresses(self):
        dry_toe = METHOD_REGISTRY["analyze_slope"](_base())
        ponded = METHOD_REGISTRY["analyze_slope"](_base(
            gwt_points=[[0, 14], [70, 14]]))   # 4 m pond over the toe bench
        assert ponded["FOS"] != pytest.approx(dry_toe["FOS"], rel=1e-3)

    def test_multiple_surcharge_zones(self):
        """E8: several distinct surcharge zones (bench + crest) apply and lower
        the FOS; a zone off the sliding mass is inert; bad zone rejected."""
        plain = METHOD_REGISTRY["analyze_slope"](_base(method="bishop"))["FOS"]
        loaded = METHOD_REGISTRY["analyze_slope"](_base(
            method="bishop",
            surcharges=[{"pressure": 40.0, "x_start": 30.0, "x_end": 45.0},
                        {"pressure": 30.0, "x_start": 45.0, "x_end": 60.0}]))["FOS"]
        assert loaded < plain          # crest-side loads add driving weight
        with pytest.raises(ValueError, match=r"surcharges\[\].*pressure"):
            METHOD_REGISTRY["analyze_slope"](_base(
                surcharges=[{"x_start": 30.0, "x_end": 45.0}]))


# ----------------------------------------------------------------------------
# Infinite slope
# ----------------------------------------------------------------------------

class TestInfiniteSlope:
    def test_dry_cohesionless(self):
        import math
        beta = math.degrees(math.atan(0.4))    # 2.5:1
        r = METHOD_REGISTRY["infinite_slope_fos"](
            {"slope_angle": beta, "phi": 30.0, "gamma": 18.85})
        assert r["FOS"] == pytest.approx(1.443, abs=0.003)   # Slide2 #79

    def test_ru_and_components(self):
        r = METHOD_REGISTRY["infinite_slope_fos"](
            {"slope_angle": 20.0, "phi": 25.0, "gamma": 19.0, "c": 10.0,
             "depth": 3.0, "water_condition": "ru", "ru": 0.3})
        assert r["FOS"] == pytest.approx(1.392, abs=0.005)
        assert r["pore_pressure_kPa"] == pytest.approx(17.1, abs=0.1)

    def test_requires_core_params_and_rejects_bad_water(self):
        with pytest.raises(ValueError):
            METHOD_REGISTRY["infinite_slope_fos"]({"slope_angle": 20.0, "phi": 30.0})
        with pytest.raises(ValueError, match="water_condition"):
            METHOD_REGISTRY["infinite_slope_fos"](
                {"slope_angle": 20.0, "phi": 30.0, "gamma": 19.0,
                 "water_condition": "bogus"})


class TestNewmark:
    def test_yield_acceleration(self):
        r = METHOD_REGISTRY["yield_acceleration"](_base(method="spencer"))
        assert r["converged"] and r["ky"] > 0
        assert r["ay_m_s2"] == pytest.approx(r["ky"] * 9.80665, rel=1e-3)
        assert r["FOS_static"] > 1.0

    def test_newmark_displacement_rectangular_pulse(self):
        # ay = 1 m/s^2 -> ky = 1/g; ap=3, T=2 -> D = ap(ap-ay)T^2/(2ay)
        ky = 1.0 / 9.80665
        dt = 0.001
        accel = [3.0] * int(2.0 / dt) + [0.0] * int(4.0 / dt)
        r = METHOD_REGISTRY["newmark_displacement"](
            {"ky": ky, "accel": accel, "dt": dt})
        ay = ky * 9.80665
        assert r["displacement_m"] == pytest.approx(3.0 * (3.0 - ay) * 4.0
                                                    / (2 * ay), rel=1e-3)

    def test_newmark_jibson2007(self):
        r = METHOD_REGISTRY["newmark_jibson2007"]({"ky": 0.10, "amax": 0.20})
        assert r["displacement_cm"] == pytest.approx(0.877, abs=1e-2)

    def test_newmark_requires_params(self):
        with pytest.raises(ValueError):
            METHOD_REGISTRY["newmark_jibson2007"]({"ky": 0.1})
        with pytest.raises(ValueError):
            METHOD_REGISTRY["newmark_displacement"]({"ky": 0.1, "dt": 0.01})


# ----------------------------------------------------------------------------
# Probabilistic
# ----------------------------------------------------------------------------

class TestProbabilistic:
    def test_fosm(self):
        r = METHOD_REGISTRY["fosm_fos"](_base(
            method="bishop",
            variables={"phi": {"cov": 0.10}, "c_prime": {"cov": 0.30}}))
        assert r["FOS_mean_values"] > 1.5
        assert r["beta_lognormal"] > 0
        assert 0 <= r["pf_lognormal"] <= 1
        assert sum(r["variable_variance_pct"].values()) == pytest.approx(
            100.0, abs=0.5)

    def test_fosm_requires_variables(self):
        with pytest.raises(ValueError, match="variables"):
            METHOD_REGISTRY["fosm_fos"](_base(method="bishop"))

    def test_monte_carlo_seeded_reproducible(self):
        kw = _base(method="bishop", n=150, seed=42,
                   variables={"phi": {"cov": 0.10}})
        a = METHOD_REGISTRY["monte_carlo_fos"](kw)
        b = METHOD_REGISTRY["monte_carlo_fos"](kw)
        assert a["FOS_mean"] == b["FOS_mean"]
        assert a["n_realizations"] == 150
        assert "histogram_counts" in a


# ----------------------------------------------------------------------------
# Error clarity
# ----------------------------------------------------------------------------

class TestErrors:
    def test_invalid_method(self):
        with pytest.raises(ValueError, match=r"method 'sarma'.*Allowed"):
            METHOD_REGISTRY["analyze_slope"](_base(method="sarma"))

    def test_invalid_f_interslice(self):
        with pytest.raises(ValueError, match="f_interslice 'parabolic'"):
            METHOD_REGISTRY["analyze_slope"](_base(f_interslice="parabolic"))


# ----------------------------------------------------------------------------
# v5.4 capabilities: pore-pressure grid, tension-crack side/model, rapid-drawdown
# stage-3 normal option, and the rapid-drawdown critical-surface search
# ----------------------------------------------------------------------------

_FT, _PSF, _PCF = 0.3048, 0.04788, 0.157087


def _dam(**over):
    """#98-style embankment dam geometry (feet -> SI) for rapid drawdown."""
    p = {
        "surface_points": [[0, 0], [100 * _FT, 40 * _FT], [140 * _FT, 60 * _FT],
                           [180 * _FT, 60 * _FT]],
        "soil_layers": [{"top_elevation": 60 * _FT, "bottom_elevation": -2 * _FT,
                         "gamma": 125 * _PCF, "c_prime": 0.0, "phi": 40.0}],
    }
    p.update(over)
    return p


class TestV54PorePressureGrid:
    def test_pore_pressure_points_lowers_fos(self):
        import math
        gw = 9.81
        base = {
            "surface_points": [[0, 0], [10, 0], [30, 10], [50, 10]],
            "soil_layers": [{"top_elevation": 10, "bottom_elevation": -8,
                             "gamma": 20, "phi": 28, "c_prime": 11}],
            "xc": 14.29, "yc": 21.86, "radius": 22.27,
            "method": "bishop", "n_slices": 40,
        }
        dry = METHOD_REGISTRY["analyze_slope"](base)["FOS"]
        grid = [[float(x), float(z), gw * max(5.0 - z, 0.0)]
                for x in range(-2, 53, 5) for z in [-8, -5, 0, 5, 10]]
        wet = METHOD_REGISTRY["analyze_slope"](
            {**base, "pore_pressure_points": grid})["FOS"]
        assert wet < dry
        assert "pore_pressure_points" in \
            METHOD_INFO["analyze_slope"]["parameters"]

    def test_pore_pressure_grid_wires_through_search(self):
        grid = [[float(x), float(z), 9.81 * max(5.0 - z, 0.0)]
                for x in range(-2, 53, 5) for z in [-8, -5, 0, 5, 10]]
        r = METHOD_REGISTRY["search_critical_surface"]({
            "surface_points": [[0, 0], [10, 0], [30, 10], [50, 10]],
            "soil_layers": [{"top_elevation": 10, "bottom_elevation": -8,
                             "gamma": 20, "phi": 28, "c_prime": 11}],
            "surface_type": "circular", "nx": 5, "ny": 5, "n_slices": 20,
            "x_entry_range": [0, 12], "x_exit_range": [28, 50],
            "pore_pressure_points": grid})
        assert r["critical"]["FOS"] > 0.5


class TestV54TensionCrack:
    def _acads1b(self, **over):
        p = {
            "surface_points": [[20, 25], [30, 25], [50, 35], [70, 35]],
            "soil_layers": [{"top_elevation": 35, "bottom_elevation": 10,
                             "gamma": 20, "phi": 10, "c_prime": 32}],
            "xc": 38.04, "yc": 42.94, "radius": 20.47,
            "method": "bishop", "n_slices": 60,
            "tension_crack_depth": 3.814, "tension_crack_water_depth": 3.814,
        }
        p.update(over)
        return p

    def test_exit_side_truncation_matches_slide2(self):
        """Exit-side crack + mass-truncation on the un-mirrored ACADS 1(b) slope
        reproduces Slide2's published Bishop water-crack FOS (1.596)."""
        r = METHOD_REGISTRY["analyze_slope"](self._acads1b(
            tension_crack_side="exit", tension_crack_model="truncation"))
        assert r["FOS"] == pytest.approx(1.597, abs=0.01)

    def test_strength_model_is_more_conservative(self):
        strength = METHOD_REGISTRY["analyze_slope"](self._acads1b(
            tension_crack_side="exit", tension_crack_model="strength"))["FOS"]
        trunc = METHOD_REGISTRY["analyze_slope"](self._acads1b(
            tension_crack_side="exit", tension_crack_model="truncation"))["FOS"]
        assert trunc > strength

    def test_allowed_values_and_bad_values(self):
        gp = METHOD_INFO["analyze_slope"]["parameters"]
        assert gp["tension_crack_side"]["allowed_values"] == ["entry", "exit"]
        assert gp["tension_crack_model"]["allowed_values"] == \
            ["strength", "truncation"]
        with pytest.raises(ValueError, match="tension_crack_side 'bogus'"):
            METHOD_REGISTRY["analyze_slope"](self._acads1b(
                tension_crack_side="bogus"))
        with pytest.raises(ValueError, match="tension_crack_model 'bogus'"):
            METHOD_REGISTRY["analyze_slope"](self._acads1b(
                tension_crack_model="bogus"))


class TestV54RapidDrawdown:
    def test_stage3_effective_normal_raises_fos(self):
        """The 3-stage 'gle' stage-3 normal option raises the FOS above the
        Fellenius default on a specified circle (Slide2 #96 dam)."""
        _FACE = [[0, 0], [220 * _FT, 73 * _FT], [312 * _FT, 110 * _FT],
                 [380 * _FT, 110 * _FT]]
        base = {
            "surface_points": _FACE,
            "soil_layers": [{"top_elevation": 110 * _FT,
                             "bottom_elevation": -1.0 * _FT,
                             "gamma": 135 * _PCF, "phi": 30.0, "c_prime": 0.0,
                             "R_c": 1200 * _PSF, "R_phi": 16.0}],
            "xc": 169.5 * _FT, "yc": 210 * _FT, "radius": 210 * _FT,
            "drawdown_from_elevation": 110 * _FT,
            "drawdown_to_elevation": 24 * _FT,
            "method": "duncan_3stage", "n_slices": 50,
        }
        fell = METHOD_REGISTRY["rapid_drawdown_fos"](base)["FOS"]
        gle = METHOD_REGISTRY["rapid_drawdown_fos"](
            {**base, "stage3_effective_normal": "gle"})["FOS"]
        assert gle > fell
        assert METHOD_INFO["rapid_drawdown_fos"]["parameters"][
            "stage3_effective_normal"]["allowed_values"] == ["fellenius", "gle"]

    def test_bad_stage3_normal_rejected(self):
        with pytest.raises(ValueError, match="stage3_effective_normal"):
            METHOD_REGISTRY["rapid_drawdown_fos"](_dam(
                drawdown_from_elevation=47 * _FT,
                drawdown_to_elevation=15 * _FT,
                stage3_effective_normal="bogus"))


class TestV54SearchRapidDrawdown:
    def test_search_runs_and_returns_detail(self):
        r = METHOD_REGISTRY["search_rapid_drawdown"](_dam(
            drawdown_from_elevation=47 * _FT, drawdown_to_elevation=15 * _FT,
            method="corps_2stage", surface_type="circular", nx=5, ny=4,
            x_range=[30 * _FT, 140 * _FT], y_range=[60 * _FT, 150 * _FT],
            n_slices=20))
        assert 0.3 < r["FOS"] < 2.5
        assert r["method"] == "corps_2stage"
        assert r["n_surfaces_evaluated"] > 0
        assert "drawdown_detail" in r and "search" in r

    def test_requires_drawdown_levels(self):
        with pytest.raises(ValueError, match="drawdown_from_elevation"):
            METHOD_REGISTRY["search_rapid_drawdown"](_dam())

    def test_bad_surface_type_rejected(self):
        with pytest.raises(ValueError, match="surface_type"):
            METHOD_REGISTRY["search_rapid_drawdown"](_dam(
                drawdown_from_elevation=47 * _FT,
                drawdown_to_elevation=15 * _FT, surface_type="pso"))

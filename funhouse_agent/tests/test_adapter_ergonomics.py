"""v5.1 adapter-ergonomics tests.

Covers the four ergonomics tasks:
1. Param-name mismatch fixes — aliases accepted, clear ValueError instead of
   raw KeyError, per adapter.
2. allowed_values rollout — coverage test across all hand-written adapters.
3. (ufc_pavement count reconcile lives in test_reference_adapters.py)
4. RW-5 earth_pressure_coefficient thin method on retaining_walls.

Plus the new module-param exposures: lateral_pile stickup, wave_equation
damping_model, seismic pga, axial cohesive_phi/uplift params, Priebe n0,
sheet_pile embedment_increase, MSE custom reinforcement type.
"""

import importlib
import math

import pytest

from funhouse_agent.adapters import (
    MODULE_REGISTRY, apply_aliases, reject_unknown_params, require_keys,
    require_params,
)


# ============================================================================
# Shared helpers (funhouse_agent.adapters.__init__)
# ============================================================================

class TestSharedHelpers:
    def test_apply_aliases_renames(self):
        assert apply_aliases({"q": 1}, {"q": "q_net"}) == {"q_net": 1}

    def test_apply_aliases_canonical_wins(self):
        out = apply_aliases({"q": 1, "q_net": 2}, {"q": "q_net"})
        assert out == {"q_net": 2}

    def test_apply_aliases_no_alias_returns_same_object(self):
        params = {"a": 1}
        assert apply_aliases(params, {"q": "q_net"}) is params

    def test_require_params_message(self):
        with pytest.raises(ValueError, match=r"my_method.*\['x'\]"):
            require_params({"y": 1}, ["x"], method="my_method")

    def test_require_keys_message(self):
        with pytest.raises(ValueError, match=r"layers\[\].*thickness"):
            require_keys({"unit_weight": 18}, ["thickness", "unit_weight"],
                         method="m")

    def test_reject_unknown_params_message(self):
        with pytest.raises(ValueError, match=r"unknown parameter.*bogus.*Valid"):
            reject_unknown_params({"bogus": 1}, {"good"}, method="m")


# ============================================================================
# Task 2 — allowed_values coverage across adapters
# ============================================================================

# Hand-written adapters that must carry allowed_values on enum-like params.
# (Reference/table adapters — dm7, gec*, ufc_*, fhwa, etc. — build METHOD_INFO
# from docstrings via _reference_common and are exempt.)
ADAPTERS_WITH_ALLOWED_VALUES = [
    "axial_pile", "bearing_capacity", "calc_package", "drilled_shaft",
    "dxf_export", "dxf_import_adapter", "ground_improvement",
    "gstools_adapter", "hvsrpy_adapter", "lateral_pile",
    "liquefaction_adapter", "liquepy_adapter", "pdf_import_adapter",
    "retaining_walls", "seismic_geotech", "settlement", "sheet_pile",
    "slope_stability", "soe", "subsurface_adapter", "swprocess_adapter",
    "wave_equation",
]


class TestAllowedValuesCoverage:
    @pytest.mark.parametrize("mod_name", ADAPTERS_WITH_ALLOWED_VALUES)
    def test_adapter_has_allowed_values(self, mod_name):
        mod = importlib.import_module(f"funhouse_agent.adapters.{mod_name}")
        found = [
            (m, p)
            for m, mi in mod.METHOD_INFO.items()
            for p, pi in (mi.get("parameters") or {}).items()
            if "allowed_values" in pi
        ]
        assert found, f"{mod_name}: no allowed_values params found"

    def test_allowed_values_well_formed_everywhere(self):
        """Every allowed_values list is non-empty and contains its default."""
        for name, spec in MODULE_REGISTRY.items():
            try:
                mod = importlib.import_module(spec["adapter"])
            except Exception:
                continue  # optional-dependency adapters
            for m, mi in getattr(mod, "METHOD_INFO", {}).items():
                for p, pi in (mi.get("parameters") or {}).items():
                    if "allowed_values" not in pi:
                        continue
                    av = pi["allowed_values"]
                    assert isinstance(av, (list, tuple)) and av, \
                        f"{name}.{m}.{p}: empty allowed_values"
                    default = pi.get("default")
                    if default is not None:
                        assert default in av, \
                            f"{name}.{m}.{p}: default {default!r} not in {av}"


# ============================================================================
# Task 1 — param-name mismatch fixes (aliases + clear errors)
# ============================================================================

class TestAxialPileAliases:
    def _layers(self):
        return [{"thickness": 20, "soil_type": "cohesionless",
                 "unit_weight": 19, "friction_angle": 33}]

    def test_pipe_width_alias_for_diameter(self):
        from funhouse_agent.adapters.axial_pile import METHOD_REGISTRY as R
        r = R["axial_pile_capacity"]({
            "pile_type": "pipe_closed", "width": 0.61, "thickness": 0.0127,
            "pile_length": 15.0, "layers": self._layers(),
        })
        assert r["Q_ultimate_kN"] > 0

    def test_concrete_diameter_alias_for_width(self):
        from funhouse_agent.adapters.axial_pile import METHOD_REGISTRY as R
        r = R["axial_pile_capacity"]({
            "pile_type": "concrete_circular", "diameter": 0.45,
            "pile_length": 15.0, "layers": self._layers(),
        })
        assert r["Q_ultimate_kN"] > 0

    def test_missing_diameter_clear_error(self):
        from funhouse_agent.adapters.axial_pile import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match="diameter"):
            R["axial_pile_capacity"]({
                "pile_type": "pipe_closed", "pile_length": 15.0,
                "layers": self._layers(),
            })

    def test_unknown_pile_type_lists_allowed(self):
        from funhouse_agent.adapters.axial_pile import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match="pipe_closed"):
            R["make_pile_section"]({"pile_type": "timber"})


class TestGroundImprovementAliases:
    def test_aggregate_piers_area_replacement_ratio(self):
        from funhouse_agent.adapters.ground_improvement import METHOD_REGISTRY as R
        r = R["aggregate_piers"]({
            "column_diameter": 0.76, "area_replacement_ratio": 0.15,
        })
        assert 0.10 < r["area_replacement_ratio"] < 0.20

    def test_wick_drains_legacy_names(self):
        from funhouse_agent.adapters.ground_improvement import METHOD_REGISTRY as R
        r = R["wick_drains"]({
            "drain_spacing": 1.5, "ch": 3.0, "cv": 1.0, "Hdr": 5.0,
            "time_years": 1.0, "drain_diameter": 0.05,
        })
        assert 0 < r["U_total_percent"] <= 100

    def test_unknown_param_clear_error(self):
        from funhouse_agent.adapters.ground_improvement import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match="unknown parameter.*Valid parameters"):
            R["wick_drains"]({"spacing": 1.5, "ch": 3.0, "cv": 1.0,
                              "Hdr": 5.0, "time": 1.0, "bogus_arg": 7})

    def test_missing_required_clear_error(self):
        from funhouse_agent.adapters.ground_improvement import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match="missing required"):
            R["wick_drains"]({"ch": 3.0})


class TestSoeClearErrors:
    def test_layer_missing_thickness(self):
        from funhouse_agent.adapters.soe import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match=r"layers\[\].*thickness"):
            R["braced_excavation"]({
                "excavation_depth": 6.0,
                "layers": [{"unit_weight": 18, "friction_angle": 30}],
            })

    def test_basal_heave_bjerrum_requires_Be_Le(self):
        from funhouse_agent.adapters.soe import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match="bjerrum_eide.*Be"):
            R["check_basal_heave"]({"method": "bjerrum_eide", "H": 6.0,
                                    "cu": 40.0, "gamma": 18.0})


class TestSettlementAliases:
    def test_elastic_q_alias(self):
        from funhouse_agent.adapters.settlement import METHOD_REGISTRY as R
        r = R["elastic_settlement"]({"q": 100.0, "B": 2.0, "Es": 15000.0})
        assert r["immediate_settlement_mm"] > 0

    def test_consolidation_layer_missing_sigma_v0(self):
        from funhouse_agent.adapters.settlement import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match="sigma_v0"):
            R["consolidation_settlement"]({
                "layers": [{"thickness": 3, "depth_to_center": 4,
                            "e0": 0.9, "Cc": 0.3, "Cr": 0.03}],
                "delta_sigma": 50.0,
            })


class TestLateralPileClearErrors:
    def test_sand_layer_missing_k(self):
        from funhouse_agent.adapters.lateral_pile import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match="SandAPI.*'k'"):
            R["lateral_pile_analysis"]({
                "pile_diameter": 0.6, "pile_length": 15.0, "Vt": 50.0,
                "layers": [{"top": 0, "bottom": 20, "model": "SandAPI",
                            "phi": 33, "gamma": 9.5}],
            })


class TestPileGroupSpacing:
    def test_single_spacing_accepted(self):
        from funhouse_agent.adapters.pile_group import METHOD_REGISTRY as R
        r = R["pile_group_simple"]({
            "n_rows": 2, "n_cols": 3, "spacing": 1.5, "Vz": 3000.0,
        })
        assert "error" not in r

    def test_missing_spacing_clear_error(self):
        from funhouse_agent.adapters.pile_group import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match="spacing"):
            R["pile_group_simple"]({"n_rows": 2, "n_cols": 3, "Vz": 3000.0})


class TestRetainingWallAliases:
    def test_cantilever_short_names(self):
        from funhouse_agent.adapters.retaining_walls import METHOD_REGISTRY as R
        r = R["cantilever_wall"]({"height": 4.0, "gamma": 19.0, "phi": 32.0})
        assert r["FOS_sliding"] > 0

    def test_missing_height_clear_error(self):
        from funhouse_agent.adapters.retaining_walls import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match="wall_height"):
            R["cantilever_wall"]({"gamma_backfill": 19.0, "phi_backfill": 32.0})


class TestSheetPileLayerAliases:
    def test_gamma_phi_aliases(self):
        from funhouse_agent.adapters.sheet_pile import METHOD_REGISTRY as R
        r = R["cantilever_wall"]({
            "excavation_depth": 4.0,
            "layers": [{"thickness": 12, "gamma": 18, "phi": 32}],
        })
        assert r["embedment_depth_m"] > 0

    def test_layer_missing_unit_weight(self):
        from funhouse_agent.adapters.sheet_pile import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match="unit_weight"):
            R["cantilever_wall"]({
                "excavation_depth": 4.0,
                "layers": [{"thickness": 12, "friction_angle": 32}],
            })


class TestSlopeStabilityOptionalName:
    def test_layer_name_optional(self):
        from funhouse_agent.adapters.slope_stability import METHOD_REGISTRY as R
        r = R["analyze_slope"]({
            "surface_points": [[0, 0], [10, 0], [25, 8], [40, 8]],
            "soil_layers": [{"top_elevation": 8, "bottom_elevation": -10,
                             "gamma": 19, "phi": 30, "c_prime": 5}],
            "xc": 17.0, "yc": 18.0, "radius": 17.0,
        })
        assert r.get("FOS", 0) > 0 or "error" not in r

    def test_layer_missing_gamma_clear_error(self):
        from funhouse_agent.adapters.slope_stability import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match=r"soil_layers\[\].*gamma"):
            R["analyze_slope"]({
                "surface_points": [[0, 0], [40, 8]],
                "soil_layers": [{"top_elevation": 8, "bottom_elevation": -10}],
                "xc": 17.0, "yc": 18.0, "radius": 17.0,
            })


class TestReferenceDbMaxResultsAlias:
    def test_max_results_maps_to_limit(self):
        from funhouse_agent.dispatch import call_agent
        res = call_agent("reference_db", "reference_search", {
            "query": "consolidation settlement", "max_results": 2,
        })
        hits = res.get("result", res)
        assert isinstance(hits, list)
        assert len(hits) <= 2


# ============================================================================
# Task 4 — RW-5: earth_pressure_coefficient
# ============================================================================

class TestEarthPressureCoefficient:
    def _run(self, params):
        from funhouse_agent.adapters.retaining_walls import METHOD_REGISTRY as R
        return R["earth_pressure_coefficient"](params)

    def test_rankine_ka(self):
        r = self._run({"phi_deg": 30.0})
        assert r["coefficient"] == "Ka"
        assert r["K"] == pytest.approx(1.0 / 3.0, abs=1e-3)

    def test_rankine_kp(self):
        r = self._run({"phi_deg": 30.0, "state": "passive"})
        assert r["coefficient"] == "Kp"
        assert r["K"] == pytest.approx(3.0, abs=1e-3)

    def test_at_rest_jaky(self):
        r = self._run({"phi_deg": 30.0, "state": "at_rest"})
        assert r["coefficient"] == "K0"
        assert r["K"] == pytest.approx(0.5, abs=1e-3)

    def test_rankine_sloped_backfill_increases_ka(self):
        flat = self._run({"phi_deg": 30.0})["K"]
        sloped = self._run({"phi_deg": 30.0, "beta_deg": 15.0})["K"]
        assert sloped > flat

    def test_coulomb_with_wall_friction_below_rankine(self):
        r = self._run({"phi_deg": 30.0, "theory": "coulomb", "delta_deg": 20.0})
        assert r["coefficient"] == "Ka"
        assert r["K"] < 1.0 / 3.0

    def test_aliases(self):
        r = self._run({"phi": 30.0, "delta": 20.0, "theory": "coulomb"})
        assert r["phi_deg"] == 30.0
        assert r["delta_deg"] == 20.0

    def test_invalid_theory(self):
        with pytest.raises(ValueError, match="rankine"):
            self._run({"phi_deg": 30.0, "theory": "terzaghi"})

    def test_invalid_state(self):
        with pytest.raises(ValueError, match="active"):
            self._run({"phi_deg": 30.0, "state": "neutral"})

    def test_missing_phi(self):
        with pytest.raises(ValueError, match="phi_deg"):
            self._run({"theory": "rankine"})

    def test_registered_in_method_info(self):
        from funhouse_agent.adapters.retaining_walls import (
            METHOD_INFO, METHOD_REGISTRY,
        )
        assert set(METHOD_INFO) == set(METHOD_REGISTRY)
        info = METHOD_INFO["earth_pressure_coefficient"]
        assert info["parameters"]["theory"]["allowed_values"] == [
            "rankine", "coulomb"]
        assert info["parameters"]["state"]["allowed_values"] == [
            "active", "passive", "at_rest"]


# ============================================================================
# New module-param exposures (v5.1 engineering-module additions)
# ============================================================================

class TestLateralPileStickup:
    BASE = {
        "pile_diameter": 0.6, "pile_length": 15.0, "Vt": 100.0,
        "layers": [{"top": 0, "bottom": 20, "model": "SandAPI",
                    "phi": 33, "gamma": 9.5, "k": 16000}],
    }

    def test_stickup_increases_deflection_and_moment(self):
        from funhouse_agent.adapters.lateral_pile import METHOD_REGISTRY as R
        base = R["lateral_pile_analysis"](dict(self.BASE))
        stick = R["lateral_pile_analysis"](dict(self.BASE, stickup=2.0))
        assert stick["max_deflection_m"] > base["max_deflection_m"]
        assert stick["max_moment_kNm"] > base["max_moment_kNm"]

    def test_free_length_alias(self):
        from funhouse_agent.adapters.lateral_pile import METHOD_REGISTRY as R
        a = R["lateral_pile_analysis"](dict(self.BASE, stickup=2.0))
        b = R["lateral_pile_analysis"](dict(self.BASE, free_length=2.0))
        assert a["max_moment_kNm"] == pytest.approx(b["max_moment_kNm"])


class TestWaveEquationDampingModel:
    BASE = {"hammer_name": None, "pile_length": 20.0, "pile_area": 0.01,
            "R_total": 1000.0}

    def _params(self, **kw):
        import wave_equation
        p = dict(self.BASE, hammer_name=wave_equation.list_hammers()[0])
        p.update(kw)
        return p

    def test_single_blow_damping_models_differ(self):
        from funhouse_agent.adapters.wave_equation import METHOD_REGISTRY as R
        smith = R["single_blow"](self._params())
        visc = R["single_blow"](self._params(damping_model="smith_viscous"))
        assert smith["permanent_set_mm"] != visc["permanent_set_mm"]

    def test_single_blow_r_ultimate_alias(self):
        from funhouse_agent.adapters.wave_equation import METHOD_REGISTRY as R
        p = self._params()
        p["R_ultimate"] = p.pop("R_total")
        r = R["single_blow"](p)
        assert r["permanent_set_mm"] > 0

    def test_single_blow_missing_resistance_clear_error(self):
        from funhouse_agent.adapters.wave_equation import METHOD_REGISTRY as R
        p = self._params()
        p.pop("R_total")
        with pytest.raises(ValueError, match="R_total"):
            R["single_blow"](p)

    def test_invalid_damping_model_rejected(self):
        from funhouse_agent.adapters.wave_equation import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match="damping_model"):
            R["single_blow"](self._params(damping_model="case"))


class TestSeismicPga:
    def test_pga_returns_fpga(self):
        from funhouse_agent.adapters.seismic_geotech import METHOD_REGISTRY as R
        r = R["site_classification"]({"vs30": 300.0, "Ss": 1.0, "S1": 0.4,
                                      "pga": 0.4})
        assert r["site_class"] == "D"
        assert r.get("Fpga") is not None


class TestAxialUpliftParams:
    BASE = {
        "pile_type": "pipe_closed", "diameter": 0.61, "wall_thickness": 0.0127,
        "pile_length": 20.0,
        "layers": [{"thickness": 25, "soil_type": "cohesionless",
                    "unit_weight": 19, "friction_angle": 33}],
    }

    def test_include_uplift_and_pile_weight(self):
        from funhouse_agent.adapters.axial_pile import METHOD_REGISTRY as R
        r0 = R["axial_pile_capacity"](dict(self.BASE, include_uplift=True))
        r1 = R["axial_pile_capacity"](dict(self.BASE, include_uplift=True,
                                           pile_weight=30.0))
        assert r1["Q_uplift_kN"] == pytest.approx(r0["Q_uplift_kN"] + 30.0,
                                                  abs=0.2)

    def test_uplift_skin_fraction(self):
        from funhouse_agent.adapters.axial_pile import METHOD_REGISTRY as R
        lo = R["axial_pile_capacity"](dict(self.BASE, include_uplift=True,
                                           uplift_skin_fraction=0.5))
        hi = R["axial_pile_capacity"](dict(self.BASE, include_uplift=True,
                                           uplift_skin_fraction=1.0))
        assert hi["Q_uplift_kN"] > lo["Q_uplift_kN"]

    def test_cohesive_phi_changes_beta_tip(self):
        """cohesive_phi drives the Nt tip term for cohesive layers (beta method)."""
        from funhouse_agent.adapters.axial_pile import METHOD_REGISTRY as R
        base = {
            "pile_type": "pipe_closed", "diameter": 0.61,
            "wall_thickness": 0.0127, "pile_length": 20.0, "method": "beta",
            "layers": [{"thickness": 25, "soil_type": "cohesive",
                        "unit_weight": 18, "cohesion": 60}],
        }
        r25 = R["axial_pile_capacity"](dict(base))
        r35 = R["axial_pile_capacity"](dict(base, cohesive_phi=35.0))
        assert r35["Q_tip_kN"] > r25["Q_tip_kN"]


class TestPriebeImprovementFactor:
    def test_published_form(self):
        from funhouse_agent.adapters.ground_improvement import METHOD_REGISTRY as R
        r = R["priebe_improvement_factor"]({"as_ratio": 0.2})
        Kac = math.tan(math.radians(45.0 - 42.5 / 2.0)) ** 2
        expected = 1.0 + 0.2 * ((5.0 - 0.2) / (4.0 * Kac * (1.0 - 0.2)) - 1.0)
        assert r["n0"] == pytest.approx(expected, abs=0.01)

    def test_area_replacement_ratio_alias(self):
        from funhouse_agent.adapters.ground_improvement import METHOD_REGISTRY as R
        a = R["priebe_improvement_factor"]({"as_ratio": 0.25})
        b = R["priebe_improvement_factor"]({"area_replacement_ratio": 0.25})
        assert a["n0"] == b["n0"]

    def test_out_of_range_raises(self):
        from funhouse_agent.adapters.ground_improvement import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match="replacement ratio"):
            R["priebe_improvement_factor"]({"as_ratio": 1.2})

    def test_registered(self):
        from funhouse_agent.adapters.ground_improvement import (
            METHOD_INFO, METHOD_REGISTRY,
        )
        assert set(METHOD_INFO) == set(METHOD_REGISTRY)
        assert "priebe_improvement_factor" in METHOD_INFO


class TestSheetPileEmbedmentIncrease:
    def test_multiplier_applied(self):
        from funhouse_agent.adapters.sheet_pile import METHOD_REGISTRY as R
        layers = [{"thickness": 12, "unit_weight": 18, "friction_angle": 32}]
        base = R["cantilever_wall"]({"excavation_depth": 4.0, "layers": layers})
        inc = R["cantilever_wall"]({"excavation_depth": 4.0, "layers": layers,
                                    "embedment_increase": 1.2})
        assert inc["embedment_depth_m"] == pytest.approx(
            1.2 * base["embedment_depth_m"], rel=1e-3)


class TestMseCustomReinforcement:
    def test_custom_metallic_strip(self):
        from funhouse_agent.adapters.retaining_walls import METHOD_REGISTRY as R
        r = R["mse_wall"]({
            "wall_height": 6.0, "gamma_backfill": 19.0, "phi_backfill": 34.0,
            "reinforcement_type": "metallic_strip",
            "reinforcement_Tallowable": 40.0,
        })
        assert r["FOS_sliding"] > 0

    def test_unknown_name_without_strength_clear_error(self):
        from funhouse_agent.adapters.retaining_walls import METHOD_REGISTRY as R
        with pytest.raises(ValueError, match="reinforcement_Tallowable"):
            R["mse_wall"]({
                "wall_height": 6.0, "gamma_backfill": 19.0,
                "phi_backfill": 34.0, "reinforcement_name": "mystery_grid",
            })

    def test_reinforcement_type_allowed_values(self):
        from funhouse_agent.adapters.retaining_walls import METHOD_INFO
        av = METHOD_INFO["mse_wall"]["parameters"]["reinforcement_type"][
            "allowed_values"]
        assert av == ["metallic_strip", "metallic_grid", "geosynthetic"]

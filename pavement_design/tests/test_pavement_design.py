"""Tests for pavement_design (AASHTO 1993 orchestration).

Validation anchors are the guide's own printed worked examples, already
digit-verified in geotech_references.aashto_1993:

- Figure 3.1 (flexible): W18=5e6, R=95% (ZR=-1.645), So=0.35, MR=5000 psi,
  dPSI=1.9 -> SN = 5.0 (nomograph; equation solve ~4.95-5.0).
- Figure 3.7 (rigid): W18=5.1e6, k=72 pci, Ec=5e6, Sc'=650, J=3.2, Cd=1.0,
  So=0.29, ZR=-1.645, dPSI=1.7 -> D = 10.0 in (nomograph; solve ~9.7-10.0).
- Figure 2.4 (effective MR): the guide's 12-month example -> 5000 psi.
"""

import math

import pytest

from pavement_design import (PavementLayer, compute_design_esals,
                             design_flexible_pavement, design_rigid_pavement,
                             growth_factor)

# The guide's Figure 3.1 printed worked example inputs.
FLEX_EXAMPLE = dict(w18=5e6, reliability_pct=95, so=0.35, mr_psi=5000,
                    delta_psi=1.9)
# The guide's Figure 3.7 printed worked example inputs.
RIGID_EXAMPLE = dict(w18=5.1e6, sc_psi=650, ec_psi=5e6, reliability_pct=95,
                     so=0.29, delta_psi=1.7, j=3.2, cd=1.0, k_pci=72,
                     pt=2.5)
# The guide's Figure 2.4 seasonal example (printed solution: 5000 psi).
MONTHLY_MR = [20000, 20000, 2500, 4000, 4000, 7000, 7000, 7000, 7000, 7000,
              4000, 20000]

THREE_LAYERS = [
    PavementLayer("asphalt", modulus_psi=400000),
    PavementLayer("granular_base", modulus_psi=30000,
                  drainage_quality="fair", pct_saturation_time="1-5%"),
    PavementLayer("granular_subbase", modulus_psi=11000, m=1.0),
]


# ---------------------------------------------------------------------------
# Flexible: guide worked example + design mode
# ---------------------------------------------------------------------------

class TestFlexibleDesign:
    def test_guide_example_sn_required(self):
        res = design_flexible_pavement(
            layers=[PavementLayer("asphalt", a=0.44)], **FLEX_EXAMPLE)
        # Printed nomograph solution SN = 5.0; equation solve 4.95-5.0.
        assert 4.9 <= res.sn_required <= 5.05
        assert res.zr == pytest.approx(-1.645, abs=1e-3)

    def test_full_depth_design(self):
        res = design_flexible_pavement(
            layers=[PavementLayer("asphalt", a=0.44)], **FLEX_EXAMPLE)
        assert res.mode == "design"
        d1 = res.layers[0]["thickness_in"]
        # SN/a1 ~ 4.98/0.44 ~ 11.3 -> rounds up to 11.5
        assert d1 == pytest.approx(11.5, abs=0.51)
        assert res.sn_provided >= res.sn_required
        assert res.adequate

    def test_three_layer_design_cascade(self):
        res = design_flexible_pavement(layers=THREE_LAYERS, **FLEX_EXAMPLE)
        assert res.mode == "design"
        assert len(res.sn_stack) == 3
        sns = [row["sn_required"] for row in res.sn_stack]
        # Weaker foundation -> larger SN: over base < over subbase < roadbed
        assert sns[0] < sns[1] < sns[2]
        assert res.sn_provided >= res.sn_required - 1e-9
        assert res.adequate
        # Every layer sized and rounded to the 0.5-in increment.
        for lay in res.layers:
            assert lay["thickness_in"] > 0
            frac = (lay["thickness_in"] / 0.5) % 1.0
            assert min(frac, 1 - frac) < 1e-6

    def test_cascade_matches_reference_arithmetic(self):
        """Round-as-you-go split reproduces SN composition exactly."""
        res = design_flexible_pavement(layers=THREE_LAYERS, **FLEX_EXAMPLE)
        a = [lay["a"] for lay in res.layers]
        m = [lay["m"] for lay in res.layers]
        d = [lay["thickness_in"] for lay in res.layers]
        sn = a[0] * d[0] + a[1] * d[1] * m[1] + a[2] * d[2] * m[2]
        assert res.sn_provided == pytest.approx(sn, abs=0.005)

    def test_layer_coefficients_from_guide_values(self):
        """a1(EAC=400k)=0.42 chart; a2(30k)=0.14, a3(15k)=0.11 printed checks."""
        res = design_flexible_pavement(
            layers=[
                PavementLayer("asphalt", modulus_psi=400000),
                PavementLayer("granular_base", modulus_psi=30000),
                PavementLayer("granular_subbase", modulus_psi=15000),
            ], **FLEX_EXAMPLE)
        assert res.layers[0]["a"] == pytest.approx(0.42, abs=0.005)
        assert res.layers[1]["a"] == pytest.approx(0.14, abs=0.005)
        assert res.layers[2]["a"] == pytest.approx(0.11, abs=0.005)

    def test_seasonal_mr_guide_example(self):
        res = design_flexible_pavement(
            w18=5e6, reliability_pct=95, so=0.35, delta_psi=1.9,
            monthly_mr_psi=MONTHLY_MR,
            layers=[PavementLayer("asphalt", a=0.44)])
        # Printed solution: effective MR = 5,000 psi.
        assert res.effective_mr_psi == pytest.approx(5000, rel=0.05)

    def test_minimum_thickness_governs_low_traffic(self):
        res = design_flexible_pavement(
            w18=100000, reliability_pct=80, so=0.45, mr_psi=15000, pt=2.0,
            layers=[
                PavementLayer("asphalt", modulus_psi=400000),
                PavementLayer("granular_base", modulus_psi=30000),
            ])
        # Section 3.1.4: 50k-150k ESALs -> AC min 2.0 in, base min 4 in.
        assert res.layers[0]["thickness_in"] >= 2.0
        assert res.layers[1]["thickness_in"] >= 4.0
        assert res.adequate

    def test_check_mode_adequate_and_inadequate(self):
        thick = [
            PavementLayer("asphalt", a=0.44, thickness_in=6.0),
            PavementLayer("granular_base", a=0.14, m=1.0, thickness_in=12.0),
        ]
        res = design_flexible_pavement(layers=thick, **FLEX_EXAMPLE)
        assert res.mode == "check"
        # SN = 0.44*6 + 0.14*12 = 4.32 < ~4.98 required
        assert res.sn_provided == pytest.approx(4.32, abs=0.01)
        assert not res.adequate

        thin_ok = [
            PavementLayer("asphalt", a=0.44, thickness_in=8.0),
            PavementLayer("granular_base", a=0.14, m=1.0, thickness_in=12.0),
        ]
        res2 = design_flexible_pavement(layers=thin_ok, **FLEX_EXAMPLE)
        assert res2.sn_provided == pytest.approx(5.20, abs=0.01)
        assert res2.adequate
        assert res2.w18_capacity > res2.w18

    def test_forward_check_consistency(self):
        """Capacity of the provided section must exceed W18 in design mode."""
        res = design_flexible_pavement(layers=THREE_LAYERS, **FLEX_EXAMPLE)
        assert res.w18_capacity >= res.w18

    def test_drainage_midpoint_policy(self):
        res = design_flexible_pavement(layers=THREE_LAYERS, **FLEX_EXAMPLE)
        # fair / 1-5% -> Table 2.4 (1.15, 1.05) -> midpoint 1.10
        assert res.layers[1]["m"] == pytest.approx(1.10, abs=0.001)
        assert "Table 2.4 midpoint" in res.layers[1]["m_basis"]

    def test_references_carried(self):
        res = design_flexible_pavement(layers=THREE_LAYERS, **FLEX_EXAMPLE)
        joined = " | ".join(res.references)
        assert "Figure 3.1" in joined
        assert "Table 4.1" in joined
        assert "Figure 3.2" in joined

    def test_to_dict_and_summary(self):
        res = design_flexible_pavement(layers=THREE_LAYERS, **FLEX_EXAMPLE)
        d = res.to_dict()
        assert d["sn_required"] == res.sn_required
        assert isinstance(res.summary(), str)
        assert "SN required" in res.summary()


class TestFlexibleValidation:
    def test_rejects_bad_w18(self):
        with pytest.raises(ValueError, match="w18"):
            design_flexible_pavement(
                w18=0, reliability_pct=95, mr_psi=5000,
                layers=[PavementLayer("asphalt", a=0.44)])

    def test_rejects_no_layers(self):
        with pytest.raises(ValueError, match="at least one"):
            design_flexible_pavement(mr_psi=5000, reliability_pct=95, w18=1e6,
                                     layers=[])

    def test_rejects_non_asphalt_top(self):
        with pytest.raises(ValueError, match="asphalt"):
            design_flexible_pavement(
                w18=1e6, reliability_pct=95, mr_psi=5000,
                layers=[PavementLayer("granular_base", modulus_psi=30000)])

    def test_rejects_mixed_thickness_spec(self):
        with pytest.raises(ValueError, match="EVERY layer"):
            design_flexible_pavement(
                w18=1e6, reliability_pct=95, mr_psi=5000,
                layers=[
                    PavementLayer("asphalt", a=0.44, thickness_in=4.0),
                    PavementLayer("granular_base", a=0.14),
                ])

    def test_rejects_missing_modulus_in_design_mode(self):
        with pytest.raises(ValueError, match="modulus_psi"):
            design_flexible_pavement(
                w18=1e6, reliability_pct=95, mr_psi=5000,
                layers=[
                    PavementLayer("asphalt", a=0.44),
                    PavementLayer("granular_base", a=0.14),
                ])

    def test_rejects_missing_reliability(self):
        with pytest.raises(ValueError, match="reliability"):
            design_flexible_pavement(
                w18=1e6, mr_psi=5000,
                layers=[PavementLayer("asphalt", a=0.44)])

    def test_rejects_missing_mr(self):
        with pytest.raises(ValueError, match="mr_psi"):
            design_flexible_pavement(
                w18=1e6, reliability_pct=95,
                layers=[PavementLayer("asphalt", a=0.44)])

    def test_rejects_subbase_without_base(self):
        with pytest.raises(ValueError, match="base course"):
            design_flexible_pavement(
                w18=1e6, reliability_pct=95, mr_psi=5000,
                layers=[
                    PavementLayer("asphalt", a=0.44),
                    PavementLayer("granular_subbase", modulus_psi=15000),
                ])

    def test_rejects_dict_layer_with_bad_type(self):
        with pytest.raises(ValueError, match="asphalt"):
            design_flexible_pavement(
                w18=1e6, reliability_pct=95, mr_psi=5000,
                layers=[{"layer_type": "concrete"}])
        with pytest.raises(ValueError, match="layer_type"):
            design_flexible_pavement(
                w18=1e6, reliability_pct=95, mr_psi=5000,
                layers=[PavementLayer("asphalt", a=0.44),
                        PavementLayer("gravel", modulus_psi=30000)])

    def test_accepts_dict_layers(self):
        res = design_flexible_pavement(
            w18=1e6, reliability_pct=90, mr_psi=8000,
            layers=[{"layer_type": "asphalt", "a": 0.44}])
        assert res.mode == "design"
        assert res.adequate


# ---------------------------------------------------------------------------
# Rigid: guide worked example
# ---------------------------------------------------------------------------

class TestRigidDesign:
    def test_guide_example_d_required(self):
        res = design_rigid_pavement(**RIGID_EXAMPLE)
        # Printed nomograph solution D = 10.0 in; equation solve ~9.7-10.0.
        assert 9.6 <= res.d_required_in <= 10.05
        assert res.mode == "design"
        assert res.d_provided_in >= res.d_required_in
        assert res.adequate

    def test_check_mode(self):
        res = design_rigid_pavement(slab_thickness_in=10.0, **RIGID_EXAMPLE)
        assert res.mode == "check"
        assert res.adequate
        thin = design_rigid_pavement(slab_thickness_in=8.0, **RIGID_EXAMPLE)
        assert not thin.adequate
        assert thin.w18_capacity < thin.w18

    def test_simple_k_from_mr(self):
        params = {k: v for k, v in RIGID_EXAMPLE.items() if k != "k_pci"}
        res = design_rigid_pavement(mr_psi=1397, **params)
        # k = 1397/19.4 = 72.0 pci -> matches the direct-k example
        assert res.k_pci == pytest.approx(72.0, abs=0.1)
        assert res.k_basis["basis"] == "simple_mr_over_19.4"
        assert 9.6 <= res.d_required_in <= 10.05

    def test_j_from_table(self):
        params = {k: v for k, v in RIGID_EXAMPLE.items() if k != "j"}
        res = design_rigid_pavement(
            pavement_type="plain_jointed_jrcp", shoulder_type="asphalt",
            load_transfer_devices=True, **params)
        # Table 2.6: dowelled JCP/JRCP w/ asphalt shoulder -> J = 3.2
        assert res.j == pytest.approx(3.2, abs=0.001)

    def test_cd_from_table(self):
        params = {k: v for k, v in RIGID_EXAMPLE.items() if k != "cd"}
        res = design_rigid_pavement(
            drainage_quality="good", pct_saturation_time="1-5%", **params)
        # Table 2.5 good/1-5% -> (1.15, 1.10) midpoint 1.125
        assert res.cd == pytest.approx(1.125, abs=0.001)

    def test_requires_exactly_one_k_source(self):
        params = {k: v for k, v in RIGID_EXAMPLE.items() if k != "k_pci"}
        with pytest.raises(ValueError, match="exactly one"):
            design_rigid_pavement(**params)
        with pytest.raises(ValueError, match="exactly one"):
            design_rigid_pavement(k_pci=72, mr_psi=1397, **params)

    def test_composite_k_spec_without_module_or_runs(self):
        """Composite-k spec either runs (module present) or raises the
        informative NotImplementedError (module absent)."""
        params = {k: v for k, v in RIGID_EXAMPLE.items() if k != "k_pci"}
        spec = {"seasonal": [{"mr_psi": 6000}] * 12, "dsb_in": 6.0,
                "esb_psi": 20000, "ls": 1.0}
        try:
            res = design_rigid_pavement(composite_k=spec, **params)
        except NotImplementedError as exc:
            assert "composite_k" in str(exc)
        else:
            assert res.k_pci > 0
            assert res.k_basis["basis"] == "composite_section_3.2"

    def test_references_carried(self):
        res = design_rigid_pavement(**RIGID_EXAMPLE)
        joined = " | ".join(res.references)
        assert "Figure 3.7" in joined
        assert "Table 4.1" in joined

    def test_to_dict_and_summary(self):
        res = design_rigid_pavement(**RIGID_EXAMPLE)
        d = res.to_dict()
        assert d["d_required_in"] == res.d_required_in
        assert "D required" in res.summary()


# ---------------------------------------------------------------------------
# Traffic / ESALs
# ---------------------------------------------------------------------------

class TestTraffic:
    def test_growth_factor_standard_values(self):
        # Classic check: 4%/yr over 20 yr -> 29.78
        assert growth_factor(4, 20)["growth_factor"] == pytest.approx(
            29.778, abs=0.01)
        # Zero growth -> n years
        assert growth_factor(0, 15)["growth_factor"] == 15.0
        # 2% over 10 -> (1.02^10-1)/0.02 = 10.9497
        assert growth_factor(2, 10)["growth_factor"] == pytest.approx(
            10.9497, abs=0.001)

    def test_direct_base_year_path(self):
        res = compute_design_esals(
            base_year_w18_two_way=200000, growth_rate_pct=0,
            design_period_yr=20, num_lanes_per_direction=1,
            directional_factor=0.5)
        # 200k * 20 = 4e6 two-way; DD 0.5, DL 1.0 (1 lane) -> 2e6
        assert res.w18_two_way_total == pytest.approx(4e6)
        assert res.lane_factor == 1.0
        assert res.w18_design_lane == pytest.approx(2e6)

    def test_truck_factor_path(self):
        res = compute_design_esals(
            vehicles=[
                {"description": "5-axle semi", "daily_count": 500,
                 "truck_factor": 1.2},
                {"description": "SU truck", "daily_count": 300,
                 "truck_factor": 0.4},
            ],
            growth_rate_pct=0, design_period_yr=1,
            num_lanes_per_direction=1, directional_factor=0.5)
        base = (500 * 1.2 + 300 * 0.4) * 365
        assert res.base_year_w18_two_way == pytest.approx(base, rel=1e-6)
        assert res.lef_basis == "truck_factors"

    def test_axle_spectrum_at_digitized_point(self):
        """18-kip single axle on SN=5/pt=2.5 has LEF = 1.0 by definition."""
        res = compute_design_esals(
            axle_groups=[{"axle_config": "single", "load_kips": 18,
                          "daily_count": 100}],
            pavement_type="flexible", sn=5.0, pt=2.5,
            growth_rate_pct=0, design_period_yr=1,
            num_lanes_per_direction=1, directional_factor=1.0,
            lane_factor=1.0)
        assert res.axle_breakdown[0]["lef"] == pytest.approx(1.0, abs=0.02)
        assert res.w18_design_lane == pytest.approx(36500, rel=0.02)

    def test_axle_spectrum_tandem(self):
        res = compute_design_esals(
            axle_groups=[{"axle_config": "tandem", "load_kips": 34,
                          "daily_count": 100}],
            pavement_type="flexible", sn=5.0, pt=2.5,
            growth_rate_pct=0, design_period_yr=1,
            num_lanes_per_direction=1, directional_factor=1.0,
            lane_factor=1.0)
        # Table D.5: 34-kip tandem @ SN5 -> 1.09
        assert res.axle_breakdown[0]["lef"] == pytest.approx(1.09, abs=0.03)

    def test_requires_exactly_one_traffic_source(self):
        with pytest.raises(ValueError, match="exactly one"):
            compute_design_esals(growth_rate_pct=2, design_period_yr=20)
        with pytest.raises(ValueError, match="exactly one"):
            compute_design_esals(
                base_year_w18_two_way=1e5,
                vehicles=[{"daily_count": 1, "truck_factor": 1}])

    def test_lane_factor_default_two_lanes(self):
        res = compute_design_esals(
            base_year_w18_two_way=100000, growth_rate_pct=0,
            design_period_yr=1, num_lanes_per_direction=2)
        # 2 lanes/dir -> 80-100% -> midpoint 0.90
        assert res.lane_factor == pytest.approx(0.90, abs=0.001)

    def test_bad_growth_inputs(self):
        with pytest.raises(ValueError):
            growth_factor(-1, 20)
        with pytest.raises(ValueError):
            growth_factor(2, 0)


# ---------------------------------------------------------------------------
# Full-table LEF integration (requires geotech_references.aashto_1993.lef)
# ---------------------------------------------------------------------------

class TestAppendixDLEF:
    def test_lef_off_design_point(self):
        pytest.importorskip("geotech_references.aashto_1993.lef")
        res = compute_design_esals(
            axle_groups=[{"axle_config": "single", "load_kips": 18,
                          "daily_count": 100}],
            pavement_type="flexible", sn=3.0, pt=2.0,
            growth_rate_pct=0, design_period_yr=1,
            num_lanes_per_direction=1, directional_factor=1.0,
            lane_factor=1.0)
        assert res.lef_basis == "appendix_d_tables"
        # LEF of the 18-kip single axle is 1.0 at every SN/pt by definition.
        assert res.axle_breakdown[0]["lef"] == pytest.approx(1.0, abs=0.02)

    def test_triple_axle(self):
        pytest.importorskip("geotech_references.aashto_1993.lef")
        res = compute_design_esals(
            axle_groups=[{"axle_config": "triple", "load_kips": 54,
                          "daily_count": 50}],
            pavement_type="flexible", sn=5.0, pt=2.5,
            growth_rate_pct=0, design_period_yr=1,
            num_lanes_per_direction=1, directional_factor=1.0,
            lane_factor=1.0)
        assert res.axle_breakdown[0]["lef"] > 0


# ---------------------------------------------------------------------------
# Composite-k integration (requires geotech_references.aashto_1993.composite_k)
# ---------------------------------------------------------------------------

class TestCompositeK:
    def test_composite_k_design_runs(self):
        pytest.importorskip("geotech_references.aashto_1993.composite_k")
        params = {k: v for k, v in RIGID_EXAMPLE.items() if k != "k_pci"}
        spec = {"seasonal": [{"mr_psi": 6000}] * 12, "dsb_in": 6.0,
                "esb_psi": 20000, "ls": 1.0}
        res = design_rigid_pavement(composite_k=spec, **params)
        assert res.k_pci > 0
        assert res.k_basis["basis"] == "composite_section_3.2"
        assert res.d_provided_in >= res.d_required_in
        assert res.iterations >= 1


# ---------------------------------------------------------------------------
# Environmental serviceability loss (requires ...aashto_1993.environmental)
# Guide anchors: Fig G.4 example (t=15, theta=0.10, Ps=60%, VR=2 -> 0.3);
# Fig G.8 example (t=15, phi=5, PF=30%, dPSI_MAX=2.0 -> 0.47).
# ---------------------------------------------------------------------------

SWELL_G4 = {"vr_in": 2.0, "ps_pct": 60, "theta": 0.10}
FROST_G8 = {"phi_mm_day": 5, "pf_pct": 30, "delta_psi_max": 2.0}


class TestEnvironmentalLoss:
    def test_flexible_swelling_reduces_budget_and_grows_sn(self):
        pytest.importorskip("geotech_references.aashto_1993.environmental")
        base = design_flexible_pavement(
            layers=[PavementLayer("asphalt", a=0.44)], **FLEX_EXAMPLE)
        env = design_flexible_pavement(
            layers=[PavementLayer("asphalt", a=0.44)],
            swelling=SWELL_G4, design_period_yr=15, **FLEX_EXAMPLE)
        # Fig G.4 printed example: dPSI_sw ~ 0.3 at 15 yr.
        assert env.environmental["delta_psi_sw"] == pytest.approx(0.31,
                                                                  abs=0.02)
        assert env.environmental["delta_psi_traffic"] == pytest.approx(
            1.9 - env.environmental["delta_psi_sw"], abs=1e-6)
        # Less serviceability for traffic -> a larger required SN.
        assert env.sn_required > base.sn_required
        assert env.adequate

    def test_rigid_frost_heave(self):
        pytest.importorskip("geotech_references.aashto_1993.environmental")
        base = design_rigid_pavement(**RIGID_EXAMPLE)
        env = design_rigid_pavement(frost=FROST_G8, design_period_yr=15,
                                    **RIGID_EXAMPLE)
        # Fig G.8 printed example: dPSI_fh ~ 0.47 at 15 yr.
        assert env.environmental["delta_psi_fh"] == pytest.approx(0.47,
                                                                  abs=0.02)
        assert env.d_required_in > base.d_required_in
        assert env.adequate

    def test_combined_loss_and_overconsumption_error(self):
        pytest.importorskip("geotech_references.aashto_1993.environmental")
        both = design_flexible_pavement(
            layers=[PavementLayer("asphalt", a=0.44)],
            swelling=SWELL_G4, frost=FROST_G8, design_period_yr=15,
            **FLEX_EXAMPLE)
        assert both.environmental["delta_psi_total"] == pytest.approx(
            0.78, abs=0.03)
        # A severe case that eats the whole budget must error, not design.
        with pytest.raises(ValueError, match="consumes"):
            design_flexible_pavement(
                layers=[PavementLayer("asphalt", a=0.44)],
                swelling={"vr_in": 10.0, "ps_pct": 100, "theta": 0.20},
                frost=FROST_G8, design_period_yr=30, **FLEX_EXAMPLE)

    def test_env_requires_design_period(self):
        pytest.importorskip("geotech_references.aashto_1993.environmental")
        with pytest.raises(ValueError, match="design_period_yr"):
            design_flexible_pavement(
                layers=[PavementLayer("asphalt", a=0.44)],
                swelling=SWELL_G4, **FLEX_EXAMPLE)

    def test_no_env_leaves_result_none(self):
        res = design_flexible_pavement(
            layers=[PavementLayer("asphalt", a=0.44)], **FLEX_EXAMPLE)
        assert res.environmental is None


class TestUfcAlternative:
    """UFC 3-250-01 alt method (refs digitization is Appendix-G-anchored;
    these tests exercise the module-side orchestration)."""

    def test_flexible_ufc_section(self):
        from pavement_design import design_flexible_pavement_ufc
        r = design_flexible_pavement_ufc(passes_18kip=1e6, cbr_subgrade=6,
                                         cbr_subbase=30)
        sec = r["section"]
        assert sec["provided_total_thickness_in"] >= \
            sec["required_total_thickness_in"] - 0.51
        names = [l["layer"] for l in sec["layers"]]
        assert names == ["asphalt_surface", "base", "subbase"]
        # Table 7-2 minimums respected.
        assert sec["layers"][0]["thickness_in"] >= 3.0
        assert sec["layers"][1]["thickness_in"] >= 4.0
        assert "Figure E-1" in " ".join(r["references"])

    def test_flexible_ufc_two_layer(self):
        from pavement_design import design_flexible_pavement_ufc
        r = design_flexible_pavement_ufc(passes_18kip=1e5, cbr_subgrade=10)
        assert len(r["section"]["layers"]) == 2

    def test_frost_governs_for_f3_subgrade(self):
        from pavement_design import design_flexible_pavement_ufc
        r = design_flexible_pavement_ufc(passes_18kip=1e6, cbr_subgrade=6,
                                         cbr_subbase=30,
                                         frost={"uscs_class": "CL"})
        assert r["frost_group"] == "F3"
        # FSI 3.5 < CBR 6 -> thicker frost section governs.
        assert r["frost_governs"] is True
        assert (r["frost_section"]["provided_total_thickness_in"]
                > r["section"]["provided_total_thickness_in"])

    def test_high_cbr_off_curve_handled(self):
        from pavement_design import design_flexible_pavement_ufc
        # CBR 80 base is off the E-1 curve at 1e6 passes -> minimums govern,
        # no exception.
        r = design_flexible_pavement_ufc(passes_18kip=1e6, cbr_subgrade=6)
        assert any("off the high-CBR end" in n for n in r["notes"])

    def test_rigid_ufc_guard_or_runs(self):
        from pavement_design import design_rigid_pavement_ufc
        try:
            r = design_rigid_pavement_ufc(passes_18kip=1e6,
                                          flexural_strength_psi=650,
                                          k_pci=200)
        except NotImplementedError as exc:
            assert "F-1" in str(exc)
        else:
            assert r["slab_provided_in"] >= r["hd_required_in"] - 0.51

    def test_rigid_ufc_appendix_g_anchors(self):
        """The F-1 solve reproduces both Appendix G worked examples
        through the orchestrator (G-7: 8.1 in; G-8.11: 6.3 in)."""
        pytest.importorskip("geotech_references.ufc_pavement")
        from geotech_references.ufc_pavement import tables as utb
        if not hasattr(utb, "figure_f1_rigid_thickness"):
            pytest.skip("F-1 not digitized in this refs install")
        from pavement_design import design_rigid_pavement_ufc
        g7 = design_rigid_pavement_ufc(passes_18kip=20000000,
                                       flexural_strength_psi=650, k_pci=100)
        assert g7["hd_required_in"] == pytest.approx(8.1, abs=0.05)
        g8 = design_rigid_pavement_ufc(passes_18kip=1200000,
                                       flexural_strength_psi=650, k_pci=325)
        assert g8["hd_required_in"] == pytest.approx(6.3, abs=0.05)

    def test_rigid_ufc_stabilized_foundation(self):
        pytest.importorskip("geotech_references.ufc_pavement")
        from geotech_references.ufc_pavement import tables as utb
        if not hasattr(utb, "figure_f1_rigid_thickness"):
            pytest.skip("F-1 not digitized in this refs install")
        from pavement_design import design_rigid_pavement_ufc
        base = design_rigid_pavement_ufc(passes_18kip=5e6,
                                         flexural_strength_psi=650,
                                         k_pci=200)
        stab = design_rigid_pavement_ufc(
            passes_18kip=5e6, flexural_strength_psi=650, k_pci=200,
            stabilized_foundation={"ef_psi": 650000, "hs_in": 6.0})
        # Eq 13-1 reduces the slab on a stabilized foundation.
        assert stab["ho_on_stabilized_in"] < base["hd_required_in"]
        assert stab["slab_provided_in"] <= base["slab_provided_in"]

    def test_compare_methods(self):
        from pavement_design import compare_flexible_pavement_methods
        c = compare_flexible_pavement_methods(passes_18kip=1e6,
                                              cbr_subgrade=6)
        assert c["aashto_1993"]["total_thickness_in"] > 0
        assert c["ufc_3_250_01"]["total_thickness_in"] > 0
        assert c["delta_total_thickness_in"] == pytest.approx(
            c["ufc_3_250_01"]["total_thickness_in"]
            - c["aashto_1993"]["total_thickness_in"], abs=0.02)
        joined = " ".join(c["notes"])
        assert "damage models" in joined      # basis caveat stated
        assert "2555" in joined               # CBR->Mr correlation stated
        assert "deterministic" in joined      # reliability asymmetry stated

    def test_compare_requires_one_subgrade_basis(self):
        from pavement_design import compare_flexible_pavement_methods
        with pytest.raises(ValueError, match="exactly one"):
            compare_flexible_pavement_methods(passes_18kip=1e6)
        with pytest.raises(ValueError, match="exactly one"):
            compare_flexible_pavement_methods(passes_18kip=1e6,
                                              cbr_subgrade=6, mr_psi=9000)


class TestFigures:
    def test_flexible_figures(self):
        pytest.importorskip("matplotlib")
        from pavement_design.calc_steps import get_figures
        res = design_flexible_pavement(
            layers=THREE_LAYERS, monthly_mr_psi=MONTHLY_MR,
            swelling=SWELL_G4, design_period_yr=15,
            **{k: v for k, v in FLEX_EXAMPLE.items() if k != "mr_psi"})
        figs = get_figures(res)
        titles = [f.title for f in figs]
        assert "Flexible design chart" in titles
        assert "Designed section" in titles
        assert "Effective roadbed MR" in titles
        assert "Environmental serviceability loss" in titles
        assert all(len(f.image_base64) > 5000 for f in figs)

    def test_rigid_figures(self):
        pytest.importorskip("matplotlib")
        from pavement_design.calc_steps import get_figures
        res = design_rigid_pavement(**RIGID_EXAMPLE)
        figs = get_figures(res)
        assert [f.title for f in figs] == ["Rigid design chart"]

    def test_no_optional_figures_without_inputs(self):
        pytest.importorskip("matplotlib")
        from pavement_design.calc_steps import get_figures
        res = design_flexible_pavement(
            layers=[PavementLayer("asphalt", a=0.44)], **FLEX_EXAMPLE)
        titles = [f.title for f in get_figures(res)]
        assert "Effective roadbed MR" not in titles
        assert "Environmental serviceability loss" not in titles


class TestPerformancePeriod:
    def test_flexible_iteration_converges(self):
        pytest.importorskip("geotech_references.aashto_1993.environmental")
        from pavement_design import estimate_performance_period
        res = estimate_performance_period(
            "flexible", delta_psi_design=1.9, base_year_w18=300000,
            growth_rate_pct=2.0, swelling=SWELL_G4, frost=FROST_G8,
            max_performance_period_yr=15, sn=5.0, mr_psi=5000,
            zr=-1.645, so=0.35)
        assert res["converged"]
        assert 0 < res["performance_period_yr"] <= 15
        assert len(res["rows"]) >= 2
        # Environmental loss must shorten the period vs the no-env horizon.
        # (With SN=5 the section carries ~5e6 ESALs on the full budget.)

    def test_rigid_iteration_runs(self):
        pytest.importorskip("geotech_references.aashto_1993.environmental")
        from pavement_design import estimate_performance_period
        res = estimate_performance_period(
            "rigid", delta_psi_design=1.7, base_year_w18=400000,
            growth_rate_pct=0.0, swelling=SWELL_G4,
            max_performance_period_yr=20, d_in=10.0, sc_psi=650,
            ec_psi=5e6, j=3.2, cd=1.0, k_pci=72, zr=-1.645, so=0.29)
        assert res["converged"]
        assert res["performance_period_yr"] > 0

    def test_requires_env_and_structure(self):
        pytest.importorskip("geotech_references.aashto_1993.environmental")
        from pavement_design import estimate_performance_period
        with pytest.raises(ValueError, match="swelling"):
            estimate_performance_period(
                "flexible", delta_psi_design=1.9, base_year_w18=1e5,
                sn=5.0, mr_psi=5000, zr=-1.645, so=0.35)
        with pytest.raises(ValueError, match="sn"):
            estimate_performance_period(
                "flexible", delta_psi_design=1.9, base_year_w18=1e5,
                swelling=SWELL_G4, mr_psi=None, sn=None, zr=-1.645, so=0.35)

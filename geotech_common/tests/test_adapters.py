"""
Tests for SoilProfile adapter methods (to_*_input).

Test classes:
    TestBearingCapacityAdapter   - to_bearing_capacity_input
    TestSettlementAdapter        - to_settlement_input
    TestAxialPileAdapter         - to_axial_pile_input
    TestLateralPileAdapter       - to_lateral_pile_input
    TestSheetPileAdapter         - to_sheet_pile_input
    TestPileGroupAdapter         - to_pile_group_input
    TestAdapterIntegration       - Round-trip: adapters produce valid module input
"""

import pytest

from geotech_common.soil_profile import (
    SoilLayer, GroundwaterCondition, SoilProfile, SoilProfileBuilder,
    _estimate_k_sand, _estimate_k_stiff_clay,
)


# ── Fixtures ──────────────────────────────────────────────────────────

def _mixed_profile():
    """3-layer profile: fill / soft clay / dense sand, GWT at 2m."""
    layers = [
        SoilLayer(0, 3, "Sandy fill (SM)", gamma=18.0, gamma_sat=19.5,
                  phi=30, uscs="SM", is_cohesive=False),
        SoilLayer(3, 10, "Soft gray clay (CH)", gamma=16.5, gamma_sat=17.0,
                  cu=30, uscs="CH", e0=1.1, Cc=0.4, Cr=0.06,
                  sigma_p=60, eps50=0.015),
        SoilLayer(10, 20, "Dense sand (SP)", gamma=19.5, gamma_sat=20.5,
                  phi=36, N_spt=35, uscs="SP", is_cohesive=False),
    ]
    gw = GroundwaterCondition(depth=2.0)
    return SoilProfile(layers=layers, groundwater=gw)


def _clay_only_profile():
    """Single thick clay layer."""
    layers = [
        SoilLayer(0, 15, "Stiff clay (CL)", gamma=19.0, gamma_sat=20.0,
                  cu=120, uscs="CL", e0=0.7, Cc=0.2, Cr=0.03,
                  sigma_p=200, eps50=0.005, Es=50000),
    ]
    gw = GroundwaterCondition(depth=3.0)
    return SoilProfile(layers=layers, groundwater=gw)


def _sand_only_profile():
    """Single thick sand layer."""
    layers = [
        SoilLayer(0, 20, "Medium dense sand (SP)", gamma=18.5, gamma_sat=20.0,
                  phi=33, N_spt=22, uscs="SP", Es=18000),
    ]
    gw = GroundwaterCondition(depth=5.0)
    return SoilProfile(layers=layers, groundwater=gw)


def _profile_with_rock():
    """Profile with rock at bottom."""
    layers = [
        SoilLayer(0, 5, "Sand", gamma=18.0, phi=30, uscs="SP"),
        SoilLayer(5, 10, "Weathered rock", gamma=23.0, is_rock=True,
                  qu=5000, RQD=60),
    ]
    gw = GroundwaterCondition(depth=3.0)
    return SoilProfile(layers=layers, groundwater=gw)


# ── TestBearingCapacityAdapter ────────────────────────────────────────

class TestBearingCapacityAdapter:

    def test_clay_at_footing_depth(self):
        profile = _mixed_profile()
        result = profile.to_bearing_capacity_input(footing_depth=4.0)
        # Footing at 4m is in soft clay
        assert result["layer1"]["cohesion"] == 30  # cu
        assert result["layer1"]["friction_angle"] == 0.0  # phi=0 for total stress
        assert result["gwt_depth"] == 2.0

    def test_sand_at_footing_depth(self):
        profile = _mixed_profile()
        result = profile.to_bearing_capacity_input(footing_depth=1.0)
        # Footing at 1m is in sandy fill
        assert result["layer1"]["friction_angle"] == 30
        assert result["layer1"]["cohesion"] == 0.0

    def test_two_layer_detection(self):
        profile = _mixed_profile()
        result = profile.to_bearing_capacity_input(footing_depth=1.0)
        # Fill at 1m, clay starts at 3m -> should get layer2
        assert result["layer2"] is not None
        assert result["layer2"]["cohesion"] == 30  # soft clay cu
        # Layer1 thickness should be distance from footing to layer boundary
        assert abs(result["layer1"]["thickness"] - 2.0) < 0.01  # 3-1=2m

    def test_single_layer_no_layer2(self):
        profile = _clay_only_profile()
        result = profile.to_bearing_capacity_input(footing_depth=2.0)
        assert result["layer2"] is None

    def test_unit_weight_present(self):
        profile = _mixed_profile()
        result = profile.to_bearing_capacity_input(footing_depth=5.0)
        assert result["layer1"]["unit_weight"] == 16.5

    def test_missing_strength_raises(self):
        layers = [SoilLayer(0, 10, "Unknown", gamma=18.0)]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        with pytest.raises(ValueError, match="neither cu nor phi"):
            profile.to_bearing_capacity_input(footing_depth=2.0)


# ── TestSettlementAdapter ─────────────────────────────────────────────

class TestSettlementAdapter:

    def test_overburden_pressure(self):
        profile = _mixed_profile()
        result = profile.to_settlement_input(footing_depth=3.0,
                                             footing_width=2.0)
        # Overburden at 3m: 18*2 (fill above GWT) + 19.5*1 (fill below GWT) = 55.5
        assert result["q_overburden"] > 0

    def test_consolidation_layers_from_clay(self):
        profile = _mixed_profile()
        result = profile.to_settlement_input(footing_depth=3.0,
                                             footing_width=2.0)
        # Influence depth = 3 + 2*2 = 7m, clay is 3-10m
        # So clay from 3-7m contributes consolidation layers
        assert len(result["consolidation_layers"]) >= 1
        cl = result["consolidation_layers"][0]
        assert cl["Cc"] == 0.4
        assert cl["Cr"] == 0.06
        assert cl["e0"] == 1.1
        assert cl["sigma_p"] == 60

    def test_no_consolidation_layers_in_sand(self):
        profile = _sand_only_profile()
        result = profile.to_settlement_input(footing_depth=1.0,
                                             footing_width=2.0)
        assert len(result["consolidation_layers"]) == 0

    def test_Es_immediate_from_clay(self):
        profile = _clay_only_profile()
        result = profile.to_settlement_input(footing_depth=1.0,
                                             footing_width=2.0)
        assert result["Es_immediate"] == 50000

    def test_Es_immediate_from_sand(self):
        profile = _sand_only_profile()
        result = profile.to_settlement_input(footing_depth=1.0,
                                             footing_width=2.0)
        assert result["Es_immediate"] == 18000

    def test_square_footing_default(self):
        profile = _clay_only_profile()
        result = profile.to_settlement_input(footing_depth=1.0,
                                             footing_width=2.0)
        assert result["B"] == 2.0
        assert result["L"] == 2.0  # Default L=B

    def test_rectangular_footing(self):
        profile = _clay_only_profile()
        result = profile.to_settlement_input(footing_depth=1.0,
                                             footing_width=2.0,
                                             footing_length=4.0)
        assert result["L"] == 4.0

    def test_depth_to_center_relative_to_footing(self):
        profile = _mixed_profile()
        result = profile.to_settlement_input(footing_depth=3.0,
                                             footing_width=2.0)
        # Clay 3-5m (clipped to influence depth 7m), center = 5m, depth_to_center = 5-3 = 2m
        if result["consolidation_layers"]:
            cl = result["consolidation_layers"][0]
            assert cl["depth_to_center"] > 0


# ── TestAxialPileAdapter ──────────────────────────────────────────────

class TestAxialPileAdapter:

    def test_mixed_profile(self):
        profile = _mixed_profile()
        result = profile.to_axial_pile_input(pile_length=15.0)
        layers = result["layers"]
        assert len(layers) == 3
        # Fill: cohesionless
        assert layers[0]["soil_type"] == "cohesionless"
        assert layers[0]["friction_angle"] == 30
        # Clay: cohesive
        assert layers[1]["soil_type"] == "cohesive"
        assert layers[1]["cohesion"] == 30
        # Sand: cohesionless
        assert layers[2]["soil_type"] == "cohesionless"
        assert layers[2]["friction_angle"] == 36

    def test_pile_clips_to_length(self):
        profile = _mixed_profile()
        result = profile.to_axial_pile_input(pile_length=5.0)
        layers = result["layers"]
        # Fill 0-3 (3m) + Clay 3-5 (2m) = 5m total
        total = sum(l["thickness"] for l in layers)
        assert abs(total - 5.0) < 0.01

    def test_gwt_passed_through(self):
        profile = _mixed_profile()
        result = profile.to_axial_pile_input(pile_length=10.0)
        assert result["gwt_depth"] == 2.0

    def test_missing_strength_raises(self):
        layers = [SoilLayer(0, 10, "Unknown", gamma=18.0)]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        with pytest.raises(ValueError, match="neither cu nor phi"):
            profile.to_axial_pile_input(pile_length=5.0)

    def test_layer_thicknesses_correct(self):
        profile = _mixed_profile()
        result = profile.to_axial_pile_input(pile_length=12.0)
        layers = result["layers"]
        assert abs(layers[0]["thickness"] - 3.0) < 0.01  # Fill: 0-3
        assert abs(layers[1]["thickness"] - 7.0) < 0.01  # Clay: 3-10
        assert abs(layers[2]["thickness"] - 2.0) < 0.01  # Sand: 10-12


# ── TestLateralPileAdapter ────────────────────────────────────────────

class TestLateralPileAdapter:

    def test_soft_clay_gets_matlock(self):
        profile = _mixed_profile()
        result = profile.to_lateral_pile_input(pile_length=8.0,
                                               pile_diameter=0.6)
        layers = result["layers"]
        # Clay layer (3-8m) should get SoftClayMatlock (cu=30 < 100)
        clay_layer = [l for l in layers if "clay" in l["description"].lower()][0]
        assert clay_layer["py_model_type"] == "SoftClayMatlock"
        assert clay_layer["py_model_params"]["c"] == 30
        assert clay_layer["py_model_params"]["eps50"] == 0.015

    def test_sand_gets_api(self):
        profile = _sand_only_profile()
        result = profile.to_lateral_pile_input(pile_length=15.0,
                                               pile_diameter=0.6)
        layers = result["layers"]
        assert layers[0]["py_model_type"] == "SandAPI"
        assert layers[0]["py_model_params"]["phi"] == 33

    def test_stiff_clay_below_gwt(self):
        profile = _clay_only_profile()
        result = profile.to_lateral_pile_input(pile_length=10.0,
                                               pile_diameter=0.6)
        layers = result["layers"]
        # cu=120 >= 100, GWT=3m, layer starts at 0
        # Part above GWT should get StiffClayAboveWT? No — single layer, midpoint checked
        # Actually the whole layer 0-10, mid = 5 > gwt=3 -> below GWT
        assert layers[0]["py_model_type"] == "StiffClayBelowWT"

    def test_rock_gets_weakrock(self):
        profile = _profile_with_rock()
        result = profile.to_lateral_pile_input(pile_length=8.0,
                                               pile_diameter=0.6)
        layers = result["layers"]
        rock_layer = [l for l in layers if "rock" in l["description"].lower()][0]
        assert rock_layer["py_model_type"] == "WeakRock"
        assert rock_layer["py_model_params"]["qu"] == 5000

    def test_cyclic_loading(self):
        profile = _sand_only_profile()
        result = profile.to_lateral_pile_input(pile_length=10.0,
                                               pile_diameter=0.6,
                                               loading="cyclic")
        assert result["layers"][0]["py_model_params"]["loading"] == "cyclic"

    def test_clips_to_pile_length(self):
        profile = _mixed_profile()
        result = profile.to_lateral_pile_input(pile_length=5.0,
                                               pile_diameter=0.6)
        layers = result["layers"]
        total_depth = sum(l["bottom"] - l["top"] for l in layers)
        assert abs(total_depth - 5.0) < 0.01

    def test_effective_gamma_below_gwt(self):
        profile = _mixed_profile()
        result = profile.to_lateral_pile_input(pile_length=8.0,
                                               pile_diameter=0.6)
        # Clay layer: mid at (3+8)/2 = 5.5m, below GWT=2m
        # gamma_eff = gamma_sat - gamma_w = 17.0 - 9.81 = 7.19
        clay_layer = [l for l in result["layers"]
                      if "clay" in l["description"].lower()][0]
        gamma = clay_layer["py_model_params"]["gamma"]
        assert abs(gamma - (17.0 - 9.81)) < 0.1

    def test_k_estimation_for_sand(self):
        """k should be estimated from phi when not provided."""
        profile = _sand_only_profile()
        result = profile.to_lateral_pile_input(pile_length=10.0,
                                               pile_diameter=0.6)
        k = result["layers"][0]["py_model_params"]["k"]
        assert k > 0


# ── TestSheetPileAdapter ──────────────────────────────────────────────

class TestSheetPileAdapter:

    def test_basic_output(self):
        profile = _mixed_profile()
        result = profile.to_sheet_pile_input(excavation_depth=4.0)
        assert result["excavation_depth"] == 4.0
        layers = result["layers"]
        assert len(layers) == 3

    def test_sand_layer_params(self):
        profile = _mixed_profile()
        result = profile.to_sheet_pile_input(excavation_depth=3.0)
        fill = result["layers"][0]
        assert fill["friction_angle"] == 30
        assert fill["cohesion"] == 0.0
        assert fill["unit_weight"] == 18.0

    def test_clay_layer_params(self):
        profile = _mixed_profile()
        result = profile.to_sheet_pile_input(excavation_depth=5.0)
        clay = result["layers"][1]
        assert clay["cohesion"] == 30  # cu
        assert clay["unit_weight"] == 16.5

    def test_thicknesses_preserved(self):
        profile = _mixed_profile()
        result = profile.to_sheet_pile_input(excavation_depth=4.0)
        thicknesses = [l["thickness"] for l in result["layers"]]
        assert abs(thicknesses[0] - 3.0) < 0.01
        assert abs(thicknesses[1] - 7.0) < 0.01
        assert abs(thicknesses[2] - 10.0) < 0.01

    def test_missing_strength_raises(self):
        layers = [SoilLayer(0, 10, "Unknown", gamma=18.0)]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        with pytest.raises(ValueError, match="neither phi nor cu"):
            profile.to_sheet_pile_input(excavation_depth=3.0)


# ── TestPileGroupAdapter ─────────────────────────────────────────────

class TestPileGroupAdapter:

    def test_mixed_profile(self):
        profile = _mixed_profile()
        result = profile.to_pile_group_input(pile_length=15.0,
                                             pile_diameter=0.3)
        assert result["pile_length"] == 15.0
        assert result["pile_diameter"] == 0.3
        assert result["gwt_depth"] == 2.0
        assert result["average_phi"] > 0
        assert result["average_cu"] > 0

    def test_cohesive_dominant(self):
        profile = _clay_only_profile()
        result = profile.to_pile_group_input(pile_length=10.0,
                                             pile_diameter=0.6)
        assert result["is_cohesive"] is True
        assert result["average_cu"] == 120

    def test_granular_dominant(self):
        profile = _sand_only_profile()
        result = profile.to_pile_group_input(pile_length=15.0,
                                             pile_diameter=0.3)
        assert result["is_cohesive"] is False
        assert result["average_phi"] == 33

    def test_weighted_average(self):
        profile = _mixed_profile()
        result = profile.to_pile_group_input(pile_length=13.0,
                                             pile_diameter=0.3)
        # Fill: 3m @ phi=30, Sand: 3m @ phi=36 (10-13m)
        # avg_phi = (3*30 + 3*36) / 6 = 33
        assert result["average_phi"] == 33.0


# ── TestAdapterIntegration ───────────────────────────────────────────

class TestAdapterIntegration:

    def test_bearing_creates_valid_module_input(self):
        """Adapter output should create valid bearing_capacity objects."""
        from bearing_capacity.soil_profile import (
            SoilLayer as BCSoilLayer, BearingSoilProfile,
        )
        profile = _mixed_profile()
        data = profile.to_bearing_capacity_input(footing_depth=4.0)

        layer1 = BCSoilLayer(**data["layer1"])
        assert layer1.cohesion == 30
        assert layer1.friction_angle == 0.0

        soil = BearingSoilProfile(
            layer1=layer1,
            gwt_depth=data["gwt_depth"],
        )
        assert soil is not None

    def test_bearing_two_layer_valid(self):
        from bearing_capacity.soil_profile import (
            SoilLayer as BCSoilLayer, BearingSoilProfile,
        )
        profile = _mixed_profile()
        data = profile.to_bearing_capacity_input(footing_depth=1.0)

        layer1 = BCSoilLayer(**data["layer1"])
        layer2 = BCSoilLayer(**data["layer2"])
        soil = BearingSoilProfile(
            layer1=layer1,
            layer2=layer2,
            gwt_depth=data["gwt_depth"],
        )
        assert soil.is_two_layer

    def test_axial_creates_valid_module_input(self):
        from axial_pile.soil_profile import AxialSoilLayer, AxialSoilProfile
        profile = _mixed_profile()
        data = profile.to_axial_pile_input(pile_length=15.0)

        layers = [AxialSoilLayer(**lyr) for lyr in data["layers"]]
        soil = AxialSoilProfile(layers=layers, gwt_depth=data["gwt_depth"])
        assert len(soil.layers) == 3

    def test_settlement_consolidation_layers_valid(self):
        from settlement.consolidation import ConsolidationLayer
        profile = _mixed_profile()
        data = profile.to_settlement_input(footing_depth=3.0,
                                           footing_width=2.0)
        for cl_data in data["consolidation_layers"]:
            cl = ConsolidationLayer(**cl_data)
            assert cl.thickness > 0
            assert cl.Cc > 0

    def test_sheet_pile_creates_valid_module_input(self):
        from sheet_pile.cantilever import WallSoilLayer
        profile = _mixed_profile()
        data = profile.to_sheet_pile_input(excavation_depth=4.0)

        layers = [WallSoilLayer(**lyr) for lyr in data["layers"]]
        assert len(layers) == 3


# ── TestHelperFunctions ───────────────────────────────────────────────

class TestHelperFunctions:

    def test_k_sand_below_gwt(self):
        k = _estimate_k_sand(30, below_gwt=True)
        assert k == 11000.0

    def test_k_sand_above_gwt(self):
        k = _estimate_k_sand(30, below_gwt=False)
        assert k == 16300.0

    def test_k_sand_high_phi(self):
        k = _estimate_k_sand(40, below_gwt=True)
        assert k == 50000.0

    def test_k_stiff_clay_moderate(self):
        k = _estimate_k_stiff_clay(150)
        assert k == 135_000.0

    def test_k_stiff_clay_very_stiff(self):
        k = _estimate_k_stiff_clay(500)
        assert k == 540_000.0


# ── TestDrilledShaftAdapter ─────────────────────────────────────────

class TestDrilledShaftAdapter:

    def test_mixed_profile_layers(self):
        profile = _mixed_profile()
        data = profile.to_drilled_shaft_input(shaft_length=15.0)
        layers = data["layers"]
        assert len(layers) == 3
        assert layers[0]["soil_type"] == "cohesionless"  # fill
        assert layers[1]["soil_type"] == "cohesive"      # clay
        assert layers[2]["soil_type"] == "cohesionless"  # sand

    def test_layer_clipping(self):
        profile = _mixed_profile()
        data = profile.to_drilled_shaft_input(shaft_length=6.0)
        layers = data["layers"]
        assert len(layers) == 2  # fill + part of clay
        assert layers[0]["thickness"] == 3.0
        assert layers[1]["thickness"] == 3.0  # 3-6m clipped

    def test_cohesive_cu_propagated(self):
        profile = _mixed_profile()
        data = profile.to_drilled_shaft_input(shaft_length=15.0)
        clay = data["layers"][1]
        assert clay["cu"] == 30.0
        assert clay["soil_type"] == "cohesive"

    def test_rock_layer(self):
        profile = _profile_with_rock()
        data = profile.to_drilled_shaft_input(shaft_length=10.0)
        layers = data["layers"]
        assert len(layers) == 2
        assert layers[1]["soil_type"] == "rock"
        assert layers[1]["qu"] == 5000
        assert layers[1]["RQD"] == 60

    def test_gwt_depth_passed(self):
        profile = _mixed_profile()
        data = profile.to_drilled_shaft_input(shaft_length=15.0)
        assert data["gwt_depth"] == 2.0

    def test_sand_has_phi(self):
        profile = _sand_only_profile()
        data = profile.to_drilled_shaft_input(shaft_length=10.0)
        assert data["layers"][0]["phi"] == 33
        assert data["layers"][0]["N60"] == 0.0  # N60 not set, only N_spt

    def test_clay_only(self):
        profile = _clay_only_profile()
        data = profile.to_drilled_shaft_input(shaft_length=10.0)
        assert len(data["layers"]) == 1
        assert data["layers"][0]["cu"] == 120


# ── TestRetainingWallAdapter ────────────────────────────────────────

class TestRetainingWallAdapter:

    def test_sand_backfill(self):
        profile = _sand_only_profile()
        data = profile.to_retaining_wall_input(wall_height=5.0)
        assert data["phi_backfill"] == 33.0
        assert data["c_backfill"] == 0.0
        assert data["gamma_backfill"] == 18.5

    def test_mixed_profile_weighted_average(self):
        profile = _mixed_profile()
        data = profile.to_retaining_wall_input(wall_height=10.0)
        # 3m of sand phi=30 + 7m of clay cu=30 (phi=0 for clay)
        # phi_backfill = (30*3 + 0*7) / 10 = 9.0
        assert abs(data["phi_backfill"] - 9.0) < 0.2
        # c_backfill = (0*3 + 30*7) / 10 = 21.0
        assert abs(data["c_backfill"] - 21.0) < 0.2

    def test_foundation_soil_from_layer_at_base(self):
        profile = _mixed_profile()
        data = profile.to_retaining_wall_input(wall_height=5.0)
        # Foundation is soft clay at 5m depth
        assert data["phi_foundation"] == 0.0
        assert data["c_foundation"] == 30.0  # cu

    def test_foundation_sand(self):
        profile = _mixed_profile()
        data = profile.to_retaining_wall_input(wall_height=12.0)
        # Foundation is dense sand layer
        assert data["phi_foundation"] == 36.0
        assert data["c_foundation"] == 0.0

    def test_surcharge_passed_through(self):
        profile = _sand_only_profile()
        data = profile.to_retaining_wall_input(wall_height=4.0, surcharge=10.0)
        assert data["surcharge"] == 10.0

    def test_clay_only_wall(self):
        profile = _clay_only_profile()
        data = profile.to_retaining_wall_input(wall_height=5.0)
        assert data["phi_backfill"] == 0.0
        assert data["c_backfill"] == 120.0
        assert data["gamma_backfill"] == 19.0


# ── TestSeismicAdapter ──────────────────────────────────────────────

class TestSeismicAdapter:

    def test_site_classification_n_values(self):
        profile = _sand_only_profile()
        # Sand has N_spt=22, N60 not set — adapter uses N_spt fallback
        data = profile.to_seismic_input()
        sc = data["site_classification"]
        assert len(sc["N_values"]) == 1
        assert sc["N_values"][0] == 22  # N_spt

    def test_site_classification_su_values(self):
        profile = _clay_only_profile()
        data = profile.to_seismic_input()
        sc = data["site_classification"]
        assert len(sc["su_values"]) == 1
        assert sc["su_values"][0] == 120  # cu

    def test_liquefaction_data_below_gwt(self):
        # Sand below GWT should appear in liquefaction data
        layers = [
            SoilLayer(0, 5, "Dry sand", gamma=18.0, phi=30,
                      is_cohesive=False, N_spt=15, N160=12, uscs="SP"),
            SoilLayer(5, 15, "Saturated sand", gamma=19.0, phi=33,
                      is_cohesive=False, N_spt=20, N160=18, uscs="SP"),
        ]
        gw = GroundwaterCondition(depth=5.0)
        profile = SoilProfile(layers=layers, groundwater=gw)
        data = profile.to_seismic_input(amax_g=0.3, magnitude=7.5)

        liq = data["liquefaction"]
        assert len(liq["depths"]) == 1  # only saturated sand
        assert liq["N160"][0] == 18.0
        assert liq["FC"][0] == 5.0  # SP → clean sand

    def test_cohesive_layers_excluded_from_liquefaction(self):
        profile = _clay_only_profile()
        data = profile.to_seismic_input(amax_g=0.2)
        assert len(data["liquefaction"]["depths"]) == 0

    def test_gwt_depth_and_params_passed(self):
        profile = _mixed_profile()
        data = profile.to_seismic_input(amax_g=0.25, magnitude=6.5)
        assert data["amax_g"] == 0.25
        assert data["magnitude"] == 6.5
        assert data["gwt_depth"] == 2.0

    def test_fines_content_from_uscs_sm(self):
        layers = [
            SoilLayer(0, 10, "Silty sand", gamma=18.0, phi=28,
                      is_cohesive=False, N_spt=12, N160=10, uscs="SM"),
        ]
        gw = GroundwaterCondition(depth=0.0)
        profile = SoilProfile(layers=layers, groundwater=gw)
        data = profile.to_seismic_input(amax_g=0.3)
        assert data["liquefaction"]["FC"][0] == 25.0  # SM → 25%

    def test_rock_excluded_from_both(self):
        profile = _profile_with_rock()
        data = profile.to_seismic_input()
        # Rock should not be in N-bar or liquefaction
        assert all(N > 0 for N in data["site_classification"]["N_values"])
        assert len(data["liquefaction"]["depths"]) == 0

    def test_mixed_profile_n_bar_and_liq(self):
        profile = _mixed_profile()
        data = profile.to_seismic_input(amax_g=0.2)
        sc = data["site_classification"]
        # Sand has N_spt=35 → in N_values; clay has cu → in su_values
        assert len(sc["N_values"]) >= 1
        assert len(sc["su_values"]) == 1
        assert sc["su_values"][0] == 30.0


# ── TestNewAdapterIntegration ───────────────────────────────────────

class TestNewAdapterIntegration:

    def test_drilled_shaft_creates_valid_module_input(self):
        from drilled_shaft.soil_profile import ShaftSoilLayer, ShaftSoilProfile
        profile = _mixed_profile()
        data = profile.to_drilled_shaft_input(shaft_length=15.0)

        layers = [ShaftSoilLayer(**lyr) for lyr in data["layers"]]
        soil = ShaftSoilProfile(layers=layers, gwt_depth=data["gwt_depth"])
        assert len(soil.layers) == 3
        assert soil.total_thickness == 15.0  # clipped from 20 to 15

    def test_drilled_shaft_rock_module_input(self):
        from drilled_shaft.soil_profile import ShaftSoilLayer, ShaftSoilProfile
        profile = _profile_with_rock()
        data = profile.to_drilled_shaft_input(shaft_length=10.0)

        layers = [ShaftSoilLayer(**lyr) for lyr in data["layers"]]
        soil = ShaftSoilProfile(layers=layers, gwt_depth=data["gwt_depth"])
        assert soil.layers[-1].soil_type == "rock"

    def test_retaining_wall_cantilever_integration(self):
        from retaining_walls.geometry import CantileverWallGeometry
        from retaining_walls.cantilever import analyze_cantilever_wall
        profile = _sand_only_profile()
        data = profile.to_retaining_wall_input(wall_height=4.0)

        geom = CantileverWallGeometry(wall_height=4.0, surcharge=data["surcharge"])
        result = analyze_cantilever_wall(
            geom,
            gamma_backfill=data["gamma_backfill"],
            phi_backfill=data["phi_backfill"],
            c_backfill=data["c_backfill"],
            phi_foundation=data["phi_foundation"],
            c_foundation=data["c_foundation"],
        )
        assert result.FOS_sliding > 0
        assert result.FOS_overturning > 0

    def test_seismic_site_classification_integration(self):
        from seismic_geotech.site_class import compute_n_bar, classify_site
        profile = _sand_only_profile()
        data = profile.to_seismic_input()
        sc = data["site_classification"]

        if sc["N_values"]:
            n_bar = compute_n_bar(sc["n_thicknesses"], sc["N_values"])
            site_class = classify_site(n_bar=n_bar)
            assert site_class in ("C", "D", "E")

    def test_seismic_liquefaction_integration(self):
        from seismic_geotech.liquefaction import evaluate_liquefaction
        layers = [
            SoilLayer(0, 3, "Dry sand", gamma=18.0, phi=30,
                      is_cohesive=False, N_spt=15, N160=12, uscs="SP"),
            SoilLayer(3, 15, "Loose saturated sand", gamma=19.0, phi=28,
                      is_cohesive=False, N_spt=10, N160=8, uscs="SP"),
        ]
        gw = GroundwaterCondition(depth=3.0)
        profile = SoilProfile(layers=layers, groundwater=gw)
        data = profile.to_seismic_input(amax_g=0.3, magnitude=7.5)

        liq = data["liquefaction"]
        if liq["depths"]:
            results = evaluate_liquefaction(
                layer_depths=liq["depths"],
                layer_N160=liq["N160"],
                layer_FC=liq["FC"],
                layer_gamma=liq["gamma"],
                amax_g=data["amax_g"],
                gwt_depth=data["gwt_depth"],
                M=data["magnitude"],
            )
            assert len(results) == 1
            assert "FOS_liq" in results[0]

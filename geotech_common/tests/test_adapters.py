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

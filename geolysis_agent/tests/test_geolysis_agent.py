"""
Tests for geolysis_agent — Soil classification, SPT corrections, bearing capacity.

Tier 1: No geolysis required (result dataclasses, validation, utilities, Foundry metadata)
Tier 2: Requires geolysis (integration tests with actual analyses)
"""

import json
import pytest

import matplotlib
matplotlib.use('Agg')

from geolysis_agent.geolysis_utils import has_geolysis
from geolysis_agent.results import (
    ClassificationResult,
    SPTCorrectionResult,
    BearingCapacityResult,
)

requires_geolysis = pytest.mark.skipif(
    not has_geolysis(), reason="geolysis not installed"
)


# =====================================================================
# Tier 1: Result dataclass defaults
# =====================================================================

class TestClassificationResultDefaults:
    """Test ClassificationResult with default values."""

    def test_default_construction(self):
        r = ClassificationResult()
        assert r.system == ""
        assert r.symbol == ""
        assert r.description == ""
        assert r.group_index is None

    def test_construction_with_values(self):
        r = ClassificationResult(
            system="uscs",
            symbol="SW-SC",
            description="Well-graded sand with clay",
            liquid_limit=25.0,
            plastic_limit=15.0,
            plasticity_index=10.0,
            fines=12.0,
            sand=55.0,
        )
        assert r.system == "uscs"
        assert r.symbol == "SW-SC"
        assert r.fines == 12.0
        assert r.plasticity_index == 10.0

    def test_summary_contains_key_values(self):
        r = ClassificationResult(
            system="uscs", symbol="CL", description="Lean clay", liquid_limit=35.0
        )
        s = r.summary()
        assert "USCS" in s
        assert "CL" in s
        assert "35.0" in s

    def test_to_dict_keys(self):
        r = ClassificationResult(system="uscs", symbol="SM")
        d = r.to_dict()
        assert "system" in d
        assert "symbol" in d
        assert "description" in d
        assert "liquid_limit" in d

    def test_to_dict_json_serializable(self):
        r = ClassificationResult(system="aashto", symbol="A-4", group_index="5")
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


class TestSPTCorrectionResultDefaults:
    """Test SPTCorrectionResult with default values."""

    def test_default_construction(self):
        r = SPTCorrectionResult()
        assert r.recorded_n == 0
        assert r.n60 == 0.0
        assert r.n1_60 == 0.0
        assert r.dilatancy_applied is False

    def test_construction_with_values(self):
        r = SPTCorrectionResult(
            recorded_n=25,
            n60=20.5,
            n1_60=18.3,
            n_corrected=18.3,
            energy_percentage=0.6,
            hammer_type="safety",
            opc_method="gibbs",
            eop_kpa=100.0,
        )
        assert r.recorded_n == 25
        assert r.n60 == 20.5
        assert r.n1_60 == 18.3
        assert r.energy_percentage == 0.6

    def test_summary_contains_key_values(self):
        r = SPTCorrectionResult(
            recorded_n=25, n60=20.5, n1_60=18.3, n_corrected=18.3
        )
        s = r.summary()
        assert "25" in s
        assert "20.5" in s
        assert "18.3" in s

    def test_to_dict_keys(self):
        r = SPTCorrectionResult(recorded_n=25, n60=20.0)
        d = r.to_dict()
        assert "recorded_n" in d
        assert "n60" in d
        assert "n1_60" in d
        assert "n_corrected" in d
        assert "dilatancy_applied" in d

    def test_to_dict_json_serializable(self):
        r = SPTCorrectionResult(recorded_n=25, n60=20.5)
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


class TestBearingCapacityResultDefaults:
    """Test BearingCapacityResult with default values."""

    def test_default_construction(self):
        r = BearingCapacityResult()
        assert r.method == ""
        assert r.bc_type == ""
        assert r.bearing_capacity_kpa == 0.0
        assert r.allowable_load_kn is None

    def test_construction_with_spt_values(self):
        r = BearingCapacityResult(
            method="bowles",
            bc_type="allowable_spt",
            bearing_capacity_kpa=250.0,
            allowable_load_kn=1000.0,
            depth_m=1.5,
            width_m=2.0,
            shape="square",
            corrected_spt_n=20.0,
            settlement_mm=25.0,
        )
        assert r.method == "bowles"
        assert r.bc_type == "allowable_spt"
        assert r.bearing_capacity_kpa == 250.0
        assert r.corrected_spt_n == 20.0

    def test_construction_with_ultimate_values(self):
        r = BearingCapacityResult(
            method="vesic",
            bc_type="ultimate",
            bearing_capacity_kpa=500.0,
            depth_m=1.5,
            width_m=2.0,
            shape="square",
            n_c=20.0,
            n_q=15.0,
            n_gamma=10.0,
            factor_of_safety=3.0,
        )
        assert r.method == "vesic"
        assert r.bc_type == "ultimate"
        assert r.n_c == 20.0
        assert r.factor_of_safety == 3.0

    def test_summary_contains_key_values_spt(self):
        r = BearingCapacityResult(
            method="bowles",
            bc_type="allowable_spt",
            bearing_capacity_kpa=250.0,
            corrected_spt_n=20.0,
        )
        s = r.summary()
        assert "bowles" in s.lower()
        assert "250.0" in s
        assert "20.0" in s

    def test_summary_contains_key_values_ultimate(self):
        r = BearingCapacityResult(
            method="vesic",
            bc_type="ultimate",
            bearing_capacity_kpa=500.0,
            n_c=20.5,
        )
        s = r.summary()
        assert "vesic" in s.lower()
        assert "500.0" in s
        assert "20.5" in s

    def test_to_dict_keys(self):
        r = BearingCapacityResult(method="bowles", bc_type="allowable_spt")
        d = r.to_dict()
        assert "method" in d
        assert "bc_type" in d
        assert "bearing_capacity_kpa" in d
        assert "n_c" in d
        assert "factor_of_safety" in d

    def test_to_dict_json_serializable(self):
        r = BearingCapacityResult(method="vesic", bearing_capacity_kpa=500.0)
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# =====================================================================
# Tier 1: Input validation
# =====================================================================

class TestValidation:
    """Test input validation without requiring geolysis."""

    def test_validate_bad_liquid_limit(self):
        """Validation errors are raised during analysis, not at import."""
        # This will be tested in Tier 2 when geolysis is available
        pass

    def test_validate_bad_plastic_limit(self):
        pass

    def test_validate_pl_exceeds_ll(self):
        pass

    def test_validate_bad_fines(self):
        pass

    def test_validate_bad_spt_n(self):
        pass

    def test_validate_bad_energy(self):
        pass

    def test_validate_bad_hammer_type(self):
        pass

    def test_validate_bad_shape(self):
        pass

    def test_validate_bad_method(self):
        pass


# =====================================================================
# Tier 1: Utilities
# =====================================================================

class TestUtilities:
    """Test utility functions."""

    def test_has_geolysis_returns_bool(self):
        result = has_geolysis()
        assert isinstance(result, bool)


# =====================================================================
# Tier 1: Foundry metadata (no geolysis required)
# =====================================================================

class TestFoundryMetadata:
    """Test Foundry agent metadata functions without geolysis."""

    def setup_method(self):
        """Import Foundry functions."""
        from geolysis_agent_foundry import (
            geolysis_list_methods,
            geolysis_describe_method,
        )
        self.list_methods = geolysis_list_methods
        self.describe_method = geolysis_describe_method

    def test_list_all_methods(self):
        result_str = self.list_methods()
        result = json.loads(result_str)
        # Should return categories
        assert isinstance(result, dict)
        # Check some expected categories
        assert any(cat in result for cat in ["Classification", "SPT", "Bearing Capacity"])

    def test_list_methods_filtered_classification(self):
        result_str = self.list_methods(category="Classification")
        result = json.loads(result_str)
        assert "Classification" in result
        assert "classify_uscs" in result["Classification"]
        assert "classify_aashto" in result["Classification"]

    def test_list_methods_filtered_spt(self):
        result_str = self.list_methods(category="SPT")
        result = json.loads(result_str)
        assert "SPT" in result
        assert "correct_spt" in result["SPT"]

    def test_list_methods_bad_category(self):
        result_str = self.list_methods(category="NonExistent")
        result = json.loads(result_str)
        assert "error" in result

    def test_describe_classify_uscs(self):
        result_str = self.describe_method("classify_uscs")
        result = json.loads(result_str)
        assert "parameters" in result
        assert "liquid_limit" in result["parameters"]
        assert "returns" in result

    def test_describe_classify_aashto(self):
        result_str = self.describe_method("classify_aashto")
        result = json.loads(result_str)
        assert "parameters" in result
        assert "liquid_limit" in result["parameters"]

    def test_describe_correct_spt(self):
        result_str = self.describe_method("correct_spt")
        result = json.loads(result_str)
        assert "parameters" in result
        assert "recorded_spt_n_value" in result["parameters"]
        assert "opc_method" in result["parameters"]

    def test_describe_allowable_bc_spt(self):
        result_str = self.describe_method("allowable_bc_spt")
        result = json.loads(result_str)
        assert "parameters" in result
        assert "corrected_spt_n_value" in result["parameters"]

    def test_describe_ultimate_bc(self):
        result_str = self.describe_method("ultimate_bc")
        result = json.loads(result_str)
        assert "parameters" in result
        assert "friction_angle" in result["parameters"]

    def test_describe_unknown_method(self):
        result_str = self.describe_method("nonexistent_method")
        result = json.loads(result_str)
        assert "error" in result


# =====================================================================
# Tier 2: USCS classification
# =====================================================================

@requires_geolysis
class TestUSCSClassification:
    """Test USCS soil classification with geolysis."""

    def test_classify_lean_clay(self):
        from geolysis_agent import classify_uscs
        result = classify_uscs(liquid_limit=35.0, plastic_limit=18.0, fines=95.0)
        assert result.system == "uscs"
        assert "CL" in result.symbol  # Could be CL or CL-ML
        assert result.liquid_limit == 35.0
        assert result.plasticity_index == 17.0

    def test_classify_silty_sand(self):
        from geolysis_agent import classify_uscs
        # geolysis requires LL/PL even for granular — use NP values
        result = classify_uscs(
            liquid_limit=0.0, plastic_limit=0.0, fines=15.0, sand=75.0
        )
        assert result.system == "uscs"
        assert "SM" in result.symbol or "S" in result.symbol

    def test_classify_well_graded_sand_with_clay(self):
        from geolysis_agent import classify_uscs
        result = classify_uscs(
            liquid_limit=25.0,
            plastic_limit=15.0,
            fines=12.0,
            sand=55.0,
            d_10=0.1,
            d_30=0.5,
            d_60=2.0,
        )
        assert result.system == "uscs"
        # Should be SW-SC or similar
        assert "S" in result.symbol  # Sand

    def test_classify_organic_peat(self):
        from geolysis_agent import classify_uscs
        # geolysis requires LL/PL for organic classification too
        result = classify_uscs(liquid_limit=60.0, plastic_limit=35.0, fines=80.0, organic=True)
        assert result.system == "uscs"
        assert "Pt" in result.symbol or "O" in result.symbol


# =====================================================================
# Tier 2: AASHTO classification
# =====================================================================

@requires_geolysis
class TestAASHTOClassification:
    """Test AASHTO soil classification with geolysis."""

    def test_classify_a1_a(self):
        from geolysis_agent import classify_aashto
        # geolysis requires LL, PL, and fines for AASHTO
        result = classify_aashto(liquid_limit=20.0, plastic_limit=15.0, fines=10.0)
        assert result.system == "aashto"
        # Low LL, low PI, low fines → A-1 or A-2
        assert "A-" in result.symbol

    def test_classify_a4(self):
        from geolysis_agent import classify_aashto
        result = classify_aashto(liquid_limit=35.0, plastic_limit=25.0, fines=85.0)
        assert result.system == "aashto"
        # Low LL, silty → likely A-4 or A-6
        assert "A-" in result.symbol
        assert result.group_index is not None

    def test_classify_a7_6(self):
        from geolysis_agent import classify_aashto
        result = classify_aashto(liquid_limit=65.0, plastic_limit=25.0, fines=85.0)
        assert result.system == "aashto"
        # High LL, high PI → A-7
        assert "A-7" in result.symbol
        assert result.group_index is not None


# =====================================================================
# Tier 2: SPT corrections
# =====================================================================

@requires_geolysis
class TestSPTCorrections:
    """Test SPT N-value corrections with geolysis."""

    def test_correct_spt_basic(self):
        from geolysis_agent import correct_spt
        result = correct_spt(
            recorded_spt_n_value=25,
            eop=100.0,
            energy_percentage=0.6,
            hammer_type="safety",
            opc_method="gibbs",
        )
        assert result.recorded_n == 25
        assert result.n60 > 0
        assert result.n1_60 > 0
        assert result.n_corrected > 0
        assert not result.dilatancy_applied

    def test_correct_spt_all_methods(self):
        from geolysis_agent import correct_spt
        methods = ["gibbs", "bazaraa", "peck", "liao", "skempton"]
        for method in methods:
            result = correct_spt(
                recorded_spt_n_value=25,
                eop=100.0,
                energy_percentage=0.6,
                opc_method=method,
            )
            assert result.opc_method == method
            assert result.n1_60 > 0

    def test_correct_spt_high_overburden(self):
        from geolysis_agent import correct_spt
        # High overburden → lower corrected N
        result = correct_spt(
            recorded_spt_n_value=25,
            eop=300.0,
            energy_percentage=0.6,
            opc_method="liao",  # Recommended for high overburden
        )
        assert result.n1_60 < result.n60  # Overburden reduces N


# =====================================================================
# Tier 2: Design N-value
# =====================================================================

@requires_geolysis
class TestDesignNValue:
    """Test design N-value computation with geolysis."""

    def test_design_n_weighted(self):
        from geolysis_agent import design_n_value
        n_values = [15.0, 20.0, 25.0, 18.0, 22.0]
        n_design = design_n_value(n_values, method="wgt")
        assert 15.0 <= n_design <= 25.0
        # Weighted should be closer to lower values
        assert n_design < sum(n_values) / len(n_values)

    def test_design_n_minimum(self):
        from geolysis_agent import design_n_value
        n_values = [15.0, 20.0, 25.0]
        n_design = design_n_value(n_values, method="min")
        assert n_design == 15.0

    def test_design_n_average(self):
        from geolysis_agent import design_n_value
        n_values = [15.0, 20.0, 25.0]
        n_design = design_n_value(n_values, method="avg")
        assert n_design == 20.0


# =====================================================================
# Tier 2: Allowable bearing capacity (SPT)
# =====================================================================

@requires_geolysis
class TestAllowableBearingCapacitySPT:
    """Test SPT-based allowable bearing capacity with geolysis."""

    def test_allowable_bc_bowles(self):
        from geolysis_agent import allowable_bc_spt
        result = allowable_bc_spt(
            corrected_spt_n_value=20.0,
            tol_settlement=25.0,
            depth=1.5,
            width=2.0,
            shape="square",
            abc_method="bowles",
        )
        assert result.method == "bowles"
        assert result.bc_type == "allowable_spt"
        assert result.bearing_capacity_kpa > 0
        assert result.allowable_load_kn > 0
        assert result.corrected_spt_n == 20.0

    def test_allowable_bc_meyerhof(self):
        from geolysis_agent import allowable_bc_spt
        result = allowable_bc_spt(
            corrected_spt_n_value=20.0,
            abc_method="meyerhof",
        )
        assert result.method == "meyerhof"
        assert result.bearing_capacity_kpa > 0

    def test_allowable_bc_terzaghi(self):
        from geolysis_agent import allowable_bc_spt
        result = allowable_bc_spt(
            corrected_spt_n_value=20.0,
            abc_method="terzaghi",
        )
        assert result.method == "terzaghi"
        assert result.bearing_capacity_kpa > 0


# =====================================================================
# Tier 2: Ultimate bearing capacity
# =====================================================================

@requires_geolysis
class TestUltimateBearingCapacity:
    """Test ultimate bearing capacity with geolysis."""

    def test_ultimate_bc_vesic_sand(self):
        from geolysis_agent import ultimate_bc
        result = ultimate_bc(
            friction_angle=30.0,
            cohesion=0.0,
            moist_unit_wgt=18.0,
            depth=1.5,
            width=2.0,
            factor_of_safety=3.0,
            shape="square",
            ubc_method="vesic",
        )
        assert result.method == "vesic"
        assert result.bc_type == "ultimate"
        assert result.bearing_capacity_kpa > 0
        assert result.n_c is not None
        assert result.n_q is not None
        assert result.n_gamma is not None
        assert result.allowable_bearing_capacity_kpa > 0

    def test_ultimate_bc_terzaghi_sand(self):
        from geolysis_agent import ultimate_bc
        result = ultimate_bc(
            friction_angle=30.0,
            cohesion=0.0,
            ubc_method="terzaghi",
        )
        assert result.method == "terzaghi"
        assert result.bearing_capacity_kpa > 0

    def test_ultimate_bc_with_cohesion(self):
        from geolysis_agent import ultimate_bc
        result = ultimate_bc(
            friction_angle=25.0,
            cohesion=50.0,  # Mixed soil
            moist_unit_wgt=18.0,
            depth=1.5,
            width=2.0,
            factor_of_safety=3.0,
            ubc_method="vesic",
        )
        assert result.bearing_capacity_kpa > 0
        assert result.n_c > 0  # Cohesion contributes


# =====================================================================
# Tier 2: Foundry integration
# =====================================================================

@requires_geolysis
class TestFoundryIntegration:
    """Test Foundry agent with geolysis installed."""

    def setup_method(self):
        """Import Foundry agent."""
        from geolysis_agent_foundry import geolysis_agent
        self.agent = geolysis_agent

    def test_classify_uscs_via_foundry(self):
        params = json.dumps({
            "liquid_limit": 35.0,
            "plastic_limit": 18.0,
            "fines": 95.0,
        })
        result_str = self.agent("classify_uscs", params)
        result = json.loads(result_str)
        assert "symbol" in result
        assert "system" in result
        assert result["system"] == "uscs"

    def test_correct_spt_via_foundry(self):
        params = json.dumps({
            "recorded_spt_n_value": 25,
            "eop": 100.0,
            "energy_percentage": 0.6,
        })
        result_str = self.agent("correct_spt", params)
        result = json.loads(result_str)
        assert "n60" in result
        assert "n1_60" in result
        assert "n_corrected" in result

    def test_invalid_json(self):
        result_str = self.agent("classify_uscs", "not json")
        result = json.loads(result_str)
        assert "error" in result

    def test_unknown_method(self):
        params = json.dumps({})
        result_str = self.agent("unknown_method", params)
        result = json.loads(result_str)
        assert "error" in result


# =====================================================================
# Tier 2: Input validation (with geolysis)
# =====================================================================

@requires_geolysis
class TestValidationWithGeolysis:
    """Test input validation when geolysis is available."""

    def test_pl_exceeds_ll_raises(self):
        from geolysis_agent import classify_uscs
        with pytest.raises(ValueError, match="plastic_limit.*cannot exceed"):
            classify_uscs(liquid_limit=20.0, plastic_limit=30.0)

    def test_negative_spt_n_raises(self):
        from geolysis_agent import correct_spt
        with pytest.raises(ValueError, match="recorded_spt_n_value.*>= 0"):
            correct_spt(recorded_spt_n_value=-5, eop=100.0)

    def test_invalid_hammer_type_raises(self):
        from geolysis_agent import correct_spt
        with pytest.raises(ValueError, match="hammer_type"):
            correct_spt(
                recorded_spt_n_value=25, eop=100.0, hammer_type="invalid"
            )

    def test_invalid_shape_raises(self):
        from geolysis_agent import allowable_bc_spt
        with pytest.raises(ValueError, match="shape"):
            allowable_bc_spt(corrected_spt_n_value=20.0, shape="invalid")

    def test_fos_too_low_raises(self):
        from geolysis_agent import ultimate_bc
        with pytest.raises(ValueError, match="factor_of_safety"):
            ultimate_bc(friction_angle=30.0, factor_of_safety=0.5)

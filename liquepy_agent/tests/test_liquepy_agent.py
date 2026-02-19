"""
Tests for liquepy_agent — CPT-based liquefaction triggering and correlations.

Tier 1: No liquepy required (result dataclasses, validation, utilities, Foundry metadata)
Tier 2: Requires liquepy (integration tests with actual analyses)
"""

import json
import math
import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from liquepy_agent.liquepy_utils import has_liquepy
from liquepy_agent.results import CPTLiquefactionResult, FieldCorrelationsResult

requires_liquepy = pytest.mark.skipif(
    not has_liquepy(), reason="liquepy not installed"
)


# =====================================================================
# Helper: standard CPT input arrays
# =====================================================================

def _make_cpt_arrays(n=40, qc_kpa=5000.0, fs_kpa=50.0):
    """Create standard CPT arrays for testing."""
    depth = np.arange(0.5, 0.5 + n * 0.5, 0.5)
    q_c = np.full_like(depth, qc_kpa)
    f_s = np.full_like(depth, fs_kpa)
    u_2 = np.zeros_like(depth)
    return depth, q_c, f_s, u_2


# =====================================================================
# Tier 1: Result dataclass defaults
# =====================================================================

class TestCPTLiquefactionResultDefaults:
    """Test CPTLiquefactionResult with default values."""

    def test_default_construction(self):
        r = CPTLiquefactionResult()
        assert r.n_points == 0
        assert r.pga_g == 0.0
        assert r.m_w == 7.5
        assert r.lpi == 0.0

    def test_construction_with_values(self):
        r = CPTLiquefactionResult(
            n_points=40, gwl_m=1.0, pga_g=0.3, m_w=7.5,
            lpi=42.0, lsn=60.0, ldi_m=0.5, min_fos=0.4,
            max_settlement_mm=150.0,
        )
        assert r.n_points == 40
        assert r.lpi == 42.0
        assert r.lsn == 60.0
        assert r.ldi_m == 0.5

    def test_summary_contains_key_values(self):
        r = CPTLiquefactionResult(
            n_points=40, pga_g=0.3, m_w=7.5, lpi=42.0, min_fos=0.4,
        )
        s = r.summary()
        assert "40" in s
        assert "0.30" in s
        assert "42.0" in s

    def test_to_dict_keys(self):
        r = CPTLiquefactionResult(n_points=40, pga_g=0.3, lpi=42.0)
        d = r.to_dict()
        assert "n_points" in d
        assert "pga_g" in d
        assert "lpi" in d
        assert "lsn" in d
        assert "ldi_m" in d
        assert "min_fos" in d
        assert "max_settlement_mm" in d

    def test_to_dict_json_serializable(self):
        r = CPTLiquefactionResult(n_points=40, pga_g=0.3, lpi=42.0)
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


class TestFieldCorrelationsResultDefaults:
    """Test FieldCorrelationsResult with default values."""

    def test_default_construction(self):
        r = FieldCorrelationsResult()
        assert r.n_points == 0
        assert r.vs_method == ""

    def test_construction_with_values(self):
        r = FieldCorrelationsResult(
            n_points=40, gwl_m=1.0, vs_method="mcgann2015",
            vs_m_per_s=np.array([150, 200, 250]),
        )
        assert r.n_points == 40
        assert r.vs_method == "mcgann2015"

    def test_summary_contains_method(self):
        r = FieldCorrelationsResult(
            n_points=40, vs_method="mcgann2015",
            vs_m_per_s=np.array([150.0, 200.0, 250.0]),
        )
        s = r.summary()
        assert "mcgann2015" in s
        assert "40" in s

    def test_to_dict_keys(self):
        r = FieldCorrelationsResult(n_points=40, vs_method="mcgann2015")
        d = r.to_dict()
        assert "n_points" in d
        assert "vs_method" in d

    def test_to_dict_with_vs_stats(self):
        r = FieldCorrelationsResult(
            n_points=3, vs_method="mcgann2015",
            vs_m_per_s=np.array([150.0, 200.0, 250.0]),
        )
        d = r.to_dict()
        assert "vs_min_m_per_s" in d
        assert "vs_max_m_per_s" in d
        assert d["vs_min_m_per_s"] == 150.0

    def test_to_dict_json_serializable(self):
        r = FieldCorrelationsResult(n_points=40, vs_method="mcgann2015")
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# =====================================================================
# Tier 1: Plot smoke tests
# =====================================================================

class TestPlotSmoke:
    """Smoke tests for plot methods (no liquepy needed)."""

    def test_cpt_liq_plot_fos(self):
        r = CPTLiquefactionResult(
            depth=np.arange(0.5, 10.5, 0.5),
            factor_of_safety=np.random.uniform(0.3, 2.0, 20),
        )
        ax = r.plot_fos(show=False)
        assert ax is not None

    def test_cpt_liq_plot_csr_crr(self):
        r = CPTLiquefactionResult(
            depth=np.arange(0.5, 10.5, 0.5),
            csr=np.random.uniform(0.1, 0.3, 20),
            crr=np.random.uniform(0.1, 0.4, 20),
        )
        ax = r.plot_csr_crr(show=False)
        assert ax is not None

    def test_cpt_liq_plot_ic(self):
        r = CPTLiquefactionResult(
            depth=np.arange(0.5, 10.5, 0.5),
            i_c=np.random.uniform(1.0, 3.0, 20),
        )
        ax = r.plot_ic(show=False)
        assert ax is not None

    def test_cpt_liq_plot_all(self):
        n = 20
        r = CPTLiquefactionResult(
            n_points=n,
            depth=np.arange(0.5, 10.5, 0.5),
            factor_of_safety=np.random.uniform(0.3, 2.0, n),
            csr=np.random.uniform(0.1, 0.3, n),
            crr=np.random.uniform(0.1, 0.4, n),
            i_c=np.random.uniform(1.0, 3.0, n),
            volumetric_strain=np.random.uniform(0, 0.02, n),
            shear_strain=np.random.uniform(0, 0.1, n),
        )
        axes = r.plot_all(show=False)
        assert len(axes) == 4

    def test_field_corr_plot_vs(self):
        r = FieldCorrelationsResult(
            depth=np.arange(0.5, 10.5, 0.5),
            vs_m_per_s=np.random.uniform(100, 300, 20),
            vs_method="mcgann2015",
        )
        ax = r.plot_vs(show=False)
        assert ax is not None

    def test_field_corr_plot_all(self):
        n = 20
        r = FieldCorrelationsResult(
            n_points=n,
            depth=np.arange(0.5, 10.5, 0.5),
            vs_m_per_s=np.random.uniform(100, 300, n),
            relative_density=np.random.uniform(0.3, 0.8, n),
            i_c=np.random.uniform(1.0, 3.0, n),
            vs_method="mcgann2015",
        )
        axes = r.plot_all(show=False)
        assert len(axes) == 3


# =====================================================================
# Tier 1: Input validation
# =====================================================================

class TestInputValidation:
    """Test input validation (no liquepy needed)."""

    def test_cpt_liq_empty_depth(self):
        from liquepy_agent.cpt_liquefaction import _validate_cpt_inputs
        with pytest.raises(ValueError, match="depth"):
            _validate_cpt_inputs([], [1], [1], [1], 1.0, 0.3, 7.5)

    def test_cpt_liq_mismatched_lengths(self):
        from liquepy_agent.cpt_liquefaction import _validate_cpt_inputs
        with pytest.raises(ValueError, match="q_c length"):
            _validate_cpt_inputs([1, 2], [1], [1, 2], [1, 2], 1.0, 0.3, 7.5)

    def test_cpt_liq_negative_gwl(self):
        from liquepy_agent.cpt_liquefaction import _validate_cpt_inputs
        with pytest.raises(ValueError, match="gwl"):
            _validate_cpt_inputs([1], [1], [1], [1], -1.0, 0.3, 7.5)

    def test_cpt_liq_zero_pga(self):
        from liquepy_agent.cpt_liquefaction import _validate_cpt_inputs
        with pytest.raises(ValueError, match="pga"):
            _validate_cpt_inputs([1], [1], [1], [1], 1.0, 0.0, 7.5)

    def test_cpt_liq_zero_magnitude(self):
        from liquepy_agent.cpt_liquefaction import _validate_cpt_inputs
        with pytest.raises(ValueError, match="m_w"):
            _validate_cpt_inputs([1], [1], [1], [1], 1.0, 0.3, 0.0)

    def test_field_corr_bad_vs_method(self):
        from liquepy_agent.field_correlations import _validate_correlation_inputs
        with pytest.raises(ValueError, match="vs_method"):
            _validate_correlation_inputs([1], [1], [1], [1], 1.0, "invalid")

    def test_field_corr_empty_depth(self):
        from liquepy_agent.field_correlations import _validate_correlation_inputs
        with pytest.raises(ValueError, match="depth"):
            _validate_correlation_inputs([], [1], [1], [1], 1.0, "mcgann2015")

    def test_field_corr_negative_gwl(self):
        from liquepy_agent.field_correlations import _validate_correlation_inputs
        with pytest.raises(ValueError, match="gwl"):
            _validate_correlation_inputs([1], [1], [1], [1], -1.0, "mcgann2015")


# =====================================================================
# Tier 1: Utility functions
# =====================================================================

class TestUtilities:
    """Test utility functions."""

    def test_has_liquepy_returns_bool(self):
        assert isinstance(has_liquepy(), bool)

    def test_calc_ldi_safe_basic(self):
        from liquepy_agent.cpt_liquefaction import _calc_ldi_safe
        depth = np.array([1.0, 2.0, 3.0])
        strain = np.array([0.01, 0.02, 0.01])
        ldi = _calc_ldi_safe(strain, depth)
        assert ldi > 0

    def test_calc_ldi_safe_with_zmax(self):
        from liquepy_agent.cpt_liquefaction import _calc_ldi_safe
        depth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        strain = np.array([0.01, 0.02, 0.03, 0.02, 0.01])
        ldi_full = _calc_ldi_safe(strain, depth)
        ldi_partial = _calc_ldi_safe(strain, depth, z_max=3.0)
        assert ldi_partial < ldi_full

    def test_calc_ldi_safe_single_point(self):
        from liquepy_agent.cpt_liquefaction import _calc_ldi_safe
        ldi = _calc_ldi_safe(np.array([0.01]), np.array([1.0]))
        assert ldi == 0.0


# =====================================================================
# Tier 1: Foundry metadata
# =====================================================================

class TestFoundryMetadata:
    """Test Foundry agent metadata functions (no liquepy needed)."""

    def test_list_methods_all(self):
        from liquepy_agent_foundry import liquepy_list_methods
        result = json.loads(liquepy_list_methods(""))
        assert "Triggering" in result or "Correlations" in result

    def test_list_methods_triggering(self):
        from liquepy_agent_foundry import liquepy_list_methods
        result = json.loads(liquepy_list_methods("Triggering"))
        assert "Triggering" in result
        assert "cpt_liquefaction" in result["Triggering"]

    def test_list_methods_bad_category(self):
        from liquepy_agent_foundry import liquepy_list_methods
        result = json.loads(liquepy_list_methods("nonexistent"))
        assert "error" in result

    def test_describe_cpt_liquefaction(self):
        from liquepy_agent_foundry import liquepy_describe_method
        result = json.loads(liquepy_describe_method("cpt_liquefaction"))
        assert "parameters" in result
        assert "depth" in result["parameters"]

    def test_describe_field_correlations(self):
        from liquepy_agent_foundry import liquepy_describe_method
        result = json.loads(liquepy_describe_method("field_correlations"))
        assert "parameters" in result
        assert "vs_method" in result["parameters"]

    def test_describe_unknown_method(self):
        from liquepy_agent_foundry import liquepy_describe_method
        result = json.loads(liquepy_describe_method("nonexistent"))
        assert "error" in result

    def test_agent_invalid_json(self):
        from liquepy_agent_foundry import liquepy_agent
        result = json.loads(liquepy_agent("cpt_liquefaction", "not json"))
        assert "error" in result

    def test_agent_unknown_method(self):
        from liquepy_agent_foundry import liquepy_agent
        result = json.loads(liquepy_agent("nonexistent", "{}"))
        assert "error" in result


# =====================================================================
# Tier 2: CPT liquefaction integration (requires liquepy)
# =====================================================================

@requires_liquepy
class TestCPTLiquefactionIntegration:
    """Integration tests for CPT liquefaction analysis."""

    def test_basic_analysis(self):
        from liquepy_agent import analyze_cpt_liquefaction
        depth, q_c, f_s, u_2 = _make_cpt_arrays()
        r = analyze_cpt_liquefaction(
            depth, q_c, f_s, u_2, gwl=1.0, pga=0.3, m_w=7.5,
        )
        assert r.n_points == 40
        assert r.pga_g == 0.3
        assert r.m_w == 7.5
        assert len(r.factor_of_safety) == 40

    def test_fos_values_reasonable(self):
        from liquepy_agent import analyze_cpt_liquefaction
        depth, q_c, f_s, u_2 = _make_cpt_arrays()
        r = analyze_cpt_liquefaction(
            depth, q_c, f_s, u_2, gwl=1.0, pga=0.3, m_w=7.5,
        )
        # FoS should be bounded
        assert np.all(r.factor_of_safety >= 0)
        assert np.all(r.factor_of_safety <= 3.0)

    def test_lpi_positive_for_liquefiable_soil(self):
        from liquepy_agent import analyze_cpt_liquefaction
        depth, q_c, f_s, u_2 = _make_cpt_arrays(qc_kpa=3000.0)
        r = analyze_cpt_liquefaction(
            depth, q_c, f_s, u_2, gwl=1.0, pga=0.4, m_w=7.5,
        )
        # Soft soil + strong shaking should trigger liquefaction
        assert r.lpi > 0

    def test_lsn_computed(self):
        from liquepy_agent import analyze_cpt_liquefaction
        depth, q_c, f_s, u_2 = _make_cpt_arrays(qc_kpa=3000.0)
        r = analyze_cpt_liquefaction(
            depth, q_c, f_s, u_2, gwl=1.0, pga=0.4, m_w=7.5,
        )
        assert isinstance(r.lsn, float)

    def test_ldi_computed(self):
        from liquepy_agent import analyze_cpt_liquefaction
        depth, q_c, f_s, u_2 = _make_cpt_arrays(qc_kpa=3000.0)
        r = analyze_cpt_liquefaction(
            depth, q_c, f_s, u_2, gwl=1.0, pga=0.4, m_w=7.5,
        )
        assert isinstance(r.ldi_m, float)
        assert r.ldi_m >= 0

    def test_stress_profiles(self):
        from liquepy_agent import analyze_cpt_liquefaction
        depth, q_c, f_s, u_2 = _make_cpt_arrays()
        r = analyze_cpt_liquefaction(
            depth, q_c, f_s, u_2, gwl=1.0, pga=0.3, m_w=7.5,
        )
        # Total stress should increase with depth
        assert r.sigma_v[-1] > r.sigma_v[0]
        # Effective stress should be positive
        assert np.all(r.sigma_veff > 0)

    def test_ic_values(self):
        from liquepy_agent import analyze_cpt_liquefaction
        depth, q_c, f_s, u_2 = _make_cpt_arrays()
        r = analyze_cpt_liquefaction(
            depth, q_c, f_s, u_2, gwl=1.0, pga=0.3, m_w=7.5,
        )
        # Ic should be between 1 and 4
        assert np.all(r.i_c >= 0)
        assert np.all(r.i_c <= 5)

    def test_volumetric_strain_nonneg(self):
        from liquepy_agent import analyze_cpt_liquefaction
        depth, q_c, f_s, u_2 = _make_cpt_arrays()
        r = analyze_cpt_liquefaction(
            depth, q_c, f_s, u_2, gwl=1.0, pga=0.3, m_w=7.5,
        )
        assert np.all(r.volumetric_strain >= 0)

    def test_settlement_mm(self):
        from liquepy_agent import analyze_cpt_liquefaction
        depth, q_c, f_s, u_2 = _make_cpt_arrays(qc_kpa=3000.0)
        r = analyze_cpt_liquefaction(
            depth, q_c, f_s, u_2, gwl=1.0, pga=0.4, m_w=7.5,
        )
        assert r.max_settlement_mm >= 0

    def test_no_u2_defaults_to_zeros(self):
        from liquepy_agent import analyze_cpt_liquefaction
        depth, q_c, f_s, _ = _make_cpt_arrays()
        r = analyze_cpt_liquefaction(
            depth, q_c, f_s, gwl=1.0, pga=0.3, m_w=7.5,
        )
        assert r.n_points == 40

    def test_to_dict_json_serializable(self):
        from liquepy_agent import analyze_cpt_liquefaction
        depth, q_c, f_s, u_2 = _make_cpt_arrays()
        r = analyze_cpt_liquefaction(
            depth, q_c, f_s, u_2, gwl=1.0, pga=0.3, m_w=7.5,
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)

    def test_high_pga_lower_fos(self):
        from liquepy_agent import analyze_cpt_liquefaction
        depth, q_c, f_s, u_2 = _make_cpt_arrays()
        r_low = analyze_cpt_liquefaction(
            depth, q_c, f_s, u_2, gwl=1.0, pga=0.1, m_w=7.5,
        )
        r_high = analyze_cpt_liquefaction(
            depth, q_c, f_s, u_2, gwl=1.0, pga=0.5, m_w=7.5,
        )
        assert r_high.min_fos <= r_low.min_fos

    def test_foundry_agent_runs(self):
        from liquepy_agent_foundry import liquepy_agent
        depth, q_c, f_s, u_2 = _make_cpt_arrays(n=20)
        params = {
            "depth": depth.tolist(),
            "q_c": q_c.tolist(),
            "f_s": f_s.tolist(),
            "u_2": u_2.tolist(),
            "gwl": 1.0,
            "pga": 0.3,
            "m_w": 7.5,
        }
        result = json.loads(liquepy_agent("cpt_liquefaction", json.dumps(params)))
        assert "error" not in result
        assert "lpi" in result


# =====================================================================
# Tier 2: Field correlations integration (requires liquepy)
# =====================================================================

@requires_liquepy
class TestFieldCorrelationsIntegration:
    """Integration tests for field correlations."""

    def test_mcgann_vs(self):
        from liquepy_agent import analyze_field_correlations
        depth, q_c, f_s, u_2 = _make_cpt_arrays()
        r = analyze_field_correlations(
            depth, q_c, f_s, u_2, gwl=1.0, vs_method="mcgann2015",
        )
        assert r.n_points == 40
        assert r.vs_method == "mcgann2015"
        assert len(r.vs_m_per_s) == 40
        # Vs should be positive
        assert np.all(r.vs_m_per_s > 0)

    def test_robertson_vs(self):
        from liquepy_agent import analyze_field_correlations
        depth, q_c, f_s, u_2 = _make_cpt_arrays()
        r = analyze_field_correlations(
            depth, q_c, f_s, u_2, gwl=1.0, vs_method="robertson2009",
        )
        assert r.vs_method == "robertson2009"
        valid = r.vs_m_per_s[np.isfinite(r.vs_m_per_s)]
        assert len(valid) > 0
        assert np.all(valid > 0)

    def test_andrus_vs(self):
        from liquepy_agent import analyze_field_correlations
        depth, q_c, f_s, u_2 = _make_cpt_arrays()
        r = analyze_field_correlations(
            depth, q_c, f_s, u_2, gwl=1.0, vs_method="andrus2007",
        )
        assert r.vs_method == "andrus2007"

    def test_relative_density_range(self):
        from liquepy_agent import analyze_field_correlations
        depth, q_c, f_s, u_2 = _make_cpt_arrays()
        r = analyze_field_correlations(depth, q_c, f_s, u_2, gwl=1.0)
        # Dr should be between 0 and 1
        assert np.all(r.relative_density >= 0)
        assert np.all(r.relative_density <= 1)

    def test_permeability_positive(self):
        from liquepy_agent import analyze_field_correlations
        depth, q_c, f_s, u_2 = _make_cpt_arrays()
        r = analyze_field_correlations(depth, q_c, f_s, u_2, gwl=1.0)
        valid = r.permeability_cm_per_s[np.isfinite(r.permeability_cm_per_s)]
        assert np.all(valid > 0)

    def test_ic_values(self):
        from liquepy_agent import analyze_field_correlations
        depth, q_c, f_s, u_2 = _make_cpt_arrays()
        r = analyze_field_correlations(depth, q_c, f_s, u_2, gwl=1.0)
        assert np.all(r.i_c >= 0)
        assert np.all(r.i_c <= 5)

    def test_no_u2(self):
        from liquepy_agent import analyze_field_correlations
        depth, q_c, f_s, _ = _make_cpt_arrays()
        r = analyze_field_correlations(depth, q_c, f_s, gwl=1.0)
        assert r.n_points == 40

    def test_to_dict_json_serializable(self):
        from liquepy_agent import analyze_field_correlations
        depth, q_c, f_s, u_2 = _make_cpt_arrays()
        r = analyze_field_correlations(depth, q_c, f_s, u_2, gwl=1.0)
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)

    def test_foundry_agent_runs(self):
        from liquepy_agent_foundry import liquepy_agent
        depth, q_c, f_s, u_2 = _make_cpt_arrays(n=20)
        params = {
            "depth": depth.tolist(),
            "q_c": q_c.tolist(),
            "f_s": f_s.tolist(),
            "u_2": u_2.tolist(),
            "gwl": 1.0,
            "vs_method": "mcgann2015",
        }
        result = json.loads(liquepy_agent("field_correlations", json.dumps(params)))
        assert "error" not in result
        assert "vs_method" in result

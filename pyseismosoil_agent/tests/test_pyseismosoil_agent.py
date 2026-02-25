"""
Tests for pyseismosoil_agent — nonlinear soil curves and Vs profiles.

Tier 1: No PySeismoSoil required (result dataclasses, validation, utilities, Foundry metadata)
Tier 2: Requires PySeismoSoil (integration tests)
"""

import json
import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from pyseismosoil_agent.pyseismosoil_utils import has_pyseismosoil
from pyseismosoil_agent.results import CurveResult, VsProfileResult

requires_pss = pytest.mark.skipif(
    not has_pyseismosoil(), reason="PySeismoSoil not installed"
)


# =====================================================================
# Tier 1: CurveResult defaults
# =====================================================================

class TestCurveResultDefaults:

    def test_default_construction(self):
        r = CurveResult()
        assert r.model == ""
        assert r.n_points == 0
        assert r.strain_pct is None

    def test_construction_with_values(self):
        r = CurveResult(
            model="MKZ",
            params={"gamma_ref": 0.05},
            n_points=50,
            strain_pct=np.geomspace(1e-4, 10, 50),
            G_Gmax=np.ones(50),
            damping_pct=np.zeros(50),
        )
        assert r.model == "MKZ"
        assert r.n_points == 50

    def test_summary_contains_model(self):
        r = CurveResult(model="MKZ", params={"gamma_ref": 0.05}, n_points=50)
        s = r.summary()
        assert "MKZ" in s

    def test_to_dict_keys(self):
        r = CurveResult(
            model="MKZ", params={"gamma_ref": 0.05}, n_points=50,
            strain_pct=np.geomspace(1e-4, 10, 10),
            G_Gmax=np.ones(10),
            damping_pct=np.zeros(10),
        )
        d = r.to_dict()
        assert "model" in d
        assert "strain_pct" in d
        assert "G_Gmax" in d
        assert "damping_pct" in d

    def test_to_dict_json_serializable(self):
        r = CurveResult(
            model="MKZ", params={"gamma_ref": 0.05}, n_points=10,
            strain_pct=np.geomspace(1e-4, 10, 10),
            G_Gmax=np.ones(10),
            damping_pct=np.zeros(10),
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# =====================================================================
# Tier 1: VsProfileResult defaults
# =====================================================================

class TestVsProfileResultDefaults:

    def test_default_construction(self):
        r = VsProfileResult()
        assert r.n_layers == 0
        assert r.vs30 == 0.0

    def test_construction_with_values(self):
        r = VsProfileResult(
            n_layers=3, vs30=250.0, f0_bh=3.0, f0_ro=3.5,
            z1=20.0, z_max=20.0,
            thicknesses=[5, 5, 10, 0],
            vs_values=[150, 200, 300, 400],
        )
        assert r.vs30 == 250.0
        assert r.n_layers == 3

    def test_summary_contains_vs30(self):
        r = VsProfileResult(vs30=257.1, f0_bh=3.4, n_layers=3)
        s = r.summary()
        assert "257.1" in s

    def test_to_dict_keys(self):
        r = VsProfileResult(
            n_layers=3, vs30=250.0, f0_bh=3.0, f0_ro=3.5,
            thicknesses=[5, 5, 10, 0], vs_values=[150, 200, 300, 400],
        )
        d = r.to_dict()
        assert "vs30" in d
        assert "f0_bh" in d
        assert "thicknesses" in d
        assert "vs_values" in d

    def test_to_dict_json_serializable(self):
        r = VsProfileResult(
            n_layers=3, vs30=250.0,
            thicknesses=[5, 5, 10, 0], vs_values=[150, 200, 300, 400],
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# =====================================================================
# Tier 1: Plot smoke test
# =====================================================================

class TestPlotSmoke:

    def test_curve_plot(self):
        strain = np.geomspace(1e-4, 10, 30)
        r = CurveResult(
            model="MKZ", params={"gamma_ref": 0.05}, n_points=30,
            strain_pct=strain,
            G_Gmax=1.0 / (1 + strain / 0.05),
            damping_pct=strain * 2,
        )
        ax = r.plot_curves(show=False)
        assert ax is not None


# =====================================================================
# Tier 1: Input validation
# =====================================================================

class TestCurveValidation:

    def test_bad_model(self):
        from pyseismosoil_agent.soil_curves import _validate_curve_inputs
        with pytest.raises(ValueError, match="model"):
            _validate_curve_inputs("BAD", {"gamma_ref": 0.05}, 50)

    def test_too_few_points(self):
        from pyseismosoil_agent.soil_curves import _validate_curve_inputs
        with pytest.raises(ValueError, match="n_points"):
            _validate_curve_inputs("MKZ", {"gamma_ref": 0.05, "beta": 1, "s": 0.9, "Gmax": 50000}, 1)

    def test_missing_mkz_params(self):
        from pyseismosoil_agent.soil_curves import _validate_curve_inputs
        with pytest.raises(ValueError, match="Missing"):
            _validate_curve_inputs("MKZ", {"gamma_ref": 0.05}, 50)

    def test_missing_hh_params(self):
        from pyseismosoil_agent.soil_curves import _validate_curve_inputs
        with pytest.raises(ValueError, match="Missing"):
            _validate_curve_inputs("HH", {"gamma_ref": 0.05}, 50)

    def test_no_params(self):
        from pyseismosoil_agent.soil_curves import generate_curves
        with pytest.raises(ValueError, match="params"):
            generate_curves(params=None)


class TestProfileValidation:

    def test_too_few_layers(self):
        from pyseismosoil_agent.soil_curves import _validate_profile_inputs
        with pytest.raises(ValueError, match="at least 2"):
            _validate_profile_inputs([0], [200])

    def test_mismatched_lengths(self):
        from pyseismosoil_agent.soil_curves import _validate_profile_inputs
        with pytest.raises(ValueError, match="same length"):
            _validate_profile_inputs([5, 0], [150, 200, 300])

    def test_last_not_halfspace(self):
        from pyseismosoil_agent.soil_curves import _validate_profile_inputs
        with pytest.raises(ValueError, match="halfspace"):
            _validate_profile_inputs([5, 10], [150, 200])

    def test_negative_thickness(self):
        from pyseismosoil_agent.soil_curves import _validate_profile_inputs
        with pytest.raises(ValueError, match="thickness"):
            _validate_profile_inputs([-5, 0], [150, 200])

    def test_negative_vs(self):
        from pyseismosoil_agent.soil_curves import _validate_profile_inputs
        with pytest.raises(ValueError, match="Vs"):
            _validate_profile_inputs([5, 0], [-150, 200])


# =====================================================================
# Tier 1: Utilities
# =====================================================================

class TestUtilities:

    def test_has_pyseismosoil_returns_bool(self):
        assert isinstance(has_pyseismosoil(), bool)


# =====================================================================
# Tier 1: Foundry metadata
# =====================================================================

class TestFoundryMetadata:

    def test_list_methods_all(self):
        from foundry.pyseismosoil_agent_foundry import pyseismosoil_list_methods
        result = json.loads(pyseismosoil_list_methods(""))
        assert "Soil Curves" in result
        assert "Vs Profile" in result

    def test_list_methods_filtered(self):
        from foundry.pyseismosoil_agent_foundry import pyseismosoil_list_methods
        result = json.loads(pyseismosoil_list_methods("Soil Curves"))
        assert "generate_curves" in result["Soil Curves"]

    def test_list_methods_bad_category(self):
        from foundry.pyseismosoil_agent_foundry import pyseismosoil_list_methods
        result = json.loads(pyseismosoil_list_methods("nonexistent"))
        assert "error" in result

    def test_describe_curves(self):
        from foundry.pyseismosoil_agent_foundry import pyseismosoil_describe_method
        result = json.loads(pyseismosoil_describe_method("generate_curves"))
        assert "parameters" in result
        assert "model" in result["parameters"]

    def test_describe_profile(self):
        from foundry.pyseismosoil_agent_foundry import pyseismosoil_describe_method
        result = json.loads(pyseismosoil_describe_method("analyze_vs_profile"))
        assert "parameters" in result
        assert "thicknesses" in result["parameters"]

    def test_describe_unknown(self):
        from foundry.pyseismosoil_agent_foundry import pyseismosoil_describe_method
        result = json.loads(pyseismosoil_describe_method("nonexistent"))
        assert "error" in result

    def test_agent_invalid_json(self):
        from foundry.pyseismosoil_agent_foundry import pyseismosoil_agent
        result = json.loads(pyseismosoil_agent("generate_curves", "not json"))
        assert "error" in result

    def test_agent_unknown_method(self):
        from foundry.pyseismosoil_agent_foundry import pyseismosoil_agent
        result = json.loads(pyseismosoil_agent("nonexistent", "{}"))
        assert "error" in result


# =====================================================================
# Tier 2: Curve integration (requires PySeismoSoil)
# =====================================================================

@requires_pss
class TestCurveIntegration:

    def test_mkz_curves(self):
        from pyseismosoil_agent import generate_curves
        r = generate_curves(
            model="MKZ",
            params={"gamma_ref": 0.05, "beta": 1.0, "s": 0.919, "Gmax": 50000},
        )
        assert r.model == "MKZ"
        assert r.n_points == 50
        assert r.G_Gmax is not None
        assert r.damping_pct is not None
        # G/Gmax should decrease with strain
        assert r.G_Gmax[0] > r.G_Gmax[-1]
        # Damping should increase with strain
        assert r.damping_pct[-1] > r.damping_pct[0]

    def test_hh_curves(self):
        from pyseismosoil_agent import generate_curves
        r = generate_curves(
            model="HH",
            params={
                "gamma_t": 0.065, "a": 1.0, "gamma_ref": 0.05,
                "beta": 1.0, "s": 0.919, "Gmax": 50000,
                "mu": 1.0, "Tmax": 0.5, "d": 1.0,
            },
        )
        assert r.model == "HH"
        assert r.G_Gmax is not None

    def test_custom_strain_range(self):
        from pyseismosoil_agent import generate_curves
        r = generate_curves(
            model="MKZ",
            params={"gamma_ref": 0.05, "beta": 1.0, "s": 0.919, "Gmax": 50000},
            strain_min=1e-3, strain_max=5.0, n_points=30,
        )
        assert r.n_points == 30
        assert abs(r.strain_pct[0] - 1e-3) < 1e-6

    def test_curves_to_dict_json(self):
        from pyseismosoil_agent import generate_curves
        r = generate_curves(
            model="MKZ",
            params={"gamma_ref": 0.05, "beta": 1.0, "s": 0.919, "Gmax": 50000},
            n_points=20,
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)

    def test_curves_plot(self):
        from pyseismosoil_agent import generate_curves
        r = generate_curves(
            model="MKZ",
            params={"gamma_ref": 0.05, "beta": 1.0, "s": 0.919, "Gmax": 50000},
            n_points=20,
        )
        ax = r.plot_curves(show=False)
        assert ax is not None


# =====================================================================
# Tier 2: Vs profile integration (requires PySeismoSoil)
# =====================================================================

@requires_pss
class TestVsProfileIntegration:

    def test_basic_profile(self):
        from pyseismosoil_agent import analyze_vs_profile
        r = analyze_vs_profile(
            thicknesses=[5, 5, 10, 0],
            vs_values=[150, 200, 300, 400],
        )
        assert r.n_layers == 3
        assert r.vs30 > 0
        assert r.f0_bh > 0
        assert r.f0_ro > 0

    def test_vs30_calculation(self):
        from pyseismosoil_agent import analyze_vs_profile
        # Simple uniform profile: Vs30 should equal layer Vs
        r = analyze_vs_profile(
            thicknesses=[30, 0],
            vs_values=[200, 500],
        )
        assert abs(r.vs30 - 200.0) < 1.0

    def test_depth_array(self):
        from pyseismosoil_agent import analyze_vs_profile
        r = analyze_vs_profile(
            thicknesses=[5, 5, 10, 0],
            vs_values=[150, 200, 300, 400],
        )
        assert r.depth_array[0] == 0.0
        assert r.depth_array[-1] == 20.0

    def test_profile_to_dict_json(self):
        from pyseismosoil_agent import analyze_vs_profile
        r = analyze_vs_profile(
            thicknesses=[5, 5, 10, 0],
            vs_values=[150, 200, 300, 400],
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# =====================================================================
# Tier 2: Foundry integration (requires PySeismoSoil)
# =====================================================================

@requires_pss
class TestFoundryIntegration:

    def test_foundry_curves(self):
        from foundry.pyseismosoil_agent_foundry import pyseismosoil_agent
        params = {
            "model": "MKZ",
            "params": {"gamma_ref": 0.05, "beta": 1.0, "s": 0.919, "Gmax": 50000},
            "n_points": 20,
        }
        result = json.loads(pyseismosoil_agent("generate_curves", json.dumps(params)))
        assert "error" not in result
        assert result["model"] == "MKZ"
        assert "G_Gmax" in result

    def test_foundry_profile(self):
        from foundry.pyseismosoil_agent_foundry import pyseismosoil_agent
        params = {
            "thicknesses": [5, 5, 10, 0],
            "vs_values": [150, 200, 300, 400],
        }
        result = json.loads(pyseismosoil_agent("analyze_vs_profile", json.dumps(params)))
        assert "error" not in result
        assert result["vs30"] > 0
        assert "f0_bh" in result

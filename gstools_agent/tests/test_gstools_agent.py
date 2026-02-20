"""
Tests for gstools_agent — geostatistical analysis.

Tier 1: No gstools required (result dataclasses, validation, utilities, Foundry metadata)
Tier 2: Requires gstools (integration tests with synthetic data)
"""

import json
import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from gstools_agent.gstools_utils import has_gstools
from gstools_agent.results import KrigingResult, VariogramResult, RandomFieldResult

requires_gstools = pytest.mark.skipif(
    not has_gstools(), reason="gstools not installed"
)


# =====================================================================
# Helpers
# =====================================================================

def _make_synthetic_data(n=20, seed=42):
    """Create synthetic SPT N-value measurements at random locations."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    # Spatial trend: higher N-values toward the east
    values = 10 + 0.1 * x + rng.randn(n) * 3
    return x, y, values


# =====================================================================
# Tier 1: KrigingResult defaults
# =====================================================================

class TestKrigingResultDefaults:

    def test_default_construction(self):
        r = KrigingResult()
        assert r.n_data == 0
        assert r.n_grid_x == 0
        assert r.model_type == ""
        assert r.field is None

    def test_construction_with_values(self):
        r = KrigingResult(
            n_data=20, n_grid_x=50, n_grid_y=50,
            model_type="Gaussian", variance=10.0, len_scale=30.0,
            nugget=0.5, kriging_type="ordinary",
        )
        assert r.n_data == 20
        assert r.model_type == "Gaussian"
        assert r.kriging_type == "ordinary"

    def test_summary_contains_model(self):
        r = KrigingResult(
            n_data=20, n_grid_x=50, n_grid_y=50,
            model_type="Gaussian", variance=10.0, len_scale=30.0,
        )
        s = r.summary()
        assert "Gaussian" in s
        assert "20" in s

    def test_to_dict_keys(self):
        r = KrigingResult(
            n_data=20, n_grid_x=50, n_grid_y=50,
            model_type="Gaussian", variance=10.0, len_scale=30.0,
        )
        d = r.to_dict()
        assert "n_data" in d
        assert "model_type" in d
        assert "variance" in d
        assert "len_scale" in d

    def test_to_dict_with_field(self):
        field = np.ones((10, 10))
        grid_x = np.linspace(0, 100, 10)
        grid_y = np.linspace(0, 100, 10)
        r = KrigingResult(
            n_data=5, n_grid_x=10, n_grid_y=10,
            model_type="Gaussian", variance=1.0, len_scale=10.0,
            field=field, krige_variance=field * 0.1,
            grid_x=grid_x, grid_y=grid_y,
        )
        d = r.to_dict()
        assert "field" in d
        assert "field_min" in d
        assert "krige_variance" in d
        assert "grid_x" in d

    def test_to_dict_no_field_when_none(self):
        r = KrigingResult(n_data=5, model_type="Gaussian")
        d = r.to_dict()
        assert "field" not in d
        assert "krige_variance" not in d

    def test_to_dict_json_serializable(self):
        field = np.ones((5, 5))
        r = KrigingResult(
            n_data=3, n_grid_x=5, n_grid_y=5,
            model_type="Gaussian", variance=1.0, len_scale=10.0,
            field=field, grid_x=np.linspace(0, 10, 5),
            grid_y=np.linspace(0, 10, 5),
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# =====================================================================
# Tier 1: VariogramResult defaults
# =====================================================================

class TestVariogramResultDefaults:

    def test_default_construction(self):
        r = VariogramResult()
        assert r.n_data == 0
        assert r.n_bins == 0
        assert r.model_type == ""

    def test_construction_with_values(self):
        r = VariogramResult(
            n_data=20, n_bins=10,
            model_type="Exponential", variance=8.0,
            len_scale=25.0, nugget=0.5,
        )
        assert r.model_type == "Exponential"
        assert r.variance == 8.0

    def test_summary_contains_model(self):
        r = VariogramResult(
            n_data=20, n_bins=10, model_type="Matern",
            variance=5.0, len_scale=20.0,
        )
        s = r.summary()
        assert "Matern" in s
        assert "20" in s

    def test_to_dict_keys(self):
        r = VariogramResult(
            n_data=20, n_bins=10, model_type="Gaussian",
            variance=5.0, len_scale=20.0,
        )
        d = r.to_dict()
        assert "n_data" in d
        assert "model_type" in d
        assert "variance" in d

    def test_to_dict_with_bins(self):
        bins = np.linspace(1, 50, 10)
        gamma = np.linspace(0, 5, 10)
        r = VariogramResult(
            n_data=20, n_bins=10, model_type="Gaussian",
            variance=5.0, len_scale=20.0,
            bin_center=bins, gamma=gamma,
        )
        d = r.to_dict()
        assert "bin_center" in d
        assert "gamma" in d
        assert len(d["bin_center"]) == 10

    def test_to_dict_json_serializable(self):
        r = VariogramResult(
            n_data=20, n_bins=10, model_type="Gaussian",
            variance=5.0, len_scale=20.0,
            bin_center=np.linspace(1, 50, 10),
            gamma=np.linspace(0, 5, 10),
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# =====================================================================
# Tier 1: RandomFieldResult defaults
# =====================================================================

class TestRandomFieldResultDefaults:

    def test_default_construction(self):
        r = RandomFieldResult()
        assert r.n_grid_x == 0
        assert r.model_type == ""
        assert r.field is None

    def test_construction_with_values(self):
        r = RandomFieldResult(
            n_grid_x=50, n_grid_y=50,
            model_type="Gaussian", variance=1.0,
            len_scale=10.0, mean=15.0, seed=42,
        )
        assert r.mean == 15.0
        assert r.seed == 42

    def test_summary_contains_model(self):
        r = RandomFieldResult(
            n_grid_x=50, n_grid_y=50,
            model_type="Spherical", variance=2.0, len_scale=15.0,
        )
        s = r.summary()
        assert "Spherical" in s
        assert "50" in s

    def test_to_dict_keys(self):
        r = RandomFieldResult(
            n_grid_x=50, n_grid_y=50,
            model_type="Gaussian", variance=1.0, len_scale=10.0,
        )
        d = r.to_dict()
        assert "n_grid_x" in d
        assert "model_type" in d
        assert "seed" in d

    def test_to_dict_with_field(self):
        field = np.ones((10, 10))
        r = RandomFieldResult(
            n_grid_x=10, n_grid_y=10,
            model_type="Gaussian", variance=1.0, len_scale=10.0,
            field=field, grid_x=np.linspace(0, 100, 10),
            grid_y=np.linspace(0, 100, 10),
        )
        d = r.to_dict()
        assert "field" in d
        assert "field_min" in d
        assert "field_std" in d

    def test_to_dict_json_serializable(self):
        r = RandomFieldResult(
            n_grid_x=5, n_grid_y=5,
            model_type="Gaussian", variance=1.0, len_scale=10.0,
            field=np.ones((5, 5)),
            grid_x=np.linspace(0, 10, 5),
            grid_y=np.linspace(0, 10, 5),
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# =====================================================================
# Tier 1: Plot smoke tests
# =====================================================================

class TestPlotSmoke:

    def test_kriging_plot(self):
        field = np.random.rand(10, 10)
        r = KrigingResult(
            n_data=5, n_grid_x=10, n_grid_y=10,
            model_type="Gaussian", variance=1.0, len_scale=10.0,
            field=field, grid_x=np.linspace(0, 100, 10),
            grid_y=np.linspace(0, 100, 10),
        )
        ax = r.plot_field(show=False)
        assert ax is not None

    def test_variogram_plot(self):
        r = VariogramResult(
            n_data=20, n_bins=10, model_type="Gaussian",
            variance=5.0, len_scale=20.0,
            bin_center=np.linspace(1, 50, 10),
            gamma=np.linspace(0, 5, 10),
        )
        ax = r.plot_variogram(show=False)
        assert ax is not None

    def test_random_field_plot(self):
        r = RandomFieldResult(
            n_grid_x=10, n_grid_y=10,
            model_type="Gaussian", variance=1.0, len_scale=10.0,
            field=np.random.rand(10, 10),
            grid_x=np.linspace(0, 100, 10),
            grid_y=np.linspace(0, 100, 10),
        )
        ax = r.plot_field(show=False)
        assert ax is not None


# =====================================================================
# Tier 1: Input validation — kriging
# =====================================================================

class TestKrigingValidation:

    def test_too_few_points(self):
        from gstools_agent.kriging import _validate_kriging_inputs
        with pytest.raises(ValueError, match="at least 3"):
            _validate_kriging_inputs(
                np.array([1, 2]), np.array([1, 2]), np.array([1, 2]),
                "Gaussian", "ordinary", 0, 100, 0, 100, 50, 50,
            )

    def test_mismatched_lengths(self):
        from gstools_agent.kriging import _validate_kriging_inputs
        with pytest.raises(ValueError, match="same length"):
            _validate_kriging_inputs(
                np.array([1, 2, 3]), np.array([1, 2]), np.array([1, 2, 3]),
                "Gaussian", "ordinary", 0, 100, 0, 100, 50, 50,
            )

    def test_bad_model_type(self):
        from gstools_agent.kriging import _validate_kriging_inputs
        with pytest.raises(ValueError, match="model_type"):
            _validate_kriging_inputs(
                np.ones(5), np.ones(5), np.ones(5),
                "InvalidModel", "ordinary", 0, 100, 0, 100, 50, 50,
            )

    def test_bad_kriging_type(self):
        from gstools_agent.kriging import _validate_kriging_inputs
        with pytest.raises(ValueError, match="kriging_type"):
            _validate_kriging_inputs(
                np.ones(5), np.ones(5), np.ones(5),
                "Gaussian", "invalid_type", 0, 100, 0, 100, 50, 50,
            )

    def test_bad_grid_bounds(self):
        from gstools_agent.kriging import _validate_kriging_inputs
        with pytest.raises(ValueError, match="grid_x_max"):
            _validate_kriging_inputs(
                np.ones(5), np.ones(5), np.ones(5),
                "Gaussian", "ordinary", 100, 0, 0, 100, 50, 50,
            )

    def test_bad_grid_size(self):
        from gstools_agent.kriging import _validate_kriging_inputs
        with pytest.raises(ValueError, match="n_grid_x"):
            _validate_kriging_inputs(
                np.ones(5), np.ones(5), np.ones(5),
                "Gaussian", "ordinary", 0, 100, 0, 100, 1, 50,
            )


# =====================================================================
# Tier 1: Input validation — variogram
# =====================================================================

class TestVariogramValidation:

    def test_too_few_points(self):
        from gstools_agent.variogram import _validate_variogram_inputs
        with pytest.raises(ValueError, match="at least 3"):
            _validate_variogram_inputs(
                np.array([1, 2]), np.array([1, 2]), np.array([1, 2]),
                "Gaussian",
            )

    def test_bad_model(self):
        from gstools_agent.variogram import _validate_variogram_inputs
        with pytest.raises(ValueError, match="model_type"):
            _validate_variogram_inputs(
                np.ones(5), np.ones(5), np.ones(5), "BadModel",
            )


# =====================================================================
# Tier 1: Input validation — random field
# =====================================================================

class TestRandomFieldValidation:

    def test_bad_model(self):
        from gstools_agent.random_field import _validate_rf_inputs
        with pytest.raises(ValueError, match="model_type"):
            _validate_rf_inputs("BadModel", 1.0, 10.0, 0, 100, 0, 100, 50, 50)

    def test_bad_variance(self):
        from gstools_agent.random_field import _validate_rf_inputs
        with pytest.raises(ValueError, match="variance"):
            _validate_rf_inputs("Gaussian", -1.0, 10.0, 0, 100, 0, 100, 50, 50)

    def test_bad_len_scale(self):
        from gstools_agent.random_field import _validate_rf_inputs
        with pytest.raises(ValueError, match="len_scale"):
            _validate_rf_inputs("Gaussian", 1.0, -5.0, 0, 100, 0, 100, 50, 50)

    def test_bad_bounds(self):
        from gstools_agent.random_field import _validate_rf_inputs
        with pytest.raises(ValueError, match="max must be > min"):
            _validate_rf_inputs("Gaussian", 1.0, 10.0, 100, 0, 0, 100, 50, 50)

    def test_bad_grid_size(self):
        from gstools_agent.random_field import _validate_rf_inputs
        with pytest.raises(ValueError, match="n_x and n_y"):
            _validate_rf_inputs("Gaussian", 1.0, 10.0, 0, 100, 0, 100, 1, 50)


# =====================================================================
# Tier 1: Utilities
# =====================================================================

class TestUtilities:

    def test_has_gstools_returns_bool(self):
        assert isinstance(has_gstools(), bool)


# =====================================================================
# Tier 1: Foundry metadata
# =====================================================================

class TestFoundryMetadata:

    def test_list_methods_all(self):
        from gstools_agent_foundry import gstools_list_methods
        result = json.loads(gstools_list_methods(""))
        assert "Kriging" in result
        assert "Variogram" in result
        assert "Random Field" in result

    def test_list_methods_filtered(self):
        from gstools_agent_foundry import gstools_list_methods
        result = json.loads(gstools_list_methods("Kriging"))
        assert "kriging" in result["Kriging"]

    def test_list_methods_bad_category(self):
        from gstools_agent_foundry import gstools_list_methods
        result = json.loads(gstools_list_methods("nonexistent"))
        assert "error" in result

    def test_describe_kriging(self):
        from gstools_agent_foundry import gstools_describe_method
        result = json.loads(gstools_describe_method("kriging"))
        assert "parameters" in result
        assert "x" in result["parameters"]
        assert "y" in result["parameters"]
        assert "values" in result["parameters"]

    def test_describe_variogram(self):
        from gstools_agent_foundry import gstools_describe_method
        result = json.loads(gstools_describe_method("variogram"))
        assert "parameters" in result
        assert "n_bins" in result["parameters"]

    def test_describe_random_field(self):
        from gstools_agent_foundry import gstools_describe_method
        result = json.loads(gstools_describe_method("random_field"))
        assert "parameters" in result
        assert "seed" in result["parameters"]

    def test_describe_unknown(self):
        from gstools_agent_foundry import gstools_describe_method
        result = json.loads(gstools_describe_method("nonexistent"))
        assert "error" in result

    def test_agent_invalid_json(self):
        from gstools_agent_foundry import gstools_agent
        result = json.loads(gstools_agent("kriging", "not json"))
        assert "error" in result

    def test_agent_unknown_method(self):
        from gstools_agent_foundry import gstools_agent
        result = json.loads(gstools_agent("nonexistent", "{}"))
        assert "error" in result


# =====================================================================
# Tier 2: Kriging integration (requires gstools)
# =====================================================================

@requires_gstools
class TestKrigingIntegration:

    def test_basic_kriging(self):
        x, y, values = _make_synthetic_data(n=20)
        from gstools_agent import analyze_kriging
        r = analyze_kriging(x, y, values, n_grid_x=20, n_grid_y=20)
        assert r.n_data == 20
        assert r.n_grid_x == 20
        assert r.field is not None
        assert r.field.shape == (20, 20)
        assert r.krige_variance is not None

    def test_auto_bounds(self):
        x, y, values = _make_synthetic_data(n=20)
        from gstools_agent import analyze_kriging
        r = analyze_kriging(x, y, values, n_grid_x=10, n_grid_y=10)
        assert r.grid_x is not None
        assert r.grid_y is not None
        # Grid should extend beyond data range
        assert r.grid_x[0] < x.min()
        assert r.grid_x[-1] > x.max()

    def test_explicit_bounds(self):
        x, y, values = _make_synthetic_data(n=20)
        from gstools_agent import analyze_kriging
        r = analyze_kriging(
            x, y, values,
            grid_x_min=0, grid_x_max=100,
            grid_y_min=0, grid_y_max=100,
            n_grid_x=15, n_grid_y=15,
        )
        assert r.field.shape == (15, 15)
        assert abs(r.grid_x[0] - 0.0) < 0.01
        assert abs(r.grid_x[-1] - 100.0) < 0.01

    def test_exponential_model(self):
        x, y, values = _make_synthetic_data(n=20)
        from gstools_agent import analyze_kriging
        r = analyze_kriging(x, y, values, model_type="Exponential",
                            n_grid_x=10, n_grid_y=10)
        assert r.model_type == "Exponential"

    def test_simple_kriging(self):
        x, y, values = _make_synthetic_data(n=20)
        from gstools_agent import analyze_kriging
        r = analyze_kriging(x, y, values, kriging_type="simple",
                            n_grid_x=10, n_grid_y=10)
        assert r.kriging_type == "simple"

    def test_no_variogram_fit(self):
        x, y, values = _make_synthetic_data(n=20)
        from gstools_agent import analyze_kriging
        r = analyze_kriging(
            x, y, values, fit_variogram=False,
            variance=10.0, len_scale=30.0,
            n_grid_x=10, n_grid_y=10,
        )
        assert r.field is not None

    def test_with_nugget(self):
        x, y, values = _make_synthetic_data(n=20)
        from gstools_agent import analyze_kriging
        r = analyze_kriging(x, y, values, nugget=1.0,
                            n_grid_x=10, n_grid_y=10)
        assert r.nugget >= 0

    def test_to_dict_json_serializable(self):
        x, y, values = _make_synthetic_data(n=20)
        from gstools_agent import analyze_kriging
        r = analyze_kriging(x, y, values, n_grid_x=10, n_grid_y=10)
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)

    def test_plot_integration(self):
        x, y, values = _make_synthetic_data(n=20)
        from gstools_agent import analyze_kriging
        r = analyze_kriging(x, y, values, n_grid_x=10, n_grid_y=10)
        ax = r.plot_field(show=False)
        assert ax is not None


# =====================================================================
# Tier 2: Variogram integration (requires gstools)
# =====================================================================

@requires_gstools
class TestVariogramIntegration:

    def test_basic_variogram(self):
        x, y, values = _make_synthetic_data(n=30)
        from gstools_agent import analyze_variogram
        r = analyze_variogram(x, y, values)
        assert r.n_data == 30
        assert r.n_bins > 0
        assert r.bin_center is not None
        assert r.gamma is not None
        assert r.variance > 0
        assert r.len_scale > 0

    def test_exponential_fit(self):
        x, y, values = _make_synthetic_data(n=30)
        from gstools_agent import analyze_variogram
        r = analyze_variogram(x, y, values, model_type="Exponential")
        assert r.model_type == "Exponential"

    def test_custom_bins(self):
        x, y, values = _make_synthetic_data(n=30)
        from gstools_agent import analyze_variogram
        r = analyze_variogram(x, y, values, n_bins=15)
        assert r.n_bins == 15

    def test_to_dict_json_serializable(self):
        x, y, values = _make_synthetic_data(n=30)
        from gstools_agent import analyze_variogram
        r = analyze_variogram(x, y, values)
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)

    def test_plot_integration(self):
        x, y, values = _make_synthetic_data(n=30)
        from gstools_agent import analyze_variogram
        r = analyze_variogram(x, y, values)
        ax = r.plot_variogram(show=False)
        assert ax is not None


# =====================================================================
# Tier 2: Random field integration (requires gstools)
# =====================================================================

@requires_gstools
class TestRandomFieldIntegration:

    def test_basic_random_field(self):
        from gstools_agent import generate_random_field
        r = generate_random_field(n_x=20, n_y=20, seed=42)
        assert r.n_grid_x == 20
        assert r.n_grid_y == 20
        assert r.field is not None
        assert r.field.shape == (20, 20)

    def test_reproducible_seed(self):
        from gstools_agent import generate_random_field
        r1 = generate_random_field(n_x=10, n_y=10, seed=123)
        r2 = generate_random_field(n_x=10, n_y=10, seed=123)
        np.testing.assert_array_equal(r1.field, r2.field)

    def test_different_seed(self):
        from gstools_agent import generate_random_field
        r1 = generate_random_field(n_x=10, n_y=10, seed=1)
        r2 = generate_random_field(n_x=10, n_y=10, seed=2)
        assert not np.allclose(r1.field, r2.field)

    def test_custom_model(self):
        from gstools_agent import generate_random_field
        r = generate_random_field(
            model_type="Exponential", variance=5.0, len_scale=20.0,
            n_x=10, n_y=10,
        )
        assert r.model_type == "Exponential"
        assert r.variance == 5.0

    def test_with_mean(self):
        from gstools_agent import generate_random_field
        r = generate_random_field(mean=15.0, n_x=20, n_y=20, seed=42)
        assert r.mean == 15.0
        # Field mean should be approximately 15
        assert abs(r.field.mean() - 15.0) < 5.0

    def test_to_dict_json_serializable(self):
        from gstools_agent import generate_random_field
        r = generate_random_field(n_x=10, n_y=10)
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)

    def test_plot_integration(self):
        from gstools_agent import generate_random_field
        r = generate_random_field(n_x=10, n_y=10)
        ax = r.plot_field(show=False)
        assert ax is not None


# =====================================================================
# Tier 2: Foundry agent integration (requires gstools)
# =====================================================================

@requires_gstools
class TestFoundryIntegration:

    def test_foundry_kriging(self):
        x, y, values = _make_synthetic_data(n=20)
        from gstools_agent_foundry import gstools_agent
        params = {
            "x": x.tolist(),
            "y": y.tolist(),
            "values": values.tolist(),
            "n_grid_x": 10,
            "n_grid_y": 10,
        }
        result = json.loads(gstools_agent("kriging", json.dumps(params)))
        assert "error" not in result
        assert result["n_data"] == 20
        assert "field" in result

    def test_foundry_variogram(self):
        x, y, values = _make_synthetic_data(n=30)
        from gstools_agent_foundry import gstools_agent
        params = {
            "x": x.tolist(),
            "y": y.tolist(),
            "values": values.tolist(),
        }
        result = json.loads(gstools_agent("variogram", json.dumps(params)))
        assert "error" not in result
        assert result["n_data"] == 30
        assert "bin_center" in result

    def test_foundry_random_field(self):
        from gstools_agent_foundry import gstools_agent
        params = {
            "n_x": 10,
            "n_y": 10,
            "seed": 42,
        }
        result = json.loads(gstools_agent("random_field", json.dumps(params)))
        assert "error" not in result
        assert "field" in result

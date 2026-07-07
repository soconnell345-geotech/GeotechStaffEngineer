"""fem2d adapter single-sourced validation (v5.4 E9).

Proves the extracted `_require_choice` / `_normalize_gwt` helpers preserve the
prior behaviour: invalid enum parameters still raise (before any FEM runs) and
the GWT-argument coercion is identical for polylines / scalars / None / empty.
"""

import numpy as np
import pytest

from funhouse_agent.adapters.fem2d_adapter import (
    METHOD_REGISTRY as R, _require_choice, _normalize_gwt,
)


class TestRequireChoice:
    def test_accepts_allowed(self):
        # returns None (no raise) for a valid value
        assert _require_choice("t6", ("t6", "cst"),
                               name="element_type", method="m") is None

    def test_rejects_and_names_param_and_method(self):
        with pytest.raises(ValueError, match=r"m: element_type must be one of"):
            _require_choice("q8", ("t6", "cst"),
                            name="element_type", method="m")


class TestEnumValidationAtRegistry:
    """Each enum is rejected before the (expensive) analysis is invoked."""

    def test_bad_element_type(self):
        with pytest.raises(ValueError, match="element_type must be one of"):
            R["fem2d_slope_srm"]({
                "surface_points": [[0, 0], [10, 0]],
                "soil_layers": [{"bottom_elevation": -5}],
                "element_type": "q8"})

    def test_bad_srm_field(self):
        with pytest.raises(ValueError, match="srm_field must be one of"):
            R["fem2d_slope_srm"]({
                "surface_points": [[0, 0], [10, 0]],
                "soil_layers": [{"bottom_elevation": -5}],
                "srm_field": "cohesion"})

    def test_bad_n_gp(self):
        with pytest.raises(ValueError, match="n_gp must be one of"):
            R["fem2d_slope_srm"]({
                "surface_points": [[0, 0], [10, 0]],
                "soil_layers": [{"bottom_elevation": -5}],
                "n_gp": 4})

    def test_bad_consolidation_scheme(self):
        with pytest.raises(ValueError, match="consolidation_scheme must be one of"):
            R["fem2d_consolidation"]({
                "width": 1.0, "depth": 1.0,
                "soil_layers": [{"bottom_elevation": -1}],
                "k": 1e-9, "load_q": 10.0, "time_points": [1.0],
                "consolidation_scheme": "foo"})


class TestNormalizeGwt:
    def test_polyline_to_ndarray(self):
        out = _normalize_gwt([[0, 1.0], [5, 1.0]])
        assert isinstance(out, np.ndarray) and out.shape == (2, 2)

    def test_scalar_passthrough(self):
        assert _normalize_gwt(3.0) == 3.0

    def test_none_passthrough(self):
        assert _normalize_gwt(None) is None

    def test_empty_list_passthrough(self):
        # empty list is not a polyline → passed through unchanged (not ndarray)
        assert _normalize_gwt([]) == []

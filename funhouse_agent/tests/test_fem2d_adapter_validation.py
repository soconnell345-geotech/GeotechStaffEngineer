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


# ---------------------------------------------------------------------------
# Eval CON-1 / MRS-1 (2026-07-13): schema completeness + depth guard
# ---------------------------------------------------------------------------

class TestMethodInfoDeclaresHandlerParams:
    """Every param a handler accepts (its ``_valid`` tuple) must be declared in
    METHOD_INFO so describe_method advertises it. Regression for CON-1: the
    Biot modulus n_w was accepted by fem2d_consolidation but undocumented, so
    the agent could not set it and defaulted to a wrong M."""

    def test_no_handler_param_missing_from_method_info(self):
        import ast
        import re
        from funhouse_agent.adapters import fem2d_adapter as fa
        src = open(fa.__file__, encoding="utf-8").read()
        offenders = {}
        for m in fa.METHOD_REGISTRY:
            declared = set((fa.METHOD_INFO.get(m, {}).get("parameters") or {}))
            mo = re.search(
                r"_valid = (\([^)]*\))\s*\n\s*reject_unknown_params\("
                r"params, _valid, method=\"" + re.escape(m) + r"\"",
                src, re.DOTALL)
            if not mo:
                continue
            missing = set(ast.literal_eval(mo.group(1))) - declared
            if missing:
                offenders[m] = sorted(missing)
        assert not offenders, f"handler params absent from METHOD_INFO: {offenders}"

    def test_consolidation_declares_biot_modulus(self):
        from funhouse_agent.adapters.fem2d_adapter import METHOD_INFO
        assert "n_w" in METHOD_INFO["fem2d_consolidation"]["parameters"]


class TestPositiveDepthGuard:
    """A negative/zero domain depth silently produced non-converged garbage
    (eval MRS-1: the agent passed depth=-20 to fem2d_slope_srm). It now raises a
    clear error before any FEM runs."""

    def test_require_positive_rejects_negative(self):
        from funhouse_agent.adapters.fem2d_adapter import _require_positive
        with pytest.raises(ValueError, match="positive distance"):
            _require_positive(-20, name="depth", method="m")

    def test_require_positive_allows_none_and_positive(self):
        from funhouse_agent.adapters.fem2d_adapter import _require_positive
        assert _require_positive(None, name="depth", method="m") is None
        assert _require_positive(5.0, name="depth", method="m") is None

    def test_slope_srm_negative_depth_hints_mesh_study(self):
        from funhouse_agent.dispatch import call_agent
        r = call_agent("fem2d", "fem2d_slope_srm", {
            "surface_points": [[0, 0], [10, 0], [30, 10], [50, 10]],
            "soil_layers": [{"name": "l", "bottom_elevation": -10, "E": 20000,
                             "nu": 0.3, "c": 10, "phi": 15, "gamma": 18}],
            "depth": -20, "nx": 16, "ny": 8})
        assert "error" in r
        assert "positive" in r["error"].lower()
        assert "srm_mesh_refinement_study" in r["error"]

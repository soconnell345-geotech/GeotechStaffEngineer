"""
Tier 4: Error handling and edge case tests.

Tests that all agents handle bad input gracefully — returning error messages
in JSON instead of crashing. This is critical for LLM agents because the
LLM may provide incorrect parameter names, types, or values.
"""

import json
import pytest

from foundry_test_harness.harness import FoundryAgentHarness, AgentError

from foundry.bearing_capacity_agent_foundry import (
    bearing_capacity_agent, bearing_capacity_list_methods,
    bearing_capacity_describe_method,
)
from foundry.settlement_agent_foundry import settlement_agent
from foundry.axial_pile_agent_foundry import axial_pile_agent
from foundry.drilled_shaft_agent_foundry import drilled_shaft_agent
from foundry.seismic_geotech_agent_foundry import seismic_geotech_agent
from foundry.retaining_walls_agent_foundry import retaining_walls_agent
from foundry.sheet_pile_agent_foundry import sheet_pile_agent
from foundry.slope_stability_agent_foundry import slope_stability_agent
from foundry.geolysis_agent_foundry import geolysis_agent
from foundry.ground_improvement_agent_foundry import ground_improvement_agent
from foundry.wave_equation_agent_foundry import wave_equation_agent
from foundry.pile_group_agent_foundry import pile_group_agent
from foundry.downdrag_agent_foundry import downdrag_agent
from foundry.pystra_agent_foundry import pystra_agent

H = FoundryAgentHarness()

# All core agent functions for looping tests
ALL_AGENTS = [
    bearing_capacity_agent,
    settlement_agent,
    axial_pile_agent,
    drilled_shaft_agent,
    seismic_geotech_agent,
    retaining_walls_agent,
    sheet_pile_agent,
    slope_stability_agent,
    geolysis_agent,
    ground_improvement_agent,
    wave_equation_agent,
    pile_group_agent,
    downdrag_agent,
    pystra_agent,
]


class TestInvalidJSON:
    """All agents must handle malformed JSON gracefully."""

    @pytest.mark.parametrize("agent_func", ALL_AGENTS)
    def test_bad_json(self, agent_func):
        """Passing non-JSON string should return error dict, not crash."""
        result = H.call_raw(agent_func, "some_method", "not valid json {{{")
        assert "error" in result
        assert "json" in result["error"].lower() or "invalid" in result["error"].lower()

    @pytest.mark.parametrize("agent_func", ALL_AGENTS)
    def test_unknown_method(self, agent_func):
        """Passing unknown method name should return error with available methods."""
        result = H.call_raw(agent_func, "nonexistent_method_xyz", "{}")
        assert "error" in result
        assert "available" in result["error"].lower() or "unknown" in result["error"].lower()


class TestMissingParameters:
    """Agents should give clear errors when required parameters are missing."""

    def test_bearing_missing_width(self):
        """bearing_capacity_analysis requires 'width'."""
        with pytest.raises(AgentError, match="width|required|missing"):
            H.call(bearing_capacity_agent, "bearing_capacity_analysis", {
                "unit_weight": 18.0,
                "friction_angle": 30.0,
            })

    def test_bearing_missing_unit_weight(self):
        """bearing_capacity_analysis requires 'unit_weight'."""
        with pytest.raises(AgentError):
            H.call(bearing_capacity_agent, "bearing_capacity_analysis", {
                "width": 2.0,
                "friction_angle": 30.0,
            })

    def test_settlement_missing_params(self):
        """elastic_settlement requires q_net, B, Es."""
        with pytest.raises(AgentError):
            H.call(settlement_agent, "elastic_settlement", {
                "q_net": 100.0,
                # Missing B and Es
            })

    def test_spt_missing_eop(self):
        """correct_spt requires eop (effective overburden pressure)."""
        with pytest.raises(AgentError):
            H.call(geolysis_agent, "correct_spt", {
                "recorded_spt_n_value": 25,
                # Missing eop
            })


class TestInvalidValues:
    """Agents should validate parameter ranges."""

    def test_negative_width(self):
        """Negative footing width should be rejected."""
        with pytest.raises(AgentError):
            H.call(bearing_capacity_agent, "bearing_capacity_analysis", {
                "width": -1.0,
                "unit_weight": 18.0,
            })

    def test_friction_angle_out_of_range(self):
        """Friction angle > 60° should be rejected or capped."""
        # This may raise or produce unusual results — either is acceptable
        try:
            r = H.call(bearing_capacity_agent, "bearing_capacity_factors", {
                "friction_angle": 89.0,
            })
            # If it doesn't raise, values should still be positive
            assert r["Nc"] > 0
        except AgentError:
            pass  # Expected

    def test_zero_pile_length(self):
        """Zero pile length should be rejected."""
        with pytest.raises(AgentError):
            H.call(axial_pile_agent, "axial_pile_capacity", {
                "pile_type": "pipe_closed",
                "pile_length": 0.0,
                "diameter": 0.356,
                "wall_thickness": 0.0127,
                "layers": [
                    {"thickness": 10.0, "soil_type": "cohesionless",
                     "unit_weight": 18.0, "friction_angle": 30.0}
                ],
            })

    def test_empty_layers(self):
        """Empty layers list should be rejected."""
        with pytest.raises(AgentError):
            H.call(axial_pile_agent, "axial_pile_capacity", {
                "pile_type": "pipe_closed",
                "pile_length": 15.0,
                "diameter": 0.356,
                "wall_thickness": 0.0127,
                "layers": [],
            })


class TestDescribeMethodErrors:
    """describe_method should handle unknown methods gracefully."""

    def test_unknown_method_describe(self):
        """Asking for unknown method description returns error."""
        desc = H.describe(bearing_capacity_describe_method, "nonexistent_xyz")
        assert "error" in desc

    def test_describe_has_parameters(self):
        """Valid method description must have 'parameters' key."""
        desc = H.describe(bearing_capacity_describe_method,
                          "bearing_capacity_analysis")
        assert "parameters" in desc
        assert len(desc["parameters"]) > 0

    def test_list_returns_categories(self):
        """list_methods returns at least one category."""
        methods = H.list_methods(bearing_capacity_list_methods)
        assert len(methods) > 0
        # Each category should have at least one method
        for cat, meths in methods.items():
            assert len(meths) > 0

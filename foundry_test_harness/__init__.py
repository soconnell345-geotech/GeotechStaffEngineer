"""
Foundry Agent Test Harness — validates the Geotech Staff Engineer agent ecosystem.

Tests the exact JSON-in / JSON-out path that the Palantir Foundry LLM agent uses.
Four tiers:
  Tier 1: Individual functions vs textbook answers
  Tier 2: Multi-function engineering workflows
  Tier 3: Cross-agent consistency checks
  Tier 4: Error handling and edge cases
"""

from foundry_test_harness.harness import FoundryAgentHarness, AgentError

__all__ = ["FoundryAgentHarness", "AgentError"]

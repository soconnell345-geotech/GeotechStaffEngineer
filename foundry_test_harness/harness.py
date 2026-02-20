"""
FoundryAgentHarness — simulates how the Palantir Foundry LLM agent calls tools.

Every Foundry agent function takes (method: str, parameters_json: str) -> str.
This harness wraps that interface so tests can pass Python dicts and get dicts back,
while still exercising the exact JSON serialization path the LLM uses.

Some agents (pystra, etc.) return dicts/lists directly instead of JSON strings.
The harness handles both formats transparently.
"""

import json


class AgentError(Exception):
    """Raised when an agent returns an error dict instead of results."""
    pass


def _parse_result(raw):
    """Parse agent result — handles both JSON strings and direct dicts/lists."""
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


class FoundryAgentHarness:
    """Simulates LLM agent calling Foundry tools via JSON strings."""

    def call(self, agent_func, method, params_dict):
        """
        Call an agent function the way the LLM would.

        Args:
            agent_func: The *_agent function (e.g., bearing_capacity_agent)
            method: Method name string (e.g., "bearing_capacity_analysis")
            params_dict: Python dict of parameters (will be JSON-serialized)

        Returns:
            dict: Parsed result from the agent

        Raises:
            AgentError: If the agent returns {"error": "..."}
        """
        params_json = json.dumps(params_dict)
        raw = agent_func(method, params_json)
        result = _parse_result(raw)
        if isinstance(result, dict) and "error" in result:
            raise AgentError(result["error"])
        return result

    def call_raw(self, agent_func, method, params_json_str):
        """
        Call an agent with a raw JSON string (for testing malformed input).

        Returns:
            dict: Parsed result (may contain "error" key)
        """
        raw = agent_func(method, params_json_str)
        return _parse_result(raw)

    def list_methods(self, list_func, category=""):
        """
        Call *_list_methods and parse the result.

        Returns:
            dict or list: Method listings (format varies by agent)
        """
        raw = list_func(category)
        return _parse_result(raw)

    def describe(self, describe_func, method):
        """
        Call *_describe_method and parse the result.

        Returns:
            dict: Full method documentation including parameters, returns, etc.
        """
        raw = describe_func(method)
        return _parse_result(raw)

"""
Claude tool_use schema definitions for the 3 meta-tools.

These tools allow Claude to interact with all 30 geotechnical agents
through a unified interface: call_agent, list_methods, describe_method.
"""

from trial_agent.agent_registry import AGENT_NAMES

# Build the enum list once at import time
_AGENT_ENUM = list(AGENT_NAMES)

TOOLS = [
    {
        "name": "call_agent",
        "description": (
            "Execute a geotechnical calculation using one of the 30 specialized agents. "
            "Pass the agent name, method name, and a parameters object. "
            "Returns a JSON object with calculation results, or an error message. "
            "IMPORTANT: Always call describe_method first for any method you haven't "
            "used before, to get the exact parameter names and types."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "enum": _AGENT_ENUM,
                    "description": "The agent to call.",
                },
                "method": {
                    "type": "string",
                    "description": "The method name within the agent (e.g., 'bearing_capacity_analysis').",
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters for the calculation. Keys and types vary per method.",
                },
            },
            "required": ["agent_name", "method", "parameters"],
        },
    },
    {
        "name": "list_methods",
        "description": (
            "List available methods for a specific agent, organized by category. "
            "Returns {category: {method_name: brief_description}}. "
            "Optionally filter by category substring (e.g., 'bearing', 'Ch5')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "enum": _AGENT_ENUM,
                    "description": "The agent to query.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter (partial match). Leave empty for all.",
                    "default": "",
                },
            },
            "required": ["agent_name"],
        },
    },
    {
        "name": "describe_method",
        "description": (
            "Get full parameter documentation for a specific method. "
            "Returns parameter names, types, descriptions, default values, "
            "related methods, typical workflow, and common mistakes. "
            "ALWAYS call this before using a method for the first time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "enum": _AGENT_ENUM,
                    "description": "The agent containing the method.",
                },
                "method": {
                    "type": "string",
                    "description": "The method name to describe.",
                },
            },
            "required": ["agent_name", "method"],
        },
    },
]

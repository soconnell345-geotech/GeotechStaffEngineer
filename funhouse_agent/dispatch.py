"""Funhouse agent tool dispatch — routes tool calls to analysis modules.

Dispatches directly to analysis modules via adapter functions.
No dependency on foundry/ files.

Tools:
    list_agents()                       → available modules
    list_methods(agent_name, category)  → methods for a module
    describe_method(agent_name, method) → parameter documentation
    call_agent(agent_name, method, params) → execute calculation
"""

import json
import importlib

from funhouse_agent.adapters import MODULE_REGISTRY


# ---------------------------------------------------------------------------
# Lazy loader — caches imported adapter modules
# ---------------------------------------------------------------------------

_loaded_adapters = {}


def _load_adapter(agent_name: str):
    """Import an adapter module on demand and cache it."""
    if agent_name in _loaded_adapters:
        return _loaded_adapters[agent_name]

    spec = MODULE_REGISTRY[agent_name]
    mod = importlib.import_module(spec["adapter"])
    _loaded_adapters[agent_name] = mod
    return mod


# ---------------------------------------------------------------------------
# Dispatch functions — same interface as chat_agent/agent_registry.py
# ---------------------------------------------------------------------------

AGENT_NAMES = sorted(MODULE_REGISTRY.keys())


def list_agents() -> dict:
    """List all available analysis modules with brief descriptions."""
    return {name: spec["brief"] for name, spec in MODULE_REGISTRY.items()}


def list_methods(agent_name: str, category: str = "") -> dict:
    """List available methods for a specific module."""
    if agent_name not in MODULE_REGISTRY:
        return {"error": f"Unknown module '{agent_name}'. Available: {AGENT_NAMES}"}
    try:
        mod = _load_adapter(agent_name)
    except Exception as e:
        return {"error": f"Failed to load module '{agent_name}': {e}"}
    # Each adapter exports METHOD_INFO with method_name -> {category, brief, ...}
    result = {}
    for method_name, info in mod.METHOD_INFO.items():
        cat = info.get("category", "General")
        if category and cat.lower() != category.lower():
            continue
        if cat not in result:
            result[cat] = {}
        result[cat][method_name] = info["brief"]
    return result


def describe_method(agent_name: str, method: str) -> dict:
    """Get full parameter documentation for a method."""
    if agent_name not in MODULE_REGISTRY:
        return {"error": f"Unknown module '{agent_name}'. Available: {AGENT_NAMES}"}
    try:
        mod = _load_adapter(agent_name)
    except Exception as e:
        return {"error": f"Failed to load module '{agent_name}': {e}"}
    if method not in mod.METHOD_INFO:
        available = sorted(mod.METHOD_INFO.keys())
        return {"error": f"Unknown method '{method}'. Available: {available}"}
    return mod.METHOD_INFO[method]


def call_agent(agent_name: str, method: str, parameters: dict) -> dict:
    """Execute a geotechnical calculation.

    Parameters
    ----------
    agent_name : str
        One of the registered module names.
    method : str
        Method name within that module.
    parameters : dict
        Flat dict of parameters.

    Returns
    -------
    dict
        Calculation results or {"error": "..."}.
    """
    if agent_name not in MODULE_REGISTRY:
        return {"error": f"Unknown module '{agent_name}'. Available: {AGENT_NAMES}"}
    try:
        mod = _load_adapter(agent_name)
    except Exception as e:
        return {"error": f"Failed to load module '{agent_name}': {e}"}
    if method not in mod.METHOD_REGISTRY:
        available = sorted(mod.METHOD_REGISTRY.keys())
        return {"error": f"Unknown method '{method}'. Available: {available}"}
    try:
        result = mod.METHOD_REGISTRY[method](parameters)
        return result
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# ToolCall dispatch — drop-in replacement for chat_agent.agent.dispatch_tool
# ---------------------------------------------------------------------------

def dispatch_tool(tool_call) -> str:
    """Route a parsed ToolCall to the adapter registry and return JSON string."""
    name = tool_call.tool_name
    args = tool_call.arguments

    if name == "call_agent":
        result = call_agent(
            agent_name=args.get("agent_name", ""),
            method=args.get("method", ""),
            parameters=args.get("parameters", {}),
        )
    elif name == "list_methods":
        result = list_methods(
            agent_name=args.get("agent_name", ""),
            category=args.get("category", ""),
        )
    elif name == "describe_method":
        result = describe_method(
            agent_name=args.get("agent_name", ""),
            method=args.get("method", ""),
        )
    elif name == "list_agents":
        result = list_agents()
    else:
        result = {"error": f"Unknown tool '{name}'"}

    return json.dumps(result, default=str)

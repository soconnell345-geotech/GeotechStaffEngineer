"""
Backward-compatibility shim — canonical copy is now in chat_agent.agent_registry.

All symbols re-exported so existing trial_agent imports continue to work.
"""

from chat_agent.agent_registry import (  # noqa: F401
    _AGENT_SPECS,
    AGENT_NAMES,
    _loaded_agents,
    _load_agent,
    _parse_result,
    call_agent,
    list_methods,
    describe_method,
    list_agents,
)

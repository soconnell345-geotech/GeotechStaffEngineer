"""Say what to call when the model calls a module METHOD as if it were a tool.

The analysis and reference modules are reached through one tool,
``call_agent(agent_name, method, parameters)``. A model that reads "the
``subsurface`` module's ``write_diggs``" sometimes calls ``write_diggs(...)``
directly; LangGraph's tool node then answers only "Error: write_diggs is not
a valid tool, try one of [...]" (``langgraph.prebuilt.tool_node.
INVALID_TOOL_NAME_ERROR_TEMPLATE``), which lists the tools but not that the
name is a real method one call away (live smoke wave 1, F20 / A15g).

:class:`UnknownToolHint` wraps every tool call. When the tool node had no tool
of that name (``request.tool is None``) and the name is a method of a module
this agent can reach -- or a name the dispatcher already redirects to one --
the error gains the ``call_agent`` form to use. Anything else passes through
untouched.
"""

from __future__ import annotations

from typing import Iterable, Optional

from langchain_core.messages import ToolMessage

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:  # pragma: no cover - older layout
    from langchain.agents.middleware.types import AgentMiddleware

#: Method name (lower-case) -> [(module, method)], per scope.
_INDEX_CACHE: dict = {}


def _method_index(scope: Optional[frozenset]) -> dict:
    """Every method of every module in ``scope`` (``None``: all), by name."""
    key = scope
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    from funhouse_agent import dispatch
    from funhouse_agent.adapters import MODULE_REGISTRY
    index: dict = {}
    for module in MODULE_REGISTRY:
        if scope is not None and module not in scope:
            continue
        try:
            mod = dispatch._load_adapter(module)
            names = list(getattr(mod, "METHOD_REGISTRY", {}) or {})
        except Exception:                              # noqa: BLE001
            continue
        for name in names:
            index.setdefault(str(name).lower(), []).append((module, name))
    # Names the dispatcher already sends to another module's method.
    for guess, (module, method) in getattr(
            dispatch, "_CROSS_MODULE_REDIRECTS", {}).items():
        if scope is None or module in scope:
            pair = (module, method)
            if pair not in index.setdefault(guess, []):
                index[guess].append(pair)
    _INDEX_CACHE[key] = index
    return index


def method_hint(name: str, scope: Optional[Iterable[str]] = None) -> str:
    """The sentence naming the ``call_agent`` form for a tool name that is
    really a module method; ``""`` when it is not one."""
    if not name:
        return ""
    frozen = frozenset(scope) if scope is not None else None
    try:
        hits = _method_index(frozen).get(str(name).strip().lower()) or []
    except Exception:                                  # noqa: BLE001
        return ""
    if not hits:
        return ""
    forms = [f"call_agent(agent_name='{m}', method='{meth}', "
             "parameters={...})" for m, meth in hits[:3]]
    module, method = hits[0]
    return (f" '{name}' is not a tool: it is a module method, reached "
            f"through the call_agent tool -- {' or '.join(forms)}. "
            f"describe_method('{module}', '{method}') lists its parameters.")


class UnknownToolHint(AgentMiddleware):
    """Add the ``call_agent`` form to an unknown-tool error whose name is a
    module method (see the module docstring). ``scope`` is the modules this
    agent's ``call_agent`` reaches (``None``: all)."""

    def __init__(self, scope: Optional[Iterable[str]] = None):
        super().__init__()
        self.scope = frozenset(scope) if scope is not None else None

    def _hinted(self, request, result):
        if getattr(request, "tool", None) is not None:
            return result
        if not isinstance(result, ToolMessage):
            return result
        call = getattr(request, "tool_call", None) or {}
        hint = method_hint(call.get("name") or "", self.scope)
        if not hint or not isinstance(result.content, str):
            return result
        return result.model_copy(update={"content": result.content + hint})

    def wrap_tool_call(self, request, handler):
        return self._hinted(request, handler(request))

    async def awrap_tool_call(self, request, handler):
        return self._hinted(request, await handler(request))


__all__ = ["UnknownToolHint", "method_hint"]

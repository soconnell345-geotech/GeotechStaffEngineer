"""v5.0 deepagents port of the geotech agent (additive — does not touch v1).

Public API::

    from funhouse_agent.deep import build_deep_agent, make_core_tools

* :func:`build_deep_agent` — construct the compiled deep agent
  (``deepagents.create_deep_agent``) with the primary scoped to
  ``ANALYSIS_MODULES`` and ``references`` + ``reviewer`` sub-agents.
* :func:`make_core_tools` / :func:`make_vision_tools` — the LangChain tool
  factories (scoped to an ``allowed_agents`` whitelist).
* :class:`DeepNotebookChat` — the Phase-4 ipywidgets chat that drives the
  compiled graph's LangGraph streaming (imported lazily so this package import
  never requires ipywidgets).

These names load on first use (a module ``__getattr__``), not when the
package is imported, so importing one of its modules (the review page's
builds, the guards, the vision engine) does not load the geotech builder,
the setup agent or the eval harness (BACKLOG INF-30, the app split's
step 2).
"""

# MUST run before ANY langchain / langgraph / deepagents import below: on
# Databricks the runtime pre-imports a stale typing_extensions (<4.13) and
# langchain-protocol builds PEP 728 TypedDicts at import time. This reloads the
# freshly-installed typing_extensions in place so no dbutils.library.restartPython()
# is needed. See funhouse_agent/runtime_check.py.
from funhouse_agent.runtime_check import ensure_typing_extensions as _ensure_te

_ensure_te()

import importlib as _importlib
import importlib.util as _importlib_util

#: Each public name -> the module that defines it.
_LAZY = {
    "build_deep_agent": "funhouse_agent.deep.agent",
    "build_primary_tools": "funhouse_agent.deep.agent",
    "build_references_subagent": "funhouse_agent.deep.agent",
    "build_reviewer_subagent": "funhouse_agent.deep.agent",
    "render_suite_markdown": "funhouse_agent.deep.eval_harness",
    "run_suite": "funhouse_agent.deep.eval_harness",
    "write_suite_results": "funhouse_agent.deep.eval_harness",
    "DeepNotebookChat": "funhouse_agent.deep.notebook",
    "build_domain_prompt": "funhouse_agent.deep.prompt",
    "run_selfcheck": "funhouse_agent.deep.selfcheck",
    "make_core_tools": "funhouse_agent.deep.tools",
    "make_vision_tools": "funhouse_agent.deep.tools",
    "LangChainVisionEngine": "funhouse_agent.deep.vision_engine",
}


def __getattr__(name):
    """A public name from its module, or a submodule by its name (as
    ``funhouse_agent.deep.agent`` was reachable when this package imported
    it)."""
    module = _LAZY.get(name)
    if module is not None:
        value = getattr(_importlib.import_module(module), name)
        globals()[name] = value
        return value
    if not name.startswith("__"):
        try:
            spec = _importlib_util.find_spec(f"{__name__}.{name}")
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            return _importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "build_deep_agent",
    "build_primary_tools",
    "build_references_subagent",
    "build_reviewer_subagent",
    "build_domain_prompt",
    "make_core_tools",
    "make_vision_tools",
    "LangChainVisionEngine",
    "DeepNotebookChat",
    "run_selfcheck",
    "run_suite",
    "render_suite_markdown",
    "write_suite_results",
]

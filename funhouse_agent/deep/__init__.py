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
"""

from funhouse_agent.deep.agent import (
    build_deep_agent,
    build_primary_tools,
    build_references_subagent,
    build_reviewer_subagent,
)
from funhouse_agent.deep.notebook import DeepNotebookChat
from funhouse_agent.deep.prompt import build_domain_prompt
from funhouse_agent.deep.tools import make_core_tools, make_vision_tools
from funhouse_agent.deep.vision_engine import LangChainVisionEngine

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
]

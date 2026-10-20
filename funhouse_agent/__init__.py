"""
Funhouse Agent — Engine-agnostic geotechnical agent with vision.

Self-contained dispatch: routes tool calls through internal adapters
directly to 50 modules (~901 methods). No dependency on foundry/ files.

Works with any AI backend satisfying the GenAIEngine protocol:
    - PrompterAPI (Databricks/Funhouse) — works natively
    - ClaudeEngine — Anthropic SDK adapter
    - Any future backend — implement chat() + analyze_image()

Usage:
    from funhouse_agent import GeotechAgent, ClaudeEngine

    # With PrompterAPI (Databricks)
    agent = GeotechAgent(genai_engine=prompter_api)

    # With Claude
    agent = GeotechAgent(genai_engine=ClaudeEngine())

    result = agent.ask("Calculate bearing capacity of 2m footing, phi=30")

The names below load on first use (a module ``__getattr__``), not when the
package is imported, so importing one of its modules (the vision tools,
the document tools, the switches) does not load the v1 agent, its engines
or the reviewers (BACKLOG INF-30, the app split's step 2).
"""

import importlib as _importlib
import importlib.util as _importlib_util

#: Each public name -> the module that defines it.
_LAZY = {
    "GenAIEngine": "funhouse_agent.engine",
    "ClaudeEngine": "funhouse_agent.engine",
    "NativeToolEngine": "funhouse_agent.engine",
    "PrompterBridgeEngine": "funhouse_agent.engine",
    "USING_SDK_ENGINES": "funhouse_agent.engine",
    "GeotechAgent": "funhouse_agent.agent",
    "AgentResult": "funhouse_agent.react_support",
    "make_seismic_reviewer": "funhouse_agent.reviewers",
    "make_seismic_reviewer_deep": "funhouse_agent.reviewers",
    "make_foundations_reviewer": "funhouse_agent.reviewers",
    "make_foundations_reviewer_deep": "funhouse_agent.reviewers",
    "make_earth_retention_reviewer": "funhouse_agent.reviewers",
    "make_earth_retention_reviewer_deep": "funhouse_agent.reviewers",
    "make_slope_fem_reviewer": "funhouse_agent.reviewers",
    "make_slope_fem_reviewer_deep": "funhouse_agent.reviewers",
    "make_pavement_specialist": "funhouse_agent.reviewers",
    "make_pavement_specialist_deep": "funhouse_agent.reviewers",
    "make_structural_specialist": "funhouse_agent.reviewers",
    "make_structural_specialist_deep": "funhouse_agent.reviewers",
    "disclaimer": "funhouse_agent._disclaimer",
}


def __getattr__(name):
    """A public name from its module, or a submodule by its name (as
    ``funhouse_agent.agent`` was reachable when this package imported it)."""
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
    "GeotechAgent",
    "GenAIEngine",
    "ClaudeEngine",
    "NativeToolEngine",
    "PrompterBridgeEngine",
    "USING_SDK_ENGINES",
    "AgentResult",
    "make_seismic_reviewer",
    "make_seismic_reviewer_deep",
    "make_foundations_reviewer",
    "make_foundations_reviewer_deep",
    "make_earth_retention_reviewer",
    "make_earth_retention_reviewer_deep",
    "make_slope_fem_reviewer",
    "make_slope_fem_reviewer_deep",
    "make_pavement_specialist",
    "make_pavement_specialist_deep",
    "make_structural_specialist",
    "make_structural_specialist_deep",
    "disclaimer",
]

# One-time (per user) professional-use disclaimer banner on first import. Fully
# self-guarding: stderr only, suppressed by GEOTECH_NO_DISCLAIMER=1 and under
# pytest, and it never raises. See funhouse_agent/_disclaimer.py.
try:
    from funhouse_agent._disclaimer import maybe_show_first_import_notice as _show_disclaimer

    _show_disclaimer()
except Exception:  # pragma: no cover - the notice must never break importing
    pass

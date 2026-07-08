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
"""

from funhouse_agent.engine import (
    GenAIEngine,
    ClaudeEngine,
    NativeToolEngine,
    PrompterBridgeEngine,
    USING_SDK_ENGINES,
)
from funhouse_agent.agent import GeotechAgent
from funhouse_agent.react_support import AgentResult
from funhouse_agent.reviewers import (
    make_seismic_reviewer,
    make_seismic_reviewer_deep,
    make_foundations_reviewer,
    make_foundations_reviewer_deep,
    make_earth_retention_reviewer,
    make_earth_retention_reviewer_deep,
    make_slope_fem_reviewer,
    make_slope_fem_reviewer_deep,
)
from funhouse_agent._disclaimer import disclaimer

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

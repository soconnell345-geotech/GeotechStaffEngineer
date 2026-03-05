"""
Funhouse Agent — Engine-agnostic geotechnical agent with vision.

Self-contained dispatch: routes tool calls through internal adapters
directly to 16 analysis modules. No dependency on foundry/ files.

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

from funhouse_agent.engine import GenAIEngine, ClaudeEngine
from funhouse_agent.agent import GeotechAgent
from chat_agent.agent import AgentResult

__all__ = [
    "GeotechAgent",
    "GenAIEngine",
    "ClaudeEngine",
    "AgentResult",
]

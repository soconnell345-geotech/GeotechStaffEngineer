"""Live (real-API) acceptance test for the figure read-off pipeline.

Opt-in and cost-aware: SKIPPED unless BOTH are true:
  - an Anthropic key is available (shell env or Windows User env), and
  - RUN_LIVE_TESTS=1 is set.

So normal ``pytest -q`` never hits the network or spends money. Run explicitly:

    # PowerShell
    $env:RUN_LIVE_TESTS=1; .venv/Scripts/python -m pytest funhouse_agent/tests/test_live_figure_readoff.py -v -s

What it proves that the mock tests can't: a real vision model, handed the
rendered DM 7.2 Figure 4-12 page, reads a sane passive earth-pressure
coefficient off it -- and the full ReAct agent finds the chart and reads it.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from funhouse_agent.run_local import _load_api_key

_MODEL = os.environ.get("LIVE_TEST_MODEL", "claude-sonnet-4-6")

live = pytest.mark.skipif(
    not (os.environ.get("RUN_LIVE_TESTS") and _load_api_key()),
    reason="set RUN_LIVE_TESTS=1 and provide ANTHROPIC_API_KEY to run live tests",
)


def _floats(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"\d+(?:\.\d+)?", text)]


@pytest.fixture(scope="module")
def engine():
    from funhouse_agent import ClaudeEngine
    return ClaudeEngine(model=_MODEL)


@live
def test_read_off_fig_4_12_direct(engine):
    """read_reference_figure on Fig 4-12 returns a plausible Kp."""
    from funhouse_agent.vision_tools import dispatch_extended_tool

    out = json.loads(dispatch_extended_tool(
        "read_reference_figure",
        {"reference": "dm7_2", "figure_number": "4-12",
         "prompt": "Read the passive coefficient Kp for phi'=35 deg, wall "
                   "batter theta=10 deg, delta/phi'=0.66. Give a single number."},
        engine, attachments={},
    ))
    assert "error" not in out, out
    assert out["figure_number"] == "4-12"
    analysis = out["analysis"]
    flat = analysis.replace(" ", "").replace("_", "").lower()
    assert analysis and ("kp" in flat or "passive" in analysis.lower())
    # Some read-off number should land in a physically sane passive band.
    assert any(1.0 < v < 25.0 for v in _floats(analysis)), analysis


@live
def test_agent_finds_and_reads_chart(engine):
    """End-to-end: the ReAct agent locates the chart and reads it off.

    Closes the former autonomy gap: with the figure_search result stripped of
    any values and signposted with the read_reference_figure next-step, the
    agent renders and reads the chart instead of answering from memory.
    """
    from funhouse_agent import GeotechAgent
    from funhouse_agent.run_local import DEMO_QUESTION

    agent = GeotechAgent(genai_engine=engine, verbose=False)
    result = agent.ask(DEMO_QUESTION)

    assert result.answer, "empty answer"
    tools = [tc.get("tool_name") for tc in result.tool_calls]
    # The agent must render the figure rather than answer from memory.
    assert "read_reference_figure" in tools, tools
    # And the read-off should land in a physically sane passive band (not the
    # ~9 the model confabulated from memory before the signpost).
    assert any(1.0 < v < 25.0 for v in _floats(result.answer)), result.answer

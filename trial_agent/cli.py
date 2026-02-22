"""
Trial Geotech Staff Engineer Agent — local CLI using Claude API.

Usage:
    pip install anthropic
    set ANTHROPIC_API_KEY=sk-...
    python -m trial_agent.cli

The agent has access to all 30 geotechnical calculation agents via
3 meta-tools: call_agent, list_methods, describe_method.
"""

import json
import sys
import os

# Fix Windows console encoding for Unicode (Greek letters, etc.)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is on the path so foundry agents can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

from trial_agent.system_prompt import SYSTEM_PROMPT
from trial_agent.tools import TOOLS
from trial_agent.agent_registry import (
    call_agent, list_methods, describe_method, list_agents,
)

# Model to use — Sonnet 4.5 for best speed/cost balance
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 4096
MAX_TOOL_ROUNDS = 8  # Prevent runaway exploration


def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    """Route a tool call to the appropriate registry function."""
    if tool_name == "call_agent":
        return call_agent(
            tool_input["agent_name"],
            tool_input["method"],
            tool_input["parameters"],
        )
    elif tool_name == "list_methods":
        return list_methods(
            tool_input["agent_name"],
            tool_input.get("category", ""),
        )
    elif tool_name == "describe_method":
        return describe_method(
            tool_input["agent_name"],
            tool_input["method"],
        )
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def _truncate(text: str, max_len: int = 500) -> str:
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _make_system_with_cache(text: str) -> list:
    """Wrap system prompt with cache_control for prompt caching."""
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def run_conversation():
    """Run an interactive conversation with the Geotech Staff Engineer agent."""
    client = anthropic.Anthropic(timeout=120.0)
    messages = []
    system = _make_system_with_cache(SYSTEM_PROMPT)

    print("=" * 60)
    print("  Geotech Staff Engineer Agent")
    print("  30 agents | 536+ methods | All geotechnical engineering")
    print("=" * 60)
    print()
    print("Available agents:")
    agents = list_agents()
    for name, brief in sorted(agents.items()):
        print(f"  {name:20s}  {brief}")
    print()
    print("Type your question, or 'quit' to exit.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Call Claude API
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError as e:
            print(f"\n[API Error] {e}")
            messages.pop()  # Remove failed user message
            continue

        # Process response — may contain tool_use blocks that need resolution
        tool_round = 0
        while response.stop_reason == "tool_use":
            tool_round += 1
            if tool_round > MAX_TOOL_ROUNDS:
                print(f"\n  [Reached {MAX_TOOL_ROUNDS} tool rounds, stopping]")
                break

            # Collect tool results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n  [Tool] {block.name}(", end="")
                    # Show a compact version of the input
                    inp_str = json.dumps(block.input, separators=(",", ":"))
                    print(f"{_truncate(inp_str, 200)})")

                    result = dispatch_tool(block.name, block.input)

                    result_str = json.dumps(result, indent=2)
                    print(f"  [Result] {_truncate(result_str, 400)}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            # Add assistant response and tool results to message history
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            # Continue the conversation with tool results
            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    tools=TOOLS,
                    messages=messages,
                )
            except anthropic.APIError as e:
                print(f"\n[API Error] {e}")
                break

        # Print final text response
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        if text_parts:
            print(f"\n{''.join(text_parts)}")

        # Add final assistant response to history
        messages.append({"role": "assistant", "content": response.content})


def main():
    """Entry point."""
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("  set ANTHROPIC_API_KEY=sk-ant-...")
        print("  python -m trial_agent.cli")
        sys.exit(1)

    run_conversation()


if __name__ == "__main__":
    main()

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

# Ensure project root is on the path so foundry agents can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

from trial_agent.system_prompt import SYSTEM_PROMPT
from trial_agent.tools import TOOLS
from trial_agent.agent_registry import (
    call_agent, list_methods, describe_method, list_agents,
)

# Model to use — Sonnet for cost efficiency during testing
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096


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


def run_conversation():
    """Run an interactive conversation with the Geotech Staff Engineer agent."""
    client = anthropic.Anthropic()
    messages = []

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
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError as e:
            print(f"\n[API Error] {e}")
            messages.pop()  # Remove failed user message
            continue

        # Process response — may contain tool_use blocks that need resolution
        while response.stop_reason == "tool_use":
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
                    system=SYSTEM_PROMPT,
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

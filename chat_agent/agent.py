"""
GeotechChatAgent — ReAct agent for text-only chat functions.

Wraps any chat function with signature (prompt, system_prompt, temp) -> str
and gives it access to 44 geotechnical Foundry agents via <tool_call> parsing.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from chat_agent.parser import parse_response, ToolCall
from chat_agent.react_prompt import build_system_prompt
from trial_agent.agent_registry import (
    call_agent,
    list_methods,
    describe_method,
    list_agents,
)


# ---------------------------------------------------------------------------
# Conversation history manager
# ---------------------------------------------------------------------------

class ConversationHistory:
    """Formats multi-turn conversation as a single prompt string."""

    def __init__(self):
        self._turns: list[tuple[str, str]] = []  # (role, content)

    def add_user(self, text: str) -> None:
        self._turns.append(("User", text))

    def add_assistant(self, text: str) -> None:
        self._turns.append(("Assistant", text))

    def add_tool_result(self, result_text: str) -> None:
        self._turns.append(("Tool", result_text))

    def format_prompt(self) -> str:
        """Pack all turns into a single string for the chat function."""
        parts = []
        for role, content in self._turns:
            if role == "Tool":
                parts.append(f"\n<tool_result>\n{content}\n</tool_result>\n")
            else:
                parts.append(f"\n[{role}]\n{content}\n")
        # Trailing tag cues the model to continue
        parts.append("\n[Assistant]\n")
        return "".join(parts)

    def clear(self) -> None:
        self._turns.clear()

    def token_estimate(self) -> int:
        """Rough token estimate (~4 chars per token)."""
        total_chars = sum(len(c) for _, c in self._turns)
        return total_chars // 4

    def __len__(self) -> int:
        return len(self._turns)


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def dispatch_tool(tool_call: ToolCall) -> str:
    """Route a parsed ToolCall to the agent registry and return JSON string."""
    name = tool_call.tool_name
    args = tool_call.arguments

    if name == "call_agent":
        result = call_agent(
            agent_name=args.get("agent_name", ""),
            method=args.get("method", ""),
            parameters=args.get("parameters", {}),
        )
    elif name == "list_methods":
        result = list_methods(
            agent_name=args.get("agent_name", ""),
            category=args.get("category", ""),
        )
    elif name == "describe_method":
        result = describe_method(
            agent_name=args.get("agent_name", ""),
            method=args.get("method", ""),
        )
    elif name == "list_agents":
        result = list_agents()
    else:
        result = {"error": f"Unknown tool '{name}'"}

    return json.dumps(result, default=str)


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding a marker if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Result from a GeotechChatAgent.ask() call."""
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    rounds: int = 0
    total_time_s: float = 0.0
    conversation_turns: int = 0


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class GeotechChatAgent:
    """ReAct agent that gives a text-only chat function access to all Foundry agents.

    Args:
        chat_fn: Function with signature (prompt, system_prompt, temp) -> str.
        max_rounds: Maximum ReAct loop iterations (default 10).
        temperature: Temperature for the chat function (default 0.1).
        system_prompt: Custom system prompt (default: full ReAct + agent catalog).
        max_result_chars: Truncate tool results beyond this length (default 8000).
        verbose: Print each round's action to stdout (default False).
        on_tool_call: Optional callback(tool_name, arguments, result_str) for logging.
    """

    def __init__(
        self,
        chat_fn: Callable,
        max_rounds: int = 10,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
        max_result_chars: int = 8000,
        verbose: bool = False,
        on_tool_call: Optional[Callable] = None,
    ):
        self._chat_fn = chat_fn
        self._max_rounds = max_rounds
        self._temperature = temperature
        self._system_prompt = system_prompt or build_system_prompt()
        self._max_result_chars = max_result_chars
        self._verbose = verbose
        self._on_tool_call = on_tool_call
        self._history = ConversationHistory()

    def ask(self, question: str) -> AgentResult:
        """Run the ReAct loop for a user question.

        Returns AgentResult with the final answer, tool call log, and timing.
        """
        t0 = time.time()
        self._history.add_user(question)

        tool_log: list[dict] = []
        rounds = 0

        for rounds in range(1, self._max_rounds + 1):
            # Build prompt and call the chat function
            prompt = self._history.format_prompt()
            response = self._chat_fn(
                prompt, self._system_prompt, self._temperature
            )

            # Parse for tool calls
            try:
                parsed = parse_response(response)
            except ValueError as e:
                # Malformed tool call — feed error back for self-correction
                self._history.add_assistant(response)
                error_msg = json.dumps({"error": f"Parse error: {e}"})
                self._history.add_tool_result(error_msg)
                if self._verbose:
                    print(f"  Round {rounds}: PARSE ERROR — {e}")
                continue

            if parsed.tool_call is None:
                # No tool call — this is the final answer
                self._history.add_assistant(response)
                if self._verbose:
                    print(f"  Round {rounds}: FINAL ANSWER")
                return AgentResult(
                    answer=response.strip(),
                    tool_calls=tool_log,
                    rounds=rounds,
                    total_time_s=time.time() - t0,
                    conversation_turns=len(self._history),
                )

            # Execute the tool call
            tc = parsed.tool_call
            if self._verbose:
                print(f"  Round {rounds}: {tc.tool_name}("
                      f"{json.dumps(tc.arguments, default=str)[:120]})")

            result_str = dispatch_tool(tc)
            result_str = _truncate(result_str, self._max_result_chars)

            # Log
            log_entry = {
                "round": rounds,
                "tool_name": tc.tool_name,
                "arguments": tc.arguments,
                "result_preview": result_str[:200],
            }
            tool_log.append(log_entry)

            if self._on_tool_call:
                self._on_tool_call(tc.tool_name, tc.arguments, result_str)

            # Append to conversation history
            self._history.add_assistant(response)
            self._history.add_tool_result(result_str)

        # Max rounds reached — return whatever we have
        if self._verbose:
            print(f"  Max rounds ({self._max_rounds}) reached")
        return AgentResult(
            answer=(
                f"[Reached maximum of {self._max_rounds} tool rounds. "
                f"Partial analysis above may be incomplete.]"
            ),
            tool_calls=tool_log,
            rounds=rounds,
            total_time_s=time.time() - t0,
            conversation_turns=len(self._history),
        )

    def reset(self) -> None:
        """Clear conversation history to start a fresh session."""
        self._history.clear()

    @property
    def history(self) -> ConversationHistory:
        """Access the conversation history (read-only)."""
        return self._history

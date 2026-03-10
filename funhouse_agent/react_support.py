"""
ReAct support classes for the funhouse agent.

Contains AgentResult, ConversationHistory, ToolCall parsing, and helpers
that were previously imported from chat_agent. Inlined here so
funhouse_agent is fully self-contained.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Tool call parser
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)


@dataclass
class ToolCall:
    """A parsed tool invocation from the LLM response."""
    tool_name: str
    arguments: dict
    raw_json: str
    reasoning: str  # text before the <tool_call> tag


@dataclass
class ParseResult:
    """Result of parsing an LLM response."""
    tool_call: Optional[ToolCall]
    full_text: str


def parse_response(text: str, valid_tools: set = None) -> ParseResult:
    """Parse an LLM response for a <tool_call> tag.

    Parameters
    ----------
    text : str
        LLM response text.
    valid_tools : set, optional
        Allowed tool names. If None, any tool name is accepted.

    Returns ParseResult with tool_call=None if no tag found (final answer).
    Raises ValueError for malformed JSON or invalid tool_name.
    """
    match = _TOOL_CALL_RE.search(text)
    if not match:
        return ParseResult(tool_call=None, full_text=text)

    raw_json = match.group(1).strip()
    reasoning = text[:match.start()].strip()

    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in <tool_call>: {e}")

    if not isinstance(payload, dict):
        raise ValueError("Tool call JSON must be an object, not "
                         f"{type(payload).__name__}")

    tool_name = payload.get("tool_name")
    if not tool_name:
        raise ValueError("Missing 'tool_name' in tool call JSON")
    if valid_tools is not None and tool_name not in valid_tools:
        raise ValueError(
            f"Invalid tool_name '{tool_name}'. "
            f"Must be one of: {sorted(valid_tools)}"
        )

    # Extract arguments (everything except tool_name)
    arguments = {k: v for k, v in payload.items() if k != "tool_name"}

    return ParseResult(
        tool_call=ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            raw_json=raw_json,
            reasoning=reasoning,
        ),
        full_text=text,
    )


# ---------------------------------------------------------------------------
# Conversation history
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
# Agent result
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Result from an agent ask() call."""
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    rounds: int = 0
    total_time_s: float = 0.0
    conversation_turns: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding a marker if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"

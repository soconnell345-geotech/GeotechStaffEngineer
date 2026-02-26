"""
Parse <tool_call> tags from LLM text responses.

Extracts the first <tool_call>JSON</tool_call> block, validates the JSON
structure, and returns a ToolCall or None (final answer).
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

VALID_TOOLS = {"call_agent", "list_methods", "describe_method", "list_agents"}

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


def parse_response(text: str) -> ParseResult:
    """Parse an LLM response for a <tool_call> tag.

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
    if tool_name not in VALID_TOOLS:
        raise ValueError(
            f"Invalid tool_name '{tool_name}'. "
            f"Must be one of: {sorted(VALID_TOOLS)}"
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

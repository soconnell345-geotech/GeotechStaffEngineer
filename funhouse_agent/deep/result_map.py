"""A large JSON tool result opens with a map of its keys (BACKLOG TA-30).

WHY. deepagents moves a tool result of about 80,000 characters or more out
of the conversation into its scratch space and shows the model a preview:
the first five and last five LINES, each cut at 1,000 characters. A JSON
result is one line, so the preview is its first 1,000 characters. In the
2026-10-15 field session kriging returned 277,000 characters; the model saw
the start of line one, guessed a key (``field_variance``; the key is
``krige_variance``), found nothing and gave up (``module_work/
field_feedback/2026-10-15_rock-excavation-kriging-mc_v5.33.0/
TRACE_REVIEW.md`` §1).

WHAT. :class:`LargeResultKeyMap` sits inside deepagents' own filesystem
middleware, so it sees a result before it is moved. A JSON object of
:data:`MIN_CHARS` or more is written one key per line (``indent=1``) under a
short map: every top-level key, the line it starts on, what it holds and its
size, packed into the preview's first five lines. ``read_file`` with
``offset = line - 1`` then reads one key, and ``grep`` finds a key by name.
Smaller results and anything that is not a JSON object pass unchanged.
"""

from __future__ import annotations

import json
from typing import Any, List, Tuple

from langchain_core.messages import ToolMessage

try:                                           # langchain >= 1.0
    from langchain.agents.middleware import AgentMiddleware
except ImportError:                            # pragma: no cover
    from langchain.agents.middleware.types import AgentMiddleware

#: Results at least this long are mapped (deepagents moves results of about
#: 80,000 characters out of the conversation; mapping a little below that
#: keeps the map whatever its token estimate).
MIN_CHARS = 60_000
#: The preview shows this many leading lines, each cut at this many
#: characters (deepagents' ``_create_content_preview``).
PREVIEW_LINES = 5
PREVIEW_LINE_CHARS = 1_000


def _what(value: Any) -> str:
    if isinstance(value, list):
        kinds = {type(v).__name__ for v in value[:50]}
        kind = ("numbers" if kinds <= {"int", "float"} else
                "records" if kinds == {"dict"} else "items")
        return f"a list of {len(value):,} {kind}"
    if isinstance(value, dict):
        return f"an object of {len(value):,} keys"
    if isinstance(value, str):
        return "text"
    return type(value).__name__


def key_map(obj: dict) -> Tuple[List[str], str]:
    """``(header lines, body)`` for a JSON object: the body is the object one
    key per line; the header names every top-level key with the line (in
    the whole text) it starts on."""
    body = json.dumps(obj, indent=1, ensure_ascii=False, default=str)
    lines = body.splitlines()
    starts = {}
    for i, line in enumerate(lines):
        if line.startswith(' "') and '": ' in line[:400]:
            key = json.loads(line[1:line.index('": ') + 1])
            starts.setdefault(key, i)
    entries = []
    for key, value in obj.items():
        size = len(json.dumps(value, ensure_ascii=False, default=str))
        entries.append((key, starts.get(key, 0), _what(value), size))

    room = PREVIEW_LINE_CHARS - 10
    head = ("KEY MAP: this JSON result is large; its keys, the line each "
            "starts on and what it holds:")
    tail = ("Read one key: read_file on this result's saved path with "
            "offset = its line - 1 and a limit; grep finds a key by name.")
    packed: List[str] = []
    n_lines = PREVIEW_LINES          # head + packed + tail, fixed below
    for _ in range(2):               # line numbers depend on the header size
        packed = []
        cur = ""
        for key, start, what, size in entries:
            item = (f"{key} (line {start + n_lines + 1:,}, {what}, "
                    f"{size:,} characters)")
            if cur and len(cur) + 2 + len(item) > room:
                packed.append(cur)
                cur = item
            else:
                cur = f"{cur}; {item}" if cur else item
        if cur:
            packed.append(cur)
        n_lines = 2 + len(packed)
    return [head] + packed + [tail], body


def mapped_text(content: str) -> str:
    """``content`` with a key map on top when it is a large JSON object;
    otherwise unchanged."""
    if not isinstance(content, str) or len(content) < MIN_CHARS:
        return content
    text = content.lstrip()
    if not text.startswith("{"):
        return content
    try:
        obj = json.loads(text)
    except ValueError:
        return content
    if not isinstance(obj, dict) or not obj:
        return content
    header, body = key_map(obj)
    return "\n".join(header) + "\n" + body


class LargeResultKeyMap(AgentMiddleware):
    """Put a key map on top of a large JSON tool result (see the module
    docstring)."""

    def _map(self, result):
        if isinstance(result, ToolMessage) and isinstance(result.content,
                                                          str):
            new = mapped_text(result.content)
            if new is not result.content:
                return result.model_copy(update={"content": new})
        return result

    def wrap_tool_call(self, request, handler):
        return self._map(handler(request))

    async def awrap_tool_call(self, request, handler):
        return self._map(await handler(request))


__all__ = ["LargeResultKeyMap", "mapped_text", "key_map", "MIN_CHARS"]

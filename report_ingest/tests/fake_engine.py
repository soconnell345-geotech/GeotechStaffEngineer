"""An :class:`~report_ingest.engine.Engine` that replays canned replies.

The two passes take an engine and never import an SDK, so the whole of both
can be exercised offline: the prompt that is built, the tool loop, the
budget, the way a structured answer is turned into labels. A script is a
list of turns; each turn is either tool calls to make or a final answer to
parse.

It is deliberately strict where a real engine is forgiving. Running past the
end of the script raises rather than returning something plausible, because
a pass that makes one more call than the test expected is a bug the test
exists to catch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from report_ingest.engine import Reply, ToolCall, Usage, tool_use_block


class ScriptExhausted(AssertionError):
    """The pass asked for more turns than the script has."""


class FakeEngine:
    """Replays ``turns`` in order.

    A turn is one of:

    ``{"tools": [(name, arguments), ...]}``
        the model asks for those tool calls;
    ``{"final": <object>}``
        the model answers, and the object becomes ``reply.parsed``;
    ``{"text": "..."}``
        the model answers in prose with nothing parsed.

    Every call is recorded in :attr:`calls` -- the messages, the system
    prompt, the tool names offered, how many images were attached and
    whether a structured answer was asked for -- so a test can assert on
    what the pass actually sent.
    """

    def __init__(self, turns: Sequence[Dict[str, Any]],
                 name: str = "fake-model",
                 usage: Optional[Usage] = None) -> None:
        self.turns = list(turns)
        self.name = name
        self.model = name
        self.usage = usage or Usage(input_tokens=1000, output_tokens=100)
        self.calls: List[Dict[str, Any]] = []
        self._next = 0

    def complete(self, messages: Sequence[Dict[str, Any]], *,
                 system: Optional[str] = None,
                 tools: Optional[Sequence[Dict[str, Any]]] = None,
                 images: Optional[Sequence[bytes]] = None,
                 output_format: Any = None,
                 max_tokens: Optional[int] = None) -> Reply:
        self.calls.append({
            "messages": [dict(m) for m in messages],
            "system": system,
            "tools": [t["name"] for t in (tools or [])],
            "n_images": len(images or []),
            "output_format": output_format,
        })
        if self._next >= len(self.turns):
            raise ScriptExhausted(
                f"the pass made call {self._next + 1} but the script has "
                f"{len(self.turns)} turn(s)")
        turn = self.turns[self._next]
        self._next += 1

        if "tools" in turn:
            wanted = turn["tools"]
            calls = [ToolCall(f"call_{self._next}_{i}", name, dict(args))
                     for i, (name, args) in enumerate(wanted)]
            return Reply(
                text=turn.get("text", ""),
                tool_calls=calls,
                stop_reason="tool_use",
                usage=self.usage,
                model=self.model,
                seconds=0.0,
                content=([{"type": "text", "text": turn["text"]}]
                         if turn.get("text") else [])
                + [tool_use_block(c.id, c.name, c.arguments) for c in calls],
            )
        parsed = turn.get("final")
        text = turn.get("text", "")
        return Reply(
            text=text,
            tool_calls=[],
            parsed=parsed,
            stop_reason="end_turn",
            usage=self.usage,
            model=self.model,
            seconds=0.0,
            content=([{"type": "text", "text": text}] if text else []),
        )

    # -- what a test asks afterwards ---------------------------------------
    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def tool_results(self) -> List[Dict[str, Any]]:
        """Every tool-result block the pass fed back, in order."""
        out: List[Dict[str, Any]] = []
        for call in self.calls:
            for message in call["messages"]:
                content = message["content"]
                if isinstance(content, str):
                    continue
                for block in content:
                    if isinstance(block, dict) and \
                            block.get("type") == "tool_result":
                        out.append(block)
        # The same message list is resent each turn, so keep first sightings.
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for block in out:
            if block["tool_use_id"] in seen:
                continue
            seen.add(block["tool_use_id"])
            unique.append(block)
        return unique

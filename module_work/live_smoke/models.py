"""The models the harness hands the app: Claude through the Anthropic API,
or a scripted fake for the $0 dry run.

The live model is built the way ``webapp.engine_config.resolve_engine``
builds its own ``ChatAnthropic`` (same ``max_tokens`` default) plus three
harness-only settings: the API key passed in directly (never put in the
environment, never printed), ``max_retries`` so the Anthropic SDK backs off
on 429 / 529 overloaded / 5xx, and the :class:`~live_smoke.spend.SpendMeter`
attached to the model object itself.

Prompt caching: deepagents 0.6.8 already adds
``AnthropicPromptCachingMiddleware`` to the primary agent and to every
sub-agent (``deepagents/graph.py``), so the agent loop is cached with no
change here; the meter's cache_read / cache_write counts say whether it
worked. Vision side calls (one image each) are not cached and gain nothing
from it.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Callable, Dict, List, Optional

#: The models a wave may use (the owner's list; each has a price on file).
LIVE_MODELS = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5")

#: Anthropic SDK retries (429, 529 overloaded, 5xx, connection errors) with
#: exponential backoff that honours ``retry-after``.
MAX_RETRIES = 8
REQUEST_TIMEOUT_S = 600.0


def api_key() -> str:
    """The Anthropic key from the process env, else the Windows USER env.
    Never printed, logged or written anywhere."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key.strip()
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "[Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY',"
             "'User')"], capture_output=True, text=True, timeout=30)
        return (out.stdout or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def build_claude(model_id: str, meter, *, max_retries: int = MAX_RETRIES,
                 timeout: float = REQUEST_TIMEOUT_S):
    """``ChatAnthropic`` for ``model_id`` with the meter attached."""
    if model_id not in LIVE_MODELS:
        raise ValueError(f"model {model_id!r} is not one of {LIVE_MODELS}")
    key = api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (process or Windows "
                           "User environment)")
    from langchain_anthropic import ChatAnthropic
    from webapp.engine_config import _default_max_tokens
    return ChatAnthropic(model=model_id, max_tokens=_default_max_tokens(),
                         api_key=key, max_retries=int(max_retries),
                         default_request_timeout=float(timeout),
                         callbacks=[meter])


# ---------------------------------------------------------------------------
# The scripted fake (dry runs, tests)
# ---------------------------------------------------------------------------

def _scripted_class():
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ScriptedChatModel(BaseChatModel):
        """A tool-calling chat model that follows a script.

        ``policy(messages) -> AIMessage`` decides every reply. The default
        policy answers plainly. :func:`turn_script_policy` builds the usual
        one: a list of steps per user turn, where the step is the number of
        AI messages after the last human message.
        """

        policy: Any = None
        model: str = "scripted-fake"
        usage: Optional[dict] = None
        calls: List[int] = []

        @property
        def _llm_type(self) -> str:
            return "scripted-fake"

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            self.calls.append(len(messages))
            pol = self.policy or (lambda msgs: AIMessage(
                content="Scripted answer (dry run)."))
            msg = pol(messages)
            if self.usage and not getattr(msg, "usage_metadata", None):
                msg.usage_metadata = dict(self.usage)
            return ChatResult(generations=[ChatGeneration(message=msg)])

    return ScriptedChatModel


def scripted_model(policy: Optional[Callable] = None, *,
                   usage: Optional[dict] = None, callbacks=None):
    """A :class:`ScriptedChatModel` (see :func:`_scripted_class`)."""
    cls = _scripted_class()
    return cls(policy=policy, usage=usage, calls=[],
               callbacks=list(callbacks or []))


def _is_vision_call(messages) -> bool:
    """A one-message call carrying an image (vision side call / probe)."""
    if len(messages) != 1:
        return False
    content = getattr(messages[0], "content", None)
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") in ("image_url", "image")
        for b in content)


def turn_script_policy(turns: List[List[Any]],
                       vision_text: str = "A drawing sheet (scripted).",
                       fallback: str = "Scripted answer (dry run).") -> Callable:
    """A policy from ``turns[i] = [step0, step1, ...]`` for user turn ``i``.

    A step is an ``AIMessage``, a dict ``{"content": str, "tool_calls":
    [...]}``, or a callable ``(messages) -> AIMessage``. The turn index is
    the number of real user messages in the history minus one (the app
    replays every earlier user turn); the step is the number of AI messages
    since the last human message. Vision side calls get ``vision_text``.
    """
    from langchain_core.messages import AIMessage

    counter = {"n": 0}

    def policy(messages):
        if _is_vision_call(messages):
            return AIMessage(content=vision_text)
        humans = [i for i, m in enumerate(messages)
                  if getattr(m, "type", "") == "human"]
        turn = max(0, len(humans) - 1)
        last_h = humans[-1] if humans else -1
        step = sum(1 for m in messages[last_h + 1:]
                   if getattr(m, "type", "") == "ai")
        steps = turns[turn] if turn < len(turns) else []
        if step >= len(steps):
            return AIMessage(content=fallback)
        spec = steps[step]
        if callable(spec):
            return spec(messages)
        if isinstance(spec, AIMessage):
            return spec
        counter["n"] += 1
        calls = []
        for j, tc in enumerate(spec.get("tool_calls") or []):
            calls.append({"name": tc["name"], "args": dict(tc.get("args")
                                                           or {}),
                          "id": tc.get("id") or f"call_{counter['n']}_{j}"})
        return AIMessage(content=spec.get("content", ""), tool_calls=calls)

    return policy


__all__ = ["LIVE_MODELS", "api_key", "build_claude", "scripted_model",
           "turn_script_policy", "MAX_RETRIES"]

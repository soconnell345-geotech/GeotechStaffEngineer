"""Live one-turn harness for the v5.0 deepagents port.

Proves ONE real tool-calling turn end-to-end: builds the deep agent on a real
chat model, asks a fixed bearing-capacity question, and checks that the agent
invoked ``call_agent`` against the ``bearing_capacity`` module and that a numeric
bearing capacity appears in the final answer.

.. warning::
   ``run()`` makes REAL LLM API calls (it spends API budget). It is intended to
   be run by hand or from a Databricks notebook — NOT from the offline test
   suite. The offline tests only import this module.

Backends
--------
``--backend anthropic`` (local, CLI)
    Builds ``ChatAnthropic(model=...)`` from ``langchain_anthropic`` and reads
    ``ANTHROPIC_API_KEY`` from the environment. Run from the worktree root::

        .venv/Scripts/python.exe -m funhouse_agent.deep.run_live --backend anthropic
        .venv/Scripts/python.exe -m funhouse_agent.deep.run_live --backend anthropic --model claude-opus-4-8

``--backend databricks`` (notebook)
    A PrompterAPI cannot be constructed off-Databricks, so the CLI path for this
    backend only explains how to wire it. In a Databricks notebook, build the
    model yourself and call :func:`run` directly::

        from funhouse_agent.deep.databricks_bridge import PrompterChatModel
        from funhouse_agent.deep.run_live import run

        model = PrompterChatModel(prompter=fh_prompter)   # fh_prompter = PrompterAPI(...)
        run(model)

    ``build_deep_agent`` default-wraps the model for vision, so a single
    ``PrompterChatModel`` drives text, tools, and vision.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from funhouse_agent.deep.agent import build_deep_agent

DEFAULT_MODEL = "claude-sonnet-4-6"

QUESTION = (
    "What is the ultimate bearing capacity of a 2.0 m wide strip footing at "
    "1.5 m depth in sand with friction angle 30 degrees and unit weight "
    "18 kN/m3? Use the tools."
)


# ---------------------------------------------------------------------------
# Trace extraction helpers (work on the LangGraph message list)
# ---------------------------------------------------------------------------

def _iter_messages(result) -> list:
    """Pull the message list out of a deepagents/LangGraph invoke result."""
    if isinstance(result, dict):
        return result.get("messages", []) or []
    return getattr(result, "messages", []) or []


def _tool_calls(messages) -> list:
    """Flatten all assistant tool calls across the message trace.

    Returns a list of ``(name, args)`` tuples. Handles the LangChain
    ``AIMessage.tool_calls`` shape (``{name, args, id}``).
    """
    calls = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            if isinstance(tc, dict):
                calls.append((tc.get("name"), tc.get("args", {})))
    return calls


def _final_text(messages) -> str:
    """Text of the last assistant message in the trace."""
    for m in reversed(messages):
        # An AIMessage with content and no tool calls is the final answer.
        if getattr(m, "type", None) == "ai" or type(m).__name__ == "AIMessage":
            content = getattr(m, "content", "")
            text = content if isinstance(content, str) else _join_blocks(content)
            if text.strip():
                return text
    return ""


def _join_blocks(content) -> str:
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )
    return str(content)


def _has_number(text: str) -> bool:
    """True if a numeric token appears in the text."""
    import re
    return bool(re.search(r"\d", text or ""))


def _print_trace(messages) -> None:
    print("-" * 70)
    print("MESSAGE / TOOL TRACE")
    print("-" * 70)
    for i, m in enumerate(messages):
        kind = type(m).__name__
        tcs = getattr(m, "tool_calls", None) or []
        if tcs:
            for tc in tcs:
                name = tc.get("name") if isinstance(tc, dict) else None
                args = tc.get("args") if isinstance(tc, dict) else None
                preview = json.dumps(args, default=str)[:160]
                print(f"[{i:02d}] {kind} -> tool_call {name}({preview})")
        else:
            content = getattr(m, "content", "")
            text = content if isinstance(content, str) else _join_blocks(content)
            label = getattr(m, "name", None) or kind
            print(f"[{i:02d}] {label}: {text[:200].strip()}")
    print("-" * 70)


# ---------------------------------------------------------------------------
# The runnable harness
# ---------------------------------------------------------------------------

def run(model, question: str = QUESTION, *, verbose: bool = True) -> bool:
    """Build the deep agent on ``model`` and prove one real tool-calling turn.

    Parameters
    ----------
    model : BaseChatModel
        A real LangChain chat model (e.g. ``ChatAnthropic`` or a
        :class:`~funhouse_agent.deep.databricks_bridge.PrompterChatModel`).
        **This makes real API calls.**
    question : str
        The prompt to send. Defaults to the fixed bearing-capacity question.
    verbose : bool
        Print the message/tool trace and a pass/fail summary.

    Returns
    -------
    bool
        ``True`` if a ``call_agent`` against ``bearing_capacity`` was invoked
        and a numeric bearing capacity appears in the final answer.
    """
    agent = build_deep_agent(model=model)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = _iter_messages(result)

    calls = _tool_calls(messages)
    final = _final_text(messages)

    called_bearing = any(
        name == "call_agent"
        and isinstance(args, dict)
        and args.get("agent_name") == "bearing_capacity"
        for name, args in calls
    )
    final_has_number = _has_number(final)
    ok = called_bearing and final_has_number

    if verbose:
        _print_trace(messages)
        print(f"call_agent -> bearing_capacity invoked: {called_bearing}")
        print(f"numeric bearing capacity in final answer: {final_has_number}")
        print("=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)
        print(final)
        print("=" * 70)
        print("RESULT:", "PASS" if ok else "FAIL")

    if not called_bearing:
        raise AssertionError(
            "Expected a call_agent tool call against 'bearing_capacity'; "
            f"got tool calls: {calls}"
        )
    if not final_has_number:
        raise AssertionError(
            "Expected a numeric bearing capacity in the final answer; "
            f"got: {final!r}"
        )
    return ok


def _build_anthropic_model(model_id: str):
    """Construct a real ChatAnthropic model (reads ANTHROPIC_API_KEY)."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:  # pragma: no cover - env-dependent
        raise SystemExit(
            "langchain_anthropic is required for --backend anthropic: "
            f"{e}\n  pip install langchain-anthropic"
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Export it before running "
            "--backend anthropic."
        )
    return ChatAnthropic(model=model_id, max_tokens=4096, temperature=0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Live one-turn harness for the v5.0 deepagents port "
                    "(makes REAL API calls)."
    )
    parser.add_argument(
        "--backend", choices=["anthropic", "databricks"], default="anthropic",
        help="anthropic: build ChatAnthropic locally (ANTHROPIC_API_KEY). "
             "databricks: notebook-only — inject a PrompterChatModel via run().",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Chat model id for --backend anthropic (default {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--question", default=QUESTION, help="Override the question.",
    )
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if args.backend == "databricks":
        print(
            "The databricks backend cannot be constructed from the CLI: a "
            "Funhouse PrompterAPI only exists inside Databricks.\n\n"
            "In a Databricks notebook, run:\n\n"
            "    from funhouse_agent.deep.databricks_bridge import "
            "PrompterChatModel\n"
            "    from funhouse_agent.deep.run_live import run\n\n"
            "    model = PrompterChatModel(prompter=fh_prompter)\n"
            "    run(model)\n",
            file=sys.stderr,
        )
        return 2

    model = _build_anthropic_model(args.model)
    print(f"[deep run_live | backend=anthropic model={args.model}]")
    print(f"Q: {args.question}")
    try:
        ok = run(model, question=args.question)
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the geotech agent locally against the real Anthropic API.

A standalone driver for exercising GeotechAgent on your own machine -- no
Funhouse, no Foundry. It uses ClaudeEngine, which talks straight to the
Anthropic API, so you can poke at the tools (including the new figure read-off)
exactly the way a user would.

Requires
--------
- ``pip install anthropic``
- ``ANTHROPIC_API_KEY`` in your shell env, or as a Windows *User* environment
  variable (set once with::

      [Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY','sk-ant-...','User')

  then open a new shell). This script reads either.

Usage
-----
    # one-shot question
    .venv/Scripts/python funhouse_agent/run_local.py "Bearing capacity of a 2m strip footing, phi=30, gamma=18?"

    # interactive REPL (no argument)
    .venv/Scripts/python funhouse_agent/run_local.py

    # the built-in figure read-off demo (find chart -> read value off it)
    .venv/Scripts/python funhouse_agent/run_local.py --demo

    # turn on the reference reviewer second-pass, on Opus
    .venv/Scripts/python funhouse_agent/run_local.py --review --model claude-opus-4-8 "..."
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

DEFAULT_MODEL = "claude-sonnet-4-6"

# A question that forces the whole figure pipeline: the agent must locate the
# right chart (figure_db.figure_search) and then read a value off it
# (read_reference_figure), rather than being handed the figure number.
DEMO_QUESTION = (
    "Using the DM7 design charts, find the figure giving active/passive earth "
    "pressure coefficients by the log spiral method for a sloping wall with a "
    "horizontal backfill, and read off the passive coefficient Kp for "
    "phi'=35 degrees, wall batter theta=10 degrees, and delta/phi'=0.66. "
    "Tell me which figure you used and show your reasoning."
)


def _load_api_key() -> str | None:
    """Return the Anthropic key from the shell env, or the Windows User env."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command",
                 "[Environment]::GetEnvironmentVariable("
                 "'ANTHROPIC_API_KEY','User')"],
                capture_output=True, text=True, timeout=15,
            )
            key = out.stdout.strip()
            if key:
                os.environ["ANTHROPIC_API_KEY"] = key
                return key
        except Exception:
            pass
    return None


def _print_result(result) -> None:
    print("-" * 70)
    print(f"rounds: {result.rounds} | tool calls: {len(result.tool_calls)} | "
          f"time: {result.total_time_s:.1f}s")
    print("=" * 70)
    print(result.answer)
    print("=" * 70)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run GeotechAgent locally via the Anthropic API.")
    parser.add_argument("question", nargs="?",
                        help="Question to ask. Omit for an interactive REPL.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Claude model id (default {DEFAULT_MODEL}). "
                             "Opus reads dense charts more reliably.")
    parser.add_argument("--review", action="store_true",
                        help="Enable the reference reviewer second-pass.")
    parser.add_argument("--quiet", action="store_true",
                        help="Hide the ReAct tool trace.")
    parser.add_argument("--demo", action="store_true",
                        help="Ask the built-in figure read-off demo question.")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not _load_api_key():
        print("ERROR: ANTHROPIC_API_KEY not found in the shell or Windows User "
              "environment.\nSet it once (PowerShell), then open a new shell:\n"
              "  [Environment]::SetEnvironmentVariable("
              "'ANTHROPIC_API_KEY','sk-ant-...','User')",
              file=sys.stderr)
        return 2

    try:
        from funhouse_agent import GeotechAgent, ClaudeEngine
    except ImportError as e:
        print(f"ERROR: {e}\nInstall the SDK:  pip install anthropic",
              file=sys.stderr)
        return 2

    engine = ClaudeEngine(model=args.model)
    agent = GeotechAgent(genai_engine=engine, review=args.review,
                         verbose=not args.quiet)
    print(f"[ClaudeEngine model={args.model} | review={args.review}]")

    question = args.question or (DEMO_QUESTION if args.demo else None)
    if question:
        print(f"\nQ: {question}\n" + "-" * 70)
        _print_result(agent.ask(question))
        return 0

    # Interactive REPL
    print("Interactive mode — type a question, or 'exit' to quit.")
    print(f"Tip — try the figure read-off:\n  {DEMO_QUESTION}")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            break
        try:
            _print_result(agent.ask(q))
        except Exception as e:  # noqa: BLE001 — keep the REPL alive
            print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

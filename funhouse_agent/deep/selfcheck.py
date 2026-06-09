"""One-call self-check for the v5.0 deepagents agent (makes REAL API calls).

A notebook-friendly "is it working?" check — the shipped, non-developer
equivalent of the offline pytest suite (which is NOT included in the wheel).
It runs the live proofs against a real chat model and prints a clear PASS/FAIL
summary.

Checks
------
- **agent_tool_call** — the agent drives a real tool call and answers a
  bearing-capacity question (via :func:`funhouse_agent.deep.run_live.run`).
- **cross_session_memory** — a fact saved in one session is recalled in a new
  session sharing the same store (via
  :func:`funhouse_agent.deep.agent.run_memory_demo`).

Notebook usage (Funhouse / Databricks)::

    from funhouse_agent.deep.databricks_bridge import PrompterChatModel
    from funhouse_agent.deep.selfcheck import run_selfcheck

    model = PrompterChatModel(prompter=fh_prompter, model="funhouse-gpt-high")
    run_selfcheck(model)               # ~3 model calls (a few cents)

Local CLI (with ``ANTHROPIC_API_KEY`` set)::

    python -m funhouse_agent.deep.selfcheck --backend anthropic

.. warning::
   Every check makes REAL LLM API calls. This is NOT part of the offline test
   suite — the offline tests only exercise the pure summary formatter.
"""

from __future__ import annotations

import argparse
import sys


CHECK_NAMES = ("agent_tool_call", "cross_session_memory")


def _format_summary(results: dict) -> str:
    """Render a human-readable PASS/FAIL summary of a self-check ``results`` dict.

    Pure function (no model / no I/O) so it is unit-testable offline.

    Parameters
    ----------
    results : dict
        ``{check_name: {"ok": bool, "error": str optional}}``.

    Returns
    -------
    str
        A multi-line summary ending in ``N/M checks passed``.
    """
    passed = sum(1 for r in results.values() if r.get("ok"))
    total = len(results)
    lines = ["", "=" * 60, "  v5.0 deep-agent self-check", "=" * 60]
    for name, r in results.items():
        mark = "PASS" if r.get("ok") else "FAIL"
        line = f"  [{mark}] {name}"
        if not r.get("ok") and r.get("error"):
            line += f"  -- {r['error']}"
        lines.append(line)
    lines.append("-" * 60)
    verdict = "ALL GOOD" if passed == total and total else "SEE FAILURES ABOVE"
    lines.append(f"  {passed}/{total} checks passed  ({verdict})")
    lines.append("=" * 60)
    return "\n".join(lines)


def run_selfcheck(model, *, include_memory: bool = True,
                  verbose: bool = True) -> dict:
    """Run the live self-check against a real chat ``model`` and summarize.

    Each check is isolated in its own ``try`` so one failure does not abort the
    rest; a failing check records its error string. **Makes real API calls**
    (roughly one model call for the tool check plus two for the memory check).

    Parameters
    ----------
    model : BaseChatModel
        A real LangChain chat model — e.g. a
        :class:`~funhouse_agent.deep.databricks_bridge.PrompterChatModel`
        (Funhouse) or ``langchain_anthropic.ChatAnthropic``.
    include_memory : bool
        Also run the cross-session memory check (2 extra model calls).
    verbose : bool
        Print the summary.

    Returns
    -------
    dict
        ``{check_name: {"ok": bool, "error": str optional}}``.
    """
    results: dict = {}

    # 1) The agent drives a real tool call and answers.
    try:
        from funhouse_agent.deep.run_live import run
        ok = run(model, verbose=False)
        results["agent_tool_call"] = {"ok": bool(ok)}
    except Exception as exc:  # noqa: BLE001 - surface any failure as a check FAIL
        results["agent_tool_call"] = {
            "ok": False, "error": f"{type(exc).__name__}: {exc}",
        }

    # 2) Cross-session memory: save in one thread, recall in another.
    if include_memory:
        try:
            from langgraph.store.memory import InMemoryStore
            from langgraph.checkpoint.memory import InMemorySaver
            from funhouse_agent.deep.agent import run_memory_demo
            ok = run_memory_demo(
                model, InMemoryStore(), InMemorySaver(), verbose=False,
            )
            results["cross_session_memory"] = {"ok": bool(ok)}
        except Exception as exc:  # noqa: BLE001
            results["cross_session_memory"] = {
                "ok": False, "error": f"{type(exc).__name__}: {exc}",
            }

    if verbose:
        print(_format_summary(results))
    return results


def _build_anthropic_model(model_id: str):
    """Construct a real ChatAnthropic model (reads ANTHROPIC_API_KEY)."""
    import os
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
        description="Live self-check for the v5.0 deepagents agent "
                    "(makes REAL API calls).",
    )
    parser.add_argument(
        "--backend", choices=["anthropic", "databricks"], default="anthropic",
        help="anthropic: build ChatAnthropic locally (ANTHROPIC_API_KEY). "
             "databricks: notebook-only — call run_selfcheck(model) directly.",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Chat model id for --backend anthropic.",
    )
    parser.add_argument(
        "--no-memory", action="store_true",
        help="Skip the cross-session memory check (cheaper).",
    )
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if args.backend == "databricks":
        print(
            "The databricks backend cannot be constructed from the CLI.\n\n"
            "In a Databricks notebook, run:\n\n"
            "    from funhouse_agent.deep.databricks_bridge import "
            "PrompterChatModel\n"
            "    from funhouse_agent.deep.selfcheck import run_selfcheck\n\n"
            "    model = PrompterChatModel(prompter=fh_prompter, "
            "model='funhouse-gpt-high')\n"
            "    run_selfcheck(model)\n",
            file=sys.stderr,
        )
        return 2

    model = _build_anthropic_model(args.model)
    results = run_selfcheck(model, include_memory=not args.no_memory)
    ok = all(r.get("ok") for r in results.values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

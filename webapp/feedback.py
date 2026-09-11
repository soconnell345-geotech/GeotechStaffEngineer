"""Feedback capture — from the user AND from the agent — saved WITH the
conversation.

Owner feedback 2026-09-11: "Give the app instructions and tools to create
feedback files based on user input or its own instances of lacking
capabilities. Feedback files would get saved in the specific conversation
folder." Until now the only feedback path was the owner exporting a
conversation and writing a Word doc by hand.

Two entry points, one writer:

* the sidebar **Feedback** box (``webapp/app.py``) — the user types a note,
  it is recorded with ``source="user"``;
* the ``record_feedback`` agent tool — the agent records a capability gap
  ("no tool draws a p-y curve"), a tool error it could not work around, or
  feedback the user gave in chat, with ``source="agent"``.

Both land in the conversation directory, next to ``transcript.jsonl``:

    <conversation>/feedback.jsonl   one JSON object per entry (machine)
    <conversation>/FEEDBACK.md      the same entries, readable (what the
                                    owner opens on SharePoint)

The SharePoint mirror walks the whole conversation directory
(``sharepoint_store``), so both files reach permanent storage after the next
turn or "Sync now" — nothing to register. Triage reads them first
(``module_work/field_feedback/README.md``).

The tool is built PER CONVERSATION with the directory closed over
(:func:`make_record_feedback_tool`) — no process-global "current thread"
state — and is injected through ``build_deep_agent(extra_tools=...)`` for the
primary and ``calc_extra_tools=`` for the calc sub-agent, where "there is no
tool for this" is most often discovered. Like every other agent-facing tool
in the web app it returns a plain string and never raises.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

JSONL_NAME = "feedback.jsonl"
MD_NAME = "FEEDBACK.md"

SOURCES = ("user", "agent")
KINDS = ("user_feedback", "capability_gap", "tool_error", "other")

#: Cap on free text per entry — feedback is a note, not a transcript.
MAX_DETAILS_CHARS = 8000
MAX_SUMMARY_CHARS = 300


def _app_version() -> Optional[str]:
    try:
        import importlib.metadata as _md
        return _md.version("geotech-staff-engineer")
    except Exception:                                  # noqa: BLE001
        return None


def _norm_kind(kind: Optional[str]) -> str:
    k = (kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    return k if k in KINDS else "other"


def _norm_source(source: Optional[str]) -> str:
    s = (source or "").strip().lower()
    return s if s in SOURCES else "agent"


def render_md(entry: dict) -> str:
    """One FEEDBACK.md section for ``entry`` (as returned by :func:`record`)."""
    ctx = entry.get("context") or {}
    ctx_line = ", ".join(f"{k}={v}" for k, v in sorted(ctx.items())
                         if v not in (None, ""))
    lines = [
        f"## {entry.get('iso', '')} · {entry.get('source', '?')} · "
        f"{entry.get('kind', '?')}",
        "",
        f"**{entry.get('summary', '').strip()}**",
    ]
    details = (entry.get("details") or "").strip()
    if details:
        lines += ["", details]
    if ctx_line:
        lines += ["", f"_{ctx_line}_"]
    lines += ["", ""]
    return "\n".join(lines)


def record(conv_dir: str, *, source: str, kind: str, summary: str,
           details: str = "", context: Optional[dict] = None) -> dict:
    """Append one feedback entry to ``<conv_dir>/feedback.jsonl`` and
    ``<conv_dir>/FEEDBACK.md``. Returns the entry written.

    ``summary`` is required (a blank summary raises ``ValueError`` — the
    tool wrapper turns that into a readable string). Unknown ``kind`` values
    are kept as ``other``; unknown ``source`` values as ``agent``.
    """
    summary = " ".join((summary or "").split())
    if not summary:
        raise ValueError("feedback needs a one-line summary")
    now = time.time()
    entry = {
        "ts": now,
        "iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "source": _norm_source(source),
        "kind": _norm_kind(kind),
        "summary": summary[:MAX_SUMMARY_CHARS],
        "details": (details or "")[:MAX_DETAILS_CHARS],
        "context": dict(context or {}),
    }
    entry["context"].setdefault("app_version", _app_version())
    os.makedirs(conv_dir, exist_ok=True)
    with open(os.path.join(conv_dir, JSONL_NAME), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    md_path = os.path.join(conv_dir, MD_NAME)
    header = "" if os.path.isfile(md_path) else (
        "# Feedback recorded in this conversation\n\n"
        "Entries from the sidebar Feedback box (`user`) and from the agent's "
        "`record_feedback` tool (`agent`: capability gaps, tool errors, "
        "feedback given in chat). Machine copy: `feedback.jsonl`.\n\n")
    with open(md_path, "a", encoding="utf-8") as fh:
        fh.write(header + render_md(entry))
    return entry


def load(conv_dir: str) -> list:
    """All entries in ``<conv_dir>/feedback.jsonl`` (oldest first); ``[]`` if
    none. Malformed lines are skipped."""
    path = os.path.join(conv_dir, JSONL_NAME)
    out: list = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return out


# ---------------------------------------------------------------------------
# Agent tool
# ---------------------------------------------------------------------------

FEEDBACK_PROMPT = (
    "FEEDBACK: you have a `record_feedback` tool. Call it — once, briefly, "
    "then carry on with the task — when (a) the user comments on the app "
    "itself (what it did well or badly, what they wish it did), (b) the task "
    "needs a tool or capability you do not have (no method for the analysis, "
    "no way to draw the figure asked for, a file type you cannot read), or "
    "(c) a tool fails in a way you cannot work around. kind = user_feedback | "
    "capability_gap | tool_error | other. The note is saved with this "
    "conversation for the developers; it never replaces your answer, and it "
    "is not for ordinary engineering findings.")


def make_record_feedback_tool(conv_dir: str,
                              context_fn: Optional[Callable[[], dict]] = None):
    """Build the ``record_feedback`` LangChain tool bound to ``conv_dir``.

    ``context_fn`` (optional) is called at record time and its dict merged
    into the entry's ``context`` (the app passes the turn index + model).
    """
    from langchain_core.tools import tool

    @tool
    def record_feedback(kind: str, summary: str, details: str = "") -> str:
        """Record feedback for the developers, saved with this conversation.

        Use when the user comments on the app itself, when you LACK a tool
        or capability the task needs, or when a tool fails in a way you
        cannot work around. Call it once, briefly, then continue the task.

        Parameters
        ----------
        kind : str
            One of ``user_feedback``, ``capability_gap``, ``tool_error``,
            ``other``.
        summary : str
            One line: what is missing / what went wrong / what the user said.
        details : str, optional
            What you were trying to do, the exact tool/method/inputs, the
            error text, or the user's words. A short paragraph is plenty.
        """
        try:
            ctx: dict = {}
            if context_fn is not None:
                try:
                    ctx = dict(context_fn() or {})
                except Exception:                      # noqa: BLE001
                    ctx = {}
            entry = record(conv_dir, source="agent", kind=kind,
                           summary=summary, details=details, context=ctx)
            return (f"Feedback recorded ({entry['kind']}): {entry['summary']} "
                    "— saved with this conversation. Continue with the task.")
        except Exception as exc:                       # noqa: BLE001
            return (f"Feedback could not be recorded ({type(exc).__name__}: "
                    f"{exc}). Continue with the task; mention the gap in your "
                    "answer instead.")

    return record_feedback


def tools_for(conv_dir: str,
              context_fn: Optional[Callable[[], dict]] = None) -> tuple:
    """``(tools, prompt)`` for a conversation — mirrors
    ``sharepoint_tools.tools_if_configured`` so ``core.build_agent`` can
    splice it in the same way."""
    return [make_record_feedback_tool(conv_dir, context_fn)], FEEDBACK_PROMPT


__all__ = ["record", "load", "render_md", "make_record_feedback_tool",
           "tools_for", "FEEDBACK_PROMPT", "KINDS", "SOURCES",
           "JSONL_NAME", "MD_NAME"]

"""Detached turn execution — analyses that survive websocket death.

Live evidence (2026-09-03, owner DevTools + the funhouse dev's probe): the
Databricks driver proxy enforces a HARD ~60-second websocket lifetime. No
keepalive — protocol pings, app-level data frames, bidirectional traffic —
prevents it; every socket dies at the same round interval and Streamlit
reconnects a moment later. Any turn longer than the TTL therefore crossed a
socket death, and because the turn ran IN the script thread, the disconnect
killed the analysis mid-flight ("the question dropped").

This module flips the architecture: the ENTIRE turn pipeline — streaming the
agent, checkpointing partials, artifact diffing, transcript/messages
persistence, tracing, the SharePoint mirror — runs in a background worker
thread that never touches Streamlit. The page merely FOLLOWS the job's
recorded events and re-attaches after every reconnect: socket deaths now
cost a one-second display blink, never the work.

Jobs are process-global (one Streamlit server process hosts every session),
keyed by conversation thread_id, one active job per conversation.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from webapp import core

__all__ = ["TurnJob", "start_turn_job", "get_turn_job"]

_JOBS: dict[str, "TurnJob"] = {}
_JOBS_LOCK = threading.Lock()


class TurnJob:
    """One running (or finished, not-yet-consumed) turn for a conversation."""

    def __init__(self, thread_id: str, prompt: str):
        self.thread_id = thread_id
        self.prompt = prompt
        self.started_at = time.time()
        self.events: list[dict] = []
        self.done = False
        self.consumed = False          # a session folded the result into its UI
        self.result: dict = {}
        self._lock = threading.Lock()

    # -- worker side ------------------------------------------------------
    def _add(self, item: dict) -> None:
        with self._lock:
            self.events.append(item)

    def _finish(self, **result) -> None:
        with self._lock:
            self.result = result
            self.done = True

    # -- UI side ----------------------------------------------------------
    def follow(self, start: int = 0, poll_s: float = 0.5):
        """Yield events from index ``start``; return once done AND drained.

        Called from a Streamlit script thread. If that thread dies with its
        websocket, the job is unaffected — the next script run re-follows.
        """
        i = start
        while True:
            with self._lock:
                chunk = self.events[i:]
                finished = self.done
            for item in chunk:
                yield item
            i += len(chunk)
            if finished and i >= len(self.events):
                return
            time.sleep(poll_s)


def get_turn_job(thread_id: str) -> Optional[TurnJob]:
    """The active or finished-unconsumed job for a conversation, if any."""
    with _JOBS_LOCK:
        job = _JOBS.get(thread_id)
    if job is not None and job.done and job.consumed:
        return None
    return job


def start_turn_job(agent, messages: list, thread_id: str,
                   recursion_limit, ctx: dict) -> TurnJob:
    """Start the detached turn worker (no-op returns the existing active job).

    ``ctx`` carries everything the post-turn pipeline needs, resolved by the
    submitting session BEFORE the socket can die:
      prompt, temp_dir, before (files snapshot), staged_inputs,
      working_dir, before_wd (or None), artifacts (the session list, mutated
      in place), artifacts_before_len, transcript (session list, mutated),
      trace_on (bool), model, behavior (dict).
    """
    with _JOBS_LOCK:
        existing = _JOBS.get(thread_id)
        if existing is not None and not existing.done:
            return existing
        job = TurnJob(thread_id, ctx.get("prompt") or "")
        _JOBS[thread_id] = job
    worker = threading.Thread(
        target=_run_turn_job, name=f"turn-{thread_id[:8]}",
        args=(job, agent, messages, thread_id, recursion_limit, ctx),
        daemon=True)
    worker.start()
    return job


def _run_turn_job(job: TurnJob, agent, messages: list, thread_id: str,
                  recursion_limit, ctx: dict) -> None:
    """The entire turn pipeline, Streamlit-free. Never raises."""
    answer = ""
    final = ""
    turn_tokens = 0
    turn_error = None
    _chunks = 0
    trace_t0 = time.time()
    trace_tools: list[dict] = []
    try:
        for item in core.with_heartbeat(core.stream_turn(
                agent, messages, thread_id,
                recursion_limit=recursion_limit)):
            kind = item.get("kind")
            job._add(item)
            if kind == "token":
                answer += item.get("text", "")
                _chunks += 1
                if _chunks % 8 == 0:
                    core.checkpoint_partial(thread_id, answer)
            elif kind in ("tool_call", "todos", "tool_result"):
                core.checkpoint_partial(thread_id, answer)
                if ctx.get("trace_on") and kind == "tool_call":
                    trace_tools.append(
                        {"t": round(time.time() - trace_t0, 3),
                         "call": (item.get("text") or "")[:80]})
            elif kind == "turn_done":
                final = item.get("answer", "")
                turn_tokens = item.get("turn_tokens", 0)
    except Exception as exc:                      # noqa: BLE001
        try:
            turn_error = core.friendly_turn_error(exc)
        except Exception:
            turn_error = f"{type(exc).__name__}: {exc}"

    final = final or answer or "(no answer text)"
    save_error = None
    sp_sync = None
    try:
        # -- artifact association (mirrors the old in-script logic) --------
        artifacts = ctx["artifacts"]
        save_new = artifacts[ctx["artifacts_before_len"]:]
        dir_new = core.new_artifacts(ctx["temp_dir"], ctx["before"],
                                     ctx["staged_inputs"])
        if ctx.get("before_wd") is not None:
            for p in core.import_external_artifacts(
                    ctx["working_dir"], ctx["temp_dir"], ctx["before_wd"],
                    ctx["staged_inputs"]):
                if p not in dir_new:
                    dir_new.append(p)
        for p in dir_new:
            if p not in artifacts:
                artifacts.append(p)
        turn_paths = core.collect_turn_artifacts(save_new, dir_new)

        assistant_entry = {"role": "assistant", "text": final,
                           "artifacts": turn_paths}
        if turn_error:
            assistant_entry["error"] = turn_error
        elif final == "(no answer text)":
            assistant_entry["error"] = (
                "The model returned no visible text. Open 'Connection "
                "diagnostics' in the sidebar and run the tests — common "
                "causes: a reasoning model spending the whole token budget "
                "before any output (raise GEOTECH_WEBAPP_MAX_TOKENS), or the "
                "proxy rejecting a request parameter.")

        messages.append({"role": "assistant", "content": final})
        ctx["transcript"].append(assistant_entry)
        try:
            core.append_transcript(thread_id, assistant_entry)
            transcript = ctx["transcript"]
            user_turns = sum(1 for e in transcript if e.get("role") == "user")
            title = (core.auto_title(ctx.get("prompt"))
                     if user_turns == 1 and ctx.get("prompt") else None)
            core.save_messages(thread_id, messages)
            core.touch_conversation(thread_id, title=title,
                                    turn_count=user_turns,
                                    model=ctx.get("model"))
            core.set_behavior(thread_id,
                              ctx.get("behavior") or core.default_behavior())
            core.clear_partial(thread_id)
        except Exception as exc:                  # noqa: BLE001
            save_error = f"{type(exc).__name__}: {exc}"

        if ctx.get("trace_on"):
            try:
                core.write_turn_trace(thread_id, {
                    "ts": time.time(),
                    "duration_s": round(time.time() - trace_t0, 3),
                    "turn_tokens": turn_tokens,
                    "n_tool_calls": len(trace_tools),
                    "tools": trace_tools,
                    "error": turn_error,
                })
            except Exception:
                pass

        try:                                       # permanent storage mirror
            from webapp import sharepoint_store
            _sp = sharepoint_store.get_store()
            if _sp.configured:
                sp_sync = _sp.mirror_conversation(thread_id)
        except Exception:
            pass
    except Exception as exc:                       # noqa: BLE001
        save_error = save_error or f"{type(exc).__name__}: {exc}"

    job._finish(final=final, turn_tokens=turn_tokens, error=turn_error,
                save_error=save_error, sp_sync=sp_sync)

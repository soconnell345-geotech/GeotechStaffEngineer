"""A turn's own messages, kept when the turn stops (BACKLOG TA-19).

Field session 2026-10-14: a 51-minute turn extracted a report's boring logs
with parallel helpers and ended on a rate-limit refusal. ``messages.json``
got the question and an error stub; every tool result lived only in
``activity.jsonl``; the app has no checkpointer; and the error said
"Nothing is lost ... ask the agent to continue". The continue turn started
from nothing and rebuilt the inventory and the grid, and wrote the 2019 and
1987 borings as "transcription pending" although the first turn's primary had
held both transcriptions.

:class:`TurnRecord` collects the messages the run ADDS (the model's replies
with their tool calls, and the tool results), from the graph's ``updates``
stream, as they arrive (``webapp.core.stream_turn``). When the turn errors,
:meth:`TurnRecord.kept` turns them into history the next turn can be
replayed on:

* **valid order** -- every tool call is followed by its result before the
  next model reply; a call that never returned gets a result saying so, and
  a result with no call is dropped (:func:`complete_pairs`);
* **plain** -- text and tool calls only (a provider's reasoning items and
  content blocks are not replayed), so any engine accepts it;
* **size** -- a result deepagents moved into its scratch space (over ~80,000
  characters) is put back whole from the turn scope
  (:mod:`funhouse_agent.deep.keep_work`); a result over
  :data:`INLINE_MAX_CHARS`, or the largest ones while the whole is over
  :data:`TOTAL_MAX_CHARS`, is written to ``kept/`` in the working folder and
  replaced in the history by its first :data:`HEAD_CHARS` characters and the
  file's name.

A turn the process never finished (a restart, a crash) is covered by a
checkpoint of the same record (:data:`CHECKPOINT_NAME`, rewritten as each
message arrives), folded into ``messages.json`` on the next open
(``webapp.core.recover_partial``).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

#: The checkpoint of a running turn's messages, beside ``partial.json``.
CHECKPOINT_NAME = "partial_messages.json"

#: A kept tool result longer than this goes to a file (the history keeps its
#: head and the file's name). Just under deepagents' own 80,000-character
#: eviction, so what the model had inline during the turn stays inline.
INLINE_MAX_CHARS = 60_000

#: The kept history's tool results, all together, at most this long: the
#: largest go to files first. Every later model call carries the history.
TOTAL_MAX_CHARS = 300_000

#: Characters of a result moved to a file that stay in the history.
HEAD_CHARS = 4_000

#: The folder in the working folder that kept results go to (the same one
#: a stopped helper's file goes to).
KEPT_DIR = "kept"

#: What a tool call that never returned is answered with.
NOT_FINISHED = ("Not finished: the turn stopped before this call returned, so "
                "it has no result. Run it again if it is still needed.")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                if b.get("type") in (None, "text") and \
                        isinstance(b.get("text"), str):
                    parts.append(b["text"])
                elif b.get("type") in ("image", "image_url"):
                    parts.append("[an image]")
        return "".join(parts)
    return "" if content is None else str(content)


def _kind(msg) -> Optional[str]:
    if isinstance(msg, dict):
        role = msg.get("role") or msg.get("type")
    else:
        role = getattr(msg, "type", None)
    return {"ai": "assistant", "assistant": "assistant",
            "AIMessageChunk": "assistant", "tool": "tool"}.get(role)


def plain(msg) -> Optional[dict]:
    """An AI reply or a tool result as a plain, replayable dict; ``None`` for
    anything else."""
    kind = _kind(msg)
    get = (msg.get if isinstance(msg, dict)
           else (lambda k, d=None: getattr(msg, k, d)))
    if kind == "assistant":
        calls = []
        for c in get("tool_calls") or []:
            if not isinstance(c, dict):
                continue
            if "function" in c:                       # OpenAI shape
                fn = c.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (TypeError, ValueError):
                    args = {}
                c = {"name": fn.get("name"), "args": args, "id": c.get("id")}
            if not c.get("id") or not c.get("name"):
                continue
            args = c.get("args")
            calls.append({"name": c["name"],
                          "args": args if isinstance(args, dict) else {},
                          "id": str(c["id"]), "type": "tool_call"})
        out = {"role": "assistant", "content": _text(get("content"))}
        if calls:
            out["tool_calls"] = calls
        return out
    if kind == "tool":
        out = {"role": "tool", "content": _text(get("content")),
               "tool_call_id": str(get("tool_call_id") or "")}
        if get("name"):
            out["name"] = str(get("name"))
        if get("status") == "error":
            out["status"] = "error"
        return out
    return None


def _msg_id(msg) -> Optional[str]:
    mid = msg.get("id") if isinstance(msg, dict) else getattr(msg, "id", None)
    return str(mid) if mid else None


def complete_pairs(messages: List[dict]) -> List[dict]:
    """``messages`` (plain AI replies and tool results, in arrival order) in
    an order every engine accepts: each AI reply's tool calls answered,
    before the next reply, by their results; a call with no result answered
    with :data:`NOT_FINISHED`; a result with no open call dropped; an empty
    reply (no text, no calls) dropped."""
    out: List[dict] = []
    open_calls: dict = {}                    # id -> name, of the last reply

    def close() -> None:
        for cid, name in open_calls.items():
            out.append({"role": "tool", "content": NOT_FINISHED,
                        "tool_call_id": cid, "name": name,
                        "status": "error"})
        open_calls.clear()

    for m in messages:
        if m.get("role") == "assistant":
            calls = m.get("tool_calls") or []
            if not calls and not str(m.get("content") or "").strip():
                continue
            close()
            out.append(m)
            open_calls.update({c["id"]: c["name"] for c in calls})
        elif m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if cid in open_calls:
                open_calls.pop(cid)
                out.append(m)
    close()
    return out


@dataclass
class KeptWork:
    """What :meth:`TurnRecord.kept` made of a stopped turn."""
    messages: List[dict] = field(default_factory=list)
    n_calls: int = 0              # tool calls the turn made
    n_results: int = 0            # of them, with a real result
    files: List[str] = field(default_factory=list)   # results moved to files
    helper_files: List[str] = field(default_factory=list)

    @property
    def any(self) -> bool:
        return bool(self.n_results or self.helper_files)

    def line(self) -> str:
        """One plain sentence saying exactly what is kept."""
        if not self.any:
            return ("No tool had returned a result this turn, so there is "
                    "nothing to pick up from; files saved or downloaded "
                    "earlier are still in the conversation.")
        bits = []
        if self.n_results:
            bits.append(
                f"its {self.n_results} finished tool "
                f"{'result is' if self.n_results == 1 else 'results are'} "
                "now in the conversation's history"
                + (f" ({len(self.files)} of the largest also as files in the "
                   f"'{KEPT_DIR}' folder)" if self.files else ""))
        if self.helper_files:
            n = len(self.helper_files)
            bits.append(f"{n} helper{'s' if n > 1 else ''} that stopped left "
                        f"what {'they' if n > 1 else 'it'} had read in the "
                        f"'{KEPT_DIR}' folder")
        return ("What this turn had done is kept: " + "; ".join(bits)
                + ", and files saved or downloaded this turn are still in "
                  "the conversation. Asked to continue, the agent picks up "
                  "from them instead of starting over.")

    def history_note(self, error: str = "") -> str:
        """The assistant message that closes a stopped turn in the history
        (what the MODEL reads on the next turn)."""
        gist = " ".join(str(error or "").split())
        if len(gist) > 300:
            gist = gist[:300] + " ..."
        why = f" ({gist})" if gist else ""
        return (f"[This turn stopped on an error before it finished{why}. "
                "The tool calls and results above are kept: when asked to "
                "continue, pick up from them rather than doing them again, "
                "and say what is still missing.]")


def _safe(name: str, room: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(name or "")).strip("-").lower()
    return (s[:room].strip("-")) or "result"


class TurnRecord:
    """The messages one turn's run added, in order (see the module
    docstring). ``add`` is called with the messages of each ``updates``
    item; a message that arrives again with the same id replaces the
    first, as the graph's own reducer does."""

    def __init__(self, question: Any = None,
                 checkpoint_path: Optional[str] = None):
        self.question = question
        self.checkpoint_path = checkpoint_path
        self.scope = None                 # keep_work.TurnScope, set by core
        self._msgs: List[dict] = []
        self._ids: dict = {}              # message id -> index
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._msgs)

    @property
    def messages(self) -> List[dict]:
        with self._lock:
            return list(self._msgs)

    def add(self, messages) -> int:
        """Add the AI replies and tool results among ``messages``; how many
        were new or changed."""
        changed = 0
        with self._lock:
            for m in messages or ():
                p = plain(m)
                if p is None:
                    continue
                mid = _msg_id(m)
                if mid and mid in self._ids:
                    i = self._ids[mid]
                    if self._msgs[i] != p:
                        self._msgs[i] = p
                        changed += 1
                    continue
                if mid:
                    self._ids[mid] = len(self._msgs)
                self._msgs.append(p)
                changed += 1
        if changed and self.checkpoint_path:
            self.checkpoint()
        return changed

    # -- the crash checkpoint ---------------------------------------------
    def checkpoint(self) -> None:
        """Write the record so far (atomically; never raises)."""
        path = self.checkpoint_path
        if not path:
            return
        try:
            data = {"updated": time.time(),
                    "question": plain_question(self.question),
                    "messages": self.messages}
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = path + ".part"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001 - a checkpoint never costs the turn
            pass

    @classmethod
    def from_checkpoint(cls, path: str) -> Optional["TurnRecord"]:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh) or {}
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        rec = cls(question=data.get("question"))
        rec.add(m for m in (data.get("messages") or [])
                if isinstance(m, dict))
        return rec

    # -- what is kept ------------------------------------------------------
    def kept(self, files_dir: Optional[str], turn: int = 0,
             inline_max: int = INLINE_MAX_CHARS,
             total_max: int = TOTAL_MAX_CHARS) -> KeptWork:
        """The turn's messages as replayable history (see the module
        docstring). Large results go to ``<files_dir>/kept/``; with no
        ``files_dir`` they are cut to their head instead."""
        msgs = [dict(m) for m in complete_pairs(self.messages)]
        full = dict(getattr(self.scope, "full_results", None) or {})
        out = KeptWork(messages=msgs)
        if self.scope is not None:
            out.helper_files = list(getattr(self.scope, "kept_files", ())
                                    or ())
        results = [m for m in msgs if m.get("role") == "tool"]
        out.n_calls = len(results)
        out.n_results = sum(1 for m in results
                            if m.get("content") != NOT_FINISHED)
        for m in results:
            whole = full.get(m.get("tool_call_id"))
            if whole and len(whole) > len(m.get("content") or ""):
                m["content"] = whole            # deepagents had evicted it
        order = sorted(results, key=lambda m: -len(m.get("content") or ""))
        total = sum(len(m.get("content") or "") for m in results)
        for m in order:
            n = len(m.get("content") or "")
            if n <= inline_max and total <= total_max:
                continue
            if n <= HEAD_CHARS:
                break
            path = self._to_file(m, files_dir, turn)
            head = m["content"][:HEAD_CHARS]
            if path:
                out.files.append(path)
                where = (f"the file '{_relative(path, files_dir)}' in the "
                         "working folder; read the rest with read_text_file "
                         f"(offset {HEAD_CHARS:,} onward)")
            else:
                where = "nowhere: it could not be written to a file"
            m["content"] = (f"{head}\n\n[This result was {n:,} characters; "
                            f"the first {HEAD_CHARS:,} are above and the "
                            f"whole of it is in {where}.]")
            total -= n - len(m["content"])
        return out

    @staticmethod
    def _to_file(msg: dict, files_dir: Optional[str],
                 turn: int) -> Optional[str]:
        if not files_dir:
            return None
        try:
            folder = os.path.join(files_dir, KEPT_DIR)
            os.makedirs(folder, exist_ok=True)
            tail = re.sub(r"[^A-Za-z0-9]", "",
                          str(msg.get("tool_call_id") or ""))[-6:] or \
                str(int(time.time() * 1000))[-6:]
            name = (f"turn{int(turn) if turn else 0}_"
                    f"{_safe(msg.get('name') or 'result')}_{tail}.txt")
            path = os.path.join(folder, name)
            tmp = path + ".part"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(msg.get("content") or "")
            os.replace(tmp, path)
            return path
        except Exception:  # noqa: BLE001
            return None


def _relative(path: str, base: Optional[str]) -> str:
    if base:
        try:
            rel = os.path.relpath(path, base)
            if not rel.startswith(".."):
                return rel.replace(os.sep, "/")
        except ValueError:
            pass
    return os.path.basename(path)


def plain_question(msg) -> Optional[dict]:
    """The turn's user message as ``{role, content}``, or ``None``."""
    if msg is None:
        return None
    if isinstance(msg, dict):
        role = msg.get("role") or msg.get("type")
        content = msg.get("content")
    else:
        role = getattr(msg, "type", None)
        content = getattr(msg, "content", None)
    if role not in ("user", "human"):
        return None
    return {"role": "user", "content": content if isinstance(content, (str, list))
            else _text(content)}


__all__ = ["TurnRecord", "KeptWork", "complete_pairs", "plain",
           "plain_question", "CHECKPOINT_NAME", "INLINE_MAX_CHARS",
           "TOTAL_MAX_CHARS", "HEAD_CHARS", "KEPT_DIR", "NOT_FINISHED"]

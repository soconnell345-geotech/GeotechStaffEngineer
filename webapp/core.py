"""Framework-agnostic logic for the geotech web chat app.

Everything here is import-testable WITHOUT streamlit and without a live model:
the streamlit shell (``app.py``) is a thin view over these functions, and the
offline tests exercise this module directly. Heavy imports (the deepagents
builder, langchain callbacks, the LangChain model) are done lazily inside the
functions that need them so ``import webapp.core`` stays cheap and side-effect
free.

Responsibilities:

* **Attachments** — an uploaded file is (a) registered in the live attachments
  dict the agent's vision tools read (by a sanitized key), AND (b) staged as a
  real file under the session temp dir so real-path tools (pdf_import /
  dxf_import / drawing_ir / read_pdf_text) can open it. A system-style note
  tells the agent both the key and the staged path.
* **Artifacts** — a ``save_fn`` captures files the agent writes via ``save_file``
  (resolved into the session temp dir), and a directory snapshot/diff catches
  anything else written there (calc packages, DXFs, plots) for download.
* **Streaming** — one turn is streamed from the compiled deep agent, reusing the
  PURE formatters and token accounting from ``funhouse_agent.deep.notebook`` so
  the parsing logic is shared with the notebook UI and already unit-tested.
* **Disclaimer** — the package's professional-use disclaimer text, captured for
  prominent rendering at the top of the app.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple
from uuid import uuid4

#: Accepted upload types (mirrors the notebook FileUpload accept list).
ACCEPTED_UPLOAD_TYPES = [
    "pdf", "png", "jpg", "jpeg", "tif", "tiff",
    "dxf", "csv", "txt", "xml", "diggs",
]


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

def disclaimer_text() -> str:
    """Return the package's full professional-use disclaimer as text.

    ``funhouse_agent.disclaimer`` PRINTS to a stream, so capture it into a
    buffer. Falls back to a short notice if the import is unavailable, so the
    banner is never empty.
    """
    import io
    try:
        from funhouse_agent import disclaimer
        buf = io.StringIO()
        disclaimer(file=buf)
        text = buf.getvalue().strip()
        if text:
            return text
    except Exception:
        pass
    return (
        "GeotechStaffEngineer is an ANALYSIS/RESEARCH AID, not a design "
        "deliverable. Every result must be independently reviewed by a "
        "licensed professional engineer familiar with the site. No warranty; "
        "no engineer-of-record relationship."
    )


# ---------------------------------------------------------------------------
# Session temp dir
# ---------------------------------------------------------------------------

def new_session_dir(prefix: str = "geotech_webapp_") -> str:
    """Create and return a fresh per-session temp directory for staged inputs
    and agent-produced artifacts."""
    return tempfile.mkdtemp(prefix=prefix)


# ---------------------------------------------------------------------------
# Attachments — register bytes + stage to disk
# ---------------------------------------------------------------------------

@dataclass
class Attachment:
    """A staged upload: attachment ``key`` (for vision / read_pdf_text) and the
    on-disk ``path`` (for real-path tools)."""
    key: str
    path: str
    size: int


def sanitize_key(name) -> str:
    """Reduce an uploaded filename to a safe attachment key (reuses the
    funhouse helper; falls back to a local implementation offline)."""
    try:
        from funhouse_agent.vision_tools import sanitize_upload_name
        return sanitize_upload_name(name)
    except Exception:
        import re
        base = os.path.basename(str(name or "").replace("\\", "/")) or "file"
        base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
        return base or "file"


def stage_upload(attachments: dict, temp_dir: str, name, data: bytes) -> Attachment:
    """Register ``data`` in the live ``attachments`` dict under a sanitized key
    AND write it as a real file under ``temp_dir``.

    Returns the :class:`Attachment`. An existing key is overwritten (both the
    dict entry and the file), so re-uploading the same filename replaces it.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("upload data must be bytes")
    data = bytes(data)
    key = sanitize_key(name)
    attachments[key] = data
    path = os.path.join(temp_dir, key)
    with open(path, "wb") as fh:
        fh.write(data)
    return Attachment(key=key, path=path, size=len(data))


def stage_uploads(attachments: dict, temp_dir: str,
                  files: Iterable[Tuple[object, bytes]]) -> List[Attachment]:
    """Stage a batch of ``(name, bytes)`` pairs. Returns the list of
    :class:`Attachment`."""
    out = []
    for name, data in files:
        out.append(stage_upload(attachments, temp_dir, name, data))
    return out


def attachment_note(atts: List[Attachment]) -> str:
    """Build the system-style note telling the agent about staged attachments —
    the attachment key (for analyze_image / analyze_pdf_page / read_pdf_text)
    AND the on-disk path (for pdf_import / dxf_import / drawing_ir tools that
    need a real file path). Returns ``""`` for an empty list."""
    atts = list(atts or [])
    if not atts:
        return ""
    lines = ["[System note] The user attached files, available to you as:"]
    for a in atts:
        lines.append(
            f"- '{a.key}': attachment key '{a.key}' "
            f"(use analyze_image / analyze_pdf_page / read_pdf_text with "
            f"attachment_key='{a.key}'); also staged on disk at '{a.path}' "
            f"(pass this path to pdf_import / dxf_import / drawing_ir tools that "
            f"need a real file path)."
        )
    return "\n".join(lines)


def assemble_user_message(pending_notes: Iterable[str], user_text: str) -> str:
    """Compose the agent-facing user message: any pending attachment notes,
    then the user's typed text. When there are no notes this is just
    ``user_text`` (byte-identical), so the no-attachment path is unchanged."""
    notes = [n for n in (pending_notes or []) if n]
    if not notes:
        return user_text
    return "\n\n".join(notes + [user_text])


# ---------------------------------------------------------------------------
# Artifacts — save_fn capture + directory watch
# ---------------------------------------------------------------------------

def make_save_fn(temp_dir: str, artifacts: List[str]) -> Callable[[str, object], str]:
    """Build a ``save_fn(path, content) -> saved_path`` for ``build_deep_agent``.

    A bare/relative path is resolved into the session ``temp_dir`` (by
    basename) so the app can serve it; an absolute path is honored as-is. Every
    resolved path is appended to ``artifacts`` (deduplicated) for the download
    list. Bytes or text content are both handled.
    """
    def save_fn(path, content) -> str:
        p = str(path)
        if not os.path.isabs(p):
            p = os.path.join(temp_dir, os.path.basename(p) or "output")
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if isinstance(content, (bytes, bytearray)):
            with open(p, "wb") as fh:
                fh.write(content)
        else:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(str(content))
        if p not in artifacts:
            artifacts.append(p)
        return p
    return save_fn


def snapshot_dir(temp_dir: str) -> set:
    """Return the set of file paths currently under ``temp_dir`` (recursive)."""
    found = set()
    for root, _dirs, names in os.walk(temp_dir):
        for n in names:
            found.add(os.path.join(root, n))
    return found


def new_artifacts(temp_dir: str, before: set, input_paths: Iterable[str]) -> List[str]:
    """Files under ``temp_dir`` that appeared since the ``before`` snapshot and
    are NOT staged inputs — i.e. agent-produced artifacts to offer for
    download. Sorted for stable display."""
    inputs = set(input_paths or ())
    after = snapshot_dir(temp_dir)
    return sorted(p for p in (after - before) if p not in inputs)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def build_agent(model, attachments: dict, temp_dir: str, artifacts: List[str],
                **build_kwargs):
    """Build the compiled deep agent wired to the SHARED attachments dict and a
    session-dir save_fn. ``build_kwargs`` pass through to ``build_deep_agent``
    (e.g. ``enable_memory``). Lazy-imports the deepagents builder."""
    from funhouse_agent.deep.agent import build_deep_agent
    return build_deep_agent(
        model,
        attachments=attachments,
        save_fn=make_save_fn(temp_dir, artifacts),
        **build_kwargs,
    )


# ---------------------------------------------------------------------------
# Streaming one turn
# ---------------------------------------------------------------------------

def new_thread_id() -> str:
    """A fresh LangGraph thread id."""
    return uuid4().hex


def token_line(turn_tokens: int, total_tokens: int) -> str:
    """Format the per-turn / running token line (reuses the notebook helper)."""
    try:
        from funhouse_agent.deep.notebook import _format_token_line
        return _format_token_line(turn_tokens, total_tokens)
    except Exception:
        return (f"tokens this turn: {turn_tokens:,} | "
                f"conversation total: {total_tokens:,}")


def stream_turn(agent, messages: list, thread_id: str,
                max_result_chars: int = 2000):
    """Stream ONE turn from the compiled deep agent.

    ``messages`` is the full agent-facing history INCLUDING the new user turn
    (the caller appends it and, on completion, appends the assistant answer from
    the ``turn_done`` item). Yields entry dicts:

    * ``{"kind": "token", "text": str}`` — a streamed answer token.
    * ``{"kind": "tool_call"|"todos"|"tool_result", "text": str}`` — activity.
    * ``{"kind": "turn_done", "answer": str, "turn_tokens": int}`` — final,
      carrying the concatenated answer and the token spend for this turn
      (aggregated across every model call in the run, sub-agents included).

    Reuses the PURE ``_format_update`` parser and ``_sum_callback_tokens`` from
    ``funhouse_agent.deep.notebook`` so the stream contract is shared with the
    notebook UI. The whole stream runs under a usage-metadata callback; if that
    is unavailable the turn still streams (token spend simply reports 0).
    """
    from funhouse_agent.deep.notebook import _format_update, _sum_callback_tokens

    answer_parts: List[str] = []
    config = {"configurable": {"thread_id": thread_id}}
    try:
        from langchain_core.callbacks import get_usage_metadata_callback
        cb_ctx = get_usage_metadata_callback()
    except Exception:
        cb_ctx = None

    if cb_ctx is None:
        for mode, chunk in agent.stream(
                {"messages": messages}, config=config,
                stream_mode=["updates", "messages"]):
            for entry in _format_update(mode, chunk,
                                        max_result_chars=max_result_chars):
                if entry["kind"] == "token":
                    answer_parts.append(entry["text"])
                yield entry
        yield {"kind": "turn_done", "answer": "".join(answer_parts),
               "turn_tokens": 0}
        return

    with cb_ctx as cb:
        run_config = dict(config)
        run_config["callbacks"] = [cb]
        for mode, chunk in agent.stream(
                {"messages": messages}, config=run_config,
                stream_mode=["updates", "messages"]):
            for entry in _format_update(mode, chunk,
                                        max_result_chars=max_result_chars):
                if entry["kind"] == "token":
                    answer_parts.append(entry["text"])
                yield entry
        turn_tokens = _sum_callback_tokens(dict(cb.usage_metadata))
    yield {"kind": "turn_done", "answer": "".join(answer_parts),
           "turn_tokens": turn_tokens}

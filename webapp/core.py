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
import re
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


#: Extension -> artifact kind (drives the card icon + which inline preview to
#: use). Anything unlisted is "other" (download-only).
_ARTIFACT_KIND_BY_EXT = {
    ".html": "html", ".htm": "html",
    ".pdf": "pdf",
    ".png": "png",
    ".jpg": "image", ".jpeg": "image", ".gif": "image", ".webp": "image",
    ".bmp": "image",
    ".svg": "svg",
    ".dxf": "dxf",
    ".csv": "csv", ".txt": "text", ".md": "text", ".json": "text",
}

#: Inline-preview size caps. Bigger files are offered download-only — a
#: self-contained calc-package HTML or a big PDF should not be inlined on every
#: rerun.
HTML_PREVIEW_MAX_BYTES = 4 * 1024 * 1024
PDF_PREVIEW_MAX_BYTES = 10 * 1024 * 1024


@dataclass
class ArtifactCard:
    """Display data for one agent-produced artifact (streamlit-free — the
    rendering lives in app.py)."""
    path: str
    name: str
    size: int
    kind: str

    @property
    def exists(self) -> bool:
        return os.path.isfile(self.path)


def classify_artifact(path) -> str:
    """Map a file path to an artifact kind by extension
    (``plotly``/``html``/``pdf``/``png``/``image``/``svg``/``dxf``/``csv``/``text``/``other``).

    A ``*.plotly.json`` sidecar (a Plotly figure serialized with
    ``figure.to_json()``) classifies as ``plotly`` so the app can render it
    natively with ``st.plotly_chart`` — checked before the plain-extension
    lookup, which would otherwise see only ``.json`` and return ``text``.
    """
    if str(path).lower().endswith(".plotly.json"):
        return "plotly"
    return _ARTIFACT_KIND_BY_EXT.get(os.path.splitext(str(path))[1].lower(),
                                     "other")


def describe_artifact(path) -> ArtifactCard:
    """Build the :class:`ArtifactCard` for ``path`` (name, size, kind)."""
    p = str(path)
    try:
        size = os.path.getsize(p)
    except OSError:
        size = 0
    return ArtifactCard(path=p, name=os.path.basename(p), size=size,
                        kind=classify_artifact(p))


def artifact_bytes(path) -> bytes:
    """Read an artifact's raw bytes (for download / preview)."""
    with open(path, "rb") as fh:
        return fh.read()


def read_text(path) -> str:
    """Read an artifact as UTF-8 text (lossy on undecodable bytes)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def pdf_data_uri(path, max_bytes: int = PDF_PREVIEW_MAX_BYTES) -> Optional[str]:
    """Return a ``data:application/pdf;base64,…`` URI for inline preview, or
    ``None`` when the file is missing/empty or exceeds ``max_bytes`` (then the
    app shows download-only)."""
    import base64
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if size <= 0 or size > max_bytes:
        return None
    return "data:application/pdf;base64," + \
        base64.b64encode(artifact_bytes(path)).decode("ascii")


def collect_turn_artifacts(save_new: Iterable[str],
                           dir_new: Iterable[str]) -> List[str]:
    """Associate a turn's artifacts: the union of the paths the save_fn recorded
    during the turn and the new files the directory diff found, deduplicated and
    order-preserving (save_fn first)."""
    out: List[str] = []
    seen = set()
    for p in list(save_new or []) + list(dir_new or []):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


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
                checkpointer=None, **build_kwargs):
    """Build the compiled deep agent wired to the SHARED attachments dict and a
    session-dir save_fn. ``build_kwargs`` pass through to ``build_deep_agent``
    (e.g. ``enable_memory``). Lazy-imports the deepagents builder.

    ``checkpointer`` (optional) is a LangGraph checkpointer forwarded to
    ``build_deep_agent`` for durable/resumable thread state. The shipped webapp
    resumes conversations by REPLAYING the persisted agent-facing message history
    (see the Persistence section) rather than depending on a durable checkpointer,
    so this defaults to ``None`` (byte-identical to the pre-persistence build);
    it is exposed so a durable saver (e.g. a LangGraph SQLite saver, an optional
    dependency) can be dropped in without touching the builder wiring.
    """
    from funhouse_agent.deep.agent import build_deep_agent
    kw = dict(build_kwargs)
    if checkpointer is not None:
        kw["checkpointer"] = checkpointer
    return build_deep_agent(
        model,
        attachments=attachments,
        save_fn=make_save_fn(temp_dir, artifacts),
        **kw,
    )


#: agent_type value -> deep reviewer builder name in funhouse_agent.reviewers.
_REVIEWER_BUILDERS = {
    "seismic": "make_seismic_reviewer_deep",
    "foundations": "make_foundations_reviewer_deep",
    "earth_retention": "make_earth_retention_reviewer_deep",
    "slope_fem": "make_slope_fem_reviewer_deep",
}


def build_reviewer_agent(kind, model, attachments: dict, temp_dir: str,
                         artifacts: List[str], **build_kwargs):
    """Build a NARROW reviewer deep-agent (A5e) for ``kind``, wired to the SAME
    shared attachments dict + session-dir save_fn as the full agent (the reviewer
    ``make_*_reviewer_deep`` builders forward these to ``build_deep_agent``).

    An unknown ``kind`` (including ``"full"``) falls back to the full agent build.
    A reviewer manages its own scope + review-mode prompt + ``reference_mode``, so
    the behavior reference/analysis-depth build-kwargs are deliberately NOT applied
    to it; the recursion cap still applies at stream time.
    """
    name = _REVIEWER_BUILDERS.get(kind)
    if name is None:
        return build_agent(model, attachments, temp_dir, artifacts, **build_kwargs)
    from funhouse_agent import reviewers as _reviewers
    builder = getattr(_reviewers, name)
    return builder(model, attachments=attachments,
                   save_fn=make_save_fn(temp_dir, artifacts), **build_kwargs)


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


# --- Mid-turn-stop fix (owner bug, 2026-07-14 retaining-wall session) --------
# A model sometimes ENDS its reply on a stated-but-unperformed next step
# ("Let me get that Ka …") with no tool call behind it — the graph sees no tool
# call and the turn ends silently mid-analysis. stream_turn detects that shape
# and re-invokes with a terse nudge, at most MAX_AUTO_CONTINUES times.

CONTINUE_NUDGE = "Continue — complete the action you just stated."
MAX_AUTO_CONTINUES = 2

_INTENT_RE = re.compile(
    r"\b(let me|let's|i'?ll|i will|now i|next,? i)\b", re.IGNORECASE)
# Endings addressed TO the user (offers/questions) must never trigger a nudge.
_ADDRESSED_RE = re.compile(
    r"\b(would you|should i|do you|let me know|if you|if needed|happy to|"
    r"feel free|want me to|prefer)\b", re.IGNORECASE)


def ends_mid_task(text: str, saw_tool_call: bool) -> bool:
    """True when an assistant reply looks like it STOPPED on a stated next step.

    Conservative by design (a false nudge costs one extra model call; a missed
    one just reproduces the old behavior): fires only when the turn actually
    used tools, the reply doesn't end with a question, and the FINAL sentence
    contains first-person intent language ("let me …", "I'll …") that is not
    addressed to the user ("let me know if …", "would you …").
    """
    t = (text or "").strip()
    if not t or not saw_tool_call or t.endswith("?"):
        return False
    tail = re.split(r"(?<=[.!?])\s+|\n+", t)[-1].strip()
    if not tail or _ADDRESSED_RE.search(tail):
        return False
    return bool(_INTENT_RE.search(tail))


def stream_turn(agent, messages: list, thread_id: str,
                max_result_chars: int = 2000,
                recursion_limit: Optional[int] = None):
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
    saw_tool = False
    work_messages = list(messages)
    continuations = 0
    config = {"configurable": {"thread_id": thread_id}}
    if recursion_limit:                     # A5(b): primary-agent step cap
        config["recursion_limit"] = int(recursion_limit)
    try:
        from langchain_core.callbacks import get_usage_metadata_callback
        cb_ctx = get_usage_metadata_callback()
    except Exception:
        cb_ctx = None

    def _run_passes(run_config):
        # One or more graph invocations: the extra passes are the bounded
        # auto-continue for replies that end on a stated-but-unperformed step.
        nonlocal work_messages, continuations, saw_tool
        while True:
            pass_parts: List[str] = []
            for mode, chunk in agent.stream(
                    {"messages": work_messages}, config=run_config,
                    stream_mode=["updates", "messages"]):
                for entry in _format_update(mode, chunk,
                                            max_result_chars=max_result_chars):
                    if entry["kind"] == "token":
                        pass_parts.append(entry["text"])
                    elif entry["kind"] == "tool_call":
                        saw_tool = True
                    yield entry
            pass_text = "".join(pass_parts)
            if answer_parts and pass_text:
                answer_parts.append("\n\n")
            answer_parts.append(pass_text)
            if (continuations < MAX_AUTO_CONTINUES
                    and ends_mid_task(pass_text, saw_tool)):
                continuations += 1
                work_messages = work_messages + [
                    {"role": "assistant", "content": pass_text},
                    {"role": "user", "content": CONTINUE_NUDGE},
                ]
                yield {"kind": "tool_call",
                       "text": (f"auto-continue {continuations}/"
                                f"{MAX_AUTO_CONTINUES}: finishing the stated "
                                "next step")}
                continue
            return

    if cb_ctx is None:
        for entry in _run_passes(dict(config)):
            yield entry
        yield {"kind": "turn_done", "answer": "".join(answer_parts),
               "turn_tokens": 0}
        return

    with cb_ctx as cb:
        run_config = dict(config)
        run_config["callbacks"] = [cb]
        for entry in _run_passes(run_config):
            yield entry
        turn_tokens = _sum_callback_tokens(dict(cb.usage_metadata))
    yield {"kind": "turn_done", "answer": "".join(answer_parts),
           "turn_tokens": turn_tokens}


# ---------------------------------------------------------------------------
# Model choices (the in-app picker)
# ---------------------------------------------------------------------------
#
# Curated Claude models offered in the sidebar picker, as DATA (id/label/blurb)
# so it is testable and trivial to extend. The first entry is the default unless
# ``GEOTECH_WEBAPP_MODEL`` is set (then that id is prepended and becomes the
# default). Keep the first id aligned with ``engine_config.DEFAULT_MODEL``.

MODEL_CHOICES = [
    {"id": "claude-opus-4-8", "label": "Opus 4.8",
     "blurb": "deepest reasoning, default"},
    {"id": "claude-sonnet-5", "label": "Sonnet 5",
     "blurb": "fast + capable"},
    {"id": "claude-haiku-4-5-20251001", "label": "Haiku 4.5",
     "blurb": "quick questions"},
]


def model_choices(env_model: Optional[str] = None) -> List[dict]:
    """The picker list ``[{id,label,blurb}, ...]``. If ``GEOTECH_WEBAPP_MODEL``
    (or the ``env_model`` override) names a model not already curated, it is
    PREPENDED (and so becomes the default). Returns fresh dicts (safe to mutate)."""
    envm = (env_model if env_model is not None
            else os.environ.get("GEOTECH_WEBAPP_MODEL"))
    choices = [dict(c) for c in MODEL_CHOICES]
    if envm and not any(c["id"] == envm for c in choices):
        choices.insert(0, {"id": envm, "label": envm,
                           "blurb": "from GEOTECH_WEBAPP_MODEL"})
    return choices


def default_model_id(env_model: Optional[str] = None) -> str:
    """The default selected model id: ``GEOTECH_WEBAPP_MODEL`` if set (whether or
    not it is already curated), else the first curated choice."""
    envm = (env_model if env_model is not None
            else os.environ.get("GEOTECH_WEBAPP_MODEL"))
    return envm or MODEL_CHOICES[0]["id"]


def model_label(model_id: Optional[str]) -> str:
    """Short display label for a model id (falls back to the id; ``""`` for
    ``None``/empty)."""
    if not model_id:
        return ""
    for c in MODEL_CHOICES:
        if c["id"] == model_id:
            return c["label"]
    return model_id


# ---------------------------------------------------------------------------
# Persistence — durable, resumable conversations
# ---------------------------------------------------------------------------
#
# A conversation is a directory ``<data_root>/conversations/<thread_id>/`` with:
#   meta.json          {thread_id, title, created, updated, turn_count}
#   transcript.jsonl   one display entry per line (append-on-turn)
#   messages.json      the agent-facing message history (replayed on resume so
#                      the model "remembers" the conversation without depending
#                      on a durable LangGraph checkpointer)
#   attachments.json   the staged upload keys (re-registered into the live
#                      attachments dict on resume)
#   files/             the working dir: staged uploads AND agent artifacts (this
#                      is the ``temp_dir`` the rest of core.py already uses, made
#                      persistent so artifacts survive restarts)
# Deleting a conversation MOVES its directory into ``<data_root>/.trash/`` rather
# than hard-deleting. The data root is ``$GEOTECH_WEBAPP_DATA`` or, by default,
# ``~/.geotech_webapp`` (TinyApp deployments override it to a writable volume).

import json as _json
import shutil as _shutil
import time as _time


def data_root() -> str:
    """Root directory for persisted conversations. ``$GEOTECH_WEBAPP_DATA`` if
    set (``~`` expanded), else ``~/.geotech_webapp``."""
    env = os.environ.get("GEOTECH_WEBAPP_DATA")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(os.path.expanduser("~"), ".geotech_webapp")


def conversations_root(root: Optional[str] = None) -> str:
    return os.path.join(root or data_root(), "conversations")


def conversation_dir(thread_id: str, root: Optional[str] = None) -> str:
    return os.path.join(conversations_root(root), thread_id)


def conversation_files_dir(thread_id: str, root: Optional[str] = None) -> str:
    """The conversation's working dir (staged uploads + artifacts) — the
    persistent replacement for ``new_session_dir()``. Created on demand."""
    d = os.path.join(conversation_dir(thread_id, root), "files")
    os.makedirs(d, exist_ok=True)
    return d


def working_dir_for(thread_id: str, meta: Optional[dict] = None,
                    root: Optional[str] = None) -> str:
    """The conversation's WORKING FOLDER — where the agent's saves (calc
    packages, plots, ``save_file``) default. ``meta['working_dir']`` if set
    (``~`` expanded, absolute), else the conversation ``files/`` dir. The
    directory is created on demand."""
    if meta is None:
        meta = load_meta(thread_id, root)
    wd = (meta or {}).get("working_dir")
    if wd and str(wd).strip():
        path = os.path.abspath(os.path.expanduser(str(wd).strip()))
    else:
        path = conversation_files_dir(thread_id, root)
    os.makedirs(path, exist_ok=True)
    return path


def set_working_dir(thread_id: str, path: Optional[str],
                    root: Optional[str] = None) -> str:
    """Persist the conversation's working folder in meta. A blank/None ``path``
    resets it to the ``files/`` default. Returns the resolved absolute dir."""
    p = str(path or "").strip()
    resolved = os.path.abspath(os.path.expanduser(p)) if p else None
    meta = ensure_conversation(thread_id, root=root)
    meta["working_dir"] = resolved          # None => default (files dir)
    meta["updated"] = _time.time()
    save_meta(thread_id, meta, root)
    return working_dir_for(thread_id, meta, root)


def apply_default_output_dir(path: Optional[str]) -> None:
    """Point the agent's default output dir at ``path`` (via the
    ``GEOTECH_DEFAULT_OUTPUT_DIR`` env the tool layer reads in
    ``funhouse_agent._fileio.default_output_dir``), so tool saves default INTO
    the conversation working folder instead of the system temp dir. Falsy clears
    it (restores the pre-app default). Precedence: an explicit tool
    ``output_path`` > this working folder > the temp fallback."""
    try:
        from funhouse_agent._fileio import DEFAULT_OUTPUT_DIR_ENV as _ENV
    except Exception:
        _ENV = "GEOTECH_DEFAULT_OUTPUT_DIR"
    if path:
        os.environ[_ENV] = str(path)
    else:
        os.environ.pop(_ENV, None)


def _unique_dest(path: str) -> str:
    """A destination path that does not overwrite a DIFFERENT existing file:
    return ``path`` if free, else append ``_1``/``_2``/… to the stem."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"


def import_external_artifacts(working_dir: str, files_dir: str, before: set,
                              input_paths: Iterable[str]) -> List[str]:
    """Copy files newly produced in ``working_dir`` (since the ``before``
    snapshot, excluding staged ``input_paths``) INTO ``files_dir`` so they
    persist with the conversation and render as durable cards. Returns the
    destination paths under ``files_dir`` (sorted). A no-op returning ``[]`` when
    the working dir IS the files dir (the default — normal capture covers it)."""
    wd = os.path.abspath(working_dir)
    fd = os.path.abspath(files_dir)
    if wd == fd:
        return []
    inputs = set(input_paths or ())
    new = sorted(p for p in (snapshot_dir(wd) - set(before or ()))
                 if p not in inputs)
    out: List[str] = []
    for src in new:
        dst = _unique_dest(os.path.join(fd, os.path.basename(src)))
        try:
            _shutil.copy2(src, dst)
        except OSError:
            continue
        out.append(dst)
    return out


# ---------------------------------------------------------------------------
# Behavior settings (A5): per-conversation pickers, persisted in meta
# ---------------------------------------------------------------------------
# Five knobs the sidebar exposes and stores under meta["behavior"]:
#   references     -- "anytime" (consult sub-agent offered) | "off" (no refs)
#   ref_max_calls  -- the reference consult model-call budget
#   recursion_limit-- the PRIMARY agent's LangGraph step cap (per-turn)
#   analysis_depth -- "screening" | "standard" | "comprehensive": a system-prompt
#                     preset applied on ALL engines (Anthropic + Prompter). This is
#                     NOT LLM "thinking" — that name is reserved for the future
#                     API-level control (deferred adaptive-thinking follow-up;
#                     budget_tokens 400s on Opus 4.8 / Sonnet 5 — see
#                     module_work/APP_PLAN.md A5 notes).
#   agent_type     -- "full" (general agent) | a narrow domain reviewer
#                     (seismic / foundations / earth_retention / slope_fem)
#   route_calc     -- bool: delegate tool-heavy calc to a `calc` sub-agent so the
#                     bulky calc trace stays out of the conversation (A2). ON by
#                     default in the app (the owner-approved A2 default-on; the
#                     build_deep_agent library default stays OFF).
# Defaults reproduce today's behavior EXACTLY EXCEPT route_calc (the one owner-
# approved A2 default-on): references anytime, ref budget 8, LangGraph's default
# recursion_limit 25, analysis_depth "standard" == no preset, agent_type "full".

ANALYSIS_DEPTHS = ("screening", "standard", "comprehensive")
REFERENCE_CHOICES = ("anytime", "off")

#: Selectable agent variants (value -> sidebar label). "full" is the default
#: general geotech agent; the rest are the narrow domain reviewers (F8/D6),
#: each scoped to one discipline's methods + references and prompted in review
#: mode (built via funhouse_agent.reviewers.make_*_reviewer_deep).
AGENT_TYPES = {
    "full": "Full geotech agent",
    "seismic": "Seismic reviewer",
    "foundations": "Foundations reviewer",
    "earth_retention": "Earth-retention reviewer",
    "slope_fem": "Slope / FEM reviewer",
}

DEFAULT_BEHAVIOR = {
    "references": "anytime",
    "ref_max_calls": 8,
    "recursion_limit": 25,
    "analysis_depth": "standard",
    "agent_type": "full",
    "route_calc": True,
}


def agent_type_label(kind: Optional[str]) -> str:
    """Human label for an agent-type value (defaults to the 'full' label)."""
    return AGENT_TYPES.get(kind or "full", str(kind))

_DEPTH_SCREENING = (
    "ANALYSIS DEPTH: SCREENING. Give a fast, concise screening answer. Run the "
    "single most appropriate method, report the result with its key assumptions, "
    "and stop. Do not run multiple methods, cross-checks, sensitivity studies, or "
    "other elective extras unless the user explicitly asks.")
_DEPTH_COMPREHENSIVE = (
    "ANALYSIS DEPTH: COMPREHENSIVE. Be thorough. Where the question warrants it: "
    "run multiple applicable methods and compare them; cross-check the governing "
    "result via a second, independent approach; run a short sensitivity on the "
    "governing inputs and state the resulting range/spread (the true answer is a "
    "distribution, not a point); state the governing conditions and your "
    "confidence; and offer to produce a calc package.")


def default_behavior() -> dict:
    """A fresh copy of the default behavior settings."""
    return dict(DEFAULT_BEHAVIOR)


def behavior_from_meta(meta: Optional[dict]) -> dict:
    """Behavior settings for a conversation: ``meta['behavior']`` merged over the
    defaults (unknown keys ignored, missing keys defaulted), so an old meta with
    no behavior block reads as today's defaults."""
    b = dict(DEFAULT_BEHAVIOR)
    src = (meta or {}).get("behavior")
    if isinstance(src, dict):
        for k in DEFAULT_BEHAVIOR:
            if src.get(k) is not None:
                b[k] = src[k]
    return b


def set_behavior(thread_id: str, behavior: dict,
                 root: Optional[str] = None) -> dict:
    """Persist a conversation's behavior settings in meta. Returns the resolved
    (defaulted) settings."""
    meta = ensure_conversation(thread_id, root=root)
    meta["behavior"] = {k: behavior[k] for k in DEFAULT_BEHAVIOR if k in behavior}
    meta["updated"] = _time.time()
    save_meta(thread_id, meta, root)
    return behavior_from_meta(meta)


def depth_prompt(depth: str) -> str:
    """The system-prompt preset appended for an analysis-depth level ("" for
    "standard"/unknown == today's default behavior, byte-identical)."""
    return {"screening": _DEPTH_SCREENING,
            "comprehensive": _DEPTH_COMPREHENSIVE}.get(depth, "")


def behavior_build_kwargs(behavior: Optional[dict]) -> dict:
    """Translate behavior settings into ``build_deep_agent`` kwargs (via
    ``build_agent``): reference mode, the reference call budget, and the
    analysis-depth prompt preset. Recursion is applied at stream time, not here.
    Defaults produce an EMPTY-of-overrides-equivalent build (reference_mode
    anytime, ref budget 8, no extra prompt)."""
    b = behavior_from_meta({"behavior": behavior}) if behavior is not None \
        else default_behavior()
    kw: dict = {
        "reference_mode": "off" if b["references"] == "off" else "anytime",
        "references_max_model_calls": int(b["ref_max_calls"]),
    }
    preset = depth_prompt(b["analysis_depth"])
    if preset:
        kw["extra_system_prompt"] = preset
    if b.get("route_calc", True):          # A2: delegate heavy calc to `calc`
        kw["enable_calc_subagent"] = True
    return kw


# ---------------------------------------------------------------------------
# Run tracing (A7 rec 1): optional, OFF by default
# ---------------------------------------------------------------------------
# Two independent paths, both opt-in:
#   * LangSmith (SaaS) — set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY; the
#     langchain/langgraph stack auto-traces every run, no code here.
#   * Local (no SaaS) — set GEOTECH_TRACE=1; the app writes ONE compact JSONL
#     line per turn (duration, tokens, tool calls incl. sub-agent hops, error)
#     to <conversation>/trace.jsonl and shows a "turn details" expander.

def tracing_enabled() -> bool:
    """True when the local per-turn tracer is on (``GEOTECH_TRACE`` truthy)."""
    return str(os.environ.get("GEOTECH_TRACE", "")).strip().lower() in (
        "1", "true", "yes", "on")


def trace_path(thread_id: str, root: Optional[str] = None) -> str:
    return _conv_path(thread_id, "trace.jsonl", root)


def write_turn_trace(thread_id: str, record: dict,
                     root: Optional[str] = None) -> None:
    """Append one per-turn trace ``record`` as a JSONL line in the conversation
    dir. Best-effort — a trace failure must NEVER affect the turn."""
    try:
        os.makedirs(conversation_dir(thread_id, root), exist_ok=True)
        with open(trace_path(thread_id, root), "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def load_recent_traces(thread_id: str, n: int = 1,
                       root: Optional[str] = None) -> List[dict]:
    """The last ``n`` per-turn trace records (oldest→newest), or ``[]``."""
    p = trace_path(thread_id, root)
    if not os.path.isfile(p):
        return []
    try:
        with open(p, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return []
    out: List[dict] = []
    for line in lines[-int(max(1, n)):]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(_json.loads(line))
        except ValueError:
            continue
    return out


def auto_title(text, n_words: int = 8) -> str:
    """First ~``n_words`` words of the first user message, as a conversation
    title. Falls back to 'New conversation' for empty input."""
    words = str(text or "").split()
    if not words:
        return "New conversation"
    title = " ".join(words[:n_words])
    if len(words) > n_words:
        title += "…"
    return title


def _conv_path(thread_id, name, root=None) -> str:
    return os.path.join(conversation_dir(thread_id, root), name)


def load_meta(thread_id: str, root: Optional[str] = None) -> Optional[dict]:
    """Load a conversation's meta dict, or ``None`` if it does not exist."""
    p = _conv_path(thread_id, "meta.json", root)
    try:
        with open(p, encoding="utf-8") as fh:
            return _json.load(fh)
    except (OSError, ValueError):
        return None


def save_meta(thread_id: str, meta: dict, root: Optional[str] = None) -> None:
    os.makedirs(conversation_dir(thread_id, root), exist_ok=True)
    with open(_conv_path(thread_id, "meta.json", root), "w",
              encoding="utf-8") as fh:
        _json.dump(meta, fh, ensure_ascii=False, indent=2)


def ensure_conversation(thread_id: str, title: Optional[str] = None,
                        root: Optional[str] = None) -> dict:
    """Return the conversation's meta, creating it (with ``title`` or a
    placeholder) on first use."""
    meta = load_meta(thread_id, root)
    if meta is None:
        now = _time.time()
        meta = {"thread_id": thread_id, "title": title or "New conversation",
                "created": now, "updated": now, "turn_count": 0, "model": None}
        save_meta(thread_id, meta, root)
    return meta


def touch_conversation(thread_id: str, *, title: Optional[str] = None,
                       turn_count: Optional[int] = None,
                       model: Optional[str] = None,
                       root: Optional[str] = None) -> dict:
    """Update ``updated`` (and optionally ``title`` / ``turn_count`` / ``model``)
    on a conversation's meta; creates it if missing."""
    meta = ensure_conversation(thread_id, title=title, root=root)
    if title is not None:
        meta["title"] = title
    if turn_count is not None:
        meta["turn_count"] = turn_count
    if model is not None:
        meta["model"] = model
    meta["updated"] = _time.time()
    save_meta(thread_id, meta, root)
    return meta


def rename_conversation(thread_id: str, title: str,
                        root: Optional[str] = None) -> dict:
    """Set a conversation's title."""
    return touch_conversation(thread_id, title=str(title), root=root)


def list_conversations(root: Optional[str] = None) -> List[dict]:
    """Every saved conversation's meta, most-recently-updated first. Skips the
    ``.trash`` folder and any dir without a readable meta.json."""
    base = conversations_root(root)
    out: List[dict] = []
    try:
        names = os.listdir(base)
    except OSError:
        return out
    for name in names:
        if not os.path.isdir(os.path.join(base, name)):
            continue
        meta = load_meta(name, root)
        if meta:
            out.append(meta)
    out.sort(key=lambda m: m.get("updated", 0), reverse=True)
    return out


def _rel_artifact(path, files_dir) -> str:
    """Store an artifact reference portably: relative to the conversation files
    dir when it lives under it, else the absolute path."""
    ap = os.path.abspath(str(path))
    fd = os.path.abspath(files_dir)
    if ap == fd or ap.startswith(fd + os.sep):
        return os.path.relpath(ap, fd)
    return ap


def _resolve_artifact(ref, files_dir) -> str:
    """Inverse of ``_rel_artifact``: resolve a stored reference back to a path
    under the (possibly relocated) conversation files dir."""
    if os.path.isabs(ref):
        return ref
    return os.path.join(files_dir, ref)


def append_transcript(thread_id: str, entry: dict,
                      root: Optional[str] = None) -> None:
    """Append ONE display entry to ``transcript.jsonl``. Artifact paths in the
    entry are stored relative to the conversation files dir (portable)."""
    os.makedirs(conversation_dir(thread_id, root), exist_ok=True)
    files_dir = conversation_files_dir(thread_id, root)
    rec = dict(entry)
    if rec.get("artifacts"):
        rec["artifacts"] = [_rel_artifact(p, files_dir) for p in rec["artifacts"]]
    with open(_conv_path(thread_id, "transcript.jsonl", root), "a",
              encoding="utf-8") as fh:
        fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")


def load_transcript(thread_id: str, root: Optional[str] = None) -> List[dict]:
    """Load the display transcript, resolving artifact refs back to absolute
    paths under the conversation files dir."""
    p = _conv_path(thread_id, "transcript.jsonl", root)
    files_dir = conversation_files_dir(thread_id, root)
    out: List[dict] = []
    try:
        fh = open(p, encoding="utf-8")
    except OSError:
        return out
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except ValueError:
                continue
            if rec.get("artifacts"):
                rec["artifacts"] = [_resolve_artifact(r, files_dir)
                                    for r in rec["artifacts"]]
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Mid-turn crash safety (A3): a per-turn "partial" checkpoint file
# ---------------------------------------------------------------------------
# A hard interruption mid-stream (process kill, OOM, browser close, forced
# rerun — not a Python exception) would lose the streamed text. ``begin_partial``
# marks an in-progress turn before streaming, ``checkpoint_partial`` overwrites
# the accumulating text every few chunks, and a clean completion calls
# ``clear_partial``. On the next boot ``recover_partial`` folds any leftover
# partial into the transcript as a clearly-marked "recovered" entry.

def partial_path(thread_id: str, root: Optional[str] = None) -> str:
    return _conv_path(thread_id, "partial.json", root)


def begin_partial(thread_id: str, prompt: str,
                  root: Optional[str] = None) -> None:
    """Mark an in-progress assistant turn BEFORE streaming (the A3 placeholder).
    Best-effort — never raises into the turn loop."""
    try:
        os.makedirs(conversation_dir(thread_id, root), exist_ok=True)
        with open(partial_path(thread_id, root), "w", encoding="utf-8") as fh:
            _json.dump({"prompt": prompt, "text": "", "started": _time.time()},
                       fh, ensure_ascii=False)
    except OSError:
        pass


def checkpoint_partial(thread_id: str, text: str,
                       root: Optional[str] = None) -> None:
    """Overwrite the in-progress assistant text (called every N stream chunks).
    Best-effort: a checkpoint failure must NEVER break the stream."""
    try:
        p = partial_path(thread_id, root)
        prompt = ""
        try:
            with open(p, encoding="utf-8") as fh:
                prompt = (_json.load(fh) or {}).get("prompt", "")
        except (OSError, ValueError):
            pass
        os.makedirs(conversation_dir(thread_id, root), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            _json.dump({"prompt": prompt, "text": text,
                        "updated": _time.time()}, fh, ensure_ascii=False)
    except OSError:
        pass


def clear_partial(thread_id: str, root: Optional[str] = None) -> None:
    """Remove the partial checkpoint after a turn is durably persisted."""
    try:
        os.remove(partial_path(thread_id, root))
    except OSError:
        pass


def recover_partial(thread_id: str, root: Optional[str] = None) -> Optional[dict]:
    """If a turn was interrupted (``partial.json`` present), return a recovered
    display entry to append to the transcript, else ``None``. Dedupes the rare
    append-succeeded-but-clear-failed case by comparing against the last
    assistant entry already on disk. Clears the partial file either way."""
    p = partial_path(thread_id, root)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            data = _json.load(fh) or {}
    except (OSError, ValueError):
        clear_partial(thread_id, root)
        return None
    text = (data.get("text") or "").strip()
    if text:
        for e in reversed(load_transcript(thread_id, root)):
            if e.get("role") == "assistant":
                if text in (e.get("text") or ""):
                    clear_partial(thread_id, root)
                    return None
                break
    clear_partial(thread_id, root)
    body = text or "_(this turn was interrupted before any output was produced)_"
    return {"role": "assistant", "recovered": True,
            "text": body + "\n\n_(recovered after an interrupted session)_"}


def _msg_to_plain(m) -> dict:
    """A LangChain BaseMessage (or a dict) -> a plain ``{role, content}`` dict."""
    if isinstance(m, dict):
        return m
    role = {"human": "user", "ai": "assistant", "system": "system",
            "tool": "tool", "function": "tool"}.get(
                getattr(m, "type", ""), "assistant")
    return {"role": role, "content": getattr(m, "content", "")}


def serialize_messages(messages: list) -> list:
    """Turn the agent-facing history into a JSON-safe list. The history may hold
    plain ``{role, content}`` dicts AND/OR LangChain message OBJECTS
    (HumanMessage / AIMessage) — the latter are NOT json-serializable, which is
    the crash this guards. ``convert_to_messages`` normalizes the mixed list to
    BaseMessage, then ``messages_to_dict`` makes it JSON-safe; on any failure a
    per-item best-effort keeps the save from ever raising."""
    msgs = list(messages or [])
    try:
        from langchain_core.messages import (messages_to_dict,
                                             convert_to_messages)
        return messages_to_dict(convert_to_messages(msgs))
    except Exception:
        out = []
        for m in msgs:
            if isinstance(m, dict):
                out.append(m)
            else:
                out.append({"role": getattr(m, "type", "assistant"),
                            "content": str(getattr(m, "content", m))})
        return out


def deserialize_messages(data) -> list:
    """Inverse of :func:`serialize_messages` -> plain ``{role, content}`` dicts
    (the form the replay path feeds ``agent.stream``). Handles BOTH the
    ``messages_to_dict`` form (``{type, data}``) and older plain-dict files."""
    if not isinstance(data, list) or not data:
        return []
    if all(isinstance(m, dict) and "type" in m and "data" in m for m in data):
        try:
            from langchain_core.messages import messages_from_dict
            return [_msg_to_plain(m) for m in messages_from_dict(data)]
        except Exception:
            pass
    return [m if isinstance(m, dict) else _msg_to_plain(m) for m in data]


def save_messages(thread_id: str, messages: list,
                  root: Optional[str] = None) -> None:
    """Persist the agent-facing message history (small; full rewrite). Robust to
    LangChain message objects in the history (see :func:`serialize_messages`)."""
    os.makedirs(conversation_dir(thread_id, root), exist_ok=True)
    with open(_conv_path(thread_id, "messages.json", root), "w",
              encoding="utf-8") as fh:
        _json.dump(serialize_messages(messages), fh, ensure_ascii=False)


def load_messages(thread_id: str, root: Optional[str] = None) -> list:
    """Load the agent-facing message history (replayed on resume) as plain
    ``{role, content}`` dicts."""
    p = _conv_path(thread_id, "messages.json", root)
    try:
        with open(p, encoding="utf-8") as fh:
            data = _json.load(fh)
    except (OSError, ValueError):
        return []
    return deserialize_messages(data)


def save_attachments_index(thread_id: str, keys: Iterable[str],
                           root: Optional[str] = None) -> None:
    """Record the staged upload keys so they can be re-registered on resume."""
    os.makedirs(conversation_dir(thread_id, root), exist_ok=True)
    with open(_conv_path(thread_id, "attachments.json", root), "w",
              encoding="utf-8") as fh:
        _json.dump(list(keys or []), fh, ensure_ascii=False)


def load_attachments(thread_id: str, attachments: dict,
                     root: Optional[str] = None) -> List[Attachment]:
    """Re-register a conversation's staged uploads into the live ``attachments``
    dict by reading each key's bytes from the conversation files dir. Returns the
    :class:`Attachment` list (so callers can rebuild the agent-facing note)."""
    p = _conv_path(thread_id, "attachments.json", root)
    files_dir = conversation_files_dir(thread_id, root)
    try:
        with open(p, encoding="utf-8") as fh:
            keys = _json.load(fh)
    except (OSError, ValueError):
        return []
    out: List[Attachment] = []
    for key in keys or []:
        fpath = os.path.join(files_dir, key)
        try:
            with open(fpath, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        attachments[key] = data
        out.append(Attachment(key=key, path=fpath, size=len(data)))
    return out


def delete_conversation(thread_id: str, root: Optional[str] = None) -> Optional[str]:
    """Move a conversation directory into ``<data_root>/.trash/`` (soft delete).
    Returns the trash path, or ``None`` if the conversation did not exist."""
    src = conversation_dir(thread_id, root)
    if not os.path.isdir(src):
        return None
    trash = os.path.join(root or data_root(), ".trash")
    os.makedirs(trash, exist_ok=True)
    dst = os.path.join(trash, f"{thread_id}_{int(_time.time())}")
    _shutil.move(src, dst)
    return dst


def artifacts_from_transcript(transcript: Iterable[dict]) -> List[str]:
    """Rebuild the download list (unique artifact paths, in order) from a loaded
    transcript's per-turn artifact references."""
    out: List[str] = []
    seen = set()
    for entry in transcript or []:
        for p in entry.get("artifacts", []) or []:
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out

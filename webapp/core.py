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
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

#: Accepted upload types (mirrors the notebook FileUpload accept list). Word
#: and Excel files are read as Markdown (``funhouse_agent.office_text``) by
#: ``open_document`` / ``read_document`` / ``read_text_file``; until live
#: smoke wave 2b (C2) the uploader refused them.
ACCEPTED_UPLOAD_TYPES = [
    "pdf", "png", "jpg", "jpeg", "tif", "tiff",
    "dxf", "csv", "txt", "xml", "diggs",
    "docx", "xlsx", "xlsm",
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


#: Names a browser gives a picture pasted from the clipboard.
_CLIPBOARD_NAMES = {"image.png", "image.jpg", "image.jpeg", "image.gif",
                    "image.webp", "image.bmp", "blob"}


def pasted_upload_name(name, index: int = 0, when: Optional[float] = None
                       ) -> str:
    """A screenshot pasted into the chat box arrives called ``image.png``
    every time; a second paste would be taken for the first (attachments are
    deduplicated by name). So a clipboard name becomes
    ``pasted_<YYYYmmdd_HHMMSS>[_<n>].<ext>``; any other name is kept."""
    base = os.path.basename(str(name or "").replace("\\", "/")).lower()
    if base not in _CLIPBOARD_NAMES:
        return str(name)
    ext = os.path.splitext(base)[1] or ".png"
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(
        time.time() if when is None else when))
    return f"pasted_{stamp}{'_' + str(index) if index else ''}{ext}"


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


def attachment_note(atts: List[Attachment], review: bool = False) -> str:
    """Build the system-style note telling the agent about staged attachments —
    the attachment key (for analyze_image / analyze_pdf_page / read_pdf_text)
    AND that the same NAME is a file in the working folder (for pdf_import /
    dxf_import / drawing_ir tools that need a file path: a bare name is
    looked up there). Returns ``""`` for an empty list.

    No server path: the note named the absolute staging path until
    2026-10-09, and the model used it and repeated it to users (live smoke
    wave 1, A6). Every file tool resolves the bare name.

    ``review=True`` is the Document Review page's wording when that page runs
    its own lean agent (``GEOTECH_REVIEW_AGENT=lean``): it names only tools
    that agent has. With the switch off the note is the one it always was, so
    the legacy page is unchanged."""
    atts = list(atts or [])
    if not atts:
        return ""
    try:
        from funhouse_agent import review_flags
        lean = review and review_flags.lean_agent()
    except Exception:
        lean = False
    if lean:
        lines = ["[System note] The user attached files, available to you as:"]
        for a in atts:
            lines.append(f"- '{a.key}': open it with open_document(source="
                         f"'{a.key}'); the page and region tools take the same "
                         f"name (it is a file of that name in the working "
                         f"folder).")
        return "\n".join(lines)
    lines = ["[System note] The user attached files, available to you as:"]
    for a in atts:
        lines.append(
            f"- '{a.key}': attachment key '{a.key}' "
            f"(to review a PDF, open_document with source='{a.key}'; use "
            f"analyze_image / analyze_pdf_page / read_pdf_text with "
            f"attachment_key='{a.key}'); it is also the file '{a.key}' in the "
            f"working folder (tools that take a file path -- pdf_import / "
            f"dxf_import / drawing_ir, a call_agent file_path -- take that "
            f"name)."
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

#: What the ``save_file`` tool result says once a file lands in the conversation
#: files dir. That directory IS the chat-attachment surface — its contents show
#: up as download cards in the transcript and in the sidebar — but the model had
#: no way to know it, so it wrote reports there and told the user in the same
#: breath that it "can't directly attach the binary PDF into the chat stream"
#: (field feedback 2026-09-04, Praia_Downdrag). Awareness gap, not a capability
#: gap: the tool result now says what the save already accomplished.
ATTACHMENT_NOTE = (
    "This file is now ATTACHED to this conversation — it appears as a download "
    "card in the chat and in the sidebar file list, so the user can already "
    "open it. Tell the user it is attached; do NOT say you cannot attach files."
)


def attachment_note_for(path, files_dir: str) -> Optional[str]:
    """``ATTACHMENT_NOTE`` when ``path`` landed inside ``files_dir`` (the
    conversation's chat-attachment surface), else ``None``.

    Saves into a custom working folder are bridged into ``files_dir`` after the
    turn (see ``import_external_artifacts``) and so become attachments too, but
    only at turn end — this note describes what is already true at save time.
    """
    try:
        ap = os.path.abspath(str(path))
        fd = os.path.abspath(files_dir)
    except (TypeError, ValueError):
        return None
    if ap == fd or ap.startswith(fd + os.sep):
        return ATTACHMENT_NOTE
    return None


def make_save_fn(temp_dir: str, artifacts: List[str]) -> Callable[[str, object], str]:
    """Build a ``save_fn(path, content) -> saved_path`` for ``build_deep_agent``.

    A bare/relative path is resolved into the session ``temp_dir`` (by
    basename) so the app can serve it; an absolute path is honored as-is. Every
    resolved path is appended to ``artifacts`` (deduplicated) for the download
    list. Bytes or text content are both handled.

    The returned callable also carries a ``saved_note(path) -> str | None``
    attribute, the hook ``funhouse_agent.deep.tools`` uses to let the host add
    a line to the ``save_file`` tool result (here: "this file is now attached").
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
        # A REWRITE of a file saved in an earlier turn is not appended again
        # (the download list stays one entry per file); it still gets this
        # turn's card, because the turn compares file times
        # (:func:`rewritten_files`, live smoke wave 1, A13). Tool scratch
        # (a dot-folder, a cache folder) is never a deliverable.
        if p not in artifacts and not is_cache_path(p, temp_dir):
            artifacts.append(p)
        return p

    save_fn.saved_note = lambda saved: attachment_note_for(saved, temp_dir)
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


#: A Plotly sidecar and the static image it supersedes share everything but
#: this suffix (``lateral_pressure.png`` / ``lateral_pressure.plotly.json``).
PLOTLY_SIDECAR_SUFFIX = ".plotly.json"
_SUPERSEDED_IMAGE_EXTS = (".png", ".jpg", ".jpeg")
#: Every twin a sidecar's card stands for: the static image, and the
#: self-contained HTML copy a plot method also writes (a 4.8 MB second card
#: for the same figure in live smoke wave 1, A15d).
_SUPERSEDED_BY_PLOTLY_EXTS = _SUPERSEDED_IMAGE_EXTS + (".html", ".htm")


def plotly_download_twin(path: str) -> str:
    """The file a plotly card's Download button serves: the figure's PNG
    (else JPG, else its HTML copy) beside the ``*.plotly.json``; the sidecar
    itself only when there is no twin -- a reader downloading a chart wants
    the picture, not Plotly's JSON (live smoke wave 1, A15d)."""
    p = str(path)
    if not p.lower().endswith(PLOTLY_SIDECAR_SUFFIX):
        return p
    stem = p[:-len(PLOTLY_SIDECAR_SUFFIX)]
    for ext in (".png", ".jpg", ".jpeg", ".html", ".htm"):
        for cand in (stem + ext, stem + ext.upper()):
            if os.path.isfile(cand):
                return cand
    return p


def plotly_picture_twin(path: str) -> Optional[str]:
    """The static picture (PNG, else JPG) beside a ``*.plotly.json`` sidecar,
    or ``None`` when ``path`` is not a sidecar or has no picture. The card
    list keeps only the sidecar (:func:`collect_turn_artifacts`), so the
    per-turn file note names the picture from here (live smoke wave 2a,
    B10: "put that plot in the memo" had no image to find)."""
    p = str(path)
    if not p.lower().endswith(PLOTLY_SIDECAR_SUFFIX):
        return None
    stem = p[:-len(PLOTLY_SIDECAR_SUFFIX)]
    for ext in _SUPERSEDED_IMAGE_EXTS:
        for cand in (stem + ext, stem + ext.upper()):
            if os.path.isfile(cand):
                return cand
    return None


def _plotly_sidecar_stems(paths: Iterable[str]) -> set:
    """Lower-cased paths-without-suffix of every ``*.plotly.json`` in ``paths``."""
    stems = set()
    for p in paths or ():
        low = str(p).lower()
        if low.endswith(PLOTLY_SIDECAR_SUFFIX):
            stems.add(low[:-len(PLOTLY_SIDECAR_SUFFIX)])
    return stems


def _superseded_by_plotly(path: str, stems: set) -> bool:
    """True when ``path`` is the static image -- or the self-contained HTML
    copy -- of a figure that also has an interactive sidecar in the same
    list."""
    low = str(path).lower()
    if not low.endswith(_SUPERSEDED_BY_PLOTLY_EXTS):
        return False
    return os.path.splitext(low)[0] in stems


def collect_turn_artifacts(save_new: Iterable[str],
                           dir_new: Iterable[str]) -> List[str]:
    """Associate a turn's artifacts: the union of the paths the save_fn recorded
    during the turn and the new files the directory diff found, deduplicated and
    order-preserving (save_fn first).

    One figure, one card: when a ``*.plotly.json`` sidecar is present, the PNG
    (or JPG, or the self-contained HTML copy) of the SAME figure is dropped
    from this list, so the chat shows the interactive chart rather than a
    chart and a picture of it; the card's Download serves the PNG
    (:func:`plotly_download_twin`). Only the
    CARD list is filtered — the caller's own artifact list still carries the
    image, so the SharePoint mirror, the sidebar downloads and ``html_to_pdf``
    are untouched."""
    out: List[str] = []
    seen = set()
    for p in list(save_new or []) + list(dir_new or []):
        if p not in seen:
            seen.add(p)
            out.append(p)
    stems = _plotly_sidecar_stems(out)
    if not stems:
        return out
    return [p for p in out if not _superseded_by_plotly(p, stems)]


#: Cache and scratch folders tools keep in the working folder: their files are
#: working data, not deliverables, so they never become download cards, are
#: never listed to the model as the conversation's files, and are not
#: mirrored to SharePoint (``sharepoint_store`` prunes them; see
#: :func:`mirror_skips_dir`). ``digest`` = the Document Review digest
#: (funhouse_agent.review_digest: code only, rebuilt from the PDF in
#: seconds); ``.scratch`` = the per-conversation tool scratch (contact
#: sheets, thumbnails). Any other dot-folder is treated the same way.
CACHE_DIRS = ("digest", ".scratch")


def _skipped_dir(name: str) -> bool:
    return name in CACHE_DIRS or name.startswith(".")


def is_cache_path(path, files_dir: str) -> bool:
    """True when ``path`` lies in a cache or scratch folder of ``files_dir``
    (a top-level :data:`CACHE_DIRS` entry, or any dot-folder at any depth)."""
    try:
        ap = os.path.abspath(str(path))
        fd = os.path.abspath(files_dir)
    except (TypeError, ValueError):
        return False
    if not ap.startswith(fd + os.sep):
        return False
    parts = os.path.relpath(ap, fd).split(os.sep)[:-1]
    return bool(parts) and (parts[0] in CACHE_DIRS
                            or any(p.startswith(".") for p in parts))


def mirror_skips_dir(rel_dir: str) -> bool:
    """Whether the SharePoint mirror leaves out a conversation sub-folder,
    given relative to the CONVERSATION folder with '/' separators: the
    working folder's cache and scratch folders (``files/digest``,
    ``files/.scratch``, any dot-folder under ``files/``)."""
    parts = [p for p in str(rel_dir or "").replace("\\", "/").split("/")
             if p and p != "."]
    if len(parts) < 2 or parts[0] != "files":
        return False
    return parts[1] in CACHE_DIRS or any(p.startswith(".")
                                         for p in parts[1:])


#: Files that exist only while a turn runs or a write is half done: the
#: turn's checkpoint in the conversation folder (:func:`begin_partial`) ...
IN_FLIGHT_FILES = ("partial.json",)
#: ... and the part files a write renames into place when it is complete
#: (``.part``: SharePoint downloads, the coverage ledger, reference PDFs;
#: ``.tmp``: the downloads ledger).
IN_FLIGHT_SUFFIXES = (".part", ".tmp")


def is_in_flight_file(rel_path) -> bool:
    """Whether a conversation file -- given relative to the CONVERSATION
    folder, '/' separators -- is an in-progress file the SharePoint mirror
    never uploads and a restore never downloads.

    Live smoke wave 2a (B2): "save it to SharePoint" mirrors the conversation
    in the MIDDLE of a turn, which uploaded that turn's ``partial.json``; the
    end-of-turn mirror never removed it, and restoring the conversation later
    turned it into a fake "interrupted" turn. ``partial.json`` counts only at
    the top of the conversation folder: a file of that name a user put in
    ``files/`` is theirs."""
    rel = str(rel_path or "").replace("\\", "/").strip("/")
    if not rel:
        return False
    if rel in IN_FLIGHT_FILES:
        return True
    return rel.lower().endswith(IN_FLIGHT_SUFFIXES)


def snapshot_dir(temp_dir: str) -> set:
    """Return the set of file paths currently under ``temp_dir`` (recursive),
    leaving out the top-level :data:`CACHE_DIRS` and every dot-folder."""
    found = set()
    top = os.path.abspath(temp_dir)
    for root, dirs, names in os.walk(temp_dir):
        if os.path.abspath(root) == top:
            dirs[:] = [d for d in dirs if not _skipped_dir(d)]
        else:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
        for n in names:
            found.add(os.path.join(root, n))
    return found


def snapshot_mtimes(temp_dir: str) -> Dict[str, Tuple[int, int]]:
    """``{path: (mtime_ns, size)}`` for :func:`snapshot_dir`'s files -- taken
    before a turn so a file REWRITTEN during it is found (A13)."""
    out: Dict[str, Tuple[int, int]] = {}
    for p in snapshot_dir(temp_dir):
        try:
            st = os.stat(p)
        except OSError:
            continue
        out[p] = (st.st_mtime_ns, st.st_size)
    return out


def rewritten_files(temp_dir: str, before: Optional[dict],
                    input_paths: Iterable[str] = ()) -> List[str]:
    """Files that existed before the turn (``before`` from
    :func:`snapshot_mtimes`) and were written again during it -- a memo
    rewritten under the same name gets its card in the turn that rewrote it
    (live smoke wave 1, A13: the only card was the earlier turn's). Staged
    inputs are left out. Sorted."""
    if not before:
        return []
    inputs = {os.path.abspath(p) for p in (input_paths or ())}
    out = []
    for p, stamp in before.items():
        if os.path.abspath(p) in inputs:
            continue
        try:
            st = os.stat(p)
        except OSError:
            continue
        if (st.st_mtime_ns, st.st_size) != tuple(stamp):
            out.append(p)
    return sorted(out)


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

def _register_reference_fetcher() -> None:
    """Reference PDFs from SharePoint ``primary_references/``, fetched the
    first time a chart needs one (owner decision 2026-09-15). Process-wide and
    harmless to repeat; a no-op without SharePoint."""
    try:
        from webapp import reference_fetch
        reference_fetch.register_if_configured()
    except Exception:                                  # noqa: BLE001
        pass


def _add_feedback_tool(kw: dict, temp_dir: str) -> None:
    """Splice the ``record_feedback`` tool and its instructions into
    ``build_deep_agent`` kwargs (owner feedback 2026-09-11).

    The agent records capability gaps / tool errors / in-chat feedback into
    the CONVERSATION directory (temp_dir is <conversation>/files, so its parent
    is the record dir). It reaches the primary (``extra_tools``) and EVERY
    sub-agent (``subagent_extra_tools``: references, reviewer, calc,
    model_setup) — a sub-agent is where most gaps are found: calc when no tool
    draws the figure, references when a chart's source PDF is missing. Until
    2026-09-15 only primary + calc had it. Best-effort: never blocks a build.
    """
    try:
        from webapp import feedback as _fb
        _conv_dir = os.path.dirname(os.path.abspath(temp_dir))
        _thread_id = os.path.basename(_conv_dir)

        def _fb_context(_tid=_thread_id):
            try:
                tr = load_transcript(_tid)
                turn = sum(1 for e in tr if e.get("role") == "user")
                meta = load_meta(_tid) or {}
                return {"thread_id": _tid, "turn": turn,
                        "model": meta.get("model")}
            except Exception:                              # noqa: BLE001
                return {"thread_id": _tid}

        _fb_tools, _fb_prompt = _fb.tools_for(_conv_dir, _fb_context)
    except Exception:
        return
    for tools_key, prompt_key in (
            ("extra_tools", "extra_system_prompt"),
            ("subagent_extra_tools", "subagent_extra_system_prompt")):
        kw[tools_key] = list(kw.get(tools_key) or []) + _fb_tools
        kw[prompt_key] = "\n\n".join(
            p for p in (kw.get(prompt_key), _fb_prompt) if p)


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
    # SharePoint file tools (browse/search/download/upload) — injected ONLY
    # when the deployment is configured for SharePoint, so everyone else
    # carries zero extra tool surface. Best-effort: a tools-layer problem
    # must never block building the agent.
    try:
        from webapp import sharepoint_tools
        # thread id = the conversation dir name (temp_dir is <conv>/files), so
        # an upload with no destination lands in this conversation's folder;
        # the conversation dir itself keeps the downloads ledger.
        _conv_dir = os.path.dirname(os.path.abspath(temp_dir))
        _sp_tools, _sp_prompt = sharepoint_tools.tools_if_configured(
            thread_id=os.path.basename(_conv_dir), record_dir=_conv_dir)
    except Exception:
        _sp_tools, _sp_prompt = [], ""
    if _sp_tools:
        kw["extra_tools"] = list(kw.get("extra_tools") or []) + _sp_tools
        kw["extra_system_prompt"] = "\n\n".join(
            p for p in (kw.get("extra_system_prompt"), _sp_prompt) if p)
    # Email tool (email a produced file to a colleague) — injected ONLY when
    # the Funhouse email SDK is importable. Same best-effort rule as above.
    try:
        from webapp import email_tools
        _em_tools, _em_prompt = email_tools.tools_if_available()
    except Exception:
        _em_tools, _em_prompt = [], ""
    if _em_tools:
        kw["extra_tools"] = list(kw.get("extra_tools") or []) + _em_tools
        kw["extra_system_prompt"] = "\n\n".join(
            p for p in (kw.get("extra_system_prompt"), _em_prompt) if p)
    _add_feedback_tool(kw, temp_dir)
    _register_reference_fetcher()
    # The coverage ledger (GEOTECH_COVERAGE, off by default) lives in the
    # conversation's folder beside activity.jsonl, so it outlives a rebuild
    # of the agent and is mirrored with the conversation. Unused when off.
    kw.setdefault("coverage_dir", os.path.dirname(os.path.abspath(temp_dir)))
    if kw.get("review_page"):
        # The Document Review page's lean agent binds its findings and
        # digests to THIS conversation's folder when it is built; the
        # process-wide working folder can be re-pointed by another tab.
        # (The legacy build accepts and ignores it.)
        kw.setdefault("working_dir", temp_dir)
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
    "pavement": "make_pavement_specialist_deep",
    "structural": "make_structural_specialist_deep",
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
    kw = dict(build_kwargs)
    # Specialists never pass through build_agent, so they had no feedback
    # tool at all before 2026-09-15.
    _add_feedback_tool(kw, temp_dir)
    _register_reference_fetcher()
    return builder(model, attachments=attachments,
                   save_fn=make_save_fn(temp_dir, artifacts), **kw)


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

#: A reply this long is an answer, not a stop on a stated step -- unless its
#: last line opens something it never delivers (it ends on ":" or "…").
MID_TASK_MAX_CHARS = 600

# The last sentence must OPEN with the speaker's own next step ("Let me …",
# "Now I'll …", "Next, I will …"), after an optional list marker or
# discourse word. "I'll" further in ("… tell me and I'll re-map them") is an
# offer, not a step being taken.
_LEAD_INTENT_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|now|next|first|then|so|right|alright|good|great)"
    r"[,:]?\s+)*"
    r"(?:let me|let's|let us|i'?ll|i will|i'?m going to|i am going to|"
    r"i need to|i'?m now going to)\b", re.IGNORECASE)
# Anything in the last sentence that makes the step wait on the user, or
# puts it to the user, makes the reply a finished answer: a condition
# ("if", "once", "when" … "then"), a request ("tell me", "give me",
# "send"), an offer ("I can", "happy to", "would you"), or the user named at
# all ("you", "your").
_ADDRESSED_RE = re.compile(
    r"\b(if|once|when|whenever|after|as soon as|then|unless|until|"
    r"you|your|yours|you'?(?:d|ll|re|ve)|let me know|tell me|give me|"
    r"send me|show me|say|ask|happy|glad|feel free|want|prefer|wish|"
    r"can|could|would|should|may|might|wait|await|ready|available|"
    r"needed|necessary|required)\b", re.IGNORECASE)
_LIST_MARK_RE = re.compile(r"^(?:[-*•]+|\d+[.)]|\(\w\))\s*")


def _last_sentence(text: str) -> str:
    """The final sentence of ``text``'s last non-empty line, list marker
    and emphasis stripped."""
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    line = _LIST_MARK_RE.sub("", lines[-1]).strip().strip("*_").strip()
    parts = [p for p in re.split(r"(?<=[.!?])\s+", line) if p.strip()]
    if not parts:
        return ""
    tail = parts[-1].strip()
    # "2. Give me the name" splits as "2." + "Give me ..."; a bare number is
    # never the sentence.
    return _LIST_MARK_RE.sub("", tail).strip().strip("*_").strip()


def ends_mid_task(text: str, saw_tool_call: bool) -> bool:
    """True when an assistant reply STOPPED on a step it announced and did
    not take ("Let me get that Ka …" with no tool call behind it).

    An offer to the user ("If your roles are different, tell me and I'll
    re-map them"), a request ("Give me the file name and I'll look for it"),
    a step that waits on the user ("Once I have the log, I'll check each
    submittal") and a question are FINISHED answers. Live smoke wave 2b
    (C1): the old rule fired on any "I'll" in the last sentence, nudged on
    three such endings and delivered each answer two or three times, once
    with a false confession that a save had not happened.

    So it fires only when the turn used tools, the reply does not end with a
    question, its last sentence OPENS with the speaker's own next step and
    has nothing in it that hands the step to the user, and the reply is
    short (narration, :data:`MID_TASK_MAX_CHARS`) or its last line opens
    something it never delivers (ends on ":" or "…"). A missed stop costs
    the old behaviour (the user asks again); a false one costs a repeated,
    contradicting answer -- so it errs towards missing.
    """
    t = (text or "").strip()
    if not t or not saw_tool_call or t.endswith("?"):
        return False
    tail = _last_sentence(t)
    if not tail or tail.endswith("?") or not _LEAD_INTENT_RE.match(tail):
        return False
    if _ADDRESSED_RE.search(tail):
        return False
    open_end = t.endswith((":", "…", "...")) or tail.endswith((":", "…",
                                                               "..."))
    return len(t) <= MID_TASK_MAX_CHARS or open_end


def _without_stated_step(text: str) -> str:
    """``text`` less the trailing sentence that announced the step a
    continuation pass then took ("Ka is next. Let me get that Ka." -> "Ka is
    next."); ``""`` when that sentence was all of it."""
    t = str(text or "").rstrip()
    tail = _last_sentence(t)
    if not tail:
        return t.strip()
    cut = t.rfind(tail)
    if cut < 0:
        return t.strip()
    body = t[:cut].rstrip()
    # a list marker or emphasis left dangling on its own
    body = re.sub(r"(?:\n|^)\s*(?:[-*•]+|\d+[.)])?\s*[*_]*\s*$", "", body)
    return body.strip()


def _norm_text(text: str) -> str:
    return " ".join(str(text or "").split()).lower()


def merge_continuation(previous: str, following: str) -> str:
    """The answer after an auto-continue pass: what the earlier reply said
    before the step it announced, then the reply that took the step.

    The announcement itself ("Let me get that Ka.") is dropped -- the next
    reply is what came of it -- and so is the earlier text when the next
    reply already repeats it, so the user never gets one answer twice. An
    empty next reply leaves the earlier one standing."""
    nxt = str(following or "").strip()
    if not nxt:
        return str(previous or "").strip()
    body = _without_stated_step(previous)
    if not body or _norm_text(body) in _norm_text(nxt):
        return nxt
    return f"{body}\n\n{nxt}"


def _run_messages(chunk) -> Optional[list]:
    """The whole message list of a ``values``-mode stream item (the graph's
    state after a step), or ``None``."""
    if isinstance(chunk, dict) and isinstance(chunk.get("messages"),
                                              (list, tuple)):
        return list(chunk["messages"])
    return None


def _new_run_messages(chunk) -> list:
    """The messages an ``updates``-mode item ADDS to the run (model and tool
    results), for a host whose stream has no ``values`` mode. A node that
    rewrites the whole history adds nothing here."""
    from funhouse_agent.deep.notebook import _update_messages
    out: list = []
    if not isinstance(chunk, dict):
        return out
    for update in chunk.values():
        for msg in _update_messages(update):
            kind = (msg.get("role") if isinstance(msg, dict)
                    else getattr(msg, "type", None))
            if kind in ("ai", "assistant", "tool", "AIMessageChunk"):
                out.append(msg)
    return out


# --- The coverage gate's held-back reply (Foundry brief 5, CV2/N4) -----------
# With GEOTECH_COVERAGE on, the gate can stop the model as it finishes, take
# its reply out of the conversation and give it a note (the pages nobody read;
# state coverage as counts) in its place. The reply streamed before that note
# is then NOT the answer: the reply after it is, whole. Brief 5 delivered
# every gated answer as the first reply glued to the second (10 of 10 on the
# new tasks, 5 of 6 on ordinary ones), sometimes mid-line.

#: The status line shown when the gate holds a reply back.
GATE_STATUS = ("coverage check: the app listed what has not been read; the "
               "answer is being written again with it")


def coverage_gate_spoke(chunk) -> bool:
    """Whether an ``updates``-mode stream item carries the coverage gate's
    note (a user-role message opening with the gate's prefix)."""
    if not isinstance(chunk, dict):
        return False
    from funhouse_agent.coverage import GATE_PREFIX
    for update in chunk.values():
        if not isinstance(update, dict):
            continue
        msgs = update.get("messages")
        if not isinstance(msgs, (list, tuple)):
            continue
        for msg in msgs:
            if isinstance(msg, dict):
                kind, content = msg.get("role"), msg.get("content")
            else:
                kind = getattr(msg, "type", None)
                content = getattr(msg, "content", None)
            if kind in ("human", "user") and isinstance(content, str) \
                    and content.startswith(GATE_PREFIX):
                return True
    return False


#: Seconds of stream silence before a heartbeat item is emitted. Behind the
#: Databricks driver proxy, a websocket with no traffic for ~1-2 min gets
#: killed ("Connecting" flaps, orphaned turns — observed live through 5.10.2):
#: a long reasoning-model call or a slow tool (big PDF read, SharePoint
#: download) produces exactly that silence. The app turns each heartbeat into
#: a status-label update, which IS websocket traffic, keeping the connection
#: alive. Override via GEOTECH_HEARTBEAT_S (<=0 disables).
HEARTBEAT_INTERVAL_S = 15.0


def heartbeat_interval() -> float:
    raw = os.environ.get("GEOTECH_HEARTBEAT_S", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return HEARTBEAT_INTERVAL_S


def with_heartbeat(gen, interval_s: Optional[float] = None):
    """Yield ``gen``'s items as they arrive, inserting
    ``{"kind": "heartbeat", "elapsed_s": float}`` items whenever the stream is
    silent for ``interval_s`` seconds.

    The wrapped generator runs in a daemon worker thread feeding a queue; the
    consumer polls with a timeout. Exceptions from ``gen`` re-raise here (same
    surface as consuming ``gen`` directly). ``interval_s <= 0`` disables the
    wrapper entirely (items pass straight through on the caller's thread).
    """
    import queue as _queue
    import threading as _threading
    import time as _time

    interval = heartbeat_interval() if interval_s is None else float(interval_s)
    if interval <= 0:
        yield from gen
        return

    q: "_queue.Queue" = _queue.Queue()

    def _pump():
        try:
            for item in gen:
                q.put(("item", item))
            q.put(("done", None))
        except BaseException as exc:                    # re-raised on consumer
            q.put(("exc", exc))

    _threading.Thread(target=_pump, daemon=True,
                      name="geotech-turn-pump").start()
    t0 = _time.monotonic()
    while True:
        try:
            kind, payload = q.get(timeout=interval)
        except _queue.Empty:
            yield {"kind": "heartbeat",
                   "elapsed_s": round(_time.monotonic() - t0, 1)}
            continue
        if kind == "item":
            yield payload
        elif kind == "done":
            return
        else:
            raise payload


def stream_turn(agent, messages: list, thread_id: str,
                max_result_chars: int = 2000,
                recursion_limit: Optional[int] = None,
                callbacks: Optional[list] = None,
                turn_note: Optional[str] = None):
    """Stream ONE turn from the compiled deep agent.

    ``callbacks`` (optional) are LangChain callback handlers attached to the
    run config alongside the usage-metadata callback; they propagate into
    sub-agent invocations (the ``activity_log.ActivityLogger`` rides here).

    ``turn_note`` (optional) is put in front of the turn's user message for
    THIS run only (:func:`with_turn_note`) -- the app's per-turn note of the
    files the conversation already holds (:func:`working_files_note`).

    ``messages`` is the full agent-facing history INCLUDING the new user turn
    (the caller appends it and, on completion, appends the assistant answer from
    the ``turn_done`` item). Yields entry dicts:

    * ``{"kind": "token", "text": str}`` — a streamed answer token.
    * ``{"kind": "tool_call"|"todos"|"tool_result", "text": str}`` — activity.
    * ``{"kind": "turn_done", "answer": str, "turn_tokens": int}`` — final,
      carrying the answer and the token spend for this turn (aggregated
      across every model call in the run, sub-agents included).

    The ANSWER is the run's final AI message -- the last one that requested
    no tool, read off the ``updates`` stream -- not every token the model
    streamed: narration between tool calls ("I'll open the page map.") is
    shown live but is not the reply, and a context-summarization call's
    handoff never becomes text at all (live smoke wave 1, A3). The live view
    puts a paragraph break between model calls. Only when the stream carried
    no such message (an agent that streams tokens alone) is the streamed text
    the answer. An auto-continue pass (:func:`ends_mid_task`) goes on from
    the run's own messages, and its reply takes the place of the step the
    pass before it announced (:func:`merge_continuation`): one answer, never
    two copies of it.

    When the coverage gate holds a reply back (:func:`coverage_gate_spoke`),
    the reply before its note is left out of ``answer``: the reply after the
    note is the answer (or, if the model gave nothing after it, the reply
    held back).

    Reuses the PURE ``_format_update`` parser and ``_sum_callback_tokens`` from
    ``funhouse_agent.deep.notebook`` so the stream contract is shared with the
    notebook UI. The whole stream runs under a usage-metadata callback; if that
    is unavailable the turn still streams (token spend simply reports 0).
    """
    from funhouse_agent.deep.notebook import (_final_answer_in_update,
                                              _format_update,
                                              _sum_callback_tokens)

    answer_parts: List[str] = []
    saw_tool = False
    work_messages = with_turn_note(messages, turn_note)
    continuations = 0
    config = {"configurable": {"thread_id": thread_id}}
    # An agent that ends long requests itself (the lean review agent's
    # model-call budget) says how many graph steps that needs; the host's cap
    # is raised to it so the budget, not a GraphRecursionError, ends the turn.
    floor = getattr(agent, "geotech_min_recursion_limit", None)
    if floor:
        recursion_limit = max(int(recursion_limit or 0), int(floor))
    # A build with the coverage gate (GEOTECH_COVERAGE) lets a turn that takes
    # data out of a document run to a higher cap (150, owner 2026-10-08). The
    # run is given that cap, and the ordinary cap travels with it as the
    # turn's step allowance: an ordinary turn is ended there with an answer
    # by the gate, and an extraction turn runs on.
    extraction = getattr(agent, "geotech_extraction_recursion_limit", None)
    if extraction:
        allowance = int(recursion_limit or DEFAULT_BEHAVIOR["recursion_limit"])
        if int(extraction) > allowance:
            from funhouse_agent.deep.coverage_tools import ALLOWANCE_KEY
            config["configurable"][ALLOWANCE_KEY] = allowance
            recursion_limit = int(extraction)
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
            held_back: Optional[str] = None
            # The reply: the last AI message of this pass that asked for no
            # tool (None until one arrives). ``boundary``: a model call has
            # finished since the last streamed token, so the next token opens
            # a new paragraph in the live view.
            final_text: Optional[str] = None
            boundary = False
            # The run's own messages -- its tool calls and their results as
            # well as its replies -- so a continuation pass goes on from what
            # the run DID, not from a text-only history (wave 2b, C1: the
            # second pass lost its own save and "confessed" it had never
            # happened). ``values`` gives the graph's whole state; a host
            # without that mode falls back to what the updates added.
            run_state: Optional[list] = None
            added: list = []
            for mode, chunk in agent.stream(
                    {"messages": work_messages}, config=run_config,
                    stream_mode=["updates", "messages", "values"]):
                if mode == "values":
                    state = _run_messages(chunk)
                    if state is not None:
                        run_state = state
                    continue
                if mode == "updates":
                    added.extend(_new_run_messages(chunk))
                if mode == "updates" and coverage_gate_spoke(chunk):
                    # The coverage gate (GEOTECH_COVERAGE) held the reply
                    # streamed so far back from the user and asked for the
                    # whole answer again: the reply after its note is THE
                    # answer, never the two glued (Foundry brief 5, CV2/N4).
                    held_back = (final_text if final_text is not None
                                 else "".join(pass_parts))
                    pass_parts = []
                    final_text = None
                    boundary = False
                    yield {"kind": "tool_call", "text": GATE_STATUS}
                    if held_back.strip():
                        # Keeps the live view from running the two replies
                        # together; the delivered answer is turn_done's.
                        yield {"kind": "token", "text": "\n\n"}
                elif mode == "updates":
                    saw_ai, final = _final_answer_in_update(chunk)
                    if saw_ai:
                        boundary = True
                    if final is not None:
                        final_text = final
                for entry in _format_update(mode, chunk,
                                            max_result_chars=max_result_chars):
                    if entry["kind"] == "token":
                        if boundary and pass_parts and \
                                not pass_parts[-1].endswith("\n\n"):
                            # Narration of one model call must not run into
                            # the next call's text mid-sentence.
                            pass_parts.append("\n\n")
                            yield {"kind": "token", "text": "\n\n"}
                        boundary = False
                        pass_parts.append(entry["text"])
                    elif entry["kind"] == "tool_call":
                        saw_tool = True
                    yield entry
            streamed = "".join(pass_parts)
            pass_text = (final_text if final_text and final_text.strip()
                         else streamed)
            if held_back is not None and not pass_text.strip():
                # The model gave nothing after the note: the reply it had is
                # better than none.
                pass_text = held_back
            # ONE answer, however many passes: a continuation's reply takes
            # the place of the step its predecessor announced, never a
            # second copy of the answer (merge_continuation).
            answer_parts[:] = [merge_continuation("".join(answer_parts),
                                                  pass_text)
                               if continuations else pass_text]
            if (continuations < MAX_AUTO_CONTINUES
                    and ends_mid_task(pass_text, saw_tool)):
                continuations += 1
                if run_state is not None:
                    history = list(run_state)
                elif added:
                    history = list(work_messages) + added
                else:
                    history = list(work_messages) + [
                        {"role": "assistant", "content": pass_text}]
                work_messages = history + [
                    {"role": "user", "content": CONTINUE_NUDGE}]
                yield {"kind": "tool_call",
                       "text": (f"auto-continue {continuations}/"
                                f"{MAX_AUTO_CONTINUES}: finishing the stated "
                                "next step")}
                continue
            return

    extra_cbs = list(callbacks or [])

    def _passes_noting_reply(run_config):
        # A turn that fails keeps the reply it had completed (an earlier
        # pass's) on the error, for the host to show labelled as cut short
        # instead of the raw stream (turn_jobs, failed_turn_text).
        try:
            yield from _run_passes(run_config)
        except BaseException as exc:
            try:
                done = "".join(answer_parts).strip()
                if done and not getattr(exc, "geotech_reply", None):
                    exc.geotech_reply = done
            except Exception:  # noqa: BLE001 - never mask the error
                pass
            raise

    if cb_ctx is None:
        run_config = dict(config)
        if extra_cbs:
            run_config["callbacks"] = extra_cbs
        for entry in _passes_noting_reply(run_config):
            yield entry
        yield {"kind": "turn_done", "answer": "".join(answer_parts),
               "turn_tokens": 0}
        return

    with cb_ctx as cb:
        run_config = dict(config)
        run_config["callbacks"] = [cb] + extra_cbs
        for entry in _passes_noting_reply(run_config):
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


#: Env listing Foundry model RIDs for the picker, comma-separated; each entry
#: is either a bare RID or "Label=RID" (e.g. "GPT 5.2=ri.language-model-
#: service..language-model.gpt-5-2"). Set in the Foundry workspace so new RIDs
#: (e.g. Claude once enabled) plug in with no code change.
FOUNDRY_MODELS_ENV = "GEOTECH_FOUNDRY_MODELS"


def _parse_model_env(text: str, blurb: str) -> List[dict]:
    """Parse a ``Label=id,...`` model-list env value into picker entries."""
    out: List[dict] = []
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            label, mid = item.split("=", 1)
            label, mid = label.strip(), mid.strip()
        else:
            label, mid = item, item
        if mid and not any(c["id"] == mid for c in out):
            out.append({"id": mid, "label": label or mid, "blurb": blurb})
    return out


def foundry_model_choices(raw: Optional[str] = None) -> List[dict]:
    """Parse ``GEOTECH_FOUNDRY_MODELS`` (or ``raw``) into picker entries."""
    text = raw if raw is not None else os.environ.get(FOUNDRY_MODELS_ENV, "")
    return _parse_model_env(text, "Foundry model")


#: Env listing Prompter model ids for the picker on a deployment-provided
#: (registered-builder) engine, comma-separated ``Label=id`` or bare ids —
#: e.g. "GPT deep=funhouse-gpt-high,GPT 5.1 fast=funhouse-gpt-medium". Only
#: honored when the registered builder accepts a model id (see
#: engine_config.register_model_builder); the databricks launcher sets it.
PROMPTER_MODELS_ENV = "GEOTECH_PROMPTER_MODELS"


def prompter_model_choices(raw: Optional[str] = None) -> List[dict]:
    """Parse ``GEOTECH_PROMPTER_MODELS`` (or ``raw``) into picker entries."""
    text = raw if raw is not None else os.environ.get(PROMPTER_MODELS_ENV, "")
    return _parse_model_env(text, "Prompter model")


def model_choices(env_model: Optional[str] = None) -> List[dict]:
    """The picker list ``[{id,label,blurb}, ...]``. Foundry RIDs from
    ``GEOTECH_FOUNDRY_MODELS`` are PREPENDED (first one = default on a Foundry
    deployment); then, if ``GEOTECH_WEBAPP_MODEL`` (or the ``env_model``
    override) names a model not already listed, it is prepended above those.
    Returns fresh dicts (safe to mutate)."""
    envm = (env_model if env_model is not None
            else os.environ.get("GEOTECH_WEBAPP_MODEL"))
    from webapp import engine_config
    # Deployment-provided engine (registered builder) with a configured
    # Prompter model list: those are the ONLY sensible choices — the curated
    # Anthropic list cannot be served by the Prompter.
    if engine_config.has_model_builder():
        pm = prompter_model_choices()
        if pm:
            return pm
    # Foundry deployment: RIDs are the ONLY model surface — the curated
    # Anthropic list is not offered (and the key path is disabled in
    # engine_config), so an unconfigured deployment shows just the RID input.
    # A Tiny Apps deployment likewise never offers the curated list: its one
    # model arrives through the Prompter builder above, or not at all.
    if engine_config.is_keyless_deployment():
        choices = foundry_model_choices()
    else:
        choices = foundry_model_choices() + [dict(c) for c in MODEL_CHOICES]
    if envm and not any(c["id"] == envm for c in choices):
        choices.insert(0, {"id": envm, "label": envm,
                           "blurb": "from GEOTECH_WEBAPP_MODEL"})
    return choices


def default_model_id(env_model: Optional[str] = None) -> str:
    """The default selected model id: ``GEOTECH_WEBAPP_MODEL`` if set (whether
    or not it is already curated), else the first ``GEOTECH_FOUNDRY_MODELS``
    entry (a Foundry deployment defaults to its own models), else the first
    curated choice."""
    envm = (env_model if env_model is not None
            else os.environ.get("GEOTECH_WEBAPP_MODEL"))
    if envm:
        return envm
    from webapp import engine_config
    if engine_config.has_model_builder():
        pm = prompter_model_choices()
        if pm:
            return pm[0]["id"]
    fm = foundry_model_choices()
    if fm:
        return fm[0]["id"]
    # Foundry deployment with nothing configured: no default model exists —
    # the app boots engineless and offers the RID input only. Same on Tiny
    # Apps: no Prompter settings, no model.
    if engine_config.is_keyless_deployment():
        return ""
    return MODEL_CHOICES[0]["id"]


def model_label(model_id: Optional[str]) -> str:
    """Short display label for a model id (falls back to the id; ``""`` for
    ``None``/empty)."""
    if not model_id:
        return ""
    for c in MODEL_CHOICES + foundry_model_choices() + prompter_model_choices():
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


# A conversation's ROOT can differ from the process default. On Tiny Apps one
# process serves many people and two pages, so each person's conversations
# for each page live under their own root (``webapp.profiles.session_root``).
# Everything downstream — the detached turn worker included — addresses a
# conversation by ``thread_id`` alone, so the root is REGISTERED against the
# thread when the conversation is opened and looked up here. A thread nobody
# registered uses the default root, byte-identical to the single-user app.
_THREAD_ROOTS: Dict[str, str] = {}
_THREAD_ROOTS_LOCK = threading.Lock()


def register_thread_root(thread_id: str, root: Optional[str]) -> None:
    """Remember that ``thread_id`` lives under ``root`` (``None`` forgets)."""
    with _THREAD_ROOTS_LOCK:
        if root:
            _THREAD_ROOTS[thread_id] = os.path.abspath(root)
        else:
            _THREAD_ROOTS.pop(thread_id, None)


def thread_root(thread_id: str) -> Optional[str]:
    """The registered root for ``thread_id``, or ``None`` (the default)."""
    with _THREAD_ROOTS_LOCK:
        return _THREAD_ROOTS.get(thread_id)


def conversation_dir(thread_id: str, root: Optional[str] = None) -> str:
    return os.path.join(conversations_root(root or thread_root(thread_id)),
                        thread_id)


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


#: Files larger than this are not copied into the conversation folder.
IMPORT_MAX_BYTES = 200 * 1024 * 1024


def _same_bytes(a: str, b: str, chunk: int = 1 << 20) -> bool:
    """True when two files hold the same bytes.

    NOT ``filecmp.cmp``: that caches by (size, mtime), so a rebuilt file of
    the SAME byte size written in the same clock tick reads as unchanged — and
    the conversation would keep showing the previous version, which is the
    very failure this import exists to prevent (caught 2026-09-15 when the
    suite's ordering made the cache hit).
    """
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
        with open(a, "rb") as fa, open(b, "rb") as fb:
            while True:
                ba, bb = fa.read(chunk), fb.read(chunk)
                if ba != bb:
                    return False
                if not ba:
                    return True
    except OSError:
        return False


def _reported_path(path, base: str) -> str:
    """A path a tool reported, made absolute: a relative one is a name in
    the working folder ``base`` -- tool results name files that way since
    2026-10-09 (live smoke wave 1, A6) -- not a path under the process cwd."""
    p = os.path.expanduser(str(path).strip().strip("'\""))
    if not os.path.isabs(p):
        p = os.path.join(base, p)
    return os.path.abspath(p)


def import_reported_outputs(paths: Iterable[str], files_dir: str,
                            exclude: Iterable[str] = (),
                            working_dir: Optional[str] = None) -> dict:
    """Copy files a tool REPORTED writing outside the conversation folder into
    ``files_dir``, so they get a download card, show inline and reach the
    SharePoint mirror (field feedback 2026-09-15, N4: the calc package and its
    figures were written to /tmp and the owner never received them).

    A RELATIVE reported path is a name in the working folder (``working_dir``,
    default ``files_dir``): results give conversation-relative names.

    Files already inside the conversation directory, missing files, excluded
    paths (staged uploads, files fetched only to be read) and files over
    ``IMPORT_MAX_BYTES`` are skipped. A same-named file with identical content
    is reused; different content gets ``_1``/``_2``... Returns
    ``{source_abs_path: destination_path}``.
    """
    fd = os.path.abspath(files_dir)
    conv = os.path.dirname(fd)
    base = os.path.abspath(working_dir) if working_dir else fd
    skip = set()
    for p in exclude or ():
        try:
            skip.add(_reported_path(p, base))
        except (TypeError, ValueError):
            continue
    copied: dict = {}
    for p in dict.fromkeys(paths or ()):
        try:
            src = _reported_path(p, base)
        except (TypeError, ValueError):
            continue
        if (src in skip or src in copied or not os.path.isfile(src)
                or src.startswith(conv + os.sep)):
            continue
        try:
            if os.path.getsize(src) > IMPORT_MAX_BYTES:
                continue
            os.makedirs(fd, exist_ok=True)
            dst = os.path.join(fd, os.path.basename(src))
            if not (os.path.isfile(dst) and _same_bytes(src, dst)):
                dst = _unique_dest(dst)
                _shutil.copy2(src, dst)
        except OSError:
            continue
        copied[src] = dst
    return copied


_LOCAL_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\((?!https?:|data:)([^)\s]+)[^)]*\)")


def displayable_markdown(text: str, artifact_paths: Iterable[str] = ()) -> str:
    """Replace markdown images that point at local files -- which the chat
    cannot display (field feedback N6: a broken-image icon where the PYWall
    plot should have been) -- with a pointer to the card shown under the
    reply, or a plain note when there is no such card.

    A figure whose PNG card was replaced by its interactive Plotly chart
    (:func:`collect_turn_artifacts`) still counts as shown: the reader sees
    that figure below the reply, so the PNG link must not read "not viewable"."""
    names = {os.path.basename(str(p)) for p in artifact_paths or ()}
    for name in list(names):                   # the sidecar stands in for its
        low = name.lower()                     # image — same figure, one card
        if low.endswith(PLOTLY_SIDECAR_SUFFIX):
            stem = name[:-len(PLOTLY_SIDECAR_SUFFIX)]
            names.update(stem + ext for ext in _SUPERSEDED_IMAGE_EXTS)

    def _swap(m):
        alt = m.group(1).strip() or "figure"
        if os.path.basename(m.group(2)) in names:
            return f"*({alt} — shown below)*"
        return f"*({alt} — local file `{m.group(2)}`, not viewable in chat)*"

    return _LOCAL_MD_IMAGE.sub(_swap, text or "")


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
#   trace          -- per-conversation "turn details" tracer toggle: True/False
#                     explicit, None = follow the GEOTECH_TRACE env default.
# Defaults: references anytime, ref budget 8, recursion_limit 50 (raised from
# the LangGraph-default 25 in 5.10.2 — see the inline note), route_calc ON
# (owner-approved A2), analysis_depth "standard" == no preset, agent_type
# "full".

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
    "pavement": "Pavement design specialist",
    "structural": "Structural calc specialist",
}

DEFAULT_BEHAVIOR = {
    "references": "anytime",
    # 10 since 2026-10-09 (funhouse_agent.deep.limits: live smoke wave 1, G5)
    "ref_max_calls": 10,
    # 25 (the LangGraph default) proved too tight once SharePoint fetch +
    # multi-page PDF reads entered normal workflows (owner hit the cap on the
    # downdrag task 2026-08); 50 covers those while still bounding runaways.
    "recursion_limit": 50,
    "analysis_depth": "standard",
    "agent_type": "full",
    "route_calc": True,
    # Per-conversation tracer override: True/False = explicit sidebar choice,
    # None = follow the GEOTECH_TRACE env default (see tracing_enabled()).
    "trace": None,
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


def _busy_kind(exc: BaseException) -> Optional[str]:
    """``rate limit`` / ``overloaded`` / ``server error …`` / ``connection``
    / ``timeout`` for an error that asking again later would fix, else
    ``None``. Read off the error's type, HTTP status and body
    (``vision_engine.busy_kind``), with the error's name and text as a
    fallback for hosts whose errors carry neither."""
    name = type(exc).__name__.lower()
    if "timeout" in name or isinstance(exc, TimeoutError):
        return "timeout"
    kind = None
    try:
        from funhouse_agent.deep.vision_engine import busy_kind
        kind = busy_kind(exc)
    except Exception:  # noqa: BLE001 - advice must never mask the error
        kind = None
    low = str(exc).lower()
    if kind is None:
        if "ratelimit" in name or "rate_limit_exceeded" in low \
                or "too many requests" in low:
            kind = "rate limit"
        elif "overloaded" in name or "overloaded_error" in low:
            kind = "overloaded"
    return kind


def friendly_turn_error(exc: BaseException) -> str:
    """Turn-failure text for the transcript: the raw error plus, for known
    cases, plain-language advice (owner ask 2026-08: a raw GraphRecursionError
    traceback reads as a crash, when the fix is one sidebar setting).

    No server path ever reaches the user (live smoke wave 2b, C3: a MuPDF
    error showed ``C:\\…\\users\\livesmoke__tester\\…``):
    :func:`funhouse_agent.error_text.scrub_paths` leaves each path's file
    name. A busy model (rate limit, overload, a 5xx, a timeout) is said in
    plain words first, with the raw text after it, shortened."""
    text = f"{type(exc).__name__}: {exc}"
    try:
        from funhouse_agent.error_text import scrub_paths
        text = scrub_paths(text)
    except Exception:  # noqa: BLE001 - the error is shown either way
        pass
    low = text.lower()
    if "recursion" in low and "limit" in low:
        return (text + "\n\nThe agent ran out of its per-turn step budget "
                "before finishing. For document-heavy or multi-part requests, "
                "raise 'Primary step cap (recursion limit)' under Behavior > "
                "Advanced caps in the sidebar (e.g. 50-100) and re-ask — "
                "or split the request across turns; work done so far "
                "(downloads, saved files) is kept.")
    if "budgetexceeded" in type(exc).__name__.lower():
        return (text + "\n\nYour monthly Funhouse AI budget is exhausted; "
                "it resets next month — contact the Funhouse admins to "
                "raise it.")
    kind = _busy_kind(exc)
    if kind is None:
        return text
    # A busy model (field session 2026-10-06; live smoke wave 2a, B7): said
    # in plain words FIRST, the raw text after it for whoever reports it.
    detail = " ".join(text.split())
    if len(detail) > 300:
        detail = detail[:300] + " …"
    kept = ("Nothing is lost: files downloaded or saved this turn are still "
            "in the conversation. Wait a minute, then ask the agent to "
            "continue.")
    if kind == "rate limit":
        lead = ("The AI model's rate limit was reached (too many requests "
                "or tokens in the last minute — several people may be "
                "sharing it) — a throttle, not a fault.")
    elif kind == "overloaded":
        lead = ("The AI model is overloaded right now (busy at the "
                "provider), so this turn stopped — a temporary condition, "
                "not a fault in your request.")
    elif kind == "timeout":
        lead = ("The AI model did not answer in time, so this turn stopped "
                "— usually a busy service, not a fault in your request.")
    elif kind == "connection":
        lead = ("A network connection to the AI model failed, so this turn "
                "stopped — usually temporary.")
    else:
        lead = (f"The AI service had a temporary problem ({kind}), so this "
                f"turn stopped — not a fault in your request.")
    return f"{lead} {kept}\n\n(Details: {detail})"


#: How long the readable part of a failed turn's streamed text must be to be
#: kept (labelled as cut short) rather than replaced by the error alone.
USEFUL_PARTIAL_CHARS = 200

#: The label put on a failed turn's partial answer.
CUT_SHORT_LABEL = ("*This answer was cut short: the turn stopped on an error "
                   "before it finished (see the message below). What follows "
                   "is what had been written by then.*")

#: What a failed turn says when nothing it wrote is worth keeping.
NOTHING_KEPT = ("This turn stopped on an error before it could answer (see "
                "the message below). Files saved or downloaded this turn are "
                "still in the conversation; ask again, or ask the agent to "
                "continue.")


def _is_narration(paragraph: str) -> bool:
    """A short paragraph that only announces a step ("I'll open the page
    map.", "Now let me zoom on the title block:")."""
    p = paragraph.strip()
    if not p:
        return True
    if len(p) > 240:
        return False
    first = _LIST_MARK_RE.sub("", p).strip()
    return bool(_LEAD_INTENT_RE.match(first)) or p.endswith(":")


def failed_turn_text(streamed: str = "", reply: Optional[str] = None) -> str:
    """What a FAILED turn shows as its answer (the error itself is shown
    under it).

    A reply the run had completed (an earlier pass's) is kept; otherwise the
    text streamed so far is kept only when, with the step announcements
    taken out, it still says something (:data:`USEFUL_PARTIAL_CHARS`) --
    raw narration ("I'll open the page map.") is not an answer. Either way
    it is labelled as cut short. Paths are scrubbed."""
    try:
        from funhouse_agent.error_text import scrub_paths
    except Exception:  # noqa: BLE001
        def scrub_paths(t):
            return t
    reply = str(reply or "").strip()
    if reply:
        return f"{CUT_SHORT_LABEL}\n\n{scrub_paths(reply)}"
    paras = [p.strip() for p in re.split(r"\n\s*\n", str(streamed or ""))
             if p.strip()]
    kept = [p for p in paras if not _is_narration(p)]
    body = "\n\n".join(kept)
    if len(body) >= USEFUL_PARTIAL_CHARS:
        return f"{CUT_SHORT_LABEL}\n\n{scrub_paths(body)}"
    return NOTHING_KEPT


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
# Run tracing (A7 rec 1): the local summary is ON by default (owner, 2026-10-06)
# ---------------------------------------------------------------------------
# Two independent paths:
#   * LangSmith (SaaS) — set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY; the
#     langchain/langgraph stack auto-traces every run, no code here.
#   * Local (no SaaS) — on unless GEOTECH_TRACE=0 (or "Show turn details" is
#     unticked for the conversation);
#     the app writes ONE compact JSONL SUMMARY line per turn (duration,
#     tokens, an 80-char one-liner per PRIMARY tool call, error) to
#     <conversation>/trace.jsonl and shows a "turn details" expander.
#     It does NOT see sub-agent internals.
# Independent of both, and ALWAYS on: <conversation>/activity.jsonl
# (webapp/activity_log.py) — every tool call with full args, every tool
# result (capped at 32 KB), every model call's usage, for the primary AND
# the calc/references sub-agents, attributed by `task` nesting. That file
# is the archive; trace.jsonl is the on-screen summary.

#: ``GEOTECH_TRACE`` values that turn the per-turn details OFF. Anything else,
#: including the variable being unset, leaves them ON.
_TRACE_OFF = ("0", "false", "no", "off")


def tracing_enabled(override: Optional[bool] = None) -> bool:
    """True when the local per-turn tracer is on.

    ``override`` is the per-conversation sidebar choice (``behavior["trace"]``):
    an explicit ``True``/``False`` wins; ``None`` falls back to the
    ``GEOTECH_TRACE`` env default, which is ON unless the variable says
    ``0``/``false``/``no``/``off`` (owner, 2026-10-06 field session: "Make
    showing turn details the default" -- the record is cheap and the person
    using the app is the one who reads it)."""
    if override is not None:
        return bool(override)
    raw = str(os.environ.get("GEOTECH_TRACE", "")).strip().lower()
    return raw not in _TRACE_OFF


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


#: Where a conversation's title came from (meta ``title_source``): the
#: attached files (an orientation turn), the first typed question, or the
#: user's own Rename. Only an attachments title is changed by a question,
#: and then only extended: the files stay in it (:func:`turn_title`).
TITLE_FROM_ATTACHMENTS = "attachments"
TITLE_FROM_QUESTION = "question"
TITLE_FROM_USER = "user"


def orientation_title(names: Iterable[str], max_names: int = 3) -> str:
    """A conversation title for an automatic orientation turn: the attached
    files' names (``21.01.pdf``, ``a.pdf + b.pdf``, ``a.pdf + b.pdf + 2
    more``), not the request the app sent for the user. Live smoke wave 1,
    A10: 18 of 28 conversations were titled "I just attached `x.pdf`.
    Before I ask anything,…", and so were their SharePoint folders."""
    shown = [str(n).strip() for n in (names or ()) if str(n or "").strip()]
    if not shown:
        return "New conversation"
    head = " + ".join(shown[:max_names])
    more = len(shown) - max_names
    return head + (f" + {more} more" if more > 0 else "")


#: Words of the first typed question added after a files title.
QUESTION_GIST_WORDS = 6

#: Between a files title and the question's gist.
TITLE_JOINER = " — "


def files_and_question_title(files_title, prompt,
                             n_words: int = QUESTION_GIST_WORDS) -> str:
    """A files title extended by the first typed question's gist:
    ``3000.pdf — What is this sheet?``.

    Live smoke wave 2a (B11): the question used to REPLACE the files title,
    so two conversations about two different sheets were both "What is this
    sheet?" in the sidebar and neither named its file. The files stay; the
    gist alone is used only when it already names the files title, and the
    files title alone when there is no question."""
    files_title = str(files_title or "").strip()
    gist = auto_title(prompt, n_words=n_words) if str(prompt or "").strip() \
        else ""
    if not files_title or files_title.lower() == "new conversation":
        return gist or "New conversation"
    if not gist:
        return files_title
    if files_title.lower() in gist.lower():
        return gist
    return f"{files_title}{TITLE_JOINER}{gist}"


_MORE_RE = re.compile(r"^(\d+) more$")
_FILE_PIECE_RE = re.compile(r"^[^\s/\\][^/\\]*\.[A-Za-z0-9]{1,6}$")


def _title_files(head: str) -> Optional[Tuple[List[str], int]]:
    """``(names, hidden)`` when ``head`` is a files title
    (:func:`orientation_title`'s ``a.pdf + b.pdf + 2 more``), else ``None``."""
    names, hidden = [], 0
    for piece in [p.strip() for p in str(head or "").split(" + ")]:
        more = _MORE_RE.match(piece)
        if more:
            hidden += int(more.group(1))
        elif _FILE_PIECE_RE.match(piece):
            names.append(piece)
        else:
            return None
    return (names, hidden) if names else None


def title_with_new_files(meta: Optional[dict], names: Iterable[str],
                         max_names: int = 3) -> Optional[str]:
    """The conversation title once files are attached to a conversation that
    already has one, or ``None`` to keep it.

    Live smoke wave 2b (C14): the 21.01 review stayed titled after
    "scan_0001.pdf", the blank wrong file, and a report added later never
    reached its conversation's name. A later upload's names join the files
    part of the title (``a.pdf + b.pdf — gist``), or go in front of a
    question title; names already in it, a title the user typed, and a
    conversation not yet titled (its first turn titles it) are left alone.
    The SharePoint folder does not follow (``meta.mirror_folder``)."""
    meta = meta or {}
    title = str(meta.get("title") or "").strip()
    if (meta.get("title_source") == TITLE_FROM_USER or not title
            or title.lower() == "new conversation"):
        return None
    low = title.lower()
    new = []
    for n in names or ():
        n = str(n or "").strip()
        if n and n.lower() not in low and n not in new:
            new.append(n)
    if not new:
        return None
    head, sep, tail = title.partition(TITLE_JOINER)
    parsed = _title_files(head)
    if parsed is not None:
        known, hidden = parsed
        allnames = known + new
        shown = allnames[:max_names]
        more = hidden + len(allnames) - len(shown)
        head = " + ".join(shown) + (f" + {more} more" if more > 0 else "")
        return head + (sep + tail if sep else "")
    return orientation_title(new, max_names=max_names) + TITLE_JOINER + title


def retitle_for_upload(thread_id: str, names: Iterable[str],
                       root: Optional[str] = None) -> Optional[str]:
    """Give a conversation that already has a title the names of files
    attached to it later (:func:`title_with_new_files`). Returns the new
    title, or ``None`` when it is kept. Never raises."""
    try:
        meta = load_meta(thread_id, root)
        if not meta:
            return None
        new = title_with_new_files(meta, names)
        if new:
            touch_conversation(thread_id, title=new, root=root)
        return new
    except Exception:  # noqa: BLE001 - a title is never worth a failure
        return None


def turn_title(meta: Optional[dict], prompt, user_turns: int,
               orientation: Optional[Iterable[str]] = None
               ) -> Tuple[Optional[str], Optional[str]]:
    """``(title, title_source)`` a finished turn gives the conversation, or
    ``(None, None)`` to keep the title it has.

    An orientation turn (``orientation`` = the attached names) titles a
    conversation that has no typed question yet after its files; the first
    TYPED question then adds its gist to that title, keeping the files in it
    (:func:`files_and_question_title`); a conversation whose first turn is
    typed is titled by the question as it always has been. A title the user
    typed (Rename) is never replaced. The SharePoint folder never follows a
    retitle: it is fixed at the first mirror (``meta.mirror_folder``)."""
    meta = meta or {}
    source = meta.get("title_source")
    if source == TITLE_FROM_USER:
        return None, None
    if orientation is not None:
        # Only a conversation's FIRST turn: a later upload's orientation
        # keeps the title the conversation already has.
        if source is None and user_turns <= 1:
            return orientation_title(orientation), TITLE_FROM_ATTACHMENTS
        return None, None
    if not prompt:
        return None, None
    if source == TITLE_FROM_ATTACHMENTS:
        return (files_and_question_title(meta.get("title"), prompt),
                TITLE_FROM_QUESTION)
    if user_turns == 1:
        return auto_title(prompt), TITLE_FROM_QUESTION
    return None, None


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
                       root: Optional[str] = None,
                       title_source: Optional[str] = None) -> dict:
    """Update ``updated`` (and optionally ``title`` / ``turn_count`` / ``model``)
    on a conversation's meta; creates it if missing. ``title_source`` records
    where a new title came from (:func:`turn_title`)."""
    meta = ensure_conversation(thread_id, title=title, root=root)
    if title is not None:
        meta["title"] = title
        if title_source is not None:
            meta["title_source"] = title_source
    if turn_count is not None:
        meta["turn_count"] = turn_count
    if model is not None:
        meta["model"] = model
    meta["updated"] = _time.time()
    save_meta(thread_id, meta, root)
    return meta


def tag_conversation(thread_id: str, root: Optional[str] = None,
                     **fields) -> dict:
    """Set descriptive fields on a conversation's meta that are not already
    set — ``owner`` (the signed-in person) and ``page`` (the app profile) on
    a multi-user host — creating the meta if needed. Existing values win, so
    a conversation keeps the owner it was created under."""
    meta = ensure_conversation(thread_id, root=root)
    changed = False
    for key, value in fields.items():
        if value and not meta.get(key):
            meta[key] = value
            changed = True
    if changed:
        save_meta(thread_id, meta, root)
    return meta


def rename_conversation(thread_id: str, title: str,
                        root: Optional[str] = None) -> dict:
    """Set a conversation's title (the user's own: never replaced by a
    question's, see :func:`turn_title`)."""
    return touch_conversation(thread_id, title=str(title), root=root,
                              title_source=TITLE_FROM_USER)


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


def partial_is_stale(data: Optional[dict], transcript: Iterable[dict]) -> bool:
    """Whether a turn checkpoint belongs to a turn the transcript shows as
    FINISHED, so it records no interruption.

    A turn writes its question to the transcript, then the checkpoint, and
    its answer last; an interrupted turn therefore always leaves its question
    as the transcript's LAST entry. So a checkpoint is stale when the last
    entry is an answer and the checkpoint's question has one (or it names no
    question at all). Such checkpoints came from a mirror taken mid-turn and
    restored later (live smoke wave 2a, B2), or a turn whose answer was saved
    but whose checkpoint could not be removed."""
    entries = [e for e in (transcript or ()) if isinstance(e, dict)]
    if not entries or entries[-1].get("role") != "assistant":
        return False
    prompt = str((data or {}).get("prompt") or "").strip()
    if not prompt:
        return True
    answered = set()
    question = None
    for e in entries:
        if e.get("role") == "user":
            question = str(e.get("text") or "").strip()
        elif e.get("role") == "assistant" and question is not None:
            answered.add(question)
            question = None
    return prompt in answered


def recover_partial(thread_id: str, root: Optional[str] = None) -> Optional[dict]:
    """If a turn was interrupted (``partial.json`` present), return a recovered
    display entry to append to the transcript, else ``None``. A checkpoint
    whose turn the transcript shows as answered is stale and ignored
    (:func:`partial_is_stale`); the rare append-succeeded-but-clear-failed
    case is also caught by comparing against the last assistant entry
    already on disk. Clears the partial file either way."""
    p = partial_path(thread_id, root)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            data = _json.load(fh) or {}
    except (OSError, ValueError):
        clear_partial(thread_id, root)
        return None
    transcript = load_transcript(thread_id, root)
    if not isinstance(data, dict) or partial_is_stale(data, transcript):
        clear_partial(thread_id, root)
        return None
    text = (data.get("text") or "").strip()
    if text:
        for e in reversed(transcript):
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
    trash = os.path.join(root or thread_root(thread_id) or data_root(), ".trash")
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


# ---------------------------------------------------------------------------
# What the conversation already has on disk, told to the agent every turn
# ---------------------------------------------------------------------------
# Field session 2026-10-06 (geotech page): the agent downloaded a report from
# SharePoint in turn 4, worked from it in turns 4 and 5, and in turn 7 told the
# user it had "only ever had the original attachment" and that its earlier
# work was "not verified". Nothing was wrong with the file -- it was still in
# the working folder. The app replays earlier turns to the model as the
# user's messages and the model's FINAL ANSWERS only (see the Persistence
# section): every tool result, the download included, is gone from the next
# turn's view, so the model had no record that the file existed. A turn-6
# SharePoint failure then read, to the model, as proof it never had the file.
#
# Two pieces of record fix that, without replaying tool results:
#   * a downloads ledger (``downloads.json`` in the conversation folder), which
#     the SharePoint download tool writes: which SharePoint file became which
#     local file, so a repeat fetch reuses it under ONE name (the session saved
#     the same 37 MB report twice under two names) and the agent can be told
#     where each input came from;
#   * :func:`working_files_note`, a short "[System note]" put in front of the
#     CURRENT turn's user message only (never saved into the history): the
#     files already in the working folder -- attached, fetched, produced --
#     by their names in that folder (never server paths: live smoke wave 1,
#     A6, a model quoted the absolute path from this note to the user).

#: The ledger of SharePoint downloads, in the conversation record folder.
DOWNLOADS_LEDGER = "downloads.json"

#: How many files the per-turn note names (the newest ones).
WORKING_FILES_NOTE_MAX = 30


def load_downloads(conv_dir: str) -> List[dict]:
    """The conversation's download ledger, ``[]`` when there is none.

    Each entry: ``{"remote": <SharePoint path>, "local": <absolute path>,
    "bytes": int, "ts": float}`` -- one per SharePoint file, the latest copy.
    """
    try:
        with open(os.path.join(conv_dir, DOWNLOADS_LEDGER),
                  encoding="utf-8") as fh:
            data = _json.load(fh)
    except (OSError, ValueError):
        return []
    return [d for d in data if isinstance(d, dict) and d.get("local")] \
        if isinstance(data, list) else []


def record_download(conv_dir: str, remote: str, local: str,
                    size: int = 0) -> None:
    """Record that SharePoint file ``remote`` is the local file ``local``.

    One entry per remote (case-insensitive); a later download of the same
    remote replaces its entry. Best-effort: a ledger that cannot be written
    must never fail the download it describes."""
    if not conv_dir or not remote or not local:
        return
    try:
        entries = [d for d in load_downloads(conv_dir)
                   if str(d.get("remote", "")).lower() != str(remote).lower()]
        entries.append({"remote": str(remote),
                        "local": os.path.abspath(str(local)),
                        "bytes": int(size or 0), "ts": _time.time()})
        os.makedirs(conv_dir, exist_ok=True)
        tmp = os.path.join(conv_dir, DOWNLOADS_LEDGER + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            _json.dump(entries, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(conv_dir, DOWNLOADS_LEDGER))
    except (OSError, TypeError, ValueError):
        pass


def _attachment_keys(conv_dir: str) -> List[str]:
    try:
        with open(os.path.join(conv_dir, "attachments.json"),
                  encoding="utf-8") as fh:
            keys = _json.load(fh)
    except (OSError, ValueError):
        return []
    return [str(k) for k in keys or [] if k]


def _note_name(path: str, files_dir: str) -> str:
    """A file as the turn note names it: relative to the working folder
    (forward slashes) when inside it -- the name the tools resolve -- and the
    path as it is only for a file elsewhere (a custom working folder)."""
    ap = os.path.abspath(path)
    fd = os.path.abspath(files_dir)
    if ap.startswith(fd + os.sep):
        return os.path.relpath(ap, fd).replace(os.sep, "/")
    return ap


#: Characters of the "already looked at" line at most.
READS_NOTE_MAX_CHARS = 900


def _page_list(pages: List[int]) -> str:
    """``[1, 2, 3, 5]`` -> ``"1-3, 5"``."""
    pages = sorted(set(int(p) for p in pages))
    out, i = [], 0
    while i < len(pages):
        j = i
        while j + 1 < len(pages) and pages[j + 1] == pages[j] + 1:
            j += 1
        out.append(str(pages[i]) if i == j else f"{pages[i]}-{pages[j]}")
        i = j + 1
    return ", ".join(out)


def reads_note(folder: Optional[str]) -> str:
    """One line naming the pages this conversation has LOOKED AT with the
    page and zoom tools, per document, in a viewer's page numbers; ``""``
    when none are recorded.

    Live smoke wave 2b (C7): asked "what pages did you look at?", a model
    that had viewed all ten sheets said its claim was "stronger than my
    record supports" and re-read all ten ($1.86, 152 s); 67 of 162 page reads
    re-read a page an earlier turn had read. The vision tools keep the record
    (``funhouse_agent.vision_tools.reads_for_conversation``); this puts it in
    front of the turn."""
    try:
        from funhouse_agent.vision_tools import reads_for_conversation
        reads = reads_for_conversation(folder) if folder else []
    except Exception:  # noqa: BLE001 - a missing record is no record
        return ""
    if not reads:
        return ""
    docs: Dict[str, Dict[str, set]] = {}
    order: List[str] = []
    for r in reads:
        name = str(r.get("document") or "").strip()
        page = r.get("pdf_page")
        if not name or not isinstance(page, int):
            continue
        if name not in docs:
            docs[name] = {"whole": set(), "tiled": set(), "zoom": set()}
            order.append(name)
        view = r.get("view")
        if isinstance(view, str) and "tiles" in view:
            docs[name]["tiled"].add(page)
            docs[name]["whole"].add(page)
        elif view == "page" or view is None:
            docs[name]["whole"].add(page)
        else:
            docs[name]["zoom"].add(page)
    parts = []
    for name in order:
        d = docs[name]
        bits = []
        if d["whole"]:
            bits.append(f"whole page p. {_page_list(sorted(d['whole']))}")
        if d["tiled"]:
            bits.append(f"also in tiles p. {_page_list(sorted(d['tiled']))}")
        if d["zoom"]:
            bits.append(f"zoomed on p. {_page_list(sorted(d['zoom']))}")
        if bits:
            parts.append(f"'{name}': " + "; ".join(bits))
    if not parts:
        return ""
    line = ("Already LOOKED AT in this conversation (page and zoom tools, "
            "PDF pages as a viewer counts them): " + " | ".join(parts) + ".")
    if len(line) > READS_NOTE_MAX_CHARS:
        line = line[:READS_NOTE_MAX_CHARS - 40].rsplit(" | ", 1)[0] + \
            " | ... (more not listed)."
    return line + (" Use this when asked what you looked at; look again only "
                   "when the question needs something those reads did not "
                   "cover. Text you read (read_document, search) is not in "
                   "this list.")


def working_files_note(files_dir: str, transcript: Iterable[dict] = (),
                       exclude_text: str = "",
                       limit: int = WORKING_FILES_NOTE_MAX,
                       reads_folder: Optional[str] = None) -> str:
    """The "[System note]" naming the files this conversation already holds,
    for the front of the current turn's message; ``""`` when there are none.

    ``reads_folder`` is the working folder the turn is bound to (the key of
    the vision tools' record of pages looked at, :func:`reads_note`);
    ``None`` uses ``files_dir``.

    Sources, all already on disk: ``attachments.json`` (the user's uploads),
    the downloads ledger (SharePoint files fetched to read, with where they
    came from), and the transcript (each turn's produced files, and the files
    a turn fetched -- which also covers conversations from before the ledger).
    Only files that still exist are named. A file whose path or name appears
    in ``exclude_text`` -- this turn's own attachment note -- is left out, so
    a fresh upload is not announced twice."""
    fd = os.path.abspath(files_dir)
    conv = os.path.dirname(fd)
    found: Dict[str, dict] = {}
    order: List[str] = []

    def add(path: str, **info) -> None:
        try:
            ap = os.path.abspath(str(path))
        except (TypeError, ValueError):
            return
        if ap not in found:
            found[ap] = {}
            order.append(ap)
        for k, v in info.items():
            if v not in (None, "") and not found[ap].get(k):
                found[ap][k] = v

    for key in _attachment_keys(conv):
        add(os.path.join(fd, key), origin="attached by the user")
    turn = 0
    for entry in transcript or ():
        role = entry.get("role")
        if role == "user":
            turn += 1
        elif role == "assistant":
            for name in entry.get("inputs") or ():
                add(os.path.join(fd, str(name)), origin="fetched to read",
                    turn=turn)
            for ref in entry.get("artifacts") or ():
                path = _resolve_artifact(str(ref), fd)
                # A chart's card is its interactive sidecar; its picture is
                # named too, beside it, for a document that needs the image.
                picture = plotly_picture_twin(path)
                if picture:
                    add(picture, origin="you produced it", turn=turn,
                        note="the picture of the chart "
                             f"'{_note_name(path, fd)}'; use it in documents")
                    add(path, origin="you produced it", turn=turn,
                        note="the interactive chart shown to the user; "
                             f"its picture is '{_note_name(picture, fd)}'")
                else:
                    add(path, origin="you produced it", turn=turn)
    for d in load_downloads(conv):
        add(d["local"], origin="fetched to read", remote=d.get("remote"))

    rows = []
    excl = str(exclude_text or "")
    for ap in order:
        if not os.path.isfile(ap) or is_cache_path(ap, fd):
            continue
        if excl and (ap in excl or f"'{os.path.basename(ap)}'" in excl):
            continue
        info = found[ap]
        where = ""
        if info.get("remote"):
            where = f" from SharePoint '{info['remote']}'"
        when = f" in turn {info['turn']}" if info.get("turn") else ""
        what = f" -- {info['note']}" if info.get("note") else ""
        try:
            size = f"{os.path.getsize(ap):,} bytes"
        except OSError:
            size = "size unknown"
        rows.append(f"- '{_note_name(ap, fd)}' ({size}): "
                    f"{info.get('origin', 'in the folder')}{where}{when}"
                    f"{what}")
    looked = reads_note(reads_folder or files_dir)
    if not rows and not looked:
        return ""
    skipped = max(0, len(rows) - int(limit))
    rows = rows[-int(limit):] if skipped else rows
    head = ("[System note] Files already in this conversation's working "
            "folder, from earlier turns. You see earlier turns only through "
            "the answers you gave, not through their tool results, so this "
            "list is the record of what you already have. Use these files "
            "by these names -- every file tool looks a name up in the "
            "working folder -- and do not fetch them again; do not tell "
            "the user a file is unavailable, or that you never had it, while "
            "it is listed here.")
    head += (" The pages you looked at are listed after the files; which "
             "pages you read as text is only in your earlier answers."
             if looked else
             " Which pages of a file you actually read is only in your "
             "earlier answers.")
    if skipped:
        head += f" (The {skipped} oldest are not listed; list_files shows all.)"
    return "\n".join([head] + rows + ([looked] if looked else []))


def with_turn_note(messages: list, note: Optional[str]) -> list:
    """A copy of ``messages`` with ``note`` put in front of the LAST message
    when that message is the user's (the turn being answered). The caller's
    list is not touched, so the note never enters the saved history -- each
    turn gets a fresh one."""
    out = list(messages or [])
    if not note or not out:
        return out
    last = out[-1]
    if isinstance(last, dict):
        if last.get("role") in ("user", "human") and \
                isinstance(last.get("content"), str):
            out[-1] = {**last, "content": f"{note}\n\n{last['content']}"}
        return out
    if getattr(last, "type", "") == "human" and \
            isinstance(getattr(last, "content", None), str):
        out[-1] = {"role": "user", "content": f"{note}\n\n{last.content}"}
    return out

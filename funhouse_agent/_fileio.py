"""Verified file-write helpers shared by tool layers that save real files.

Why this exists: on Databricks, plain Python writes to ``/Workspace/...``
paths go through the workspace FUSE mount, which on several compute/access
modes does NOT durably store the content — the call succeeds, but the
workspace keeps a literal ``PLACEHOLDER`` file (text) or a corrupt payload
(binary, e.g. PDFs). Confirmed live 2026-06-12: a 241 kB calc-package HTML
written to /Workspace read back as the 11-byte string "PLACEHOLDER".
Additionally, on DBR 14+ a notebook's working directory IS its /Workspace
folder, so even bare default filenames land on the unreliable mount.

Tool responses must therefore (a) default outputs away from /Workspace,
(b) verify what actually landed on disk, and (c) rescue the content when
the target did not store it -- into the conversation's working folder when
a host bound one, else the local temp dir (:func:`rescue_write`).
"""

import contextlib
import contextvars
import os
import tempfile


#: The working folder bound to the turn running in THIS context. On a host
#: where several people share one process (Tiny Apps), a process-wide env var
#: is repointed by any other session's rerun while a turn is writing, so a
#: file could land in another person's folder. The web app's turn worker binds
#: its conversation's folder here (:func:`bind_working_dir`); LangGraph copies
#: the context into the threads it runs tools in. It wins over the env var.
_WORKING_DIR: "contextvars.ContextVar" = contextvars.ContextVar(
    "geotech_working_dir", default=None)


def bind_working_dir(path):
    """Bind ``path`` as this context's working folder; returns the token for
    :func:`unbind_working_dir`. A falsy ``path`` binds nothing (the env var,
    then the old fallbacks, apply)."""
    return _WORKING_DIR.set(os.path.abspath(os.path.expanduser(str(path)))
                            if path else None)


def unbind_working_dir(token) -> None:
    try:
        _WORKING_DIR.reset(token)
    except (ValueError, RuntimeError):     # set in another context
        pass


@contextlib.contextmanager
def working_dir_bound(path):
    """``with working_dir_bound(folder):`` -- the folder for this context."""
    token = bind_working_dir(path)
    try:
        yield
    finally:
        unbind_working_dir(token)


#: Set on a host where several people share one process (Tiny Apps): with
#: no folder bound to the running context, refuse rather than fall back to
#: the process-wide env var, which belongs to whoever reran last.
_REQUIRE_BINDING = False


class UnboundWorkingDir(RuntimeError):
    """No working folder is bound to this turn on a shared host."""


def require_turn_binding(on: bool = True) -> None:
    """Fail closed from now on (process-wide): an unbound context raises
    :class:`UnboundWorkingDir` instead of using the env var."""
    global _REQUIRE_BINDING
    _REQUIRE_BINDING = bool(on)


def turn_binding_required() -> bool:
    return _REQUIRE_BINDING


def _bound_or_env():
    bound = _WORKING_DIR.get()
    if bound:
        return bound
    if _REQUIRE_BINDING:
        raise UnboundWorkingDir(
            "no working folder is bound to this turn on a shared host; "
            "refusing to guess whose folder to use")
    env = os.environ.get(DEFAULT_OUTPUT_DIR_ENV)
    if env and env.strip():
        return os.path.abspath(os.path.expanduser(env.strip()))
    return None


def is_databricks() -> bool:
    """True when running on a Databricks cluster/serverless runtime."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def _is_workspace_path(path: str) -> bool:
    return str(path).replace("\\", "/").startswith("/Workspace")


#: Env var naming the directory auto-generated tool outputs default INTO — set
#: by a host (e.g. the web app, per conversation) so calc packages / plots land
#: in a chosen working folder instead of the system temp dir. An explicit
#: ``output_path`` passed to a tool still takes precedence over this default.
DEFAULT_OUTPUT_DIR_ENV = "GEOTECH_DEFAULT_OUTPUT_DIR"


def default_output_dir() -> str:
    """Directory for auto-generated output files.

    Resolution (first that applies):

    1. the working folder bound to this turn (:func:`bind_working_dir`), else
       ``$GEOTECH_DEFAULT_OUTPUT_DIR`` if set — the host-chosen working folder
       (``~`` expanded, made absolute). This is the highest-precedence DEFAULT;
       an explicit ``output_path`` passed to a tool still wins over it.
    2. the system temp dir on Databricks or whenever the current directory is a
       /Workspace FUSE path, where plain file writes are unreliable.
    3. the current directory ("") locally.
    """
    host = _bound_or_env()
    if host:
        return host
    if is_databricks() or _is_workspace_path(os.getcwd()):
        return tempfile.gettempdir()
    return ""


def host_output_dir():
    """The working folder a HOST set for this conversation -- bound to the
    turn (:func:`bind_working_dir`), else ``$GEOTECH_DEFAULT_OUTPUT_DIR`` (the
    web app per conversation, the review suite per run) -- absolute, or
    ``None`` for a library caller that set none (then
    :func:`default_output_dir` falls back as it always has)."""
    return _bound_or_env()


def _inside(path: str, folder: str) -> bool:
    try:
        p = os.path.normcase(os.path.abspath(path))
        f = os.path.normcase(os.path.abspath(folder))
        return os.path.commonpath([p, f]) == f
    except ValueError:              # another drive (Windows)
        return False


def into_working_folder(path: str, folder=None, is_dir: bool = False) -> str:
    """Where a tool writes the file (or, ``is_dir``, the folder of files) a
    model named ``path``, once a host has set a working folder.

    Inside that folder (a bare name, a sub-path, or an absolute path already
    in it) the path is kept. Anywhere else — ``/tmp/x.xml``, another
    conversation's folder, a home directory — the FILE NAME is kept and the
    directory is replaced by the working folder; a folder outside it becomes
    the working folder itself. Foundry brief 5 (2026-10-08): every model
    copied a ``/tmp`` example into ``write_diggs``, the file was written there
    and the run never delivered it.

    With no host folder (``folder`` not given and :func:`host_output_dir`
    ``None``) ``path`` is returned unchanged: library callers keep exactly the
    behaviour they had.
    """
    folder = folder or host_output_dir()
    if not folder or not isinstance(path, str) or not path.strip():
        return path
    folder = os.path.abspath(folder)
    # join() keeps an absolute (or, on Windows, a rooted "/tmp/...") path
    # whole and puts a relative one inside the folder.
    candidate = os.path.abspath(os.path.join(
        folder, os.path.expanduser(path.strip())))
    if _inside(candidate, folder):
        return candidate
    if is_dir:
        return folder
    name = os.path.basename(candidate.rstrip("/\\"))
    return os.path.join(folder, name) if name else folder


def find_in_working_folder(name: str):
    """Absolute path of a bare or relative file ``name`` that exists in
    :func:`default_output_dir`, else ``None``.

    A downloaded or saved file is announced by name ("the file is now in the
    working folder"), and agents then pass that name to the file tools; before
    2026-09-15 those tools accepted only a full path or an attachment key and
    answered "not found" (field feedback N9).
    """
    if not name:
        return None
    p = os.path.expanduser(str(name).strip())
    if not p or os.path.isabs(p):
        return None
    base = default_output_dir()
    if not base:
        return None
    candidate = os.path.join(base, p)
    return os.path.abspath(candidate) if os.path.isfile(candidate) else None


def resolve_output_path(output_path: str) -> str:
    """Resolve a tool's ``output_path`` against :func:`default_output_dir`.

    An absolute path (or a ``~`` path) is honored as given. A bare or relative
    path is placed INTO ``default_output_dir()`` — so when a host has set the
    working-folder env, a bare filename such as ``"cross_section.html"`` lands
    in that folder (e.g. the web app's per-conversation ``files/`` dir) instead
    of the process CWD. With no env set and off Databricks, ``default_output_dir``
    is ``""`` and the path is returned unchanged, i.e. byte-identical to the
    prior CWD-relative behavior.
    """
    p = os.path.expanduser(str(output_path))
    if os.path.isabs(p):
        return p
    base = default_output_dir()
    return os.path.join(base, p) if base else p


def written_file_problem(abs_path: str, expected: bytes = None):
    """Return a human-readable problem string if the file at ``abs_path``
    does not hold the expected content, else ``None``.

    ``expected`` is the intended content as bytes (text already encoded
    utf-8). The size check tolerates CRLF inflation from text-mode writes on
    Windows (newline translation only ever adds bytes), and the head
    comparison is newline-normalized.
    """
    if not os.path.isfile(abs_path):
        return f"no file exists at '{abs_path}' after writing"
    size = os.path.getsize(abs_path)
    if expected is None:
        return None if size > 0 else f"the file at '{abs_path}' is empty"
    if size < len(expected):
        return (
            f"the file on disk is {size} bytes but {len(expected)} bytes of "
            "content were written — the target filesystem did not store the "
            "content"
        )
    try:
        with open(abs_path, "rb") as f:
            head = f.read(257)
    except OSError as exc:
        return f"the file at '{abs_path}' could not be read back ({exc})"
    # Exact match first: BINARY content (PNG figures, PDFs) may legitimately
    # contain \r\n — the PNG signature does — and newline-normalizing it would
    # report a perfectly good file as corrupt.
    if expected.startswith(head[: min(len(head), 200)]):
        return None
    norm = head.replace(b"\r\n", b"\n")
    # A fixed-size read can split a CRLF pair; drop the orphaned \r rather
    # than fail the comparison on a correctly-written file.
    if norm.endswith(b"\r"):
        norm = norm[:-1]
    if not expected.startswith(norm[: min(len(norm), 200)]):
        return (
            "the file on disk does not start with the written content "
            f"(it begins with {head[:40]!r})"
        )
    return None


#: Longest full path a rescue copy is given. Windows refuses paths over 260
#: characters unless long paths are enabled; a rescue exists because a write
#: failed, so its own name must not fail the same way (live smoke wave 1,
#: A14: a 266-character figure path).
MAX_RESCUE_PATH = 200


def rescue_dir() -> str:
    """Where a rescue copy goes: the working folder a host bound (the
    conversation's folder in the web app, where the user receives files),
    else the system temp dir (a library caller)."""
    return host_output_dir() or tempfile.gettempdir()


def _short_stem(stem: str, room: int) -> str:
    """``stem`` cut to ``room`` characters, kept unique by a short hash."""
    if len(stem) <= room:
        return stem
    import hashlib
    tag = hashlib.sha1(stem.encode("utf-8", "replace")).hexdigest()[:6]
    return stem[:max(room - 7, 8)].rstrip("_- .") + "_" + tag


def rescue_write(filename: str, expected: bytes):
    """Write ``expected`` as a rescue copy in :func:`rescue_dir`.

    Since 2026-10-09 the copy goes into the conversation's working folder
    when a host bound one -- where the user gets a card for it and the
    mirror carries it -- instead of the system temp folder, where it reached
    nobody (live smoke wave 1, A14). Its name is shortened when the full path
    would exceed :data:`MAX_RESCUE_PATH`, and it never overwrites the path
    that failed.

    Returns the verified absolute rescue path, or ``None`` if even this
    write failed.
    """
    folder = os.path.abspath(rescue_dir())
    base = os.path.basename(str(filename)) or "rescued_file"
    stem, ext = os.path.splitext(base)
    stem = _short_stem(stem, MAX_RESCUE_PATH - len(folder) - len(ext) - 5)
    original = os.path.abspath(str(filename))
    n = 0
    while True:
        name = f"{stem}_{n}{ext}" if n else f"{stem}{ext}"
        rescue = os.path.join(folder, name)
        if rescue == original:
            n += 1
            continue
        if not os.path.exists(rescue):
            break
        try:
            with open(rescue, "rb") as f:
                if f.read() == expected:
                    return rescue  # identical rescue already present
        except OSError:
            pass
        n += 1
        if n > 100:
            return None
    try:
        os.makedirs(folder, exist_ok=True)
        with open(rescue, "wb") as f:
            f.write(expected)
    except OSError:
        return None
    return rescue if written_file_problem(rescue, expected) is None else None


def rescue_note(rescue: str) -> str:
    """The sentence a failed save's error carries about its rescue copy.

    It names the copy as the user will see it -- by its file name in the
    working folder -- and never asks the model to hand a server path to the
    user (until 2026-10-09: "report THAT path to the user", with the temp
    path)."""
    shown = conversation_name(rescue)
    if shown != rescue:
        return (f" A verified copy was saved in the working folder as "
                f"'{shown}' -- that copy is the file; call it by that name "
                "(it is attached to the conversation).")
    return f" A verified copy was saved at '{rescue}' -- that copy is the file."


def conversation_name(path):
    """How a file is named to the model: relative to the working folder a
    host bound, when the file is inside it (forward slashes; ``.`` for the
    folder itself); anything else unchanged. The file tools resolve such a
    name in the working folder (:func:`find_in_working_folder`), so absolute
    server paths stay internal (live smoke wave 1, A6)."""
    folder = host_output_dir()
    if not folder or not isinstance(path, str) or not path.strip():
        return path
    try:
        ap = os.path.abspath(os.path.expanduser(path.strip()))
    except (TypeError, ValueError):
        return path
    if not os.path.isabs(os.path.expanduser(path.strip())) or \
            not _inside(ap, folder):
        return path
    rel = os.path.relpath(ap, os.path.abspath(folder))
    return "." if rel == "." else rel.replace(os.sep, "/")


def _folder_patterns(folder: str):
    """``(inside, bare)`` regexes: the folder followed by a separator (a
    path inside it), and the folder on its own."""
    import re
    folder = os.path.abspath(folder).rstrip("\\/")
    spellings = {folder, folder.replace("\\", "/"), folder.replace("/", "\\")}
    alts = "|".join(re.escape(s) for s in sorted(spellings, key=len,
                                                 reverse=True))
    flags = re.IGNORECASE if os.name == "nt" else 0
    inside = re.compile(f"(?:{alts})[\\\\/]+", flags)
    bare = re.compile(f"(?:{alts})(?![\\w.\\-])", flags)
    return inside, bare


def hide_working_folder(obj):
    """``obj`` (a tool result: dicts, lists, strings) with every mention of
    the host's working folder taken out: a path inside it becomes its
    conversation-relative name (``figs/x.png``, which the tools resolve), a
    value that IS the folder becomes ``.``, and the folder named in prose
    becomes "the working folder". With no host folder ``obj`` is returned
    as it is. The input is never modified."""
    folder = host_output_dir()
    if not folder:
        return obj
    inside, bare = _folder_patterns(folder)
    import re
    # the rest of a path after the folder: up to a space, quote or bracket
    rest = re.compile(r"[^\s\"'<>|]*")

    def _relative(m) -> str:
        tail = rest.match(m.string, m.end()).group(0)
        return tail.replace("\\", "/")

    def _text(s: str) -> str:
        if not s:
            return s
        if bare.fullmatch(s.strip()):
            return "."
        out, pos = [], 0
        for m in inside.finditer(s):
            if m.start() < pos:
                continue
            out.append(s[pos:m.start()])
            tail = _relative(m)
            out.append(tail)
            pos = m.end() + len(tail)
        out.append(s[pos:])
        return bare.sub("the working folder", "".join(out))

    def _walk(o, depth=0):
        if depth > 40:
            return o
        if isinstance(o, str):
            return _text(o)
        if isinstance(o, dict):
            return {k: _walk(v, depth + 1) for k, v in o.items()}
        if isinstance(o, list):
            return [_walk(v, depth + 1) for v in o]
        if isinstance(o, tuple):
            return tuple(_walk(v, depth + 1) for v in o)
        return o

    try:
        return _walk(obj)
    except Exception:                                  # noqa: BLE001
        return obj


def workspace_write_hint(path: str) -> str:
    """Extra guidance appended to write-failure errors for /Workspace paths."""
    if _is_workspace_path(path):
        return (
            " Note: on Databricks, /Workspace paths written with plain file "
            "I/O are often not durably stored (the workspace keeps a literal "
            "PLACEHOLDER file, and binary files such as PDFs come out "
            "corrupt). Save to /tmp or a /Volumes path instead, then copy it "
            "out with dbutils.fs.cp('file:/tmp/<name>', ...) or download it."
        )
    return ""


# ---------------------------------------------------------------------------
# Databricks-aware durable /Workspace writes (optional databricks-sdk)
# ---------------------------------------------------------------------------
#
# Plain file I/O to /Workspace is unreliable (see module docstring). The
# authenticated Workspace API stores arbitrary file bytes durably. The SDK is
# an OPTIONAL dependency — it is preinstalled on Databricks runtimes but must
# never become a hard requirement, so every import and call is guarded and any
# failure degrades to a plain write + verify + rescue by the caller.


def _workspace_upload_bytes(client, path: str, content: bytes) -> None:
    """Upload raw bytes to a workspace path via the SDK's import/upload API.

    Isolated so the exact SDK surface lives in one place (and is trivial to
    mock in tests). The SDK's ``workspace.upload`` expects a binary stream, so
    the bytes are wrapped in ``BytesIO``. ``ImportFormat.AUTO`` imports
    arbitrary files as-is by extension; it is imported defensively so a missing
    enum does not block the upload.
    """
    import io

    fmt = None
    try:  # pragma: no cover - trivial import shim, exercised via mocks
        from databricks.sdk.service.workspace import ImportFormat
        fmt = ImportFormat.AUTO
    except Exception:
        fmt = None
    kwargs = {"overwrite": True}
    if fmt is not None:
        kwargs["format"] = fmt
    client.workspace.upload(path, io.BytesIO(content), **kwargs)


def _workspace_size(client, path: str):
    """Best-effort byte size of a workspace object; ``None`` if unavailable."""
    try:
        status = client.workspace.get_status(path)
    except Exception:
        return None
    return getattr(status, "size", None)


def workspace_api_upload(path: str, content) -> dict:
    """Durably write ``content`` to a ``/Workspace`` path via the Databricks SDK.

    Uses ``WorkspaceClient`` (default in-notebook auth). Returns a result dict
    and NEVER raises:

    * ``{"ok": True, "size": int, "verified": bool}`` on success — ``verified``
      is ``True`` when the stored size was read back and matches.
    * ``{"ok": False, "error": str}`` on any failure, including
      ``databricks-sdk`` not being importable.

    ``databricks-sdk`` stays an optional dependency: an import failure simply
    yields ``ok=False`` so the caller falls back to a plain filesystem write.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    try:
        from databricks.sdk import WorkspaceClient
    except Exception as e:  # ImportError or a partial/broken install
        return {"ok": False, "error": f"databricks-sdk not importable: {e}"}
    try:
        client = WorkspaceClient()
    except Exception as e:  # missing/failed auth off-cluster
        return {"ok": False, "error": f"WorkspaceClient auth failed: {e}"}

    norm = str(path).replace("\\", "/")
    parent = norm.rsplit("/", 1)[0]
    if parent and parent not in ("", "/", "/Workspace"):
        try:
            client.workspace.mkdirs(parent)
        except Exception:
            pass  # upload may still succeed; report the real error below

    try:
        _workspace_upload_bytes(client, norm, content)
    except Exception as e:
        return {"ok": False,
                "error": f"workspace upload failed: {type(e).__name__}: {e}"}

    size = _workspace_size(client, norm)
    if size is None:
        return {"ok": True, "size": len(content), "verified": False}
    if size < len(content):
        return {"ok": False,
                "error": (f"workspace API stored {size} bytes but "
                          f"{len(content)} were sent")}
    return {"ok": True, "size": size, "verified": True}


# ---------------------------------------------------------------------------
# High-level reusable verified save
# ---------------------------------------------------------------------------

def _local_write(path: str, content) -> str:
    """Write bytes/str to the local filesystem; returns the absolute path."""
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if isinstance(content, bytes):
        with open(abs_path, "wb") as f:
            f.write(content)
    else:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
    return abs_path


def save_verified(path: str, content) -> dict:
    """Write ``content`` to a REAL path, verify it landed, rescue on failure.

    The reusable verified-save entry point for tool/adapter code that produces
    a real output file (e.g. a self-contained Plotly HTML). Mirrors the
    ``save_file`` tool's default-writer behavior without the tool envelope:

    * a ``/Workspace`` target is routed through the durable Databricks
      workspace API first (falls back to a plain write when ``databricks-sdk``
      is unavailable);
    * the write is read back and compared (size + head) — a target that stored
      only a ``PLACEHOLDER`` is caught;
    * on a verify failure or a writer exception, a verified copy is staged to
      :func:`rescue_dir` (the working folder in the web app) and returned
      as ``rescue_path``.

    ``content`` may be ``str`` or ``bytes``. Returns a structured dict:
    ``{saved, file_exists, file_size_bytes, [save_method], [error],
    [rescue_path], [workspace_api_note]}``.
    """
    expected = (content if isinstance(content, bytes)
                else content.encode("utf-8", errors="replace"))

    api_error = None
    if _is_workspace_path(path):
        api = workspace_api_upload(path, expected)
        if api.get("ok"):
            return {
                "saved": path,
                "file_exists": True,
                "file_size_bytes": api.get("size", len(expected)),
                "save_method": "workspace_api",
            }
        api_error = api.get("error")

    try:
        saved_path = _local_write(path, content)
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {e}" + workspace_write_hint(path)}
        rescue = rescue_write(os.path.abspath(path), expected)
        if rescue:
            out["rescue_path"] = rescue
            out["error"] += rescue_note(rescue)
        if api_error:
            out["workspace_api_note"] = (
                f"The Databricks workspace API was tried first and failed "
                f"({api_error}); the plain write then also failed.")
        return out

    abs_path = os.path.abspath(saved_path)
    exists = os.path.isfile(abs_path)
    out = {
        "saved": abs_path if exists else saved_path,
        "file_size_bytes": os.path.getsize(abs_path) if exists else 0,
    }
    problem = written_file_problem(abs_path, expected)
    out["file_exists"] = exists and problem is None
    if problem:
        out["error"] = f"save ran but {problem}." + workspace_write_hint(abs_path)
        rescue = rescue_write(abs_path, expected)
        if rescue:
            out["rescue_path"] = rescue
            out["error"] += rescue_note(rescue)
    if api_error:
        out["workspace_api_note"] = (
            f"The Databricks workspace API was unavailable ({api_error}); used "
            "a plain filesystem write instead.")
    return out


__all__ = [
    "is_databricks", "default_output_dir", "resolve_output_path",
    "DEFAULT_OUTPUT_DIR_ENV", "host_output_dir", "into_working_folder",
    "bind_working_dir", "unbind_working_dir", "working_dir_bound",
    "written_file_problem", "rescue_write", "rescue_dir", "rescue_note",
    "MAX_RESCUE_PATH", "conversation_name", "hide_working_folder",
    "find_in_working_folder", "workspace_write_hint",
    "workspace_api_upload", "save_verified",
]

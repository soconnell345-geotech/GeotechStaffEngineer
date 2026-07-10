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
(b) verify what actually landed on disk, and (c) rescue the content to the
local temp dir when the target did not store it.
"""

import os
import tempfile


def is_databricks() -> bool:
    """True when running on a Databricks cluster/serverless runtime."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def _is_workspace_path(path: str) -> bool:
    return str(path).replace("\\", "/").startswith("/Workspace")


def default_output_dir() -> str:
    """Directory for auto-generated output files.

    Empty string (current directory) locally; the system temp dir on
    Databricks or whenever the current directory is a /Workspace FUSE path,
    where plain file writes are unreliable.
    """
    if is_databricks() or _is_workspace_path(os.getcwd()):
        return tempfile.gettempdir()
    return ""


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


def rescue_write(filename: str, expected: bytes):
    """Write ``expected`` to the system temp dir as a rescue copy.

    Returns the verified absolute rescue path, or ``None`` if even the temp
    dir write failed (or would overwrite the original path).
    """
    base = os.path.basename(filename)
    stem, ext = os.path.splitext(base)
    rescue = os.path.abspath(os.path.join(tempfile.gettempdir(), base))
    # Uniquify so two rescues with the same basename never clobber each other.
    n = 1
    while os.path.exists(rescue) and rescue != os.path.abspath(filename):
        try:
            with open(rescue, "rb") as f:
                if f.read() == expected:
                    return rescue  # identical rescue already present
        except OSError:
            pass
        rescue = os.path.abspath(
            os.path.join(tempfile.gettempdir(), f"{stem}_{n}{ext}"))
        n += 1
        if n > 100:
            return None
    if rescue == os.path.abspath(filename):
        return None
    try:
        with open(rescue, "wb") as f:
            f.write(expected)
    except OSError:
        return None
    return rescue if written_file_problem(rescue, expected) is None else None


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


__all__ = [
    "is_databricks", "default_output_dir", "written_file_problem",
    "rescue_write", "workspace_write_hint", "workspace_api_upload",
]

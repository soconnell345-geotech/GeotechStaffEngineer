"""SharePoint permanent storage for conversations (Funhouse SDK backed).

The Databricks driver's disk is ephemeral — conversations, uploads and calc
packages die with the cluster. This module mirrors each conversation's
directory (meta, transcript, trace, staged uploads, produced artifacts) to a
per-conversation SharePoint folder after every turn:

    <ROOT>/conversations/<thread_id>/...   (same layout as the local dir)

Design rules:

* **Best-effort, never raises into the app** — a SharePoint hiccup must not
  affect a turn. Errors are captured on the summary/status for the sidebar.
* **Incremental** — a local manifest (``sp_manifest.json`` in the conversation
  dir) records each mirrored file's (size, mtime); unchanged files are skipped.
* **Streamlit-free** — importable and testable without the app.

Configuration (env vars; the launcher subprocess inherits the notebook env):

    GEOTECH_SHAREPOINT_SITE_URL       https://<tenant>.sharepoint.com/sites/<site>   (required)
    GEOTECH_SHAREPOINT_TOKEN_FILE     **the delegated-OAuth path (State/Funhouse
                                      setup)**: a driver-local file holding the
                                      current Graph bearer token, staged AND
                                      kept fresh by the notebook (see
                                      databricks_launcher.stage_sharepoint)
    GEOTECH_SHAREPOINT_CLIENT_ID      \\ app-registration auth (office365 backend --
    GEOTECH_SHAREPOINT_CLIENT_SECRET  /  non-interactive; only if the tenant has one)
    GEOTECH_SHAREPOINT_TOKEN          a one-shot Graph bearer token (expires
                                      ~60-90 min; prefer TOKEN_FILE)
    GEOTECH_SHAREPOINT_DRIVE_NAME     optional (Graph): document-library name
    GEOTECH_SHAREPOINT_ROOT           base folder, default
                                      "Shared Documents/GeotechStaffEngineer"

Auth notes (from the Funhouse SDK source, researched 2026-07-30): the
``office365`` backend authenticates with plain ``client_id``/``client_secret``
strings (ACS app-only) — the SharePoint analog of the Prompter NTLM strings,
and safe to hand to the app subprocess. The SDK's default ``graph`` backend is
device-code interactive (browser) and is NOT used here; a pre-minted Graph
token is accepted as the alternative. No SharePoint code path touches
dbutils/Py4J, and ``FunhouseConfig`` degrades gracefully off-notebook.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from webapp import core

ENV_SITE = "GEOTECH_SHAREPOINT_SITE_URL"
ENV_CLIENT_ID = "GEOTECH_SHAREPOINT_CLIENT_ID"
ENV_CLIENT_SECRET = "GEOTECH_SHAREPOINT_CLIENT_SECRET"
ENV_TOKEN = "GEOTECH_SHAREPOINT_TOKEN"
ENV_TOKEN_FILE = "GEOTECH_SHAREPOINT_TOKEN_FILE"
ENV_DRIVE = "GEOTECH_SHAREPOINT_DRIVE_NAME"
ENV_ROOT = "GEOTECH_SHAREPOINT_ROOT"

DEFAULT_ROOT = "Shared Documents/GeotechStaffEngineer"

#: Local per-conversation mirror manifest (never itself uploaded).
MANIFEST_NAME = "sp_manifest.json"


def configured() -> bool:
    """True when the env carries enough to build a SharePoint client."""
    if not os.environ.get(ENV_SITE, "").strip():
        return False
    if (os.environ.get(ENV_CLIENT_ID, "").strip()
            and os.environ.get(ENV_CLIENT_SECRET, "").strip()):
        return True
    if os.environ.get(ENV_TOKEN_FILE, "").strip():
        return True
    return bool(os.environ.get(ENV_TOKEN, "").strip())


def _build_file_manager():
    """Construct the Funhouse SharePoint file manager from env strings.

    Prefers the non-interactive ``office365`` client-credential backend; falls
    back to a pre-minted Graph token. Raises on missing config/SDK.
    """
    site = os.environ.get(ENV_SITE, "").strip()
    if not site:
        raise RuntimeError(f"{ENV_SITE} is not set")
    cid = os.environ.get(ENV_CLIENT_ID, "").strip()
    secret = os.environ.get(ENV_CLIENT_SECRET, "").strip()
    if cid and secret:
        from funhouse.services.sharepoint import SharePointClient
        client = SharePointClient(site_url=site, client_id=cid,
                                  client_secret=secret, backend="office365")
        return client.file_manager

    drive_kwargs: Dict[str, Any] = {}
    drive = os.environ.get(ENV_DRIVE, "").strip()
    if drive:
        drive_kwargs["drive_name"] = drive

    token_file = os.environ.get(ENV_TOKEN_FILE, "").strip()
    if token_file:
        # Delegated-OAuth path: the notebook stages the current Graph token in
        # a driver-local file and keeps it FRESH with a refresher thread (see
        # databricks_launcher.stage_sharepoint). A provider that re-reads the
        # file per request means the client's 401-retry picks up refreshed
        # tokens automatically.
        from funhouse.services.sharepoint.graph.graph_client import (
            create_sharepoint_client_from_token_provider)

        def _read_token() -> str:
            with open(token_file, "r", encoding="utf-8") as fh:
                return fh.read().strip()

        client = create_sharepoint_client_from_token_provider(
            site, _read_token, **drive_kwargs)
        return client.file_manager

    token = os.environ.get(ENV_TOKEN, "").strip()
    if token:
        from funhouse.services.sharepoint.graph.graph_client import (
            create_sharepoint_client_from_token)
        client = create_sharepoint_client_from_token(site, token,
                                                     **drive_kwargs)
        return client.file_manager
    raise RuntimeError(
        f"SharePoint storage needs {ENV_CLIENT_ID}+{ENV_CLIENT_SECRET}, "
        f"{ENV_TOKEN_FILE}, or {ENV_TOKEN}")


def _manifest_path(conv_dir: str) -> str:
    return os.path.join(conv_dir, MANIFEST_NAME)


def _load_manifest(conv_dir: str) -> Dict[str, List]:
    try:
        with open(_manifest_path(conv_dir), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_manifest(conv_dir: str, manifest: Dict[str, List]) -> None:
    try:
        with open(_manifest_path(conv_dir), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
    except OSError:
        pass                                   # best-effort


def _stamp(path: str) -> List:
    st = os.stat(path)
    return [st.st_size, round(st.st_mtime, 3)]


class SharePointStore:
    """Mirrors conversation directories to SharePoint, incrementally.

    ``file_manager`` may be injected (tests / a caller with a live client);
    otherwise it is built lazily from the env on first use.
    """

    def __init__(self, file_manager: Any = None):
        self._fm = file_manager
        self._fm_error: Optional[str] = None
        self._lock = threading.Lock()
        self._made_dirs: set = set()
        self._folder_urls: Dict[str, str] = {}
        self.last_sync: Optional[dict] = None

    # -- configuration / client -------------------------------------------

    @property
    def configured(self) -> bool:
        return self._fm is not None or configured()

    def _file_manager(self):
        if self._fm is None:
            self._fm = _build_file_manager()
        return self._fm

    def root(self) -> str:
        return (os.environ.get(ENV_ROOT, "").strip() or DEFAULT_ROOT).strip("/")

    def session_folder(self, thread_id: str) -> str:
        """The remote folder for one conversation."""
        return f"{self.root()}/conversations/{thread_id}"

    # -- the mirror --------------------------------------------------------

    def mirror_conversation(self, thread_id: str,
                            root: Optional[str] = None) -> dict:
        """Upload this conversation's new/changed files. Never raises.

        Returns a summary dict: ``{"uploaded", "skipped", "errors",
        "web_url", "duration_s"}`` (also stored as ``self.last_sync``).
        """
        t0 = time.time()
        summary: dict = {"uploaded": 0, "skipped": 0, "errors": [],
                         "web_url": self._folder_urls.get(thread_id),
                         "duration_s": 0.0}
        with self._lock:
            try:
                self._mirror_locked(thread_id, root, summary)
            except Exception as exc:  # backstop — a sync must never crash a turn
                summary["errors"].append(f"{type(exc).__name__}: {exc}")
        summary["duration_s"] = round(time.time() - t0, 3)
        self.last_sync = summary
        return summary

    def _mirror_locked(self, thread_id: str, root: Optional[str],
                       summary: dict) -> None:
        try:
            fm = self._file_manager()
        except Exception as exc:
            summary["errors"].append(
                f"SharePoint client: {type(exc).__name__}: {exc}")
            return
        conv_dir = core.conversation_dir(thread_id, root)
        if not os.path.isdir(conv_dir):
            return
        manifest = _load_manifest(conv_dir)
        remote_base = self.session_folder(thread_id)

        for dirpath, _dirnames, filenames in os.walk(conv_dir):
            rel_dir = os.path.relpath(dirpath, conv_dir).replace(os.sep, "/")
            for name in sorted(filenames):
                if name == MANIFEST_NAME:
                    continue
                local = os.path.join(dirpath, name)
                rel = name if rel_dir == "." else f"{rel_dir}/{name}"
                try:
                    stamp = _stamp(local)
                except OSError:
                    continue                    # vanished mid-walk
                if manifest.get(rel) == stamp:
                    summary["skipped"] += 1
                    continue
                remote_dir = (remote_base if rel_dir == "."
                              else f"{remote_base}/{rel_dir}")
                try:
                    self._ensure_folder(fm, remote_dir)
                    fm.upload_file(local, f"{remote_dir}/{name}",
                                   overwrite=True)
                    manifest[rel] = stamp
                    summary["uploaded"] += 1
                except Exception as exc:
                    summary["errors"].append(
                        f"{rel}: {type(exc).__name__}: {exc}")

        _save_manifest(conv_dir, manifest)
        if thread_id not in self._folder_urls:
            try:
                self._folder_urls[thread_id] = fm.get_web_url(remote_base)
            except Exception:
                pass
        summary["web_url"] = self._folder_urls.get(thread_id)

    def _ensure_folder(self, fm, remote_dir: str) -> None:
        """Create ``remote_dir`` (and parents) once per process, idempotent."""
        if remote_dir in self._made_dirs:
            return
        parts = remote_dir.split("/")
        # Both backends create nested paths, but building up the chain keeps
        # office365 add_using_path happy on deep first-time trees.
        for i in range(2, len(parts) + 1):      # skip the library root itself
            partial = "/".join(parts[:i])
            if partial in self._made_dirs:
                continue
            fm.create_folder(partial)
            self._made_dirs.add(partial)


_STORE: Optional[SharePointStore] = None


def get_store(refresh: bool = False) -> SharePointStore:
    """The process-wide store (lazy). ``refresh=True`` rebuilds it (tests /
    changed env)."""
    global _STORE
    if refresh or _STORE is None:
        _STORE = SharePointStore()
    return _STORE


__all__ = ["SharePointStore", "get_store", "configured", "DEFAULT_ROOT",
           "MANIFEST_NAME"]

"""SharePoint permanent storage for conversations (Funhouse SDK backed).

The Databricks driver's disk is ephemeral — conversations, uploads and calc
packages die with the cluster. This module mirrors each conversation's
directory (meta, transcript, trace, activity log, feedback, staged uploads,
produced artifacts) to a
per-conversation SharePoint folder after every turn:

    <ROOT>/conversations/<name>_<YYYY-MM-DD>/...  (same layout as the local dir)

``<name>`` is the conversation's sidebar title, sanitized for SharePoint, and
the date is when the conversation was created; an unnamed conversation falls
back to its thread id (owner request 2026-09-04 — the hex thread ids made the
library unbrowsable). See :func:`conversation_folder`.

The mirror also runs BACKWARDS (owner feedback 2026-09-11: "after I close the
browser and the notebook clears, old conversations are gone from the
toolbar"): :meth:`SharePointStore.list_remote_conversations` lists the
mirrored folders — the folder name IS the search key — and
:meth:`SharePointStore.restore_conversation` downloads one back into the
local conversations directory (record files AND ``files/``), writing the
manifest so the next mirror does not re-upload everything.

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

    GRAPH_TENANT_ID / GRAPH_CLIENT_ID / GRAPH_CLIENT_SECRET / SHAREPOINT_SITE_URL
        the Tiny Apps path (5.26): an Entra app registration granted
        Sites.Selected on the team's site, served by the SDK-free
        :mod:`webapp.graph_sharepoint` client. These are CfA's own setting
        names, read through :mod:`webapp.tinyapps_settings` (env var, else
        Key Vault on a deployment, else a local ``.env``), and they take
        precedence over every path below — a deployment that has them needs
        neither the Funhouse SDK nor GEOTECH_SHAREPOINT_SITE_URL.

    GEOTECH_FOUNDRY_CONVERSATION_DATASET
        Foundry (INF-38): on a Foundry deployment with this set (the
        dataset's workspace alias), the conversations go to that Foundry
        dataset of raw files instead, through
        :class:`webapp.foundry_dataset_store.DatasetFileManager`; the sidebar
        names it "Foundry storage". The module-level :func:`configured`, the
        gate of the agent's SharePoint tools, is then False: those tools never
        run on the dataset. A save there takes seconds, so the turn's save
        runs after the turn, in the background (:func:`save_after_change`).

A file manager may offer a batch hook, ``begin_batch()`` and
``end_batch(commit=True)``: the mirror then sends all of one sync's files
between the two (the dataset's one transaction per save), and records them
as mirrored only once ``end_batch`` succeeds; a restore runs between them
too, so the dataset is listed once. One that offers ``download_files``
gets a restore's files in one call (in parallel). SharePoint's file
managers have neither and are driven exactly as before.

Auth notes (from the Funhouse SDK source, researched 2026-07-30): the
``office365`` backend authenticates with plain ``client_id``/``client_secret``
strings (ACS app-only) — the SharePoint analog of the Prompter NTLM strings,
and safe to hand to the app subprocess. The SDK's default ``graph`` backend is
device-code interactive (browser) and is NOT used here; a pre-minted Graph
token is accepted as the alternative. No SharePoint code path touches
dbutils/Py4J, and ``FunhouseConfig`` degrades gracefully off-notebook.
"""

from __future__ import annotations

import atexit
import collections
import json
import os
import re
import tempfile
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from webapp import core

ENV_SITE = "GEOTECH_SHAREPOINT_SITE_URL"
ENV_CLIENT_ID = "GEOTECH_SHAREPOINT_CLIENT_ID"
ENV_CLIENT_SECRET = "GEOTECH_SHAREPOINT_CLIENT_SECRET"
ENV_TOKEN = "GEOTECH_SHAREPOINT_TOKEN"
ENV_TOKEN_FILE = "GEOTECH_SHAREPOINT_TOKEN_FILE"
ENV_DRIVE = "GEOTECH_SHAREPOINT_DRIVE_NAME"
ENV_ROOT = "GEOTECH_SHAREPOINT_ROOT"

DEFAULT_ROOT = "Shared Documents/GeotechStaffEngineer"

#: The page whose conversations keep the flat layout: the geotech page,
#: ``webapp.profiles.DEFAULT.name`` (spelled out here so this module stays
#: free of the app's page machinery).
DEFAULT_PAGE = "geotech"

#: Local per-conversation mirror manifest (never itself uploaded).
MANIFEST_NAME = "sp_manifest.json"

#: Dropped into a conversation's PREVIOUS remote folder after a rename.
MOVED_NAME = "MOVED.txt"

#: Characters SharePoint/OneDrive reject in a file or folder name, plus the two
#: that survive the API but break the resulting URL (``#`` and ``%``).
_FORBIDDEN_CHARS = '"*:<>?/\\|#%'

#: Characters a title carries that do not belong in a folder name: Markdown
#: code marks, commas and the ellipsis ``core.auto_title`` ends a cut title
#: with. Dropped (a comma becomes a word break).
_DROPPED_CHARS = ("`", ",", "…")

#: ``meta.json`` field holding the conversation's mirror folder NAME, fixed
#: at its first mirror under a real title and never changed after: a link
#: handed out to ``<folder>/files/<name>`` must not go stale when the
#: conversation is retitled (live smoke wave 1, A10 -- an orientation turn's
#: title is replaced by the first typed question). Renaming changes the
#: displayed title only.
MIRROR_FOLDER_KEY = "mirror_folder"

#: Titles that mean "this conversation has no name yet" — mirror under the
#: thread id instead of making a folder called "New conversation_2026-09-04".
_PLACEHOLDER_TITLES = {"", "new conversation", "untitled"}

#: Cap on the name half of the folder. SharePoint's own limit is far higher,
#: but the full path is capped at 400 chars and these folders nest.
MAX_NAME_CHARS = 64


def sanitize_folder_name(name, max_len: int = MAX_NAME_CHARS) -> str:
    """A SharePoint-safe folder segment for ``name``; ``""`` if none survives.

    Replaces the rejected characters (``" * : < > ? / \\ | # %``), control
    characters and whitespace runs with a single underscore — underscores
    rather than spaces so the folder's URL carries no ``%20`` — breaks up the
    reserved ``_vti_`` token and a leading ``~$``, caps the length, and strips
    the leading/trailing dots, spaces and underscores SharePoint also rejects.
    Non-ASCII letters are kept; SharePoint accepts them. Backticks, commas
    and the ellipsis "…" are dropped: a title is prose (Markdown code marks,
    a truncated question), and they made folders like
    ``I_just_attached_`21.01.pdf`._Before_I_ask_anything,…`` (live smoke
    wave 1, A10).
    """
    text = str(name or "")
    for ch in _DROPPED_CHARS:
        text = text.replace(ch, " " if ch == "," else "")
    cleaned = "".join(
        "_" if (ch in _FORBIDDEN_CHARS or ch.isspace() or ord(ch) < 32) else ch
        for ch in text
    )
    cleaned = re.sub(r"_+", "_", cleaned).replace("_vti_", "_vti-")
    if cleaned.startswith("~$"):
        cleaned = cleaned[2:]
    return cleaned[:max_len].strip(" ._")


def folder_date(meta: Optional[dict]) -> str:
    """The conversation's creation date as ``YYYY-MM-DD`` (today if unknown)."""
    stamp = (meta or {}).get("created") or (meta or {}).get("updated")
    try:
        stamp = float(stamp)
    except (TypeError, ValueError):
        stamp = time.time()
    return time.strftime("%Y-%m-%d", time.localtime(stamp))


def _base_folder(thread_id: str, meta: Optional[dict]) -> str:
    """``<sanitized title>_<YYYY-MM-DD>``, or the thread id when unnamed."""
    title = str((meta or {}).get("title") or "").strip()
    if title.lower() in _PLACEHOLDER_TITLES:
        return str(thread_id)
    safe = sanitize_folder_name(title)
    return f"{safe}_{folder_date(meta)}" if safe else str(thread_id)


def conversation_folder(thread_id: str, meta: Optional[dict] = None,
                        siblings: Iterable[dict] = ()) -> str:
    """The remote folder NAME for one conversation.

    ``<sanitized title>_<YYYY-MM-DD created>`` when the conversation carries a
    real name — the sidebar title, whether the user typed it via Rename or it
    was derived from their first question — and the bare thread id when it does
    not. The name is computed here once: the first mirror under a real title
    fixes it in ``meta.json`` (:data:`MIRROR_FOLDER_KEY`), and a later rename
    changes the displayed title only, so links into the folder keep working.
    (A changed root or owner still moves the mirror; see
    ``SharePointStore._mirror_locked``.)

    ``siblings`` is the other conversations' metas (``core.list_conversations``).
    A short thread-id shard is appended when one of them would claim the same
    folder, because two conversations mirroring into ONE folder would overwrite
    each other's ``meta.json`` / ``transcript.jsonl``. The earliest-created
    claimant keeps the bare name so an existing folder does not move just
    because a same-named conversation was started later.

    A conversation whose meta already carries its fixed folder gets that
    name back. ``siblings`` are THIS host's conversations only; the first
    mirror also checks the remote library (``SharePointStore._pin_folder``),
    because a wiped or redeployed host, or a second host, cannot see the
    folders its predecessors made (live smoke wave 2a, B12).
    """
    pinned = str((meta or {}).get(MIRROR_FOLDER_KEY) or "").strip()
    if pinned:
        return pinned                    # fixed at the first mirror
    base = _base_folder(thread_id, meta)
    if base == str(thread_id):
        return base                      # already unique
    # A folder another conversation has already fixed as its own is taken,
    # whatever that conversation is titled now (see MIRROR_FOLDER_KEY).
    taken = {str(m.get(MIRROR_FOLDER_KEY) or "") for m in (siblings or [])
             if str(m.get("thread_id")) != str(thread_id)}
    if base in taken:
        return f"{base}_{str(thread_id)[:6]}"
    rivals = [
        m for m in (siblings or [])
        if str(m.get("thread_id")) != str(thread_id)
        and _base_folder(str(m.get("thread_id")), m) == base
    ]
    if not rivals:
        return base
    mine = (float((meta or {}).get("created") or 0.0), str(thread_id))
    first = min([mine] + [(float(m.get("created") or 0.0),
                           str(m.get("thread_id"))) for m in rivals])
    if first == mine:
        return base
    return f"{base}_{str(thread_id)[:6]}"


def _graph_configured() -> bool:
    """The Tiny Apps app-registration path — SDK-free, checked first."""
    try:
        from webapp import graph_sharepoint
        return graph_sharepoint.configured()
    except Exception:
        return False


def _foundry_selected() -> bool:
    """Foundry with the conversation dataset set (INF-38)."""
    try:
        from webapp import foundry_dataset_store
        return foundry_dataset_store.selected()
    except Exception:                                  # noqa: BLE001
        return False


def configured() -> bool:
    """True when the env carries enough to build a SharePoint client.

    False whenever the store's file manager is the Foundry dataset
    (:func:`_foundry_selected`): this is the gate of the agent's SharePoint
    tools and the reference-PDF fetch, which must not run on it."""
    if _foundry_selected():
        return False
    if _graph_configured():
        return True
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

    The Tiny Apps app-registration settings (``GRAPH_*`` +
    ``SHAREPOINT_SITE_URL``) win over all of these: they build the SDK-free
    :class:`webapp.graph_sharepoint.GraphFileManager`.

    On Foundry with the conversation dataset set, the Foundry dataset's file
    manager is built instead (INF-38); it raises with the plain reason when
    something it needs is missing.
    """
    if _foundry_selected():
        from webapp import foundry_dataset_store
        reason = foundry_dataset_store.unconfigured_reason()
        if reason:
            raise RuntimeError(reason)
        return foundry_dataset_store.DatasetFileManager()
    if _graph_configured():
        from webapp.graph_sharepoint import GraphFileManager
        return GraphFileManager()
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


def _load_manifest(conv_dir: str) -> Dict[str, Any]:
    try:
        with open(_manifest_path(conv_dir), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _manifest_files(manifest: Dict[str, Any]) -> Dict[str, List]:
    """The ``rel path -> [size, mtime]`` map.

    Manifests written before folder naming were that map at the top level; the
    current form nests it under ``"files"`` alongside the remote folder. Old
    manifests are read in place rather than discarded, so upgrading the app
    does not trigger a full re-upload of every conversation.
    """
    files = manifest.get("files")
    if isinstance(files, dict):
        return dict(files)
    return {k: v for k, v in manifest.items() if isinstance(v, list)}


def _save_manifest(conv_dir: str, files: Dict[str, List],
                   folder: Optional[str] = None) -> None:
    try:
        with open(_manifest_path(conv_dir), "w", encoding="utf-8") as fh:
            json.dump({"folder": folder, "files": files}, fh)
    except OSError:
        pass                                   # best-effort


def _stamp(path: str) -> List:
    st = os.stat(path)
    return [st.st_size, round(st.st_mtime, 3)]


def fix_web_url(url) -> str:
    """Repair the Funhouse SDK's redaction-dodging web URLs.

    The SDK intentionally emits ``https:/host/...`` (ONE slash) so Databricks
    log redaction doesn't eat the link — but that is an invalid URL: browsers
    resolve it as a RELATIVE path against the current page, producing e.g.
    ``https://adb-dp-.../<tenant>.sharepoint.com/...`` (owner-reported dead
    sidebar link, 2026-09). Normalize back to ``https://``.
    """
    text = str(url or "")
    for scheme in ("https", "http"):
        broken = f"{scheme}:/"
        if text.startswith(broken) and not text.startswith(f"{scheme}://"):
            return f"{scheme}://" + text[len(broken):]
    return text


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
        #: The latest mirror of ANY conversation in this process -- kept for
        #: callers that read it, never shown to a user (D3: on a shared host
        #: it is whoever mirrored last). The sidebar reads
        #: :meth:`last_sync_for` instead.
        self.last_sync: Optional[dict] = None
        self._syncs: Dict[str, dict] = {}
        # Saves after a turn, in the background (save_soon): one worker,
        # in order.
        self._bg_cv = threading.Condition()
        self._bg_queue: collections.deque = collections.deque()
        self._bg_waiting: set = set()
        self._bg_saving: Optional[str] = None
        self._bg_worker: Optional[threading.Thread] = None

    def last_sync_for(self, thread_id: Optional[str]) -> Optional[dict]:
        """This conversation's own latest mirror summary in this process,
        or ``None`` -- never another conversation's (live smoke wave 2c,
        D3: a fresh session's sidebar showed the last tester's folder link)."""
        if not thread_id:
            return None
        return self._syncs.get(str(thread_id))

    # -- configuration / client -------------------------------------------

    @property
    def configured(self) -> bool:
        if self._fm is not None:
            return True
        if _foundry_selected():
            from webapp import foundry_dataset_store
            return foundry_dataset_store.configured()
        return configured()

    @property
    def backend(self) -> str:
        """``"foundry"`` when the conversations go to a Foundry dataset
        (INF-38), else ``"sharepoint"``."""
        if self._fm is not None:
            return ("foundry" if getattr(self._fm, "BACKEND", "") == "foundry"
                    else "sharepoint")
        return "foundry" if _foundry_selected() else "sharepoint"

    @property
    def label(self) -> str:
        """What the sidebar calls the permanent storage."""
        return "Foundry storage" if self.backend == "foundry" else "SharePoint"

    @property
    def background(self) -> bool:
        """True when the saves after a turn run in the background: on the
        Foundry dataset, where one takes 5-7 s and the chat must not wait
        for it. SharePoint saves as before, at the end of the turn."""
        return self.backend == "foundry"

    def not_kept_reason(self) -> Optional[str]:
        """On a Foundry deployment, why conversations are NOT kept across
        restarts, with the fix, in one line (the text of Connection
        diagnostics' line): no alias set, a resource ID in its place, the
        library missing, an alias the workspace does not define, or a
        dataset its latest read could not read. ``None`` when they are
        kept, as far as is known (see :meth:`check_state`), and on every
        other host. Never raises."""
        if self._fm is not None and self.backend != "foundry":
            return None
        try:
            from webapp import foundry_dataset_store
            fm = self.file_manager() if self.backend == "foundry" \
                and self.configured else None
            return foundry_dataset_store.not_kept_reason(fm)
        except Exception:                              # noqa: BLE001
            return None

    def check_state(self) -> Optional[str]:
        """``"checking"`` while the Foundry dataset's first read in this
        process runs in the background (started by :meth:`not_kept_reason`;
        INF-65), else ``None``. Never raises."""
        if self.backend != "foundry" or not self.configured:
            return None
        try:
            checking = getattr(self.file_manager(), "checking", None)
            return "checking" if callable(checking) and checking() else None
        except Exception:                              # noqa: BLE001
            return None

    def _file_manager(self):
        if self._fm is None:
            self._fm = _build_file_manager()
        return self._fm

    def file_manager(self):
        """The Funhouse SharePoint file manager (built lazily from env).
        Raises on missing config/SDK — callers wanting best-effort must wrap."""
        return self._file_manager()

    def root(self) -> str:
        """The base folder: ``GEOTECH_SHAREPOINT_ROOT``, else the backend's
        own default (the Foundry dataset's has no document library)."""
        default = DEFAULT_ROOT
        if self.backend == "foundry":
            from webapp import foundry_dataset_store
            default = foundry_dataset_store.DEFAULT_ROOT
        return (os.environ.get(ENV_ROOT, "").strip() or default).strip("/")

    def folder_name(self, thread_id: str, root: Optional[str] = None) -> str:
        """This conversation's folder NAME (see :func:`conversation_folder`).
        Falls back to the thread id if the local metadata can't be read."""
        root = root or core.thread_root(thread_id)
        try:
            meta = core.load_meta(thread_id, root)
            pinned = str((meta or {}).get(MIRROR_FOLDER_KEY) or "").strip()
            if pinned:
                return pinned            # fixed at the first mirror
            siblings = core.list_conversations(root)
        except Exception:
            return str(thread_id)
        return conversation_folder(thread_id, meta, siblings)

    def _pin_folder(self, thread_id: str, root: Optional[str],
                    manifest: dict, fm: Any = None) -> None:
        """Fix the conversation's folder name (``MIRROR_FOLDER_KEY``) the
        first time it is mirrored under a real title -- or, for a
        conversation mirrored before this field existed, the titled folder it
        already has, so it does not move. An untitled conversation is not
        fixed yet: it mirrors under its thread id until it has a title.
        A NEW name is also checked against the remote library
        (:meth:`_unclaimed_remotely`). Best-effort."""
        try:
            root = root or core.thread_root(thread_id)
            meta = core.load_meta(thread_id, root)
            if not meta or str(meta.get(MIRROR_FOLDER_KEY) or "").strip():
                return
            name = None
            previous = manifest.get("folder") if isinstance(manifest,
                                                            dict) else None
            if isinstance(previous, str) and previous.strip("/"):
                last = previous.rstrip("/").rsplit("/", 1)[-1]
                if last and last != str(thread_id):
                    name = last
            if name is None:
                name = conversation_folder(thread_id, meta,
                                           core.list_conversations(root))
                if name == str(thread_id):
                    return
                name = self._unclaimed_remotely(fm, thread_id, meta, name)
            # Read again: the remote check takes seconds, and a save in the
            # background may run while the next turn retitles the
            # conversation; only the folder field is set here.
            meta = core.load_meta(thread_id, root) or meta
            if str(meta.get(MIRROR_FOLDER_KEY) or "").strip():
                return
            meta[MIRROR_FOLDER_KEY] = name
            core.save_meta(thread_id, meta, root)
        except Exception:                              # noqa: BLE001
            pass

    def _unclaimed_remotely(self, fm: Any, thread_id: str, meta: dict,
                            name: str) -> str:
        """``name``, or ``name_<first 6 of the thread id>`` when a folder of
        that name already exists in the remote library and is not this
        conversation's.

        The local dedupe (:func:`conversation_folder`) sees only this host's
        conversations; after a wiped or redeployed host, or on a second host,
        two conversations on the same file the same day chose the same
        ``<file>_<date>`` folder and would overwrite each other's record
        (live smoke wave 2a, B12). Only a NEW name is checked: a folder
        already fixed in a conversation's meta is never changed. SharePoint
        names are case-insensitive, so the comparison is too. When the
        library cannot be read the shard is added -- a unique name is the
        safe choice; a file manager with no listing at all is not checked."""
        tid = str(thread_id)
        shard = f"{name}_{tid[:6]}"
        if fm is None or not callable(getattr(fm, "ls", None)) \
                or name.endswith(f"_{tid[:6]}"):
            return name
        base = self.conversations_base(meta.get("owner"), meta.get("page"))
        try:
            entries = fm.ls(base) or []
        except Exception:                              # noqa: BLE001
            return shard
        wanted = name.lower()
        same = [str(e.get("name") or "").strip() for e in entries
                if isinstance(e, dict) and _is_folder(e)
                and str(e.get("name") or "").strip().lower() == wanted]
        if not same:
            return name
        # The folder exists: keep the name only if it is this conversation's.
        try:
            with tempfile.TemporaryDirectory() as td:
                local = os.path.join(td, "meta.json")
                fm.download_file(f"{base}/{same[0]}/meta.json",
                                 local_path=local, return_bytes=False,
                                 overwrite=True)
                with open(local, "r", encoding="utf-8") as fh:
                    remote = json.load(fh)
        except Exception:                              # noqa: BLE001
            return shard
        owner_tid = str((remote or {}).get("thread_id") or "") \
            if isinstance(remote, dict) else ""
        return name if owner_tid == tid else shard

    def session_folder(self, thread_id: str, root: Optional[str] = None) -> str:
        """The remote folder path for one conversation.

        ``<root>/conversations/<name>`` — the layout every deployment has had.
        A conversation whose meta names an ``owner`` (the signed-in person on
        a multi-user host, set by the app) gets it as a segment in between,
        so one shared site folder does not mix people; one whose meta names
        a ``page`` other than the geotech page gets that too, so the review
        page's conversations do not mix into the geotech list:
        ``<root>/conversations/[<owner>/][<page>/]<name>``.
        """
        root = root or core.thread_root(thread_id)
        meta = core.load_meta(thread_id, root) or {}
        return (self.conversations_base(meta.get("owner"), meta.get("page"))
                + "/" + self.folder_name(thread_id, root))

    def conversations_base(self, owner: Optional[str] = None,
                           page: Optional[str] = None) -> str:
        """The remote folder holding one person's conversations for one page:
        ``<root>/conversations/[<owner>/][<page>/]`` — the page segment only
        for a page other than the geotech page (see :meth:`session_folder`)."""
        segments = [self.root(), "conversations"]
        owner = sanitize_folder_name(str(owner or ""))
        if owner:
            segments.append(owner)
        page = sanitize_folder_name(str(page or ""))
        if page and page != DEFAULT_PAGE:
            segments.append(page)
        return "/".join(segments)

    # -- the mirror --------------------------------------------------------

    def mirror_conversation(self, thread_id: str,
                            root: Optional[str] = None) -> dict:
        """Upload this conversation's new/changed files. Never raises.

        Returns a summary dict: ``{"uploaded", "skipped", "errors", "web_url",
        "duration_s", "folder", "thread_id", "at"}`` (``at``: when it ended,
        epoch seconds) — plus ``"renamed_from"`` on
        the first sync after a rename — also kept as this conversation's own
        record (:meth:`last_sync_for`) and as ``self.last_sync``.
        """
        t0 = time.time()
        summary: dict = {"uploaded": 0, "skipped": 0, "errors": [],
                         "web_url": self._folder_urls.get(thread_id),
                         "duration_s": 0.0, "folder": None,
                         "thread_id": str(thread_id)}
        with self._lock:
            try:
                self._mirror_locked(thread_id, root, summary)
            except Exception as exc:  # backstop — a sync must never crash a turn
                summary["errors"].append(f"{type(exc).__name__}: {exc}")
        summary["at"] = time.time()
        summary["duration_s"] = round(summary["at"] - t0, 3)
        self.last_sync = summary
        self._syncs[str(thread_id)] = summary
        if summary["errors"] and self.backend == "sharepoint":
            try:                   # TA-35: kept in the app's SharePoint log
                self._log_save_errors(thread_id, root, summary)
            except Exception:                          # noqa: BLE001
                pass
        return summary

    def _log_save_errors(self, thread_id: str, root: Optional[str],
                         summary: dict) -> None:
        """TA-35: a save with errors, in the app's SharePoint log under the
        conversation's THREAD ID. Its folder's name -- the owner and a
        title made from the first question -- never goes in: inside the
        error texts it is replaced, as written and URL-quoted, by the
        thread id (``.../conversations/<thread id>``). The entry is tagged
        in memory with the owner, so only they see it in the diagnostics."""
        from urllib.parse import quote
        from webapp import sharepoint_log
        tid = str(thread_id)
        text = (f"{len(summary['errors'])} error(s): "
                + "; ".join(str(e) for e in summary["errors"][:5]))
        for folder in (summary.get("folder"), summary.get("renamed_from")):
            folder = str(folder or "")
            if "/conversations/" not in folder:
                continue
            tail = folder.split("/conversations/", 1)[1].strip("/")
            for part in (tail, tail.rsplit("/", 1)[-1]):  # owner/title, title
                if not part or part == tid:
                    continue
                for form in sorted({part, quote(part), quote(part, safe="")},
                                   key=len, reverse=True):
                    text = text.replace(form, tid)
        try:
            meta = core.load_meta(tid, root or core.thread_root(tid)) or {}
            owner = str(meta.get("owner") or "").strip()
        except Exception:                              # noqa: BLE001
            owner = ""
        sharepoint_log.record("conversation save", text,
                              f"conversations/{tid}", who=owner)

    # -- saving in the background ------------------------------------------

    def save_soon(self, thread_id: str, root: Optional[str] = None) -> None:
        """Queue a mirror of this conversation, run by one background
        worker, in order. A conversation already waiting is not queued
        again: that save will send this change too. The worker outlives the
        turn that asked, and at exit the queue is finished (up to a minute).
        Never raises."""
        tid = str(thread_id)
        with self._bg_cv:
            if tid in self._bg_waiting:
                return
            self._bg_waiting.add(tid)
            self._bg_queue.append((tid, root))
            if self._bg_worker is None or not self._bg_worker.is_alive():
                if self._bg_worker is None:
                    atexit.register(self.flush, 60.0)
                self._bg_worker = threading.Thread(
                    target=self._bg_run, name="permanent-storage-save",
                    daemon=True)
                self._bg_worker.start()
            self._bg_cv.notify_all()

    def _bg_run(self) -> None:
        while True:
            with self._bg_cv:
                while not self._bg_queue:
                    self._bg_cv.wait()
                tid, root = self._bg_queue.popleft()
                self._bg_waiting.discard(tid)
                self._bg_saving = tid
            try:
                self.mirror_conversation(tid, root=root)
            except Exception:                          # noqa: BLE001
                pass                                   # it never raises
            finally:
                with self._bg_cv:
                    self._bg_saving = None
                    self._bg_cv.notify_all()

    def save_state(self, thread_id: Optional[str]) -> Optional[str]:
        """``"saving"`` or ``"waiting"`` while this conversation has a save
        in the background, else ``None``."""
        tid = str(thread_id or "")
        with self._bg_cv:
            if tid and tid == self._bg_saving:
                return "saving"
            if tid in self._bg_waiting:
                return "waiting"
        return None

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Wait until the background saves are done; True when they are."""
        end = None if timeout is None else time.time() + timeout
        with self._bg_cv:
            while self._bg_queue or self._bg_saving is not None:
                left = None if end is None else end - time.time()
                if left is not None and left <= 0:
                    return False
                self._bg_cv.wait(left)
        return True

    def _mirror_locked(self, thread_id: str, root: Optional[str],
                       summary: dict) -> None:
        try:
            fm = self._file_manager()
        except Exception as exc:
            summary["errors"].append(
                f"{self.label} client: {type(exc).__name__}: {exc}")
            return
        conv_dir = core.conversation_dir(thread_id, root)
        if not os.path.isdir(conv_dir):
            return
        # A file manager with the batch hook (the Foundry dataset) lists
        # once for the whole sync (the folder-name check included) and gets
        # this sync's files in ONE save; they count as mirrored only once it
        # is committed, so a failed commit leaves them to be sent next time.
        batched = _begin_batch(fm)
        try:
            self._mirror_files(fm, thread_id, root, conv_dir, summary,
                               batched)
        except BaseException:
            if batched:
                _end_batch(fm, commit=False)
            raise

    def _mirror_files(self, fm, thread_id: str, root: Optional[str],
                      conv_dir: str, summary: dict, batched: bool) -> None:
        manifest = _load_manifest(conv_dir)
        files = _manifest_files(manifest)
        self._pin_folder(thread_id, root, manifest, fm)
        remote_base = self.session_folder(thread_id, root)
        summary["folder"] = remote_base

        # A changed GEOTECH_SHAREPOINT_ROOT (or, before folder names were
        # fixed at the first mirror, a rename) moves the mirror to a
        # new folder. Rather than move it server-side — the Funhouse file
        # manager exposes no rename/move, and a half-finished move is worse
        # than a duplicate — re-upload into the new folder and leave a pointer
        # behind. A conversation's files are small, and renames are rare.
        previous = manifest.get("folder")
        if isinstance(previous, str) and previous and previous != remote_base:
            summary["renamed_from"] = previous
            self._leave_moved_pointer(fm, previous, remote_base, summary)
            files = {}
            self._folder_urls.pop(thread_id, None)

        # In-progress files (core.is_in_flight_file: the turn checkpoint
        # partial.json, half-written *.part / *.tmp) are never uploaded. One
        # an older version uploaded -- the manifest says so -- is deleted
        # from the folder where the file manager can, and forgotten either
        # way: a restore skips it too (live smoke wave 2a, B2).
        for rel in [r for r in files if core.is_in_flight_file(r)]:
            files.pop(rel, None)
            delete = getattr(fm, "delete_file", None)
            if callable(delete):
                try:
                    delete(f"{remote_base}/{rel}")
                except Exception:                      # noqa: BLE001
                    pass

        sent = {} if batched else files
        self._upload_changed(fm, conv_dir, remote_base, files, sent, summary)
        if batched:
            try:
                fm.end_batch(commit=True)
                files.update(sent)
            except Exception as exc:
                summary["uploaded"] -= len(sent)
                summary["errors"].append(
                    f"{len(sent)} file(s) not saved: "
                    f"{type(exc).__name__}: {exc}")
            take_note = getattr(fm, "take_note", None)
            note = take_note() if callable(take_note) else None
            if note:
                summary["note"] = note           # said once per process

        _save_manifest(conv_dir, files, remote_base)
        if thread_id not in self._folder_urls:
            try:
                self._folder_urls[thread_id] = fix_web_url(
                    fm.get_web_url(remote_base))
            except Exception:
                pass
        summary["web_url"] = self._folder_urls.get(thread_id)

    def _upload_changed(self, fm, conv_dir: str, remote_base: str,
                        files: Dict[str, List], sent: Dict[str, List],
                        summary: dict) -> None:
        """Upload each file whose stamp differs from ``files`` (the
        manifest), recording what went up in ``sent`` (the manifest itself
        unless the sync is batched)."""
        for dirpath, _dirnames, filenames in os.walk(conv_dir):
            rel_dir = os.path.relpath(dirpath, conv_dir).replace(os.sep, "/")
            # The working folder's tool scratch (files/.scratch, any dot-
            # folder) and rebuildable caches (files/digest) stay local
            # (core.mirror_skips_dir). Pruned here so the walk never enters.
            _dirnames[:] = [
                d for d in _dirnames
                if not core.mirror_skips_dir(
                    d if rel_dir == "." else f"{rel_dir}/{d}")]
            for name in sorted(filenames):
                if name == MANIFEST_NAME:
                    continue
                local = os.path.join(dirpath, name)
                rel = name if rel_dir == "." else f"{rel_dir}/{name}"
                if core.is_in_flight_file(rel):
                    continue                    # a turn checkpoint, a part file
                try:
                    stamp = _stamp(local)
                except OSError:
                    continue                    # vanished mid-walk
                if files.get(rel) == stamp:
                    summary["skipped"] += 1
                    continue
                remote_dir = (remote_base if rel_dir == "."
                              else f"{remote_base}/{rel_dir}")
                try:
                    self._ensure_folder(fm, remote_dir)
                    fm.upload_file(local, f"{remote_dir}/{name}",
                                   overwrite=True)
                    sent[rel] = stamp
                    summary["uploaded"] += 1
                except Exception as exc:
                    summary["errors"].append(
                        f"{rel}: {type(exc).__name__}: {exc}")

    # -- the mirror, backwards: list + restore ------------------------------

    def list_remote_conversations(self, owner: Optional[str] = None,
                                  page: Optional[str] = None) -> List[dict]:
        """Mirrored conversation folders of one person's page, newest date
        first. Never raises.

        ``owner`` / ``page`` pick the folder the mirror files that page's
        conversations under (:meth:`conversations_base`); the defaults are the
        geotech page on a single-user host, ``<root>/conversations``. At the
        geotech page's level the other pages' own folders (``document_review``)
        sit beside the conversations and are left out — they are not
        conversations.

        Each entry: ``{"name": "<title>_<YYYY-MM-DD>", "path": <remote>}``.
        The folder name carries the title and the creation date, so it is
        the search key — no per-folder reads are made here.
        """
        if not self.configured:
            return []
        try:
            fm = self.file_manager()
            base = self.conversations_base(owner, page)
            entries = fm.ls(base) or []
        except Exception:
            return []
        skip = (page_folder_names() if _is_default_page(page) else set())
        out = []
        for e in entries:
            if not _is_folder(e):
                continue
            name = str(e.get("name") or "").strip()
            if not name or name in skip:
                continue
            out.append({"name": name,
                        "path": str(e.get("path") or f"{base}/{name}"),
                        "date": _trailing_date(name)})
        out.sort(key=lambda d: (d["date"], d["name"]), reverse=True)
        return out

    def restore_conversation(self, folder_name: str,
                             root: Optional[str] = None,
                             overwrite: bool = False,
                             owner: Optional[str] = None,
                             page: Optional[str] = None) -> dict:
        """Download one mirrored conversation back into the local store.

        ``folder_name`` is a name from :meth:`list_remote_conversations` with
        the SAME ``owner`` / ``page``; ``root`` is that page's local root.
        Downloads ``meta.json`` first to learn the thread id, then every other
        file (record files and ``files/`` — uploads and artifacts) into
        ``conversations/<thread_id>/``, and writes ``sp_manifest.json`` from
        the downloaded stamps so the next mirror reports everything
        up-to-date. Never raises. Returns ``{"status": "restored" |
        "exists" | "moved" | "error", "thread_id", "title", "downloaded",
        "errors", "duration_s", "folder", "moved_to"}``.
        """
        t0 = time.time()
        summary: dict = {"status": "error", "thread_id": None, "title": None,
                         "downloaded": 0, "errors": [], "folder": None,
                         "duration_s": 0.0}
        try:
            self._restore(folder_name, root, overwrite, summary, owner, page)
        except Exception as exc:
            summary["errors"].append(f"{type(exc).__name__}: {exc}")
        summary["duration_s"] = round(time.time() - t0, 2)
        return summary

    def _restore(self, folder_name: str, root, overwrite: bool,
                 summary: dict, owner: Optional[str] = None,
                 page: Optional[str] = None) -> None:
        from webapp import core

        if not self.configured:
            summary["errors"].append(f"{self.label} is not configured")
            return
        fm = self.file_manager()
        # In a batch the Foundry dataset lists the folder once for the walk
        # and the downloads, not once per subfolder (INF-38).
        batched = _begin_batch(fm)
        try:
            self._restore_from(fm, folder_name, root, overwrite, summary,
                               owner, page)
        finally:
            if batched:
                _end_batch(fm, commit=False)

    def _restore_from(self, fm, folder_name: str, root, overwrite: bool,
                      summary: dict, owner: Optional[str] = None,
                      page: Optional[str] = None) -> None:
        name = str(folder_name or "").strip().strip("/")
        if not name or "/" in name or name in (".", ".."):
            summary["errors"].append(f"not a conversation folder: {name!r}")
            return
        if _is_default_page(page) and name in page_folder_names():
            summary["errors"].append(
                f"{name!r} is another page's folder of conversations, not a "
                "conversation — open that page to restore from it")
            return
        remote_base = f"{self.conversations_base(owner, page)}/{name}"
        summary["folder"] = remote_base

        files = _walk_remote(fm, remote_base)
        if not files:
            summary["errors"].append(f"empty or missing folder: {remote_base}")
            return
        names = {rel for rel, _ in files}
        if "meta.json" not in names:
            if MOVED_NAME in names:
                summary["status"] = "moved"
                summary["moved_to"] = _read_moved_target(fm, remote_base)
                summary["errors"].append(
                    "this folder is the copy left behind by a rename — "
                    "restore the folder named in MOVED.txt instead")
            else:
                summary["errors"].append("no meta.json in the folder — not a "
                                         "conversation record")
            return

        with tempfile.TemporaryDirectory() as td:
            meta_local = os.path.join(td, "meta.json")
            fm.download_file(f"{remote_base}/meta.json", local_path=meta_local,
                             return_bytes=False, overwrite=True)
            with open(meta_local, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        thread_id = str((meta or {}).get("thread_id") or "").strip()
        if not thread_id or "/" in thread_id or "\\" in thread_id \
                or thread_id in (".", ".."):
            summary["errors"].append("meta.json carries no usable thread_id")
            return
        summary["thread_id"] = thread_id
        summary["title"] = (meta or {}).get("title")

        conv_dir = core.conversation_dir(thread_id, root)
        if os.path.isfile(os.path.join(conv_dir, "meta.json")) and not overwrite:
            summary["status"] = "exists"
            return

        stamps: Dict[str, List] = {}
        refused: List[str] = []
        wanted: List[Tuple[str, str, str]] = []      # rel, remote, local
        for rel, remote_path in files:
            if rel == MANIFEST_NAME:
                continue                     # never restore a stale manifest
            if core.is_in_flight_file(rel):
                # a turn checkpoint mirrored mid-turn would come back as a
                # fake "interrupted" turn (live smoke wave 2a, B2)
                continue
            local = _local_path_inside(conv_dir, rel)
            if local is None:
                refused.append(rel)
                continue
            os.makedirs(os.path.dirname(local), exist_ok=True)
            wanted.append((rel, remote_path, local))
        for (rel, _remote, local), exc in zip(wanted,
                                               _download_all(fm, wanted)):
            if exc is None:
                try:
                    stamps[rel] = _stamp(local)
                    summary["downloaded"] += 1
                    continue
                except OSError as missing:
                    exc = missing
            summary["errors"].append(f"{rel}: {type(exc).__name__}: {exc}")
        if refused:
            summary["errors"].append(
                f"{len(refused)} file(s) not restored, their paths lead "
                f"outside the conversation's folder: "
                + ", ".join(repr(r) for r in refused[:5])
                + (" …" if len(refused) > 5 else ""))
        _save_manifest(conv_dir, stamps, remote_base)
        summary["status"] = "restored" if summary["downloaded"] else "error"
        if summary["downloaded"]:
            # The record says the folder came back from the mirror (BACKLOG
            # EV-19: a restart and a restore were invisible in it).
            try:
                from webapp import activity_log
                activity_log.note_restored(
                    conv_dir, source="sharepoint",
                    files=summary["downloaded"],
                    bytes=sum(int(s[0]) for s in stamps.values()
                              if isinstance(s, (list, tuple)) and s),
                    errors=len(summary["errors"]), replaced=bool(overwrite))
            except Exception:  # noqa: BLE001 - a record, never the restore
                pass

    def _leave_moved_pointer(self, fm, old_base: str, new_base: str,
                             summary: dict) -> None:
        """Write ``MOVED.txt`` into the conversation's previous folder so the
        stale copy explains itself to whoever browses the library."""
        text = (
            "This conversation was renamed in GeotechStaffEngineer.\n\n"
            f"Its files now mirror to:\n    {new_base}\n\n"
            "The files in THIS folder are the copy made before the rename and "
            "are no longer updated. Delete this folder once you have checked "
            "the new one.\n"
        )
        try:
            with tempfile.TemporaryDirectory() as td:
                local = os.path.join(td, MOVED_NAME)
                with open(local, "w", encoding="utf-8") as fh:
                    fh.write(text)
                fm.upload_file(local, f"{old_base}/{MOVED_NAME}",
                               overwrite=True)
        except Exception as exc:
            summary["errors"].append(
                f"{MOVED_NAME}: {type(exc).__name__}: {exc}")

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


def _begin_batch(fm: Any) -> bool:
    """Start the file manager's batch when it offers the hook
    (``begin_batch`` and ``end_batch``); True when one was started."""
    begin = getattr(fm, "begin_batch", None)
    if not (callable(begin) and callable(getattr(fm, "end_batch", None))):
        return False
    try:
        begin()
    except Exception:                                  # noqa: BLE001
        return False
    return True


def _end_batch(fm: Any, commit: bool) -> None:
    """End a batch without letting its failure out (the abort paths)."""
    try:
        fm.end_batch(commit=commit)
    except Exception:                                  # noqa: BLE001
        pass


def _download_all(fm: Any, wanted: List[Tuple[str, str, str]]
                  ) -> List[Optional[BaseException]]:
    """Download each ``(rel, remote, local)``; one entry per file, ``None``
    or what went wrong. Through the file manager's ``download_files`` when
    it has one (the Foundry dataset: several at once, each about 3-4 s),
    else one ``download_file`` after another."""
    bulk = getattr(fm, "download_files", None)
    if callable(bulk):
        try:
            out = list(bulk([(remote, local) for _r, remote, local in wanted]))
        except Exception as exc:                       # noqa: BLE001
            out = [exc] * len(wanted)
        return (out + [None] * len(wanted))[:len(wanted)]
    out: List[Optional[BaseException]] = []
    for _rel, remote, local in wanted:
        try:
            fm.download_file(remote, local_path=local, return_bytes=False,
                             overwrite=True)
            out.append(None)
        except Exception as exc:                       # noqa: BLE001
            out.append(exc)
    return out


def _is_folder(entry: dict) -> bool:
    """Best-effort folder test over the Funhouse file-manager entry shape."""
    if not isinstance(entry, dict):
        return False
    kind = str(entry.get("type") or entry.get("kind") or "").lower()
    if kind:
        return kind.startswith("folder") or kind.startswith("dir")
    return bool(entry.get("is_folder") or entry.get("folder"))


def _is_default_page(page: Optional[str]) -> bool:
    return not page or sanitize_folder_name(str(page)) == DEFAULT_PAGE


def listing_cache_key(owner: Optional[str] = None,
                      page: Optional[str] = None) -> str:
    """The session-state key a page caches its list of mirrored
    conversations under: one per (person, page).

    The app's pages share ONE ``st.session_state`` (Streamlit's multipage
    design). Until 2026-10-07 the listing was cached under a single key, so
    whichever page a session opened first -- the Document Review page, the
    app's root -- filled it, and the geotech page showed that page's list
    instead of its own (owner, 2026-10-06: "the geotech page's past
    conversations no longer show")."""
    page_key = DEFAULT_PAGE if _is_default_page(page) else \
        sanitize_folder_name(str(page))
    owner_key = sanitize_folder_name(str(owner or ""))
    return f"sp_remote_list:{owner_key}:{page_key}"


def page_folder_names() -> set:
    """Folder names the mirror gives pages other than the geotech page; they
    sit beside the geotech page's conversations under ``conversations/``."""
    try:
        from webapp.profiles import PROFILES
        names = {sanitize_folder_name(n) for n in PROFILES}
    except Exception:                                  # noqa: BLE001
        names = {"document_review"}
    return {n for n in names if n and n != DEFAULT_PAGE}


def _trailing_date(name: str) -> str:
    """``YYYY-MM-DD`` suffix of a mirrored folder name, else ``""``."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})$", name or "")
    return m.group(1) if m else ""


def _local_path_inside(conv_dir: str, rel: str) -> Optional[str]:
    """Where remote file ``rel`` goes under ``conv_dir``, or ``None`` when
    its path is absolute, holds a ``..`` segment, or resolves outside the
    folder. A SharePoint library cannot hold such a name; a Foundry
    dataset's file paths are free text (INF-38), so a restore checks."""
    text = str(rel or "").replace("\\", "/")
    parts = text.split("/")
    if not text or text.startswith("/") or ".." in parts \
            or os.path.isabs(text) or re.match(r"^[A-Za-z]:", text):
        return None
    local = os.path.join(conv_dir, *parts)
    root = os.path.realpath(conv_dir)
    try:
        if os.path.commonpath([root, os.path.realpath(local)]) != root:
            return None
    except ValueError:                                 # another drive
        return None
    return local


def _walk_remote(fm, base: str, _depth: int = 0) -> List[tuple]:
    """``[(rel_path, remote_path), ...]`` for every file under ``base``."""
    out: List[tuple] = []
    if _depth > 6:
        return out
    for e in (fm.ls(base) or []):
        name = str(e.get("name") or "").strip()
        if not name:
            continue
        remote = str(e.get("path") or f"{base}/{name}")
        if _is_folder(e):
            for rel, rp in _walk_remote(fm, remote, _depth + 1):
                out.append((f"{name}/{rel}", rp))
        else:
            out.append((name, remote))
    return out


def _read_moved_target(fm, base: str) -> Optional[str]:
    try:
        with tempfile.TemporaryDirectory() as td:
            local = os.path.join(td, MOVED_NAME)
            fm.download_file(f"{base}/{MOVED_NAME}", local_path=local,
                             return_bytes=False, overwrite=True)
            with open(local, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if "/conversations/" in line:
                        return line
    except Exception:
        pass
    return None


def get_store(refresh: bool = False) -> SharePointStore:
    """The process-wide store (lazy). ``refresh=True`` rebuilds it (tests /
    changed env)."""
    global _STORE
    if refresh or _STORE is None:
        _STORE = SharePointStore()
    return _STORE


def save_after_change(store: Any, thread_id: str) -> Optional[dict]:
    """Save conversation ``thread_id`` to permanent storage after a turn or
    a feedback note: in the background where the store says so (the Foundry
    dataset, :attr:`SharePointStore.background`), else at once. Returns the
    summary of a save made at once, ``None`` for one queued (the sidebar
    reads its state from the store). Never raises."""
    try:
        if getattr(store, "background", False) is True \
                and callable(getattr(store, "save_soon", None)):
            store.save_soon(thread_id)
            return None
        return store.mirror_conversation(thread_id)
    except Exception:                                  # noqa: BLE001
        return None


def sync_for_conversation(store: Any, thread_id: Optional[str],
                          session_sync: Optional[dict] = None
                          ) -> Optional[dict]:
    """The mirror summary the sidebar shows for conversation ``thread_id``:
    the store's own record of this conversation's latest save when it has
    one (a save in the background reaches no session), else the session's
    record when it is this conversation's -- never another conversation's.

    Live smoke wave 2c (D3): the sidebar fell back to the store-wide
    ``last_sync``, which on a shared host is whoever mirrored last, so a
    fresh session showed another tester's folder name and "Open session
    folder" link. A session record that names a different conversation is
    ignored; one with no ``thread_id`` (an older summary shape) is taken as
    the session's own, since the app clears it on a new or opened
    conversation."""
    tid = str(thread_id or "")
    if not tid:
        return None
    getter = getattr(store, "last_sync_for", None)
    found = None
    if callable(getter):
        try:
            found = getter(tid)
        except Exception:                              # noqa: BLE001
            found = None
    if isinstance(found, dict):
        return found
    if isinstance(session_sync, dict):
        owner = str(session_sync.get("thread_id") or "")
        if not owner or owner == tid:
            return session_sync
    return None


__all__ = ["SharePointStore", "get_store", "configured", "DEFAULT_ROOT",
           "MANIFEST_NAME", "MOVED_NAME", "conversation_folder",
           "sanitize_folder_name", "folder_date", "save_after_change",
           "sync_for_conversation"]

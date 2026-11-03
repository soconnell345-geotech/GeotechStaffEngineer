"""Conversations kept in a Foundry dataset: permanent storage on Foundry.

On Funhouse the app mirrors each conversation folder to SharePoint after
every turn and restores past conversations from the sidebar
(:mod:`webapp.sharepoint_store`). On Foundry a published app's conversations
live in its container and are lost when the container is rebuilt (BACKLOG
INF-38). This module is the Foundry side of the same mirror: a file manager
over a Foundry dataset of raw files (no schema), with the methods
:class:`webapp.sharepoint_store.SharePointStore` calls on SharePoint's
(``ls``, ``create_folder``, ``upload_file``, ``download_file``,
``get_web_url``), so the store, the sidebar's permanent storage, "Sync now",
the past-conversations list and restore work unchanged on top of it.

How a dataset differs from a SharePoint library (the AI FDE's answer,
2026-11-03):

* Paths with slashes act as folders, so creating a folder does nothing.
* A listing returns files only, never folders; :meth:`DatasetFileManager.ls`
  makes the folder entries from the paths.
* Files are written inside a transaction, and a branch has one open
  transaction at a time. Every file one mirror changes therefore goes up in
  ONE ``UPDATE`` transaction (:meth:`DatasetFileManager.begin_batch` /
  :meth:`DatasetFileManager.end_batch`, which the store calls when a file
  manager offers them), and opening one is retried with a growing pause
  while another is open.
* There is no per-file web address, so :meth:`get_web_url` gives none.

Configuration (on a Foundry deployment, ``GEOTECH_DEPLOYMENT=foundry``):

    GEOTECH_FOUNDRY_CONVERSATION_DATASET   the dataset's RID (required)
    GEOTECH_FOUNDRY_CONVERSATION_BRANCH    its branch, default ``master``

The token and the host are the ones the app already uses for the language
model (:func:`webapp.engine_config.foundry_token`,
:func:`webapp.engine_config.foundry_base_url`); the token needs the datasets
read and write scopes. The Foundry platform SDK (pip
``foundry-platform-sdk``, import ``foundry_sdk``) is imported only when
storage is used and is not one of the package's requirements: without it,
permanent storage stays off and Connection diagnostics says why.

**None of the SDK calls has been run from a published app yet.** Every one
is in :meth:`DatasetFileManager._build_client` and the ``_sdk_*`` methods
below it, so a name the AI FDE's test corrects changes in one place.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

ENV_DATASET = "GEOTECH_FOUNDRY_CONVERSATION_DATASET"
ENV_BRANCH = "GEOTECH_FOUNDRY_CONVERSATION_BRANCH"
DEFAULT_BRANCH = "master"

#: The base folder inside the dataset (``GEOTECH_SHAREPOINT_ROOT`` still
#: overrides it, as it does on SharePoint).
DEFAULT_ROOT = "GeotechStaffEngineer"

#: The file the SDK reads to find the platform's services in a Foundry
#: container; when it is set and the app names no host of its own, the SDK
#: is left to find the host itself.
SERVICE_DISCOVERY_ENV = "FOUNDRY_SERVICE_DISCOVERY_V2"

#: The pip package, for the messages.
SDK_PACKAGE = "foundry-platform-sdk"

#: Seconds before one SDK request gives up.
TIMEOUT_S = 120

#: Opening a transaction while another is open on the branch: tries, and the
#: first pause (doubled each time: 0.5, 1, 2, 4 s, about 8 s in all).
OPEN_TRIES = 5
OPEN_BACKOFF_S = 0.5

#: Files per listing page.
PAGE_SIZE = 1000


def _env(name: str) -> str:
    return str(os.environ.get(name) or "").strip()


def dataset_rid() -> str:
    """The conversation dataset's RID, or ``""``."""
    return _env(ENV_DATASET)


def branch_name() -> str:
    """The branch conversations are written to (default ``master``)."""
    return _env(ENV_BRANCH) or DEFAULT_BRANCH


def selected() -> bool:
    """True on a Foundry deployment with the conversation dataset set: then
    the permanent storage is this dataset, whether or not it works."""
    from webapp import engine_config
    return engine_config.is_foundry_deployment() and bool(dataset_rid())


def sdk_target() -> Tuple[Optional[str], str]:
    """Where the SDK's requests go: ``(host, the setting it came from)``.

    The host is ``None`` when the SDK finds it itself. First that is set:
    ``GEOTECH_FOUNDRY_HOST`` (named on purpose, as for the model route);
    the SDK's service-discovery file (``FOUNDRY_SERVICE_DISCOVERY_V2``, set
    in Foundry containers); ``FOUNDRY_HOSTNAME`` / ``FOUNDRY_URL``, read by
    :func:`webapp.engine_config.foundry_base_url`. ``(None, "")`` when none
    is.
    """
    from webapp import engine_config
    explicit = engine_config.FOUNDRY_HOST_ENVS[0]
    if _env(explicit):
        return engine_config.foundry_base_url(), explicit
    if _env(SERVICE_DISCOVERY_ENV):
        return None, SERVICE_DISCOVERY_ENV
    source = next((e for e in engine_config.FOUNDRY_HOST_ENVS if _env(e)), "")
    return (engine_config.foundry_base_url(), source) if source else (None, "")


def sdk_installed() -> bool:
    try:
        import foundry_sdk  # noqa: F401
    except Exception:                                  # noqa: BLE001
        return False
    return True


def unconfigured_reason() -> Optional[str]:
    """Why the dataset cannot be used, in a sentence; ``None`` when it can."""
    from webapp import engine_config
    if not engine_config.is_foundry_deployment():
        return "this is not a Foundry deployment"
    if not dataset_rid():
        return (f"{ENV_DATASET} is not set, so conversations stay in the "
                "app's container and are lost when it is rebuilt")
    if not sdk_installed():
        return (f"the Foundry platform SDK (pip package {SDK_PACKAGE}, "
                "import foundry_sdk) is not installed in the app's "
                "environment; add it in the workspace Libraries panel and "
                "republish")
    if not engine_config.foundry_token():
        return ("no Foundry token (GEOTECH_FOUNDRY_TOKEN or FOUNDRY_TOKEN; "
                "Foundry normally provides FOUNDRY_TOKEN)")
    if not sdk_target()[1]:
        return ("no Foundry host (GEOTECH_FOUNDRY_HOST, "
                f"{SERVICE_DISCOVERY_ENV}, FOUNDRY_HOSTNAME or FOUNDRY_URL)")
    return None


def configured() -> bool:
    """True when the dataset is selected and everything it needs is there."""
    return selected() and unconfigured_reason() is None


def _is_open_transaction_conflict(exc: BaseException) -> bool:
    """The branch already has an open transaction (another save is running).
    Matched by the error's name, which is what the SDK's error classes and
    its generic 409 both carry."""
    marker = "OpenTransactionAlreadyExists"
    return (type(exc).__name__ == marker
            or str(getattr(exc, "name", "") or "") == marker
            or marker in str(exc))


def _clean(path: Optional[str]) -> str:
    """A dataset file path: forward slashes, no leading or trailing slash."""
    return str(path or "").replace("\\", "/").strip().strip("/")


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


class _Batch:
    """One mirror's or restore's state: its listings, and the transaction
    its uploads share, opened at the first upload."""

    def __init__(self, active: bool = False):
        self.active = active
        self.listings: Dict[str, List[dict]] = {}
        self.txn: Optional[str] = None
        self.error: Optional[BaseException] = None
        self.paths: set = set()


class DatasetFileManager:
    """Files in one Foundry dataset of raw files, as the app's mirror uses
    them. Nothing is contacted until the first call.

    ``client`` may be injected (tests, or a caller holding a live
    ``foundry_sdk.FoundryClient``); otherwise it is built from the app's
    Foundry token and host, and rebuilt when the token changes.
    """

    BACKEND = "foundry"
    DEFAULT_ROOT = DEFAULT_ROOT

    def __init__(self, dataset: Optional[str] = None,
                 branch: Optional[str] = None, *, client: Any = None,
                 open_tries: int = OPEN_TRIES,
                 open_backoff_s: float = OPEN_BACKOFF_S,
                 sleep: Callable[[float], None] = time.sleep,
                 page_size: int = PAGE_SIZE):
        self.dataset = (dataset or dataset_rid()).strip()
        if not self.dataset:
            raise RuntimeError(f"{ENV_DATASET} is not set")
        self.branch = (branch or branch_name()).strip()
        self._client_obj = client
        self._client_token: Optional[str] = None
        self._injected = client is not None
        self._open_tries = max(1, int(open_tries))
        self._open_backoff_s = float(open_backoff_s)
        self._sleep = sleep
        self._page_size = page_size
        self._prefix_ok = True
        # The batch, per thread: one process serves every viewer, and a
        # restore in one session must not end another session's save.
        self._local = threading.local()

    def _batch(self) -> "_Batch":
        batch = getattr(self._local, "batch", None)
        if batch is None:
            batch = self._local.batch = _Batch()
        return batch

    # ------------------------------------------------------------ the client
    def _client(self) -> Any:
        if self._injected:
            return self._client_obj
        from webapp import engine_config
        token = engine_config.foundry_token()
        if not token:
            raise RuntimeError(unconfigured_reason() or "no Foundry token")
        if self._client_obj is None or token != self._client_token:
            self._client_obj = self._build_client(token)
            self._client_token = token
        return self._client_obj

    @staticmethod
    def _build_client(token: str) -> Any:
        """``foundry_sdk.FoundryClient`` with the app's token and host. The
        CA file is the one the model route uses (``foundry_ca_bundle``)."""
        try:
            import foundry_sdk
        except ImportError as exc:
            raise RuntimeError(
                f"the Foundry platform SDK (pip package {SDK_PACKAGE}) is "
                "not installed") from exc
        from webapp import engine_config
        host, source = sdk_target()
        if not source:
            raise RuntimeError(unconfigured_reason() or "no Foundry host")
        config: Dict[str, Any] = {"timeout": TIMEOUT_S}
        ca = engine_config.foundry_ca_bundle()
        if ca:
            config["verify"] = ca
        kwargs: Dict[str, Any] = {
            "auth": foundry_sdk.UserTokenAuth(token=token)}
        if host:
            kwargs["hostname"] = host
            if host.lower().startswith("http://"):
                config["scheme"] = "http"   # the SDK drops a given scheme
        kwargs["config"] = foundry_sdk.Config(**config)
        return foundry_sdk.FoundryClient(**kwargs)

    # ----------------------------------------- the SDK calls, all of them
    def _files(self) -> Any:
        return self._client().datasets.Dataset.File

    def _transactions(self) -> Any:
        return self._client().datasets.Dataset.Transaction

    def _sdk_list(self, prefix: str, page_size: Optional[int] = None):
        """The files whose path starts with ``prefix`` (all pages)."""
        kwargs: Dict[str, Any] = {"branch_name": self.branch,
                                  "page_size": page_size or self._page_size}
        if prefix and self._prefix_ok:
            try:
                return self._files().list(self.dataset, path_prefix=prefix,
                                          **kwargs)
            except Exception as exc:                   # noqa: BLE001
                # An SDK without path_prefix refuses the argument (a
                # TypeError, or pydantic's ValidationError): list
                # everything and filter here.
                if not (isinstance(exc, TypeError)
                        or type(exc).__name__ == "ValidationError"):
                    raise
                self._prefix_ok = False
        return self._files().list(self.dataset, **kwargs)

    def _sdk_open(self) -> str:
        txn = self._transactions().create(self.dataset,
                                          transaction_type="UPDATE",
                                          branch_name=self.branch)
        return str(getattr(txn, "rid", None) or txn)

    def _sdk_upload(self, path: str, data: bytes, txn: str) -> None:
        self._files().upload(self.dataset, path, data, transaction_rid=txn)

    def _sdk_commit(self, txn: str) -> None:
        self._transactions().commit(self.dataset, txn)

    def _sdk_abort(self, txn: str) -> None:
        self._transactions().abort(self.dataset, txn)

    def _sdk_content(self, path: str) -> bytes:
        data = self._files().content(self.dataset, path,
                                     branch_name=self.branch)
        return data if isinstance(data, (bytes, bytearray)) else bytes(data)

    # ---------------------------------------------------------- transactions
    def _open_transaction(self) -> str:
        """An open ``UPDATE`` transaction, retried while another is open."""
        pause = self._open_backoff_s
        for attempt in range(1, self._open_tries + 1):
            try:
                return self._sdk_open()
            except Exception as exc:                   # noqa: BLE001
                if not _is_open_transaction_conflict(exc) \
                        or attempt == self._open_tries:
                    raise
            self._sleep(pause)
            pause *= 2
        raise RuntimeError("unreachable")              # pragma: no cover

    def _abort_quietly(self, txn: Optional[str]) -> None:
        if not txn:
            return
        try:
            self._sdk_abort(txn)
        except Exception:                              # noqa: BLE001
            pass

    # ------------------------------------------------------- the batch hook
    def begin_batch(self) -> None:
        """Start one mirror or restore: listings are fetched once and kept,
        and every upload until :meth:`end_batch` goes into ONE transaction,
        opened at the first upload (a mirror with nothing new opens none).
        The batch belongs to the calling thread."""
        self._local.batch = _Batch(active=True)

    def end_batch(self, commit: bool = True) -> int:
        """Commit (or, with ``commit=False``, abort) the batch's transaction,
        if one was opened, and drop its listings. Returns the number of
        files committed. A commit that fails is aborted and re-raised, so
        the caller records none of the batch's files as saved."""
        batch = self._batch()
        self._local.batch = _Batch()
        txn, paths = batch.txn, len(batch.paths)
        if not txn:
            return 0
        if not commit:
            self._abort_quietly(txn)
            return 0
        try:
            self._sdk_commit(txn)
        except Exception:
            self._abort_quietly(txn)
            raise
        return paths

    # -------------------------------------------------------- the file calls
    def _listing(self, prefix: str) -> List[dict]:
        """``[{"path", "size", "updated"}]`` for the files at or under
        ``prefix``; in a batch a listing already fetched for a folder above
        is reused."""
        batch = self._batch()
        if batch.active:
            for cached, files in batch.listings.items():
                if not cached or prefix == cached \
                        or prefix.startswith(cached + "/"):
                    return [f for f in files
                            if not prefix or f["path"] == prefix
                            or f["path"].startswith(prefix + "/")]
        files = []
        for item in self._sdk_list(prefix):
            path = _clean(getattr(item, "path", None)
                          or (item.get("path") if isinstance(item, dict)
                              else ""))
            if prefix and path != prefix \
                    and not path.startswith(prefix + "/"):
                continue
            files.append({"path": path,
                          "size": getattr(item, "size_bytes", None) or 0,
                          "updated": _iso(getattr(item, "updated_time", ""))})
        if batch.active:
            batch.listings[prefix] = files
        return files

    def ls(self, path: Optional[str] = None, recursive: bool = False,
           type: str = "both", **_ignored) -> List[Dict[str, Any]]:
        """The files and folders directly in ``path`` (``[]`` when there is
        nothing there), in the shape the store reads: ``name``, ``type``
        (``"file"`` / ``"folder"``), ``is_folder``, ``path``, ``size``.
        ``recursive`` lists everything below; ``type`` is ``files`` /
        ``folders`` / ``both``."""
        base = _clean(path)
        files = self._listing(base)
        if len(files) == 1 and files[0]["path"] == base and base:
            return [self._entry(files[0]["path"], files[0])]
        out: List[Dict[str, Any]] = []
        folders: set = set()
        for f in files:
            rest = f["path"][len(base) + 1:] if base else f["path"]
            if not rest:
                continue
            parts = rest.split("/")
            depth = len(parts) if recursive else 1
            for i in range(1, min(depth, len(parts) - 1) + 1):
                folder = "/".join(parts[:i])
                if folder not in folders:
                    folders.add(folder)
                    if type != "files":
                        out.append(self._entry(
                            f"{base}/{folder}" if base else folder, None))
            if (recursive or len(parts) == 1) and type != "folders":
                out.append(self._entry(f["path"], f))
        return out

    @staticmethod
    def _entry(path: str, info: Optional[dict]) -> Dict[str, Any]:
        is_folder = info is None
        return {"name": path.rsplit("/", 1)[-1],
                "type": "folder" if is_folder else "file",
                "is_folder": is_folder,
                "path": path,
                "size": 0 if is_folder else info.get("size", 0),
                "web_url": "",
                "lastModifiedDateTime": "" if is_folder
                else info.get("updated", "")}

    def create_folder(self, folder_path: str) -> Dict[str, Any]:
        """Nothing to do: a path with slashes is its own folder."""
        return {"name": _clean(folder_path).rsplit("/", 1)[-1],
                "path": _clean(folder_path), "type": "folder",
                "web_url": ""}

    def upload_file(self, local_path: str, path: str, overwrite: bool = False,
                    **_ignored) -> bool:
        """Write ``local_path`` to ``path``, replacing a file already there
        (a dataset keeps the newest version of a path; the store always
        passes ``overwrite=True``). In a batch it joins the batch's
        transaction; otherwise it is a transaction of its own. Raises on
        failure, so the store records the file as not saved."""
        target = _clean(path)
        if not target:
            raise ValueError("an empty dataset path")
        with open(local_path, "rb") as fh:
            data = fh.read()
        batch = self._batch()
        batch.listings = {}          # a listing from before this is stale
        if not batch.active:
            txn = self._open_transaction()
            try:
                self._sdk_upload(target, data, txn)
                self._sdk_commit(txn)
            except Exception:
                self._abort_quietly(txn)
                raise
            return True
        if batch.error is not None:
            raise RuntimeError(f"no transaction for this save: "
                               f"{type(batch.error).__name__}: {batch.error}")
        if batch.txn is None:
            try:
                batch.txn = self._open_transaction()
            except Exception as exc:
                batch.error = exc            # the other files fail fast
                raise
        if target in batch.paths:
            return True                      # one transaction, one write
        self._sdk_upload(target, data, batch.txn)
        batch.paths.add(target)
        return True

    def download_file(self, path: str, local_path: Optional[str] = None,
                      return_bytes: bool = True, overwrite: bool = False,
                      **_ignored) -> Any:
        """Bytes of ``path``; written to ``local_path`` when given (``True``
        back unless ``return_bytes``). An existing local file is kept
        unless ``overwrite``."""
        if local_path and os.path.exists(local_path) and not overwrite:
            if not return_bytes:
                return True
            with open(local_path, "rb") as fh:
                return fh.read()
        content = self._sdk_content(_clean(path))
        if local_path:
            os.makedirs(os.path.dirname(os.path.abspath(local_path)),
                        exist_ok=True)
            with open(local_path, "wb") as fh:
                fh.write(content)
            if not return_bytes:
                return True
        return content

    def get_web_url(self, path: str) -> str:
        """None to give: a dataset has no address per file or folder."""
        return ""

    # ------------------------------------------------------------ diagnostics
    def probe(self) -> str:
        """One small read (the first listing page under the base folder),
        for Connection diagnostics. Raises what the SDK raises."""
        first = next(iter(self._sdk_list("", page_size=1)), None)
        return "readable, " + ("not empty" if first is not None else "empty")

    def describe(self) -> str:
        """Where the files go, for Connection diagnostics."""
        source = "an injected client" if self._injected else sdk_target()[1]
        return (f"dataset {self.dataset}, branch {self.branch}, host from "
                f"{source or 'nowhere'}")


__all__ = ["DatasetFileManager", "ENV_DATASET", "ENV_BRANCH",
           "DEFAULT_BRANCH", "DEFAULT_ROOT", "configured",
           "dataset_rid", "branch_name", "sdk_target", "sdk_installed",
           "selected", "unconfigured_reason"]

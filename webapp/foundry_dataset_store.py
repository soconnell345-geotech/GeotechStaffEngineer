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

How it reaches the dataset (the AI FDE's test from a published app,
2026-11-03): the platform's datasets API is turned off inside app
containers, so the app uses the workspace dataset library a Code Workspace
provides, ``foundry.transforms.Dataset``. It finds the dataset by its
workspace ALIAS: the dataset is imported into the workspace (Data panel,
read and write), which names it in ``.foundry/aliases.yml``, and pushed.
Writes run as the publisher's account, as far as that test could tell.

How a dataset differs from a SharePoint library:

* Paths with slashes act as folders, so creating a folder does nothing.
* A listing returns every file in the dataset (no prefix filter), never
  folders; :meth:`DatasetFileManager.ls` filters on the path and makes the
  folder entries. A mirror or a restore lists once
  (:meth:`DatasetFileManager.begin_batch`).
* Every write is a transaction, and a dataset takes one open transaction at
  a time. One save therefore stages the files it changed under the outputs
  folder (``/foundry/outputs``), at their dataset paths, and uploads that
  folder in ONE transaction (``upload_directory``), under a process-wide
  lock. A save that meets another transaction open retries for about 30 s,
  then gives up until the next save.
* ``upload_directory`` uploads everything under the outputs folder, so the
  app never uses it while anything it did not stage is there; it then saves
  file by file (``upload_file``, one transaction each) and says so once.
* There is no per-file web address, so :meth:`get_web_url` gives none.
* A new dataset with no committed transaction yet answers every read with
  ``400 FoundryContainerService:FailedToResolveDataset`` ("does not have
  any committed transaction"); the app reads it as an empty store, and the
  first save makes the first commit (INF-65). The error's other causes (an
  alias pointing at the wrong dataset, no access) still fail plainly.
* What the latest read of the dataset found is kept for the process, so the
  sidebar and Connection diagnostics say the same (:meth:`known_problem`);
  the sidebar's first look starts one listing in the background.

Configuration (on a Foundry deployment, ``GEOTECH_DEPLOYMENT=foundry``):

    GEOTECH_FOUNDRY_CONVERSATION_DATASET   the dataset's workspace alias (required)

Nothing to install: the library comes with a Code Workspace. It is imported
only when storage is used; without it permanent storage stays off and
Connection diagnostics says why. Every library call is in a ``_lib_*``
method, :meth:`DatasetFileManager._dataset` (``Dataset.get``) or
:meth:`DatasetFileManager._fetch` (``File.download``).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, \
    Tuple

ENV_DATASET = "GEOTECH_FOUNDRY_CONVERSATION_DATASET"

#: The base folder inside the dataset (``GEOTECH_SHAREPOINT_ROOT`` still
#: overrides it, as it does on SharePoint).
DEFAULT_ROOT = "GeotechStaffEngineer"

#: The workspace dataset library, for the messages.
LIBRARY = "foundry.transforms"

#: The only folder ``upload_directory`` takes; a file's dataset path is its
#: path relative to it.
OUTPUTS_DIR = "/foundry/outputs"

#: Pauses between tries while another transaction holds the dataset (about
#: 30 s in all), then :class:`StorageBusy`.
BUSY_PAUSES_S = (1.0, 2.0, 4.0, 8.0, 15.0)

#: Downloads at once in a restore (each takes about 3-4 s).
PARALLEL_DOWNLOADS = 6

#: What to do when the alias does not work, for every message that says so.
ALIAS_HELP = ("import the dataset into the workspace (Data panel, read and "
              f"write), push, and set {ENV_DATASET} to its alias")

#: Process-wide: one save at a time touches the outputs folder and the
#: dataset (every instance; the store and report ingest's mirror alike).
_SAVE_LOCK = threading.Lock()


class _Reads:
    """What the latest read of each alias found, for every file manager in
    the process (the store's and Connection diagnostics' alike), and the
    sidebar's one background check per alias (INF-65)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.found: Dict[str, Tuple[bool, str]] = {}   # alias -> (ok, text)
        self.checks: Dict[str, threading.Thread] = {}


_READS = _Reads()


def wait_checks(timeout: float = 10.0) -> bool:
    """Wait for the background checks to end; True when they have."""
    end = time.time() + timeout
    for thread in list(_READS.checks.values()):
        thread.join(max(0.0, end - time.time()))
    return not any(t.is_alive() for t in _READS.checks.values())


class StorageBusy(RuntimeError):
    """Another transaction held the dataset for the whole wait."""


class AliasNotDefined(RuntimeError):
    """The workspace has no dataset under the configured alias."""


def _env(name: str) -> str:
    return str(os.environ.get(name) or "").strip()


def dataset_alias() -> str:
    """The conversation dataset's workspace alias, or ``""``."""
    return _env(ENV_DATASET)


def looks_like_resource_id(value: str) -> bool:
    """A resource ID (``ri.…``) where the alias belongs."""
    return str(value or "").strip().lower().startswith("ri.")


def selected() -> bool:
    """True on a Foundry deployment with the conversation dataset set: then
    the permanent storage is this dataset, whether or not it works."""
    from webapp import engine_config
    return engine_config.is_foundry_deployment() and bool(dataset_alias())


def library_installed() -> bool:
    try:
        from foundry.transforms import Dataset  # noqa: F401
    except Exception:                                  # noqa: BLE001
        return False
    return True


def unconfigured_reason() -> Optional[str]:
    """Why the dataset cannot be used, in a sentence; ``None`` when it can."""
    from webapp import engine_config
    if not engine_config.is_foundry_deployment():
        return "this is not a Foundry deployment"
    alias = dataset_alias()
    if not alias:
        return (f"{ENV_DATASET} is not set, so conversations stay in the "
                "app's container and are lost when it is rebuilt")
    if looks_like_resource_id(alias):
        return (f"{ENV_DATASET} holds a resource ID, not the dataset's "
                f"workspace alias: {ALIAS_HELP}")
    if not library_installed():
        return (f"the workspace dataset library ({LIBRARY}) is not in the "
                "app's environment; it comes with a Foundry Code Workspace, "
                "so publish the app from one")
    return None


def configured() -> bool:
    """True when the dataset is selected and everything it needs is there."""
    return selected() and unconfigured_reason() is None


def not_kept_reason(fm: Any = None) -> Optional[str]:
    """On a Foundry deployment, why conversations are NOT kept across
    restarts, with the fix, in one line: Connection diagnostics' line and
    the sidebar show this same text. ``None`` when they are kept, as far as
    is known, and off Foundry. ``fm`` (the store's file manager, when the
    dataset is set up) adds why the dataset could not be read, as its latest
    read found (:meth:`DatasetFileManager.known_problem`: an alias the
    workspace does not define, or any other refusal)."""
    from webapp import engine_config
    if not engine_config.is_foundry_deployment():
        return None
    if not dataset_alias():
        return (f"{unconfigured_reason()}. To keep them, see "
                "docs/FOUNDRY.md, 'Conversations that survive a rebuild'; "
                "meanwhile 'Download this conversation' in the sidebar "
                "keeps one.")
    reason = unconfigured_reason()
    if not reason:
        known = getattr(fm, "known_problem", None)
        reason = known() if callable(known) else None
    return f"{ENV_DATASET} is set, but {reason}." if reason else None


def _named(exc: BaseException, marker: str) -> bool:
    return (marker in type(exc).__name__
            or marker in str(getattr(exc, "name", "") or "")
            or marker in str(exc))


def _is_busy(exc: BaseException) -> bool:
    """Another transaction is open on the dataset (400
    ``Catalog:SimultaneousOpenTransactionsNotAllowed``)."""
    return _named(exc, "SimultaneousOpenTransactionsNotAllowed")


def _is_alias_missing(exc: BaseException) -> bool:
    """404 ``FoundryContainerSidecar:AliasNotDefined``."""
    return _named(exc, "AliasNotDefined")


_NO_COMMIT = re.compile(r"\b(?:no|not have any) committed transaction", re.I)


def _is_no_commit(exc: BaseException) -> bool:
    """A new dataset with no committed transaction yet (the FDE's build i,
    2026-11-03): ``400 FoundryContainerService:FailedToResolveDataset``,
    "The dataset does not have any committed transaction on the requested
    branch or across fallback branches". The error name AND that wording,
    so its other causes (an alias pointing at the wrong dataset, no access)
    still fail plainly. Looked for in the error and the errors it wraps."""
    texts: List[str] = []
    seen = 0
    while exc is not None and seen < 4:
        texts += [type(exc).__name__, str(getattr(exc, "name", "") or ""),
                  str(exc), str(getattr(exc, "message", "") or ""),
                  str(getattr(exc, "parameters", "") or "")]
        exc = exc.__cause__ or exc.__context__
        seen += 1
    return any("FailedToResolveDataset" in t for t in texts) \
        and any(_NO_COMMIT.search(t) for t in texts)


def _read_reason(exc: BaseException) -> str:
    """Why a read failed, in the words the sidebar and diagnostics show."""
    if isinstance(exc, AliasNotDefined):
        return str(exc)
    return (f"the conversation dataset could not be read: "
            f"{type(exc).__name__}: {exc}")


def _clean(path: Optional[str]) -> str:
    """A dataset file path: forward slashes, no leading or trailing slash."""
    return str(path or "").replace("\\", "/").strip().strip("/")


def _parts(path: str) -> List[str]:
    """The segments of a dataset path the app may write, or ValueError: one
    with an empty or ``..`` segment, or a drive, would stage outside the
    outputs folder."""
    parts = path.split("/")
    if not path or any(p in ("", ".", "..") for p in parts) \
            or ":" in parts[0]:
        raise ValueError(f"not a dataset path the app writes: {path!r}")
    return parts


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def _app_top() -> str:
    """The first segment of the store's base folder: the one folder in the
    outputs folder that is the app's own."""
    from webapp.sharepoint_store import ENV_ROOT
    root = _clean(os.environ.get(ENV_ROOT) or DEFAULT_ROOT)
    return root.split("/")[0] or DEFAULT_ROOT


class _Batch:
    """One mirror's or restore's state: the listing, read once, and the
    files the save will send, each copied when the store handed it over."""

    def __init__(self, active: bool = False):
        self.active = active
        self.files: Optional[List[dict]] = None
        self.index: Dict[str, Any] = {}
        self.staged: Dict[str, str] = {}      # dataset path -> local copy
        self.copies: Optional[str] = None     # the folder holding the copies

    def discard(self) -> None:
        if self.copies:
            shutil.rmtree(self.copies, ignore_errors=True)
            self.copies = None


class DatasetFileManager:
    """Files in one Foundry dataset of raw files, as the app's mirror uses
    them, through the workspace dataset library. Nothing is contacted until
    the first call.

    ``outputs_dir`` replaces :data:`OUTPUTS_DIR` (tests); ``busy_pauses``
    and ``sleep`` the wait while another transaction is open; ``workers``
    the downloads at once in a restore.
    """

    BACKEND = "foundry"
    DEFAULT_ROOT = DEFAULT_ROOT

    #: The outputs folder held something the app did not stage: said once
    #: per process.
    _warned_foreign = False

    def __init__(self, alias: Optional[str] = None, *,
                 outputs_dir: Optional[str] = None,
                 busy_pauses: Sequence[float] = BUSY_PAUSES_S,
                 sleep: Callable[[float], None] = time.sleep,
                 workers: int = PARALLEL_DOWNLOADS):
        self.alias = (alias or dataset_alias()).strip()
        if not self.alias:
            raise RuntimeError(f"{ENV_DATASET} is not set")
        self.outputs_dir = outputs_dir or OUTPUTS_DIR
        self._busy_pauses = tuple(float(p) for p in busy_pauses)
        self._sleep = sleep
        self._workers = max(1, int(workers))
        self._ds: Any = None
        self._note: Optional[str] = None
        #: The workspace does not define the alias (the last call said so;
        #: a call that works clears it).
        self.alias_error: Optional[str] = None
        #: The latest listing found a new dataset with nothing committed.
        self.uncommitted = False
        #: Shared by every instance: what the latest read found
        #: (:meth:`known_problem`).
        self._reads = _READS
        # The batch, per thread: one process serves every viewer, and a
        # restore in one session must not end another session's save.
        self._local = threading.local()

    def _batch(self) -> _Batch:
        batch = getattr(self._local, "batch", None)
        if batch is None:
            batch = self._local.batch = _Batch()
        return batch

    # ------------------------------------------------- the library calls
    def _dataset(self) -> Any:
        """``Dataset.get(alias)``, once per instance."""
        if self._ds is None:
            try:
                from foundry.transforms import Dataset
            except ImportError as exc:
                raise RuntimeError(
                    f"the workspace dataset library ({LIBRARY}) is not "
                    "installed") from exc
            self._ds = self._call(lambda: Dataset.get(self.alias))
        return self._ds

    def _lib_files(self) -> List[Any]:
        """Every file in the dataset: items with ``.path``, ``.size_bytes``,
        ``.updated_time`` and ``.download()``. Hidden ones too (a name
        starting ``_`` or ``.``), so a restore misses none."""
        ds = self._dataset()
        try:
            return list(ds.files(show_hidden_files=True))
        except TypeError:
            return list(ds.files())

    def _lib_upload_file(self, local: str, path: str) -> Any:
        return self._dataset().upload_file(local, path)

    def _lib_upload_directory(self) -> Any:
        return self._dataset().upload_directory(self.outputs_dir)

    def _call(self, fn: Callable[[], Any]) -> Any:
        """``fn()``, with an alias the workspace does not define said
        plainly (and kept in :attr:`alias_error`)."""
        try:
            out = fn()
        except Exception as exc:
            if _is_alias_missing(exc):
                self.alias_error = (
                    f"the workspace has no dataset alias {self.alias!r} "
                    f"(AliasNotDefined): {ALIAS_HELP}")
                raise AliasNotDefined(self.alias_error) from exc
            raise
        self.alias_error = None
        return out

    def known_problem(self) -> Optional[str]:
        """Why the dataset cannot be read, as far as this process knows: an
        alias the workspace does not define, or what the latest listing of
        this alias raised (Connection diagnostics' probe included), in the
        words its line shows. ``None`` when the latest read worked, and
        while nothing has read the alias yet: the first call then starts one
        listing in the background (about 3-4 s; :meth:`checking` is True
        until it ends), so the sidebar never says storage is on before a
        read has worked, and its first draw does not wait. Never raises."""
        if self.alias_error:
            return self.alias_error
        reads = self._reads
        with reads.lock:
            found = reads.found.get(self.alias)
            if found is None and self.alias not in reads.checks:
                check = threading.Thread(target=self._check, daemon=True,
                                         name="permanent-storage-check")
                reads.checks[self.alias] = check
                check.start()
        return found[1] if found is not None and not found[0] else None

    def _check(self) -> None:
        try:
            self._listing()                    # records what it found
        except Exception:                              # noqa: BLE001
            pass

    def checking(self) -> bool:
        """True while the background check of :meth:`known_problem` runs
        and nothing else has read the alias."""
        reads = self._reads
        with reads.lock:
            check = reads.checks.get(self.alias)
            return (check is not None and check.is_alive()
                    and self.alias not in reads.found)

    def _record_read(self, ok: bool, text: str) -> None:
        with self._reads.lock:
            self._reads.found[self.alias] = (ok, text)

    def _write(self, fn: Callable[[], Any]) -> Any:
        """One write, tried again while another transaction holds the
        dataset (:data:`BUSY_PAUSES_S`); then :class:`StorageBusy`."""
        pauses = list(self._busy_pauses)
        while True:
            try:
                return self._call(fn)
            except Exception as exc:
                if not _is_busy(exc):
                    raise
                if not pauses:
                    raise StorageBusy(
                        "another save holds the conversation dataset "
                        "(Foundry takes one open save at a time); waited "
                        f"about {round(sum(self._busy_pauses))} s; these "
                        "files go up with the next save") from exc
                self._sleep(pauses.pop(0))

    # ------------------------------------------------------- the batch hook
    def begin_batch(self) -> None:
        """Start one mirror or restore: the dataset is listed once, and
        every upload until :meth:`end_batch` goes up in ONE save (none when
        nothing changed). The batch belongs to the calling thread."""
        self._batch().discard()
        self._local.batch = _Batch(active=True)

    def end_batch(self, commit: bool = True) -> int:
        """Save the batch's files (with ``commit=False``, drop them) and its
        listing. Returns the number of files saved; raises when the save
        failed, so the caller records none of them as saved."""
        batch = self._batch()
        self._local.batch = _Batch()
        try:
            if not commit or not batch.staged:
                return 0
            with _SAVE_LOCK:
                self._save(batch.staged)
            return len(batch.staged)
        finally:
            batch.discard()

    def take_note(self) -> Optional[str]:
        """A one-time line for the sidebar (the outputs folder held files
        the app did not put there), or ``None``."""
        note, self._note = self._note, None
        return note

    # ---------------------------------------------------------------- a save
    def _save(self, staged: Dict[str, str]) -> None:
        """``staged`` ({dataset path: local copy}) in ONE transaction, when
        the outputs folder holds nothing but the app's own folder; else one
        transaction per file. Called under :data:`_SAVE_LOCK`."""
        top = _app_top()
        foreign = self._others_in_outputs(top)
        if foreign is None or foreign \
                or any(_parts(p)[0] != top for p in staged):
            if foreign:
                self._warn_foreign(foreign)
            for path, local in staged.items():
                self._write(lambda l=local, p=path: self._lib_upload_file(l, p))
            return
        placed: List[str] = []
        try:
            for path, local in staged.items():
                dst = os.path.join(self.outputs_dir, *_parts(path))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(local, dst)
                placed.append(dst)
            self._write(self._lib_upload_directory)
        finally:
            self._clear(placed)

    def _others_in_outputs(self, top: str) -> Optional[List[str]]:
        """What the outputs folder holds besides the app's own folder, after
        clearing that folder (anything there is left from a save that
        stopped). ``None`` when the folder cannot be used, or the app's own
        folder could not be cleared: then nothing may be staged there."""
        try:
            os.makedirs(self.outputs_dir, exist_ok=True)
            names = os.listdir(self.outputs_dir)
            if top in names:
                own = os.path.join(self.outputs_dir, top)
                if os.path.isdir(own) and not os.path.islink(own):
                    shutil.rmtree(own, ignore_errors=True)
                else:
                    os.remove(own)
                if os.path.lexists(own):
                    return None
        except OSError:
            return None
        return sorted(n for n in names if n != top)

    def _clear(self, placed: Iterable[str]) -> None:
        """Remove the files this save staged and the folders that held only
        them; nothing else in the outputs folder."""
        stop = os.path.normpath(self.outputs_dir)
        folders = set()
        for dst in placed:
            try:
                os.remove(dst)
            except OSError:
                pass
            folders.add(os.path.dirname(dst))
        for folder in sorted(folders, key=len, reverse=True):
            while os.path.normpath(folder) != stop \
                    and os.path.normpath(folder).startswith(stop):
                try:
                    os.rmdir(folder)
                except OSError:
                    break
                folder = os.path.dirname(folder)

    def _warn_foreign(self, foreign: List[str]) -> None:
        if DatasetFileManager._warned_foreign:
            return
        DatasetFileManager._warned_foreign = True
        self._note = (f"The outputs folder ({self.outputs_dir}) holds "
                      f"{len(foreign)} item(s) the app did not put there, so "
                      "saves go file by file (one transaction each, "
                      "slower). Move them out to save each turn in one step.")

    # ------------------------------------------------------------ listing
    def _listing(self) -> Tuple[List[dict], Dict[str, Any]]:
        """``([{"path", "size", "updated"}], {path: library file})`` for the
        whole dataset; in a batch, read once. A new dataset with nothing
        committed reads as empty. What it found is recorded for the process
        (:meth:`known_problem`)."""
        batch = self._batch()
        if batch.active and batch.files is not None:
            return batch.files, batch.index
        files: List[dict] = []
        index: Dict[str, Any] = {}
        try:
            items = self._call(self._lib_files)
            self.uncommitted = False
        except Exception as exc:
            if not _is_no_commit(exc):
                self._record_read(False, _read_reason(exc))
                raise
            items = []                 # the first save makes the first commit
            self.uncommitted = True
        for item in items:
            path = _clean(getattr(item, "path", None))
            if not path:
                continue
            files.append({"path": path,
                          "size": getattr(item, "size_bytes", None) or 0,
                          "updated": _iso(getattr(item, "updated_time", ""))})
            index[path] = item
        self._record_read(True, self._state(files))
        if batch.active:
            batch.files, batch.index = files, index
        return files, index

    def _under(self, prefix: str) -> List[dict]:
        files, _ = self._listing()
        return [f for f in files if not prefix or f["path"] == prefix
                or f["path"].startswith(prefix + "/")]

    def ls(self, path: Optional[str] = None, recursive: bool = False,
           type: str = "both", **_ignored) -> List[Dict[str, Any]]:
        """The files and folders directly in ``path`` (``[]`` when there is
        nothing there), in the shape the store reads: ``name``, ``type``
        (``"file"`` / ``"folder"``), ``is_folder``, ``path``, ``size``.
        ``recursive`` lists everything below; ``type`` is ``files`` /
        ``folders`` / ``both``."""
        base = _clean(path)
        files = self._under(base)
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

    # -------------------------------------------------------- the file calls
    def create_folder(self, folder_path: str) -> Dict[str, Any]:
        """Nothing to do: a path with slashes is its own folder."""
        return {"name": _clean(folder_path).rsplit("/", 1)[-1],
                "path": _clean(folder_path), "type": "folder",
                "web_url": ""}

    def upload_file(self, local_path: str, path: str, overwrite: bool = False,
                    **_ignored) -> bool:
        """Write ``local_path`` to ``path``, replacing a file already there
        (a dataset keeps the newest version of a path, and the files it
        does not name). In a batch the file is copied now and goes up with
        the batch; otherwise it is a transaction of its own. Raises on
        failure, so the store records the file as not saved."""
        target = _clean(path)
        parts = _parts(target)
        batch = self._batch()
        if not batch.active:
            with _SAVE_LOCK:
                self._write(lambda: self._lib_upload_file(local_path, target))
            return True
        # A copy now: the file may change before the save, or be a
        # temporary one gone by then (the MOVED.txt pointer).
        if batch.copies is None:
            batch.copies = tempfile.mkdtemp(prefix="gse_save_")
        copy = os.path.join(batch.copies, *parts)
        os.makedirs(os.path.dirname(copy), exist_ok=True)
        shutil.copyfile(local_path, copy)
        batch.staged[target] = copy
        return True

    def _fetch(self, path: str, index: Optional[Dict[str, Any]] = None
               ) -> str:
        """The local copy the library downloads of ``path``."""
        if index is None:
            _, index = self._listing()
        item = index.get(_clean(path))
        if item is None:
            raise FileNotFoundError(f"not in the conversation dataset: {path}")
        return self._call(item.download)

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
        src = self._fetch(path)
        if local_path:
            os.makedirs(os.path.dirname(os.path.abspath(local_path)),
                        exist_ok=True)
            shutil.copyfile(src, local_path)
            if not return_bytes:
                return True
        with open(src, "rb") as fh:
            return fh.read()

    def download_files(self, pairs: Iterable[Tuple[str, str]]
                       ) -> List[Optional[BaseException]]:
        """Each ``(dataset path, local path)`` downloaded and copied to its
        local path, up to :data:`PARALLEL_DOWNLOADS` at once; the dataset is
        listed once, in the caller's thread. One entry per pair: ``None``,
        or what went wrong. The caller has checked the local paths."""
        pairs = list(pairs)
        if not pairs:
            return []
        _, index = self._listing()

        def one(pair: Tuple[str, str]) -> None:
            remote, local = pair
            src = self._fetch(remote, index)
            os.makedirs(os.path.dirname(os.path.abspath(local)),
                        exist_ok=True)
            shutil.copyfile(src, local)

        workers = min(self._workers, len(pairs))
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="gse-restore") as pool:
            futures = [pool.submit(one, p) for p in pairs]
        return [f.exception() for f in futures]

    def get_web_url(self, path: str) -> str:
        """None to give: a dataset has no address per file or folder."""
        return ""

    # ------------------------------------------------------------ diagnostics
    def probe(self) -> str:
        """One listing, for Connection diagnostics. Raises what the library
        raises, after recording it for the sidebar."""
        files, _ = self._listing()
        return self._state(files)

    def _state(self, files: List[dict]) -> str:
        if not files and self.uncommitted:
            return ("readable, 0 files: a new dataset with nothing "
                    "committed yet; the first save makes its first commit")
        top = _app_top()
        mine = sum(1 for f in files if f["path"].startswith(top + "/"))
        return f"readable, {len(files)} file(s), {mine} of them the app's"

    def describe(self) -> str:
        """Where the files go, for Connection diagnostics."""
        return (f"workspace dataset alias {self.alias}, saves staged in "
                f"{self.outputs_dir}")


__all__ = ["DatasetFileManager", "ENV_DATASET", "DEFAULT_ROOT",
           "AliasNotDefined", "StorageBusy", "configured", "dataset_alias",
           "library_installed", "looks_like_resource_id", "not_kept_reason",
           "selected", "unconfigured_reason", "wait_checks"]

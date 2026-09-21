"""A durable copy of a run's output, because the cluster's disk is not one.

WHY THIS EXISTS. The first full cluster run wrote 38 label-review runs into
``/tmp/report_ingest_520`` -- about $17 of model calls -- and a cluster
restart wiped them. The owner's words, 2026-09-20: *"We should really be
saving the files somewhere other than tmp. Big waste of money. You know
we've had this issue elsewhere."* The app has mirrored every conversation to
SharePoint after every turn since 5.10.0 for exactly this reason
(``webapp/sharepoint_store.py``); this is the same idea for the scoring runs,
in the shipped package, with no dependency on the app.

WHAT IT IS. One :class:`Mirror` over one or two backends:

* a **SharePoint file manager** -- the live Funhouse object, or anything
  exposing ``create_folder`` / ``upload_file`` / ``ls`` / ``download_file``.
  The ``fh_sp_client`` itself is accepted too: its ``.file_manager`` is taken
  when it has one.
* a **plain filesystem path** (``durable_dir``) -- the owner's workspace
  folder under ``geotech_app/``, which persists, or a Volume. Never ``/tmp``:
  the owner's rule of 2026-09-20 names the workspace folder and SharePoint as
  the two places that keep things here. Files are copied.

Both are driven through one duck type, so a run can mirror to either, or to
both at once.

INCREMENTAL, LIKE THE APP'S MIRROR. ``mirror_manifest.json`` in the local
folder records each mirrored file's ``(size, mtime)``; a file whose stamp is
unchanged is skipped. The manifest is never itself uploaded, and it is keyed
by backend, so two backends do not credit each other's uploads.

IT NEVER RAISES. A mirror is insurance, not the work: every error is caught
and returned on the summary. A SharePoint outage costs the durability of a
run, and a run that stopped because of one would cost the run.

IT ALSO RUNS BACKWARDS. :meth:`Mirror.restore_dir` downloads every remote
file that is missing locally, which is what makes a wiped ``/tmp`` resume
from what the mirror already holds rather than paying for it twice.

PRIVACY. It sends the WHOLE folder, which is the point: the files worth
saving are the expensive ones, and those are the run files whose reasons and
rationales can name a firm, a project or a person. The remote folder is
therefore exactly as private as the local one, and belongs wherever the
reports themselves already live. Nothing here decides that; the caller names
the folder.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

__all__ = ["Mirror", "MANIFEST_NAME", "DEFAULT_FOLDER", "file_manager_of"]

#: The local manifest of what has been mirrored. Never uploaded itself.
MANIFEST_NAME = "mirror_manifest.json"

#: The remote folder a run's output goes under, unless the caller says
#: otherwise. The owner's hand-made copy of the 2026-09-20 sheet-mode run
#: sits at ``GeotechStaffEngineer/report_ingest/521_sheet``, so a run named
#: ``521_sheet`` lands exactly where they already put it.
DEFAULT_FOLDER = "GeotechStaffEngineer/report_ingest"

#: How deep a remote listing may go before it stops. A run's output is two
#: levels (``runs/<ID>.json``); this is a guard against a cycle, not a limit
#: anything real reaches.
MAX_REMOTE_DEPTH = 6


def file_manager_of(sharepoint: Any) -> Any:
    """The file manager inside whatever the caller passed, or ``None``.

    The owner's notebook holds ``fh_sp_client`` (a Funhouse SharePoint
    client, whose ``.file_manager`` is the thing with the methods) and the
    app holds a :class:`webapp.sharepoint_store.SharePointStore` (whose
    ``file_manager`` is a METHOD). Either is accepted, and so is a file
    manager passed directly, because asking the owner to remember which of
    the three this parameter wants is asking for the run that mirrors
    nowhere.
    """
    if sharepoint is None:
        return None
    if hasattr(sharepoint, "upload_file"):
        return sharepoint
    inner = getattr(sharepoint, "file_manager", None)
    if inner is None:
        return sharepoint            # let the first call fail onto the summary
    if callable(inner) and not hasattr(inner, "upload_file"):
        try:
            inner = inner()
        except Exception:
            return None
    return inner


def _stamp(path: Path) -> List:
    st = path.stat()
    return [st.st_size, round(st.st_mtime, 3)]


def _is_folder(entry: dict) -> bool:
    """Best-effort folder test over the Funhouse file-manager entry shape."""
    if not isinstance(entry, dict):
        return False
    kind = str(entry.get("type") or entry.get("kind") or "").lower()
    if kind:
        return kind.startswith("folder") or kind.startswith("dir")
    return bool(entry.get("is_folder") or entry.get("folder"))


def _walk_remote(fm: Any, base: str, _depth: int = 0) -> List[Tuple[str, str]]:
    """``[(rel path, remote path), ...]`` for every file under ``base``."""
    out: List[Tuple[str, str]] = []
    if _depth > MAX_REMOTE_DEPTH:
        return out
    for entry in (fm.ls(base) or []):
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        remote = str(entry.get("path") or f"{base}/{name}")
        if _is_folder(entry):
            for rel, path in _walk_remote(fm, remote, _depth + 1):
                out.append((f"{name}/{rel}", path))
        else:
            out.append((name, remote))
    return out


# ---------------------------------------------------------------------------
# the two backends, behind one duck type
# ---------------------------------------------------------------------------

class _FileManagerBackend:
    """A SharePoint (or SharePoint-shaped) file manager as a mirror."""

    def __init__(self, fm: Any) -> None:
        self._fm = fm
        self._made: set = set()

    @property
    def key(self) -> str:
        return "sharepoint"

    def describe(self, remote_dir: str) -> str:
        return f"SharePoint {remote_dir}"

    def ensure_dir(self, remote_dir: str) -> None:
        parts = [p for p in str(remote_dir).split("/") if p]
        for i in range(1, len(parts) + 1):
            partial = "/".join(parts[:i])
            if partial in self._made:
                continue
            self._fm.create_folder(partial)
            self._made.add(partial)

    def put(self, local: Path, remote_path: str) -> None:
        self._fm.upload_file(str(local), remote_path, overwrite=True)

    def walk(self, remote_dir: str) -> List[Tuple[str, str]]:
        return _walk_remote(self._fm, str(remote_dir).strip("/"))

    def get(self, remote_path: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self._fm.download_file(remote_path, local_path=str(local),
                               return_bytes=False, overwrite=True)


class _DirBackend:
    """A plain folder as a mirror: the owner's workspace folder or a Volume.

    Never ``/tmp`` -- the owner's rule of 2026-09-20.

    ``strip`` is the remote prefix a SharePoint mirror carries and a folder
    does not need: a run mirrored to ``GeotechStaffEngineer/report_ingest/
    521_sheet`` on SharePoint lands in ``<durable_dir>/521_sheet``, not in a
    copy of the library's folder tree. A remote path that does not begin with
    the prefix is used whole.

    THE RUN NAME IS NOT ADDED TWICE. ``durable_dir`` gets written both ways
    -- as the PARENT (``.../results``, with the run's own folder made inside
    it) and as this run's folder named in full (``.../results/521_sheet``)
    -- and both mean the same place. So where the folder's own name already
    IS the run's name, the run is not nested inside itself. Nothing wants
    ``521_sheet/521_sheet``, and a restore that looked there would find an
    empty folder and let the run pay for everything twice.
    """

    def __init__(self, root: Any, strip: str = "") -> None:
        self._root = Path(root)
        self._strip = str(strip or "").strip("/")

    @property
    def key(self) -> str:
        return f"dir:{self._root}"

    def resolve(self, remote_dir: str) -> Path:
        rel = str(remote_dir).replace("\\", "/").strip("/")
        if self._strip and (rel == self._strip
                            or rel.startswith(self._strip + "/")):
            rel = rel[len(self._strip):].strip("/")
        parts = [p for p in rel.split("/") if p]
        if parts and self._root.name == parts[0]:
            parts = parts[1:]
        return self._root.joinpath(*parts) if parts else self._root

    def describe(self, remote_dir: str) -> str:
        return str(self.resolve(remote_dir))

    def ensure_dir(self, remote_dir: str) -> None:
        self.resolve(remote_dir).mkdir(parents=True, exist_ok=True)

    def put(self, local: Path, remote_path: str) -> None:
        target = self.resolve(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(local), str(target))

    def walk(self, remote_dir: str) -> List[Tuple[str, str]]:
        base = self.resolve(remote_dir)
        out: List[Tuple[str, str]] = []
        if not base.is_dir():
            return out
        for dirpath, _dirnames, filenames in os.walk(base):
            rel_dir = os.path.relpath(dirpath, base).replace(os.sep, "/")
            for name in sorted(filenames):
                rel = name if rel_dir == "." else f"{rel_dir}/{name}"
                out.append((rel, str(Path(dirpath) / name)))
        return out

    def get(self, remote_path: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_path, str(local))


# ---------------------------------------------------------------------------
# the mirror
# ---------------------------------------------------------------------------

class Mirror:
    """Keeps a local folder copied somewhere that survives a restart.

    Parameters
    ----------
    sharepoint
        The live ``fh_sp_client``, its ``.file_manager``, or the app's
        ``SharePointStore``. ``None`` for no SharePoint backend.
    durable_dir
        A folder that survives the driver: the owner's workspace folder
        under ``geotech_app/`` or a Volume; never ``/tmp``. ``None``
        for no filesystem backend.
    base_folder
        The remote folder the run folders sit under. It prefixes the
        SharePoint paths and is stripped from the filesystem ones, so one
        ``remote_dir`` names the same run in both.

    With neither backend the mirror is INACTIVE: every call returns an empty
    summary and does nothing, which is what a run with no ``sharepoint`` and
    no ``durable_dir`` should cost.
    """

    def __init__(self, sharepoint: Any = None, durable_dir: Any = None,
                 base_folder: str = DEFAULT_FOLDER) -> None:
        self.base_folder = str(base_folder or "").strip("/")
        self.backends: List[Any] = []
        fm = file_manager_of(sharepoint)
        if fm is not None:
            self.backends.append(_FileManagerBackend(fm))
        if durable_dir:
            self.backends.append(_DirBackend(durable_dir,
                                             strip=self.base_folder))

    @property
    def active(self) -> bool:
        return bool(self.backends)

    def remote_for(self, name: str) -> str:
        """The remote folder for a run whose ``out_dir`` is named ``name``."""
        name = str(name).strip("/")
        return f"{self.base_folder}/{name}" if self.base_folder else name

    def describe(self, remote_dir: str) -> str:
        """Where this mirror points, in one line, for the run's first print."""
        if not self.backends:
            return "nowhere (no sharepoint= and no durable_dir=)"
        return " and ".join(b.describe(remote_dir) for b in self.backends)

    # -- forwards ---------------------------------------------------------

    def mirror_dir(self, local_dir: Any, remote_dir: str) -> Dict[str, Any]:
        """Upload every new or changed file under ``local_dir``. Never raises.

        Returns ``{"uploaded", "skipped", "errors", "remote", "duration_s"}``.
        """
        started = time.time()
        remote = str(remote_dir).strip("/")
        summary: Dict[str, Any] = {"uploaded": 0, "skipped": 0,
                                   "errors": [], "remote": remote,
                                   "duration_s": 0.0}
        try:
            self._mirror(Path(local_dir), remote, summary)
        except Exception as exc:                 # a mirror must never stop a run
            summary["errors"].append(f"{type(exc).__name__}: {exc}")
        summary["duration_s"] = round(time.time() - started, 3)
        return summary

    def _mirror(self, local: Path, remote: str,
                summary: Dict[str, Any]) -> None:
        if not self.backends or not local.is_dir():
            return
        manifest = self._manifest(local, remote)
        local_files = _walk_local(local)
        for backend in self.backends:
            stamps = manifest["files"].setdefault(backend.key, {})
            for rel, path in local_files:
                try:
                    stamp = _stamp(path)
                except OSError:
                    continue                     # vanished mid-walk
                if stamps.get(rel) == stamp:
                    summary["skipped"] += 1
                    continue
                remote_path = f"{remote}/{rel}"
                try:
                    backend.ensure_dir(_parent(remote_path))
                    backend.put(path, remote_path)
                    stamps[rel] = stamp
                    summary["uploaded"] += 1
                except Exception as exc:
                    summary["errors"].append(
                        f"{rel}: {type(exc).__name__}: {exc}")
        _save_manifest(local, manifest)

    # -- backwards --------------------------------------------------------

    def restore_dir(self, remote_dir: str, local_dir: Any,
                    overwrite: bool = False) -> Dict[str, Any]:
        """Download every remote file that is missing locally. Never raises.

        Returns ``{"downloaded", "skipped", "errors", "remote",
        "duration_s"}``. The stamps of what was downloaded go into the
        manifest, so the mirror that follows does not send it all straight
        back again.
        """
        started = time.time()
        remote = str(remote_dir).strip("/")
        summary: Dict[str, Any] = {"downloaded": 0, "skipped": 0,
                                   "errors": [], "remote": remote,
                                   "duration_s": 0.0}
        try:
            self._restore(remote, Path(local_dir), overwrite, summary)
        except Exception as exc:
            summary["errors"].append(f"{type(exc).__name__}: {exc}")
        summary["duration_s"] = round(time.time() - started, 3)
        return summary

    def _restore(self, remote: str, local: Path, overwrite: bool,
                 summary: Dict[str, Any]) -> None:
        if not self.backends:
            return
        local.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest(local, remote)
        for backend in self.backends:
            try:
                entries = backend.walk(remote)
            except Exception as exc:
                summary["errors"].append(
                    f"{backend.key}: {type(exc).__name__}: {exc}")
                continue
            stamps = manifest["files"].setdefault(backend.key, {})
            for rel, remote_path in entries:
                if rel == MANIFEST_NAME:
                    continue                     # never restore a stale one
                target = local.joinpath(*rel.split("/"))
                if target.exists() and not overwrite:
                    summary["skipped"] += 1
                    continue
                try:
                    backend.get(remote_path, target)
                    stamps[rel] = _stamp(target)
                    summary["downloaded"] += 1
                except Exception as exc:
                    summary["errors"].append(
                        f"{rel}: {type(exc).__name__}: {exc}")
        _save_manifest(local, manifest)

    # -- the manifest ------------------------------------------------------

    def _manifest(self, local: Path, remote: str) -> Dict[str, Any]:
        """The local manifest, reset if it belongs to a different remote."""
        manifest = _load_manifest(local)
        if manifest.get("remote") != remote:
            manifest = {"remote": remote, "files": {}}
        if not isinstance(manifest.get("files"), dict):
            manifest["files"] = {}
        return manifest


def _parent(remote_path: str) -> str:
    head, _, _tail = str(remote_path).rpartition("/")
    return head


def _walk_local(local: Path) -> List[Tuple[str, Path]]:
    """``[(rel path, local path), ...]``, the manifest itself left out."""
    out: List[Tuple[str, Path]] = []
    for dirpath, _dirnames, filenames in os.walk(local):
        rel_dir = os.path.relpath(dirpath, local).replace(os.sep, "/")
        for name in sorted(filenames):
            if rel_dir == "." and name == MANIFEST_NAME:
                continue
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            out.append((rel, Path(dirpath) / name))
    return out


def _load_manifest(local: Path) -> Dict[str, Any]:
    try:
        data = json.loads((local / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_manifest(local: Path, manifest: Dict[str, Any]) -> None:
    try:
        (local / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    except OSError:
        pass                                     # best-effort, like the rest

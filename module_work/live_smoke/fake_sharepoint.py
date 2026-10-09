"""A SharePoint file manager backed by a local folder.

Stands in for the Funhouse SDK's file manager (and for
``webapp.graph_sharepoint.GraphFileManager`` on Tiny Apps) behind
``webapp.sharepoint_store.SharePointStore(file_manager=...)``, so the
conversation mirror, the restore and the agent's four SharePoint tools all
run their real code against it. It answers the way the tests' fakes in
``webapp/tests/test_sharepoint_*.py`` do (the SDK's shapes):

* ``ls`` -> ``[{"name", "type": "file"|"folder", "size", "path"}]``, ``[]``
  for a missing folder;
* ``download_file`` raises ``FileNotFoundError("... Download failed: 404 ...")``
  for a missing file;
* ``upload_file(..., overwrite=False)`` returns False when the name is taken;
* ``search_filenames`` matches on the file NAME (substring, any case).

Every call is logged in :attr:`calls` so a detector can see what the app and
the agent asked of SharePoint. Paths are library paths
(``Shared Documents/...``), ``/sites/<site>/...`` server-relative paths, or a
web URL this fake itself handed out.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from typing import List, Optional
from urllib.parse import quote, unquote, urlsplit

#: Synthetic tenant (Microsoft's documentation name, as the app's tests use).
SITE_URL = "https://contoso.sharepoint.com/sites/LiveSmoke"
SITE_PATH = "/sites/LiveSmoke"
#: The app's base folder in the library (the layout the deployment uses).
ROOT = "Shared Documents/General/GSE_app"


def _long(path: str) -> str:
    if os.name == "nt" and not path.startswith(_LONG_PREFIX):
        return _LONG_PREFIX + path
    return path


#: Windows' extended-length path prefix (backslash backslash ? backslash).
_LONG_PREFIX = "\\\\?\\"


class LocalSharePointFM:
    """A SharePoint document library kept in ``store_dir``."""

    def __init__(self, store_dir: str, site_url: str = SITE_URL):
        # SharePoint allows 400-character paths; Windows' 260 would make the
        # FAKE fail where the real library would not, so the store uses the
        # extended-length form on Windows.
        self.store_dir = _long(os.path.abspath(store_dir))
        self.site_url = site_url.rstrip("/")
        self.site_path = urlsplit(self.site_url).path.rstrip("/")
        os.makedirs(self.store_dir, exist_ok=True)
        self.calls: List[dict] = []
        self._lock = threading.Lock()

    # -- paths -------------------------------------------------------------
    def norm(self, path: str) -> str:
        """A library path (``Shared Documents/...``) for any accepted form."""
        p = str(path or "").strip()
        if p.lower().startswith("http"):
            parts = urlsplit(p)
            p = unquote(parts.path)
        p = p.replace("\\", "/")
        sp = self.site_path.lower()
        if p.lower().startswith(sp + "/"):
            p = p[len(sp) + 1:]
        p = p.strip("/")
        segs = [s for s in p.split("/") if s not in ("", ".")]
        if any(s == ".." for s in segs):
            raise ValueError(f"path escapes the library: {path!r}")
        return "/".join(segs)

    def local(self, path: str) -> str:
        return os.path.join(self.store_dir, *self.norm(path).split("/"))

    def _log(self, op: str, path: str, ok: bool, **extra) -> None:
        with self._lock:
            self.calls.append({"ts": time.time(), "op": op, "path": str(path),
                               "ok": bool(ok), **extra})

    def _entry(self, lib_path: str) -> dict:
        loc = self.local(lib_path)
        name = lib_path.rsplit("/", 1)[-1]
        if os.path.isdir(loc):
            return {"name": name, "type": "folder",
                    "path": f"{self.site_path}/{lib_path}"}
        return {"name": name, "type": "file", "size": os.path.getsize(loc),
                "path": f"{self.site_path}/{lib_path}"}

    # -- the SDK surface ---------------------------------------------------
    def ls(self, path: str) -> list:
        lib = self.norm(path)
        loc = self.local(lib)
        if not os.path.isdir(loc):
            self._log("ls", path, False, reason="missing")
            return []
        out = [self._entry(f"{lib}/{n}" if lib else n)
               for n in sorted(os.listdir(loc))]
        self._log("ls", path, True, n=len(out))
        return out

    def get_folder_details(self, path: str) -> dict:
        loc = self.local(path)
        if os.path.isdir(loc):
            self._log("get_folder_details", path, True)
            return {"id": f"fake:{self.norm(path)}",
                    "folder": {"childCount": len(os.listdir(loc))}}
        self._log("get_folder_details", path, False)
        return {}

    def get_file_details(self, path: str) -> dict:
        loc = self.local(path)
        if os.path.isfile(loc):
            self._log("get_file_details", path, True)
            return {"name": os.path.basename(loc),
                    "size": os.path.getsize(loc)}
        self._log("get_file_details", path, False)
        raise FileNotFoundError(f"File not found: {path}. 404")

    def download_file(self, path: str, local_path: Optional[str] = None,
                      return_bytes: bool = True, overwrite: bool = False):
        loc = self.local(path)
        if not os.path.isfile(loc):
            self._log("download_file", path, False, reason="missing")
            raise FileNotFoundError(
                f"Failed to download file from SharePoint: {path}. Download "
                "failed: 404 - {\"error\": {\"code\": \"itemNotFound\"}}")
        with open(loc, "rb") as fh:
            data = fh.read()
        self._log("download_file", path, True, bytes=len(data),
                  local_path=local_path)
        if local_path:
            if os.path.exists(local_path) and not overwrite:
                raise FileExistsError(local_path)
            os.makedirs(os.path.dirname(os.path.abspath(local_path)),
                        exist_ok=True)
            with open(local_path, "wb") as fh:
                fh.write(data)
            return local_path if not return_bytes else data
        return data

    def upload_file(self, local_path: str, remote_path: str,
                    overwrite: bool = False) -> bool:
        if not os.path.isfile(local_path):
            self._log("upload_file", remote_path, False, reason="no local file",
                      local_path=local_path)
            raise FileNotFoundError(local_path)
        loc = self.local(remote_path)
        if os.path.exists(loc) and not overwrite:
            self._log("upload_file", remote_path, False, reason="exists",
                      local_path=local_path)
            return False
        os.makedirs(os.path.dirname(loc), exist_ok=True)
        shutil.copyfile(local_path, loc)
        self._log("upload_file", remote_path, True, local_path=local_path,
                  bytes=os.path.getsize(loc))
        return True

    def create_folder(self, path: str) -> bool:
        os.makedirs(self.local(path), exist_ok=True)
        self._log("create_folder", path, True)
        return True

    def get_web_url(self, path: str) -> str:
        lib = self.norm(path)
        self._log("get_web_url", path, True)
        return f"{self.site_url}/{quote(lib)}"

    def search_filenames(self, query: str, path: Optional[str] = None) -> list:
        base = self.norm(path) if path else ""
        q = str(query or "").strip().lower()
        hits = []
        loc = self.local(base) if base else self.store_dir
        for dirpath, _dirs, files in os.walk(loc):
            for n in files:
                if q and q in n.lower():
                    rel = os.path.relpath(os.path.join(dirpath, n),
                                          self.store_dir).replace(os.sep, "/")
                    hits.append(self._entry(rel))
        self._log("search_filenames", path or "", True, query=query,
                  n=len(hits))
        return hits

    # -- harness helpers ---------------------------------------------------
    def seed(self, lib_path: str, data: bytes) -> str:
        """Put a file into the library (before a flow starts)."""
        loc = self.local(lib_path)
        os.makedirs(os.path.dirname(loc), exist_ok=True)
        with open(loc, "wb") as fh:
            fh.write(data)
        return self.norm(lib_path)

    def exists(self, path: str) -> bool:
        try:
            return os.path.isfile(self.local(path))
        except ValueError:
            return False

    def size(self, path: str) -> Optional[int]:
        loc = self.local(path)
        return os.path.getsize(loc) if os.path.isfile(loc) else None

    def resolve_url(self, url: str) -> Optional[str]:
        """The library path behind a web URL this fake handed out (``None``
        for another host)."""
        try:
            parts = urlsplit(str(url))
        except ValueError:
            return None
        own = urlsplit(self.site_url)
        if parts.netloc.lower() != own.netloc.lower():
            return None
        try:
            return self.norm(url)
        except ValueError:
            return None

    def listing(self) -> List[dict]:
        """Every file in the library: ``[{"path", "bytes"}]``."""
        out = []
        for dirpath, _dirs, files in os.walk(self.store_dir):
            for n in sorted(files):
                full = os.path.join(dirpath, n)
                out.append({"path": os.path.relpath(full, self.store_dir)
                            .replace(os.sep, "/"),
                            "bytes": os.path.getsize(full)})
        return sorted(out, key=lambda d: d["path"])


__all__ = ["LocalSharePointFM", "SITE_URL", "SITE_PATH", "ROOT"]

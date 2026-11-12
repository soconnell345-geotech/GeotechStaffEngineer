"""A fake ``foundry.transforms``, the workspace dataset library, for the
offline tests of the Foundry dataset store,
:mod:`aec_agent_core.foundry_dataset_store` (BACKLOG INF-38; kept beside
the core's tests since the module moved into the core, INF-74). The core's
tests import it by name; GeotechStaffEngineer's store tests
(``webapp/tests/test_foundry_dataset_store.py``) load it from this folder.

No network, and every alias, path and name in it is made up. The fake keeps
what the AI FDE measured from a published app (2026-11-03):

* ``Dataset.get`` of an alias the workspace does not define fails with
  ``404 FoundryContainerSidecar:AliasNotDefined``;
* ``upload_directory`` refuses a folder outside the outputs folder
  (``IllegalArgumentError``), uploads EVERYTHING under it in one
  transaction, and returns ``{local path: dataset path}``, the dataset path
  being the path relative to the outputs folder;
* ``upload_file`` is a transaction of its own;
* a write while another transaction is open fails with
  ``400 Catalog:SimultaneousOpenTransactionsNotAllowed``;
* ``files()`` lists the whole dataset (no prefix filter, hidden files only
  when asked), and ``File.download()`` returns a local path;
* a later write overwrites a path and keeps every other file;
* a new dataset with no committed transaction answers a read with
  ``400 FoundryContainerService:FailedToResolveDataset`` ("does not have
  any committed transaction", the FDE's build i), and its first write
  makes the first commit.
"""

import os
import sys
import threading
import time
import types
from datetime import datetime, timezone

import aec_agent_core.foundry_dataset_store as fds

#: The one alias the fake workspace defines.
ALIAS = "made_up_conversations"


class IllegalArgumentError(Exception):
    pass


class SidecarError(Exception):
    """The library's refusals carry the status and the error name."""


#: A new dataset's answer to a read, in the library's shape.
NO_COMMIT = ("400 FoundryContainerService:FailedToResolveDataset {'message': "
             "'The dataset does not have any committed transaction on the "
             "requested branch or across fallback branches'}")
#: The same error name for another cause (made-up wording).
NOT_RESOLVED = ("400 FoundryContainerService:FailedToResolveDataset "
                "{'message': 'made-up: the alias resolves to a dataset this "
                "app cannot read'}")


class FakeFile:
    def __init__(self, ds, path, data):
        self.path = path
        self.size_bytes = len(data)
        self.updated_time = datetime(2026, 1, 2, tzinfo=timezone.utc)
        self._ds = ds

    def download(self):
        return self._ds.download(self.path)


class FakeDataset:
    """One dataset's files, with the one-open-transaction rule."""

    def __init__(self, lib):
        self.lib = lib
        self.files_ = {}             # dataset path -> bytes
        self.calls = []
        self.busy = 0                # writes refused before one goes through
        self.open = False
        self.download_delay = 0.0
        self.gate = None             # an Event a write waits on
        self.uncommitted = False     # new: reads refused until a write
        self.get_error = None        # what Dataset.get raises
        self.files_error = None      # what files() raises
        self.files_gate = None       # an Event a listing waits on
        self.active = 0
        self.peak = 0
        self.n = 0
        self._lock = threading.Lock()

    def _begin(self, what):
        self.calls.append((what,))
        if self.busy:
            self.busy -= 1
            raise SidecarError("400 Catalog:SimultaneousOpenTransactionsNotAllowed"
                               " {'openTransactionRid': 'ri.made-up.1'}")
        if self.open:
            raise SidecarError("400 Catalog:SimultaneousOpenTransactionsNotAllowed")
        self.open = True
        if self.gate is not None:
            self.gate.wait(5)

    def upload_file(self, path, logical_path=None):
        self._begin("upload_file")
        try:
            logical = (logical_path or os.path.basename(path)).lstrip("/")
            with open(path, "rb") as fh:
                self.files_[logical] = fh.read()
            self.calls[-1] = ("upload_file", logical)
            self.uncommitted = False             # the first commit
            return logical
        finally:
            self.open = False

    def upload_directory(self, local_dir_path):
        out = os.path.normpath(self.lib.outputs)
        given = os.path.normpath(os.path.abspath(local_dir_path))
        if given != out and not given.startswith(out + os.sep):
            raise IllegalArgumentError(
                f"Provided path {local_dir_path} should be in {out}")
        self._begin("upload_directory")
        try:
            result = {}
            for dirpath, _dirs, names in os.walk(given):
                for name in names:
                    local = os.path.join(dirpath, name)
                    logical = os.path.relpath(local, out).replace(os.sep, "/")
                    with open(local, "rb") as fh:
                        self.files_[logical] = fh.read()
                    result[local] = logical
            self.calls[-1] = ("upload_directory", sorted(result.values()))
            self.uncommitted = False             # the first commit
            return result
        finally:
            self.open = False

    def files(self, mode="current", show_hidden_files=False):
        self.calls.append(("files",))
        if self.files_gate is not None:
            self.files_gate.wait(5)
        if self.files_error:
            raise SidecarError(self.files_error)
        if self.uncommitted:
            raise SidecarError(NO_COMMIT)
        return iter([FakeFile(self, p, b)
                     for p, b in sorted(self.files_.items())
                     if show_hidden_files or not any(
                         s.startswith(("_", ".")) for s in p.split("/"))])

    def download(self, path):
        with self._lock:
            self.calls.append(("download", path))
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.n += 1
            n = self.n
        try:
            if self.download_delay:
                time.sleep(self.download_delay)
            if path not in self.files_:
                raise SidecarError(f"404 FileNotFound {path}")
            dst = os.path.join(self.lib.cache, f"{n}_{path.rsplit('/', 1)[-1]}")
            with open(dst, "wb") as fh:
                fh.write(self.files_[path])
            return dst
        finally:
            with self._lock:
                self.active -= 1

    def count(self, kind):
        return sum(1 for c in self.calls if c[0] == kind)

    def uploads(self):
        return [c for c in self.calls
                if c[0] in ("upload_file", "upload_directory") and len(c) > 1]


class FakeLibrary:
    """``foundry.transforms`` with one dataset under :data:`ALIAS`."""

    def __init__(self, tmp_path):
        self.outputs = str(tmp_path / "foundry" / "outputs")
        self.cache = str(tmp_path / "library_cache")
        os.makedirs(self.cache, exist_ok=True)
        self.ds = FakeDataset(self)
        lib = self

        class Dataset:
            @staticmethod
            def get(alias):
                lib.ds.calls.append(("get", alias))
                if alias != ALIAS:
                    raise SidecarError("404 FoundryContainerSidecar:"
                                       f"AliasNotDefined {{'alias': '{alias}'}}")
                if lib.ds.get_error:
                    raise SidecarError(lib.ds.get_error)
                return lib.ds

        pkg = types.ModuleType("foundry")
        pkg.__path__ = []
        mod = types.ModuleType("foundry.transforms")
        mod.Dataset = Dataset
        pkg.transforms = mod
        self.modules = {"foundry": pkg, "foundry.transforms": mod}

    def install(self, monkeypatch):
        for name, module in self.modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(fds, "OUTPUTS_DIR", self.outputs)
        return self

    def leftovers(self):
        """Every file left in the outputs folder."""
        return sorted(os.path.relpath(os.path.join(d, n), self.outputs)
                      .replace(os.sep, "/")
                      for d, _s, names in os.walk(self.outputs)
                      for n in names)


def write(path, data=b"x"):
    """``data`` written to ``path`` (its folders made); the path back."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path

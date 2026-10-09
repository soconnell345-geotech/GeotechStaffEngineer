"""Snapshots of the places a file should NOT land, taken around every turn.

Detector (a) compares a before and an after snapshot of:

* ``temp``  -- ``tempfile.gettempdir()`` (depth 3; other processes write
  here too, so a new file there is only charged to the turn when the turn's
  own record names it);
* ``cwd``   -- the scenario's working directory (the app's process cwd; a
  relative-path save lands here);
* ``home``  -- the home folder's top level, plus ``~/.geotech_webapp`` (the
  app's default data root: anything new there means a code path ignored
  ``GEOTECH_WEBAPP_DATA``);
* ``c_tmp`` -- ``C:\\tmp``, which is where ``/tmp/...`` lands on Windows;
* ``repo``  -- the source checkout's top level.

The harness's own output and pytest's folders are left out.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

#: File extensions that look like a deliverable (a finding when they turn
#: up outside the conversation folder).
DELIVERABLE_EXT = (".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".png",
                   ".jpg", ".jpeg", ".svg", ".html", ".htm", ".dxf", ".xml",
                   ".diggs", ".json", ".md", ".txt", ".zip", ".pptx", ".tex",
                   ".gef", ".ags")

#: Directory names never walked (noise, or the harness/pytest itself).
SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules", ".pytest_cache",
             "geotech-references", "build", "dist"}
SKIP_PREFIXES = ("pytest-of-", "claude")


def watched_roots(cwd: Optional[str] = None) -> Dict[str, Tuple[str, int]]:
    """``{label: (path, depth)}`` for the turn's outside-folder watch."""
    home = os.path.expanduser("~")
    roots = {
        "temp": (tempfile.gettempdir(), 3),
        "home": (home, 0),
        "home_webapp": (os.path.join(home, ".geotech_webapp"), 6),
        "c_tmp": ("C:\\tmp" if os.name == "nt" else "/tmp", 3),
        "repo": (REPO, 0),
    }
    if cwd:
        roots["cwd"] = (os.path.abspath(cwd), 8)
    return roots


def _excluded(path: str, excludes: Iterable[str]) -> bool:
    ap = os.path.normcase(os.path.abspath(path))
    for e in excludes:
        e = os.path.normcase(os.path.abspath(e))
        if ap == e or ap.startswith(e + os.sep):
            return True
    return False


def snapshot(roots: Dict[str, Tuple[str, int]],
             excludes: Iterable[str] = ()) -> Dict[str, Dict[str, list]]:
    """``{label: {path: [size, mtime_ns]}}`` for every file under each root
    down to its depth. Never raises."""
    excludes = list(excludes)
    out: Dict[str, Dict[str, list]] = {}
    for label, (base, depth) in roots.items():
        files: Dict[str, list] = {}
        if not os.path.isdir(base) or _excluded(base, excludes):
            out[label] = files
            continue
        base_depth = os.path.abspath(base).rstrip(os.sep).count(os.sep)
        try:
            for root, dirs, names in os.walk(base):
                d = os.path.abspath(root).rstrip(os.sep).count(os.sep) \
                    - base_depth
                dirs[:] = [x for x in dirs
                           if x not in SKIP_DIRS
                           and not x.startswith(SKIP_PREFIXES)
                           and not _excluded(os.path.join(root, x), excludes)]
                if d >= depth:
                    dirs[:] = []
                for n in names:
                    p = os.path.join(root, n)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    files[p] = [st.st_size, st.st_mtime_ns]
        except OSError:
            pass
        out[label] = files
    return out


def diff(before: Dict[str, Dict[str, list]],
         after: Dict[str, Dict[str, list]]) -> List[dict]:
    """New or changed files: ``[{"root", "path", "bytes", "change"}]``."""
    out = []
    for label, now in after.items():
        was = before.get(label) or {}
        for p, (size, mtime) in now.items():
            if p not in was:
                out.append({"root": label, "path": p, "bytes": size,
                            "change": "new"})
            elif was[p][1] != mtime and p.lower().endswith(DELIVERABLE_EXT):
                out.append({"root": label, "path": p, "bytes": size,
                            "change": "modified"})
    return sorted(out, key=lambda d: (d["root"], d["path"]))


__all__ = ["watched_roots", "snapshot", "diff", "DELIVERABLE_EXT", "REPO"]

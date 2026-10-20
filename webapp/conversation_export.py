"""Download this conversation: one zip of its folder (BACKLOG INF-38).

On a Foundry deployment conversations live inside the published app's
container (``GEOTECH_WEBAPP_DATA``): they cannot be seen from the workspace
and are lost when the container is rebuilt. The geotech page's sidebar
**Download this conversation** keeps one as a zip of the conversation's
folder, laid out as the folder is, under one top folder named like the zip:
the record (``meta.json``, ``activity.jsonl``, the transcript, the messages
the agent replays, the attachment and download ledgers, any feedback) and
``files/`` (uploads, deliverables, figures). It goes through the app's
normal download (:mod:`webapp.file_widgets`), which on Foundry is made in
the browser.

Left out:

* caches that are rebuilt when needed: the working folder's
  (``files/digest``, ``files/.scratch``; :func:`webapp.core.mirror_skips_dir`)
  and ``__pycache__``;
* files still being written (the turn checkpoint, ``*.part``, ``*.tmp``;
  :func:`webapp.core.is_in_flight_file`);
* the SharePoint mirror's manifest (``sp_manifest.json``): this machine's
  list of what it uploaded, which a restore writes afresh;
* anything that may hold a key: hidden files and folders (``.env``,
  ``.netrc``), key and certificate files (``*.pem``, ``*.key``, ``*.pfx``,
  ``*.p12``) and links. The app keeps no key in a conversation's folder;
  this is for a file a user or a tool put there;
* anything that is not a regular file (a pipe would stop the zip).

The zip is built only when asked for (:func:`build`): on Foundry its bytes
travel inside the page, and putting them there on every rerun is heavy
(INF-44). Over the host's download limit :func:`build` returns no bytes and
says so; ``record_only=True`` then builds the record and the deliverables
that fit (uploads and SharePoint downloads stay out: the user has those),
with a ``LEFT_OUT.txt`` naming what did not go in.

Streamlit-free; the sidebar block is in ``webapp/app.py``.
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
import zlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from webapp import core
from webapp.sharepoint_store import MANIFEST_NAME, sanitize_folder_name

#: The record files a record-only zip takes first, in this order; the
#: conversation folder's other top-level files follow, then deliverables.
RECORD_FIRST = ("meta.json", "activity.jsonl")

#: Key and certificate files, left out by their extension.
KEY_SUFFIXES = (".pem", ".key", ".pfx", ".p12")

#: Folders rebuilt when needed, left out wherever they are (beside the
#: working folder's own caches, ``core.CACHE_DIRS``).
CACHE_FOLDERS = ("__pycache__",)

#: The note a record-only zip carries: what it leaves out, and why.
NOTE_NAME = "LEFT_OUT.txt"

#: Titles that mean the conversation has no name yet (as the mirror's).
_UNTITLED = {"", "new conversation", "untitled"}

#: Bytes a zip adds per entry besides the name, written twice (local header
#: 30 and central directory 46, rounded up), and once at the end (22, rounded
#: up). The record-only zip counts with these before it is built.
_ENTRY_BYTES = 96
_END_BYTES = 64
#: Kept back from a record-only zip's budget for its note.
_NOTE_RESERVE = 16 * 1024
#: The most left-out paths the note names one by one (it then counts the
#: rest), so the note stays inside its reserve.
NOTE_MAX_LINES = 400


class _OverCap(Exception):
    """The zip passed the download limit while it was being written."""


class _CappedBuffer(io.BytesIO):
    """The zip in memory, refusing to grow past ``cap`` bytes: a file larger
    than the limit is never deflated whole into memory only to be dropped."""

    def __init__(self, cap: Optional[int]):
        super().__init__()
        self.cap = cap

    def write(self, b) -> int:
        if self.cap is not None and self.tell() + len(b) > self.cap:
            raise _OverCap()
        return super().write(b)


@dataclass
class Export:
    """One conversation zip, built or refused for size."""
    thread_id: str
    #: The zip's file name, ``<title slug>_<YYYY-MM-DD>.zip``.
    name: str
    #: The zip, or ``None`` when it would be over ``cap``.
    data: Optional[bytes]
    #: Paths in the conversation folder the zip holds ('/' separators), or
    #: would hold when it is refused for size.
    files: List[str] = field(default_factory=list)
    #: ``(path, why)`` for each file or folder (``path/``) left out.
    left_out: List[Tuple[str, str]] = field(default_factory=list)
    #: The size on disk of the files meant to go in.
    folder_bytes: int = 0
    #: The host's download limit in bytes, ``None`` for none.
    cap: Optional[int] = None
    record_only: bool = False

    @property
    def size(self) -> int:
        return len(self.data) if self.data is not None else 0


def zip_name(meta: Optional[dict], thread_id: str,
             when: Optional[float] = None) -> str:
    """``<title slug>_<YYYY-MM-DD>.zip``: the title spelled as the SharePoint
    mirror spells a folder (:func:`webapp.sharepoint_store.
    sanitize_folder_name`) and the day the zip is made, so two downloads on
    two days do not share a name; ``conversation_<first 8 of the thread id>``
    for a conversation with no title yet."""
    title = str((meta or {}).get("title") or "").strip()
    slug = "" if title.lower() in _UNTITLED else sanitize_folder_name(title)
    slug = slug or f"conversation_{str(thread_id)[:8]}"
    return f"{slug}_{time.strftime('%Y-%m-%d', time.localtime(when))}.zip"


def _folder_reason(rel_dir: str) -> Optional[str]:
    name = rel_dir.rsplit("/", 1)[-1]
    if core.mirror_skips_dir(rel_dir) or name in CACHE_FOLDERS:
        return "a cache, rebuilt when needed"
    if name.startswith("."):
        return "a hidden folder (it may hold a key)"
    return None


def _file_reason(rel: str, path: str) -> Optional[str]:
    name = rel.rsplit("/", 1)[-1]
    if rel == MANIFEST_NAME:
        return "this machine's SharePoint mirror list (a restore writes it)"
    if core.is_in_flight_file(rel):
        return "still being written"
    if name.startswith("."):
        return "a hidden file (it may hold a key)"
    if name.lower().endswith(KEY_SUFFIXES):
        return "a key or certificate file"
    if os.path.islink(path):
        return "a link to a file outside the conversation"
    if not os.path.isfile(path):
        return "not a regular file"
    return None


def collect(conv_dir: str) -> Tuple[List[Tuple[str, str]],
                                    List[Tuple[str, str]]]:
    """``(included, left_out)`` for a conversation folder: ``included`` is
    ``[(path in the folder, path on disk)]`` in folder order, ``left_out``
    is ``[(path, why)]``, a folder written with a trailing ``/``."""
    included: List[Tuple[str, str]] = []
    left_out: List[Tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(conv_dir):
        rel_dir = os.path.relpath(dirpath, conv_dir).replace(os.sep, "/")
        kept = []
        for d in sorted(dirnames):
            rel = d if rel_dir == "." else f"{rel_dir}/{d}"
            why = _folder_reason(rel)
            if why or os.path.islink(os.path.join(dirpath, d)):
                left_out.append((f"{rel}/", why or "a link to a folder "
                                 "outside the conversation"))
            else:
                kept.append(d)
        dirnames[:] = kept
        for name in sorted(filenames):
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            path = os.path.join(dirpath, name)
            why = _file_reason(rel, path)
            if why:
                left_out.append((rel, why))
            else:
                included.append((rel, path))
    return included, left_out


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _arcname(top: str, rel: str) -> str:
    """``rel``'s name in the zip, under ``top/``. A backslash in a file name
    (possible on Linux) becomes ``_``: an unzip on Windows would read it as
    a folder separator."""
    return f"{top}/{rel.replace(chr(92), '_')}"


def _write_zip(top: str, entries: List[Tuple[str, str]],
               cap: Optional[int], left_out: List[Tuple[str, str]],
               note: Optional[str] = None
               ) -> Tuple[Optional[bytes], List[str]]:
    """The zip of ``entries`` under ``top/`` and the paths that went in, or
    ``(None, [])`` as soon as it passes ``cap``. A file that cannot be read
    is added to ``left_out``."""
    buf = _CappedBuffer(cap)
    written: List[str] = []
    try:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED,
                             strict_timestamps=False) as zf:
            for rel, path in entries:
                try:
                    zf.write(path, _arcname(top, rel))
                except OSError as exc:
                    left_out.append((rel, f"could not be read "
                                          f"({type(exc).__name__})"))
                    continue
                written.append(rel)
            if note is not None:
                zf.writestr(f"{top}/{NOTE_NAME}", note)
    except _OverCap:
        return None, []
    return buf.getvalue(), written


def _deflated_size(path: str) -> int:
    """The bytes ``path`` takes deflated as the zip deflates it."""
    comp = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            n += len(comp.compress(chunk))
    return n + len(comp.flush())


def _inputs(conv_dir: str) -> dict:
    """``{path in the folder: why}`` for the conversation's inputs: its
    uploads (``attachments.json``) and its SharePoint downloads."""
    out = {}
    try:
        with open(os.path.join(conv_dir, "attachments.json"),
                  encoding="utf-8") as fh:
            keys = json.load(fh)
    except (OSError, ValueError):
        keys = []
    for key in keys if isinstance(keys, list) else []:
        out[f"files/{key}"] = "an upload (you have the original)"
    for entry in core.load_downloads(conv_dir):
        try:
            rel = os.path.relpath(entry["local"], conv_dir)
        except ValueError:          # another drive: not in this folder
            continue
        out[rel.replace(os.sep, "/")] = ("fetched from SharePoint (it is "
                                         "still there)")
    return out


def _record_only(top: str, conv_dir: str,
                 included: List[Tuple[str, str]], cap: Optional[int],
                 left_out: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """The record first (:data:`RECORD_FIRST`, then the folder's other
    top-level files), then the deliverables newest first, each taken while
    the zip stays under ``cap``; inputs and what does not fit go to
    ``left_out``."""
    inputs = _inputs(conv_dir)
    first = {rel: i for i, rel in enumerate(RECORD_FIRST)}
    record = sorted((e for e in included if "/" not in e[0]),
                    key=lambda e: (first.get(e[0], len(first)), e[0]))
    deliverables = []
    for rel, path in included:
        if "/" not in rel:
            continue
        if rel in inputs:
            left_out.append((rel, inputs[rel]))
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        deliverables.append((-mtime, rel, path))
    deliverables.sort()
    budget = (None if cap is None
              else cap - _END_BYTES - _NOTE_RESERVE)
    chosen = []
    for rel, path in record + [(r, p) for _m, r, p in deliverables]:
        if budget is not None:
            try:
                cost = (_deflated_size(path) + _ENTRY_BYTES
                        + 2 * len(_arcname(top, rel).encode("utf-8")))
            except OSError as exc:
                left_out.append((rel, f"could not be read "
                                      f"({type(exc).__name__})"))
                continue
            if cost > budget:
                left_out.append((rel, f"did not fit ({_size(path) / 1048576:.1f}"
                                      " MB)"))
                continue
            budget -= cost
        chosen.append((rel, path))
    return chosen


def _note(meta: dict, cap: Optional[int],
          left_out: List[Tuple[str, str]]) -> str:
    title = str(meta.get("title") or "this conversation")
    limit = (f"the {cap / 1048576:.3g} MB a download in this app can carry"
             if cap is not None else "the download limit")
    lines = [f"This zip holds the record of \"{title}\" and the deliverables "
             f"that fit under {limit}. The whole conversation did not fit.",
             "", "Left out:"]
    lines += [f"  {rel}: {why}" for rel, why in left_out[:NOTE_MAX_LINES]]
    if len(left_out) > NOTE_MAX_LINES:
        lines.append(f"  and {len(left_out) - NOTE_MAX_LINES} more")
    return "\n".join(lines) + "\n"


def build(thread_id: str, *, cap: Optional[int] = None,
          record_only: bool = False, root: Optional[str] = None,
          when: Optional[float] = None) -> Export:
    """Zip the conversation ``thread_id`` (see the module docstring).

    ``cap`` is the host's download limit in bytes (``None``: none). The whole
    folder over it gives an :class:`Export` with ``data=None``;
    ``record_only=True`` gives the record and the deliverables that fit,
    with a ``LEFT_OUT.txt`` naming the rest."""
    conv_dir = core.conversation_dir(thread_id, root)
    meta = core.load_meta(thread_id, root) or {}
    name = zip_name(meta, thread_id, when)
    top = name[:-len(".zip")]
    included, left_out = collect(conv_dir)
    note = None
    if record_only:
        included = _record_only(top, conv_dir, included, cap, left_out)
        note = _note(meta, cap, left_out)
    data, written = _write_zip(top, included, cap, left_out, note)
    return Export(thread_id=str(thread_id), name=name, data=data,
                  files=(written if data is not None
                         else [rel for rel, _p in included]),
                  left_out=left_out,
                  folder_bytes=sum(_size(p) for _r, p in included),
                  cap=cap, record_only=record_only)


__all__ = ["Export", "build", "collect", "zip_name", "RECORD_FIRST",
           "KEY_SUFFIXES", "CACHE_FOLDERS", "NOTE_NAME", "NOTE_MAX_LINES"]

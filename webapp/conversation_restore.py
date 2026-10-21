"""Restore a conversation from its downloaded zip (BACKLOG INF-45).

On a Foundry deployment a container rebuild loses the conversations folder,
and the zip from **Download this conversation** (:mod:`webapp.
conversation_export`) is the user's copy. The sidebar's **Restore a
downloaded conversation** box takes that zip back:

* it is recognised by its own layout: one top folder holding ``meta.json``
  and ``activity.jsonl`` (a record-alone zip whose ``LEFT_OUT.txt`` names
  ``activity.jsonl`` may lack it); any other zip is refused with a plain
  message;
* it unpacks into this person's conversations root (the root the export
  reads), under the thread id in its ``meta.json``, so a restored
  conversation is the same one;
* refused: a path outside the folder, an absolute path, a link (or anything
  not a file or folder), an encrypted entry, more than :func:`max_bytes`
  unpacked, a thread id this person already has (unless ``replace``: the
  copy here then goes to the trash folder, as a delete does), a conversation
  open under another root on this server, a conversation of another page;
* left out, as the export leaves them out: the SharePoint mirror's manifest
  (the next mirror writes it), files still being written (a turn checkpoint
  would come back as a fake interrupted turn), hidden files and key files;
* a record-alone zip (``LEFT_OUT.txt``) restores what it has, and the
  conversation gains a note naming what was left out.

The zip is unpacked into a staging folder beside ``conversations/`` and
moved into place only once all of it is out, so a refusal or a damaged zip
leaves nothing behind. Files keep the times the zip records.

Streamlit-free; the sidebar box is in ``webapp/app.py``.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import stat
import tempfile
import time
import zipfile
import zlib
from dataclasses import dataclass, field
from typing import BinaryIO, Callable, List, Optional, Tuple, Union

from webapp import core
from webapp.conversation_export import KEY_SUFFIXES, NOTE_NAME
from webapp.sharepoint_store import MANIFEST_NAME

#: The most a restore unpacks, in MB, unless :data:`MAX_MB_ENV` says.
DEFAULT_MAX_MB = 200.0
MAX_MB_ENV = "GEOTECH_RESTORE_MAX_MB"

#: The files that mark a conversation zip's top folder.
MARKERS = ("meta.json", "activity.jsonl")

#: A thread id the restore accepts as a folder name (``core.new_thread_id``
#: makes 32 hex digits).
_THREAD_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")

#: The most of ``LEFT_OUT.txt`` read into the conversation's note.
_NOTE_MAX_BYTES = 256 * 1024
#: The most left-out paths the agent's note names.
AGENT_NOTE_MAX_NAMES = 30

_COPY_CHUNK = 1 << 20


def max_bytes() -> int:
    """The unpacked-size cap in bytes (:data:`MAX_MB_ENV`, else
    :data:`DEFAULT_MAX_MB`; a blank or unreadable value is the default)."""
    raw = (os.environ.get(MAX_MB_ENV) or "").strip()
    try:
        mb = float(raw) if raw else DEFAULT_MAX_MB
    except ValueError:
        mb = DEFAULT_MAX_MB
    if mb <= 0:
        mb = DEFAULT_MAX_MB
    return int(mb * 1048576)


class Refused(Exception):
    """The zip is not restored; the message says why, for the user."""


@dataclass
class Restore:
    """What a restore did."""
    #: ``"restored"``, ``"exists"`` (already here, ``replace`` not set) or
    #: ``"refused"``.
    status: str
    #: For the user: why it was refused, or what happened.
    message: str = ""
    thread_id: Optional[str] = None
    title: Optional[str] = None
    #: Paths restored in the conversation folder ('/' separators).
    files: List[str] = field(default_factory=list)
    #: ``(path, why)`` for each file in the zip not restored.
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    #: The zip was a record-alone one (it held ``LEFT_OUT.txt``).
    partial: bool = False
    #: The paths ``LEFT_OUT.txt`` names (a partial zip only).
    left_out: List[str] = field(default_factory=list)
    #: True when a copy already here was replaced (moved to the trash).
    replaced: bool = False

    def agent_note(self) -> Optional[str]:
        """A note for the agent's next turn after a partial restore: the
        files the conversation had that are not in its working folder."""
        if not self.partial or not self.left_out:
            return None
        names = [p for p in self.left_out if not p.endswith("/")]
        if not names:
            return None
        shown = ", ".join(names[:AGENT_NOTE_MAX_NAMES])
        more = len(names) - AGENT_NOTE_MAX_NAMES
        return ("[System note] This conversation was restored from a "
                "downloaded zip that left some files out, so these are not "
                "in its folder: " + shown
                + (f", and {more} more" if more > 0 else "")
                + ". Ask the user for any of them you need.")


# -- reading the zip ---------------------------------------------------------


def _parts(name: str) -> List[str]:
    """The path parts of a zip entry name, refused when the name could land
    outside the folder."""
    if "\\" in name:
        raise Refused(f"It holds a path with a backslash ({name!r}); a "
                      "conversation zip has none.")
    if name.startswith("/") or re.match(r"[A-Za-z]:", name):
        raise Refused(f"It holds an absolute path ({name!r}).")
    parts = name.rstrip("/").split("/")
    if any(p == ".." for p in parts):
        raise Refused(f"It holds a path that leads out of its folder "
                      f"({name!r}).")
    if any(p in ("", ".") for p in parts):
        raise Refused(f"It holds a malformed path ({name!r}).")
    return parts


def _check_entry(info: zipfile.ZipInfo) -> None:
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode):
        raise Refused(f"It holds a link ({info.filename!r}); a conversation "
                      "zip holds files only.")
    if kind and kind not in (stat.S_IFREG, stat.S_IFDIR):
        raise Refused(f"It holds something that is not a file "
                      f"({info.filename!r}).")
    if info.flag_bits & 0x1:
        raise Refused(f"It holds an encrypted file ({info.filename!r}).")


def _skip_reason(rel: str) -> Optional[str]:
    """Why a file of the zip is not restored (the export's own exclusions,
    for a zip made by hand), or ``None``."""
    parts = rel.split("/")
    if rel == MANIFEST_NAME:
        return "this machine's SharePoint mirror list (the next mirror writes it)"
    if rel == NOTE_NAME:
        return "the list of what the zip left out (kept as a note)"
    if core.is_in_flight_file(rel):
        return "a file that was still being written"
    if any(p.startswith(".") for p in parts):
        return "a hidden file (it may hold a key)"
    if parts[-1].lower().endswith(KEY_SUFFIXES):
        return "a key or certificate file"
    return None


def _left_out_paths(note: str) -> List[str]:
    """The paths a ``LEFT_OUT.txt`` names (its ``  path: why`` lines)."""
    out = []
    for line in note.splitlines():
        if line.startswith("  ") and ": " in line:
            out.append(line.strip().rsplit(": ", 1)[0])
    return out


def _read_layout(zf: zipfile.ZipFile, cap: int):
    """``(files, meta, note)`` of a conversation zip: ``files`` is
    ``[(path in the folder, ZipInfo)]``; refused unless the layout is the
    export's."""
    tops = set()
    files = []
    total = 0
    for info in zf.infolist():
        # orig_filename: on Windows zipfile turns a backslash into '/'
        parts = _parts(info.orig_filename)
        _check_entry(info)
        tops.add(parts[0])
        if info.is_dir():
            continue
        if len(parts) < 2:
            raise Refused("It is not a conversation zip: its files are not "
                          "inside one folder.")
        total += info.file_size
        files.append(("/".join(parts[1:]), info))
    if len(tops) != 1:
        raise Refused("It is not a conversation zip: a conversation zip "
                      "holds one folder.")
    names = {rel for rel, _i in files}
    note = None
    if NOTE_NAME in names:
        info = dict(files)[NOTE_NAME]
        with zf.open(info) as fh:
            note = fh.read(_NOTE_MAX_BYTES).decode("utf-8", "replace")
    missing = [m for m in MARKERS if m not in names]
    if missing == ["activity.jsonl"] and note is not None \
            and "activity.jsonl" in _left_out_paths(note):
        missing = []
    if missing:
        raise Refused("It is not a conversation zip from Download this "
                      "conversation: its folder has no "
                      + " or ".join(missing) + ".")
    if total > cap:
        raise Refused(f"It unpacks to {total / 1048576:.1f} MB, over the "
                      f"{cap / 1048576:.3g} MB a restore takes "
                      f"({MAX_MB_ENV}).")
    try:
        with zf.open(dict(files)["meta.json"]) as fh:
            meta = json.loads(fh.read().decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        meta = None
    if not isinstance(meta, dict):
        raise Refused("Its meta.json could not be read.")
    return files, meta, note


# -- unpacking -----------------------------------------------------------------


def _zip_time(info: zipfile.ZipInfo) -> Optional[float]:
    try:
        return time.mktime(tuple(info.date_time) + (0, 0, -1))
    except (OverflowError, ValueError):
        return None


def _unpack(zf: zipfile.ZipFile, files, stage: str, cap: int,
            result: Restore) -> None:
    """Write the files into ``stage``; the size is counted as it is written,
    whatever the zip's headers claim."""
    written = 0
    base = os.path.abspath(stage)
    for rel, info in files:
        why = _skip_reason(rel)
        if why:
            result.skipped.append((rel, why))
            continue
        dest = os.path.abspath(os.path.join(base, *rel.split("/")))
        if os.path.commonpath([base, dest]) != base or dest == base:
            raise Refused(f"It holds a path that leads out of its folder "
                          f"({info.filename!r}).")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with zf.open(info) as src, open(dest, "wb") as out:
            while True:
                chunk = src.read(_COPY_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > cap:
                    raise Refused(f"It unpacks to more than the "
                                  f"{cap / 1048576:.3g} MB a restore takes "
                                  f"({MAX_MB_ENV}).")
                out.write(chunk)
        when = _zip_time(info)
        if when is not None:
            try:
                os.utime(dest, (when, when))
            except OSError:
                pass
        result.files.append(rel)


def _note_entry(zip_name: str, note: str, when: Optional[float]) -> dict:
    """The transcript entry a partial restore adds: what the zip left out."""
    day = time.strftime("%Y-%m-%d", time.localtime(when))
    lines = [f"_Restored on {day} from `{zip_name or 'a downloaded zip'}`, "
             "a partial copy: the whole conversation was over the download "
             "limit, so the zip held the record and the deliverables that "
             "fit. Left out:_", ""]
    for line in note.splitlines():
        if line.startswith("  ") and ": " in line:
            path, why = line.strip().rsplit(": ", 1)
            lines.append(f"- `{path}`: {why}")
        elif line.startswith("  "):
            lines.append(f"- {line.strip()}")
    if len(lines) == 2:
        lines.append("- (the zip's list was empty)")
    return {"role": "assistant", "restored": True, "text": "\n".join(lines)}


def _append_entry(stage: str, entry: dict) -> None:
    path = os.path.join(stage, "transcript.jsonl")
    prefix = ""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell():
                fh.seek(-1, os.SEEK_END)
                prefix = "" if fh.read(1) == b"\n" else "\n"
    except OSError:
        pass
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(prefix + json.dumps(entry, ensure_ascii=False) + "\n")


def _retag(stage: str, meta: dict, owner: Optional[str],
           owner_name: Optional[str]) -> None:
    """Make the restored meta this person's: the owner fields are the ones
    this session tags (none on a single-user host), so the next mirror goes
    to their own folder. Rewritten only when they differ."""
    want = {"owner": owner, "owner_name": owner_name}
    if all(meta.get(k) == v for k, v in want.items()):
        return
    for k, v in want.items():
        if v:
            meta[k] = v
        else:
            meta.pop(k, None)
    with open(os.path.join(stage, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def restore(data: Union[bytes, BinaryIO], *, root: Optional[str] = None,
            replace: bool = False, zip_name: str = "",
            page: Optional[str] = None, owner: Optional[str] = None,
            owner_name: Optional[str] = None,
            busy: Optional[Callable[[str], bool]] = None,
            when: Optional[float] = None) -> Restore:
    """Restore the conversation zip ``data`` (bytes or a binary file) into
    the conversations root ``root`` (see the module docstring). Never raises
    for a bad zip: a refusal is a :class:`Restore` with ``status="refused"``.

    ``page`` is this page's name: a zip whose ``meta.json`` names another
    page is refused. ``owner``/``owner_name`` are what this session tags a
    conversation with (``None`` on a single-user host). ``busy(thread_id)``
    says whether a turn is running in that conversation (it is then never
    replaced)."""
    try:
        return _restore(data, root, replace, zip_name, page, owner,
                        owner_name, busy, when)
    except Refused as exc:
        return Restore(status="refused", message=str(exc))
    except (zipfile.BadZipFile, zlib.error, EOFError, RuntimeError,
            NotImplementedError) as exc:
        return Restore(status="refused",
                       message=f"The zip could not be read "
                               f"({type(exc).__name__}: {exc}).")
    except OSError as exc:
        return Restore(status="refused",
                       message=f"The conversation could not be written "
                               f"({type(exc).__name__}: {exc}).")


def _restore(data, root, replace, zip_name, page, owner, owner_name, busy,
             when) -> Restore:
    cap = max_bytes()
    src = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
    try:
        zf = zipfile.ZipFile(src)
    except zipfile.BadZipFile:
        raise Refused("It is not a zip file.")
    with zf:
        files, meta, note = _read_layout(zf, cap)
        tid = str(meta.get("thread_id") or "").strip()
        if not _THREAD_ID.fullmatch(tid):
            raise Refused("Its meta.json names no usable conversation id.")
        their_page = meta.get("page")
        if page and their_page and their_page != page:
            raise Refused(f"It is a conversation of the {their_page!r} page; "
                          "restore it there.")
        target_root = os.path.abspath(root or core.data_root())
        registered = core.thread_root(tid)
        if registered is not None and os.path.abspath(registered) != \
                target_root:
            raise Refused("That conversation is open under another page or "
                          "account on this server.")
        result = Restore(status="restored", thread_id=tid,
                         title=meta.get("title"), partial=note is not None,
                         left_out=_left_out_paths(note or ""))
        dest = core.conversation_dir(tid, target_root)
        here = os.path.isfile(os.path.join(dest, "meta.json"))
        if here and not replace:
            result.status = "exists"
            result.message = (
                f"'{result.title or tid}' is already here. Tick Replace the "
                "existing copy to put the zip's copy in its place.")
            return result
        if here and busy is not None and busy(tid):
            raise Refused("A turn is running in that conversation; replace "
                          "it once the turn has finished.")
        os.makedirs(target_root, exist_ok=True)
        stage = tempfile.mkdtemp(prefix=".restoring-", dir=target_root)
        try:
            _unpack(zf, files, stage, cap, result)
            _retag(stage, meta, owner, owner_name)
            if note is not None:
                _append_entry(stage, _note_entry(zip_name, note, when))
            if os.path.exists(dest):
                # the copy here goes where a delete puts it (soft)
                core.delete_conversation(tid, target_root)
                result.replaced = here
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.replace(stage, dest)
            stage = None
        finally:
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)
    n = len(result.files)
    # The record says the conversation came back from a zip (BACKLOG EV-19:
    # a restore was invisible in it).
    try:
        from webapp import activity_log
        activity_log.note_restored(dest, source="zip", files=n,
                                   skipped=len(result.skipped),
                                   partial=result.partial,
                                   replaced=result.replaced)
    except Exception:  # noqa: BLE001 - a record, never the restore
        pass
    result.message = (
        f"Restored '{result.title or tid}': {n} {'file' if n == 1 else 'files'}"
        + (", replacing the copy that was here" if result.replaced else "")
        + (". The zip was a partial copy; the conversation lists what it "
           "left out." if result.partial else "."))
    return result


__all__ = ["Restore", "Refused", "restore", "max_bytes", "DEFAULT_MAX_MB",
           "MAX_MB_ENV", "MARKERS", "AGENT_NOTE_MAX_NAMES"]

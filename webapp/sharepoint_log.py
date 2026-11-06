"""SharePoint errors kept in SharePoint (TA-35).

The host's admin cannot read the app's console on a shared host, so every
SharePoint error the app meets -- a tool's failure, a conversation save that
failed, a site a user pointed at that the app cannot open -- and each
connection test's report is appended to a monthly log file in the app's own
SharePoint folder::

    <GEOTECH_SHAREPOINT_ROOT, else the library root>/_app_log/sharepoint-YYYY-MM.log

One line an entry, tab-separated: the time in UTC, the step or tool, the
message in plain words, the path (empty when there is none). A connection
test's report is one entry, its lines joined with `` | ``.

What is never written: a secret, a token, a key, or conversation text. The
messages are the app's own plain words and Graph's error text; anything
that looks like a bearer token, a JWT, a client secret or assertion, or the
configured secret's value itself is replaced by ``[removed]``.

``GEOTECH_SHAREPOINT_LOG=0`` (or ``off`` / ``false`` / ``no``) turns it off;
it is on whenever SharePoint is configured.

It never raises into the caller and never loops: the write runs on its own
thread, a write that fails is logged only to the process log (never back to
SharePoint), and while a write is under way nothing new is sent to it from
that write. The last entries stay in memory (:func:`recent`), so the
sidebar's diagnostics can offer them as a download when SharePoint itself
cannot be reached.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

ENV_SWITCH = "GEOTECH_SHAREPOINT_LOG"
FOLDER = "_app_log"
#: Longest message kept per entry (a connection test's report is longer).
MAX_MESSAGE = 600
MAX_REPORT = 3000

_recent: deque = deque(maxlen=200)
_pending: List[str] = []
_lock = threading.Lock()
_cv = threading.Condition(_lock)
_worker: Optional[threading.Thread] = None
_writing = threading.local()
_busy = False
#: The last write's outcome: None (none yet), True, or the reason it failed.
last_write: Any = None


def enabled() -> bool:
    """On when SharePoint is configured, unless the switch says off."""
    if os.environ.get(ENV_SWITCH, "").strip().lower() in (
            "0", "off", "false", "no"):
        return False
    try:
        from webapp import sharepoint_store
        return sharepoint_store.configured()
    except Exception:                                      # noqa: BLE001
        return False


def log_path(when: Optional[float] = None) -> str:
    """This month's log file in the app's SharePoint folder."""
    root = (os.environ.get("GEOTECH_SHAREPOINT_ROOT", "").strip()
            or "Shared Documents").strip("/")
    month = time.strftime("%Y-%m", time.gmtime(when or time.time()))
    return f"{root}/{FOLDER}/sharepoint-{month}.log"


_SCRUB = [
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*"),
    re.compile(r"(?i)(client_secret|client_assertion|access_token|"
               r"refresh_token|api[-_]?key|password)([\"']?\s*[=:]\s*[\"']?)"
               r"[^\s&\"',}]+"),
    re.compile(r"-----BEGIN [A-Z ]+-----.*?-----END [A-Z ]+-----", re.S),
]


def scrub(text: str) -> str:
    """``text`` with anything secret-shaped replaced by ``[removed]``."""
    out = str(text or "")
    for name in ("GRAPH_CLIENT_SECRET", "GRAPH_CLIENT_KEY_PASSWORD",
                 "GEOTECH_SHAREPOINT_CLIENT_SECRET",
                 "GEOTECH_SHAREPOINT_TOKEN"):
        value = os.environ.get(name, "").strip()
        if len(value) >= 6:
            out = out.replace(value, "[removed]")
    for pattern in _SCRUB:
        out = pattern.sub(lambda m: (m.group(1) + m.group(2) + "[removed]")
                          if m.re.groups >= 2 else "[removed]", out)
    return out


def _clean(text: str, limit: int) -> str:
    one = re.sub(r"\s*\n\s*", " | ", scrub(text)).replace("\t", " ").strip()
    return one if len(one) <= limit else one[:limit - 1] + "…"


def format_entry(step: str, message: str, path: str = "",
                 when: Optional[float] = None, limit: int = MAX_MESSAGE) -> str:
    """One log line: UTC time, step, message, path, tab-separated."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when or time.time()))
    return "\t".join([stamp, _clean(step, 120), _clean(message, limit),
                      _clean(path, 400)])


def record(step: str, message: str, path: str = "", wait: bool = False,
           limit: int = MAX_MESSAGE) -> Optional[str]:
    """Append one entry; returns the line (None when nothing was kept).
    Never raises. ``wait`` blocks until the write is done (tests, the CLI)."""
    try:
        if getattr(_writing, "active", False):
            return None                    # from inside a log write: no loop
        line = format_entry(step, message, path, limit=limit)
        _recent.append(line)
        if not enabled():
            return line
        _queue(line)
        if wait:
            flush(60.0)
        return line
    except Exception:                                      # noqa: BLE001
        return None


def record_check(results: List[Dict[str, Any]], wait: bool = False
                 ) -> Optional[str]:
    """Append a connection test's report as one entry."""
    try:
        from webapp.graph_sharepoint import format_check
        text = format_check(results)
        ok = bool(results) and all(r.get("ok") for r in results)
    except Exception:                                      # noqa: BLE001
        return None
    return record("connection test", ("PASS | " if ok else "FAIL | ") + text,
                  wait=wait, limit=MAX_REPORT)


def recent() -> List[str]:
    """The entries kept in memory this process, oldest first."""
    return list(_recent)


def recent_text() -> str:
    return "\n".join(recent()) + ("\n" if _recent else "")


# ------------------------------------------------------------------ writing

def _queue(line: str) -> None:
    global _worker
    with _cv:
        _pending.append(line)
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, name="sharepoint-log",
                                       daemon=True)
            _worker.start()
        _cv.notify_all()


def _run() -> None:
    global _busy
    while True:
        with _cv:
            if not _pending:
                _cv.notify_all()
                return
            lines = list(_pending)
            _pending.clear()
            _busy = True
        try:
            _write(lines)
        finally:
            with _cv:
                _busy = False
                _cv.notify_all()


def flush(timeout: Optional[float] = None) -> bool:
    """Wait for queued entries to be written; True when none are left."""
    end = None if timeout is None else time.time() + timeout
    with _cv:
        while _pending or _busy:
            left = None if end is None else end - time.time()
            if left is not None and left <= 0:
                return False
            _cv.wait(left if left is not None else 1.0)
    return True


def _missing(exc: BaseException) -> bool:
    status = getattr(exc, "status", None)
    return (status == 404 or isinstance(exc, FileNotFoundError)
            or "itemnotfound" in str(exc).lower())


def _write(lines: List[str]) -> None:
    """Download this month's file, append, upload it back. A failure goes
    to the process log only; the lines are kept in memory either way."""
    global last_write
    _writing.active = True
    tmp = None
    try:
        from webapp import sharepoint_store
        fm = sharepoint_store.get_store().file_manager()
        remote = log_path()
        try:
            old = fm.download_file(remote, return_bytes=True) or b""
            if not isinstance(old, (bytes, bytearray)):
                old = b""
        except Exception as exc:                           # noqa: BLE001
            if not _missing(exc):
                raise                      # never overwrite a log not read
            old = b""
        body = bytes(old)
        if body and not body.endswith(b"\n"):
            body += b"\n"
        body += ("\n".join(lines) + "\n").encode("utf-8")
        fd, tmp = tempfile.mkstemp(prefix="sp_log_", suffix=".log")
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
        ok = fm.upload_file(tmp, remote, overwrite=True)
        if ok is False:
            raise RuntimeError("the upload was refused")
        last_write = True
    except Exception as exc:                               # noqa: BLE001
        last_write = f"{type(exc).__name__}: {scrub(str(exc))[:300]}"
        log.warning("The SharePoint log could not be written (%s); %d "
                    "entr(ies) kept in memory only.", last_write, len(lines))
    finally:
        _writing.active = False
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass


def reset() -> None:
    """Forget what is in memory (tests)."""
    global last_write
    flush(10.0)
    _recent.clear()
    last_write = None


__all__ = ["record", "record_check", "recent", "recent_text", "flush",
           "enabled", "log_path", "format_entry", "scrub", "ENV_SWITCH",
           "FOLDER", "reset"]

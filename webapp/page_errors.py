"""A page rerun that raised, kept in the open conversation's record.

Field session 2026-10-15 (BACKLOG EV-19): a rerun of the page crashed on a
repeated download-button key (TA-29), and the traceback reached the record
only because the owner pasted it. ``activity.jsonl`` logs turns; a page
rerun is not a turn.

A ``page_error`` record (``webapp.activity_log.append_event``) goes into
the log of the conversation the rerun had open (:func:`bind`, called by
``webapp/app.py`` on every rerun): the error, the frame it came from and the
end of the traceback, with paths reduced to file names (one glued to a
widget key's text too), web addresses to their host, email addresses
dropped and anything shaped like a key masked. Two routes reach it:

* :func:`recording`, around the run of ``webapp/app.py`` in the entry
  scripts (``webapp.pages_entry``, ``webapp.foundry_entry``): an exception
  escaping the page is recorded, then raised on unchanged for Streamlit to
  show. Streamlit's own stop and rerun signals are not exceptions of this
  kind and pass untouched.
* :class:`PageErrorRecorder`, a handler on Streamlit's
  ``streamlit.error_util`` logger, for ``streamlit run webapp/app.py`` on
  its own: Streamlit logs "Uncaught app execution" there when it does not
  print the traceback with ``rich``.

An exception is recorded once whichever route sees it. Nothing here changes
what the page does, and a failure to record is swallowed. Streamlit-free
(plain :mod:`logging` and a context manager).
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import threading
import traceback
from typing import Optional

#: Streamlit's logger for an exception that escapes the page script.
LOGGER_NAME = "streamlit.error_util"

#: Most characters of a traceback kept (its end, where the error is).
TRACEBACK_CHARS = 4000

#: Most characters of the error's own text kept.
ERROR_CHARS = 1000

_BOUND = threading.local()
_INSTALL_LOCK = threading.Lock()


def bind(conv_dir: Optional[str]) -> None:
    """The conversation folder this thread's page run has open (Streamlit
    runs a session's page script on its own thread)."""
    _BOUND.conv_dir = conv_dir


def bound() -> Optional[str]:
    return getattr(_BOUND, "conv_dir", None)


#: A Windows path glued to a word, as in a widget key built from a file's
#: path (``dlcard_C:\\...``), which ``scrub_paths`` leaves alone.
_GLUED_WIN = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/][^\s'\"<>|()\[\]{}]+")

#: The same on the Linux hosts (``dlcard_/Workspace/Users/<email>/...``,
#: ``dbfs:/...``): three or more parts, so a unit such as ``kN/m/m`` stays.
_GLUED_POSIX = re.compile(
    r"(?<=[\w=:\-])/(?!/)(?:[^\s/'\"<>|()\[\]{}]+/){2,}"
    r"[^\s/'\"<>|()\[\]{}]*")

#: A web address (a SharePoint link names the site and the project's
#: folders): kept as its scheme and host.
_URL = re.compile(r"\b[A-Za-z][A-Za-z0-9+.\-]*://[^\s'\"<>]+")

#: An email address (a user's own, in a host path or a service's message).
_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")


def _name(path: str) -> str:
    parts = [p for p in re.split(r"[\\/]+", path) if p]
    return parts[-1] if parts else ""


def _host(url: str) -> str:
    scheme, _, rest = url.partition("://")
    host = re.split(r"[/?#]", rest, maxsplit=1)[0].rsplit("@", 1)[-1]
    return f"{scheme}://{host}" + ("/…" if len(host) < len(rest) else "")


def _clean(text: str) -> str:
    try:
        from funhouse_agent.error_text import mask_secrets, scrub_paths
        text = _URL.sub(lambda m: _host(m.group(0)), text)
        text = scrub_paths(text)
        text = _GLUED_WIN.sub(lambda m: _name(m.group(0)), text)
        text = _GLUED_POSIX.sub(lambda m: _name(m.group(0)), text)
        text = _EMAIL.sub("(an email address)", text)
        return mask_secrets(text)
    except Exception:  # noqa: BLE001 - never a raw path or key
        return ""


def error_facts(exc: BaseException) -> dict:
    """``error`` (type and message), ``where`` (the last frame: file name,
    line and function) and ``traceback`` (its last
    :data:`TRACEBACK_CHARS` characters), paths and keys scrubbed."""
    out = {"error": _clean(f"{type(exc).__name__}: {exc}")[:ERROR_CHARS]}
    try:
        frames = traceback.extract_tb(exc.__traceback__)
        if frames:
            f = frames[-1]
            out["where"] = (f"{os.path.basename(f.filename)}:{f.lineno} "
                            f"in {f.name}")
        tb = "".join(traceback.format_exception(type(exc), exc,
                                                exc.__traceback__))
        out["traceback"] = _clean(tb)[-TRACEBACK_CHARS:]
    except Exception:  # noqa: BLE001
        pass
    return out


#: Set on an exception once it is on the record.
_MARK = "_geotech_page_error_recorded"


def record(exc: Optional[BaseException],
           conv_dir: Optional[str] = None) -> bool:
    """Write a ``page_error`` record for ``exc`` into ``conv_dir`` (default:
    the conversation bound to this thread), once per exception. Only into a
    conversation that has a folder: a page that fails before its first turn
    leaves no folder behind. ``True`` when written; never raises."""
    try:
        conv = conv_dir or bound()
        if (not isinstance(exc, Exception) or not conv
                or not os.path.isdir(conv) or getattr(exc, _MARK, False)):
            return False
        try:
            setattr(exc, _MARK, True)
        except Exception:  # noqa: BLE001 - an exception that refuses it
            pass
        from webapp import activity_log
        return activity_log.append_event(conv, "page_error",
                                         **error_facts(exc))
    except Exception:  # noqa: BLE001 - a record, never the page
        return False


@contextlib.contextmanager
def recording():
    """Around one run of the page script: an exception escaping it is
    recorded (:func:`record`), then raised on unchanged. The binding is
    cleared first, so a run that fails before ``webapp/app.py`` binds its
    conversation is not filed under the previous run's."""
    bind(None)
    try:
        yield
    except Exception as exc:
        record(exc)
        raise


class PageErrorRecorder(logging.Handler):
    """Writes a ``page_error`` record for each exception Streamlit logs as
    escaping the page script (see the module docstring)."""

    def emit(self, rec: logging.LogRecord) -> None:
        try:
            record(rec.exc_info[1] if rec.exc_info else None)
        except Exception:  # noqa: BLE001 - a record, never the page
            pass


def install() -> None:
    """Attach the recorder to Streamlit's logger, once per process."""
    with _INSTALL_LOCK:
        logger = logging.getLogger(LOGGER_NAME)
        if not any(isinstance(h, PageErrorRecorder) for h in logger.handlers):
            logger.addHandler(PageErrorRecorder(level=logging.ERROR))


__all__ = ["bind", "bound", "install", "record", "recording", "error_facts",
           "PageErrorRecorder", "LOGGER_NAME", "TRACEBACK_CHARS"]

"""Error text fit to show: no server paths, and a tool error the turn survives.

Live smoke wave 2b (C3): ``annotate_document`` raised MuPDF's
``FzErrorSystem: cannot remove file 'C:\\...\\users\\livesmoke__tester\\...
\\review_set_marked.pdf'``. Nothing between the tool and the graph caught it,
so the whole turn ended on "(no answer text)" and the raw text -- the server
path with the tester's folder in it -- was shown to the tester. On Tiny Apps
the same text would show ``/home/data/geotech_webapp/users/<domain>__<user>/``.

* :func:`scrub_paths` takes every absolute path out of a text: the host's
  working folder becomes the file's conversation-relative name
  (``_fileio.hide_working_folder``), and any other absolute path -- Windows,
  UNC or POSIX, quoted or not -- becomes its file name. URLs are left alone.
* :func:`tool_error` is the JSON a tool returns instead of raising: what
  failed, said plainly, and that the turn goes on.

Pure Python, no third-party import: the document tools, the agent's tool
middleware and the web app all use it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

#: A quoted absolute path (Windows drive, UNC or POSIX): its quotes kept.
_QUOTED = re.compile(
    r"""(?P<q>['"])(?P<p>(?:[A-Za-z]:[\\/]|\\\\|/)[^'"\n]*?)(?P=q)""")
#: An unquoted Windows or UNC path (up to whitespace or a closing bracket).
_WIN = re.compile(r"(?<![\w/\\])(?:[A-Za-z]:[\\/]|\\\\)[^\s'\"<>|()\[\]{}]+")
#: An unquoted POSIX path of two or more parts, not part of a URL
#: ("https://h/x" has a ':' or '/' before each of its slashes).
_POSIX = re.compile(
    r"(?<![\w:/.~\\])/(?:[^\s/'\"<>|()\[\]{}]+/)+[^\s/'\"<>|()\[\]{}]*")


def _name_of(path: str) -> str:
    """The last part of ``path``; a folder's own name for a trailing slash."""
    parts = [p for p in re.split(r"[\\/]+", path.strip()) if p]
    return parts[-1] if parts else "(a folder)"


def scrub_paths(text: Any) -> str:
    """``text`` with every absolute path replaced by its file name.

    The host's working folder goes first (a path inside it keeps its
    conversation-relative name, e.g. ``figs/x.png``); then any other
    absolute path, quoted or not, becomes its last part. Never raises."""
    s = "" if text is None else str(text)
    if not s:
        return s
    try:
        from funhouse_agent._fileio import hide_working_folder
        hidden = hide_working_folder(s)
        if isinstance(hidden, str):
            s = hidden
    except Exception:  # noqa: BLE001 - a scrub never fails the caller
        pass
    try:
        s = _QUOTED.sub(lambda m: f"{m.group('q')}{_name_of(m.group('p'))}"
                                  f"{m.group('q')}", s)
        s = _WIN.sub(lambda m: _name_of(m.group(0)), s)
        s = _POSIX.sub(lambda m: _name_of(m.group(0)), s)
    except Exception:  # noqa: BLE001
        pass
    return s


def error_line(exc: BaseException, limit: int = 300) -> str:
    """``"<Type>: <message>"`` on one line, paths scrubbed, at most
    ``limit`` characters of message."""
    msg = " ".join(scrub_paths(str(exc)).split())
    if len(msg) > limit:
        msg = msg[:limit] + " …"
    name = type(exc).__name__
    return f"{name}: {msg}" if msg else name


def tool_error(tool: str, exc: BaseException,
               hint: Optional[str] = None) -> Dict[str, Any]:
    """The result a tool gives instead of raising ``exc``: what failed (no
    server path in it), and that the conversation goes on."""
    out: Dict[str, Any] = {
        "error": f"{tool or 'the tool'} failed: {error_line(exc)}",
        "hint": hint or (
            "The tool raised an error instead of answering; nothing it was "
            "doing was finished. Try it again once if the cause looks "
            "passing, or another way; tell the user plainly what could not "
            "be done rather than stopping."),
    }
    return out


def tool_error_json(tool: str, exc: BaseException,
                    hint: Optional[str] = None) -> str:
    return json.dumps(tool_error(tool, exc, hint), ensure_ascii=False)


__all__ = ["scrub_paths", "error_line", "tool_error", "tool_error_json"]

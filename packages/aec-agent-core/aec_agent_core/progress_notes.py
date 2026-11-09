"""A line a long-running tool leaves for the app's "Working..." label.

While a turn is quiet the app shows "Working... 3m 20s (long step)" (the
heartbeat, :func:`webapp.core.with_heartbeat`). A tool that runs for minutes
(the plotting box, :mod:`funhouse_agent.plot_box`) leaves a note saying what
it is doing, so the person sees "still drawing" rather than a frozen screen.

Notes are keyed by the conversation's working folder (the folder the turn
binds, :func:`funhouse_agent._fileio.host_output_dir`), so on a shared host
nobody sees another person's note. With no folder bound the key is "".
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

_LOCK = threading.Lock()
_NOTES: Dict[str, str] = {}


def current_key() -> str:
    """The key of the conversation running in this context."""
    try:
        from funhouse_agent._fileio import host_output_dir
        return host_output_dir() or ""
    except Exception:                                      # noqa: BLE001
        return ""


def set_note(text: str, key: Optional[str] = None) -> None:
    with _LOCK:
        _NOTES[current_key() if key is None else key] = str(text)


def clear_note(key: Optional[str] = None) -> None:
    with _LOCK:
        _NOTES.pop(current_key() if key is None else key, None)


def note_for(key: str) -> Optional[str]:
    with _LOCK:
        return _NOTES.get(key)


__all__ = ["current_key", "set_note", "clear_note", "note_for"]

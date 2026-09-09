"""Websocket-based file uploader — the driver-proxy-safe attach path.

Streamlit's built-in ``st.file_uploader`` sends the file with an HTTP PUT to
``/_stcore/upload_file/...``. Behind the Databricks driver proxy that PUT
comes back 403 (live-observed since 5.10.0; the proxy is suspected of
rejecting the method — POSTs demonstrably pass). This component sidesteps
HTTP entirely: a tiny custom component reads the file in the browser and
sends the bytes base64-encoded OVER THE ALREADY-OPEN WEBSOCKET as its
component value — the same channel every button click uses, which the proxy
demonstrably forwards.

Trade-offs vs the native uploader: base64 inflates the payload ~33%, and the
whole file rides one websocket message — so uploads are capped (default
25 MB per file) to stay well inside Streamlit's message limits. Fine for
reports, boring logs, DXF/PDF sections; not for point clouds.

THE ONE-SHOT RULE (5.11.3). A component value is WIDGET STATE, and Streamlit's
browser client re-sends **every** widget state on **every** rerun and on every
reconnect — ``createWidgetStatesMsg()`` pushes the whole map with no delta, and
``AppSession`` reads ``client_state.widget_states`` off each rerun BackMsg. So
a file left sitting in this component's value is re-uploaded forever: a 22 MB
attachment (~29 MB base64) cannot finish crossing the driver proxy before the
socket is torn down, the client reconnects, re-sends the same 29 MB, and dies
again — an unbreakable ~14 s reconnect loop (live-observed 2026-09-09 on
5.11.2). The caller MUST therefore retire the widget key once the bytes are
staged (``next_upload_key``), which drops the payload from the client's state
map (``WidgetStateManager.removeInactive``) as well as the server's.

Selection: ``GEOTECH_UPLOAD_MODE`` env — ``http`` (default; native
uploader) or ``ws`` (this component). The Databricks launcher bootstrap
sets ``ws`` because the native path 403s there anyway.
"""

from __future__ import annotations

import base64
import binascii
import os
from typing import List, Optional, Tuple

_COMPONENT_DIR = os.path.join(os.path.dirname(__file__), "ws_upload_component")

MAX_FILE_MB = 25          # per-file cap enforced browser-side AND re-checked here

_component_func = None


def upload_mode() -> str:
    """Return the active upload mode: 'http' (native) or 'ws' (component)."""
    mode = (os.environ.get("GEOTECH_UPLOAD_MODE") or "http").strip().lower()
    return mode if mode in ("http", "ws") else "http"


def _get_component():
    global _component_func
    if _component_func is None:
        import streamlit.components.v1 as components
        _component_func = components.declare_component(
            "geotech_ws_upload", path=_COMPONENT_DIR)
    return _component_func


def decode_component_value(value) -> Tuple[List[Tuple[str, bytes]], List[str]]:
    """Turn the component's JSON value into (name, bytes) pairs + error list.

    The component sends ``[{"name": str, "b64": str, "size": int}, ...]``.
    Anything malformed is reported, never raised — an upload widget must not
    crash a render.
    """
    pairs: List[Tuple[str, bytes]] = []
    errors: List[str] = []
    if not value:
        return pairs, errors
    if not isinstance(value, list):
        return pairs, [f"unexpected component value type {type(value).__name__}"]
    for item in value:
        try:
            name = str(item["name"])
            raw = base64.b64decode(item["b64"], validate=True)
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            errors.append(f"undecodable upload entry: {type(exc).__name__}: {exc}")
            continue
        if len(raw) > MAX_FILE_MB * 1024 * 1024:
            errors.append(f"{name}: exceeds the {MAX_FILE_MB} MB websocket-upload cap")
            continue
        if not raw:
            errors.append(f"{name}: empty file")
            continue
        pairs.append((name, raw))
    return pairs, errors


def next_upload_key(thread_id: str, epoch: int) -> str:
    """Widget key for the uploader on ``thread_id`` at staging generation ``epoch``.

    Bumping ``epoch`` after a successful stage gives the component a NEW widget
    id, which is the only way to get the previous file's bytes out of the
    browser's widget-state map — see THE ONE-SHOT RULE in the module docstring.
    """
    return f"ws_uploader_{thread_id}_{int(epoch)}"


def ws_file_uploader(accepted_types: Optional[List[str]] = None,
                     key: str = "ws_upload") -> Tuple[List[Tuple[str, bytes]], List[str]]:
    """Render the websocket uploader; return staged (name, bytes) + errors.

    Mirrors the shape the app builds from ``st.file_uploader`` so the
    downstream staging code is identical for both modes.
    """
    accept = ",".join(f".{t.lstrip('.')}" for t in (accepted_types or []))
    value = _get_component()(accept=accept, max_mb=MAX_FILE_MB,
                             key=key, default=None)
    return decode_component_value(value)

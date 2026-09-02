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

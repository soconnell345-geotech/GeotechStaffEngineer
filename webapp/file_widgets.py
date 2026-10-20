"""Download buttons and the PDF preview, wherever the app runs (INF-37).

Streamlit's ``st.download_button`` has the browser fetch the file from the
app (``GET /media/<id>``, served as an attachment). Inside the Foundry app
frame that request comes back as an HTML error page: Chrome saves
``<id>.htm`` and says "File wasn't available on site". Images from the same
route display, and a file the page builds in the browser saves: the owner
checked PDF, PNG, HTML and CSV that way in the published app (2026-10-20).
So on a Foundry deployment (``GEOTECH_DEPLOYMENT=foundry``, set by
:mod:`webapp.foundry_entry`) :func:`download_button` draws a small component
with two buttons: the download itself (the bytes, inlined as base64, become
a Blob saved under the file's own name) and **Open in new tab** (the same
Blob in the browser's own viewer). The bytes travel inside the page, so a
file over :func:`download_cap_bytes` gets a message instead. Everywhere else
it is ``st.download_button``, as before. Every download in ``webapp`` goes
through here (``webapp/tests/test_file_widgets.py`` checks).

The inline PDF preview is a ``data:application/pdf`` frame, which the
Foundry frame blocks as well ("This content is blocked"; a ``data:`` HTML
page of page images was blocked too). :func:`pdf_page_preview` shows the
first pages as images instead, which the frame displays.
"""

from __future__ import annotations

import base64
import html
import json
import os
from typing import Optional, Union

import streamlit as st

from webapp import core, engine_config

#: Largest file (MB) the Foundry download component carries in the page.
DOWNLOAD_MAX_MB_ENV = "GEOTECH_FOUNDRY_DOWNLOAD_MAX_MB"
DEFAULT_DOWNLOAD_MAX_MB = 20.0

#: Height (px) of the component: one row of buttons.
_COMPONENT_HEIGHT = 44


def download_cap_bytes() -> int:
    """The Foundry download component's size limit in bytes."""
    try:
        mb = float(os.environ.get(DOWNLOAD_MAX_MB_ENV) or
                   DEFAULT_DOWNLOAD_MAX_MB)
    except ValueError:
        mb = DEFAULT_DOWNLOAD_MAX_MB
    return int(max(0.0, mb) * 1024 * 1024)


def _js_string(value: str) -> str:
    """A JavaScript string literal that cannot end the script element."""
    return json.dumps(value).replace("</", "<\\/")


def browser_download_html(data: bytes, file_name: str, mime: Optional[str],
                          label: str) -> str:
    """The component page: a button saving ``data`` as ``file_name`` from a
    Blob made in the browser, and one opening it in a new tab."""
    b64 = base64.b64encode(bytes(data)).decode("ascii")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
:root {{ color-scheme: light dark; }}
body {{ margin: 0; background: transparent;
  font: 14px "Source Sans Pro", system-ui, sans-serif; }}
.row {{ display: flex; gap: 8px; align-items: center; flex-wrap: nowrap;
  padding: 2px 1px; }}
button {{ font: inherit; padding: 4px 12px; border-radius: 8px;
  border: 1px solid rgba(128, 128, 128, 0.45); background: #fff;
  color: #31333f; cursor: pointer; white-space: nowrap; }}
button:hover {{ border-color: #ff4b4b; color: #ff4b4b; }}
#save {{ min-width: 0; overflow: hidden; text-overflow: ellipsis; }}
#note {{ color: #808495; font-size: 12px; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }}
</style></head><body><div class="row">
<button id="save" type="button" title="Save {html.escape(file_name)}">{html.escape(label)}</button>
<button id="open" type="button" title="Opens in the browser's own viewer; the first button keeps the file's name">Open in new tab</button>
<span id="note"></span></div>
<script>
const B64 = {_js_string(b64)};
const NAME = {_js_string(file_name)};
const MIME = {_js_string(mime or "application/octet-stream")};
let url = null;
function blobUrl() {{
  if (!url) {{
    const s = atob(B64);
    const u = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    url = URL.createObjectURL(new Blob([u], {{type: MIME}}));
  }}
  return url;
}}
function say(text) {{ document.getElementById("note").textContent = text; }}
document.getElementById("save").onclick = function () {{
  try {{
    const a = document.createElement("a");
    a.href = blobUrl(); a.download = NAME;
    document.body.appendChild(a); a.click(); a.remove();
  }} catch (e) {{ say("Download failed: " + e); }}
}};
document.getElementById("open").onclick = function () {{
  try {{
    if (!window.open(blobUrl(), "_blank"))
      say("The browser blocked the new tab: allow pop-ups for this app.");
  }} catch (e) {{ say("Could not open it: " + e); }}
}};
</script></body></html>"""


def _component(page: str, height: int) -> None:
    """Draw ``page`` in a component frame (scripts run in it)."""
    try:
        import streamlit.components.v1 as components
        components.html(page, height=height)
    except (ImportError, AttributeError):        # a Streamlit without it
        st.iframe(page, height=height)


def download_button(label: str, data: Union[bytes, str], *, file_name: str,
                    mime: Optional[str] = None,
                    key: Optional[str] = None) -> None:
    """A download for ``data`` saved as ``file_name``: Streamlit's own button,
    or on Foundry the in-browser component (see the module docstring)."""
    if not engine_config.is_foundry_deployment():
        st.download_button(label, data=data, file_name=file_name, mime=mime,
                           key=key)
        return
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    cap = download_cap_bytes()
    if len(raw) > cap:
        st.caption(
            f"{file_name} is {len(raw) / 1048576:.1f} MB, over the "
            f"{cap / 1048576:g} MB that a download in the Foundry app can "
            f"carry (the file travels inside the page). Ask for a smaller "
            f"file, or raise {DOWNLOAD_MAX_MB_ENV} in the app file.")
        return
    _component(browser_download_html(raw, file_name, mime, label),
               _COMPONENT_HEIGHT)


def pdf_page_preview(path: str) -> None:
    """The first pages of the PDF at ``path`` as images (Foundry), with a
    note on how to see the whole file."""
    pngs, count = core.pdf_page_pngs(path)
    if not pngs:
        st.caption("No preview: the PDF could not be read. Download it "
                   "above.")
        return
    for number, png in enumerate(pngs, 1):
        st.image(png, caption=f"Page {number} of {count}")
    shown = (f"Showing the first {len(pngs)} of {count} pages. "
             if count > len(pngs) else "")
    try:
        whole = os.path.getsize(path) <= download_cap_bytes()
    except OSError:
        whole = False
    st.caption(shown + ("Use Open in new tab above to read the whole file in "
                        "the browser's PDF viewer." if whole else
                        "The file is too large to open here."))

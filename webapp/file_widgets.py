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

Since INF-44 on Foundry: a file over :func:`download_inline_bytes` (2 MB)
gets a **Prepare download** button first, because every rerun draws every
card and the sidebar again, and each carries the bytes. Pressed, the bytes
go into the page until the conversation moves on (the app calls
:func:`forget_prepared`), as the conversation zip does. A file given by its
path (:func:`download_file`) is not read until then. **Open in new tab**
shows an HTML or SVG file inside a sandboxed frame (:data:`HTML_SANDBOX`,
no ``allow-same-origin``), so a page the agent wrote cannot act as the app:
a Blob opened directly carries the app's origin. The HTML preview on a card
is framed the same way (:func:`sandboxed_html_page`) on every host, since
Streamlit's own frames allow the app's origin.

The inline PDF preview is a ``data:application/pdf`` frame, which the
Foundry frame blocks as well ("This content is blocked"; a ``data:`` HTML
page of page images was blocked too). :func:`pdf_page_preview` shows the
first pages as images instead, which the frame displays, when its switch is
turned on.
"""

from __future__ import annotations

import base64
import html
import json
import os
from typing import Callable, Optional, Union

import streamlit as st

from webapp import core, engine_config

#: Largest file (MB) the Foundry download component carries in the page.
DOWNLOAD_MAX_MB_ENV = "GEOTECH_FOUNDRY_DOWNLOAD_MAX_MB"
DEFAULT_DOWNLOAD_MAX_MB = 20.0

#: Largest file (MB) put into the page at once on Foundry; a larger one
#: waits for its Prepare download button (INF-44).
DOWNLOAD_INLINE_MB_ENV = "GEOTECH_FOUNDRY_DOWNLOAD_INLINE_MB"
DEFAULT_DOWNLOAD_INLINE_MB = 2.0

#: What an HTML page the app shows (a card's preview, Open in new tab) may
#: do in its frame: run its scripts, open links, print, save. Never
#: ``allow-same-origin``: the page gets an origin of its own and cannot
#: reach the app, its session or its other pages.
HTML_SANDBOX = ("allow-scripts allow-popups allow-popups-to-escape-sandbox "
                "allow-modals allow-downloads")

#: File types a browser runs scripts in when it opens them in a tab.
ACTIVE_MIMES = ("text/html", "application/xhtml+xml", "image/svg+xml")

#: Session-state key: the downloads prepared since the conversation last
#: moved on.
_PREPARED = "file_widgets_prepared"

#: Height (px) of the component: one row of buttons.
_COMPONENT_HEIGHT = 44


def _mb_setting(env: str, default: float) -> int:
    try:
        mb = float(os.environ.get(env) or default)
    except ValueError:
        mb = default
    return int(max(0.0, mb) * 1024 * 1024)


def download_cap_bytes() -> int:
    """The Foundry download component's size limit in bytes."""
    return _mb_setting(DOWNLOAD_MAX_MB_ENV, DEFAULT_DOWNLOAD_MAX_MB)


def download_inline_bytes() -> int:
    """The largest file (bytes) the Foundry component carries without a
    Prepare download press first."""
    return _mb_setting(DOWNLOAD_INLINE_MB_ENV, DEFAULT_DOWNLOAD_INLINE_MB)


def forget_prepared() -> None:
    """Drop every prepared download: the app calls this when the
    conversation moves on (a turn, another conversation)."""
    st.session_state.pop(_PREPARED, None)


def sandboxed_html_page(text: str) -> str:
    """A page showing the HTML ``text`` in a frame sandboxed by
    :data:`HTML_SANDBOX`: its scripts run, with an origin of their own."""
    return ('<!doctype html><html><head><meta charset="utf-8"><style>'
            "html, body { margin: 0; height: 100%; } "
            "iframe { display: block; border: 0; width: 100%; height: 100%; }"
            f'</style></head><body><iframe sandbox="{HTML_SANDBOX}" '
            f'srcdoc="{html.escape(text, quote=True)}"></iframe>'
            "</body></html>")


def download_limit_bytes() -> Optional[int]:
    """The largest file :func:`download_button` can carry on this host: the
    Foundry component's limit, or ``None`` where Streamlit's own button
    serves the file."""
    if engine_config.is_foundry_deployment():
        return download_cap_bytes()
    return None


def _js_string(value: str) -> str:
    """A JavaScript string literal that cannot end the script element or
    change how the browser reads it: ``<``, ``>`` and ``&`` are written as
    ``\\u`` escapes, so neither ``</script>`` nor ``<!--<script>`` (which
    would keep the element open past its end tag) can appear."""
    return (json.dumps(value).replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("&", "\\u0026"))


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
const SANDBOX = {_js_string(HTML_SANDBOX)};
const ACTIVE = JSON.parse({_js_string(json.dumps(list(ACTIVE_MIMES)))});
let raw = null, url = null, framed = null;
function bytes() {{
  if (!raw) {{
    const s = atob(B64);
    raw = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) raw[i] = s.charCodeAt(i);
  }}
  return raw;
}}
function blobUrl() {{
  if (!url) url = URL.createObjectURL(new Blob([bytes()], {{type: MIME}}));
  return url;
}}
// An HTML or SVG file opened as a Blob would run its scripts with the app's
// origin: it opens inside a sandboxed frame instead (no allow-same-origin).
function openUrl() {{
  if (!ACTIVE.includes(MIME.split(";")[0].trim().toLowerCase()))
    return blobUrl();
  if (!framed) {{
    const d = document.implementation.createHTMLDocument(NAME);
    const meta = d.createElement("meta");
    meta.setAttribute("charset", "utf-8");
    d.head.prepend(meta);
    const style = d.createElement("style");
    style.textContent = "html, body {{ margin: 0; height: 100%; }} " +
      "iframe {{ display: block; border: 0; width: 100%; height: 100%; }}";
    d.head.appendChild(style);
    const frame = d.createElement("iframe");
    frame.setAttribute("sandbox", SANDBOX);
    frame.setAttribute("srcdoc", new TextDecoder("utf-8").decode(bytes()));
    d.body.appendChild(frame);
    framed = URL.createObjectURL(new Blob(
      ["<!doctype html>" + d.documentElement.outerHTML],
      {{type: "text/html;charset=utf-8"}}));
  }}
  return framed;
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
    if (!window.open(openUrl(), "_blank"))
      say("The browser blocked the new tab: allow pop-ups for this app.");
  }} catch (e) {{ say("Could not open it: " + e); }}
}};
</script></body></html>"""


def _component(page: str, height: int) -> None:
    """Draw ``page`` in a frame (scripts run in it): ``st.iframe``, which
    replaced the deprecated ``components.v1.html`` and draws the same frame,
    else the older call (as ``app._embed_frame`` does)."""
    if hasattr(st, "iframe"):
        st.iframe(page, height=height)
        return
    import streamlit.components.v1 as components
    components.html(page, height=height)


def _over_cap(file_name: str, size: int, cap: int) -> None:
    st.caption(
        f"{file_name} is {size / 1048576:.1f} MB, over the "
        f"{cap / 1048576:g} MB that a download in the Foundry app can "
        f"carry (the file travels inside the page). Ask for a smaller "
        f"file, or raise {DOWNLOAD_MAX_MB_ENV} in the app file.")


def _foundry_download(label: str, read: Callable[[], bytes], size: int,
                      file_name: str, mime: Optional[str],
                      key: Optional[str], prepare: bool) -> None:
    """The Foundry component for a file of ``size`` bytes that ``read``
    returns, behind a Prepare download press when it is over
    :func:`download_inline_bytes` and ``prepare`` is set."""
    cap = download_cap_bytes()
    if size > cap:
        _over_cap(file_name, size, cap)
        return
    ready = key or f"{file_name}:{size}"
    prepared = st.session_state.setdefault(_PREPARED, set())
    if prepare and size > download_inline_bytes() and ready not in prepared:
        what = "download" if label == "Download" else label
        if not st.button(
                f"Prepare {what} ({size / 1048576:.1f} MB)",
                key=f"prepare_{ready}",
                help="A large file is put into the page only when asked "
                     "for: in the Foundry app the file travels inside the "
                     "page, and the page is sent again on every action."):
            return
        prepared.add(ready)
    raw = read()
    if len(raw) > cap:
        _over_cap(file_name, len(raw), cap)
        return
    _component(browser_download_html(raw, file_name, mime, label),
               _COMPONENT_HEIGHT)


def download_button(label: str, data: Union[bytes, str], *, file_name: str,
                    mime: Optional[str] = None, key: Optional[str] = None,
                    prepare: bool = True) -> None:
    """A download for ``data`` saved as ``file_name``: Streamlit's own button,
    or on Foundry the in-browser component (see the module docstring).
    ``prepare=False`` skips the Prepare download press on Foundry, for bytes
    the user already asked to have made (the conversation zip)."""
    if not engine_config.is_foundry_deployment():
        st.download_button(label, data=data, file_name=file_name, mime=mime,
                           key=key)
        return
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    _foundry_download(label, lambda: raw, len(raw), file_name, mime, key,
                      prepare)


def download_file(label: str, path: str, *, file_name: Optional[str] = None,
                  mime: Optional[str] = None,
                  key: Optional[str] = None) -> None:
    """:func:`download_button` for the file at ``path``. On Foundry a file
    waiting for its Prepare download press is not read. Raises
    :class:`OSError` when the file cannot be read."""
    name = file_name or os.path.basename(path)
    if not engine_config.is_foundry_deployment():
        st.download_button(label, data=core.artifact_bytes(path),
                           file_name=name, mime=mime, key=key)
        return
    _foundry_download(label, lambda: core.artifact_bytes(path),
                      os.path.getsize(path), name, mime, key, True)


def pdf_page_preview(path: str, key: Optional[str] = None) -> None:
    """A switch that shows the first pages of the PDF at ``path`` as images
    (Foundry), with a note on how to see the whole file.

    The pages are drawn only once the switch is on: every rerun draws every
    card again, and a conversation with more PDFs than the page cache holds
    would otherwise draw every page of every PDF on every rerun."""
    if not st.toggle(f"Preview the first {core.PDF_PREVIEW_PAGES} pages",
                     key=key):
        return
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
        size = os.path.getsize(path)
    except OSError:
        size = None
    if size is None or size > download_cap_bytes():
        how = "The file is too large to open here."
    elif size > download_inline_bytes():
        how = ("Prepare the download above, then use Open in new tab to read "
               "the whole file in the browser's PDF viewer.")
    else:
        how = ("Use Open in new tab above to read the whole file in the "
               "browser's PDF viewer.")
    st.caption(shown + how)

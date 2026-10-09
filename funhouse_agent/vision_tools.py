"""
Extended tool definitions and dispatch for GeotechAgent.

Extends the standard 4 ReAct tools (call_agent, list_methods, describe_method,
list_agents) with vision-capable tools and file output tools.
"""

import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Uploaded-file helpers (shared by the notebook chat FileUpload widgets)
# ---------------------------------------------------------------------------

def sanitize_upload_name(name) -> str:
    """Reduce an uploaded filename to a safe attachment key.

    Drops any directory component and replaces characters outside
    ``[A-Za-z0-9._-]`` with ``_`` so the key is stable and shell/path safe.
    """
    base = os.path.basename(str(name or "").replace("\\", "/")) or "file"
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base or "file"


def iter_upload_files(value):
    """Yield ``(name, bytes)`` from an ipywidgets ``FileUpload.value``.

    Handles both widget generations: ipywidgets 7.x (``{name: {"content":
    bytes}}``) and 8.x (a tuple of ``{"name", "content", ...}`` dicts).
    """
    if not value:
        return
    if isinstance(value, dict):
        for name, info in value.items():
            content = info.get("content", b"") if isinstance(info, dict) else b""
            yield name, bytes(content)
    else:
        for info in value:
            yield info.get("name", "file"), bytes(info.get("content", b""))

# Standard 4 ReAct tools (defined locally to avoid foundry import chain)
STANDARD_TOOLS = {"call_agent", "list_methods", "describe_method", "list_agents"}

# Standard tools + vision + output extensions
EXTENDED_TOOLS = STANDARD_TOOLS | {
    "list_files",
    "read_pdf_text",
    "analyze_image",
    "analyze_pdf_page",
    "render_region",
    "find_like",
    "read_reference_figure",
    "save_file",
    "write_docx",
}


def docx_available() -> bool:
    """Whether Word output can be produced in this install.

    ``write_docx`` needs python-docx, which is a core dependency but can be
    absent from an install trimmed by hand or from an older wheel. A tool that
    would fail is worse than one that was never offered, so the agent layers
    ask this before advertising it.
    """
    try:
        import docx  # noqa: F401
        import markdown_it  # noqa: F401
    except ImportError:
        return False
    return True

VISION_TOOL_DESCRIPTIONS = """
### 5. list_files
Browse a REAL directory (read-only) so you can find where the user's files are
and where to save output. Returns each entry's name, type (dir/file), size, and
modified time. `path` is a real filesystem directory (e.g. `/Workspace/Users/...`,
`/Volumes/...`, `/tmp`, or `.`); `max_entries` caps the count (default 200);
optional `depth` (0-2, default 0) lists sub-directories too. Use this to
DISCOVER the user's folder structure before reading a report or choosing a save
destination — the agent scratch filesystem (`ls`/`read_file`) does NOT see real
paths, this tool does.
```
<tool_call>
{"tool_name": "list_files", "path": "/Workspace/Users/me/geotech_project"}
</tool_call>
```

### 6. read_pdf_text
Extract the TEXT LAYER of a PDF (PyMuPDF — cheap, no vision). **This is the
first-choice reader for a text-based report** (boring logs, lab summaries,
recommendations, specs): read the text directly instead of vision-reading every
page. `source` is an attachment key OR a real filesystem path (driver-local
`/tmp/...` or a `/Volumes/...` path; `/Workspace` reads are unreliable). `pages`
is an int, a list, or a "start-end" range (e.g. "0-9"); omit it for the first
several pages. A page with no text layer (scanned image) is flagged per-page —
use `analyze_pdf_page` on those.
```
<tool_call>
{"tool_name": "read_pdf_text", "source": "/tmp/geotech_report.pdf", "pages": "0-4"}
</tool_call>
```

### 7. analyze_image
Analyze an attached image using vision. Returns text description/analysis.
`attachment_key` is an attachment key OR a real filesystem path.
```
<tool_call>
{"tool_name": "analyze_image", "attachment_key": "site_plan", "prompt": "Extract the cross-section geometry"}
</tool_call>
```

### 8. analyze_pdf_page
Render ONE PDF page and analyze it using vision — for a SCANNED page, a figure,
a boring-log sheet, or a plotted cross-section. For a text-layer report prefer
`read_pdf_text` (cheaper). `attachment_key` is an attachment key OR a real path.
```
<tool_call>
{"tool_name": "analyze_pdf_page", "attachment_key": "report", "page": 0, "prompt": "Extract geometry from this cross-section"}
</tool_call>
```

### 9. render_region
Render a ZOOMED-IN crop of a PDF page and analyze it with vision — the
"geometry says WHERE, vision says WHAT" primitive for drawings. Get exact
coordinates first from the `drawing_ir` module (digitize_drawing →
query_drawing: e.g. a leader's tip, a title block's region), then zoom here to
see WHAT is at that location. `bbox` is [x0,y0,x1,y1] in PDF points, TOP-LEFT
origin with y DOWN (drawing_ir query coordinates are bottom-left/y-up: convert
with y_pdf = page_height − y_ir, or use the drawing_ir `snip_region` method
which converts for you). Optional `marks` = [[x,y,label], ...] draws numbered
circles so the question becomes "what is mark 1 pointing at?". The zoom is
always drawn as large as the vision model reads. Every vision result
carries a `view` (the page rect its image showed) and asks the model to give
locations as 0-999 boxes on that image: to zoom on one, pass `view` +
`image_box` instead of `bbox`.
```
<tool_call>
{"tool_name": "render_region", "attachment_key": "sheet.pdf", "page": 0, "bbox": [400, 180, 480, 240], "marks": [[440, 210, "1"]], "prompt": "What is mark 1 pointing at?"}
</tool_call>
```

### 9b. find_like
Find EVERY copy of one mark — a tag, a code, a symbol — across a drawing set,
including sheets whose lettering is drawn as lines (no text layer). Zoom until
ONE copy is legible (a legend row is fine), then pass its `bbox` (PDF points) —
or a vision result's `view` + the copy's 0-999 `image_box` — and `text` (what it
reads). Every page is image-matched; every candidate is then READ by vision on
enlarged contact sheets; the result counts instances by page, callouts (with
where each leader points) apart from legend entries, and lists uncertain reads
to zoom on. Use it instead of paging through whole-sheet views.
```
<tool_call>
{"tool_name": "find_like", "attachment_key": "sheets.pdf", "page": 5, "bbox": [929, 97, 940, 102], "text": "GCE"}
</tool_call>
```

### 10. read_reference_figure
Render a digitized reference figure (e.g. a DM7 design chart) and read a value
off it with vision. **Use this whenever a numeric value must come from a chart —
do not read values off a chart from the caption or from memory.** Find the figure
first with `call_agent` → `figure_db.figure_search`, then pass its `reference` +
`figure_number` here with a `prompt` describing the value(s) you need. Returns a
chart read-off **estimate** and, where the chart's axes and curve can be found
in the drawing, a value measured from it with its +/- beside the estimate
(`code_reading`, flagged where they disagree) — verify against a
closed-form/digitized method where one exists.
```
<tool_call>
{"tool_name": "read_reference_figure", "reference": "dm7_2", "figure_number": "4-12", "prompt": "Read Kp for phi'=35 deg, theta=10 deg, delta/phi=0.66"}
</tool_call>
```

### 11. save_file
Save raw text or data to a file. Returns the saved file path. The write is
VERIFIED on the real filesystem; report the `saved` path the tool returns (or
its `rescue_path` if the target could not store the file). A `/Workspace` save
on Databricks goes through the authenticated workspace API when available.
For formatted calculation documents, use the `calc_package` module instead.
```
<tool_call>
{"tool_name": "save_file", "path": "output/data.csv", "content": "x,y\n1,2\n3,4"}
</tool_call>
```
For binary content, set encoding to "base64":
```
<tool_call>
{"tool_name": "save_file", "path": "output/image.png", "content": "iVBORw0KGgo...", "encoding": "base64"}
</tool_call>
```

### 12. write_docx
Write a Word (.docx) document from markdown — a memo, a findings summary, a
review response, anything the reader will open in Word, track changes in, or
paste into a report template. Write the markdown as you normally would:
headings (`#` to `####`), `**bold**`, `*italic*`, `` `code` ``, bullet and
numbered lists (one level of nesting), pipe tables (an italic line right after
a table becomes its caption), `> quotes`, fenced code, and `![alt](file.png)`
for a figure you already saved — a bare image name is looked up in the working
folder, and a figure that is not there is reported in `warnings` rather than
losing you the document. `---` becomes a PAGE BREAK. `title` (optional) is
rendered as the document title. For a Mathcad-style CALCULATION package use
the `calc_package` module instead; this is for prose.
```
<tool_call>
{"tool_name": "write_docx", "path": "review_memo.docx", "title": "Foundation Review", "markdown": "# Findings\n\n- Bearing elevation is unconfirmed\n\n![Profile](profile.png)\n"}
</tool_call>
```
"""


# ---------------------------------------------------------------------------
# Default save function (local filesystem)
# ---------------------------------------------------------------------------

def _default_save_fn(path: str, content: bytes | str) -> str:
    """Save to local filesystem. Returns the absolute path of the saved file."""
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if isinstance(content, bytes):
        with open(abs_path, "wb") as f:
            f.write(content)
    else:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
    return abs_path


def dispatch_extended_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    engine,
    attachments: Dict[str, bytes],
    save_fn: Optional[Callable] = None,
) -> str:
    """Dispatch an extended tool call (vision or output).

    Parameters
    ----------
    tool_name : str
        One of the extended tool names.
    arguments : dict
        Tool arguments from the parsed tool call.
    engine : GenAIEngine
        AI engine with analyze_image() capability.
    attachments : dict
        {key: bytes} of attached files.
    save_fn : callable, optional
        File save function ``(path, content) -> saved_path``.
        Defaults to local filesystem write.

    Returns
    -------
    str
        JSON string result.
    """
    if tool_name == "list_files":
        return _dispatch_list_files(arguments)
    elif tool_name == "read_pdf_text":
        return _dispatch_read_pdf_text(arguments, attachments)
    elif tool_name == "read_text_file":
        return _dispatch_read_text_file(arguments)
    elif tool_name == "analyze_image":
        return _dispatch_analyze_image(arguments, engine, attachments)
    elif tool_name == "analyze_pdf_page":
        return _dispatch_analyze_pdf_page(arguments, engine, attachments)
    elif tool_name == "render_region":
        return _dispatch_render_region(arguments, engine, attachments)
    elif tool_name == "find_like":
        return _dispatch_find_like(arguments, engine, attachments)
    elif tool_name == "read_reference_figure":
        return _dispatch_read_reference_figure(arguments, engine)
    elif tool_name == "view_worked_example_source":
        return _dispatch_view_worked_example(arguments, engine)
    elif tool_name == "save_file":
        return _dispatch_save_file(arguments, save_fn or _default_save_fn)
    elif tool_name == "write_docx":
        return _dispatch_write_docx(arguments, save_fn or _default_save_fn)
    else:
        return json.dumps({"error": f"Unknown extended tool: {tool_name}"})


# Keep old name as alias for backwards compatibility
dispatch_vision_tool = dispatch_extended_tool


# ---------------------------------------------------------------------------
# Source resolution: attachment key OR real filesystem path
# ---------------------------------------------------------------------------

def _pdf_page(page) -> Dict[str, int]:
    """``{"pdf_page": page + 1}`` — the number a PDF viewer shows for a
    0-based page index, so an answer can cite it (see
    :func:`funhouse_agent.document_tools.with_viewer_pages`)."""
    try:
        return {"pdf_page": int(page) + 1}
    except (TypeError, ValueError):
        return {}


def _page_count(data) -> Optional[int]:
    """How many pages ``data`` (PDF or image bytes) has, or ``None`` when it
    cannot be opened here (the renderer then says what is wrong)."""
    try:
        import fitz
    except ImportError:
        return None
    for kind in ("pdf", None):
        try:
            doc = (fitz.open(stream=bytes(data), filetype=kind) if kind
                   else fitz.open(stream=bytes(data)))
        except Exception:  # noqa: BLE001 - not this kind of file
            continue
        try:
            return int(doc.page_count)
        finally:
            doc.close()
    return None


def _checked_page(arguments, data):
    """``(page, None)`` -- the 0-based page a page tool's ``page`` or
    ``pdf_page`` (1-based) names, checked against the document -- or
    ``(None, error_json)``: a page the document does not have is a clear
    error, with "pages are 0-based here; PDF page 5 is page 4" when it is
    one past the end, never a raise (live smoke wave 2c: analyze_pdf_page
    RAISED IndexError on PDF page 5 of 5)."""
    from funhouse_agent.page_numbers import page_error, resolve_page
    given_pdf = arguments.get("pdf_page") not in (None, "")
    given_page = arguments.get("page") not in (None, "")
    page, problem = resolve_page(arguments.get("page"),
                                 arguments.get("pdf_page"))
    if problem is not None:
        return None, json.dumps(problem)
    n = _page_count(data)
    if n is not None:
        problem = page_error(page, n, as_pdf_page=given_pdf and not given_page)
        if problem is not None:
            return None, json.dumps(problem)
    return page, None


def _render_error(exc) -> str:
    """A render that failed, as the tool's JSON error (with the 0-based
    hint when it was a page out of range)."""
    from funhouse_agent.page_numbers import range_hint
    out = {"error": str(exc)}
    hint = range_hint(str(exc))
    if hint:
        out["hint"] = hint
    return json.dumps(out)


# ---------------------------------------------------------------------------
# What the read tools may read: this conversation's files (live smoke 1, A2)
# ---------------------------------------------------------------------------
#
# On a host that has bound a working folder for the turn (the web app, per
# conversation; the review suite, per run) the real-disk tools read only:
# that folder (the conversation's uploads, saves and downloads, and its tool
# scratch), the reference library the app fetches PDFs into, and any folder a
# deployment names in GEOTECH_EXTRA_READ_ROOTS. A relative path means a path
# inside the working folder -- never the server process's own folder. One
# process serves many people on Tiny Apps; before this, ``list_files('.')``
# listed the server's folder and a calc sub-agent walked other conversations
# (live smoke 2026-10-08, F23/F26). Library and notebook callers that bind no
# folder keep the old behaviour: any readable path, relative to the cwd.

#: The per-conversation folder tool scratch images go in (page thumbnails,
#: find_like contact sheets): inside the working folder, named to the model
#: by a short relative name (``.scratch/<file>.png``) that ``analyze_image``
#: resolves. The host keeps it off the download cards.
SCRATCH_DIR = ".scratch"

#: More folders the read tools may read while a working folder is bound
#: (separated by ``os.pathsep``) -- e.g. a single-user deployment that wants
#: its agent to browse a ``/Volumes`` share. Never set it on a shared host.
EXTRA_READ_ROOTS_ENV = "GEOTECH_EXTRA_READ_ROOTS"

#: How many of the working folder's file names a refusal lists.
_NAMES_IN_REFUSAL = 25


class PathRefused(FileNotFoundError):
    """A real path outside what this conversation's tools may read. A
    ``FileNotFoundError``, so every read tool already reports it as an
    error result rather than raising."""


def _host_folder() -> Optional[str]:
    """The working folder a host bound for this turn, or ``None``."""
    from funhouse_agent._fileio import host_output_dir
    return host_output_dir()


def scratch_dir(create: bool = True) -> Optional[str]:
    """This conversation's tool scratch folder (:data:`SCRATCH_DIR` inside
    the bound working folder), or ``None`` when no folder is bound."""
    host = _host_folder()
    if not host:
        return None
    path = os.path.join(host, SCRATCH_DIR)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def reference_read_roots() -> List[str]:
    """The folders reference PDFs are read from: ``GEOTECH_REFERENCES_DOCS``,
    the cache the app fetches them into, and a source checkout's
    ``geotech-references/docs``."""
    roots: List[str] = []
    env = os.environ.get("GEOTECH_REFERENCES_DOCS", "").strip()
    if env:
        roots.append(env)
    try:
        from funhouse_agent import reference_docs
        roots.append(reference_docs.cache_dir())
    except Exception:  # noqa: BLE001
        pass
    try:
        from geotech_references import _figures_db
        roots.append(os.path.join(str(_figures_db._REPO_ROOT), "docs"))
    except Exception:  # noqa: BLE001
        pass
    roots.append(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "geotech-references", "docs"))
    out: List[str] = []
    for r in roots:
        a = os.path.abspath(os.path.expanduser(r))
        if a not in out:
            out.append(a)
    return out


#: Folders the read tools may also read on Databricks: there the app serves
#: ONE person (the owner's own Funhouse cluster), whose data lives in Unity
#: Catalog volumes and their workspace folder. Tiny Apps -- one process for
#: many people -- is not Databricks and gets none of these.
DATABRICKS_READ_ROOTS = ("/Volumes", "/Workspace")


def _extra_read_roots() -> List[str]:
    """The folders a deployment exposes beyond the conversation: the
    Databricks data folders on Databricks, and :data:`EXTRA_READ_ROOTS_ENV`."""
    roots: List[str] = []
    try:
        from funhouse_agent._fileio import is_databricks
        if is_databricks():
            roots += list(DATABRICKS_READ_ROOTS)
    except Exception:  # noqa: BLE001
        pass
    raw = os.environ.get(EXTRA_READ_ROOTS_ENV, "")
    roots += [p.strip() for p in raw.split(os.pathsep) if p.strip()]
    out: List[str] = []
    for r in roots:
        a = os.path.abspath(os.path.expanduser(r))
        if a not in out:
            out.append(a)
    return out


def read_roots() -> Optional[List[str]]:
    """What the read tools may read, the working folder FIRST; ``None`` when
    no working folder is bound (library use: unconfined)."""
    host = _host_folder()
    if not host:
        return None
    return ([os.path.abspath(host)] + reference_read_roots()
            + _extra_read_roots())


def _within(path: str, root: str) -> bool:
    """``path`` is ``root`` or inside it, symlinks and ``..`` resolved."""
    try:
        p = os.path.normcase(os.path.realpath(path))
        r = os.path.normcase(os.path.realpath(root))
        return os.path.commonpath([p, r]) == r
    except (ValueError, OSError):           # another drive (Windows)
        return False


def working_folder_names(limit: int = 200) -> List[str]:
    """The files of the bound working folder by their names (one level of
    sub-folders as ``sub/name``), scratch and partial downloads left out."""
    host = _host_folder()
    if not host or not os.path.isdir(host):
        return []
    names: List[str] = []
    try:
        top = sorted(os.scandir(host), key=lambda e: e.name.lower())
    except OSError:
        return []
    for e in top:
        if e.name == SCRATCH_DIR or e.name.startswith(".") \
                or e.name.endswith(".part"):
            continue
        if _safe_is_dir(e):
            try:
                for c in sorted(os.scandir(e.path), key=lambda c: c.name.lower()):
                    if not _safe_is_dir(c) and not c.name.startswith("."):
                        names.append(f"{e.name}/{c.name}")
            except OSError:
                pass
        else:
            names.append(e.name)
        if len(names) >= limit:
            break
    return names


def files_here_text() -> str:
    """``"a.pdf, memo.docx, …"`` -- what a refusal or a miss names."""
    names = working_folder_names()
    if not names:
        return "none yet"
    shown = ", ".join(names[:_NAMES_IN_REFUSAL])
    more = len(names) - _NAMES_IN_REFUSAL
    return shown + (f" … and {more} more (list_files)" if more > 0 else "")


def refusal_message(name: str) -> str:
    """Why a real path is not read, and where the user's files are -- by
    name, never by server path."""
    extra = _extra_read_roots()
    opened = (f", the reference library and these folders: "
              f"{', '.join(extra)}" if extra else " and the reference library")
    return (
        f"'{name}' is outside this conversation's files. The file tools read "
        "only this conversation's own files (its uploads, and what was saved "
        f"or downloaded here){opened}; other folders on the server are not "
        "available. Pass a file by its name. Files here: "
        f"{files_here_text()}.")


def confine_read_path(name: str) -> str:
    """The absolute path a read tool may open for ``name``.

    With a working folder bound, a relative ``name`` is inside that folder
    and an absolute one must lie inside :func:`read_roots`, else
    :class:`PathRefused` (whose message names the conversation's files).
    With none bound, ``name`` resolves against the cwd as it always has.
    """
    p = os.path.expanduser(str(name or "").strip())
    roots = read_roots()
    if roots is None:
        return os.path.abspath(p)
    cand = os.path.abspath(p if os.path.isabs(p) else os.path.join(roots[0], p))
    if any(_within(cand, r) for r in roots):
        return cand
    raise PathRefused(refusal_message(str(name)))


def find_readable_file(name) -> Optional[str]:
    """The real file ``name`` names, or ``None``; raises :class:`PathRefused`
    for a path outside what this conversation may read.

    With a working folder bound: the name inside that folder (a bare name,
    ``sub/x.png``, ``.scratch/x.png``, or an absolute path inside it), else
    a relative name inside a reference folder (a reference PDF by its file
    name). With none bound: the path as given, else the name in the working
    folder -- the old behaviour.
    """
    s = str(name or "").strip()
    if not s:
        return None
    roots = read_roots()
    if roots is None:
        p = os.path.expanduser(s)
        if os.path.isfile(p):
            return os.path.abspath(p)
        from funhouse_agent._fileio import find_in_working_folder
        return find_in_working_folder(p)
    cand = confine_read_path(s)
    if os.path.isfile(cand):
        return cand
    p = os.path.expanduser(s)
    if not os.path.isabs(p):
        for r in roots[1:]:
            c = os.path.abspath(os.path.join(r, p))
            if _within(c, r) and os.path.isfile(c):
                return c
    return None


def display_path(path) -> str:
    """How a result names a real file: relative to the working folder
    (``memo.docx``, ``.scratch/x.png``; ``.`` for the folder itself), a
    reference PDF by its name in the library -- never the server path. With
    no working folder bound, the path as it was (library use)."""
    if not isinstance(path, str) or not path:
        return path
    roots = read_roots()
    if roots is None:
        return path
    for i, root in enumerate(roots):
        if not _within(path, root):
            continue
        if i >= len(roots) - len(_extra_read_roots()) and i > 0:
            return os.path.abspath(path)     # a folder the deployment exposed
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
        if rel.startswith(".."):             # reached through a symlink
            rel = os.path.relpath(os.path.realpath(path),
                                  os.path.realpath(root))
        return "." if rel == "." else rel.replace(os.sep, "/")
    return os.path.basename(path.rstrip("/\\")) or path


def _handle_source(key) -> Optional[str]:
    """The source (attachment key or file) behind a document handle THIS
    conversation opened, else ``None`` (A7: a handle works wherever a file
    is expected)."""
    if not key or not str(key).startswith("doc_"):
        return None
    try:
        from funhouse_agent import document_tools
        return document_tools.handle_source(str(key))
    except Exception:  # noqa: BLE001
        return None


def _resolve_attachment_or_path(key, attachments):
    """Return ``(bytes, source_type)`` for an attachment key, an open
    document handle, or a real file.

    The ``attachments`` dict takes PRECEDENCE. A document handle this
    conversation opened (``doc_…``) stands for the document it was opened
    from. Anything else is a file: with a working folder bound, a name in
    that folder (or a reference PDF by its name) -- a path outside what this
    conversation may read raises :class:`PathRefused`; with none bound, any
    readable path. The dict is never written to. Raises
    ``FileNotFoundError`` naming the attachment keys (and, on a host, the
    conversation's files by name).
    """
    attachments = attachments or {}
    if key and key in attachments:
        return attachments[key], "attachment"
    target = key
    source = _handle_source(key)
    if source and source != key:
        if source in attachments:
            return attachments[source], "attachment"
        target = source
    path = find_readable_file(target) if target else None
    if path:
        try:
            with open(path, "rb") as fh:
                return fh.read(), "path"
        except OSError as e:
            raise FileNotFoundError(
                f"'{key}' exists but could not be read: {e}")
    available = sorted(attachments.keys())
    if read_roots() is not None:
        raise FileNotFoundError(
            f"'{key}' is not an attachment key, a document handle this "
            f"conversation opened, or a file in this conversation's working "
            f"folder. Attachment keys: {available}. Files here: "
            f"{files_here_text()}.")
    raise FileNotFoundError(
        f"'{key}' not found as an attachment key, a readable file path, or a "
        f"file name in the working folder. Available attachment keys: "
        f"{available}. Real filesystem paths are also accepted (driver-local "
        f"/tmp/... or a /Volumes/... path; /Workspace reads are unreliable on "
        f"Databricks)."
    )


# ---------------------------------------------------------------------------
# read_text_file — a REAL text file from disk (HTML, TXT, CSV, JSON, MD)
# ---------------------------------------------------------------------------

#: Characters returned per call. JSON escaping can nearly double HTML, so this
#: stays well under the 16,000-character reference cap and the result is
#: never cut into invalid JSON; longer files page with ``offset``.
_TEXT_READ_MAX_CHARS = 6000
#: Files larger than this are refused rather than read into memory.
_TEXT_READ_MAX_BYTES = 20 * 1024 * 1024


def _real_path_for(path: str) -> str:
    """The real file ``path`` names. With a working folder bound: inside it
    (or a reference PDF by name), :class:`PathRefused` outside what this
    conversation may read. With none: ``path`` as given if it exists, else a
    bare or relative name tried in the working folder."""
    if read_roots() is not None:
        return find_readable_file(path) or confine_read_path(path)
    p = os.path.expanduser(path)
    if os.path.exists(p) or os.path.isabs(p):
        return os.path.abspath(p)
    from funhouse_agent._fileio import find_in_working_folder
    return find_in_working_folder(p) or os.path.abspath(p)


def _dispatch_read_text_file(arguments):
    """Read a real text file in pages of characters.

    Field feedback 2026-09-15 (N1): the calc sub-agent had no way to read the
    HTML report it had written earlier -- deepagents' ``read_file`` sees only
    scratch space and said "not found" -- so it rebuilt the report with
    placeholders where the numbers belonged.
    """
    path = str(arguments.get("path") or arguments.get("file_path") or "").strip()
    if not path:
        return json.dumps({"error": "'path' is required (a real file path)."})
    try:
        resolved = _real_path_for(path)
    except PathRefused as exc:
        return json.dumps({"error": str(exc)})
    shown = display_path(resolved)
    if not os.path.isfile(resolved):
        if read_roots() is not None:
            return json.dumps({"error": (
                f"No such file: '{path}' in this conversation's files. Files "
                f"here: {files_here_text()}. Use list_files to find it.")})
        return json.dumps({"error": (
            f"No such file: '{path}' (looked at '{resolved}'). Use list_files "
            "to find it.")})
    try:
        size = os.path.getsize(resolved)
    except OSError as exc:
        return json.dumps({"error": f"Could not stat '{shown}': {exc}"})
    if size > _TEXT_READ_MAX_BYTES:
        return json.dumps({"error": (
            f"'{shown}' is {size / 1e6:.1f} MB, too large to read as text.")})
    ext = os.path.splitext(resolved)[1].lower()
    extra: Dict[str, Any] = {}
    from funhouse_agent import office_text
    if office_text.is_office_file(resolved):
        # A .docx / .xlsx as Markdown (B3); a docx's pictures go to this
        # conversation's scratch folder, named so write_docx embeds them.
        image_dir = scratch_dir(create=False)
        try:
            text, extra = office_text.read_office(
                resolved, image_dir=image_dir,
                image_ref=SCRATCH_DIR if image_dir else None)
        except office_text.OfficeReadError as exc:
            return json.dumps({"error": str(exc)})
    else:
        try:
            with open(resolved, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            return json.dumps({"error": f"Could not read '{shown}': {exc}"})
        if b"\x00" in data[:4096]:
            if ext in office_text.LEGACY_OFFICE:
                return json.dumps({"error": (
                    f"'{shown}': {office_text.LEGACY_OFFICE[ext]}, then "
                    f"read_text_file it.")})
            hint = ("read_pdf_text or open_document" if ext == ".pdf"
                    else "analyze_image" if ext in (".png", ".jpg", ".jpeg",
                                                    ".gif", ".bmp", ".tif",
                                                    ".tiff", ".webp")
                    else "a tool made for that file type")
            return json.dumps({"error": (
                f"'{shown}' is a binary file, not text. Use {hint}.")})
        text = data.decode("utf-8", errors="replace")
    try:
        offset = max(0, int(arguments.get("offset", 0) or 0))
    except (TypeError, ValueError):
        offset = 0
    try:
        max_chars = int(arguments.get("max_chars", _TEXT_READ_MAX_CHARS)
                        or _TEXT_READ_MAX_CHARS)
    except (TypeError, ValueError):
        max_chars = _TEXT_READ_MAX_CHARS
    max_chars = max(200, min(max_chars, _TEXT_READ_MAX_CHARS))
    chunk = text[offset:offset + max_chars]
    result = {"path": shown, "chars_total": len(text), "offset": offset,
              "returned_chars": len(chunk), "text": chunk}
    if extra:
        note = extra.pop("note", None)
        result.update(extra)
        if note and offset == 0:
            result["note"] = note
    if offset + max_chars < len(text):
        result["truncated"] = True
        result["next_offset"] = offset + max_chars
    return json.dumps(result)


# ---------------------------------------------------------------------------
# list_files — read-only REAL-directory listing (discovery, no engine needed)
# ---------------------------------------------------------------------------

#: Default cap on the number of entries returned.
_LIST_FILES_DEFAULT_MAX = 200
#: Absolute ceiling on entries, even for a recursive listing with a big cap.
_LIST_FILES_HARD_CAP = 1000
#: Character budget for the serialized entries, kept below the reference result
#: cap (16000) so the JSON result is never string-truncated into invalid JSON.
_LIST_FILES_CHAR_BUDGET = 14000
#: Deepest recursion allowed (0 = immediate children only).
_LIST_FILES_MAX_DEPTH = 2


def _fmt_mtime(ts):
    """Format a POSIX mtime as 'YYYY-MM-DD HH:MM' (or None if out of range)."""
    import datetime
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return None


def _list_entry_for(dir_entry, name):
    """Build a listing record for an ``os.DirEntry`` (dir/file, size, mtime)."""
    try:
        is_dir = dir_entry.is_dir(follow_symlinks=False)
    except OSError:
        is_dir = False
    try:
        st = dir_entry.stat(follow_symlinks=False)
    except OSError as e:
        return {"name": name, "type": "dir" if is_dir else "file",
                "error": f"stat failed: {e}"}
    return {
        "name": name,
        "type": "dir" if is_dir else "file",
        "size_bytes": None if is_dir else st.st_size,
        "modified": _fmt_mtime(st.st_mtime),
    }


def _collect_entries(root, depth, max_entries, char_budget, skip_top=()):
    """Collect listing records under ``root`` (BFS, dirs-first within each dir).

    Descends up to ``depth`` levels (0 = immediate children only). Stops at
    ``max_entries`` OR when the serialized size would exceed ``char_budget``,
    returning ``(entries, truncated)``. ``skip_top`` names children of
    ``root`` itself to leave out.
    """
    entries = []
    used = 0
    truncated = False
    queue = [(root, "", 0)]
    while queue and not truncated:
        current, rel, lvl = queue.pop(0)
        try:
            with os.scandir(current) as it:
                children = sorted(
                    (c for c in it if lvl or c.name not in skip_top),
                    key=lambda e: (not _safe_is_dir(e), e.name.lower()),
                )
        except OSError as e:
            entries.append({"name": (rel + "/") if rel else ".",
                            "type": "dir", "error": f"unreadable: {e}"})
            continue
        for child in children:
            if len(entries) >= max_entries:
                truncated = True
                break
            name = (rel + "/" + child.name) if rel else child.name
            entry = _list_entry_for(child, name)
            entry_len = len(json.dumps(entry, default=str)) + 2
            if entries and used + entry_len > char_budget:
                truncated = True
                break
            entries.append(entry)
            used += entry_len
            if entry.get("type") == "dir" and lvl < depth:
                queue.append((os.path.join(current, child.name), name, lvl + 1))
    return entries, truncated


def _safe_is_dir(dir_entry) -> bool:
    try:
        return dir_entry.is_dir(follow_symlinks=False)
    except OSError:
        return False


def _dispatch_list_files(arguments):
    """List a REAL directory so the agent can locate files and save targets.

    Read-only. Accepts ``path`` (real filesystem directory), ``max_entries``
    (default 200, hard-capped), and ``depth`` (0-2, default 0). Returns entry
    records with type/size/mtime, a clear error for a missing/unreadable path,
    and a truncation nudge to narrow to a subdirectory when the cap is hit.
    """
    path = (arguments.get("path") or arguments.get("directory")
            or arguments.get("dir") or ".")
    path = str(path).strip() or "."
    try:
        max_entries = int(arguments.get("max_entries", _LIST_FILES_DEFAULT_MAX))
    except (ValueError, TypeError):
        max_entries = _LIST_FILES_DEFAULT_MAX
    max_entries = max(1, min(max_entries, _LIST_FILES_HARD_CAP))
    try:
        depth = int(arguments.get("depth", 0))
    except (ValueError, TypeError):
        depth = 0
    depth = max(0, min(depth, _LIST_FILES_MAX_DEPTH))

    # With a working folder bound, '.' and every relative path are inside
    # it, and nothing outside this conversation's files is listed (A2).
    roots = read_roots()
    try:
        abs_path = confine_read_path(path)
    except PathRefused as e:
        return json.dumps({"error": str(e)})
    shown = display_path(abs_path)
    if not os.path.exists(abs_path) and roots is not None \
            and os.path.normcase(os.path.abspath(abs_path)) == \
            os.path.normcase(os.path.abspath(roots[0])):
        return json.dumps({"path": ".", "depth": depth, "n_entries": 0,
                           "entries": [],
                           "note": "The conversation's working folder is "
                                   "empty: nothing has been uploaded, saved "
                                   "or downloaded here yet."})
    if not os.path.exists(abs_path):
        if roots is not None:
            return json.dumps({"error": (
                f"Path not found: '{path}' is not in this conversation's "
                f"working folder. Files here: {files_here_text()}. "
                "list_files() with no path lists the working folder.")})
        return json.dumps({
            "error": (
                f"Path not found: '{path}' (resolved to '{abs_path}'). Give an "
                "existing directory — browse from a known root like '/Workspace', "
                "'/Volumes', '/tmp', or '.' (the agent scratch filesystem's "
                "ls/read_file do NOT see real paths; this tool does)."
            )
        })
    if os.path.isfile(abs_path):
        try:
            st = os.stat(abs_path)
            info = {"size_bytes": st.st_size, "modified": _fmt_mtime(st.st_mtime)}
        except OSError as e:
            info = {"stat_error": str(e)}
        return json.dumps({
            "path": shown,
            "is_file": True,
            **info,
            "note": ("This is a file, not a directory. Read it with "
                     "read_pdf_text / analyze_pdf_page / analyze_image, or "
                     "list its parent directory."),
        })

    # The tool scratch folder (thumbnails, contact sheets) is working data,
    # not the user's files: left out of a listing of the working folder.
    at_home = roots is not None and _within(abs_path, roots[0]) \
        and os.path.normcase(os.path.realpath(abs_path)) == \
        os.path.normcase(os.path.realpath(roots[0]))
    try:
        entries, truncated = _collect_entries(
            abs_path, depth, max_entries, _LIST_FILES_CHAR_BUDGET,
            skip_top=(SCRATCH_DIR,) if at_home else ())
    except PermissionError as e:
        return json.dumps({"error": f"Permission denied reading '{shown}': {e}"})
    except OSError as e:
        return json.dumps({
            "error": f"Could not read '{shown}': {type(e).__name__}: {e}"})

    result = {
        "path": shown,
        "depth": depth,
        "n_entries": len(entries),
        "entries": entries,
    }
    if at_home:
        result["note"] = ("This is the conversation's working folder: the "
                          "user's uploads and the files saved or downloaded "
                          "here. Pass a file to the other tools by its name.")
    if truncated:
        result["truncated"] = True
        result["truncated_note"] = (
            f"Listing hit the {max_entries}-entry / size cap. Narrow to a "
            "specific subdirectory (pass its path) to see the rest.")
    return json.dumps(result)


# ---------------------------------------------------------------------------
# read_pdf_text — PyMuPDF text-layer extraction (no vision engine needed)
# ---------------------------------------------------------------------------

#: Default per-call character budget for the concatenated page text. Sized to
#: fit inside the deep reference-read cap (DEFAULT_REFERENCE_RESULT_CHARS=16000)
#: and the v1 read_pdf_text cap, with headroom for the JSON envelope.
_PDF_TEXT_MAX_CHARS = 12000
#: Pages returned when ``pages`` is not given.
_PDF_TEXT_DEFAULT_PAGES = 8
#: A page whose stripped text is shorter than this is treated as "no text layer"
#: (scanned image) and flagged for analyze_pdf_page.
_SCANNED_TEXT_THRESHOLD = 20


def _parse_pages(pages, n_total, default_n):
    """Resolve the ``pages`` argument to a sorted list of valid 0-based indices.

    Accepts ``None`` (first ``default_n`` pages), an int, a list of ints, or a
    ``"start-end"`` / ``"N"`` range string. Returns ``(page_list, note)`` where
    ``note`` is a message about dropped out-of-range requests (or None).
    """
    if n_total <= 0:
        return [], "document has no pages"
    note = None
    if pages is None or pages == "":
        return list(range(min(default_n, n_total))), None
    raw = []
    try:
        if isinstance(pages, int):
            raw = [pages]
        elif isinstance(pages, (list, tuple)):
            raw = [int(p) for p in pages]
        elif isinstance(pages, str):
            s = pages.strip()
            if "-" in s:
                a, b = s.split("-", 1)
                raw = list(range(int(a), int(b) + 1))
            else:
                raw = [int(s)]
        else:
            raw = [int(pages)]
    except (ValueError, TypeError):
        return list(range(min(default_n, n_total))), (
            f"could not parse pages={pages!r}; returned the first "
            f"{min(default_n, n_total)} pages instead")
    valid = sorted({p for p in raw if 0 <= p < n_total})
    dropped = sorted({p for p in raw if not (0 <= p < n_total)})
    if dropped:
        note = (f"requested page(s) {dropped} are out of range "
                f"(document has {n_total} pages, 0-{n_total - 1})")
    if not valid:
        return [], note or "no valid pages requested"
    return valid, note


def _dispatch_read_pdf_text(arguments, attachments):
    """Extract a PDF's text layer with PyMuPDF (no vision engine required)."""
    key = (arguments.get("source") or arguments.get("attachment_key")
           or arguments.get("path") or "")
    pages_arg = arguments.get("pages")
    try:
        max_chars = int(arguments.get("max_chars", _PDF_TEXT_MAX_CHARS))
    except (ValueError, TypeError):
        max_chars = _PDF_TEXT_MAX_CHARS
    try:
        max_pages = int(arguments.get("max_pages", _PDF_TEXT_DEFAULT_PAGES))
    except (ValueError, TypeError):
        max_pages = _PDF_TEXT_DEFAULT_PAGES

    try:
        data, source_type = _resolve_attachment_or_path(key, attachments)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})

    try:
        import fitz
    except ImportError:
        return json.dumps({
            "error": "PyMuPDF required for PDF text extraction. "
                     "pip install PyMuPDF"
        })
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        return json.dumps({
            "error": f"Could not open '{key}' as a PDF: {type(e).__name__}: {e}"
        })

    try:
        n_total = doc.page_count
        page_list, page_note = _parse_pages(pages_arg, n_total, max_pages)

        out_pages = []
        total_chars = 0
        truncated = False
        for p in page_list:
            text = doc[p].get_text("text")
            if len(text.strip()) < _SCANNED_TEXT_THRESHOLD:
                out_pages.append({
                    "page": p, "has_text_layer": False, "text": "",
                    "note": (f"page {p} has no text layer — use "
                             f"analyze_pdf_page for this page"),
                })
                continue
            remaining = max_chars - total_chars
            if remaining <= 0:
                truncated = True
                break
            if len(text) > remaining:
                text = text[:remaining] + "\n...[page text truncated]"
                truncated = True
            out_pages.append({
                "page": p, "has_text_layer": True,
                "chars": len(text), "text": text,
            })
            total_chars += len(text)
            if truncated:
                break
    finally:
        doc.close()

    scanned = [e["page"] for e in out_pages if not e.get("has_text_layer")]
    result = {
        "source": key,
        "source_type": source_type,
        "n_pages_total": n_total,
        "pages_returned": [e["page"] for e in out_pages],
        "pages": out_pages,
    }
    if scanned:
        result["scanned_pages"] = scanned
        result["scanned_note"] = (
            f"{len(scanned)} page(s) have no text layer "
            f"(scanned images): {scanned}. Use analyze_pdf_page for those.")
    if truncated:
        result["truncated"] = True
        result["truncated_note"] = (
            "Output hit the character budget. Request specific later pages "
            "(e.g. pages='8-15') to continue reading.")
    if page_note:
        result["page_request_note"] = page_note
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Repeat reads: an identical vision read in the same conversation (B6)
# ---------------------------------------------------------------------------
#
# Live smoke wave 2a: 24 of 80 page reads re-viewed a page an earlier turn had
# read (433 s of tool time). A read whose file content, page, view or region,
# prompt and vision setup (model, image budget, detail) are ALL the same as
# one made earlier in the SAME conversation returns that reading again,
# marked ``repeat``, instead of asking the model. A different prompt is a
# different read and is never served from here. The cache is per
# conversation (the bound working folder, as the document toolkits are
# kept), never shared between conversations, held in memory and bounded;
# with no working folder bound (library use) nothing is cached.

#: ``0`` / ``off`` turns repeat reads off.
REPEAT_CACHE_ENV = "GEOTECH_VISION_REPEAT_CACHE"
#: Readings kept per conversation (least recently used dropped first).
REPEAT_MAX_ENTRIES = 64
#: Characters of readings kept per conversation.
REPEAT_MAX_CHARS = 2_000_000
#: Conversations whose readings are kept at once.
REPEAT_MAX_CONVERSATIONS = 32

REPEAT_NOTE = ("Identical to a read made earlier in this conversation (same "
               "file, page, view and prompt): that reading is returned "
               "again, not a new look. Change the prompt to look afresh.")


class _RepeatReads:
    """Per-conversation, size-bounded store of vision tool results."""

    def __init__(self):
        self._lock = threading.Lock()
        self._convs: "OrderedDict[str, OrderedDict]" = OrderedDict()

    def get(self, conv, key):
        with self._lock:
            store = self._convs.get(conv)
            if store is None or key not in store:
                return None
            self._convs.move_to_end(conv)
            store.move_to_end(key)
            return store[key][0]

    def put(self, conv, key, text):
        with self._lock:
            store = self._convs.get(conv)
            if store is None:
                store = self._convs[conv] = OrderedDict()
            self._convs.move_to_end(conv)
            store[key] = (text, time.time())
            store.move_to_end(key)
            while len(store) > 1 and (
                    len(store) > REPEAT_MAX_ENTRIES
                    or sum(len(v[0]) for v in store.values())
                    > REPEAT_MAX_CHARS):
                store.popitem(last=False)
            while len(self._convs) > REPEAT_MAX_CONVERSATIONS:
                self._convs.popitem(last=False)

    def clear(self):
        with self._lock:
            self._convs.clear()

    def size(self, conv=None) -> int:
        with self._lock:
            if conv is None:
                return sum(len(s) for s in self._convs.values())
            return len(self._convs.get(conv) or ())


_REPEATS = _RepeatReads()


def clear_repeat_reads() -> None:
    """Forget every conversation's kept readings."""
    _REPEATS.clear()


def _vision_setup(engine) -> Dict[str, Any]:
    """What else decides a reading besides the file, view and prompt: the
    model, the image budget and detail, the vision policy and switches."""
    model = getattr(engine, "model", None)
    name = None
    for attr in ("model_name", "model", "deployment_name", "azure_deployment",
                 "model_id"):
        v = getattr(model, attr, None) if model is not None else None
        if isinstance(v, str) and v:
            name = v
            break
    setup: Dict[str, Any] = {"engine": type(engine).__name__, "model": name}
    try:
        from funhouse_agent import review_flags, vision_view
        setup.update(budget=vision_view.budget_name(engine),
                     detail=vision_view.detail(engine),
                     max_px=vision_view.max_px(engine),
                     policy=vision_view.policy(),
                     patch_align=vision_view.patch_align(),
                     text_context=review_flags.vision_text_context(),
                     structured=review_flags.vision_structured())
    except Exception:  # noqa: BLE001 - an unknown setup just keys narrower
        setup["setup"] = "unknown"
    return setup


def _repeat_key(tool, data, normalized, arguments, engine):
    """``(conversation, key)`` for a vision read, or ``None`` when it must
    not be served again (switched off, inline looks, no conversation)."""
    flag = (os.environ.get(REPEAT_CACHE_ENV) or "").strip().lower()
    if flag in ("0", "off", "false", "no"):
        return None
    if arguments.get("_inline"):
        return None
    try:
        from funhouse_agent.deep.vision_engine import conversation_key
        conv = conversation_key()
    except Exception:  # noqa: BLE001
        return None
    if not conv:
        return None
    try:
        blob = json.dumps({"tool": tool,
                           "file": hashlib.sha256(bytes(data)).hexdigest(),
                           "args": normalized,
                           "setup": _vision_setup(engine)},
                          sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        return None
    return conv, hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _repeat_hit(rkey, tool) -> Optional[str]:
    """The earlier reading for ``rkey``, marked as a repeat, or ``None``."""
    if rkey is None:
        return None
    raw = _REPEATS.get(*rkey)
    if raw is None:
        return None
    try:
        out = json.loads(raw)
    except ValueError:
        return None
    out["repeat"] = REPEAT_NOTE
    if tool == "analyze_pdf_page":
        _fit_located(out, TILED_RESULT_CHARS)
        if out.get("tiles"):
            _fit_tiles(out)
    else:
        _fit_region(out)
    return json.dumps(out)


def _repeat_store(rkey, raw) -> None:
    """Keep a COMPLETE reading: never an error, an inline look, or a page
    read missing a tile or its overview (asking again might fill it)."""
    if rkey is None:
        return
    try:
        out = json.loads(raw)
    except (TypeError, ValueError):
        return
    if not isinstance(out, dict) or "error" in out or "image_id" in out \
            or out.get("tiles_not_read") or out.get("overview_not_read") \
            or out.get("cut_off"):
        return
    if any(isinstance(t, dict) and "error" in t
           for t in out.get("tiles") or ()):
        return
    _REPEATS.put(rkey[0], rkey[1], raw)


def _repeat_or_read(tool, data, normalized, arguments, engine, read):
    """``read()``'s result, or the same reading made earlier in this
    conversation (:data:`REPEAT_NOTE`)."""
    rkey = _repeat_key(tool, data, normalized, arguments, engine)
    hit = _repeat_hit(rkey, tool)
    if hit is not None:
        return hit
    raw = read()
    _repeat_store(rkey, raw)
    return raw


# -- What was read in this conversation (wave 2b, C7) -------------------------

#: Reads remembered per conversation, newest last.
READ_LOG_MAX = 200
#: Conversations whose read logs are kept in memory at once (the record on
#: disk is kept for every conversation: see :func:`reads_file`).
READ_LOG_CONVERSATIONS = 64

#: The conversation's record of its page and region reads (live smoke wave
#: 2c, D4: the log lived in memory only, so a restart or an App Service
#: recycle lost it and a restored conversation never had it). In the web
#: app it sits in the CONVERSATION folder beside ``activity.jsonl`` -- not a
#: download card, mirrored to SharePoint and restored with the rest; for a
#: bare working folder (a library or eval host) in its ``.scratch``.
READS_FILE = "reads.json"

_READ_LOG_LOCK = threading.Lock()
_READ_LOG: "OrderedDict[str, list]" = OrderedDict()


def reads_file(folder=None) -> Optional[str]:
    """Where the read record of the conversation whose working folder is
    ``folder`` (``None`` = the one bound in this context) is kept, or
    ``None`` with no folder. The web app's working folder is
    ``<conversation>/files`` beside the conversation's ``meta.json``: the
    record goes in the conversation folder. Any other folder keeps it in
    its own :data:`SCRATCH_DIR`."""
    if folder is None:
        try:
            folder = _host_folder()
        except Exception:  # noqa: BLE001 - an unbound shared host
            folder = None
    if not folder:
        return None
    folder = os.path.abspath(str(folder))
    parent = os.path.dirname(folder)
    if os.path.basename(folder) == "files" and \
            os.path.isfile(os.path.join(parent, "meta.json")):
        return os.path.join(parent, READS_FILE)
    return os.path.join(folder, SCRATCH_DIR, READS_FILE)


def _load_reads(path: Optional[str]) -> Optional[list]:
    """The reads kept at ``path``; ``None`` when there is no readable
    record."""
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    reads = data.get("reads") if isinstance(data, dict) else data
    if not isinstance(reads, list):
        return None
    return [r for r in reads if isinstance(r, dict)]


def _save_reads(path: Optional[str], reads: list) -> None:
    """Write the record whole, through a temporary file. Best-effort: a
    record that cannot be written never fails the read it describes."""
    if not path:
        return
    tmp = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"reads": reads[-READ_LOG_MAX:]}, fh,
                      ensure_ascii=False)
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError):
        try:
            os.remove(tmp)
        except OSError:
            pass


def _conversation_for(folder=None) -> str:
    if folder:
        try:
            return os.path.normcase(os.path.realpath(str(folder)))
        except (OSError, ValueError):
            return str(folder)
    try:
        from funhouse_agent.deep.vision_engine import conversation_key
        return conversation_key()
    except Exception:  # noqa: BLE001
        return ""


def _document_name(key) -> str:
    """A document as the user knows it: its file name (a handle stands for
    the file it was opened from), never a server path."""
    src = _handle_source(key) or key
    name = os.path.basename(str(src or "").replace("\\", "/").rstrip("/"))
    return name or str(key or "")


def _note_read(tool, key, page, view, prompt) -> None:
    """Remember one successful page or region read for this conversation:
    in memory, and in the conversation's record on disk
    (:func:`reads_file`), which outlives the process."""
    conv = _conversation_for()
    if not conv:
        return
    entry = {"document": _document_name(key), "page": page,
             **_pdf_page(page), "view": view, "tool": tool,
             "prompt": " ".join(str(prompt or "").split())[:100],
             "when": round(time.time(), 1)}
    try:
        path = reads_file()
    except Exception:  # noqa: BLE001 - memory only, then
        path = None
    with _READ_LOG_LOCK:
        log = _newest_record(_load_reads(path), _READ_LOG.get(conv))
        log = (log + [entry])[-READ_LOG_MAX:]
        _READ_LOG[conv] = log
        _READ_LOG.move_to_end(conv)
        while len(_READ_LOG) > READ_LOG_CONVERSATIONS:
            _READ_LOG.popitem(last=False)
        _save_reads(path, log)


def _newest_record(on_disk: Optional[list], in_memory: Optional[list]) -> list:
    """The fuller of the two records of one conversation's reads: the one
    on disk (it outlives the process, and a restore brings it back) unless
    memory holds more (a write that failed)."""
    mem = list(in_memory or [])
    if on_disk is not None and len(on_disk) >= len(mem):
        return list(on_disk)
    return mem


def reads_for_conversation(folder=None) -> List[Dict[str, Any]]:
    """The page and region reads made in one conversation, newest last:
    ``document`` (file name), ``page`` (0-based), ``pdf_page`` (1-based),
    ``view`` (``"page"``, ``"page+tiles NxN"``, or ``[x0, y0, x1, y1]`` in
    PDF points for a zoom), ``tool``, ``prompt`` (first 100 characters) and
    ``when`` (epoch seconds). ``folder`` is the conversation's working
    folder; ``None`` = the one bound in this context. Only successful
    ``analyze_pdf_page`` / ``render_region`` reads; per conversation, never
    shared, at most :data:`READ_LOG_MAX` of them. Read back from the
    conversation's record on disk (:func:`reads_file`), so a restart, a
    recycled host or a restored conversation keeps it (live smoke wave 2c,
    D4)."""
    conv = _conversation_for(folder)
    if not conv:
        return []
    try:
        on_disk = _load_reads(reads_file(folder))
    except Exception:  # noqa: BLE001 - memory only, then
        on_disk = None
    with _READ_LOG_LOCK:
        log = _newest_record(on_disk, _READ_LOG.get(conv))
    return [dict(e) for e in log]


def clear_read_log() -> None:
    """Forget every conversation's read log IN MEMORY (what a restart does;
    each conversation's record on disk stays)."""
    with _READ_LOG_LOCK:
        _READ_LOG.clear()


# -- A vision answer cut off at the model's output limit (wave 2b, C5) --------

CUT_OFF_NOTE = ("the vision model's answer was CUT OFF at its output limit, "
                "so the end of this reading is missing: ask about a smaller "
                "part (render_region) or a narrower question before relying "
                "on what is not here")


def _is_cut_off(text) -> bool:
    return bool(getattr(text, "cut_off", False))


# -- Only images go to analyze_image (wave 2b, C6) -----------------------------

#: Image types sent as they are; GIF, TIFF and BMP are converted to PNG.
_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"), (b"GIF89a", "gif"), (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"), (b"BM", "bmp"),
)


class NotAnImage(ValueError):
    """Bytes analyze_image cannot send as a picture; the message says which
    tool reads them instead."""


def _image_kind(data: bytes) -> Optional[str]:
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    for magic, kind in _IMAGE_MAGIC:
        if data[:len(magic)] == magic:
            return kind
    return None


def _not_an_image_message(data: bytes, name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    shown = os.path.basename(name.replace("\\", "/")) or name
    if data[:5] == b"%PDF-" or ext == ".pdf":
        return (f"'{shown}' is a PDF, not an image: look at a page with "
                f"analyze_pdf_page, or open it with open_document.")
    from funhouse_agent import office_text
    if office_text.is_office_file(name) or ext in office_text.LEGACY_OFFICE:
        return (f"'{shown}' is a Word/Excel file, not an image: read it with "
                f"read_text_file.")
    if data[:4] == b"PK\x03\x04":
        return (f"'{shown}' is a zip-based document (an Office file?), not "
                f"an image.")
    if data and b"\x00" not in data[:4096]:
        return (f"'{shown}' is a text file, not an image: read it with "
                f"read_text_file.")
    return (f"'{shown}' is not an image this tool can send (PNG, JPEG, "
            f"WebP, GIF, TIFF or BMP).")


def _image_for_vision(data: bytes, name: str, engine=None):
    """``(bytes, note)`` to send for an image file: PNG, JPEG and WebP as
    they are; GIF, TIFF and BMP converted to PNG (the first frame or page;
    held to the vision model's largest edge). Raises :class:`NotAnImage`
    for anything else — before, every file went out labelled PNG, and an
    .xlsx came back as a 400 from the API."""
    data = bytes(data or b"")
    kind = _image_kind(data)
    if kind in ("png", "jpeg", "webp"):
        return data, None
    if kind is None:
        raise NotAnImage(_not_an_image_message(data, str(name or "")))
    import io
    try:
        from PIL import Image
    except ImportError as exc:                   # pragma: no cover - env
        raise NotAnImage(f"a {kind.upper()} image needs Pillow to be "
                         f"converted: {exc}") from exc
    try:
        im = Image.open(io.BytesIO(data))
        frames = getattr(im, "n_frames", 1) or 1
        im.seek(0)
        if im.mode not in ("1", "L", "LA", "RGB", "RGBA"):
            im = im.convert("RGBA" if im.mode in ("P", "PA") else "RGB")
        cap = None
        try:
            from funhouse_agent import vision_view
            cap = vision_view.max_px(engine)
        except Exception:  # noqa: BLE001
            cap = None
        if cap and max(im.size) > cap:
            im.thumbnail((cap, cap))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    except Exception as exc:  # noqa: BLE001 - a damaged image
        raise NotAnImage(f"'{os.path.basename(str(name))}' looks like a "
                         f"{kind.upper()} image but could not be read: "
                         f"{type(exc).__name__}: {exc}") from exc
    note = f"converted from {kind.upper()} to PNG"
    if frames > 1:
        note += f" (page 1 of {frames}: only the first was sent)"
    return buf.getvalue(), note


def _dispatch_analyze_image(arguments, engine, attachments):
    """Handle analyze_image tool call."""
    key = arguments.get("attachment_key", "")
    prompt = arguments.get("prompt", "Describe this image.")

    try:
        image_data, src = _resolve_attachment_or_path(key, attachments)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    try:
        image_data, image_note = _image_for_vision(image_data, key, engine)
    except NotAnImage as e:
        return json.dumps({"error": str(e)})

    if arguments.get("_inline") and src == "path":
        inline = _inline_image_file_result(image_data, key)
        if inline is not None:
            return json.dumps(inline)

    def read():
        try:
            result = engine.analyze_image(image_data, prompt)
            out = {"analysis": result}
            if image_note:
                out["image_note"] = image_note
            if _is_cut_off(result):
                out["cut_off"] = CUT_OFF_NOTE
            return json.dumps(out)
        except (NotImplementedError, AttributeError) as e:
            return json.dumps({
                "error": f"Vision not available on this engine: {e}"
            })
        except Exception as e:
            return json.dumps({"error": _vision_error(e)})

    return _repeat_or_read("analyze_image", image_data, {"prompt": prompt},
                           {}, engine, read)


def _dispatch_render_region(arguments, engine, attachments):
    """Handle a render_region tool call — the region-snip "zoom in" primitive.

    Live in the agent catalog since Phase 2/B6 of the drawing-intelligence
    build (module_work/DRAWING_INTELLIGENCE_DESIGN.md): registered in
    ``EXTENDED_TOOLS``/``VISION_TOOL_DESCRIPTIONS``, routed by
    ``dispatch_extended_tool``, and exposed on both the deep-agent
    (``deep/tools.py``) and native (``native_tools.py``) surfaces. Same
    shape/conventions as ``_dispatch_analyze_pdf_page`` (same
    attachment-or-path resolution, same "render then vision-analyze" flow).

    Arguments: ``attachment_key`` (or a real path), ``page`` (0-indexed,
    default 0), and WHERE, one of two ways: ``bbox`` ([x0,y0,x1,y1] in PDF
    points, PyMuPDF page space — see ``planlens.ir.render`` for the coordinate
    contract), or ``view`` + ``image_box`` (the ``view`` an earlier vision
    result returned and a 0-999 box the analysis gave on that image —
    :mod:`funhouse_agent.vision_view`). Also ``dpi`` (default: fill the image
    budget; 300 without one), ``pad_frac`` (default 0.15), ``marks``
    (optional list of [x,y,label] for set-of-marks prompting), ``prompt``.

    A zoom on ``view`` + ``image_box`` is padded by the location error of
    the view the box came from (:func:`vision_view.zoom_pad`: a tenth of that
    view each way, at least 12 pt), not by 15 % of the box: a box read off a
    whole sheet can be tens of points off, and a window the size of the box
    showed blank paper (11 first zooms in 11, 2026-10-07).

    A ``dpi`` the caller gives is NOT used while an image budget is in force
    (the result says so): planlens lowers a dpi that would overshoot the
    budget but never raises one, so an agent's ``dpi: 300`` on a 100 pt
    window rendered a quarter of the pixels the budget allowed — 45 of 106
    zooms in Foundry brief 4 (2026-10-07), at a median 4.2 px per point
    against 15-17 without, and the hedged readings that dropped two tags.
    """
    from funhouse_agent import vision_view

    key = arguments.get("attachment_key", "")
    bbox = arguments.get("bbox")
    view = arguments.get("view")
    image_box = arguments.get("image_box")
    dpi = arguments.get("dpi")
    dpi_note = None
    if dpi is not None and vision_view.budget(engine) is not None:
        dpi_note = (f"dpi={dpi} was not used: the zoom is drawn as large as "
                    f"the vision model reads, and a dpi could only make it "
                    f"smaller and harder to read")
        dpi = None
    pad_frac = arguments.get("pad_frac", 0.15)
    marks = arguments.get("marks")
    prompt = arguments.get("prompt", "Describe what this zoomed-in region shows.")

    window, render_pad, padded = bbox, pad_frac, None
    if view is not None or image_box is not None:
        if bbox is not None:
            return json.dumps({"error": "pass bbox OR view + image_box, not both"})
        try:
            thing = vision_view.image_box_to_page(view or [], image_box or [])
            px, py = vision_view.zoom_pad(view, thing, float(pad_frac))
        except (TypeError, ValueError) as e:
            return json.dumps({
                "error": f"view + image_box: {e}",
                "hint": "view = the 'view' of an earlier vision result; "
                        "image_box = a 0-999 box the analysis gave on it"})
        bbox = [round(v, 2) for v in thing]
        window = [thing[0] - px, thing[1] - py, thing[2] + px, thing[3] + py]
        render_pad, padded = 0.0, [round(px, 1), round(py, 1)]

    try:
        pdf_bytes, _src = _resolve_attachment_or_path(key, attachments)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    page, refused = _checked_page(arguments, pdf_bytes)
    if refused is not None:
        return refused

    rkey = _repeat_key("render_region", pdf_bytes, {
        "page": page, "bbox": arguments.get("bbox"), "view": view,
        "image_box": image_box, "dpi": arguments.get("dpi"),
        "pad_frac": pad_frac, "marks": marks, "prompt": prompt},
        arguments, engine)
    hit = _repeat_hit(rkey, "render_region")
    if hit is not None:
        return hit

    try:
        image_bytes, info = vision_view.render_view(
            pdf_bytes, page=page, bbox=window, marks=marks, dpi=dpi,
            pad_frac=render_pad,
            allow_jpeg=getattr(engine, "accepts_jpeg", False), engine=engine)
    except ImportError:
        return json.dumps({
            "error": "PyMuPDF required for PDF rendering. pip install PyMuPDF"
        })
    except (ValueError, IndexError) as e:
        return _render_error(e)

    lines = _lines_for_context(pdf_bytes, page)
    if arguments.get("_inline"):
        return json.dumps(_inline_result(image_bytes, info, page, engine,
                                         lines, bbox=bbox))
    try:
        result = engine.analyze_image(
            image_bytes, _vision_prompt(prompt, info["clip"], lines,
                                        _sent_size(info)))
        out = {"page": page, **_pdf_page(page), "bbox": bbox,
               "analysis": result, **vision_view.view_payload(info, engine)}
        if _is_cut_off(result):
            out["cut_off"] = CUT_OFF_NOTE
        if padded is not None:
            out["window_padding_pt"] = padded
            out["window_note"] = (
                f"the window is the box padded by {padded[0]:g} x "
                f"{padded[1]:g} pt each side, the location error of the view "
                f"the box was read off; if the thing is not in it, look "
                f"again rather than concluding it is absent")
        if dpi_note:
            out["dpi_note"] = dpi_note
        _finish_answer(out, info)
        if padded is not None:
            _say_how_far_from_the_aim(out, view, bbox)
        if "located" in out:
            _fit_region(out)
        raw = json.dumps(out)
        _repeat_store(rkey, raw)
        _note_read("render_region", key, page,
                   [round(float(v), 1) for v in info["clip"]], prompt)
        return raw
    except (NotImplementedError, AttributeError) as e:
        return json.dumps({
            "error": f"Vision not available on this engine: {e}"
        })
    except Exception as e:
        return json.dumps({"error": _vision_error(e)})


def _say_how_far_from_the_aim(out, source_view, aim_box) -> None:
    """For a zoom on ``view`` + ``image_box``: how far the answer's nearest
    box is from the box the zoom was aimed at, and a note when that is more
    than a box from the source view is off (:func:`vision_view.aim_tolerance`)
    — the window is padded widely enough to hold a neighbour too, and in
    Foundry brief 4 a window aimed at a look-alike was answered about a tag
    98 pt away, which was then ringed twice."""
    from funhouse_agent import vision_view
    try:
        ax = (float(aim_box[0]) + float(aim_box[2])) / 2.0
        ay = (float(aim_box[1]) + float(aim_box[3])) / 2.0
        found = vision_view.answer_boxes(out.get("analysis") or "",
                                         out["view"])
        tol = vision_view.aim_tolerance(source_view)
    except (KeyError, TypeError, ValueError):
        return
    out["aim"] = [round(ax, 1), round(ay, 1)]
    if not found:
        return

    def dist(b):
        return ((b[0] + b[2]) / 2.0 - ax) ** 2 + ((b[1] + b[3]) / 2.0 - ay) ** 2

    label, near = min(found, key=lambda lb: dist(lb[1]))
    d = dist(near) ** 0.5
    out["nearest_box_from_aim_pt"] = round(d, 1)
    if len(found) > 1:
        out["boxes_in_answer"] = len(found)
    if d > tol:
        out["aim_note"] = (
            f"the answer's nearest box{(' (' + label + ')') if label else ''} "
            f"is {d:.0f} pt from the box this zoom was aimed at — more than a "
            f"box from that view is off (~{tol:.0f} pt). The window can hold "
            f"a neighbour: this answer may be about a different thing than "
            f"the one you zoomed on. Check what it read before using it, or "
            f"zoom on a smaller box round your target.")
    elif len(found) > 1:
        out["aim_note"] = (
            f"the answer gives {len(found)} boxes; the one nearest the box "
            f"this zoom was aimed at is {d:.0f} pt from it — make sure you "
            f"use that one for the thing you zoomed on.")


#: Two located things from the page and its tiles are one thing when their
#: page boxes come within this fraction of the page view's longer side of
#: each other (at least :data:`vision_view.MIN_LOCATION_ERROR_PT`), and their
#: labels share a code or one has none.
MERGE_FRAC = 0.02

#: Most entries in a tiled result's merged list of located things.
MAX_MERGED = 60

_CODE = re.compile(r"\b(?=[A-Z0-9-]*[A-Z])[A-Z0-9][A-Z0-9-]{1,7}\b")


def _codes(label: str) -> set:
    """The code-like words of a label (GCE, FPG-2, B-14) — what tells two
    located things apart by name; empty when it carries none."""
    return {c.strip("-") for c in _CODE.findall(str(label or ""))
            if len(c.strip("-")) >= 2}


def _merge_found(out) -> None:
    """One list of the things the whole-page answer and its tiles located,
    each with its page box and where it was seen (``found``) — a thing seen
    ONLY in tiles marked ``tiles_only`` and named in ``found_note``.

    Foundry brief 4 (2026-10-07): the whole-page answer missed the tag drawn
    turned 90 degrees; a tile found it; the result nested the tiles under the
    page answer and nothing said "the tiles found one the page did not", so
    the agent worked from the page's list and the tag was never ringed (two
    runs of three on one model, one on the other). The same happened to a
    note the page answer did not name."""
    from funhouse_agent import vision_view
    page_view = out.get("view")
    if not page_view or not out.get("tiles"):
        return
    vx0, vy0, vx1, vy1 = (float(v) for v in page_view)
    reach = max(vision_view.MIN_LOCATION_ERROR_PT,
                MERGE_FRAC * max(vx1 - vx0, vy1 - vy0))
    entries = [("page", lab, box, (vx1 - vx0) * (vy1 - vy0))
               for lab, box in vision_view.answer_boxes(
                   out.get("analysis") or "", page_view)]
    for t in out["tiles"]:
        if not t.get("analysis") or not t.get("view"):
            continue
        tx0, ty0, tx1, ty1 = (float(v) for v in t["view"])
        entries += [(t["tile"], lab, box, (tx1 - tx0) * (ty1 - ty0))
                    for lab, box in vision_view.answer_boxes(
                        t["analysis"], t["view"])]
    if not entries:
        return

    def near(a, b):
        return (a[0] - reach <= b[2] and b[0] - reach <= a[2]
                and a[1] - reach <= b[3] and b[1] - reach <= a[3])

    groups: List[Dict[str, Any]] = []
    for src, label, box, area in entries:
        codes = _codes(label)
        for g in groups:
            if near(g["box"], box) and (not codes or not g["codes"]
                                        or codes & g["codes"]):
                if src not in g["seen_in"]:
                    g["seen_in"].append(src)
                if area < g["area"]:       # a smaller view's box is better
                    g.update(box=box, area=area, what=label or g["what"])
                g["codes"] |= codes
                break
        else:
            groups.append({"what": label, "box": box, "area": area,
                           "seen_in": [src], "codes": set(codes)})
    groups.sort(key=lambda g: (g["box"][1], g["box"][0]))
    rows = []
    for g in groups:
        row = {"what": g["what"] or "?",
               "page_bbox": [round(v, 1) for v in g["box"]],
               "seen_in": g["seen_in"]}
        if "page" not in g["seen_in"]:
            row["tiles_only"] = True
        rows.append(row)
    only = [r for r in rows if r.get("tiles_only")]
    # The things only the tiles saw come first, so a cut list keeps them.
    kept = (only + [r for r in rows if not r.get("tiles_only")])[:MAX_MERGED]
    out["found"] = sorted(kept, key=lambda r: (r["page_bbox"][1],
                                               r["page_bbox"][0]))
    if len(rows) > MAX_MERGED:
        out["found_cut"] = f"{len(rows) - MAX_MERGED} more not listed"
    if only:
        named = "; ".join(f"{r['what']} at {r['page_bbox']} "
                          f"({', '.join(r['seen_in'])})" for r in only[:12])
        out["found_note"] = (
            f"{len(only)} thing(s) were found ONLY in the tiles, not in the "
            f"whole-page answer: {named}"
            + (" …" if len(only) > 12 else "")
            + ". The tiles read small lettering larger — treat these as "
              "found, and zoom on each before relying on it. 'found' lists "
              "every located thing with its page box and where it was seen.")
    else:
        out["found_note"] = ("'found' lists every located thing with its "
                             "page box and where it was seen (the page "
                             "answer and the tiles agree on what is there).")


def _lines_for_context(pdf_bytes, page):
    """The page's text-layer lines, when vision calls are to be told them
    (``GEOTECH_VISION_TEXT_CONTEXT``); ``None`` otherwise."""
    from funhouse_agent import review_flags, vision_view
    if not review_flags.vision_text_context():
        return None
    return vision_view.page_lines(pdf_bytes, page)


def _sent_size(info):
    """``(width_px, height_px)`` of the image as rendered and sent — the size
    a pixel box is converted with (never the size a model says it saw)."""
    try:
        return int(info["width_px"]), int(info["height_px"])
    except (KeyError, TypeError, ValueError):
        return None


def _vision_prompt(prompt, clip, lines, size=None) -> str:
    """The prompt one page/region vision call gets: the text layer inside its
    view (when switched on), the agent's own prompt, the location
    instruction — boxes in pixels of an image of ``size`` (w, h), or with no
    size the old 0-999 grid — and the LOCATED instruction (when switched
    on)."""
    from funhouse_agent import review_flags, vision_view
    parts = []
    if lines is not None:
        parts.append(vision_view.text_context(lines[0], clip, lines[1],
                                              size=size))
    parts.append(prompt)
    text = vision_view.with_location("\n\n".join(parts), size)
    if review_flags.vision_structured():
        text = vision_view.with_locations(text, size)
    return text


def _split_located(out, clip, size=None) -> None:
    """Move a structured LOCATED list out of ``out['analysis']`` into
    ``out['located']`` with page boxes (``GEOTECH_VISION_STRUCTURED``)."""
    from funhouse_agent import review_flags, vision_view
    if not review_flags.vision_structured() or not out.get("analysis"):
        return
    try:
        text, located = vision_view.split_located(str(out["analysis"]), clip,
                                                  size)
    except Exception:  # noqa: BLE001 - an unparsed answer is still an answer
        return
    out["analysis"] = text
    if located:
        out["located"] = located
        if any("zoom_bbox" in it for it in located):
            out["located_note"] = (
                "each located item's page_bbox is PDF points on this page, "
                "read off a view too wide to place a mark from: zoom with "
                "render_region(bbox=<its zoom_bbox>) — the box padded by "
                "this view's location error — and place a mark from the "
                "zoom")
        else:
            out["located_note"] = (
                "each located item's page_bbox is PDF points on this page: "
                "pass it as render_region's bbox to zoom, or as an "
                "annotate_document box")


def _finish_answer(out, info, note: bool = True,
                   tagged_only: bool = False) -> None:
    """Turn a vision answer's locations into what the agent works with: the
    structured LOCATED list split off (when switched on), then every pixel
    box in the prose rewritten on the 0-999 grid of this view
    (:func:`vision_view.boxes_to_grid`), converted with the size SENT.
    ``note`` adds the line saying so (a tile's rows leave it to the page);
    ``tagged_only`` (chart read-offs) converts only boxes tagged ``px=``."""
    from funhouse_agent import vision_view
    size = _sent_size(info)
    _split_located(out, info["clip"], size)
    if not size or not isinstance(out.get("analysis"), str):
        return
    try:
        text, counts = vision_view.boxes_to_grid(out["analysis"], size,
                                                 tagged_only=tagged_only)
    except Exception:  # noqa: BLE001 - an unconverted answer is still one
        return
    out["analysis"] = text
    msg = vision_view.boxes_note(counts, size) if note else None
    if msg:
        out["boxes"] = msg
    hedge = vision_view.bracketed_note(text) if note and not tagged_only \
        else None
    if hedge:
        out["reading_note"] = hedge


#: A render_region result stays under this (the general tool cap is 8,000 —
#: ``deep.tools.DEFAULT_MAX_RESULT_CHARS``), so it is never cut mid-JSON.
REGION_RESULT_CHARS = 7800


def _fit_located(out, limit: int) -> None:
    """Halve the longest ``located`` list (top level or a tile's) until the
    result fits ``limit``; the analysis text is shortened after, by the
    caller. Each list that was cut says so."""
    while len(json.dumps(out)) > limit:
        holders = [d for d in [out, *(out.get("tiles") or [])]
                   if isinstance(d, dict) and d.get("located")]
        if not holders:
            return
        h = max(holders, key=lambda d: len(json.dumps(d["located"])))
        n = len(h["located"])
        if n <= 1:
            h.pop("located", None)
            h.pop("located_note", None)
        else:
            h["located"] = h["located"][: n // 2]
        h["located_cut"] = "some located items left out to fit; zoom closer"


def _fit_region(out) -> None:
    """Keep a render_region result under :data:`REGION_RESULT_CHARS`."""
    _fit_located(out, REGION_RESULT_CHARS)
    while len(json.dumps(out)) > REGION_RESULT_CHARS and \
            len(out.get("analysis") or "") > 200:
        a = out["analysis"]
        out["analysis"] = a[: int(len(a) * 0.8)] + " …[shortened]"


#: What the model is told when the image itself is shown to it (inline mode).
#: ``{in_view}`` says how long: only the newest N views are shown at each
#: call (``GEOTECH_VISION_INLINE_KEEP``, default 2); an older one is not.
INLINE_NOTE = (
    "The image of this view is shown to you with your next step {in_view}: "
    "look at it yourself now. To zoom, call render_region with a bbox in PDF "
    "points inside this view, or with this view + a 0-999 image_box on the "
    "image.")


def _in_view() -> str:
    from funhouse_agent import inline_store
    keep = inline_store.keep_from_env()
    return ("while it is the newest view" if keep == 1 else
            f"while it is among the newest {keep} views")


def inline_note() -> str:
    """:data:`INLINE_NOTE` saying how many of the newest views stay shown."""
    return INLINE_NOTE.format(in_view=_in_view())


def _inline_result(image_bytes, info, page, engine, lines, bbox=None):
    """The result a page/region tool returns when the main model looks itself
    (``GEOTECH_VISION_INLINE``): no one-shot vision call — the image is
    stored and shown to the model at its next call
    (:mod:`funhouse_agent.deep.inline_images`)."""
    from funhouse_agent import inline_store, vision_view
    image_id = inline_store.put(image_bytes, {
        "page": page, "pdf_page": _pdf_page(page).get("pdf_page"),
        "view": [round(float(v), 1) for v in info["clip"]]})
    out = {"page": page, **_pdf_page(page), "image_id": image_id,
           **({"bbox": bbox} if bbox is not None else {}),
           **vision_view.view_payload(info, engine), "note": inline_note()}
    out.pop("zoom_hint", None)
    if lines is not None:
        out["text_layer"] = vision_view.text_context(lines[0], info["clip"],
                                                     lines[1])
    return out


#: What the model is told when an image FILE is shown to it (inline mode);
#: ``{in_view}`` as in :data:`INLINE_NOTE`.
INLINE_IMAGE_NOTE = (
    "The image is shown to you with your next step {in_view}: look at it "
    "yourself now. On a contact sheet each thumbnail is captioned with its "
    "0-based page index; to read a page, use the page tools on the document "
    "itself.")


def inline_image_note() -> str:
    """:data:`INLINE_IMAGE_NOTE` saying how many of the newest views stay
    shown."""
    return INLINE_IMAGE_NOTE.format(in_view=_in_view())

#: An image file larger than this keeps the one-shot vision call: a request
#: body has a size limit on some gateways.
INLINE_IMAGE_MAX_BYTES = 4 * 1024 * 1024


def _inline_image_file_result(data, path):
    """``analyze_image`` of an image FILE when the main model looks itself
    (``GEOTECH_VISION_INLINE``, lean agent): the image is stored and shown at
    the model's next call, as :func:`_inline_result` does for a page. ``None``
    (keep the vision call) for bytes that are not a PNG, JPEG, GIF or WebP
    image, or that are too large to send."""
    from funhouse_agent import inline_store, vision_view
    data = bytes(data or b"")
    if not data or len(data) > INLINE_IMAGE_MAX_BYTES:
        return None
    is_image = (data[:8] == b"\x89PNG\r\n\x1a\n"
                or vision_view.image_media_type(data) != "image/png")
    if not is_image:
        return None
    name = os.path.basename(str(path))
    image_id = inline_store.put(data, {"label": f"image file {name}",
                                       "source": str(path)})
    return {"image_id": image_id, "source": str(path),
            "note": inline_image_note()}


def render_region_to_file(path, filepath=None, content=None, page=0,
                          bbox=None, dpi=300, pad_frac=0.15, marks=None,
                          save_fn=None):
    """Render a PDF region (``planlens.ir.render.render_region``) and save it.

    The save-to-file counterpart to ``render_region``/``_dispatch_render_region``,
    following the same local-write convention as ``save_file``/
    ``_default_save_fn``: makes parent directories, writes the PNG bytes, and
    returns the absolute saved path.
    """
    from planlens.ir.render import render_region as _render_region
    png_bytes = _render_region(filepath=filepath, content=content, page=page,
                               bbox=bbox, dpi=dpi, pad_frac=pad_frac,
                               marks=marks)
    writer = save_fn or _default_save_fn
    return writer(path, png_bytes)


#: Most threads one page read starts: the whole-page call and up to 4 x 4
#: tiles, all at once (live smoke wave 2a, B5: the tiles waited for the
#: whole-page answer they do not use, 10-30 s a read). The process's
#: in-flight cap (``deep.vision_engine.call_slot``) still bounds how many
#: are asking the model at any moment.
TILE_WORKERS = 17

#: The whole tiled result stays under this (the vision cap is 32,000 —
#: ``deep.tools.DEFAULT_VISION_RESULT_CHARS``).
TILED_RESULT_CHARS = 30000


class _Failed:
    """A vision call that raised: the error, kept for the result."""

    def __init__(self, exc):
        self.exc = exc


def _vision_error(exc) -> str:
    """Why a vision call gave no reading, in plain words (a busy model is
    said to be busy, not dumped as the SDK's error text)."""
    from funhouse_agent.deep.vision_engine import describe_error
    try:
        return describe_error(exc)
    except Exception:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def _run_together(jobs):
    """Run zero-argument ``jobs`` at once, each in a copy of the caller's
    context (so the run's callbacks — the activity log, the token count —
    see it; a plain pool starts every worker with an empty one). Results in
    the jobs' order; a job that raised is a :class:`_Failed`."""
    from concurrent.futures import ThreadPoolExecutor
    import contextvars

    def guarded(job):
        try:
            return job()
        except Exception as exc:  # noqa: BLE001 - handed back in order
            return _Failed(exc)

    if len(jobs) == 1:
        return [guarded(jobs[0])]
    with ThreadPoolExecutor(max_workers=min(len(jobs), TILE_WORKERS)) as ex:
        futs = [ex.submit(contextvars.copy_context().run, guarded, j)
                for j in jobs]
        return [f.result() for f in futs]


def _dispatch_analyze_pdf_page(arguments, engine, attachments):
    """Handle analyze_pdf_page tool call.

    ``tiles`` — ``"auto"`` (default), ``"off"``, or N / ``"NxN"`` for an
    N x N split (N 2-4; a larger N is held to 4 and the result says so; any
    other value is an error naming what is accepted — ``"6x6"`` used to be
    read silently as no tiles). On ``auto`` the page is also read in
    overlapping tiles when its small lettering would arrive under
    ``vision_view.LEGIBLE_TEXT_PX`` in the whole-page image AS SENT (measured
    from the text layer, or assumed 0.06 in on a sheet whose lettering is
    drawn as lines): each tile goes at the same budget, so the lettering is N
    times larger, and the result carries the whole-page overview AND every
    tile's reading, each with its ``view`` for zooming. Policy ``efficient``
    turns ``auto`` off.

    The whole-page call and the tiles run AT ONCE (B5), within the process's
    in-flight cap; the result keeps their order and meaning. A tile that
    could not be read is named at the top (``tiles_not_read``), and a failed
    whole-page call with tiles read is said too (``overview_not_read``) —
    the rest of the reading still comes back (B7). An identical read earlier
    in the same conversation is returned again, marked ``repeat`` (B6).
    """
    key = arguments.get("attachment_key", "")
    prompt = arguments.get("prompt", "Describe the content of this page.")
    tiles, tiles_note, tiles_error = _parse_tiles(arguments.get("tiles", "auto"))
    if tiles_error:
        return json.dumps({"error": tiles_error, "accepted": TILES_ACCEPTED})

    try:
        pdf_bytes, _src = _resolve_attachment_or_path(key, attachments)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    page, refused = _checked_page(arguments, pdf_bytes)
    if refused is not None:
        return refused

    raw = _repeat_or_read(
        "analyze_pdf_page", pdf_bytes,
        {"page": page, "prompt": prompt, "tiles": tiles}, arguments, engine,
        lambda: _read_pdf_page(pdf_bytes, page, prompt, tiles, tiles_note,
                               arguments, engine))
    try:
        out = json.loads(raw)
    except ValueError:
        out = None
    if isinstance(out, dict) and "error" not in out and "repeat" not in out \
            and "image_id" not in out:
        n = int(round(len(out.get("tiles") or ()) ** 0.5))
        _note_read("analyze_pdf_page", key, page,
                   f"page+tiles {n}x{n}" if n > 1 else "page", prompt)
    return raw


def _read_pdf_page(pdf_bytes, page, prompt, tiles, tiles_note, arguments,
                   engine):
    """:func:`_dispatch_analyze_pdf_page` once the document is in hand."""
    # Render the page at the vision model's image budget.
    from funhouse_agent import vision_view
    try:
        image_bytes, info = vision_view.render_view(
            pdf_bytes, page=page,
            allow_jpeg=getattr(engine, "accepts_jpeg", False),
            engine=engine)
    except ImportError:
        return json.dumps({
            "error": "PyMuPDF required for PDF rendering. pip install PyMuPDF"
        })
    except (ValueError, IndexError) as e:
        return _render_error(e)

    lines = _lines_for_context(pdf_bytes, page)
    if arguments.get("_inline"):
        # The main model looks at the whole page itself and zooms with
        # render_region where the lettering is small: no tiles.
        return json.dumps(_inline_result(image_bytes, info, page, engine,
                                         lines))
    page_prompt = _vision_prompt(prompt, info["clip"], lines,
                                 _sent_size(info))
    n = _tile_count(tiles, info)
    if n > 1 and not _has_ink(image_bytes):
        # A blank page has nothing to read closer (wave 2b, C11).
        n = 1
        tiles_note = "; ".join(filter(None, [
            tiles_note, "the page has no ink (it is blank), so it was not "
                        "read in tiles"]))
    # The whole page and its tiles at once: the tiles never use the
    # whole-page answer, so the read takes the longest call, not the sum.
    jobs = [lambda: engine.analyze_image(image_bytes, page_prompt)]
    if n > 1:
        jobs += _tile_jobs(pdf_bytes, page, info, n, prompt, engine, lines)
    results = _run_together(jobs)
    result, tile_rows = results[0], (results[1:] if n > 1 else None)
    if isinstance(result, _Failed):
        exc = result.exc
        if isinstance(exc, (NotImplementedError, AttributeError)):
            return json.dumps({
                "error": f"Vision not available on this engine: {exc}"
            })
        if not tile_rows or all("error" in t for t in tile_rows):
            return json.dumps({"error": _vision_error(exc)})
    out = {"page": page, **_pdf_page(page),
           "analysis": "" if isinstance(result, _Failed) else result,
           **vision_view.view_payload(info, engine)}
    if isinstance(result, _Failed):
        out["overview_not_read"] = (
            f"the whole-page view was NOT read ({_vision_error(result.exc)}); "
            f"the tiles below were read")
    cut = (["the whole-page view"] if _is_cut_off(result) else []) + [
        f"tile {t['tile']}" for t in tile_rows or ()
        if isinstance(t, dict) and t.get("cut_off")]
    if cut:
        out["cut_off"] = f"{', '.join(cut)}: {CUT_OFF_NOTE}"
    _finish_answer(out, info)
    if tiles_note:
        out["tiles_note"] = tiles_note

    if n > 1:
        out["tiles"] = tile_rows
        failed = _tiles_not_read_note(tile_rows)
        if failed:
            out["tiles_not_read"] = failed
        why = ("the page's small lettering was too small in the whole-page "
               "image, so it was ALSO" if tiles == "auto" else "it was ALSO")
        out["tiling"] = (
            f"{why} read in {n}x{n} overlapping tiles (each at the same "
            f"image size, lettering {n}x larger). Trust a tile over the "
            f"overview for small lettering. Each tile's boxes are on that "
            f"tile's own 0-999 grid: pass them with the TILE's view, to zoom "
            f"with render_region or, once the thing is legible in a view of "
            f"{vision_view.MARK_VIEW_PT:.0f} pt or less, to place a mark.")
        out.pop("legibility", None)
        _merge_found(out)
        if "reading_note" not in out:
            hedge = next((vision_view.bracketed_note(t.get("analysis"))
                          for t in out["tiles"]
                          if vision_view.bracketed_note(t.get("analysis"))),
                         None)
            if hedge:
                out["reading_note"] = hedge
    # Located lists first (they can outgrow the text), then the text.
    _fit_located(out, TILED_RESULT_CHARS)
    if n > 1:
        _fit_tiles(out)
    return json.dumps(out)


#: What ``analyze_pdf_page(tiles=...)`` accepts, in the words the agent sees.
TILES_ACCEPTED = ("'auto' (the default: tiles only when the lettering is too "
                  "small in the whole-page image), 'off', or N or 'NxN' for "
                  "an N x N split with N from 2 to 4, e.g. '3' or '3x3'")

_TILES_OFF = ("off", "none", "no", "false", "0", "1", "1x1")
_TILES_N = re.compile(r"^\s*(\d+)\s*(?:[x×*]\s*(\d+))?\s*$", re.IGNORECASE)


def _parse_tiles(tiles):
    """``(value, note, error)`` for an ``analyze_pdf_page(tiles=...)``
    argument: value is ``"auto"``, ``"off"`` or N (2-4); ``note`` says when
    N was held to the maximum; ``error`` names what is accepted when the
    value means nothing (it used to be read silently as no tiles)."""
    from funhouse_agent import vision_view
    top = vision_view.MAX_TILES_PER_SIDE
    if tiles is None or tiles is True:
        return "auto", None, None
    if tiles is False:
        return "off", None, None
    if isinstance(tiles, (int, float)):
        text = str(int(tiles))
    else:
        text = str(tiles).strip().lower()
    if text in ("", "auto"):
        return "auto", None, None
    if text in _TILES_OFF:
        return "off", None, None
    m = _TILES_N.match(text)
    if not m:
        return None, None, f"tiles={tiles!r} is not a tiling."
    rows, cols = int(m.group(1)), int(m.group(2) or m.group(1))
    if rows != cols:
        return None, None, (f"tiles={tiles!r}: a page is split N x N, the "
                            f"same both ways.")
    if rows <= 1:
        return "off", None, None
    if rows > top:
        return top, (f"tiles={tiles!r} asked for; {top}x{top} is the most "
                     f"made (each tile already 1/{top} of the page each way): "
                     f"read with {top}x{top} — zoom further with "
                     f"render_region"), None
    return rows, None, None


def _tile_count(tiles, info) -> int:
    """Tiles per side for a parsed ``tiles`` value (``_parse_tiles``); a raw
    argument is parsed here too, and one that means nothing is 1."""
    from funhouse_agent import vision_view
    if tiles not in ("auto", "off") and not isinstance(tiles, int):
        tiles, _note, error = _parse_tiles(tiles)
        if error:
            return 1
    if tiles == "off":
        return 1
    if tiles == "auto":
        if vision_view.policy() != "robust":
            return 1
        return vision_view.tile_grid(info)
    return max(1, min(vision_view.MAX_TILES_PER_SIDE, int(tiles)))


#: A pixel is ink when its darkest channel is below this (0-255): faint
#: pencil and a yellow highlight count, paper does not.
INK_LEVEL = 230
#: A page is blank when less than this fraction of its pixels is ink
#: (a few specks of scanner dust are not something to read in tiles).
BLANK_INK_FRACTION = 1e-5


def _has_ink(image_bytes) -> bool:
    """Whether a rendered page shows anything at all. Unknown = yes, so a
    page is never left untiled on a guess."""
    try:
        import fitz
        import numpy as np
        pix = fitz.Pixmap(image_bytes)
        colour = pix.n - (1 if pix.alpha else 0)
        a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, :max(1, colour)]
        ink = int((a.min(axis=2) < INK_LEVEL).sum())
        return ink >= BLANK_INK_FRACTION * pix.width * pix.height
    except Exception:  # noqa: BLE001
        return True


def _read_tiles(pdf_bytes, page, info, n, prompt, engine, lines=None):
    """Read the page in ``n`` x ``n`` overlapping tiles, in parallel."""
    return _run_together(_tile_jobs(pdf_bytes, page, info, n, prompt, engine,
                                    lines))


def _tiles_not_read_note(rows) -> Optional[str]:
    """``"tile r2c3 failed: <why>; the rest were read …"`` for the tiles
    that came back with an error, or ``None`` (B7: a failed tile used to be
    a silent hole in the page read)."""
    failed = [t for t in rows or () if isinstance(t, dict) and "error" in t]
    if not failed:
        return None
    names = ", ".join(t.get("tile", "?") for t in failed)
    why = failed[0]["error"]
    if len(failed) == len(rows):
        return (f"NO tile was read ({why}); only the whole-page view "
                f"above was")
    word = "tile" if len(failed) == 1 else "tiles"
    return (f"{word} {names} failed: {why}; the rest were read. That part "
            f"of the page was NOT read in close-up: read it with "
            f"render_region(bbox=<the failed tile's view>) before relying "
            f"on it")


def _tile_jobs(pdf_bytes, page, info, n, prompt, engine, lines=None):
    """One job per tile of an ``n`` x ``n`` overlapping split, in tile
    order; each renders its tile and reads it, and returns its row (an
    ``error`` row, naming why in plain words, when it could not)."""
    from funhouse_agent import vision_view
    boxes = vision_view.tile_boxes(info["clip"], n)

    def one(k):
        r, c = divmod(k, n)
        try:
            img, tinfo = vision_view.render_view(
                pdf_bytes, page=page, bbox=boxes[k], pad_frac=0.0,
                allow_jpeg=getattr(engine, "accepts_jpeg", False),
                engine=engine)
            where = (f"This image is tile row {r + 1} of {n}, column {c + 1} "
                     f"of {n} of the sheet (tiles overlap slightly). Report "
                     f"only what is IN this tile, briefly.")
            text = engine.analyze_image(
                img, _vision_prompt(f"{prompt}\n\n{where}", tinfo["clip"],
                                    lines, _sent_size(tinfo)))
            row = {"tile": f"r{r + 1}c{c + 1}",
                   "view": [round(v, 1) for v in tinfo["clip"]],
                   "view_px": [tinfo["width_px"], tinfo["height_px"]],
                   **({"text_px": tinfo["text_px"]}
                      if tinfo.get("text_px") else {}),
                   "analysis": text,
                   **({"cut_off": True} if _is_cut_off(text) else {})}
            _finish_answer(row, tinfo, note=False)
            row.pop("located_note", None)
            return row
        except Exception as exc:                  # one tile, not the page
            return {"tile": f"r{r + 1}c{c + 1}",
                    "view": [round(v, 1) for v in boxes[k]],
                    "error": _vision_error(exc)}

    return [(lambda k=k: one(k)) for k in range(n * n)]


def _fit_tiles(out) -> None:
    """Shorten each tile's text evenly until the result fits."""
    while len(json.dumps(out)) > TILED_RESULT_CHARS:
        longest = max((t for t in out["tiles"] if t.get("analysis")),
                      key=lambda t: len(t["analysis"]), default=None)
        if longest is None or len(longest["analysis"]) < 200:
            out["analysis"] = (out.get("analysis") or "")[:1500]
            break
        a = longest["analysis"]
        longest["analysis"] = a[: int(len(a) * 0.8)] + " …[shortened]"


def _dispatch_find_like(arguments, engine, attachments):
    """Every copy of one mark across a drawing set, verified — see
    :mod:`funhouse_agent.find_like`.

    Arguments: ``attachment_key`` (or a real path), ``page`` and ``bbox``
    (PDF points round ONE copy of the mark) — or ``view`` + ``image_box`` from
    an earlier vision result — ``text`` (what the mark reads, e.g. "GCE"),
    ``pages`` (default all), ``include_legend``, ``threshold``.
    """
    from funhouse_agent import find_like as _fl
    from funhouse_agent import vision_view
    from funhouse_agent.page_numbers import pdf_pages_to_pages, range_hint
    key = arguments.get("attachment_key", "")
    bbox = arguments.get("bbox")
    view, image_box = arguments.get("view"), arguments.get("image_box")
    if view is not None or image_box is not None:
        if bbox is not None:
            return json.dumps({"error": "pass bbox OR view + image_box, not both"})
        try:
            bbox = list(vision_view.image_box_to_page(view or [], image_box or []))
        except (TypeError, ValueError) as e:
            return json.dumps({"error": f"view + image_box: {e}"})
    if not bbox or len(bbox) != 4:
        return json.dumps({
            "error": "find_like needs the box round ONE copy of the mark",
            "hint": "zoom (render_region) until one copy is legible, then pass "
                    "its bbox in PDF points — or that result's view + the 0-999 "
                    "image_box of the copy; a legend row is a fine example"})
    try:
        pdf_bytes, _src = _resolve_attachment_or_path(key, attachments)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    page, refused = _checked_page(arguments, pdf_bytes)
    if refused is not None:
        return refused
    pages = arguments.get("pages")
    one_based = arguments.get("pdf_pages") not in (None, "", [], ())
    if one_based:
        if pages not in (None, "", [], ()):
            return json.dumps({"error": "give pages (0-based) or pdf_pages "
                                        "(1-based), not both"})
        pages, problem = pdf_pages_to_pages(arguments.get("pdf_pages"))
        if problem is not None:
            return json.dumps(problem)
    try:
        # The conversation's tool scratch folder (A9: working images, not
        # download cards); with no working folder bound, the contact sheets
        # are not written anywhere.
        save_dir = scratch_dir()
    except OSError:
        save_dir = None
    try:
        out = _fl.find_like(
            pdf_bytes, int(page), [float(v) for v in bbox], engine,
            text=(arguments.get("text") or None),
            pages=pages,
            include_legend=bool(arguments.get("include_legend", False)),
            threshold=arguments.get("threshold"),
            save_dir=save_dir)
    except ImportError as e:
        return json.dumps({"error": f"find_like cannot run here: {e}"})
    except (ValueError, IndexError) as e:
        hint = range_hint(str(e), one_based)
        return json.dumps({"error": str(e),
                           "hint": hint or (
                               "box the mark's lettering tightly, with no "
                               "leader or table rule inside the box")})
    if isinstance(out, dict) and out.get("contact_sheets"):
        # Named by their place in the conversation, which analyze_image
        # resolves; the server path stays internal (A6).
        out["contact_sheets"] = [display_path(p)
                                 for p in out["contact_sheets"]]
        out.setdefault("contact_sheets_note", (
            "the contact sheets are working images in this conversation's "
            "scratch folder: view one with analyze_image(attachment_key=<its "
            "name>)"))
    return json.dumps(out)


_READ_OFF_NOTE = (
    "Value(s) are a vision read-off estimate from the chart — accurate to a few "
    "percent on linear axes, looser on log axes or dense curve families. Verify "
    "against a closed-form or digitized method where one exists."
)

#: The note on a reference chart read-off that also carries code's reading
#: (W3 step 7): which value rests on what.
_READ_OFF_NOTE_WITH_CODE = (
    "Two readings. 'analysis' is a vision read-off estimate from the chart "
    "image (by eye: a few percent on linear axes, looser on log axes or "
    "dense curve families). 'code_reading' is measured from the chart's own "
    "drawing - "
    "its axes fitted to the gridlines and printed labels, the curve found "
    "where it crosses the input - with its own +/-: where it gives a value, "
    "that is the value to use, with the vision value beside it; one marked "
    "'disagree' differs by more than about 3x code's +/-, so check it before "
    "relying on either. Where code gives no value it says why, and the "
    "vision read-off stands alone. Verify against a closed-form or digitized "
    "method where one exists."
)


def _dispatch_view_worked_example(arguments, engine):
    """Chart read-off: rendered and sent at the chart budget
    (``vision_view.chart_reading``) — see ``_dispatch_view_worked_example_at_budget``."""
    from funhouse_agent import vision_view
    with vision_view.chart_reading():
        return _dispatch_view_worked_example_at_budget(arguments, engine)


def _dispatch_view_worked_example_at_budget(arguments, engine):
    """Render a worked example's printed source page and analyze it via vision.

    The corpus (funhouse_agent/worked_examples.json) catalogues, per entry with
    a source PDF in the docs folder, ``source_doc`` + 1-based
    ``source_pdf_pages`` (page-search located — treated as estimated). This is
    the sample-calc twin of ``read_reference_figure``: same PDF resolution
    convention (GEOTECH_REFERENCES_DOCS / repo docs), same 220-dpi render, same
    read-off caveat.
    """
    example_id = arguments.get("example_id", "")
    pdf_page = arguments.get("pdf_page")           # 1-based; optional
    question = (arguments.get("prompt", "")
                or "Describe the worked example on this page: the given "
                   "values, the calculation steps shown, and any figures.")

    if not example_id:
        return json.dumps({"error": "'example_id' is required "
                                    "(find one via find_worked_examples)."})
    try:
        from funhouse_agent import worked_examples as _we
        entry = _we.get_example(example_id)
        if entry is None:
            known = [e.get("id") for e in _we.load_examples()]
            return json.dumps({"error": f"Unknown worked example "
                                        f"'{example_id}'. Known ids: {known}"})
        pages = entry.get("source_pdf_pages") or []
        if not pages and pdf_page is None:
            return json.dumps({
                "error": f"Worked example {example_id} has no catalogued "
                         "source pages (source PDF not in the docs folder — "
                         "e.g. the Slide2 verification manual). Its problem/"
                         "dispatch_calls text is still available via "
                         "get_worked_example."})
        page_1b = int(pdf_page) if pdf_page is not None else int(pages[0])
        try:
            pdf_abs = _we.resolve_source_pdf(entry)
        except FileNotFoundError as e:
            from funhouse_agent import reference_docs
            pdf_abs = reference_docs.fetch(entry.get("source_doc", ""))
            if not pdf_abs:
                return json.dumps({"error": reference_docs.missing_message(
                    str(e), entry.get("source_doc", ""))})
    except (KeyError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    from funhouse_agent import vision_view
    try:
        image_bytes, info = vision_view.render_view(
            str(pdf_abs), page=page_1b - 1,
            allow_jpeg=getattr(engine, "accepts_jpeg", False),
            engine=engine)
    except ImportError:
        return json.dumps({
            "error": "PyMuPDF required for PDF rendering. pip install PyMuPDF"})
    except ValueError as e:
        return json.dumps({"error": str(e)})

    framing = (
        f"This is page {page_1b} (of the pages catalogued for worked example "
        f"{entry['id']}: {pages or [page_1b]}) from \"{entry.get('source', '')}\" "
        f"— the printed source of: {entry.get('title', '')}. The catalogued "
        "page numbers were located by text search and may be off by a page — "
        "if this page does not show the expected content, say so plainly. "
        f"Task: {question}"
    )
    try:
        analysis = engine.analyze_image(
            image_bytes, vision_view.with_location(framing, _sent_size(info)))
    except (NotImplementedError, AttributeError) as e:
        return json.dumps({"error": f"Vision not available on this engine: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    out = {
        "example_id": entry["id"],
        "source_doc": entry.get("source_doc"),
        "pdf_page": page_1b,
        "catalogued_pages": pages,
        "analysis": analysis,
        "note": _READ_OFF_NOTE,
        # The PDF by its name in the reference library, which render_region
        # resolves; the server path stays internal.
        "source": display_path(str(pdf_abs)),
        "page": page_1b - 1,
        **vision_view.view_payload(info, engine),
    }
    _finish_answer(out, info, tagged_only=True)
    return json.dumps(out)


def _dispatch_read_reference_figure(arguments, engine):
    """Chart read-off: rendered and sent at the chart budget
    (``vision_view.chart_reading``) — see ``_dispatch_read_reference_figure_at_budget``."""
    from funhouse_agent import vision_view
    with vision_view.chart_reading():
        return _dispatch_read_reference_figure_at_budget(arguments, engine)


def _dispatch_read_reference_figure_at_budget(arguments, engine):
    """Render a catalogued reference figure and read value(s) off it via vision."""
    reference = arguments.get("reference", "")
    figure_number = arguments.get("figure_number", "")
    question = arguments.get("prompt", "") or "Read the relevant value(s) from this chart."

    if not reference or not figure_number:
        return json.dumps({
            "error": "Both 'reference' and 'figure_number' are required "
                     "(find them via figure_db.figure_search)."
        })

    # Resolve the figure to its source PDF page.
    try:
        from geotech_references import _figures_db
        rec = _figures_db.figure_get(reference, figure_number)
        pdf_abs, page_idx = _figures_db.resolve_pdf(reference, figure_number)
    except KeyError as e:
        return json.dumps({"error": f"Figure not found: {e}"})
    except FileNotFoundError as e:
        # Not in the local docs folder: a host-registered source (the web
        # app's SharePoint primary_references/) fetches it on first use.
        from funhouse_agent import reference_docs
        pdf_abs = reference_docs.fetch(rec.get("pdf_path", ""))
        if not pdf_abs:
            return json.dumps({"error": reference_docs.missing_message(
                str(e), rec.get("pdf_path", ""))})
        page_idx = int(rec["pdf_page_index"])
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    # Render the page at the vision model's image budget: the largest image
    # it reads without shrinking, so curves and axis labels stay legible.
    from funhouse_agent import vision_view
    try:
        image_bytes, info = vision_view.render_view(
            str(pdf_abs), page=page_idx,
            allow_jpeg=getattr(engine, "accepts_jpeg", False),
            engine=engine)
    except ImportError:
        return json.dumps({
            "error": "PyMuPDF required for PDF rendering. pip install PyMuPDF"
        })
    except ValueError as e:
        return json.dumps({"error": str(e)})

    # Some catalog pages are estimated (not caption-confirmed); warn the vision
    # model so it verifies the figure is present instead of reading a wrong page.
    estimated = rec.get("page_estimated")
    locate_clause = "that should contain" if estimated else "containing"
    est_caveat = (
        f" NOTE: the page for Figure {rec['figure_number']} was located by "
        "ESTIMATE and may be off by a page or two."
        if estimated else ""
    )
    step_one = (
        f"1. FIRST confirm Figure {rec['figure_number']} actually appears on this "
        "page. If it does NOT, say so plainly and do not read a value — report "
        "that the page lookup was estimated and an adjacent page should be tried.\n"
        if estimated else
        f"1. Locate Figure {rec['figure_number']} on the page; ignore other "
        "figures and body text.\n"
    )
    full_prompt = (
        f"This image is a rendered page from {reference} {locate_clause} Figure "
        f"{rec['figure_number']}: \"{rec['caption']}\".{est_caveat}\n\n"
        "Read the requested value(s) off this engineering chart:\n"
        + step_one +
        "2. Identify the axes (note any logarithmic scales) and the family of "
        "curves and what parameter distinguishes them.\n"
        "3. For the requested inputs, select the correct curve (interpolating "
        "between curves where needed) and read the value at the right axis "
        "position.\n"
        "4. Report the value(s) clearly, state which curve/axis you used, and "
        "flag that this is a chart read-off estimate.\n\n"
        f"Request: {question}"
    )
    # The second voter (W3 step 7): the side call also says, per curve it
    # read, where on the image the curve crosses the input, so code can
    # measure the same thing from the drawing.
    from funhouse_agent import chart_reading
    code_voter = chart_reading.voter_on()
    if code_voter:
        full_prompt = chart_reading.with_read_lines(full_prompt)

    try:
        result = engine.analyze_image(
            image_bytes, vision_view.with_location(full_prompt,
                                                   _sent_size(info)))
    except (NotImplementedError, AttributeError) as e:
        return json.dumps({"error": f"Vision not available on this engine: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    out = {
        "reference": reference,
        "figure_number": rec["figure_number"],
        "caption": rec["caption"],
        "pdf_page_index": page_idx,
        "page_estimated": rec.get("page_estimated", False),
        "analysis": result,
        "note": _READ_OFF_NOTE,
        # Zooming on the chart (an axis, a curve label) goes through
        # render_region on the same PDF page -- named by its file name in
        # the reference library, which render_region resolves.
        "source": display_path(str(pdf_abs)),
        "page": page_idx,
        **vision_view.view_payload(info, engine),
    }
    if code_voter:
        # Read off the RAW answer, before its pixel boxes are rewritten on
        # the 0-999 grid for the agent: code places them with the size sent.
        try:
            block = chart_reading.code_reading(
                str(pdf_abs), page_idx, str(result or ""), info["clip"],
                _sent_size(info), engine)
        except Exception as e:  # noqa: BLE001 - a second voter, never the answer
            block = {"status": "unavailable",
                     "note": f"code could not read the chart: "
                             f"{type(e).__name__}: {e}"}
        out["code_reading"] = block
        out["note"] = _READ_OFF_NOTE_WITH_CODE
    _finish_answer(out, info, tagged_only=True)
    return json.dumps(out)


def _dispatch_write_docx(arguments, save_fn):
    """Handle write_docx tool call: markdown in, a Word file on disk out.

    The rendering is ``calc_package.docx_renderer``; the SAVING is
    ``save_file``'s, byte for byte — the document is rendered to a scratch file,
    read back and handed to the same writer, so it gets the same path
    resolution into the conversation's working folder, the same /Workspace
    handling, the same verification and the same rescue copy when the target
    filesystem does not store it.
    """
    path = str(arguments.get("path", "") or "")
    markdown = arguments.get("markdown", "")
    title = arguments.get("title") or None
    # Who File > Info names as the author (B4: it said "python-docx"): the
    # signed-in person via the app (the host passes ``_author``), else the
    # deployment's markup author, else the renderer's "GeotechStaffEngineer".
    author = (str(arguments.get("_author") or "").strip()
              or os.environ.get("GEOTECH_MARKUP_AUTHOR", "").strip() or None)

    if not path:
        return json.dumps({"error": "Missing required parameter: path"})
    if not markdown:
        return json.dumps({"error": "Missing required parameter: markdown"})
    try:
        from calc_package.docx_renderer import markdown_to_docx
    except ImportError as exc:
        return json.dumps({
            "error": f"Word output is not available here: {exc}",
            "hint": "write the text with save_file (.md or .html) instead",
        })
    if os.path.splitext(path)[1].lower() != ".docx":
        path += ".docx"

    # Relative image paths resolve against the working folder, which is where
    # plot_data, render_figures and save_file have already put the agent's own
    # figures — so `![](profile.png)` finds the PNG written minutes earlier.
    from funhouse_agent._fileio import default_output_dir
    base_dir = default_output_dir() or os.path.dirname(os.path.abspath(path))

    import base64
    import shutil
    import tempfile
    warnings = []
    scratch = tempfile.mkdtemp(prefix="geotech_docx_")
    try:
        built = markdown_to_docx(markdown, os.path.join(scratch, "out.docx"),
                                 base_dir=base_dir, title=title,
                                 warnings=warnings, author=author)
        with open(built, "rb") as fh:
            blob = fh.read()
    except Exception as exc:
        return json.dumps({
            "error": f"could not render the Word document: "
                     f"{type(exc).__name__}: {exc}"})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    result = json.loads(_dispatch_save_file(
        {"path": path, "encoding": "base64",
         "content": base64.b64encode(blob).decode("ascii")}, save_fn))
    if warnings:
        # Not an error: the document was written. These are the things the
        # reader will NOT see, and the agent must say so rather than claim a
        # figure is in a file that carries a line of text where it should be.
        result["warnings"] = warnings
    return json.dumps(result)


def _dispatch_save_file(arguments, save_fn):
    """Handle save_file tool call."""
    path = arguments.get("path", "")
    content = arguments.get("content", "")
    encoding = arguments.get("encoding", "text")

    if not path:
        return json.dumps({"error": "Missing required parameter: path"})
    if not content:
        return json.dumps({"error": "Missing required parameter: content"})

    # Decode base64 binary content
    if encoding == "base64":
        import base64
        try:
            content = base64.b64decode(content)
        except Exception as e:
            return json.dumps({"error": f"Invalid base64 content: {e}"})

    from funhouse_agent._fileio import (
        into_working_folder, rescue_write, workspace_write_hint,
        written_file_problem, workspace_api_upload, _is_workspace_path,
    )

    # With a working folder bound, a save lands in this conversation: a path
    # elsewhere on the server (another conversation's folder, /tmp, the
    # server's own folder) keeps its file name and goes into the working
    # folder instead (A2). A host's own writer still decides where a bare
    # name goes, as it always has.
    host = _host_folder()
    if host:
        given = os.path.expanduser(str(path))
        if save_fn is _default_save_fn:
            path = into_working_folder(str(path), host)
        elif os.path.isabs(given) and not _within(given, host):
            path = os.path.basename(given.rstrip("/\\")) or path

    expected = (content if isinstance(content, bytes)
                else content.encode("utf-8", errors="replace"))

    # Databricks durable /Workspace write: plain FUSE writes to /Workspace
    # often store only a PLACEHOLDER, so route through the authenticated
    # workspace API first — but ONLY for the DEFAULT writer. A custom save_fn
    # is the caller's explicit backend choice and must never be bypassed.
    api_error = None
    if save_fn is _default_save_fn and _is_workspace_path(path):
        api = workspace_api_upload(path, expected)
        if api.get("ok"):
            result = {
                "saved": path,
                "file_exists": True,
                "file_size_bytes": api.get("size", len(expected)),
                "save_method": "workspace_api",
                "note": (
                    "Written durably via the Databricks workspace API."
                    if api.get("verified")
                    else "Written via the Databricks workspace API; the stored "
                    "size could not be read back, so file_size_bytes is the "
                    "number of bytes sent."
                ),
            }
            return json.dumps(result)
        api_error = api.get("error")  # fall through to plain write + verify

    # Write via save_fn. On a hard failure, still rescue the content to /tmp
    # so the user does not lose it, and return a structured error naming both.
    try:
        saved_path = save_fn(path, content)
    except Exception as e:
        err = {"error": f"{type(e).__name__}: {e}" + workspace_write_hint(path)}
        rescue = rescue_write(os.path.abspath(path), expected)
        if rescue:
            err["rescue_path"] = rescue
            err["error"] += (
                f" A verified copy was saved to '{rescue}' — report THAT path "
                "to the user."
            )
        if api_error:
            err["workspace_api_note"] = (
                f"The Databricks workspace API was tried first and failed "
                f"({api_error}); the plain write then also failed.")
        return json.dumps(err)

    # Verify the write on the REAL filesystem so the agent never needs a
    # separate ls/read to trust the save — agent-side filesystem tools may be
    # sandboxed/virtual and cannot see this file. For the default local
    # writer the CONTENT is verified too (Databricks /Workspace FUSE writes
    # can "succeed" while the workspace stores a literal PLACEHOLDER file).
    # A custom save_fn may write somewhere not locally visible (e.g. DBFS),
    # so only flag an error for the default writer.
    result = {"saved": saved_path}
    if isinstance(saved_path, str):
        abs_path = os.path.abspath(saved_path)
        file_exists = os.path.isfile(abs_path)
        result["saved"] = abs_path if file_exists else saved_path
        result["file_size_bytes"] = (
            os.path.getsize(abs_path) if file_exists else 0)
        if save_fn is _default_save_fn:
            problem = written_file_problem(abs_path, expected)
            result["file_exists"] = file_exists and problem is None
            if problem:
                result["error"] = (
                    f"save_file ran but {problem}."
                    + workspace_write_hint(abs_path)
                )
                rescue = rescue_write(abs_path, expected)
                if rescue:
                    result["rescue_path"] = rescue
                    result["error"] += (
                        f" A verified copy was saved to '{rescue}' — report "
                        "THAT path to the user."
                    )
            if api_error:
                # We tried the durable workspace API for a /Workspace path and
                # fell back to a plain write — tell the agent, since a "clean"
                # plain /Workspace write can still be non-durable on Databricks.
                result["workspace_api_note"] = (
                    f"The Databricks workspace API was unavailable "
                    f"({api_error}); used a plain filesystem write instead.")
        else:
            result["file_exists"] = file_exists
            if not file_exists:
                result["note"] = (
                    "File was saved via a custom save function; the path is "
                    "not visible on the local filesystem (e.g. remote/DBFS), "
                    "so file_exists/file_size_bytes reflect the local view "
                    "only."
                )
    if host:
        _name_saved_file(result, save_fn)
    return json.dumps(result)


def _name_saved_file(result: dict, save_fn) -> None:
    """A save made in a conversation names the file by its place in the
    conversation (``memo.docx``), which every file tool resolves -- the
    server path stays internal and never reaches the user (A6). The host's
    ``saved_note`` hook is asked about the REAL path first, so its note
    ("this file is attached to the chat") survives the renaming."""
    saved = result.get("saved")
    if not isinstance(saved, str) or not os.path.isabs(saved):
        return
    hook = getattr(save_fn, "saved_note", None)
    if callable(hook) and not result.get("error"):
        try:
            note = hook(saved)
        except Exception:  # noqa: BLE001 - a note, never the save
            note = None
        if note:
            existing = result.get("note")
            result["note"] = f"{existing} {note}" if existing else str(note)
    result["saved"] = display_path(saved)

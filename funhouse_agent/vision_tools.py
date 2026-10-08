"""
Extended tool definitions and dispatch for GeotechAgent.

Extends the standard 4 ReAct tools (call_agent, list_methods, describe_method,
list_agents) with vision-capable tools and file output tools.
"""

import json
import os
import re
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
chart read-off **estimate** — verify it against a closed-form/digitized method
where one exists.
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


def _resolve_attachment_or_path(key, attachments):
    """Return ``(bytes, source_type)`` for an attachment key OR a real file path.

    The ``attachments`` dict takes PRECEDENCE; only if the key is not an
    attachment is it tried as a filesystem path (driver-local ``/tmp/...`` or a
    ``/Volumes/...`` path). The dict is never written to. Raises
    ``FileNotFoundError`` with an informative message listing the available
    attachment keys AND noting that real paths are accepted.
    """
    attachments = attachments or {}
    if key and key in attachments:
        return attachments[key], "attachment"
    path = key if key and os.path.isfile(key) else None
    if path is None and key:
        from funhouse_agent._fileio import find_in_working_folder
        path = find_in_working_folder(key)
    if path:
        try:
            with open(path, "rb") as fh:
                return fh.read(), "path"
        except OSError as e:
            raise FileNotFoundError(f"'{key}' exists but could not be read: {e}")
    available = sorted(attachments.keys())
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
    """``path`` as given if it exists; a bare or relative name that does not
    is tried in the working folder (``default_output_dir``)."""
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
    resolved = _real_path_for(path)
    if not os.path.isfile(resolved):
        return json.dumps({"error": (
            f"No such file: '{path}' (looked at '{resolved}'). Use list_files "
            "to find it.")})
    try:
        size = os.path.getsize(resolved)
    except OSError as exc:
        return json.dumps({"error": f"Could not stat '{resolved}': {exc}"})
    if size > _TEXT_READ_MAX_BYTES:
        return json.dumps({"error": (
            f"'{resolved}' is {size / 1e6:.1f} MB, too large to read as text.")})
    try:
        with open(resolved, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        return json.dumps({"error": f"Could not read '{resolved}': {exc}"})
    if b"\x00" in data[:4096]:
        ext = os.path.splitext(resolved)[1].lower()
        hint = ("read_pdf_text or open_document" if ext == ".pdf"
                else "analyze_image" if ext in (".png", ".jpg", ".jpeg", ".gif",
                                                ".bmp", ".tif", ".tiff", ".webp")
                else "a tool made for that file type")
        return json.dumps({"error": (
            f"'{resolved}' is a binary file, not text. Use {hint}.")})
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
    result = {"path": resolved, "chars_total": len(text), "offset": offset,
              "returned_chars": len(chunk), "text": chunk}
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


def _collect_entries(root, depth, max_entries, char_budget):
    """Collect listing records under ``root`` (BFS, dirs-first within each dir).

    Descends up to ``depth`` levels (0 = immediate children only). Stops at
    ``max_entries`` OR when the serialized size would exceed ``char_budget``,
    returning ``(entries, truncated)``.
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
                    it,
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

    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(abs_path):
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
            "path": abs_path,
            "is_file": True,
            **info,
            "note": ("This is a file, not a directory. Read it with "
                     "read_pdf_text / analyze_pdf_page / analyze_image, or "
                     "list its parent directory."),
        })

    try:
        entries, truncated = _collect_entries(
            abs_path, depth, max_entries, _LIST_FILES_CHAR_BUDGET)
    except PermissionError as e:
        return json.dumps({"error": f"Permission denied reading '{abs_path}': {e}"})
    except OSError as e:
        return json.dumps({
            "error": f"Could not read '{abs_path}': {type(e).__name__}: {e}"})

    result = {
        "path": abs_path,
        "depth": depth,
        "n_entries": len(entries),
        "entries": entries,
    }
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


def _dispatch_analyze_image(arguments, engine, attachments):
    """Handle analyze_image tool call."""
    key = arguments.get("attachment_key", "")
    prompt = arguments.get("prompt", "Describe this image.")

    try:
        image_data, src = _resolve_attachment_or_path(key, attachments)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})

    if arguments.get("_inline") and src == "path":
        inline = _inline_image_file_result(image_data, key)
        if inline is not None:
            return json.dumps(inline)

    try:
        result = engine.analyze_image(image_data, prompt)
        return json.dumps({"analysis": result})
    except (NotImplementedError, AttributeError) as e:
        return json.dumps({
            "error": f"Vision not available on this engine: {e}"
        })
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


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
    page = arguments.get("page", 0)
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

    try:
        image_bytes, info = vision_view.render_view(
            pdf_bytes, page=page, bbox=window, marks=marks, dpi=dpi,
            pad_frac=render_pad,
            allow_jpeg=getattr(engine, "accepts_jpeg", False), engine=engine)
    except ImportError:
        return json.dumps({
            "error": "PyMuPDF required for PDF rendering. pip install PyMuPDF"
        })
    except ValueError as e:
        return json.dumps({"error": str(e)})

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
        return json.dumps(out)
    except (NotImplementedError, AttributeError) as e:
        return json.dumps({
            "error": f"Vision not available on this engine: {e}"
        })
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


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


#: Parallel vision calls for the tiles of one page.
TILE_WORKERS = 4

#: The whole tiled result stays under this (the vision cap is 32,000 —
#: ``deep.tools.DEFAULT_VISION_RESULT_CHARS``).
TILED_RESULT_CHARS = 30000


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
    """
    key = arguments.get("attachment_key", "")
    page = arguments.get("page", 0)
    prompt = arguments.get("prompt", "Describe the content of this page.")
    tiles, tiles_note, tiles_error = _parse_tiles(arguments.get("tiles", "auto"))
    if tiles_error:
        return json.dumps({"error": tiles_error, "accepted": TILES_ACCEPTED})

    try:
        pdf_bytes, _src = _resolve_attachment_or_path(key, attachments)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})

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
    except ValueError as e:
        return json.dumps({"error": str(e)})

    lines = _lines_for_context(pdf_bytes, page)
    if arguments.get("_inline"):
        # The main model looks at the whole page itself and zooms with
        # render_region where the lettering is small: no tiles.
        return json.dumps(_inline_result(image_bytes, info, page, engine,
                                         lines))
    try:
        result = engine.analyze_image(
            image_bytes, _vision_prompt(prompt, info["clip"], lines,
                                        _sent_size(info)))
    except (NotImplementedError, AttributeError) as e:
        return json.dumps({
            "error": f"Vision not available on this engine: {e}"
        })
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    out = {"page": page, **_pdf_page(page), "analysis": result,
           **vision_view.view_payload(info, engine)}
    _finish_answer(out, info)
    if tiles_note:
        out["tiles_note"] = tiles_note

    n = _tile_count(tiles, info)
    if n > 1:
        out["tiles"] = _read_tiles(pdf_bytes, page, info, n, prompt, engine,
                                   lines)
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


def _read_tiles(pdf_bytes, page, info, n, prompt, engine, lines=None):
    """Read the page in ``n`` x ``n`` overlapping tiles, in parallel."""
    from concurrent.futures import ThreadPoolExecutor
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
                   "analysis": text}
            _finish_answer(row, tinfo, note=False)
            row.pop("located_note", None)
            return row
        except Exception as exc:                  # one tile, not the page
            return {"tile": f"r{r + 1}c{c + 1}",
                    "view": [round(v, 1) for v in boxes[k]],
                    "error": f"{type(exc).__name__}: {exc}"}

    # Each tile's call runs in a copy of the caller's context, so the run's
    # callbacks (the activity log, the turn's token count) see it; a plain
    # thread pool starts every worker with an empty context.
    import contextvars
    with ThreadPoolExecutor(max_workers=TILE_WORKERS) as ex:
        futs = [ex.submit(contextvars.copy_context().run, one, k)
                for k in range(n * n)]
        return [f.result() for f in futs]


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
    key = arguments.get("attachment_key", "")
    page = arguments.get("page", 0)
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
    save_dir = None
    try:
        from funhouse_agent._fileio import default_output_dir
        # The conversation's working folder (the host sets it); with none,
        # the contact sheets are not written anywhere.
        save_dir = default_output_dir() or None
    except Exception:
        save_dir = None
    try:
        out = _fl.find_like(
            pdf_bytes, int(page), [float(v) for v in bbox], engine,
            text=(arguments.get("text") or None),
            pages=arguments.get("pages"),
            include_legend=bool(arguments.get("include_legend", False)),
            threshold=arguments.get("threshold"),
            save_dir=save_dir)
    except ImportError as e:
        return json.dumps({"error": f"find_like cannot run here: {e}"})
    except (ValueError, IndexError) as e:
        return json.dumps({"error": str(e),
                           "hint": "box the mark's lettering tightly, with no "
                                   "leader or table rule inside the box"})
    return json.dumps(out)


_READ_OFF_NOTE = (
    "Value(s) are a vision read-off estimate from the chart — accurate to a few "
    "percent on linear axes, looser on log axes or dense curve families. Verify "
    "against a closed-form or digitized method where one exists."
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
        "source": str(pdf_abs),
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
        # render_region on the same PDF page.
        "source": str(pdf_abs),
        "page": page_idx,
        **vision_view.view_payload(info, engine),
    }
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
                                 warnings=warnings)
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
        rescue_write, workspace_write_hint, written_file_problem,
        workspace_api_upload, _is_workspace_path,
    )

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
    return json.dumps(result)

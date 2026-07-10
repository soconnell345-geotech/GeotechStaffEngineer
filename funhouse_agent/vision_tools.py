"""
Extended tool definitions and dispatch for GeotechAgent.

Extends the standard 4 ReAct tools (call_agent, list_methods, describe_method,
list_agents) with vision-capable tools and file output tools.
"""

import json
import os
import re
from typing import Any, Callable, Dict, Optional


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
    "read_pdf_text",
    "analyze_image",
    "analyze_pdf_page",
    "read_reference_figure",
    "save_file",
}

VISION_TOOL_DESCRIPTIONS = """
### 5. read_pdf_text
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

### 6. analyze_image
Analyze an attached image using vision. Returns text description/analysis.
`attachment_key` is an attachment key OR a real filesystem path.
```
<tool_call>
{"tool_name": "analyze_image", "attachment_key": "site_plan", "prompt": "Extract the cross-section geometry"}
</tool_call>
```

### 7. analyze_pdf_page
Render ONE PDF page and analyze it using vision — for a SCANNED page, a figure,
a boring-log sheet, or a plotted cross-section. For a text-layer report prefer
`read_pdf_text` (cheaper). `attachment_key` is an attachment key OR a real path.
```
<tool_call>
{"tool_name": "analyze_pdf_page", "attachment_key": "report", "page": 0, "prompt": "Extract geometry from this cross-section"}
</tool_call>
```

### 8. read_reference_figure
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

### 9. save_file
Save raw text or data to a file. Returns the saved file path.
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
    if tool_name == "read_pdf_text":
        return _dispatch_read_pdf_text(arguments, attachments)
    elif tool_name == "analyze_image":
        return _dispatch_analyze_image(arguments, engine, attachments)
    elif tool_name == "analyze_pdf_page":
        return _dispatch_analyze_pdf_page(arguments, engine, attachments)
    elif tool_name == "read_reference_figure":
        return _dispatch_read_reference_figure(arguments, engine)
    elif tool_name == "save_file":
        return _dispatch_save_file(arguments, save_fn or _default_save_fn)
    else:
        return json.dumps({"error": f"Unknown extended tool: {tool_name}"})


# Keep old name as alias for backwards compatibility
dispatch_vision_tool = dispatch_extended_tool


# ---------------------------------------------------------------------------
# Source resolution: attachment key OR real filesystem path
# ---------------------------------------------------------------------------

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
    if key and os.path.isfile(key):
        try:
            with open(key, "rb") as fh:
                return fh.read(), "path"
        except OSError as e:
            raise FileNotFoundError(f"'{key}' exists but could not be read: {e}")
    available = sorted(attachments.keys())
    raise FileNotFoundError(
        f"'{key}' not found as an attachment key or a readable file path. "
        f"Available attachment keys: {available}. Real filesystem paths are "
        f"also accepted (driver-local /tmp/... or a /Volumes/... path; "
        f"/Workspace reads are unreliable on Databricks)."
    )


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
        image_data, _src = _resolve_attachment_or_path(key, attachments)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})

    try:
        result = engine.analyze_image(image_data, prompt)
        return json.dumps({"analysis": result})
    except (NotImplementedError, AttributeError) as e:
        return json.dumps({
            "error": f"Vision not available on this engine: {e}"
        })
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def _dispatch_analyze_pdf_page(arguments, engine, attachments):
    """Handle analyze_pdf_page tool call."""
    key = arguments.get("attachment_key", "")
    page = arguments.get("page", 0)
    prompt = arguments.get("prompt", "Describe the content of this page.")

    try:
        pdf_bytes, _src = _resolve_attachment_or_path(key, attachments)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})

    # Render PDF page to PNG
    try:
        from pdf_import.vision import _render_pdf_page
        image_bytes = _render_pdf_page(content=pdf_bytes, page=page)
    except ImportError:
        return json.dumps({
            "error": "PyMuPDF required for PDF rendering. pip install PyMuPDF"
        })
    except ValueError as e:
        return json.dumps({"error": str(e)})

    try:
        result = engine.analyze_image(image_bytes, prompt)
        return json.dumps({"page": page, "analysis": result})
    except (NotImplementedError, AttributeError) as e:
        return json.dumps({
            "error": f"Vision not available on this engine: {e}"
        })
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


_READ_OFF_NOTE = (
    "Value(s) are a vision read-off estimate from the chart — accurate to a few "
    "percent on linear axes, looser on log axes or dense curve families. Verify "
    "against a closed-form or digitized method where one exists."
)


def _dispatch_read_reference_figure(arguments, engine):
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
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    # Render the page at high DPI for legible curves/labels.
    try:
        from pdf_import.vision import _render_pdf_page
        image_bytes = _render_pdf_page(filepath=str(pdf_abs), page=page_idx, dpi=220)
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
        result = engine.analyze_image(image_bytes, full_prompt)
    except (NotImplementedError, AttributeError) as e:
        return json.dumps({"error": f"Vision not available on this engine: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    return json.dumps({
        "reference": reference,
        "figure_number": rec["figure_number"],
        "caption": rec["caption"],
        "pdf_page_index": page_idx,
        "page_estimated": rec.get("page_estimated", False),
        "analysis": result,
        "note": _READ_OFF_NOTE,
    })


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

    try:
        saved_path = save_fn(path, content)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    # Verify the write on the REAL filesystem so the agent never needs a
    # separate ls/read to trust the save — agent-side filesystem tools may be
    # sandboxed/virtual and cannot see this file. For the default local
    # writer the CONTENT is verified too (Databricks /Workspace FUSE writes
    # can "succeed" while the workspace stores a literal PLACEHOLDER file).
    # A custom save_fn may write somewhere not locally visible (e.g. DBFS),
    # so only flag an error for the default writer.
    from funhouse_agent._fileio import (
        rescue_write, workspace_write_hint, written_file_problem,
    )

    result = {"saved": saved_path}
    if isinstance(saved_path, str):
        abs_path = os.path.abspath(saved_path)
        file_exists = os.path.isfile(abs_path)
        result["saved"] = abs_path if file_exists else saved_path
        result["file_size_bytes"] = (
            os.path.getsize(abs_path) if file_exists else 0)
        if save_fn is _default_save_fn:
            expected = (content if isinstance(content, bytes)
                        else content.encode("utf-8", errors="replace"))
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

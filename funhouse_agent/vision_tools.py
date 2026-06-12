"""
Extended tool definitions and dispatch for GeotechAgent.

Extends the standard 4 ReAct tools (call_agent, list_methods, describe_method,
list_agents) with vision-capable tools and file output tools.
"""

import json
import os
from typing import Any, Callable, Dict, Optional

# Standard 4 ReAct tools (defined locally to avoid foundry import chain)
STANDARD_TOOLS = {"call_agent", "list_methods", "describe_method", "list_agents"}

# Standard tools + vision + output extensions
EXTENDED_TOOLS = STANDARD_TOOLS | {
    "analyze_image",
    "analyze_pdf_page",
    "read_reference_figure",
    "save_file",
}

VISION_TOOL_DESCRIPTIONS = """
### 5. analyze_image
Analyze an attached image using vision. Returns text description/analysis.
```
<tool_call>
{"tool_name": "analyze_image", "attachment_key": "site_plan", "prompt": "Extract the cross-section geometry"}
</tool_call>
```

### 6. analyze_pdf_page
Render a PDF page and analyze it using vision.
```
<tool_call>
{"tool_name": "analyze_pdf_page", "attachment_key": "report", "page": 0, "prompt": "Extract geometry from this cross-section"}
</tool_call>
```

### 7. read_reference_figure
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

### 8. save_file
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
    if tool_name == "analyze_image":
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


def _dispatch_analyze_image(arguments, engine, attachments):
    """Handle analyze_image tool call."""
    key = arguments.get("attachment_key", "")
    prompt = arguments.get("prompt", "Describe this image.")

    if key not in attachments:
        available = list(attachments.keys()) if attachments else []
        return json.dumps({
            "error": f"Attachment '{key}' not found. Available: {available}"
        })

    try:
        result = engine.analyze_image(attachments[key], prompt)
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

    if key not in attachments:
        available = list(attachments.keys()) if attachments else []
        return json.dumps({
            "error": f"Attachment '{key}' not found. Available: {available}"
        })

    pdf_bytes = attachments[key]

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
    # sandboxed/virtual and cannot see this file. A custom save_fn may write
    # somewhere not locally visible (e.g. DBFS), so only flag an error when
    # the DEFAULT local writer produced no file.
    result = {"saved": saved_path}
    if isinstance(saved_path, str):
        abs_path = os.path.abspath(saved_path)
        file_exists = os.path.isfile(abs_path)
        result["saved"] = abs_path if file_exists else saved_path
        result["file_exists"] = file_exists
        result["file_size_bytes"] = (
            os.path.getsize(abs_path) if file_exists else 0)
        if not file_exists:
            if save_fn is _default_save_fn:
                result["error"] = (
                    f"save_file ran but no file was found at '{abs_path}' "
                    "after writing. The target location may not be writable "
                    "from this process; retry with a local path (e.g. under "
                    "/tmp)."
                )
            else:
                result["note"] = (
                    "File was saved via a custom save function; the path is "
                    "not visible on the local filesystem (e.g. remote/DBFS), "
                    "so file_exists/file_size_bytes reflect the local view "
                    "only."
                )
    return json.dumps(result)

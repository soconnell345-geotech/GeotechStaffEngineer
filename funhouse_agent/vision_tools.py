"""
Extended tool definitions and dispatch for GeotechAgent.

Extends the standard 4 ReAct tools (call_agent, list_methods, describe_method,
list_agents) with vision-capable tools and file output tools.
"""

import json
import os
from typing import Any, Callable, Dict, Optional

from chat_agent.parser import VALID_TOOLS

# Standard tools + vision + output extensions
EXTENDED_TOOLS = VALID_TOOLS | {
    "analyze_image",
    "analyze_pdf_page",
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

### 7. save_file
Save content to a file (calc packages, HTML reports, Plotly figures, etc.).
Content can be text (str) or base64-encoded binary. Returns the saved file path.
```
<tool_call>
{"tool_name": "save_file", "path": "output/bearing_capacity_report.html", "content": "<html>...</html>"}
</tool_call>
```
For binary content (e.g. PDF), set encoding to "base64":
```
<tool_call>
{"tool_name": "save_file", "path": "output/report.pdf", "content": "JVBERi0xLjQ...", "encoding": "base64"}
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
        return json.dumps({"saved": saved_path})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

"""
Vision tool definitions and dispatch for GeotechAgent.

Extends the standard 4 ReAct tools (call_agent, list_methods, describe_method,
list_agents) with vision-capable tools that use the engine's analyze_image().
"""

import json
from typing import Any, Dict

from chat_agent.parser import VALID_TOOLS

# Standard tools + vision extensions
EXTENDED_TOOLS = VALID_TOOLS | {
    "analyze_image",
    "analyze_pdf_page",
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
"""


def dispatch_vision_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    engine,
    attachments: Dict[str, bytes],
) -> str:
    """Dispatch a vision tool call.

    Parameters
    ----------
    tool_name : str
        One of the vision tool names.
    arguments : dict
        Tool arguments from the parsed tool call.
    engine : GenAIEngine
        AI engine with analyze_image() capability.
    attachments : dict
        {key: bytes} of attached files.

    Returns
    -------
    str
        JSON string result.
    """
    if tool_name == "analyze_image":
        return _dispatch_analyze_image(arguments, engine, attachments)
    elif tool_name == "analyze_pdf_page":
        return _dispatch_analyze_pdf_page(arguments, engine, attachments)
    else:
        return json.dumps({"error": f"Unknown vision tool: {tool_name}"})


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

"""
OpenAI-native tool schemas and dispatch for NativeToolEngine.

Defines the 7 tools (4 dispatch + 3 extended) as OpenAI function-calling
schemas, and routes native tool_call responses to the existing adapter
dispatch functions.
"""

import json
from typing import Any, Callable, Dict, Optional

from funhouse_agent.dispatch import (
    call_agent, describe_method, list_agents, list_methods,
)

# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_agents",
            "description": (
                "List all available geotechnical analysis modules with "
                "brief descriptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_methods",
            "description": (
                "List available methods for a specific analysis module."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": (
                            "Module name (e.g. 'bearing_capacity', "
                            "'settlement', 'subsurface')."
                        ),
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Optional category filter. Empty string for all."
                        ),
                        "default": "",
                    },
                },
                "required": ["agent_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_method",
            "description": (
                "Get full parameter documentation for a method. Always call "
                "this before using a method for the first time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Module name.",
                    },
                    "method": {
                        "type": "string",
                        "description": "Method name within the module.",
                    },
                },
                "required": ["agent_name", "method"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_agent",
            "description": (
                "Execute a geotechnical calculation. All method-specific "
                "inputs (width, depth, phi, gamma, etc.) MUST go inside "
                "the 'parameters' object — never as top-level arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Module name.",
                    },
                    "method": {
                        "type": "string",
                        "description": "Method name.",
                    },
                    "parameters": {
                        "type": "object",
                        "description": (
                            "Nested dict of method-specific inputs (all SI "
                            "units). Example: {\"width\": 2.0, \"depth\": "
                            "1.5, \"phi\": 30, \"gamma\": 18.0}. Use "
                            "describe_method first to see the required keys."
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": ["agent_name", "method", "parameters"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": (
                "Analyze an attached image using vision. Returns text "
                "description/analysis of the image content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_key": {
                        "type": "string",
                        "description": "Key of the attached image file.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "What to extract or analyze from the image."
                        ),
                        "default": "Describe this image.",
                    },
                },
                "required": ["attachment_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_pdf_page",
            "description": (
                "Render a PDF page and analyze it using vision."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_key": {
                        "type": "string",
                        "description": "Key of the attached PDF file.",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number (0-indexed).",
                        "default": 0,
                    },
                    "prompt": {
                        "type": "string",
                        "description": "What to extract from the page.",
                        "default": "Describe the content of this page.",
                    },
                },
                "required": ["attachment_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_file",
            "description": (
                "Save raw text or data to a file. Returns the saved file "
                "path. For formatted calculation documents, use the "
                "calc_package module via call_agent instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Output file path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content (text or base64).",
                    },
                    "encoding": {
                        "type": "string",
                        "description": (
                            "'text' (default) or 'base64' for binary."
                        ),
                        "default": "text",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]

# Tool names that are dispatched via vision_tools (not dispatch.py)
EXTENDED_TOOL_NAMES = {"analyze_image", "analyze_pdf_page", "save_file"}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch_native_tool(
    name: str,
    arguments: dict,
    engine=None,
    attachments: Optional[Dict[str, bytes]] = None,
    save_fn: Optional[Callable] = None,
) -> str:
    """Execute a native tool call and return a JSON string result.

    Routes to the existing dispatch functions (for the 4 standard tools)
    or vision_tools dispatch (for the 3 extended tools).

    Parameters
    ----------
    name : str
        Tool/function name from the model's tool_call.
    arguments : dict
        Parsed arguments from the model's tool_call.
    engine : GenAIEngine, optional
        Required for analyze_image / analyze_pdf_page.
    attachments : dict, optional
        {key: bytes} of attached files.  Required for vision tools.
    save_fn : callable, optional
        File save function for save_file tool.

    Returns
    -------
    str
        JSON string result.
    """
    if name == "list_agents":
        return json.dumps(list_agents(), default=str)

    if name == "list_methods":
        return json.dumps(
            list_methods(
                agent_name=arguments.get("agent_name", ""),
                category=arguments.get("category", ""),
            ),
            default=str,
        )

    if name == "describe_method":
        return json.dumps(
            describe_method(
                agent_name=arguments.get("agent_name", ""),
                method=arguments.get("method", ""),
            ),
            default=str,
        )

    if name == "call_agent":
        return json.dumps(
            call_agent(
                agent_name=arguments.get("agent_name", ""),
                method=arguments.get("method", ""),
                parameters=arguments.get("parameters", {}),
            ),
            default=str,
        )

    # Extended tools — delegate to vision_tools dispatch
    if name in EXTENDED_TOOL_NAMES:
        from funhouse_agent.vision_tools import (
            dispatch_extended_tool, _default_save_fn,
        )
        return dispatch_extended_tool(
            tool_name=name,
            arguments=arguments,
            engine=engine,
            attachments=attachments or {},
            save_fn=save_fn or _default_save_fn,
        )

    return json.dumps({"error": f"Unknown tool: {name}"})

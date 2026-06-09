"""LangChain tools for the deepagents (v5.0) port of the geotech agent.

Wraps the same dispatch surface the v1 agent uses
(:mod:`funhouse_agent.dispatch` + :mod:`funhouse_agent.vision_tools`) as
LangChain :class:`~langchain_core.tools.StructuredTool` objects so they can be
handed to ``deepagents.create_deep_agent``.

Design goals (mirrors v1):

* Every tool returns a **JSON string** (``json.dumps(..., default=str)``)
  exactly like ``dispatch.dispatch_tool`` / ``native_tools.dispatch_native_tool``.
* ``call_agent`` preserves the flattened-parameter auto-nesting quirk from
  ``native_tools.dispatch_native_tool`` (LLMs frequently hoist method params to
  the top level instead of nesting them under ``parameters``).
* Tools are produced by factories bound to an ``allowed_agents`` scope so the
  references sub-agent sees ``REFERENCE_MODULES`` while the primary sees
  ``ANALYSIS_MODULES`` — the same scoping v1 enforces at the dispatcher.
* Tool descriptions are kept close to the (tuned) wording in
  ``native_tools.OPENAI_TOOLS`` and ``vision_tools.VISION_TOOL_DESCRIPTIONS``.

The vision tools need a GenAIEngine + attachments + save_fn. For offline
construction those are injected at build time (closured); ``read_reference_figure``
and ``save_file`` work without a live engine where possible, and ``analyze_*``
return a clear error if no engine was wired.
"""

import json
from typing import Any, Callable, Dict, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from funhouse_agent.dispatch import (
    call_agent as _call_agent,
    describe_method as _describe_method,
    list_agents as _list_agents,
    list_methods as _list_methods,
)
from funhouse_agent.vision_tools import (
    dispatch_extended_tool as _dispatch_extended_tool,
    _default_save_fn,
)


# ---------------------------------------------------------------------------
# Tool-result size cap (token-bloat guard)
# ---------------------------------------------------------------------------

#: Default cap on the size of a tool result fed back to the model, mirroring
#: v1's ``GeotechAgent._max_result_chars`` (``funhouse_agent/agent.py``). Large
#: reference-text dumps from the references sub-agent would otherwise be
#: re-sent on every ReAct round and compound the input-token cost.
DEFAULT_MAX_RESULT_CHARS = 8000


def _truncate(text: str, max_chars: int) -> str:
    """Truncate a tool-result string to ``max_chars``, marking any cut.

    Mirrors :func:`funhouse_agent.react_support._truncate` (the v1 behavior),
    but appends a count of the dropped characters so the marker is informative.

    Parameters
    ----------
    text : str
        The tool result (already a JSON string in this module).
    max_chars : int
        Maximum number of characters to keep. A value ``<= 0`` disables
        truncation and the full ``text`` is returned unchanged.

    Returns
    -------
    str
        ``text`` unchanged when it is short enough (or truncation is disabled),
        otherwise the first ``max_chars`` characters followed by a
        ``\\n...[truncated N chars]`` marker.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


# ---------------------------------------------------------------------------
# Core dispatch tools (the 4 meta-tools), scoped to allowed_agents
# ---------------------------------------------------------------------------

# Keys that belong to call_agent itself; anything else the model passes at the
# top level is treated as a flattened method parameter and auto-nested.
_CALL_AGENT_KEYS = {"agent_name", "method", "parameters"}


class _CallAgentArgs(BaseModel):
    """Args schema for the call_agent tool.

    ``extra="allow"`` is the key piece: LLMs frequently flatten the method
    parameters to the top level instead of nesting them under ``parameters``.
    A plain ``**kwargs`` function signature does NOT survive
    ``StructuredTool.from_function`` (it collapses to a single ``extra`` dict
    field), so we declare the schema explicitly and let pydantic keep the
    extras — they flow into the function's ``**extra`` and get auto-nested,
    mirroring ``native_tools.dispatch_native_tool``.
    """

    model_config = ConfigDict(extra="allow")

    agent_name: str = Field(description="Module name.")
    method: str = Field(description="Method name.")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Nested dict of method-specific inputs (all SI units). Example: "
            '{"width": 2.0, "depth": 1.5, "friction_angle": 30, '
            '"unit_weight": 18.0}. Use describe_method first to see the '
            "required keys."
        ),
    )


def make_core_tools(
    allowed_agents=None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
) -> list:
    """Build the 4 core dispatch tools bound to an ``allowed_agents`` scope.

    Parameters
    ----------
    allowed_agents : iterable of str, optional
        Whitelist of module names. If provided, modules outside this set are
        invisible to ``list_agents`` / ``list_methods`` / ``describe_method``
        and refused by ``call_agent`` (same semantics as v1
        ``dispatch.dispatch_tool``). ``None`` exposes the full registry.
    max_result_chars : int, optional
        Cap on the size of each tool's JSON-string result fed back to the model
        (default ``8000``, mirroring v1's ``GeotechAgent._max_result_chars``).
        Results longer than this are truncated with a clear marker, which keeps
        large reference-text dumps (especially from the references sub-agent,
        which also uses this factory) from compounding the input-token cost as
        they are re-sent on every round. A value ``<= 0`` disables truncation.

    Returns
    -------
    list[StructuredTool]
        ``[list_agents, list_methods, describe_method, call_agent]``.
    """

    def list_agents() -> str:
        """List all available geotechnical analysis modules with brief
        descriptions."""
        return _truncate(
            json.dumps(_list_agents(allowed_agents=allowed_agents), default=str),
            max_result_chars,
        )

    def list_methods(agent_name: str, category: str = "") -> str:
        """List available methods for a specific analysis module.

        ``agent_name`` is the module name (e.g. 'bearing_capacity',
        'settlement', 'subsurface'). ``category`` is an optional category
        filter; empty string for all.
        """
        return _truncate(
            json.dumps(
                _list_methods(
                    agent_name=agent_name,
                    category=category or "",
                    allowed_agents=allowed_agents,
                ),
                default=str,
            ),
            max_result_chars,
        )

    def describe_method(agent_name: str, method: str) -> str:
        """Get full parameter documentation for a method. Always call this
        before using a method for the first time.

        ``agent_name`` is the module name; ``method`` is the method name
        within that module.
        """
        return _truncate(
            json.dumps(
                _describe_method(
                    agent_name=agent_name,
                    method=method,
                    allowed_agents=allowed_agents,
                ),
                default=str,
            ),
            max_result_chars,
        )

    def call_agent(
        agent_name: str,
        method: str,
        parameters: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> str:
        """Execute a geotechnical calculation. All method-specific inputs
        (width, depth, friction_angle, unit_weight, etc.) MUST go inside the
        'parameters' object (all SI units: meters, kPa, kN, kN/m3, degrees).
        Use describe_method first to see the required keys.

        ``agent_name`` is the module name; ``method`` is the method name.
        ``parameters`` is a nested dict of method-specific inputs, e.g.
        ``{"width": 2.0, "depth": 1.5, "friction_angle": 30,
        "unit_weight": 18.0}``.
        """
        # LLMs sometimes flatten method parameters to the top level instead of
        # nesting them under "parameters". Auto-nest any extra keys — mirrors
        # native_tools.dispatch_native_tool's call_agent handling.
        params = dict(parameters or {})
        extras = {k: v for k, v in extra.items() if k not in _CALL_AGENT_KEYS}
        if extras:
            params.update(extras)
        return _truncate(
            json.dumps(
                _call_agent(
                    agent_name=agent_name,
                    method=method,
                    parameters=params,
                    attachments=None,
                    allowed_agents=allowed_agents,
                ),
                default=str,
            ),
            max_result_chars,
        )

    return [
        StructuredTool.from_function(
            list_agents,
            name="list_agents",
            description=(
                "List all available geotechnical analysis modules with "
                "brief descriptions."
            ),
        ),
        StructuredTool.from_function(
            list_methods,
            name="list_methods",
            description=(
                "List available methods for a specific analysis module."
            ),
        ),
        StructuredTool.from_function(
            describe_method,
            name="describe_method",
            description=(
                "Get full parameter documentation for a method. Always call "
                "this before using a method for the first time."
            ),
        ),
        StructuredTool.from_function(
            call_agent,
            name="call_agent",
            description=(
                "Execute a geotechnical calculation. All method-specific "
                "inputs (width, depth, friction_angle, unit_weight, etc.) "
                "MUST go inside the 'parameters' object — never as top-level "
                "arguments. All units are SI."
            ),
            args_schema=_CallAgentArgs,
        ),
    ]


# ---------------------------------------------------------------------------
# Vision / file-output tools (wrap vision_tools.dispatch_extended_tool)
# ---------------------------------------------------------------------------

def make_vision_tools(
    engine=None,
    attachments: Optional[Dict[str, bytes]] = None,
    save_fn: Optional[Callable] = None,
    include: Optional[set] = None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
) -> list:
    """Build the vision / file-output tools as LangChain tools.

    The engine, attachments, and save_fn are closured in at build time so the
    tools match the no-extra-args calling convention deepagents expects.

    Parameters
    ----------
    engine : GenAIEngine, optional
        Vision-capable engine (``analyze_image``). Required for
        ``analyze_image`` / ``analyze_pdf_page`` / ``read_reference_figure``;
        when ``None`` those tools return a clear "vision not available" error
        instead of raising (so offline construction still works).
    attachments : dict, optional
        ``{key: bytes}`` of attached files (for ``analyze_image`` /
        ``analyze_pdf_page``). May be a live reference the host mutates.
    save_fn : callable, optional
        ``(path, content) -> saved_path``. Defaults to local filesystem write.
    include : set of str, optional
        Subset of tool names to build. Defaults to all four
        (``analyze_image``, ``analyze_pdf_page``, ``read_reference_figure``,
        ``save_file``).
    max_result_chars : int, optional
        Cap on the size of each tool's result fed back to the model (default
        ``8000``, mirroring v1). Results longer than this are truncated with a
        clear marker; a value ``<= 0`` disables truncation. Applied to every
        vision/file tool via the shared ``_dispatch`` helper.

    Returns
    -------
    list[StructuredTool]
    """
    attachments = attachments if attachments is not None else {}
    save_fn = save_fn or _default_save_fn
    include = include if include is not None else {
        "analyze_image", "analyze_pdf_page", "read_reference_figure",
        "save_file",
    }

    def _dispatch(tool_name: str, arguments: dict) -> str:
        return _truncate(
            _dispatch_extended_tool(
                tool_name=tool_name,
                arguments=arguments,
                engine=engine,
                attachments=attachments,
                save_fn=save_fn,
            ),
            max_result_chars,
        )

    def analyze_image(attachment_key: str,
                      prompt: str = "Describe this image.") -> str:
        """Analyze an attached image using vision. Returns text
        description/analysis of the image content.

        ``attachment_key`` is the key of the attached image file; ``prompt``
        is what to extract or analyze from the image.
        """
        return _dispatch(
            "analyze_image",
            {"attachment_key": attachment_key, "prompt": prompt},
        )

    def analyze_pdf_page(
        attachment_key: str,
        page: int = 0,
        prompt: str = "Describe the content of this page.",
    ) -> str:
        """Render a PDF page and analyze it using vision.

        ``attachment_key`` is the key of the attached PDF file; ``page`` is the
        0-indexed page number; ``prompt`` is what to extract from the page.
        """
        return _dispatch(
            "analyze_pdf_page",
            {"attachment_key": attachment_key, "page": page, "prompt": prompt},
        )

    def read_reference_figure(
        reference: str,
        figure_number: str,
        prompt: str = "",
    ) -> str:
        """Render a digitized reference figure (e.g. a DM7 design chart) and
        read a value off it with vision. Use this whenever a numeric value must
        come from a chart — do not read values off a chart from the caption or
        from memory. Find the figure first with ``call_agent`` →
        ``figure_db.figure_search``, then pass its ``reference`` +
        ``figure_number`` here with a ``prompt`` describing the value(s) you
        need. Returns a chart read-off estimate — verify it against a
        closed-form/digitized method where one exists.
        """
        return _dispatch(
            "read_reference_figure",
            {
                "reference": reference,
                "figure_number": figure_number,
                "prompt": prompt,
            },
        )

    def save_file(path: str, content: str, encoding: str = "text") -> str:
        """Save raw text or data to a file. Returns the saved file path. For
        formatted calculation documents, use the ``calc_package`` module via
        ``call_agent`` instead.

        ``path`` is the output file path; ``content`` is the file content
        (text or base64); ``encoding`` is 'text' (default) or 'base64' for
        binary.
        """
        return _dispatch(
            "save_file",
            {"path": path, "content": content, "encoding": encoding},
        )

    _builders = {
        "analyze_image": (
            analyze_image,
            "Analyze an attached image using vision. Returns text "
            "description/analysis of the image content.",
        ),
        "analyze_pdf_page": (
            analyze_pdf_page,
            "Render a PDF page and analyze it using vision.",
        ),
        "read_reference_figure": (
            read_reference_figure,
            "Render a digitized reference figure (e.g. a DM7 design chart) "
            "and read a value off it with vision. Find the figure first via "
            "figure_db.figure_search, then read the value off the actual "
            "chart. Returns a chart read-off estimate.",
        ),
        "save_file": (
            save_file,
            "Save raw text or data to a file. Returns the saved file path. "
            "For formatted calculation documents, use the calc_package module "
            "via call_agent instead.",
        ),
    }

    tools = []
    for name, (fn, desc) in _builders.items():
        if name in include:
            tools.append(
                StructuredTool.from_function(fn, name=name, description=desc)
            )
    return tools


__all__ = [
    "make_core_tools",
    "make_vision_tools",
    "DEFAULT_MAX_RESULT_CHARS",
]

"""Panel chatbot interface for GeotechAgent.

Browser-based chat UI that works standalone or inline in Databricks notebooks.
Uses Panel's built-in ChatInterface for native chat UX with streaming,
file upload, and async support.

Usage (standalone)::

    from funhouse_agent import GeotechAgent, ClaudeEngine
    from funhouse_agent.panel_chat import ChatApp

    agent = GeotechAgent(genai_engine=ClaudeEngine())
    chat = ChatApp(agent)
    chat.show(port=8052)      # opens browser

Usage (Databricks notebook)::

    import panel as pn
    pn.extension()

    from funhouse_agent import GeotechAgent
    from funhouse_agent.panel_chat import ChatApp

    agent = GeotechAgent(genai_engine=prompter_api)
    chat = ChatApp(agent)
    chat.panel()              # inline display

Requires: ``pip install panel``
"""

import json
import os
import threading
from typing import Optional

import panel as pn

from funhouse_agent.agent import GeotechAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_summary(tool_name: str, arguments: dict) -> str:
    """Build a short one-line summary for a tool call."""
    if tool_name == "call_agent":
        return (f"call_agent({arguments.get('agent_name', '?')}, "
                f"{arguments.get('method', '?')})")
    elif tool_name == "list_methods":
        return f"list_methods({arguments.get('agent_name', '?')})"
    elif tool_name == "describe_method":
        return (f"describe_method({arguments.get('agent_name', '?')}, "
                f"{arguments.get('method', '?')})")
    elif tool_name == "analyze_image":
        return f"analyze_image({arguments.get('attachment_key', '?')})"
    elif tool_name == "save_file":
        return f"save_file({arguments.get('path', '?')})"
    else:
        return tool_name


def _format_tool_call(tool_name: str, arguments: dict,
                      result_preview: str) -> str:
    """Format a tool call as a collapsible markdown block."""
    summary = _tool_summary(tool_name, arguments)
    args_text = json.dumps(arguments, indent=2, default=str)
    return (
        f"<details><summary><b>Tool:</b> <code>{summary}</code></summary>\n\n"
        f"```json\n{args_text}\n```\n\n"
        f"**Result preview:**\n```\n{result_preview}\n```\n"
        f"</details>"
    )


def _format_file_output(path: str, fmt: str) -> str:
    """Format a file output notification."""
    return f"**Output:** `{path}` ({fmt})"


# ---------------------------------------------------------------------------
# ChatApp
# ---------------------------------------------------------------------------

class ChatApp:
    """Panel chat interface wrapping a :class:`GeotechAgent`.

    Parameters
    ----------
    agent : GeotechAgent
        Pre-configured agent instance (works with PrompterAPI or ClaudeEngine).
    title : str
        Header title displayed above the chat area.
    height : int
        Height in pixels for the chat area.
    """

    def __init__(
        self,
        agent: GeotechAgent,
        title: str = "GeotechAgent Chat",
        height: int = 550,
    ):
        self._agent = agent
        self._title = title
        self._height = height
        self._tool_calls: list[dict] = []
        self._output_files: list[dict] = []

        # Build Panel widgets
        self._chat = pn.chat.ChatInterface(
            callback=self._on_message,
            callback_user="Agent",
            show_rerun=False,
            show_undo=False,
            sizing_mode="stretch_width",
            height=height,
            placeholder_text="Ask a geotechnical question...",
        )

        self._file_input = pn.widgets.FileInput(
            accept=".png,.jpg,.jpeg,.pdf,.tif,.tiff",
            multiple=True,
            name="Attach files",
            width=250,
        )
        self._file_input.param.watch(self._on_upload, "value")

        self._attachment_badges = pn.pane.HTML("", sizing_mode="stretch_width")

        self._stats = pn.pane.HTML(
            self._render_stats(),
            sizing_mode="stretch_width",
        )

        self._reset_btn = pn.widgets.Button(
            name="Reset", button_type="warning", width=80,
        )
        self._reset_btn.on_click(self._on_reset)

        self._layout = pn.Column(
            pn.pane.HTML(
                f'<div style="padding:8px 0;border-bottom:1px solid #e0e0e0;">'
                f'<h3 style="margin:0;color:#1e40af;font-size:16px;">'
                f'{title}</h3>'
                f'<div style="color:#666;font-size:12px;margin-top:2px;">'
                f'50 geotechnical modules | Text + Vision | ReAct agent'
                f'</div></div>',
                sizing_mode="stretch_width",
            ),
            self._stats,
            self._chat,
            pn.Row(
                self._file_input,
                self._attachment_badges,
                self._reset_btn,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
            max_width=900,
        )

    # -- Public API ---------------------------------------------------------

    def panel(self):
        """Return the Panel layout for inline notebook display.

        Usage in a Databricks/Jupyter cell::

            chat.panel()
        """
        return self._layout

    def show(self, port: int = 8052, **kwargs):
        """Launch as a standalone browser app.

        Parameters
        ----------
        port : int
            Port number (default 8052).
        **kwargs
            Passed to ``pn.serve()``.
        """
        pn.serve(self._layout, port=port, show=True, title=self._title,
                 **kwargs)

    def servable(self):
        """Make the layout servable for ``panel serve dash_chat.py``.

        Returns the layout with ``.servable()`` called.
        """
        return self._layout.servable(title=self._title)

    @property
    def output_files(self) -> list[str]:
        """File paths produced during this session."""
        return [f["path"] for f in self._output_files]

    # -- Callbacks ----------------------------------------------------------

    def _on_message(self, contents: str, user: str, instance: pn.chat.ChatInterface):
        """Called when the user sends a message.

        Runs agent.ask() synchronously (Panel handles threading).
        Yields intermediate status and tool call messages.
        """
        if not contents or not contents.strip():
            return

        question = contents.strip()

        # Capture tool calls during this query
        self._tool_calls = []
        files_before = len(self._output_files)

        original_cb = self._agent._on_tool_call
        self._agent._on_tool_call = self._capture_tool_call

        try:
            result = self._agent.ask(question)
        except Exception as exc:
            self._agent._on_tool_call = original_cb
            yield f"**Error:** {type(exc).__name__}: {exc}"
            return
        finally:
            self._agent._on_tool_call = original_cb

        # Yield tool calls as collapsible details
        if self._tool_calls:
            tool_md = "\n\n".join(
                _format_tool_call(
                    tc["tool_name"], tc["arguments"], tc["result_preview"],
                )
                for tc in self._tool_calls
            )
            yield {"user": "Agent", "value": tool_md}

        # Yield output file notifications
        for f in self._output_files[files_before:]:
            yield {"user": "Agent",
                    "value": _format_file_output(f["path"], f["format"])}

        # Update stats
        self._stats.object = self._render_stats(result)

        # Yield the final answer
        yield result.answer

    def _capture_tool_call(self, tool_name, arguments, result_str):
        """Injected into agent._on_tool_call during a query."""
        self._tool_calls.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "result_preview": result_str[:300],
        })

        # Detect output files
        try:
            result_data = json.loads(result_str)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(result_data, dict):
            return
        if tool_name == "save_file" and "saved" in result_data:
            ext = ""
            path_arg = arguments.get("path", "")
            if "." in path_arg:
                ext = path_arg.rsplit(".", 1)[-1]
            self._output_files.append({
                "path": result_data["saved"], "format": ext or "file",
            })
        if "output_path" in result_data and result_data.get("status") == "success":
            self._output_files.append({
                "path": result_data["output_path"],
                "format": result_data.get("format", "html"),
            })

    def _on_upload(self, event):
        """Handle file upload — add as agent attachments."""
        if not event.new:
            return

        # FileInput with multiple=True: value is bytes of last file,
        # filename is str or list. We use the param values directly.
        filename = self._file_input.filename
        value = self._file_input.value

        if isinstance(filename, str):
            # Single file
            self._agent.add_attachment(filename, value)
        elif isinstance(filename, list):
            # Multiple files — value is bytes of last only with FileInput.
            # For multiple, we get only the last file.
            # Note: Panel FileInput multiple=True gives last file in .value.
            # For full multi-file, user uploads one at a time.
            self._agent.add_attachment(filename[-1], value)

        self._update_badges()

    def _on_reset(self, event):
        """Clear conversation, attachments, and history."""
        self._agent.reset()
        self._tool_calls.clear()
        self._output_files.clear()
        self._chat.clear()
        self._attachment_badges.object = ""
        self._stats.object = self._render_stats()

    # -- Rendering ----------------------------------------------------------

    def _render_stats(self, result=None) -> str:
        """Render stats bar HTML."""
        tokens = self._agent.history.token_estimate()
        turns = len(self._agent.history)
        parts = [f"Tokens: ~{tokens:,}", f"Turns: {turns}"]
        if result:
            parts.append(
                f"Last: {result.rounds} rounds, "
                f"{result.total_time_s:.1f}s"
            )
            if result.errors:
                parts.append(f"{len(result.errors)} error(s)")
        return (
            f'<div style="font-size:11px;color:#888;font-family:monospace;'
            f'padding:4px 8px;">{" | ".join(parts)}</div>'
        )

    def _update_badges(self):
        """Refresh attachment badge display."""
        keys = list(self._agent.attachments.keys())
        if keys:
            badges = " ".join(
                f'<span style="background:#e0e0e0;padding:2px 8px;'
                f'border-radius:4px;font-size:12px;margin:2px;">'
                f'{k}</span>'
                for k in keys
            )
            self._attachment_badges.object = badges
        else:
            self._attachment_badges.object = ""


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting GeotechAgent Panel Chat...")
    print("Initializing ClaudeEngine...")

    from funhouse_agent import GeotechAgent, ClaudeEngine

    _agent = GeotechAgent(
        genai_engine=ClaudeEngine(),
        verbose=True,
        review=False,
    )
    _chat = ChatApp(_agent)
    _chat.show(port=8052)

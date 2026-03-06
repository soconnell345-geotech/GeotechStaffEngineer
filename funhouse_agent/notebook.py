"""ipywidgets chat interface for GeotechAgent in Jupyter/Databricks notebooks.

Usage::

    from funhouse_agent import GeotechAgent
    from funhouse_agent.notebook import NotebookChat

    agent = GeotechAgent(genai_engine=prompter_api)
    chat = NotebookChat(agent)
    chat.display()

    # After running queries:
    chat.output_files        # list of produced file paths
    chat.preview_file(path)  # inline HTML preview of a calc package

Requires: ``pip install ipywidgets``
"""

import html as _html
import json
import re
from typing import Optional

import ipywidgets as widgets

from funhouse_agent.agent import GeotechAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _escape(text: str) -> str:
    """HTML-escape text for safe rendering."""
    return _html.escape(str(text))


def _format_answer(text: str) -> str:
    """Light markdown-to-HTML for agent answers (bold + line breaks)."""
    text = _escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = text.replace("\n", "<br>")
    return text


# Inline CSS for chat bubbles — works in Databricks and local Jupyter.
_CSS = """<style>
.nb-msg{margin:6px 0;padding:8px 12px;border-radius:8px;
        font-family:sans-serif;font-size:13px;line-height:1.5}
.nb-user{background:#e3f2fd}
.nb-agent{background:#f5f5f5}
.nb-status{color:#888;font-style:italic}
.nb-tool{background:#fff8e1;font-size:12px;margin-left:16px;
         padding:4px 8px;border-radius:6px;margin-top:2px;margin-bottom:2px}
.nb-file{background:#e8f5e9;padding:6px 12px;margin-left:16px;
         border-radius:6px;font-size:12px;margin-top:2px;margin-bottom:2px}
.nb-tool summary{cursor:pointer}
</style>"""


# ---------------------------------------------------------------------------
# NotebookChat
# ---------------------------------------------------------------------------

class NotebookChat:
    """ipywidgets chat interface wrapping a :class:`GeotechAgent`.

    Parameters
    ----------
    agent : GeotechAgent
        Pre-configured agent instance.
    height : str
        CSS height for the scrollable chat area (default ``"500px"``).
    """

    def __init__(self, agent: GeotechAgent, height: str = "500px"):
        self._agent = agent
        self._height = height

        # State
        self._messages: list[dict] = []
        self._output_files: list[dict] = []
        self._pending_tool_calls: list[dict] = []
        self._is_processing: bool = False

        # Inject our callback (chain to any existing one)
        self._original_on_tool_call = agent._on_tool_call
        agent._on_tool_call = self._handle_tool_call

        self._build_widgets()
        self._wire_events()

    # -- public API ---------------------------------------------------------

    def display(self):
        """Return the widget container for notebook display."""
        return self._container

    def _repr_mimebundle_(self, **kwargs):
        """Auto-display when the object is the last expression in a cell."""
        return self._container._repr_mimebundle_(**kwargs)

    @property
    def output_files(self) -> list[str]:
        """File paths produced during this session."""
        return [f["path"] for f in self._output_files]

    def attach(self, path: str, key: str = None):
        """Attach a file from the filesystem (local, DBFS, workspace).

        Parameters
        ----------
        path : str
            File path (e.g. ``"/dbfs/my_project/plan.png"`` or local path).
        key : str, optional
            Attachment key for the agent.  Defaults to the filename.
        """
        import os
        if key is None:
            key = os.path.basename(path)
        with open(path, "rb") as f:
            data = f.read()
        self._agent.add_attachment(key, data)
        self._update_attachment_badges()

    def preview_file(self, path: str):
        """Render an HTML file inline in the notebook."""
        try:
            from IPython.display import HTML, display as ipy_display
        except ImportError:
            print("IPython not available.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            ipy_display(HTML(content))
        except FileNotFoundError:
            print(f"File not found: {path}")

    # -- widget construction ------------------------------------------------

    def _build_widgets(self):
        self._stats_html = widgets.HTML(
            value=self._render_stats(),
            layout=widgets.Layout(padding="4px 8px"),
        )

        self._chat_html = widgets.HTML(
            value="",
            layout=widgets.Layout(
                height=self._height,
                overflow_y="auto",
                border="1px solid #ddd",
                padding="8px",
            ),
        )

        self._upload = widgets.FileUpload(
            accept=".png,.jpg,.jpeg,.pdf,.tif,.tiff",
            multiple=True,
            description="Attach",
            layout=widgets.Layout(width="auto"),
        )
        self._attachment_badges = widgets.HTML(value="")
        upload_row = widgets.HBox(
            [self._upload, self._attachment_badges],
            layout=widgets.Layout(padding="4px 0"),
        )

        self._input = widgets.Text(
            placeholder="Ask a geotechnical question...",
            continuous_update=False,
            layout=widgets.Layout(flex="1"),
        )
        self._send_btn = widgets.Button(
            description="Send",
            button_style="primary",
            layout=widgets.Layout(width="80px"),
        )
        self._reset_btn = widgets.Button(
            description="Reset",
            button_style="warning",
            tooltip="Clear conversation and attachments",
            layout=widgets.Layout(width="80px"),
        )
        input_row = widgets.HBox(
            [self._input, self._send_btn, self._reset_btn],
            layout=widgets.Layout(padding="4px 0"),
        )

        self._container = widgets.VBox([
            self._stats_html,
            self._chat_html,
            upload_row,
            input_row,
        ])

    def _wire_events(self):
        self._send_btn.on_click(self._on_send)
        self._input.observe(self._on_input_change, names="value")
        self._upload.observe(self._on_upload_change, names="value")
        self._reset_btn.on_click(self._on_reset)

    # -- event handlers -----------------------------------------------------

    def _on_input_change(self, change):
        """Trigger send when Enter is pressed (continuous_update=False)."""
        if change["new"]:
            self._on_send()

    def _on_send(self, _=None):
        question = self._input.value.strip()
        if not question or self._is_processing:
            return

        self._is_processing = True
        self._send_btn.disabled = True
        self._input.value = ""

        # Show user message + thinking indicator
        self._messages.append({"role": "user", "content": question})
        self._messages.append({"role": "status", "content": "Thinking..."})
        self._refresh_chat()

        # Track which output files are new from this query
        self._pending_tool_calls = []
        files_before = len(self._output_files)

        try:
            result = self._agent.ask(question)
        except Exception as exc:
            result = None
            # Remove thinking indicator and show error
            self._messages = [
                m for m in self._messages
                if not (m["role"] == "status" and m["content"] == "Thinking...")
            ]
            self._messages.append({
                "role": "assistant",
                "content": f"Error: {type(exc).__name__}: {exc}",
            })

        if result is not None:
            # Remove thinking indicator
            self._messages = [
                m for m in self._messages
                if not (m["role"] == "status" and m["content"] == "Thinking...")
            ]
            # Append tool calls
            for tc in self._pending_tool_calls:
                self._messages.append({"role": "tool", **tc})
            # Append new output files
            for f in self._output_files[files_before:]:
                self._messages.append({"role": "file", **f})
            # Append agent answer
            self._messages.append({"role": "assistant", "content": result.answer})
            self._update_stats(result)

        self._refresh_chat()
        self._is_processing = False
        self._send_btn.disabled = False

    def _on_reset(self, _=None):
        self._agent.reset()
        self._messages.clear()
        self._output_files.clear()
        self._pending_tool_calls.clear()
        self._refresh_chat()
        self._update_stats()
        self._attachment_badges.value = ""

    def _on_upload_change(self, change):
        if not change["new"]:
            return
        value = change["new"]
        if isinstance(value, dict):
            # ipywidgets 7.x: {name: {"content": bytes}}
            for name, info in value.items():
                self._agent.add_attachment(name, info["content"])
        else:
            # ipywidgets 8.x: tuple of FileInfo dicts
            for file_info in value:
                name = file_info.get("name", "file")
                content = file_info.get("content", b"")
                self._agent.add_attachment(name, bytes(content))
        self._update_attachment_badges()

    # -- agent callback -----------------------------------------------------

    def _handle_tool_call(self, tool_name: str, arguments: dict,
                          result_str: str):
        """Injected into agent._on_tool_call to capture tool calls."""
        # Chain to original callback
        if self._original_on_tool_call:
            self._original_on_tool_call(tool_name, arguments, result_str)

        self._pending_tool_calls.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "result_preview": result_str[:300],
        })

        # Detect output files
        try:
            result = json.loads(result_str)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(result, dict):
            return

        # save_file tool returns {"saved": path}
        if tool_name == "save_file" and "saved" in result:
            ext = ""
            path_arg = arguments.get("path", "")
            if "." in path_arg:
                ext = path_arg.rsplit(".", 1)[-1]
            self._output_files.append({
                "path": result["saved"],
                "format": ext or "file",
            })

        # calc_package adapter returns {"output_path": path, "status": "success"}
        if "output_path" in result and result.get("status") == "success":
            self._output_files.append({
                "path": result["output_path"],
                "format": result.get("format", "html"),
            })

    # -- rendering ----------------------------------------------------------

    def _render_stats(self, result=None) -> str:
        tokens = self._agent.history.token_estimate()
        turns = len(self._agent.history)
        rounds = result.rounds if result else 0
        elapsed = result.total_time_s if result else 0.0
        last_part = (
            f" | Last: {rounds} rounds, {elapsed:.1f}s" if result else ""
        )
        return (
            f'<div style="font-size:12px;color:#666;font-family:monospace;">'
            f"Tokens: ~{tokens:,} | Turns: {turns}{last_part}"
            f"</div>"
        )

    def _update_stats(self, result=None):
        self._stats_html.value = self._render_stats(result)

    def _update_attachment_badges(self):
        keys = list(self._agent.attachments.keys())
        if keys:
            badges = " ".join(
                f'<span style="background:#e0e0e0;padding:2px 6px;'
                f'border-radius:4px;font-size:12px;margin:2px;">'
                f"{_escape(k)}</span>"
                for k in keys
            )
            self._attachment_badges.value = badges
        else:
            self._attachment_badges.value = ""

    def _render_chat_html(self) -> str:
        parts = [_CSS]
        for msg in self._messages:
            role = msg["role"]
            if role == "user":
                parts.append(
                    f'<div class="nb-msg nb-user">'
                    f'<b>You:</b> {_escape(msg["content"])}</div>'
                )
            elif role == "assistant":
                parts.append(
                    f'<div class="nb-msg nb-agent">'
                    f'<b>Agent:</b> {_format_answer(msg["content"])}</div>'
                )
            elif role == "tool":
                parts.append(self._render_tool_call(msg))
            elif role == "file":
                parts.append(self._render_file_link(msg))
            elif role == "status":
                parts.append(
                    f'<div class="nb-msg nb-status">'
                    f'<em>{_escape(msg["content"])}</em></div>'
                )
        return "\n".join(parts)

    def _render_tool_call(self, msg: dict) -> str:
        name = msg.get("tool_name", "unknown")
        args = msg.get("arguments", {})
        preview = msg.get("result_preview", "")

        # Build a short summary line
        if name == "call_agent":
            summary = (
                f"call_agent({args.get('agent_name', '?')}, "
                f"{args.get('method', '?')})"
            )
        elif name == "list_methods":
            summary = f"list_methods({args.get('agent_name', '?')})"
        elif name == "describe_method":
            summary = (
                f"describe_method({args.get('agent_name', '?')}, "
                f"{args.get('method', '?')})"
            )
        elif name == "analyze_image":
            summary = f"analyze_image({args.get('attachment_key', '?')})"
        elif name == "save_file":
            summary = f"save_file({args.get('path', '?')})"
        else:
            summary = name

        args_json = json.dumps(args, indent=2, default=str)
        return (
            f'<details class="nb-tool">'
            f"<summary>Tool: {_escape(summary)}</summary>"
            f'<pre style="font-size:11px;max-height:200px;overflow:auto;">'
            f"Arguments:\n{_escape(args_json)}\n\n"
            f"Result preview:\n{_escape(preview)}"
            f"</pre></details>"
        )

    def _render_file_link(self, msg: dict) -> str:
        path = msg.get("path", "")
        fmt = msg.get("format", "")
        return (
            f'<div class="nb-file">'
            f"<b>Output:</b> <code>{_escape(path)}</code>"
            f" ({_escape(fmt)})</div>"
        )

    def _refresh_chat(self):
        self._chat_html.value = self._render_chat_html()

"""Keep the deepagents scratch-filesystem tools off real paths.

deepagents gives every agent ``ls``, ``read_file``, ``write_file``,
``edit_file``, ``glob`` and ``grep`` over an in-memory scratch space. Pointed
at a real file they answer "not found" or "No matches found", which read as
true answers. In the 2026-09-15 Nairobi session agents ran them about 35 times
on real files (25+ greps over a 260-page PDF, all empty), and a sub-agent that
"could not find" the earlier report on disk rebuilt it with placeholder text
where the numbers belonged (field feedback N1 and N12). Prompts already said
the scratch space is not the disk; sub-agents never saw those prompts.

:class:`ScratchFilesystemGuard` intercepts those tool calls when the path is a
real one and answers with what to use instead. Scratch paths (anything the
agent wrote itself, ``/memories/``, deepagents' evicted large results) pass
through untouched.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from langchain_core.messages import ToolMessage

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:  # pragma: no cover - older layout
    from langchain.agents.middleware.types import AgentMiddleware

#: Scratch tool -> the argument that carries its path.
PATH_ARG = {"ls": "path", "read_file": "file_path", "write_file": "file_path",
            "edit_file": "file_path", "glob": "path", "grep": "path"}

#: Roots that are never scratch paths.
REAL_ROOTS = ("/tmp", "/root", "/home", "/Workspace", "/Volumes", "/dbfs",
              "/mnt", "/var", "/opt", "/databricks", "/Users")

_WINDOWS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def looks_real(path: str) -> bool:
    """True for a path under a real root (or a Windows drive/UNC path)."""
    p = (path or "").strip()
    if not p or p in ("/", "."):
        return False
    if _WINDOWS_PATH.match(p):
        return True
    norm = p.replace("\\", "/")
    return any(norm == r or norm.startswith(r + "/") for r in REAL_ROOTS)


def _in_scratch(state, path: str) -> bool:
    try:
        files = state.get("files") if state is not None else None
    except Exception:  # noqa: BLE001
        files = None
    if not files:
        return False
    p = path.rstrip("/") or "/"
    if p in files:
        return True
    prefix = p + "/"
    return any(str(k).startswith(prefix) for k in files)


def _message(tool: str, path: str) -> str:
    where = f"'{path}' is on the real disk"
    if tool == "read_file":
        return (f"read_file only reads this agent's scratch notes; {where}. "
                "Read it with read_text_file (HTML, TXT, CSV, JSON, MD), "
                "read_pdf_text (PDF) or analyze_image (image).")
    if tool in ("ls", "glob"):
        return (f"{tool} only lists this agent's scratch notes; {where}. "
                "Use list_files to see real folders.")
    if tool == "grep":
        return (f"grep only searches this agent's scratch notes; {where}, so "
                "'No matches' would have meant nothing. For a PDF use "
                "search_document after open_document, or read_pdf_text; for a "
                "text file use read_text_file.")
    return (f"{tool} only writes to scratch space, which the user never sees; "
            f"{where}. To save a real file use save_file, or pass output_path "
            "to the tool that builds it.")


class ScratchFilesystemGuard(AgentMiddleware):
    """Answer scratch-filesystem calls on real paths with the right tool."""

    def _intercept(self, request) -> Optional[ToolMessage]:
        call = getattr(request, "tool_call", None) or {}
        name = call.get("name")
        arg = PATH_ARG.get(name)
        if not arg:
            return None
        path = (call.get("args") or {}).get(arg)
        if not isinstance(path, str) or not path.strip():
            return None
        if _in_scratch(getattr(request, "state", None), path):
            return None
        if name == "write_file":
            hit = looks_real(path)
        else:
            hit = looks_real(path) or (path.strip() != "/"
                                       and os.path.exists(path))
        if not hit:
            return None
        return ToolMessage(content=_message(name, path),
                           tool_call_id=call.get("id") or "", name=name,
                           status="error")

    def wrap_tool_call(self, request, handler):
        blocked = self._intercept(request)
        return blocked if blocked is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        blocked = self._intercept(request)
        return blocked if blocked is not None else await handler(request)


__all__ = ["ScratchFilesystemGuard", "looks_real", "PATH_ARG", "REAL_ROOTS"]

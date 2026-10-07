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
agent wrote itself, ``/memories/``, deepagents' evicted large results, and the
scratch root itself — ``/`` or ``.``) pass through untouched.

It also answers a ``read_file`` that finds nothing. deepagents saves a tool
result too large to return under ``/large_tool_results/<tool_call_id>`` and
says so in that tool's message; the 2026-10 Foundry eval caught the model
reading ``/large_tool_results/<id>.txt`` — a name deepagents never writes (it
turns every '.' in an id into '_') — straight after a SMALL result that was
never saved at all, and getting only a bare "not found". A missed read now
resolves to the saved file it plainly meant (the same name without the
extension), or says what IS saved and that the result it wants is the tool
message itself.

And it explains an EMPTY scratch search. Field session 2026-10-06: asked for
two published sources, the references sub-agent ran ``ls('/')`` and four
``grep`` calls over the empty scratch space, got "No files found" / "No
matches found", and reported that its searches had found nothing -- it never
called a reference tool. An ``ls`` / ``glob`` / ``grep`` that finds nothing
now says what it searched and where to search instead
(:data:`EMPTY_SCRATCH_NOTE`).
"""

from __future__ import annotations

import os
import posixpath
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

#: deepagents' own scratch areas: offloaded large tool results, evicted
#: conversation history, and the persistent memories route.
LARGE_RESULTS = "/large_tool_results"
SCRATCH_PREFIXES = (LARGE_RESULTS, "/conversation_history", "/memories")

_WINDOWS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def _virtual(path: str) -> str:
    """The scratch path deepagents makes of ``path``: forward slashes, one
    leading '/', no '.' segments ('.' and './' are the scratch root)."""
    p = (path or "").strip().replace("\\", "/")
    p = posixpath.normpath("/" + p.lstrip("/")) if p else "/"
    return "/" if p in ("/.", "//") else p


def _is_scratch_area(path: str) -> bool:
    """The scratch root ('/', '.', './') or one of deepagents' own areas."""
    p = _virtual(path)
    if p == "/":
        return True
    return any(p == r or p.startswith(r + "/") for r in SCRATCH_PREFIXES)


def looks_real(path: str) -> bool:
    """True for a path under a real root (or a Windows drive/UNC path)."""
    p = (path or "").strip()
    if not p or p in ("/", "."):
        return False
    if _WINDOWS_PATH.match(p):
        return True
    norm = p.replace("\\", "/")
    return any(norm == r or norm.startswith(r + "/") for r in REAL_ROOTS)


def _scratch_files(state) -> dict:
    try:
        files = state.get("files") if state is not None else None
    except Exception:  # noqa: BLE001
        files = None
    return files if isinstance(files, dict) else {}


def _in_scratch(state, path: str) -> bool:
    files = _scratch_files(state)
    if not files:
        return False
    p = path.rstrip("/") or "/"
    if p in files:
        return True
    prefix = p + "/"
    return any(str(k).startswith(prefix) for k in files)


def _text(result) -> str:
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content)
    return ""


def _is_not_found(result) -> bool:
    """A short deepagents 'Error: File ... not found' reply."""
    if not isinstance(result, ToolMessage):
        return False
    text = _text(result).strip()
    low = text.lower()
    return (len(text) < 600 and low.startswith("error")
            and ("not found" in low or "does not exist" in low))


def _saved_match(path: str, files: dict) -> Optional[str]:
    """The saved scratch file a missed ``read_file`` path plainly meant: the
    same path without its extension, or with '.' written as '_' the way
    deepagents names an offloaded tool result. ``None`` otherwise — never a
    merely similar name, which could hand back a DIFFERENT call's result."""
    if not files:
        return None
    p = _virtual(path)
    head, base = posixpath.split(p)
    stem = posixpath.splitext(base)[0]
    for cand in (p, posixpath.join(head, stem),
                 posixpath.join(head, base.replace(".", "_"))):
        if stem and cand != p and cand in files:
            return cand
    return None


def _missing_read_message(path: str, original: str, files: dict) -> str:
    p = _virtual(path)
    if p == LARGE_RESULTS or p.startswith(LARGE_RESULTS + "/"):
        saved = sorted(k for k in files
                       if str(k).startswith(LARGE_RESULTS + "/"))
        msg = (f"Error: no tool result is saved at '{path}'. A tool result is "
               f"saved under {LARGE_RESULTS}/ only when it was too large to "
               "return, and then that tool's own message says so and gives "
               "the exact path (no file extension). ")
        if saved:
            return msg + (f"Saved results in this conversation: {saved[:10]}. "
                          "Read one of those, or use the result in the tool "
                          "message itself.")
        return msg + ("Nothing has been saved there in this conversation, so "
                      "every result you have was returned in full in its tool "
                      "message — use that message, or run the tool again. Do "
                      "not report a value you have not read.")
    names = sorted(str(k) for k in files)
    if names:
        return (f"{original.strip()} Scratch files in this conversation: "
                f"{names[:20]}. A file on the real disk is read with "
                "read_text_file, read_pdf_text or analyze_image.")
    return (f"{original.strip()} Nothing has been written to the scratch space "
            "in this conversation. A file on the real disk is read with "
            "read_text_file, read_pdf_text or analyze_image.")


#: What an empty answer from the scratch space does NOT mean. Field session
#: 2026-10-06: the references sub-agent, asked for Youd et al. (2001) and
#: ASCE 7, ran ``ls('/')`` and four ``grep``s over the (empty) scratch space,
#: got "No files found" / "No matches found", and reported that "tool
#: searches returned no matches" -- without calling one reference tool.
EMPTY_SCRATCH_NOTE = (
    "(That searched only the scratch space: notes written in this "
    "conversation with write_file, and saved large tool results. It is NOT "
    "the reference library, the user's documents or SharePoint, so it says "
    "nothing about what those hold. Search them with the tools made for "
    "them: the reference tools (list_agents, list_methods, describe_method, "
    "call_agent) for the references, search_document after open_document "
    "for a PDF, list_files for real folders.)")

_EMPTY_ANSWER = re.compile(r"^\s*(?:no (?:files|matches|results)\b[^\n]*|)\s*$",
                           re.IGNORECASE)


def _is_empty_search(name: Optional[str], result) -> bool:
    """A scratch ``ls`` / ``glob`` / ``grep`` that found nothing."""
    if name not in ("ls", "glob", "grep") or not isinstance(result,
                                                             ToolMessage):
        return False
    return bool(_EMPTY_ANSWER.match(_text(result)))


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
                "text file use read_text_file. To search your own scratch "
                "notes and any saved large tool results, call grep with no "
                "path.")
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
        # deepagents reads every relative path as a scratch path ('.' is the
        # scratch root), so '.' is never "the real disk" — the 2026-10 eval
        # caught grep(path='.') refused four times running (LP-2).
        if _is_scratch_area(path) and not looks_real(path):
            return None
        if _in_scratch(getattr(request, "state", None), path):
            return None
        if name == "write_file":
            hit = looks_real(path)
        else:
            hit = looks_real(path) or os.path.exists(path)
        if not hit:
            return None
        return ToolMessage(content=_message(name, path),
                           tool_call_id=call.get("id") or "", name=name,
                           status="error")

    @staticmethod
    def _missed_read(request):
        """``(path, args, files)`` when this is a ``read_file`` call, else
        ``None``."""
        call = getattr(request, "tool_call", None) or {}
        if call.get("name") != "read_file":
            return None
        args = call.get("args") or {}
        path = args.get("file_path")
        if not isinstance(path, str) or not path.strip():
            return None
        return path, args, _scratch_files(getattr(request, "state", None))

    @staticmethod
    def _retarget(request, args, alt, arg="file_path"):
        call = dict(request.tool_call)
        call["args"] = {**args, arg: alt}
        if hasattr(request, "override"):
            return request.override(tool_call=call)
        import dataclasses
        return dataclasses.replace(request, tool_call=call)

    def _root_for_dot(self, request):
        """'.' / './' mean the scratch root, but deepagents reads them as a
        path named '/.' that holds nothing, so grep/ls/glob there answer
        'No matches' about files that exist. Send them to '/' instead."""
        call = getattr(request, "tool_call", None) or {}
        name = call.get("name")
        arg = PATH_ARG.get(name)
        if not arg or name in ("write_file", "edit_file", "read_file"):
            return request
        args = call.get("args") or {}
        path = args.get(arg)
        if (isinstance(path, str) and path.strip() not in ("", "/")
                and _virtual(path) == "/"):
            return self._retarget(request, args, "/", arg)
        return request

    @staticmethod
    def _noted(result, note: str):
        content = result.content
        if isinstance(content, list):       # e.g. an image read: keep blocks
            content = [{"type": "text", "text": note}] + list(content)
        else:
            content = f"{note}\n{content}"
        return result.model_copy(update={"content": content})

    @staticmethod
    def _explained(request, result, path, files):
        call = getattr(request, "tool_call", None) or {}
        return ToolMessage(
            content=_missing_read_message(path, _text(result), files),
            tool_call_id=getattr(result, "tool_call_id", None)
            or call.get("id") or "",
            name="read_file", status="error")

    @staticmethod
    def _empty_explained(request, result):
        name = (getattr(request, "tool_call", None) or {}).get("name")
        if _is_empty_search(name, result):
            return ScratchFilesystemGuard._noted(result, EMPTY_SCRATCH_NOTE)
        return result

    def wrap_tool_call(self, request, handler):
        blocked = self._intercept(request)
        if blocked is not None:
            return blocked
        request = self._root_for_dot(request)
        result = self._empty_explained(request, handler(request))
        missed = self._missed_read(request)
        if missed is None or not _is_not_found(result):
            return result
        path, args, files = missed
        alt = _saved_match(path, files)
        if alt is not None:
            retry = handler(self._retarget(request, args, alt))
            if not _is_not_found(retry) and isinstance(retry, ToolMessage):
                return self._noted(retry, f"(Nothing is saved at '{path}'; "
                                          f"read '{alt}', the saved file of "
                                          "that name.)")
        return self._explained(request, result, path, files)

    async def awrap_tool_call(self, request, handler):
        blocked = self._intercept(request)
        if blocked is not None:
            return blocked
        request = self._root_for_dot(request)
        result = self._empty_explained(request, await handler(request))
        missed = self._missed_read(request)
        if missed is None or not _is_not_found(result):
            return result
        path, args, files = missed
        alt = _saved_match(path, files)
        if alt is not None:
            retry = await handler(self._retarget(request, args, alt))
            if not _is_not_found(retry) and isinstance(retry, ToolMessage):
                return self._noted(retry, f"(Nothing is saved at '{path}'; "
                                          f"read '{alt}', the saved file of "
                                          "that name.)")
        return self._explained(request, result, path, files)


__all__ = ["ScratchFilesystemGuard", "looks_real", "PATH_ARG", "REAL_ROOTS",
           "LARGE_RESULTS", "SCRATCH_PREFIXES", "EMPTY_SCRATCH_NOTE"]

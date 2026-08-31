"""Agent-facing SharePoint tools for the web app (Funhouse SDK backed).

Four LangChain tools that let the AGENT work with the team SharePoint during a
conversation — fetch a project file into the session, push a deliverable out,
browse and search folders:

    sharepoint_list_files(path)            list a folder
    sharepoint_download_file(path, ...)    SharePoint -> working folder
    sharepoint_upload_file(local_path, ..) working folder -> SharePoint
    sharepoint_search_files(query, path)   filename search

They are injected into the deep agent by ``webapp.core.build_agent`` via
``build_deep_agent(extra_tools=...)`` ONLY when SharePoint is configured
(``sharepoint_store.configured()``), so unconfigured deployments carry zero
extra tool surface. The client/auth comes from :mod:`webapp.sharepoint_store`
(same delegated-OAuth token file or client credentials as the mirror).

Path convention (kept simple for the model): a RELATIVE path ("borings/",
"reports/site_A.pdf") resolves under the configured base folder
(``GEOTECH_SHAREPOINT_ROOT``); an absolute form — "Shared Documents/...",
"/sites/<site>/...", or a full "https://..." URL — is passed through
unchanged (the SDK accepts all three).

Every tool returns a plain string and NEVER raises: errors come back as
readable text so the agent can report/retry rather than crash the turn.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

from langchain_core.tools import tool

from webapp import sharepoint_store

#: Cap on listed/search entries so a huge folder cannot blow up the context.
MAX_ENTRIES = 60

_NOT_CONFIGURED = ("SharePoint is not configured for this session — no "
                   "site/credentials were provided at launch.")


def _fm():
    return sharepoint_store.get_store().file_manager()


def _root() -> str:
    return sharepoint_store.get_store().root()


def _resolve(path: Optional[str]) -> str:
    """Resolve a tool-supplied path against the configured base folder.

    Absolute forms (full URL, "/sites/...", "Shared Documents/...") pass
    through UNCHANGED (a leading slash is meaningful to the SDK); anything
    else is joined under the configured root.
    """
    p = (path or "").strip().rstrip("/")
    if not p or p == "/":
        return _root()
    low = p.lstrip("/").lower()
    if (low.startswith("http") or low.startswith("sites/")
            or low.startswith("shared documents")):
        return p
    return f"{_root()}/{p.lstrip('/')}"


def _working_dir() -> str:
    """Where downloads land: the conversation's working folder when set."""
    try:
        from funhouse_agent._fileio import default_output_dir
        d = default_output_dir()
        if d:
            return str(d)
    except Exception:
        pass
    import tempfile
    return tempfile.gettempdir()


def _fmt_entry(e: dict) -> str:
    name = e.get("name") or "?"
    kind = e.get("type") or ("folder" if e.get("folder") else "file")
    size = e.get("size")
    size_s = f", {int(size):,} B" if isinstance(size, (int, float)) else ""
    path = e.get("path") or e.get("url") or ""
    return f"- [{kind}] {name}{size_s}" + (f"  ({path})" if path else "")


@tool
def sharepoint_list_files(path: str = "") -> str:
    """List files and folders in a SharePoint folder.

    path: folder to list — relative to the app's base SharePoint folder
    (default "" = the base folder itself), or an absolute
    "Shared Documents/..." / "/sites/..." / full-URL path.
    """
    if not sharepoint_store.configured():
        return _NOT_CONFIGURED
    try:
        remote = _resolve(path)
        entries = _fm().ls(remote) or []
        if not entries:
            return f"(empty or missing folder: {remote})"
        lines = [f"Contents of {remote} ({len(entries)} items"
                 + (f", first {MAX_ENTRIES} shown" if len(entries) > MAX_ENTRIES
                    else "") + "):"]
        lines += [_fmt_entry(e) for e in entries[:MAX_ENTRIES]]
        return "\n".join(lines)
    except Exception as exc:
        return f"SharePoint list error: {type(exc).__name__}: {exc}"


@tool
def sharepoint_download_file(path: str, save_as: str = "") -> str:
    """Download a file from SharePoint into the session working folder, so it
    can be read/analyzed with the file tools (read_pdf_text, subsurface
    parsers, ...) or attached to results.

    path: the SharePoint file — relative to the base folder, or absolute.
    save_as: optional local filename override (defaults to the SharePoint name).
    """
    if not sharepoint_store.configured():
        return _NOT_CONFIGURED
    try:
        remote = _resolve(path)
        name = (save_as or os.path.basename(remote.rstrip("/"))).strip()
        dest_dir = _working_dir()
        os.makedirs(dest_dir, exist_ok=True)
        local = os.path.join(dest_dir, name)
        _fm().download_file(remote, local_path=local, return_bytes=False,
                            overwrite=True)
        size = os.path.getsize(local) if os.path.exists(local) else 0
        return (f"Downloaded {remote} -> {local} ({size:,} bytes). The file "
                "is now in the working folder and available to the file "
                "tools.")
    except FileNotFoundError:
        return (f"SharePoint file not found: {_resolve(path)} — check the "
                "path with sharepoint_list_files or sharepoint_search_files.")
    except Exception as exc:
        return f"SharePoint download error: {type(exc).__name__}: {exc}"


@tool
def sharepoint_upload_file(local_path: str, dest_folder: str = "") -> str:
    """Upload a local file (e.g. a calc package or plot from the working
    folder) to SharePoint.

    local_path: the local file to upload (as returned by save_file /
    calc-package tools).
    dest_folder: SharePoint folder — relative to the base folder (default ""
    = the base folder), or absolute. Created if missing. An existing file of
    the same name is NOT overwritten — a timestamped name is used instead.
    """
    if not sharepoint_store.configured():
        return _NOT_CONFIGURED
    if not os.path.isfile(local_path):
        return f"Local file not found: {local_path}"
    try:
        fm = _fm()
        folder = _resolve(dest_folder)
        try:
            fm.create_folder(folder)
        except Exception:
            pass                                    # may already exist
        name = os.path.basename(local_path)
        remote = f"{folder}/{name}"
        ok = fm.upload_file(local_path, remote, overwrite=False)
        if not ok:                                  # name taken -> unique name
            stem, ext = os.path.splitext(name)
            remote = f"{folder}/{stem}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
            ok = fm.upload_file(local_path, remote, overwrite=False)
        if not ok:
            return f"SharePoint upload failed for {remote} (upload rejected)."
        try:
            url = fm.get_web_url(remote)
        except Exception:
            url = ""
        return (f"Uploaded {local_path} -> {remote}."
                + (f" Link: {url}" if url else ""))
    except Exception as exc:
        return f"SharePoint upload error: {type(exc).__name__}: {exc}"


@tool
def sharepoint_search_files(query: str, path: str = "") -> str:
    """Search SharePoint for files by name.

    query: filename text to search for (e.g. "boring log", "Kinshasa").
    path: optional folder to scope the search — relative to the base folder,
    or absolute. Default searches from the base folder.

    NOTE: search uses an index that lags NEW uploads by several minutes; for
    a file added in the last ~15 minutes use sharepoint_list_files instead.
    """
    if not sharepoint_store.configured():
        return _NOT_CONFIGURED
    try:
        fm = _fm()
        remote = _resolve(path)
        try:
            hits = fm.search_filenames(query, remote)
        except TypeError:                    # backend without a path arg
            hits = fm.search_filenames(query)
        hits = hits or []
        if not hits:
            return f"No files matching '{query}' under {remote}."
        lines = [f"Matches for '{query}' ({len(hits)}"
                 + (f", first {MAX_ENTRIES} shown" if len(hits) > MAX_ENTRIES
                    else "") + "):"]
        for h in hits[:MAX_ENTRIES]:
            lines.append(_fmt_entry(h) if isinstance(h, dict) else f"- {h}")
        return "\n".join(lines)
    except Exception as exc:
        return f"SharePoint search error: {type(exc).__name__}: {exc}"


#: Prompt block injected alongside the tools (build_agent), so the agent knows
#: the capability exists and the path convention.
SHAREPOINT_PROMPT = (
    "SHAREPOINT: This deployment is connected to the team SharePoint. You "
    "have sharepoint_list_files / sharepoint_search_files (browse + find), "
    "sharepoint_download_file (fetch a project file into the working folder "
    "for analysis), and sharepoint_upload_file (publish a deliverable such "
    "as a calc package). Paths are relative to the app's base SharePoint "
    "folder unless given as 'Shared Documents/...', '/sites/...', or a full "
    "URL. When the user references project files 'on SharePoint', use these "
    "tools rather than asking for an upload. For files uploaded RECENTLY "
    "(within the last ~15 minutes), prefer sharepoint_list_files over "
    "sharepoint_search_files: search rides an index that lags new uploads by "
    "several minutes, while listing a folder sees them immediately.")


def tools_if_configured() -> tuple:
    """``(tools, prompt)`` when SharePoint is configured, else ``([], "")``."""
    if not sharepoint_store.configured():
        return [], ""
    return ([sharepoint_list_files, sharepoint_download_file,
             sharepoint_upload_file, sharepoint_search_files],
            SHAREPOINT_PROMPT)


__all__ = ["tools_if_configured", "SHAREPOINT_PROMPT", "MAX_ENTRIES",
           "sharepoint_list_files", "sharepoint_download_file",
           "sharepoint_upload_file", "sharepoint_search_files"]

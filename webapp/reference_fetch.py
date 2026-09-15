"""Reference PDFs from the team SharePoint, fetched when first needed.

Owner decision 2026-09-15: keep the published reference PDFs (DM7, the GECs,
the UFCs...) in SharePoint ``GSE_app/primary_references`` and download each
one only when a chart or worked-example page from it is first needed -- never
all ~750 MB at launch. :func:`register_if_configured` installs a fetcher with
:mod:`funhouse_agent.reference_docs`, which keeps the file in its local cache
for the rest of the cluster session. A local ``GEOTECH_REFERENCES_DOCS``
folder, when set, is still tried first.
"""

from __future__ import annotations

import os
from typing import Optional, Set

from webapp import sharepoint_store

#: Folder holding the reference PDFs, relative to the SharePoint base folder
#: (or absolute "Shared Documents/..." / "sites/...").
ENV_DIR = "GEOTECH_REFERENCES_SHAREPOINT_DIR"
DEFAULT_DIR = "primary_references"


def folder() -> str:
    return os.environ.get(ENV_DIR, "").strip().strip("/") or DEFAULT_DIR


def remote_folder() -> str:
    f = folder()
    if f.lower().startswith(("shared documents", "sites/")):
        return f
    return f"{sharepoint_store.get_store().root()}/{f}"


def _fetch(name: str, dest: str) -> Optional[str]:
    """Download ``name`` from the SharePoint folder to ``dest``; the file only
    appears under its final name once complete."""
    fm = sharepoint_store.get_store().file_manager()
    part = dest + ".part"
    fm.download_file(f"{remote_folder()}/{name}", local_path=part,
                     return_bytes=False, overwrite=True)
    if not os.path.isfile(part) or os.path.getsize(part) == 0:
        return None
    os.replace(part, dest)
    return dest


def register_if_configured() -> bool:
    """Install the SharePoint fetcher when SharePoint is configured."""
    from funhouse_agent import reference_docs
    if not sharepoint_store.configured():
        return False
    _fetch.description = f"SharePoint {folder()}/"
    reference_docs.register_fetcher(_fetch)
    return True


def available_names() -> Optional[Set[str]]:
    """File names in the SharePoint folder, or None if it cannot be listed."""
    try:
        entries = sharepoint_store.get_store().file_manager().ls(
            remote_folder()) or []
    except Exception:  # noqa: BLE001
        return None
    return {str(e["name"]) for e in entries
            if isinstance(e, dict) and e.get("name")}


__all__ = ["register_if_configured", "available_names", "folder",
           "remote_folder", "ENV_DIR", "DEFAULT_DIR"]

"""A deployment's local reference packs, as the app sees them (BACKLOG REF-64).

geotech-references 1.6.0 indexes the packs named by
``GEOTECH_REFERENCES_PACKS`` beside its built-in references
(``geotech_references._packs``; the format is the refs repo's
``docs/REFERENCE_PACKS.md``). Search, citation and figure lookups need no app
change. What the app adds (plan ``module_work/REFERENCE_PACKS_PLAN.md``
sections 2.4 and 2.7):

- :func:`packs_line`: one line for the librarian's prompt and its
  description, so the main agent routes questions about the deployment's own
  manual to the librarian and the librarian can scope a search to it;
- :func:`pack_records`: the packs in force, for the build record.

Both are empty with no packs, and on a refs older than 1.6.0 (feature
detected by ``hasattr(_retrieval_db, "pack_dirs")``), so the app's prompts
and records are then exactly what they were.
"""

from typing import List


def _packs_module():
    """``geotech_references._packs``, or None on a refs without packs."""
    try:
        from geotech_references import _retrieval_db
        if not hasattr(_retrieval_db, "pack_dirs"):
            return None
        from geotech_references import _packs
        return _packs
    except Exception:  # noqa: BLE001 - no refs, or an older one: no packs
        return None


def pack_records() -> List[dict]:
    """``[{id, title, reference_id, edition, files, sha}]`` per valid pack;
    ``[]`` with none."""
    mod = _packs_module()
    if mod is None:
        return []
    try:
        return list(mod.pack_records())
    except Exception:  # noqa: BLE001 - a record, never a failed build
        return []


def pack_problems() -> List[str]:
    """The packs (or pack files) the library skipped, and why."""
    mod = _packs_module()
    if mod is None:
        return []
    try:
        return list(mod.problems())
    except Exception:  # noqa: BLE001
        return []


def packs_line() -> str:
    """``"This deployment adds: <title> (`local_x`), ..."``; ``""`` with no
    packs."""
    mod = _packs_module()
    if mod is None:
        return ""
    names = []
    try:
        for pid, folder in mod.packs():
            try:
                title = str(mod.read_pack_json(folder).get(
                    "reference_title") or "").strip()
            except (OSError, ValueError):
                title = ""
            names.append(f"{title} (`{pid}`)" if title else f"`{pid}`")
    except Exception:  # noqa: BLE001
        return ""
    if not names:
        return ""
    return "This deployment adds: " + ", ".join(names) + "."

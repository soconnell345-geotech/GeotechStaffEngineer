"""Runtime guard: make ``funhouse_agent.deep`` importable on Databricks WITHOUT
``dbutils.library.restartPython()``.

Background
----------
Databricks cluster runtimes PRE-IMPORT an old ``typing_extensions`` (<4.13) at
kernel startup. When you then ``%pip install geotech-staff-engineer[deep]`` the
fresh ``typing_extensions>=4.13`` lands on disk, but the STALE module object is
already sitting in ``sys.modules``. ``langchain-protocol`` (pulled in by
langgraph / deepagents) calls ``typing_extensions.TypedDict(..., extra_items=...)``
(PEP 728) at import time, and the stale <4.13 module rejects it with::

    TypeError: TypedDict() got an unexpected keyword argument 'extra_items'

The historical workaround was ``dbutils.library.restartPython()``. This module
removes that requirement for the normal case: it reloads the already-installed
newer ``typing_extensions`` IN PLACE. ``importlib.reload`` re-executes the same
module object, so the attribute ``typing_extensions.TypedDict`` is rebound to the
new (PEP 728-aware) implementation and every *later* ``typing_extensions.TypedDict(...)``
call — including langchain-protocol's, which happens at ITS import time — sees the
new keyword. That is why :func:`ensure_typing_extensions` must run BEFORE the first
langchain / deepagents import (it is called at the top of
``funhouse_agent/deep/__init__.py``).

Failure modes are all reported as clear ``RuntimeError`` messages that fall back
to the restart instruction (see :class:`TypingExtensionsError`).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from dataclasses import dataclass
from typing import Optional

#: Distribution / top-level module name (they coincide for typing_extensions).
_TE_NAME = "typing_extensions"


class TypingExtensionsError(RuntimeError):
    """Raised when a usable ``typing_extensions>=4.13`` cannot be made available
    in-process (disk copy too old, or an in-place reload did not take)."""


@dataclass(frozen=True)
class TypingExtensionsCheck:
    """Result of a successful :func:`ensure_typing_extensions` call.

    ``action`` is one of:

    * ``"not-loaded"`` — ``typing_extensions`` was not imported yet; the next
      ``import`` will pick up the on-disk version, so there is nothing to fix.
    * ``"already-ok"`` — the loaded module already supports PEP 728.
    * ``"reloaded"`` — a stale module was reloaded in place and now supports it.
    """

    action: str
    loaded_version: Optional[str]
    installed_version: Optional[str]
    message: str


def _installed_dist_version(name: str = _TE_NAME) -> Optional[str]:
    """Version of the *installed* (on-disk) distribution, or ``None`` if it is
    not found. Isolated in its own function so tests can monkeypatch the disk
    lookup independently of whatever object is loaded in ``sys.modules``."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_tuple(version: object) -> tuple:
    """Best-effort numeric tuple for a dotted version string. Non-numeric
    suffixes (``rc1``, ``.dev0``, …) are truncated per-segment; missing/garbage
    segments read as 0. Good enough for a ``>=4.13`` floor comparison."""
    parts = []
    for chunk in str(version).split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _version_lt(a: object, b: object) -> bool:
    return _version_tuple(a) < _version_tuple(b)


def _supports_pep728(module: object) -> bool:
    """Cheapest probe that distinguishes ``typing_extensions>=4.13``.

    4.13+ ``TypedDict`` accepts the PEP 728 ``closed=`` keyword (the same release
    that added ``extra_items=``); older versions raise ``TypeError``. Constructing
    an empty closed TypedDict is cheap and side-effect-free. Never raises — an old
    module's ``TypeError`` (or any other surprise) is swallowed and read as "no
    support", which is exactly the "do not probe ``extra_items`` on old versions in
    a way that raises uncaught" requirement."""
    TypedDict = getattr(module, "TypedDict", None)
    if TypedDict is None:
        return False
    try:
        TypedDict("_GeotechRuntimeProbe", {}, closed=True)
        return True
    except Exception:
        return False


def _restart_hint(min_version: str) -> str:
    return (
        "Restart the Python kernel so the freshly installed "
        f"typing_extensions>={min_version} is picked up cleanly:\n"
        "    dbutils.library.restartPython()   # Databricks\n"
        "(or restart your interpreter / Jupyter kernel). Alternatively, install "
        "typing_extensions as a CLUSTER-scoped library so it is present before "
        "the runtime pre-imports the old copy — see the funhouse agent guide."
    )


def ensure_typing_extensions(min_version: str = "4.13") -> TypingExtensionsCheck:
    """Ensure the in-process ``typing_extensions`` supports PEP 728 TypedDict.

    Call this BEFORE importing anything that touches ``langchain`` / ``langgraph``
    / ``deepagents`` (they build PEP 728 ``TypedDict``\\s at import time).

    Returns a :class:`TypingExtensionsCheck` on every non-error path. Raises
    :class:`TypingExtensionsError` (a ``RuntimeError``) with an actionable message
    when the situation cannot be auto-repaired:

    * the on-disk distribution is itself older than ``min_version`` (tells the
      user to ``pip install 'typing_extensions>=<min_version>'``), or
    * an in-place reload failed / did not take (falls back to the restart hint).
    """
    module = sys.modules.get(_TE_NAME)
    installed = _installed_dist_version()

    # (1) Not imported yet — the next `import typing_extensions` reads the on-disk
    # version, so there is nothing loaded to reload. Leave disk-too-old detection
    # to the real import + downstream use rather than pre-empting it here.
    if module is None:
        return TypingExtensionsCheck(
            action="not-loaded",
            loaded_version=None,
            installed_version=installed,
            message="typing_extensions not yet imported; nothing to reload.",
        )

    loaded_version = getattr(module, "__version__", None)

    # (2) Decide staleness. The feature probe is authoritative when a TypedDict is
    # present; a known-old __version__ is also treated as stale (belt and braces,
    # and it honors the version-vs-installed comparison the fix is described by).
    feature_ok = _supports_pep728(module)
    version_stale = loaded_version is not None and _version_lt(loaded_version, min_version)
    if feature_ok and not version_stale:
        return TypingExtensionsCheck(
            action="already-ok",
            loaded_version=loaded_version,
            installed_version=installed,
            message="Loaded typing_extensions already supports PEP 728 TypedDict.",
        )

    # (3) The loaded module is stale. A reload only helps if the on-disk copy is
    # new enough — otherwise say so plainly.
    if installed is not None and _version_lt(installed, min_version):
        raise TypingExtensionsError(
            f"funhouse_agent.deep needs typing_extensions>={min_version} "
            "(langchain/langgraph build PEP 728 TypedDicts at import), but the "
            f"installed version is {installed}. Install a newer one:\n"
            f"    %pip install --upgrade 'typing_extensions>={min_version}'\n"
            "then re-import. " + _restart_hint(min_version)
        )

    # (4) Disk is adequate (or unknown) but the in-memory object is stale — reload
    # it in place so existing `typing_extensions` references see the new TypedDict.
    try:
        reloaded = importlib.reload(module)
    except Exception as exc:  # noqa: BLE001 - any reload failure -> restart fallback
        raise TypingExtensionsError(
            "funhouse_agent.deep tried to reload a stale typing_extensions "
            f"(loaded={loaded_version!r}, installed={installed!r}) in place, but "
            f"the reload failed: {exc}\n" + _restart_hint(min_version)
        ) from exc

    if not _supports_pep728(reloaded):
        raise TypingExtensionsError(
            "funhouse_agent.deep reloaded typing_extensions in place but it still "
            "does not support PEP 728 TypedDict "
            f"(loaded={getattr(reloaded, '__version__', None)!r}, "
            f"installed={installed!r}). An even older copy is probably winning at "
            "cluster/library scope.\n" + _restart_hint(min_version)
        )

    return TypingExtensionsCheck(
        action="reloaded",
        loaded_version=loaded_version,
        installed_version=getattr(reloaded, "__version__", installed),
        message="Reloaded a stale typing_extensions in place; PEP 728 TypedDict now available.",
    )

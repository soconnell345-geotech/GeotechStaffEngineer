"""Runtime version-drift guard for the web app.

pip caps (pyproject) bound what OUR install resolves, but a deployment can
still run untested versions — cluster base images shadow packages, unpinned
installs regress, mirrors serve stale metadata. Twice this produced
hard-to-diagnose outages (Pygments 2.20 kernel SIGTERMs; deepagents 0.7.11's
missing write_todos looping every turn to GraphRecursionError). This module
detects it AT RUNTIME and says so in plain language.

``TESTED_MAX_EXCLUSIVE`` mirrors the pyproject agent-stack caps and is bumped
in the same commit that raises them (after a green deep+webapp run on the
newer version). ``check_versions()`` returns human-readable warnings; the
sidebar shows them and the diagnostics report includes them.
"""

from __future__ import annotations

import importlib.metadata as _md
from typing import List, Optional, Tuple

#: package -> (min_supported, first_UNTESTED version). None min = don't check.
#: Keep in lockstep with the pyproject caps + the last drift-gate run
#: (2026-09: deepagents 0.7.11 / langchain 1.3.18 / langgraph 1.2.11 /
#: streamlit 1.62 / openai 2.x all exercised offline).
TESTED_MAX_EXCLUSIVE = {
    "deepagents": ("0.6.8", "0.8"),
    "langchain": ("1.3", "1.4"),
    "langgraph": ("1.2", "1.3"),
    "openai": (None, "3"),
    "streamlit": ("1.36", "1.63"),
}


def _ver_tuple(text: str) -> Tuple[int, ...]:
    """Lenient release-tuple parse ("1.62.0" -> (1, 62, 0)); ignores suffixes."""
    parts: List[int] = []
    for token in text.split("."):
        digits = ""
        for ch in token:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or (0,)


def installed_version(package: str) -> Optional[str]:
    try:
        return _md.version(package)
    except Exception:
        return None


def check_versions(overrides: Optional[dict] = None) -> List[str]:
    """Return one warning line per agent-stack package running OUTSIDE the
    tested range (empty list = all good). ``overrides`` injects versions for
    tests: ``{package: version_str}``."""
    warnings: List[str] = []
    for package, (vmin, vmax) in TESTED_MAX_EXCLUSIVE.items():
        ver = ((overrides or {}).get(package)
               if overrides and package in (overrides or {})
               else installed_version(package))
        if ver is None:
            continue                       # not installed / not detectable
        vt = _ver_tuple(ver)
        if vmax is not None and vt >= _ver_tuple(vmax):
            warnings.append(
                f"{package} {ver} is NEWER than the tested range (<{vmax}) — "
                "agent behavior may break in untested ways (this exact "
                "situation caused the 2026-08 every-question failures). "
                "Prefer reinstalling with the pinned app version.")
        elif vmin is not None and vt < _ver_tuple(vmin):
            warnings.append(
                f"{package} {ver} is OLDER than the supported floor "
                f"(>={vmin}) — upgrade the app install.")
    return warnings


def drift_summary(overrides: Optional[dict] = None) -> str:
    """One-line status for captions/diagnostics: 'OK' or 'N package(s) outside
    tested range'."""
    warnings = check_versions(overrides)
    if not warnings:
        return "agent-stack versions: OK"
    return f"agent-stack versions: {len(warnings)} outside tested range"


__all__ = ["check_versions", "drift_summary", "installed_version",
           "TESTED_MAX_EXCLUSIVE"]

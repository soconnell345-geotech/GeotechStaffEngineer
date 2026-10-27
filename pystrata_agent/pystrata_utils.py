"""
Utility functions for the pystrata agent module.

Provides runtime dependency checking and import helpers for pystrata,
following the same pattern as opensees_agent/opensees_utils.py.
"""

from geotech_common.optional_packages import import_error


def pystrata_import_error():
    """Why pystrata cannot be imported (not installed, or its import's own
    error), or None when it can (BACKLOG INF-54)."""
    return import_error("pystrata")


def has_pystrata() -> bool:
    """Return True if pystrata is importable."""
    return pystrata_import_error() is None


def import_pystrata():
    """Import and return pystrata with a helpful error message.

    Returns
    -------
    module
        The pystrata module.

    Raises
    ------
    ImportError
        If pystrata is not installed or its import fails.
    """
    error = pystrata_import_error()
    if error:
        raise ImportError(
            f"pystrata is required for equivalent-linear site response. {error}"
        )
    import pystrata
    return pystrata

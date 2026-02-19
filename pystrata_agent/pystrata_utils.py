"""
Utility functions for the pystrata agent module.

Provides runtime dependency checking and import helpers for pystrata,
following the same pattern as opensees_agent/opensees_utils.py.
"""


def has_pystrata() -> bool:
    """Return True if pystrata is importable."""
    try:
        import pystrata  # noqa: F401
        return True
    except ImportError:
        return False


def import_pystrata():
    """Import and return pystrata with a helpful error message.

    Returns
    -------
    module
        The pystrata module.

    Raises
    ------
    ImportError
        If pystrata is not installed.
    """
    try:
        import pystrata
        return pystrata
    except ImportError:
        raise ImportError(
            "pystrata is required for equivalent-linear site response. "
            "Install with: pip install pystrata"
        )

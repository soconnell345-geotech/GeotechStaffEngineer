"""
Utility functions for GSTools agent.
"""

from geotech_common.optional_packages import import_error


def gstools_import_error():
    """Why gstools cannot be imported (not installed, or its import's own
    error), or None when it can (BACKLOG INF-54)."""
    return import_error("gstools")


def has_gstools():
    """Check if gstools is installed and importable."""
    return gstools_import_error() is None


def import_gstools():
    """Import and return the gstools module."""
    error = gstools_import_error()
    if error:
        raise ImportError(error)
    import gstools
    return gstools

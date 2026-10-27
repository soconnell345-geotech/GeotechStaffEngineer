"""
Utility functions for SALib agent.
"""

from geotech_common.optional_packages import import_error


def salib_import_error():
    """Why SALib cannot be imported (not installed, or its import's own
    error), or None when it can (BACKLOG INF-54: pandas 3 failing on numpy
    1.26 read as "SALib is not installed")."""
    return import_error("SALib")


def has_salib():
    """Check if SALib is installed and importable."""
    return salib_import_error() is None


def _require_salib():
    error = salib_import_error()
    if error:
        raise ImportError(error)


def import_salib_sobol_sample():
    """Import and return SALib.sample.sobol."""
    _require_salib()
    from SALib.sample import sobol
    return sobol


def import_salib_sobol_analyze():
    """Import and return SALib.analyze.sobol."""
    _require_salib()
    from SALib.analyze import sobol
    return sobol


def import_salib_morris_sample():
    """Import and return SALib.sample.morris."""
    _require_salib()
    from SALib.sample import morris
    return morris


def import_salib_morris_analyze():
    """Import and return SALib.analyze.morris."""
    _require_salib()
    from SALib.analyze import morris
    return morris

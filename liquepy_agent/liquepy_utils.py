"""
Utility functions for liquepy agent.

Provides optional-dependency guards and import helpers following
the same pattern as opensees_agent and pystrata_agent.
"""

from geotech_common.optional_packages import import_error


def liquepy_import_error():
    """Why liquepy cannot be imported (not installed, or its import's own
    error), or None when it can (BACKLOG INF-54)."""
    return import_error("liquepy")


def has_liquepy():
    """Return True if liquepy is installed and importable."""
    return liquepy_import_error() is None


def import_liquepy():
    """Import and return the liquepy package, raising RuntimeError if it is
    missing or its import fails."""
    error = liquepy_import_error()
    if error:
        raise RuntimeError(error)
    import liquepy
    return liquepy


def import_liquepy_trigger():
    """Import and return liquepy.trigger module."""
    import_liquepy()
    import liquepy.trigger as trigger
    return trigger


def import_liquepy_field():
    """Import and return liquepy.field module."""
    import_liquepy()
    import liquepy.field as field
    return field
